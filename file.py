import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# BASIC CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Mutual Fund Risk Classification",
    layout="wide"
)

st.title("📊 Machine Learning for Mutual Fund Risk Classification")
st.write(
    """
    This app replicates your notebook as an interactive dashboard:
    - Cleans and preprocesses the mutual fund dataset  
    - Performs EDA  
    - Trains a Logistic Regression model  
    - Lets you **predict risk level** for new inputs  
    """
)

# --------------------------------------------------
# DATA LOADING & PREPROCESSING
# --------------------------------------------------

@st.cache_data
def load_default_data():
    """
    Try to load the default CSV from the repo.
    If not found, return None (user can upload).
    """
    try:
        df = pd.read_csv("Mutual_fund Data.csv")
        return df
    except FileNotFoundError:
        return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. AUM cleaning: create AUM_in_cr as in notebook
    if "AUM" in df.columns:
        df["AUM_in_cr"] = df["AUM"].astype(str).str.lower()
        df["AUM_in_cr"] = df["AUM_in_cr"].str.replace(" cr", "", regex=False).str.replace(",", "", regex=False)
        df["AUM_in_cr"] = df["AUM_in_cr"].str.replace("lakh", "0.01", regex=False).str.replace("k", "", regex=False)
        df["AUM_in_cr"] = pd.to_numeric(df["AUM_in_cr"], errors="coerce")
        df["AUM_in_cr"] = df["AUM_in_cr"].fillna(0)

    # 2. Return columns cleaning
    return_cols = ["1 month return", "1 Year return", "3 Year Return"]
    for col in return_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 3. Strip text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def get_feature_target(df: pd.DataFrame):
    """
    Prepare X, y, label encoder and scaler based on your notebook logic.
    """
    feature_cols = [
        "Morning star rating",
        "Value Research rating",
        "1 month return",
        "1 Year return",
        "3 Year Return",
        "AUM_in_cr",
    ]
    target_col = "Risk"

    # Ensure required columns exist
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, y_encoded, feature_cols, target_col, label_encoder, scaler


# --------------------------------------------------
# SIDEBAR: DATA SOURCE
# --------------------------------------------------
st.sidebar.header("📂 Data Source")

data_option = st.sidebar.radio(
    "Choose dataset",
    ("Use default CSV in repo", "Upload your own CSV")
)

df = None

if data_option == "Use default CSV in repo":
    df = load_default_data()
    if df is None:
        st.warning("Default file 'Mutual_fund Data.csv' not found. Please upload a CSV from sidebar.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload mutual fund CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

if df is None:
    st.stop()

# Preprocess
df = preprocess_data(df)

# --------------------------------------------------
# TABS LAYOUT
# --------------------------------------------------
tab_eda, tab_model, tab_predict = st.tabs(["🔍 EDA", "🤖 Model Training", "🎯 Predict Risk"])

# --------------------------------------------------
# TAB 1: EDA
# --------------------------------------------------
with tab_eda:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Dataset Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    with col2:
        st.write("### Summary Statistics")
        st.dataframe(df.describe(include="all").transpose())

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

    # Plots
    st.write("### Risk Level Distribution")
    if "Risk" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x="Risk", data=df, ax=ax)
        ax.set_title("Distribution of Risk Categories")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if "Category" in df.columns and "AUM_in_cr" in df.columns:
        st.write("### AUM by Fund Category (Bar Chart)")
        aum_by_cat = df.groupby("Category")["AUM_in_cr"].sum().reset_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(x="Category", y="AUM_in_cr", data=aum_by_cat, ax=ax2)
        ax2.set_title("Total AUM (in crores) by Category")
        ax2.tick_params(axis="x", rotation=45)
        st.pyplot(fig2)

    if "AMC" in df.columns and "NAV" in df.columns:
        st.write("### Average NAV by AMC (Top 15)")
        avg_nav_by_amc = (
            df.groupby("AMC")["NAV"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(avg_nav_by_amc["AMC"], avg_nav_by_amc["NAV"], marker="o")
        ax3.set_title("Average NAV by AMC (Top 15)")
        ax3.set_xlabel("AMC")
        ax3.set_ylabel("Average NAV")
        plt.xticks(rotation=45, ha="right")
        ax3.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig3)

# --------------------------------------------------
# TAB 2: MODEL TRAINING
# --------------------------------------------------
with tab_model:
    st.subheader("Model: Logistic Regression (from your notebook)")

    try:
        X, X_scaled, y, y_encoded, feature_cols, target_col, label_encoder, scaler = get_feature_target(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.3, 0.05)

    if st.button("Train Logistic Regression Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42
        )

        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.write("### Model Performance Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")

        st.write("### Detailed Classification Report")
        report_df = pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        st.dataframe(report_df)

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # Save in session for prediction tab
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["feature_cols"] = feature_cols
        st.session_state["label_encoder"] = label_encoder

        st.success("Model trained and stored for prediction tab ✅")

# --------------------------------------------------
# TAB 3: PREDICTION
# --------------------------------------------------
with tab_predict:
    st.subheader("Predict Risk Level for a New Mutual Fund Scheme")

    if "model" not in st.session_state:
        st.info("Train the model in the **Model Training** tab first.")
    else:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        feature_cols = st.session_state["feature_cols"]
        label_encoder = st.session_state["label_encoder"]

        # Use ranges from data to guide user inputs
        X_current = df[feature_cols]

        col_left, col_right = st.columns(2)
        inputs = {}

        with col_left:
            min_mstar = float(X_current["Morning star rating"].min())
            max_mstar = float(X_current["Morning star rating"].max())
            inputs["Morning star rating"] = st.slider(
                "Morning star rating",
                min_value=min_mstar,
                max_value=max_mstar,
                value=float(X_current["Morning star rating"].median()),
                step=0.5,
            )

            min_vr = float(X_current["Value Research rating"].min())
            max_vr = float(X_current["Value Research rating"].max())
            inputs["Value Research rating"] = st.slider(
                "Value Research rating",
                min_value=min_vr,
                max_value=max_vr,
                value=float(X_current["Value Research rating"].median()),
                step=0.5,
            )

            min_1m = float(X_current["1 month return"].min())
            max_1m = float(X_current["1 month return"].max())
            inputs["1 month return"] = st.slider(
                "1 month return (%)",
                min_value=min_1m,
                max_value=max_1m,
                value=float(X_current["1 month return"].median()),
            )

        with col_right:
            min_1y = float(X_current["1 Year return"].min())
            max_1y = float(X_current["1 Year return"].max())
            inputs["1 Year return"] = st.slider(
                "1 Year return (%)",
                min_value=min_1y,
                max_value=max_1y,
                value=float(X_current["1 Year return"].median()),
            )

            min_3y = float(X_current["3 Year Return"].min())
            max_3y = float(X_current["3 Year Return"].max())
            inputs["3 Year Return"] = st.slider(
                "3 Year Return (%)",
                min_value=min_3y,
                max_value=max_3y,
                value=float(X_current["3 Year Return"].median()),
            )

            min_aum = float(X_current["AUM_in_cr"].min())
            max_aum = float(X_current["AUM_in_cr"].max())
            inputs["AUM_in_cr"] = st.slider(
                "AUM (in crores)",
                min_value=min_aum,
                max_value=max_aum,
                value=float(X_current["AUM_in_cr"].median()),
            )

        if st.button("Predict Risk"):
            sample_df = pd.DataFrame([inputs])[feature_cols]
            sample_scaled = scaler.transform(sample_df)
            pred_encoded = model.predict(sample_scaled)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            st.write("### Predicted Risk Category")
            st.success(f"🔮 The model predicts: **{pred_label}**")
