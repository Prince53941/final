import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    """
    This app performs **EDA**, **outlier detection**, and **supervised ML models**  
    (Logistic Regression, Random Forest, XGBoost) for credit card fraud detection.
    """
)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_sample_data():
    """Generate a synthetic dataset if user doesn't upload."""
    np.random.seed(42)
    n_samples = 2000
    fraud_ratio = 0.015
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    V_cols = {f"V{i}": np.random.normal(0, 1, n_samples) for i in range(1, 29)}
    amounts = np.abs(np.random.normal(50, 40, n_samples)).round(2)
    labels = np.array([0] * n_normal + [1] * n_fraud)
    np.random.shuffle(labels)

    df = pd.DataFrame(V_cols)
    df["Amount"] = amounts
    df["Class"] = labels
    return df

def eda_basic(df: pd.DataFrame):
    st.subheader("1️⃣ Basic EDA")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Preview**")
        st.dataframe(df.head())

    with col2:
        st.write("**Info**")
        buffer = []
        df.info(buf=buffer)
        s = "\n".join(buffer)
        st.text(s)

    st.write("**Missing Values**")
    st.write(df.isnull().sum())

    st.write("**Summary Statistics**")
    st.write(df.describe())

    st.write("**Class Distribution**")
    st.write(df[TARGET_COLUMN].value_counts())
    st.write("Class distribution (%)")
    st.write(df[TARGET_COLUMN].value_counts(normalize=True) * 100)

    # Plots
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        sns.countplot(x=TARGET_COLUMN, data=df, ax=ax)
        ax.set_title("Class Distribution (0 = Normal, 1 = Fraud)")
        st.pyplot(fig)

    if "Amount" in df.columns:
        with col4:
            fig, ax = plt.subplots()
            sns.boxplot(x=TARGET_COLUMN, y="Amount", data=df, ax=ax)
            ax.set_title("Transaction Amount vs Class")
            st.pyplot(fig)

def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def run_isolation_forest(X_train_scaled, X_test_scaled, y_test):
    st.subheader("2️⃣ Outlier Detection: Isolation Forest")

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_train_scaled)

    y_pred_if = iso.predict(X_test_scaled)
    y_pred_if_binary = np.where(y_pred_if == -1, 1, 0)

    st.write("**Classification Report**")
    report = classification_report(y_test, y_pred_if_binary, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_test, y_pred_if_binary)
    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    st.write("### Logistic Regression")
    lr = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("**Classification Report**")
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    auc = roc_auc_score(y_test, y_prob)
    st.write(f"**ROC-AUC:** {auc:.4f}")

    return lr, y_prob

def train_random_forest(X_train, X_test, y_train, y_test):
    st.write("### Random Forest")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("**Classification Report**")
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    auc = roc_auc_score(y_test, y_prob)
    st.write(f"**ROC-AUC:** {auc:.4f}")

    return rf, y_prob

def train_xgboost(X_train, X_test, y_train, y_test):
    st.write("### XGBoost Classifier")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=10
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("**Classification Report**")
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    auc = roc_auc_score(y_test, y_prob)
    st.write(f"**ROC-AUC:** {auc:.4f}")

    return xgb, y_prob

def plot_roc_curves(y_test, probs_dict):
    st.subheader("4️⃣ ROC Curve Comparison")
    fig, ax = plt.subplots()

    for name, y_prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(fig)

def predict_single_transaction(model, scaler, sample_dict, feature_columns, scaled=True):
    sample_df = pd.DataFrame([sample_dict])
    sample_df = sample_df[feature_columns]

    if scaled:
        sample_scaled = scaler.transform(sample_df)
        prob = model.predict_proba(sample_scaled)[:, 1][0]
    else:
        prob = model.predict_proba(sample_df)[:, 1][0]

    return prob

# -----------------------------
# SIDEBAR: DATA INPUT
# -----------------------------
st.sidebar.header("Data Options")

data_source = st.sidebar.radio(
    "Choose dataset",
    ["Use sample synthetic data", "Upload your own CSV"],
)

if data_source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or switch to sample data.")
        st.stop()
else:
    df = load_sample_data()

if TARGET_COLUMN not in df.columns:
    st.error(f"'{TARGET_COLUMN}' column not found in the dataset.")
    st.stop()

# -----------------------------
# MAIN WORKFLOW
# -----------------------------
with st.expander("🔍 Exploratory Data Analysis (EDA)", expanded=True):
    eda_basic(df)

# Prepare features
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_features(df)

with st.expander("🧪 Outlier Detection (Isolation Forest)", expanded=False):
    if st.button("Run Isolation Forest"):
        run_isolation_forest(X_train_scaled, X_test_scaled, y_test)

st.subheader("3️⃣ Supervised Machine Learning Models")

model_choice = st.selectbox(
    "Select model to train",
    ["Logistic Regression", "Random Forest", "XGBoost (Gradient Boosting)"]
)

trained_model = None
probs_dict = {}

if st.button("Train Selected Model"):
    if model_choice == "Logistic Regression":
        model, y_prob = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        trained_model = model
        probs_dict["Logistic Regression"] = y_prob
        st.session_state["model_type"] = "lr"
    elif model_choice == "Random Forest":
        model, y_prob = train_random_forest(X_train, X_test, y_train, y_test)
        trained_model = model
        probs_dict["Random Forest"] = y_prob
        st.session_state["model_type"] = "rf"
    else:
        model, y_prob = train_xgboost(X_train, X_test, y_train, y_test)
        trained_model = model
        probs_dict["XGBoost"] = y_prob
        st.session_state["model_type"] = "xgb"

    st.session_state["trained_model"] = trained_model
    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = list(df.drop(columns=[TARGET_COLUMN]).columns)
    st.success("Model trained and stored in session.")

    plot_roc_curves(y_test, probs_dict)

# -----------------------------
# REAL-TIME STYLE PREDICTION
# -----------------------------
st.subheader("5️⃣ Real-Time Style Prediction")

if "trained_model" in st.session_state:
    model_type = st.session_state.get("model_type", "lr")
    model = st.session_state["trained_model"]
    scaler = st.session_state["scaler"]
    feature_columns = st.session_state["feature_columns"]

    st.write("Use a row from the **test set** as a sample transaction.")

    # Pick index from test set
    idx = st.number_input(
        "Select row index from test set",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    sample_series = pd.DataFrame(X_test).iloc[idx]
    st.write("**Transaction Features:**")
    st.json(sample_series.to_dict())

    if st.button("Predict Fraud Probability for this Transaction"):
        sample_dict = sample_series.to_dict()
        # Logistic Regression uses scaled features, RF/XGB can use raw
        use_scaled = True if model_type == "lr" else False

        prob = predict_single_transaction(
            model=model,
            scaler=scaler,
            sample_dict=sample_dict,
            feature_columns=feature_columns,
            scaled=use_scaled
        )

        st.write(f"### Fraud Probability: `{prob:.4f}`")
        if prob > 0.5:
            st.error("⚠️ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction is likely NORMAL.")
else:
    st.info("Train a model first to enable real-time prediction section.")
