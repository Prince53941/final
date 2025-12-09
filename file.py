import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

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

# ------------------------------------
# CONFIG
# ------------------------------------
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

# ------------------------------------
# LOAD SAMPLE DATA
# ------------------------------------
def load_sample_data():
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

# ------------------------------------
# FIXED EDA FUNCTION (works now)
# ------------------------------------
def eda_basic(df: pd.DataFrame):
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Dataset Preview")
        st.dataframe(df.head())

    with col2:
        st.write("### Dataset Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Class Distribution")
    st.write(df[TARGET_COLUMN].value_counts())

    # Charts
    fig, ax = plt.subplots()
    sns.countplot(x=TARGET_COLUMN, data=df, ax=ax)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

    if "Amount" in df.columns:
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=TARGET_COLUMN, y="Amount", data=df, ax=ax2)
        ax2.set_title("Amount vs Class")
        st.pyplot(fig2)

# ------------------------------------
# PREPARE FEATURES
# ------------------------------------
def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# ------------------------------------
# ISOLATION FOREST
# ------------------------------------
def run_isolation_forest(X_train_scaled, X_test_scaled, y_test):
    st.subheader("Outlier Detection: Isolation Forest")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    iso.fit(X_train_scaled)

    y_pred = iso.predict(X_test_scaled)
    y_pred = np.where(y_pred == -1, 1, 0)

    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)

# ------------------------------------
# SUPERVISED ML MODELS
# ------------------------------------
def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    st.subheader("Logistic Regression")
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]

    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    auc = roc_auc_score(y_test, y_prob)
    st.write("ROC-AUC:", auc)

    return lr, y_prob

def train_random_forest(X_train, X_test, y_train, y_test):
    st.subheader("Random Forest")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    auc = roc_auc_score(y_test, y_prob)
    st.write("ROC-AUC:", auc)

    return rf, y_prob

def train_xgboost(X_train, X_test, y_train, y_test):
    st.subheader("XGBoost")
    xgb = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss",
        scale_pos_weight=10
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    auc = roc_auc_score(y_test, y_prob)
    st.write("ROC-AUC:", auc)

    return xgb, y_prob

# ------------------------------------
# ROC PLOT
# ------------------------------------
def plot_roc(y_test, model_probs):
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots()

    for label, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    st.pyplot(fig)

# ------------------------------------
# REAL-TIME PREDICTION
# ------------------------------------
def predict_single(model, scaler, sample_dict, feature_columns, scaled=True):
    df_sample = pd.DataFrame([sample_dict])[feature_columns]
    if scaled:
        df_sample = scaler.transform(df_sample)
    prob = model.predict_proba(df_sample)[:, 1][0]
    return prob

# ------------------------------------
# SIDEBAR
# ------------------------------------
st.sidebar.header("Dataset Options")

choice = st.sidebar.radio(
    "Select Data Source",
    ["Use sample synthetic data", "Upload CSV"]
)

if choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()
else:
    df = load_sample_data()

# ------------------------------------
# MAIN APP WORKFLOW
# ------------------------------------
st.header("1️⃣ Data Exploration")
eda_basic(df)

st.header("2️⃣ Prepare Features")
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_features(df)

st.header("3️⃣ Outlier Detection")
if st.button("Run Isolation Forest"):
    run_isolation_forest(X_train_scaled, X_test_scaled, y_test)

st.header("4️⃣ Train ML Models")
model_option = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

if st.button("Train Model"):
    if model_option == "Logistic Regression":
        model, y_prob = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        scaled = True
    elif model_option == "Random Forest":
        model, y_prob = train_random_forest(X_train, X_test, y_train, y_test)
        scaled = False
    else:
        model, y_prob = train_xgboost(X_train, X_test, y_train, y_test)
        scaled = False

    st.session_state["model"] = model
    st.session_state["scaled"] = scaled
    st.success("Model trained successfully!")

st.header("5️⃣ Real-Time Transaction Prediction")

if "model" in st.session_state:
    idx = st.number_input("Pick sample index", 0, len(X_test) - 1, 0)
    sample_dict = pd.DataFrame(X_test).iloc[idx].to_dict()

    if st.button("Predict Fraud Probability"):
        prob = predict_single(
            model=st.session_state["model"],
            scaler=scaler,
            sample_dict=sample_dict,
            feature_columns=list(df.drop(columns=["Class"]).columns),
            scaled=st.session_state["scaled"]
        )
        st.write("Fraud Probability:", prob)
        if prob > 0.5:
            st.error("⚠️ Fraudulent Transaction")
        else:
            st.success("✅ Normal Transaction")
