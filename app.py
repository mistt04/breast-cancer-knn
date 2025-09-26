# app.py
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Page config
st.set_page_config(
    page_title="Breast Cancer KNN",
    layout="wide"
)


# Utilities

@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # clean columns
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis=1)
    # encode diagnosis to 0/1
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

@st.cache_resource
def train_model(df: pd.DataFrame, k: int, test_size: float, seed: int):
    X = df.iloc[:, 1:].copy()
    y = df["diagnosis"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)

    y_pred = knn.predict(X_test_s)
    y_prob = knn.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred, labels=[0, 1]),  # [[TN, FP],[FN, TP]]
        "X_cols": list(X.columns),
        "scaler": scaler,
        "model": knn
    }
    return metrics

def cm_to_dataframe(cm):
    # cm = [[TN, FP], [FN, TP]]
    df_cm = pd.DataFrame(
        cm,
        index=pd.Index(["Benign (0)", "Malignant (1)"], name="True"),
        columns=pd.Index(["Benign (0)", "Malignant (1)"], name="Predicted"),
    )
    return df_cm


# Sidebar controls

st.sidebar.title("Settings")
csv_path = st.sidebar.text_input(
    "CSV path",
    value="Breast Cancer Wisconsin Data.csv",
    help="Path to the Kaggle CSV in your repo."
)
k = st.sidebar.slider("Neighbors (k)", min_value=1, max_value=25, value=13, step=2)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.33, step=0.01)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

df = load_data(csv_path)


# Header + metrics

st.title("Breast Cancer Predictor (KNN)")
st.caption("Wisconsin Diagnostic Breast Cancer dataset • K-Nearest Neighbors classifier")

metrics = train_model(df, k=k, test_size=test_size, seed=seed)
acc = metrics["accuracy"] * 100
auc = metrics["roc_auc"] * 100

m1, m2, m3 = st.columns(3)
m1.metric("Test Accuracy", f"{acc:.2f}%")
m2.metric("ROC-AUC", f"{auc:.2f}%")
m3.metric("Neighbors (k)", k)

with st.expander("Confusion Matrix"):
    st.dataframe(cm_to_dataframe(metrics["cm"]))


# Inputs (two modes): Preset row OR manual

st.subheader("Make a Prediction")

X_cols = metrics["X_cols"]
X_full = df.iloc[:, 1:].copy()  # all numeric features
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Choose a preset (auto-fills from a real row)**")
    idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(df)-1, value=0, step=1)
    preset_vals = X_full.iloc[idx].to_dict()
    if st.button("Use Preset Row"):
        st.session_state["inputs"] = preset_vals

with col_right:
    st.markdown("**Or enter values manually**")

# Prepare default state
if "inputs" not in st.session_state:
    # sensible defaults = feature means
    st.session_state["inputs"] = {col: float(X_full[col].mean()) for col in X_cols}

# Build inputs grid
user_vals = []
n_cols = 3
rows = (len(X_cols) + n_cols - 1) // n_cols
for r in range(rows):
    cols = st.columns(n_cols)
    for c in range(n_cols):
        i = r * n_cols + c
        if i >= len(X_cols):
            continue
        colname = X_cols[i]
        col_min = float(X_full[colname].min())
        col_max = float(X_full[colname].max())
        col_mean = float(X_full[colname].mean())
        default = st.session_state["inputs"].get(colname, col_mean)
        with cols[c]:
            val = st.number_input(
                colname,
                value=float(round(default, 3)),
                min_value=float(round(col_min, 3)),
                max_value=float(round(col_max, 3)),
                step=0.001,
                format="%.3f",
            )
            st.session_state["inputs"][colname] = val
            user_vals.append(val)


# Predict

predict_col, explain_col = st.columns([1, 1])

with predict_col:
    if st.button("Predict"):
        scaler = metrics["scaler"]
        model = metrics["model"]
        X_user = scaler.transform([user_vals])
        pred = int(model.predict(X_user)[0])
        proba = float(model.predict_proba(X_user)[0][pred])
        label = "Malignant (1)" if pred == 1 else "Benign (0)"
        st.success(f"Prediction: **{label}**")
        st.caption(f"Approx. confidence: **{proba*100:.1f}%**")

with explain_col:
    st.markdown("**What affects this prediction?**")
    st.write(
        "KNN has no built-in feature importance. For insight, try toggling inputs and watch how the prediction changes. "
        "You can also add permutation importance or SHAP in a future version."
    )


# Data peek

with st.expander("Peek at the dataset"):
    st.write(df.head())
    st.write(df.describe().T)
