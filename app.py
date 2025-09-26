# app.py (simplified, calmer UI)
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Breast Cancer KNN", page_icon="🧬", layout="wide")

# --------- Data / Model helpers ---------
@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    for c in ["id", "Unnamed: 32"]:
        if c in df.columns:
            df = df.drop(c, axis=1)
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

@st.cache_resource
def train(df: pd.DataFrame, k: int, test_size: float, seed: int):
    X, y = df.iloc[:, 1:].copy(), df["diagnosis"].copy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=k).fit(scaler.transform(Xtr), ytr)

    ypred = knn.predict(scaler.transform(Xte))
    yprob = knn.predict_proba(scaler.transform(Xte))[:, 1]

    metrics = dict(
        accuracy=accuracy_score(yte, ypred),
        roc_auc=roc_auc_score(yte, yprob),
        cm=confusion_matrix(yte, ypred, labels=[0, 1]),
    )
    return knn, scaler, X.columns.tolist(), metrics, (Xtr, Xte, ytr, yte)

def cm_df(cm):
    return pd.DataFrame(cm,
        index=pd.Index(["Benign (0)", "Malignant (1)"], name="True"),
        columns=pd.Index(["Benign (0)", "Malignant (1)"], name="Predicted"),
    )

def top_features(df: pd.DataFrame, n: int = 6):
    """Pick n most correlated features with diagnosis (absolute Pearson)."""
    corr = df.corr(numeric_only=True)["diagnosis"].drop("diagnosis").abs().sort_values(ascending=False)
    return corr.index[:n].tolist(), corr

# --------- Sidebar (minimal) ---------
st.sidebar.title("⚙️ Settings")
csv_path = st.sidebar.text_input("CSV path", "Breast Cancer Wisconsin Data.csv")
k = st.sidebar.slider("Neighbors (k)", 1, 25, 13, step=2)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.33, step=0.01)
seed = st.sidebar.number_input("Random seed", 0, value=42, step=1)

# --------- Load + Train ---------
df = load_data(csv_path)
model, scaler, feature_names, metrics, splits = train(df, k, test_size, seed)

# --------- Header metrics ---------
st.title("🧬 Breast Cancer Predictor (KNN)")
c1, c2, c3 = st.columns(3)
c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
c2.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")
c3.metric("Neighbors (k)", k)

with st.expander("Confusion Matrix", expanded=False):
    st.dataframe(cm_df(metrics["cm"]))

# --------- Tabs ---------
tab_predict, tab_model, tab_data = st.tabs(["🔮 Predict", "🧠 Model", "📄 Data"])

# ============ PREDICT TAB ============
with tab_predict:
    st.subheader("Make a Prediction")

    X = df.iloc[:, 1:].copy()
    top_cols, corr_series = top_features(df, n=6)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Preset from dataset**")
        row_idx = st.number_input("Row index (0-based)", 0, len(df)-1, 0)
        if st.button("Use Preset Row", key="preset"):
            st.session_state["inputs"] = X.iloc[row_idx].to_dict()

    with right:
        st.markdown("**Feature selection**")
        use_top = st.toggle("Show only top features first (recommended)", value=True)

    # initialize session state
    if "inputs" not in st.session_state:
        st.session_state["inputs"] = {c: float(X[c].mean()) for c in feature_names}

    # input form (prevents UI thrash)
    with st.form("predict_form"):
        st.caption("Adjust the sliders, then press **Predict**.")
        chosen_cols = top_cols if use_top else feature_names

        # show chosen features first
        cols_per_row = 3
        for i, col in enumerate(chosen_cols):
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row)
            j = i % cols_per_row
            vmin, vmax = float(X[col].min()), float(X[col].max())
            default = float(st.session_state["inputs"].get(col, X[col].mean()))
            st.session_state["inputs"][col] = row[j].slider(
                col, min_value=vmin, max_value=vmax, value=default, step=(vmax-vmin)/200
            )

        # remaining (advanced) features tucked away
        remaining = [c for c in feature_names if c not in chosen_cols]
        if remaining:
            with st.expander("Advanced features (optional)"):
                for i, col in enumerate(remaining):
                    if i % cols_per_row == 0:
                        row = st.columns(cols_per_row)
                    j = i % cols_per_row
                    vmin, vmax = float(X[col].min()), float(X[col].max())
                    default = float(st.session_state["inputs"].get(col, X[col].mean()))
                    st.session_state["inputs"][col] = row[j].slider(
                        col, min_value=vmin, max_value=vmax, value=default, step=(vmax-vmin)/200
                    )

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        vals = [st.session_state["inputs"][c] for c in feature_names]
        X_user = scaler.transform([vals])
        pred = int(model.predict(X_user)[0])
        proba = float(model.predict_proba(X_user)[0][pred])
        label = "Malignant (1)" if pred == 1 else "Benign (0)"
        st.success(f"Prediction: **{label}**  ·  Confidence ~ {proba*100:.1f}%")

        with st.expander("Why only a few sliders up top?"):
            st.write(
                "To avoid overwhelming inputs, the top features (by absolute correlation with the label) are "
                "shown first. Toggle off the switch to expose all 30, or open **Advanced features**."
            )

# ============ MODEL TAB ============
with tab_model:
    st.subheader("How the model is trained")
    st.write(
        "- Train/test split with stratification\n"
        "- Standardization (`StandardScaler`) so KNN distances are fair\n"
        f"- KNN with **k = {k}**\n"
        "- Metrics: Accuracy and ROC-AUC on the held-out test set"
    )
    st.markdown("**Top feature correlations with diagnosis**")
    st.dataframe(corr_series.to_frame("abs_corr").style.background_gradient(cmap="Blues"))

# ============ DATA TAB ============
with tab_data:
    st.subheader("Dataset")
    st.caption("Wisconsin Diagnostic Breast Cancer (WDBC). `diagnosis`: 1=Malignant, 0=Benign.")
    st.dataframe(df.head(20))
    st.caption("Summary (transpose)")
    st.dataframe(df.describe().T)
