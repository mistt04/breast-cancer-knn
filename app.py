import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Add custom CSS to load OpenAI Sans from Google Fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Open Sans', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# ------------------------
# Helpers
# ------------------------
@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    for c in ["id", "Unnamed: 32"]:
        if c in df.columns:
            df = df.drop(c, axis=1)
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # 1=malignant, 0=benign
    return df

@st.cache_resource
def train(df: pd.DataFrame, k: int, test_size: float, seed: int):
    X, y = df.iloc[:, 1:].copy(), df["diagnosis"].copy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=k).fit(scaler.transform(Xtr), ytr)

    # test metrics
    ypred = knn.predict(scaler.transform(Xte))
    acc = accuracy_score(yte, ypred)
    tn, fp, fn, tp = confusion_matrix(yte, ypred, labels=[0, 1]).ravel()
    # plain-English metrics
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # catches cancer cases
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # avoids false alarms

    return knn, scaler, X.columns.tolist(), dict(
        accuracy=acc, sensitivity=sensitivity, specificity=specificity
    )

def group_features(cols):
    """Roughly group features into size / texture / shape using name keywords."""
    size_keys     = ["radius", "perimeter", "area"]
    texture_keys  = ["texture", "smoothness", "symmetry"]
    shape_keys    = ["compactness", "concavity", "concave", "fractal"]

    def pick(keys): return [c for c in cols if any(k in c for k in keys)]
    size = pick(size_keys)
    texture = [c for c in pick(texture_keys) if c not in size]
    shape = [c for c in pick(shape_keys) if c not in size + texture]
    remaining = [c for c in cols if c not in size + texture + shape]
    return size, texture, shape, remaining

FRIENDLY = {
    # Size-related
    "radius_mean": "Average nucleus size (radius) — mean",
    "radius_worst": "Largest nucleus size (radius) — worst",
    "perimeter_mean": "Average perimeter — mean",
    "perimeter_worst": "Largest perimeter — worst",
    "area_mean": "Average cell area — mean",
    "area_worst": "Largest cell area — worst",

    # Texture-related
    "texture_mean": "Image texture (variation) — mean",
    "texture_worst": "Image texture (variation) — worst",
    "smoothness_mean": "Smoothness (small-scale variation) — mean",
    "smoothness_worst": "Smoothness — worst",
    "symmetry_mean": "Symmetry — mean",
    "symmetry_worst": "Symmetry — worst",

    # Shape/irregularity
    "compactness_mean": "Compactness (edge thickness) — mean",
    "compactness_worst": "Compactness — worst",
    "concavity_mean": "Concavity (indentation) — mean",
    "concavity_worst": "Concavity — worst",
    "concave points_mean": "Concave points (count of indents) — mean",
    "concave points_worst": "Concave points — worst",
    "fractal_dimension_mean": "Fractal dimension (edge complexity) — mean",
    "fractal_dimension_worst": "Fractal dimension — worst",
}

HELP = {
    "radius_mean": "Larger average nuclei often associate with malignancy.",
    "texture_mean": "How varied the pixel intensity is; more variation can indicate malignancy.",
    "smoothness_mean": "Higher = smoother edges; lower = bumpier edges.",
    "compactness_mean": "Higher can indicate thicker/tighter edges.",
    "concave points_mean": "More ‘indents’ along the edge can indicate malignancy.",
    "fractal_dimension_mean": "Higher = more complex/irregular edges.",
}

# ------------------------
# Sidebar (simple)
# ------------------------
st.sidebar.title("Settings")
csv_path = st.sidebar.text_input("CSV path", "Breast Cancer Wisconsin Data.csv")
k = st.sidebar.slider("Neighbors (how many examples to compare)", 1, 25, 13, step=2)
test_size = st.sidebar.slider("Test size (kept for checking)", 0.1, 0.4, 0.33, step=0.01)
seed = st.sidebar.number_input("Random seed", 0, value=42, step=1)

# ------------------------
# Load + Train
# ------------------------
df = load_data(csv_path)
model, scaler, feature_names, m = train(df, k, test_size, seed)

# ------------------------
# Header
# ------------------------
st.title("Breast Cancer Predictor")
st.caption("This tool uses patterns from past biopsy measurements to estimate whether a tumor is likely **Benign (0)** or **Malignant (1)**. \
It compares your inputs to similar past cases (K-Nearest Neighbors). It is **not** a medical diagnosis.")

# Friendly metrics
c1, c2, c3 = st.columns(3)
c1.metric("Overall accuracy", f"{m['accuracy']*100:.1f}%")
c2.metric("Catches cancer case (sensitivity)", f"{m['sensitivity']*100:.1f}%")
c3.metric("Avoids false alarms (specificity)", f"{m['specificity']*100:.1f}%")

st.divider()

# ------------------------
# Prediction UI
# ------------------------
st.subheader("Try a prediction")

X = df.iloc[:, 1:].copy()
size_cols, texture_cols, shape_cols, other_cols = group_features(feature_names)

# defaults (means)
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {c: float(X[c].mean()) for c in feature_names}

# Preset row or manual
preset_col, _ = st.columns([1, 2])
with preset_col:
    st.markdown("**Use a real example**")
    row_idx = st.number_input("Pick a row from the dataset (0–568)", 0, len(df)-1, 0)
    if st.button("Fill from this row"):
        st.session_state["inputs"] = X.iloc[row_idx].astype(float).to_dict()

# Form to prevent constant re-run
with st.form("predict_form"):
    st.markdown("**Adjust a few key measurements** (others are optional).")

    def sliders_for(group_name, cols):
        if not cols:
            return
        with st.expander(group_name, expanded=True):
            grid = st.columns(3)
            for i, col in enumerate(cols):
                vmin, vmax = float(X[col].min()), float(X[col].max())
                default = float(st.session_state["inputs"].get(col, X[col].mean()))
                label = FRIENDLY.get(col, col.replace("_", " ").title())
                help_text = HELP.get(col, None)
                st.session_state["inputs"][col] = grid[i % 3].slider(
                    label,
                    min_value=vmin, max_value=vmax, value=default,
                    step=(vmax - vmin) / 200 if vmax > vmin else 0.001,
                    help=help_text
                )

    # Show the most intuitive groups first
    sliders_for("Size (Nucleus size & area)", size_cols[:6])
    sliders_for("Texture (Variation & smoothness of biopsy)", texture_cols[:6])
    with st.expander("Shape / Irregularity (optional)"):
        sliders_for("Edge shape features", shape_cols)
    with st.expander("Advanced features (rarely needed)"):
        sliders_for("Other features", other_cols)

    submitted = st.form_submit_button("Predict")

if submitted:
    vals = [st.session_state["inputs"][c] for c in feature_names]
    X_user = scaler.transform([vals])
    pred = int(model.predict(X_user)[0])
    proba = float(model.predict_proba(X_user)[0][pred])
    label = "Malignant (1)" if pred == 1 else "Benign (0)"

    st.success(f"Prediction: **{label}**")
    st.caption(f"Approximate confidence: **{proba*100:.1f}%** (from similar past cases).")

with st.expander("How it works (quickly)"):
    st.write(
        "- **You enter measurements** from a biopsy image.\n"
        "- The model **finds past cases with similar measurements**.\n"
        "- If most nearby cases were malignant, it predicts **malignant**; otherwise **benign**.\n\n"
        "**Notes:**\n"
        "- This is for learning only and **not medical advice**.\n"
        "- Accuracy shows how often the app is correct overall.\n"
        "- *Catches cancer cases* (sensitivity) is the share of cancer cases it correctly flags.\n"
        "- *Avoids false alarms* (specificity) is the share of benign cases it correctly clears."
    )
