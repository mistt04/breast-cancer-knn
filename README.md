# breast-cancer-knn
# Breast Cancer Prediction with K-Nearest Neighbors (KNN)

This project uses the **Wisconsin Diagnostic Breast Cancer** dataset and the **K-Nearest Neighbors** algorithm to predict whether a breast tumor is **malignant (cancerous)** or **benign** based on biopsy-derived cell-nucleus features.

---

## Project Overview
- **Dataset:** [Wisconsin Diagnostic Breast Cancer (WDBC)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
  - 569 samples × 30 numeric features describing cell nuclei (e.g., `radius_mean`, `texture_mean`, etc.)
  - Labels: `M` → malignant, `B` → benign
- **Goal:** Build a supervised ML model that predicts tumor type from the numeric features.
- **Algorithm:** K-Nearest Neighbors classifier (KNN) implemented with **scikit-learn**.
- **Language / Libraries:** Python, NumPy, Pandas, Matplotlib/Seaborn, scikit-learn.

---

## Workflow
1. **Data Cleaning**
   - Dropped unused columns (`id`, `Unnamed: 32`)
   - Encoded `diagnosis` as 1 = malignant, 0 = benign
2. **Exploratory Data Analysis**
   - Used `sns.lmplot`, scatterplots, and correlation heatmaps to visualize feature separation.
3. **Model Preparation**
   - Split data into `X` (features) and `y` (labels)
   - Train/Test split (67% / 33%) with `random_state=42`
   - Scaled features using `StandardScaler` for fair distance calculations
4. **Model Training**
   - Trained `KNeighborsClassifier` with various values of **k**
   - Chose **k = 13** based on misclassification-error curve
5. **Evaluation**
   - Reported **accuracy**, confusion matrix, ROC-AUC
   - Visualized error vs. neighbors plot to pick optimal k

---

## Results
- **Optimal k:** 13 neighbors
- Achieved high accuracy (~94-96% on test set, may vary per split)
- Malignant tumors tend to cluster with **higher radius_mean** and **higher texture_mean**

---

## How to Run Locally
```bash
# Clone this repository
git clone https://github.com/<your-username>/breast-cancer-knn.git
cd breast-cancer-knn

# Create & activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the script
python knn_cancer.py
