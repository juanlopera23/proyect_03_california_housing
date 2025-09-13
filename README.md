# 🏠 California Housing Regression Project

This project applies **Linear Regression** (and its regularized versions **Ridge** and **Lasso**) to the **California Housing dataset**, following a structured Machine Learning workflow.

---

---

## 🔎 Workflow

### **Block 1: Preparation**
- Created virtual environment and installed dependencies:
  - `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `numpy`.
- Loaded dataset with `fetch_california_housing`.

### **Block 2: Exploratory Data Analysis (EDA)**
- Inspected dataset shape and descriptive statistics.
- Checked missing values and outliers.
- Visualized feature distributions and correlation heatmap.
- Identified strong correlation of **MedInc (median income)** with housing value.

### **Block 3: Preprocessing**
- **Outliers handling**: winsorization + log-transform for `Population`, clipping for `AveRooms`, `AveOccup`, `AveBedrms`.
- **Multicollinearity**: Variance Inflation Factor (VIF) → removed `Longitude`, `Latitude`, `AveRooms`, `AveOccup`.
- **Scaling**: StandardScaler applied inside pipeline.
- **Train/Test split**: 80/20.

### **Block 4: Baseline Model**
- Trained **Linear Regression** with scikit-learn.
- Achieved baseline performance:
  - Train R² ≈ 0.51
  - Test R² ≈ 0.49

### **Block 5: Evaluation**
- Compared Train vs Test → no overfitting.
- Metrics:
  - **MSE**, **RMSE**, **R²**
- Scatter plot of `y_true vs y_pred` → good fit for middle values, underestimation for expensive houses.

### **Block 6: Model Interpretation**
- Analyzed coefficients (standardized):
  - `MedInc`: **0.8284** → strongest predictor.
  - `HouseAge`: smaller positive contribution.
  - `Population` & `AveBedrms`: negligible effect.
- Detected censored target at **5.0** → explains ceiling effect.

### **Block 7: Regularization (Extension)**
- Implemented Ridge and Lasso pipelines.
- Hyperparameter tuning with **cross-validation**.
- Results:
  - Linear Regression (baseline): Test R² ≈ 0.50
  - Ridge: Test R² ≈ 0.58 (best stability, mitigates collinearity).
  - Lasso: Test R² ≈ 0.58 (similar performance, performs feature selection).
- Best alpha values: Ridge ≈ 100, Lasso ≈ 0.1–1.

---

## 📊 Results

| Model              | Train R² | Test R² | Notes                                |
|--------------------|----------|---------|--------------------------------------|
| Linear Regression  | 0.51     | 0.49    | Baseline                             |
| Ridge Regression   | 0.61     | 0.58    | Best stability, handles collinearity |
| Lasso Regression   | 0.61     | 0.58    | Similar to Ridge, does feature selection |

---

## 🚀 Conclusions
- **Median income (MedInc)** is the most important feature for predicting housing value.
- Regularization (Ridge/Lasso) improves generalization compared to baseline linear regression.
- Ridge is more stable under multicollinearity, while Lasso offers simpler models by removing irrelevant features.
- Dataset limitation: capped housing values at 5.0, which reduces performance for expensive areas.

---

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project_03_california_housing.git
   cd project_03_california_housing
