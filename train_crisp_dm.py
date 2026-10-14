"""
50 Startups - Multiple Linear Regression Pipeline
Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)
Library: scikit-learn, pandas, numpy, matplotlib, seaborn
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print(" CRISP-DM: 50 STARTUPS MULTIPLE LINEAR REGRESSION PIPELINE")
print("=" * 80)

# ==============================================================================
# PHASE 1: BUSINESS UNDERSTANDING (商業理解)
# ==============================================================================
print("\n" + "=" * 50)
print("PHASE 1: BUSINESS UNDERSTANDING")
print("=" * 50)
print("""
[Business Problem]:
A venture capital firm / corporate investment committee wants to evaluate 50 startups
across New York, California, and Florida to determine which expenditure areas 
(R&D Spend, Administration, Marketing Spend) drive Profitability.

[Objectives]:
1. Quantify the marginal return of each dollar spent on R&D, Administration, and Marketing.
2. Determine whether geographic location (State) significantly affects profitability.
3. Build an accurate, interpretable Multiple Linear Regression model (target R² > 0.85).
4. Provide a production-ready scoring pipeline to evaluate future startup investment pitches.
""")

# ==============================================================================
# PHASE 2: DATA UNDERSTANDING (資料理解)
# ==============================================================================
print("=" * 50)
print("PHASE 2: DATA UNDERSTANDING")
print("=" * 50)

data_path = os.path.join(OUTPUT_DIR, "50_Startups.csv")
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 Records:")
print(df.head())

print("\nData Types & Null Counts:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])

print("\nCategorical Feature Distribution ('State'):")
print(df['State'].value_counts())

# Check for zero expenditures (special business note: early-stage bootstrapped / non-marketing startups)
zeros_rd = (df['R&D Spend'] == 0).sum()
zeros_mkt = (df['Marketing Spend'] == 0).sum()
print(f"\nZero spend count -> R&D Spend: {zeros_rd}, Marketing Spend: {zeros_mkt}")

# Correlation analysis for numerical features
num_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
corr_matrix = df[num_cols].corr()
print("\nCorrelation with Profit:")
print(corr_matrix['Profit'].sort_values(ascending=False))

# Generate EDA Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Heatmap of correlations
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".3f", ax=axes[0], cbar=True)
axes[0].set_title("Feature Correlation Heatmap", fontsize=12, fontweight='bold')

# Plot 2: Scatter plot of R&D Spend vs Profit colored by State
sns.scatterplot(
    data=df, 
    x='R&D Spend', 
    y='Profit', 
    hue='State', 
    style='State', 
    s=70, 
    palette='Set1', 
    ax=axes[1]
)
axes[1].set_title("R&D Spend vs. Profit by State", fontsize=12, fontweight='bold')
plt.tight_layout()
eda_plot_path = os.path.join(OUTPUT_DIR, "eda_analysis.png")
plt.savefig(eda_plot_path, dpi=200)
plt.close()
print(f"\n[Saved EDA visual]: {eda_plot_path}")

# ==============================================================================
# PHASE 3: DATA PREPARATION (資料準備)
# ==============================================================================
print("\n" + "=" * 50)
print("PHASE 3: DATA PREPARATION")
print("=" * 50)

# Separate predictors and target
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = df['Profit']

numerical_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Preprocessing:
# 1. Numerical: Keep unscaled for direct dollar-to-dollar coefficient interpretation,
#    or pass through within ColumnTransformer.
# 2. Categorical: OneHotEncoder with drop='first' to avoid the Dummy Variable Trap (Multicollinearity).
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# Fit preprocessor to inspect encoded feature names
preprocessor.fit(X_train)
cat_encoder = preprocessor.named_transformers_['cat']
cat_encoded_names = list(cat_encoder.get_feature_names_out(categorical_features))
feature_names = numerical_features + cat_encoded_names
print("\nPrepared Feature Space (Avoiding Dummy Variable Trap):")
for idx, name in enumerate(feature_names, 1):
    print(f"  {idx}. {name}")

# ==============================================================================
# PHASE 4: MODELING (模型建立)
# ==============================================================================
print("\n" + "=" * 50)
print("PHASE 4: MODELING")
print("=" * 50)

# Build end-to-end Pipeline
mlr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression(fit_intercept=True))
])

# Train Model on Train split
mlr_pipeline.fit(X_train, y_train)

# Perform 5-Fold Cross Validation on the training set
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mlr_pipeline, X_train, y_train, cv=cv, scoring='r2')
print(f"5-Fold Cross-Validation R² Scores: {np.round(cv_scores, 4)}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# Extract Model Parameters (Intercept and Coefficients)
regressor = mlr_pipeline.named_steps['regressor']
intercept = regressor.intercept_
coefficients = regressor.coef_

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient (Beta)': coefficients
})
print("\nTrained Linear Regression Equation Parameters:")
print(f"Intercept (Beta_0): ${intercept:,.2f}")
print(coef_df.to_string(index=False))

# Regularized Models Comparison (Ridge & Lasso)
# Note: For Ridge & Lasso, we use standardized numerical features to ensure fair penalization
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)
ridge_pipeline = Pipeline([('prep', preprocessor_scaled), ('reg', Ridge(alpha=1.0))]).fit(X_train, y_train)
lasso_pipeline = Pipeline([('prep', preprocessor_scaled), ('reg', Lasso(alpha=100.0, max_iter=10000))]).fit(X_train, y_train)

# ==============================================================================
# PHASE 5: EVALUATION (模型評估)
# ==============================================================================
print("\n" + "=" * 50)
print("PHASE 5: EVALUATION")
print("=" * 50)

# Predict on Train and Test sets
y_train_pred = mlr_pipeline.predict(X_train)
y_test_pred = mlr_pipeline.predict(X_test)

# Metrics calculation function
def calc_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'R2': r2, 'Adj_R2': adj_r2, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

p = len(feature_names)
train_metrics = calc_metrics(y_train, y_train_pred, p)
test_metrics = calc_metrics(y_test, y_test_pred, p)

eval_summary = pd.DataFrame([train_metrics, test_metrics], index=['Train Set', 'Test Set'])
print("Model Performance Metrics:")
print(eval_summary.round(4).to_string())

# Regularized model performance on test set
r2_ridge = r2_score(y_test, ridge_pipeline.predict(X_test))
r2_lasso = r2_score(y_test, lasso_pipeline.predict(X_test))
print(f"\nModel Benchmark Comparison (Test R²):")
print(f"  - OLS Multiple Linear Regression: {test_metrics['R2']:.4f}")
print(f"  - Ridge Regression (alpha=1.0):   {r2_ridge:.4f}")
print(f"  - Lasso Regression (alpha=100.0): {r2_lasso:.4f}")

# Residual Analysis & Diagnostics
residuals_test = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Residuals vs Predicted (Homoscedasticity check)
axes[0].scatter(y_test_pred, residuals_test, color='royalblue', edgecolor='k', s=60)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title("Residuals vs. Fitted Values (Homoscedasticity)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Fitted Values (Predicted Profit)")
axes[0].set_ylabel("Residuals ($)")

# 2. Actual vs Predicted
axes[1].scatter(y_test, y_test_pred, color='darkgreen', edgecolor='k', s=60, label='Test Data')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')
axes[1].set_title(f"Actual vs. Predicted Profit (R² = {test_metrics['R2']:.3f})", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Actual Profit ($)")
axes[1].set_ylabel("Predicted Profit ($)")
axes[1].legend()

plt.tight_layout()
diag_plot_path = os.path.join(OUTPUT_DIR, "evaluation_diagnostics.png")
plt.savefig(diag_plot_path, dpi=200)
plt.close()
print(f"\n[Saved Diagnostics visual]: {diag_plot_path}")

# Multicollinearity Check: Variance Inflation Factor (VIF) using scikit-learn
# VIF_j = 1 / (1 - R_j^2), where R_j^2 is obtained by regressing feature j on all other features
def calculate_vif(df_features):
    vif_results = []
    cols = df_features.columns
    for i, col in enumerate(cols):
        X_other = df_features.drop(columns=[col])
        y_target = df_features[col]
        reg = LinearRegression().fit(X_other, y_target)
        r2 = reg.score(X_other, y_target)
        vif = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-10 else float('inf')
        vif_results.append({'Feature': col, 'VIF': vif})
    return pd.DataFrame(vif_results)

X_num = X_train[numerical_features].copy()
vif_data = calculate_vif(X_num)
print("\nVariance Inflation Factor (VIF) for Numerical Features:")
print(vif_data.round(2).to_string(index=False))

# ==============================================================================
# PHASE 6: DEPLOYMENT (模型部署與應用)
# ==============================================================================
print("\n" + "=" * 50)
print("PHASE 6: DEPLOYMENT")
print("=" * 50)

# 1. Serialize Pipeline
model_file = os.path.join(OUTPUT_DIR, "startup_profit_pipeline.joblib")
joblib.dump(mlr_pipeline, model_file)
print(f"[Model Artifact Exported]: {model_file}")

# 2. Load and verify inference function
loaded_pipeline = joblib.load(model_file)

# 3. Simulate New Startup Profiles for Investment Evaluation
new_startups = pd.DataFrame([
    {
        'R&D Spend': 160000.0,
        'Administration': 130000.0,
        'Marketing Spend': 450000.0,
        'State': 'California'
    },
    {
        'R&D Spend': 75000.0,
        'Administration': 110000.0,
        'Marketing Spend': 180000.0,
        'State': 'New York'
    },
    {
        'R&D Spend': 10000.0,
        'Administration': 150000.0,
        'Marketing Spend': 50000.0,
        'State': 'Florida'
    }
])

predictions = loaded_pipeline.predict(new_startups)
new_startups['Predicted Profit'] = predictions

print("\nDeployment Test: Profit Predictions for 3 Candidate Startups:")
print(new_startups[['State', 'R&D Spend', 'Marketing Spend', 'Administration', 'Predicted Profit']].to_string())

print("\n" + "=" * 80)
print(" CRISP-DM PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
