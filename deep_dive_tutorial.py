"""
Deep-Dive Tutorial: 50 Startups Multiple Linear Regression
Covers:
1. Preprocessing: One-Hot Encoding (Nominal), train_test_split necessity, astype & reshape
2. Load Data + Random Data Scatter Plot
3. Build Model (Pipeline Architecture)
4. Evaluation: MAE, MSE, RMSE, R² interpretations
5. Overfitting vs. Underfitting diagnosis (Learning Curve & Complexity Curve)
6. Hyperparameter Optimization with Optuna (ElasticNet / Ridge / Lasso)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import optuna
# Suppress verbose Optuna logging output
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("DEEP-DIVE: PREPROCESSING, EVALUATION, OVERFIT/UNDERFIT, AND OPTUNA")
print("=" * 80)

# ==============================================================================
# SECTION 1: PREPROCESSING DEEP-DIVE
# ==============================================================================
print("\n" + "=" * 50)
print("SECTION 1: PREPROCESSING CONCEPTS")
print("=" * 50)

# 1.1 One-Hot Encoding for Nominal Features vs Ordinal Features
print("""
[1.1 One-Hot Encoding vs Label Encoding for Nominal Features]
- Nominal (名義變數): Categories without inherent mathematical ranking (e.g. State: 'California', 'Florida', 'New York').
  * If we used Label Encoding (0, 1, 2), a linear model assumes:
      Profit = Beta * State = Beta * 2 (for New York), implying New York has twice the weight of Florida.
  * One-Hot Encoding creates binary indicator columns [0, 1] for each category.
- Dummy Variable Trap (虛擬變數陷阱):
  * If a category has k levels, sum of all k dummy columns equals 1 (a constant).
  * This is collinear with the intercept term (Beta_0 * 1), causing a singular matrix (X^T X)^{-1}.
  * Solution: drop='first' (drop 1 category as reference base, leaving k - 1 dummies).
""")

# 1.2 Train-Test Split 的必要性
print("""
[1.2 train_test_split 的必要性]
1. Generalization Evaluation (衡量泛化能力):
   A model that achieves 100% training accuracy might simply memorize noise. 
   Testing on unseen hold-out data verifies real-world predictive validity.
2. Preventing Data Leakage (防止資料外洩):
   Feature transformations (e.g. StandardScaler mean & std, OneHotEncoder categories)
   MUST be fit only on the training set, then applied to the test set via transform().
   Fitting on the entire dataset leaks target/distribution information into validation.
3. Diagnostic Baseline for Bias-Variance (辨別 Overfit / Underfit):
   Comparing train score vs test score is the primary method to detect variance/bias.
""")

# 1.3 Change Type and Reshape
print("""
[1.3 Change Type (astype) & Reshape (.reshape(-1, 1))]
- Type Casting: Ensures numeric columns are float64/int64 rather than object/string,
  e.g., df['R&D Spend'] = df['R&D Spend'].astype(np.float64)
- Reshaping: Scikit-learn estimators expect a 2D matrix for X with shape (n_samples, n_features).
  * A 1D pandas Series or numpy array has shape (n_samples,).
  * Calling array.reshape(-1, 1) converts shape (50,) -> (50, 1), creating a valid 2D feature matrix.
""")

# Code demo of Reshape
sample_series = pd.Series([100.0, 200.0, 300.0], name="spend")
print(f"Original Series shape: {sample_series.values.shape}")
reshaped_matrix = sample_series.values.reshape(-1, 1)
print(f"Reshaped 2D matrix shape: {reshaped_matrix.shape}")

# ==============================================================================
# SECTION 2: LOAD DATA + GENERATE RANDOM DATA SCATTER PLOT
# ==============================================================================
print("\n" + "=" * 50)
print("SECTION 2: LOAD DATA + SCATTER PLOTS")
print("=" * 50)

csv_path = os.path.join(PROJECT_DIR, "50_Startups.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded existing dataset from {csv_path}")
else:
    # Fallback synthetic generation matching 50 Startups characteristics
    print("Generating synthetic 50 Startups dataset...")
    np.random.seed(42)
    n = 50
    rd = np.random.uniform(10000, 170000, n)
    admin = np.random.uniform(50000, 180000, n)
    mkt = np.random.uniform(50000, 450000, n)
    states = np.random.choice(['New York', 'California', 'Florida'], n)
    noise = np.random.normal(0, 8000, n)
    profit = 49000 + 0.85 * rd - 0.05 * admin + 0.03 * mkt + noise
    df = pd.DataFrame({
        'R&D Spend': rd,
        'Administration': admin,
        'Marketing Spend': mkt,
        'State': states,
        'Profit': profit
    })

# Ensure correct datatypes
df['R&D Spend'] = df['R&D Spend'].astype(float)
df['Administration'] = df['Administration'].astype(float)
df['Marketing Spend'] = df['Marketing Spend'].astype(float)
df['Profit'] = df['Profit'].astype(float)
df['State'] = df['State'].astype(str)

# Generate Random Synthetic Comparison Data to illustrate relationship
np.random.seed(42)
random_rd = np.random.uniform(df['R&D Spend'].min(), df['R&D Spend'].max(), 30)
# Synthetic baseline profit
random_profit = 50000 + 0.82 * random_rd + np.random.normal(0, 12000, 30)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual 50 Startups Data with Regression Trendline
sns.regplot(
    data=df, 
    x='R&D Spend', 
    y='Profit', 
    ax=axes[0], 
    color='darkblue', 
    scatter_kws={'alpha': 0.7, 's': 50}, 
    line_kws={'color': 'red', 'linewidth': 2}
)
axes[0].set_title("50 Startups: R&D Spend vs. Profit (Actual)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("R&D Spend ($)")
axes[0].set_ylabel("Profit ($)")

# Plot 2: Random / Synthetic Perturbation Scatter Plot
axes[1].scatter(random_rd, random_profit, color='forestgreen', s=60, label='Generated Synthetic Points', alpha=0.8)
z = np.polyfit(random_rd, random_profit, 1)
p = np.poly1d(z)
axes[1].plot(random_rd, p(random_rd), "r--", linewidth=2, label=f"Fit: y = {z[0]:.2f}x + {z[1]:,.0f}")
axes[1].set_title("Synthetic Random Data Generator: Trend Scatter", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Random R&D Spend ($)")
axes[1].set_ylabel("Simulated Profit ($)")
axes[1].legend()

plt.tight_layout()
scatter_plot_path = os.path.join(PROJECT_DIR, "random_scatter_plot.png")
plt.savefig(scatter_plot_path, dpi=200)
plt.close()
print(f"[Generated Scatter Plot]: {scatter_plot_path}")

# ==============================================================================
# SECTION 3: BUILD MODEL (PREPROCESSING + REGRESSION PIPELINE)
# ==============================================================================
print("\n" + "=" * 50)
print("SECTION 3: BUILD MODEL")
print("=" * 50)

X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = df['Profit']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Build Preprocessor:
# - Passthrough numeric features
# - OneHotEncode nominal 'State', dropping first level ('California') to prevent dummy variable trap
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['R&D Spend', 'Administration', 'Marketing Spend']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['State'])
    ]
)

# Standard Linear Regression Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
print("Trained Scikit-learn Linear Regression Pipeline successfully.")

# ==============================================================================
# SECTION 4: EVALUATION METRICS (MSE, MAE, R²)
# ==============================================================================
print("\n" + "=" * 50)
print("SECTION 4: EVALUATION METRICS & INTERPRETATION")
print("=" * 50)

y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"Test Set Evaluation Results:")
print(f"  - MAE  (Mean Absolute Error):       ${mae:,.2f}")
print(f"  - MSE  (Mean Squared Error):        ${mse:,.2f}")
print(f"  - RMSE (Root Mean Squared Error):   ${rmse:,.2f}")
print(f"  - R²   (Coefficient of Determ.):    {r2:.4f}")

print("""
[Mathematical Meanings]:
1. MAE (Mean Absolute Error) = (1/n) * SUM |y_i - y_hat_i|
   - Meaning: Average magnitude of the errors in dollar units ($).
   - Robustness: Linear penalty, resilient to severe outliers.
2. MSE (Mean Squared Error) = (1/n) * SUM (y_i - y_hat_i)^2
   - Meaning: Average squared deviation in ($)^2.
   - Robustness: Heavily penalizes large errors. Essential for mathematical optimization (gradient descent).
3. RMSE (Root Mean Squared Error) = SQRT(MSE)
   - Meaning: Restores MSE back to the original dollar unit ($) while preserving sensitivity to large errors.
4. R² (Coefficient of Determination) = 1 - (SS_res / SS_tot) = 1 - [SUM(y_i - y_hat_i)^2 / SUM(y_i - y_mean)^2]
   - Meaning: The percentage of total variation in Profit explained by the linear combination of inputs.
   - Range: <= 1.0. R² = 1 is perfect; R² = 0 is no better than predicting the average profit; R² < 0 means worse than the mean.
""")

# ==============================================================================
# SECTION 5: OVERFIT & UNDERFIT DIAGNOSIS (CURVE PLOTTING)
# ==============================================================================
print("\n" + "=" * 50)
print("SECTION 5: OVERFIT & UNDERFIT DIAGNOSIS")
print("=" * 50)

# 1. Learning Curve (Increasing Training Size)
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.2, 1.0, 6),
    scoring='r2',
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# 2. Complexity Curve (Varying Polynomial Degree to demonstrate Underfit vs Overfit)
# Using single feature (R&D Spend) to clearly show Underfitting vs Ideal vs Overfitting
degrees = [1, 2, 4, 10]
poly_train_r2 = []
poly_test_r2 = []

X_rd_train = X_train[['R&D Spend']]
X_rd_test = X_test[['R&D Spend']]

for deg in degrees:
    poly_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('reg', LinearRegression())
    ])
    poly_pipe.fit(X_rd_train, y_train)
    poly_train_r2.append(r2_score(y_train, poly_pipe.predict(X_rd_train)))
    poly_test_r2.append(r2_score(y_test, poly_pipe.predict(X_rd_test)))

# Generate Curves Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot A: Learning Curve (Training samples vs Score)
axes[0].plot(train_sizes, train_mean, 'o-', color='navy', label='Training R² Score')
axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='navy')
axes[0].plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation R² Score (5-Fold CV)')
axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='darkorange')
axes[0].set_title("Learning Curve: Training vs. Validation R²", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Number of Training Samples")
axes[0].set_ylabel("R² Score")
axes[0].set_ylim(0.7, 1.02)
axes[0].legend(loc='lower right')

# Plot B: Model Complexity Curve (Overfitting Demonstration)
axes[1].plot(degrees, poly_train_r2, 'bo-', label='Train R² (Memorization)', linewidth=2)
axes[1].plot(degrees, poly_test_r2, 'ro--', label='Test R² (Generalization)', linewidth=2)
axes[1].axvline(1, color='green', linestyle=':', label='Optimal Complexity (Degree=1)')
axes[1].set_title("Complexity Curve: Identifying Overfitting", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Polynomial Degree (Model Complexity)")
axes[1].set_ylabel("R² Score")
axes[1].set_ylim(-0.5, 1.05)
axes[1].legend()

plt.tight_layout()
curve_plot_path = os.path.join(PROJECT_DIR, "overfit_underfit_curves.png")
plt.savefig(curve_plot_path, dpi=200)
plt.close()
print(f"[Generated Learning & Complexity Curves]: {curve_plot_path}")

print("""
[Overfit vs. Underfit Diagnosis Criteria]:
- Underfitting (高偏差 / High Bias):
  * Train Score: Low
  * Test Score: Low
  * The model is too simple to capture the underlying pattern.
- Overfitting (高變異 / High Variance):
  * Train Score: Very High (approaching 1.0)
  * Test Score: Poor / Negative (Generalization gap is wide)
  * As seen in Degree=10, the polynomial fits training noise, causing catastrophic test error!
- Good Fit (適度擬合):
  * Train Score is high (~0.95), Test Score is high (~0.90), and both converge smoothly.
""")

# ==============================================================================
# SECTION 6: HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ==============================================================================
print("\n" + "=" * 50)
print("SECTION 6: MODEL OPTIMIZATION WITH OPTUNA")
print("=" * 50)

# Preprocessor with StandardScaler for regularized regression
prep_scaled = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['R&D Spend', 'Administration', 'Marketing Spend']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['State'])
    ]
)

def objective(trial):
    # Suggest model type and hyperparameters
    model_type = trial.suggest_categorical('model_type', ['Ridge', 'Lasso', 'ElasticNet'])
    
    if model_type == 'Ridge':
        alpha = trial.suggest_float('ridge_alpha', 1e-3, 100.0, log=True)
        reg = Ridge(alpha=alpha, random_state=42)
    elif model_type == 'Lasso':
        alpha = trial.suggest_float('lasso_alpha', 1e-1, 500.0, log=True)
        reg = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    else:
        alpha = trial.suggest_float('elastic_alpha', 1e-3, 100.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.05, 0.95)
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
    
    opt_pipeline = Pipeline([
        ('prep', prep_scaled),
        ('reg', reg)
    ])
    
    # 5-fold cross validation score (optimizing negative RMSE / maximizing R²)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(opt_pipeline, X_train, y_train, cv=cv, scoring='r2')
    return scores.mean()

# Create Optuna study to MAXIMIZE mean cross-validation R²
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Optuna Trials Completed: {len(study.trials)}")
print(f"Best Trial #{study.best_trial.number}:")
print(f"  Best Cross-Validation R²: {study.best_value:.4f}")
print("  Best Parameters:")
for k, v in study.best_params.items():
    print(f"    - {k}: {v}")

# Fit best model on full training set and evaluate on test set
best_params = study.best_params
best_type = best_params['model_type']
if best_type == 'Ridge':
    best_reg = Ridge(alpha=best_params['ridge_alpha'], random_state=42)
elif best_type == 'Lasso':
    best_reg = Lasso(alpha=best_params['lasso_alpha'], random_state=42, max_iter=10000)
else:
    best_reg = ElasticNet(alpha=best_params['elastic_alpha'], l1_ratio=best_params['l1_ratio'], random_state=42)

best_pipeline = Pipeline([
    ('prep', prep_scaled),
    ('reg', best_reg)
])
best_pipeline.fit(X_train, y_train)

best_test_r2 = r2_score(y_test, best_pipeline.predict(X_test))
best_test_rmse = np.sqrt(mean_squared_error(y_test, best_pipeline.predict(X_test)))

print(f"\nFinal Optuna Optimized Model Test Evaluation:")
print(f"  - Model Selected: {best_type}")
print(f"  - Test Set R²:    {best_test_r2:.4f}")
print(f"  - Test Set RMSE:  ${best_test_rmse:,.2f}")

print("\n" + "=" * 80)
print("DEEP-DIVE TUTORIAL RUN COMPLETED SUCCESSFULLY")
print("=" * 80)
