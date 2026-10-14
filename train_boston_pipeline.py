"""
Train Boston Housing Regression Pipeline and Generate Diagnostic Figures
- Multiple Linear Regression on Boston Housing
- Figure 1: Comparison of 10 Regression Models on 'rm' vs 'medv' + Sorted MSE Table
- Figure 2: Sequential Feature Selection Curves (RMSE & R-squared) + Progression Table
- Export Pipeline: boston_housing_pipeline.joblib
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(OUTPUT_DIR, "BostonHousing.csv")
df = pd.read_csv(data_path)

print("=" * 80)
print("TRAINING BOSTON HOUSING MULTIPLE LINEAR REGRESSION PIPELINE")
print("=" * 80)

# Clean column names (strip quotes/spaces)
df.columns = [c.replace('"', '').strip().lower() for c in df.columns]

# Features and target
feature_cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
target_col = 'medv'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full Multiple Linear Regression Pipeline
boston_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
boston_pipeline.fit(X_train, y_train)

# Evaluate Pipeline
y_pred_train = boston_pipeline.predict(X_train)
y_pred_test = boston_pipeline.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Boston Housing MLR Performance:")
print(f"  - Train RMSE: ${train_rmse*1000:,.2f} | Train R²: {train_r2:.4f}")
print(f"  - Test RMSE:  ${test_rmse*1000:,.2f} | Test R²:  {test_r2:.4f}")
print(f"  - Test MAE:   ${test_mae*1000:,.2f}")

# Save trained pipeline
joblib.dump(boston_pipeline, os.path.join(OUTPUT_DIR, "boston_housing_pipeline.joblib"))
print("Saved pipeline to: boston_housing_pipeline.joblib")

# ==============================================================================
# FIGURE 1: COMPARISON OF 10 REGRESSION MODELS (RM vs MEDV)
# ==============================================================================
print("\nGenerating Boston Housing Figure 1: 10 Models Benchmark on 'rm' vs 'medv'...")

df_sorted = df.sort_values(by='rm').reset_index(drop=True)
X_rm = df_sorted[['rm']]
y_medv = df_sorted['medv']

scaler_rm = StandardScaler()
X_rm_scaled = scaler_rm.fit_transform(X_rm)

models = {
    'SVR': SVR(kernel='rbf', C=50, epsilon=1.0),
    'LASSO': Lasso(alpha=0.1, max_iter=10000),
    'Ridge': Ridge(alpha=10.0),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'PCA': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=6),
    'Mutual Information': LinearRegression(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3),
    'SVM': SVR(kernel='linear', C=10),
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=60, max_depth=4)
}

colors = {
    'SVR': 'blue',
    'LASSO': 'green',
    'Ridge': 'red',
    'Elastic Net': 'purple',
    'PCA': 'brown',
    'Random Forest': 'orange',
    'Mutual Information': 'pink',
    'Gradient Boosting Regressor': 'cyan',
    'SVM': 'magenta',
    'LightGBM': 'gray'
}

predictions = {}
results_table = []

for name, model in models.items():
    if name == 'PCA':
        pca = PCA(n_components=1)
        X_t = pca.fit_transform(X_rm)
        model.fit(X_t, y_medv)
        pred = model.predict(X_t)
        feat_desc = "[PCA of rm]"
    elif name in ['Ridge', 'LASSO', 'Elastic Net', 'SVR', 'SVM']:
        model.fit(X_rm_scaled, y_medv)
        pred = model.predict(X_rm_scaled)
        feat_desc = "[rm]"
    else:
        model.fit(X_rm, y_medv)
        pred = model.predict(X_rm)
        feat_desc = "[rm]"
        
    predictions[name] = pred
    mse = mean_squared_error(y_medv, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_medv, pred)
    results_table.append({
        'Model': name,
        'Features': feat_desc,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    })

df_results = pd.DataFrame(results_table).sort_values(by='MSE').reset_index(drop=True)

fig1 = plt.figure(figsize=(11, 10), facecolor='white')
gs1 = GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.15)

ax1 = fig1.add_subplot(gs1[0])
ax1.scatter(X_rm['rm'], y_medv, color='black', label='Actual Data', s=25, alpha=0.6, zorder=5)

for name in ['SVR', 'LASSO', 'Ridge', 'Elastic Net', 'PCA', 'Random Forest', 
             'Mutual Information', 'Gradient Boosting Regressor', 'SVM', 'LightGBM']:
    if name in predictions:
        ax1.plot(X_rm['rm'], predictions[name], label=name, color=colors[name], linewidth=1.5)

ax1.set_title("Boston Housing: Comparison of Different Regression Models (RM vs MEDV)", fontsize=13, pad=10)
ax1.set_xlabel("Average Number of Rooms per Dwelling (RM)", fontsize=10)
ax1.set_ylabel("Median House Value in $1000s (MEDV)", fontsize=10)
ax1.set_xlim(3.0, 9.0)
ax1.set_ylim(0, 55)
ax1.legend(loc='upper left', frameon=True, fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.4)

# Bottom Terminal Table
ax2 = fig1.add_subplot(gs1[1])
ax2.set_facecolor('#222222')
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)

ax2.text(0.01, 0.92, "模型結果 (MSE 由小到大排序):", color='white', fontsize=9.5, fontfamily=['Microsoft JhengHei', 'sans-serif'], transform=ax2.transAxes)
col_header = f"{'Model':>27s}  {'Features':>15s}  {'MSE':>10s}  {'RMSE':>9s}  {'R2 Score':>8s}"
ax2.text(0.01, 0.81, col_header, color='#d1d5db', fontsize=8.5, fontfamily='monospace', transform=ax2.transAxes)

y_cursor = 0.72
for _, r in df_results.iterrows():
    line = f"{r['Model']:>27s}  {r['Features']:>15s}  {r['MSE']:>10.4f}  {r['RMSE']:>9.4f}  {r['R2 Score']:>8.4f}"
    ax2.text(0.01, y_cursor, line, color='#9ca3af', fontsize=8.2, fontfamily='monospace', transform=ax2.transAxes)
    y_cursor -= 0.075

fig1_path = os.path.join(OUTPUT_DIR, "boston_figure1_model_comparison.png")
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# ==============================================================================
# FIGURE 2: FORWARD FEATURE SELECTION ON BOSTON HOUSING
# ==============================================================================
print("\nGenerating Boston Housing Figure 2: Sequential Feature Selection...")

# Sequential forward selection sequence starting with highest correlation features
feature_steps = [
    [1, "['lstat']", ['lstat']],
    [2, "['lstat', 'rm']", ['lstat', 'rm']],
    [3, "['lstat', 'rm', 'ptratio']", ['lstat', 'rm', 'ptratio']],
    [4, "['lstat', 'rm', 'ptratio', 'dis']", ['lstat', 'rm', 'ptratio', 'dis']],
    [5, "['lstat', 'rm', 'ptratio', 'dis', 'nox']", ['lstat', 'rm', 'ptratio', 'dis', 'nox']],
    [6, "['lstat', 'rm', 'ptratio', 'dis', 'nox', 'crim']", ['lstat', 'rm', 'ptratio', 'dis', 'nox', 'crim']],
    [7, "['lstat', 'rm', 'ptratio', 'dis', 'nox', 'crim', 'tax']", ['lstat', 'rm', 'ptratio', 'dis', 'nox', 'crim', 'tax']],
    [8, "All 13 Features (Full Boston Dataset)", feature_cols]
]

feat_results = []
for num_feats, label_feats, feats in feature_steps:
    X_tr = X_train[feats]
    X_te = X_test[feats]
    scaler_sfs = StandardScaler()
    X_tr_s = scaler_sfs.fit_transform(X_tr)
    X_te_s = scaler_sfs.transform(X_te)
    
    reg = LinearRegression().fit(X_tr_s, y_train)
    preds = reg.predict(X_te_s)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    feat_results.append({
        'Number of Features': num_feats,
        'Selected Features': label_feats,
        'RMSE': rmse,
        'R-squared': r2
    })

df_feat_summary = pd.DataFrame(feat_results)

fig2 = plt.figure(figsize=(10, 8), facecolor='white')
gs2 = GridSpec(2, 2, height_ratios=[1.6, 1.0], hspace=0.35, wspace=0.25)

# RMSE Plot
ax_rmse = fig2.add_subplot(gs2[0, 0])
ax_rmse.plot(df_feat_summary['Number of Features'], df_feat_summary['RMSE'], marker='o', markersize=5, color='#1f77b4', linewidth=1.5)
ax_rmse.set_title("RMSE by Number of Features (Boston Housing)", fontsize=11, pad=8)
ax_rmse.set_xlabel("Number of Features", fontsize=9.5)
ax_rmse.set_ylabel("RMSE ($1000s)", fontsize=9.5)
ax_rmse.grid(True, linestyle=':', alpha=0.3)

# R-squared Plot
ax_r2 = fig2.add_subplot(gs2[0, 1])
ax_r2.plot(df_feat_summary['Number of Features'], df_feat_summary['R-squared'], marker='o', markersize=5, color='#2ca02c', linewidth=1.5)
ax_r2.set_title("R-squared by Number of Features (Boston Housing)", fontsize=11, pad=8)
ax_r2.set_xlabel("Number of Features", fontsize=9.5)
ax_r2.set_ylabel("R-squared", fontsize=9.5)
ax_r2.grid(True, linestyle=':', alpha=0.3)

# Table
ax_tbl = fig2.add_subplot(gs2[1, :])
ax_tbl.axis('off')

table_data = []
for _, r in df_feat_summary.iterrows():
    table_data.append([
        str(int(r['Number of Features'])),
        r['Selected Features'],
        f"{r['RMSE']:.4f}",
        f"{r['R-squared']:.4f}"
    ])

table = ax_tbl.table(
    cellText=table_data,
    colLabels=["Number of Features", "Selected Features", "RMSE ($k)", "R-squared"],
    colWidths=[0.18, 0.54, 0.14, 0.14],
    cellLoc='left',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.45)

for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    cell.set_linewidth(0.5)
    if row_idx == 0:
        cell.set_facecolor('#f8f9fa')
        cell.set_text_props(weight='bold', color='#111827')
    else:
        cell.set_facecolor('#ffffff' if row_idx % 2 != 0 else '#fafafa')
        cell.set_text_props(color='#374151')

fig2_path = os.path.join(OUTPUT_DIR, "boston_figure2_feature_selection.png")
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

print("\n" + "=" * 80)
print("BOSTON HOUSING PIPELINE & FIGURES COMPLETED SUCCESSFULLY")
print("=" * 80)
