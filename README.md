# 50 Startups Multiple Linear Regression (CRISP-DM)

This project provides an end-to-end Machine Learning pipeline using `scikit-learn` following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** lifecycle.

## Project Structure
- `50_Startups.csv`: Benchmark dataset containing spending records across R&D, Administration, Marketing, and State locations.
- `train_crisp_dm.py`: Full CRISP-DM training, diagnostics, evaluation, and pipeline export script.
- `predict_startup.py`: Lightweight inference scoring script for new startup candidate inputs.
- `startup_profit_pipeline.joblib`: Serialized `scikit-learn` pipeline (`ColumnTransformer` + `LinearRegression`).
- `eda_analysis.png`: Correlation matrix heatmap and bivariate exploratory visualizations.
- `evaluation_diagnostics.png`: Residual plots, homoscedasticity checks, and actual vs. predicted diagnostics.

## Quickstart

### 1. Run Complete CRISP-DM Pipeline
```bash
python train_crisp_dm.py
```

### 2. Predict Profit for Candidate Startups
```bash
python predict_startup.py
```
