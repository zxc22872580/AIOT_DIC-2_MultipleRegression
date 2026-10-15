"""
Multi-Dataset Multiple Linear Regression Analytics Engine
Framework: Flask
The browser does the modelling: docs/ is a self-contained static app that fits
OLS, Lasso and ElasticNet and runs five feature-selection algorithms in plain
JavaScript, on the built-in datasets or any CSV you point it at.

This server exists for two things:
  - serving that static app locally and on Vercel
  - the JSON prediction API backed by the pre-trained joblib pipelines
"""

import os
import sys
import io
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

def _find_model(filename):
    for d in [BASE_DIR, os.getcwd(), os.path.dirname(BASE_DIR),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

startups_path = _find_model("startup_profit_pipeline.joblib")
boston_path   = _find_model("boston_housing_pipeline.joblib")
startups_pipeline = joblib.load(startups_path) if startups_path else None
boston_pipeline   = joblib.load(boston_path)   if boston_path   else None



# ══════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════

@app.route("/")
def home():
    """Serve the static explorer. docs/ is the single source of truth for the
    UI -- the same directory GitHub Pages publishes -- so local, Vercel and
    Pages all run a byte-identical front end."""
    return send_from_directory(DOCS_DIR, "index.html")


@app.route("/assets/<path:filename>")
def docs_assets(filename):
    return send_from_directory(os.path.join(DOCS_DIR, "assets"), filename)

@app.route("/figures/<path:filename>")
def get_figure(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy",
                    "models_loaded": {"startups": startups_pipeline is not None,
                                      "boston": boston_pipeline is not None},
                    "version": "3.0.0"})

@app.route("/api/predict/startups", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_startups():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([{
            'R&D Spend':       float(data.get("rd_spend", 0.0)),
            'Administration':  float(data.get("administration", 0.0)),
            'Marketing Spend': float(data.get("marketing_spend", 0.0)),
            'State':           str(data.get("state", "California"))
        }])
        prediction = float(startups_pipeline.predict(input_df)[0])
        return jsonify({"status": "success", "dataset": "50_startups",
                        "predicted_profit": round(prediction, 2)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/predict/boston", methods=["POST"])
def predict_boston():
    try:
        data = request.get_json(force=True)
        feature_cols = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']
        defaults = {'crim':0.25,'zn':11.36,'indus':11.14,'chas':0.0,'nox':0.55,'rm':6.28,
                    'age':68.57,'dis':3.79,'rad':9.55,'tax':408.2,'ptratio':18.46,'b':356.67,'lstat':12.65}
        input_data = {col: float(data.get(col, defaults[col])) for col in feature_cols}
        input_df = pd.DataFrame([[input_data[col] for col in feature_cols]], columns=feature_cols)
        medv_pred = float(boston_pipeline.predict(input_df)[0])
        return jsonify({"status": "success", "dataset": "boston_housing",
                        "predicted_medv": round(medv_pred, 2),
                        "predicted_value_usd": round(max(3000.0, medv_pred * 1000.0), 2),
                        "inputs": input_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ── In-memory model store for custom analysis ──
_custom_models = {}

@app.route("/api/analyze", methods=["POST"])
def analyze_custom():
    """
    Accepts arbitrary CSV rows + target/feature config.
    Runs scikit-learn LinearRegression (with StandardScaler) and returns metrics + coefficients.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import uuid

        data = request.get_json(force=True)
        target   = data.get("target")
        features = data.get("features", [])
        rows     = data.get("rows", [])

        if not target or not features or not rows:
            return jsonify({"status": "error", "message": "Missing target, features, or rows"}), 400

        df = pd.DataFrame(rows)
        df = df[[target] + features].dropna()
        if len(df) < 10:
            return jsonify({"status": "error", "message": f"Not enough valid rows after dropna (got {len(df)})"}), 400

        X = df[features].values.astype(float)
        y = df[target].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2   = float(r2_score(y_test, y_pred))
        n    = len(y_test)
        p    = len(features)
        r2_adj = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))

        # Unstandardize coefficients for display
        raw_coefs = model.coef_ / scaler.scale_
        raw_intercept = float(model.intercept_ - np.dot(raw_coefs, scaler.mean_))

        coefficients = [
            {"feature": f, "coef": float(raw_coefs[i]), "std_coef": float(model.coef_[i])}
            for i, f in enumerate(features)
        ]

        feature_means = {f: float(np.mean(X[:, i])) for i, f in enumerate(features)}

        # Store model for subsequent prediction calls
        model_id = str(uuid.uuid4())
        _custom_models[model_id] = {"model": model, "scaler": scaler, "features": features}

        return jsonify({
            "status": "success",
            "model_id": model_id,
            "r2": r2, "r2_adj": r2_adj,
            "rmse": rmse, "mae": mae,
            "intercept": raw_intercept,
            "coefficients": coefficients,
            "feature_means": feature_means,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/analyze/predict", methods=["POST"])
def analyze_custom_predict():
    try:
        data = request.get_json(force=True)
        model_id = data.get("model_id")
        inputs   = data.get("inputs", {})
        if model_id not in _custom_models:
            return jsonify({"status": "error", "message": "Model not found. Please re-run analysis."}), 400

        store   = _custom_models[model_id]
        model   = store["model"]
        scaler  = store["scaler"]
        features = store["features"]

        X_row = np.array([[float(inputs.get(f, 0)) for f in features]])
        X_scaled = scaler.transform(X_row)
        pred = float(model.predict(X_scaled)[0])
        return jsonify({"status": "success", "prediction": pred})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
