<img width="1882" height="862" alt="image" src="https://github.com/user-attachments/assets/10b81a48-e7f5-45d2-a6ca-c8f1d71d9edc" />
<img width="1867" height="860" alt="image" src="https://github.com/user-attachments/assets/3e0fb0b9-cd28-4a78-b940-bb9da8fbac88" />

# 📈 Multiple Regression Explorer

> Feature selection and parsimony analysis for multiple linear regression — five selection
> algorithms, a k-sweep from 1 to every available predictor, full OLS inference, and a live
> predictor. **Point it at any CSV** and it runs the same analysis.

Built as an extension of the course reference build,
[AIoT-DIC2-MultipleLR](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/), whose dashboard
layout and five-algorithm comparison this project reproduces — then generalises from one
hard-coded dataset to any dataset you supply.

---

## 🌐 Live demo

**GitHub Pages:** https://zxc22872580.github.io/AIOT_DIC-2_MultipleRegression/

Everything runs in the browser. There is no backend call and nothing is precomputed: every
coefficient, p-value and curve on screen is fitted on page load.

---

## ✨ What it does

| Feature | Detail |
|---|---|
| 🔬 **Five selection algorithms** | Lasso (L1 ranking), Forward SFS, RFE, ElasticNet, Exhaustive best-subset |
| 📉 **k-sweep** | Every model refitted for k = 1 … p, with the minimum-RMSE "sweet spot" and the worst k marked |
| 🧮 **Full OLS inference** | β, standard error, t-statistic, p-value, 95% confidence interval, VIF, global F-test |
| 📊 **Dual charts** | RMSE and R² against k, with a compare-all mode overlaying all five methods |
| 🎛️ **Live predictor** | Sliders for numeric predictors, dropdowns for categorical ones, updating in real time |
| 🔍 **Residual diagnostics** | Largest hold-out residuals ranked, with % error and a flag above 1.5 × MAE |
| 📥 **Any dataset** | Load by URL, file upload, drag-and-drop or paste |
| 🧷 **Reproducible splits** | The train/test split matches `sklearn.train_test_split` row for row |

### Bringing your own dataset

Paste a URL, drop a file, or paste the text. The app then:

1. sniffs the delimiter (`,` `;` tab `|`) and parses quoted fields;
2. infers each column as numeric or categorical;
3. picks a target — the last numeric column, skipping id/year-style bookkeeping columns — which you can override;
4. expands categorical columns to *k−1* dummies, holding out the alphabetically first level as the reference (same convention as `pandas.get_dummies(drop_first=True)`);
5. drops rows with missing values and constant columns, and reports what it dropped;
6. refits all five algorithms across every k.

The URL the assignment suggests works directly:

```
https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv
```

> **CORS note.** A static page can only read URLs whose host allows cross-origin reads.
> `raw.githubusercontent.com` does. For hosts that don't, use **Upload file** or **Paste CSV** —
> the app says so explicitly rather than failing with a bare network error.

---

## 🗂️ Project structure

```
AIOT_DIC-2_MultipleRegression/
├── docs/                           # the app — this directory is what GitHub Pages serves
│   ├── index.html
│   ├── selftest.html               # numerical self-test (see "Verification")
│   └── assets/
│       ├── css/app.css
│       ├── js/stats.js             # MT19937 + numpy parity, linear algebra, t and F tails
│       ├── js/regression.js        # OLS inference, coordinate descent, the five algorithms
│       ├── js/data.js              # CSV parsing, schema inference, design matrix
│       ├── js/app.js               # dashboard wiring and charts
│       └── data/                   # built-in datasets
├── tools/
│   └── reference_check.py          # numpy/scipy/sklearn mirror of the JS spec
├── app.py                          # Flask: serves docs/ + the JSON prediction API
├── api/index.py                    # Vercel serverless entrypoint
├── train_crisp_dm.py               # 50 Startups CRISP-DM training pipeline
├── train_boston_pipeline.py        # Boston Housing training pipeline
├── generate_analysis_figures.py    # model comparison & feature selection figures
├── *.joblib                        # serialized pipelines used by the JSON API
├── requirements.txt
└── vercel.json
```

---

## 🧠 How the statistics are computed

Everything the dashboard shows is computed client-side in plain JavaScript — no scikit-learn,
no WASM, no precomputed JSON.

| Piece | Implementation |
|---|---|
| Train/test split | MT19937 + numpy's `random_interval` rejection sampling, reproducing `RandomState(seed).permutation(n)` exactly, then sklearn's `ShuffleSplit` slicing |
| OLS | Normal equations on a cached Gram matrix, Gauss-Jordan inverse with partial pivoting |
| Standard errors / t / p | σ² (XᵀX)⁻¹ diagonal; Student-t tail via the regularized incomplete beta (Lentz continued fraction) |
| Confidence intervals | t quantile by bisection on the CDF |
| F-test | Fisher-F tail from the same incomplete beta |
| VIF | Each predictor regressed on the others, read out of the training Gram without rescanning rows |
| Lasso / ElasticNet | Coordinate descent on standardized predictors, α chosen by 5-fold CV over a 40-point log grid |
| Forward SFS | Greedy forward selection maximising mean 5-fold CV R² |
| RFE | Recursive elimination of the smallest standardized \|β\| |
| Exhaustive | Best subset over all C(p, k) combinations, ranked by hold-out RMSE |

Caching the Gram matrix is what makes exhaustive search practical in a browser: fitting a
subset costs O(k³) rather than O(n·k²), so Boston Housing's 2¹³ subsets finish in well under a
second.

> **On the exhaustive search.** It selects the subset with the lowest *hold-out* RMSE, matching
> the reference build. That makes it an optimistic bound rather than an honest generalisation
> estimate, and the interface labels it accordingly.

---

## ✅ Verification

`tools/reference_check.py` is an independent implementation of the same specification using
numpy, scipy and scikit-learn. `docs/selftest.html` runs the JavaScript against its output.

```bash
pip install numpy pandas scipy scikit-learn
python tools/reference_check.py          # writes the fixtures into docs/
python -m http.server 8000 --directory docs
# then open http://localhost:8000/selftest.html
```

The suite checks, and currently passes:

- `numpyPermutation` against `numpy.random.RandomState(seed).permutation(n)` for n ∈ {10, 50, 100, 506} and seeds {0, 42} — exact;
- the t and F tails and the t quantile against scipy — to 1e-9;
- every coefficient, standard error, t-statistic, p-value, confidence bound, VIF, RMSE, MAE, R², adjusted R², F-statistic, F p-value and residual row, for all five algorithms at every k, on both built-in datasets — to 1e-8.

The reference numbers also reproduce the course build exactly, including its 50 Startups
landmarks: `8274.868` (k=1), `8198.797` (k=2), `8803.776`, `8997.199`, `9068.541`, `9137.990`
(full model), and the best-subset results `8171.646` (k=3) and `8178.369` (k=4).

---

## 🚀 Running locally

**Static only** — no Python needed:

```bash
python -m http.server 8000 --directory docs
```

**With the Flask API:**

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000. Flask serves `docs/` directly, so the local, Vercel and
GitHub Pages front ends are the same files.

### Retraining the joblib pipelines used by the API

```bash
python train_crisp_dm.py          # 50 Startups
python train_boston_pipeline.py   # Boston Housing
python generate_analysis_figures.py
```

### Publishing to GitHub Pages

Settings → Pages → Source: **Deploy from a branch**, branch `main`, folder **`/docs`**.

---

## 🔄 CRISP-DM workflow

```mermaid
flowchart LR
    A([🏢 Business<br/>Understanding]) --> B([📦 Data<br/>Understanding])
    B --> C([🔧 Data<br/>Preparation])
    C --> D([🤖 Modeling])
    D --> E([📊 Evaluation])
    E --> F([🚀 Deployment])
    F -->|Retrain| A

    style A fill:#1e3a5f,color:#fff,stroke:#3b82f6
    style B fill:#1e3a5f,color:#fff,stroke:#3b82f6
    style C fill:#1e3a5f,color:#fff,stroke:#3b82f6
    style D fill:#1e3a5f,color:#fff,stroke:#3b82f6
    style E fill:#1e3a5f,color:#fff,stroke:#3b82f6
    style F fill:#065f46,color:#fff,stroke:#10b981
```

| Phase | In this project |
|---|---|
| **1. Business understanding** | Predict a continuous target; find the smallest predictor set that does not cost accuracy |
| **2. Data understanding** | Schema inference, per-column statistics, correlation with the target, data preview |
| **3. Data preparation** | Dummy encoding with a held-out reference level, missing-value and constant-column handling, standardization for the penalised fits |
| **4. Modeling** | OLS plus five selection algorithms across every k |
| **5. Evaluation** | Hold-out RMSE / MAE / R² / adjusted R², F-test, VIF, residual and outlier diagnostics |
| **6. Deployment** | GitHub Pages (static) · Flask (local) · Vercel serverless (JSON API) |

---

## 🌐 REST API reference

The JSON API is unchanged and still backed by the trained `joblib` pipelines.

### Health check

```http
GET /api/health
```

```json
{ "status": "healthy", "models_loaded": { "startups": true, "boston": true }, "version": "3.0.0" }
```

### 50 Startups — profit prediction

```http
POST /api/predict/startups
Content-Type: application/json

{ "rd_spend": 160000, "administration": 130000, "marketing_spend": 300000, "state": "California" }
```

```json
{ "status": "success", "dataset": "50_startups", "predicted_profit": 182943.06 }
```

### Boston Housing — price prediction

```http
POST /api/predict/boston
Content-Type: application/json

{ "rm": 6.8, "lstat": 8.5, "ptratio": 16.0, "dis": 4.2, "tax": 350.0 }
```

```json
{ "status": "success", "dataset": "boston_housing", "predicted_medv": 29.44, "predicted_value_usd": 29435.26 }
```

Unspecified Boston features fall back to the training-set mean.

---

## 📦 Tech stack

| Layer | Technology |
|---|---|
| **Modelling (browser)** | Plain JavaScript — no ML library |
| **Charts** | Chart.js 4.4.1 |
| **Type** | Outfit · Inter · JetBrains Mono |
| **API server** | Python 3, Flask |
| **Training / verification** | scikit-learn, pandas, NumPy, SciPy, joblib |
| **Deployment** | GitHub Pages · Vercel (Python serverless) |

---

## 📁 Built-in datasets

| Dataset | Rows | Predictors | Target |
|---|---|---|---|
| `50_Startups.csv` | 50 | R&D Spend, Administration, Marketing Spend, State | **Profit** |
| `BostonHousing.csv` | 506 | CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT | **MEDV** (median value × $1,000) |

---

## 🙏 Acknowledgments

- **Reference build:** [AIoT-DIC2-MultipleLR](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/) by Huan-Chen — the dashboard layout and the five-algorithm comparison this project reproduces and extends.
- **Table style:** [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/).
- **Dataset:** [50 Startups](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv) — Avik Jain, *100-Days-Of-ML-Code*.
- **Methodology:** CRISP-DM.
