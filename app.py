"""
Web Application for 50 Startups Profit Prediction Service
Framework: Flask
Pipeline: scikit-learn Pipeline (startup_profit_pipeline.joblib)
Includes: Interactive Simulator + Model Benchmark & Feature Selection Analysis Figures
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "startup_profit_pipeline.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_crisp_dm.py first.")

pipeline = joblib.load(MODEL_PATH)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>50 Startups Profit Predictor & Model Analytics</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen py-10 px-4">
  <div class="max-w-5xl mx-auto space-y-10">
    
    <!-- Header -->
    <div class="text-center space-y-2">
      <span class="inline-block py-1 px-3 rounded-full bg-blue-500/10 text-blue-400 text-xs font-semibold tracking-wide border border-blue-500/20">
        CRISP-DM Phase 6: Production Web Service
      </span>
      <h1 class="text-3xl font-extrabold text-white sm:text-4xl">50 Startups Profit Prediction Engine</h1>
      <p class="text-slate-400 text-sm max-w-2xl mx-auto">
        Interactive Multiple Linear Regression deployment powered by Scikit-Learn, featuring real-time inference and model benchmark analytics.
      </p>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex justify-center">
      <div class="bg-slate-800 p-1.5 rounded-xl border border-slate-700 inline-flex gap-2">
        <button id="tab-btn-sim" onclick="switchTab('sim')" class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white transition">
          🚀 Interactive Predictor
        </button>
        <button id="tab-btn-bench" onclick="switchTab('bench')" class="px-4 py-2 text-sm font-medium rounded-lg text-slate-400 hover:text-white transition">
          📊 Model Comparison (10 Models)
        </button>
        <button id="tab-btn-feats" onclick="switchTab('feats')" class="px-4 py-2 text-sm font-medium rounded-lg text-slate-400 hover:text-white transition">
          🎯 Feature Selection Curves
        </button>
      </div>
    </div>

    <!-- TAB 1: INTERACTIVE SIMULATOR -->
    <div id="tab-sim" class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Input Controls Column (7 cols) -->
      <div class="lg:col-span-7 bg-slate-800/80 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl space-y-6">
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
          <h2 class="text-lg font-semibold text-white flex items-center gap-2">
            <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
            Investment Parameters
          </h2>
          <div class="flex gap-1.5">
            <button onclick="applyPreset('tech')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">AI Tech</button>
            <button onclick="applyPreset('d2c')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Marketing D2C</button>
            <button onclick="applyPreset('lean')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Lean</button>
          </div>
        </div>

        <form id="predict-form" class="space-y-5" onsubmit="event.preventDefault(); triggerPrediction();">
          <!-- R&D Spend Slider -->
          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span>
                R&D Spend
              </label>
              <span id="rd-val" class="font-mono text-emerald-400 font-semibold">$120,000</span>
            </div>
            <input type="range" id="rd-spend" min="0" max="250000" step="5000" value="120000" class="w-full accent-emerald-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>$0</span>
              <span>$125k</span>
              <span>$250k</span>
            </div>
          </div>

          <!-- Marketing Spend Slider -->
          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-blue-400 inline-block"></span>
                Marketing Spend
              </label>
              <span id="mkt-val" class="font-mono text-blue-400 font-semibold">$250,000</span>
            </div>
            <input type="range" id="mkt-spend" min="0" max="500000" step="10000" value="250000" class="w-full accent-blue-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>$0</span>
              <span>$250k</span>
              <span>$500k</span>
            </div>
          </div>

          <!-- Administration Slider -->
          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block"></span>
                Administration Spend
              </label>
              <span id="admin-val" class="font-mono text-amber-400 font-semibold">$115,000</span>
            </div>
            <input type="range" id="admin-spend" min="0" max="250000" step="5000" value="115000" class="w-full accent-amber-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>$0</span>
              <span>$125k</span>
              <span>$250k</span>
            </div>
          </div>

          <!-- State Selection -->
          <div class="space-y-2 pt-1">
            <label class="block text-sm font-medium text-slate-300">State Headquarters</label>
            <div class="grid grid-cols-3 gap-3">
              <label class="relative flex items-center justify-center p-2.5 border border-slate-700 rounded-xl cursor-pointer hover:border-slate-500 transition text-sm">
                <input type="radio" name="state" value="California" class="peer sr-only" checked>
                <span class="text-slate-400 peer-checked:text-blue-400 peer-checked:font-semibold">California</span>
                <div class="absolute inset-0 border-2 border-transparent peer-checked:border-blue-500 rounded-xl pointer-events-none"></div>
              </label>
              <label class="relative flex items-center justify-center p-2.5 border border-slate-700 rounded-xl cursor-pointer hover:border-slate-500 transition text-sm">
                <input type="radio" name="state" value="New York" class="peer sr-only">
                <span class="text-slate-400 peer-checked:text-blue-400 peer-checked:font-semibold">New York</span>
                <div class="absolute inset-0 border-2 border-transparent peer-checked:border-blue-500 rounded-xl pointer-events-none"></div>
              </label>
              <label class="relative flex items-center justify-center p-2.5 border border-slate-700 rounded-xl cursor-pointer hover:border-slate-500 transition text-sm">
                <input type="radio" name="state" value="Florida" class="peer sr-only">
                <span class="text-slate-400 peer-checked:text-blue-400 peer-checked:font-semibold">Florida</span>
                <div class="absolute inset-0 border-2 border-transparent peer-checked:border-blue-500 rounded-xl pointer-events-none"></div>
              </label>
            </div>
          </div>
        </form>
      </div>

      <!-- Prediction & Analytics Column (5 cols) -->
      <div class="lg:col-span-5 flex flex-col gap-6">
        <div class="bg-gradient-to-br from-slate-800 to-slate-800/90 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
          <span class="text-xs uppercase tracking-wider font-semibold text-slate-400">Predicted Annual Profit</span>
          <div class="flex items-baseline gap-2">
            <span id="predicted-profit" class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200">
              $150,260.40
            </span>
          </div>
          <div class="flex items-center gap-2 text-xs text-slate-400 border-t border-slate-700/60 pt-3">
            <span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              R² = 0.8987
            </span>
            <span>Evaluated by Scikit-learn Pipeline</span>
          </div>
        </div>

        <div class="bg-slate-800/80 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4 flex-1">
          <h3 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">Marginal Impact Breakdown</h3>
          <div class="space-y-3 text-xs">
            <div>
              <div class="flex justify-between mb-1">
                <span class="text-slate-400">R&D Spend Contribution (β = +0.81)</span>
                <span id="rd-contrib" class="font-mono text-emerald-400 font-semibold">+$96,675</span>
              </div>
              <div class="w-full bg-slate-700/50 rounded-full h-1.5">
                <div id="rd-bar" class="bg-emerald-500 h-1.5 rounded-full" style="width: 80%"></div>
              </div>
            </div>

            <div>
              <div class="flex justify-between mb-1">
                <span class="text-slate-400">Marketing Contribution (β = +0.03)</span>
                <span id="mkt-contrib" class="font-mono text-blue-400 font-semibold">+$7,464</span>
              </div>
              <div class="w-full bg-slate-700/50 rounded-full h-1.5">
                <div id="mkt-bar" class="bg-blue-500 h-1.5 rounded-full" style="width: 25%"></div>
              </div>
            </div>

            <div>
              <div class="flex justify-between mb-1">
                <span class="text-slate-400">Admin Cost Drag (β = -0.07)</span>
                <span id="admin-contrib" class="font-mono text-amber-400 font-semibold">-$7,910</span>
              </div>
              <div class="w-full bg-slate-700/50 rounded-full h-1.5">
                <div id="admin-bar" class="bg-amber-500 h-1.5 rounded-full" style="width: 15%"></div>
              </div>
            </div>

            <div class="pt-2 border-t border-slate-700/50 text-[11px] text-slate-400">
              <span class="font-semibold text-slate-300">Baseline Intercept (β₀):</span> $54,028.04
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: BENCHMARK FIGURE 1 -->
    <div id="tab-bench" class="hidden space-y-6">
      <div class="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
          <div>
            <h2 class="text-lg font-semibold text-white">Comparison of Different Regression Models</h2>
            <p class="text-xs text-slate-400">Evaluation across SVR, LASSO, Ridge, Elastic Net, PCA, Random Forest, GBR, SVM, and LightGBM</p>
          </div>
          <span class="text-xs px-2.5 py-1 bg-blue-500/10 text-blue-400 rounded-lg border border-blue-500/20">Analysis Fig 1</span>
        </div>
        <div class="flex justify-center bg-slate-900/50 p-2 rounded-xl">
          <img src="/figures/figure1_model_comparison.png" alt="Model Comparison" class="max-w-full h-auto rounded-lg shadow-md border border-slate-700/60">
        </div>
      </div>
    </div>

    <!-- TAB 3: FEATURE SELECTION FIGURE 2 -->
    <div id="tab-feats" class="hidden space-y-6">
      <div class="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
          <div>
            <h2 class="text-lg font-semibold text-white">Sequential Forward Feature Selection Curves</h2>
            <p class="text-xs text-slate-400">RMSE and R² progression as additional features are added (Demonstrating optimal 2-feature subset)</p>
          </div>
          <span class="text-xs px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">Analysis Fig 2</span>
        </div>
        <div class="flex justify-center bg-slate-900/50 p-2 rounded-xl">
          <img src="/figures/figure2_feature_selection.png" alt="Feature Selection Curves" class="max-w-full h-auto rounded-lg shadow-md border border-slate-700/60">
        </div>
      </div>
    </div>

    <!-- API Reference Footnote -->
    <div class="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-3">
      <div>
        <span class="font-semibold text-slate-300">REST API Endpoint:</span> <code class="bg-slate-900 px-2 py-0.5 rounded text-blue-400">POST /api/predict</code>
      </div>
      <div class="font-mono text-slate-500">
        Status: <span class="text-emerald-400 font-medium">Model Online & Serving</span>
      </div>
    </div>

  </div>

  <script>
    function switchTab(tabId) {
      document.getElementById('tab-sim').classList.add('hidden');
      document.getElementById('tab-bench').classList.add('hidden');
      document.getElementById('tab-feats').classList.add('hidden');

      document.getElementById('tab-btn-sim').className = "px-4 py-2 text-sm font-medium rounded-lg text-slate-400 hover:text-white transition";
      document.getElementById('tab-btn-bench').className = "px-4 py-2 text-sm font-medium rounded-lg text-slate-400 hover:text-white transition";
      document.getElementById('tab-btn-feats').className = "px-4 py-2 text-sm font-medium rounded-lg text-slate-400 hover:text-white transition";

      if (tabId === 'sim') {
        document.getElementById('tab-sim').classList.remove('hidden');
        document.getElementById('tab-btn-sim').className = "px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white transition";
      } else if (tabId === 'bench') {
        document.getElementById('tab-bench').classList.remove('hidden');
        document.getElementById('tab-btn-bench').className = "px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white transition";
      } else {
        document.getElementById('tab-feats').classList.remove('hidden');
        document.getElementById('tab-btn-feats').className = "px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white transition";
      }
    }

    const rdInput = document.getElementById('rd-spend');
    const mktInput = document.getElementById('mkt-spend');
    const adminInput = document.getElementById('admin-spend');

    const rdVal = document.getElementById('rd-val');
    const mktVal = document.getElementById('mkt-val');
    const adminVal = document.getElementById('admin-val');

    const profitDisplay = document.getElementById('predicted-profit');
    const rdContrib = document.getElementById('rd-contrib');
    const mktContrib = document.getElementById('mkt-contrib');
    const adminContrib = document.getElementById('admin-contrib');
    const rdBar = document.getElementById('rd-bar');
    const mktBar = document.getElementById('mkt-bar');
    const adminBar = document.getElementById('admin-bar');

    function formatCurrency(num) {
      return '$' + Math.round(num).toLocaleString();
    }

    const presets = {
      tech: { rd: 165000, mkt: 300000, admin: 100000, state: 'California' },
      d2c:  { rd: 45000,  mkt: 480000, admin: 130000, state: 'New York' },
      lean: { rd: 90000,  mkt: 80000,  admin: 50000,  state: 'Florida' }
    };

    function applyPreset(name) {
      const p = presets[name];
      if (!p) return;
      rdInput.value = p.rd;
      mktInput.value = p.mkt;
      adminInput.value = p.admin;
      const radio = document.querySelector(`input[name="state"][value="${p.state}"]`);
      if (radio) radio.checked = true;
      triggerPrediction();
    }

    async function triggerPrediction() {
      const rd = parseFloat(rdInput.value);
      const mkt = parseFloat(mktInput.value);
      const admin = parseFloat(adminInput.value);
      const state = document.querySelector('input[name="state"]:checked').value;

      rdVal.textContent = formatCurrency(rd);
      mktVal.textContent = formatCurrency(mkt);
      adminVal.textContent = formatCurrency(admin);

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rd_spend: rd,
            administration: admin,
            marketing_spend: mkt,
            state: state
          })
        });
        const data = await response.json();
        if (data.status === 'success') {
          profitDisplay.textContent = formatCurrency(data.predicted_profit);
        }
      } catch (err) {
        const pred = 54028.04 + (0.805630 * rd) - (0.068788 * admin) + (0.029855 * mkt) + (state === 'Florida' ? 938.79 : (state === 'New York' ? 6.99 : 0));
        profitDisplay.textContent = formatCurrency(pred);
      }

      const rdEffect = 0.805630 * rd;
      const mktEffect = 0.029855 * mkt;
      const adminEffect = -0.068788 * admin;

      rdContrib.textContent = '+' + formatCurrency(rdEffect);
      mktContrib.textContent = '+' + formatCurrency(mktEffect);
      adminContrib.textContent = '-' + formatCurrency(Math.abs(adminEffect));

      const maxScale = 200000;
      rdBar.style.width = Math.min(100, Math.max(5, (rdEffect / maxScale) * 100)) + '%';
      mktBar.style.width = Math.min(100, Math.max(5, (mktEffect / 20000) * 100)) + '%';
      adminBar.style.width = Math.min(100, Math.max(5, (Math.abs(adminEffect) / 20000) * 100)) + '%';
    }

    [rdInput, mktInput, adminInput].forEach(inp => {
      inp.addEventListener('input', triggerPrediction);
    });
    document.querySelectorAll('input[name="state"]').forEach(radio => {
      radio.addEventListener('change', triggerPrediction);
    });

    triggerPrediction();
  </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/figures/<path:filename>")
def get_figure(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "scikit-learn LinearRegression",
        "version": "1.0.0"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        rd_spend = float(data.get("rd_spend", 0.0))
        administration = float(data.get("administration", 0.0))
        marketing_spend = float(data.get("marketing_spend", 0.0))
        state = str(data.get("state", "California"))

        input_df = pd.DataFrame([{
            'R&D Spend': rd_spend,
            'Administration': administration,
            'Marketing Spend': marketing_spend,
            'State': state
        }])

        prediction = float(pipeline.predict(input_df)[0])

        return jsonify({
            "status": "success",
            "predicted_profit": round(prediction, 2),
            "inputs": {
                "rd_spend": rd_spend,
                "administration": administration,
                "marketing_spend": marketing_spend,
                "state": state
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
