"""
Web Application for 50 Startups & Boston Housing Multi-Dataset Regression Engine
Framework: Flask
Pipelines: 
  - 50 Startups (startup_profit_pipeline.joblib)
  - Boston Housing (boston_housing_pipeline.joblib)
Features:
  - Interactive Dataset Selector Dropdown (50 Startups vs. Boston Housing)
  - Dynamic Predictor Simulator with Sliders & Presets
  - Model Comparison Benchmarks (10 Regression Models)
  - Sequential Forward Feature Selection Curves
  - REST API Endpoints for both datasets
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

def _find_model(filename):
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(BASE_DIR), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

startups_path = _find_model("startup_profit_pipeline.joblib")
boston_path = _find_model("boston_housing_pipeline.joblib")

startups_pipeline = joblib.load(startups_path) if startups_path else None
boston_pipeline = joblib.load(boston_path) if boston_path else None

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-Dataset Regression & Analytics Engine</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen py-10 px-4">
  <div class="max-w-5xl mx-auto space-y-8">
    
    <!-- Top Bar with Dataset Dropdown Selector -->
    <div class="flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-800/90 border border-slate-700 p-4 rounded-2xl shadow-xl">
      <div class="flex items-center gap-3">
        <span class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
        <div>
          <h1 class="text-lg font-bold text-white">Multiple Linear Regression Analytics</h1>
          <p class="text-xs text-slate-400">CRISP-DM Machine Learning Platform</p>
        </div>
      </div>

      <!-- Dataset Selector Dropdown -->
      <div class="flex items-center gap-2.5 w-full sm:w-auto">
        <label for="dataset-select" class="text-xs font-semibold text-slate-300 uppercase tracking-wide whitespace-nowrap">
          選擇資料集 (Dataset):
        </label>
        <div class="relative w-full sm:w-64">
          <select id="dataset-select" onchange="onDatasetChange(this.value)" class="w-full bg-slate-900 border border-blue-500/50 hover:border-blue-400 text-blue-300 font-medium py-2 px-3 pr-8 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition cursor-pointer">
            <option value="startups" selected>📈 50 Startups (Profit Prediction)</option>
            <option value="boston">🏡 Boston Housing (Price Prediction)</option>
          </select>
        </div>
      </div>
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

    <!-- ===================================================================== -->
    <!-- TAB 1: INTERACTIVE PREDICTOR (DYNAMIC FOR STARTUPS & BOSTON) -->
    <!-- ===================================================================== -->
    <div id="tab-sim" class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- CONTROLS COLUMN (7 cols) -->
      <div class="lg:col-span-7 bg-slate-800/80 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl space-y-6">
        
        <!-- Controls Header & Presets -->
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
          <h2 id="controls-title" class="text-base font-semibold text-white flex items-center gap-2">
            Investment Parameters
          </h2>
          <!-- Preset buttons (updated dynamically) -->
          <div id="presets-container" class="flex gap-1.5">
            <!-- Injected via JS -->
          </div>
        </div>

        <!-- 50 Startups Controls Form -->
        <div id="form-startups" class="space-y-5">
          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span> R&D Spend
              </label>
              <span id="rd-val" class="font-mono text-emerald-400 font-semibold">$120,000</span>
            </div>
            <input type="range" id="rd-spend" min="0" max="250000" step="5000" value="120000" class="w-full accent-emerald-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>$0</span><span>$125k</span><span>$250k</span></div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span> Marketing Spend
              </label>
              <span id="mkt-val" class="font-mono text-blue-400 font-semibold">$250,000</span>
            </div>
            <input type="range" id="mkt-spend" min="0" max="500000" step="10000" value="250000" class="w-full accent-blue-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>$0</span><span>$250k</span><span>$500k</span></div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span> Administration Spend
              </label>
              <span id="admin-val" class="font-mono text-amber-400 font-semibold">$115,000</span>
            </div>
            <input type="range" id="admin-spend" min="0" max="250000" step="5000" value="115000" class="w-full accent-amber-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>$0</span><span>$125k</span><span>$250k</span></div>
          </div>

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
        </div>

        <!-- Boston Housing Controls Form -->
        <div id="form-boston" class="space-y-5 hidden">
          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span> RM (Avg Rooms per Dwelling)
              </label>
              <span id="rm-val" class="font-mono text-emerald-400 font-semibold">6.28 rooms</span>
            </div>
            <input type="range" id="b-rm" min="3.5" max="8.8" step="0.1" value="6.28" class="w-full accent-emerald-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>3.5</span><span>6.2</span><span>8.8</span></div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-rose-400"></span> LSTAT (% Lower Status Population)
              </label>
              <span id="lstat-val" class="font-mono text-rose-400 font-semibold">12.6%</span>
            </div>
            <input type="range" id="b-lstat" min="1.0" max="38.0" step="0.5" value="12.6" class="w-full accent-rose-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>1%</span><span>19%</span><span>38%</span></div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span> PTRATIO (Pupil-Teacher Ratio)
              </label>
              <span id="ptratio-val" class="font-mono text-blue-400 font-semibold">18.5</span>
            </div>
            <input type="range" id="b-ptratio" min="12.0" max="22.0" step="0.5" value="18.5" class="w-full accent-blue-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>12.0</span><span>17.0</span><span>22.0</span></div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span> DIS (Distance to Employment Centers)
              </label>
              <span id="dis-val" class="font-mono text-amber-400 font-semibold">3.8 km</span>
            </div>
            <input type="range" id="b-dis" min="1.0" max="12.0" step="0.2" value="3.8" class="w-full accent-amber-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>1.0 km</span><span>6.5 km</span><span>12.0 km</span></div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between items-center text-sm">
              <label class="font-medium text-slate-300 flex items-center gap-1.5">
                <span class="w-2.5 h-2.5 rounded-full bg-purple-400"></span> TAX (Property Tax Rate / $10k)
              </label>
              <span id="tax-val" class="font-mono text-purple-400 font-semibold">$408</span>
            </div>
            <input type="range" id="b-tax" min="180" max="711" step="10" value="408" class="w-full accent-purple-500 cursor-pointer">
            <div class="flex justify-between text-[11px] text-slate-500 font-mono"><span>$180</span><span>$450</span><span>$711</span></div>
          </div>
        </div>

      </div>

      <!-- PREDICTION OUTPUT COLUMN (5 cols) -->
      <div class="lg:col-span-5 flex flex-col gap-6">
        
        <!-- Score Card -->
        <div class="bg-gradient-to-br from-slate-800 to-slate-800/90 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
          <span id="pred-card-label" class="text-xs uppercase tracking-wider font-semibold text-slate-400">
            Predicted Annual Profit
          </span>
          <div class="flex items-baseline gap-2">
            <span id="predicted-output" class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200">
              $150,260.40
            </span>
          </div>
          <div class="flex items-center gap-2 text-xs text-slate-400 border-t border-slate-700/60 pt-3">
            <span id="r2-badge" class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              R² = 0.8987
            </span>
            <span id="model-desc">Scikit-learn Regression Pipeline</span>
          </div>
        </div>

        <!-- Marginal Contribution Card -->
        <div class="bg-slate-800/80 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4 flex-1">
          <h3 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">Key Predictor Breakdown</h3>
          <div id="breakdown-container" class="space-y-3 text-xs">
            <!-- Dynamic breakdown rows injected via JS -->
          </div>
        </div>

      </div>

    </div>

    <!-- ===================================================================== -->
    <!-- TAB 2: MODEL COMPARISON (FIGURE 1) -->
    <!-- ===================================================================== -->
    <div id="tab-bench" class="hidden space-y-6">
      <div class="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
          <div>
            <h2 id="fig1-title" class="text-lg font-semibold text-white">Comparison of Different Regression Models</h2>
            <p id="fig1-desc" class="text-xs text-slate-400">10 Regression Models evaluated and ranked by MSE ascending</p>
          </div>
          <span class="text-xs px-2.5 py-1 bg-blue-500/10 text-blue-400 rounded-lg border border-blue-500/20">Analysis Fig 1</span>
        </div>
        <div class="flex justify-center bg-slate-900/50 p-2 rounded-xl">
          <img id="fig1-img" src="/figures/figure1_model_comparison.png" alt="Model Comparison" class="max-w-full h-auto rounded-lg shadow-md border border-slate-700/60">
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 3: FEATURE SELECTION (FIGURE 2) -->
    <!-- ===================================================================== -->
    <div id="tab-feats" class="hidden space-y-6">
      <div class="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
          <div>
            <h2 id="fig2-title" class="text-lg font-semibold text-white">Sequential Forward Feature Selection Curves</h2>
            <p id="fig2-desc" class="text-xs text-slate-400">Progression of RMSE and R² metrics as features are added step-by-step</p>
          </div>
          <span class="text-xs px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">Analysis Fig 2</span>
        </div>
        <div class="flex justify-center bg-slate-900/50 p-2 rounded-xl">
          <img id="fig2-img" src="/figures/figure2_feature_selection.png" alt="Feature Selection Curves" class="max-w-full h-auto rounded-lg shadow-md border border-slate-700/60">
        </div>
      </div>
    </div>

    <!-- API Reference Footnote -->
    <div class="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-3">
      <div>
        <span class="font-semibold text-slate-300">REST API Endpoints:</span> 
        <code class="bg-slate-900 px-2 py-0.5 rounded text-blue-400">POST /api/predict/startups</code>
        <span class="mx-1 text-slate-600">|</span>
        <code class="bg-slate-900 px-2 py-0.5 rounded text-blue-400">POST /api/predict/boston</code>
      </div>
      <div class="font-mono text-slate-500">
        Status: <span class="text-emerald-400 font-medium">Dual Models Active & Serving</span>
      </div>
    </div>

  </div>

  <script>
    let currentDataset = 'startups';

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

    function onDatasetChange(dataset) {
      currentDataset = dataset;
      const formStartups = document.getElementById('form-startups');
      const formBoston = document.getElementById('form-boston');
      const presetsContainer = document.getElementById('presets-container');
      const controlsTitle = document.getElementById('controls-title');
      const predCardLabel = document.getElementById('pred-card-label');
      const r2Badge = document.getElementById('r2-badge');
      const fig1Img = document.getElementById('fig1-img');
      const fig2Img = document.getElementById('fig2-img');
      const fig1Title = document.getElementById('fig1-title');
      const fig2Title = document.getElementById('fig2-title');

      if (dataset === 'startups') {
        formStartups.classList.remove('hidden');
        formBoston.classList.add('hidden');
        controlsTitle.textContent = "Investment Parameters (50 Startups)";
        predCardLabel.textContent = "Predicted Annual Profit";
        r2Badge.textContent = "R² = 0.8987";
        fig1Img.src = "/figures/figure1_model_comparison.png";
        fig2Img.src = "/figures/figure2_feature_selection.png";
        fig1Title.textContent = "50 Startups: Comparison of Different Regression Models";
        fig2Title.textContent = "50 Startups: Sequential Feature Selection Curves";

        presetsContainer.innerHTML = `
          <button onclick="applyPreset('tech')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">AI Tech</button>
          <button onclick="applyPreset('d2c')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Marketing D2C</button>
          <button onclick="applyPreset('lean')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Lean</button>
        `;
        triggerStartupsPrediction();
      } else {
        formStartups.classList.add('hidden');
        formBoston.classList.remove('hidden');
        controlsTitle.textContent = "Housing Indicators (Boston Housing)";
        predCardLabel.textContent = "Predicted Median Home Value";
        r2Badge.textContent = "R² = 0.6688";
        fig1Img.src = "/figures/boston_figure1_model_comparison.png";
        fig2Img.src = "/figures/boston_figure2_feature_selection.png";
        fig1Title.textContent = "Boston Housing: Comparison of Different Regression Models (RM vs MEDV)";
        fig2Title.textContent = "Boston Housing: Sequential Feature Selection Curves";

        presetsContainer.innerHTML = `
          <button onclick="applyBostonPreset('suburban')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Suburban Luxury</button>
          <button onclick="applyBostonPreset('urban')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Urban Core</button>
          <button onclick="applyBostonPreset('affordable')" class="text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 font-medium transition">Affordable</button>
        `;
        triggerBostonPrediction();
      }
    }

    // 50 Startups Logic
    const rdInput = document.getElementById('rd-spend');
    const mktInput = document.getElementById('mkt-spend');
    const adminInput = document.getElementById('admin-spend');

    function applyPreset(name) {
      const presets = {
        tech: { rd: 165000, mkt: 300000, admin: 100000, state: 'California' },
        d2c:  { rd: 45000,  mkt: 480000, admin: 130000, state: 'New York' },
        lean: { rd: 90000,  mkt: 80000,  admin: 50000,  state: 'Florida' }
      };
      const p = presets[name];
      if (!p) return;
      rdInput.value = p.rd;
      mktInput.value = p.mkt;
      adminInput.value = p.admin;
      const radio = document.querySelector(`input[name="state"][value="${p.state}"]`);
      if (radio) radio.checked = true;
      triggerStartupsPrediction();
    }

    async function triggerStartupsPrediction() {
      const rd = parseFloat(rdInput.value);
      const mkt = parseFloat(mktInput.value);
      const admin = parseFloat(adminInput.value);
      const state = document.querySelector('input[name="state"]:checked').value;

      document.getElementById('rd-val').textContent = '$' + Math.round(rd).toLocaleString();
      document.getElementById('mkt-val').textContent = '$' + Math.round(mkt).toLocaleString();
      document.getElementById('admin-val').textContent = '$' + Math.round(admin).toLocaleString();

      try {
        const resp = await fetch('/api/predict/startups', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rd_spend: rd, administration: admin, marketing_spend: mkt, state: state })
        });
        const res = await resp.json();
        if (res.status === 'success') {
          document.getElementById('predicted-output').textContent = '$' + Math.round(res.predicted_profit).toLocaleString();
        }
      } catch (e) {
        const pred = 54028.04 + (0.805630 * rd) - (0.068788 * admin) + (0.029855 * mkt) + (state === 'Florida' ? 938.79 : (state === 'New York' ? 6.99 : 0));
        document.getElementById('predicted-output').textContent = '$' + Math.round(pred).toLocaleString();
      }

      const rdEff = 0.805630 * rd;
      const mktEff = 0.029855 * mkt;
      const admEff = -0.068788 * admin;

      document.getElementById('breakdown-container').innerHTML = `
        <div>
          <div class="flex justify-between mb-1"><span class="text-slate-400">R&D Spend Contribution (β = +0.81)</span><span class="font-mono text-emerald-400 font-semibold">+$${Math.round(rdEff).toLocaleString()}</span></div>
          <div class="w-full bg-slate-700/50 rounded-full h-1.5"><div class="bg-emerald-500 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(5, (rdEff/200000)*100))}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between mb-1"><span class="text-slate-400">Marketing Contribution (β = +0.03)</span><span class="font-mono text-blue-400 font-semibold">+$${Math.round(mktEff).toLocaleString()}</span></div>
          <div class="w-full bg-slate-700/50 rounded-full h-1.5"><div class="bg-blue-500 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(5, (mktEff/20000)*100))}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between mb-1"><span class="text-slate-400">Admin Overhead Drag (β = -0.07)</span><span class="font-mono text-amber-400 font-semibold">-$${Math.round(Math.abs(admEff)).toLocaleString()}</span></div>
          <div class="w-full bg-slate-700/50 rounded-full h-1.5"><div class="bg-amber-500 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(5, (Math.abs(admEff)/20000)*100))}%"></div></div>
        </div>
        <div class="pt-2 border-t border-slate-700/50 text-[11px] text-slate-400">
          <span class="font-semibold text-slate-300">Baseline Intercept (β₀):</span> $54,028.04
        </div>
      `;
    }

    // Boston Housing Logic
    const rmInp = document.getElementById('b-rm');
    const lstatInp = document.getElementById('b-lstat');
    const ptratioInp = document.getElementById('b-ptratio');
    const disInp = document.getElementById('b-dis');
    const taxInp = document.getElementById('b-tax');

    function applyBostonPreset(name) {
      const presets = {
        suburban:   { rm: 7.8, lstat: 4.5, ptratio: 14.5, dis: 5.5, tax: 280 },
        urban:      { rm: 6.2, lstat: 12.0, ptratio: 18.0, dis: 2.8, tax: 420 },
        affordable: { rm: 5.2, lstat: 24.0, ptratio: 20.2, dis: 2.1, tax: 660 }
      };
      const p = presets[name];
      if (!p) return;
      rmInp.value = p.rm;
      lstatInp.value = p.lstat;
      ptratioInp.value = p.ptratio;
      disInp.value = p.dis;
      taxInp.value = p.tax;
      triggerBostonPrediction();
    }

    async function triggerBostonPrediction() {
      const rm = parseFloat(rmInp.value);
      const lstat = parseFloat(lstatInp.value);
      const ptratio = parseFloat(ptratioInp.value);
      const dis = parseFloat(disInp.value);
      const tax = parseFloat(taxInp.value);

      document.getElementById('rm-val').textContent = rm.toFixed(2) + ' rooms';
      document.getElementById('lstat-val').textContent = lstat.toFixed(1) + '%';
      document.getElementById('ptratio-val').textContent = ptratio.toFixed(1);
      document.getElementById('dis-val').textContent = dis.toFixed(1) + ' km';
      document.getElementById('tax-val').textContent = '$' + Math.round(tax);

      try {
        const resp = await fetch('/api/predict/boston', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rm: rm, lstat: lstat, ptratio: ptratio, dis: dis, tax: tax, crim: 0.25 })
        });
        const res = await resp.json();
        if (res.status === 'success') {
          document.getElementById('predicted-output').textContent = '$' + Math.round(res.predicted_value_usd).toLocaleString();
        }
      } catch (e) {
        // Fallback MLR formula for Boston Housing
        const medv = 36.459 + (3.809 * (rm - 6.28)) - (0.547 * (lstat - 12.65)) - (0.952 * (ptratio - 18.45));
        document.getElementById('predicted-output').textContent = '$' + Math.round(Math.max(5, medv) * 1000).toLocaleString();
      }

      const rmImpact = (rm - 6.28) * 3800;
      const lstatImpact = (12.65 - lstat) * 550;
      const ptratioImpact = (18.45 - ptratio) * 950;

      document.getElementById('breakdown-container').innerHTML = `
        <div>
          <div class="flex justify-between mb-1"><span class="text-slate-400">Rooms (RM vs Avg 6.28)</span><span class="font-mono ${rmImpact >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-semibold">${rmImpact >= 0 ? '+' : ''}$${Math.round(rmImpact).toLocaleString()}</span></div>
          <div class="w-full bg-slate-700/50 rounded-full h-1.5"><div class="bg-emerald-500 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(5, (rm/8.8)*100))}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between mb-1"><span class="text-slate-400">Socioeconomic Status (LSTAT)</span><span class="font-mono ${lstatImpact >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-semibold">${lstatImpact >= 0 ? '+' : ''}$${Math.round(lstatImpact).toLocaleString()}</span></div>
          <div class="w-full bg-slate-700/50 rounded-full h-1.5"><div class="bg-rose-500 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(5, (lstat/38)*100))}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between mb-1"><span class="text-slate-400">School Quality (PTRATIO)</span><span class="font-mono ${ptratioImpact >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-semibold">${ptratioImpact >= 0 ? '+' : ''}$${Math.round(ptratioImpact).toLocaleString()}</span></div>
          <div class="w-full bg-slate-700/50 rounded-full h-1.5"><div class="bg-blue-500 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(5, (ptratio/22)*100))}%"></div></div>
        </div>
        <div class="pt-2 border-t border-slate-700/50 text-[11px] text-slate-400">
          <span class="font-semibold text-slate-300">Baseline Town Median Price:</span> $22,532.80
        </div>
      `;
    }

    [rdInput, mktInput, adminInput].forEach(inp => inp.addEventListener('input', triggerStartupsPrediction));
    document.querySelectorAll('input[name="state"]').forEach(radio => radio.addEventListener('change', triggerStartupsPrediction));

    [rmInp, lstatInp, ptratioInp, disInp, taxInp].forEach(inp => inp.addEventListener('input', triggerBostonPrediction));

    // Initialize Default View
    onDatasetChange('startups');
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
        "models_loaded": {
            "startups": startups_pipeline is not None,
            "boston": boston_pipeline is not None
        },
        "version": "2.0.0"
    })

# Startups Prediction API
@app.route("/api/predict/startups", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_startups():
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

        prediction = float(startups_pipeline.predict(input_df)[0])

        return jsonify({
            "status": "success",
            "dataset": "50_startups",
            "predicted_profit": round(prediction, 2),
            "inputs": {
                "rd_spend": rd_spend,
                "administration": administration,
                "marketing_spend": marketing_spend,
                "state": state
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Boston Housing Prediction API
@app.route("/api/predict/boston", methods=["POST"])
def predict_boston():
    try:
        data = request.get_json(force=True)
        # Default typical means for Boston Housing
        input_data = {
            'crim': float(data.get('crim', 0.25)),
            'zn': float(data.get('zn', 11.36)),
            'indus': float(data.get('indus', 11.14)),
            'chas': float(data.get('chas', 0.0)),
            'nox': float(data.get('nox', 0.55)),
            'rm': float(data.get('rm', 6.28)),
            'age': float(data.get('age', 68.57)),
            'dis': float(data.get('dis', 3.79)),
            'rad': float(data.get('rad', 9.55)),
            'tax': float(data.get('tax', 408.2)),
            'ptratio': float(data.get('ptratio', 18.46)),
            'b': float(data.get('b', 356.67)),
            'lstat': float(data.get('lstat', 12.65))
        }

        feature_cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
        input_df = pd.DataFrame([[input_data[col] for col in feature_cols]], columns=feature_cols)

        medv_pred = float(boston_pipeline.predict(input_df)[0])
        value_usd = max(3000.0, medv_pred * 1000.0)

        return jsonify({
            "status": "success",
            "dataset": "boston_housing",
            "predicted_medv": round(medv_pred, 2),
            "predicted_value_usd": round(value_usd, 2),
            "inputs": input_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
