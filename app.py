"""
Multi-Dataset Multiple Linear Regression Analytics Engine
Framework: Flask
Features:
  - Redesigned light UI matching HuanChen1107 AIoT-DIC2 reference layout
  - 50 Startups + Boston Housing built-in datasets
  - Custom CSV Upload with auto multiple regression
  - 10-model comparison, feature selection curves
  - REST API endpoints
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
from flask import Flask, request, jsonify, render_template_string, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multiple Linear Regression Analytics Engine</title>
  <meta name="description" content="CRISP-DM machine learning platform with interactive regression, custom dataset upload, and model comparison.">

  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">

  <!-- Icons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

  <!-- PapaParse (CSV parsing) -->
  <script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>

  <style>
    :root {
      --primary-dark: #343090;
      --primary: #4f46e5;
      --primary-light: #6366f1;
      --primary-subtle: #eef2ff;
      --accent-blue: #3b82f6;
      --accent-emerald: #10b981;
      --accent-emerald-subtle: #ecfdf5;
      --accent-amber: #f59e0b;
      --accent-amber-subtle: #fffbeb;
      --accent-rose: #f43f5e;
      --accent-rose-subtle: #fff1f2;
      --slate-50: #f8fafc; --slate-100: #f1f5f9; --slate-200: #e2e8f0;
      --slate-300: #cbd5e1; --slate-400: #94a3b8; --slate-500: #64748b;
      --slate-600: #475569; --slate-700: #334155; --slate-800: #1e293b;
      --slate-900: #0f172a;
      --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px; --radius-xl: 20px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.05);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -2px rgba(0,0,0,.05);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,.08),0 4px 6px -4px rgba(0,0,0,.04);
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',-apple-system,sans-serif;background:var(--slate-50);color:var(--slate-800);line-height:1.5;-webkit-font-smoothing:antialiased}
    h1,h2,h3,h4{font-family:'Outfit',sans-serif;font-weight:700;color:var(--slate-900)}

    /* ── Sticky Header ── */
    .top-header{background:#fff;border-bottom:1px solid var(--slate-200);padding:14px 0;position:sticky;top:0;z-index:100;box-shadow:0 1px 2px rgba(0,0,0,.03)}
    .header-inner{max-width:1340px;margin:0 auto;padding:0 24px;display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
    .breadcrumbs{font-size:12px;color:var(--slate-400);margin-bottom:2px;display:flex;align-items:center;gap:6px}
    .breadcrumbs a{color:var(--slate-400);text-decoration:none}
    .breadcrumbs a:hover{color:var(--primary)}
    .header-title-group h1{font-size:22px;letter-spacing:-.02em;display:flex;align-items:center;gap:10px}
    .brand-badge{background:linear-gradient(135deg,var(--primary-dark),var(--primary));color:#fff;font-size:10px;font-weight:700;padding:3px 9px;border-radius:9999px;letter-spacing:.04em;text-transform:uppercase}
    .header-actions{display:flex;align-items:center;gap:10px}
    .ref-btn{display:inline-flex;align-items:center;gap:7px;background:var(--slate-100);color:var(--slate-700);border:1px solid var(--slate-200);padding:8px 14px;border-radius:var(--radius-md);font-size:13px;font-weight:600;text-decoration:none;transition:.2s}
    .ref-btn:hover{background:#fff;border-color:var(--primary-light);color:var(--primary);box-shadow:var(--shadow-sm)}
    .live-dot{width:8px;height:8px;border-radius:50%;background:var(--accent-emerald);display:inline-block;animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.6;transform:scale(.85)}}

    /* ── Container & Hero ── */
    .container{max-width:1340px;margin:0 auto;padding:28px 24px 64px}
    .hero-card{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 42%,#3730a3 100%);color:#fff;border-radius:var(--radius-xl);padding:36px 44px;margin-bottom:28px;box-shadow:var(--shadow-lg);position:relative;overflow:hidden}
    .hero-card::after{content:'';position:absolute;top:-70px;right:-70px;width:340px;height:340px;background:radial-gradient(circle,rgba(99,102,241,.22) 0%,transparent 70%);pointer-events:none}
    .hero-card h2{font-size:30px;color:#fff;margin-bottom:10px;line-height:1.25}
    .hero-card p{font-size:14.5px;color:#c7d2fe;max-width:860px;line-height:1.65}
    .hero-tags{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
    .hero-pill{background:rgba(255,255,255,.12);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.18);color:#e0e7ff;font-size:12px;font-weight:500;padding:5px 12px;border-radius:9999px;display:flex;align-items:center;gap:6px}

    /* ── Tab Bar ── */
    .tab-bar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
    .tab-btn{padding:10px 18px;background:var(--slate-50);border:1px solid var(--slate-200);border-radius:var(--radius-md);font-size:13.5px;font-weight:600;color:var(--slate-700);cursor:pointer;display:inline-flex;align-items:center;gap:8px;transition:.2s;font-family:'Inter',sans-serif}
    .tab-btn:hover{background:var(--primary-subtle);border-color:var(--primary-light);color:var(--primary)}
    .tab-btn.active{background:var(--primary-dark);border-color:var(--primary-dark);color:#fff;box-shadow:0 4px 10px rgba(52,48,144,.25)}
    .tab-section{display:none}.tab-section.active{display:block}

    /* ── Cards ── */
    .card{background:#fff;border-radius:var(--radius-lg);border:1px solid var(--slate-200);box-shadow:var(--shadow-sm);padding:28px;transition:.2s}
    .card:hover{box-shadow:var(--shadow-md)}
    .card-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:12px}
    .card-title-group h3{font-size:17px;display:flex;align-items:center;gap:8px}
    .card-subtitle{font-size:13px;color:var(--slate-500);margin-top:3px}
    .badge-tag{font-size:11px;font-weight:600;padding:4px 10px;border-radius:9999px;border:1px solid}
    .badge-indigo{background:var(--primary-subtle);color:var(--primary);border-color:#c7d2fe}
    .badge-emerald{background:var(--accent-emerald-subtle);color:#065f46;border-color:#a7f3d0}
    .badge-amber{background:var(--accent-amber-subtle);color:#92400e;border-color:#fde68a}
    .badge-rose{background:var(--accent-rose-subtle);color:#9f1239;border-color:#fecdd3}

    /* ── Grid layouts ── */
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .metrics-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px}
    @media(max-width:860px){.grid-2{grid-template-columns:1fr}.grid-3{grid-template-columns:1fr 1fr}}
    @media(max-width:540px){.grid-3{grid-template-columns:1fr}}

    /* ── Metric Cards ── */
    .metric-card{background:#fff;border-radius:var(--radius-lg);border:1px solid var(--slate-200);padding:20px;position:relative;overflow:hidden;box-shadow:var(--shadow-sm);transition:.2s}
    .metric-card:hover{transform:translateY(-2px);box-shadow:var(--shadow-md)}
    .metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:var(--primary)}
    .metric-card.sweet::before{background:var(--accent-emerald)}
    .metric-card.amber::before{background:var(--accent-amber)}
    .metric-card.rose::before{background:var(--accent-rose)}
    .metric-title{font-size:11px;font-weight:700;color:var(--slate-500);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
    .metric-val{font-size:26px;font-family:'Outfit',sans-serif;font-weight:700;color:var(--slate-900);line-height:1.1;margin-bottom:4px}
    .metric-note{font-size:12px;color:var(--slate-500)}

    /* ── Slider ── */
    .slider-group{margin-bottom:20px}
    .slider-label-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:13px}
    .slider-label{font-weight:600;color:var(--slate-700);display:flex;align-items:center;gap:8px}
    .slider-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
    .slider-val{font-family:'Fira Code',monospace;font-size:13px;font-weight:600;color:var(--primary)}
    input[type=range]{width:100%;accent-color:var(--primary);cursor:pointer;height:5px;border-radius:9999px}
    .slider-ticks{display:flex;justify-content:space-between;font-size:10.5px;color:var(--slate-400);margin-top:3px;font-family:'Fira Code',monospace}

    /* ── State radio ── */
    .state-group{display:flex;gap:8px;margin-top:4px}
    .state-btn{padding:9px 16px;border:1.5px solid var(--slate-200);border-radius:var(--radius-md);font-size:13px;font-weight:600;color:var(--slate-600);cursor:pointer;transition:.2s;background:#fff}
    .state-btn:hover{border-color:var(--primary);color:var(--primary)}
    .state-btn.active{background:var(--primary);border-color:var(--primary);color:#fff;box-shadow:0 4px 10px rgba(79,70,229,.25)}

    /* ── Prediction Result ── */
    .pred-output-card{background:linear-gradient(135deg,#0f172a,#1e1b4b);border-radius:var(--radius-lg);padding:24px;color:#fff;text-align:center;margin-bottom:16px}
    .pred-label{font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
    .pred-amount{font-size:48px;font-family:'Outfit',sans-serif;font-weight:800;color:#fff;line-height:1;margin-bottom:6px;background:linear-gradient(90deg,#a5b4fc,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .pred-badge{display:inline-block;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-radius:9999px;padding:4px 12px;font-size:12px;color:#c7d2fe;font-weight:500}

    /* ── Breakdown bars ── */
    .breakdown-item{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:13px}
    .breakdown-label{color:var(--slate-600)}
    .breakdown-val{font-family:'Fira Code',monospace;font-weight:600}
    .breakdown-bar-wrap{width:100%;background:var(--slate-100);border-radius:9999px;height:6px;margin-top:4px;margin-bottom:12px;overflow:hidden}
    .breakdown-bar{height:6px;border-radius:9999px;transition:width .4s ease}

    /* ── Preset buttons ── */
    .preset-group{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
    .preset-btn{padding:7px 14px;border:1px solid var(--slate-200);border-radius:var(--radius-md);font-size:12.5px;font-weight:600;color:var(--slate-600);cursor:pointer;background:#fff;transition:.2s;font-family:'Inter',sans-serif}
    .preset-btn:hover{border-color:var(--primary);color:var(--primary);background:var(--primary-subtle)}

    /* ── Figure image ── */
    .fig-wrap{background:var(--slate-50);border:1px solid var(--slate-100);border-radius:var(--radius-lg);padding:12px;overflow:hidden}
    .fig-wrap img{max-width:100%;height:auto;border-radius:var(--radius-md)}

    /* ── Custom Upload Section ── */
    .upload-zone{border:2px dashed var(--slate-300);border-radius:var(--radius-lg);padding:40px 24px;text-align:center;cursor:pointer;transition:.2s;background:#fff;position:relative}
    .upload-zone:hover,.upload-zone.dragging{border-color:var(--primary);background:var(--primary-subtle)}
    .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
    .upload-icon{font-size:36px;color:var(--slate-300);margin-bottom:12px}
    .upload-zone:hover .upload-icon,.upload-zone.dragging .upload-icon{color:var(--primary)}
    .upload-title{font-size:16px;font-weight:700;color:var(--slate-700);margin-bottom:4px;font-family:'Outfit',sans-serif}
    .upload-sub{font-size:13px;color:var(--slate-400)}

    /* CSV Preview Table */
    .table-wrap{overflow-x:auto;border-radius:var(--radius-md);border:1px solid var(--slate-200)}
    table{width:100%;border-collapse:collapse;font-size:13px}
    thead{background:var(--slate-900)}
    thead th{padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:var(--slate-300);text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
    tbody tr:nth-child(even){background:var(--slate-50)}
    tbody tr:hover{background:#eef2ff}
    tbody td{padding:9px 14px;color:var(--slate-700);font-family:'Fira Code',monospace;font-size:12.5px;white-space:nowrap}

    /* Column selectors */
    .col-selector-grid{display:flex;flex-direction:column;gap:14px}
    .col-select-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
    .col-select-label{font-size:13px;font-weight:600;color:var(--slate-700);min-width:100px}
    select.col-pick{padding:8px 12px;border:1.5px solid var(--slate-200);border-radius:var(--radius-md);font-size:13px;color:var(--slate-800);background:#fff;cursor:pointer;transition:.2s;font-family:'Inter',sans-serif}
    select.col-pick:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px rgba(79,70,229,.1)}
    .checkboxes-grid{display:flex;flex-wrap:wrap;gap:8px;max-width:720px}
    .feat-checkbox{display:flex;align-items:center;gap:7px;padding:6px 12px;border:1.5px solid var(--slate-200);border-radius:var(--radius-md);cursor:pointer;transition:.2s;font-size:13px;color:var(--slate-700);background:#fff;font-weight:500}
    .feat-checkbox:hover{border-color:var(--primary);color:var(--primary);background:var(--primary-subtle)}
    .feat-checkbox input{accent-color:var(--primary);width:14px;height:14px;cursor:pointer}
    .feat-checkbox.selected{border-color:var(--primary);background:var(--primary-subtle);color:var(--primary)}

    /* Run button */
    .run-btn{display:inline-flex;align-items:center;gap:10px;background:linear-gradient(135deg,var(--primary-dark),var(--primary));color:#fff;border:none;padding:13px 28px;border-radius:var(--radius-lg);font-size:14px;font-weight:700;cursor:pointer;transition:.2s;font-family:'Outfit',sans-serif;box-shadow:0 4px 12px rgba(79,70,229,.3)}
    .run-btn:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(79,70,229,.35)}
    .run-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

    /* Coeff table */
    .coeff-badge{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:12px;font-weight:700;font-family:'Fira Code',monospace}
    .coeff-pos{background:#ecfdf5;color:#065f46}
    .coeff-neg{background:#fff1f2;color:#9f1239}

    /* Results section */
    .results-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:24px}

    /* Separator */
    .section-divider{height:1px;background:var(--slate-200);margin:24px 0}

    /* Toast */
    #toast{position:fixed;bottom:28px;right:28px;background:var(--slate-900);color:#fff;padding:12px 20px;border-radius:var(--radius-lg);font-size:13.5px;font-weight:500;box-shadow:var(--shadow-lg);z-index:999;transform:translateY(80px);opacity:0;transition:.3s}
    #toast.show{transform:translateY(0);opacity:1}
    #toast.error{background:#9f1239}
    #toast.success{background:#065f46}

    /* Responsive */
    @media(max-width:640px){.hero-card{padding:24px 20px}.hero-card h2{font-size:22px}.tab-btn{font-size:12.5px;padding:8px 13px}}
  </style>
</head>
<body>

<!-- ══════════════════════════ STICKY HEADER ══════════════════════════ -->
<header class="top-header">
  <div class="header-inner">
    <div class="header-title-group">
      <div class="breadcrumbs">
        <a href="#">AIoT</a> <i class="fa fa-chevron-right" style="font-size:9px"></i>
        <a href="#">DIC-2</a> <i class="fa fa-chevron-right" style="font-size:9px"></i>
        <span>Multiple Regression</span>
      </div>
      <h1>
        <span class="live-dot"></span>
        Multiple Regression Analytics Engine
        <span class="brand-badge">CRISP-DM</span>
      </h1>
    </div>
    <div class="header-actions">
      <a href="https://github.com/zxc22872580/AIOT_DIC-2_MultipleRegression" target="_blank" class="ref-btn">
        <i class="fab fa-github"></i> Source Code
      </a>
      <a href="/api/health" target="_blank" class="ref-btn">
        <i class="fa fa-heartbeat"></i> API Health
      </a>
    </div>
  </div>
</header>

<!-- ══════════════════════════ MAIN CONTENT ══════════════════════════ -->
<main class="container">

  <!-- Hero Banner -->
  <div class="hero-card">
    <h2><i class="fa fa-chart-line" style="margin-right:10px;opacity:.85"></i>Multiple Linear Regression Analytics</h2>
    <p>End-to-end machine learning platform following the <strong style="color:#e0e7ff">CRISP-DM</strong> lifecycle. Explore built-in datasets (50 Startups &amp; Boston Housing), benchmark 10 regression models, inspect Sequential Forward Feature Selection curves, and run regression analysis on <strong style="color:#e0e7ff">any custom CSV dataset</strong> you upload.</p>
    <div class="hero-tags">
      <span class="hero-pill"><i class="fa fa-robot"></i> scikit-learn Pipelines</span>
      <span class="hero-pill"><i class="fa fa-upload"></i> Custom CSV Upload</span>
      <span class="hero-pill"><i class="fa fa-chart-bar"></i> 10-Model Comparison</span>
      <span class="hero-pill"><i class="fa fa-filter"></i> Feature Selection</span>
      <span class="hero-pill"><i class="fa fa-plug"></i> REST API</span>
    </div>
  </div>

  <!-- ── Tab Navigation ── -->
  <div class="tab-bar">
    <button class="tab-btn active" id="tb-predict" onclick="switchTab('predict')">
      <i class="fa fa-sliders-h"></i> Interactive Predictor
    </button>
    <button class="tab-btn" id="tb-bench" onclick="switchTab('bench')">
      <i class="fa fa-trophy"></i> Model Comparison <span style="font-size:11px;background:rgba(0,0,0,.08);padding:2px 7px;border-radius:9999px">10 models</span>
    </button>
    <button class="tab-btn" id="tb-feats" onclick="switchTab('feats')">
      <i class="fa fa-project-diagram"></i> Feature Selection Curves
    </button>
    <button class="tab-btn" id="tb-custom" onclick="switchTab('custom')">
      <i class="fa fa-file-csv"></i> Custom Dataset Analysis
    </button>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- TAB 1 — INTERACTIVE PREDICTOR                     -->
  <!-- ══════════════════════════════════════════════════ -->
  <div id="tab-predict" class="tab-section active">

    <!-- Dataset picker row -->
    <div class="card" style="margin-bottom:18px;padding:18px 24px">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <span style="font-size:12px;font-weight:700;color:var(--slate-400);text-transform:uppercase;letter-spacing:.05em">選擇資料集</span>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--slate-700)">
          <input type="radio" name="ds" value="startups" checked onchange="onDatasetChange('startups')" style="accent-color:var(--primary)">
          📈 50 Startups
        </label>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--slate-700)">
          <input type="radio" name="ds" value="boston" onchange="onDatasetChange('boston')" style="accent-color:var(--primary)">
          🏡 Boston Housing
        </label>
        <span id="ds-r2-badge" class="badge-tag badge-indigo" style="margin-left:auto">R² = 0.8987</span>
      </div>
    </div>

    <div class="grid-2">
      <!-- LEFT: Controls -->
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3 id="ctrl-title"><i class="fa fa-sliders-h" style="color:var(--primary)"></i> Investment Parameters (50 Startups)</h3>
            <p class="card-subtitle" id="ctrl-sub">Adjust feature values to simulate profit predictions</p>
          </div>
          <div class="preset-group" id="preset-container"></div>
        </div>

        <!-- 50 Startups form -->
        <div id="form-startups">
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#10b981"></span>R&D Spend</span>
              <span class="slider-val" id="rd-val">$120,000</span>
            </div>
            <input type="range" id="rd-spend" min="0" max="250000" step="5000" value="120000" oninput="triggerStartupsPrediction()">
            <div class="slider-ticks"><span>$0</span><span>$125k</span><span>$250k</span></div>
          </div>
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#3b82f6"></span>Marketing Spend</span>
              <span class="slider-val" id="mkt-val">$250,000</span>
            </div>
            <input type="range" id="mkt-spend" min="0" max="500000" step="10000" value="250000" oninput="triggerStartupsPrediction()">
            <div class="slider-ticks"><span>$0</span><span>$250k</span><span>$500k</span></div>
          </div>
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#f59e0b"></span>Administration Spend</span>
              <span class="slider-val" id="admin-val">$115,000</span>
            </div>
            <input type="range" id="admin-spend" min="0" max="250000" step="5000" value="115000" oninput="triggerStartupsPrediction()">
            <div class="slider-ticks"><span>$0</span><span>$125k</span><span>$250k</span></div>
          </div>
          <div style="margin-bottom:16px">
            <div class="slider-label" style="margin-bottom:10px"><i class="fa fa-map-marker-alt" style="color:var(--primary)"></i> State Headquarters</div>
            <div class="state-group">
              <button class="state-btn active" onclick="selectState(this,'California')">California</button>
              <button class="state-btn" onclick="selectState(this,'New York')">New York</button>
              <button class="state-btn" onclick="selectState(this,'Florida')">Florida</button>
            </div>
            <input type="hidden" id="state-val" value="California">
          </div>
        </div>

        <!-- Boston Housing form -->
        <div id="form-boston" style="display:none">
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#10b981"></span>Avg Rooms (RM)</span>
              <span class="slider-val" id="rm-val">6.28 rooms</span>
            </div>
            <input type="range" id="b-rm" min="3" max="8.8" step="0.1" value="6.28" oninput="triggerBostonPrediction()">
            <div class="slider-ticks"><span>3.0</span><span>5.9</span><span>8.8</span></div>
          </div>
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#f43f5e"></span>Low-Status Pop. (LSTAT %)</span>
              <span class="slider-val" id="lstat-val">12.7%</span>
            </div>
            <input type="range" id="b-lstat" min="1" max="38" step="0.5" value="12.65" oninput="triggerBostonPrediction()">
            <div class="slider-ticks"><span>1%</span><span>19.5%</span><span>38%</span></div>
          </div>
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#6366f1"></span>Pupil-Teacher Ratio (PTRATIO)</span>
              <span class="slider-val" id="ptratio-val">18.5</span>
            </div>
            <input type="range" id="b-ptratio" min="12" max="22" step="0.1" value="18.46" oninput="triggerBostonPrediction()">
            <div class="slider-ticks"><span>12</span><span>17</span><span>22</span></div>
          </div>
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#3b82f6"></span>Distance to Employment (DIS)</span>
              <span class="slider-val" id="dis-val">3.8 km</span>
            </div>
            <input type="range" id="b-dis" min="1" max="12" step="0.1" value="3.79" oninput="triggerBostonPrediction()">
            <div class="slider-ticks"><span>1</span><span>6.5</span><span>12</span></div>
          </div>
          <div class="slider-group">
            <div class="slider-label-row">
              <span class="slider-label"><span class="slider-dot" style="background:#f59e0b"></span>Property Tax Rate (TAX)</span>
              <span class="slider-val" id="tax-val">$408</span>
            </div>
            <input type="range" id="b-tax" min="187" max="711" step="5" value="408" oninput="triggerBostonPrediction()">
            <div class="slider-ticks"><span>$187</span><span>$449</span><span>$711</span></div>
          </div>
          <div style="margin-bottom:4px">
            <div class="preset-group">
              <button class="preset-btn" onclick="applyBostonPreset('suburban')">🌳 Suburban</button>
              <button class="preset-btn" onclick="applyBostonPreset('urban')">🏙️ Urban</button>
              <button class="preset-btn" onclick="applyBostonPreset('affordable')">💰 Affordable</button>
            </div>
          </div>
        </div>
      </div>

      <!-- RIGHT: Prediction Output -->
      <div>
        <div class="pred-output-card" style="margin-bottom:18px">
          <div class="pred-label" id="pred-label">PREDICTED ANNUAL PROFIT</div>
          <div class="pred-amount" id="pred-output">—</div>
          <span class="pred-badge" id="pred-badge">R² = 0.8987 · scikit-learn Pipeline</span>
        </div>

        <div class="card">
          <div class="card-header" style="margin-bottom:14px">
            <div class="card-title-group">
              <h3><i class="fa fa-chart-pie" style="color:var(--accent-amber)"></i> Key Predictor Breakdown</h3>
            </div>
          </div>
          <div id="breakdown-container">
            <div style="color:var(--slate-400);font-size:13px;text-align:center;padding:20px">Adjust sliders to see breakdown...</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- TAB 2 — MODEL COMPARISON                          -->
  <!-- ══════════════════════════════════════════════════ -->
  <div id="tab-bench" class="tab-section">

    <div class="card" style="margin-bottom:18px;padding:18px 24px">
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
        <span style="font-size:12px;font-weight:700;color:var(--slate-400);text-transform:uppercase;letter-spacing:.05em">Dataset for Comparison</span>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--slate-700)">
          <input type="radio" name="bench-ds" value="startups" checked onchange="updateFig1('startups')" style="accent-color:var(--primary)">
          📈 50 Startups
        </label>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--slate-700)">
          <input type="radio" name="bench-ds" value="boston" onchange="updateFig1('boston')" style="accent-color:var(--primary)">
          🏡 Boston Housing
        </label>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <h3 id="fig1-title"><i class="fa fa-trophy" style="color:var(--accent-amber)"></i> Comparison of 10 Regression Models</h3>
          <p class="card-subtitle" id="fig1-desc">Models evaluated and ranked by MSE ascending</p>
        </div>
        <span class="badge-tag badge-indigo">Analysis Fig 1</span>
      </div>
      <div class="fig-wrap">
        <img id="fig1-img" src="/figures/figure1_model_comparison.png" alt="Model Comparison" style="max-width:100%;height:auto">
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- TAB 3 — FEATURE SELECTION CURVES                  -->
  <!-- ══════════════════════════════════════════════════ -->
  <div id="tab-feats" class="tab-section">

    <div class="card" style="margin-bottom:18px;padding:18px 24px">
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
        <span style="font-size:12px;font-weight:700;color:var(--slate-400);text-transform:uppercase;letter-spacing:.05em">Dataset</span>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--slate-700)">
          <input type="radio" name="feat-ds" value="startups" checked onchange="updateFig2('startups')" style="accent-color:var(--primary)">
          📈 50 Startups
        </label>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--slate-700)">
          <input type="radio" name="feat-ds" value="boston" onchange="updateFig2('boston')" style="accent-color:var(--primary)">
          🏡 Boston Housing
        </label>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <h3 id="fig2-title"><i class="fa fa-project-diagram" style="color:var(--accent-emerald)"></i> Sequential Forward Feature Selection Curves</h3>
          <p class="card-subtitle" id="fig2-desc">Progression of RMSE and R² as features are added step-by-step</p>
        </div>
        <span class="badge-tag badge-emerald">Analysis Fig 2</span>
      </div>
      <div class="fig-wrap">
        <img id="fig2-img" src="/figures/figure2_feature_selection.png" alt="Feature Selection Curves" style="max-width:100%;height:auto">
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- TAB 4 — CUSTOM DATASET ANALYSIS                   -->
  <!-- ══════════════════════════════════════════════════ -->
  <div id="tab-custom" class="tab-section">

    <div class="grid-2" style="margin-bottom:24px">
      <!-- Upload zone -->
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3><i class="fa fa-file-csv" style="color:var(--primary)"></i> Upload Your Dataset</h3>
            <p class="card-subtitle">CSV files up to 10 MB · Numeric columns are auto-detected</p>
          </div>
        </div>
        <div class="upload-zone" id="upload-zone" ondragover="event.preventDefault();this.classList.add('dragging')" ondragleave="this.classList.remove('dragging')" ondrop="handleDrop(event)">
          <input type="file" id="csv-input" accept=".csv" onchange="handleFile(this.files[0])">
          <div class="upload-icon"><i class="fa fa-cloud-upload-alt"></i></div>
          <div class="upload-title">Drag & Drop or Click to Upload</div>
          <div class="upload-sub">Supports any CSV with a header row. Numeric columns will be used for regression.</div>
        </div>
        <div id="file-info" style="margin-top:14px;font-size:13px;color:var(--slate-500);display:none">
          <i class="fa fa-check-circle" style="color:var(--accent-emerald)"></i>
          <span id="file-name-label"></span> — <span id="file-stats-label"></span>
        </div>
      </div>

      <!-- Column configuration -->
      <div class="card" id="col-config-card" style="opacity:.4;pointer-events:none">
        <div class="card-header">
          <div class="card-title-group">
            <h3><i class="fa fa-columns" style="color:var(--accent-amber)"></i> Configure Columns</h3>
            <p class="card-subtitle">Select target variable and feature columns</p>
          </div>
        </div>
        <div class="col-selector-grid">
          <div class="col-select-row">
            <span class="col-select-label">🎯 Target (Y)</span>
            <select id="target-col" class="col-pick" style="min-width:200px" onchange="updateFeatureList()">
              <option value="">— upload CSV first —</option>
            </select>
          </div>
          <div>
            <span class="col-select-label" style="display:block;margin-bottom:10px">📊 Features (X) — select one or more</span>
            <div class="checkboxes-grid" id="feature-checkboxes"></div>
          </div>
        </div>
        <div class="section-divider"></div>
        <button class="run-btn" id="run-btn" onclick="runCustomRegression()" disabled>
          <i class="fa fa-play"></i> Run Regression Analysis
        </button>
      </div>
    </div>

    <!-- Data Preview -->
    <div class="card" id="preview-card" style="display:none;margin-bottom:24px">
      <div class="card-header">
        <div class="card-title-group">
          <h3><i class="fa fa-table" style="color:var(--accent-blue)"></i> Data Preview</h3>
          <p class="card-subtitle" id="preview-sub">First 8 rows</p>
        </div>
        <span class="badge-tag badge-indigo" id="preview-shape-badge"></span>
      </div>
      <div class="table-wrap" id="preview-table"></div>
    </div>

    <!-- Analysis Results -->
    <div id="custom-results" style="display:none">
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3><i class="fa fa-chart-bar" style="color:var(--accent-emerald)"></i> Regression Analysis Results</h3>
            <p class="card-subtitle" id="result-dataset-label"></p>
          </div>
          <span class="badge-tag badge-emerald">Multiple Linear Regression</span>
        </div>

        <!-- Metric cards -->
        <div class="results-grid" id="results-metrics"></div>
        <div class="section-divider"></div>

        <!-- Coefficients Table -->
        <h3 style="font-size:15px;margin-bottom:14px"><i class="fa fa-list" style="color:var(--primary)"></i> Model Coefficients</h3>
        <div class="table-wrap" id="coeff-table"></div>

        <div class="section-divider"></div>

        <!-- Chart -->
        <div class="grid-2">
          <div>
            <h3 style="font-size:15px;margin-bottom:14px"><i class="fa fa-chart-scatter" style="color:var(--accent-amber)"></i> Actual vs. Predicted</h3>
            <div style="background:var(--slate-50);border-radius:var(--radius-md);padding:16px;border:1px solid var(--slate-100)">
              <canvas id="scatter-chart" height="280"></canvas>
            </div>
          </div>
          <div>
            <h3 style="font-size:15px;margin-bottom:14px"><i class="fa fa-wave-square" style="color:var(--accent-rose)"></i> Residuals Distribution</h3>
            <div style="background:var(--slate-50);border-radius:var(--radius-md);padding:16px;border:1px solid var(--slate-100)">
              <canvas id="residuals-chart" height="280"></canvas>
            </div>
          </div>
        </div>

        <div class="section-divider"></div>

        <!-- Feature Importance Bar Chart -->
        <h3 style="font-size:15px;margin-bottom:14px"><i class="fa fa-star" style="color:var(--accent-amber)"></i> Feature Importance (|Standardized Coefficient|)</h3>
        <div style="background:var(--slate-50);border-radius:var(--radius-md);padding:16px;border:1px solid var(--slate-100)">
          <canvas id="importance-chart" height="200"></canvas>
        </div>

        <!-- Custom Prediction -->
        <div class="section-divider"></div>
        <h3 style="font-size:15px;margin-bottom:14px"><i class="fa fa-magic" style="color:var(--primary)"></i> Predict with Custom Inputs</h3>
        <div id="custom-input-fields" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px"></div>
        <button class="run-btn" style="background:linear-gradient(135deg,#065f46,#10b981);box-shadow:0 4px 12px rgba(16,185,129,.3)" onclick="predictCustom()">
          <i class="fa fa-calculator"></i> Predict
        </button>
        <div id="custom-pred-result" style="margin-top:18px;display:none">
          <div class="pred-output-card" style="max-width:380px">
            <div class="pred-label">PREDICTED VALUE</div>
            <div class="pred-amount" id="custom-pred-value">—</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- API Footer -->
  <div style="margin-top:32px;background:#fff;border:1px solid var(--slate-200);border-radius:var(--radius-lg);padding:16px 24px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;font-size:13px;color:var(--slate-500)">
    <div>
      <span style="font-weight:700;color:var(--slate-700)">REST API Endpoints: </span>
      <code style="background:var(--slate-100);padding:2px 8px;border-radius:6px;color:var(--primary)">POST /api/predict/startups</code>
      <span style="margin:0 8px;color:var(--slate-300)">|</span>
      <code style="background:var(--slate-100);padding:2px 8px;border-radius:6px;color:var(--primary)">POST /api/predict/boston</code>
      <span style="margin:0 8px;color:var(--slate-300)">|</span>
      <code style="background:var(--slate-100);padding:2px 8px;border-radius:6px;color:var(--accent-emerald)">POST /api/analyze</code>
    </div>
    <span style="color:var(--accent-emerald);font-weight:600"><i class="fa fa-circle" style="font-size:8px;margin-right:5px"></i>Dual Models Active &amp; Serving</span>
  </div>
</main>

<!-- Toast notification -->
<div id="toast"></div>

<script>
// ═══════════════════════════════════════
//  UTILITY
// ═══════════════════════════════════════
let scatterChart = null, residualsChart = null, importanceChart = null;

function showToast(msg, type='') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show ' + type;
  setTimeout(() => t.className = '', 3000);
}

function switchTab(id) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.getElementById('tb-' + id).classList.add('active');
}

// ═══════════════════════════════════════
//  TAB 1 — PREDICTOR
// ═══════════════════════════════════════
let currentDataset = 'startups';

function onDatasetChange(ds) {
  currentDataset = ds;
  const isBoston = ds === 'boston';
  document.getElementById('form-startups').style.display = isBoston ? 'none' : 'block';
  document.getElementById('form-boston').style.display   = isBoston ? 'block' : 'none';
  document.getElementById('ctrl-title').innerHTML = isBoston
    ? '<i class="fa fa-home" style="color:var(--primary)"></i> Housing Parameters (Boston)'
    : '<i class="fa fa-sliders-h" style="color:var(--primary)"></i> Investment Parameters (50 Startups)';
  document.getElementById('ctrl-sub').textContent = isBoston
    ? 'Adjust housing features to predict median home value'
    : 'Adjust feature values to simulate profit predictions';
  document.getElementById('pred-label').textContent = isBoston ? 'PREDICTED HOME VALUE' : 'PREDICTED ANNUAL PROFIT';
  const r2 = isBoston ? 'R² = 0.67' : 'R² = 0.8987';
  document.getElementById('ds-r2-badge').textContent = r2;
  document.getElementById('pred-badge').textContent = r2 + ' · scikit-learn Pipeline';
  document.getElementById('pred-output').textContent = '—';
  document.getElementById('breakdown-container').innerHTML = '<div style="color:var(--slate-400);font-size:13px;text-align:center;padding:20px">Adjust sliders to see breakdown...</div>';

  const pc = document.getElementById('preset-container');
  if (!isBoston) {
    pc.innerHTML = `
      <button class="preset-btn" onclick="applyStartupPreset('tech')">🤖 AI Tech</button>
      <button class="preset-btn" onclick="applyStartupPreset('d2c')">🛍️ Marketing D2C</button>
      <button class="preset-btn" onclick="applyStartupPreset('lean')">🌿 Lean</button>`;
  } else {
    pc.innerHTML = '';
  }

  if (isBoston) triggerBostonPrediction();
  else triggerStartupsPrediction();
}

function applyStartupPreset(name) {
  const p = {tech:{rd:165000,mkt:300000,admin:100000,state:'California'}, d2c:{rd:45000,mkt:480000,admin:130000,state:'New York'}, lean:{rd:90000,mkt:80000,admin:50000,state:'Florida'}}[name];
  document.getElementById('rd-spend').value = p.rd;
  document.getElementById('mkt-spend').value = p.mkt;
  document.getElementById('admin-spend').value = p.admin;
  selectStateByName(p.state);
  triggerStartupsPrediction();
}

function selectState(el, name) {
  document.querySelectorAll('.state-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('state-val').value = name;
  triggerStartupsPrediction();
}
function selectStateByName(name) {
  document.querySelectorAll('.state-btn').forEach(b => {
    b.classList.remove('active');
    if (b.textContent.trim() === name) b.classList.add('active');
  });
  document.getElementById('state-val').value = name;
}

async function triggerStartupsPrediction() {
  const rd = +document.getElementById('rd-spend').value;
  const mkt = +document.getElementById('mkt-spend').value;
  const admin = +document.getElementById('admin-spend').value;
  const state = document.getElementById('state-val').value;
  document.getElementById('rd-val').textContent = '$' + Math.round(rd).toLocaleString();
  document.getElementById('mkt-val').textContent = '$' + Math.round(mkt).toLocaleString();
  document.getElementById('admin-val').textContent = '$' + Math.round(admin).toLocaleString();

  let profit;
  try {
    const r = await fetch('/api/predict/startups', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rd_spend:rd,administration:admin,marketing_spend:mkt,state})});
    const d = await r.json();
    profit = d.status === 'success' ? d.predicted_profit : null;
  } catch(e){}
  if (!profit) profit = 54028.04 + 0.80563*rd - 0.068788*admin + 0.029855*mkt;
  document.getElementById('pred-output').textContent = '$' + Math.round(profit).toLocaleString();

  const rdEff = 0.80563 * rd, mktEff = 0.029855 * mkt, admEff = -0.068788 * admin;
  const total = Math.abs(rdEff) + Math.abs(mktEff) + Math.abs(admEff) || 1;
  document.getElementById('breakdown-container').innerHTML = `
    ${breakdown('R&D Spend Contribution (β = +0.81)', rdEff, '#10b981', Math.abs(rdEff)/total*100)}
    ${breakdown('Marketing Contribution (β = +0.03)', mktEff, '#3b82f6', Math.abs(mktEff)/total*100)}
    ${breakdown('Admin Overhead Drag (β = -0.07)', admEff, '#f59e0b', Math.abs(admEff)/total*100, true)}
    <div style="border-top:1px solid var(--slate-100);padding-top:10px;font-size:12px;color:var(--slate-400)">
      Baseline Intercept (β₀): $54,028.04
    </div>`;
}

function breakdown(label, val, color, pct, forceNeg=false) {
  const isPos = val >= 0;
  const sign = isPos && !forceNeg ? '+' : '';
  const dispColor = isPos && !forceNeg ? '#065f46' : '#9f1239';
  return `<div style="margin-bottom:12px">
    <div class="breakdown-item">
      <span class="breakdown-label">${label}</span>
      <span class="breakdown-val" style="color:${dispColor}">${sign}$${Math.round(Math.abs(val)).toLocaleString()}</span>
    </div>
    <div class="breakdown-bar-wrap"><div class="breakdown-bar" style="width:${Math.max(2,pct)}%;background:${color}"></div></div>
  </div>`;
}

function applyBostonPreset(name) {
  const p = {suburban:{rm:7.8,lstat:4.5,ptratio:14.5,dis:5.5,tax:280}, urban:{rm:6.2,lstat:12,ptratio:18,dis:2.8,tax:420}, affordable:{rm:5.2,lstat:24,ptratio:20.2,dis:2.1,tax:660}}[name];
  document.getElementById('b-rm').value = p.rm;
  document.getElementById('b-lstat').value = p.lstat;
  document.getElementById('b-ptratio').value = p.ptratio;
  document.getElementById('b-dis').value = p.dis;
  document.getElementById('b-tax').value = p.tax;
  triggerBostonPrediction();
}

async function triggerBostonPrediction() {
  const rm = +document.getElementById('b-rm').value;
  const lstat = +document.getElementById('b-lstat').value;
  const ptratio = +document.getElementById('b-ptratio').value;
  const dis = +document.getElementById('b-dis').value;
  const tax = +document.getElementById('b-tax').value;
  document.getElementById('rm-val').textContent = rm.toFixed(2) + ' rooms';
  document.getElementById('lstat-val').textContent = lstat.toFixed(1) + '%';
  document.getElementById('ptratio-val').textContent = ptratio.toFixed(1);
  document.getElementById('dis-val').textContent = dis.toFixed(1) + ' km';
  document.getElementById('tax-val').textContent = '$' + Math.round(tax);

  let value;
  try {
    const r = await fetch('/api/predict/boston', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rm,lstat,ptratio,dis,tax,crim:0.25})});
    const d = await r.json();
    value = d.status === 'success' ? d.predicted_value_usd : null;
  } catch(e){}
  if (!value) {
    const medv = 36.459 + 3.809*(rm-6.28) - 0.547*(lstat-12.65) - 0.952*(ptratio-18.45);
    value = Math.max(3000, medv*1000);
  }
  document.getElementById('pred-output').textContent = '$' + Math.round(value).toLocaleString();

  const rmI = (rm-6.28)*3800, lstatI = (12.65-lstat)*550, ptI = (18.45-ptratio)*950;
  const total = Math.abs(rmI)+Math.abs(lstatI)+Math.abs(ptI)||1;
  document.getElementById('breakdown-container').innerHTML = `
    ${breakdown('Rooms (RM vs Avg 6.28)', rmI, '#10b981', Math.abs(rmI)/total*100)}
    ${breakdown('Socioeconomic Status (LSTAT)', lstatI, '#f43f5e', Math.abs(lstatI)/total*100)}
    ${breakdown('School Quality (PTRATIO)', ptI, '#6366f1', Math.abs(ptI)/total*100)}
    <div style="border-top:1px solid var(--slate-100);padding-top:10px;font-size:12px;color:var(--slate-400)">
      Baseline Town Median: $22,532.80
    </div>`;
}

// ═══════════════════════════════════════
//  TAB 2 & 3 — FIGURES
// ═══════════════════════════════════════
function updateFig1(ds) {
  const img = document.getElementById('fig1-img');
  const t = document.getElementById('fig1-title');
  if (ds==='boston') {
    img.src = '/figures/boston_figure1_model_comparison.png';
    t.innerHTML = '<i class="fa fa-trophy" style="color:var(--accent-amber)"></i> Boston Housing — 10 Model Comparison';
  } else {
    img.src = '/figures/figure1_model_comparison.png';
    t.innerHTML = '<i class="fa fa-trophy" style="color:var(--accent-amber)"></i> 50 Startups — 10 Model Comparison';
  }
}
function updateFig2(ds) {
  const img = document.getElementById('fig2-img');
  const t = document.getElementById('fig2-title');
  if (ds==='boston') {
    img.src = '/figures/boston_figure2_feature_selection.png';
    t.innerHTML = '<i class="fa fa-project-diagram" style="color:var(--accent-emerald)"></i> Boston Housing — Feature Selection Curves';
  } else {
    img.src = '/figures/figure2_feature_selection.png';
    t.innerHTML = '<i class="fa fa-project-diagram" style="color:var(--accent-emerald)"></i> 50 Startups — Feature Selection Curves';
  }
}

// ═══════════════════════════════════════
//  TAB 4 — CUSTOM DATASET
// ═══════════════════════════════════════
let csvData = null;         // array of row objects
let csvHeaders = [];        // all column names
let numericHeaders = [];    // numeric-only column names
let regressionModel = null; // server response

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('dragging');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
}

function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.csv')) { showToast('Please upload a .csv file', 'error'); return; }
  Papa.parse(file, {
    header: true, skipEmptyLines: true, dynamicTyping: true,
    complete(results) {
      csvData = results.data;
      csvHeaders = results.meta.fields || [];
      numericHeaders = csvHeaders.filter(h => csvData.slice(0,20).every(r => typeof r[h] === 'number' || r[h] === null || r[h] === ''));

      // Show file info
      document.getElementById('file-info').style.display = 'block';
      document.getElementById('file-name-label').textContent = file.name;
      document.getElementById('file-stats-label').textContent = `${csvData.length} rows × ${csvHeaders.length} columns (${numericHeaders.length} numeric)`;

      // Populate target selector
      const tc = document.getElementById('target-col');
      tc.innerHTML = numericHeaders.map(h => `<option value="${h}">${h}</option>`).join('');
      updateFeatureList();

      // Enable config card
      const cc = document.getElementById('col-config-card');
      cc.style.opacity = '1'; cc.style.pointerEvents = 'auto';
      document.getElementById('run-btn').disabled = false;

      // Show preview table
      renderPreview();
      showToast('CSV loaded successfully! ' + csvData.length + ' rows detected.', 'success');
    },
    error(e) { showToast('Error parsing CSV: ' + e.message, 'error'); }
  });
}

function updateFeatureList() {
  const target = document.getElementById('target-col').value;
  const available = numericHeaders.filter(h => h !== target);
  const box = document.getElementById('feature-checkboxes');
  box.innerHTML = available.map(h => `
    <label class="feat-checkbox" id="fc-${h.replace(/\W/g,'_')}">
      <input type="checkbox" value="${h}" checked onchange="this.closest('.feat-checkbox').classList.toggle('selected',this.checked)">
      ${h}
    </label>`).join('');
  box.querySelectorAll('.feat-checkbox').forEach(l => l.classList.add('selected'));
}

function getSelectedFeatures() {
  return [...document.querySelectorAll('#feature-checkboxes input:checked')].map(i => i.value);
}

function renderPreview() {
  if (!csvData || !csvData.length) return;
  const rows = csvData.slice(0, 8);
  const cols = csvHeaders.slice(0, 12); // max 12 cols for readability
  document.getElementById('preview-card').style.display = 'block';
  document.getElementById('preview-sub').textContent = `First ${rows.length} of ${csvData.length} rows`;
  document.getElementById('preview-shape-badge').textContent = `${csvData.length} × ${csvHeaders.length}`;
  document.getElementById('preview-table').innerHTML = `
    <table>
      <thead><tr>${cols.map(h => `<th>${h}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(r => `<tr>${cols.map(h => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>
    </table>`;
}

async function runCustomRegression() {
  const target = document.getElementById('target-col').value;
  const features = getSelectedFeatures();
  if (!target) { showToast('Please select a target column', 'error'); return; }
  if (features.length === 0) { showToast('Please select at least one feature column', 'error'); return; }
  if (!csvData || !csvData.length) { showToast('No data loaded', 'error'); return; }

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Running...';

  // Build payload — send raw CSV rows
  const payload = { target, features, rows: csvData.slice(0, 5000) };

  try {
    const r = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const result = await r.json();
    if (result.status === 'success') {
      regressionModel = result;
      renderResults(result, target, features);
      showToast('Regression complete! R² = ' + result.r2.toFixed(4), 'success');
    } else {
      showToast('Error: ' + result.message, 'error');
    }
  } catch(e) {
    showToast('Network error: ' + e.message, 'error');
  }
  btn.disabled = false;
  btn.innerHTML = '<i class="fa fa-play"></i> Run Regression Analysis';
}

function renderResults(res, target, features) {
  document.getElementById('custom-results').style.display = 'block';
  document.getElementById('result-dataset-label').textContent = `Target: ${target} | Features: ${features.join(', ')}`;

  // Metric cards
  document.getElementById('results-metrics').innerHTML = `
    ${metricCard('R² Score', res.r2.toFixed(4), 'Higher is better (max 1.0)', 'sweet', 'fa-bullseye')}
    ${metricCard('Adj. R²', res.r2_adj.toFixed(4), 'Adjusted for feature count', 'sweet', 'fa-chart-line')}
    ${metricCard('RMSE', fmtNum(res.rmse), 'Root Mean Squared Error', '', 'fa-wave-square')}
    ${metricCard('MAE', fmtNum(res.mae), 'Mean Absolute Error', '', 'fa-ruler')}
    ${metricCard('Training Samples', res.n_train, 'Used for model fitting', 'amber', 'fa-database')}
    ${metricCard('Test Samples', res.n_test, 'Held-out for evaluation', 'amber', 'fa-vial')}`;

  // Coefficient table
  const rows = res.coefficients.map(c => {
    const cls = c.coef >= 0 ? 'coeff-pos' : 'coeff-neg';
    const sign = c.coef >= 0 ? '+' : '';
    return `<tr>
      <td><strong>${c.feature}</strong></td>
      <td><span class="coeff-badge ${cls}">${sign}${c.coef.toFixed(6)}</span></td>
      <td style="font-family:'Fira Code',monospace;font-size:12px">${c.std_coef !== null ? c.std_coef.toFixed(4) : '—'}</td>
    </tr>`;
  });
  document.getElementById('coeff-table').innerHTML = `
    <table>
      <thead><tr><th>Feature</th><th>Coefficient</th><th>Standardized Coef.</th></tr></thead>
      <tbody>
        <tr><td><em>Intercept</em></td><td><span class="coeff-badge coeff-pos">+${res.intercept.toFixed(6)}</span></td><td>—</td></tr>
        ${rows.join('')}
      </tbody>
    </table>`;

  // Charts
  renderScatter(res);
  renderResiduals(res);
  renderImportance(res);

  // Custom input fields
  document.getElementById('custom-pred-result').style.display = 'none';
  document.getElementById('custom-input-fields').innerHTML = features.map(f => `
    <div>
      <label style="display:block;font-size:12px;font-weight:700;color:var(--slate-600);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em">${f}</label>
      <input type="number" id="inp-${f.replace(/\W/g,'_')}" data-feat="${f}"
        value="${(res.feature_means && res.feature_means[f] !== undefined ? +res.feature_means[f].toFixed(4) : 0)}"
        style="width:100%;padding:9px 12px;border:1.5px solid var(--slate-200);border-radius:var(--radius-md);font-size:13px;font-family:'Fira Code',monospace;transition:.2s"
        onfocus="this.style.borderColor='var(--primary)'" onblur="this.style.borderColor='var(--slate-200)'">
    </div>`).join('');

  document.getElementById('custom-results').scrollIntoView({behavior:'smooth',block:'start'});
}

function metricCard(title, val, note, cls='', icon='fa-chart-bar') {
  return `<div class="metric-card ${cls}">
    <div class="metric-title"><span>${title}</span><i class="fa ${icon}" style="color:var(--slate-300)"></i></div>
    <div class="metric-val">${val}</div>
    <div class="metric-note">${note}</div>
  </div>`;
}

function fmtNum(n) {
  if (n === undefined || n === null) return '—';
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined,{maximumFractionDigits:2});
  return (+n).toFixed(4);
}

function renderScatter(res) {
  if (scatterChart) scatterChart.destroy();
  const ctx = document.getElementById('scatter-chart').getContext('2d');
  const data = res.actual.map((a,i) => ({x:a, y:res.predicted[i]}));
  const mn = Math.min(...res.actual), mx = Math.max(...res.actual);
  scatterChart = new Chart(ctx, {
    type:'scatter',
    data:{
      datasets:[
        {label:'Predictions', data, backgroundColor:'rgba(79,70,229,.5)', pointRadius:5, pointHoverRadius:7},
        {label:'Perfect Fit', data:[{x:mn,y:mn},{x:mx,y:mx}], type:'line', borderColor:'#f43f5e', borderDash:[6,4], borderWidth:2, pointRadius:0, fill:false}
      ]
    },
    options:{responsive:true, plugins:{legend:{position:'bottom',labels:{font:{size:12}}}}, scales:{x:{title:{display:true,text:'Actual'}}, y:{title:{display:true,text:'Predicted'}}}}
  });
}

function renderResiduals(res) {
  if (residualsChart) residualsChart.destroy();
  const ctx = document.getElementById('residuals-chart').getContext('2d');
  const residuals = res.actual.map((a,i) => a - res.predicted[i]);
  const bins = 20;
  const mn = Math.min(...residuals), mx = Math.max(...residuals);
  const step = (mx - mn) / bins;
  const counts = Array(bins).fill(0);
  const labels = [];
  for (let i=0;i<bins;i++) labels.push((mn + i*step).toFixed(1));
  residuals.forEach(r => { const idx = Math.min(bins-1, Math.floor((r-mn)/step)); counts[idx]++; });
  residualsChart = new Chart(ctx, {
    type:'bar',
    data:{labels, datasets:[{label:'Residuals', data:counts, backgroundColor:'rgba(244,63,94,.55)', borderColor:'rgba(244,63,94,.8)', borderWidth:1}]},
    options:{responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Residual'}}, y:{title:{display:true,text:'Count'}}}}
  });
}

function renderImportance(res) {
  if (importanceChart) importanceChart.destroy();
  const ctx = document.getElementById('importance-chart').getContext('2d');
  const sorted = [...res.coefficients].filter(c => c.std_coef !== null).sort((a,b) => Math.abs(b.std_coef)-Math.abs(a.std_coef));
  const colors = sorted.map(c => c.std_coef >= 0 ? 'rgba(79,70,229,.7)' : 'rgba(244,63,94,.7)');
  importanceChart = new Chart(ctx, {
    type:'bar',
    data:{labels:sorted.map(c=>c.feature), datasets:[{label:'|Standardized Coef.|', data:sorted.map(c=>Math.abs(c.std_coef)), backgroundColor:colors, borderRadius:6}]},
    options:{indexAxis:'y', responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'|Std. Coefficient|'}}}}
  });
}

async function predictCustom() {
  if (!regressionModel) { showToast('Run regression first', 'error'); return; }
  const vals = {};
  document.querySelectorAll('#custom-input-fields input').forEach(inp => { vals[inp.dataset.feat] = parseFloat(inp.value)||0; });
  try {
    const r = await fetch('/api/analyze/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: regressionModel.model_id, inputs: vals})
    });
    const d = await r.json();
    if (d.status === 'success') {
      document.getElementById('custom-pred-result').style.display = 'block';
      document.getElementById('custom-pred-value').textContent = fmtNum(d.prediction);
      showToast('Prediction: ' + fmtNum(d.prediction), 'success');
    } else showToast('Error: ' + d.message, 'error');
  } catch(e) { showToast('Error: ' + e.message, 'error'); }
}

// ─── Init ───
onDatasetChange('startups');
triggerStartupsPrediction();
</script>
</body>
</html>
"""

# ══════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

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
