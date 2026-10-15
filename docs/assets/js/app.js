/* =============================================================================
 * app.js — dashboard wiring
 *
 * Owns the page state (dataset -> schema -> split -> fitted grid of models),
 * renders every panel from the result of Regression.analyse(), and keeps the
 * two Chart.js charts in sync with the active algorithm and k.
 *
 * Depends on: stats.js, regression.js, data.js, Chart.js
 * ========================================================================== */
(function () {
  'use strict';

  const ALGO_META = {
    'Lasso': { icon: 'i-filter', tag: 'L1 ranking' },
    'Forward SFS': { icon: 'i-trend', tag: 'Greedy' },
    'RFE': { icon: 'i-scissors', tag: 'Backward' },
    'ElasticNet': { icon: 'i-net', tag: 'L1 + L2' },
    'Exhaustive': { icon: 'i-search', tag: 'Best subset' }
  };

  const SERIES_COLORS = ['#4f46e5', '#f97316', '#10b981', '#e11d48', '#8b5cf6'];

  const PRESETS = [
    {
      id: 'startups',
      name: '50 Startups',
      note: '50 rows · profit',
      path: 'assets/data/50_Startups.csv',
      target: 'Profit',
      prefix: '$',
      crumb: '50 Startups Feature Progression'
    },
    {
      id: 'boston',
      name: 'Boston Housing',
      note: '506 rows · median value',
      path: 'assets/data/BostonHousing.csv',
      target: 'medv',
      prefix: '',
      suffix: 'k',
      crumb: 'Boston Housing Feature Progression'
    }
  ];

  const state = {
    datasetName: '',
    crumb: '',
    prefix: '',
    suffix: '',
    table: null,
    cols: null,
    target: null,
    features: [],
    design: null,
    result: null,
    algo: 'Lasso',
    k: 1,
    inputs: {},
    charts: { rmse: null, r2: null }
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
    );
  const icon = (id, style) =>
    '<svg class="ic"' + (style ? ' style="' + style + '"' : '') + '><use href="#' + id + '"/></svg>';

  /* ---------------------------------------------------------------------------
   * Formatting
   * ------------------------------------------------------------------------ */

  function fmt(v, decimals) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const d = decimals === undefined ? 2 : decimals;
    return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  /* Target-space values carry the dataset's unit affixes so "29.44" reads as
   * $29.44k on Boston and 182,943.06 reads as $182,943.06 on 50 Startups. */
  function fmtTarget(v, decimals) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const abs = Math.abs(v);
    const d = decimals !== undefined ? decimals : abs >= 1000 ? 2 : abs >= 1 ? 3 : 4;
    return state.prefix + fmt(v, d) + (state.suffix || '');
  }

  /* Coefficients span many orders of magnitude (0.0008 to 48,416), so pick the
   * precision from the value instead of fixing it. */
  function fmtCoef(v) {
    if (!isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1000) return fmt(v, 2);
    if (a >= 1) return fmt(v, 4);
    if (a >= 0.001) return v.toFixed(6);
    return v.toExponential(3);
  }

  function fmtP(p) {
    if (!isFinite(p)) return '—';
    if (p < 0.0001) return '< 0.0001';
    return p.toFixed(4);
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ---------------------------------------------------------------------------
   * Status messages
   * ------------------------------------------------------------------------ */

  function message(kind, html) {
    const box = $('sourceMsg');
    if (!kind) {
      box.hidden = true;
      box.innerHTML = '';
      return;
    }
    const ic = { err: 'i-warn', warn: 'i-warn', info: 'i-info', ok: 'i-check2' }[kind];
    box.hidden = false;
    box.className = 'notice ' + kind;
    box.innerHTML = icon(ic) + '<div>' + html + '</div>';
  }

  function busy(on, text) {
    $('busyText').textContent = text || 'Fitting models…';
    $('busyOverlay').hidden = !on;
  }

  /* ---------------------------------------------------------------------------
   * Loading
   * ------------------------------------------------------------------------ */

  function adoptTable(text, meta) {
    const loaded = DataKit.load(text);
    state.table = loaded.table;
    state.cols = loaded.cols;
    state.datasetName = meta.name;
    state.crumb = meta.crumb || meta.name;
    state.prefix = meta.prefix || '';
    state.suffix = meta.suffix || '';

    const wanted = meta.target && loaded.cols.some((c) => c.name === meta.target && c.type === 'numeric');
    state.target = wanted ? meta.target : loaded.target;
    state.features = DataKit.defaultFeatures(loaded.cols, state.target);

    renderPresets(meta.id || null);
    renderSchema();
    $('crumbDataset').textContent = state.crumb;

    const skipped = loaded.cols.filter((c) => c.highCardinality);
    let note =
      'Loaded <strong>' + esc(meta.name) + '</strong> — ' +
      loaded.table.rows.length + ' rows × ' + loaded.table.columns.length + ' columns' +
      (loaded.table.delimiter !== ',' ? ' (delimiter <code>' + esc(loaded.table.delimiter === '\t' ? 'tab' : loaded.table.delimiter) + '</code>)' : '') +
      '. Target set to <strong>' + esc(state.target) + '</strong>.';
    if (skipped.length) {
      note += ' Left out as identifier-like: ' +
        skipped.map((c) => '<code>' + esc(c.name) + '</code>').join(', ') + '.';
    }
    message('ok', note);
    runAnalysis();
  }

  function loadPreset(preset) {
    busy(true, 'Loading ' + preset.name + '…');
    fetch(preset.path)
      .then((r) => {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' reading ' + preset.path);
        return r.text();
      })
      .then((text) => adoptTable(text, preset))
      .catch((err) => {
        busy(false);
        message('err', 'Could not load ' + esc(preset.name) + '. ' + esc(err.message));
      });
  }

  function loadFromUrl() {
    const url = $('csvUrl').value.trim();
    if (!url) {
      message('err', 'Enter a CSV URL first.');
      return;
    }
    const btn = $('loadUrlBtn');
    btn.disabled = true;
    busy(true, 'Fetching CSV…');
    DataKit.fetchCSV(url)
      .then((text) =>
        adoptTable(text, {
          name: url.split('/').pop().split('?')[0] || 'Remote CSV',
          crumb: 'Custom dataset',
          prefix: '',
          suffix: ''
        })
      )
      .catch((err) => {
        busy(false);
        message('err', esc(err.message));
      })
      .finally(() => {
        btn.disabled = false;
      });
  }

  function loadFromFile(file) {
    busy(true, 'Reading ' + file.name + '…');
    DataKit.readFile(file)
      .then((text) => adoptTable(text, { name: file.name, crumb: 'Custom dataset' }))
      .catch((err) => {
        busy(false);
        message('err', esc(err.message));
      });
  }

  function loadFromPaste() {
    const text = $('pasteBox').value;
    if (!text.trim()) {
      message('err', 'Paste some delimited text first.');
      return;
    }
    busy(true, 'Parsing…');
    try {
      adoptTable(text, { name: 'Pasted data', crumb: 'Custom dataset' });
    } catch (err) {
      busy(false);
      message('err', esc(err.message));
    }
  }

  /* ---------------------------------------------------------------------------
   * Schema panel
   * ------------------------------------------------------------------------ */

  function renderPresets(activeId) {
    $('presetRow').innerHTML = PRESETS.map(
      (p) =>
        '<button type="button" class="preset-btn' + (p.id === activeId ? ' active' : '') +
        '" data-preset="' + p.id + '">' + esc(p.name) + '<small>' + esc(p.note) + '</small></button>'
    ).join('');
  }

  function renderSchema() {
    $('targetSelect').innerHTML = state.cols
      .filter((c) => c.type === 'numeric')
      .map(
        (c) =>
          '<option value="' + esc(c.name) + '"' + (c.name === state.target ? ' selected' : '') +
          '>' + esc(c.name) + '</option>'
      )
      .join('');

    $('colList').innerHTML = state.cols
      .map((c) => {
        const isTarget = c.name === state.target;
        const checked = state.features.indexOf(c.name) !== -1;
        let meta;
        if (isTarget) meta = 'target';
        else if (c.type === 'numeric') meta = 'numeric';
        else if (c.levelsCapped) meta = DataKit.MAX_LEVELS + '+ levels · excluded';
        else meta = c.nLevels + ' levels';
        const miss = c.missing ? ' · ' + c.missing + ' missing' : '';
        return (
          '<label class="col-row' + (isTarget ? ' is-target' : '') + '">' +
          '<input type="checkbox" data-col="' + esc(c.name) + '"' +
          (checked ? ' checked' : '') + (isTarget ? ' disabled' : '') + '>' +
          '<span class="cname">' + esc(c.name) + '</span>' +
          '<span class="cmeta">' + esc(meta + miss) + '</span>' +
          '<span class="badge-tag ' + (isTarget ? 'badge-blue' : c.type === 'numeric' ? 'badge-grey' : 'badge-amber') +
          '">' + (isTarget ? 'y' : c.type === 'numeric' ? 'num' : 'cat') + '</span>' +
          '</label>'
        );
      })
      .join('');
  }

  /* ---------------------------------------------------------------------------
   * Analysis
   * ------------------------------------------------------------------------ */

  function runAnalysis() {
    busy(true, 'Fitting models…');
    /* let the overlay paint before the synchronous fit blocks the thread */
    setTimeout(() => {
      try {
        const design = DataKit.buildDesign(state.table, state.cols, state.target, state.features);
        const result = Regression.analyse(design, {
          testSize: parseFloat($('testSize').value),
          seed: parseInt($('seedInput').value, 10) || 0
        });
        state.design = design;
        state.result = result;

        if (!result.models[state.algo]) state.algo = 'Lasso';
        const best = result.bestGlobal;
        state.algo = best ? best.algo : state.algo;
        state.k = best ? best.k : 1;

        buildPredictorInputs();
        renderAll();

        const extras = [];
        if (design.droppedRows) {
          extras.push(design.droppedRows + ' row(s) dropped for missing values');
        }
        if (design.droppedConstant.length) {
          extras.push('constant column(s) removed: ' + design.droppedConstant.join(', '));
        }
        if (result.poolNote) extras.push(result.poolNote);
        if (extras.length) {
          message('info', esc(extras.join('. ')) + '.');
        }
      } catch (err) {
        message('err', esc(err.message));
        state.result = null;
      } finally {
        busy(false);
      }
    }, 20);
  }

  /* ---------------------------------------------------------------------------
   * Render
   * ------------------------------------------------------------------------ */

  /* Which algorithm the metric ribbon, equation and tables describe. In
   * "compare all" mode the charts show every method at once, so the detail
   * panels follow whichever one wins at the selected k — and say so. */
  function detailAlgo() {
    const r = state.result;
    if (!r) return 'Lasso';
    if (state.algo !== 'ALL') return state.algo;
    let best = null;
    r.algorithms.forEach((a) => {
      const m = r.models[a][state.k - 1];
      if (!m || !isFinite(m.rmse)) return;
      if (!best || m.rmse < r.models[best][state.k - 1].rmse) best = a;
    });
    return best || 'Lasso';
  }

  function activeModel() {
    if (!state.result) return null;
    const list = state.result.models[detailAlgo()];
    if (!list) return null;
    return list[state.k - 1] || null;
  }

  /* Best RMSE at each k across all methods — the envelope the compare view
   * marks its sweet spot and cliff against. */
  function envelope() {
    const r = state.result;
    const out = [];
    for (let k = 1; k <= r.maxK; k++) {
      let best = null;
      r.algorithms.forEach((a) => {
        const m = r.models[a][k - 1];
        if (!m || !isFinite(m.rmse)) return;
        if (!best || m.rmse < best.rmse) best = { rmse: m.rmse, algo: a };
      });
      out.push(best);
    }
    return out;
  }

  function renderAll() {
    renderHero();
    renderControls();
    renderMetrics();
    renderPredictors();
    renderEquation();
    renderCoefTable();
    renderOutliers();
    renderDesignPreview();
    renderCharts();
    updatePrediction();
  }

  function renderHero() {
    const r = state.result;
    if (!r) return;
    const best = r.bestGlobal;
    const full = r.models.Lasso[r.maxK - 1];
    const tags = [];
    tags.push(icon('i-db') + ' <span>' + esc(state.datasetName) + '</span>');
    tags.push(icon('i-layers') + ' <span>' + r.algorithms.length + ' selection methods · k = 1…' + r.maxK + '</span>');
    tags.push(
      icon('i-split') + ' <span>' + r.split.nTrain + ' train / ' + r.split.nTest +
      ' test · seed ' + r.split.seed + '</span>'
    );
    if (best) {
      tags.push(
        icon('i-star', 'color:#6ee7b7') + ' <span>Sweet spot: k = ' + best.k +
        ' <b>' + fmtTarget(best.rmse) + '</b> RMSE (' + esc(best.algo) + ')</span>'
      );
    }
    if (full && best && full.rmse > best.rmse) {
      const pct = ((full.rmse / best.rmse - 1) * 100).toFixed(1);
      tags.push(
        icon('i-warn', 'color:#fcd34d') + ' <span>Full ' + r.maxK + '-feature model: <b>' +
        fmtTarget(full.rmse) + '</b> (+' + pct + '%)</span>'
      );
    }
    $('heroTags').innerHTML = tags.map((t) => '<span class="hero-pill">' + t + '</span>').join('');

    $('splitSummary').innerHTML =
      ' Currently <strong>' + r.split.nTrain + '</strong> train / <strong>' +
      r.split.nTest + '</strong> test rows.';
  }

  function renderControls() {
    const r = state.result;
    if (!r) return;

    $('algoTabs').innerHTML =
      r.algorithms
        .map((a) => {
          const m = ALGO_META[a];
          const bestK = r.perAlgoBest[a];
          const tag = bestK ? 'best k=' + bestK.k : m.tag;
          return (
            '<button type="button" class="tab-btn' + (state.algo === a ? ' active' : '') +
            '" data-algo="' + esc(a) + '">' + icon(m.icon) + ' ' + esc(a) +
            ' <span class="badge-pill">' + esc(tag) + '</span></button>'
          );
        })
        .join('') +
      '<button type="button" class="tab-btn compare' + (state.algo === 'ALL' ? ' active' : '') +
      '" data-algo="ALL">' + icon('i-layers') + ' Compare all ' + r.algorithms.length + '</button>';

    /* k buttons carry the RMSE at each k for the active method, or the best
     * across methods in compare mode, and mark that curve's min and max. */
    const curve =
      state.algo === 'ALL'
        ? envelope()
        : r.models[state.algo].map((m) => (m && isFinite(m.rmse) ? { rmse: m.rmse } : null));

    let bestK = null;
    let worstK = null;
    for (let i = 0; i < curve.length; i++) {
      if (!curve[i]) continue;
      if (bestK === null || curve[i].rmse < curve[bestK - 1].rmse) bestK = i + 1;
      if (worstK === null || curve[i].rmse > curve[worstK - 1].rmse) worstK = i + 1;
    }

    const buttons = [];
    for (let k = 1; k <= r.maxK; k++) {
      const point = curve[k - 1];
      let cls = 'k-btn';
      let sub = '';
      if (k === bestK) {
        cls += ' sweet';
        sub = '<span class="sub">★ min RMSE</span>';
      } else if (k === worstK && r.maxK > 1) {
        cls += ' cliff';
        sub = '<span class="sub">⚠ worst</span>';
      } else if (point) {
        sub = '<span class="sub">' + esc(fmtTarget(point.rmse)) + '</span>';
      }
      if (state.k === k) cls += ' active';
      buttons.push(
        '<button type="button" class="' + cls + '" data-k="' + k + '"' + (point ? '' : ' disabled') +
        '>k = ' + k + ' ' + sub + '</button>'
      );
    }
    $('kSelectorGroup').innerHTML = buttons.join('');
  }

  function renderMetrics() {
    const m = activeModel();
    const r = state.result;
    if (!m || !r) return;

    const algo = detailAlgo();
    const bestK = r.perAlgoBest[algo];
    const isBest = bestK && bestK.k === state.k;

    $('valRMSE').textContent = fmtTarget(m.rmse);
    $('valR2').textContent = m.r2.toFixed(4);
    $('valAdjR2').textContent = isFinite(m.adj_r2) ? m.adj_r2.toFixed(4) : '—';
    $('valMAE').textContent = fmtTarget(m.mae);
    $('valFStat').textContent = isFinite(m.f_stat) ? fmt(m.f_stat, 1) : '—';

    const badge = $('rmseBadge');
    const card = $('cardRMSE');
    card.className = 'metric-card' + (isBest ? ' sweet' : '');
    if (isBest) {
      badge.className = 'badge-tag badge-green';
      badge.innerHTML = icon('i-award') + ' Optimal';
    } else {
      badge.className = 'badge-tag badge-blue';
      badge.textContent = 'k = ' + state.k;
    }
    $('noteRMSE').textContent =
      'Lower is better · ' + r.split.nTest + ' held-out rows';

    $('r2Badge').className = 'badge-tag ' + (m.r2 >= 0.8 ? 'badge-green' : m.r2 >= 0.5 ? 'badge-blue' : 'badge-amber');
    $('r2Badge').textContent = (m.r2 * 100).toFixed(1) + '% fit';
    $('noteR2').textContent =
      'Explains ' + (m.r2 * 100).toFixed(2) + '% of out-of-sample variance';

    const fb = $('fBadge');
    if (isFinite(m.f_pvalue)) {
      fb.className = 'badge-tag ' + (m.f_pvalue < 0.05 ? 'badge-green' : 'badge-amber');
      fb.textContent = 'p ' + (m.f_pvalue < 0.0001 ? '< 0.0001' : '= ' + m.f_pvalue.toFixed(4));
    } else {
      fb.className = 'badge-tag badge-grey';
      fb.textContent = '—';
    }
    $('noteF').textContent = 'Joint significance · training R² = ' + m.r2_train.toFixed(4);

    /* active predictor chips, prefixed in compare mode by the method the
     * detail panels below are actually describing */
    $('featuresCount').textContent = m.features.length;
    const lead =
      state.algo === 'ALL'
        ? '<span class="badge-tag badge-blue">' + icon('i-award') + ' ' + esc(algo) +
          ' wins at k = ' + state.k + '</span>'
        : '';
    $('featuresListChips').innerHTML =
      lead + m.features.map((f) => '<span class="feature-chip">' + esc(f) + '</span>').join('');

    const notice = $('parsimonyNotice');
    if (isBest && state.k < r.maxK) {
      notice.className = 'badge-tag badge-green';
      notice.innerHTML =
        icon('i-star') + " Occam's razor: " + state.k + ' of ' + r.maxK +
        ' predictors give the lowest hold-out error for ' + esc(algo) + '.';
    } else if (state.k === r.maxK && bestK && bestK.k < r.maxK) {
      const pct = ((m.rmse / bestK.rmse - 1) * 100).toFixed(1);
      notice.className = 'badge-tag badge-amber';
      notice.innerHTML =
        icon('i-warn') + ' Overfitting: the full model costs ' + pct +
        '% more error than k = ' + bestK.k + '.';
    } else {
      notice.className = 'badge-tag badge-blue';
      notice.innerHTML = icon('i-info') + ' Evaluated with ' + state.k + ' predictor(s).';
    }
  }

  function renderEquation() {
    const m = activeModel();
    if (!m) return;
    const parts = [];
    m.coefficients.forEach((c, i) => {
      if (i === 0) {
        parts.push('<span class="eq-intercept">' + esc(fmtCoef(c.coef)) + '</span>');
        return;
      }
      const sign = c.coef >= 0 ? '<span class="eq-pos">+</span>' : '<span class="eq-neg">−</span>';
      parts.push(
        sign + ' ' + esc(fmtCoef(Math.abs(c.coef))) + ' × <span class="eq-var">(' + esc(c.feature) + ')</span>'
      );
    });
    $('equationBox').innerHTML =
      '<strong>' + esc(state.target) + '</strong> = ' + parts.join(' ');
  }

  function renderCoefTable() {
    const m = activeModel();
    if (!m) return;
    $('coefTableBody').innerHTML = m.coefficients
      .map((c) => {
        const isIntercept = c.vif === null;
        const pBadge = !isFinite(c.pvalue)
          ? '<span class="badge-tag badge-grey">—</span>'
          : c.pvalue < 0.05
            ? '<span class="badge-tag badge-green">' + fmtP(c.pvalue) + ' ' + icon('i-check2') + '</span>'
            : '<span class="badge-tag badge-amber">' + fmtP(c.pvalue) + ' (n.s.)</span>';

        let vifCell = '<span style="color:var(--fg-faint)">—</span>';
        if (!isIntercept) {
          if (!isFinite(c.vif)) {
            vifCell = '<span class="badge-tag badge-red">∞ (singular)</span>';
          } else if (c.vif < 5) {
            vifCell = '<span class="badge-tag badge-green">' + fmt(c.vif, 2) + ' safe</span>';
          } else if (c.vif < 10) {
            vifCell = '<span class="badge-tag badge-amber">' + fmt(c.vif, 2) + ' moderate</span>';
          } else {
            vifCell = '<span class="badge-tag badge-red">' + fmt(c.vif, 2) + ' collinear</span>';
          }
        }

        return (
          '<tr><td><strong>' + esc(c.feature) + '</strong></td>' +
          '<td class="num" style="font-weight:600">' + esc(fmtCoef(c.coef)) + '</td>' +
          '<td class="num">' + esc(fmtCoef(c.stderr)) + '</td>' +
          '<td class="num">' + (isFinite(c.tstat) ? c.tstat.toFixed(3) : '—') + '</td>' +
          '<td>' + pBadge + '</td>' +
          '<td class="ci">[' + esc(fmtCoef(c.ci_lower)) + ', ' + esc(fmtCoef(c.ci_upper)) + ']</td>' +
          '<td>' + vifCell + '</td></tr>'
        );
      })
      .join('');
  }

  function renderOutliers() {
    const m = activeModel();
    if (!m) return;
    const threshold = m.mae * 1.5;
    $('outlierSub').textContent =
      'Largest absolute residuals among the ' + state.result.split.nTest +
      ' rows the model never saw. Flagged above 1.5 × MAE.';
    $('outliersTableBody').innerHTML = m.outliers
      .map((o) => {
        const high = o.abs_error > threshold;
        const status = high
          ? '<span class="badge-tag badge-amber"><span class="status-dot"></span> High residual</span>'
          : '<span class="badge-tag badge-green"><span class="status-dot"></span> In range</span>';
        const col = o.residual >= 0 ? 'var(--good)' : 'var(--bad)';
        return (
          '<tr><td style="font-weight:700">#' + o.rank + '</td>' +
          '<td>row ' + o.index + '</td>' +
          '<td class="num">' + esc(fmtTarget(o.actual)) + '</td>' +
          '<td class="num">' + esc(fmtTarget(o.predicted)) + '</td>' +
          '<td class="num" style="color:' + col + ';font-weight:600">' +
          (o.residual >= 0 ? '+' : '−') + esc(fmtTarget(Math.abs(o.residual))) + '</td>' +
          '<td class="num">' + esc(fmtTarget(o.abs_error)) + '</td>' +
          '<td class="num">' + (isFinite(o.pct_error) ? o.pct_error.toFixed(2) + '%' : '—') + '</td>' +
          '<td>' + status + '</td></tr>'
        );
      })
      .join('');
  }

  function renderDesignPreview() {
    const d = state.design;
    if (!d) return;
    $('previewBadge').textContent =
      d.nRows + ' rows × ' + d.featureNames.length + ' design columns';

    const maxAbs = Math.max.apply(
      null,
      d.columnsMeta.map((c) => Math.abs(c.corr))
    ) || 1;

    $('designBody').innerHTML = d.columnsMeta
      .map((c) => {
        const w = Math.round((Math.abs(c.corr) / maxAbs) * 54);
        const kind =
          c.kind === 'dummy'
            ? '<span class="badge-tag badge-amber">dummy of ' + esc(c.source) + '</span>'
            : '<span class="badge-tag badge-grey">numeric</span>';
        return (
          '<tr><td><strong>' + esc(c.label) + '</strong></td><td>' + kind + '</td>' +
          '<td class="num">' + esc(fmt(c.min, 3)) + '</td>' +
          '<td class="num">' + esc(fmt(c.max, 3)) + '</td>' +
          '<td class="num">' + esc(fmt(c.mean, 3)) + '</td>' +
          '<td class="num">' + esc(fmt(c.std, 3)) + '</td>' +
          '<td class="num">' + c.corr.toFixed(3) +
          ' <span class="corr-bar" style="width:' + w + 'px;background:' +
          (c.corr >= 0 ? 'var(--accent)' : 'var(--bad)') + '"></span></td></tr>'
        );
      })
      .join('');

    const cols = state.table.columns;
    $('rawHead').innerHTML = cols.map((c) => '<th>' + esc(c) + '</th>').join('');
    $('rawBody').innerHTML = state.table.rows
      .slice(0, 10)
      .map(
        (row) =>
          '<tr>' + cols.map((c) => '<td>' + esc(row[c]) + '</td>').join('') + '</tr>'
      )
      .join('');
  }

  /* ---------------------------------------------------------------------------
   * Live predictor
   *
   * Controls are built per *source* column, not per design column: a numeric
   * column gets a slider, a categorical column gets a dropdown whose options
   * are the held-out reference level plus one per dummy.
   * ------------------------------------------------------------------------ */

  function buildPredictorInputs() {
    const d = state.design;
    const groups = [];
    const bySource = {};

    d.columnsMeta.forEach((c, j) => {
      if (c.kind === 'numeric') {
        groups.push({ kind: 'numeric', source: c.source, col: j, meta: c });
        return;
      }
      if (!bySource[c.source]) {
        bySource[c.source] = {
          kind: 'categorical',
          source: c.source,
          reference: c.reference,
          dummies: []
        };
        groups.push(bySource[c.source]);
      }
      bySource[c.source].dummies.push({ col: j, label: c.label, level: c.label.slice(c.source.length + 1) });
    });

    state.groups = groups;
    state.inputs = {};

    const html = groups
      .map((g, gi) => {
        if (g.kind === 'numeric') {
          const m = g.meta;
          const span = m.max - m.min || 1;
          const step = span / 200;
          const start = m.median;
          state.inputs[g.col] = start;
          return (
            '<div class="input-group" data-group="' + gi + '">' +
            '<div class="input-label-row"><span>' + esc(m.label) + '</span>' +
            '<span class="val" id="val-' + gi + '">' + esc(fmt(start, 3)) + '</span></div>' +
            '<input type="number" class="input-field" id="num-' + gi + '" value="' + start +
            '" step="' + step + '">' +
            '<input type="range" class="range-slider" id="rng-' + gi + '" value="' + start +
            '" min="' + m.min + '" max="' + m.max + '" step="' + step + '">' +
            '</div>'
          );
        }
        g.dummies.forEach((dm) => {
          state.inputs[dm.col] = 0;
        });
        const opts =
          '<option value="">' + esc(g.reference) + ' (reference)</option>' +
          g.dummies
            .map((dm) => '<option value="' + dm.col + '">' + esc(dm.level) + '</option>')
            .join('');
        return (
          '<div class="input-group" data-group="' + gi + '">' +
          '<div class="input-label-row"><span>' + esc(g.source) + '</span></div>' +
          '<select class="input-field" id="sel-' + gi + '">' + opts + '</select>' +
          '</div>'
        );
      })
      .join('');

    $('predictorInputs').innerHTML = html;
    $('predTargetName').textContent = state.target;

    groups.forEach((g, gi) => {
      if (g.kind === 'numeric') {
        const num = $('num-' + gi);
        const rng = $('rng-' + gi);
        const lab = $('val-' + gi);
        const sync = (v) => {
          const val = parseFloat(v);
          if (!isFinite(val)) return;
          state.inputs[g.col] = val;
          lab.textContent = fmt(val, 3);
          updatePrediction();
        };
        rng.addEventListener('input', (e) => {
          num.value = e.target.value;
          sync(e.target.value);
        });
        num.addEventListener('input', (e) => {
          rng.value = e.target.value;
          sync(e.target.value);
        });
      } else {
        $('sel-' + gi).addEventListener('change', (e) => {
          g.dummies.forEach((dm) => {
            state.inputs[dm.col] = 0;
          });
          if (e.target.value !== '') state.inputs[parseInt(e.target.value, 10)] = 1;
          updatePrediction();
        });
      }
    });
  }

  /* Grey out controls whose design columns are not in the active model. */
  function renderPredictors() {
    const m = activeModel();
    if (!m || !state.groups) return;
    const active = {};
    m.featIdx.forEach((j) => {
      active[j] = true;
    });
    state.groups.forEach((g, gi) => {
      const el = document.querySelector('.input-group[data-group="' + gi + '"]');
      if (!el) return;
      const used =
        g.kind === 'numeric' ? !!active[g.col] : g.dummies.some((dm) => active[dm.col]);
      el.classList.toggle('inactive', !used);
    });
  }

  function updatePrediction() {
    const m = activeModel();
    if (!m) return;
    let pred = m.coefficients[0].coef;
    const terms = [];
    m.featIdx.forEach((j, i) => {
      const x = state.inputs[j] || 0;
      const b = m.coefficients[i + 1].coef;
      const contrib = b * x;
      pred += contrib;
      terms.push(state.design.featureNames[j] + ': ' + (contrib >= 0 ? '+' : '−') + fmtTarget(Math.abs(contrib), 2));
    });
    $('predOutput').textContent = fmtTarget(pred);
    $('predNote').innerHTML =
      'Active predictors: ' + esc(m.features.join(', ')) +
      '<span class="terms">intercept ' + esc(fmtTarget(m.coefficients[0].coef)) +
      (terms.length ? ' · ' + esc(terms.join(' · ')) : '') + '</span>';
  }

  /* ---------------------------------------------------------------------------
   * Charts
   * ------------------------------------------------------------------------ */

  function chartTheme() {
    return {
      grid: cssVar('--line'),
      text: cssVar('--fg-muted'),
      faint: cssVar('--fg-faint'),
      surface: cssVar('--surface'),
      fg: cssVar('--fg')
    };
  }

  function buildSeries(metric) {
    const r = state.result;
    const labels = [];
    for (let k = 1; k <= r.maxK; k++) labels.push('k=' + k);

    const pick = (m) => (m ? (metric === 'rmse' ? m.rmse : m.r2) : null);

    if (state.algo === 'ALL') {
      return {
        labels: labels,
        datasets: r.algorithms.map((a, i) => ({
          label: a,
          data: r.models[a].map(pick),
          borderColor: SERIES_COLORS[i % SERIES_COLORS.length],
          backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length],
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
          tension: 0.25,
          spanGaps: true
        }))
      };
    }

    const list = r.models[state.algo];
    const bestK = r.perAlgoBest[state.algo];
    const accent = metric === 'rmse' ? cssVar('--accent') : cssVar('--good');
    return {
      labels: labels,
      datasets: [
        {
          label: state.algo,
          data: list.map(pick),
          borderColor: accent,
          backgroundColor: accent + '22',
          borderWidth: 2.5,
          fill: true,
          tension: 0.25,
          spanGaps: true,
          pointRadius: (ctx) => {
            const k = ctx.dataIndex + 1;
            if (k === state.k) return 7;
            if (bestK && k === bestK.k) return 6;
            return 3.5;
          },
          pointBackgroundColor: (ctx) => {
            const k = ctx.dataIndex + 1;
            if (k === state.k) return cssVar('--fg');
            if (bestK && k === bestK.k) return cssVar('--good');
            return accent;
          },
          pointHoverRadius: 8
        }
      ]
    };
  }

  function chartOptions(metric) {
    const t = chartTheme();
    const r = state.result;
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: state.algo === 'ALL',
          position: 'top',
          labels: { color: t.text, boxWidth: 12, boxHeight: 12, font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: t.fg,
          titleColor: t.surface,
          bodyColor: t.surface,
          padding: 11,
          cornerRadius: 8,
          displayColors: state.algo === 'ALL',
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y;
              const head = state.algo === 'ALL' ? ctx.dataset.label + ': ' : '';
              return head + (metric === 'rmse' ? fmtTarget(v) : v.toFixed(4));
            },
            afterLabel: (ctx) => {
              const algo = state.algo === 'ALL' ? ctx.dataset.label : state.algo;
              const m = r.models[algo][ctx.dataIndex];
              if (!m) return '';
              return 'Features: ' + m.features.join(', ');
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Number of features (k)', color: t.faint, font: { size: 11.5 } },
          grid: { color: t.grid },
          ticks: { color: t.text, font: { size: 11 } }
        },
        y: {
          title: {
            display: true,
            text: metric === 'rmse' ? 'Hold-out RMSE' : 'Hold-out R²',
            color: t.faint,
            font: { size: 11.5 }
          },
          grid: { color: t.grid },
          ticks: {
            color: t.text,
            font: { size: 11 },
            callback: (v) => (metric === 'rmse' ? fmt(v, 0) : Number(v).toFixed(2))
          }
        }
      }
    };
  }

  function renderCharts() {
    const r = state.result;
    if (!r || typeof Chart === 'undefined') return;

    ['rmse', 'r2'].forEach((metric) => {
      const canvas = $(metric === 'rmse' ? 'rmseChart' : 'r2Chart');
      if (state.charts[metric]) state.charts[metric].destroy();
      state.charts[metric] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: buildSeries(metric),
        options: chartOptions(metric)
      });
    });

    const algo = state.algo === 'ALL' ? 'all methods' : state.algo;
    const bestAll = r.bestGlobal;
    $('rmseChartBadge').textContent = state.algo === 'ALL' ? 'All methods' : algo;
    $('r2ChartBadge').textContent = bestAll ? 'Best k = ' + bestAll.k : '—';

    if (state.algo === 'ALL') {
      $('rmseChartSub').textContent = 'Lower is better. Each line is one selection method.';
      $('r2ChartSub').textContent = 'Higher is better. Divergence shows where the methods disagree.';
    } else {
      const bk = r.perAlgoBest[state.algo];
      $('rmseChartSub').textContent =
        'Lower is better.' + (bk ? ' Minimum at k = ' + bk.k + ' (' + fmtTarget(bk.rmse) + ').' : '');
      $('r2ChartSub').textContent = 'Higher is better. Filled marker is the selected k.';
    }
  }

  /* ---------------------------------------------------------------------------
   * Events
   * ------------------------------------------------------------------------ */

  function bind() {
    $('presetRow').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-preset]');
      if (!btn) return;
      const preset = PRESETS.find((p) => p.id === btn.dataset.preset);
      if (preset) loadPreset(preset);
    });

    $('sourceTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-pane]');
      if (!btn) return;
      Array.from($('sourceTabs').children).forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      ['url', 'file', 'paste'].forEach((p) => {
        $('pane-' + p).hidden = p !== btn.dataset.pane;
      });
    });

    $('loadUrlBtn').addEventListener('click', loadFromUrl);
    $('csvUrl').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') loadFromUrl();
    });
    $('loadPasteBtn').addEventListener('click', loadFromPaste);

    const dz = $('dropZone');
    dz.addEventListener('click', () => $('fileInput').click());
    dz.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        $('fileInput').click();
      }
    });
    ['dragenter', 'dragover'].forEach((ev) =>
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        dz.classList.add('over');
      })
    );
    ['dragleave', 'drop'].forEach((ev) =>
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        dz.classList.remove('over');
      })
    );
    dz.addEventListener('drop', (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) loadFromFile(f);
    });
    $('fileInput').addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) loadFromFile(f);
    });

    $('targetSelect').addEventListener('change', (e) => {
      state.target = e.target.value;
      state.features = DataKit.defaultFeatures(state.cols, state.target);
      renderSchema();
      runAnalysis();
    });

    $('colList').addEventListener('change', (e) => {
      const cb = e.target.closest('input[data-col]');
      if (!cb) return;
      const name = cb.dataset.col;
      const at = state.features.indexOf(name);
      if (cb.checked && at === -1) state.features.push(name);
      if (!cb.checked && at !== -1) state.features.splice(at, 1);
      if (!state.features.length) {
        message('err', 'Keep at least one predictor selected.');
        cb.checked = true;
        state.features.push(name);
        return;
      }
      runAnalysis();
    });

    $('refitBtn').addEventListener('click', runAnalysis);

    $('algoTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-algo]');
      if (!btn) return;
      state.algo = btn.dataset.algo;
      const r = state.result;
      if (r) {
        /* jump to the k that method does best at, or the global best in
         * compare mode, so switching tabs always lands on something useful */
        const best = state.algo === 'ALL' ? r.bestGlobal : r.perAlgoBest[state.algo];
        if (best) state.k = best.k;
      }
      renderAll();
    });

    $('kSelectorGroup').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-k]');
      if (!btn || btn.disabled) return;
      state.k = parseInt(btn.dataset.k, 10);
      renderAll();
    });

    $('themeToggle').addEventListener('click', () => {
      const root = document.documentElement;
      const explicit = root.getAttribute('data-theme');
      const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const current = explicit || (systemDark ? 'dark' : 'light');
      root.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
      try {
        localStorage.setItem('mre-theme', root.getAttribute('data-theme'));
      } catch (err) {
        /* private mode or blocked storage — the toggle still works for this view */
      }
      if (state.result) renderCharts();
    });
  }

  function restoreTheme() {
    try {
      const saved = localStorage.getItem('mre-theme');
      if (saved === 'dark' || saved === 'light') {
        document.documentElement.setAttribute('data-theme', saved);
      }
    } catch (err) {
      /* ignore */
    }
  }

  restoreTheme();
  bind();
  renderPresets('startups');
  loadPreset(PRESETS[0]);
})();
