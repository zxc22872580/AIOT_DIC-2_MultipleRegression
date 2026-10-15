/* =============================================================================
 * regression.js — OLS inference + five feature-selection algorithms
 *
 * Everything runs against a pre-computed Gram cache: Xᵀ X, Xᵀ y and yᵀ y are
 * accumulated once per split, after which fitting any subset of k predictors
 * costs O(k³) instead of O(n·k²). That is what makes exhaustive best-subset
 * search (2^p fits) viable in the browser.
 *
 * Depends on: stats.js
 * ========================================================================== */
(function (global) {
  'use strict';

  const S = global.Stats;

  /* Hard ceiling on the exhaustive search. 2^16 = 65536 subsets is still
   * sub-second with the Gram cache; beyond that we shrink the candidate pool. */
  const EXHAUSTIVE_MAX_POOL = 16;

  /* ---------------------------------------------------------------------------
   * Gram cache
   * ------------------------------------------------------------------------ */

  /* Accumulate the augmented normal-equation pieces for one design matrix.
   * Column 0 of the augmented basis is the intercept. */
  function accumulate(X, y) {
    const n = X.length;
    const p = n ? X[0].length : 0;
    const m = p + 1;
    const G = [];
    for (let i = 0; i < m; i++) G.push(new Float64Array(m));
    const b = new Float64Array(m);
    let yty = 0;
    let ysum = 0;

    for (let r = 0; r < n; r++) {
      const row = X[r];
      const yr = y[r];
      yty += yr * yr;
      ysum += yr;
      b[0] += yr;
      G[0][0] += 1;
      for (let i = 0; i < p; i++) {
        const vi = row[i];
        G[0][i + 1] += vi;
        b[i + 1] += vi * yr;
        for (let j = i; j < p; j++) G[i + 1][j + 1] += vi * row[j];
      }
    }
    for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) G[i][j] = G[j][i];

    const ybar = n ? ysum / n : 0;
    return { G: G, b: b, yty: yty, n: n, ybar: ybar, tss: yty - n * ybar * ybar };
  }

  function buildCache(Xtr, ytr, Xte, yte) {
    return {
      p: Xtr.length ? Xtr[0].length : 0,
      train: accumulate(Xtr, ytr),
      test: accumulate(Xte, yte),
      Xte: Xte,
      yte: yte
    };
  }

  /* Pull the [intercept, ...featIdx] submatrix out of an accumulated Gram. */
  function subGram(acc, cols) {
    const k = cols.length;
    const Gs = [];
    for (let i = 0; i < k; i++) {
      const row = new Array(k);
      for (let j = 0; j < k; j++) row[j] = acc.G[cols[i]][cols[j]];
      Gs.push(row);
    }
    const bs = new Array(k);
    for (let i = 0; i < k; i++) bs[i] = acc.b[cols[i]];
    return { G: Gs, b: bs };
  }

  function quadForm(G, v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) {
      let inner = 0;
      for (let j = 0; j < v.length; j++) inner += G[i][j] * v[j];
      s += v[i] * inner;
    }
    return s;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  /* Fit one subset and return only what the search criteria need. */
  function fitFast(cache, featIdx) {
    const cols = [0].concat(featIdx.map((i) => i + 1));
    const tr = subGram(cache.train, cols);
    const Ginv = S.matInverse(tr.G);
    if (!Ginv) return null;
    const beta = S.matVec(Ginv, tr.b);

    const rssTrain = cache.train.yty - dot(beta, tr.b);
    const te = subGram(cache.test, cols);
    const rssTest =
      cache.test.yty - 2 * dot(beta, te.b) + quadForm(te.G, beta);

    const nTe = cache.test.n;
    return {
      featIdx: featIdx,
      beta: beta,
      Ginv: Ginv,
      rssTrain: rssTrain,
      r2Train: 1 - rssTrain / cache.train.tss,
      rssTest: rssTest,
      rmse: Math.sqrt(Math.max(rssTest, 0) / nTe),
      r2Test: 1 - rssTest / cache.test.tss
    };
  }

  /* Variance inflation factor for feature `j` against the rest, read straight
   * out of the training Gram (no row scan). */
  function vifFromGram(acc, featIdx, j) {
    if (featIdx.length < 2) return 1;
    const others = featIdx.filter((f) => f !== j);
    const cols = [0].concat(others.map((i) => i + 1));
    const jc = j + 1;

    const k = cols.length;
    const G = [];
    for (let i = 0; i < k; i++) {
      const row = new Array(k);
      for (let q = 0; q < k; q++) row[q] = acc.G[cols[i]][cols[q]];
      G.push(row);
    }
    const b = cols.map((c) => acc.G[c][jc]);
    const Ginv = S.matInverse(G);
    if (!Ginv) return Infinity;
    const beta = S.matVec(Ginv, b);

    const n = acc.n;
    const jmean = acc.G[0][jc] / n;
    const tss = acc.G[jc][jc] - n * jmean * jmean;
    if (tss <= 0) return Infinity;
    const rss = acc.G[jc][jc] - dot(beta, b);
    const r2 = 1 - rss / tss;
    if (r2 >= 1 - 1e-12) return Infinity;
    return 1 / (1 - r2);
  }

  /* Full inference for a subset: coefficient table, F test, hold-out metrics
   * and the per-row residual diagnostics. */
  function fitFull(cache, featIdx, names, testRowIds, opts) {
    const base = fitFast(cache, featIdx);
    if (!base) return null;
    const o = opts || {};
    const beta = base.beta;
    const p = featIdx.length;
    const nTr = cache.train.n;
    const dof = nTr - p - 1;

    /* --- coefficient inference on the training set ----------------------- */
    const sigma2 = dof > 0 ? base.rssTrain / dof : NaN;
    const tcrit = dof > 0 ? S.studentTInv(0.975, dof) : NaN;
    const coefficients = [];
    for (let i = 0; i < beta.length; i++) {
      const se = dof > 0 ? Math.sqrt(Math.max(base.Ginv[i][i] * sigma2, 0)) : NaN;
      const t = se > 0 ? beta[i] / se : NaN;
      coefficients.push({
        feature: i === 0 ? 'Intercept (β₀)' : names[featIdx[i - 1]],
        coef: beta[i],
        stderr: se,
        tstat: t,
        pvalue: isFinite(t) ? S.studentTTwoSided(t, dof) : NaN,
        ci_lower: beta[i] - tcrit * se,
        ci_upper: beta[i] + tcrit * se,
        vif: i === 0 ? null : vifFromGram(cache.train, featIdx, featIdx[i - 1])
      });
    }

    /* --- global F test (training set, as in a classic OLS summary) ------- */
    const r2Train = base.r2Train;
    let fStat = NaN;
    let fP = NaN;
    if (dof > 0 && r2Train < 1) {
      fStat = (r2Train / p) / ((1 - r2Train) / dof);
      fP = S.fSf(fStat, p, dof);
    }

    /* --- hold-out metrics + residual diagnostics ------------------------- */
    const Xte = cache.Xte;
    const yte = cache.yte;
    const nTe = yte.length;
    const resid = new Array(nTe);
    const preds = new Array(nTe);
    let absSum = 0;
    for (let r = 0; r < nTe; r++) {
      let yhat = beta[0];
      for (let i = 0; i < p; i++) yhat += beta[i + 1] * Xte[r][featIdx[i]];
      preds[r] = yhat;
      resid[r] = yte[r] - yhat;
      absSum += Math.abs(resid[r]);
    }
    const mae = nTe ? absSum / nTe : NaN;
    const r2Test = base.r2Test;
    const adjR2 =
      nTe - p - 1 > 0 ? 1 - ((1 - r2Test) * (nTe - 1)) / (nTe - p - 1) : NaN;

    const order = [];
    for (let r = 0; r < nTe; r++) order.push(r);
    order.sort((a, b) => Math.abs(resid[b]) - Math.abs(resid[a]));
    const topN = Math.min(o.topOutliers || 10, nTe);
    const outliers = [];
    for (let q = 0; q < topN; q++) {
      const r = order[q];
      outliers.push({
        rank: q + 1,
        index: testRowIds ? testRowIds[r] : r,
        actual: yte[r],
        predicted: preds[r],
        residual: resid[r],
        abs_error: Math.abs(resid[r]),
        pct_error: yte[r] !== 0 ? Math.abs(resid[r] / yte[r]) * 100 : NaN
      });
    }

    return {
      k: p,
      featIdx: featIdx.slice(),
      features: featIdx.map((i) => names[i]),
      rmse: base.rmse,
      mae: mae,
      r2: r2Test,
      adj_r2: adjR2,
      r2_train: r2Train,
      f_stat: fStat,
      f_pvalue: fP,
      coefficients: coefficients,
      outliers: outliers
    };
  }

  /* ---------------------------------------------------------------------------
   * Standardisation + penalised coordinate descent
   * ------------------------------------------------------------------------ */

  /* StandardScaler: mean 0, population std (ddof = 0), as scikit-learn does.
   * Zero-variance columns are left at 0 rather than producing NaN. */
  function standardize(X) {
    const n = X.length;
    const p = n ? X[0].length : 0;
    const mu = new Array(p).fill(0);
    const sd = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += X[r][j];
      mu[j] = s / n;
      let v = 0;
      for (let r = 0; r < n; r++) {
        const d = X[r][j] - mu[j];
        v += d * d;
      }
      sd[j] = Math.sqrt(v / n) || 1;
    }
    const Z = [];
    for (let r = 0; r < n; r++) {
      const row = new Array(p);
      for (let j = 0; j < p; j++) row[j] = (X[r][j] - mu[j]) / sd[j];
      Z.push(row);
    }
    return { Z: Z, mu: mu, sd: sd };
  }

  function softThreshold(z, g) {
    if (z > g) return z - g;
    if (z < -g) return z + g;
    return 0;
  }

  /* Coordinate descent for
   *   1/(2n)·||y - Zw||² + alpha·l1Ratio·||w||₁ + alpha(1-l1Ratio)/2·||w||²
   * which is scikit-learn's ElasticNet objective (Lasso at l1Ratio = 1).
   * Z must be standardized and y centred. */
  function coordinateDescent(Z, yc, alpha, l1Ratio, maxIter, tol) {
    const n = Z.length;
    const p = n ? Z[0].length : 0;
    const w = new Array(p).fill(0);
    const resid = yc.slice();
    const l1 = alpha * l1Ratio;
    const l2 = alpha * (1 - l1Ratio);
    /* Z is standardized so Zⱼ·Zⱼ = n exactly, hence the denominator is 1 + l2. */
    const denom = 1 + l2;
    const iters = maxIter || 1000;
    const eps = tol === undefined ? 1e-7 : tol;

    for (let it = 0; it < iters; it++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const wj = w[j];
        let rho = 0;
        for (let r = 0; r < n; r++) rho += Z[r][j] * resid[r];
        rho = rho / n + wj; /* add back own contribution */
        const newW = softThreshold(rho, l1) / denom;
        const delta = newW - wj;
        if (delta !== 0) {
          for (let r = 0; r < n; r++) resid[r] -= Z[r][j] * delta;
          w[j] = newW;
          const ad = Math.abs(delta);
          if (ad > maxDelta) maxDelta = ad;
        }
      }
      if (maxDelta < eps) break;
    }
    return w;
  }

  /* Smallest alpha at which every coefficient is still zero. */
  function alphaMax(Z, yc, l1Ratio) {
    const n = Z.length;
    const p = n ? Z[0].length : 0;
    let m = 0;
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += Z[r][j] * yc[r];
      m = Math.max(m, Math.abs(s) / n);
    }
    return m / Math.max(l1Ratio, 1e-3);
  }

  /* Contiguous folds — scikit-learn's KFold(n_splits) with shuffle=False. */
  function kFold(n, nSplits) {
    const k = Math.min(nSplits, n);
    const sizes = new Array(k).fill(Math.floor(n / k));
    for (let i = 0; i < n % k; i++) sizes[i] += 1;
    const folds = [];
    let start = 0;
    for (let i = 0; i < k; i++) {
      const stop = start + sizes[i];
      const test = [];
      const train = [];
      for (let r = 0; r < n; r++) (r >= start && r < stop ? test : train).push(r);
      folds.push({ train: train, test: test });
      start = stop;
    }
    return folds;
  }

  function takeRows(X, idx) {
    return idx.map((i) => X[i]);
  }

  /* Pick alpha by 5-fold CV over a log grid, re-standardising inside each fold
   * so the selection stays honest, then refit on the whole training set. */
  function fitPenalisedCV(Xtr, ytr, l1Ratio) {
    const full = standardize(Xtr);
    const ybar = S.mean(ytr);
    const yc = ytr.map((v) => v - ybar);
    const aMax = alphaMax(full.Z, yc, l1Ratio);
    if (!(aMax > 0)) {
      return { coef: new Array(full.Z[0] ? full.Z[0].length : 0).fill(0), alpha: 0 };
    }

    const nGrid = 40;
    const grid = [];
    for (let i = 0; i < nGrid; i++) {
      grid.push(aMax * Math.pow(1e-3, i / (nGrid - 1)));
    }

    const folds = kFold(Xtr.length, 5);
    let bestAlpha = grid[0];
    let bestMse = Infinity;
    for (let g = 0; g < grid.length; g++) {
      let sse = 0;
      let cnt = 0;
      for (let f = 0; f < folds.length; f++) {
        const Xa = takeRows(Xtr, folds[f].train);
        const ya = folds[f].train.map((i) => ytr[i]);
        const sc = standardize(Xa);
        const mA = S.mean(ya);
        const w = coordinateDescent(sc.Z, ya.map((v) => v - mA), grid[g], l1Ratio, 500, 1e-6);
        for (let q = 0; q < folds[f].test.length; q++) {
          const r = folds[f].test[q];
          let yhat = mA;
          for (let j = 0; j < w.length; j++) {
            if (w[j] === 0) continue;
            yhat += w[j] * ((Xtr[r][j] - sc.mu[j]) / sc.sd[j]);
          }
          const d = ytr[r] - yhat;
          sse += d * d;
          cnt++;
        }
      }
      const mse = sse / cnt;
      if (mse < bestMse - 1e-12) {
        bestMse = mse;
        bestAlpha = grid[g];
      }
    }

    return {
      coef: coordinateDescent(full.Z, yc, bestAlpha, l1Ratio, 5000, 1e-8),
      alpha: bestAlpha,
      cvMse: bestMse
    };
  }

  /* ---------------------------------------------------------------------------
   * Feature-selection algorithms
   *
   * Each returns an array `sets` where sets[k-1] is the chosen feature-index
   * array of size k.
   * ------------------------------------------------------------------------ */

  /* |coef| ranking from a penalised fit, ties broken by the standardized OLS
   * magnitude so that shrunk-to-zero features still get a stable order. */
  function rankingSets(coef, tieBreak, p) {
    const order = [];
    for (let i = 0; i < p; i++) order.push(i);
    order.sort((a, b) => {
      const d = Math.abs(coef[b]) - Math.abs(coef[a]);
      if (Math.abs(d) > 1e-12) return d;
      return Math.abs(tieBreak[b]) - Math.abs(tieBreak[a]);
    });
    const sets = [];
    for (let k = 1; k <= p; k++) sets.push(order.slice(0, k).sort((a, b) => a - b));
    return sets;
  }

  /* Greedy forward selection maximising the mean 5-fold CV R² on the training
   * set — scikit-learn's SequentialFeatureSelector(direction='forward'). */
  function forwardSFS(Xtr, ytr, p) {
    const folds = kFold(Xtr.length, 5);
    const chosen = [];
    const remaining = [];
    for (let i = 0; i < p; i++) remaining.push(i);
    const sets = [];

    /* Per-fold Gram caches so each candidate score is O(k³). */
    const caches = folds.map((f) =>
      buildCache(
        takeRows(Xtr, f.train),
        f.train.map((i) => ytr[i]),
        takeRows(Xtr, f.test),
        f.test.map((i) => ytr[i])
      )
    );

    while (remaining.length) {
      let bestFeat = remaining[0];
      let bestScore = -Infinity;
      for (let q = 0; q < remaining.length; q++) {
        const trial = chosen.concat([remaining[q]]).sort((a, b) => a - b);
        let sum = 0;
        let ok = true;
        for (let c = 0; c < caches.length; c++) {
          const fit = fitFast(caches[c], trial);
          if (!fit) {
            ok = false;
            break;
          }
          sum += fit.r2Test;
        }
        const score = ok ? sum / caches.length : -Infinity;
        if (score > bestScore) {
          bestScore = score;
          bestFeat = remaining[q];
        }
      }
      chosen.push(bestFeat);
      remaining.splice(remaining.indexOf(bestFeat), 1);
      sets.push(chosen.slice().sort((a, b) => a - b));
    }
    return sets;
  }

  /* Recursive feature elimination: drop the smallest |coef| of an OLS fit on
   * standardized predictors, repeat. */
  function rfe(Xtr, ytr, p) {
    const sc = standardize(Xtr);
    const cache = buildCache(sc.Z, ytr, sc.Z, ytr);
    let alive = [];
    for (let i = 0; i < p; i++) alive.push(i);
    const sets = new Array(p);
    sets[p - 1] = alive.slice();

    while (alive.length > 1) {
      const fit = fitFast(cache, alive);
      let dropAt = 0;
      if (fit) {
        let smallest = Infinity;
        for (let i = 0; i < alive.length; i++) {
          const m = Math.abs(fit.beta[i + 1]);
          if (m < smallest) {
            smallest = m;
            dropAt = i;
          }
        }
      }
      alive = alive.slice(0, dropAt).concat(alive.slice(dropAt + 1));
      sets[alive.length - 1] = alive.slice();
    }
    return sets;
  }

  function* combinations(pool, k) {
    const n = pool.length;
    if (k > n) return;
    const idx = [];
    for (let i = 0; i < k; i++) idx.push(i);
    while (true) {
      yield idx.map((i) => pool[i]);
      let i = k - 1;
      while (i >= 0 && idx[i] === i + n - k) i--;
      if (i < 0) return;
      idx[i]++;
      for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
    }
  }

  /* Best-subset search. The criterion is hold-out RMSE, matching the reference
   * implementation this dashboard is benchmarked against — it is therefore an
   * optimistic bound, and the UI labels it as such. */
  function exhaustive(cache, p, pool) {
    const candidates = pool || (function () {
      const a = [];
      for (let i = 0; i < p; i++) a.push(i);
      return a;
    })();
    const sets = new Array(p).fill(null);
    for (let k = 1; k <= candidates.length; k++) {
      let best = null;
      let bestRmse = Infinity;
      for (const combo of combinations(candidates, k)) {
        const fit = fitFast(cache, combo);
        if (fit && fit.rmse < bestRmse) {
          bestRmse = fit.rmse;
          best = combo;
        }
      }
      if (best) sets[k - 1] = best;
    }
    return sets;
  }

  const ALGORITHMS = [
    'Lasso',
    'Forward SFS',
    'RFE',
    'ElasticNet',
    'Exhaustive'
  ];

  /* ---------------------------------------------------------------------------
   * Top-level driver
   * ------------------------------------------------------------------------ */

  /* design: { X, y, featureNames, rowLabels }
   * options: { testSize, seed, maxK, topOutliers } */
  function analyse(design, options) {
    const o = options || {};
    const testSize = o.testSize === undefined ? 0.2 : o.testSize;
    const seed = o.seed === undefined ? 0 : o.seed;
    const X = design.X;
    const y = design.y;
    const names = design.featureNames;
    const n = X.length;
    const p = names.length;

    if (n < 8) throw new Error('Need at least 8 complete rows to fit and validate a model.');
    if (p < 1) throw new Error('Need at least one predictor column.');

    const split = S.trainTestSplit(n, testSize, seed);
    if (split.train.length < p + 2) {
      throw new Error(
        'Training split (' + split.train.length + ' rows) is too small for ' +
        p + ' predictors. Reduce the test size or drop some features.'
      );
    }

    const Xtr = takeRows(X, split.train);
    const ytr = split.train.map((i) => y[i]);
    const Xte = takeRows(X, split.test);
    const yte = split.test.map((i) => y[i]);
    const testRowIds = split.test.map((i) =>
      design.rowLabels ? design.rowLabels[i] : i
    );

    const cache = buildCache(Xtr, ytr, Xte, yte);
    const maxK = Math.max(1, Math.min(o.maxK || p, p, split.train.length - 2));

    /* OLS on standardized predictors: the tie-breaker for shrunk coefficients
     * and the fallback importance ranking. */
    const scTr = standardize(Xtr);
    const olsStd = S.lstsq(S.addIntercept(scTr.Z), ytr);
    const stdCoef = olsStd ? olsStd.slice(1) : new Array(p).fill(0);

    const lassoFit = fitPenalisedCV(Xtr, ytr, 1.0);
    const enetFit = fitPenalisedCV(Xtr, ytr, 0.5);

    const sets = {};
    sets.Lasso = rankingSets(lassoFit.coef, stdCoef, p);
    sets.ElasticNet = rankingSets(enetFit.coef, stdCoef, p);
    sets['Forward SFS'] = forwardSFS(Xtr, ytr, p);
    sets.RFE = rfe(Xtr, ytr, p);

    /* Keep the exhaustive search tractable: when there are too many predictors,
     * restrict the pool to the strongest ones by Lasso magnitude. */
    let pool = null;
    let poolNote = null;
    if (p > EXHAUSTIVE_MAX_POOL) {
      const order = [];
      for (let i = 0; i < p; i++) order.push(i);
      order.sort(
        (a, b) =>
          Math.abs(lassoFit.coef[b]) - Math.abs(lassoFit.coef[a]) ||
          Math.abs(stdCoef[b]) - Math.abs(stdCoef[a])
      );
      pool = order.slice(0, EXHAUSTIVE_MAX_POOL).sort((a, b) => a - b);
      poolNote =
        'Best-subset search restricted to the top ' + EXHAUSTIVE_MAX_POOL +
        ' predictors by Lasso magnitude (' + p + ' available) to keep 2^p tractable.';
    }
    sets.Exhaustive = exhaustive(cache, p, pool);

    /* Expand every selected subset into a full inference bundle. */
    const models = {};
    for (let a = 0; a < ALGORITHMS.length; a++) {
      const algo = ALGORITHMS[a];
      models[algo] = [];
      for (let k = 1; k <= maxK; k++) {
        const featIdx = sets[algo][k - 1];
        if (!featIdx) {
          models[algo].push(null);
          continue;
        }
        models[algo].push(
          fitFull(cache, featIdx, names, testRowIds, { topOutliers: o.topOutliers || 10 })
        );
      }
    }

    /* Global sweet spot = lowest hold-out RMSE anywhere in the grid. */
    let bestGlobal = null;
    for (let a = 0; a < ALGORITHMS.length; a++) {
      const algo = ALGORITHMS[a];
      for (let k = 0; k < models[algo].length; k++) {
        const m = models[algo][k];
        if (!m || !isFinite(m.rmse)) continue;
        if (!bestGlobal || m.rmse < bestGlobal.rmse) {
          bestGlobal = { algo: algo, k: k + 1, rmse: m.rmse, r2: m.r2 };
        }
      }
    }

    /* Per-algorithm sweet spot and the full-model reference point. */
    const perAlgoBest = {};
    for (let a = 0; a < ALGORITHMS.length; a++) {
      const algo = ALGORITHMS[a];
      let best = null;
      for (let k = 0; k < models[algo].length; k++) {
        const m = models[algo][k];
        if (!m || !isFinite(m.rmse)) continue;
        if (!best || m.rmse < best.rmse) best = { k: k + 1, rmse: m.rmse };
      }
      perAlgoBest[algo] = best;
    }

    const fullModel = models.Lasso[maxK - 1] || null;

    return {
      algorithms: ALGORITHMS,
      models: models,
      maxK: maxK,
      featureNames: names,
      split: {
        nTrain: split.train.length,
        nTest: split.test.length,
        testSize: testSize,
        seed: seed,
        trainIdx: split.train,
        testIdx: split.test
      },
      tuning: { lassoAlpha: lassoFit.alpha, enetAlpha: enetFit.alpha },
      bestGlobal: bestGlobal,
      perAlgoBest: perAlgoBest,
      fullModel: fullModel,
      poolNote: poolNote
    };
  }

  global.Regression = {
    ALGORITHMS: ALGORITHMS,
    buildCache: buildCache,
    fitFast: fitFast,
    fitFull: fitFull,
    standardize: standardize,
    coordinateDescent: coordinateDescent,
    fitPenalisedCV: fitPenalisedCV,
    kFold: kFold,
    forwardSFS: forwardSFS,
    rfe: rfe,
    exhaustive: exhaustive,
    combinations: combinations,
    analyse: analyse
  };
})(typeof window !== 'undefined' ? window : globalThis);
