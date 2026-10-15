/* =============================================================================
 * stats.js — numerics layer
 *
 *  1. MT19937 + numpy.random.RandomState parity  -> identical train/test splits
 *     to scikit-learn's train_test_split(random_state=N)
 *  2. Dense linear algebra (Gauss-Jordan inverse, matmul helpers)
 *  3. Distribution tails (Student-t, Fisher-F) via the regularized incomplete
 *     beta function -> p-values and confidence intervals without a server
 *
 * No dependencies. Everything here is pure and side-effect free.
 * ========================================================================== */
(function (global) {
  'use strict';

  /* ---------------------------------------------------------------------------
   * 1. Mersenne Twister — bit-compatible with numpy's legacy RandomState
   * ------------------------------------------------------------------------ */

  const N = 624;
  const M = 397;
  const MATRIX_A = 0x9908b0df;
  const UPPER_MASK = 0x80000000;
  const LOWER_MASK = 0x7fffffff;

  function MT19937(seed) {
    this.mt = new Uint32Array(N);
    this.mti = N + 1;
    this.initGenrand(seed >>> 0);
  }

  /* numpy's mt19937_seed() is the reference init_genrand() written as a
   * forward recurrence; both produce the same state vector. */
  MT19937.prototype.initGenrand = function (s) {
    let seed = s >>> 0;
    for (let pos = 0; pos < N; pos++) {
      this.mt[pos] = seed;
      // seed = 1812433253 * (seed ^ (seed >> 30)) + pos + 1
      const x = (seed ^ (seed >>> 30)) >>> 0;
      seed = (Math.imul(1812433253, x) + pos + 1) >>> 0;
    }
    this.mti = N;
  };

  MT19937.prototype.genrandUint32 = function () {
    let y;
    if (this.mti >= N) {
      let kk;
      for (kk = 0; kk < N - M; kk++) {
        y = ((this.mt[kk] & UPPER_MASK) | (this.mt[kk + 1] & LOWER_MASK)) >>> 0;
        this.mt[kk] = (this.mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      }
      for (; kk < N - 1; kk++) {
        y = ((this.mt[kk] & UPPER_MASK) | (this.mt[kk + 1] & LOWER_MASK)) >>> 0;
        this.mt[kk] =
          (this.mt[kk + (M - N)] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      }
      y = ((this.mt[N - 1] & UPPER_MASK) | (this.mt[0] & LOWER_MASK)) >>> 0;
      this.mt[N - 1] = (this.mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      this.mti = 0;
    }
    y = this.mt[this.mti++];
    y = (y ^ (y >>> 11)) >>> 0;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y = (y ^ (y >>> 18)) >>> 0;
    return y >>> 0;
  };

  /* Smallest all-ones mask >= max (numpy gen_mask). */
  function genMask(max) {
    let mask = max >>> 0;
    mask |= mask >>> 1;
    mask |= mask >>> 2;
    mask |= mask >>> 4;
    mask |= mask >>> 8;
    mask |= mask >>> 16;
    return mask >>> 0;
  }

  /* numpy random_interval(): masked rejection sampling on [0, max]. */
  MT19937.prototype.randomInterval = function (max) {
    if (max === 0) return 0;
    const mask = genMask(max);
    let value;
    do {
      value = (this.genrandUint32() & mask) >>> 0;
    } while (value > max);
    return value;
  };

  /* numpy RandomState.permutation(n): arange(n) then legacy Fisher-Yates. */
  function numpyPermutation(n, seed) {
    const rng = new MT19937(seed >>> 0);
    const arr = new Int32Array(n);
    for (let i = 0; i < n; i++) arr[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = rng.randomInterval(i);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return Array.from(arr);
  }

  /* sklearn.model_selection.train_test_split(shuffle=True, random_state=seed).
   * _validate_shuffle_split rounds the test count up and gives the remainder to
   * train; ShuffleSplit then takes perm[:n_test] as test and the next n_train
   * entries as train. */
  function trainTestSplit(n, testSize, seed) {
    const nTest = Math.ceil(testSize * n);
    const nTrain = n - nTest;
    const perm = numpyPermutation(n, seed);
    return {
      test: perm.slice(0, nTest),
      train: perm.slice(nTest, nTest + nTrain)
    };
  }

  /* ---------------------------------------------------------------------------
   * 2. Linear algebra
   * ------------------------------------------------------------------------ */

  /* Gauss-Jordan with partial pivoting. Returns null when (near-)singular so
   * callers can degrade gracefully instead of propagating NaN. */
  function matInverse(A) {
    const n = A.length;
    const a = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j];
      r[n + i] = 1;
      return r;
    });

    for (let col = 0; col < n; col++) {
      let piv = col;
      let best = Math.abs(a[col][col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(a[r][col]);
        if (v > best) {
          best = v;
          piv = r;
        }
      }
      if (best < 1e-12) return null;
      if (piv !== col) {
        const t = a[piv];
        a[piv] = a[col];
        a[col] = t;
      }
      const d = a[col][col];
      for (let j = 0; j < 2 * n; j++) a[col][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = a[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[col][j];
      }
    }
    return a.map((row) => Array.from(row.subarray(n)));
  }

  /* Xᵀ·X for an n×p matrix given as an array of rows. */
  function gram(X) {
    const n = X.length;
    const p = n ? X[0].length : 0;
    const out = [];
    for (let i = 0; i < p; i++) out.push(new Array(p).fill(0));
    for (let r = 0; r < n; r++) {
      const row = X[r];
      for (let i = 0; i < p; i++) {
        const vi = row[i];
        if (vi === 0) continue;
        for (let j = i; j < p; j++) out[i][j] += vi * row[j];
      }
    }
    for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) out[i][j] = out[j][i];
    return out;
  }

  /* Xᵀ·y */
  function matTVec(X, y) {
    const n = X.length;
    const p = n ? X[0].length : 0;
    const out = new Array(p).fill(0);
    for (let r = 0; r < n; r++) {
      const row = X[r];
      const yr = y[r];
      if (yr === 0) continue;
      for (let j = 0; j < p; j++) out[j] += row[j] * yr;
    }
    return out;
  }

  /* A·v */
  function matVec(A, v) {
    return A.map((row) => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += row[j] * v[j];
      return s;
    });
  }

  /* Prepend a column of ones. */
  function addIntercept(X) {
    return X.map((row) => [1].concat(row));
  }

  /* Least squares via the normal equations; null when rank-deficient. */
  function lstsq(X, y) {
    const G = gram(X);
    const Ginv = matInverse(G);
    if (!Ginv) return null;
    return matVec(Ginv, matTVec(X, y));
  }

  /* ---------------------------------------------------------------------------
   * 3. Distributions
   * ------------------------------------------------------------------------ */

  const LANCZOS = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7
  ];

  function logGamma(x) {
    if (x < 0.5) {
      // reflection: Γ(x)Γ(1-x) = π / sin(πx)
      return (
        Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x)
      );
    }
    const z = x - 1;
    let a = 0.99999999999980993;
    const t = z + 7.5;
    for (let i = 0; i < LANCZOS.length; i++) a += LANCZOS[i] / (z + i + 1);
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
  }

  /* Continued fraction for the incomplete beta (Lentz's method, NR §6.4). */
  function betacf(a, b, x) {
    const FPMIN = 1e-300;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-16) break;
    }
    return h;
  }

  /* Regularized incomplete beta I_x(a, b). */
  function incompleteBeta(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbt =
      logGamma(a + b) - logGamma(a) - logGamma(b) +
      a * Math.log(x) + b * Math.log1p(-x);
    const bt = Math.exp(lbt);
    if (x < (a + 1) / (a + b + 2)) return (bt * betacf(a, b, x)) / a;
    return 1 - (bt * betacf(b, a, 1 - x)) / b;
  }

  /* Two-sided p-value for a t statistic:  P(|T| >= |t|). */
  function studentTTwoSided(t, df) {
    if (!isFinite(t) || df <= 0) return NaN;
    const x = df / (df + t * t);
    return incompleteBeta(df / 2, 0.5, x);
  }

  /* P(T <= t) */
  function studentTCdf(t, df) {
    const half = studentTTwoSided(t, df) / 2;
    return t >= 0 ? 1 - half : half;
  }

  /* Upper tail of the F distribution: P(F >= f). */
  function fSf(f, d1, d2) {
    if (!isFinite(f) || f <= 0) return 1;
    const x = d2 / (d2 + d1 * f);
    return incompleteBeta(d2 / 2, d1 / 2, x);
  }

  /* Inverse t CDF by bisection — only ever called for the 0.975 quantile, so
   * robustness beats speed here. */
  function studentTInv(p, df) {
    if (df <= 0) return NaN;
    let lo = -1e3;
    let hi = 1e3;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (studentTCdf(mid, df) < p) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /* ---------------------------------------------------------------------------
   * Small descriptive helpers
   * ------------------------------------------------------------------------ */

  function mean(v) {
    if (!v.length) return NaN;
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i];
    return s / v.length;
  }

  function std(v, ddof) {
    const n = v.length;
    const d = ddof === undefined ? 0 : ddof;
    if (n - d <= 0) return 0;
    const m = mean(v);
    let s = 0;
    for (let i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
    return Math.sqrt(s / (n - d));
  }

  function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  /* Pearson correlation. */
  function pearson(a, b) {
    const n = a.length;
    const ma = mean(a);
    const mb = mean(b);
    let num = 0;
    let da = 0;
    let db = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma;
      const y = b[i] - mb;
      num += x * y;
      da += x * x;
      db += y * y;
    }
    const den = Math.sqrt(da * db);
    return den === 0 ? 0 : num / den;
  }

  global.Stats = {
    MT19937: MT19937,
    numpyPermutation: numpyPermutation,
    trainTestSplit: trainTestSplit,
    matInverse: matInverse,
    gram: gram,
    matTVec: matTVec,
    matVec: matVec,
    addIntercept: addIntercept,
    lstsq: lstsq,
    logGamma: logGamma,
    incompleteBeta: incompleteBeta,
    studentTTwoSided: studentTTwoSided,
    studentTCdf: studentTCdf,
    studentTInv: studentTInv,
    fSf: fSf,
    mean: mean,
    std: std,
    quantile: quantile,
    pearson: pearson
  };
})(typeof window !== 'undefined' ? window : globalThis);
