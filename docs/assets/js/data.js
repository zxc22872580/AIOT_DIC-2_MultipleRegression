/* =============================================================================
 * data.js — CSV ingestion, schema inference and design-matrix construction
 *
 * Turns an arbitrary CSV (URL, file upload or pasted text) into the numeric
 * design matrix the regression layer expects, applying the same conventions as
 * pandas: one-hot encoding with the first level dropped, numeric columns left
 * in their original units.
 *
 * Depends on: stats.js
 * ========================================================================== */
(function (global) {
  'use strict';

  const S = global.Stats;

  /* Treat a categorical column with more levels than this as an identifier and
   * leave it out of the default feature set. */
  const MAX_LEVELS = 20;

  const MISSING_TOKENS = ['', 'na', 'n/a', 'nan', 'null', 'none', '?', '-', '.'];

  function isMissing(v) {
    return MISSING_TOKENS.indexOf(String(v).trim().toLowerCase()) !== -1;
  }

  /* ---------------------------------------------------------------------------
   * CSV parsing — RFC 4180-ish: quoted fields, escaped quotes, CR/LF/CRLF.
   * Also sniffs between ',', ';' and tab so European exports work.
   * ------------------------------------------------------------------------ */

  function sniffDelimiter(text) {
    const head = text.slice(0, 8192).split(/\r?\n/).slice(0, 5).join('\n');
    const candidates = [',', ';', '\t', '|'];
    let best = ',';
    let bestScore = -1;
    for (let i = 0; i < candidates.length; i++) {
      const d = candidates[i];
      /* count only delimiters outside quotes */
      let count = 0;
      let inQ = false;
      for (let c = 0; c < head.length; c++) {
        const ch = head[c];
        if (ch === '"') inQ = !inQ;
        else if (ch === d && !inQ) count++;
      }
      if (count > bestScore) {
        bestScore = count;
        best = d;
      }
    }
    return best;
  }

  function parseCSV(text, delimiter) {
    let src = text.replace(/^﻿/, '');
    const d = delimiter || sniffDelimiter(src);
    const rows = [];
    let row = [];
    let field = '';
    let inQuotes = false;
    let i = 0;

    while (i < src.length) {
      const ch = src[i];
      if (inQuotes) {
        if (ch === '"') {
          if (src[i + 1] === '"') {
            field += '"';
            i += 2;
            continue;
          }
          inQuotes = false;
          i++;
          continue;
        }
        field += ch;
        i++;
        continue;
      }
      if (ch === '"') {
        inQuotes = true;
        i++;
        continue;
      }
      if (ch === d) {
        row.push(field);
        field = '';
        i++;
        continue;
      }
      if (ch === '\n' || ch === '\r') {
        row.push(field);
        field = '';
        rows.push(row);
        row = [];
        if (ch === '\r' && src[i + 1] === '\n') i += 2;
        else i++;
        continue;
      }
      field += ch;
      i++;
    }
    if (field.length || row.length) {
      row.push(field);
      rows.push(row);
    }

    /* drop fully blank lines */
    const clean = rows.filter((r) => r.some((c) => String(c).trim() !== ''));
    if (!clean.length) throw new Error('The file contains no data rows.');

    const header = clean[0].map((h, idx) => {
      const name = String(h).trim();
      return name || 'column_' + (idx + 1);
    });

    /* de-duplicate repeated header names */
    const seen = {};
    const columns = header.map((name) => {
      if (seen[name] === undefined) {
        seen[name] = 0;
        return name;
      }
      seen[name] += 1;
      return name + '.' + seen[name];
    });

    const data = [];
    for (let r = 1; r < clean.length; r++) {
      const raw = clean[r];
      const obj = {};
      for (let c = 0; c < columns.length; c++) {
        obj[columns[c]] = raw[c] === undefined ? '' : String(raw[c]).trim();
      }
      data.push(obj);
    }
    if (!data.length) throw new Error('The file has a header but no data rows.');
    return { columns: columns, rows: data, delimiter: d };
  }

  /* ---------------------------------------------------------------------------
   * Schema inference
   * ------------------------------------------------------------------------ */

  function parseNumber(v) {
    const s = String(v).trim().replace(/,/g, '');
    if (s === '') return NaN;
    const n = Number(s);
    return isFinite(n) ? n : NaN;
  }

  function profile(table) {
    return table.columns.map((name) => {
      const values = table.rows.map((r) => r[name]);
      let missing = 0;
      let numericOk = 0;
      let present = 0;
      const levels = {};
      const nums = [];

      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (isMissing(v)) {
          missing++;
          continue;
        }
        present++;
        const n = parseNumber(v);
        if (!isNaN(n)) {
          numericOk++;
          nums.push(n);
        }
        if (Object.keys(levels).length <= MAX_LEVELS + 1) {
          levels[v] = (levels[v] || 0) + 1;
        }
      }

      const isNumeric = present > 0 && numericOk === present;
      const levelNames = Object.keys(levels).sort();
      const levelsCapped = levelNames.length > MAX_LEVELS;
      const sorted = nums.slice().sort((a, b) => a - b);
      const allIntegers = nums.length > 0 && nums.every((v) => Number.isInteger(v));

      return {
        name: name,
        type: isNumeric ? 'numeric' : 'categorical',
        missing: missing,
        present: present,
        nLevels: isNumeric ? null : levelNames.length,
        levels: isNumeric ? null : levelNames,
        /* counting stops just past the cap, so the exact total is unknown */
        levelsCapped: !isNumeric && levelsCapped,
        allIntegers: isNumeric && allIntegers,
        highCardinality: !isNumeric && levelsCapped,
        min: isNumeric && sorted.length ? sorted[0] : null,
        max: isNumeric && sorted.length ? sorted[sorted.length - 1] : null,
        mean: isNumeric && nums.length ? S.mean(nums) : null,
        std: isNumeric && nums.length ? S.std(nums, 1) : null,
        median: isNumeric && sorted.length ? S.quantile(sorted, 0.5) : null
      };
    });
  }

  /* Columns that are almost never the thing you want to predict, even though
   * they are numeric: row ids, keys, years and similar bookkeeping. */
  const BOOKKEEPING = /^(id|index|idx|no|num|key|code|year|yr|model[_ ]?year|zip|zipcode|postcode|rank)$/i;

  function looksLikeBookkeeping(col) {
    return col.allIntegers && BOOKKEEPING.test(col.name.trim());
  }

  /* Default target: the last numeric column — the convention in virtually every
   * teaching dataset, including 50 Startups (Profit) and Boston (medv) — but
   * skipping id/year-style columns, which sit last often enough to matter. */
  function defaultTarget(cols) {
    const numeric = cols.filter((c) => c.type === 'numeric');
    if (!numeric.length) return null;
    for (let i = numeric.length - 1; i >= 0; i--) {
      if (!looksLikeBookkeeping(numeric[i])) return numeric[i].name;
    }
    return numeric[numeric.length - 1].name;
  }

  function defaultFeatures(cols, target) {
    return cols
      .filter(
        (c) =>
          c.name !== target &&
          c.present > 0 &&
          !c.highCardinality &&
          !(c.type === 'categorical' && c.nLevels < 2)
      )
      .map((c) => c.name);
  }

  /* ---------------------------------------------------------------------------
   * Design matrix
   *
   * Numeric features pass through unchanged; categorical features become k-1
   * dummies named "<col>_<level>" with the alphabetically first level held out
   * as the reference, matching pandas.get_dummies(drop_first=True).
   * ------------------------------------------------------------------------ */

  function buildDesign(table, cols, target, featureNames) {
    const byName = {};
    cols.forEach((c) => {
      byName[c.name] = c;
    });

    const targetCol = byName[target];
    if (!targetCol) throw new Error('Target column "' + target + '" not found.');
    if (targetCol.type !== 'numeric') {
      throw new Error(
        'Target "' + target + '" is not numeric. Linear regression needs a continuous target.'
      );
    }
    if (!featureNames.length) throw new Error('Select at least one predictor column.');

    /* plan the expanded columns */
    const plan = [];
    featureNames.forEach((name) => {
      const c = byName[name];
      if (!c) return;
      if (c.type === 'numeric') {
        plan.push({ kind: 'numeric', source: name, label: name });
      } else {
        const levels = c.levels || [];
        if (levels.length < 2) return;
        const reference = levels[0];
        for (let i = 1; i < levels.length; i++) {
          plan.push({
            kind: 'dummy',
            source: name,
            level: levels[i],
            reference: reference,
            label: name + '_' + levels[i]
          });
        }
      }
    });
    if (!plan.length) throw new Error('The selected predictors expand to zero usable columns.');

    const X = [];
    const y = [];
    const rowLabels = [];
    const keptRows = [];
    let dropped = 0;

    for (let r = 0; r < table.rows.length; r++) {
      const row = table.rows[r];
      const yv = parseNumber(row[target]);
      if (isMissing(row[target]) || isNaN(yv)) {
        dropped++;
        continue;
      }
      let bad = false;
      const vec = new Array(plan.length);
      for (let j = 0; j < plan.length; j++) {
        const pl = plan[j];
        const raw = row[pl.source];
        if (isMissing(raw)) {
          bad = true;
          break;
        }
        if (pl.kind === 'numeric') {
          const v = parseNumber(raw);
          if (isNaN(v)) {
            bad = true;
            break;
          }
          vec[j] = v;
        } else {
          vec[j] = raw === pl.level ? 1 : 0;
        }
      }
      if (bad) {
        dropped++;
        continue;
      }
      X.push(vec);
      y.push(yv);
      rowLabels.push(r);
      keptRows.push(row);
    }

    if (X.length < 8) {
      throw new Error(
        'Only ' + X.length + ' complete rows remain after dropping missing values — ' +
        'not enough to train and validate.'
      );
    }

    /* drop constant columns: they are collinear with the intercept */
    const constant = [];
    for (let j = plan.length - 1; j >= 0; j--) {
      let allSame = true;
      for (let r = 1; r < X.length; r++) {
        if (X[r][j] !== X[0][j]) {
          allSame = false;
          break;
        }
      }
      if (allSame) {
        constant.push(plan[j].label);
        for (let r = 0; r < X.length; r++) X[r].splice(j, 1);
        plan.splice(j, 1);
      }
    }
    if (!plan.length) throw new Error('Every selected predictor is constant.');

    /* per-feature correlation with the target, for the EDA panel */
    const columnsMeta = plan.map((pl, j) => {
      const col = X.map((row) => row[j]);
      const sorted = col.slice().sort((a, b) => a - b);
      return {
        label: pl.label,
        kind: pl.kind,
        source: pl.source,
        reference: pl.reference || null,
        min: sorted[0],
        max: sorted[sorted.length - 1],
        mean: S.mean(col),
        std: S.std(col, 1),
        median: S.quantile(sorted, 0.5),
        corr: S.pearson(col, y)
      };
    });

    const ysorted = y.slice().sort((a, b) => a - b);

    return {
      X: X,
      y: y,
      rowLabels: rowLabels,
      keptRows: keptRows,
      featureNames: plan.map((pl) => pl.label),
      columnsMeta: columnsMeta,
      target: target,
      targetStats: {
        min: ysorted[0],
        max: ysorted[ysorted.length - 1],
        mean: S.mean(y),
        std: S.std(y, 1),
        median: S.quantile(ysorted, 0.5)
      },
      droppedRows: dropped,
      droppedConstant: constant.reverse(),
      nRows: X.length
    };
  }

  /* ---------------------------------------------------------------------------
   * Loading
   * ------------------------------------------------------------------------ */

  /* Fetch a CSV over the network. Many hosts (raw.githubusercontent.com,
   * most CDNs, data portals) send permissive CORS headers; those that do not
   * cannot be read from a static page, so the error says so plainly instead of
   * surfacing a bare "Failed to fetch". */
  function fetchCSV(url) {
    return fetch(url, { mode: 'cors', redirect: 'follow' })
      .then((res) => {
        if (!res.ok) {
          throw new Error('The server answered HTTP ' + res.status + ' ' + res.statusText + '.');
        }
        return res.text();
      })
      .catch((err) => {
        if (err instanceof TypeError) {
          throw new Error(
            'Could not read that URL from the browser. The host either blocked the ' +
            'request or sent no CORS header. Download the file and use "Upload file", ' +
            'or paste the contents into the Paste CSV tab.'
          );
        }
        throw err;
      });
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error('Could not read "' + file.name + '".'));
      reader.readAsText(file);
    });
  }

  function load(text) {
    const table = parseCSV(text);
    const cols = profile(table);
    const target = defaultTarget(cols);
    if (!target) {
      throw new Error(
        'No numeric column found, so there is nothing to regress. ' +
        'Check the delimiter and the decimal separator.'
      );
    }
    return {
      table: table,
      cols: cols,
      target: target,
      features: defaultFeatures(cols, target)
    };
  }

  global.DataKit = {
    MAX_LEVELS: MAX_LEVELS,
    parseCSV: parseCSV,
    sniffDelimiter: sniffDelimiter,
    parseNumber: parseNumber,
    isMissing: isMissing,
    profile: profile,
    defaultTarget: defaultTarget,
    defaultFeatures: defaultFeatures,
    buildDesign: buildDesign,
    fetchCSV: fetchCSV,
    readFile: readFile,
    load: load
  };
})(typeof window !== 'undefined' ? window : globalThis);
