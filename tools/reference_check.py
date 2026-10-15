"""Reference implementation of the spec in docs/assets/js, written against
numpy/scipy/scikit-learn.

The JavaScript in docs/assets/js and this script are two independent
implementations of one specification. Run this to regenerate the fixtures that
docs/selftest.html compares against:

    python tools/reference_check.py

then serve docs/ and open selftest.html. It checks the MT19937 train/test split
against numpy, the t and F tails against scipy, and every coefficient, standard
error, t-statistic, p-value, confidence bound, VIF, RMSE, MAE, R2, adjusted R2,
F-statistic and residual row for all five algorithms at every k on both
built-in datasets.
"""
import itertools, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

L1_GRID_N = 40
L1_GRID_MIN_RATIO = 1e-3


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


def soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def coord_descent(Z, yc, alpha, l1_ratio, max_iter=5000, tol=1e-8):
    n, p = Z.shape
    w = np.zeros(p)
    resid = yc.copy()
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)
    denom = 1.0 + l2
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            wj = w[j]
            rho = Z[:, j] @ resid / n + wj
            new = soft(rho, l1) / denom
            d = new - wj
            if d != 0.0:
                resid -= Z[:, j] * d
                w[j] = new
                max_delta = max(max_delta, abs(d))
        if max_delta < tol:
            break
    return w


def alpha_max(Z, yc, l1_ratio):
    n = Z.shape[0]
    return np.max(np.abs(Z.T @ yc) / n) / max(l1_ratio, 1e-3)


def kfold(n, k=5):
    k = min(k, n)
    sizes = [n // k] * k
    for i in range(n % k):
        sizes[i] += 1
    folds, start = [], 0
    for s in sizes:
        stop = start + s
        test = list(range(start, stop))
        train = [r for r in range(n) if r < start or r >= stop]
        folds.append((train, test))
        start = stop
    return folds


def fit_penalised_cv(Xtr, ytr, l1_ratio):
    Z, mu, sd = standardize(Xtr)
    ybar = ytr.mean()
    yc = ytr - ybar
    amax = alpha_max(Z, yc, l1_ratio)
    if not amax > 0:
        return np.zeros(Xtr.shape[1]), 0.0
    grid = [amax * (L1_GRID_MIN_RATIO ** (i / (L1_GRID_N - 1))) for i in range(L1_GRID_N)]
    folds = kfold(Xtr.shape[0], 5)
    best_alpha, best_mse = grid[0], np.inf
    for a in grid:
        sse, cnt = 0.0, 0
        for tr, te in folds:
            Xa = Xtr[tr]
            ya = ytr[tr]
            Za, mua, sda = standardize(Xa)
            mA = ya.mean()
            w = coord_descent(Za, ya - mA, a, l1_ratio, max_iter=500, tol=1e-6)
            for r in te:
                yhat = mA + float(w @ ((Xtr[r] - mua) / sda))
                sse += (ytr[r] - yhat) ** 2
                cnt += 1
        mse = sse / cnt
        if mse < best_mse - 1e-12:
            best_mse, best_alpha = mse, a
    w = coord_descent(Z, yc, best_alpha, l1_ratio, max_iter=5000, tol=1e-8)
    return w, best_alpha


class Cache:
    def __init__(self, Xtr, ytr, Xte, yte):
        self.Xte, self.yte = Xte, yte
        self.tr = self._acc(Xtr, ytr)
        self.te = self._acc(Xte, yte)

    @staticmethod
    def _acc(X, y):
        n = X.shape[0]
        A = np.column_stack([np.ones(n), X])
        G = A.T @ A
        b = A.T @ y
        ybar = y.mean()
        return dict(G=G, b=b, yty=float(y @ y), n=n, ybar=ybar,
                    tss=float(y @ y) - n * ybar ** 2)


def fit_fast(cache, feat):
    cols = [0] + [i + 1 for i in feat]
    G = cache.tr['G'][np.ix_(cols, cols)]
    b = cache.tr['b'][cols]
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return None
    beta = Ginv @ b
    rss_tr = cache.tr['yty'] - beta @ b
    Gte = cache.te['G'][np.ix_(cols, cols)]
    bte = cache.te['b'][cols]
    rss_te = cache.te['yty'] - 2 * beta @ bte + beta @ Gte @ beta
    return dict(beta=beta, Ginv=Ginv, rss_train=rss_tr,
                r2_train=1 - rss_tr / cache.tr['tss'],
                rmse=np.sqrt(max(rss_te, 0) / cache.te['n']),
                r2_test=1 - rss_te / cache.te['tss'])


def vif_from_gram(acc, feat, j):
    if len(feat) < 2:
        return 1.0
    others = [f for f in feat if f != j]
    cols = [0] + [i + 1 for i in others]
    jc = j + 1
    G = acc['G'][np.ix_(cols, cols)]
    b = acc['G'][np.ix_(cols, [jc])].ravel()
    beta = np.linalg.inv(G) @ b
    n = acc['n']
    jmean = acc['G'][0][jc] / n
    tss = acc['G'][jc][jc] - n * jmean ** 2
    rss = acc['G'][jc][jc] - beta @ b
    r2 = 1 - rss / tss
    return float('inf') if r2 >= 1 - 1e-12 else 1 / (1 - r2)


def fit_full(cache, feat, names, test_ids):
    base = fit_fast(cache, feat)
    beta = base['beta']
    p = len(feat)
    n_tr = cache.tr['n']
    dof = n_tr - p - 1
    sigma2 = base['rss_train'] / dof
    tcrit = stats.t.ppf(0.975, dof)
    coefs = []
    for i in range(len(beta)):
        se = np.sqrt(max(base['Ginv'][i][i] * sigma2, 0))
        t = beta[i] / se if se > 0 else np.nan
        coefs.append(dict(
            feature='Intercept (β₀)' if i == 0 else names[feat[i - 1]],
            coef=float(beta[i]), stderr=float(se), tstat=float(t),
            pvalue=float(2 * stats.t.sf(abs(t), dof)),
            ci_lower=float(beta[i] - tcrit * se), ci_upper=float(beta[i] + tcrit * se),
            vif=None if i == 0 else vif_from_gram(cache.tr, feat, feat[i - 1])))
    r2_tr = base['r2_train']
    f_stat = (r2_tr / p) / ((1 - r2_tr) / dof)
    f_p = float(stats.f.sf(f_stat, p, dof))

    Xte, yte = cache.Xte, cache.yte
    pred = beta[0] + Xte[:, feat] @ beta[1:]
    resid = yte - pred
    n_te = len(yte)
    mae = float(np.abs(resid).mean())
    r2_te = base['r2_test']
    adj = 1 - (1 - r2_te) * (n_te - 1) / (n_te - p - 1) if n_te - p - 1 > 0 else np.nan
    order = np.argsort(-np.abs(resid), kind='stable')
    outliers = [dict(rank=q + 1, index=int(test_ids[r]), actual=float(yte[r]),
                     predicted=float(pred[r]), residual=float(resid[r]),
                     abs_error=float(abs(resid[r])),
                     pct_error=float(abs(resid[r] / yte[r]) * 100))
                for q, r in enumerate(order[:min(10, n_te)])]
    return dict(k=p, features=[names[i] for i in feat], rmse=float(base['rmse']),
                mae=mae, r2=float(r2_te), adj_r2=float(adj), r2_train=float(r2_tr),
                f_stat=float(f_stat), f_pvalue=f_p,
                coefficients=coefs, outliers=outliers)


def ranking_sets(coef, tie, p):
    order = sorted(range(p), key=lambda i: (-abs(coef[i]), -abs(tie[i]), i))
    # mirror JS comparator: primary |coef| desc with 1e-12 tolerance, then |tie| desc
    import functools

    def cmp(a, b):
        d = abs(coef[b]) - abs(coef[a])
        if abs(d) > 1e-12:
            return -1 if d < 0 else 1
        d2 = abs(tie[b]) - abs(tie[a])
        return -1 if d2 < 0 else (1 if d2 > 0 else 0)

    order = sorted(range(p), key=functools.cmp_to_key(cmp))
    return [sorted(order[:k]) for k in range(1, p + 1)]


def forward_sfs(Xtr, ytr, p):
    folds = kfold(Xtr.shape[0], 5)
    caches = [Cache(Xtr[tr], ytr[tr], Xtr[te], ytr[te]) for tr, te in folds]
    chosen, remaining, sets = [], list(range(p)), []
    while remaining:
        best_feat, best_score = remaining[0], -np.inf
        for cand in remaining:
            trial = sorted(chosen + [cand])
            tot, ok = 0.0, True
            for c in caches:
                f = fit_fast(c, trial)
                if f is None:
                    ok = False
                    break
                tot += f['r2_test']
            score = tot / len(caches) if ok else -np.inf
            if score > best_score:
                best_score, best_feat = score, cand
        chosen.append(best_feat)
        remaining.remove(best_feat)
        sets.append(sorted(chosen))
    return sets


def rfe(Xtr, ytr, p):
    Z, _, _ = standardize(Xtr)
    cache = Cache(Z, ytr, Z, ytr)
    alive = list(range(p))
    sets = [None] * p
    sets[p - 1] = alive[:]
    while len(alive) > 1:
        f = fit_fast(cache, alive)
        drop_at = int(np.argmin([abs(f['beta'][i + 1]) for i in range(len(alive))]))
        alive = alive[:drop_at] + alive[drop_at + 1:]
        sets[len(alive) - 1] = alive[:]
    return sets


def exhaustive(cache, p):
    sets = [None] * p
    for k in range(1, p + 1):
        best, best_rmse = None, np.inf
        for combo in itertools.combinations(range(p), k):
            f = fit_fast(cache, list(combo))
            if f and f['rmse'] < best_rmse:
                best_rmse, best = f['rmse'], list(combo)
        sets[k - 1] = best
    return sets


def analyse(df, target, test_size=0.2, seed=0):
    y = df[target].to_numpy(dtype=float)
    feats = df.drop(columns=[target])
    X = pd.get_dummies(feats, drop_first=True).astype(float)
    names = X.columns.tolist()
    Xv = X.to_numpy(dtype=float)
    n, p = Xv.shape
    idx = np.arange(n)
    itr, ite = train_test_split(idx, test_size=test_size, random_state=seed)
    Xtr, ytr, Xte, yte = Xv[itr], y[itr], Xv[ite], y[ite]
    cache = Cache(Xtr, ytr, Xte, yte)

    Zt, _, _ = standardize(Xtr)
    A = np.column_stack([np.ones(len(Zt)), Zt])
    std_coef = (np.linalg.inv(A.T @ A) @ A.T @ ytr)[1:]

    las, a_las = fit_penalised_cv(Xtr, ytr, 1.0)
    enet, a_en = fit_penalised_cv(Xtr, ytr, 0.5)

    sets = {
        'Lasso': ranking_sets(las, std_coef, p),
        'ElasticNet': ranking_sets(enet, std_coef, p),
        'Forward SFS': forward_sfs(Xtr, ytr, p),
        'RFE': rfe(Xtr, ytr, p),
        'Exhaustive': exhaustive(cache, p),
    }
    models = {a: [fit_full(cache, sets[a][k - 1], names, ite) for k in range(1, p + 1)]
              for a in sets}
    return dict(featureNames=names, split=dict(nTrain=len(itr), nTest=len(ite),
                testIdx=ite.tolist(), trainIdx=itr.tolist()),
                tuning=dict(lassoAlpha=float(a_las), enetAlpha=float(a_en)),
                models=models,
                permutation={'50_0': None})


if __name__ == '__main__':
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    docs = os.path.join(here, os.pardir, 'docs')
    data = os.path.join(docs, 'assets', 'data')

    out = {}
    df1 = pd.read_csv(os.path.join(data, '50_Startups.csv'))
    out['startups'] = analyse(df1, 'Profit')
    df2 = pd.read_csv(os.path.join(data, 'BostonHousing.csv'))
    df2.columns = [c.strip().strip('"') for c in df2.columns]
    out['boston'] = analyse(df2, 'medv')

    with open(os.path.join(docs, 'ref2_out.json'), 'w') as fh:
        json.dump(out, fh)

    perms = {}
    for n in (10, 50, 100, 506):
        for seed in (0, 42):
            perms['%d_%d' % (n, seed)] = np.random.RandomState(seed).permutation(n).tolist()
    with open(os.path.join(docs, 'perm_truth.json'), 'w') as fh:
        json.dump(perms, fh)

    print('wrote docs/ref2_out.json and docs/perm_truth.json')
    print()
    for name in out:
        r = out[name]
        print('===', name, '| features:', r['featureNames'])
        print('    n_train/n_test', r['split']['nTrain'], r['split']['nTest'],
              '| alpha lasso %.4f enet %.4f' % (r['tuning']['lassoAlpha'],
                                                r['tuning']['enetAlpha']))
        for a in ('Lasso', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive'):
            row = [(m['k'], round(m['rmse'], 4)) for m in r['models'][a][:6]]
            print('   ', a.ljust(12), row)
