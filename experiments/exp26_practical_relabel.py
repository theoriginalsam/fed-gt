#!/usr/bin/env python3
"""Exp 26: a practical relabelling attack on real adapter spectra.

Proposition 2 proves that a relabelling client produces a transcript with the
honest distribution, averaged over the auditor's model P of the learning. Its
map T and posterior sampler exist, but for real adapters nobody has built them.
This experiment builds them from public adapters and asks whether an auditor
that knows the attack can still tell the cheater apart.

Setting. In the client's declared basis (its top singular vectors) the core
learning is diag(s), with s the r singular values of B A; off-diagonal core
entries carry no learning and stay honestly noised. Singular values are
normalised by the median update norm of their (layer, rank) group, computed on
public adapters, and then clipped to norm 1 (honest cores and the cheater's
public pool alike), so sigma from exact Gaussian calibration with sensitivity 2
is the honest mechanism's real budget. --no-clip gives the earlier unclipped
diagnostic (exp26b), whose budgets are nominal only.

    honest     Y = s + Z,          commits (L, N) = (s, Z)
    shaping    Y = s + cZ,         commits (s, cZ)              (positive control)
    relabel    Y = T(s) + cZ,      commits (L', Y - L')

The relabeller is built only from public adapters (seeds 42 and 43):
    P_h = public sample smoothed by N(0, h^2 I)   (an atomless model of P)
    T   = optimal assignment from public s_j to draws of P_h * N(0, (1-c^2) sigma^2),
          extended to a new s by its nearest public neighbour
    L'  = exact posterior draw of the learning given Y under P_h.
Test clients come from seed 44, never seen by the relabeller.

Auditor. The strongest auditor we can build: it sees the whole diagonal
transcript (Y, L, N) (or only N), knows the attack, and trains a
gradient-boosted classifier on attack and honest transcripts. A client's
decision sums the log-odds of T = 10 of its module transcripts: 10 distinct
saved module adapters of that client, one noise draw each. These are modules
of one or two saved rounds, not ten chronological training rounds.

Evaluation (--split strict, the default). Test clients are split into five
folds. For each test fold and each of the other four folds as calibration
fold, a classifier is trained on the remaining three folds and frozen; the
threshold is the 95th percentile of the honest pooled scores of the
calibration clients, and the test clients are scored against it. No model or
threshold that decides a test client ever saw that client's data. Results
average over the four calibration choices. --split crossfit gives the earlier
exp26b procedure (leave-one-client-out thresholds from out-of-fold scores).

CPU-only. First run caches singular values (a few minutes).
"""
import argparse, glob, json, pathlib, re, sys, time, zlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from fedgt.privacy_calibration import sigma_of_eps

RANKS = [4, 8, 16, 32]
EPS = [1, 8, 80, 800, 24000]
C_KEEP = 0.5
ALPHA = 0.05
T_POOL = 10
M_OT = 1500
DRAWS = 6                   # noise draws per test adapter
CLIP = True
SPLIT = "strict"
CACHE = ROOT / "results" / "cache_exp26_singular_values.npz"
PAT = re.compile(r"campaign/(\w+?)_(flexlora|hetero_spa)_s(\d+)/adapters/round(\d+)_client(\d+)_r(\d+)__.*layers_(\d+)_self_attn_q_proj\.pt")


def singular_values():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["meta"], list(z["svals"])
    meta, svals = [], []
    files = sorted(glob.glob(str(ROOT / "campaign/*/adapters/*q_proj.pt")))
    t0 = time.time()
    for i, f in enumerate(files):
        m = PAT.search(f)
        ds, meth, seed, rnd, cid, r, layer = m.groups()
        st = torch.load(f, map_location="cpu")
        A = st["lora_A"].double().numpy(); B = st["lora_B"].double().numpy()
        _, Rb = np.linalg.qr(B); _, Ra = np.linalg.qr(A.T)
        s = np.linalg.svd(Rb @ Ra.T, compute_uv=False)
        meta.append((ds, meth, int(seed), int(rnd), int(cid), int(r), int(layer)))
        svals.append(s)
        if i % 2000 == 0:
            print(f"  {i}/{len(files)} adapters  [{time.time()-t0:.0f}s]", flush=True)
    meta = np.array(meta, dtype=object)
    np.savez(CACHE, meta=meta, svals=np.array(svals, dtype=object))
    return meta, svals


def group(meta, svals, r):
    """Normalised singular-value vectors of rank r, with client keys."""
    idx = [i for i, m in enumerate(meta) if m[5] == r]
    S = np.stack([svals[i][:r] for i in idx])
    seeds = np.array([meta[i][2] for i in idx]); layers = np.array([meta[i][6] for i in idx])
    keys = np.array([f"{meta[i][0]}_{meta[i][1]}_{meta[i][2]}_{meta[i][4]}" for i in idx])
    pub = seeds != 44
    for L in np.unique(layers):                     # scale C = median norm, from public data
        m = layers == L
        C = np.median(np.linalg.norm(S[m & pub], axis=1))
        S[m] /= C
    if CLIP:                                        # clip to norm 1: honest cores and public pool
        S *= np.minimum(1.0, 1.0 / np.linalg.norm(S, axis=1))[:, None]
    return S, pub, keys


def seed_of(*parts):
    """Stable seed (Python's str hash changes between processes)."""
    return [26] + [zlib.crc32(p.encode()) if isinstance(p, str) else int(p) for p in parts]


class Relabeller:
    def __init__(self, P, sigma, h, rng):
        self.P, self.sigma, self.h, self.rng = P, sigma, h, rng
        n = min(M_OT, len(P))
        self.src = P[rng.choice(len(P), n, replace=False)]
        tau = np.sqrt(h ** 2 + (1 - C_KEEP ** 2) * sigma ** 2)
        tgt = P[rng.choice(len(P), n)] + rng.normal(0, tau, self.src.shape)
        cost = (self.src ** 2).sum(1)[:, None] + (tgt ** 2).sum(1)[None, :] - 2 * self.src @ tgt.T
        row, col = linear_sum_assignment(cost)
        self.tgt = tgt[col[np.argsort(row)]]

    def T(self, s):
        j = np.argmin(((self.src - s) ** 2).sum(1))
        return self.tgt[j] + (s - self.src[j])

    def posterior(self, y):
        if self.h == 0:                            # exact posterior under the empirical model
            lw = -((self.P - y) ** 2).sum(1) / (2 * self.sigma ** 2)
            w = np.exp(lw - lw.max()); w /= w.sum()
            return self.P[self.rng.choice(len(self.P), p=w)].copy()
        v = self.sigma ** 2 + self.h ** 2
        lw = -((self.P - y) ** 2).sum(1) / (2 * v)
        w = np.exp(lw - lw.max()); w /= w.sum()
        j = self.rng.choice(len(self.P), p=w)
        shrink = self.h ** 2 / v
        mean = self.P[j] + shrink * (y - self.P[j])
        sd = np.sqrt(self.h ** 2 * self.sigma ** 2 / v)
        return mean + self.rng.normal(0, sd, y.shape)


def transcript(kind, s, sigma, rel, rng):
    z = rng.normal(0, sigma, s.shape)
    if kind == "honest":
        y, L = s + z, s
    elif kind == "shaping":
        y, L = s + C_KEEP * z, s
    else:
        y = rel.T(s) + C_KEEP * z
        L = rel.posterior(y)
    return y, L, y - L


FEATS = "all"


def features(y, L, N, sigma, P_sorted_ref):
    r = len(y)
    if FEATS == "noise_only":                      # what the paper's audits read: the noise part
        return np.concatenate([N / sigma, [np.sum(N ** 2) / (r * sigma ** 2)]])
    base = [y / sigma, L, N / sigma,
            [np.sum(N ** 2) / (r * sigma ** 2), np.sum(L ** 2), np.sum(y ** 2) / sigma ** 2,
             L.min(), L[0] / max(abs(L.sum()), 1e-12)]]
    if FEATS == "all":
        base.append([float(np.sum(np.diff(L) > 0))])   # real singular values are sorted
    return np.concatenate(base)


def crossfit_eval(honest, attack, rng, k_folds=5, n_pool=20, n_boot=1000):
    """Client-disjoint evaluation of an informed auditor.

    Every test client is scored by a classifier trained on other clients only
    (k-fold over clients). Its pooled-T decision uses a threshold fitted on the
    honest pooled scores of all other clients (leave-one-client-out). So the
    clients used to train the classifier, to set the threshold and to measure
    power never overlap. Uncertainty is a bootstrap over clients, the
    independent unit here (transcripts of one client share its adapters).
    """
    keys = sorted({k for k, _ in honest})
    order = rng.permutation(len(keys))
    fold = {keys[i]: j % k_folds for j, i in enumerate(order)}
    oof = {0: {}, 1: {}}
    for f in range(k_folds):
        Xtr, ytr = [], []
        for lab, rows in ((0, honest), (1, attack)):
            for k, x in rows:
                if fold[k] != f:
                    Xtr.append(x); ytr.append(lab)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
        clf.fit(np.array(Xtr), np.array(ytr))
        for lab, rows in ((0, honest), (1, attack)):
            ks = [k for k, _ in rows if fold[k] == f]
            if not ks:
                continue
            X = np.array([x for k, x in rows if fold[k] == f])
            p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            for k, v in zip(ks, np.log(p / (1 - p))):
                oof[lab].setdefault(k, []).append(v)
    pooled = {lab: {k: np.array([np.random.default_rng(seed_of(k, lab, i)).choice(v, T_POOL).sum()
                                 for i in range(n_pool)]) for k, v in oof[lab].items()} for lab in (0, 1)}
    pw, fp = {}, {}
    for k in keys:
        thr = np.quantile(np.concatenate([pooled[0][j] for j in keys if j != k]), 1 - ALPHA)
        pw[k] = float(np.mean(pooled[1][k] > thr))
        fp[k] = float(np.mean(pooled[0][k] > thr))
    per_round = [(0, v) for k in keys for v in oof[0][k]] + [(1, v) for k in keys for v in oof[1][k]]
    auc = roc_auc_score([l for l, _ in per_round], [v for _, v in per_round])
    boot_auc, boot_pw = [], []
    for _ in range(n_boot):
        ks = rng.choice(keys, len(keys))
        y = [0] * sum(len(oof[0][k]) for k in ks) + [1] * sum(len(oof[1][k]) for k in ks)
        sc = np.concatenate([oof[0][k] for k in ks] + [oof[1][k] for k in ks])
        boot_auc.append(roc_auc_score(y, sc))
        boot_pw.append(np.mean([pw[k] for k in ks]))
    return {"n_clients": len(keys), "auc": float(auc),
            "auc_ci": [float(np.quantile(boot_auc, 0.025)), float(np.quantile(boot_auc, 0.975))],
            "power": float(np.mean(list(pw.values()))),
            "power_ci": [float(np.quantile(boot_pw, 0.025)), float(np.quantile(boot_pw, 0.975))],
            "fpr": float(np.mean(list(fp.values())))}


def pool_scores(sc, aid, n_pool, seed):
    """Pooled scores of one client: T_POOL distinct module adapters, one noise draw each."""
    g = np.random.default_rng(seed)
    ids = np.unique(aid)
    by = {a: np.where(aid == a)[0] for a in ids}
    out = np.empty(n_pool)
    for i in range(n_pool):
        pick = g.choice(ids, T_POOL, replace=len(ids) < T_POOL)
        out[i] = sum(sc[g.choice(by[a])] for a in pick)
    return out


def strict_eval(honest, attack, rng, k_folds=5, n_pool=200, n_boot=1000):
    """Three-way client-disjoint evaluation: train / calibrate / test.

    For test fold f and calibration fold c != f, a classifier trained on the
    other three folds is frozen; the threshold is the honest pooled scores'
    95th percentile over calibration clients; test clients are scored with
    both. Nothing that decides a test client saw its data. Results average
    over the k-1 calibration choices; intervals bootstrap over test clients.
    """
    keys = sorted({k for k, _, _ in honest})
    order = rng.permutation(len(keys))
    fold = {keys[i]: j % k_folds for j, i in enumerate(order)}
    X = {lab: {} for lab in (0, 1)}
    for lab, rows in ((0, honest), (1, attack)):
        for k, a, x in rows:
            X[lab].setdefault(k, ([], []))
            X[lab][k][0].append(a); X[lab][k][1].append(x)
    X = {lab: {k: (np.array(a), np.array(x)) for k, (a, x) in d.items()} for lab, d in X.items()}
    pw = {k: [] for k in keys}; fp = {k: [] for k in keys}; logit = {lab: {k: [] for k in keys} for lab in (0, 1)}
    for f in range(k_folds):
        for c in range(k_folds):
            if c == f:
                continue
            tr = [k for k in keys if fold[k] not in (f, c)]
            Xtr = np.concatenate([X[lab][k][1] for lab in (0, 1) for k in tr])
            ytr = np.concatenate([np.full(len(X[lab][k][1]), lab) for lab in (0, 1) for k in tr])
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
            clf.fit(Xtr, ytr)

            def score(lab, k):
                p = np.clip(clf.predict_proba(X[lab][k][1])[:, 1], 1e-6, 1 - 1e-6)
                return np.log(p / (1 - p))
            cal = [k for k in keys if fold[k] == c]
            thr = np.quantile(np.concatenate([pool_scores(score(0, k), X[0][k][0], n_pool, seed_of(k, 0, f, c))
                                              for k in cal]), 1 - ALPHA)
            for k in (k for k in keys if fold[k] == f):
                s0, s1 = score(0, k), score(1, k)
                logit[0][k].append(s0); logit[1][k].append(s1)
                fp[k].append(np.mean(pool_scores(s0, X[0][k][0], n_pool, seed_of(k, 0, f, c, 1)) > thr))
                pw[k].append(np.mean(pool_scores(s1, X[1][k][0], n_pool, seed_of(k, 1, f, c, 1)) > thr))
    pw = {k: float(np.mean(v)) for k, v in pw.items()}; fp = {k: float(np.mean(v)) for k, v in fp.items()}
    tsc = {lab: {k: np.mean(v, axis=0) for k, v in logit[lab].items()} for lab in (0, 1)}
    def auc_of(ks):
        y = np.concatenate([np.full(len(tsc[lab][k]), lab) for lab in (0, 1) for k in ks])
        return roc_auc_score(y, np.concatenate([tsc[lab][k] for lab in (0, 1) for k in ks]))
    boot = {"auc": [], "power": [], "fpr": []}
    for _ in range(n_boot):
        ks = rng.choice(keys, len(keys))
        boot["auc"].append(auc_of(ks))
        boot["power"].append(np.mean([pw[k] for k in ks])); boot["fpr"].append(np.mean([fp[k] for k in ks]))
    ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"n_clients": len(keys), "auc": float(auc_of(keys)), "auc_ci": ci(boot["auc"]),
            "power": float(np.mean(list(pw.values()))), "power_ci": ci(boot["power"]),
            "fpr": float(np.mean(list(fp.values()))), "fpr_ci": ci(boot["fpr"])}


def run_rank(r, S, pub, keys, rng, h_factor=0.1, public="other_seeds"):
    if public == "same_seed":
        # the cheater's model comes from other clients of the test seed (it knows P well)
        tk = np.unique(keys[~pub]); rng.shuffle(tk)
        model_k = set(tk[: len(tk) // 3])
        model = np.array([k in model_k for k in keys])
        P = S[model]
        pub = pub | model                          # those clients are no longer test clients
    else:
        P = S[pub]
    sd = P.std(0).mean()
    h = h_factor * sd
    rows = []
    for eps in EPS:
        sigma = sigma_of_eps(eps, 2.0)
        rel = Relabeller(P, sigma, h, rng)
        data = {k: [] for k in ("honest", "shaping", "relabel")}
        for kind in data:
            for i in np.where(~pub)[0]:
                for _ in range(DRAWS):
                    data[kind].append((keys[i], i, features(*transcript(kind, S[i], sigma, rel, rng), sigma, None)))
        out = {"eps": eps, "sigma": sigma, "h": h}
        for adv in ("shaping", "relabel"):
            if SPLIT == "strict":
                out[adv] = strict_eval(data["honest"], data[adv], rng)
            else:
                drop = lambda rows: [(k, x) for k, _, x in rows]
                out[adv] = crossfit_eval(drop(data["honest"]), drop(data[adv]), rng)
        # how much does the relabeller move the learning, relative to the noise it withholds
        move = np.mean([np.linalg.norm(rel.T(s) - s) for s in S[~pub][:300]])
        out["T_shift_over_sigma_sqrt_r"] = float(move / (sigma * np.sqrt(r)))
        rows.append(out)
        R = out["relabel"]; Sh = out["shaping"]
        print(f"  r={r:2d} eps={eps:>8}  clients={R['n_clients']}  shaping power {Sh['power']:.2f}  |  relabel AUC "
              f"{R['auc']:.3f} [{R['auc_ci'][0]:.2f},{R['auc_ci'][1]:.2f}]  power {R['power']:.2f} "
              f"[{R['power_ci'][0]:.2f},{R['power_ci'][1]:.2f}]  FPR {R['fpr']:.3f}", flush=True)
    runs = sorted({k.rsplit("_", 1)[0] for k in keys[~pub]})
    return {"rank": r, "n_public": int(pub.sum()), "n_test_adapters": int((~pub).sum()),
            "n_test_runs": len(runs), "rows": rows}


def main():
    global FEATS, EPS, CLIP, SPLIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.1, help="smoothing bandwidth / mean coordinate sd (0 = none)")
    ap.add_argument("--public", choices=["other_seeds", "same_seed"], default="other_seeds")
    ap.add_argument("--feats", choices=["all", "no_sort", "noise_only"], default="all")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--eps", type=float, nargs="+", default=EPS)
    ap.add_argument("--no-clip", action="store_true", help="skip the norm-1 clip (exp26b diagnostic)")
    ap.add_argument("--split", choices=["strict", "crossfit"], default="strict")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    FEATS, EPS, CLIP, SPLIT = args.feats, args.eps, not args.no_clip, args.split
    rng = np.random.default_rng(26)
    meta, svals = singular_values()
    print(f"{len(svals)} adapters; h={args.h} public={args.public} feats={args.feats} clip={CLIP} split={SPLIT}")
    res = []
    for r in args.ranks:
        S, pub, keys = group(meta, svals, r)
        res.append(run_rank(r, S, pub, keys, rng, args.h, args.public))
    out = {"config": {"c": C_KEEP, "alpha": ALPHA, "T": T_POOL, "eps": EPS, "draws": DRAWS,
                      "public": args.public, "test": "seed 44", "h_factor": args.h, "feats": args.feats,
                      "clip": CLIP, "split": SPLIT,
                      "pooling": "T distinct module adapters of a client, one noise draw each"
                                 if SPLIT == "strict" else "T module/noise transcripts sampled with replacement"},
           "ranks": res}
    stem = "exp26c" if (CLIP and SPLIT == "strict") else "exp26b"
    path = ROOT / "results" / f"{stem}_practical_relabel{args.tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
