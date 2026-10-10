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
normalised by the median update norm C of their (layer, rank) group, so the
clip norm is 1 and sigma follows from exact Gaussian calibration with
sensitivity 2.

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
transcript (Y, L, N), knows the attack, and trains a gradient-boosted classifier
on transcripts of half the test clients; it is scored on the other half.
Per-client decisions pool T rounds (module adapters of that client) by summing
log-odds, with the threshold at the honest clients' 95th percentile.

CPU-only. First run caches singular values (a few minutes).
"""
import argparse, glob, json, pathlib, re, sys, time
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
    for L in np.unique(layers):                     # clip norm C = median norm, from public data
        m = layers == L
        C = np.median(np.linalg.norm(S[m & pub], axis=1))
        S[m] /= C
    return S, pub, keys


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
    pooled = {lab: {k: np.array([np.random.default_rng(hash((k, lab, i)) % 2**32).choice(v, T_POOL).sum()
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
                    data[kind].append((keys[i], features(*transcript(kind, S[i], sigma, rel, rng), sigma, None)))
        out = {"eps": eps, "sigma": sigma, "h": h}
        for adv in ("shaping", "relabel"):
            out[adv] = crossfit_eval(data["honest"], data[adv], rng)
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
    global FEATS, EPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.1, help="smoothing bandwidth / mean coordinate sd (0 = none)")
    ap.add_argument("--public", choices=["other_seeds", "same_seed"], default="other_seeds")
    ap.add_argument("--feats", choices=["all", "no_sort", "noise_only"], default="all")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--eps", type=float, nargs="+", default=EPS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    FEATS, EPS = args.feats, args.eps
    rng = np.random.default_rng(26)
    meta, svals = singular_values()
    print(f"{len(svals)} adapters; h={args.h} public={args.public} feats={args.feats}")
    res = []
    for r in args.ranks:
        S, pub, keys = group(meta, svals, r)
        res.append(run_rank(r, S, pub, keys, rng, args.h, args.public))
    out = {"config": {"c": C_KEEP, "alpha": ALPHA, "T": T_POOL, "eps": EPS, "draws": DRAWS,
                      "public": args.public, "test": "seed 44", "h_factor": args.h, "feats": args.feats},
           "ranks": res}
    path = ROOT / "results" / f"exp26b_practical_relabel{args.tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
