#!/usr/bin/env python3
"""Exp 22: does the design survive realistic privacy noise?

Every experiment so far used sigma = 5e-5 per entry, about the size of a
typical entry of a real update. That makes the detection problem hard and
keeps the learning visible, but it is not a privacy level. With the Gaussian
mechanism, sensitivity equal to a typical module's update norm and
delta = 1e-5, it corresponds to epsilon of roughly 23,000 for one module in
one round. Meaningful privacy (epsilon 1 to 8) needs thousands of times more
noise per entry.

This sweeps the noise from our setting up to that regime and re-measures
every check that could depend on the noise size:

  tail estimator     recovery of sigma, and power against a client that
                     scales all its noise by 0.5
  declaration check  energy ratio of the declared subspace, against a client
                     that starves its true core and declares a random
                     subspace, or one tilted 15 degrees from the truth
  rank check         s_{r+1} against the noise floor, against a loud decoy
                     (planted at s_1) and a weak one (planted at s_r)

The challenge itself is not re-run: its statistic reads U^T N V, which is
exactly r x r i.i.d. Gaussian at any sigma, and is normalised by sigma^2, so
its power does not depend on the noise size (shown in exp19).

Predictions recorded before the run: the tail estimator gets more accurate
as noise grows (signal leakage into the tail becomes negligible); the
declaration check and the rank check both need the learning to stand above
the noise floor, and should fail once the visibility of the learning drops
well below 1.

Uses FlexLoRA adapters (rounds 6 and 12), which are full rank. SPA adapters
from those rounds carry dead zero-padded components and would bias the
visibility numbers.

CPU-only.
"""
import argparse
import glob
import json
import time

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from fedgt.spectral_audit import tail_energy_dof

D = 3584
EDGE = 2.0 * np.sqrt(D)
SIG0 = 5e-5
MULTS = [1, 10, 100, 1000, 3000, 10000]
RANKS = [4, 8, 16, 32]
CHEAT = 0.5
ALPHA = 0.05
DELTA = 1e-5
KGAUSS = float(np.sqrt(2 * np.log(1.25 / DELTA)))
N_ITER = 4
N_CAL, N_TEST, N_ADV = 120, 80, 80
ADVS = ["naive", "random_decl", "tilt_15", "decoy_loud", "decoy_weak"]
ADVS_DEFAULT = ["naive", "random_decl", "tilt_15", "decoy_loud"]


def load_pool(rank, n, rng, pattern):
    files = sorted(glob.glob(pattern.format(r=rank)))
    if not files:
        raise SystemExit(f"no adapters for rank {rank}")
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out


def basis(A, B):
    Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T)
    u, s, vt = np.linalg.svd(Rb @ Ra.T)
    return Qb @ u, Qa @ vt.T, s


def orth_dirs(U, k, rng):
    X = rng.normal(size=(U.shape[0], k)); X -= U @ (U.T @ X)
    return np.linalg.qr(X)[0][:, :k]


def tilt(U, deg, rng):
    P = orth_dirs(U, U.shape[1], rng); th = np.deg2rad(deg)
    return np.linalg.qr(U * np.cos(th) + P * np.sin(th))[0]


def starve(N, U, V):
    C = U.T @ N @ V
    return N + U @ ((CHEAT - 1.0) * C) @ V.T


def measure(W, rank, Ud, Vd, sigma):
    """Every statistic from one submitted matrix, using one shared SVD."""
    k = rank + 2
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    fro2 = float(np.sum(W * W))
    sig2 = (fro2 - float(np.sum(s ** 2))) / tail_energy_dof(D, D, k)
    top2 = float(np.sum(s[:rank] ** 2))
    ratio = float(np.linalg.norm(Ud.T @ W @ Vd) ** 2) / max(top2, 1e-300)
    excess = float(s[rank] / (sigma * EDGE))
    return sig2, ratio, excess


def trial(pool, rank, adv, sigma, rng):
    A, B = pool[rng.integers(len(pool))]
    Ut, Vt_, st = basis(A, B)
    W0 = B @ A
    N = rng.normal(0.0, sigma, size=W0.shape)
    if adv == "honest":
        return measure(W0 + N, rank, Ut, Vt_, sigma)
    if adv == "naive":
        return measure(W0 + CHEAT * N, rank, Ut, Vt_, sigma)
    if adv == "random_decl":
        W = W0 + starve(N, Ut, Vt_)
        return measure(W, rank, orth_dirs(Ut, rank, rng), orth_dirs(Vt_, rank, rng), sigma)
    if adv == "tilt_15":
        W = W0 + starve(N, Ut, Vt_)
        return measure(W, rank, tilt(Ut, 15, rng), tilt(Vt_, 15, rng), sigma)
    strength = st[0] if adv == "decoy_loud" else st[rank - 1]
    ud, vd = orth_dirs(Ut, 1, rng), orth_dirs(Vt_, 1, rng)
    W = W0 + strength * (ud @ vd.T) + N
    Ud = np.concatenate([Ut[:, :rank - 1], ud], 1); Vd = np.concatenate([Vt_[:, :rank - 1], vd], 1)
    return measure(W, rank, Ud, Vd, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--mults", type=float, nargs="+", default=MULTS)
    ap.add_argument("--pattern", default="campaign/*_flexlora_*/adapters/round0[01][62]_*_r{r}__*q_proj.pt")
    ap.add_argument("--cal", type=int, default=N_CAL)
    ap.add_argument("--test", type=int, default=N_TEST)
    ap.add_argument("--adv", type=int, default=N_ADV)
    ap.add_argument("--advs", nargs="+", default=ADVS_DEFAULT, choices=ADVS)
    ap.add_argument("--out", default="results/exp22_realistic_noise.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 112)
    print(f"Exp 22: the checks at realistic privacy noise  (cheat={CHEAT}, alpha={ALPHA}, "
          f"cal={args.cal}, test={args.test}, adv={args.adv})")
    print("=" * 112)
    rows = []
    for rank in args.ranks:
        pool = load_pool(rank, 80, rng, args.pattern)
        norms = [np.linalg.norm(B @ A) for A, B in pool]
        C = float(np.median(norms))
        svals = [basis(A, B)[2][rank - 1] for A, B in pool]
        print(f"\nrank {rank}: typical update norm C = {C:.3f}")
        print(f"  {'noise x':>8} {'eps/module/round':>17} {'visibility':>10} {'sigma err':>10} | "
              f"{'tail vs naive':>13} {'decl vs random':>14} {'decl vs tilt15':>14} "
              f"{'rank vs loud decoy':>18} {'rank vs weak decoy':>18} | honest FA (tail, decl, rank)")
        for m in args.mults:
            t0 = time.time()
            sigma = SIG0 * m
            eps = C * KGAUSS / sigma
            vis = float(np.median(svals) / (sigma * EDGE))
            cal = np.array([trial(pool, rank, "honest", sigma, rng) for _ in range(args.cal)])
            thr_tail = np.quantile(cal[:, 0], ALPHA)
            thr_decl = np.quantile(cal[:, 1], ALPHA)
            thr_rank = np.quantile(cal[:, 2], 1 - ALPHA)
            hon = np.array([trial(pool, rank, "honest", sigma, rng) for _ in range(args.test)])
            fa = (float(np.mean(hon[:, 0] < thr_tail)), float(np.mean(hon[:, 1] < thr_decl)),
                  float(np.mean(hon[:, 2] > thr_rank)))
            err = float(np.median(np.sqrt(hon[:, 0]) / sigma - 1.0))
            res = {}
            for adv in args.advs:
                a = np.array([trial(pool, rank, adv, sigma, rng) for _ in range(args.adv)])
                res[adv] = {"tail": float(np.mean(a[:, 0] < thr_tail)),
                            "decl": float(np.mean(a[:, 1] < thr_decl)),
                            "rank": float(np.mean(a[:, 2] > thr_rank))}
            row = {"rank": rank, "mult": m, "sigma": sigma, "eps_module_round": eps,
                   "visibility": vis, "sigma_rel_err": err, "honest_fa": fa,
                   "tail_vs_naive": res["naive"]["tail"],
                   "decl_vs_random": res["random_decl"]["decl"],
                   "decl_vs_tilt15": res["tilt_15"]["decl"],
                   "rank_vs_decoy_loud": res["decoy_loud"]["rank"],
                   "rank_vs_decoy_weak": res.get("decoy_weak", {}).get("rank"), "all": res}
            rows.append(row)
            print(f"  {m:>8g} {eps:>17,.1f} {vis:>10.3f} {100*err:>9.3f}% | "
                  f"{row['tail_vs_naive']:>13.2f} {row['decl_vs_random']:>14.2f} "
                  f"{row['decl_vs_tilt15']:>14.2f} {row['rank_vs_decoy_loud']:>18.2f} "
                  f"{(row['rank_vs_decoy_weak'] if row['rank_vs_decoy_weak'] is not None else float('nan')):>18.2f}"
                  f" | {fa[0]:.2f} {fa[1]:.2f} {fa[2]:.2f}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
            with open(args.out, "w") as f:
                json.dump({"config": {"sig0": SIG0, "cheat": CHEAT, "alpha": ALPHA,
                                      "delta": DELTA, "pattern": args.pattern}, "rows": rows}, f, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
