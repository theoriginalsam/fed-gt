#!/usr/bin/env python3
"""Exp 6: the ADAPTIVE cheater -- can a client shape its noise to fool the
tail test, and what does the evasion cost it?

Threat model
------------
The auditor's statistic is the tail energy of the submitted update:

    sigma_hat^2 = (||W||_F^2 - sum_{i<=k} s_i^2) / dof,     k = r + margin

A naive cheater scales *all* its noise down by c < 1, which lowers the tail
energy and is caught (exp5b: power 1.00). An adaptive cheater knows the
auditor only reads the tail, so it:

  1. injects only c * sigma into the top-k signal subspace, where the
     private information actually concentrates (this is the real privacy
     loss), and
  2. injects compensating noise *orthogonal* to that subspace, scaled so the
     tail energy matches exactly what an honest client would produce.

The submitted matrix then passes the audit while offering far less real
protection on the directions that matter.

What this experiment measures
-----------------------------
  evasion  : flag rate of the adaptive cheater under a correctly calibrated
             (real-signal) threshold. Near alpha = evasion succeeds.
  privacy  : effective noise std on the top-k signal subspace, relative to
             the contracted sigma. Lower = more privacy actually lost.
  cost     : SPA aggregation truncates the update to rank r, so noise living
             outside the top-r subspace is largely discarded. We measure the
             post-truncation distortion ||trunc_r(W_sent) - DeltaW||_F /
             ||DeltaW||_F. If the adaptive cheater's distortion is NOT higher
             than an honest client's, the evasion is essentially free -- which
             would be the strongest possible warning for the mechanism.

CPU-only. Reads ./adapters/ (q_proj, square 3584x3584).
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
RANKS = [4, 8, 16, 32]
MARGIN = 2
SIGMA = 5e-5        # contracted noise std (~1x per-entry RMS of real DeltaW)
CHEAT = 0.5         # signal-subspace noise = CHEAT * SIGMA
POOL = 4
N_NULL = 40
N_TEST = 40
ALPHA = 0.05
N_ITER = 4


def load_pool(rank, n, rng):
    files = sorted(glob.glob(f"adapters/*_r{rank}__*q_proj.pt"))
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    pool = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        A = st["lora_A"].float().numpy().astype(np.float64)
        B = st["lora_B"].float().numpy().astype(np.float64)
        pool.append((A, B))
    return pool


def topk_basis(W, k):
    """Top-k left/right singular subspaces of W."""
    U, s, Vt = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=0)
    return U, Vt.T


def fast_sigma2(W, rank, margin=MARGIN):
    k = min(rank + margin, min(W.shape) - 1)
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    fro2 = float(np.sum(W * W))
    return (fro2 - float(np.sum(s ** 2))) / tail_energy_dof(W.shape[0], W.shape[1], k)


def project_out(E, U, V):
    """Remove the components of E lying in span(U) (rows) or span(V) (cols)."""
    E = E - U @ (U.T @ E)
    E = E - (E @ V) @ V.T
    return E


def make_submission(W, rank, mode, sigma, rng):
    """Build what the client actually sends, under each strategy."""
    k = min(rank + MARGIN, D - 1)
    if mode == "honest":
        return W + rng.normal(0.0, sigma, size=W.shape)
    if mode == "naive":
        return W + rng.normal(0.0, CHEAT * sigma, size=W.shape)

    # adaptive: weak noise inside the signal subspace, compensating noise
    # outside it, scaled so the tail statistic matches an honest client.
    U, V = topk_basis(W, k)
    E = rng.normal(0.0, 1.0, size=W.shape)
    E_perp = project_out(E, U, V)          # lives (almost) entirely in the tail
    dof = tail_energy_dof(D, D, k)
    target = sigma * np.sqrt(dof)          # honest expected tail energy
    E_perp *= target / (np.linalg.norm(E_perp) + 1e-30)

    G = rng.normal(0.0, CHEAT * sigma, size=W.shape)
    G_top = G - project_out(G, U, V)       # confine the weak noise to the top-k
    return W + G_top + E_perp


def signal_noise_std(W_sent, W, rank):
    """Effective noise std inside the top-k signal subspace (the privacy loss)."""
    k = min(rank + MARGIN, D - 1)
    U, V = topk_basis(W, k)
    N = W_sent - W
    N_top = N - project_out(N, U, V)
    # variance per entry over the k*(2D-k) dimensions the subspace spans
    dims = k * (2 * D - k)
    return float(np.linalg.norm(N_top) / np.sqrt(dims))


def trunc_distortion(W_sent, W, rank):
    """Distortion after SPA's rank-r truncation: what aggregation actually keeps."""
    U, s, Vt = randomized_svd(W_sent, n_components=rank, n_iter=N_ITER,
                              random_state=0)
    W_trunc = (U * s) @ Vt
    return float(np.linalg.norm(W_trunc - W) / np.linalg.norm(W))


def pooled_stat(pool, rank, mode, sigma, rng):
    vals = []
    for _ in range(POOL):
        A, B = pool[rng.integers(len(pool))]
        W = B @ A
        vals.append(fast_sigma2(make_submission(W, rank, mode, sigma, rng), rank))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    print("=" * 84)
    print(f"Exp 6: adaptive cheater vs the tail test  "
          f"(d={D}, sigma={SIGMA}, cheat={CHEAT}, pool={POOL}, alpha={ALPHA})")
    print("=" * 84)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng)

        # correctly calibrated threshold (real signal + honest noise)
        null = np.array([pooled_stat(pool, rank, "honest", SIGMA, rng)
                         for _ in range(N_NULL)])
        thr = float(np.quantile(null, ALPHA))

        row = {"rank": rank, "threshold": thr}
        for mode in ("honest", "naive", "adaptive"):
            stats = np.array([pooled_stat(pool, rank, mode, SIGMA, rng)
                              for _ in range(N_TEST)])
            row[mode] = {"flag_rate": float(np.mean(stats < thr)),
                         "sigma_hat": float(np.sqrt(np.mean(stats)))}

        # privacy loss + truncation cost, measured on single draws
        priv, cost = {}, {}
        for mode in ("honest", "naive", "adaptive"):
            ps, cs = [], []
            for _ in range(12):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                Ws = make_submission(W, rank, mode, SIGMA, rng)
                ps.append(signal_noise_std(Ws, W, rank))
                cs.append(trunc_distortion(Ws, W, rank))
            priv[mode] = float(np.mean(ps))
            cost[mode] = float(np.mean(cs))
            row[mode]["signal_noise_std"] = priv[mode]
            row[mode]["trunc_distortion"] = cost[mode]

        out.append(row)
        print(f"\nrank {rank}   threshold={thr:.4e}   [{time.time()-t0:.0f}s]")
        print(f"  {'strategy':<10} {'flagged':>8} {'sigma_hat':>12} "
              f"{'noise on signal':>16} {'post-trunc cost':>16}")
        for mode in ("honest", "naive", "adaptive"):
            r = row[mode]
            print(f"  {mode:<10} {r['flag_rate']:>8.2f} {r['sigma_hat']:>12.3e} "
                  f"{r['signal_noise_std']:>16.3e} {r['trunc_distortion']:>16.4f}")

    with open("results/exp6_adaptive_cheater.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT,
                              "pool": POOL, "alpha": ALPHA,
                              "n_null": N_NULL, "n_test": N_TEST},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp6_adaptive_cheater.json")


if __name__ == "__main__":
    main()
