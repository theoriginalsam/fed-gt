#!/usr/bin/env python3
"""Exp 5, part B: the HONEST kill test, on real SPA adapters.

For each rank, using genuine trained adapters as the signal:
  1. Estimator recovery: add honest Gaussian noise sigma to a real Delta_W,
     estimate sigma_hat from the tail, report recovery.
  2. Detection power: can we catch a client injecting 0.5*sigma (50% under)?
  3. False alarm: how often is an honest client flagged?
under two calibrations of the null threshold:
  - matched: null simulated with REAL adapter signal + honest noise,
  - flat:    null simulated with synthetic flat rank-r signal (scaled to the
             same Frobenius energy) + honest noise -- what the package does
             if it assumes idealized adapters.

The estimator is the exact tail statistic sigma_hat^2 =
(||W||_F^2 - sum of top-(r+margin) squared singular values) / dof, computed
with a fast randomized top-k SVD (validated identical to the full-SVD path).

CPU-only. Reads ./adapters/. A few minutes per rank; run in background.
"""
import argparse
import glob
import json
import time

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from fedgt.spectral_audit import tail_energy_dof

D = 3584
RANKS = [4, 8, 16, 32]
MARGIN = 2
SIGMA = 5e-5           # contract noise std ~ 1x per-entry RMS of real DeltaW
CHEAT = 0.5            # 50% under-noiser
POOL = 4               # adapters pooled per audit (like rounds_per_audit)
N_NULL = 60            # MC draws to calibrate the null threshold
N_TEST = 60            # MC draws to measure power / false alarm
ALPHA = 0.05
N_ITER = 4             # randomized_svd power iterations


def load_pool(rank, n, rng):
    """Preload n real (A,B) factor pairs for a rank into memory.

    Restricted to q_proj so all Delta_W are square 3584x3584 (v_proj is the
    rectangular 512x3584 GQA case, analyzed separately)."""
    files = sorted(glob.glob(f"adapters/*_r{rank}__*q_proj.pt"))
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    pool = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        A = st["lora_A"].float().numpy().astype(np.float64)
        B = st["lora_B"].float().numpy().astype(np.float64)
        pool.append((A, B))
    return pool


def fast_sigma2(W, rank, margin=MARGIN):
    """Exact tail statistic via top-k randomized SVD."""
    k = min(rank + margin, min(W.shape) - 1)
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    fro2 = float(np.sum(W * W))
    return (fro2 - float(np.sum(s ** 2))) / tail_energy_dof(W.shape[0], W.shape[1], k)


def flat_signal(fro, rank, rng):
    """Synthetic flat rank-r Delta_W with Frobenius norm `fro`."""
    B = rng.standard_normal((D, rank))
    A = rng.standard_normal((rank, D))
    W = B @ A
    return W * (fro / (np.linalg.norm(W) + 1e-30))


def pooled(signal_fn, sigma, rank, rng, pool=POOL):
    """Average sigma_hat^2 over `pool` fresh (signal + noise) draws."""
    vals = []
    for _ in range(pool):
        W = signal_fn(rng)
        W = W + rng.normal(0.0, sigma, size=W.shape)
        vals.append(fast_sigma2(W, rank))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    print("=" * 82)
    print(f"Exp 5B: honest kill test on REAL adapters  "
          f"(d={D}, sigma={SIGMA}, cheat={CHEAT}, pool={POOL}, alpha={ALPHA})")
    print("=" * 82)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 80, rng)
        fros = [np.linalg.norm(B @ A) for A, B in pool]
        fro_mean = float(np.mean(fros))

        def real_sig(rng, _pool=pool):
            A, B = _pool[rng.integers(len(_pool))]
            return B @ A

        def flat_sig(rng, _fro=fro_mean, _r=rank):
            return flat_signal(_fro, _r, rng)

        # --- estimator recovery on honest real signal (single draws) ---
        rec = []
        for _ in range(20):
            A, B = pool[rng.integers(len(pool))]
            W = B @ A + rng.normal(0.0, SIGMA, size=(D, D))
            rec.append(np.sqrt(fast_sigma2(W, rank)))
        sigma_hat = float(np.mean(rec))

        # --- calibrate thresholds ---
        null_matched = np.array([pooled(real_sig, SIGMA, rank, rng)
                                 for _ in range(N_NULL)])
        null_flat = np.array([pooled(flat_sig, SIGMA, rank, rng)
                              for _ in range(N_NULL)])
        thr_matched = float(np.quantile(null_matched, ALPHA))
        thr_flat = float(np.quantile(null_flat, ALPHA))

        # --- power (cheater 0.5 sigma) and false alarm (honest), REAL signal ---
        cheat_stats = np.array([pooled(real_sig, CHEAT * SIGMA, rank, rng)
                                for _ in range(N_TEST)])
        honest_stats = np.array([pooled(real_sig, SIGMA, rank, rng)
                                 for _ in range(N_TEST)])
        power_m = float(np.mean(cheat_stats < thr_matched))
        fa_m = float(np.mean(honest_stats < thr_matched))
        power_f = float(np.mean(cheat_stats < thr_flat))
        fa_f = float(np.mean(honest_stats < thr_flat))

        row = {"rank": rank, "sigma_hat": sigma_hat, "sigma_true": SIGMA,
               "recovery_err": abs(sigma_hat - SIGMA) / SIGMA,
               "matched": {"power": power_m, "false_alarm": fa_m},
               "flat": {"power": power_f, "false_alarm": fa_f}}
        out.append(row)
        print(f"\nrank {rank}  [{time.time()-t0:.0f}s]")
        print(f"  sigma_hat={sigma_hat:.3e} vs sigma={SIGMA:.1e}  "
              f"(recovery err {row['recovery_err']*100:.1f}%)")
        print(f"  MATCHED calib: detection power={power_m:.2f}  "
              f"false alarm={fa_m:.2f}")
        print(f"  FLAT   calib: detection power={power_f:.2f}  "
              f"false alarm={fa_f:.2f}   <- honest clients wrongly flagged if high")

    with open("results/exp5_real_kill_test.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT,
                              "pool": POOL, "alpha": ALPHA,
                              "n_null": N_NULL, "n_test": N_TEST},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp5_real_kill_test.json")


if __name__ == "__main__":
    main()
