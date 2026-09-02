#!/usr/bin/env python3
"""Exp 8: a geometric defense -- spike-consistency auditing.

exp7 showed a calibrated noise-shaping adversary matches the tail statistic
exactly while halving the real privacy. The tail measures how *much* noise
there is; it cannot see *where* it lives. This experiment tests a second
statistic that can.

Theory
------
For DeltaW of rank r plus iid N(0, sigma^2) noise on a d x d matrix, the
spiked model gives the observed i-th singular value

    s_i  =  theta_i + d * sigma^2 / theta_i

By AM-GM this is minimised at theta_i = sigma*sqrt(d), so for ANY honest
client and any signal strength,

    s_i  >=  2 * sigma * sqrt(d)                            (*)

That is a hard floor: an honest submission cannot place a signal spike
below twice the noise scale. An adversary that injects only c*sigma inside
the signal subspace produces

    s_i  =  theta_i + d * (c*sigma)^2 / theta_i

whose minimum is 2*c*sigma*sqrt(d) < 2*sigma*sqrt(d). So when its weakest
signal direction sits near the noise floor, it *violates* (*) relative to
the sigma its own tail is advertising.

Statistic
---------
    T  =  s_r / ( sigma_hat_tail * (sqrt(d1) + sqrt(d2)) )

with sigma_hat_tail read from the tail (the value the adversary controls).
Honest: T >= 1 up to fluctuation. Shaping adversary: T < 1 when theta_r is
near the floor. Flag when T falls below an honest-calibrated quantile.

The adversary here is strictly stronger than exp7's: it protects exactly
the r true signal directions (not r + margin), so no defense can exploit
padding placed in non-signal directions.

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
SIGMA = 5e-5
CHEAT = 0.5
POOL = 4
N_NULL = 40
N_TEST = 40
ALPHA = 0.05
N_ITER = 4
EDGE = np.sqrt(D) + np.sqrt(D)      # sqrt(d1)+sqrt(d2)


def load_pool(rank, n, rng):
    files = sorted(glob.glob(f"adapters/*_r{rank}__*q_proj.pt"))
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out


def kk(rank):
    return min(rank + MARGIN, D - 1)


def spectrum(W, rank):
    """Return (top-k singular values, sigma_hat^2 from the tail)."""
    k = kk(rank)
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    s2 = (float(np.sum(W * W)) - float(np.sum(s ** 2))) / tail_energy_dof(D, D, k)
    return s, s2


def both_stats(W, rank):
    """(tail sigma_hat^2, spike-consistency ratio T)."""
    s, s2 = spectrum(W, rank)
    sigma_hat = np.sqrt(max(s2, 1e-300))
    return s2, float(s[rank - 1] / (sigma_hat * EDGE))


def topr_basis(W, r):
    U, _, Vt = randomized_svd(W, n_components=r, n_iter=N_ITER, random_state=0)
    return U, Vt.T


def project_out(E, U, V):
    E = E - U @ (U.T @ E)
    return E - (E @ V) @ V.T


def honest(W, sigma, rng):
    return W + rng.normal(0.0, sigma, size=W.shape)


def shaped(W, rank, sigma, rng):
    """Calibrated adversary protecting exactly the r true signal directions."""
    k = kk(rank)
    dof = tail_energy_dof(D, D, k)
    U, V = topr_basis(W, rank)            # exactly the signal subspace

    G = rng.normal(0.0, CHEAT * sigma, size=W.shape)
    base = W + (G - project_out(G, U, V))

    E = project_out(rng.normal(0.0, 1.0, size=W.shape), U, V)
    E /= (np.linalg.norm(E) + 1e-30)

    target = both_stats(honest(W, sigma, rng), rank)[0]
    c0 = both_stats(base, rank)[0] * dof
    a1 = np.sqrt(max(target * dof - c0, 0.0))
    c1 = both_stats(base + a1 * E, rank)[0] * dof - a1 ** 2
    a2 = np.sqrt(max(target * dof - c1, 0.0))
    return base + a2 * E


def pooled(pool, rank, mode, rng):
    tails, spikes = [], []
    for _ in range(POOL):
        A, B = pool[rng.integers(len(pool))]
        W = B @ A
        Ws = honest(W, SIGMA, rng) if mode == "honest" else shaped(W, rank, SIGMA, rng)
        t, sp = both_stats(Ws, rank)
        tails.append(t); spikes.append(sp)
    return float(np.mean(tails)), float(np.mean(spikes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 92)
    print(f"Exp 8: spike-consistency defense vs the calibrated shaping adversary "
          f"(sigma={SIGMA}, cheat={CHEAT})")
    print("=" * 92)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng)

        null = [pooled(pool, rank, "honest", rng) for _ in range(N_NULL)]
        thr_tail = float(np.quantile([x[0] for x in null], ALPHA))
        thr_spike = float(np.quantile([x[1] for x in null], ALPHA))

        row = {"rank": rank, "thr_tail": thr_tail, "thr_spike": thr_spike}
        for mode in ("honest", "shaped"):
            st = [pooled(pool, rank, mode, rng) for _ in range(N_TEST)]
            tails = np.array([x[0] for x in st]); spikes = np.array([x[1] for x in st])
            row[mode] = {
                "tail_flag": float(np.mean(tails < thr_tail)),
                "spike_flag": float(np.mean(spikes < thr_spike)),
                "T_mean": float(np.mean(spikes)),
                "sigma_hat": float(np.sqrt(np.mean(tails))),
            }
        out.append(row)

        print(f"\nrank {rank}   [{time.time()-t0:.0f}s]   spike threshold T<{thr_spike:.4f}")
        print(f"  {'strategy':<10} {'T (mean)':>10} {'sigma_hat':>13} "
              f"{'TAIL flags':>12} {'SPIKE flags':>13}")
        for mode in ("honest", "shaped"):
            r = row[mode]
            print(f"  {mode:<10} {r['T_mean']:>10.4f} {r['sigma_hat']:>13.4e} "
                  f"{r['tail_flag']:>12.2f} {r['spike_flag']:>13.2f}")

    with open("results/exp8_spike_consistency.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "pool": POOL,
                              "alpha": ALPHA, "n_null": N_NULL, "n_test": N_TEST},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp8_spike_consistency.json")


if __name__ == "__main__":
    main()
