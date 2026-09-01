#!/usr/bin/env python3
"""Exp 7: the CALIBRATED adaptive cheater -- the real adversary.

exp6 showed a noise-shaping attack is caught, but only by a 0.04%
systematic shortfall in the tail reading, and only because the padding was
scaled to the theoretical target sigma*sqrt(dof). That is a strawman: a
real adversary tunes the padding empirically.

This adversary is strictly stronger. For each submission it:

  1. computes what an honest version of *its own* adapter would read,
     target = sigma_hat^2(DeltaW + N_honest)   -- it can do this exactly,
     since DeltaW is its own and the contract sigma is public;
  2. injects only c*sigma inside the top-k signal subspace (the real
     privacy loss);
  3. solves for the padding scale alpha, in the orthogonal complement, that
     makes its tail reading equal `target` -- not a formula, the actual
     number an honest client of the same adapter would have produced.

The statistic is near-quadratic in alpha (padding is orthogonal to the
top-k, so it contributes only to the tail), giving
    stat(alpha) ~ (C + alpha^2) / dof,
solved in closed form from stat(0) plus one Newton correction.

If this adversary evades a correctly calibrated auditor, level-only
auditing is not sufficient and the defense must inspect noise geometry.

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


def fast_sigma2(W, rank):
    k = kk(rank)
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    return (float(np.sum(W * W)) - float(np.sum(s ** 2))) / tail_energy_dof(D, D, k)


def topk_basis(W, k):
    U, _, Vt = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=0)
    return U, Vt.T


def project_out(E, U, V):
    E = E - U @ (U.T @ E)
    return E - (E @ V) @ V.T


def honest(W, sigma, rng):
    return W + rng.normal(0.0, sigma, size=W.shape)


def shaped(W, rank, sigma, rng, calibrate):
    """Noise-shaping submission. calibrate=False reproduces exp6's strawman."""
    k = kk(rank)
    dof = tail_energy_dof(D, D, k)
    U, V = topk_basis(W, k)

    # weak noise confined to the top-k signal subspace = the real privacy loss
    G = rng.normal(0.0, CHEAT * sigma, size=W.shape)
    G_top = G - project_out(G, U, V)
    base = W + G_top

    # unit-norm padding living in the orthogonal complement
    E = project_out(rng.normal(0.0, 1.0, size=W.shape), U, V)
    E /= (np.linalg.norm(E) + 1e-30)

    if not calibrate:
        alpha = sigma * np.sqrt(dof)          # exp6: theoretical target
        Ws = base + alpha * E
        return Ws, fast_sigma2(Ws, rank)

    # what an honest version of THIS adapter would have read
    target = fast_sigma2(honest(W, sigma, rng), rank)

    c0 = fast_sigma2(base, rank) * dof                       # stat(0)*dof
    a1 = np.sqrt(max(target * dof - c0, 0.0))
    s1 = fast_sigma2(base + a1 * E, rank)
    c1 = s1 * dof - a1 ** 2                                  # Newton correction
    a2 = np.sqrt(max(target * dof - c1, 0.0))
    Ws = base + a2 * E
    return Ws, fast_sigma2(Ws, rank)


def signal_noise_std(W_sent, W, rank):
    k = kk(rank)
    U, V = topk_basis(W, k)
    N = W_sent - W
    N_top = N - project_out(N, U, V)
    return float(np.linalg.norm(N_top) / np.sqrt(k * (2 * D - k)))


def pooled(pool, rank, mode, rng):
    vals = []
    for _ in range(POOL):
        A, B = pool[rng.integers(len(pool))]
        W = B @ A
        if mode == "honest":
            vals.append(fast_sigma2(honest(W, SIGMA, rng), rank))
        else:
            vals.append(shaped(W, rank, SIGMA, rng, mode == "calibrated")[1])
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 88)
    print(f"Exp 7: calibrated noise-shaping adversary  "
          f"(d={D}, sigma={SIGMA}, cheat={CHEAT}, pool={POOL}, alpha={ALPHA})")
    print("=" * 88)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng)
        null = np.array([pooled(pool, rank, "honest", rng) for _ in range(N_NULL)])
        thr = float(np.quantile(null, ALPHA))
        row = {"rank": rank, "threshold": thr}

        for mode in ("honest", "uncalibrated", "calibrated"):
            st = np.array([pooled(pool, rank, mode, rng) for _ in range(N_TEST)])
            row[mode] = {"flag_rate": float(np.mean(st < thr)),
                         "sigma_hat": float(np.sqrt(np.mean(st)))}

        # real privacy delivered, per strategy
        for mode in ("honest", "calibrated"):
            ps = []
            for _ in range(10):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                Ws = (honest(W, SIGMA, rng) if mode == "honest"
                      else shaped(W, rank, SIGMA, rng, True)[0])
                ps.append(signal_noise_std(Ws, W, rank))
            row[mode]["signal_noise_std"] = float(np.mean(ps))

        out.append(row)
        print(f"\nrank {rank}   threshold={thr:.4e}   [{time.time()-t0:.0f}s]")
        print(f"  {'strategy':<14} {'flagged':>8} {'sigma_hat':>13} {'noise on signal':>17}")
        for mode in ("honest", "uncalibrated", "calibrated"):
            r = row[mode]
            sn = f"{r['signal_noise_std']:.3e}" if "signal_noise_std" in r else "-"
            print(f"  {mode:<14} {r['flag_rate']:>8.2f} {r['sigma_hat']:>13.4e} {sn:>17}")

    with open("results/exp7_calibrated_cheater.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "pool": POOL,
                              "alpha": ALPHA, "n_null": N_NULL, "n_test": N_TEST},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp7_calibrated_cheater.json")


if __name__ == "__main__":
    main()
