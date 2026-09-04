#!/usr/bin/env python3
"""Exp 15: is it rank, or is it the signal-to-noise-floor ratio?

Every failure so far has been reported as rank dependent. Flat calibration
blew up at r32 (exp5b). Spike consistency died at r32 (exp8). The estimated
subspace collapsed at r32 (exp11). But rank is probably a proxy. exp5a showed
higher-rank adapters have weaker trailing singular values relative to the
noise floor, and that ratio is the quantity the mathematics actually depends
on.

Define the visibility of the weakest signal direction as

    visibility  =  s_r  /  ( sigma * (sqrt(d1) + sqrt(d2)) )

Above 1 the weakest signal stands clear of the noise; below 1 it is buried.
At the sigma used throughout, visibility runs 5.5, 2.1, 0.89, 0.52 across
ranks 4, 8, 16, 32, and subspace alignment ran 0.9995, 0.9984, 0.9142,
0.5257. Those track each other closely, which is the hypothesis.

This sweeps sigma over a wide range at every rank. If the measurements
collapse onto a single curve when plotted against visibility rather than
rank, then rank was never the variable and four separate rank-dependent
findings become one statement about signal visibility.

It also checks the practical question underneath: the challenge uses a
DECLARED subspace so it should not care about visibility, but the energy
check that verifies the declaration does need the signal to be visible. If
that check degrades at high sigma, the declaration guarantee weakens.

CPU-only. Reads ./adapters/ (q_proj).
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

D = 3584
EDGE = 2.0 * np.sqrt(D)
RANKS = [4, 8, 16, 32]
SIGMA0 = 5e-5
MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]
CHEAT = 0.5
Q = 20
ALPHA = 0.05
N_TRIALS = 25
N_ITER = 4


def load_pool(rank, n, rng, adir):
    files = sorted(glob.glob(f"{adir}/*_r{rank}__*q_proj.pt"))
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out


def basis(W, r):
    U, s, Vt = randomized_svd(W, n_components=r, n_iter=N_ITER, random_state=0)
    return U, Vt.T, s


def orth_dirs(U, k, rng):
    X = rng.normal(size=(U.shape[0], k))
    X -= U @ (U.T @ X)
    Qm, _ = np.linalg.qr(X)
    return Qm[:, :k]


def align(Ua, Ub):
    return float(np.mean(np.clip(np.linalg.svd(Ua.T @ Ub, compute_uv=False), 0, 1)))


def chal(C, q, rng):
    r = C.shape[0]
    A = rng.normal(size=(r, q)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.normal(size=(r, q)); B /= np.linalg.norm(B, axis=0, keepdims=True)
    return float(np.sum(np.einsum("ij,ij->j", A, C @ B) ** 2))


def energy(W_sent, Ud, Vd, rank):
    _, s, _ = randomized_svd(W_sent, n_components=rank, n_iter=N_ITER, random_state=0)
    return float(np.linalg.norm(Ud.T @ W_sent @ Vd) ** 2) / max(float(np.sum(s ** 2)), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 106)
    print("Exp 15: sigma sweep. Does rank dependence collapse onto signal visibility?")
    print("=" * 106)
    print(f"  {'rank':>5} {'sigma/base':>11} {'visibility':>11} {'alignment':>10} "
          f"{'chal power':>11} {'energy hon':>11} {'energy decoy':>13} {'decoy caught':>13}")

    out = []
    for rank in RANKS:
        pool = load_pool(rank, 40, rng, args.adapters)
        for m in MULTS:
            sig = SIGMA0 * m
            t0 = time.time()
            vis, al, hs, ss, eh, ed = [], [], [], [], [], []
            for _ in range(args.trials):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                Ut, Vt_, sv = basis(W, rank)
                vis.append(sv[rank - 1] / (sig * EDGE))

                # honest
                N = rng.normal(0.0, sig, size=W.shape)
                Ue, _, _ = basis(W + N, rank)
                al.append(align(Ut, Ue))
                hs.append(chal(Ut.T @ N @ Vt_, Q, rng) / sig ** 2)
                eh.append(energy(W + N, Ut, Vt_, rank))

                # shaped, declared honestly: the working mechanism
                Ns = N + Ut @ ((CHEAT - 1.0) * (Ut.T @ N @ Vt_)) @ Vt_.T
                ss.append(chal(Ut.T @ Ns @ Vt_, Q, rng) / sig ** 2)

                # single decoy: fake direction declared in place of the weakest
                ud, vd = orth_dirs(Ut, 1, rng), orth_dirs(Vt_, 1, rng)
                Wd = W + sv[0] * (ud @ vd.T)
                Ud = np.concatenate([Ut[:, :rank - 1], ud], axis=1)
                Vd = np.concatenate([Vt_[:, :rank - 1], vd], axis=1)
                ed.append(energy(Wd + N, Ud, Vd, rank))

            hs, ss, eh, ed = map(np.array, (hs, ss, eh, ed))
            thr_c = float(np.quantile(hs, ALPHA))
            thr_e = float(np.quantile(eh, ALPHA))
            row = {"rank": rank, "sigma_mult": m, "visibility": float(np.mean(vis)),
                   "alignment": float(np.mean(al)),
                   "challenge_power": float(np.mean(ss < thr_c)),
                   "energy_honest": float(np.mean(eh)),
                   "energy_decoy": float(np.mean(ed)),
                   "decoy_caught": float(np.mean(ed < thr_e))}
            out.append(row)
            print(f"  {rank:>5} {m:>11.2f} {row['visibility']:>11.2f} "
                  f"{row['alignment']:>10.4f} {row['challenge_power']:>11.2f} "
                  f"{row['energy_honest']:>11.4f} {row['energy_decoy']:>13.4f} "
                  f"{row['decoy_caught']:>13.2f}   [{time.time()-t0:.0f}s]")

    with open("results/exp15_sigma_sweep.json", "w") as f:
        json.dump({"config": {"d": D, "sigma_base": SIGMA0, "mults": MULTS,
                              "cheat": CHEAT, "q": Q, "alpha": ALPHA,
                              "trials": args.trials}, "rows": out}, f, indent=2)
    print("\nSaved: results/exp15_sigma_sweep.json")
    print("Collapse test: sort by visibility and check whether alignment follows it")
    print("regardless of rank. If so, rank was a proxy all along.")


if __name__ == "__main__":
    main()
