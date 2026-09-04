#!/usr/bin/env python3
"""Exp 14: does the decoy attack scale?

exp13 built the declaration check and it defeated every crude lie. One
adversary survived: a decoy that plants a fake rank-1 component, declares
(r-1 true directions plus the decoy), and starves the excluded true
direction. It passed both the energy check and the challenge.

Its payoff was poor, delivering 87 to 98 percent of contracted privacy,
because it starves only one direction out of r. That conclusion only holds
if the attack does not scale. This experiment scales it.

A k-decoy adversary plants k fake components, declares (r-k true directions
plus the k decoys), and starves the k excluded true directions. Privacy lost
should grow roughly with k/r. The question is whether the checks keep
passing, and what the attack costs.

Cost is measured two ways, both of which the adversary actually pays:
  update distortion  how far the submitted update sits from the client's
                     true update after SPA's rank-r truncation, relative to
                     the true update. Decoys are fake signal, so this is
                     real damage to the client's own contribution.
  energy ratio       how close the declaration looks to honest. If large k
                     pushes this down, the check catches it after all.

The outcome that matters: is there a k where privacy loss is large AND the
combined flag rate stays low AND distortion stays tolerable? If not, the
decoy family is economically dead and the mechanism holds.

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
RANKS = [8, 16, 32]
SIGMA = 5e-5
CHEAT = 0.5
Q = 20
ALPHA = 0.05
N_TRIALS = 40
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


def build_k_decoy(W, rank, k, rng):
    """k decoys: starve the k weakest true directions, declare them out."""
    Ut, Vt_, st = basis(W, rank)
    N = rng.normal(0.0, SIGMA, size=W.shape)
    if k == 0:
        return W + N, N, Ut, Vt_, Ut, Vt_

    starved = np.arange(rank - k, rank)          # the k weakest directions
    C = Ut.T @ N @ Vt_
    d = np.where(np.isin(np.arange(rank), starved), CHEAT, 1.0)
    N = N + Ut @ ((d[:, None] - 1.0) * C) @ Vt_.T

    Ud_extra = orth_dirs(Ut, k, rng)
    Vd_extra = orth_dirs(Vt_, k, rng)
    # decoys sized like the signal directions they replace, so the declared
    # subspace still looks like it carries the update's top-r energy
    mags = st[:k]
    Wd = W + (Ud_extra * mags) @ Vd_extra.T

    Ud = np.concatenate([Ut[:, :rank - k], Ud_extra], axis=1)
    Vd = np.concatenate([Vt_[:, :rank - k], Vd_extra], axis=1)
    return Wd + N, N, Ud, Vd, Ut, Vt_


def energy_ratio(W_sent, Ud, Vd, rank):
    _, s, _ = randomized_svd(W_sent, n_components=rank, n_iter=N_ITER, random_state=0)
    return float(np.linalg.norm(Ud.T @ W_sent @ Vd) ** 2) / max(float(np.sum(s ** 2)), 1e-300)


def challenge_stat(N, Ud, Vd, q, rng):
    C = Ud.T @ N @ Vd
    r = C.shape[0]
    A = rng.normal(size=(r, q)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.normal(size=(r, q)); B /= np.linalg.norm(B, axis=0, keepdims=True)
    return float(np.sum(np.einsum("ij,ij->j", A, C @ B) ** 2) / SIGMA ** 2)


def distortion(W_sent, W, rank):
    """Damage to the client's own contribution after rank-r truncation."""
    U, s, Vt = randomized_svd(W_sent, n_components=rank, n_iter=N_ITER, random_state=0)
    return float(np.linalg.norm((U * s) @ Vt - W) / np.linalg.norm(W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 104)
    print(f"Exp 14: does the decoy attack scale?  (sigma={SIGMA}, c={CHEAT}, q={Q}, alpha={ALPHA})")
    print("=" * 104)

    out = []
    for rank in args.ranks:
        pool = load_pool(rank, 60, rng, args.adapters)
        ks = sorted({0, 1, max(1, rank // 4), rank // 2, (3 * rank) // 4, rank - 1})
        ks = [k for k in ks if 0 <= k <= rank - 1]
        t0 = time.time()

        data = {}
        for k in ks:
            er, ch, pv, ds = [], [], [], []
            for _ in range(args.trials):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                W_sent, N, Ud, Vd, Ut, Vt_ = build_k_decoy(W, rank, k, rng)
                er.append(energy_ratio(W_sent, Ud, Vd, rank))
                ch.append(challenge_stat(N, Ud, Vd, Q, rng))
                pv.append(float(np.sqrt(np.mean((Ut.T @ N @ Vt_) ** 2))))
                ds.append(distortion(W_sent, W, rank))
            data[k] = (np.array(er), np.array(ch), float(np.mean(pv)), float(np.mean(ds)))

        thr_e = float(np.quantile(data[0][0], ALPHA))
        thr_c = float(np.quantile(data[0][1], ALPHA))
        row = {"rank": rank, "thr_energy": thr_e, "k": {}}
        print(f"\nrank {rank}   energy threshold {thr_e:.4f}   [{time.time()-t0:.0f}s]")
        print(f"  {'k decoys':>9} {'starved':>8} {'energy':>8} {'privacy':>11} "
              f"{'% of contract':>14} {'distortion':>11} {'EITHER flag':>12}")
        for k in ks:
            er, ch, pv, ds = data[k]
            fb = float(np.mean((er < thr_e) | (ch < thr_c)))
            pct = 100.0 * pv / SIGMA
            row["k"][str(k)] = {"energy_ratio": float(np.mean(er)), "privacy": pv,
                                "pct_of_contract": pct, "distortion": ds,
                                "combined_flag": fb}
            tag = "  <-- both checks passed" if (fb < 0.30 and pct < 70) else ""
            print(f"  {k:>9} {k}/{rank:<6} {np.mean(er):>8.4f} {pv:>11.3e} "
                  f"{pct:>13.1f}% {ds:>11.4f} {fb:>12.2f}{tag}")
        out.append(row)

    with open("results/exp14_multi_decoy.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "q": Q,
                              "alpha": ALPHA, "trials": args.trials},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp14_multi_decoy.json")
    print("A dangerous row is one with LOW percent of contract, LOW flag rate, and")
    print("LOW distortion. If none exists, the decoy family is economically dead.")


if __name__ == "__main__":
    main()
