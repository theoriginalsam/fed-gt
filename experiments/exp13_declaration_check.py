#!/usr/bin/env python3
"""Exp 13: can the client lie about its declared signal subspace?

exp12 showed the challenge works at every rank once the client DECLARES the
signal subspace instead of the server estimating it. That result assumes the
declaration is honest. Nothing checked it.

A cheating client can simply point the auditor somewhere harmless: declare a
subspace where it did inject full noise, and starve the real signal
directions, which are then never challenged. This experiment builds the
verification step and tries to break it.

The check
---------
A rank-r update puts essentially all of its energy in its true top-r
subspace. So the server compares the energy the DECLARED subspace captures
against the update's own top-r singular values:

    ratio = || U_dec^T W_sent V_dec ||_F^2  /  sum_{i<=r} s_i(W_sent)^2

An honest declaration gives ratio near 1. A false one captures less and is
rejected before any challenge is issued. The threshold is calibrated on
honest clients, since the theoretical value is not exact once noise is added.

Adversaries tested
------------------
  honest        declares truthfully, injects full noise (baseline)
  shaped        declares truthfully, starves the declared subspace
                (exp12's adversary: should be caught by the CHALLENGE)
  random_decl   declares a random subspace, starves the true one
                (should be caught by the ENERGY CHECK)
  tilt_XX       declares a subspace rotated away from the true one by angle
                XX degrees, starves the true one; sweeps how far it can tilt
  decoy         adds a fake high-energy rank-1 component in a harmless
                direction, declares (r-1 true directions + the decoy), and
                starves the excluded true direction. The decoy supplies the
                missing energy, so a pure energy check may pass it. This is
                the attack the check is most likely to miss.

Reported per adversary: energy-check flag rate, challenge flag rate, the
combined rate (either fires), and the privacy actually delivered.

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
RANKS = [4, 8, 16, 32]
SIGMA = 5e-5
CHEAT = 0.5
Q = 20
ALPHA = 0.05
N_TRIALS = 40
N_ITER = 4
TILTS = [15, 40]
ADVERSARIES = ["honest", "shaped", "random_decl"] + [f"tilt_{t}" for t in TILTS] + ["decoy"]


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


def orth_complement_dirs(U, k, rng):
    """k orthonormal directions orthogonal to the columns of U."""
    X = rng.normal(size=(U.shape[0], k))
    X -= U @ (U.T @ X)
    Qm, _ = np.linalg.qr(X)
    return Qm[:, :k]


def tilt(U, deg, rng):
    """Rotate each column of U away from its own span by `deg` degrees."""
    P = orth_complement_dirs(U, U.shape[1], rng)
    th = np.deg2rad(deg)
    Ur = U * np.cos(th) + P * np.sin(th)
    Qm, _ = np.linalg.qr(Ur)
    return Qm


def build(W, rank, adv, rng):
    """Return (W_sent, N, U_dec, V_dec, starved_mask_in_declared_basis)."""
    Ut, Vt_, st = basis(W, rank)
    N = rng.normal(0.0, SIGMA, size=W.shape)

    def starve(U, V, cols=None):
        C = U.T @ N @ V
        d = np.full(C.shape[0], CHEAT) if cols is None else np.where(
            np.isin(np.arange(C.shape[0]), cols), CHEAT, 1.0)
        return N + U @ ((d[:, None] - 1.0) * C) @ V.T

    if adv == "honest":
        return W + N, N, Ut, Vt_
    if adv == "shaped":
        N2 = starve(Ut, Vt_)
        return W + N2, N2, Ut, Vt_
    if adv == "random_decl":
        N2 = starve(Ut, Vt_)
        Ud = orth_complement_dirs(Ut, rank, rng)
        Vd = orth_complement_dirs(Vt_, rank, rng)
        return W + N2, N2, Ud, Vd
    if adv.startswith("tilt_"):
        N2 = starve(Ut, Vt_)
        deg = float(adv.split("_")[1])
        return W + N2, N2, tilt(Ut, deg, rng), tilt(Vt_, deg, rng)
    if adv == "decoy":
        # fake rank-1 component, as strong as the top real signal
        ud = orth_complement_dirs(Ut, 1, rng)
        vd = orth_complement_dirs(Vt_, 1, rng)
        Wd = W + st[0] * (ud @ vd.T)
        # starve only the weakest true direction, which we then exclude
        N2 = starve(Ut, Vt_, cols=[rank - 1])
        Ud = np.concatenate([Ut[:, :rank - 1], ud], axis=1)
        Vd = np.concatenate([Vt_[:, :rank - 1], vd], axis=1)
        return Wd + N2, N2, Ud, Vd
    raise ValueError(adv)


def energy_ratio(W_sent, Ud, Vd, rank):
    _, s, _ = randomized_svd(W_sent, n_components=rank, n_iter=N_ITER, random_state=0)
    top = float(np.sum(s ** 2))
    got = float(np.linalg.norm(Ud.T @ W_sent @ Vd) ** 2)
    return got / max(top, 1e-300)


def challenge_stat(N, Ud, Vd, q, rng):
    C = Ud.T @ N @ Vd
    r = C.shape[0]
    A = rng.normal(size=(r, q)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.normal(size=(r, q)); B /= np.linalg.norm(B, axis=0, keepdims=True)
    z = np.einsum("ij,ij->j", A, C @ B)
    return float(np.sum(z ** 2) / SIGMA ** 2)


def true_privacy(N, Ut, Vt_):
    """Effective noise std inside the TRUE signal subspace."""
    return float(np.sqrt(np.mean((Ut.T @ N @ Vt_) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 100)
    print(f"Exp 13: verifying the declared subspace  "
          f"(sigma={SIGMA}, c={CHEAT}, q={Q}, alpha={ALPHA})")
    print("=" * 100)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng, args.adapters)
        data = {}
        for adv in ADVERSARIES:
            er, ch, pv = [], [], []
            for _ in range(args.trials):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                Ut, Vt_, _ = basis(W, rank)
                W_sent, N, Ud, Vd = build(W, rank, adv, rng)
                er.append(energy_ratio(W_sent, Ud, Vd, rank))
                ch.append(challenge_stat(N, Ud, Vd, Q, rng))
                pv.append(true_privacy(N, Ut, Vt_))
            data[adv] = (np.array(er), np.array(ch), float(np.mean(pv)))

        # thresholds calibrated on honest clients
        thr_e = float(np.quantile(data["honest"][0], ALPHA))       # reject if BELOW
        thr_c = float(np.quantile(data["honest"][1], ALPHA))       # reject if BELOW

        row = {"rank": rank, "thr_energy": thr_e, "thr_challenge": thr_c, "adversaries": {}}
        print(f"\nrank {rank}   energy threshold {thr_e:.4f}   [{time.time()-t0:.0f}s]")
        print(f"  {'adversary':<14} {'energy ratio':>13} {'privacy':>11} "
              f"{'ENERGY flag':>12} {'CHALLENGE flag':>15} {'EITHER':>8}")
        for adv in ADVERSARIES:
            er, ch, pv = data[adv]
            fe = float(np.mean(er < thr_e))
            fc = float(np.mean(ch < thr_c))
            fb = float(np.mean((er < thr_e) | (ch < thr_c)))
            row["adversaries"][adv] = {"energy_ratio": float(np.mean(er)),
                                       "privacy": pv, "energy_flag": fe,
                                       "challenge_flag": fc, "combined_flag": fb}
            print(f"  {adv:<14} {np.mean(er):>13.4f} {pv:>11.3e} "
                  f"{fe:>12.2f} {fc:>15.2f} {fb:>8.2f}")
        out.append(row)

    with open("results/exp13_declaration_check.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "q": Q,
                              "alpha": ALPHA, "trials": args.trials},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp13_declaration_check.json")
    print("Honest privacy is 5.0e-05. Any adversary with lower privacy and a low")
    print("EITHER rate has defeated both checks.")


if __name__ == "__main__":
    main()
