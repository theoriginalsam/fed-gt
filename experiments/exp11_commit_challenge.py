#!/usr/bin/env python3
"""Exp 11 (Route B): commit-and-challenge, tested against the exp7 adversary.

Three passive defenses failed (exp7 tail, exp8 spike consistency, exp9/10
temporal) because inside the signal subspace a missing-noise shortfall is
confounded with an unknown signal. Route B removes the inference entirely.

Protocol
--------
  1. On submission the client also commits to the noise N it used. The
     commitment is binding, so N can no longer be changed. (Modelled here
     as: the revealed values must come from the N actually used. No crypto
     is implemented; binding is assumed.)
  2. At audit time the server estimates the signal subspace FROM THE NOISY
     SUBMISSION -- it does not know Delta_W.
  3. It draws q random directions inside that estimated subspace and asks
     the client to reveal the scalar projections z_j = u_j^T N v_j.
  4. It rejects when sum(z_j^2)/sigma^2 falls below the chi-squared
     alpha-quantile with q degrees of freedom.

For isotropic noise every unit direction gives z ~ N(0, sigma^2), so an
honest client passes wherever the server looks. An adversary injecting
c*sigma inside the true signal subspace gives z ~ N(0, c^2 sigma^2) --
but only to the extent the server's ESTIMATED subspace overlaps the true
one. That overlap is the thing an idealized chi-squared calculation
assumes away, and it degrades exactly where signal approaches the noise
floor. This experiment measures the real power against the idealized
curve, and records the subspace alignment that explains any gap.

CPU-only. Reads ./adapters/ (6-round set, q_proj).
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
from scipy.stats import chi2
from sklearn.utils.extmath import randomized_svd

from fedgt.spectral_audit import tail_energy_dof

D = 3584
RANKS = [4, 8, 16, 32]
MARGIN = 2
SIGMA = 5e-5
CHEAT = 0.5
ALPHA = 0.05
QS = [5, 10, 20, 40]
N_TRIALS = 150
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
    U, _, Vt = randomized_svd(W, n_components=r, n_iter=N_ITER, random_state=0)
    return U, Vt.T


def project_out(E, U, V):
    E = E - U @ (U.T @ E)
    return E - (E @ V) @ V.T


def make_noise(W, rank, mode, rng):
    """Return the noise N the client actually injected."""
    if mode == "honest":
        return rng.normal(0.0, SIGMA, size=W.shape)
    # shaped: c*sigma inside the TRUE signal subspace, padding outside,
    # scaled so the tail statistic still matches an honest client (exp7).
    U, V = basis(W, rank)
    G = rng.normal(0.0, CHEAT * SIGMA, size=W.shape)
    G_top = G - project_out(G, U, V)
    E = project_out(rng.normal(0.0, 1.0, size=W.shape), U, V)
    E /= (np.linalg.norm(E) + 1e-30)
    k = min(rank + MARGIN, D - 1)
    dof = tail_energy_dof(D, D, k)
    return G_top + E * (SIGMA * np.sqrt(dof))


def challenge(N, U_est, V_est, q, rng):
    """Reveal q scalar projections onto random directions in the estimated subspace."""
    r = U_est.shape[1]
    A = rng.normal(size=(r, q)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.normal(size=(r, q)); B /= np.linalg.norm(B, axis=0, keepdims=True)
    UA, VB = U_est @ A, V_est @ B            # unit directions in the estimate
    return np.einsum("ij,ij->j", UA, N @ VB)  # z_j = u_j^T N v_j


def alignment(U_true, U_est):
    """Mean cosine of principal angles between two subspaces (1.0 = identical)."""
    s = np.linalg.svd(U_true.T @ U_est, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 92)
    print(f"Exp 11 (Route B): commit-and-challenge vs the shaping adversary  "
          f"(sigma={SIGMA}, cheat={CHEAT}, alpha={ALPHA})")
    print("=" * 92)
    ideal = {q: float(chi2.cdf(chi2.ppf(ALPHA, q) / CHEAT ** 2, q)) for q in QS}
    print("  idealized power (exact chi-squared, perfect subspace): "
          + "  ".join(f"q={q}:{ideal[q]:.3f}" for q in QS))

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng, args.adapters)
        zs = {"honest": [], "shaped": []}
        aligns = []
        for mode in ("honest", "shaped"):
            for _ in range(args.trials):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                N = make_noise(W, rank, mode, rng)
                W_sent = W + N
                U_est, V_est = basis(W_sent, rank)      # server sees only W_sent
                if mode == "shaped":
                    aligns.append(alignment(basis(W, rank)[0], U_est))
                zs[mode].append(challenge(N, U_est, V_est, max(QS), rng))
        zh = np.array(zs["honest"]); zsh = np.array(zs["shaped"])

        row = {"rank": rank, "subspace_alignment": float(np.mean(aligns)), "q": {}}
        print(f"\nrank {rank}   subspace alignment (true vs server estimate) = "
              f"{np.mean(aligns):.4f}   [{time.time()-t0:.0f}s]")
        print("  Challenge directions are derived from W_sent, so they correlate with")
        print("  the noise being measured and inflate every projection. The theoretical")
        print("  chi-squared threshold is therefore invalid; calibrate on honest clients.")
        print(f"  {'q':>4} {'thr(chi2)':>10} {'thr(calib)':>11} {'FA':>7} "
              f"{'power':>8} {'ideal':>8} {'gap':>8}")
        for q in QS:
            sh = np.sum(zh[:, :q] ** 2, axis=1) / SIGMA ** 2
            ss = np.sum(zsh[:, :q] ** 2, axis=1) / SIGMA ** 2
            thr_t = float(chi2.ppf(ALPHA, q))
            thr_c = float(np.quantile(sh, ALPHA))      # empirically calibrated
            fa_t = float(np.mean(sh < thr_t)); pw_t = float(np.mean(ss < thr_t))
            fa_c = float(np.mean(sh < thr_c)); pw_c = float(np.mean(ss < thr_c))
            row["q"][str(q)] = {"thr_chi2": thr_t, "thr_calibrated": thr_c,
                                "fa_chi2": fa_t, "power_chi2": pw_t,
                                "false_alarm": fa_c, "power": pw_c,
                                "ideal": ideal[q]}
            print(f"  {q:>4} {thr_t:>10.2f} {thr_c:>11.2f} {fa_c:>7.3f} "
                  f"{pw_c:>8.3f} {ideal[q]:>8.3f} {pw_c-ideal[q]:>+8.3f}")
        out.append(row)

    with open("results/exp11_commit_challenge.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "alpha": ALPHA,
                              "qs": QS, "trials": args.trials},
                   "idealized_power": ideal, "rows": out}, f, indent=2)
    print("\nSaved: results/exp11_commit_challenge.json")


if __name__ == "__main__":
    main()
