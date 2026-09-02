#!/usr/bin/env python3
"""Exp 12: declared-subspace challenge, and the adversary's counter-move.

exp11 showed commit-and-challenge defeats the shaping adversary, but only
where the server can locate the signal subspace: power tracked subspace
alignment (0.9995 -> 0.95, but 0.5122 -> 0.03 at r32, where the weakest
signal sits below the noise floor).

Two questions follow, one for each side.

DEFENSE. The server does not have to guess. The client knows its own
signal subspace exactly and can DECLARE it as part of the commitment; the
server checks the declaration explains the submitted update's top-r energy
before challenging inside it. Does that restore power at high rank?

ATTACK. exp11's adversary starves the whole signal subspace. A smarter one
starves only a fraction f of the r directions, so random challenges often
miss, trading privacy gain for evasion. Detection should fall roughly as
the challenge's expected variance ratio f*c^2 + (1-f).

A third defense variant answers the attack: instead of random mixed
directions inside the subspace, challenge along the DECLARED BASIS
directions individually, so a single starved direction shows up sharply
rather than being averaged away.

Also recorded: the fraction of total injected noise energy the adversary
actually alters. The privacy-relevant noise lives in the r x r core block,
r^2 of d^2 entries, so an adversary can halve the protection on the signal
while touching a vanishing share of the noise energy. That is precisely
why an energy-based tail test cannot see it.

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
FRACS = [1.0, 0.5, 0.25]
QS = [10, 20, 40]
ALPHA = 0.05
N_TRIALS = 60
N_ITER = 4
DEFENSES = ["estimated", "declared", "declared_basis"]


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


def core_scales(r, f, rng):
    """Per-direction noise scale inside the signal subspace."""
    d = np.ones(r)
    m = max(1, int(round(f * r)))
    d[rng.choice(r, size=m, replace=False)] = CHEAT
    return d


def challenge_z(C, q, mode, rng):
    """Revealed projections from the r x r core block C = U^T N V."""
    r = C.shape[0]
    if mode == "basis":
        # probe declared basis directions individually, cycling if q > r
        ii = np.arange(q) % r
        jj = rng.integers(0, r, size=q)
        return C[ii, jj]
    A = rng.normal(size=(r, q)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.normal(size=(r, q)); B /= np.linalg.norm(B, axis=0, keepdims=True)
    return np.einsum("ij,ij->j", A, C @ B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 96)
    print(f"Exp 12: declared-subspace challenge vs the partial-subspace adversary "
          f"(sigma={SIGMA}, c={CHEAT}, alpha={ALPHA})")
    print("=" * 96)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng, args.adapters)
        Q = max(QS)

        # ---- honest trials: calibrate each defense ----
        hz = {d: [] for d in DEFENSES}
        aligns = []
        for _ in range(args.trials):
            A, B = pool[rng.integers(len(pool))]
            W = B @ A
            Ut, Vt_ = basis(W, rank)                      # true (declared)
            N = rng.normal(0.0, SIGMA, size=W.shape)
            Ue, Ve = basis(W + N, rank)                   # server estimate
            aligns.append(float(np.mean(np.clip(
                np.linalg.svd(Ut.T @ Ue, compute_uv=False), 0, 1))))
            hz["estimated"].append(challenge_z(Ue.T @ N @ Ve, Q, "rand", rng))
            hz["declared"].append(challenge_z(Ut.T @ N @ Vt_, Q, "rand", rng))
            hz["declared_basis"].append(challenge_z(Ut.T @ N @ Vt_, Q, "basis", rng))
        hz = {d: np.array(v) for d, v in hz.items()}

        # ---- adversary trials at each fraction f ----
        sz = {(d, f): [] for d in DEFENSES for f in FRACS}
        eff_priv, energy_frac = {}, None
        for f in FRACS:
            privs = []
            for _ in range(args.trials):
                A, B = pool[rng.integers(len(pool))]
                W = B @ A
                Ut, Vt_ = basis(W, rank)
                N = rng.normal(0.0, SIGMA, size=W.shape)
                # starve a fraction f of the declared directions in the core block
                C = Ut.T @ N @ Vt_
                d = core_scales(rank, f, rng)
                C_new = (d[:, None]) * C
                N = N + Ut @ (C_new - C) @ Vt_.T
                privs.append(float(np.sqrt(np.mean(C_new ** 2))))
                Ue, Ve = basis(W + N, rank)
                sz[("estimated", f)].append(challenge_z(Ue.T @ N @ Ve, Q, "rand", rng))
                sz[("declared", f)].append(challenge_z(Ut.T @ N @ Vt_, Q, "rand", rng))
                sz[("declared_basis", f)].append(
                    challenge_z(Ut.T @ N @ Vt_, Q, "basis", rng))
            eff_priv[f] = float(np.mean(privs))
            if energy_frac is None:
                energy_frac = rank * rank / float(D * D)
        sz = {k: np.array(v) for k, v in sz.items()}

        row = {"rank": rank, "alignment": float(np.mean(aligns)),
               "core_energy_fraction": energy_frac,
               "effective_core_sigma": eff_priv, "results": {}}
        print(f"\nrank {rank}   alignment={np.mean(aligns):.4f}   "
              f"core block = {energy_frac:.2e} of noise energy   [{time.time()-t0:.0f}s]")
        print(f"  {'defense':<16} {'f':>5} {'core sigma':>11} "
              + " ".join(f"{'pw q='+str(q):>9}" for q in QS) + f" {'FA':>7}")
        for dfn in DEFENSES:
            for f in FRACS:
                cells, fa_last = [], None
                for q in QS:
                    h = np.sum(hz[dfn][:, :q] ** 2, axis=1)
                    s = np.sum(sz[(dfn, f)][:, :q] ** 2, axis=1)
                    thr = float(np.quantile(h, ALPHA))
                    cells.append(float(np.mean(s < thr)))
                    fa_last = float(np.mean(h < thr))
                row["results"][f"{dfn}|f={f}"] = {
                    "power": dict(zip(map(str, QS), cells)), "false_alarm": fa_last}
                print(f"  {dfn:<16} {f:>5.2f} {eff_priv[f]:>11.3e} "
                      + " ".join(f"{c:>9.3f}" for c in cells) + f" {fa_last:>7.3f}")
        out.append(row)

    with open("results/exp12_declared_subspace.json", "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "alpha": ALPHA,
                              "qs": QS, "fracs": FRACS, "trials": args.trials},
                   "rows": out}, f, indent=2)
    print("\nSaved: results/exp12_declared_subspace.json")


if __name__ == "__main__":
    main()
