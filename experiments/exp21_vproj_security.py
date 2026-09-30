#!/usr/bin/env python3
"""Exp 21: does any of this depend on the update being square?

Every experiment so far reads q_proj, which under Qwen2.5-7B is 3584 x 3584.
Several of them hard-code that number. The other adapted module, v_proj, is
512 x 3584 under grouped-query attention, and it is a genuinely different
object rather than a smaller copy:

                          q_proj            v_proj
  shape                   3584 x 3584       512 x 3584
  noise floor             119.7 sigma       82.5 sigma
  core block r^2/(d1 d2)  1.2 - 80 ppm      8.7 - 558 ppm
  rank 32 vs short side   0.9 %             6.2 %

So the core block is up to seven times less negligible, the Marchenko-Pastur
edge sits lower, and at rank 32 the update is no longer especially low rank
relative to its short side. If any conclusion was an artifact of squareness,
this is where it shows.

The same script runs on either module, chosen with --module, so the q_proj
and v_proj columns are produced by identical code. A difference between them
is then geometry, not a code difference.

Four questions, each mirroring an earlier experiment
----------------------------------------------------
  1. spectrum      decay gamma, decay depth and the visibility of the weakest
                   signal direction                          (exp5a, exp15, exp16)
  2. attack        does the calibrated noise-shaping cheater still evade the
                   tail-energy test                                 (exp7, exp17)
  3. defence       does the declared-subspace challenge still catch it
                                                                  (exp12, exp17)
  4. decoy         does the rank check still see a planted decoy, and what
                   does the decoy steal and cost            (exp13, exp18, exp20)

Predictions worth recording before the run. Questions 2 and 3 should be
shape independent: the evasion argument is about where noise sits rather than
how much, and the challenge reads U^T N V which is exactly r x r iid Gaussian
whatever the shape. Question 4 should NOT be shape independent, because the
rank check depends on the weakest signal standing clear of the noise floor,
and both the floor and the rank-to-dimension ratio move.

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

RANKS = [4, 8, 16, 32]
MARGIN = 2
SIGMA = 5e-5
CHEAT = 0.5
POOL = 4
Q = 20
ALPHA = 0.05
N_ITER = 4
N_CAL = 120
N_TEST = 60


def load_pool(module, rank, n, rng, adir):
    files = sorted(glob.glob(f"{adir}/*_r{rank}__*{module}.pt"))
    if not files:
        raise SystemExit(f"no rank-{rank} {module} adapters under {adir}/")
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out


def basis(A, B):
    """Exact SVD of W = B @ A via two thin QRs. Shape agnostic."""
    Qb, Rb = np.linalg.qr(B)
    Qa, Ra = np.linalg.qr(A.T)
    u, s, vt = np.linalg.svd(Rb @ Ra.T)
    return Qb @ u, Qa @ vt.T, s


def orth_dirs(U, k, rng):
    X = rng.normal(size=(U.shape[0], k))
    X -= U @ (U.T @ X)
    Qm, _ = np.linalg.qr(X)
    return Qm[:, :k]


def topk_basis(W, k):
    U, _, Vt = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=0)
    return U, Vt.T


def project_out(E, U, V):
    E = E - U @ (U.T @ E)
    return E - (E @ V) @ V.T


def fast_sigma2(W, rank):
    d1, d2 = W.shape
    k = min(rank + MARGIN, min(d1, d2) - 1)
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    return ((float(np.sum(W * W)) - float(np.sum(s ** 2)))
            / tail_energy_dof(d1, d2, k))


# ------------------------------------------------------------ 1. spectrum
def spectrum_stats(pool, rank, edge, n, rng):
    gam, depth, vis = [], [], []
    for _ in range(n):
        A, B = pool[rng.integers(len(pool))]
        _, _, s = basis(A, B)
        i = np.arange(1, rank + 1)
        g = -np.polyfit(np.log(i), np.log(s / s[0]), 1)[0]
        gam.append(g)
        depth.append(s[-1] / s[0])
        vis.append(s[-1] / (SIGMA * edge))
    return (float(np.mean(gam)), float(np.mean(depth)), float(np.mean(vis)))


# -------------------------------------------------------------- 2. attack
def shaped_submission(W, rank, rng):
    """exp7's calibrated cheater: core noise at CHEAT*sigma, padding outside
    tuned so the tail statistic matches what an honest draw would have read."""
    d1, d2 = W.shape
    k = min(rank + MARGIN, min(d1, d2) - 1)
    dof = tail_energy_dof(d1, d2, k)
    U, V = topk_basis(W, k)

    G = rng.normal(0.0, CHEAT * SIGMA, size=W.shape)
    base = W + (G - project_out(G, U, V))

    E = project_out(rng.normal(0.0, 1.0, size=W.shape), U, V)
    E /= (np.linalg.norm(E) + 1e-30)

    target = fast_sigma2(W + rng.normal(0.0, SIGMA, size=W.shape), rank)
    c0 = fast_sigma2(base, rank) * dof
    a1 = np.sqrt(max(target * dof - c0, 0.0))
    c1 = fast_sigma2(base + a1 * E, rank) * dof - a1 ** 2
    a2 = np.sqrt(max(target * dof - c1, 0.0))
    Ws = base + a2 * E
    return Ws, fast_sigma2(Ws, rank)


def pooled_stat(pool, rank, mode, rng):
    vals = []
    for _ in range(POOL):
        A, B = pool[rng.integers(len(pool))]
        W = B @ A
        if mode == "honest":
            vals.append(fast_sigma2(W + rng.normal(0.0, SIGMA, size=W.shape), rank))
        else:
            vals.append(shaped_submission(W, rank, rng)[1])
    return float(np.mean(vals))


# ----------------------------------------------- 3 and 4. challenge, decoy
def starve(N, U, V, cols=None):
    C = U.T @ N @ V
    d = (np.full(C.shape[0], CHEAT) if cols is None
         else np.where(np.isin(np.arange(C.shape[0]), cols), CHEAT, 1.0))
    return N + U @ ((d[:, None] - 1.0) * C) @ V.T


def challenge_stat(N, Ud, Vd, rng):
    C = Ud.T @ N @ Vd
    r = C.shape[0]
    a = rng.normal(size=(r, Q)); a /= np.linalg.norm(a, axis=0)
    b = rng.normal(size=(r, Q)); b /= np.linalg.norm(b, axis=0)
    z = np.einsum("iq,ij,jq->q", a, C, b)
    return float(np.sum(z ** 2) / SIGMA ** 2)


def head_svd(W, rank):
    return randomized_svd(W, n_components=rank + 1, n_iter=N_ITER,
                          random_state=None)


def tilt_to(W_sent, Ut, Vt_, ud, vd, rank, top2, target, rng):
    """Tilt the declared decoy direction until the energy ratio hits target."""
    Uk = np.concatenate([Ut[:, :rank - 1], ud], axis=1)
    Vk = np.concatenate([Vt_[:, :rank - 1], vd], axis=1)
    pu, pv = orth_dirs(Uk, 1, rng), orth_dirs(Vk, 1, rng)
    M = np.concatenate([Uk, pu], axis=1).T @ W_sent @ np.concatenate([Vk, pv], axis=1)

    def ratio_at(d):
        S = np.zeros((rank + 1, rank))
        S[:rank - 1, :rank - 1] = np.eye(rank - 1)
        S[rank - 1, rank - 1], S[rank, rank - 1] = np.cos(d), np.sin(d)
        return float(np.linalg.norm(S.T @ M @ S) ** 2) / top2

    lo, hi = 0.0, np.pi / 2
    if ratio_at(hi) > target:
        d = hi
    else:
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ratio_at(mid) > target else (lo, mid)
        d = 0.5 * (lo + hi)
    return (np.concatenate([Ut[:, :rank - 1], ud * np.cos(d) + pu * np.sin(d)], 1),
            np.concatenate([Vt_[:, :rank - 1], vd * np.cos(d) + pv * np.sin(d)], 1))


def declared_run(pool, rank, adv, edge, n, rng, target=None):
    """adv in {honest, shaped, decoy}. Returns ratio, excess, challenge,
    privacy delivered, distortion."""
    er, ex, ch, pv, ds = [], [], [], [], []
    for _ in range(n):
        A, B = pool[rng.integers(len(pool))]
        Ut, Vt_, st = basis(A, B)
        W = B @ A
        N = rng.normal(0.0, SIGMA, size=W.shape)

        if adv == "honest":
            W_sent, N2, ud, vd = W + N, N, None, None
        elif adv == "shaped":
            N2 = starve(N, Ut, Vt_); W_sent, ud, vd = W + N2, None, None
        else:                                    # strongest decoy from exp18
            ud, vd = orth_dirs(Ut, 1, rng), orth_dirs(Vt_, 1, rng)
            N2 = starve(N, Ut, Vt_, cols=[rank - 1])
            W_sent = W + st[rank - 1] * (ud @ vd.T) + N2

        U, s, Vt = head_svd(W_sent, rank)
        top2 = float(np.sum(s[:rank] ** 2))
        ex.append(float(s[rank] / (SIGMA * edge)))

        if adv == "decoy":
            Ud, Vd = tilt_to(W_sent, Ut, Vt_, ud, vd, rank, top2, target, rng)
        else:
            Ud, Vd = Ut, Vt_
        er.append(float(np.linalg.norm(Ud.T @ W_sent @ Vd) ** 2) / max(top2, 1e-300))
        ch.append(challenge_stat(N2, Ud, Vd, rng))
        pv.append(float(np.sqrt(np.mean((Ut.T @ N2 @ Vt_) ** 2))))
        ds.append(float(np.linalg.norm((U[:, :rank] * s[:rank]) @ Vt[:rank] - W)
                        / np.linalg.norm(W)))
    return {"ratio": np.array(er), "excess": np.array(ex), "chal": np.array(ch),
            "privacy": float(np.mean(pv)), "distortion": float(np.mean(ds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="v_proj", choices=["v_proj", "q_proj"])
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--cal", type=int, default=N_CAL)
    ap.add_argument("--test", type=int, default=N_TEST)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"results/exp21_{args.module}_security.json"
    rng = np.random.default_rng(0)

    print("=" * 112)
    print(f"Exp 21: the full battery on {args.module}  "
          f"(sigma={SIGMA}, c={CHEAT}, q={Q}, alpha={ALPHA}, "
          f"cal={args.cal}, n={args.test})")
    print("=" * 112)

    rows = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(args.module, rank, 60, rng, args.adapters)
        d1, d2 = pool[0][1].shape[0], pool[0][0].shape[1]
        edge = np.sqrt(d1) + np.sqrt(d2)
        gam, depth, vis = spectrum_stats(pool, rank, edge, 40, rng)

        # 2. attack against the tail-energy test
        null = np.array([pooled_stat(pool, rank, "honest", rng)
                         for _ in range(args.cal)])
        thr_a = float(np.quantile(null, ALPHA))
        hon_a = np.array([pooled_stat(pool, rank, "honest", rng)
                          for _ in range(args.test)])
        atk = np.array([pooled_stat(pool, rank, "shaped", rng)
                        for _ in range(args.test)])
        fa = float(np.mean(hon_a < thr_a)); evade = float(np.mean(atk < thr_a))

        # 3 and 4. declared subspace, challenge, rank check
        cal = declared_run(pool, rank, "honest", edge, args.cal, rng)
        tgt = float(np.median(cal["ratio"]))
        a3 = ALPHA / 3.0                 # union of three checks sits at alpha
        cut = {"chal": float(np.quantile(cal["chal"], a3)),
               "exc": float(np.quantile(cal["excess"], 1 - a3)),
               "e_lo": float(np.quantile(cal["ratio"], a3))}

        def fire(d):
            return float(np.mean((d["chal"] < cut["chal"]) |
                                 (d["excess"] > cut["exc"]) |
                                 (d["ratio"] < cut["e_lo"])))

        hon = declared_run(pool, rank, "honest", edge, args.test, rng)
        shp = declared_run(pool, rank, "shaped", edge, args.test, rng)
        dec = declared_run(pool, rank, "decoy", edge, args.test, rng, tgt)
        stolen = 1.0 - (dec["privacy"] / SIGMA) ** 2

        row = {"rank": rank, "d1": d1, "d2": d2, "edge": float(edge),
               "gamma": gam, "decay_depth": depth, "visibility": vis,
               "attack_evades": evade, "attack_honest_fa": fa,
               "defence_power": fire(shp), "defence_honest_fa": fire(hon),
               "decoy_caught": fire(dec), "decoy_excess": float(np.mean(dec["excess"])),
               "honest_excess": float(np.mean(hon["excess"])),
               "decoy_stolen": stolen, "decoy_distortion": dec["distortion"],
               "penalty_to_deter_decoy": stolen / max(fire(dec) - ALPHA, 1e-9)}
        rows.append(row)
        print(f"\nrank {rank}  ({d1}x{d2})  gamma {gam:.3f}  depth {depth:.4f}  "
              f"visibility {vis:.2f}   [{time.time()-t0:.0f}s]")
        print(f"   attack evades {evade:.2f} (honest {fa:.2f})   "
              f"defence power {row['defence_power']:.2f} "
              f"(honest {row['defence_honest_fa']:.2f})")
        print(f"   decoy caught {row['decoy_caught']:.2f}   "
              f"excess {row['decoy_excess']:.2f} vs honest {row['honest_excess']:.2f}   "
              f"steals {100*stolen:.1f}%   distorts {dec['distortion']:.3f}   "
              f"fine to deter {row['penalty_to_deter_decoy']:.2f} V")

    with open(out_path, "w") as f:
        json.dump({"config": {"module": args.module, "sigma": SIGMA,
                              "cheat": CHEAT, "q": Q, "alpha": ALPHA,
                              "cal": args.cal, "test": args.test},
                   "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
