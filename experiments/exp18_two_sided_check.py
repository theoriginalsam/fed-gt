#!/usr/bin/env python3
"""Exp 18: closing the decoy. A two-sided energy check, and a rank check.

Where we are
------------
exp13 built an energy check on the declared subspace:

    ratio = || U_dec^T W_sent V_dec ||_F^2  /  sum_{i<=r} s_i(W_sent)^2

and rejected when the ratio fell BELOW an honest-calibrated threshold. That
caught every lying declaration except one. The decoy plants a fake rank-1
component in a harmless direction, declares it in place of a real signal
direction, and starves the noise on the direction it left out. It supplies
exactly the energy the check is looking for, so it passes: flag rate 0.03 to
0.08 on Yelp, 0.00 to 0.30 across the full campaign.

The observation this experiment starts from
-------------------------------------------
The decoy does not merely pass. It passes too well. Its energy ratio sits
ABOVE the honest mean in all four ranks of exp13, in every decoy count of
exp14, and in all twenty cells of exp15. Never once below. The reason is
mechanical: the decoy's planted direction is exactly a top singular direction
of the matrix it sends, while an honest client declares the CLEAN update's
subspace, which the injected noise rotates slightly away from the noisy top-r.
Honest clients lose a little energy to that rotation. The decoy does not.

So the one-sided test is looking in the wrong tail. This experiment adds the
other tail.

The adaptive reply, and why it is the real test
-----------------------------------------------
A two-sided check is only worth reporting if it survives an adversary that
knows about it. The decoy has a free knob: it can tilt its DECLARED decoy
direction away from the planted one by an angle delta, which lowers the
captured energy continuously while leaving the matrix it sends untouched.
We give the adversary the honest median ratio as its target and bisect on
delta until it lands there. That is the strongest version of this adversary,
and it should defeat any test built on the energy ratio alone.

The check the tilt cannot dodge
-------------------------------
Tilting changes the DECLARATION. It does not change the MATRIX. A client that
declares rank r must send a matrix holding r spikes above the noise floor,
because B @ A is exactly rank r and the injected Gaussian fills the rest to
the Marchenko-Pastur edge. Any planted decoy is an extra spike:

    excess = s_{r+1}(W_sent) / ( sigma * (sqrt(d1) + sqrt(d2)) )

Honest clients sit at the edge, excess near 1. A decoy pushes s_{r+1} up to
whatever it planted. This is read off the submitted matrix, so re-declaring
cannot move it.

Adversaries
-----------
  honest            truthful declaration, full noise
  shaped            truthful declaration, starves it (exp12's, caught by challenge)
  decoy             exp13's: plants at s_1 strength, declares it, starves the
                    weakest true direction
  decoy_tuned       the same, but tilts the declaration to hit the honest median
  decoy_weak        plants at s_r strength instead. Cheaper: truncation keeps
                    almost the same matrix, so the distortion cost collapses
  decoy_weak_tuned  the cheap decoy, tilted to hit the honest median

Checks, all calibrated on honest clients at this rank
-----------------------------------------------------
  energy_lo   ratio below the alpha quantile           (exp13's check)
  energy_2s   ratio outside the alpha/2 quantiles      (new)
  rank_exc    excess above the 1-alpha quantile        (new)
  challenge   challenge statistic below alpha quantile (exp12's check)

Also reported per adversary: the noise actually delivered on the true signal
subspace (the privacy theft), and the post-truncation distortion, which is
what the theft costs the client in model quality.

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

D = 3584
EDGE = 2.0 * np.sqrt(D)          # sqrt(d1) + sqrt(d2) for the square case
RANKS = [4, 8, 16, 32]
SIGMA = 5e-5
CHEAT = 0.5
Q = 20
ALPHA = 0.05
N_TRIALS = 40
N_CALIB = 400
N_ITER = 4
ADVERSARIES = ["honest", "shaped", "decoy", "decoy_tuned",
               "decoy_weak", "decoy_weak_tuned"]
CHECKS = ["energy_lo", "energy_2s", "rank_exc", "challenge"]


def load_pool(rank, n, rng, adir):
    files = sorted(glob.glob(f"{adir}/*_r{rank}__*q_proj.pt"))
    if not files:
        raise SystemExit(f"no rank-{rank} q_proj adapters under {adir}/")
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out


def basis(A, B):
    """Exact SVD of W = B @ A via two thin QRs. W is exactly rank r, so this
    is both cheaper and more accurate than a randomized SVD on the D x D
    product."""
    Qb, Rb = np.linalg.qr(B)
    Qa, Ra = np.linalg.qr(A.T)
    u, s, vt = np.linalg.svd(Rb @ Ra.T)
    return Qb @ u, Qa @ vt.T, s


def orth_dirs(U, k, rng):
    """k orthonormal directions orthogonal to the columns of U."""
    X = rng.normal(size=(U.shape[0], k))
    X -= U @ (U.T @ X)
    Qm, _ = np.linalg.qr(X)
    return Qm[:, :k]


def starve(N, U, V, cols=None):
    """Scale the noise inside span(U) x span(V) down by CHEAT, on `cols` only."""
    C = U.T @ N @ V
    d = (np.full(C.shape[0], CHEAT) if cols is None
         else np.where(np.isin(np.arange(C.shape[0]), cols), CHEAT, 1.0))
    return N + U @ ((d[:, None] - 1.0) * C) @ V.T


def tilt_to_target(W_sent, Ut, Vt_, ud, vd, rank, top2, target, rng):
    """Tilt the declared decoy direction until the energy ratio equals `target`.

    The tilt is done in the (r+1)-dimensional span of the declared directions
    plus one escape direction on each side, so the search costs one projection
    of W_sent and then only small-matrix work.
    """
    Uk = np.concatenate([Ut[:, :rank - 1], ud], axis=1)
    Vk = np.concatenate([Vt_[:, :rank - 1], vd], axis=1)
    pu = orth_dirs(Uk, 1, rng)
    pv = orth_dirs(Vk, 1, rng)
    Uf = np.concatenate([Uk, pu], axis=1)          # D x (r+1)
    Vf = np.concatenate([Vk, pv], axis=1)
    M = Uf.T @ W_sent @ Vf                          # (r+1) x (r+1)

    def ratio_at(d):
        Su = np.zeros((rank + 1, rank))
        Su[:rank - 1, :rank - 1] = np.eye(rank - 1)
        Su[rank - 1, rank - 1] = np.cos(d)
        Su[rank, rank - 1] = np.sin(d)
        got = float(np.linalg.norm(Su.T @ M @ Su) ** 2)
        return got / top2

    lo, hi = 0.0, np.pi / 2
    if ratio_at(hi) > target:                       # cannot reach it; tilt fully
        d = hi
    else:
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if ratio_at(mid) > target:
                lo = mid
            else:
                hi = mid
        d = 0.5 * (lo + hi)
    u2 = ud * np.cos(d) + pu * np.sin(d)
    v2 = vd * np.cos(d) + pv * np.sin(d)
    Ud = np.concatenate([Ut[:, :rank - 1], u2], axis=1)
    Vd = np.concatenate([Vt_[:, :rank - 1], v2], axis=1)
    return Ud, Vd, float(d)


def head_svd(W_sent, rank):
    """Top r+1 singular triplets of the submitted matrix, computed once and
    reused for the energy ratio, the rank check and the truncation cost."""
    return randomized_svd(W_sent, n_components=rank + 1, n_iter=N_ITER,
                          random_state=None)


def challenge_stat(N, Ud, Vd, q, rng):
    C = Ud.T @ N @ Vd
    r = C.shape[0]
    A = rng.normal(size=(r, q)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.normal(size=(r, q)); B /= np.linalg.norm(B, axis=0, keepdims=True)
    z = np.einsum("ij,ij->j", A, C @ B)
    return float(np.sum(z ** 2) / SIGMA ** 2)


def true_privacy(N, Ut, Vt_):
    """Noise actually delivered inside the TRUE signal subspace."""
    return float(np.sqrt(np.mean((Ut.T @ N @ Vt_) ** 2)))


def distortion(U, s, Vt, W, rank):
    """What aggregation keeps, against the real update. Rank-r truncation."""
    Wt = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    return float(np.linalg.norm(Wt - W) / np.linalg.norm(W))


def submit(A, B, rank, adv, rng, fmult=None):
    """Build the matrix the client sends, before any declaration is made.

    Returns (W, Ut, Vt_, st, W_sent, N, ud, vd). ud/vd are the planted decoy
    directions, or None for the non-decoy strategies.
    """
    Ut, Vt_, st = basis(A, B)
    W = B @ A
    N = rng.normal(0.0, SIGMA, size=W.shape)

    if adv == "honest":
        return W, Ut, Vt_, st, W + N, N, None, None
    if adv == "shaped":
        N2 = starve(N, Ut, Vt_)
        return W, Ut, Vt_, st, W + N2, N2, None, None

    # every decoy variant plants one fake direction and starves the noise on
    # the weakest true direction, which it will then leave out of its
    # declaration. The weak variants plant at s_r instead of s_1, which costs
    # far less in truncation distortion.
    if fmult is not None:
        strength = fmult * st[rank - 1]       # sweep mode: f x the weakest signal
    else:
        strength = st[rank - 1] if adv.startswith("decoy_weak") else st[0]
    ud = orth_dirs(Ut, 1, rng)
    vd = orth_dirs(Vt_, 1, rng)
    N2 = starve(N, Ut, Vt_, cols=[rank - 1])
    W_sent = W + strength * (ud @ vd.T) + N2
    return W, Ut, Vt_, st, W_sent, N2, ud, vd


def declare(adv, rank, Ut, Vt_, ud, vd, W_sent, top2, target, rng):
    """What the client tells the auditor its signal subspace is."""
    if adv in ("honest", "shaped"):
        return Ut, Vt_, 0.0
    if not adv.endswith("_tuned"):
        return (np.concatenate([Ut[:, :rank - 1], ud], axis=1),
                np.concatenate([Vt_[:, :rank - 1], vd], axis=1), 0.0)
    Ud, Vd, d = tilt_to_target(W_sent, Ut, Vt_, ud, vd, rank, top2, target, rng)
    return Ud, Vd, float(np.rad2deg(d))


def run_adv(pool, rank, adv, n, rng, target, fmult=None):
    er, ex, ch, pv, ds, tl = [], [], [], [], [], []
    for _ in range(n):
        A, B = pool[rng.integers(len(pool))]
        W, Ut, Vt_, st, W_sent, N, ud, vd = submit(A, B, rank, adv, rng, fmult)

        U, s, Vt = head_svd(W_sent, rank)
        top2 = float(np.sum(s[:rank] ** 2))
        ex.append(float(s[rank] / (SIGMA * EDGE)))

        Ud, Vd, deg = declare(adv, rank, Ut, Vt_, ud, vd, W_sent, top2,
                              target, rng)
        er.append(float(np.linalg.norm(Ud.T @ W_sent @ Vd) ** 2) / max(top2, 1e-300))
        ch.append(challenge_stat(N, Ud, Vd, Q, rng))
        pv.append(true_privacy(N, Ut, Vt_))
        ds.append(distortion(U, s, Vt, W, rank))
        tl.append(deg)
    return {"ratio": np.array(er), "excess": np.array(ex), "chal": np.array(ch),
            "privacy": float(np.mean(pv)), "distortion": float(np.mean(ds)),
            "tilt_deg": float(np.mean(tl))}


SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def sweep_strength(pool, rank, rng, target, solo, b2, n):
    """Plant the decoy at f x the weakest true singular value and walk f from
    invisible to loud. The client always tilts its declaration as far as it
    needs to hit the honest median ratio, so it is playing its best reply at
    every strength. If the energy check fires at low f and the rank check
    fires at high f with nothing surviving in between, the decoy is closed
    rather than merely detected."""
    rows = []
    print(f"\n  strength sweep, rank {rank}   (decoy planted at f x s_r, "
          f"best-reply tilt at every f)")
    print(f"    {'f':>6} {'ratio':>9} {'excess':>8} {'tilt':>6} {'stolen':>7} "
          f"{'distort':>8} | {'energy_2s':>9} {'rank_exc':>9} {'CH+RK':>7} "
          f"{'EITHER':>7}")
    for f in SWEEP:
        d = run_adv(pool, rank, "decoy_sweep_tuned", n, rng, target, fmult=f)
        e2 = ((d["ratio"] < solo["e2s_lo"]) | (d["ratio"] > solo["e2s_hi"]))
        rk = d["excess"] > solo["rank_exc"]
        chrk = ((d["excess"] > b2["rank_exc"]) | (d["chal"] < b2["challenge"]))
        either = e2 | rk
        pct = 100.0 * d["privacy"] / SIGMA
        rows.append({"f": f, "energy_ratio": float(np.mean(d["ratio"])),
                     "excess": float(np.mean(d["excess"])),
                     "tilt_deg": d["tilt_deg"], "stolen_pct": 100.0 - pct,
                     "distortion": d["distortion"],
                     "energy_2s": float(np.mean(e2)), "rank_exc": float(np.mean(rk)),
                     "ch_rk": float(np.mean(chrk)), "either": float(np.mean(either))})
        r = rows[-1]
        print(f"    {f:>6.2f} "
              f"{r['energy_ratio']:>9.5f} {r['excess']:>8.3f} {r['tilt_deg']:>6.1f} "
              f"{r['stolen_pct']:>6.1f}% {r['distortion']:>8.3f} | "
              f"{r['energy_2s']:>9.2f} {r['rank_exc']:>9.2f} {r['ch_rk']:>7.2f} "
              f"{r['either']:>7.2f}")
    worst = min(rows, key=lambda r: r["either"])
    print(f"    weakest point for the auditor: f={worst['f']}, caught "
          f"{worst['either']:.2f}, steals {worst['stolen_pct']:.1f}%, "
          f"distorts {worst['distortion']:.3f}")
    return rows


def quant(a, q):
    return float(np.quantile(a, np.clip(q, 0.0, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--calib", type=int, default=N_CALIB)
    ap.add_argument("--sweep-trials", type=int, default=30)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default="results/exp18_two_sided_check.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 118)
    print(f"Exp 18: two-sided energy check and rank check  (sigma={SIGMA}, "
          f"c={CHEAT}, q={Q}, alpha={ALPHA}, calib={args.calib}, n={args.trials})")
    print("Thresholds are fitted on the calibration draws and scored on a "
          "separate honest sample, so the honest row is a genuine false alarm rate.")
    print("Union columns are Bonferroni corrected so the union, not each check, "
          "sits at alpha.")
    print("=" * 118)

    out = []
    for rank in args.ranks:
        t0 = time.time()
        pool = load_pool(rank, 60, rng, args.adapters)

        cal = run_adv(pool, rank, "honest", args.calib, rng, None)
        target = float(np.median(cal["ratio"]))       # what a tuned decoy aims at

        def cuts(a):
            """Thresholds for every check at per-check level `a`."""
            return {"energy_lo": quant(cal["ratio"], a),
                    "e2s_lo": quant(cal["ratio"], a / 2),
                    "e2s_hi": quant(cal["ratio"], 1 - a / 2),
                    "rank_exc": quant(cal["excess"], 1 - a),
                    "challenge": quant(cal["chal"], a)}

        solo = cuts(ALPHA)                 # each check alone, honest rate = alpha
        b3 = cuts(ALPHA / 3)               # union of three, honest rate = alpha
        b2 = cuts(ALPHA / 2)               # union of two, honest rate = alpha

        def flags(d):
            def e2s(c):
                return (d["ratio"] < c["e2s_lo"]) | (d["ratio"] > c["e2s_hi"])
            f = {"energy_lo": d["ratio"] < solo["energy_lo"],
                 "energy_2s": e2s(solo),
                 "rank_exc": d["excess"] > solo["rank_exc"],
                 "challenge": d["chal"] < solo["challenge"]}
            f["ALL3"] = (e2s(b3) | (d["excess"] > b3["rank_exc"]) |
                         (d["chal"] < b3["challenge"]))
            f["CH+RK"] = ((d["excess"] > b2["rank_exc"]) |
                          (d["chal"] < b2["challenge"]))
            return {k: float(np.mean(v)) for k, v in f.items()}

        row = {"rank": rank, "thresholds": {"solo": solo, "bonf3": b3, "bonf2": b2},
               "calib_median_ratio": target, "adversaries": {}}
        print(f"\nrank {rank}   honest ratio {target:.5f}   excess cut "
              f"{solo['rank_exc']:.3f}   [calib {time.time()-t0:.0f}s]")
        print(f"  {'adversary':<17} {'ratio':>9} {'excess':>8} {'tilt':>6} "
              f"{'privacy':>9} {'stolen':>7} {'distort':>8} | " +
              " ".join(f"{c:>9}" for c in CHECKS) + f" {'ALL3':>7} {'CH+RK':>7}")

        for adv in ADVERSARIES:
            d = run_adv(pool, rank, adv, args.trials, rng, target)
            fl = flags(d)
            pct = 100.0 * d["privacy"] / SIGMA
            row["adversaries"][adv] = {
                "energy_ratio": float(np.mean(d["ratio"])),
                "excess": float(np.mean(d["excess"])),
                "tilt_deg": d["tilt_deg"], "privacy": d["privacy"],
                "pct_of_contract": pct, "distortion": d["distortion"],
                "flags": fl}
            print(f"  {adv:<17} {np.mean(d['ratio']):>9.5f} "
                  f"{np.mean(d['excess']):>8.3f} {d['tilt_deg']:>6.1f} "
                  f"{d['privacy']:>9.2e} {100.0-pct:>6.1f}% {d['distortion']:>8.3f} | " +
                  " ".join(f"{fl[c]:>9.2f}" for c in CHECKS) +
                  f" {fl['ALL3']:>7.2f} {fl['CH+RK']:>7.2f}")
        if not args.no_sweep:
            row["sweep"] = sweep_strength(pool, rank, rng, target, solo, b2,
                                          args.sweep_trials)
        out.append(row)

    with open(args.out, "w") as f:
        json.dump({"config": {"d": D, "sigma": SIGMA, "cheat": CHEAT, "q": Q,
                              "alpha": ALPHA, "trials": args.trials,
                              "calib": args.calib, "adapters": args.adapters},
                   "rows": out}, f, indent=2)
    print(f"\nSaved: {args.out}")
    print("stolen = percent of the contracted noise withheld from the true signal")
    print("subspace. An adversary only wins if it steals something, keeps a union")
    print("column near alpha, and keeps distortion low.")


if __name__ == "__main__":
    main()
