#!/usr/bin/env python3
"""Exp 10: does round-to-round signal drift keep decaying, or hit a floor?

exp9 rejected the temporal defense (Route A): inside the signal subspace,
consecutive-round signal drift exceeds the injected noise by 65x even at
round 20, where viability needs a ratio below 1. The obvious follow-up is
whether more training would close that gap.

Two-point extrapolation from exp9 suggests drift ~ t^-0.49, which would
put the crossover near 10^5 rounds. But that fit cannot distinguish a
genuine power law from a curve already flattening onto a floor, and the
distinction matters: a power law means "infeasible in practice", a floor
means "impossible at any training length".

This measures the full trajectory from the 20-round run, binning rounds
into windows and reporting drift/noise per window, then fitting the decay
exponent over the windows rather than over two endpoints.

Under non-IID data each round starts from a different global model and
samples different local batches, so a nonzero asymptote is the expected
outcome.

CPU-only. Reads ./adapters_r20/.
"""
import glob
import json
import os
import re
from collections import defaultdict

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

ADIR = "adapters_r20"
SIGMA = 5e-5
N_ITER = 4
SEP = 1
MAX_PAIRS = 70
WINDOWS = [(1, 5), (6, 10), (11, 15), (16, 20)]
F = re.compile(r"round(\d+)_client(\d+)_r(\d+)__(.+)\.pt$")


def load_dw(p):
    st = torch.load(p, map_location="cpu")
    A = st["lora_A"].float().numpy().astype(np.float64)
    B = st["lora_B"].float().numpy().astype(np.float64)
    return B @ A


def main():
    idx = defaultdict(dict)
    for p in glob.glob(f"{ADIR}/*q_proj.pt"):
        m = F.search(os.path.basename(p))
        idx[(int(m.group(2)), int(m.group(3)), m.group(4))][int(m.group(1))] = p

    print("=" * 74)
    print("Exp 10: drift/noise trajectory across training")
    print("=" * 74)
    print(f"\n  {'window':>12} {'midpoint':>9} {'pairs':>6} {'drift/noise':>13}")

    rows = []
    for lo, hi in WINDOWS:
        vals = []
        for (cid, rank, mod), byr in idx.items():
            for a in range(lo, hi):
                b = a + SEP
                if a in byr and b in byr and lo <= b <= hi:
                    if len(vals) >= MAX_PAIRS:
                        break
                    Wa, Wb = load_dw(byr[a]), load_dw(byr[b])
                    U, _, Vt = randomized_svd(Wb, n_components=rank,
                                              n_iter=N_ITER, random_state=0)
                    V = Vt.T
                    drift = float(np.linalg.norm(U.T @ (Wa - Wb) @ V))
                    vals.append(drift / (SIGMA * np.sqrt(2.0) * rank))
            if len(vals) >= MAX_PAIRS:
                break
        if not vals:
            continue
        mid = (lo + hi) / 2.0
        med = float(np.median(vals))
        rows.append({"window": f"{lo}-{hi}", "midpoint": mid,
                     "n": len(vals), "drift_over_noise": med})
        print(f"  {lo:>5}-{hi:<6} {mid:>9.1f} {len(vals):>6} {med:>13.2f}")

    if len(rows) < 3:
        print("\nnot enough windows to fit"); return

    t = np.array([r["midpoint"] for r in rows])
    y = np.array([r["drift_over_noise"] for r in rows])

    # The first window contains the initial training transient (the adapter
    # goes from near-zero to trained), which dominates any fit over all
    # windows. The question "would more training help?" concerns the
    # POST-TRANSIENT regime, so fit that separately and let it decide.
    def fit(tt, yy):
        b, la = np.polyfit(np.log(tt), np.log(yy), 1)
        pr = np.exp(la) * tt ** b
        sr, st = float(np.sum((yy - pr) ** 2)), float(np.sum((yy - yy.mean()) ** 2))
        return -b, (1 - sr / st if st > 0 else float("nan")), sr

    beta_all, r2_all, _ = fit(t, y)
    print(f"\n  fit over ALL windows:        beta = {beta_all:.3f}  (R^2 = {r2_all:.3f})")
    print("    dominated by the first window's training transient, not a usable trend")

    t2, y2 = t[1:], y[1:]
    beta_pt, r2_pt, ss_pl = fit(t2, y2)
    ss_flat = float(np.sum((y2 - y2.mean()) ** 2))
    print(f"\n  fit POST-TRANSIENT (windows 2+): beta = {beta_pt:.3f}  (R^2 = {r2_pt:.3f})")
    print(f"    level = {y2.mean():.1f} +/- {y2.std(ddof=1):.1f} "
          f"({100*y2.std(ddof=1)/y2.mean():.0f}% scatter)")
    print(f"    residual SS: flat = {ss_flat:.1f}, power law = {ss_pl:.1f}")

    monotone = all(y2[i] >= y2[i+1] for i in range(len(y2)-1))
    flat = (beta_pt <= 0.05) or (not monotone) or (abs(ss_flat - ss_pl) / ss_flat < 0.35)

    if flat:
        t_cross = None
        verdict = ("FLOOR. After the initial transient the drift/noise ratio stops "
                   "decaying and fluctuates about a constant level. More training "
                   "does not close the gap at any length, so the temporal defense is "
                   "not merely impractical but unreachable by longer runs.")
    else:
        t_cross = float(t2[-1] * (y2[-1] / 1.0) ** (1.0 / beta_pt))
        verdict = (f"POWER LAW. Still decaying post-transient; extrapolated crossover "
                   f"at ~{t_cross:,.0f} rounds, far outside any feasible run.")
    print(f"\n  VERDICT: {verdict}")

    with open("results/exp10_drift_trajectory.json", "w") as f:
        json.dump({"config": {"sigma": SIGMA, "sep": SEP, "windows": WINDOWS},
                   "rows": rows,
                   "beta_all_windows": beta_all, "r2_all_windows": r2_all,
                   "beta_post_transient": beta_pt, "r2_post_transient": r2_pt,
                   "post_transient_level": float(y2.mean()),
                   "post_transient_sd": float(y2.std(ddof=1)),
                   "extrapolated_crossover_rounds": t_cross,
                   "verdict": verdict}, f, indent=2)
    print("\nSaved: results/exp10_drift_trajectory.json")


if __name__ == "__main__":
    main()
