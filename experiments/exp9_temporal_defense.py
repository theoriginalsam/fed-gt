#!/usr/bin/env python3
"""Exp 9 (Route A): can temporal structure catch the shaping adversary?

The confound that defeated exp7 and exp8 holds within a single snapshot:
inside the signal subspace, an adversary's missing noise is
indistinguishable from a slightly weaker signal. Across rounds it need not
hold. Noise is redrawn every round; the signal is not.

If a client's clean update is nearly static between two audited rounds,

    W_t - W_s  =  (DeltaW_t - DeltaW_s)  +  (N_t - N_s)
                        ^ drift, small at convergence     ^ variance 2 sigma^2

so projecting the difference onto the signal subspace estimates the noise
variance *there* -- the quantity single-shot auditing cannot reach.
Honest gives 2 sigma^2; an adversary injecting c*sigma gives 2 c^2 sigma^2,
a factor 1/c^2 gap (4x at c=0.5) rather than exp8's 8 percent.

The whole idea rests on one empirical question, which STAGE 1 answers
before any detection is attempted:

    is the round-to-round signal drift small compared to the noise?

If drift dominates, the difference is signal, not noise, and Route A is
dead regardless of how the test is built. Stage 1 reports the ratio
   ||P(DeltaW_t - DeltaW_s)||  /  ||P(N_t - N_s)||
inside the signal subspace, as a function of training progress and round
separation. Stage 2 runs the detection test only where stage 1 permits.

Requires multi-round adapters from the SAME client. Reads ./adapters_r20/.
CPU-only.
"""
import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

ADIR = "adapters_r20"
D = 3584
SIGMA = 5e-5
CHEAT = 0.5
ALPHA = 0.05
N_ITER = 4
FNAME = re.compile(r"round(\d+)_client(\d+)_r(\d+)__(.+)\.pt$")


def index_adapters(adir):
    """{(client, rank, module): {round: path}} for q_proj only."""
    idx = defaultdict(dict)
    for p in glob.glob(f"{adir}/*q_proj.pt"):
        m = FNAME.search(os.path.basename(p))
        if not m:
            continue
        rnd, cid, rank, mod = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        idx[(cid, rank, mod)][rnd] = p
    return idx


def load_dw(path):
    st = torch.load(path, map_location="cpu")
    A = st["lora_A"].float().numpy().astype(np.float64)
    B = st["lora_B"].float().numpy().astype(np.float64)
    return B @ A


def basis(W, r):
    U, _, Vt = randomized_svd(W, n_components=r, n_iter=N_ITER, random_state=0)
    return U, Vt.T


def proj_norm(M, U, V):
    """Frobenius norm of M restricted to the (U,V) subspace."""
    return float(np.linalg.norm(U.T @ M @ V))


def subspace_noise_var(M, U, V, r):
    """Per-direction variance of M inside the (U,V) subspace."""
    return proj_norm(M, U, V) ** 2 / (r * r)


def pairs_for(rounds, sep):
    rs = sorted(rounds)
    return [(a, b) for a in rs for b in rs if b - a == sep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=ADIR)
    ap.add_argument("--late-from", type=int, default=15,
                    help="rounds >= this are treated as converged")
    ap.add_argument("--early-to", type=int, default=6,
                    help="rounds <= this are treated as early training")
    ap.add_argument("--max-pairs", type=int, default=60)
    args = ap.parse_args()

    idx = index_adapters(args.adapters)
    if not idx:
        print(f"No q_proj adapters found in {args.adapters}/"); return
    rng = np.random.default_rng(0)

    print("=" * 88)
    print("Exp 9 (Route A): temporal separation of noise from signal")
    print("=" * 88)

    # ---------------- STAGE 1: is drift small enough? ----------------
    print("\nSTAGE 1  signal drift vs injected noise, inside the signal subspace")
    print("  ratio < 1 means the difference is noise-dominated and Route A is viable\n")
    print(f"  {'regime':<12} {'sep':>4} {'pairs':>6} {'drift/noise':>13} {'verdict':>12}")

    stage1 = []
    for label, lo, hi in (("early", 1, args.early_to),
                          ("converged", args.late_from, 999)):
        for sep in (1, 2, 3):
            ratios = []
            for (cid, rank, mod), byround in idx.items():
                rounds = [r for r in byround if lo <= r <= hi]
                for a, b in pairs_for(rounds, sep):
                    if len(ratios) >= args.max_pairs:
                        break
                    Wa, Wb = load_dw(byround[a]), load_dw(byround[b])
                    U, V = basis(Wb, rank)
                    drift = proj_norm(Wa - Wb, U, V)
                    # expected norm of (N_t - N_s) projected into an r x r block
                    noise = SIGMA * np.sqrt(2.0) * rank
                    ratios.append(drift / noise)
                if len(ratios) >= args.max_pairs:
                    break
            if not ratios:
                continue
            med = float(np.median(ratios))
            verdict = "VIABLE" if med < 1 else ("MARGINAL" if med < 3 else "DEAD")
            stage1.append({"regime": label, "sep": sep, "n": len(ratios),
                           "drift_over_noise": med, "verdict": verdict})
            print(f"  {label:<12} {sep:>4} {len(ratios):>6} {med:>13.3f} {verdict:>12}")

    viable = [s for s in stage1 if s["regime"] == "converged" and s["verdict"] != "DEAD"]
    if not viable:
        print("\nSTAGE 2 skipped: signal drift dominates the noise even at convergence,")
        print("so the round-to-round difference carries signal, not noise. Route A fails.")
        with open("results/exp9_temporal_defense.json", "w") as f:
            json.dump({"stage1": stage1, "stage2": None,
                       "conclusion": "drift dominates; Route A not viable"}, f, indent=2)
        print("\nSaved: results/exp9_temporal_defense.json")
        return

    # ---------------- STAGE 2: detection ----------------
    sep = min(v["sep"] for v in viable)
    print(f"\nSTAGE 2  detection via difference variance (converged, separation {sep})\n")
    print(f"  {'rank':>5} {'pairs':>6} {'honest var':>13} {'shaped var':>13} "
          f"{'ratio':>7} {'shaped flagged':>15}")

    stage2 = []
    byrank = defaultdict(list)
    for (cid, rank, mod), byround in idx.items():
        rounds = [r for r in byround if r >= args.late_from]
        for a, b in pairs_for(rounds, sep):
            byrank[rank].append((byround[a], byround[b], rank))

    for rank in sorted(byrank):
        items = byrank[rank][:args.max_pairs]
        if len(items) < 8:
            continue
        t0 = time.time()
        hon, shp = [], []
        for pa, pb, r in items:
            Wa, Wb = load_dw(pa), load_dw(pb)
            U, V = basis(Wb, r)
            # honest: full isotropic noise both rounds
            Na = rng.normal(0, SIGMA, size=Wa.shape)
            Nb = rng.normal(0, SIGMA, size=Wb.shape)
            hon.append(subspace_noise_var((Wa + Na) - (Wb + Nb), U, V, r))
            # shaped: only CHEAT*sigma inside the signal subspace
            hon_scale = CHEAT
            Ma = rng.normal(0, SIGMA * hon_scale, size=Wa.shape)
            Mb = rng.normal(0, SIGMA * hon_scale, size=Wb.shape)
            shp.append(subspace_noise_var((Wa + Ma) - (Wb + Mb), U, V, r))
        hon, shp = np.array(hon), np.array(shp)
        thr = float(np.quantile(hon, ALPHA))
        flag = float(np.mean(shp < thr))
        stage2.append({"rank": rank, "n": len(items),
                       "honest_var": float(np.mean(hon)),
                       "shaped_var": float(np.mean(shp)),
                       "ratio": float(np.mean(hon) / max(np.mean(shp), 1e-30)),
                       "shaped_flag_rate": flag})
        print(f"  {rank:>5} {len(items):>6} {np.mean(hon):>13.4e} {np.mean(shp):>13.4e} "
              f"{np.mean(hon)/max(np.mean(shp),1e-30):>7.2f} {flag:>15.2f}  "
              f"[{time.time()-t0:.0f}s]")

    with open("results/exp9_temporal_defense.json", "w") as f:
        json.dump({"stage1": stage1, "stage2": stage2,
                   "config": {"sigma": SIGMA, "cheat": CHEAT, "alpha": ALPHA,
                              "separation": sep, "late_from": args.late_from}}, f, indent=2)
    print("\nSaved: results/exp9_temporal_defense.json")
    print("Expected if the mechanism works: ratio near 1/c^2 = 4.0 and a high flag rate.")


if __name__ == "__main__":
    main()
