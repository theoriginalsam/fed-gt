#!/usr/bin/env python3
"""Exp 20: what penalty makes honesty the best strategy, and is it affordable?

Until exp19 the incentive argument could not be written down, because we had
the probability of catching a cheater at exactly one cheat level. exp19 turned
that into a function of how much the client withholds and how many rounds are
pooled. This experiment puts that function into the mechanism and reads off
the penalty the mechanism needs.

The model
---------
A client delivers c * sigma of noise inside the private core block instead of
the promised sigma. Write the gain from that as the share of core noise
energy it did not have to absorb,

    G(c) = 1 - c^2 ,      G(1) = 0 honest,  G(0) = 1 removes it all

measured in units of V, the value to the client of removing the core noise
entirely. The server audits a client with probability rho and fines a flagged
client P. Detection is D(c, T) from exp19. An honest client is not safe: it is
flagged at the false alarm rate alpha, so it expects to pay alpha * rho * P.

Honesty is the best reply when deviating to any c < 1 is not worth it:

    V G(c) - rho P D(c,T)  <=  - rho P alpha        for every c < 1

which rearranges to the penalty the mechanism must be able to impose,

    P / V  >=  (1 / rho) * max_c [ G(c) / (D(c,T) - alpha) ]

Note what sits in the denominator. It is not the detection rate, it is the
detection rate ABOVE the false alarm rate. A test that flags honest and
cheating clients equally often deters nothing however often it fires.

The limit that matters
----------------------
As c approaches 1 both G(c) and D(c,T) - alpha go to zero, so the worst
deviation is not obviously the greediest one. Taking the ratio in the limit,
with separation s = sqrt(mT/2)(1-c^2) and m = min(q, r^2),

    P / V  ->  1 / ( phi(z_alpha) * sqrt(mT/2) )   =   13.7 / sqrt(mT)

So the required penalty falls as one over the square root of the rounds
pooled. The measured maximum is printed against this line.

Affordability
-------------
A penalty cannot be unlimited, because honest clients pay it too by mistake.
If participating is worth W to a client, it stays only while alpha rho P <= W.
Combining the two bounds, and noting rho cancels:

    W / V  >=  alpha * max_c [ G(c) / (D(c,T) - alpha) ]

That is the feasibility condition, and it does not depend on how often the
server audits. Auditing less often only makes the fine larger, not the scheme
impossible.

CPU-only. Reuses exp19's fast path. Reads exp18's results if present, to check
whether the decoy is a more attractive deviation than plain shaping.
"""
import argparse
import importlib.util
import json
import os
import time

import sys
import pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from scipy.stats import norm

spec = importlib.util.spec_from_file_location(
    "exp19", HERE / "exp19_cheat_level_sweep.py")
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)

RANKS = [4, 8, 16, 32]
Q = 20
ALPHA = 0.05
ROUNDS = [1, 2, 5, 10, 20, 50, 100]
# the grid must reach c=0, a client putting no noise at all in the core.
# Its gain is 1 and it is caught with certainty, so it pins a floor of
# 1/(1-alpha) on the fine that no amount of detection power removes.
CGRID = np.concatenate([np.arange(0.00, 0.50, 0.05),
                        np.arange(0.50, 0.96, 0.02),
                        np.arange(0.96, 0.9951, 0.005)])
BANK = 1_200_000


def gain(c):
    """Share of core noise energy the client withheld."""
    return 1.0 - c * c


def frontier(stats, T, m, alpha=ALPHA, cgrid=CGRID, k_se=4.0):
    """max_c G(c)/(D(c,T)-alpha), and the c that attains it.

    Near c=1 both G and D-alpha vanish, so the ratio tends to a finite limit
    but the empirical estimate of a sub-0.001 excess is pure Monte Carlo
    noise. We therefore maximise only over grid points whose excess is
    resolved to at least k_se standard errors, and take the analytic c->1
    limit for the rest of the range. The reported requirement is the larger
    of the two, so nothing is understated.
    """
    cal, test = exp19.pooled(stats, T)
    if len(test) < 500:
        return None
    n = len(test)
    thr = float(np.quantile(cal, alpha))
    best, best_c, curve, dropped = -np.inf, None, [], 0
    for c in cgrid:
        d = float(np.mean(c * c * test < thr))
        ex = d - alpha
        se = float(np.sqrt(max(d * (1 - d), 1e-12) / n))
        ok = ex > k_se * se
        r = gain(c) / ex if ex > 0 else np.inf
        curve.append({"c": float(c), "detect": d, "excess": ex, "se": se,
                      "resolved": bool(ok),
                      "ratio": float(r) if np.isfinite(r) else None})
        if ok and r > best:
            best, best_c = r, float(c)
        if not ok:
            dropped += 1
    lim = theory(m, T, alpha)
    if best < 0:                       # nothing resolved at all
        best, best_c = lim, None
    return {"measured_max": float(best), "c_star": best_c, "limit": float(lim),
            "requirement": float(max(best, lim)), "unresolved_points": dropped,
            "curve": curve}


def theory(m, T, alpha=ALPHA):
    """1 / (phi(z_alpha) * sqrt(mT/2))."""
    return 1.0 / (norm.pdf(norm.ppf(alpha)) * np.sqrt(m * T / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--rounds", type=int, nargs="+", default=ROUNDS)
    ap.add_argument("--q", type=int, default=Q)
    ap.add_argument("--bank", type=int, default=BANK)
    ap.add_argument("--rho", type=float, default=1.0,
                    help="probability a given client is audited")
    ap.add_argument("--out", default="results/exp20_incentive_frontier.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 104)
    print(f"Exp 20: the penalty that makes honesty the best strategy  "
          f"(q={args.q}, alpha={ALPHA}, audit prob rho={args.rho})")
    print("=" * 104)
    print("P/V is the fine needed, in units of what removing ALL the core noise")
    print("is worth. 'lim c->1' is the closed form for a near-honest cheater; once")
    print("enough rounds are pooled the binding deviation moves to the greedy end")
    print("instead, and the measured maximum is the number that matters.")
    print("is worth to the client. c* is the client's most attractive cheat level.")
    print("W/V is how much participation must be worth for an honest client to stay.")

    res = {"config": {"q": args.q, "alpha": ALPHA, "rho": args.rho,
                      "bank": args.bank, "rounds": args.rounds}, "ranks": {}}

    for rank in args.ranks:
        t0 = time.time()
        stats = exp19.bank_stats(rank, args.q, args.bank, rng)
        m = min(args.q, rank * rank)
        print(f"\nrank {rank}   m = min(q, r^2) = {m}   [{time.time()-t0:.0f}s]")
        print(f"  {'rounds':>7} {'P/V needed':>12} {'lim c->1':>9} {'binds at':>9} "
              f"{'D there':>8} {'W/V needed':>12}")
        rows = []
        for T in args.rounds:
            f = frontier(stats, T, m)
            if f is None:
                print(f"  {T:>7}   (bank too small to score this many rounds)")
                continue
            req = f["requirement"]
            pv = req / args.rho
            wv = ALPHA * req
            cs = f["c_star"]
            d_at = (next(x["detect"] for x in f["curve"] if x["c"] == cs)
                    if cs is not None else float("nan"))
            binds = "c->1" if f["limit"] >= f["measured_max"] else f"c={cs:.3f}"
            rows.append({"rounds": T, "penalty_over_V": pv,
                         "participation_over_V": wv, "binds_at": binds, **f})
            print(f"  {T:>7} {pv:>12.2f} {f['limit']:>9.2f} "
                  f"{binds:>9} {d_at:>8.3f} {wv:>12.3f}")
        res["ranks"][str(rank)] = rows

    # ---- is the decoy a better deviation than plain shaping? ----
    dec = "results/exp18_two_sided_check.json"
    if os.path.exists(dec):
        print("\n" + "-" * 104)
        print("Is the decoy a better deviation than plain shaping? "
              "(from exp18, one round)")
        print("-" * 104)
        d18 = json.load(open(dec))
        print(f"  {'rank':>5} {'strategy':>18} {'stolen':>8} {'caught':>8} "
              f"{'distortion':>11} {'P/V to deter':>13}")
        out = []
        for row in d18["rows"]:
            rk = row["rank"]
            for adv, a in row["adversaries"].items():
                if adv == "honest":
                    continue
                g = max(a["pct_of_contract"], 0.0)
                g = 1.0 - (g / 100.0) ** 2          # gain in the same G units
                ex = a["flags"]["CH+RK"] - ALPHA
                pv = g / ex if ex > 1e-4 else float("inf")
                out.append({"rank": rk, "strategy": adv, "gain": g,
                            "caught": a["flags"]["CH+RK"],
                            "distortion": a["distortion"],
                            "penalty_over_V": pv})
                print(f"  {rk:>5} {adv:>18} {g:>8.3f} "
                      f"{a['flags']['CH+RK']:>8.2f} {a['distortion']:>11.3f} "
                      f"{pv:>13.2f}" if np.isfinite(pv) else
                      f"  {rk:>5} {adv:>18} {g:>8.3f} "
                      f"{a['flags']['CH+RK']:>8.2f} {a['distortion']:>11.3f} "
                      f"{'n/a':>13}")
        res["decoy_comparison"] = out
        print("  distortion is what the deviation costs the client in model quality,")
        print("  and it is paid whether or not the client is caught.")
    else:
        print(f"\n  ({dec} not present yet, skipping the decoy comparison)")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
