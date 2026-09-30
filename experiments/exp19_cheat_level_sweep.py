#!/usr/bin/env python3
"""Exp 19: how small a shortfall can the challenge actually catch?

Every security experiment from exp11 to exp18 tests one cheat level, c = 0.5,
a client delivering half its promised noise. That is a greedy cheater. The
realistic one skims a little: c = 0.9 delivers ninety percent of the promised
noise and keeps the rest. Nothing so far says whether we can see that, and it
is the first thing a reviewer will ask.

The reduction this experiment rests on
--------------------------------------
The challenge only ever reads the core block

    C = U^T N V,     U, V orthonormal (D x r),  N iid Gaussian(0, sigma^2)

By rotational invariance of the iid Gaussian, C is exactly an r x r iid
Gaussian(0, sigma^2) matrix, whatever U and V happen to be. So for THIS
question the real adapter drops out of the mathematics entirely, and the
whole sweep collapses to r x r matrices. We do not assume that: the run
starts by checking it against real adapters and full 3584 x 3584 noise.

A second exact fact makes the sweep nearly free. A client at level c sends
c * C, so its statistic is exactly c^2 times the honest statistic computed on
the same draw. One bank of honest statistics therefore answers every c at
once, as a paired comparison.

What gets measured
------------------
  A. detection power at the protocol we have now (q = 20, one round), swept
     over c and rank. This locates the breaking point.
  B. the number of pooled rounds needed to reach power 0.9, per c and rank.
     This says what it costs to fix.
  C. whether spending more challenge directions helps instead, at c = 0.9.

There is a ceiling on C worth stating up front. The core block holds only
r^2 independent numbers, so once q passes r^2 the extra directions re-measure
what is already known and the power stops improving. Past that point pooling
rounds is the only lever. The saturated case has a closed form,

    separation  =  r * sqrt(T/2) * (1 - c^2)

which is printed next to the measurements as a check.

CPU-only. Table A/B/C need no adapters; the validation step reads ./adapters/.
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

D = 3584
RANKS = [4, 8, 16, 32]
CHEATS = [0.5, 0.7, 0.8, 0.9, 0.95]
Q = 20
Q_SWEEP = [20, 50, 100, 200, 400]
ALPHA = 0.05
TARGET_POWER = 0.9
MAX_ROUNDS = 400
BANK = 400_000
CHUNK = 20_000
Z = 1.6449 + 1.2816          # z(1-alpha) + z(power), for the closed form


# --------------------------------------------------------------- fast path
def bank_stats(rank, q, m, rng, chunk=None):
    """m independent single-round challenge statistics for an HONEST client.

    S = sum_j (a_j^T C b_j)^2 / sigma^2 with C ~ iid N(0,1) (sigma scales out).
    """
    if chunk is None:                      # keep each block near 20M floats
        chunk = int(np.clip(2e7 / (2 * rank * q + rank * rank), 1000, CHUNK))
    out = np.empty(m)
    done = 0
    while done < m:
        n = min(chunk, m - done)
        C = rng.standard_normal((n, rank, rank))
        A = rng.standard_normal((n, rank, q))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        B = rng.standard_normal((n, rank, q))
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        z = np.einsum("nrq,nrs,nsq->nq", A, C, B, optimize=True)
        out[done:done + n] = np.sum(z * z, axis=1)
        done += n
    return out


def pooled(stats, T):
    """Independent T-round pooled statistics, split into calibration and test."""
    k = (len(stats) // T) * T
    p = stats[:k].reshape(-1, T).sum(axis=1)
    h = len(p) // 2
    return p[:h], p[h:]                      # calibrate on one half, score on other


def power_at(stats, T, c, alpha=ALPHA):
    """Power against a level-c client. Its statistic is exactly c^2 x honest."""
    cal, test = pooled(stats, T)
    if len(test) < 200:
        return None, len(test)
    thr = float(np.quantile(cal, alpha))
    return float(np.mean(c * c * test < thr)), len(test)


def rounds_needed(stats, c, target=TARGET_POWER, cap=MAX_ROUNDS):
    """Smallest T reaching the target. Power rises with T, so bisect."""
    cap = min(cap, max(1, len(stats) // 400))   # T the bank can actually score
    p1, _ = power_at(stats, 1, c)
    if p1 is not None and p1 >= target:
        return 1, p1
    pc, _ = power_at(stats, cap, c)
    if pc is None or pc < target:
        return None, None
    lo, hi = 1, cap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        p, _ = power_at(stats, mid, c)
        if p is not None and p >= target:
            hi = mid
        else:
            lo = mid
    return hi, power_at(stats, hi, c)[0]


def theory_rounds(rank, c):
    """Saturated-q closed form: r * sqrt(T/2) * (1-c^2) = Z."""
    g = 1.0 - c * c
    return 2.0 * (Z / (rank * g)) ** 2


# ------------------------------------------------------- real-adapter check
def validate(adir, rank, cheats, q, n, rng):
    """Run the full pipeline on real adapters and full D x D noise, and compare
    the measured power with the fast path. If the reduction is right these
    agree to within Monte Carlo error."""
    files = sorted(glob.glob(f"{adir}/*_r{rank}__*q_proj.pt"))
    if not files:
        print(f"  (no rank-{rank} adapters under {adir}/, skipping validation)")
        return None
    pick = rng.choice(len(files), size=min(40, len(files)), replace=False)
    pool = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        pool.append((st["lora_A"].float().numpy().astype(np.float64),
                     st["lora_B"].float().numpy().astype(np.float64)))

    hon = np.empty(n)
    for t in range(n):
        A, B = pool[rng.integers(len(pool))]
        U, _ = np.linalg.qr(B)        # orthonormal basis of the column space
        V, _ = np.linalg.qr(A.T)      # orthonormal basis of the row space
        N = rng.normal(0.0, 1.0, size=(D, D))
        C = U.T @ N @ V
        a = rng.standard_normal((rank, q)); a /= np.linalg.norm(a, axis=0)
        b = rng.standard_normal((rank, q)); b /= np.linalg.norm(b, axis=0)
        z = np.einsum("iq,ij,jq->q", a, C, b)
        hon[t] = float(np.sum(z * z))

    fast = bank_stats(rank, q, 200_000, rng)
    print(f"  real adapters, rank {rank}, n={n}:  mean {hon.mean():8.3f}  "
          f"sd {hon.std():7.3f}")
    print(f"  fast path,     rank {rank}, n=200000: mean {fast.mean():8.3f}  "
          f"sd {fast.std():7.3f}")
    rows = []
    for c in cheats:
        thr = float(np.quantile(fast, ALPHA))
        pr = float(np.mean(c * c * hon < thr))
        pf = float(np.mean(c * c * fast < thr))
        se = float(np.sqrt(max(pf * (1 - pf), 1e-9) / n))
        ok = "ok" if abs(pr - pf) <= 3 * se + 0.02 else "MISMATCH"
        rows.append({"cheat": c, "power_real": pr, "power_fast": pf, "ok": ok})
        print(f"    c={c:<5} power on real adapters {pr:.3f}   "
              f"fast path {pf:.3f}   (+/-{3*se:.3f})  {ok}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--cheats", type=float, nargs="+", default=CHEATS)
    ap.add_argument("--bank", type=int, default=BANK)
    ap.add_argument("--validate-n", type=int, default=300)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--out", default="results/exp19_cheat_level_sweep.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 96)
    print(f"Exp 19: detection power vs how much noise the client withholds "
          f"(q={Q}, alpha={ALPHA})")
    print("=" * 96)
    print("c is the fraction of the promised noise the client actually delivers")
    print("in the private core block. c=1.0 is honest, c=0.5 halves it, c=0.9")
    print("skims ten percent.")

    res = {"config": {"d": D, "q": Q, "alpha": ALPHA, "bank": args.bank,
                      "cheats": args.cheats, "target_power": TARGET_POWER}}

    if not args.no_validate:
        print("\n" + "-" * 96)
        print("Step 0: does the r x r reduction actually hold on real adapters?")
        print("-" * 96)
        t0 = time.time()
        res["validation"] = validate(args.adapters, 8, [0.5, 0.9], Q,
                                     args.validate_n, rng)
        print(f"  [{time.time()-t0:.0f}s]")

    banks = {}
    for rank in args.ranks:
        t0 = time.time()
        banks[rank] = bank_stats(rank, Q, args.bank, rng)
        print(f"\n  built bank for rank {rank}  [{time.time()-t0:.0f}s]")

    # ---------------------------------------------------------------- A
    print("\n" + "-" * 96)
    print(f"A. Power with the protocol we have now: q={Q} directions, ONE round")
    print("-" * 96)
    print(f"  {'rank':>5} " + " ".join(f"{'c='+str(c):>9}" for c in args.cheats))
    tabA = {}
    for rank in args.ranks:
        row = [power_at(banks[rank], 1, c)[0] for c in args.cheats]
        tabA[rank] = row
        print(f"  {rank:>5} " + " ".join(f"{v:>9.3f}" for v in row))
    print(f"  honest clients are flagged at {ALPHA:.2f} by construction")
    res["table_a_power_q20_T1"] = {str(k): v for k, v in tabA.items()}

    # ---------------------------------------------------------------- B
    print("\n" + "-" * 96)
    print(f"B. Rounds that must be pooled to reach power {TARGET_POWER}, q={Q}")
    print("   (measured, with the saturated-q closed form in brackets)")
    print("-" * 96)
    print(f"  {'rank':>5} " + " ".join(f"{'c='+str(c):>14}" for c in args.cheats))
    tabB = {}
    for rank in args.ranks:
        cells, row = [], []
        for c in args.cheats:
            T, p = rounds_needed(banks[rank], c)
            th = theory_rounds(rank, c)
            row.append({"cheat": c, "rounds": T, "power": p, "theory": th})
            cells.append(f"{'>'+str(MAX_ROUNDS) if T is None else T:>7} "
                         f"[{th:>5.1f}]")
        tabB[rank] = row
        print(f"  {rank:>5} " + " ".join(f"{s:>14}" for s in cells))
    res["table_b_rounds_to_power"] = {str(k): v for k, v in tabB.items()}

    # ---------------------------------------------------------------- C
    print("\n" + "-" * 96)
    print("C. At c=0.9, does asking about MORE directions help instead? "
          "(one round)")
    print("-" * 96)
    print(f"  {'rank':>5} {'r^2':>6} " +
          " ".join(f"{'q='+str(q):>8}" for q in Q_SWEEP))
    tabC = {}
    for rank in args.ranks:
        row = []
        for q in Q_SWEEP:
            b = banks[rank] if q == Q else bank_stats(rank, q, 120_000, rng)
            row.append(power_at(b, 1, 0.9)[0])
        tabC[rank] = row
        print(f"  {rank:>5} {rank*rank:>6} " +
              " ".join(f"{v:>8.3f}" for v in row))
    print("  power stops improving once q passes r^2: the core block only holds")
    print("  r^2 independent numbers, so extra directions re-measure the same ones.")
    res["table_c_q_sweep_c090_T1"] = {str(k): v for k, v in tabC.items()}

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
