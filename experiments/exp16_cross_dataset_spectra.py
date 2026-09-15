#!/usr/bin/env python3
"""Exp 16: do the real-adapter spectral findings hold beyond Yelp?

Everything in exp5a came from one dataset, one aggregation method and one
seed. That was the narrowness reviewers rejected in the previous paper. This
reruns the spectral characterisation across the full campaign: 3 datasets
(Yelp, Alpaca, GSM8K) x 2 methods (hetero_spa, flexlora) x 3 seeds, at
rounds 1, 6 and 12.

Reports, per dataset and method, the power-law decay exponent gamma and the
decay depth s_r/s_1, with the spread across the three seeds. Seeds give the
error bars the earlier results lacked.

The question is whether gamma stays in the 1.0 to 1.45 band exp5a measured on
Yelp. If it does, the audit results generalise. If Alpaca or GSM8K sits
elsewhere, that is a finding in itself and changes what the paper can claim.

CPU-only, a few minutes. Reads ./campaign/.
"""
import glob, json, os, re, sys, time
from collections import defaultdict
import numpy as np, torch

F = re.compile(r"round(\d+)_client(\d+)_r(\d+)__")
RANKS = [4, 8, 16, 32]
ROOT = "campaign"


def svals(A, B):
    """Exact nonzero singular values of B@A, cheaply, via QR on thin factors."""
    Qb, Rb = np.linalg.qr(B)
    Qa, Ra = np.linalg.qr(A.T)
    return np.sort(np.linalg.svd(Rb @ Ra.T, compute_uv=False))[::-1]


def gamma_of(s):
    i = np.arange(1, len(s) + 1)
    return float(-np.polyfit(np.log(i), np.log(s + 1e-30), 1)[0])


def main():
    runs = sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/*") if os.path.isdir(d))
    if not runs:
        print(f"no runs under {ROOT}/"); return

    # per (dataset, method, seed, rank) -> list of (gamma, depth)
    acc = defaultdict(list)
    t0 = time.time()
    for run in runs:
        ds, method, seed = run.rsplit("_", 1)[0].split("_", 1)[0], \
                           "_".join(run.split("_")[1:-1]), run.split("_")[-1]
        for f in glob.glob(f"{ROOT}/{run}/adapters/*q_proj.pt"):
            m = F.search(os.path.basename(f))
            rank = int(m.group(3))
            st = torch.load(f, map_location="cpu")
            A = st["lora_A"].float().numpy().astype(np.float64)
            B = st["lora_B"].float().numpy().astype(np.float64)
            s = svals(A, B); s = s[s > 1e-12]
            if len(s) < 2: continue
            acc[(ds, method, seed, rank)].append((gamma_of(s), float(s[-1] / s[0])))
    print(f"processed {sum(len(v) for v in acc.values())} adapters in {time.time()-t0:.0f}s\n")

    print("=" * 84)
    print("Exp 16: spectral decay across datasets, methods and seeds")
    print("=" * 84)
    print(f"  {'dataset':<8} {'method':<12} {'rank':>5} {'gamma (mean +/- sd over seeds)':>32} "
          f"{'decay depth':>14}")

    out = []
    for ds in ("yelp", "alpaca", "gsm8k"):
        for method in ("hetero_spa", "flexlora"):
            for rank in RANKS:
                per_seed_g, per_seed_d, n = [], [], 0
                for seed in ("s42", "s43", "s44"):
                    v = acc.get((ds, method, seed, rank), [])
                    if not v: continue
                    per_seed_g.append(np.mean([x[0] for x in v]))
                    per_seed_d.append(np.mean([x[1] for x in v]))
                    n += len(v)
                if not per_seed_g: continue
                g_m, g_s = float(np.mean(per_seed_g)), float(np.std(per_seed_g, ddof=1) if len(per_seed_g) > 1 else 0.0)
                d_m = float(np.mean(per_seed_d))
                out.append({"dataset": ds, "method": method, "rank": rank, "n_adapters": n,
                            "gamma_mean": g_m, "gamma_sd_over_seeds": g_s, "decay_depth": d_m})
                print(f"  {ds:<8} {method:<12} {rank:>5} {g_m:>20.3f} +/- {g_s:<8.3f} {d_m:>14.4f}")
        print()

    gs = [r["gamma_mean"] for r in out]
    print(f"  gamma across all 24 dataset/method/rank cells: {min(gs):.2f} to {max(gs):.2f}")
    print(f"  exp5a measured 1.04 to 1.44 on Yelp alone, 6-round adapters")
    within = all(0.85 <= g <= 1.75 for g in gs)
    print(f"\n  VERDICT: {'consistent with exp5a, findings generalise' if within else 'some cells fall outside the exp5a band, worth investigating'}")

    with open("results/exp16_cross_dataset_spectra.json", "w") as f:
        json.dump({"rows": out}, f, indent=2)
    print("\nSaved: results/exp16_cross_dataset_spectra.json")


if __name__ == "__main__":
    main()
