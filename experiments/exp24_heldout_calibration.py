#!/usr/bin/env python3
"""Exp 24: do the thresholds survive calibration on adapters the auditor never tests?

Every earlier experiment fitted thresholds on honest clients simulated from
the same adapter pool it then tested, with fresh noise. That gives the
auditor access to the very clean adapters it audits, which a deployed auditor
never has. This calibrates at four levels of separation and scores every
threshold on one fixed test set:

  same       the test adapters themselves (the setup of earlier experiments)
  clients    other clients of the same training run
  run        another training run (different seed), same dataset
  dataset    a different dataset (GSM8K instead of Yelp)

Test set: FlexLoRA, Yelp, seed 44, rounds 6 and 12, half of the clients at
each rank (alternating by client id). The "clients" source uses the other half, so test and
calibration adapters never overlap except in "same".

Checks scored (sigma = 5e-5, where these checks are meant to work):
  tail      single-round tail estimate vs a client scaling all noise by 0.5
  decl      declaration energy ratio vs random and 15-degree-tilted lies
  rank      s_{r+1} excess vs a loud decoy
Also recorded: per-update relative error of the tail estimate, so that
average accuracy is not mistaken for a per-update guarantee.

CPU-only.
"""
import argparse, glob, json, re, sys, time, pathlib
import importlib.util
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("e22", HERE / "exp22_realistic_noise.py")
e22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e22)

SIGMA = 5e-5
ALPHA = 0.05
SOURCES = {
    "same":    ("yelp", 44, "odd"),
    "clients": ("yelp", 44, "even"),
    "run":     ("yelp", 42, "all"),
    "dataset": ("gsm8k", 42, "all"),
}
TEST = ("yelp", 44, "odd")
ADVS = ["naive", "random_decl", "tilt_15", "decoy_loud"]


def pool(dataset, seed, parity, rank, n, rng):
    fs = sorted(glob.glob(f"campaign/{dataset}_flexlora_s{seed}/adapters/round0[01][62]_*_r{rank}__*q_proj.pt"))
    if parity != "all":
        # ranks are tied to client ids, so split by alternating position among the
        # clients present at this rank rather than by id parity
        ids = sorted({int(re.search(r"client(\d+)", f).group(1)) for f in fs})
        keep = set(ids[1::2] if parity == "odd" else ids[0::2])
        fs = [f for f in fs if int(re.search(r"client(\d+)", f).group(1)) in keep]
    pick = rng.choice(len(fs), size=min(n, len(fs)), replace=False)
    out = []
    for i in pick:
        st = torch.load(fs[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out, len(fs)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 32])
    ap.add_argument("--cal", type=int, default=200)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--adv", type=int, default=100)
    ap.add_argument("--out", default="results/exp24_heldout_calibration.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print("=" * 104)
    print(f"Exp 24: held-out calibration  (sigma={SIGMA}, alpha={ALPHA}, cal={args.cal}, "
          f"test={args.test}, adv={args.adv})")
    print("=" * 104)
    rows = []
    for rank in args.ranks:
        t0 = time.time()
        tpool, ntest = pool(*TEST, rank, 80, rng)
        hon = np.array([e22.trial(tpool, rank, "honest", SIGMA, rng) for _ in range(args.test)])
        adv = {a: np.array([e22.trial(tpool, rank, a, SIGMA, rng) for _ in range(args.adv)]) for a in ADVS}
        err = np.sqrt(hon[:, 0]) / SIGMA - 1.0
        print(f"\nrank {rank}: test adapters available {ntest}; per-update tail-estimate error "
              f"median {100*np.median(err):+.3f}%, 95% range [{100*np.quantile(err,.025):+.3f}%, "
              f"{100*np.quantile(err,.975):+.3f}%]   [{time.time()-t0:.0f}s]")
        print(f"  {'calibrated on':<10} {'tail FA':>15} {'tail vs naive':>14} {'decl FA':>15} "
              f"{'decl vs rand':>12} {'decl vs tilt':>12} {'rank FA':>15} {'rank vs decoy':>13}")
        row = {"rank": rank, "per_update_err": {"median": float(np.median(err)),
               "q025": float(np.quantile(err, .025)), "q975": float(np.quantile(err, .975))},
               "sources": {}}
        for name, src in SOURCES.items():
            cpool, _ = pool(*src, rank, 80, rng)
            cal = np.array([e22.trial(cpool, rank, "honest", SIGMA, rng) for _ in range(args.cal)])
            t_tail, t_decl = np.quantile(cal[:, 0], ALPHA), np.quantile(cal[:, 1], ALPHA)
            t_rank = np.quantile(cal[:, 2], 1 - ALPHA)
            fa = {"tail": int(np.sum(hon[:, 0] < t_tail)), "decl": int(np.sum(hon[:, 1] < t_decl)),
                  "rank": int(np.sum(hon[:, 2] > t_rank))}
            pw = {"tail_naive": float(np.mean(adv["naive"][:, 0] < t_tail)),
                  "decl_random": float(np.mean(adv["random_decl"][:, 1] < t_decl)),
                  "decl_tilt": float(np.mean(adv["tilt_15"][:, 1] < t_decl)),
                  "rank_decoy": float(np.mean(adv["decoy_loud"][:, 2] > t_rank))}
            ci = {k: wilson(v, args.test) for k, v in fa.items()}
            row["sources"][name] = {"fa": {k: v / args.test for k, v in fa.items()},
                                    "fa_ci": ci, "power": pw}
            f = lambda k: f"{fa[k]/args.test:.2f} [{ci[k][0]:.2f},{ci[k][1]:.2f}]"
            print(f"  {name:<10} {f('tail'):>15} {pw['tail_naive']:>14.2f} {f('decl'):>15} "
                  f"{pw['decl_random']:>12.2f} {pw['decl_tilt']:>12.2f} {f('rank'):>15} "
                  f"{pw['rank_decoy']:>13.2f}", flush=True)
        rows.append(row)
        with open(args.out, "w") as fh:
            json.dump({"config": {"sigma": SIGMA, "alpha": ALPHA, "test": TEST, "sources": SOURCES},
                       "rows": rows}, fh, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
