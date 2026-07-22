#!/usr/bin/env python3
"""Exp 4: does the audit survive decaying (realistic) adapter spectra?

Real trained LoRA updates are still exactly rank r (Delta_W = B @ A), but
their singular values *within* the top r decay -- they are not all O(large)
like the flat synthetic generator assumes. With total signal energy held
fixed, a decay s_i ~ i^(-gamma) pushes the trailing signal directions down
toward the noise floor sigma*(sqrt(d1)+sqrt(d2)). Directions below that
floor never separate from the Marchenko-Pastur bulk, so their energy leaks
into the tail estimate, biases sigma_hat upward, and (dangerous direction)
makes an under-noising cheater look honest.

This experiment sweeps gamma and measures, at the kill-test operating point:
  - detection power at cheat factors 0.5 / 0.75 / 0.9
  - false-alarm rate on honest clients (cheat factor 1.0)
under two calibration regimes:
  - "flat":    threshold calibrated on the flat generator (deployment
               mismatch -- what happens if we assume the synthetic model)
  - "matched": threshold calibrated on the decaying generator with the
               same gamma (oracle calibration; requires knowing the decay)

Output: results/exp4_decay_robustness.json + a console table.
CPU-only, a few minutes.
"""

import argparse
import json
import time

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from fedgt import LoRAConfig, AuditConfig
from fedgt.lora_update import add_gaussian_noise, make_lora_update
from fedgt.spectral_audit import pooled_sigma2

GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
CHEATS = [0.5, 0.75, 0.9, 1.0]          # 1.0 == honest -> false-alarm rate
POOL_ROUNDS = [5, 10]
SIGMA_CONTRACT = 0.05
N_NULL_SIMS = 300
N_SIMS = 200

# --fast: first-look numbers on a weak machine (wider CIs: ~±0.04 on power)
FAST_POOL_ROUNDS = [5]
FAST_N_NULL_SIMS = 200
FAST_N_SIMS = 150


def make_decaying_update(cfg: LoRAConfig, gamma: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Rank-r update with singular values s_i ~ i^(-gamma), i = 1..r.

    Total energy matches the flat generator: sum s_i^2 = d1*d2*signal_scale^2,
    so gamma only redistributes energy across the top-r directions.
    """
    r = cfg.rank
    s = np.arange(1, r + 1, dtype=float) ** (-gamma)
    s *= np.sqrt(cfg.d_out * cfg.d_in) * cfg.signal_scale / np.linalg.norm(s)
    U, _ = np.linalg.qr(rng.normal(size=(cfg.d_out, r)))
    V, _ = np.linalg.qr(rng.normal(size=(cfg.d_in, r)))
    return (U * s) @ V.T


def decaying_round(cfg, gamma, sigma_actual, rng):
    return add_gaussian_noise(make_decaying_update(cfg, gamma, rng),
                              sigma_actual, rng)


def calibrate(cfg, audit, sigma_contract, n_rounds, gamma=None, seed=0):
    """alpha-quantile of pooled sigma2_hat under an honest client.

    gamma=None -> flat generator (the package's own calibration model);
    otherwise the matched decaying generator.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(N_NULL_SIMS)
    for i in range(N_NULL_SIMS):
        if gamma is None:
            ups = [add_gaussian_noise(make_lora_update(cfg, rng),
                                      sigma_contract, rng)
                   for _ in range(n_rounds)]
        else:
            ups = [decaying_round(cfg, gamma, sigma_contract, rng)
                   for _ in range(n_rounds)]
        stats[i] = pooled_sigma2(ups, cfg.rank, audit.rank_margin)
    return float(np.quantile(stats, audit.alpha))


def flag_rate(cfg, audit, gamma, cheat, sigma_contract, n_rounds,
              threshold, seed=123):
    """P(sigma2_hat < threshold) for a client injecting cheat*sigma_contract."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_SIMS):
        ups = [decaying_round(cfg, gamma, cheat * sigma_contract, rng)
               for _ in range(n_rounds)]
        hits += pooled_sigma2(ups, cfg.rank, audit.rank_margin) < threshold
    return hits / N_SIMS


def n_below_noise_floor(cfg, gamma, sigma):
    """How many of the top-r signal singular values sit below the MP edge."""
    r = cfg.rank
    s = np.arange(1, r + 1, dtype=float) ** (-gamma)
    s *= np.sqrt(cfg.d_out * cfg.d_in) * cfg.signal_scale / np.linalg.norm(s)
    floor = sigma * (np.sqrt(cfg.d_out) + np.sqrt(cfg.d_in))
    return int(np.sum(s < floor))


def main():
    global POOL_ROUNDS, N_NULL_SIMS, N_SIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="fewer sims / m=5 only, for a first look on a slow machine")
    args = ap.parse_args()
    if args.fast:
        POOL_ROUNDS = FAST_POOL_ROUNDS
        N_NULL_SIMS = FAST_N_NULL_SIMS
        N_SIMS = FAST_N_SIMS

    cfg = LoRAConfig(d_out=256, d_in=256, rank=8)
    audit = AuditConfig(alpha=0.05, n_null_sims=N_NULL_SIMS)

    print("=" * 76)
    print("Exp 4 — audit robustness to decaying adapter spectra "
          f"({cfg.d_out}x{cfg.d_in}, r={cfg.rank}, sigma={SIGMA_CONTRACT})")
    print("=" * 76)

    out = []
    for m in POOL_ROUNDS:
        thr_flat = calibrate(cfg, audit, SIGMA_CONTRACT, m, gamma=None)
        for gamma in GAMMAS:
            t0 = time.time()
            thr_matched = calibrate(cfg, audit, SIGMA_CONTRACT, m, gamma=gamma)
            row = {"pool_rounds": m, "gamma": gamma,
                   "n_sv_below_floor": n_below_noise_floor(cfg, gamma,
                                                           SIGMA_CONTRACT),
                   "flat": {}, "matched": {}}
            for cheat in CHEATS:
                key = "false_alarm" if cheat == 1.0 else f"power_c{cheat}"
                row["flat"][key] = flag_rate(cfg, audit, gamma, cheat,
                                             SIGMA_CONTRACT, m, thr_flat)
                row["matched"][key] = flag_rate(cfg, audit, gamma, cheat,
                                                SIGMA_CONTRACT, m, thr_matched)
            out.append(row)
            f, mt = row["flat"], row["matched"]
            print(f"m={m:>2} gamma={gamma:.1f} (sv<floor: "
                  f"{row['n_sv_below_floor']}/{cfg.rank})  "
                  f"flat: pow.5={f['power_c0.5']:.2f} fa={f['false_alarm']:.2f} | "
                  f"matched: pow.5={mt['power_c0.5']:.2f} "
                  f"pow.75={mt['power_c0.75']:.2f} "
                  f"pow.9={mt['power_c0.9']:.2f} fa={mt['false_alarm']:.2f}"
                  f"  [{time.time()-t0:.0f}s]")

    with open("results/exp4_decay_robustness.json", "w") as fp:
        json.dump({"config": {"shape": cfg.shape, "rank": cfg.rank,
                              "sigma": SIGMA_CONTRACT, "alpha": audit.alpha,
                              "gammas": GAMMAS, "cheats": CHEATS,
                              "pool_rounds": POOL_ROUNDS},
                   "rows": out}, fp, indent=2)
    print("Saved: results/exp4_decay_robustness.json")


if __name__ == "__main__":
    main()
