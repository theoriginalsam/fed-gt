#!/usr/bin/env python3
"""Experiment 1 — How accurately does the tail spectrum read the noise?

Sweeps LoRA rank and true noise level; reports relative error of sigma_hat.
CPU-only, ~1 minute.
"""

import json

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


import numpy as np

from fedgt import LoRAConfig, estimate_sigma2, client_round_update


def main():
    rng = np.random.default_rng(0)
    out = []
    for rank in (4, 8, 16, 32, 64):
        for sigma in (0.01, 0.05, 0.1):
            cfg = LoRAConfig(d_out=512, d_in=512, rank=rank)
            errs = []
            for _ in range(30):
                W = client_round_update(cfg, sigma, rng)
                s2 = estimate_sigma2(W, rank)
                errs.append(abs(np.sqrt(s2) - sigma) / sigma)
            rel = float(np.mean(errs))
            out.append({"rank": rank, "sigma": sigma, "mean_rel_err": rel})
            print(f"rank={rank:>2} sigma={sigma:.2f}  "
                  f"mean relative error of sigma_hat = {rel:6.3f}")

    with open("results/exp1_noise_estimation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/exp1_noise_estimation.json")


if __name__ == "__main__":
    main()
