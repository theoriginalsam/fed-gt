#!/usr/bin/env python3
"""Experiment 2 — Detection power vs. cheat size and audit window.

Produces the table behind Result 1 (detection power). CPU-only, a few minutes.
"""

import json

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


from fedgt import LoRAConfig, AuditConfig, SpectralAuditor


def main():
    lora = LoRAConfig(d_out=256, d_in=256, rank=8)
    auditor = SpectralAuditor(lora, AuditConfig(alpha=0.05, n_null_sims=300))
    sigma = 0.05

    rows = []
    print(f"{'cheat_factor':>12} | " +
          " | ".join(f"m={m:>2}" for m in (1, 3, 5, 10)))
    for cheat in (0.25, 0.5, 0.75, 0.9, 1.0):
        powers = {m: auditor.detection_power(cheat, sigma, m, n_sims=200)
                  for m in (1, 3, 5, 10)}
        rows.append({"cheat_factor": cheat, **{f"m{m}": p
                                               for m, p in powers.items()}})
        print(f"{cheat:>12.2f} | " +
              " | ".join(f"{powers[m]:4.2f}" for m in (1, 3, 5, 10)))
    print("(cheat_factor=1.0 row shows false-alarm rate; should be ~alpha)")

    with open("results/exp2_detection_power.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved: results/exp2_detection_power.json")


if __name__ == "__main__":
    main()
