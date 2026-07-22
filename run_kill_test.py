#!/usr/bin/env python3
"""THE TWO-WEEK KILL TEST (CPU-only, runs in minutes).

Question: one client secretly injects 50% less noise than contracted.
Can the spectral audit detect this reliably from 5-10 rounds of LoRA updates?

PASS: power >= 0.9 at cheat factor 0.5 with <= 10 rounds, false alarms <= alpha.
FAIL: power stays low -> the detector idea is too weak -> stop the project.

Usage:
    python run_kill_test.py                # small/fast (256x256, rank 8)
    python run_kill_test.py --big          # closer to real LoRA (1024x1024, rank 16)
"""

import argparse
import json
import time

from fedgt import LoRAConfig, AuditConfig, SpectralAuditor

PASS_POWER = 0.90
CHEAT = 0.5          # the kill-test cheater: half the promised noise std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="1024x1024 rank-16 matrices (slower, more realistic)")
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="contracted noise std (default 0.05)")
    args = ap.parse_args()

    if args.big:
        lora = LoRAConfig(d_out=1024, d_in=1024, rank=16)
    else:
        lora = LoRAConfig(d_out=256, d_in=256, rank=8)

    audit = AuditConfig(alpha=0.05, n_null_sims=300)
    auditor = SpectralAuditor(lora, audit)

    print("=" * 68)
    print("FedGT KILL TEST  —  can we catch a 50% under-noiser?")
    print(f"matrix {lora.d_out}x{lora.d_in}, rank {lora.rank}, "
          f"sigma_contract={args.sigma}, alpha={audit.alpha}")
    print("=" * 68)

    results = {}
    verdict_rounds = None
    for n_rounds in (1, 3, 5, 10):
        t0 = time.time()
        power = auditor.detection_power(CHEAT, args.sigma, n_rounds, n_sims=200)
        false_alarm = auditor.detection_power(1.0, args.sigma, n_rounds, n_sims=200)
        dt = time.time() - t0
        results[n_rounds] = {"power": power, "false_alarm": false_alarm}
        ok = power >= PASS_POWER and false_alarm <= audit.alpha + 0.03
        if ok and verdict_rounds is None:
            verdict_rounds = n_rounds
        print(f"rounds={n_rounds:>2}  detection power={power:5.2f}  "
              f"false alarms={false_alarm:5.2f}  [{dt:4.1f}s]"
              f"  {'<-- PASS threshold met' if ok else ''}")

    print("-" * 68)
    if verdict_rounds is not None:
        print(f"VERDICT: PASS — 50% under-noising caught with power >= "
              f"{PASS_POWER} using {verdict_rounds} audited round(s). "
              f"Proceed with the project.")
    else:
        print("VERDICT: FAIL at these settings — detector too weak. "
              "Investigate (bigger window, canary hybrid) or stop.")

    with open("results/kill_test.json", "w") as f:
        json.dump({"config": {"shape": lora.shape, "rank": lora.rank,
                              "sigma": args.sigma},
                   "results": results,
                   "pass_rounds": verdict_rounds}, f, indent=2)
    print("Saved: results/kill_test.json")


if __name__ == "__main__":
    main()
