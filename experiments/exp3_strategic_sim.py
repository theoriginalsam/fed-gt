#!/usr/bin/env python3
"""Experiment 3 — The audit game with strategic clients.

(a) Check incentive compatibility of a candidate mechanism.
(b) Compute the minimal audit probability that keeps honesty rational.
(c) Best-response dynamics of 50 clients (do they converge to honesty?).
(d) End-to-end multi-round simulation: honest vs. cheating payoffs.

CPU-only, a few minutes.
"""

import json

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


import numpy as np

from fedgt import (LoRAConfig, AuditConfig, MechanismConfig, PrivacyContract,
                   SimulationConfig, SpectralAuditor,
                   check_incentive_compatibility, minimal_audit_prob,
                   best_response_dynamics, run_simulation)


def main():
    lora = LoRAConfig(d_out=256, d_in=256, rank=8)
    audit = AuditConfig(alpha=0.05, n_null_sims=300, rounds_per_audit=5)
    auditor = SpectralAuditor(lora, audit)
    sigma = 0.05
    window = audit.rounds_per_audit

    # Cache detection power on a grid once (the expensive part).
    grid = np.linspace(0.05, 1.0, 20)
    power_cache = {}

    def power_fn(c):
        c = round(float(c), 6)
        if c not in power_cache:
            power_cache[c] = auditor.detection_power(c, sigma, window,
                                                     n_sims=150)
        return power_cache[c]

    # (a) IC check for a candidate mechanism
    mech = MechanismConfig(reward=1.0, penalty=10.0, audit_prob=0.2,
                           gain_scale=1.0)
    ic = check_incentive_compatibility(power_fn, mech, grid)
    print(f"(a) IC with p={mech.audit_prob}, penalty={mech.penalty}: "
          f"{ic.is_ic}  (worst deviation c={ic.worst_c:.2f}, "
          f"net advantage {ic.worst_gain_minus_loss:+.3f})")

    # (b) minimal audit probability for this penalty
    p_min = minimal_audit_prob(power_fn, mech, grid[:-1])
    print(f"(b) minimal audit probability at penalty={mech.penalty}: "
          f"p_min = {p_min:.3f}")

    # (c) best-response dynamics under the (possibly corrected) mechanism
    mech_ic = MechanismConfig(reward=1.0, penalty=mech.penalty,
                              audit_prob=max(mech.audit_prob,
                                             min(1.0, 1.05 * p_min)),
                              gain_scale=1.0)
    traj = best_response_dynamics(power_fn, mech_ic, n_clients=50, n_iters=30)
    final_honest = float(np.mean(traj[-1] >= 0.999))
    print(f"(c) best-response dynamics with p={mech_ic.audit_prob:.3f}: "
          f"{100 * final_honest:.0f}% of 50 clients honest at convergence "
          f"(mean cheat factor {traj[-1].mean():.3f})")

    # (d) end-to-end simulation: 5 cheaters among 50 clients
    cheats = [0.5] * 5 + [1.0] * 45
    sim_cfg = SimulationConfig(
        n_clients=50, n_rounds=60, lora=lora, audit=audit,
        contract=PrivacyContract(sigma_contract=sigma), mechanism=mech_ic)
    _, summary = run_simulation(sim_cfg, cheats)
    print(f"(d) end-to-end 60 rounds, 5/50 cheaters: "
          f"audits={summary.n_audits}, "
          f"honest false-alarm rate={summary.honest_flag_rate:.3f}, "
          f"cheater detection rate={summary.cheater_flag_rate:.3f}")
    print(f"    mean payoff honest={summary.payoffs_honest:.1f}  "
          f"cheaters={summary.payoffs_cheaters:.1f}  "
          f"(cheating should NOT pay)")

    with open("results/exp3_strategic_sim.json", "w") as f:
        json.dump({
            "ic": {"is_ic": ic.is_ic, "worst_c": ic.worst_c,
                   "net_advantage": ic.worst_gain_minus_loss},
            "p_min": p_min,
            "brd_final_honest_frac": final_honest,
            "sim": summary.__dict__,
        }, f, indent=2)
    print("Saved: results/exp3_strategic_sim.json")


if __name__ == "__main__":
    main()
