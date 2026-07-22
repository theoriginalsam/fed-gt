"""CPU-only test suite. Run:  pytest tests/ -v   (takes ~1-2 minutes)."""

import numpy as np
import pytest

from fedgt import (LoRAConfig, AuditConfig, MechanismConfig, PrivacyContract,
                   SimulationConfig, SpectralAuditor, estimate_sigma2,
                   make_lora_update, add_gaussian_noise, client_round_update,
                   check_incentive_compatibility, minimal_audit_prob,
                   best_response_dynamics, run_simulation)

LORA = LoRAConfig(d_out=256, d_in=256, rank=8)
SIGMA = 0.05


# ---------------- lora_update ----------------

def test_clean_update_has_low_rank():
    rng = np.random.default_rng(0)
    W = make_lora_update(LORA, rng)
    s = np.linalg.svd(W, compute_uv=False)
    assert s[LORA.rank] < 1e-8 * s[0]          # exactly rank r


def test_noise_addition_changes_matrix():
    rng = np.random.default_rng(0)
    W = make_lora_update(LORA, rng)
    Wn = add_gaussian_noise(W, SIGMA, rng)
    assert not np.allclose(W, Wn)
    assert add_gaussian_noise(W, 0.0, rng) is not W  # copy on sigma=0


# ---------------- spectral_audit ----------------

def test_sigma_estimator_is_accurate():
    rng = np.random.default_rng(1)
    ests = [np.sqrt(estimate_sigma2(client_round_update(LORA, SIGMA, rng),
                                    LORA.rank))
            for _ in range(20)]
    assert abs(np.mean(ests) - SIGMA) / SIGMA < 0.05   # within 5%


def test_estimator_monotone_in_noise():
    rng = np.random.default_rng(2)
    e_small = estimate_sigma2(client_round_update(LORA, 0.02, rng), LORA.rank)
    e_big = estimate_sigma2(client_round_update(LORA, 0.10, rng), LORA.rank)
    assert e_big > e_small


def test_false_alarm_rate_close_to_alpha():
    audit = AuditConfig(alpha=0.05, n_null_sims=200)
    auditor = SpectralAuditor(LORA, audit)
    fa = auditor.detection_power(1.0, SIGMA, n_rounds=5, n_sims=200)
    assert fa <= 0.12          # honest clients rarely flagged


def test_detects_half_noise_cheater():
    audit = AuditConfig(alpha=0.05, n_null_sims=200)
    auditor = SpectralAuditor(LORA, audit)
    power = auditor.detection_power(0.5, SIGMA, n_rounds=5, n_sims=100)
    assert power >= 0.9        # the kill-test criterion


def test_audit_flags_actual_cheater_and_passes_honest():
    rng = np.random.default_rng(3)
    audit = AuditConfig(alpha=0.05, n_null_sims=200)
    auditor = SpectralAuditor(LORA, audit)
    cheater_ups = [client_round_update(LORA, 0.4 * SIGMA, rng)
                   for _ in range(5)]
    honest_ups = [client_round_update(LORA, SIGMA, rng) for _ in range(5)]
    assert auditor.audit_client(cheater_ups, SIGMA).flagged
    assert not auditor.audit_client(honest_ups, SIGMA).flagged


# ---------------- mechanism ----------------

def perfect_detector(c):
    """Idealized power function for fast unit tests."""
    return 0.05 if c >= 0.999 else min(1.0, 2.0 * (1.0 - c) + 0.1)


def test_ic_holds_with_strong_mechanism():
    mech = MechanismConfig(reward=1.0, penalty=50.0, audit_prob=0.5,
                           gain_scale=1.0)
    rep = check_incentive_compatibility(perfect_detector, mech)
    assert rep.is_ic


def test_ic_fails_with_weak_mechanism():
    mech = MechanismConfig(reward=1.0, penalty=0.1, audit_prob=0.01,
                           gain_scale=1.0)
    rep = check_incentive_compatibility(perfect_detector, mech)
    assert not rep.is_ic


def test_minimal_audit_prob_restores_ic():
    mech = MechanismConfig(reward=1.0, penalty=10.0, audit_prob=0.01,
                           gain_scale=1.0)
    p_min = minimal_audit_prob(perfect_detector, mech)
    assert 0.0 < p_min <= 1.0
    mech_fixed = MechanismConfig(reward=1.0, penalty=10.0,
                                 audit_prob=min(1.0, p_min * 1.05),
                                 gain_scale=1.0)
    assert check_incentive_compatibility(perfect_detector, mech_fixed).is_ic


def test_best_response_converges_to_honesty_under_ic():
    mech = MechanismConfig(reward=1.0, penalty=50.0, audit_prob=0.5,
                           gain_scale=1.0)
    traj = best_response_dynamics(perfect_detector, mech, n_clients=20,
                                  n_iters=40)
    assert np.mean(traj[-1] >= 0.999) > 0.9


# ---------------- simulation ----------------

def test_end_to_end_simulation_runs_and_punishes_cheating():
    cfg = SimulationConfig(
        n_clients=10, n_rounds=40, seed=1, lora=LORA,
        contract=PrivacyContract(sigma_contract=SIGMA),
        audit=AuditConfig(alpha=0.05, n_null_sims=150, rounds_per_audit=5),
        mechanism=MechanismConfig(reward=1.0, penalty=15.0, audit_prob=0.5))
    cheats = [0.4, 0.4] + [1.0] * 8
    clients, summary = run_simulation(cfg, cheats)
    assert summary.n_audits > 0
    assert summary.cheater_flag_rate > summary.honest_flag_rate
    assert summary.payoffs_cheaters < summary.payoffs_honest
