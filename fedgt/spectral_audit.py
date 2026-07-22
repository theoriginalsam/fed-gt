"""The detector: estimate injected noise from the tail spectrum of a LoRA update.

Key facts used
--------------
1. A clean LoRA update has rank r, so singular values s_{r+1}, s_{r+2}, ...
   of the *noisy* update come (almost) entirely from the injected Gaussian
   noise.
2. For a d1 x d2 matrix of iid N(0, sigma^2) entries, the total spectral
   energy is E[||E||_F^2] = d1 * d2 * sigma^2, and the energy the top-r
   directions of the noise can absorb is roughly r * (d1 + d2 - r) * sigma^2
   (the parameter count of a rank-r matrix). The remaining "tail energy"
   therefore estimates sigma^2 with known degrees of freedom.

Estimator
---------
    sigma_hat^2 = sum_{i > r + margin} s_i^2 / dof,
    dof = d1*d2 - (r+margin)*(d1 + d2 - (r+margin))

Audit test
----------
H0: the client injected (at least) the contracted noise  (honest)
H1: the client injected less                              (under-noising)

We pool `rounds_per_audit` recent updates, average sigma_hat^2, and reject H0
when the average falls below a threshold calibrated by Monte-Carlo simulation
of the null (honest) distribution -- giving an exact-by-construction
false-alarm rate alpha. No asymptotic approximations that could quietly fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AuditConfig, LoRAConfig
from .lora_update import client_round_update


def tail_energy_dof(d1: int, d2: int, k: int) -> int:
    """Degrees of freedom of the tail beyond the top-k singular values."""
    return d1 * d2 - k * (d1 + d2 - k)


def estimate_sigma2(W_noisy: np.ndarray, rank: int, margin: int = 2) -> float:
    """Point estimate of the injected per-entry noise variance sigma^2."""
    d1, d2 = W_noisy.shape
    k = min(rank + margin, min(d1, d2) - 1)
    s = np.linalg.svd(W_noisy, compute_uv=False)
    tail = s[k:]
    dof = tail_energy_dof(d1, d2, k)
    return float(np.sum(tail ** 2) / dof)


def pooled_sigma2(updates: list[np.ndarray], rank: int, margin: int = 2) -> float:
    """Average sigma^2 estimate over several rounds of one client's updates."""
    return float(np.mean([estimate_sigma2(W, rank, margin) for W in updates]))


@dataclass
class AuditResult:
    sigma2_hat: float          # pooled estimate of injected noise variance
    sigma2_contract: float     # what was promised
    threshold: float           # calibrated rejection threshold
    flagged: bool              # True -> client accused of under-noising
    n_rounds: int


class SpectralAuditor:
    """Calibrated audit test with false-alarm rate <= alpha by construction."""

    def __init__(self, lora: LoRAConfig, audit: AuditConfig):
        self.lora = lora
        self.audit = audit
        self._threshold_cache: dict[tuple[float, int], float] = {}

    # ---------------- calibration ----------------
    def _null_threshold(self, sigma_contract: float, n_rounds: int) -> float:
        """alpha-quantile of the pooled estimator under an honest client.

        Simulated with fresh signal each round (signal leakage into the tail
        is therefore *included* in the calibration -- the test stays honest
        even though the estimator is slightly biased upward by the signal).
        """
        key = (round(sigma_contract, 10), n_rounds)
        if key in self._threshold_cache:
            return self._threshold_cache[key]

        rng = np.random.default_rng(self.audit.seed)
        stats = np.empty(self.audit.n_null_sims)
        for i in range(self.audit.n_null_sims):
            ups = [client_round_update(self.lora, sigma_contract, rng)
                   for _ in range(n_rounds)]
            stats[i] = pooled_sigma2(ups, self.lora.rank, self.audit.rank_margin)
        thr = float(np.quantile(stats, self.audit.alpha))
        self._threshold_cache[key] = thr
        return thr

    # ---------------- the audit ----------------
    def audit_client(self, updates: list[np.ndarray],
                     sigma_contract: float) -> AuditResult:
        """Run the audit on a client's recent updates."""
        n = len(updates)
        est = pooled_sigma2(updates, self.lora.rank, self.audit.rank_margin)
        thr = self._null_threshold(sigma_contract, n)
        return AuditResult(
            sigma2_hat=est,
            sigma2_contract=sigma_contract ** 2,
            threshold=thr,
            flagged=bool(est < thr),
            n_rounds=n,
        )

    # ---------------- power analysis ----------------
    def detection_power(self, cheat_factor: float, sigma_contract: float,
                        n_rounds: int, n_sims: int = 200,
                        seed: int = 123) -> float:
        """P(flag) for a client injecting sigma_actual = cheat_factor * contract.

        cheat_factor = 1.0 -> honest (power should be ~alpha).
        cheat_factor = 0.5 -> injects half the promised noise std.
        """
        rng = np.random.default_rng(seed)
        sigma_actual = cheat_factor * sigma_contract
        thr = self._null_threshold(sigma_contract, n_rounds)
        hits = 0
        for _ in range(n_sims):
            ups = [client_round_update(self.lora, sigma_actual, rng)
                   for _ in range(n_rounds)]
            est = pooled_sigma2(ups, self.lora.rank, self.audit.rank_margin)
            hits += est < thr
        return hits / n_sims
