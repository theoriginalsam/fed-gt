"""End-to-end multi-round simulation: clients -> updates -> random audits ->
flags, rewards, penalties. This is the harness Experiment 3/4 build on, and
later the place where real SPA training rounds replace `client_round_update`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig
from .lora_update import client_round_update
from .spectral_audit import SpectralAuditor


@dataclass
class Client:
    idx: int
    cheat_factor: float = 1.0           # 1.0 = honest
    updates: list = field(default_factory=list)   # recent updates (audit window)
    total_payoff: float = 0.0
    times_audited: int = 0
    times_flagged: int = 0

    @property
    def is_honest(self) -> bool:
        return self.cheat_factor >= 0.999


@dataclass
class SimulationSummary:
    n_rounds: int
    n_audits: int
    honest_flag_rate: float     # empirical false-alarm rate (want <= alpha)
    cheater_flag_rate: float    # empirical detection rate on cheaters
    payoffs_honest: float       # mean payoff of honest clients
    payoffs_cheaters: float     # mean payoff of cheating clients (want lower)


def run_simulation(cfg: SimulationConfig,
                   cheat_factors: list[float] | None = None) -> tuple[
                       list[Client], SimulationSummary]:
    """Run the audit game for n_rounds.

    cheat_factors: per-client cheat factor (len n_clients). Default: all honest
    except client 0 who under-noises at 0.5.
    """
    rng = np.random.default_rng(cfg.seed)
    if cheat_factors is None:
        cheat_factors = [0.5] + [1.0] * (cfg.n_clients - 1)
    assert len(cheat_factors) == cfg.n_clients

    clients = [Client(idx=i, cheat_factor=c)
               for i, c in enumerate(cheat_factors)]
    auditor = SpectralAuditor(cfg.lora, cfg.audit)
    sigma_c = cfg.contract.sigma_contract
    window = cfg.audit.rounds_per_audit

    n_audits = 0
    honest_audits = honest_flags = 0
    cheat_audits = cheat_flags = 0

    for _ in range(cfg.n_rounds):
        for cl in clients:
            W = client_round_update(cfg.lora, cl.cheat_factor * sigma_c, rng)
            cl.updates.append(W)
            if len(cl.updates) > window:
                cl.updates.pop(0)
            cl.total_payoff += cfg.mechanism.reward

            # Random audit once the window is full.
            if len(cl.updates) == window and rng.random() < (
                    cfg.mechanism.audit_prob / window):
                res = auditor.audit_client(cl.updates, sigma_c)
                n_audits += 1
                cl.times_audited += 1
                if cl.is_honest:
                    honest_audits += 1
                    honest_flags += res.flagged
                else:
                    cheat_audits += 1
                    cheat_flags += res.flagged
                if res.flagged:
                    cl.times_flagged += 1
                    cl.total_payoff -= cfg.mechanism.penalty

    honest_pay = [c.total_payoff for c in clients if c.is_honest]
    cheat_pay = [c.total_payoff for c in clients if not c.is_honest]
    summary = SimulationSummary(
        n_rounds=cfg.n_rounds,
        n_audits=n_audits,
        honest_flag_rate=honest_flags / honest_audits if honest_audits else 0.0,
        cheater_flag_rate=cheat_flags / cheat_audits if cheat_audits else 0.0,
        payoffs_honest=float(np.mean(honest_pay)) if honest_pay else 0.0,
        payoffs_cheaters=float(np.mean(cheat_pay)) if cheat_pay else 0.0,
    )
    return clients, summary
