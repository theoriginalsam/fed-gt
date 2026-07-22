"""Configuration objects for FedGT (Audit-the-Noise).

Everything is a plain dataclass so configs are easy to log, serialize,
and sweep over in experiments. No GPU / torch dependency.
"""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Shape of the (simulated or real) LoRA update Delta_W = B @ A."""
    d_out: int = 256          # rows of Delta_W  (e.g., 4096 in real Qwen; small for CPU)
    d_in: int = 256           # cols of Delta_W
    rank: int = 8             # LoRA rank r
    signal_scale: float = 1.0 # magnitude of the top (signal) singular values

    @property
    def shape(self):
        return (self.d_out, self.d_in)


@dataclass
class PrivacyContract:
    """The noise level a client has agreed to inject (Gaussian mechanism).

    sigma_contract is the per-entry std-dev of Gaussian noise the client
    promised to add to its LoRA update before sending it to the server.
    """
    sigma_contract: float = 0.05


@dataclass
class AuditConfig:
    """Settings for the spectral audit test."""
    alpha: float = 0.05        # max false-alarm rate on honest clients
    rounds_per_audit: int = 5  # how many recent rounds are pooled per audit
    n_null_sims: int = 400     # Monte-Carlo draws to calibrate the null threshold
    rank_margin: int = 2       # extra singular values discarded beyond the known rank
    seed: int = 0


@dataclass
class MechanismConfig:
    """The audit game: rewards, penalties, audit probability."""
    reward: float = 1.0        # per-round reward for participating honestly
    penalty: float = 10.0      # fine when caught under-noising
    audit_prob: float = 0.2    # probability a client is audited each window
    # Client's benefit from cheating: gain(c) = gain_scale * (1 - c**2),
    # where c = sigma_actual / sigma_contract (c < 1 means under-noising).
    gain_scale: float = 1.0


@dataclass
class SimulationConfig:
    n_clients: int = 50
    n_rounds: int = 100
    seed: int = 42
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    contract: PrivacyContract = field(default_factory=PrivacyContract)
    audit: AuditConfig = field(default_factory=AuditConfig)
    mechanism: MechanismConfig = field(default_factory=MechanismConfig)
