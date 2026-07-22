"""FedGT — Audit the Noise: verifiable local DP for federated LoRA fine-tuning.

CPU-only core: spectral noise auditing + audit-game mechanism + simulation.
"""

from .config import (LoRAConfig, PrivacyContract, AuditConfig,
                     MechanismConfig, SimulationConfig)
from .lora_update import (make_lora_update, add_gaussian_noise,
                          client_round_update, load_real_adapter)
from .spectral_audit import SpectralAuditor, estimate_sigma2, pooled_sigma2
from .mechanism import (check_incentive_compatibility, minimal_audit_prob,
                        minimal_penalty, best_response_dynamics, utility, gain)
from .simulation import run_simulation, Client, SimulationSummary

__version__ = "0.1.0"
