"""Generate (or load) LoRA updates and apply the DP noise a client injects.

CPU-only. Real adapters from the SPA / HetLoRA-M codebase can be plugged in
via `load_real_adapter` (torch is imported lazily and is optional).
"""

from __future__ import annotations

import numpy as np

from .config import LoRAConfig


def make_lora_update(cfg: LoRAConfig, rng: np.random.Generator) -> np.ndarray:
    """Simulate a client's clean LoRA update Delta_W = B @ A.

    B is (d_out, r), A is (r, d_in). Entries drawn so that the top-r
    singular values are O(signal_scale * sqrt(d)) -- i.e., clearly above
    the noise floor, matching what trained adapters look like.
    """
    r = cfg.rank
    B = rng.normal(0.0, 1.0, size=(cfg.d_out, r))
    A = rng.normal(0.0, 1.0, size=(r, cfg.d_in))
    W = B @ A
    # Normalize so the average squared entry is signal_scale**2.
    W *= cfg.signal_scale / (np.sqrt(r) + 1e-12)
    return W


def add_gaussian_noise(W: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """The Gaussian mechanism: what an honest client does with sigma_contract,
    and what a cheater does with sigma_actual < sigma_contract."""
    if sigma <= 0:
        return W.copy()
    return W + rng.normal(0.0, sigma, size=W.shape)


def client_round_update(cfg: LoRAConfig, sigma_actual: float,
                        rng: np.random.Generator) -> np.ndarray:
    """One round: fresh clean update + the noise the client actually injects."""
    return add_gaussian_noise(make_lora_update(cfg, rng), sigma_actual, rng)


def load_real_adapter(path: str) -> np.ndarray:
    """Load a real LoRA Delta_W from a saved adapter (SPA / HetLoRA-M).

    Accepts a .pt / .bin file containing tensors named '*lora_A*' and
    '*lora_B*' (PEFT convention) or a single 'delta_w'. Returns Delta_W
    as float64 numpy. Requires torch (CPU build is fine).
    """
    try:
        import torch  # lazy, optional
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading real adapters needs torch (CPU build is enough): "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from e

    state = torch.load(path, map_location="cpu")
    if "delta_w" in state:
        return state["delta_w"].double().numpy()

    a_key = next((k for k in state if "lora_A" in k), None)
    b_key = next((k for k in state if "lora_B" in k), None)
    if a_key is None or b_key is None:
        raise KeyError(
            f"Could not find lora_A / lora_B tensors in {path}. "
            f"Keys present: {list(state)[:10]}"
        )
    A = state[a_key].double().numpy()
    B = state[b_key].double().numpy()
    return B @ A
