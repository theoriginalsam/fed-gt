"""The audit game: payoffs and incentive-compatibility (Result 2 machinery).

Model (per audit window)
------------------------
A client chooses a cheat factor c in (0, 1], meaning it injects
sigma_actual = c * sigma_contract.

    utility(c) = reward + gain(c) - audit_prob * D(c) * penalty

where
    gain(c)  = gain_scale * (1 - c^2)   -- benefit of a cleaner update
                                           (0 when honest, max when c -> 0)
    D(c)     = detection power of the spectral audit at cheat factor c
               (estimated empirically by SpectralAuditor.detection_power)

Honesty (c = 1) is incentive-compatible iff for every c < 1:

    gain(c) <= audit_prob * D(c) * penalty            (IC condition)

This module checks IC on a grid of c, finds the minimal audit probability
that makes honesty a best response for a given penalty (and vice versa),
and runs best-response dynamics for a population of clients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MechanismConfig


def gain(c: np.ndarray | float, gain_scale: float) -> np.ndarray | float:
    """Client's benefit from under-noising by factor c (0 when honest)."""
    return gain_scale * (1.0 - np.asarray(c, dtype=float) ** 2)


def utility(c: np.ndarray | float, power_fn, mech: MechanismConfig):
    """Expected per-window utility of playing cheat factor c.

    power_fn(c) -> detection probability D(c) in [0, 1].
    """
    c_arr = np.atleast_1d(np.asarray(c, dtype=float))
    D = np.array([power_fn(ci) for ci in c_arr])
    u = mech.reward + gain(c_arr, mech.gain_scale) - mech.audit_prob * D * mech.penalty
    return u if u.size > 1 else float(u[0])


@dataclass
class ICReport:
    is_ic: bool                 # honesty is a best response on the grid
    worst_c: float              # most profitable deviation found
    worst_gain_minus_loss: float  # its net advantage over honesty (<=0 if IC)
    grid: np.ndarray
    utilities: np.ndarray


def check_incentive_compatibility(power_fn, mech: MechanismConfig,
                                  grid: np.ndarray | None = None) -> ICReport:
    """Is honest play (c = 1) a best response, over a grid of deviations?"""
    if grid is None:
        grid = np.linspace(0.05, 1.0, 20)
    u = np.asarray(utility(grid, power_fn, mech))
    u_honest = u[-1] if np.isclose(grid[-1], 1.0) else float(
        utility(1.0, power_fn, mech))
    net = u - u_honest
    i_worst = int(np.argmax(net[:-1])) if len(grid) > 1 else 0
    return ICReport(
        is_ic=bool(np.all(net[:-1] <= 1e-12)),
        worst_c=float(grid[i_worst]),
        worst_gain_minus_loss=float(net[i_worst]),
        grid=grid,
        utilities=u,
    )


def minimal_audit_prob(power_fn, mech: MechanismConfig,
                       grid: np.ndarray | None = None) -> float:
    """Smallest audit probability p making honesty IC at the given penalty.

    From the IC condition: p >= gain(c) / (D(c) * penalty) for all c.
    Note the correct IC condition: honest clients are also falsely flagged
    at rate D(1) ~ alpha, which taxes honesty. Cheating at c is unprofitable
    iff   gain(c) <= p * (D(c) - D(1)) * penalty,
    i.e., only the *excess* detection power over the false-alarm rate
    deters cheating. Returns np.inf if no p in (0, 1] suffices.
    """
    if grid is None:
        # Same deviation points as check_incentive_compatibility (minus c=1),
        # so the returned p provably covers the deviations the IC check tests.
        grid = np.linspace(0.05, 1.0, 20)[:-1]
    D = np.array([power_fn(c) for c in grid])
    D_honest = power_fn(1.0)
    excess = D - D_honest
    g = gain(grid, mech.gain_scale)
    with np.errstate(divide="ignore"):
        req = np.where(excess > 0, g / (excess * mech.penalty), np.inf)
    p_min = float(np.max(req))
    return p_min if p_min <= 1.0 else float("inf")


def minimal_penalty(power_fn, mech: MechanismConfig,
                    grid: np.ndarray | None = None) -> float:
    """Smallest penalty making honesty IC at the given audit probability.

    Uses the excess-power IC condition (see minimal_audit_prob).
    """
    if grid is None:
        grid = np.linspace(0.05, 1.0, 20)[:-1]
    D = np.array([power_fn(c) for c in grid])
    excess = D - power_fn(1.0)
    g = gain(grid, mech.gain_scale)
    with np.errstate(divide="ignore"):
        req = np.where(excess > 0, g / (excess * mech.audit_prob), np.inf)
    return float(np.max(req))


def best_response_dynamics(power_fn, mech: MechanismConfig, n_clients: int = 50,
                           n_iters: int = 30, seed: int = 7,
                           grid: np.ndarray | None = None) -> np.ndarray:
    """Population of clients repeatedly switching to their best cheat factor.

    Returns the (n_iters+1, n_clients) trajectory of chosen cheat factors.
    Under an IC mechanism this converges to all-ones (everyone honest).
    """
    if grid is None:
        grid = np.linspace(0.05, 1.0, 20)
    u_grid = np.asarray(utility(grid, power_fn, mech))
    best_c = float(grid[int(np.argmax(u_grid))])

    rng = np.random.default_rng(seed)
    traj = np.empty((n_iters + 1, n_clients))
    traj[0] = rng.uniform(0.05, 1.0, size=n_clients)  # random initial strategies
    for t in range(1, n_iters + 1):
        # Inertia: each client revises with prob 0.5 per iteration.
        revise = rng.random(n_clients) < 0.5
        traj[t] = np.where(revise, best_c, traj[t - 1])
    return traj
