"""Exact privacy calibration for the Gaussian mechanism.

The classical bound sigma = Delta sqrt(2 ln(1.25/delta)) / eps is only valid
for eps < 1. Every privacy level in this project is computed from the exact
privacy curve instead (Balle and Wang, ICML 2018), with mu = Delta / sigma:

    delta(eps) = Phi(mu/2 - eps/mu) - e^eps Phi(-mu/2 - eps/mu)

Sensitivity convention: client-level, replace-one. The client's update is
clipped to Frobenius norm C, and a neighbouring dataset may produce any other
clipped update, so Delta = 2C.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def delta_of_eps(eps, mu):
    return norm.cdf(mu / 2 - eps / mu) - np.exp(eps + norm.logcdf(-mu / 2 - eps / mu))


def eps_of_sigma(sigma, sensitivity, delta=1e-5):
    """Smallest eps such that Gaussian noise sigma gives (eps, delta)-DP."""
    mu = sensitivity / sigma
    if delta_of_eps(0.0, mu) <= delta:
        return 0.0
    hi = 1.0
    while delta_of_eps(hi, mu) > delta:
        hi *= 2.0
    return brentq(lambda e: delta_of_eps(e, mu) - delta, 0.0, hi)


def sigma_of_eps(eps, sensitivity, delta=1e-5):
    """Smallest Gaussian sigma giving (eps, delta)-DP."""
    f = lambda s: delta_of_eps(eps, sensitivity / s) - delta
    lo, hi = 1e-12 * sensitivity, sensitivity
    while f(hi) > 0:
        hi *= 2.0
    return brentq(f, lo, hi)
