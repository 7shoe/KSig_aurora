"""Closed-form NumPy oracles for the static kernels.

Parametrizations match ``ksig.static.kernels`` exactly (defaults: ``scale=1``,
``gamma=1``, ``degree=3``, ``bandwidth=1``, ``alpha=1``).

IMPORTANT — Matern12 / Matern32
-------------------------------
The legacy ``ksig.utils.euclid_dist(self, X, Y=None)`` carries a spurious
``self`` parameter, so ``Matern12Kernel`` and ``Matern32Kernel`` are **broken**
in the CuPy code (the args shift; a single-arg call raises).  These oracles
compute the *correct* Matern values from the textbook formulas; golden for those
two kernels is therefore sourced from the oracle (not the broken legacy code),
and the discrepancy is recorded so the torch port is held to the right answer.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _flatten(X: np.ndarray) -> np.ndarray:
    """Static kernels merge all but the first axis (ksig ``_validate_data``)."""
    return X.reshape(X.shape[0], -1) if X.ndim > 2 else X


def _sq_dist(X, Y):
    X = _flatten(X)
    Y = _flatten(X if Y is None else Y)
    d2 = (np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :]
          - 2.0 * X @ Y.T)
    return np.maximum(d2, 0.0)   # clamp cancellation, matches robust behavior


def _dist(X, Y):
    return np.sqrt(_sq_dist(X, Y))


def linear(X, Y=None, scale=1.0):
    X = _flatten(X); Y = _flatten(X if Y is None else Y)
    return scale * (X @ Y.T)


def polynomial(X, Y=None, degree=3.0, gamma=1.0, scale=1.0):
    X = _flatten(X); Y = _flatten(X if Y is None else Y)
    return scale * np.power(X @ Y.T + gamma, degree)


def rbf(X, Y=None, bandwidth=1.0):
    return np.exp(-_sq_dist(X, Y) / max(2.0 * bandwidth ** 2, _EPS))


def matern12(X, Y=None, bandwidth=1.0):
    r = _dist(X, Y) / max(bandwidth, _EPS)
    return np.exp(-r)


def matern32(X, Y=None, bandwidth=1.0):
    r = np.sqrt(3.0) * _dist(X, Y) / max(bandwidth, _EPS)
    return (1.0 + r) * np.exp(-r)


def matern52(X, Y=None, bandwidth=1.0):
    d2s = 5.0 * _sq_dist(X, Y) / max(bandwidth ** 2, _EPS)
    ds = np.sqrt(d2s)
    return (1.0 + ds + d2s / 3.0) * np.exp(-ds)


def rational_quadratic(X, Y=None, bandwidth=1.0, alpha=1.0):
    d2s = _sq_dist(X, Y) / max(2.0 * alpha * bandwidth ** 2, _EPS)
    return np.power(1.0 + d2s, -alpha)


# Map ksig class names -> (oracle fn, whether legacy is known-broken).
KERNELS = {
    "LinearKernel": (linear, False),
    "PolynomialKernel": (polynomial, False),
    "RBFKernel": (rbf, False),
    "Matern12Kernel": (matern12, True),    # legacy euclid_dist bug
    "Matern32Kernel": (matern32, True),    # legacy euclid_dist bug
    "Matern52Kernel": (matern52, False),
    "RationalQuadraticKernel": (rational_quadratic, False),
}


def gram(kernel_name, X, Y=None, **kwargs):
    fn, _ = KERNELS[kernel_name]
    return fn(X, Y, **kwargs)


def diag(kernel_name, X, **kwargs):
    """Diagonal of the self-Gram (matches ``_Kdiag``)."""
    fn, _ = KERNELS[kernel_name]
    G = fn(X, None, **kwargs)
    return np.diag(G).copy()
