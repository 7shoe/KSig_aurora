"""Deterministic input builders shared by the freeze (Stage 1) and compare
(Stage 2) stages.

CRITICAL CONTRACT
-----------------
Every input array is produced by a *NumPy* ``default_rng(seed)`` and returned as
a host ``np.ndarray``.  The freeze stage (legacy CuPy) and the compare stage
(future torch port) both call these *same* functions, so the bytes line up and
the golden oracle is reproducible on either stack.  Per ``TEST_PLAN.md`` Sec. 5.1 we
**never** rely on a backend's own RNG stream as ground truth, because CuPy
``RandomState`` and ``torch.Generator`` diverge for the same seed.

All builders return ``float64`` sequence arrays of shape ``[n, L, d]`` (the
``ndim == 3`` contract the public kernels expect) unless documented otherwise.

Memory note: keep ``n`` and ``L`` small here.  The signature / DP kernels
materialize an ``[n_X, n_Y, l_X, l_Y]`` tensor, i.e. ``O(n^2 L^2)`` — see
``fixtures/matrix.py`` for the size caps used during golden generation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian", "constant", "near_zero", "large", "ramp",
    "ragged_lengths", "ragged_with_nan", "flat_2d",
    "labelled_set",
]


def _rng(seed: int) -> np.random.Generator:
    """The single source of randomness for every fixture."""
    return np.random.default_rng(seed)


# -----------------------------------------------------------------------------
# Dense archetypes (the "what-shape-broke" axis).
# -----------------------------------------------------------------------------
def gaussian(n: int, L: int, d: int, seed: int = 0) -> np.ndarray:
    """Nominal correctness: standard normal sequences ``[n, L, d]``."""
    return _rng(seed).standard_normal((n, L, d)).astype(np.float64)


def constant(n: int, L: int, d: int, c: float = 1.0, seed: int = 0) -> np.ndarray:
    """Zero-variance sequences. Stresses normalization ``/0``, ``robust_sqrt``
    clamps and GAK degeneracy. ``seed`` is accepted for a uniform signature."""
    return np.full((n, L, d), float(c), dtype=np.float64)


def near_zero(n: int, L: int, d: int, scale: float = 1e-12,
              seed: int = 0) -> np.ndarray:
    """Tiny-magnitude inputs: ``_EPS`` clamps, ``robust_nonzero``, and
    catastrophic cancellation in ``squared_euclid_dist``."""
    return (scale * _rng(seed).standard_normal((n, L, d))).astype(np.float64)


def large(n: int, L: int, d: int, scale: float = 1e6,
          seed: int = 0) -> np.ndarray:
    """Large-magnitude inputs: ``exp`` overflow in RBF/GAK, float32 range."""
    return (scale * _rng(seed).standard_normal((n, L, d))).astype(np.float64)


def ramp(n: int, L: int, d: int, seed: int = 0) -> np.ndarray:
    """Smooth monotone ramps + small jitter — a well-conditioned non-random
    signal good for DP recurrences (no cancellation, finite, deterministic)."""
    base = np.linspace(0.0, 1.0, L)[None, :, None]
    chan = np.linspace(1.0, 2.0, d)[None, None, :]
    samp = (1.0 + np.arange(n))[:, None, None]
    jitter = 1e-3 * _rng(seed).standard_normal((n, L, d))
    return (samp * base * chan + jitter).astype(np.float64)


# -----------------------------------------------------------------------------
# Ragged / NaN / flat archetypes (preprocessing paths).
# -----------------------------------------------------------------------------
def ragged_lengths(lengths, d: int, seed: int = 0):
    """A python list of variable-length sequences ``[L_i, d]`` for the
    tabulation / interpolation path (incl. the ``L=1`` DTW edge)."""
    rng = _rng(seed)
    return [rng.standard_normal((int(L), d)).astype(np.float64) for L in lengths]


def ragged_with_nan(lengths, d: int, nan_frac: float = 0.1, seed: int = 0):
    """Variable-length sequences with scattered NaNs — only valid through the
    NaN-filtering preprocessing path (``SequenceTabulator``)."""
    rng = _rng(seed)
    out = []
    for L in lengths:
        a = rng.standard_normal((int(L), d)).astype(np.float64)
        mask = rng.random((int(L), d)) < nan_frac
        a[mask] = np.nan
        out.append(a)
    return out


def flat_2d(n: int, L: int, d: int, seed: int = 0) -> np.ndarray:
    """Flattened ``[n, L*d]`` array for the explicit-``n_features`` reshape path
    and for static kernels/features that take 2-D feature matrices."""
    return _rng(seed).standard_normal((n, L * d)).astype(np.float64)


# -----------------------------------------------------------------------------
# A small labelled set for the models layer (E2E_SCORE).
# -----------------------------------------------------------------------------
def labelled_set(n: int, L: int, d: int, n_classes: int = 2, seed: int = 0):
    """Two (or more) Gaussian blobs of sequences with integer labels — a fixed,
    separable classification problem for ``PrecomputedKernelSVC`` etc."""
    rng = _rng(seed)
    per = n // n_classes
    Xs, ys = [], []
    for c in range(n_classes):
        shift = (c - (n_classes - 1) / 2.0) * 3.0
        m = per if c < n_classes - 1 else n - per * (n_classes - 1)
        Xs.append(rng.standard_normal((m, L, d)).astype(np.float64) + shift)
        ys.append(np.full(m, c, dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
