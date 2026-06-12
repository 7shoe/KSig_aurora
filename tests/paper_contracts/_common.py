"""Shared tiny deterministic fixtures for the paper-contract layer.

Kept deliberately small (n<=16, L<=12) so every test is fast and the random
draws are reproducible.
"""
from __future__ import annotations

import numpy as np


def paths(n=6, L=8, d=3, seed=0, scale=1.0, dtype=np.float64):
    """`n` smooth, unit-ish bounded random-walk paths of shape [n, L, d]."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((n, L, d)), 1) / np.sqrt(L)
    X = X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), 1e-6, None)
    return (X * scale).astype(dtype)


def constant_paths(n=5, L=8, d=3, seed=0, dtype=np.float64):
    """`n` paths that are CONSTANT along time (zero increments) but differ across
    the batch -- the signature collapses to level 0, so every kernel entry is 1."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 1, d))
    return np.repeat(pts, L, axis=1).astype(dtype)


def labels(n, seed=1):
    return np.random.default_rng(seed).integers(0, 2, n)
