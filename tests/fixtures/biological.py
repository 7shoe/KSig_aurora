"""Biological one-hot sequence builders + the cumulative-sum embedding.

Per ``TEST_PLAN.md`` Sec. 6/7.7: one-hot DNA/RNA/protein sequences, optionally passed
through a ``cumsum``-over-the-sequence-axis transform (the standard "bag of
prefixes" embedding that turns a categorical sequence into a monotone path the
signature kernel can consume).  Everything is deterministic NumPy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DNA", "RNA", "PROTEIN", "onehot", "onehot_dna", "onehot_protein",
           "cumsum_embed"]

DNA = "ACGT"
RNA = "ACGU"
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def onehot(n: int, L: int, alphabet: str = DNA, seed: int = 0) -> np.ndarray:
    """``[n, L, |alphabet|]`` one-hot float64 sequences over ``alphabet``."""
    d = len(alphabet)
    idx = _rng(seed).integers(0, d, size=(n, L))
    out = np.zeros((n, L, d), dtype=np.float64)
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


def onehot_dna(n: int, L: int, seed: int = 0) -> np.ndarray:
    return onehot(n, L, DNA, seed)


def onehot_protein(n: int, L: int, seed: int = 0) -> np.ndarray:
    return onehot(n, L, PROTEIN, seed)


def cumsum_embed(X: np.ndarray) -> np.ndarray:
    """Cumulative sum along the sequence axis: a categorical one-hot sequence
    becomes a monotone non-decreasing path in ``R^|alphabet|`` (running letter
    counts).  Shape preserved, values finite, non-decreasing along axis 1."""
    return np.cumsum(X, axis=1)
