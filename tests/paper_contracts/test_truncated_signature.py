"""Truncated signature contracts: level-0 semantics and level indexing (F1).

Tiny deterministic inputs, one behavior per test. These target the dense
algorithm (`signature_kern_first_order` / `_higher_order`) and the facade's
`truncation=0` path. PSD / normalize-once-vs-per_level / phi(0)-freezing live in
`tests/test_general_signature.py` and are NOT duplicated here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import ksig
ksig.set_default_device("cpu")
from ksig.algorithms import (signature_kern_first_order,
                             signature_kern_higher_order)
from ksig.generalized import GeneralSignatureKernel

from tests.paper_contracts._common import paths

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- F1: level-0 contract
@pytest.mark.parametrize("higher_order", [False, True])
def test_truncation_zero_is_level_zero_only(higher_order):
    """n_levels=0 returns ONLY the constant level (Contract 1), not [K0, K1]."""
    M = torch.tensor([[[[2.0]]]], dtype=torch.float64)  # single increment
    if higher_order:
        levels = signature_kern_higher_order(M, 0, order=1, return_levels=True)
    else:
        levels = signature_kern_first_order(M, 0, return_levels=True)
    assert levels.shape[0] == 1, "n_levels=0 must yield a single (level-0) slice"
    assert torch.allclose(levels[0], torch.ones_like(levels[0]))


def test_truncation_zero_summed_is_constant():
    """Non-return_levels path: n_levels=0 sums to the level-0 constant only."""
    M = torch.randn(4, 5, 5, dtype=torch.float64)
    K = signature_kern_first_order(M, 0, return_levels=False)
    assert torch.allclose(K, torch.ones_like(K))


def test_truncation_zero_matches_higher_n_level_zero():
    """The level-0 slice for n_levels=0 equals the level-0 slice for n_levels>=1."""
    M = torch.randn(4, 5, 5, dtype=torch.float64)
    l0 = signature_kern_first_order(M, 0, return_levels=True)[0]
    l_big = signature_kern_first_order(M, 3, return_levels=True)[0]
    assert torch.allclose(l0, l_big)


@pytest.mark.parametrize("fn,kw", [
    (signature_kern_first_order, {}),
    (signature_kern_higher_order, {"order": 1}),
])
def test_negative_n_levels_rejected(fn, kw):
    M = torch.randn(2, 4, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        fn(M, -1, **kw)


def test_facade_truncation_zero_is_constant_gram():
    """GSK(const, truncation=0) normalized Gram is all ones (constant baseline)."""
    X = paths()
    K = GeneralSignatureKernel(phi="const", truncation=0)(X)
    assert np.allclose(K, np.ones_like(K), atol=1e-5)


# ---------------------------------------------------------------- level indexing
def test_level_increment_difference_exact():
    """K^{<=N} - K^{<=N-1} == K_N: the summed path and the leveled path agree, so
    a level-extraction off-by-one cannot hide between the two code branches."""
    M = torch.randn(5, 6, 6, dtype=torch.float64)
    N = 4
    K_leN = signature_kern_first_order(M, N, return_levels=False)
    K_leNm1 = signature_kern_first_order(M, N - 1, return_levels=False)
    K_N = signature_kern_first_order(M, N, return_levels=True)[N]
    assert torch.allclose(K_leN - K_leNm1, K_N, atol=1e-9)
