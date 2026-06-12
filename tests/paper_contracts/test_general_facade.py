"""General facade call-contract: delegated `diag` (F2) and dtype preservation (F3).

Plus two cheap invariant sentinels (Gram symmetry, diag-vs-full consistency) on a
couple of representative columns. The full normalization/PSD/convention matrix is
covered by `tests/test_general_signature.py`; not duplicated here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import ksig
ksig.set_default_device("cpu")
from ksig.generalized import GeneralSignatureKernel

from tests.paper_contracts._common import paths

pytestmark = pytest.mark.unit

# The two delegated "ksig"-kind engines plus a native generalized column.
KSIG_CONFIGS = [
    pytest.param(dict(phi="const", truncation=None), id="sig-PDE"),
    pytest.param(dict(phi="const", truncation=3, normalize="per_level"), id="sig-EXACT"),
]
ALL_CONFIGS = KSIG_CONFIGS + [
    pytest.param(dict(phi="const", truncation=3), id="sig-TRUNC"),
]


# ---------------------------------------------------------------- F2: delegated diag
@pytest.mark.parametrize("kwargs", KSIG_CONFIGS)
def test_delegated_diag_shape(kwargs):
    """diag=True reaches delegated ksig engines -> shape (n,) not (n, n)."""
    X = paths()
    out = GeneralSignatureKernel(**kwargs)(X, diag=True)
    assert np.asarray(out).shape == (X.shape[0],)


@pytest.mark.parametrize("kwargs", ALL_CONFIGS)
def test_diag_matches_full_diagonal(kwargs):
    """The diag path agrees with the diagonal of the full Gram (not just the
    shape) -- catches a `diag` branch that returns a cheap-but-wrong value."""
    X = paths(n=6)
    k = GeneralSignatureKernel(**kwargs)
    full = np.asarray(k(X, X))
    dg = np.asarray(k(X, diag=True))
    assert np.allclose(np.diag(full), dg, atol=1e-5)


# ---------------------------------------------------------------- F3: dtype preservation
def test_facade_preserves_float64():
    """float64 input -> float64 output (no silent float32 downcast in blocks)."""
    X = paths().astype(np.float64)
    out = GeneralSignatureKernel(phi="const", truncation=3)(X, return_on_gpu=True)
    assert out.dtype == torch.float64


def test_facade_float32_stays_float32():
    """float32 input is not silently promoted (dtype is preserved both ways)."""
    X = paths().astype(np.float32)
    out = GeneralSignatureKernel(phi="const", truncation=3)(
        torch.as_tensor(X), return_on_gpu=True)
    assert out.dtype == torch.float32


def test_fit_phi_runs_in_float64():
    """fit_phi works when dtype preservation (F3) carries float64 into the CKA
    loss -- the centering matrix/target must match the kernel dtype, not float32."""
    X = paths(n=12, seed=3).astype(np.float64)      # numpy default dtype
    y = (np.arange(12) % 2)
    k = GeneralSignatureKernel(phi="free", truncation=4, bw=1.0).fit_phi(X, y, steps=20)
    phi = k.phi_profile()
    assert np.isfinite(phi).all() and np.isclose(phi[0], 1.0, atol=1e-5)
    assert np.isfinite(np.asarray(k(X, X))).all()


# ---------------------------------------------------------------- invariant sentinels
@pytest.mark.parametrize("kwargs", ALL_CONFIGS)
def test_gram_is_symmetric(kwargs):
    """K(X, X) is symmetric for representative columns."""
    X = paths(n=7, seed=2)
    K = np.asarray(GeneralSignatureKernel(**kwargs)(X, X))
    assert np.allclose(K, K.T, atol=1e-5)
