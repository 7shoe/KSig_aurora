"""Low-level tensor-algebra primitives (``ksig/utils.py``) — the highest-leverage
tests (``TEST_PLAN.md`` Sec. 7.1).  Every kernel is built from these, and they are
where CuPy->torch axis/pad/dtype mismatches surface first.  Validated against
hand-written NumPy oracles (no golden needed); ``unit`` marker.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._backend import host, to_backend
from tests.harness import DEVICE, assert_close

ksig = pytest.importorskip("ksig")
utils = pytest.importorskip("ksig.utils")

pytestmark = pytest.mark.unit


def _rng(seed=0):
    return np.random.default_rng(seed)


# -----------------------------------------------------------------------------
# multi_cumsum — the single most port-sensitive primitive (pad ordering).
# -----------------------------------------------------------------------------
def _multi_cumsum_oracle(M, exclusive, axis):
    axis = [axis] if np.isscalar(axis) else list(axis)
    ndim = M.ndim
    axis = [ndim + a if a < 0 else a for a in axis]
    out = M
    if exclusive:
        sl = tuple(slice(-1) if a in axis else slice(None) for a in range(ndim))
        out = out[sl]
    for a in axis:
        out = np.cumsum(out, axis=a)
    if exclusive:
        pads = tuple((1, 0) if a in axis else (0, 0) for a in range(ndim))
        out = np.pad(out, pads)
    return out


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("axis", [-1, -2, [-2, -1], [-1, -2]])
@pytest.mark.parametrize("shape", [(2, 4, 5), (3, 1, 6), (2, 5, 1), (4,)])
def test_multi_cumsum(exclusive, axis, shape):
    if isinstance(axis, list) and len(shape) < 2:
        pytest.skip("multi-axis needs >=2 dims")
    M = _rng(0).standard_normal(shape)
    got = host(utils.multi_cumsum(to_backend(M, DEVICE), exclusive=exclusive,
                                  axis=axis))
    exp = _multi_cumsum_oracle(M, exclusive, axis)
    assert got.shape == exp.shape, (got.shape, exp.shape)
    assert_close(got, exp, family="DP_CUMSUM", device=DEVICE,
                 case_id=f"multi_cumsum_ex{exclusive}_ax{axis}_{shape}")


def test_multi_cumsum_exclusive_shape_and_zero():
    # exclusive prepends a zero and drops the last -> shape preserved, [..,0]==0.
    M = _rng(1).standard_normal((2, 4, 4))
    got = host(utils.multi_cumsum(to_backend(M, DEVICE), exclusive=True,
                                  axis=[-2, -1]))
    assert got.shape == M.shape
    assert np.allclose(got[:, 0, :], 0.0) and np.allclose(got[:, :, 0], 0.0)


# -----------------------------------------------------------------------------
# matrix_mult — all transpose combos vs einsum.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("tX,tY", [(False, False), (True, False),
                                   (False, True), (True, True)])
def test_matrix_mult(tX, tY):
    A = _rng(2).standard_normal((4, 5))
    B = _rng(3).standard_normal((4, 5) if (tX == tY) else (5, 4))
    # pick compatible shapes per transpose combo
    A = _rng(2).standard_normal((4, 5))
    B = _rng(3).standard_normal((5, 6))
    Xa = A.T if tX else A           # [k,m] or [m,k]
    Yb = B.T if tY else B           # [n,k] or [k,n]
    got = host(utils.matrix_mult(to_backend(Xa, DEVICE), to_backend(Yb, DEVICE),
                                 transpose_X=tX, transpose_Y=tY))
    lhs = Xa.T if tX else Xa
    rhs = Yb.T if tY else Yb
    exp = lhs @ rhs
    assert_close(got, exp, family="EXACT_ALGEBRA", device=DEVICE,
                 case_id=f"matrix_mult_tX{tX}_tY{tY}")


def test_matrix_mult_self_symmetry():
    A = _rng(4).standard_normal((6, 5))
    got = host(utils.matrix_mult(to_backend(A, DEVICE), transpose_Y=True))
    assert_close(got, A @ A.T, family="EXACT_ALGEBRA", device=DEVICE,
                 case_id="matrix_mult_self")
    assert np.allclose(got, got.T), "self product must be symmetric"


# -----------------------------------------------------------------------------
# squared_euclid_dist — non-negativity, zero diagonal, cancellation.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1.0, 1e-12, 1e6])
def test_squared_euclid_dist(scale):
    X = scale * _rng(5).standard_normal((5, 3))
    Y = scale * _rng(6).standard_normal((4, 3))
    got = host(utils.squared_euclid_dist(to_backend(X, DEVICE),
                                         to_backend(Y, DEVICE)))
    exp = (np.sum(X * X, 1)[:, None] + np.sum(Y * Y, 1)[None, :] - 2 * X @ Y.T)
    exp = np.maximum(exp, 0.0)
    assert (got >= -1e-9).all(), "distance must be clamped non-negative"
    assert_close(np.maximum(got, 0.0), exp, family="DP_CUMSUM", device=DEVICE,
                 case_id=f"sqdist_scale{scale}")


def test_squared_euclid_dist_zero_diag():
    X = _rng(7).standard_normal((5, 3))
    got = host(utils.squared_euclid_dist(to_backend(X, DEVICE)))
    assert np.allclose(np.diag(got), 0.0, atol=1e-9), "self-distance diag != 0"
    assert np.allclose(got, got.T, atol=1e-9), "self-distance not symmetric"


# -----------------------------------------------------------------------------
# outer_prod — shape [..., d1*d2], value vs np.outer reshaped.
# -----------------------------------------------------------------------------
def test_outer_prod():
    X = _rng(8).standard_normal((3, 4))
    Y = _rng(9).standard_normal((3, 5))
    got = host(utils.outer_prod(to_backend(X, DEVICE), to_backend(Y, DEVICE)))
    exp = (X[:, :, None] * Y[:, None, :]).reshape(3, 20)
    assert got.shape == (3, 20)
    assert_close(got, exp, family="EXACT_ALGEBRA", device=DEVICE,
                 case_id="outer_prod")
