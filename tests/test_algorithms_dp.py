"""The three dynamic-programming kernels vs the brute-force NumPy DP oracles
(``TEST_PLAN.md`` Sec. 7.5).  This is the highest-risk module: the Numba ``@cuda.jit``
kernels become torch wavefronts / SYCL in the port, so they are validated at the
algorithm level (feeding an identical ``M``/``D``) against hardware-independent
recurrences — no golden ``.npz`` needed, works on any backend.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness import DEVICE, IS_LEGACY, assert_close
from tests.oracles import dp_numpy as dp
from tests.oracles import signature_numpy as sn

ksig = pytest.importorskip("ksig")
alg = pytest.importorskip("ksig.algorithms")


def _host(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _to_backend(a):
    """Move a numpy array onto whatever array type ksig expects."""
    ag = ksig.utils.ArrayOnGPU
    try:
        import cupy as cp
        if ag is cp.ndarray:
            return cp.asarray(a)
    except Exception:
        pass
    try:
        import torch
        if ag is torch.Tensor:
            return torch.as_tensor(a, device=DEVICE)
    except Exception:
        pass
    return a


def _rng(seed=0):
    return np.random.default_rng(seed)


# -----------------------------------------------------------------------------
# SigPDE
# -----------------------------------------------------------------------------
# The legacy Numba SigPDE kernel launches min(l_X,l_Y) threads; after a
# `difference` on an L=1 input that is 0 threads -> cuLaunchKernel fails. The
# torch wavefront (TORCH_PORT 4.2) handles L=1 and must pass. Mirror the
# Matern policy: xfail on legacy, require on the port.
def _xfail_legacy_sigpde_edge(lX, lY, difference):
    if IS_LEGACY and difference and min(lX, lY) <= 1:
        pytest.xfail("legacy SigPDE Numba kernel cannot launch on L=1 "
                     "(0-thread grid); the torch wavefront must handle it.")


@pytest.mark.parametrize("difference", [True, False])
@pytest.mark.parametrize("shape", [(3, 2, 6, 6), (2, 2, 1, 1), (4, 3, 5, 7)])
def test_sigpde_vs_oracle(difference, shape):
    nX, nY, lX, lY = shape
    _xfail_legacy_sigpde_edge(lX, lY, difference)
    X = _rng(1).standard_normal((nX, lX, 3))
    Y = _rng(2).standard_normal((nY, lY, 3))
    M = sn.static_gram(X, Y, "rbf")
    got = _host(alg.signature_kern_pde(_to_backend(M), difference=difference))
    exp = dp.sig_pde_gram(M, difference=difference)
    assert_close(got, exp, family="DP_CUMSUM", device=DEVICE,
                 case_id=f"sigpde_diff{difference}_{shape}")


def test_sigpde_boundary_L1():
    # L=1 base case: difference makes M empty -> K == 1 everywhere.
    _xfail_legacy_sigpde_edge(1, 1, True)
    X = _rng(3).standard_normal((3, 1, 2))
    M = sn.static_gram(X, X, "rbf")
    got = _host(alg.signature_kern_pde(_to_backend(M), difference=True))
    assert np.allclose(got, 1.0), got


# -----------------------------------------------------------------------------
# GAK (log-space)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(3, 2, 6, 6), (2, 2, 1, 1), (4, 3, 5, 7)])
def test_gak_log_vs_oracle(shape):
    nX, nY, lX, lY = shape
    X = _rng(4).standard_normal((nX, lX, 3))
    Y = _rng(5).standard_normal((nY, lY, 3))
    M = sn.static_gram(X, Y, "rbf")
    got = _host(alg.global_align_kern_log(_to_backend(M)))
    exp = dp.gak_log_gram(M)
    assert_close(got, exp, family="DP_CUMSUM", device=DEVICE,
                 case_id=f"gak_{shape}")
    assert np.isfinite(got).all(), "GAK leaked -inf/nan"


def test_gak_corner_seed():
    # (0,0): up=left=-inf, diag=0 => logsumexp=0 => logK(0,0)=logM(0,0).
    M = np.full((1, 1, 1, 1), 0.5)
    got = float(_host(alg.global_align_kern_log(_to_backend(M)))[0, 0])
    m = 0.5 / (2.0 - 0.5)
    assert abs(got - np.log(m)) < 1e-10, got


# -----------------------------------------------------------------------------
# RWS / DTW  (min,+ is exact -> DTW_EXACT tolerance)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("warp", [[1, 3, 2, 4], [1, 1, 1], [2], [5, 1]])
def test_rws_dtw_vs_oracle(warp):
    warp = np.array(warp)
    nX, lX = 4, 6
    D = _rng(6).random((nX, lX, int(warp.sum())))
    got = _host(alg.random_warping_series(_to_backend(D), _to_backend(warp)))
    exp = dp.rws_dtw(D, warp)
    assert_close(got, exp, family="DTW_EXACT", device=DEVICE,
                 case_id=f"rws_warp{list(warp)}")


def test_rws_dtw_lY1_edge():
    # The l_Y = 1 edge + unequal warp lengths (the scatter-index risk).
    warp = np.array([1, 4, 1, 2])
    D = _rng(7).random((3, 5, int(warp.sum())))
    got = _host(alg.random_warping_series(_to_backend(D), _to_backend(warp)))
    exp = dp.rws_dtw(D, warp)
    assert_close(got, exp, family="DTW_EXACT", device=DEVICE, case_id="rws_lY1")


def test_dtw_hand_computed():
    # Fully hand-traceable: D=[[1,3],[4,2]] -> optimal accumulated cost 3.
    D = np.array([[[1.0, 3.0], [4.0, 2.0]]])
    got = _host(alg.random_warping_series(_to_backend(D), _to_backend(np.array([2]))))
    assert abs(float(got[0, 0]) - 3.0) < 1e-12, got
