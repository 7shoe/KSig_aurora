"""SYCL fast-path vs the torch wavefront (``TEST_PLAN.md`` Sec. 7.10, marker
``sycl``).

The torch wavefront is the numerical oracle here: the SYCL kernels must agree
with it within the f64 band of each family (``TORCH_PORT`` Sec. 12, criterion 1).
Auto-skipped unless an XPU is present *and* ``ksig._sycl`` builds, so the suite
stays green on CPU dev boxes / login nodes and only exercises SYCL on an Aurora
compute node with the extension compiled.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
if not _HAS_XPU:
    pytest.skip("no XPU device available", allow_module_level=True)

from ksig._sycl import loader  # noqa: E402

if not loader.available():
    pytest.skip("ksig._sycl extension did not build", allow_module_level=True)

import ksig.algorithms as alg  # noqa: E402
from tests.oracles import signature_numpy as sn  # noqa: E402
from tests.harness import assert_close  # noqa: E402

pytestmark = [pytest.mark.xpu, pytest.mark.sycl]
DEV = "xpu"


def _rng(seed=0):
    return np.random.default_rng(seed)


def _xpu(a):
    return torch.as_tensor(a, device=DEV)


def _wavefront_sig_pde(M, difference):
    # Force the torch wavefront (the oracle) by computing on CPU.
    return alg.signature_kern_pde(torch.as_tensor(M, device="cpu"),
                                  difference=difference).cpu().numpy()


@pytest.mark.parametrize("difference", [True, False])
@pytest.mark.parametrize("shape", [(3, 2, 6, 6), (4, 3, 5, 7), (2, 2, 1, 1)])
def test_sycl_sigpde_vs_wavefront(difference, shape):
    nX, nY, lX, lY = shape
    X = _rng(1).standard_normal((nX, lX, 3))
    Y = _rng(2).standard_normal((nY, lY, 3))
    M = sn.static_gram(X, Y, "rbf")
    got = alg.signature_kern_pde(_xpu(M), difference=difference).cpu().numpy()
    exp = _wavefront_sig_pde(M, difference)
    assert_close(got, exp, family="DP_CUMSUM", device=DEV,
                 case_id=f"sycl_sigpde_diff{difference}_{shape}")


@pytest.mark.parametrize("shape", [(3, 2, 6, 6), (4, 3, 5, 7)])
def test_sycl_gak_vs_wavefront(shape):
    nX, nY, lX, lY = shape
    X = _rng(4).standard_normal((nX, lX, 3))
    Y = _rng(5).standard_normal((nY, lY, 3))
    M = sn.static_gram(X, Y, "rbf")
    got = alg.global_align_kern_log(_xpu(M)).cpu().numpy()
    exp = alg.global_align_kern_log(torch.as_tensor(M, device="cpu")).cpu().numpy()
    assert_close(got, exp, family="DP_CUMSUM", device=DEV, case_id=f"sycl_gak_{shape}")
    assert np.isfinite(got).all()


@pytest.mark.parametrize("warp", [[1, 3, 2, 4], [1, 4, 1, 2], [2], [5, 1]])
def test_sycl_rws_vs_wavefront(warp):
    warp = np.array(warp)
    D = _rng(6).random((4, 6, int(warp.sum())))
    got = alg.random_warping_series(_xpu(D), _xpu(warp)).cpu().numpy()
    exp = alg.random_warping_series(torch.as_tensor(D, device="cpu"),
                                    torch.as_tensor(warp, device="cpu")).cpu().numpy()
    assert_close(got, exp, family="DTW_EXACT", device=DEV,
                 case_id=f"sycl_rws_{list(warp)}")


def test_sycl_torch_stream_ordering():
    """Interleave a SYCL DP call with surrounding torch ops on the same XPU
    stream; the result must not see stale data (run a few times)."""
    X = _rng(7).standard_normal((3, 5, 3))
    M = sn.static_gram(X, X, "rbf")
    ref = alg.signature_kern_pde(torch.as_tensor(M, device="cpu")).cpu().numpy()
    for _ in range(8):
        Mx = _xpu(M) * 1.0          # a torch op feeding the SYCL kernel
        K = alg.signature_kern_pde(Mx)
        K = K + 0.0                 # a torch op consuming the SYCL output
        np.testing.assert_allclose(K.cpu().numpy(), ref, rtol=1e-8, atol=1e-10)
