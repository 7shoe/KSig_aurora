"""Static kernels (``ksig/static/kernels.py``) — Gram symmetry, diagonal
contracts, and value vs the closed-form NumPy oracle (``TEST_PLAN.md`` Sec. 7.2).

Includes explicit coverage of the **``euclid_dist`` bug**: ``Matern12``/
``Matern32`` are ``xfail`` on the legacy CuPy backend (broken) and required to
pass on the torch port — the suite drives the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._backend import host, to_backend
from tests.harness import DEVICE, IS_LEGACY, assert_close
from tests.oracles import static_numpy as st

ksig = pytest.importorskip("ksig")

pytestmark = pytest.mark.unit

KERNELS = ["LinearKernel", "PolynomialKernel", "RBFKernel", "Matern12Kernel",
           "Matern32Kernel", "Matern52Kernel", "RationalQuadraticKernel"]
STATIONARY = {"RBFKernel", "Matern12Kernel", "Matern32Kernel", "Matern52Kernel",
              "RationalQuadraticKernel"}
FAMILY = {"LinearKernel": "EXACT_ALGEBRA", "PolynomialKernel": "EXACT_ALGEBRA"}


def _rng(seed=0):
    return np.random.default_rng(seed)


def _maybe_xfail_legacy(name):
    if IS_LEGACY and st.KERNELS[name][1]:
        pytest.xfail(f"legacy euclid_dist bug breaks {name}; port must fix it")


@pytest.mark.parametrize("name", KERNELS)
def test_static_gram_vs_oracle(name):
    _maybe_xfail_legacy(name)
    cls = getattr(ksig.static.kernels, name)
    X = _rng(0).standard_normal((5, 4))
    Y = _rng(1).standard_normal((3, 4))
    got = host(cls()(to_backend(X, DEVICE), to_backend(Y, DEVICE)))
    exp = st.gram(name, X, Y)
    assert_close(got, exp, family=FAMILY.get(name, "DP_CUMSUM"), device=DEVICE,
                 case_id=f"static_{name}_xy")


@pytest.mark.parametrize("name", KERNELS)
def test_static_gram_symmetric(name):
    _maybe_xfail_legacy(name)
    cls = getattr(ksig.static.kernels, name)
    X = _rng(2).standard_normal((6, 4))
    G = host(cls()(to_backend(X, DEVICE)))
    assert np.allclose(G, G.T, atol=1e-9), f"{name} self-Gram not symmetric"


@pytest.mark.parametrize("name", KERNELS)
def test_static_diag_matches_gram_diag(name):
    _maybe_xfail_legacy(name)
    cls = getattr(ksig.static.kernels, name)
    X = _rng(3).standard_normal((5, 4))
    d = host(cls()(to_backend(X, DEVICE), diag=True))
    G = host(cls()(to_backend(X, DEVICE)))
    assert_close(d, np.diag(G), family=FAMILY.get(name, "DP_CUMSUM"),
                 device=DEVICE, case_id=f"static_{name}_diag")


@pytest.mark.parametrize("name", sorted(STATIONARY))
def test_stationary_unit_diagonal(name):
    _maybe_xfail_legacy(name)
    cls = getattr(ksig.static.kernels, name)
    X = _rng(4).standard_normal((5, 4))
    d = host(cls()(to_backend(X, DEVICE), diag=True))
    assert np.allclose(d, 1.0, atol=1e-9), f"{name} self-value != 1"


def test_matern12_is_broken_on_legacy_or_fixed_on_port():
    """Pin the bug explicitly so it can never silently regress.

    On legacy: calling Matern12 with a single argument raises (the spurious
    ``self`` in ``utils.euclid_dist`` shifts the args).  On the port: it must
    produce the correct value.
    """
    cls = ksig.static.kernels.Matern12Kernel
    X = _rng(5).standard_normal((4, 3))
    if IS_LEGACY:
        with pytest.raises(Exception):
            host(cls()(to_backend(X, DEVICE)))
    else:
        got = host(cls()(to_backend(X, DEVICE)))
        assert_close(got, st.matern12(X), family="DP_CUMSUM", device=DEVICE,
                     case_id="matern12_port")
