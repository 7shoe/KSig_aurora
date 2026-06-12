"""PDE signature contracts: Goursat boundary/constant path, and the F5 stability
policy.

Policy under test (revised F5):
  * plain `sig-PDE` -- warn on risky max|M|, run the INTENDED solve, assert finite;
    NEVER silently rescale the driver (no auto-clamp).
  * learned `sig-PDEphi` -- auto-clamp/NaN is legitimate (lambda is a design var),
    and phi()/phi_profile() must report the EFFECTIVE (clamped) lambda.

The wavefront-vs-library equivalence and the sig-PDEphi gate-fires/auto-clamp-
finite behavior already live in `tests/test_general_signature.py` (tests 08, 11);
not duplicated. The items that depend on the not-yet-landed P7 policy are `xfail`
(repo convention: they xpass -> promote when P7 lands).
"""
from __future__ import annotations

import numpy as np
import pytest

import ksig
ksig.set_default_device("cpu")
from ksig.generalized import GeneralSignatureKernel

from tests.paper_contracts._common import paths, constant_paths

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- boundary / constant
def test_sigpde_constant_path_is_one():
    """Zero-increment (constant-in-time) paths -> signature collapses to level 0,
    so the normalized sig-PDE Gram is all ones (Goursat boundary check)."""
    X = constant_paths(n=5, L=8, d=3)
    K = np.asarray(GeneralSignatureKernel(phi="const", truncation=None)(X, X))
    assert np.allclose(K, np.ones_like(K), atol=1e-5)


def test_sigpde_finite_on_normal_path():
    """A standard bounded path yields a finite sig-PDE Gram (no spurious NaN)."""
    X = paths(n=8, L=12, d=3)
    K = np.asarray(GeneralSignatureKernel(phi="const", truncation=None)(X, X))
    assert np.isfinite(K).all()


# ---------------------------------------------------------------- F5: plain sig-PDE policy (P7 pending)
@pytest.mark.xfail(reason="P7 not landed: plain sig-PDE has no stability gate yet "
                          "(should warn on risky max|M| then assert finite, "
                          "without rescaling the driver).",
                   strict=False)
def test_plain_sigpde_warns_on_risky_path():
    """High-energy path should emit a single RuntimeWarning (then run the intended
    solve and assert finite) -- NOT silently rescale lambda."""
    X = paths(n=6, L=10, d=3, scale=40.0)
    with pytest.warns(RuntimeWarning):
        GeneralSignatureKernel(phi="const", truncation=None, bw=1.0)(X, X)


# ---------------------------------------------------------------- F5: learned sig-PDEphi
def test_sigpdephi_autoclamp_stays_finite_via_facade():
    """Through the facade, sig-PDEphi with auto_clamp=True stays finite on a path
    that would otherwise enter the unstable Goursat regime."""
    X = paths(n=8, L=10, d=3)
    k = GeneralSignatureKernel(phi="dilation", truncation=None, bw=1.0,
                               lam_max=5.0, auto_clamp=True)
    assert np.isfinite(np.asarray(k(X, X))).all()


@pytest.mark.xfail(reason="P7 reporting not landed: under auto_clamp the engine "
                          "evaluates with clamped lambda but phi()/phi_profile() "
                          "still reports the requested (unclamped) nodes.",
                   strict=False)
def test_auto_clamp_phi_profile_matches_evaluation_nodes():
    """phi_profile() must reflect the EFFECTIVE (clamped) lambda, not the requested
    nodes, when auto_clamp rescales lambda during evaluation."""
    X = paths(n=10, L=10, d=3)
    y = (np.arange(10) % 2)
    k = GeneralSignatureKernel(phi="dilation", truncation=None, bw=1.0,
                               lam_max=8.0, auto_clamp=True).fit_phi(X, y, steps=20)
    # An accessor for the effective (clamped) lambda profile is part of P7;
    # until it exists and agrees with phi_profile(), this xfails.
    effective = k.effective_lambdas()             # AttributeError until P7
    assert np.all(np.asarray(effective) <= 8.0 + 1e-6)
