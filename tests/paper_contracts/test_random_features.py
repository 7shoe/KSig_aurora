"""Random-feature contracts: per-level RNG independence (F6), realized sparsity
(F7), base-RFF sanity (paper-3 Contract 5), and a seeded Monte-Carlo convergence
smoke.

Core checks (independence/sparsity/RFF) are tiny and deterministic and always
run. The convergence smoke is `@pytest.mark.random_feature`: seeded and
tolerance-loose so it can be deselected and can never flake a core gate.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import ksig
ksig.set_default_device("cpu")
from ksig.kernels import SignatureFeatures, SignatureKernel
from ksig.static.features import RandomFourierFeatures
from ksig.static.kernels import RBFKernel
from ksig.projections import VerySparseRandomProjection
from ksig.torch_backend import to_numpy

from tests.paper_contracts._common import paths

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- F6: per-level independence
@pytest.mark.parametrize("seed", [0, 123, None])
def test_rfsf_independent_weights_per_level(seed):
    """Static-feature copies use INDEPENDENT weights per level (Contract 4)."""
    sf = SignatureFeatures(
        n_levels=4,
        static_features=RandomFourierFeatures(n_components=16, random_state=seed))
    sf.fit(paths())
    ws = [s.random_weights_ for s in sf.static_features_]
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            assert not torch.allclose(ws[i], ws[j]), \
                f"levels {i},{j} share weights -> biased RFSF estimator"


def test_rfsf_reproducible_for_integer_seed():
    """A fixed integer seed gives identical per-level weights across two fits."""
    def fit_once():
        sf = SignatureFeatures(
            n_levels=3,
            static_features=RandomFourierFeatures(n_components=12, random_state=7))
        sf.fit(paths())
        return [s.random_weights_ for s in sf.static_features_]
    a, b = fit_once(), fit_once()
    for wa, wb in zip(a, b):
        assert torch.allclose(wa, wb), "integer seed must be reproducible"


def test_rfsf_projection_independent_per_level():
    """Projection copies are also independent per level (Contracts 8/9)."""
    from ksig.projections import GaussianRandomProjection
    sf = SignatureFeatures(
        n_levels=3, n_features=4,
        projection=GaussianRandomProjection(n_components=8, random_state=0))
    sf.fit(paths(d=4))
    comps = [to_numpy(p.components_) for p in sf.projections_]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i].shape == comps[j].shape:
                assert not np.allclose(comps[i], comps[j])


# ---------------------------------------------------------------- F7: very sparse projection
def test_very_sparse_projection_realized_sparsity():
    """Realized component density ~= prob_nonzero, not dense (audit F7)."""
    n_features, n_components = 1000, 100
    U = torch.randn(20, n_features, dtype=torch.float64)
    proj = VerySparseRandomProjection(
        n_components=n_components, sparsity="log", random_state=0).fit(U)
    prob_nonzero = np.log(n_features) / n_features
    density = float((proj.components_ != 0).double().mean())
    assert density < 10 * prob_nonzero, f"not sparse (density={density:.3f})"
    assert density > 0.0


# ---------------------------------------------------------------- Contract 5: base RFF sanity
def test_rff_self_kernel_one_and_shape():
    """RFF map has dim 2*n_components and unit self-inner-product (k(x,x)=1)."""
    X = paths(n=10, L=1, d=3).reshape(10, 3)        # flat [n, d]
    rff = RandomFourierFeatures(n_components=64, bandwidth=1.0, random_state=0).fit(X)
    F = to_numpy(rff.transform(X, return_on_gpu=False))
    assert F.shape[-1] == 2 * 64
    self_inner = (F * F).sum(-1)
    assert np.allclose(self_inner, 1.0, atol=1e-5)


# ---------------------------------------------------------------- MC convergence smoke
@pytest.mark.random_feature
def test_rfsf_converges_to_exact_signature_kernel():
    """As n_components grows the RFF-based signature feature Gram approaches the
    exact RBF signature kernel. Seeded + averaged over seeds + loose margin: a
    smoke that the estimator is consistent, not a tight numerical gate."""
    X = paths(n=8, L=6, d=2, seed=4)
    N, bw = 3, 1.0
    exact = np.asarray(SignatureKernel(
        n_levels=N, normalize=False, static_kernel=RBFKernel(bandwidth=bw))(X, X))

    def mean_err(d_tilde):
        errs = []
        for s in range(3):
            sf = SignatureFeatures(
                n_levels=N, normalize=False,
                static_features=RandomFourierFeatures(
                    n_components=d_tilde, bandwidth=bw, random_state=s)).fit(X)
            P = to_numpy(sf.transform(X, return_on_gpu=False))
            errs.append(np.abs(P @ P.T - exact).mean())
        return float(np.mean(errs))

    err_low, err_high = mean_err(16), mean_err(256)
    assert err_high < err_low, f"no MC convergence: {err_low:.4f} -> {err_high:.4f}"
