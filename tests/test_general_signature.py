"""Acceptance tests for the General Signature Kernel facade and the learnable-phi engines.

Covers the 12 criteria from the migration plan (8 original + F-A/F-B/D4/cross-norm). Run:

    KSIG_DEVICE=cpu KSIG_DTYPE=float32 KSIG_USE_SYCL=0 pytest tests/test_general_signature.py
"""
import os
import numpy as np
import pytest
import torch

import ksig
ksig.set_default_device("cpu")
from ksig.generalized import (GeneralSignatureKernel, WeightedSignatureKernel,
                              LearnedPhiSignaturePDEKernel, sigpde_wavefront,
                              _static_block, _second_diff, signature_kern, _normalize)
from ksig.algorithms import signature_kern_pde


# ---------------------------------------------------------------- fixtures / helpers
def _paths(n=10, L=12, d=4, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((n, L, d)).astype(np.float32), 1) / np.sqrt(L)
    X = X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), 1e-6, None)
    return (X * scale).astype(np.float32)


def _labels(n, seed=1):
    return np.random.default_rng(seed).integers(0, 2, n)


def _offdiag_absmean(A, B):
    D = np.abs(np.asarray(A) - np.asarray(B))
    m = ~np.eye(D.shape[0], dtype=bool) if D.shape[0] == D.shape[1] else np.ones_like(D, bool)
    return float(D[m].mean())


# ---------------------------------------------------------------- tests
def test_01_convention_oracle():
    """GSK(const, N, per_level) reproduces ksig.kernels.SignatureKernel(N, normalize=True)."""
    X = _paths(8); bw = 1.0
    gsk = GeneralSignatureKernel(phi="const", truncation=3, normalize="per_level", bw=bw)
    from ksig.static.kernels import RBFKernel
    ref = ksig.kernels.SignatureKernel(n_levels=3, order=1, normalize=True,
                                       static_kernel=RBFKernel(bandwidth=bw))
    Kg = np.asarray(gsk(X, X, return_on_gpu=False))
    Kr = np.asarray(ref(torch.as_tensor(X), torch.as_tensor(X), return_on_gpu=False))
    assert np.max(np.abs(Kg - Kr)) < 1e-6


def test_02_confound_is_real():
    """normalize='once' (phi==1) differs measurably from normalize='per_level'."""
    X = _paths(10); bw = 1.0
    once = GeneralSignatureKernel(phi="const", truncation=3, normalize="once", bw=bw)(X, X)
    perl = GeneralSignatureKernel(phi="const", truncation=3, normalize="per_level", bw=bw)(X, X)
    assert _offdiag_absmean(once, perl) > 1e-3


def test_03_truncation_tail_vanishes():
    """C3 (corrected premise): the truncated phi==1 kernel CONVERGES as N grows -- the depth tail
    vanishes (Cauchy): ||K(N=8)-K(N=6)|| is tiny and far below ||K(N=3)-K(N=2)||.

    NOTE for the team: it converges to its OWN discrete-untruncated limit, which sits ~0.04
    (off-diag abs-mean) away from the Goursat-PDE solver -- the discrete-tensor vs continuous-PDE
    discretization gap, which does NOT shrink with N. So the `sig-PDE - sig-TRUNC-phi1` axis is
    truncation+solver, not a pure depth-tail effect; the clean pure-tail probe is within the
    truncated family at increasing N (see response/plan note)."""
    X = _paths(10); bw = 1.0
    K = {N: np.asarray(GeneralSignatureKernel(phi="const", truncation=N, bw=bw)(X, X))
         for N in (2, 3, 6, 8)}
    early = _offdiag_absmean(K[3], K[2])
    late = _offdiag_absmean(K[8], K[6])
    assert late < early                 # tail shrinking with depth
    assert late < 1e-3                  # converged (Cauchy in N)


def test_04_constant_cancels_under_once():
    """phi==const is invariant to the phi scale under normalize-once."""
    X = _paths(8); bw = 1.0; N = 3
    k1 = WeightedSignatureKernel(n_levels=N, bw=bw, phi_fixed=np.ones(N + 1))(X, X)
    k5 = WeightedSignatureKernel(n_levels=N, bw=bw, phi_fixed=5.0 * np.ones(N + 1))(X, X)
    assert np.max(np.abs(np.asarray(k1) - np.asarray(k5))) < 1e-5


@pytest.mark.parametrize("cfg", [
    dict(phi="level_one", truncation=3), dict(phi="const", truncation=3),
    dict(phi="const", truncation=3, normalize="per_level"),
    dict(phi="const", truncation=None), dict(phi="free", truncation=3),
    dict(phi="dilation", truncation=None),
])
def test_05_psd(cfg):
    """Every config is PSD (min eig >= -1e-6) after a short fit."""
    X = _paths(14); y = _labels(14)
    gsk = GeneralSignatureKernel(bw=1.0, **cfg).fit_phi(X, y, steps=40)
    K = np.asarray(gsk(X, X)); K = 0.5 * (K + K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-6


def test_06_phi0_frozen():
    """C1: after fit, phi(0)==1 for sig-Wphi (free) and sig-PDEphi (dilation). Self-enforcing gate."""
    X = _paths(16); y = _labels(16)
    wphi = GeneralSignatureKernel(phi="free", truncation=4, bw=1.0).fit_phi(X, y, steps=80)
    pdephi = GeneralSignatureKernel(phi="dilation", truncation=None, bw=1.0).fit_phi(X, y, steps=60)
    assert np.isclose(wphi.phi_profile()[0], 1.0, atol=1e-5)
    assert np.isclose(pdephi.phi_profile()[0], 1.0, atol=1e-5)
    # theta[0] carries no gradient signal (phi(0) is pinned, not learned)
    wk = WeightedSignatureKernel(n_levels=4, bw=1.0)
    Kl, dXl, _ = wk._levels(X)
    th = wk.theta.clone().requires_grad_(True)
    phi = torch.cat([torch.ones(1), torch.nn.functional.softplus(th[1:])])
    from ksig.generalized import _cka_loss, _target
    _cka_loss((phi[:, None, None] * Kl).sum(0), _target(y, "classification", torch.device("cpu"))).backward()
    assert th.grad[0].abs().item() < 1e-12


def test_07_regression_cka():
    """Continuous y centered; regression fit runs, phi finite & phi(0)==1; the learned phi aligns
    at least as well as the flat phi==1 init. (Exact phi recovery is under-determined -- the level
    Grams are collinear, per phi_recovery.py -- so the criterion is alignment, not coefficients.)"""
    X = _paths(40, seed=3)
    rng = np.random.default_rng(7)
    Kl, dXl, _ = WeightedSignatureKernel(n_levels=4, bw=1.0)._levels(X)
    base = np.asarray(_normalize(Kl[2], dXl[2], dXl[2])).mean(1)        # kernel-aligned signal
    yv = (0.7 * (base - base.mean()) / (base.std() + 1e-8)
          + 0.3 * rng.standard_normal(40)).astype(np.float32)          # + noise

    from ksig.generalized import _cka_loss, _target

    def alignment(gsk):
        K = torch.as_tensor(np.asarray(gsk(X, X)))
        return -float(_cka_loss(K, _target(yv, "regression", torch.device("cpu"))))

    flat = GeneralSignatureKernel(phi="const", truncation=4, bw=1.0)
    wk = GeneralSignatureKernel(phi="free", truncation=4, bw=1.0).fit_phi(
        X, yv, task="regression", steps=200)
    phi = wk.phi_profile()
    assert np.isfinite(phi).all() and np.isclose(phi[0], 1.0, atol=1e-5)
    assert alignment(wk) >= alignment(flat) - 1e-4


def test_08_wavefront_unchanged():
    """sigpde_wavefront matches the in-place library signature_kern_pde (engine untouched)."""
    X = _paths(6, L=10); bw = 1.0
    M = _second_diff(_static_block(X, X, bw, torch.device("cpu")))    # [n,n,L-1,L-1]
    ours = sigpde_wavefront(M).cpu().numpy()
    lib = signature_kern_pde(M, difference=False).cpu().numpy()       # M already differenced
    assert np.max(np.abs(ours - lib)) < 1e-5


def test_09_sig_l1_purity():
    """F-A: GSK(level_one) equals the normalized level-1 signature kernel, with NO K0 contamination
    (phi(0)==0). Differs from a phi(0)=1 variant by exactly the rank-1 constant level."""
    X = _paths(10); bw = 1.0; N = 3
    l1 = GeneralSignatureKernel(phi="level_one", truncation=N, bw=bw)
    assert l1.phi_profile()[0] == 0.0
    K_l1 = np.asarray(l1(X, X))
    # direct: normalized level-1 signature kernel
    M = _second_diff(_static_block(X, X, bw, torch.device("cpu")))
    Kl = torch.stack(list(signature_kern(M, N, 1, False, True)), 0)
    from ksig.generalized import _diag_block
    dl = torch.stack(list(signature_kern(_diag_block(X, bw, torch.device("cpu")), N, 1, False, True)), 0)
    K_direct = _normalize(Kl[1], dl[1], dl[1]).cpu().numpy()
    assert np.max(np.abs(K_l1 - K_direct)) < 1e-5
    # contamination check: adding phi(0)=1 changes the kernel
    contaminated = WeightedSignatureKernel(n_levels=N, bw=bw,
                                           phi_fixed=np.array([1, 1, 0, 0], np.float32))(X, X)
    assert _offdiag_absmean(K_l1, contaminated) > 1e-3


def test_10_linear_static_guard():
    """F-B (broadened, C2): untruncated + linear is rejected for ANY phi (const and dilation)."""
    with pytest.raises(ValueError):
        GeneralSignatureKernel(phi="dilation", truncation=None, static="linear")
    with pytest.raises(ValueError):
        GeneralSignatureKernel(phi="const", truncation=None, static="linear")


def test_11_stability_gate_fires():
    """D4: when lam_max*max|m| > 1 the gate WARNS and the cell is NaN'd (never raises)."""
    X = _paths(8, scale=1.0)
    pk = LearnedPhiSignaturePDEKernel(m_nodes=4, lam_max=5.0, bw=1.0, auto_clamp=False)
    with pytest.warns(RuntimeWarning):
        K = pk(X, X)
    assert not np.isfinite(np.asarray(K)).all()           # cell NaN'd, no exception
    # auto_clamp path stays finite instead
    pk2 = LearnedPhiSignaturePDEKernel(m_nodes=4, lam_max=5.0, bw=1.0, auto_clamp=True)
    assert np.isfinite(np.asarray(pk2(X, X))).all()


def test_12_cross_gram_normalization():
    """Cross block normalized with test-self & train-self diagonals lies in [-1,1]; train diag==1."""
    Xtr, Xte = _paths(12, seed=4), _paths(7, seed=5)
    k = GeneralSignatureKernel(phi="const", truncation=3, bw=1.0)
    Ktt = np.asarray(k(Xtr, Xtr))
    Kxt = np.asarray(k(Xte, Xtr))                          # test x train cross block
    assert np.allclose(np.diag(Ktt), 1.0, atol=1e-4)       # unit train-self diagonal
    assert Kxt.max() <= 1.0 + 1e-4 and Kxt.min() >= -1.0 - 1e-4
