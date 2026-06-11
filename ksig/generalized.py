"""Generalized (learnable-phi) signature kernels — the General Signature Kernel (GSK) family.

AUGMENTS KSig_aurora additively -- this module imports from the package
(`from .algorithms import signature_kern`, lazily `from . import kernels`) but modifies nothing
existing. The frozen engines (`sigpde_wavefront`, `_normalize`, `_cka_loss`, the static/diff
blocks) are reproduced verbatim and must not be rewritten.

A signature kernel's level-weighting  phi(k) = sum_i w_i lam_i^k  is exactly a mixture of
DILATED signature kernels   K_phi = sum_i w_i K^(lam_i)   (dilating a path by lam multiplies
its level-k signature by lam^k; Cass-Lyons-Xu 2021). So learning phi is learning (w, lam).
In the kernelized world "dilate by lam" == "scale the second-differenced static Gram by lam":
M -> lam M.

  WeightedSignatureKernel        : TRUNCATED. phi(0)=1 PINNED, phi(1:N)=softplus(theta[1:])>=0
                                   (fitted) -- or a LITERAL phi_fixed vector (allows exact zeros,
                                   e.g. e_1 for the level-1-only reference). The per-level Grams
                                   from signature_kern(return_levels=True) are constants, so
                                   K_phi = sum_k phi(k) K_k -- MKL over signature levels.
  LearnedPhiSignaturePDEKernel   : UNTRUNCATED. m FIXED nodes; learn (w, lam) by backprop through
                                   an OUT-OF-PLACE (autograd-safe) Goursat wavefront. phi(k)=sum
                                   w_i lam_i^k; phi(0)=sum w_i=1 is structural (softmax).
  GeneralSignatureKernel         : config-driven facade. One object reproduces every sweep column
                                   (sig-L1 / sig-TRUNC-phi1 / sig-PDE / sig-Wphi / sig-PDEphi /
                                   sig-EXACT-per-level) by delegating to the engines above and to
                                   ksig.kernels.{SignatureKernel,SignaturePDEKernel}.

Conventions (load-bearing for valid cross-method deltas):
  * normalize ONCE, after combining (D^{-1/2} K D^{-1/2}); never per-level / per-node. The legacy
    per-level-normalize-then-average convention is reachable ONLY via normalize="per_level"
    (the sig-EXACT column) and is a normalization probe, not a phi member.
  * phi(0)=1 on EVERY signature arm (WeightedSignatureKernel pins it; SigPDE has it via the
    Goursat init H==1; the dilation mixture has it via softmax). This keeps the rank-1 level
    K_0=11^T weighted identically across arms, so Delta(learn-phi) carries no level-0 mismatch.
  * Use the RBF static kernel for the UNTRUNCATED (Goursat) path: it needs bounded increments
    (|m|<=2) and is stable only for |lam*m|<~1, so clamp lam to [0, lam_max], lam_max<~1
    (default 0.5). Linear static + untruncated is rejected.

Both engines expose the KSig `__call__(X, Y=None, diag=False, return_on_gpu=False)` contract, so
after a `fit_phi(Xtr, ytr)` phase they are ordinary precomputed kernels (drop into gak_gram + SVC).
"""
from __future__ import annotations
import warnings
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .algorithms import signature_kern


# ---------------------------------------------------------------- shared pieces (FROZEN)
def _static_block(X, Y, bw, dev):
    """RBF Gram block, NOT yet differenced. X:[nX,L,d] Y:[nY,L,d] -> [nX,nY,L,L] torch."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float32, device=dev)
    d2 = (Xt[:, None, :, None, :] - Yt[None, :, None, :, :]).pow(2).sum(-1)
    return torch.exp(-d2 / (2.0 * bw * bw))                      # [nX,nY,L,L]


def _second_diff(G):
    return torch.diff(torch.diff(G, dim=-2), dim=-1)            # -> [.,.,L-1,L-1]


def _diag_block(X, bw, dev):
    """Self-pair differenced blocks, [n, L-1, L-1]: the i-th path against itself. Computed in
    O(n L^2) (NOT via the full [n,n,L,L] cross block), so it is cheap to call per eval row-block."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)   # [n,L,d]
    d2 = (Xt[:, :, None, :] - Xt[:, None, :, :]).pow(2).sum(-1)            # [n,L,L]
    return _second_diff(torch.exp(-d2 / (2.0 * bw * bw)))                  # [n,L-1,L-1]


def _cka_loss(K, y):
    """Negative kernel-target alignment (centered). Minimizing this maximizes CKA(K, yy^T).
    Valid for binary (+/-1) AND centered-continuous y (regression): the centering Hc handles
    the target's mean, so a centered continuous y yields the linear-in-y alignment direction."""
    n = K.shape[0]
    Hc = torch.eye(n, device=K.device) - 1.0 / n
    Kc, Yc = Hc @ K @ Hc, Hc @ torch.outer(y, y) @ Hc
    return -(Kc * Yc).sum() / (Kc.norm() * Yc.norm() + 1e-12)


def _normalize(K, dx, dy):
    return K / torch.sqrt(dx[:, None].clamp_min(1e-12) * dy[None, :].clamp_min(1e-12))


def sigpde_wavefront(M):
    """Second-order Goursat scheme, OUT-OF-PLACE (autograd-safe). M:[...,lx,ly] already
    differenced & lam-scaled -> [...]. Same recurrence as ksig.algorithms.signature_kern_pde, but
    that library version writes H in place and is NOT backprop-safe. Stability: |M| <~ 1."""
    *batch, lx, ly = M.shape
    Mf = M.reshape(-1, lx, ly); P = Mf.shape[0]
    H = torch.ones(P, lx + 1, ly + 1, dtype=M.dtype, device=M.device)
    ar = torch.arange(P, device=M.device)
    for it in range(lx + ly - 1):
        i = torch.arange(max(0, it - ly + 1), min(it, lx - 1) + 1, device=M.device)
        j = it - i
        m = Mf[:, i, j]
        up, left, dg = H[:, i, j + 1], H[:, i + 1, j], H[:, i, j]
        new = (up + left) * (1 + 0.5 * m + m * m / 12) - dg * (1 - m * m / 12)
        pi = ar[:, None].expand(P, len(i)).reshape(-1)
        H = H.index_put((pi, (i + 1).repeat(P), (j + 1).repeat(P)), new.reshape(-1))
    return H[:, lx, ly].reshape(*batch)


# ---------------------------------------------------------------- new shared helpers
def _target(y, task, dev):
    """CKA alignment target. classification -> +/-1 labels; regression -> centered continuous y."""
    y = np.asarray(y, dtype=np.float64)
    yt = (y - y.mean()) if task == "regression" else (2.0 * y - 1.0)
    return torch.as_tensor(yt, dtype=torch.float32, device=dev)


def _stability_ok(M, lam_max, where=""):
    """Goursat stability guard: the 2nd-order scheme is stable only for |lam*m|<~1. Warns (never
    raises) when lam_max*max|m| > 1 so the caller can NaN the cell -- a hard raise would kill a
    long sweep. Returns False if the bound is exceeded."""
    s = float(lam_max * M.abs().max())
    if s > 1.0:
        warnings.warn(f"sig-PDEphi stability: lam_max*max|m|={s:.2f}>1 in {where}; Goursat may "
                      f"diverge (cell will be NaN'd). Lower lam_max or use float64.",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


# ------------------------------------------------- Piece 1: truncated, learnable phi
class WeightedSignatureKernel:
    """TRUNCATED GSK: K_phi = sum_{k=0..N} phi(k) K_k, normalize ONCE.

    Fitted mode (phi_fixed=None): phi(0)=1 PINNED (never softplus(theta_0)); phi(1:N)=
        softplus(theta[1:]) >= 0, learned by CKA. Pinning phi(0)=1 keeps the rank-1 level
        K_0=11^T weighted as in every other signature arm, so the learn-phi delta is single-factor.
    Fixed mode (phi_fixed given): a LITERAL nonneg phi vector of length N+1 (allows exact zeros --
        e.g. e_1 for the level-1-only reference, which softplus cannot represent). No fitting.
    """

    def __init__(self, n_levels=5, bw=1.0, normalize=True, dev="cpu", phi_fixed=None):
        self.N, self.bw, self.normalize, self.dev = n_levels, bw, normalize, torch.device(dev)
        self.theta = torch.zeros(n_levels + 1)          # theta[0] is inert: phi(0)=1 is pinned
        if phi_fixed is None:
            self.phi_fixed = None
        else:
            self.phi_fixed = torch.as_tensor(np.asarray(phi_fixed, dtype=np.float32),
                                             dtype=torch.float32)
            if self.phi_fixed.numel() != n_levels + 1:
                raise ValueError(f"phi_fixed must have length n_levels+1={n_levels + 1}, "
                                 f"got {self.phi_fixed.numel()}")
            if (self.phi_fixed < 0).any():
                raise ValueError("phi_fixed must be nonnegative (PSD).")

    def _phi_vec(self, theta=None):
        """phi as a length-(N+1) tensor on self.dev. phi(0)=1 pinned in fitted mode."""
        if self.phi_fixed is not None:
            return self.phi_fixed.to(self.dev)
        th = (self.theta if theta is None else theta).to(self.dev)
        return torch.cat([torch.ones(1, device=self.dev),
                          torch.nn.functional.softplus(th[1:])])

    def _levels(self, X, Y=None):
        """Cross level Grams [N+1,nX,nY] and self-diagonals [N+1,nX],[N+1,nY]. The self-diagonals
        use signature_kern's native ndim==3 'diag' mode on the O(n L^2) self block (no [n,n,L,L])."""
        M = _second_diff(_static_block(X, X if Y is None else Y, self.bw, self.dev))
        K = torch.stack(list(signature_kern(M, self.N, 1, False, True)), 0)            # [N+1,nX,nY]
        dX = torch.stack(list(signature_kern(_diag_block(X, self.bw, self.dev),
                                             self.N, 1, False, True)), 0)              # [N+1,nX]
        dY = dX if Y is None else torch.stack(list(signature_kern(
                 _diag_block(Y, self.bw, self.dev), self.N, 1, False, True)), 0)
        return K, dX, dY

    def fit_phi(self, X, y, steps=300, lr=5e-2, task="classification", inner_idx=None,
                fit_unnormalized=False):
        """Learn phi(1:N) by CKA (phi(0)=1 pinned). task in {classification, regression}.
        inner_idx: optional row indices to restrict the fit to (disjoint inner fold, D2).
        fit_unnormalized: DIAGNOSTIC ONLY (D1) -- fit the convex unnormalized alignment; never a
        reported run (its argmax is a level-magnitude artifact). Default False."""
        if self.phi_fixed is not None:
            return self                                          # fixed phi -> nothing to learn
        if inner_idx is not None:
            X, y = X[inner_idx], np.asarray(y)[inner_idx]
        Kl, dXl, _ = self._levels(X)                             # constants; computed ONCE
        yt = _target(y, task, self.dev)
        th = self.theta.clone().to(self.dev).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=lr)
        do_norm = self.normalize and not fit_unnormalized
        for _ in range(steps):
            opt.zero_grad()
            phi = torch.cat([torch.ones(1, device=self.dev),    # phi(0)=1 pinned (no grad)
                             torch.nn.functional.softplus(th[1:])])
            K = (phi[:, None, None] * Kl).sum(0)
            if do_norm:
                d = (phi[:, None] * dXl).sum(0)
                K = _normalize(K, d, d)
            _cka_loss(K, yt).backward(); opt.step()
        self.theta = th.detach()
        return self

    def phi(self, kmax=None):
        with torch.no_grad():
            return self._phi_vec().cpu().numpy()[: (kmax or self.N) + 1]

    def __call__(self, X, Y=None, diag=False, return_on_gpu=False):
        with torch.no_grad():
            phi = self._phi_vec()
            if diag:
                if self.normalize:
                    out = torch.ones(len(X), device=self.dev)
                else:
                    Kl = torch.stack(list(signature_kern(_diag_block(X, self.bw, self.dev),
                                                         self.N, 1, False, True)), 0)   # [N+1,n]
                    out = (phi[:, None] * Kl).sum(0)
            else:
                Kl, dX, dY = self._levels(X, Y)
                out = (phi[:, None, None] * Kl).sum(0)
                if self.normalize:
                    out = _normalize(out, (phi[:, None] * dX).sum(0), (phi[:, None] * dY).sum(0))
        return out if return_on_gpu else out.cpu().numpy()


# --------------------------------------- Piece 2: untruncated, learnable phi (dilations)
class LearnedPhiSignaturePDEKernel:
    """K_phi = sum_i w_i SigPDE(lam_i M), phi(k)=sum_i w_i lam_i^k.  Fit (w,lam) by CKA.
    lam_i = lam_max * sigmoid(.) (clamped cone), w = softmax(.).  m FIXED.
    phi(0)=sum_i w_i=1 is STRUCTURAL (softmax) -- so freeze_phi0 is automatic here, not enforced.
    Stability (D4): lam_max default 0.5; gate warns + NaNs the cell when lam_max*max|m|>1 (or
    auto_clamp=True rescales lam). Use static RBF only (the facade rejects linear)."""

    def __init__(self, m_nodes=5, lam_max=0.5, bw=1.0, normalize=True, dev="cpu", auto_clamp=False):
        self.m, self.lam_max, self.bw, self.normalize, self.dev = \
            m_nodes, lam_max, bw, normalize, torch.device(dev)
        self.auto_clamp = auto_clamp
        u = (np.arange(1, m_nodes + 1) - 0.5) / m_nodes
        self.raw_l = torch.tensor(np.log(u / (1 - u)), dtype=torch.float32)   # spread nodes in (0,lam_max)
        self.raw_w = torch.zeros(m_nodes)                                     # uniform weights at init

    def _eff_lam_max(self, M):
        """lam_max actually used: clamped down so lam_max*max|m|<=1 when auto_clamp, else lam_max."""
        if not self.auto_clamp:
            return self.lam_max
        mx = float(M.abs().max())
        return self.lam_max if mx * self.lam_max <= 1.0 else (1.0 / max(mx, 1e-12))

    def _nodes(self, raw_l, raw_w, lam_max=None):
        lm = self.lam_max if lam_max is None else lam_max
        return torch.softmax(raw_w, 0), lm * torch.sigmoid(raw_l)

    def _combine(self, M, Mx, My, raw_l, raw_w, ckpt=False, lam_max=None):
        """Mixture of dilated SigPDE solves on CACHED differenced blocks. ckpt=True trades compute
        for memory (recompute the wavefront in backward) -- essential when autograd-ing at scale."""
        w, lam = self._nodes(raw_l, raw_w, lam_max=lam_max)
        wf = (lambda Z: checkpoint(sigpde_wavefront, Z, use_reentrant=False)) if ckpt else sigpde_wavefront
        K = sum(wi * wf(li * M) for wi, li in zip(w, lam))
        if not self.normalize:
            return K
        dX = sum(wi * wf(li * Mx) for wi, li in zip(w, lam))
        dY = dX if My is None else sum(wi * wf(li * My) for wi, li in zip(w, lam))
        return _normalize(K, dX, dY)

    def fit_phi(self, X, y, steps=200, lr=5e-2, ckpt=True, learn_lam=False,
                task="classification", inner_idx=None):
        """Learn the dilation mixture by CKA on (X, y). task in {classification, regression};
        inner_idx restricts the fit to a disjoint inner fold (D2).
        learn_lam=False (default, FAST & convex-in-w): freeze lam at the init grid, precompute the
            m per-node SigPDE Grams ONCE, learn only the nonneg weights w. Describe as "convex
            quadrature weights on a fixed lambda-grid", NOT "learned dilation nodes".
        learn_lam=True (faithful, SLOW): also learn the node positions lam by backprop through the
            autograd-safe wavefront (use ckpt=True to bound memory)."""
        if inner_idx is not None:
            X, y = X[inner_idx], np.asarray(y)[inner_idx]
        yt = _target(y, task, self.dev)
        M = _second_diff(_static_block(X, X, self.bw, self.dev))    # hoisted: built ONCE
        Mx = _diag_block(X, self.bw, self.dev)
        _stability_ok(M, self.lam_max, "fit_phi")                  # warn (eval will NaN if it diverges)
        if not learn_lam:                                          # ---- FAST convex weights-only ----
            with torch.no_grad():
                lam = self.lam_max * torch.sigmoid(self.raw_l.to(self.dev))
                Gi = torch.stack([sigpde_wavefront(li * M) for li in lam], 0)     # [m,n,n] constants
                di = torch.stack([sigpde_wavefront(li * Mx) for li in lam], 0)    # [m,n]
            rw = self.raw_w.clone().to(self.dev).requires_grad_(True)
            opt = torch.optim.Adam([rw], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                w = torch.softmax(rw, 0)
                K = (w[:, None, None] * Gi).sum(0)
                if self.normalize:
                    dd = (w[:, None] * di).sum(0)
                    K = _normalize(K, dd, dd)
                _cka_loss(K, yt).backward(); opt.step()
            self.raw_w = rw.detach()                              # lam (raw_l) unchanged
            return self
        rl = self.raw_l.clone().to(self.dev).requires_grad_(True)  # ---- faithful (w, lam) ----
        rw = self.raw_w.clone().to(self.dev).requires_grad_(True)
        opt = torch.optim.Adam([rl, rw], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            _cka_loss(self._combine(M, Mx, None, rl, rw, ckpt=ckpt), yt).backward()
            opt.step()
        self.raw_l, self.raw_w = rl.detach(), rw.detach()
        return self

    def phi(self, kmax=12):
        with torch.no_grad():
            w, lam = self._nodes(self.raw_l, self.raw_w)
            ks = torch.arange(kmax + 1, dtype=torch.float32, device=self.dev)[:, None]
            return (w[None] * lam[None] ** ks).sum(-1).cpu().numpy()

    def __call__(self, X, Y=None, diag=False, return_on_gpu=False):
        with torch.no_grad():
            if diag:
                out = torch.ones(len(X), device=self.dev)          # normalized diag == 1
                if not self.normalize:
                    Mx = _diag_block(X, self.bw, self.dev)
                    if not _stability_ok(Mx, self.lam_max, "__call__(diag)"):
                        out = torch.full((len(X),), float("nan"), device=self.dev)
                    else:
                        w, lam = self._nodes(self.raw_l, self.raw_w)
                        out = sum(wi * sigpde_wavefront(li * Mx) for wi, li in zip(w, lam))
            else:
                M = _second_diff(_static_block(X, X if Y is None else Y, self.bw, self.dev))
                lm = self._eff_lam_max(M)
                if not self.auto_clamp and not _stability_ok(M, self.lam_max, "__call__"):
                    nY = len(X) if Y is None else len(Y)
                    out = torch.full((len(X), nY), float("nan"), device=self.dev)
                else:
                    Mx = _diag_block(X, self.bw, self.dev)
                    My = None if Y is None else _diag_block(Y, self.bw, self.dev)
                    out = self._combine(M, Mx, My, self.raw_l, self.raw_w, ckpt=False, lam_max=lm)
        return out if return_on_gpu else out.cpu().numpy()


# ------------------------------------------- the config-driven General Signature Kernel facade
class GeneralSignatureKernel:
    """Unified GSK. One object reproduces every sweep column by configuration; delegates to the
    frozen engines (no engine is rewritten):

        phi        : "const" | "level_one" | "free" | "dilation"
        truncation : int N (truncated) | None (untruncated / Goursat PDE)
        normalize  : "once" (GSK-correct, default) | "per_level" (legacy avg convention)
        static     : "rbf" | "linear"

    Column map:
        sig-L1            phi="level_one", N,    once     (phi(0)=0; order-blind level-1 reference)
        sig-TRUNC-phi1    phi="const",     N,    once     (phi==1 truncated; the CORRECT phi==1 base)
        sig-PDE           phi="const",     None, once
        sig-Wphi          phi="free",      N,    once     (learned phi(1:N), phi(0)=1 pinned)
        sig-PDEphi        phi="dilation",  None, once     (learned dilation mixture, phi(0)=1)
        sig-EXACT         phi="const",     N,    per_level(per-level-normalize-then-average probe)
    """
    _PHIS = ("const", "level_one", "free", "dilation")
    _NORMS = ("once", "per_level")
    _STATICS = ("rbf", "linear")

    def __init__(self, phi="const", truncation=3, normalize="once", static="rbf",
                 bw=1.0, m_nodes=6, lam_max=0.5, dev="cpu",
                 freeze_phi0=True, fit_unnormalized=False, auto_clamp=False):
        if phi not in self._PHIS:
            raise ValueError(f"phi must be one of {self._PHIS}, got {phi!r}")
        if normalize not in self._NORMS:
            raise ValueError(f"normalize must be one of {self._NORMS}, got {normalize!r}")
        if static not in self._STATICS:
            raise ValueError(f"static must be one of {self._STATICS}, got {static!r}")
        # C2 -- BROADENED linear-static guard: the untruncated Goursat scheme needs bounded
        # increments for ANY phi (plain sig-PDE on linear increments diverges just like dilation).
        if truncation is None and static == "linear":
            raise ValueError("untruncated (Goursat PDE) signature kernels require static='rbf' "
                             "(bounded increments |m|<=2); linear increments are unbounded and the "
                             "wavefront can diverge for any phi (const or dilation).")
        if phi == "dilation" and truncation is not None:
            raise ValueError("phi='dilation' is the untruncated mixture; set truncation=None.")
        if phi in ("level_one", "free") and truncation is None:
            raise ValueError(f"phi='{phi}' is truncated; set truncation to an int N.")
        if normalize == "per_level" and not (phi == "const" and truncation is not None):
            raise ValueError("normalize='per_level' (legacy per-level-average convention) is "
                             "defined only for phi='const', truncated -- the sig-EXACT column.")

        self.phi, self.truncation, self.normalize, self.static = phi, truncation, normalize, static
        self.bw, self.dev = bw, torch.device(dev)
        self.fit_unnormalized = fit_unnormalized
        self._fitted = phi in ("free", "dilation")

        if normalize == "per_level":                              # sig-EXACT: delegate to live lib
            from . import kernels as _k
            from .static.kernels import RBFKernel, LinearKernel
            sk = RBFKernel(bandwidth=bw) if static == "rbf" else LinearKernel()
            self._engine = _k.SignatureKernel(n_levels=truncation, order=1, normalize=True,
                                              static_kernel=sk)
            self._kind = "ksig"
        elif truncation is None and phi == "const":              # sig-PDE: live SignaturePDEKernel
            from . import kernels as _k
            from .static.kernels import RBFKernel
            self._engine = _k.SignaturePDEKernel(static_kernel=RBFKernel(bandwidth=bw),
                                                 normalize=True, difference=True)
            self._kind = "ksig"
        elif truncation is None and phi == "dilation":           # sig-PDEphi
            self._engine = LearnedPhiSignaturePDEKernel(m_nodes=m_nodes, lam_max=lam_max, bw=bw,
                                                        normalize=True, dev=str(dev),
                                                        auto_clamp=auto_clamp)
            self._kind = "gen"
        else:                                                    # truncated, once: const/level_one/free
            phi_fixed = None
            if phi == "const":
                phi_fixed = np.ones(truncation + 1, dtype=np.float32)            # phi==1, phi(0)=1
            elif phi == "level_one":
                phi_fixed = np.zeros(truncation + 1, dtype=np.float32)           # e_1, phi(0)=0 (F-A)
                phi_fixed[1] = 1.0
            self._engine = WeightedSignatureKernel(n_levels=truncation, bw=bw, normalize=True,
                                                   dev=str(dev), phi_fixed=phi_fixed)
            self._kind = "gen"

    def fit_phi(self, X, y, task="classification", inner_idx=None, **kw):
        """No-op for fixed-phi columns; dispatches to the engine for free / dilation."""
        if not self._fitted:
            return self
        if isinstance(self._engine, WeightedSignatureKernel):
            self._engine.fit_phi(X, y, task=task, inner_idx=inner_idx,
                                 fit_unnormalized=self.fit_unnormalized, **kw)
        else:
            self._engine.fit_phi(X, y, task=task, inner_idx=inner_idx, **kw)
        return self

    def phi_profile(self, kmax=None):
        """The interpretable artifact phi(k). const->1s; level_one->e_1; free/dilation->learned;
        per_level/SigPDE -> phi==1 (the constant weighting they realize)."""
        eng = self._engine
        if isinstance(eng, WeightedSignatureKernel):
            return eng.phi(kmax)
        if isinstance(eng, LearnedPhiSignaturePDEKernel):
            return eng.phi(kmax if kmax is not None else 12)
        N = self.truncation if self.truncation is not None else (kmax if kmax is not None else 12)
        return np.ones(N + 1, dtype=np.float32)

    def __call__(self, X, Y=None, diag=False, return_on_gpu=False):
        if self._kind == "ksig":
            return self._engine(X, Y, return_on_gpu=return_on_gpu)
        return self._engine(X, Y, diag=diag, return_on_gpu=return_on_gpu)


# ---------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import math, os
    dev = os.environ.get("KSIG_DEVICE", "cpu")
    rng = np.random.default_rng(0)
    n, L, d = 64, 24, 6
    y = rng.integers(0, 2, n)
    X = np.cumsum(rng.standard_normal((n, L, d)).astype(np.float32), 1) / math.sqrt(L)
    X *= (1 + 0.5 * y[:, None, None])
    X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-6)
    bw = 1.0

    print("== truncated, learnable phi (WeightedSignatureKernel) ==")
    wk = WeightedSignatureKernel(n_levels=5, bw=bw, dev=dev).fit_phi(X, y, steps=200)
    print("phi(0:5) =", np.round(wk.phi(), 4), " phi(0)==1:", np.isclose(wk.phi()[0], 1.0))
    K = wk(X); print("Gram", K.shape, "unit-diag:", np.allclose(np.diag(K), 1, atol=1e-4),
                     "finite:", np.isfinite(K).all())

    print("== untruncated, learnable phi (LearnedPhiSignaturePDEKernel) ==")
    pk = LearnedPhiSignaturePDEKernel(m_nodes=5, lam_max=0.5, bw=bw, dev=dev).fit_phi(X, y, steps=120)
    print("phi(0:8) =", np.round(pk.phi(8), 4), " phi(0)==1:", np.isclose(pk.phi(8)[0], 1.0))
    K = pk(X); print("Gram", K.shape, "unit-diag:", np.allclose(np.diag(K), 1, atol=1e-4),
                     "finite:", np.isfinite(K).all())

    print("== facade GeneralSignatureKernel ==")
    for cfg in [dict(phi="level_one", truncation=3),
                dict(phi="const", truncation=3),
                dict(phi="const", truncation=3, normalize="per_level"),
                dict(phi="const", truncation=None),
                dict(phi="free", truncation=3),
                dict(phi="dilation", truncation=None)]:
        gsk = GeneralSignatureKernel(bw=bw, dev=dev, **cfg).fit_phi(X, y, steps=60)
        K = gsk(X, X)
        print(f"  {cfg}: Gram{K.shape} finite={np.isfinite(K).all()} "
              f"phi0={gsk.phi_profile()[0]:.3f}")
