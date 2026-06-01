"""Brute-force NumPy oracles for the signature kernel.

Two independent layers of ground truth:

1. :func:`sig_kernel_bruteforce` — the **first-principles** truth for the
   first-order, unnormalized signature kernel: an explicit sum over strictly
   increasing index tuples
       k_m(x,y) = sum_{i_1<...<i_m, j_1<...<j_m} prod_t dM[i_t, j_t]
   where ``dM`` is the doubly-differenced static Gram.  This is exponential in
   ``m`` (O(L^{2m})) so it is only used for tiny ``n_levels<=3, L<=5``.

2. :func:`signature_kernel` — a clean NumPy re-implementation of the
   Kiraly-Oberhauser recursion used by ``ksig.algorithms.signature_kern``,
   covering normalization, higher order, the diagonal, and RBF/linear static
   kernels.  Layer 2 is validated against layer 1 on the core case, so trust
   propagates outward.

Both are hardware-independent (pure NumPy, float64).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

_EPS = 1e-12


# -----------------------------------------------------------------------------
# Static Gram over time points (matches ksig._compute_embedding).
# -----------------------------------------------------------------------------
def static_gram(X: np.ndarray, Y: np.ndarray | None = None,
                static: str = "rbf", bandwidth: float = 1.0,
                diag: bool = False) -> np.ndarray:
    """Pairwise static-kernel evaluations between time points.

    Returns ``[n_X, l_X, l_Y]`` if ``diag`` else ``[n_X, n_Y, l_X, l_Y]``.
    """
    if diag:
        # M[a, i, j] = k(X[a, i], X[a, j])
        A = X[:, :, None, :]
        B = X[:, None, :, :]
        return _static(A, B, static, bandwidth)
    if Y is None:
        Y = X
    # M[a, b, i, j] = k(X[a, i], Y[b, j])
    A = X[:, None, :, None, :]
    B = Y[None, :, None, :, :]
    return _static(A, B, static, bandwidth)


def _static(A, B, static, bandwidth):
    if static == "linear":
        return np.sum(A * B, axis=-1)
    if static == "rbf":
        d2 = np.sum((A - B) ** 2, axis=-1)
        return np.exp(-d2 / max(2.0 * bandwidth ** 2, _EPS))
    raise ValueError(static)


# -----------------------------------------------------------------------------
# Layer 1: explicit iterated-sum brute force (first order, unnormalized).
# -----------------------------------------------------------------------------
def _sig_pair_bruteforce(dM: np.ndarray, n_levels: int) -> float:
    """dM is the differenced ``[l_X-1, l_Y-1]`` increment Gram for one pair."""
    lX, lY = dM.shape
    total = 1.0  # level 0
    for m in range(1, n_levels + 1):
        if m > lX or m > lY:
            break
        acc = 0.0
        for I in combinations(range(lX), m):
            for J in combinations(range(lY), m):
                acc += float(np.prod([dM[I[t], J[t]] for t in range(m)]))
        total += acc
    return total


def sig_kernel_bruteforce(X: np.ndarray, Y: np.ndarray | None = None,
                          n_levels: int = 3, static: str = "linear",
                          bandwidth: float = 1.0) -> np.ndarray:
    """First-principles first-order unnormalized signature kernel (tiny only)."""
    M = static_gram(X, Y, static, bandwidth)        # [nX, nY, lX, lY]
    dM = np.diff(np.diff(M, axis=-2), axis=-1)      # increments
    nX, nY = dM.shape[:2]
    K = np.empty((nX, nY), dtype=np.float64)
    for a in range(nX):
        for b in range(nY):
            K[a, b] = _sig_pair_bruteforce(dM[a, b], n_levels)
    return K


# -----------------------------------------------------------------------------
# Layer 2: the recursion (matches ksig.algorithms.signature_kern).
# -----------------------------------------------------------------------------
def _cumsum_excl(R: np.ndarray, axes) -> np.ndarray:
    """Exclusive cumulative sum over the given axes (strictly-below-left
    rectangle for axes=(-2,-1)).  Shape preserved; the first slice is 0.

    Handles size-0 (L=1 -> differenced axis is empty) and size-1 axes, which is
    where the L=1 recurrence base case lives.
    """
    if np.isscalar(axes):
        axes = [axes]
    out = R
    for ax in axes:
        n = out.shape[ax]
        shifted = np.zeros_like(out)
        if n > 1:
            c = np.cumsum(out, axis=ax)
            src = [slice(None)] * out.ndim
            src[ax] = slice(0, n - 1)        # c[..., :-1, ...]
            dst = [slice(None)] * out.ndim
            dst[ax] = slice(1, n)            # -> shifted[..., 1:, ...]
            shifted[tuple(dst)] = c[tuple(src)]
        out = shifted                         # n<=1 -> all zeros (exclusive)
    return out


def _signature_kern_recursion(M: np.ndarray, n_levels: int, order: int,
                              difference: bool, return_levels: bool):
    """NumPy port of ``signature_kern`` (first + higher order)."""
    order = n_levels if order <= 0 or order >= n_levels else order
    if difference:
        M = np.diff(np.diff(M, axis=-2), axis=-1)
    is_gram = (M.ndim == 4)
    lead = M.shape[:2] if is_gram else M.shape[:1]
    K0 = np.ones(lead, dtype=np.float64)

    if order == 1:
        levels = [K0, np.sum(M, axis=(-2, -1))]
        R = M.copy()
        for _ in range(1, n_levels):
            R = M * _cumsum_excl(R, (-2, -1))
            levels.append(np.sum(R, axis=(-2, -1)))
    else:
        levels = [K0, np.sum(M, axis=(-2, -1))]
        R = M[None, None, ...].copy()
        for i in range(1, n_levels):
            d = min(i + 1, order)
            R_next = np.empty((d, d) + M.shape, dtype=np.float64)
            R_next[0, 0] = M * _cumsum_excl(np.sum(R, axis=(0, 1)), (-2, -1))
            for r in range(1, d):
                R_next[0, r] = (1.0 / (r + 1)) * M * _cumsum_excl(
                    np.sum(R[:, r - 1], axis=0), -2)
                R_next[r, 0] = (1.0 / (r + 1)) * M * _cumsum_excl(
                    np.sum(R[r - 1, :], axis=0), -1)
                for s in range(1, d):
                    R_next[r, s] = (1.0 / ((r + 1) * (s + 1))) * M * R[r - 1, s - 1]
            R = R_next
            levels.append(np.sum(R, axis=(0, 1, -2, -1)))

    if return_levels:
        return np.stack(levels, axis=0)
    return np.sum(np.stack(levels, axis=0), axis=0)


def signature_kernel(X: np.ndarray, Y: np.ndarray | None = None,
                     n_levels: int = 4, order: int = 1,
                     difference: bool = True, normalize: bool = True,
                     static: str = "rbf", bandwidth: float = 1.0,
                     diag: bool = False) -> np.ndarray:
    """Full NumPy oracle for ``ksig.kernels.SignatureKernel`` (the ``_K`` /
    ``_Kdiag`` pipeline incl. per-level normalization + averaging)."""
    if diag:
        if normalize:
            return np.ones(X.shape[0], dtype=np.float64)
        M = static_gram(X, None, static, bandwidth, diag=True)
        return _signature_kern_recursion(M, n_levels, order, difference, False)

    M = static_gram(X, Y, static, bandwidth)
    K = _signature_kern_recursion(M, n_levels, order, difference,
                                  return_levels=normalize)
    if normalize:
        # normalize each level by sqrt of its own diagonal, then average levels
        if Y is None:
            diag_vals = np.diagonal(K, axis1=-2, axis2=-1)        # [lvl, n]
            s = np.maximum(np.sqrt(np.maximum(diag_vals, 0.0)), _EPS)
            K = K / (s[..., :, None] * s[..., None, :])
        else:
            dX = signature_kernel_levels_diag(X, n_levels, order, difference,
                                              static, bandwidth)
            dY = signature_kernel_levels_diag(Y, n_levels, order, difference,
                                              static, bandwidth)
            sX = np.maximum(np.sqrt(np.maximum(dX, 0.0)), _EPS)
            sY = np.maximum(np.sqrt(np.maximum(dY, 0.0)), _EPS)
            K = K / (sX[..., :, None] * sY[..., None, :])
        K = np.mean(K, axis=0)
    return K


def signature_kernel_levels_diag(X, n_levels, order, difference, static,
                                 bandwidth):
    """Per-level diagonal ``[n_levels+1, n]`` used for cross (X,Y)
    normalization."""
    M = static_gram(X, None, static, bandwidth, diag=True)
    lvls = _signature_kern_recursion(M, n_levels, order, difference,
                                     return_levels=True)
    return lvls  # [n_levels+1, n]
