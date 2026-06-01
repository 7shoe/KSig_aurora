"""Hardware-independent brute-force NumPy DP oracles for the three dynamic-
programming kernels (SigPDE, GAK, RWS/DTW).

These are written **directly from the recurrences in ``docs/TORCH_PORT.md`` Sec. 4**
with plain Python ``for`` loops — no CuPy, no Numba, no torch, no vectorized
wavefront.  They are the ground truth the legacy kernels are validated against
(``test_algorithms_dp.py``) and, later, the oracle the torch port must match
when no golden ``.npz`` is present.

Everything runs in float64 on tiny inputs.  The padded-table convention mirrors
TORCH_PORT Sec. 4.1: data cell ``(i, j)`` lives at ``H[i+1, j+1]``.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# -----------------------------------------------------------------------------
# SigPDE  (TORCH_PORT Sec. 4.2)
# -----------------------------------------------------------------------------
def sig_pde_pair(M: np.ndarray, difference: bool = True) -> float:
    """Single sequence-pair SigPDE solve.

    Args:
      M: ``[l_X, l_Y]`` static-kernel grid for one pair.
      difference: take second differences of ``M`` first (the lifted increment).

    Recurrence (m = M[i, j]):
      K(i,j) = (K(i-1,j)+K(i,j-1))*(1 + m/2 + m^2/12) - K(i-1,j-1)*(1 - m^2/12)
    with K(i-1,j)=K(i,j-1)=K(i-1,j-1)=1 on the borders.
    """
    if difference:
        M = np.diff(np.diff(M, axis=-2), axis=-1)
    lX, lY = M.shape
    H = np.ones((lX + 1, lY + 1), dtype=np.float64)  # borders pre-filled with 1
    for i in range(lX):
        for j in range(lY):
            m = M[i, j]
            up = H[i, j + 1]
            left = H[i + 1, j]
            diag = H[i, j]
            H[i + 1, j + 1] = ((up + left) * (1.0 + 0.5 * m + m * m / 12.0)
                               - diag * (1.0 - m * m / 12.0))
    return float(H[lX, lY])


def sig_pde_gram(M: np.ndarray, difference: bool = True) -> np.ndarray:
    """Batched SigPDE over ``M`` of shape ``[n_X, n_Y, l_X, l_Y]`` (or
    ``[n, l_X, l_Y]`` for the diagonal)."""
    if M.ndim == 3:
        return np.array([sig_pde_pair(M[a], difference) for a in range(M.shape[0])])
    nX, nY = M.shape[:2]
    K = np.empty((nX, nY), dtype=np.float64)
    for a in range(nX):
        for b in range(nY):
            K[a, b] = sig_pde_pair(M[a, b], difference)
    return K


# -----------------------------------------------------------------------------
# GAK in log-space  (TORCH_PORT Sec. 4.3)
# -----------------------------------------------------------------------------
def _logsumexp3(a: float, b: float, c: float) -> float:
    m = max(a, b, c)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))


def gak_log_pair(M: np.ndarray) -> float:
    """Single-pair GAK log-kernel.

    Driver transform: M <- M/(2-M); logM <- log(clamp(M, _EPS)).
    Recurrence: logK(i,j) = logM(i,j) + logsumexp(logK(i-1,j), logK(i,j-1),
                                                   logK(i-1,j-1))
    Borders -inf; seed H(0,0)=0.
    """
    M = M / (2.0 - M)
    logM = np.log(np.clip(M, _EPS, None))
    lX, lY = M.shape
    H = np.full((lX + 1, lY + 1), -np.inf, dtype=np.float64)
    H[0, 0] = 0.0
    for i in range(lX):
        for j in range(lY):
            H[i + 1, j + 1] = logM[i, j] + _logsumexp3(
                H[i, j + 1], H[i + 1, j], H[i, j])
    return float(H[lX, lY])


def gak_log_gram(M: np.ndarray) -> np.ndarray:
    """Batched GAK log-kernel; returns the *log-space* result (pre-exp /
    pre-normalization), matching ``ksig.algorithms.global_align_kern_log``."""
    if M.ndim == 3:
        return np.array([gak_log_pair(M[a]) for a in range(M.shape[0])])
    nX, nY = M.shape[:2]
    K = np.empty((nX, nY), dtype=np.float64)
    for a in range(nX):
        for b in range(nY):
            K[a, b] = gak_log_pair(M[a, b])
    return K


# -----------------------------------------------------------------------------
# RWS / DTW  (TORCH_PORT Sec. 4.4)
# -----------------------------------------------------------------------------
def dtw_pair(D: np.ndarray) -> float:
    """Single-pair DTW accumulated cost over a ``[l_X, l_Y]`` local-cost matrix.

    Recurrence: P(i,j) = D(i,j) + min(P(i-1,j), P(i,j-1), P(i-1,j-1));
    borders +inf, seed P(0,0)=0.  ``min,+`` is exact (no rounding).
    """
    lX, lY = D.shape
    H = np.full((lX + 1, lY + 1), np.inf, dtype=np.float64)
    H[0, 0] = 0.0
    for i in range(lX):
        for j in range(lY):
            H[i + 1, j + 1] = D[i, j] + min(H[i, j + 1], H[i + 1, j], H[i, j])
    return float(H[lX, lY])


def rws_dtw(D: np.ndarray, warp_lens: np.ndarray) -> np.ndarray:
    """RWS over ``D`` of shape ``[n_X, l_X, sum(warp_lens)]`` and integer
    ``warp_lens`` of shape ``[n_Y]``.  Series ``y`` occupies columns
    ``[seg[y], seg[y+1])``; returns ``[n_X, n_Y]`` DTW costs.

    This mirrors the pad-and-gather contract: each series is read at its own
    true terminal column ``l_Y(y)``.
    """
    warp_lens = np.asarray(warp_lens).astype(np.int64)
    seg = np.concatenate([[0], np.cumsum(warp_lens)])
    nX, lX = D.shape[:2]
    nY = warp_lens.shape[0]
    P = np.empty((nX, nY), dtype=np.float64)
    for a in range(nX):
        for y in range(nY):
            Dxy = D[a, :, seg[y]:seg[y + 1]]    # [l_X, l_Y(y)]
            P[a, y] = dtw_pair(Dxy)
    return P
