"""Signature computation dynamic programming algorithms.

The three dynamic-programming kernels (SigPDE, GAK log-space, RWS/DTW) used to be
hand-written Numba ``@cuda.jit`` kernels (NVIDIA-only). They are now **vectorized
torch "wavefront" recurrences** (``docs/TORCH_PORT.md`` Sec. 4): a single Python
loop over antidiagonals, each step one batched elementwise op over the whole
``n_X * n_Y`` batch and the entire current antidiagonal. This is portable
(CUDA / XPU / MPS / CPU) and generally better GPU utilization than the old
per-pair-block kernels.

On Aurora XPU an optional native SYCL fast-path (``ksig._sycl``) can do the whole
DP in one launch / fuse the static-kernel evaluation; it is dispatched only when
``device.type == "xpu"`` and the extension is built, otherwise these torch
wavefronts run (and serve as the numerical oracle for the SYCL path).
"""

import os

import numpy as np
import torch

from .projections import (DiagonalProjection, RandomProjection,
                          TensorizedRandomProjection)
from .utils import _EPS, ArrayOnGPU, multi_cumsum
from .torch_backend import as_index, eps_for
from typing import List, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# Signature Algorithms.
# -----------------------------------------------------------------------------

def signature_kern(M: ArrayOnGPU, n_levels: int, order: int = -1,
                   difference: bool = True, return_levels: bool = False
                   ) -> ArrayOnGPU:
  """Computes the full-rank signature kernel using the kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the kernel for each level separately.

  Returns:
    The signature kernel matrix of shape `[n_X, n_Y]` or `[n]`, see `M` above.
  """
  order = n_levels if order <= 0 or order >= n_levels else order
  if order==1:
    return signature_kern_first_order(
      M, n_levels, difference=difference, return_levels=return_levels)
  else:
    return signature_kern_higher_order(
      M, n_levels, order=order, difference=difference,
      return_levels=return_levels)


def signature_kern_first_order(M: ArrayOnGPU, n_levels: int,
                               difference: bool = True,
                               return_levels: bool = False) -> ArrayOnGPU:
  """Computes the first-order full-rank signature kernel using a kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    n_levels: Number of signature levels.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the kernel for each level separately.

  Returns:
    The signature kernel matrix of shape `[..., n_X, n_Y]` or `[..., n]`,
      depending on `M` above, and `...` is `n_levels` when `return_levels`.
  """

  if difference:
    M = torch.diff(torch.diff(M, dim=-2), dim=-1)
  if M.ndim == 4:
    n_X, n_Y  = M.shape[:2]
    K = torch.ones((n_X, n_Y), dtype=M.dtype, device=M.device)
  else:
    n_X = M.shape[0]
    K = torch.ones((n_X,), dtype=M.dtype, device=M.device)

  if return_levels:
    K = [K, torch.sum(M, dim=(-2, -1))]
  else:
    K = K + torch.sum(M, dim=(-2, -1))

  R = M.clone()
  for i in range(1, n_levels):
    R = M * multi_cumsum(R, exclusive=True, axis=(-2, -1))
    if return_levels:
      K.append(torch.sum(R, dim=(-2, -1)))
    else:
      K = K + torch.sum(R, dim=(-2, -1))

  return torch.stack(K, dim=0) if return_levels else K


def signature_kern_higher_order(M: ArrayOnGPU, n_levels: int, order: int,
                                difference: bool = True,
                                return_levels: bool = False) -> ArrayOnGPU:
  """Computes the higher-order full rank signature kernel using a kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the kernel for each level separately.

  Returns:
    The signature kernel matrix of shape `[..., n_X, n_Y]` or `[..., n]`,
      depending on `M` above, and `...` is `n_levels` when `return_levels`.
  """

  if difference:
    M = torch.diff(torch.diff(M, dim=-2), dim=-1)

  if M.ndim == 4:
    n_X, n_Y = M.shape[0], M.shape[1]
    K = torch.ones((n_X, n_Y), dtype=M.dtype, device=M.device)
  else:
    n_X = M.shape[0]
    K = torch.ones((n_X,), dtype=M.dtype, device=M.device)

  if return_levels:
    K = [K, torch.sum(M, dim=(-2, -1))]
  else:
    K = K + torch.sum(M, dim=(-2, -1))

  R = M.clone()[None, None, ...]
  for i in range(1, n_levels):
    d = min(i+1, order)
    R_next = torch.empty((d, d) + tuple(M.shape), dtype=M.dtype,
                         device=M.device)
    # Both time axes are non-repeating.
    R_next[0, 0] = M * multi_cumsum(
      torch.sum(R, dim=(0, 1)), exclusive=True, axis=(-2, -1))
    for r in range(1, d):
      R_next[0, r] = 1./(r+1) * M * multi_cumsum(
        torch.sum(R[:, r-1], dim=0), exclusive=True, axis=-2)
      R_next[r, 0] = 1./(r+1) * M * multi_cumsum(
        torch.sum(R[r-1, :], dim=0), exclusive=True, axis=-1)
      for s in range(1, d):
        R_next[r, s] = 1./((r+1)*(s+1)) * M * R[r-1, s-1]
    R = R_next
    if return_levels:
      K.append(torch.sum(R, dim=(0, 1, -2, -1)))
    else:
      K = K + torch.sum(R, dim=(0, 1, -2, -1))

  return torch.stack(K, dim=0) if return_levels else K


# -----------------------------------------------------------------------------
# Low-Rank Signature Algorithms.
# -----------------------------------------------------------------------------

def signature_kern_low_rank(
  U: ArrayOnGPU, n_levels: int, order: int = -1, difference: bool = True,
  return_levels: bool = False,
  projections : Optional[List[RandomProjection]] = None
  ) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
  """Computes the low-rank signature kernel in feature space.

  Args:
    U: Transformed sequences of shape `[n_X, l_X, n_d]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the features for each level separately.
    projections: Random projections for the outer product approximation.

  Returns:
    The signature features Sig(X).
  """
  order = n_levels if order <= 0 or order >= n_levels else order
  if order==1:
    return signature_kern_first_order_low_rank(
      U, n_levels, difference=difference, return_levels=return_levels,
      projections=projections)
  else:
    return signature_kern_higher_order_low_rank(
      U, n_levels, order=order, difference=difference,
      return_levels=return_levels, projections=projections)


def signature_kern_first_order_low_rank(
  U: ArrayOnGPU, n_levels: int, difference: bool = True,
  return_levels: bool = False,
  projections: Optional[List[RandomProjection]] = None
  ) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
  """Computes the first-order low-rank signature kernel in feature space.

  Args:
    U: Transformed sequences of shape `[n_X, l_X, n_d]`.
    n_levels: Number of signature levels.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the features for each level separately.
    projections: Random projections for outer product approximation.

  Returns:
    The first-order signature features Sig(X).
  """

  if isinstance(U, list):
    if difference:
      U = [torch.diff(U[i], dim=1) for i in range(n_levels)]

    n_X, l_X, n_d = U[0].shape
    P = torch.ones((n_X, 1), dtype=U[0].dtype, device=U[0].device)
    R = (projections[0](U[0], return_on_gpu=True) if projections is not None
         else U[0].clone())
  else:
    if difference:
      U = torch.diff(U, dim=1)
    n_X, l_X, n_d = U.shape
    P = torch.ones((n_X, 1), dtype=U.dtype, device=U.device)
    R = (projections[0](U, return_on_gpu=True) if projections is not None else
         U.clone())

  if (projections is not None and
      isinstance(projections[0], TensorizedRandomProjection)):
    R_reshaped = R.reshape(
      [n_X, l_X, projections[0].n_components, projections[0].rank])
    R_sum = torch.sum(R_reshaped, dim=(1, -1))
  else:
    R_sum = torch.sum(R, dim=1)

  if return_levels:
    P = [P, R_sum.reshape([n_X, -1])]
  else:
    P = torch.cat((P, R_sum.reshape([n_X, -1])), dim=-1)

  for i in range(1, n_levels):
    R = multi_cumsum(R, axis=1, exclusive=True)
    if projections is None:
      if isinstance(U, list):
        R = torch.reshape(R[..., :, None] * U[i][..., None, :], (n_X, l_X, -1))
      else:
        R = torch.reshape(R[..., :, None] * U[..., None, :], (n_X, l_X, -1))
      R_sum = torch.sum(R, dim=1)
    else:
      if isinstance(U, list):
        R = projections[i](R, U[i], return_on_gpu=True)
      else:
        R = projections[i](R, U, return_on_gpu=True)
      if isinstance(projections[i], TensorizedRandomProjection):
        R_reshaped = R.reshape(
          [n_X, l_X, projections[i].n_components, projections[i].rank])
        R_sum = torch.sum(R_reshaped, dim=(1, -1))
      else:
        R_sum = torch.sum(R, dim=1)
    R_sum = R_sum.reshape([n_X, -1])
    if return_levels:
      P.append(R_sum)
    else:
      P = torch.cat((P, R_sum), dim=-1)
  return P

def signature_kern_higher_order_low_rank(
  U: ArrayOnGPU, n_levels: int, order: int = -1, difference: bool = True,
  return_levels: bool = False,
  projections: Optional[List[RandomProjection]] = None
  ) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
  """Computes the higher-order low-rank signature kernel in feature space.

  Args:
    U: Transformed sequences of shape `[n_X, l_X, n_d]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the features for each level separately.
    projections: Random projections for outer product approximation.

  Returns:
    The higher-order signature features Sig(X).
  """
  if isinstance(U, list):
    if difference:
      U = [torch.diff(U[i], dim=1) for i in range(n_levels)]
    n_X, l_X = U[0].shape[:2]
    n_d = U[0].shape[-1]
    dtype, device = U[0].dtype, U[0].device
    P = torch.ones((n_X, 1), dtype=dtype, device=device)
    R = (projections[0](U[0], return_on_gpu=True) if projections is not None
         else U[0].clone())
  else:
    if difference:
      U = torch.diff(U, dim=1)
    n_X, l_X = U.shape[:2]
    n_d = U.shape[-1]
    dtype, device = U.dtype, U.device
    P = torch.ones((n_X, 1), dtype=dtype, device=device)
    R = (projections[0](U, return_on_gpu=True) if projections is not None else
         U.clone())

  if (projections is not None and
      isinstance(projections[0], TensorizedRandomProjection)):
    R_reshaped = R.reshape(
      [n_X, l_X, projections[0].n_components, projections[0].rank])
    R_sum = torch.sum(R_reshaped, dim=(1, -1))
  else:
    R_sum = torch.sum(R, dim=1)

  R_sum = R_sum.reshape([n_X, -1])
  if return_levels:
    P = [P, R_sum]
  else:
    P = torch.cat((P, R_sum), dim=-1)

  R = R[None]
  for i in range(1, n_levels):
    d = min(i+1, order)
    n_components = R.shape[-1] if projections is not None else n_d**(i+1)
    if (projections is not None and
        isinstance(projections[i], DiagonalProjection)):
      internal_size = projections[i].internal_size
      R_next = torch.empty((d, n_X, l_X, internal_size**(i+1), n_components),
                           dtype=dtype, device=device)
    else:
      R_next = torch.empty((d, n_X, l_X, n_components), dtype=dtype,
                           device=device)
    U_next = U[i] if isinstance(U, list) else U
    Q = multi_cumsum(torch.sum(R, dim=0), axis=1, exclusive=True)
    if projections is None:
      R_next[0] = torch.reshape(
        Q[..., :, None] * U_next[..., None, :], (n_X, l_X, -1))
    else:
      R_next[0] = projections[i](Q, U_next, return_on_gpu=True)
    if projections is None:
      for r in range(1, d):
        R_next[r] = 1./(r+1) * torch.reshape(
          R[r-1, ..., :, None] * U_next[..., None, :],
          (n_X, l_X, n_components))
    else:
      for r in range(1, d):
        R_next[r] = 1./(r+1) * projections[i](
          R[r-1], U_next, return_on_gpu=True)
    R = R_next
    if (projections is not None and
        isinstance(projections[i], TensorizedRandomProjection)):
      R_reshaped = R.reshape(
        [d, n_X, l_X, projections[i].n_components, projections[i].rank])
      R_sum = torch.sum(R_reshaped, dim=(0, 2, -1))
    else:
      R_sum = torch.sum(R, dim=(0, 2))
    R_sum = R_sum.reshape([n_X, -1])
    if return_levels:
      P.append(R_sum)
    else:
      P = torch.cat((P, R_sum), dim=-1)
  return P


# -----------------------------------------------------------------------------
# Vectorized antidiagonal "wavefront" DP (shared by SigPDE / GAK / RWS).
# -----------------------------------------------------------------------------

def _antidiag_indices(it: int, l_X: int, l_Y: int, device
                      ) -> Tuple[ArrayOnGPU, ArrayOnGPU]:
  """Row/column index tensors of the cells on antidiagonal `it` (i + j == it).

  Returns `(i, j)`, each of shape `[d]`, with `0 <= i < l_X`, `0 <= j < l_Y`.
  """
  i_lo = max(0, it - (l_Y - 1))
  i_hi = min(it, l_X - 1)
  i = torch.arange(i_lo, i_hi + 1, device=device)
  j = it - i
  return i, j


def _sycl_enabled() -> bool:
  """Whether the SYCL fast-path is allowed to engage.

  Controlled by the ``KSIG_USE_SYCL`` env var (read per-call so benchmarks can
  flip it between processes). Default is on: SYCL auto-engages whenever the ext
  is built and the inputs are on XPU. Set ``KSIG_USE_SYCL=0`` (also ``false`` /
  ``no`` / ``off``) to force the torch wavefront -- this is how the ``monitoring/``
  acceptance gate (SYCL_HANDOFF.md Sec. 7) measures the torch-XPU baseline
  against the SYCL fast-path: run the suite twice, ``KSIG_USE_SYCL=0`` then ``=1``.
  """
  val = os.environ.get('KSIG_USE_SYCL')
  if val is None:
    return True
  return val.strip().lower() not in ('0', 'false', 'no', 'off', '')


def _try_sycl(name: str, *args):
  """Dispatch to the native SYCL fast-path when on XPU and the ext is built.

  Returns the SYCL result, or ``None`` to signal "fall through to the torch
  wavefront". Any import/availability/runtime error falls through silently —
  the torch wavefront is always correct and is the numerical oracle.
  """
  if not _sycl_enabled():
    return None
  first = args[0]
  if not (isinstance(first, torch.Tensor) and first.device.type == 'xpu'):
    return None
  try:
    from ._sycl import loader
    if not loader.available():
      return None
    return getattr(loader.get_ext(), name)(*args)
  except Exception:
    return None


# -----------------------------------------------------------------------------
# Signature-PDE Kernel.
# -----------------------------------------------------------------------------

def signature_kern_pde(M: ArrayOnGPU, difference: bool = True) -> ArrayOnGPU:
  """Computes the signature-PDE kernel using a kernel trick.

  Vectorized antidiagonal wavefront (TORCH_PORT Sec. 4.2). With `m = M[..,i,j]`:
    K(i,j) = (K(i-1,j) + K(i,j-1)) * (1 + m/2 + m^2/12)
             - K(i-1,j-1) * (1 - m^2/12)
  with all border neighbors equal to 1 (encoded as the padded table borders).

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    difference: Whether to take increments of lifted sequences in the RKHS.

  Returns:
    The SigPDE kernel matrix of shape `[n_X, n_Y]` or `[n]`, see `M` above.
  """
  is_diag = (M.ndim == 3)
  if is_diag:
    M = M[:, None]
  if M.ndim != 4:
    raise ValueError('The `M` matrix must have `.ndim==3` or `.ndim==4`.')

  sycl = _try_sycl('sig_pde', M.contiguous(), difference)
  if sycl is not None:
    return sycl.squeeze(1) if is_diag else sycl

  if difference:
    M = torch.diff(torch.diff(M, dim=-2), dim=-1)
  nX, nY, lX, lY = M.shape
  dev = M.device
  # Padded table: data cell (i, j) lives at H[.., i+1, j+1]; borders are 1.
  H = torch.ones((nX, nY, lX + 1, lY + 1), dtype=M.dtype, device=dev)
  for it in range(lX + lY - 1):
    i, j = _antidiag_indices(it, lX, lY, dev)
    up   = H[:, :, i,     j + 1]
    left = H[:, :, i + 1, j]
    diag = H[:, :, i,     j]
    m    = M[:, :, i, j]
    H[:, :, i + 1, j + 1] = (
      (up + left) * (1 + 0.5 * m + m * m / 12) - diag * (1 - m * m / 12))
  K = H[:, :, lX, lY]
  return K.squeeze(1) if is_diag else K


# -----------------------------------------------------------------------------
# Global Alignment Kernel.
# -----------------------------------------------------------------------------

def global_align_kern_log(M: ArrayOnGPU) -> ArrayOnGPU:
  """Computes the (log-space) Global Alignment Kernel.

  Vectorized antidiagonal wavefront (TORCH_PORT Sec. 4.3):
    logK(i,j) = logM(i,j) + logsumexp(logK(i-1,j), logK(i,j-1), logK(i-1,j-1))
  borders -inf, corner seed logK(-1,-1)=0. The driver transform is
  `M <- M/(2-M)` then `logM = log(clamp(M, _EPS))`.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.

  Returns:
    The (log-space) GA kernel matrix of shape `[n_X, n_Y]` or `[n]`.
  """
  is_diag = (M.ndim == 3)
  if is_diag:
    M = M[:, None]
  if M.ndim != 4:
    raise ValueError('The `M` matrix must have `.ndim==3` or `.ndim==4`.')

  sycl = _try_sycl('gak_log', M.contiguous())
  if sycl is not None:
    return sycl.squeeze(1) if is_diag else sycl

  # Transform `M` to make it "infinitely divisible", then work in log-space.
  M = M / (2. - M)
  logM = torch.log(torch.clamp(M, min=eps_for(M.dtype)))
  nX, nY, lX, lY = M.shape
  dev = M.device
  H = torch.full((nX, nY, lX + 1, lY + 1), float('-inf'), dtype=M.dtype,
                 device=dev)
  H[:, :, 0, 0] = 0.
  for it in range(lX + lY - 1):
    i, j = _antidiag_indices(it, lX, lY, dev)
    up   = H[:, :, i,     j + 1]
    left = H[:, :, i + 1, j]
    diag = H[:, :, i,     j]
    H[:, :, i + 1, j + 1] = logM[:, :, i, j] + torch.logsumexp(
      torch.stack([up, left, diag], dim=0), dim=0)
  logK = H[:, :, lX, lY]
  return logK.squeeze(1) if is_diag else logK


# -----------------------------------------------------------------------------
# Random Warping Series.
# -----------------------------------------------------------------------------

def random_warping_series(D: ArrayOnGPU, warp_lens: ArrayOnGPU) -> ArrayOnGPU:
  """Computes the (log of) Random Warping Series features via DTW.

  Vectorized antidiagonal wavefront (TORCH_PORT Sec. 4.4):
    P(i,j) = D(i,j) + min(P(i-1,j), P(i,j-1), P(i-1,j-1))
  borders +inf, corner seed 0. Variable warp lengths are handled by a
  pad-and-gather over a padded `[n_X, n_Y, l_X, l_Y_max]` cost table; each series
  is read at its own true terminal column `l_Y(y)`.

  Args:
    D: Squared distances of shape `[n_X, l_X, sum of l_Y]`.
    warp_lens: Lengths of each warping series, array of shape `[n_Y]`.

  Returns:
    Random Warping Series features of shape `[n_X, n_Y]`.

  Raises:
    ValueError: If `D` does not have `.ndim==3`.
  """
  if D.ndim != 3:
    raise ValueError('`D` distances array must have `.ndim==3`.')
  dev = D.device
  warp_lens = as_index(warp_lens, device=dev)

  sycl = _try_sycl('rws_dtw', D.contiguous(), warp_lens)
  if sycl is not None:
    return sycl

  nX, lX = D.shape[:2]
  nY = warp_lens.shape[0]
  seg = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                   torch.cumsum(warp_lens, 0)])        # [nY + 1]
  lY = seg[1:] - seg[:-1]                              # [nY]
  lY_max = int(lY.max())
  # Scatter D -> Dpad[x, y, i, 0:lY(y)]; right-pad with +inf (never read by the
  # min,+ recurrence at the valid terminal columns).
  Dpad = torch.full((nX, nY, lX, lY_max), float('inf'), dtype=D.dtype,
                    device=dev)
  for y in range(nY):
    Dpad[:, y, :, :int(lY[y])] = D[:, :, int(seg[y]):int(seg[y+1])]
  H = torch.full((nX, nY, lX + 1, lY_max + 1), float('inf'), dtype=D.dtype,
                 device=dev)
  H[:, :, 0, 0] = 0.
  for it in range(lX + lY_max - 1):
    i, j = _antidiag_indices(it, lX, lY_max, dev)
    up   = H[:, :, i,     j + 1]
    left = H[:, :, i + 1, j]
    diag = H[:, :, i,     j]
    H[:, :, i + 1, j + 1] = Dpad[:, :, i, j] + torch.minimum(
      torch.minimum(up, left), diag)
  # Gather each series at its own terminal column l_Y(y).
  term = lY.view(1, nY, 1).expand(nX, nY, 1)
  P = H[:, :, lX, :].gather(-1, term).squeeze(-1)      # [nX, nY]
  return P


# -----------------------------------------------------------------------------
