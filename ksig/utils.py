"""Validation, linear algebra and probability utilities."""

import numpy as np
import torch

from numbers import Number
from typing import Optional, Sequence, Tuple, Union

from . import torch_backend as tb
from .torch_backend import (  # noqa: F401  (public re-exports, see TORCH_PORT Sec. 3)
    _EPS, RandomStateOrSeed, TorchRandomState, as_index, as_tensor,
    check_random_state, eps_for, to_numpy)


# Array-type aliases. The whole library now speaks torch; ``ArrayOnGPU`` becomes
# ``torch.Tensor`` (this is the flag the test harness keys backend detection on).
ArrayOnCPU = np.ndarray
ArrayOnGPU = torch.Tensor
ArrayOnCPUOrGPU = Union[torch.Tensor, np.ndarray]


# -----------------------------------------------------------------------------
# Type checking.
# -----------------------------------------------------------------------------

def check_positive_value(scalar: Number, name: str) -> Number:
  """Checks whether `scalar` is a positive number.

  Args:
    scalar: A variable to check.
    name: The name of the variable.

  Returns:
    The variable unchanged or raises an error if it is not positive.
  """
  if scalar <= 0:
    raise ValueError(f'The parameter \'{name}\' should have a positive value.')
  return scalar


# -----------------------------------------------------------------------------
# Linear Algebra.
# -----------------------------------------------------------------------------

def multi_cumsum(M: ArrayOnGPU, exclusive: bool = False, axis: int = -1
         ) -> ArrayOnGPU:
  """Computes the cumulative sum along a given set of axes.

  Args:
    M: A data array.
    axis: An axis or a set of axes.
  """

  ndim = M.ndim
  axis = [axis] if np.isscalar(axis) else axis
  axis = [ndim+ax if ax < 0 else ax for ax in axis]

  if exclusive:
    # Slice off last element.
    slices = tuple(
      slice(-1) if ax in axis else slice(None) for ax in range(ndim))
    M = M[slices]

  for ax in axis:
    M = torch.cumsum(M, dim=ax)

  if exclusive:
    # Pre-pad with a leading zero along each cumsum axis (the exclusive shift).
    # We prepend a zero-slice per axis with ``cat`` rather than a single flat
    # ``F.pad`` call: ``F.pad`` (constant) raises on empty tensors when a
    # zero-size trailing axis isn't padded -- which happens for the ``L=1`` +
    # ``difference`` base case where the increment grid is empty. ``cat`` is
    # empty-safe and matches numpy's ``pad((1, 0))`` semantics exactly.
    # Iterate physical axes and test membership (as the legacy code did), so a
    # degenerate out-of-range ``axis`` reduces to a plain cumsum, not a pad.
    for ax in range(ndim):
      if ax in axis:
        shape = list(M.shape)
        shape[ax] = 1
        zeros = torch.zeros(shape, dtype=M.dtype, device=M.device)
        M = torch.cat([zeros, M], dim=ax)

  return M


def matrix_diag(A: ArrayOnGPU) -> ArrayOnGPU:
  """Extracts the diagonals from a batch of matrices.

  Args:
    A: A batch of matrices of shape `[..., d, d]`.

  Returns:
    The extracted diagonals of shape `[..., d]`.
  """
  return torch.einsum('...ii->...i', A)


def matrix_mult(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                transpose_X: bool = False, transpose_Y: bool = False
                ) -> ArrayOnGPU:
  """Performs batch matrix multiplication.

  Args:
    X: A batch of matrices.
    Y: Another batch of matrices (if not given uses `X`).
    transpose_X: Whether to transpose `X`.
    transpose_Y: Whether to transpose `Y`.

  Returns:
    The result of matrix multiplication, another batch of matrices.
  """
  subscript_X = '...ji' if transpose_X else '...ij'
  subscript_Y = '...kj' if transpose_Y else '...jk'
  return torch.einsum(
    f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)


def squared_norm(X: ArrayOnGPU, axis: int = -1) -> ArrayOnGPU:
  """Computes the squared norm by reducing over a given axis.

  Args:
    X: An n-dim. array to compute the norm of.
    axis: An axis to perform the reduction over.

  Returns:
    An (n-1)-dim. array containing the squared norms.
  """
  return torch.sum(torch.square(X), dim=axis)


def squared_euclid_dist(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None
                        ) -> ArrayOnGPU:
  """Computes pairwise squared Euclidean distances.

  Args:
    X: An array of shape `[..., m, d]`.
    Y: Another array of shape `[..., n, d]`. Uses `X` if not given.

  Returns:
    An array of shape `[..., m, n]`.
  """
  X_n2 = squared_norm(X, axis=-1)
  if Y is None:
    D2 = (X_n2[..., :, None] + X_n2[..., None, :]
          - 2 * matrix_mult(X, X, transpose_Y=True))
    # Cancellation can drive nominally-nonnegative distances slightly below 0.
    D2 = torch.clamp(D2, min=0.)
    # The self-distance diagonal is exactly zero; einsum vs. squared_norm
    # accumulate the `||x||^2 + ||x||^2 - 2<x,x>` terms in different orders, so
    # the diagonal carries a ~1e-15 residual. Harmless for exp(-d^2) kernels but
    # `sqrt` amplifies it to ~1e-7, breaking the Matern12/32 unit diagonal.
    torch.diagonal(D2, dim1=-2, dim2=-1).zero_()
    return D2
  Y_n2 = squared_norm(Y, axis=-1)
  D2 = (X_n2[..., :, None] + Y_n2[..., None, :]
        - 2 * matrix_mult(X, Y, transpose_Y=True))
  return torch.clamp(D2, min=0.)


def outer_prod(X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
  """Computes the outer product of two batch of vectors along the last axes.

  Args:
    X: A batch of vectors of shape `[..., d1]`.
    Y: A batch of vectors of shape `[..., d2]`.

  Returns:
    A batch of vectors of shape `[..., d1 * d2]`.
  """
  return torch.reshape(X[..., :, None] * Y[..., None, :],
                       tuple(X.shape[:-1]) + (-1,))


def robust_sqrt(X: ArrayOnGPU) -> ArrayOnGPU:
  """Robust elementwise square root.

  Clamps at zero (not a positive epsilon): the inputs are already nonnegative
  (``squared_euclid_dist``/``squared_norm`` clamp to ``>= 0``), and a positive
  floor would turn an exact zero distance into a ``sqrt(eps)`` offset that the
  Matern12/32 exponentials carry as a first-order ``~1e-6`` error on the unit
  diagonal -- the textbook (golden) value there is exactly 1.

  Args:
      X: An array to take the elementwise square root of.

  Returns:
      An array of the same shape.
  """
  return torch.sqrt(torch.clamp(X, min=0.))


def euclid_dist(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
  """Computes pairwise Euclidean distances.

  Args:
    X: An array of shape `[..., m, d]`.
    Y: Another array of shape `[..., n, d]`. Uses `X` if not given.

  Returns:
    An array of shape `[..., m, n]`.
  """
  return robust_sqrt(squared_euclid_dist(X, Y))


def robust_nonzero(X: ArrayOnGPU) -> ArrayOnGPU:
  """Robust elementwise nonzero check.

  Args:
      X: An array to check the elements of.

  Returns:
      A boolean array of the same shape.
  """
  return torch.abs(X) > _EPS


# -----------------------------------------------------------------------------
# Probability.
# -----------------------------------------------------------------------------

def draw_rademacher_matrix(shape: Sequence[int], prob: float = 0.5,
                           random_state: Optional[RandomStateOrSeed] = None
                           ) -> ArrayOnGPU:
  """Draw a random matrix with i.i.d. Rademacher entries.

  Args:
    shape: Shape of the matrix.
    prob: Probability of an entry being 1.
    random_state: A `TorchRandomState` or an integer seed or `None`.

  Returns:
    A matrix of shape `shape`.
  """
  random_state = check_random_state(random_state)
  u = random_state.uniform(size=shape)
  return torch.where(u < prob, torch.ones_like(u), -torch.ones_like(u))


def draw_bernoulli_matrix(shape: Sequence[int], prob: float = 0.5,
                          random_state: Optional[RandomStateOrSeed] = None
                          ) -> ArrayOnGPU:
  """Draw a random matrix with i.i.d. Bernoulli entries.

  Args:
    shape: Shape of the matrix.
    prob: Probability of an entry being 1.
    random_state: A `TorchRandomState` or an integer seed or `None`.

  Returns:
    A matrix of shape `shape`.
  """
  random_state = check_random_state(random_state)
  u = random_state.uniform(size=shape)
  return torch.where(u < prob, torch.ones_like(u), torch.zeros_like(u))


# -----------------------------------------------------------------------------
# Projection utils.
# -----------------------------------------------------------------------------

def subsample_outer_prod(X: ArrayOnGPU, Y: ArrayOnGPU,
                          sampled_idx: Union[ArrayOnGPU, Sequence[int]]
                          ) -> ArrayOnGPU:
  """Computes a subsampled outer product of two batch of features.

  Args:
    X: A data array.
    Y: An optional data array.
    sampled_idx: Indices to sample from the Cartesian product.

  Returns:
    The outer product of `X` and `Y` subsampled.
  """
  dev = X.device
  idx_X = torch.arange(X.shape[-1], device=dev).reshape([-1, 1, 1])
  idx_Y = torch.arange(Y.shape[-1], device=dev).reshape([1, -1, 1])
  idx_pairs = torch.reshape(torch.cat(
    (idx_X + torch.zeros_like(idx_Y), idx_Y + torch.zeros_like(idx_X)),
    dim=-1), (-1, 2))
  sampled_idx = as_index(sampled_idx, device=dev)
  sampled_idx_pairs = torch.squeeze(
    torch.index_select(idx_pairs, 0, sampled_idx))
  X_proj = torch.index_select(X, -1, sampled_idx_pairs[:, 0])
  Y_proj = torch.index_select(Y, -1, sampled_idx_pairs[:, 1])
  return X_proj * Y_proj


def compute_count_sketch(X: ArrayOnGPU, hash_idx: ArrayOnGPU,
                         hash_bit: ArrayOnGPU,
                         n_components: Optional[int] = None) -> ArrayOnGPU:
  """Computes the count sketch of a feature array.

  Args:
    X: A data array.
    hash_idx: The hash indices.
    hash_bit: The hash bits.
    n_components: The number of sketch components.

  Returns:
    Sketched features.
  """
  # If `n_components is None`, get it from `hash_idx`.
  n_components = n_components or int(torch.max(hash_idx))
  hash_mask = (hash_idx[:, None]
               == torch.arange(n_components, device=X.device)[None, :]).to(
                 dtype=X.dtype)
  X_count_sketch = torch.einsum('...i,ij,i->...j', X, hash_mask, hash_bit)
  return X_count_sketch


def convolve_fft(X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
  """Convolves two feature arrays via FFT.

  Args:
    X: A data array.
    Y: An optional data array.

  Returns:
    Convolved features."""
  X_fft = torch.fft.fft(X, dim=-1)
  Y_fft = torch.fft.fft(Y, dim=-1)
  XY = torch.real(torch.fft.ifft(X_fft * Y_fft, dim=-1))
  return XY


# -----------------------------------------------------------------------------
