"""Sequential data preprocessing utilities."""

import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union

from .utils import ArrayOnCPUOrGPU, ArrayOnGPU, check_positive_value
from .torch_backend import as_tensor, to_numpy


# -----------------------------------------------------------------------------
# Helpers for the torch/numpy dispatch.
#
# ``torch`` has no ``interp`` / ``apply_along_axis``; per ``docs/TORCH_PORT.md``
# Sec. 5.2 the interpolation paths are done on the host with numpy and moved
# back to a tensor when the input was a tensor. The non-interp ops dispatch to
# torch or numpy depending on the input array type so the output type matches.
# -----------------------------------------------------------------------------

def _is_torch(x) -> bool:
  return isinstance(x, torch.Tensor)


def _interp_axis0(x_np: np.ndarray, target_len: int) -> np.ndarray:
  """Channel-wise linear interpolation of a `[L, d]` sequence to `target_len`."""
  return np.stack(
    [np.interp(np.linspace(0, 1, target_len),
               np.linspace(0, 1, x_np.shape[0]), x_np[:, i_c])
     for i_c in range(x_np.shape[1])], axis=1)


# -----------------------------------------------------------------------------

class SequenceTabulator(BaseEstimator, TransformerMixin):
  """Transformer that tabulates sequences to even length."""

  def __init__(self, max_len: Optional[int] = None):
    """Initializes the `SequenceTabulator` object.

    Args:
      max_len: Maximum length of sequences.
    """
    self.max_len = (check_positive_value(max_len, 'max_len')
                    if max_len is not None else None)

  def fit(self, X_seq: Union[ArrayOnCPUOrGPU, List[ArrayOnCPUOrGPU]]
          ) -> 'SequenceTabulator':
    """Fits the `SequenceTabulator` to the data and returns the fitted object.

    Args:
      X_seq: An array or list of sequences on CPU or GPU.
    """
    max_seq_len = int(np.max([x.shape[0] for x in X_seq]))
    self.max_len_ = (min(self.max_len, max_seq_len)
                     if self.max_len is not None else max_seq_len)
    return self

  def transform(self, X_seq: Union[ArrayOnCPUOrGPU, List[ArrayOnCPUOrGPU]]
                ) -> ArrayOnCPUOrGPU:
    """Tabulates sequences contained in `X_seq` to uniform length.

    Args:
      X_seq: An array or list of sequences on CPU or GPU.

    Returns:
      A tabulated array of sequences on CPU or GPU.
    """
    was_torch = _is_torch(X_seq[0]) if len(X_seq) else _is_torch(X_seq)
    device = X_seq[0].device if was_torch else None
    # Interpolation has no torch equivalent: work on the host in numpy.
    seqs = [to_numpy(x) for x in X_seq]
    needs_interp = any(
      x.shape[0] != seqs[0].shape[0] or np.any(np.isnan(x))
      or x.shape[0] > self.max_len_ for x in seqs)
    if needs_interp:
      # Filter NaN rows, then channel-wise interpolate to `max_len_`.
      seqs = [x[np.all(~np.isnan(x), axis=-1)] for x in seqs]
      seqs = [_interp_axis0(x, self.max_len_) for x in seqs]
    out = np.stack(seqs, axis=0)
    return as_tensor(out, device=device) if was_torch else out


# -----------------------------------------------------------------------------

class SequenceAugmentor(BaseEstimator, TransformerMixin):
  """Transformer that tabulates sequences to even length."""

  def __init__(self, add_time: bool = True, lead_lag: bool = True,
               basepoint: bool = True, normalize: bool = True,
               max_time: float = 1., max_len: Optional[int] = None):
    """Initializes the `SequenceAugmentor` object.

    Args:
      add_time: Whether to augment with time coordinate.
      lead_lag: Whether to augment with lead-lag.
      basepoint: Whether to augment with basepoint.
      normalize: Whether to normalize time series.
      max_time: Maximum time if `add_time is True`.
      max_len: Maximum length of sequences.
    """
    self.add_time = add_time
    self.lead_lag = lead_lag
    self.basepoint = basepoint
    self.normalize = normalize
    self.max_time = max_time
    self.max_len = max_len

  def fit(self, X_seq: ArrayOnCPUOrGPU) -> 'SequenceAugmentor':
    """Fits the `SequenceAugmentor` to the data and returns the fitted object.

    Args:
      X_seq: An array sequences on CPU or GPU.
    """
    if self.normalize:
      self.scale_ = (torch.max(X_seq) if _is_torch(X_seq)
                     else np.max(X_seq))
    return self

  def transform(self, X_seq: ArrayOnCPUOrGPU) -> ArrayOnCPUOrGPU:
    """Augments sequences in `X_seq` by adding time and lead-lag.

    Args:
        X_seq: An array of sequences on CPU or GPU.

    Returns:
        An augmented array of sequences on CPU or GPU.
    """
    is_torch = _is_torch(X_seq)
    # Normalization (avoid in-place to not mutate the caller's array).
    if self.normalize:
      X_seq = X_seq / self.scale_
    # Lead-lag augmentation.
    if self.lead_lag:
      if is_torch:
        X_seq = torch.repeat_interleave(X_seq, 2, dim=1)
        X_seq = torch.cat((X_seq[:, 1:], X_seq[:, :-1]), dim=-1)
      else:
        X_seq = np.repeat(X_seq, 2, axis=1)
        X_seq = np.concatenate((X_seq[:, 1:], X_seq[:, :-1]), axis=-1)
    # Time augmentation.
    if self.add_time and self.max_time > 1e-6:
      if is_torch:
        time = torch.linspace(0., self.max_time, X_seq.shape[1],
                              dtype=X_seq.dtype, device=X_seq.device)
        time = torch.tile(time[None, :, None], (X_seq.shape[0], 1, 1))
        X_seq = torch.cat((time, X_seq), dim=-1)
      else:
        time = np.linspace(0., self.max_time, X_seq.shape[1])
        time = np.tile(time[None, :, None], [X_seq.shape[0], 1, 1])
        X_seq = np.concatenate((time, X_seq), axis=-1)
    # Basepoint augmentation.
    if self.basepoint:
      if is_torch:
        X_seq = torch.cat((torch.zeros_like(X_seq[:, :1]), X_seq), dim=1)
      else:
        X_seq = np.concatenate((np.zeros_like(X_seq[:, :1]), X_seq), axis=1)
    # If after augmentation exceeded max length, interpolate back (numpy host
    # path; torch has no `interp`/`apply_along_axis`).
    if self.max_len is not None and X_seq.shape[1] > self.max_len:
      device = X_seq.device if is_torch else None
      X_np = to_numpy(X_seq)
      current = np.linspace(0, 1, X_np.shape[1])
      target = np.linspace(0, 1, self.max_len)
      interp_fn = lambda x: np.interp(target, current, x)
      X_np = np.apply_along_axis(interp_fn, 1, X_np)
      X_seq = as_tensor(X_np, device=device) if is_torch else X_np
    return X_seq


# -----------------------------------------------------------------------------
