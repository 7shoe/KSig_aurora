"""Device, dtype, and RNG policy for the torch backend.

Centralizes everything device/dtype/RNG. Every other module imports from here
(see ``docs/TORCH_PORT.md`` Sec. 3).

The library runs on NVIDIA CUDA, Intel XPU (Aurora) and Apple MPS through a
single torch backend; this module hides the per-device differences (no fp64 on
MPS, Generator placement, etc.) behind a small surface the rest of the code
calls instead of touching ``torch.*`` device/dtype/RNG directly.
"""
from __future__ import annotations

import numpy as np
import torch

from numbers import Integral
from typing import Optional, Union

_EPS = 1e-12


# -----------------------------------------------------------------------------
# Device policy.
# -----------------------------------------------------------------------------
def get_device(prefer: Optional[str] = None) -> torch.device:
  """Select the compute device, honoring an explicit ``prefer`` override.

  Preference order when ``prefer is None``: CUDA, then XPU (Aurora), then MPS
  (Apple), then CPU.
  """
  if prefer is not None:
    return torch.device(prefer)
  if torch.cuda.is_available():
    return torch.device('cuda')
  if hasattr(torch, 'xpu') and torch.xpu.is_available():  # Aurora.
    return torch.device('xpu')
  mps = getattr(torch.backends, 'mps', None)
  if mps is not None and mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


_DEFAULT_DEVICE = get_device()


def current_device() -> torch.device:
  """Return the process-wide default compute device."""
  return _DEFAULT_DEVICE


def set_default_device(dev) -> None:
  """Override the process-wide default compute device."""
  global _DEFAULT_DEVICE
  _DEFAULT_DEVICE = torch.device(dev)


# -----------------------------------------------------------------------------
# Dtype policy.
# -----------------------------------------------------------------------------
def supports_float64(device=None) -> bool:
  """Whether `device` has a native float64; MPS does not."""
  device = device or current_device()
  return torch.device(device).type != 'mps'


def default_float_dtype(device=None) -> torch.dtype:
  """The default floating dtype for `device` (float64 except on MPS)."""
  device = device or current_device()
  return torch.float64 if supports_float64(device) else torch.float32


def eps_for(dtype) -> float:
  """A numerically safe epsilon for `dtype` (1e-12 underflows float32)."""
  return 1e-12 if dtype == torch.float64 else 1e-7


# -----------------------------------------------------------------------------
# Array conversion.
# -----------------------------------------------------------------------------
def as_tensor(x, dtype=None, device=None) -> torch.Tensor:
  """Coerce `x` to a float tensor on `device` (default device if omitted).

  Existing tensors are moved/cast only when needed; numpy/lists are materialized
  with the device's default float dtype unless `dtype` is given.
  """
  device = device or current_device()
  if isinstance(x, torch.Tensor):
    if dtype is not None or x.device != torch.device(device):
      return x.to(device=device, dtype=dtype)
    return x
  return torch.as_tensor(np.asarray(x),
                         dtype=dtype or default_float_dtype(device),
                         device=device)


def as_index(x, device=None) -> torch.Tensor:
  """Coerce `x` to a ``long`` index tensor on `device`."""
  device = device or current_device()
  if isinstance(x, torch.Tensor):
    return x.to(device=device, dtype=torch.long)
  return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)


def to_numpy(x) -> np.ndarray:
  """Bring a tensor back to a host numpy array (no-op for numpy/lists)."""
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().numpy()
  return np.asarray(x)


# -----------------------------------------------------------------------------
# RNG policy.
# -----------------------------------------------------------------------------
class TorchRandomState:
  """Drop-in for ``cupy.random.RandomState`` used by KSig.

  Implements only the surface KSig relies on: ``normal``/``uniform``/
  ``randint``/``choice``. Note that the torch ``Generator`` stream differs from
  CuPy's for the same seed, so randomized outputs are NOT bit-compatible with
  the legacy backend (the test-suite compares the estimand, not raw draws).
  """

  def __init__(self, seed=None, device=None):
    self.device = torch.device(device) if device is not None else current_device()
    # MPS cannot host a Generator: generate on CPU then move.
    self._gen_device = 'cpu' if self.device.type == 'mps' else self.device
    self.generator = torch.Generator(device=self._gen_device)
    if seed is not None:
      self.generator.manual_seed(int(seed))

  def _move(self, t: torch.Tensor) -> torch.Tensor:
    return t.to(self.device)

  def normal(self, size, dtype=None) -> torch.Tensor:
    dt = dtype or default_float_dtype(self.device)
    return self._move(torch.randn(*size, generator=self.generator,
                                  device=self._gen_device, dtype=dt))

  def uniform(self, size, low=0., high=1., dtype=None) -> torch.Tensor:
    dt = dtype or default_float_dtype(self.device)
    u = torch.rand(*size, generator=self.generator, device=self._gen_device,
                   dtype=dt)
    return self._move(low + (high - low) * u)

  def randint(self, low, high=None, size=None) -> torch.Tensor:
    if high is None:
      low, high = 0, low
    return self._move(torch.randint(low, high, tuple(size),
                                    generator=self.generator,
                                    device=self._gen_device))

  def choice(self, n, size, replace=False) -> torch.Tensor:
    assert not replace, 'KSig only uses replace=False'
    return self._move(torch.randperm(n, generator=self.generator,
                                     device=self._gen_device)[:size])


RandomStateOrSeed = Union[Integral, TorchRandomState]


def check_random_state(rs=None) -> TorchRandomState:
  """Coerce `rs` (``None``/int seed/``TorchRandomState``) to a state object."""
  if rs is None or isinstance(rs, Integral):
    return TorchRandomState(rs)
  if isinstance(rs, TorchRandomState):
    return rs
  raise ValueError(f'{rs} cannot seed a TorchRandomState')
