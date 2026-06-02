"`LinearSVC` with precomputed features."
import warnings

import numpy as np
import torch

from ..static.features import KernelFeatures
from ..utils import ArrayOnCPUOrGPU, ArrayOnCPU, ArrayOnGPU
from ..torch_backend import as_tensor, to_numpy
from .pre_base import PrecomputedSVCBase
from sklearn.svm import LinearSVC
from typing import Dict, Optional

# cuML is NVIDIA/CuPy-only and optional in the torch port: import lazily so the
# same code runs on XPU / MPS / CPU (TORCH_PORT Sec. 8).
try:
  from cuml.svm import LinearSVC as LinearSVCOnGPU
  _HAS_CUML = True
except ImportError:
  LinearSVCOnGPU = None
  _HAS_CUML = False


# Default `LinearSVC` hyperparameter values.
_DEFAULT_LIN_SVC_HPARAMS = {
  'dual': False,
  'fit_intercept': False,
  'tol': 1e-3,
}
_DEFAULT_LIN_SVC_GPU_HPARAMS = {
  'fit_intercept': False,
  'tol': 1e-3,
}


class PrecomputedFeatureLinSVC(PrecomputedSVCBase):
  """`LinearSVC` with precomputed features with optional cross-validation."""

  def __init__(self,
               kernel: KernelFeatures,
               svc_hparams: Dict = {},
               svc_grid: Optional[Dict] = None,
               cv: int = 5,
               n_jobs: int = -1,
               need_kernel_fit: bool = False,
               batch_size: Optional[int] = None,
               on_gpu: bool = False):
    """Initializer for `PrecomputedFeatureLinSVC`.

    Args:
      kernel: A callable for computing features.
      svc_grid: Hyperparameter grid to cross-validate over, set to `None`
        for no cross-validation.
      svc_hparams: Additional hyperparameters for `LinearSVC`.
      cv: Number of CV splits.
      n_jobs: Number of jobs for `GridSearchCV`.
      need_kernel_fit: Whether the features need to be fitted to the data.
      batch_size: If given, compute the feature matrix in chunks of shape
        `[batch_size, ...]` in order to save memory.
      on_gpu: Whether to use the GPU implementation or not.
    """
    n_jobs = 1 if on_gpu else n_jobs  # GPU implementation does not support it.
    super().__init__(kernel, svc_hparams=svc_hparams, svc_grid=svc_grid, cv=cv,
                     n_jobs=n_jobs, need_kernel_fit=need_kernel_fit,
                     batch_size=batch_size)
    self.on_gpu = on_gpu
    # Set default `LinearSVC` hparams.
    if on_gpu:
      for key, val in _DEFAULT_LIN_SVC_GPU_HPARAMS.items():
        self.svc_hparams.setdefault(key, val)
    else:
      for key, val in _DEFAULT_LIN_SVC_HPARAMS.items():
        self.svc_hparams.setdefault(key, val)

  def _get_svc_model(self) -> object:
    """Returns a new instance of a linear SVC model.

    Falls back to the sklearn `LinearSVC` when the GPU path was requested but
    cuML is unavailable (e.g. on XPU / MPS / CPU), so the same script runs
    everywhere.

    Returns:
      An instance of a linear SVC model, which is to be fitted to the data.
    """
    if self.on_gpu and _HAS_CUML:
      return LinearSVCOnGPU(**self.svc_hparams)
    if self.on_gpu and not _HAS_CUML:
      warnings.warn('cuML is unavailable; falling back to `sklearn.svm.'
                    'LinearSVC` on the CPU. Set `on_gpu=False` to silence.')
      # cuML-only hparams (e.g. no `dual`) may not be accepted by sklearn; keep
      # only those sklearn understands by re-applying its safe defaults.
      hparams = {k: v for k, v in self.svc_hparams.items()
                 if k in ('fit_intercept', 'tol', 'C', 'max_iter',
                          'loss', 'penalty', 'dual')}
      hparams.setdefault('dual', False)
      return LinearSVC(**hparams)
    return LinearSVC(**self.svc_hparams)

  def _precompute_model_inputs(self, X: Optional[ArrayOnCPUOrGPU] = None
                              ) -> ArrayOnCPU:
    """Precomputes the feature matrix, which is used as input for the LinearSVC.

    If `X` is not provided, training is assumed and the kernel matrix is
    computed using the stored training data `self.X`, otherwise it is computed
    using the provided `X` data matrix.

    Args:
      X: Optional array of inputs of shape `[num_test, ...]`.

    Returns:
      Feature matrix of shape `[num_train, ...]` when `X is None` else
        it is of shape `[num_test, ...]`.
    """
    if X is None:  # Training.
      feature_mat = self._precompute_feature_mat(self.X)
    else:  # Testing.
      feature_mat = self._precompute_feature_mat(X)
    # cuML consumes cupy arrays; sklearn (the fallback) consumes numpy. The
    # feature matrix here is a torch tensor (`return_on_gpu=True`), so unless we
    # are on the real cuML path, hand sklearn a host numpy array.
    if self.on_gpu and _HAS_CUML:
      import cupy as cp  # only reachable on an NVIDIA/CuPy stack.
      return cp.asarray(to_numpy(feature_mat))
    return to_numpy(feature_mat)

# ------------------------------------------------------------------------------