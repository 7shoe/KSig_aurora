"""KSig: GPU-accelerated sequence kernels, torch-native backend.

Runs on NVIDIA CUDA, Intel XPU (Aurora), Apple MPS and CPU through a single
torch backend. On older Aurora stacks the XPU device only appears after
importing Intel's torch extension; we try that import and ignore it if the
installed torch already exposes XPU natively (recent ALCF modules do).
"""

# On older Aurora software stacks ``import intel_extension_for_pytorch`` is
# required before ``torch.xpu`` works; on recent native-XPU torch builds it is
# unnecessary (and absent). Guard it so neither case errors.
try:  # pragma: no cover - hardware/stack dependent.
  import intel_extension_for_pytorch  # noqa: F401
except Exception:
  pass

from . import torch_backend
from .torch_backend import set_default_device, current_device  # noqa: F401

from . import algorithms
from . import static
from . import kernels
from . import preprocessing
from . import projections
from . import utils
