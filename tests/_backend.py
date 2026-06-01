"""Backend-agnostic array helpers used by the Stage-2 tests: move a numpy array
onto whatever array type the ksig-under-test expects (cupy today, torch after
the port) and bring results back to host numpy."""
from __future__ import annotations

import numpy as np


def host(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def to_backend(a, device="cpu"):
    import ksig.utils as u
    ag = getattr(u, "ArrayOnGPU", None)
    try:
        import cupy as cp
        if ag is cp.ndarray:
            return cp.asarray(a)
    except Exception:
        pass
    try:
        import torch
        if ag is torch.Tensor:
            return torch.as_tensor(a, device=device)
    except Exception:
        pass
    return np.asarray(a)
