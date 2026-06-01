"""Shared entry-point runners.

The freeze stage (legacy CuPy ``ksig``) and the compare stage (future torch
``ksig``) call the **same** runner so the two stacks exercise identical code
paths.  A runner takes the resolved numpy inputs + the case kwargs, constructs
the public object, calls it, and returns a **host numpy** result (the kernels
already return host arrays; we defensively coerce).

``static_kernel`` kwargs are passed as short strings (``"rbf"``/``"linear"``)
and mapped to a kernel instance here, so the spec stays JSON-serializable.
"""
from __future__ import annotations

import numpy as np


def _host(a):
    """Coerce any backend array (cupy/torch/numpy) to host float64 ndarray."""
    if isinstance(a, np.ndarray):
        return np.ascontiguousarray(a)
    # cupy
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    # torch
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _static_kernel(ksig, name):
    sk = ksig.static.kernels
    return {"rbf": sk.RBFKernel, "linear": sk.LinearKernel}[name]()


def _static_kernel_cls(ksig, name):
    return getattr(ksig.static.kernels, name)


# -----------------------------------------------------------------------------
# Runner registry.  Each returns a host numpy array.
# -----------------------------------------------------------------------------
def run_SignatureKernel(ksig, inputs, kwargs):
    kw = dict(kwargs)
    static = _static_kernel(ksig, kw.pop("static_kernel", "rbf"))
    k = ksig.kernels.SignatureKernel(static_kernel=static, **kw)
    return _call_kernel(k, inputs)


def run_SignaturePDEKernel(ksig, inputs, kwargs):
    kw = dict(kwargs)
    static = _static_kernel(ksig, kw.pop("static_kernel", "rbf"))
    k = ksig.kernels.SignaturePDEKernel(static_kernel=static, **kw)
    return _call_kernel(k, inputs)


def run_GlobalAlignmentKernel(ksig, inputs, kwargs):
    kw = dict(kwargs)
    static = _static_kernel(ksig, kw.pop("static_kernel", "rbf"))
    k = ksig.kernels.GlobalAlignmentKernel(static_kernel=static, **kw)
    return _call_kernel(k, inputs)


def run_static(ksig, inputs, kwargs):
    kw = dict(kwargs)
    cls = _static_kernel_cls(ksig, kw.pop("kernel"))
    k = cls(**kw)
    X = inputs["X"]
    Y = inputs.get("Y", None)
    return _host(k(X, Y) if Y is not None else k(X))


def run_static_diag(ksig, inputs, kwargs):
    kw = dict(kwargs)
    cls = _static_kernel_cls(ksig, kw.pop("kernel"))
    k = cls(**kw)
    return _host(k(inputs["X"], diag=True))


def _call_kernel(k, inputs):
    """Sequence kernels: emit a dict of the three README contracts so one
    golden artifact validates Gram, diag, and the cross kernel at once."""
    X = inputs["X"]
    out = {"gram": _host(k(X)), "diag": _host(k(X, diag=True))}
    if "Y" in inputs:
        out["xy"] = _host(k(X, inputs["Y"]))
    return out


_REGISTRY = {
    "SignatureKernel": run_SignatureKernel,
    "SignaturePDEKernel": run_SignaturePDEKernel,
    "GlobalAlignmentKernel": run_GlobalAlignmentKernel,
    "static": run_static,
    "static_diag": run_static_diag,
}


def run(ksig, entry, inputs, kwargs):
    """Dispatch ``entry`` to its runner. Returns a numpy array or a dict of
    named numpy arrays (sequence kernels)."""
    return _REGISTRY[entry](ksig, inputs, kwargs)
