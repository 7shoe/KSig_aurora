"""Shared helpers for the ``./notebooks`` demos.

Backend- and device-agnostic by design: the same notebooks run on the legacy
CuPy stack (the NVIDIA reference box that froze ``cuda_reference.json``) **and**
on the torch-native port on Aurora (Intel XPU), CUDA, MPS or plain CPU.  Nothing
here imports ``cupy``/``torch``/``numba`` at module load — every backend is
probed at call time and guarded — so importing this module never fails on a box
that is missing one of them.

The notebooks lean on four things:

* :func:`detect_env`  — what backend / device / SYCL support is live now.
* :func:`simulate`    — deterministic, portable input data (NumPy RNG, identical
  bit-for-bit on every machine), so a value computed on Aurora can be compared
  against the value frozen on the H100.
* :func:`timeit`      — warmed-up, device-synchronised median wall time.
* :func:`scaling_plot`— the green/blue(/orange) scaling figure:
    - **green**  = the baked NVIDIA-CUDA reference (``cuda_reference.json``),
    - **blue**   = what this machine computes live (torch-native),
    - **orange** = the optional SYCL fast-path — drawn *only* when SYCL is
      actually available (see :func:`sycl_available`).  If the SYCL path never
      lands, the orange curve never appears.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REF_PATH = _HERE / "cuda_reference.json"

# Colour contract used across every notebook (kept in one place on purpose).
C_REF = "tab:green"     # frozen NVIDIA CUDA reference
C_LIVE = "tab:blue"     # this machine, torch-native
C_SYCL = "tab:orange"   # this machine, SYCL fast-path (only if supported)


# ---------------------------------------------------------------------------
# Environment / capability probing.
# ---------------------------------------------------------------------------
def detect_env() -> dict:
    """Return a dict describing the live stack: ``backend``, ``device``,
    ``sycl``, ``ksig_version``.  Never raises."""
    info = {"backend": "unknown", "device": "cpu", "sycl": False,
            "ksig_version": None}
    try:
        import ksig
        info["ksig_version"] = getattr(ksig, "__version__", "unknown")
        import ksig.utils as u
        ag = getattr(u, "ArrayOnGPU", None)
        try:
            import torch
            if ag is torch.Tensor:
                info["backend"] = "torch"
                if getattr(torch, "xpu", None) and torch.xpu.is_available():
                    info["device"] = "xpu"
                elif torch.cuda.is_available():
                    info["device"] = "cuda"
                elif getattr(torch.backends, "mps", None) and \
                        torch.backends.mps.is_available():
                    info["device"] = "mps"
                else:
                    info["device"] = "cpu"
        except Exception:
            pass
        if info["backend"] == "unknown":
            try:
                import cupy
                if ag is cupy.ndarray:
                    info["backend"], info["device"] = "legacy_cupy", "cuda"
            except Exception:
                pass
    except Exception:
        pass
    info["sycl"] = sycl_available()
    return info


def sycl_available() -> bool:
    """True iff the port exposes a usable SYCL fast-path.

    The torch port (``docs/TORCH_PORT.md`` §SYCL) will ship a ``ksig._sycl``
    extension; until it lands this is ``False`` everywhere, so the notebooks
    never draw a SYCL curve.  When the extension is built, this check starts
    returning ``True`` and the second (orange) curve appears automatically.

    Per ``tests/TEST_PLAN.md`` §12, SYCL is only ever *dispatched to* if it is
    both correct and measurably faster; if it is judged "never useful" the
    extension is simply absent and this stays ``False`` — i.e. no second curve,
    by construction.
    """
    try:
        # Import the submodule explicitly: ``ksig._sycl/__init__.py`` does not
        # bind ``loader``, so ``import ksig._sycl; ksig._sycl.loader`` would
        # raise ``AttributeError`` in a fresh process and wrongly report False.
        from ksig._sycl import loader  # type: ignore
        return bool(loader.available())
    except Exception:
        return False


def enable_sycl(flag: bool = True) -> bool:
    """Toggle the SYCL fast-path for a scaling sweep.

    Returns whether SYCL is active afterwards (``False`` if unavailable), so a
    caller can gate the orange curve on the return value.

    The real switch is the ``KSIG_USE_SYCL`` env var: ``ksig.algorithms``
    reads it **per kernel call** (``_sycl_enabled()``), so flipping it here
    changes dispatch for the *next* evaluation in this same process — exactly
    what the blue (off) vs orange (on) sweeps need.  ``KSIG_USE_SYCL=0`` (also
    ``false``/``no``/``off``) forces the torch wavefront; default/``1`` lets
    SYCL auto-engage on XPU.
    """
    if not sycl_available():
        return False
    os.environ["KSIG_USE_SYCL"] = "1" if flag else "0"
    return flag


# ---------------------------------------------------------------------------
# Deterministic, portable data.
# ---------------------------------------------------------------------------
def simulate(n: int, L: int, d: int, seed: int = 0) -> np.ndarray:
    """A batch of ``n`` random-walk paths of length ``L`` in ``d`` dimensions.

    Built from ``numpy.random.default_rng(seed)`` — bit-identical on every
    platform — and integrated (``cumsum``) so the paths have non-trivial
    signatures.  Returns a host ``float64`` array of shape ``[n, L, d]``; the
    public ksig kernels accept host arrays directly and move them to the device
    internally, so the very same call works on CuPy, torch-XPU, CUDA or CPU.
    """
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((n, L, d)), axis=1) / np.sqrt(L)
    return np.ascontiguousarray(X, dtype=np.float64)


def as_host(a) -> np.ndarray:
    """Coerce any backend array (cupy / torch / numpy) to a host ndarray."""
    if isinstance(a, np.ndarray):
        return a
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------
def synchronize(device: str | None = None) -> None:
    """Block until queued device work is done, so timings are real."""
    device = device or detect_env()["device"]
    try:
        import torch
        if device == "xpu" and getattr(torch, "xpu", None):
            torch.xpu.synchronize(); return
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(); return
    except Exception:
        pass
    try:
        import cupy as cp
        if device == "cuda":
            cp.cuda.Device().synchronize()
    except Exception:
        pass


def timeit(fn, reps: int = 5, warmup: int = 1, device: str | None = None) -> float:
    """Median wall time (seconds) of ``fn()`` over ``reps`` runs after
    ``warmup`` untimed runs, synchronising the device around each call."""
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---------------------------------------------------------------------------
# Reference data + plotting.
# ---------------------------------------------------------------------------
def cuda_reference(feature: str | None = None):
    """Load the frozen NVIDIA-CUDA reference (``cuda_reference.json``).

    With ``feature`` -> that feature's block (or ``None`` if absent); without
    -> the whole dict (``{}`` if the file is missing)."""
    if not _REF_PATH.exists():
        return None if feature else {}
    ref = json.loads(_REF_PATH.read_text())
    return ref.get(feature) if feature else ref


def reference_grid(feature: str, axis_key: str = "x", default=None):
    """The reference's scaling x-grid for ``feature`` (so a notebook's live
    sweep lines up with the green curve by default)."""
    ref = cuda_reference(feature) or {}
    return list(ref.get("scaling", {}).get(axis_key, default or []))


def scaling_plot(x, target_seconds, feature, *, sycl_seconds=None,
                 ylabel="median wall time (s)", title=None, ax=None,
                 env=None, logx=True, logy=True):
    """The green/blue(/orange) scaling figure.

    Args:
      x:              the problem-size grid the live sweep used.
      target_seconds: live median times on THIS machine (the blue curve).
      feature:        key into ``cuda_reference.json`` for the green curve.
      sycl_seconds:   optional live SYCL times (orange).  Pass ``None`` (the
                      default) to omit the curve — do that whenever
                      :func:`sycl_available` is ``False``.
    """
    import matplotlib.pyplot as plt
    env = env or detect_env()
    ref = cuda_reference(feature) or {}
    sc = ref.get("scaling", {})
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))

    # GREEN — the frozen NVIDIA CUDA reference.
    if sc.get("x") and sc.get("seconds"):
        gpu = ref.get("meta", {}).get("gpu") or "NVIDIA CUDA"
        ax.plot(sc["x"], sc["seconds"], "-o", color=C_REF,
                label=f"CUDA reference · {gpu}")

    # BLUE — live on this machine (torch-native, or whatever backend is loaded).
    live_label = f"this machine · {env['backend']}-{env['device']}"
    ax.plot(x, target_seconds, "-o", color=C_LIVE, label=live_label)

    # ORANGE — optional SYCL fast-path, only when supported.
    if sycl_seconds is not None:
        ax.plot(x, sycl_seconds, "-o", color=C_SYCL,
                label=f"this machine · SYCL-{env['device']}")

    ax.set_xlabel(sc.get("axis", "problem size"))
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title or feature)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def print_env_banner(env: dict | None = None) -> None:
    """One-line capability banner for the top of each notebook."""
    env = env or detect_env()
    sycl = "available" if env["sycl"] else "absent (no 2nd curve)"
    print(f"ksig backend : {env['backend']}  |  device: {env['device']}  |  "
          f"SYCL: {sycl}  |  ksig {env['ksig_version']}")
    if env["backend"] == "unknown":
        print("  ! ksig is not importable yet — port a module first, then re-run.")
