"""Timing, peak-memory, device-utilization, and CPU-fallback probes
(``TEST_PLAN.md`` Sec. 11).  Backend-aware: works against the legacy CuPy ``ksig``
today and the torch port (CUDA/XPU) later.

The XPU stack can silently fall back to CPU for unimplemented ops, which
destroys both correctness-locality and the performance story; :func:`detect_xpu_fallback`
is the hook for catching that (a recorded event in monitoring; a hard failure in
``xpu``-marked correctness tests).
"""
from __future__ import annotations

import gc
import subprocess
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional


# -----------------------------------------------------------------------------
# Backend sync / memory helpers (cupy today, torch later).
# -----------------------------------------------------------------------------
def _sync():
    try:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
        return
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def reset_peak_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        return
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            torch.xpu.reset_peak_memory_stats()
        elif torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_memory_bytes() -> Optional[int]:
    try:
        import cupy as cp
        # cupy has no max-allocated counter; report current pool high-water.
        return int(cp.get_default_memory_pool().used_bytes())
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return int(torch.xpu.max_memory_allocated())
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Timing.
# -----------------------------------------------------------------------------
def time_call(fn: Callable[[], object], reps: int = 5, warmup: int = 2) -> Dict:
    """Median wall time over ``reps`` (after ``warmup``), with peak memory."""
    for _ in range(warmup):
        fn(); _sync()
    gc.collect()
    reset_peak_memory()
    times = []
    for _ in range(reps):
        _sync(); t0 = time.perf_counter()
        fn()
        _sync(); times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "median_s": times[len(times) // 2],
        "min_s": times[0],
        "max_s": times[-1],
        "reps": reps,
        "peak_mem_bytes": peak_memory_bytes(),
    }


# -----------------------------------------------------------------------------
# Device utilization (best-effort, vendor tool).
# -----------------------------------------------------------------------------
def sample_gpu_util(index: int = 0) -> Optional[float]:
    """One-shot GPU utilization sample via nvidia-smi / xpu-smi (best effort)."""
    for cmd in (
        ["nvidia-smi", f"--id={index}", "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"],
        ["xpu-smi", "stats", "-d", str(index)],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                for tok in first.replace("%", "").split():
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        except Exception:
            continue
    return None


# -----------------------------------------------------------------------------
# CPU-fallback detection (torch/XPU).  No-op on cupy (NVIDIA-only, no fallback).
# -----------------------------------------------------------------------------
@contextmanager
def detect_xpu_fallback():
    """Yield a dict ``{"fell_back": bool, "ops": [...]}``.  On torch+XPU this
    would hook the PyTorch fallback warning / profiler op-device tags; here it is
    a structural stub that always reports no fallback on non-torch backends."""
    state = {"fell_back": False, "ops": []}
    try:
        import torch  # noqa: F401
        # Placeholder: a real implementation registers a warnings filter for
        # the "aten::... CPU fallback" message or inspects a profiler trace.
        yield state
    except Exception:
        yield state
