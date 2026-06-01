"""Reusable Stage-2 test harness (importable as a normal module, so test files
use stable absolute imports rather than fragile ``..`` relatives).

Holds: backend detection, the golden loader, and the pinpointing comparison.
``conftest.py`` re-exports these and adds the pytest fixtures/markers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.tolerances import get_tol

GOLDEN_DIR = Path(__file__).parent / "golden"


# -----------------------------------------------------------------------------
# Backend detection.
# -----------------------------------------------------------------------------
def detect_backend():
    """Return ('legacy_cupy' | 'torch' | 'unknown', device_str)."""
    try:
        import ksig.utils as u
        ag = getattr(u, "ArrayOnGPU", None)
        try:
            import torch
            if ag is torch.Tensor:
                dev = ("xpu" if getattr(torch, "xpu", None)
                       and torch.xpu.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
                return "torch", dev
        except Exception:
            pass
        try:
            import cupy
            if ag is cupy.ndarray:
                return "legacy_cupy", "cuda"
        except Exception:
            pass
    except Exception:
        pass
    return "unknown", "cpu"


BACKEND, DEVICE = detect_backend()
IS_LEGACY = (BACKEND == "legacy_cupy")


# -----------------------------------------------------------------------------
# Golden loader.
# -----------------------------------------------------------------------------
class Golden:
    def __init__(self, case_id, arrays, meta):
        self.case_id = case_id
        self._arrays = arrays
        self.meta = meta

    def __getitem__(self, key):
        return self._arrays[key]

    def get(self, key, default=None):
        return self._arrays[key] if key in self._arrays else default

    @property
    def keys(self):
        return list(self._arrays.keys())

    @property
    def family(self):
        return self.meta.get("tolerance_class", "DP_CUMSUM")


def load_golden(case_id):
    npz = GOLDEN_DIR / f"{case_id}.npz"
    sidecar = GOLDEN_DIR / f"{case_id}.json"
    if not npz.exists():
        return None
    arrays = {k: v for k, v in np.load(npz).items()}
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return Golden(case_id, arrays, meta)


# -----------------------------------------------------------------------------
# Pinpointing comparison.
# -----------------------------------------------------------------------------
def _pinpoint(got, exp, rtol, atol):
    got = np.asarray(got, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    if got.shape != exp.shape:
        return False, f"shape mismatch: got {got.shape} vs exp {exp.shape}"
    absdiff = np.abs(got - exp)
    tol = atol + rtol * np.abs(exp)
    outside = absdiff > tol
    n_out = int(outside.sum())
    if n_out == 0:
        return True, ""
    flat = int(np.argmax(absdiff))
    coord = np.unravel_index(flat, absdiff.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = absdiff / np.where(np.abs(exp) > 0, np.abs(exp), np.inf)
    msg = (f"{n_out}/{got.size} elements outside tol "
           f"(rtol={rtol:g}, atol={atol:g}); "
           f"worst at {coord}: got={got[coord]:.6g} exp={exp[coord]:.6g} "
           f"|Δ|={absdiff[coord]:.3e} relΔ={np.nanmax(rel):.3e}")
    return False, msg


def assert_close(got, exp, family, dtype="float64", device="cpu",
                 case_id="", note="", rtol_override=None, atol_override=None):
    """Family-tolerance comparison with a pinpointing failure message."""
    rtol, atol = get_tol(family, dtype=dtype, device=device)
    if rtol_override is not None:
        rtol = max(rtol, float(rtol_override))
    if atol_override is not None:
        atol = max(atol, float(atol_override))
    ok, msg = _pinpoint(got, exp, rtol, atol)
    if not ok:
        header = f"[{case_id}] family={family} dtype={dtype} device={device}"
        if note:
            header += f" ({note})"
        raise AssertionError(f"{header}\n  {msg}")
