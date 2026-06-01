"""Tolerance policy — chosen by *algorithm family*, not per-test, so the whole
policy is auditable in one table (``TEST_PLAN.md`` Sec. 5).

The family is recorded in every golden sidecar (``tolerance_class``).  The
``(rtol, atol)`` depends on the family, the compute dtype, and the device:

* float64 on CUDA/XPU/CPU  -> the tight column;
* float32 / MPS            -> the loose column (FMA ordering + vendor
  transcendentals diverge; MPS has no native fp64).

Rationale for the XPU loosening: oneAPI fp64 is IEEE-754 and should match CUDA
fp64 to the DP_CUMSUM band; fp32 differs by design, so it gets the loose column.
"""
from __future__ import annotations

from typing import Tuple

# family -> (float64 (rtol, atol), float32/MPS (rtol, atol))
_TABLE = {
    # Exact algebra: matmul, outer products, linear/poly static kernels.
    "EXACT_ALGEBRA":  ((1e-10, 1e-12), (1e-5, 1e-6)),
    # Cumulative-sum DP: signature kernels/features, SigPDE, GAK.
    "DP_CUMSUM":      ((1e-8,  1e-10), (1e-3, 1e-4)),
    # min,+ DP (RWS/DTW) — exact integer/real arithmetic.
    "DTW_EXACT":      ((1e-12, 0.0),   (1e-5, 1e-5)),
    # Random-feature maps: compared as a Monte-Carlo band, not elementwise
    # (handled in the test, this entry is the *injected-weights* exact band).
    "RANDOM_FEATURE": ((1e-8,  1e-10), (1e-3, 1e-4)),
    # End-to-end classifier scores.
    "E2E_SCORE":      ((0.0,   0.0),   (0.01, 0.01)),
    # SYCL vs torch wavefront — inherits the f64 band of its family.
    "SYCL":           ((1e-8,  1e-10), (1e-3, 1e-4)),
}


def _is_low_precision(dtype: str, device: str) -> bool:
    dtype = str(dtype)
    return ("16" in dtype) or ("32" in dtype) or (str(device) == "mps")


def get_tol(family: str, dtype="float64", device="cpu") -> Tuple[float, float]:
    """Return ``(rtol, atol)`` for a family at the given dtype/device."""
    if family not in _TABLE:
        raise KeyError(f"unknown tolerance family {family!r}; "
                       f"known: {sorted(_TABLE)}")
    f64, f32 = _TABLE[family]
    return f32 if _is_low_precision(dtype, device) else f64


# Tolerance used when cross-checking the legacy CuPy oracle against the
# brute-force NumPy oracle during freeze: both are float64, agreement should be
# at the DP_CUMSUM f64 level or tighter.
CROSS_CHECK_RTOL = 1e-7
CROSS_CHECK_ATOL = 1e-9
