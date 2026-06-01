"""Benchmark grid definitions (``TEST_PLAN.md`` Sec. 10).

Performance is **measured, not asserted** — these grids drive
``run_benchmarks.py`` to produce JSONL/CSV artifacts the maintainers read to
decide whether the SYCL fast-path earns its keep.  Tiers are explicit so a
truncated run can never read as "covered everything".

MEMORY: the signature/DP kernels materialize ``O(n^2 L^2)``.  The ``small`` and
``medium`` tiers below are bounded so a single H100/PVC is never threatened; the
``stress`` tier is opt-in and logs its frontier (largest shape completed before
OOM / wall-time ceiling) rather than crashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class GridPoint:
    entry: str            # public kernel name
    n: int                # n_samples (Gram is n x n)
    L: int                # seq_len
    d: int                # n_features
    n_levels: int = 4
    order: int = 1
    dtype: str = "float64"


def _signature_points(ns, Ls, ds, orders, n_levels=4):
    pts = []
    for n in ns:
        for L in Ls:
            for d in ds:
                for o in orders:
                    pts.append(GridPoint("SignatureKernel", n, L, d,
                                         n_levels=n_levels, order=o))
    return pts


def _dp_points(entry, ns, Ls, ds):
    return [GridPoint(entry, n, L, d) for n in ns for L in Ls for d in ds]


# Memory-bounded tiers.  (n^2 L^2 cells; worst small=8^2*16^2=16k, fine.)
TIERS = {
    "small": (
        _signature_points([4, 8], [8, 16], [3], [1, 2])
        + _dp_points("SignaturePDEKernel", [4, 8], [8, 16], [3])
        + _dp_points("GlobalAlignmentKernel", [4, 8], [8, 16], [3])
    ),
    "medium": (
        _signature_points([16, 32], [32, 64], [5], [1, 2])
        + _dp_points("SignaturePDEKernel", [16, 32], [32, 64], [5])
        + _dp_points("GlobalAlignmentKernel", [16, 32], [32, 64], [5])
    ),
    # Opt-in only. 64^2 * 128^2 ≈ 67e6 cells * 8B ≈ 0.5GB per materialized
    # tensor; run deliberately and watch the frontier log.
    "stress": (
        _signature_points([64], [128, 256], [5], [1])
        + _dp_points("SignaturePDEKernel", [64], [128, 256], [5])
    ),
}


def get_tier(name: str) -> List[GridPoint]:
    if name not in TIERS:
        raise KeyError(f"unknown tier {name!r}; known: {sorted(TIERS)}")
    return TIERS[name]


def estimate_bytes(pt: GridPoint) -> int:
    """Rough working-set estimate for the materialized M tensor (the dominant
    allocation), used to skip a point that would exceed a memory budget."""
    itemsize = 8 if pt.dtype == "float64" else 4
    cells = pt.n * pt.n * pt.L * pt.L
    factor = pt.order * pt.order if pt.entry == "SignatureKernel" else 1
    return cells * itemsize * factor
