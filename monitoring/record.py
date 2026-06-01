"""JSONL + CSV writers for benchmark results (``TEST_PLAN.md`` Sec. 10).

One JSONL row per ``(entry, backend, fixture, dtype, rep-aggregate)`` with full
provenance, plus a flat CSV mirror for quick plotting.  ``results/`` is
git-ignored.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
from pathlib import Path
from typing import Dict, List

SCHEMA_VERSION = 1

# The flat columns mirrored into CSV (nested provenance stays JSONL-only).
CSV_COLUMNS = [
    "schema_version", "timestamp", "backend", "device", "entry",
    "n", "L", "d", "n_levels", "order", "dtype",
    "median_s", "min_s", "max_s", "reps", "peak_mem_bytes", "gpu_util_pct",
    "est_bytes", "skipped", "note",
]


def provenance() -> Dict:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cupy", "numba", "torch", "scipy", "sklearn"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def new_row(backend, device, point, timing=None, gpu_util=None,
            est_bytes=None, skipped=False, note="") -> Dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "backend": backend, "device": device,
        "entry": point.entry, "n": point.n, "L": point.L, "d": point.d,
        "n_levels": point.n_levels, "order": point.order, "dtype": point.dtype,
        "est_bytes": est_bytes, "skipped": skipped, "note": note,
        "gpu_util_pct": gpu_util,
        "provenance": provenance(),
    }
    if timing:
        row.update({k: timing.get(k) for k in
                    ("median_s", "min_s", "max_s", "reps", "peak_mem_bytes")})
    return row


def write(rows: List[Dict], out_dir: str, stem: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{stem}.jsonl"
    with jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    csv_path = out / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return jsonl, csv_path
