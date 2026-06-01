"""A ``monitoring``-marked smoke test so the perf harness can't rot: import each
probe and run it once on a tiny grid (``TEST_PLAN.md`` Sec. 10).  This asserts the
harness *runs*, never a performance threshold (timings are noisy/hardware-bound).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.monitoring

ksig = pytest.importorskip("ksig")


def test_probes_import_and_run():
    from monitoring import probes
    out = probes.time_call(lambda: np.sqrt(np.arange(1000)), reps=2, warmup=1)
    assert out["median_s"] >= 0.0 and out["reps"] == 2
    # peak_memory_bytes + sample_gpu_util are best-effort (None allowed).
    _ = probes.peak_memory_bytes()
    _ = probes.sample_gpu_util()


def test_grids_are_memory_bounded():
    from monitoring import grids
    small = grids.get_tier("small")
    assert len(small) > 0
    # Every 'small' point must fit comfortably (< 256 MB working set).
    for pt in small:
        assert grids.estimate_bytes(pt) < 256 * 1024 ** 2, pt


def test_record_roundtrip(tmp_path):
    from monitoring import grids, record
    pt = grids.get_tier("small")[0]
    rows = [record.new_row("legacy_cupy", "cuda", pt,
                           timing={"median_s": 0.01, "min_s": 0.01,
                                   "max_s": 0.02, "reps": 3,
                                   "peak_mem_bytes": 1234})]
    jsonl, csv_path = record.write(rows, str(tmp_path), "smoke")
    assert jsonl.exists() and csv_path.exists()
    assert jsonl.read_text().strip().startswith("{")


def test_benchmark_one_point():
    """Drive a single real benchmark point end-to-end on the current backend."""
    from monitoring import grids, probes
    from monitoring.run_benchmarks import _make_kernel, _to_backend
    pt = grids.GridPoint("SignatureKernel", n=4, L=8, d=3, order=1)
    X = _to_backend(np.random.default_rng(0).standard_normal((pt.n, pt.L, pt.d)))
    kernel = _make_kernel(ksig, pt)
    timing = probes.time_call(lambda: kernel(X), reps=2, warmup=1)
    assert timing["median_s"] >= 0.0
