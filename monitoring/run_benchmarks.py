"""Benchmark driver (``TEST_PLAN.md`` Sec. 10) — performance is measured here, never
asserted in the pytest gate.

Iterates a grid tier, runs each public kernel on the current ``ksig`` backend,
and records wall time + peak memory + GPU utilization to JSONL/CSV.  Respects a
memory budget and **logs every point it skips** (no silent caps): a truncated
run must not read as "covered everything".

    CUDA_VISIBLE_DEVICES=0 python -m monitoring.run_benchmarks \
        --tier small --out monitoring/results --max-gb 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from monitoring import grids, probes, record


def _backend_device():
    try:
        import ksig.utils as u
        import cupy  # noqa: F401
        if u.ArrayOnGPU is cupy.ndarray:
            return "legacy_cupy", "cuda"
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return "torch", "xpu"
        if torch.cuda.is_available():
            return "torch", "cuda"
        return "torch", "cpu"
    except Exception:
        return "unknown", "cpu"


def _sycl_variant(device):
    """Tag distinguishing the SYCL fast-path from the torch wavefront, so the
    run-twice acceptance comparison (SYCL_HANDOFF.md Sec. 7) writes to distinct
    files instead of overwriting. Returns "sycl" only when SYCL would actually
    engage for this run (XPU + ext built + KSIG_USE_SYCL not disabled), else
    "torch"."""
    if device != "xpu":
        return "torch"
    try:
        from ksig.algorithms import _sycl_enabled
        from ksig._sycl import loader
        if _sycl_enabled() and loader.available():
            return "sycl"
    except Exception:
        pass
    return "torch"


def _to_backend(a):
    import ksig.utils as u
    try:
        import cupy as cp
        if u.ArrayOnGPU is cp.ndarray:
            return cp.asarray(a)
    except Exception:
        pass
    try:
        import torch
        if u.ArrayOnGPU is torch.Tensor:
            dev = "xpu" if getattr(torch, "xpu", None) and torch.xpu.is_available() \
                else "cuda" if torch.cuda.is_available() else "cpu"
            return torch.as_tensor(a, device=dev)
    except Exception:
        pass
    return a


def _make_kernel(ksig, pt):
    sk = ksig.static.kernels.RBFKernel()
    if pt.entry == "SignatureKernel":
        return ksig.kernels.SignatureKernel(n_levels=pt.n_levels, order=pt.order,
                                            static_kernel=sk)
    if pt.entry == "SignaturePDEKernel":
        return ksig.kernels.SignaturePDEKernel(static_kernel=sk)
    if pt.entry == "GlobalAlignmentKernel":
        return ksig.kernels.GlobalAlignmentKernel(static_kernel=sk)
    raise KeyError(pt.entry)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small", choices=sorted(grids.TIERS))
    ap.add_argument("--out", default="monitoring/results")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-gb", type=float, default=2.0,
                    help="skip points whose estimated working set exceeds this")
    args = ap.parse_args(argv)

    import ksig
    backend, device = _backend_device()
    budget = int(args.max_gb * 1024 ** 3)
    points = grids.get_tier(args.tier)

    rows, n_run, n_skip = [], 0, 0
    for pt in points:
        est = grids.estimate_bytes(pt)
        if est > budget:
            n_skip += 1
            rows.append(record.new_row(backend, device, pt, est_bytes=est,
                                       skipped=True,
                                       note=f"est {est/1e9:.2f}GB > budget "
                                            f"{args.max_gb}GB"))
            print(f"SKIP  {pt.entry:22s} n={pt.n} L={pt.L} d={pt.d} "
                  f"o={pt.order}  (est {est/1e9:.2f}GB > {args.max_gb}GB)")
            continue
        X = _to_backend(np.random.default_rng(0).standard_normal(
            (pt.n, pt.L, pt.d)).astype(pt.dtype))
        kernel = _make_kernel(ksig, pt)
        timing = probes.time_call(lambda: kernel(X), reps=args.reps,
                                  warmup=args.warmup)
        util = probes.sample_gpu_util()
        rows.append(record.new_row(backend, device, pt, timing=timing,
                                   gpu_util=util, est_bytes=est))
        n_run += 1
        print(f"RUN   {pt.entry:22s} n={pt.n} L={pt.L} d={pt.d} o={pt.order}  "
              f"{timing['median_s']*1e3:8.2f} ms  "
              f"peak={(timing['peak_mem_bytes'] or 0)/1e6:7.1f} MB")

    variant = _sycl_variant(device)
    stem = f"{backend}_{device}_{variant}_{args.tier}"
    jsonl, csv_path = record.write(rows, args.out, stem)
    print(f"\nTier '{args.tier}' on {backend}/{device} [{variant}]: "
          f"{n_run} run, {n_skip} skipped (over budget).")
    print(f"  -> {jsonl}\n  -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
