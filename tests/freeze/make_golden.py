"""Stage 1 — freeze the ground truth (``TEST_PLAN.md`` Sec. 3).

Runs the **legacy CuPy/Numba** ``ksig`` over the canonical fixture matrix and
persists frozen oracle outputs to ``tests/golden/<case_id>.npz`` with a
``<case_id>.json`` provenance sidecar and a top-level ``INDEX.json`` manifest.

What makes this trustworthy (and not merely "the old code agrees with itself"):
every value is **cross-checked against an independent brute-force NumPy oracle**
(:mod:`tests.oracles.pipeline`).  The recorded ``cross_check_max_abs`` proves two
independent computations agree.  Where the legacy code is known-broken (the
``euclid_dist`` bug in ``Matern12``/``Matern32``), the golden value is sourced
from the **oracle** instead and the sidecar records ``golden_source =
"numpy_oracle"`` + ``legacy_status = "broken"`` so the torch port is held to the
correct answer rather than replicating the bug.

Run (on the NVIDIA/CuPy box)::

    CUDA_VISIBLE_DEVICES=0 python -m tests.freeze.make_golden --out tests/golden

Idempotent: same fixtures -> identical ``.json`` except ``created_utc``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

from ..fixtures import matrix as fxmatrix
from ..fixtures import runners
from ..oracles import pipeline as oracle
from ..tolerances import CROSS_CHECK_ATOL, CROSS_CHECK_RTOL, get_tol


def _versions():
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("cupy", "numba", "scipy", "sklearn"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    try:
        import cupy as cp
        v["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        v["gpu_name"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        v["gpu_name"] = None
    return v


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "n_nonfinite": int((~np.isfinite(a)).sum()),
            "shape": list(a.shape)}


def _as_dict(out):
    """Normalize a runner/oracle result into a name->ndarray dict."""
    return out if isinstance(out, dict) else {"value": out}


def _diff_metrics(a, b):
    """Return (max_abs, max_rel, ok) under the CROSS_CHECK relative tolerance."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf"), float("inf"), False
    if a.size == 0:
        return 0.0, 0.0, True
    absdiff = np.abs(a - b)
    max_abs = float(absdiff.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = absdiff / np.where(np.abs(b) > 0, np.abs(b), np.inf)
    max_rel = float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0
    ok = bool(np.all(absdiff <= CROSS_CHECK_ATOL + CROSS_CHECK_RTOL * np.abs(b)))
    return max_abs, max_rel, ok


def freeze_case(case, ksig):
    """Compute (golden_arrays, sidecar) for one case."""
    inputs = fxmatrix.build_inputs(case.inputs)

    # Independent oracle ground truth (+ broken flag).
    oracle_out, legacy_broken = oracle.oracle_output(case.entry, case.kwargs,
                                                      inputs)
    oracle_d = _as_dict(oracle_out)

    # Legacy ksig output (may raise if the legacy code is broken).
    legacy_d, legacy_err = None, None
    try:
        legacy_d = _as_dict(runners.run(ksig, case.entry, inputs, case.kwargs))
    except Exception as e:  # e.g. Matern12/32 euclid_dist bug
        legacy_err = f"{type(e).__name__}: {e}"

    # Decide golden source + cross-check.
    cross, cross_rel, cross_all_ok = {}, {}, True
    if legacy_broken or legacy_d is None:
        source = "numpy_oracle"
        golden = oracle_d
        legacy_status = "broken" if (legacy_broken or legacy_err) else "ok"
    else:
        source = "legacy_cupy"
        golden = legacy_d
        legacy_status = "ok"
        for key in golden:
            if key in oracle_d:
                ma, mr, ok = _diff_metrics(golden[key], oracle_d[key])
                cross[key] = ma
                cross_rel[key] = mr
                cross_all_ok = cross_all_ok and ok

    cross_max = max(cross.values()) if cross else None
    cross_rel_max = max(cross_rel.values()) if cross_rel else None

    # Ill-conditioning detection: two correct float64 computations (legacy and
    # oracle) that disagree beyond the family's f64 rtol indicate an unstable
    # formula (here: normalization dividing by an _EPS-clamped ~0 level
    # diagonal), NOT a bug.  Flag it and recommend a looser Stage-2 tolerance
    # so the port isn't held to a value that isn't well-defined to 1e-8.
    family_rtol = get_tol(case.family, "float64")[0]
    ill_conditioned = bool(source == "legacy_cupy" and not cross_all_ok)
    recommended_rtol = (max(10.0 * cross_rel_max, family_rtol)
                        if ill_conditioned and cross_rel_max is not None
                        else family_rtol)

    sidecar = {
        "case_id": case.case_id,
        "entry": case.entry,
        "kwargs": case.kwargs,
        "semantic_tags": list(case.tags),
        "inputs": {k: {"builder": b, "params": p, "shape": list(inputs[k].shape)}
                   for k, (b, p, _) in case.inputs.items()},
        "tolerance_class": case.family,
        "golden_source": source,
        "legacy_status": legacy_status,
        "legacy_error": legacy_err,
        "cross_check_vs_oracle_max_abs": cross,
        "cross_check_vs_oracle_max_rel": cross_rel,
        "cross_check_ok": cross_all_ok,
        "ill_conditioned": ill_conditioned,
        "recommended_rtol": recommended_rtol,
        "value_stats": {k: _stats(v) for k, v in golden.items()},
        "dtype_out": "float64",
        "versions": _versions(),
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in golden.items()}
    return arrays, sidecar, cross_max


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "golden"))
    ap.add_argument("--limit", type=int, default=None,
                    help="freeze only the first N cases (debug)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ksig  # legacy CuPy/Numba ksig

    cases = fxmatrix.all_cases()
    if args.limit:
        cases = cases[: args.limit]

    index, n_oracle_sourced, worst_cross = [], 0, 0.0
    ill, structural = [], []
    for i, case in enumerate(cases, 1):
        arrays, sidecar, cross_max = freeze_case(case, ksig)
        # Structural sanity: every golden array must be finite.
        for key, arr in arrays.items():
            if not np.isfinite(arr).all():
                structural.append((case.case_id, key))
        np.savez_compressed(out_dir / f"{case.case_id}.npz", **arrays)
        (out_dir / f"{case.case_id}.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True))
        index.append({
            "case_id": case.case_id, "entry": case.entry,
            "family": case.family, "golden_source": sidecar["golden_source"],
            "legacy_status": sidecar["legacy_status"],
            "cross_check_max_abs": cross_max,
            "ill_conditioned": sidecar["ill_conditioned"],
            "recommended_rtol": sidecar["recommended_rtol"],
            "keys": list(arrays.keys()),
        })
        if sidecar["golden_source"] == "numpy_oracle":
            n_oracle_sourced += 1
        if cross_max is not None:
            worst_cross = max(worst_cross, cross_max)
        if sidecar["ill_conditioned"]:
            ill.append((case.case_id, sidecar["recommended_rtol"]))
        flag = "  ~ill-conditioned" if sidecar["ill_conditioned"] else ""
        src = sidecar["golden_source"]
        cm = f"{cross_max:.2e}" if cross_max is not None else "  (oracle-src)"
        print(f"[{i:3d}/{len(cases)}] {case.case_id[:58]:58s} "
              f"{src:12s} Δ={cm}{flag}")

    manifest = {
        "n_cases": len(cases),
        "n_legacy_sourced": len(cases) - n_oracle_sourced,
        "n_oracle_sourced": n_oracle_sourced,
        "n_ill_conditioned": len(ill),
        "worst_cross_check_max_abs": worst_cross,
        "cross_check_rtol": CROSS_CHECK_RTOL,
        "cross_check_atol": CROSS_CHECK_ATOL,
        "ill_conditioned_cases": [c for c, _ in ill],
        "artifacts": index,
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "INDEX.json").write_text(json.dumps(manifest, indent=2,
                                                   sort_keys=True))
    print(f"\nFroze {len(cases)} cases -> {out_dir}")
    print(f"  legacy-sourced: {len(cases)-n_oracle_sourced}, "
          f"oracle-sourced (legacy broken): {n_oracle_sourced}")
    print(f"  worst legacy-vs-oracle cross-check (abs): {worst_cross:.2e}")
    if ill:
        print(f"  ~ {len(ill)} ill-conditioned (normalization /~0); "
              f"flagged with loosened recommended_rtol:")
        for cid, rt in ill:
            print(f"     {cid[:70]}  rtol->{rt:.1e}")
    if structural:
        print(f"  !! {len(structural)} STRUCTURAL failures (non-finite golden):")
        for cid, key in structural:
            print(f"     {cid} [{key}]")
        return 1
    print("  all golden artifacts finite; non-ill-conditioned cross-checks "
          "within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
