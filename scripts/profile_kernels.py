"""Quick capability + performance profile of every KSig kernel on the live device.

Usually launched via the one-shot driver (sources the Aurora env for you):

    bash scripts/run_kernel_benchmarks.sh               # profile + scaling, both

Or run this stage directly on a compute node (after sourcing the env -- see
`memory: aurora-torch-env`):

    source /home/siebenschuh/Projects/Aurora_HPC/environment/activate_ddp_venv.sh
    python scripts/profile_kernels.py                   # auto device; fp32+fp64 on xpu
    python scripts/profile_kernels.py --device cpu      # force a device
    python scripts/profile_kernels.py --quick           # smaller sweep (~1 min)
    python scripts/profile_kernels.py --dtypes float32

Two sections, both printed as plain tables:

  CAPABILITY  -- each kernel builds + computes a Gram on the device and is checked
                 for: finite, symmetric, unit diagonal (normalized kernels). For
                 DETERMINISTIC kernels the device Gram is also cross-checked against
                 the CPU Gram to a rel-err tolerance (random-feature maps use a
                 device-specific RNG stream, so they get structural checks only).
                 The two learnable GSK columns run a short `fit_phi` first.
  PERFORMANCE -- median wall time over a small size sweep, one row per (kernel,
                 dtype). On XPU both float32 and float64 are timed by default --
                 fp64 is heavily throttled on PVC, so the gap is the headline.

This is the BROAD, quick snapshot across *all* kernels including the
GeneralSignatureKernel family and the random-feature maps. For the rigorous,
memory-budgeted, JSONL/CSV artifact on the three DP kernels, use the dedicated
harness instead:

    python -m monitoring.run_benchmarks --tier medium --out monitoring/results
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib
import time
import traceback

import numpy as np

# --- self-pathing: import the in-repo ksig + the notebook helpers without pip.
# This script lives in ./scripts; the shared helpers (_nbtools, _gsk_demo) live
# in ./notebooks, so both dirs and the repo root go on sys.path. ---
_HERE = pathlib.Path(__file__).resolve().parent           # ./scripts
_ROOT = _HERE.parent                                      # repo root
_NB = _ROOT / "notebooks"                                 # shared demo helpers
for _p in (str(_HERE), str(_NB), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ksig                                   # noqa: E402
import _nbtools as nb                         # noqa: E402
import _gsk_demo as g                         # noqa: E402


# ---------------------------------------------------------------------------
# Device / dtype plumbing.
# ---------------------------------------------------------------------------
def pick_device(prefer):
    """Resolve the compute device: explicit override, else CUDA -> XPU -> CPU."""
    import torch
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "xpu", None) and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def device_count(dev):
    import torch
    if dev.type == "xpu":
        return torch.xpu.device_count()
    if dev.type == "cuda":
        return torch.cuda.device_count()
    return 1


def multi_device_smoke(dev):
    """Confirm EVERY visible accelerator can run a matmul (the '6 XPUs alive'
    check). A dead/unreachable device surfaces here, not 30 kernels later."""
    import torch
    n = device_count(dev)
    if dev.type not in ("xpu", "cuda") or n <= 1:
        print(f"  (single-device run on {dev}) ")
        return
    print(f"  {n} {dev.type.upper()} devices visible -- per-device matmul smoke:")
    for i in range(n):
        d = f"{dev.type}:{i}"
        try:
            t0 = time.perf_counter()
            a = torch.randn(1024, 1024, device=d)
            (a @ a).sum().item()
            nb.synchronize(dev.type)
            print(f"    {d}:  OK   ({1e3*(time.perf_counter()-t0):6.1f} ms, "
                  f"1024x1024 matmul)")
        except Exception as e:
            print(f"    {d}:  FAIL  {type(e).__name__}: {e}")


def set_dtype_env(dt):
    """The ported kernels read KSIG_DTYPE per call; set it so a sweep can flip
    fp32/fp64 in one process. (GSK kernels instead inherit the input array's
    dtype -- we cast the input too, so both paths agree.)"""
    os.environ["KSIG_DTYPE"] = dt


# ---------------------------------------------------------------------------
# Kernel registry: name -> spec. `build(dev)` returns a callable Gram(X).
#   kind        : 'dp' (full-rank), 'feature' (needs .fit), 'gsk', 'gsk-learned'
#   deterministic: cross-check vs CPU is meaningful (no per-device RNG)
#   normalized  : a unit diagonal is expected
# ---------------------------------------------------------------------------
def _rbf(bw=1.0):
    return ksig.static.kernels.RBFKernel(bandwidth=bw)


def _build_dp(dev):
    """The three deterministic full-rank kernels (set_default_device drives them)."""
    ksig.set_default_device(str(dev))
    K = ksig.kernels
    return {
        "SignatureKernel":      dict(kind="dp", deterministic=True, normalized=True,
                                     fn=K.SignatureKernel(n_levels=4, order=2,
                                                          normalize=True, static_kernel=_rbf())),
        "SignaturePDEKernel":   dict(kind="dp", deterministic=True, normalized=True,
                                     fn=K.SignaturePDEKernel(static_kernel=_rbf(),
                                                             normalize=True, difference=True)),
        "GlobalAlignmentKernel": dict(kind="dp", deterministic=True, normalized=True,
                                      fn=K.GlobalAlignmentKernel(static_kernel=_rbf(bw=5.0))),
    }


def _build_feature(dev):
    """Random-feature maps: need .fit(X) before a call; output is device-RNG
    dependent, so cross-device cross-check is NOT meaningful (structural only)."""
    ksig.set_default_device(str(dev))
    K, F, P = ksig.kernels, ksig.static.features, ksig.projections
    nc = 100
    specs = {
        "RFSF-TRP":   K.SignatureFeatures(
                          n_levels=4, order=1, normalize=True,
                          static_features=F.RandomFourierFeatures(n_components=nc, random_state=0),
                          projection=P.TensorizedRandomProjection(n_components=nc, rank=1, random_state=0)),
        "LowRankSig": K.SignatureFeatures(
                          n_levels=4, order=1, normalize=True, static_features=None,
                          projection=P.TensorizedRandomProjection(n_components=nc, rank=1, random_state=0)),
        "RandomWarpingSeries": K.RandomWarpingSeries(
                          n_components=nc, stdev=1.0, max_warp=32, normalize=True, random_state=0),
    }
    return {name: dict(kind="feature", deterministic=False, normalized=True, fn=obj)
            for name, obj in specs.items()}


# GSK family columns (config, learned?). Fixed columns are deterministic.
_GSK_COLUMNS = [
    ("GSK:sig-L1",     dict(phi="level_one", truncation=4),               False),
    ("GSK:sig-TRUNC",  dict(phi="const",     truncation=4),               False),
    ("GSK:sig-PDE",    dict(phi="const",     truncation=None),            False),
    ("GSK:sig-EXACT",  dict(phi="const",     truncation=4, normalize="per_level"), False),
    ("GSK:sig-Wphi",   dict(phi="free",      truncation=4),               True),
    ("GSK:sig-PDEphi", dict(phi="dilation",  truncation=None),            True),
]


def _build_gsk(dev):
    out = {}
    for name, cfg, learned in _GSK_COLUMNS:
        out[name] = dict(kind="gsk-learned" if learned else "gsk",
                         deterministic=not learned, normalized=True,
                         cfg=cfg, learned=learned)
    return out


def build_registry(dev):
    reg = {}
    reg.update(_build_dp(dev))
    reg.update(_build_feature(dev))
    reg.update(_build_gsk(dev))
    return reg


def make_gram_callable(name, spec, dev, dtype_np):
    """Return a no-arg-after-X callable computing the Gram for one kernel, with
    any one-time setup (feature .fit / GSK construction / learned fit_phi) done."""
    if spec["kind"] in ("dp", "feature"):
        k = spec["fn"]
        if spec["kind"] == "feature":
            return lambda X: (k.fit(X), k(X))[1]
        return lambda X: k(X)
    # GSK column: construct on the device; input dtype drives the kernel dtype.
    cfg = dict(spec["cfg"]); norm = cfg.pop("normalize", "once")
    bw = 1.0
    if spec["learned"]:
        # one short fit so the eval exercises a real learned phi
        def _call(X):
            from ksig.generalized import GeneralSignatureKernel
            ker = GeneralSignatureKernel(static="rbf", bw=bw, dev=str(dev),
                                         normalize=norm, **cfg)
            y = (np.arange(len(X)) % 2)
            ker.fit_phi(X, y, task="classification", steps=30)
            return np.asarray(ker(X))
        return _call

    def _call(X):
        from ksig.generalized import GeneralSignatureKernel
        ker = GeneralSignatureKernel(static="rbf", bw=bw, dev=str(dev),
                                     normalize=norm, **cfg)
        return np.asarray(ker(X))
    return _call


# ---------------------------------------------------------------------------
# Capability / correctness.
# ---------------------------------------------------------------------------
def capability(reg, dev, dtype_np, tol):
    """Build + run each kernel on a tiny input; check finite/symmetric/unit-diag
    and (deterministic kernels) cross-check vs CPU. Returns rows for printing."""
    import torch
    Xn = nb.simulate(8, 12, 3, seed=0).astype(dtype_np)
    rows = []
    for name, spec in reg.items():
        try:
            set_dtype_env("float64" if dtype_np == np.float64 else "float32")
            K = nb.as_host(make_gram_callable(name, spec, dev, dtype_np)(Xn))
            finite = bool(np.isfinite(K).all())
            sym = float(np.abs(K - K.T).max())
            diag = float(np.diag(K).mean())
            relerr = None
            if spec["deterministic"]:
                # CPU reference (same kernel, CPU device) for a numeric cross-check.
                cpu = torch.device("cpu")
                cpu_reg = {**_build_dp(cpu), **_build_gsk(cpu)}
                set_dtype_env("float64" if dtype_np == np.float64 else "float32")
                Kc = nb.as_host(make_gram_callable(name, cpu_reg[name], cpu, dtype_np)(Xn))
                ksig.set_default_device(str(dev))            # restore
                relerr = float(np.linalg.norm(K - Kc) / (np.linalg.norm(Kc) + 1e-30))
            ok = finite and sym < 1e-3 and (relerr is None or relerr < tol)
            if spec["normalized"] and abs(diag - 1.0) > 1e-2:
                ok = False
            rows.append((name, "PASS" if ok else "FAIL", finite, sym, diag, relerr, ""))
        except Exception as e:
            rows.append((name, "ERROR", False, float("nan"), float("nan"), None,
                         f"{type(e).__name__}: {e}"))
            if os.environ.get("PROFILE_DEBUG"):
                traceback.print_exc()
    return rows


def print_capability(rows, dtype_name):
    print(f"\n=== CAPABILITY  (dtype={dtype_name}, tiny n=8 L=12 d=3) ===")
    print(f"  {'kernel':22s} {'status':6s} {'finite':6s} {'max|K-Kt|':>10s} "
          f"{'diag':>7s} {'relerr/CPU':>11s}  note")
    for name, status, finite, sym, diag, relerr, note in rows:
        re = "-" if relerr is None else f"{relerr:.2e}"
        print(f"  {name:22s} {status:6s} {str(finite):6s} {sym:10.2e} "
              f"{diag:7.4f} {re:>11s}  {note}")


# ---------------------------------------------------------------------------
# Performance.
# ---------------------------------------------------------------------------
def perf_grid(quick):
    """Per-kernel size sweep (n is the Gram dimension). Full-rank kernels are
    O(n^2 L^2); feature maps are ~linear in n, so they get larger n."""
    if quick:
        return dict(dp_n=[16, 32], dp_L=24, feat_n=[64, 128], feat_L=50, fit_n=64, fit_steps=30)
    return dict(dp_n=[16, 32, 64], dp_L=32, feat_n=[64, 128, 256], feat_L=50, fit_n=120, fit_steps=50)


def performance(reg, dev, dtypes, grid, reps, warmup):
    rows = []
    for name, spec in reg.items():
        feature_like = spec["kind"] == "feature"
        ns = grid["feat_n"] if feature_like else grid["dp_n"]
        L = grid["feat_L"] if feature_like else grid["dp_L"]
        d = 5 if feature_like else 3
        for dt in dtypes:
            dtype_np = np.float64 if dt == "float64" else np.float32
            set_dtype_env(dt)
            times = []
            for n in ns:
                Xn = nb.simulate(n, L, d, seed=1).astype(dtype_np)
                try:
                    call = make_gram_callable(name, spec, dev, dtype_np)
                    t = nb.timeit(lambda: call(Xn), reps=reps, warmup=warmup, device=dev.type)
                    times.append(t)
                except Exception as e:
                    times.append(float("nan"))
                    if os.environ.get("PROFILE_DEBUG"):
                        print(f"    [{name} n={n} {dt}] {type(e).__name__}: {e}")
            rows.append((name, dt, ns, L, times))
    return rows


def print_performance(rows, dev):
    print(f"\n=== PERFORMANCE  (median wall time, s; device={dev}) ===")
    # collect the union of n's used per row but print compactly
    print(f"  {'kernel':22s} {'dtype':8s} {'(n: seconds)':<40s}")
    for name, dt, ns, L, times in rows:
        cells = "  ".join(f"{n}:{(t if t==t else float('nan')):.4f}" if t == t else f"{n}:nan"
                          for n, t in zip(ns, times))
        print(f"  {name:22s} {dt:8s} L={L:<4d} {cells}")


def fit_timing(dev, dtypes, grid, reps):
    """Headline cost of the LEARNABLE GSK columns is fit_phi, not the eval. Time it
    separately on one size."""
    from ksig.generalized import GeneralSignatureKernel
    print(f"\n=== LEARNED-phi FIT  (fit_phi, n={grid['fit_n']}, "
          f"steps={grid['fit_steps']}, device={dev}) ===")
    Xtr, ytr = g.make_peak(grid["fit_n"], seed=0)
    bw = g.median_bw(Xtr)
    for dt in dtypes:
        dtype_np = np.float64 if dt == "float64" else np.float32
        Xc = Xtr.astype(dtype_np)
        for name, cfg, learned in _GSK_COLUMNS:
            if not learned:
                continue
            c = dict(cfg); norm = c.pop("normalize", "once")
            def _fit():
                ker = GeneralSignatureKernel(static="rbf", bw=bw, dev=str(dev),
                                             normalize=norm, **c)
                ker.fit_phi(Xc, ytr, task="classification", steps=grid["fit_steps"])
                return ker
            try:
                t = nb.timeit(_fit, reps=max(1, reps // 2), warmup=1, device=dev.type)
                print(f"  {name:22s} {dt:8s}  {t:.3f} s")
            except Exception as e:
                print(f"  {name:22s} {dt:8s}  ERROR {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto | xpu | cuda | cpu | xpu:0 ...")
    ap.add_argument("--dtypes", default="", help="comma list; default fp32,fp64 on xpu else fp64")
    ap.add_argument("--quick", action="store_true", help="smaller sweep (~1 min)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tol", type=float, default=None, help="rel-err tol for CPU cross-check")
    # parse_known_args so the shared driver can forward flags meant for the
    # scaling stage (e.g. --big, --seeds) without erroring here.
    args, _ = ap.parse_known_args(argv)

    dev = pick_device(args.device)
    if args.dtypes:
        dtypes = [s.strip() for s in args.dtypes.split(",") if s.strip()]
    else:
        dtypes = ["float32", "float64"] if dev.type == "xpu" else ["float64"]
    # MPS has no fp64; guard.
    if dev.type == "mps":
        dtypes = ["float32"]
    tol = args.tol if args.tol is not None else 1e-9

    print("=" * 72)
    print(f"KSig kernel profile  |  device={dev}  |  dtypes={dtypes}")
    print(f"torch device count: {device_count(dev)}  |  quick={args.quick}")
    print("=" * 72)
    multi_device_smoke(dev)

    reg = build_registry(dev)
    print(f"\nkernels under test ({len(reg)}): {', '.join(reg)}")

    # Capability is checked once per dtype (fp32 cross-check is looser).
    for dt in dtypes:
        dtype_np = np.float64 if dt == "float64" else np.float32
        t = tol if dt == "float64" else max(tol, 2e-3)   # fp32 cross-check looser
        rows = capability(reg, dev, dtype_np, t)
        print_capability(rows, dt)

    grid = perf_grid(args.quick)
    rows = performance(reg, dev, dtypes, grid, args.reps, args.warmup)
    print_performance(rows, dev)
    fit_timing(dev, dtypes, grid, args.reps)
    print("\ndone.")


if __name__ == "__main__":
    main()
