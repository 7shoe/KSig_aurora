"""Data-scaling sanity check: do the kernels stay numerically sane AND
statistically sensible as the sample size N grows?

Usually launched via the one-shot driver (sources the Aurora env for you):

    bash scripts/run_kernel_benchmarks.sh               # profile + scaling, both

Or run this stage directly on a compute node (after sourcing the env):

    source /home/siebenschuh/Projects/Aurora_HPC/environment/activate_ddp_venv.sh
    python scripts/scaling_check.py                      # N in {50,100,200}
    python scripts/scaling_check.py --big                # add N=400 (heavier)
    python scripts/scaling_check.py --quick              # tiny, ~30 s

Three parts:

  PART A  INVARIANTS vs N  (every kernel) -- at each sample size, the Gram must
          stay finite, symmetric, unit-diagonal (normalized kernels) and PSD
          (min eigenvalue >= -tol). A kernel that silently destabilizes at scale
          shows up as a failing cell.

  PART B  STATISTICS vs N  (signature family) -- out-of-sample CKA on a pure
          ORDER signal (`D_area`: loop orientation). Over several data seeds we
          report mean +/- std. The statistically-reasonable signature:
            * order-AWARE kernels: CKA high and STABLE as N grows;
            * order-BLIND baseline (pooled-RBF): pinned at chance;
            * the seed-to-seed STD SHRINKS as N grows (Monte-Carlo consistency).

  PART C  ORDER-RECOVERY RATE vs N  (learnable sig-Wphi) -- fraction of seeds on
          which the learned phi(k) peaks at the planted level. Should climb
          toward 1.0 as N grows -- the learner gets the order right more reliably
          with more data.

Complements `profile_kernels.py` (capability + wall-time) and
`monitoring/run_benchmarks.py` (rigorous DP-kernel timing artifacts).
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib

import numpy as np

# Self-pathing: this script is in ./scripts; shared helpers (_nbtools, _gsk_demo)
# live in ./notebooks, and profile_kernels.py sits beside this file in ./scripts.
_HERE = pathlib.Path(__file__).resolve().parent           # ./scripts
_ROOT = _HERE.parent                                      # repo root
_NB = _ROOT / "notebooks"                                 # shared demo helpers
for _p in (str(_HERE), str(_NB), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ksig                                          # noqa: E402
import _nbtools as nb                                # noqa: E402
import _gsk_demo as g                                # noqa: E402
import profile_kernels as pk                         # noqa: E402  (reuse the registry)


# ---------------------------------------------------------------------------
# PART A -- numerical invariants as N grows (every kernel).
# ---------------------------------------------------------------------------
def invariants_vs_N(dev, Ns, dtype_np, psd_tol):
    """For each kernel and N: finite, symmetry, unit-diag, min eigenvalue."""
    pk.set_dtype_env("float64" if dtype_np == np.float64 else "float32")
    reg = pk.build_registry(dev)
    print(f"\n=== PART A — INVARIANTS vs N  (dtype={dtype_np.__name__}, L=22, d=2) ===")
    print(f"  {'kernel':22s} " + "  ".join(f"N={n:<4d}" for n in Ns) +
          "     (cell = min-eig ; * = invariant break)")
    for name, spec in reg.items():
        cells = []
        for n in Ns:
            Xn = _paths(n, dtype_np)
            try:
                pk.set_dtype_env("float64" if dtype_np == np.float64 else "float32")
                K = np.asarray(pk.make_gram_callable(name, spec, dev, dtype_np)(Xn))
                K = K.astype(np.float64)
                finite = np.isfinite(K).all()
                sym = np.abs(K - K.T).max()
                diag_ok = (not spec["normalized"]) or abs(np.diag(K).mean() - 1) < 1e-2
                mineig = float(np.linalg.eigvalsh((K + K.T) / 2).min())
                broke = (not finite) or sym > 1e-2 or (not diag_ok) or mineig < -psd_tol
                cells.append(f"{mineig:+.1e}{'*' if broke else ' '}")
            except Exception as e:
                cells.append(f"ERR:{type(e).__name__[:6]}")
        print(f"  {name:22s} " + "  ".join(f"{c:>9s}" for c in cells))
    print(f"  (PSD ok when min-eig >= -{psd_tol:g}; '*' marks a finite/symmetry/diag/PSD break)")


def _paths(n, dtype_np):
    """Generic, NON-DEGENERATE paths for the invariant check: integrated random
    walks (nonzero net displacement, so every signature level carries energy).
    Deliberately NOT the closed-loop DGPs — those have ~zero level-1 energy and
    trip the *documented* per-level/level-1 normalization fragility (see
    `test_perlevel_path_whitens_each_level`), which would mask genuine
    scaling-stability with a known-degenerate-data artifact."""
    return nb.simulate(n, g.L, g.D, seed=0).astype(dtype_np)


# ---------------------------------------------------------------------------
# PART B -- discriminative statistics as N grows (signature family).
# ---------------------------------------------------------------------------
def _score_sig(name, cfg, learned, Xtr, ytr, Xte, yte, bw, dev, steps):
    """Out-of-sample CKA of one GSK column (device-aware build)."""
    import torch
    torch.manual_seed(0)
    c = dict(cfg); norm = c.pop("normalize", "once")
    ker = g.gsk(c["phi"], c["truncation"], normalize=norm, bw=bw, dev=str(dev))
    if learned:
        ker.fit_phi(Xtr, ytr, task="classification", steps=steps)
    return g.cka(np.asarray(ker(Xte)), yte)


# (column name, config, learned?)  — the order-blind ref is handled separately.
_COLUMNS = [
    ("sig-L1",     dict(phi="level_one", truncation=g.N_LEVELS), False),
    ("sig-TRUNC",  dict(phi="const",     truncation=g.N_LEVELS), False),
    ("sig-PDE",    dict(phi="const",     truncation=None),       False),
    ("sig-EXACT",  dict(phi="const",     truncation=g.N_LEVELS, normalize="per_level"), False),
    ("sig-Wphi",   dict(phi="free",      truncation=g.N_LEVELS), True),
    ("sig-PDEphi", dict(phi="dilation",  truncation=None),       True),
]


def statistics_vs_N(dev, Ns, seeds, dtype_np, steps):
    """CKA mean +/- std on D_area (pure order signal) vs N, over data seeds."""
    pk.set_dtype_env("float64" if dtype_np == np.float64 else "float32")
    cols = ["pooled-RBF(chance)"] + [c[0] for c in _COLUMNS]
    print(f"\n=== PART B — CKA vs N on D_area (order signal), mean±std over {len(seeds)} seeds ===")
    print(f"  {'N':>5s}  " + "  ".join(f"{c:>18s}" for c in cols))
    table = {c: [] for c in cols}
    for n in Ns:
        acc = {c: [] for c in cols}
        for s in seeds:
            Xtr, ytr, Xte, yte = g.split(g.make_area, seed=s, n_train=n, n_test=n)
            Xtr = Xtr.astype(dtype_np); Xte = Xte.astype(dtype_np)
            bw = g.median_bw(Xtr)
            acc["pooled-RBF(chance)"].append(g.cka(g.pooled_rbf_gram(Xte, bw), yte))
            for cname, cfg, learned in _COLUMNS:
                acc[cname].append(_score_sig(cname, cfg, learned, Xtr, ytr,
                                             Xte, yte, bw, dev, steps))
        cellstrs = []
        for c in cols:
            m, sd = float(np.mean(acc[c])), float(np.std(acc[c]))
            table[c].append((m, sd)); cellstrs.append(f"{m:+.3f}±{sd:.3f}")
        print(f"  {n:>5d}  " + "  ".join(f"{cs:>18s}" for cs in cellstrs))
    # Headline reasonableness #1: the order-aware mean is HIGH & STABLE in N
    # (a consistent estimator does not drift/degrade as data grows), and the
    # order-BLIND baseline is pinned near 0.
    print("\n  mean(CKA) vs N — order-aware: high & STABLE; pooled-RBF: pinned ~0:")
    for c in cols:
        ms = [m for m, _ in table[c]]
        print(f"    {c:20s} " + " -> ".join(f"{m:+.3f}" for m in ms))
    # Headline reasonableness #2: the seed-to-seed std shrinks (Monte-Carlo
    # consistency). Cleanest on the baselines; the strong-signal kernels already
    # sit near their variance floor (~0.02), so we only flag a clear BLOW-UP.
    print("\n  std(CKA) vs N — should not GROW with N (strong kernels are near the floor):")
    for c in cols:
        sds = [sd for _, sd in table[c]]
        grew = sds[-1] > 1.5 * sds[0] + 2e-2
        print(f"    {c:20s} " + " -> ".join(f"{sd:.3f}" for sd in sds) +
              f"   [{'GROWS (!)' if grew else 'stable/shrinks'}]")
    return table, cols


# ---------------------------------------------------------------------------
# PART C -- order-recovery reliability as N grows (sig-Wphi).
# ---------------------------------------------------------------------------
def recovery_vs_N(dev, Ns, seeds, dtype_np, steps):
    import torch
    print(f"\n=== PART C — order-recovery RATE vs N  (sig-Wphi, over {len(seeds)} seeds) ===")
    print(f"  fraction of seeds where argmax phi(1:N) == planted level\n")
    print(f"  {'planted':>8s}  " + "  ".join(f"N={n:<5d}" for n in Ns))
    for level in (1, 2, 3):
        rates = []
        for n in Ns:
            hits = 0
            for s in seeds:
                Xtr, ytr, _, _ = g.split(lambda nn, ss: g.make_planted(level, nn, ss),
                                         seed=s, n_train=n, n_test=2)
                Xtr = Xtr.astype(dtype_np); bw = g.median_bw(Xtr)
                torch.manual_seed(0)
                k = g.gsk("free", g.N_LEVELS, bw=bw, dev=str(dev)).fit_phi(
                    Xtr, ytr, task="classification", steps=steps)
                phi = np.asarray(k.phi_profile())
                if 1 + int(np.argmax(phi[1:])) == level:
                    hits += 1
            rates.append(hits / len(seeds))
        print(f"  level {level:>2d}  " + "  ".join(f"{r:>6.2f} " for r in rates))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="scaling runs default to float32 (lighter at large N)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120, help="fit_phi steps for learned columns")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--big", action="store_true", help="add N=400")
    ap.add_argument("--psd-tol", type=float, default=1e-3)
    # parse_known_args so the shared driver can forward profile-stage flags
    # (e.g. --dtypes, --reps) without erroring here.
    args, _ = ap.parse_known_args(argv)

    dev = pk.pick_device(args.device)
    dtype_np = np.float64 if args.dtype == "float64" else np.float32
    if args.quick:
        Ns, seeds, steps = [40, 80], 3, 80
    else:
        Ns, seeds, steps = [50, 100, 200], args.seeds, args.steps
        if args.big:
            Ns = Ns + [400]
    seedlist = list(range(seeds))

    print("=" * 72)
    print(f"KSig data-scaling check  |  device={dev}  |  dtype={args.dtype}")
    print(f"N grid: {Ns}  |  seeds: {seeds}  |  fit steps: {steps}")
    print("=" * 72)

    invariants_vs_N(dev, Ns, dtype_np, args.psd_tol)
    statistics_vs_N(dev, Ns, seedlist, dtype_np, steps)
    recovery_vs_N(dev, Ns, seedlist, dtype_np, steps)
    print("\ndone.")


if __name__ == "__main__":
    main()
