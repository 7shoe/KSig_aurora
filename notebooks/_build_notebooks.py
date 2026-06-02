"""Generate the demo notebooks from one place, so they stay consistent and can
be regenerated whenever the API or the frozen reference changes.

    python notebooks/_build_notebooks.py        # (re)writes notebooks/0X_*.ipynb

Each notebook embeds the NVIDIA-CUDA reference (``cuda_reference.json``) *as text*
in the markdown cell directly above the matching code cell, so a reader sees the
expected H100 output next to the code before running it.  Running the code cell
on the target machine prints the live output to compare, and the scaling cell at
the bottom draws green (CUDA reference) vs blue (this machine) vs — only when a
SYCL build is present — orange.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = json.loads((HERE / "cuda_reference.json").read_text())
GPU = REF["meta"]["gpu"]
CUPY_V = REF["meta"]["versions"].get("cupy")


# ---------------------------------------------------------------------------
# nbformat-v4 cell/dict helpers (no nbformat dependency).
# ---------------------------------------------------------------------------
def md(stem, i, text):
    return {"cell_type": "markdown", "id": f"{stem}-md{i}",
            "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(stem, i, text):
    return {"cell_type": "code", "id": f"{stem}-cd{i}", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write(stem, cells):
    path = HERE / f"{stem}.ipynb"
    path.write_text(json.dumps(notebook(cells), indent=1))
    print(f"wrote {path.name}  ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# Shared cells.
# ---------------------------------------------------------------------------
SETUP = """
import sys, pathlib
# Make `_nbtools` and the in-repo `ksig` importable whether the notebook is
# launched from ./notebooks or from the repo root (no `pip install` needed).
_nbdir = pathlib.Path.cwd()
_root = _nbdir.parent if (_nbdir / "_nbtools.py").exists() else _nbdir
_nbdir = _root / "notebooks"
for _p in (str(_nbdir), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import ksig
import _nbtools as nb
%matplotlib inline

ENV = nb.detect_env()
nb.print_env_banner(ENV)
"""


def ref_block_text(shape, block, diag_mean):
    return ("gram shape : " + str(tuple(shape)) + "\n"
            "K[:3,:3]   :\n " + str(np.round(np.array(block), 4)) + "\n"
            "diag mean  : " + str(diag_mean))


def scaling_md(stem, ref_key, sycl_supported=False):
    sc = REF[ref_key]["scaling"]
    orange = ("""
* 🟧 **orange** — the **SYCL** fast-path (this kernel dispatches to `ksig._sycl`),
  drawn **only if** a build is present (`nb.sycl_available()`); the blue curve is
  the same kernel with SYCL forced off, so blue-vs-orange is the head-to-head.
""" if sycl_supported else """
> This kernel has **no SYCL fast-path** — only SigPDE / GAK / RWS dispatch to
> `ksig._sycl` — so there is no orange curve here, just green vs blue.
""")
    return md(stem, 90, f"""
## Scaling — green = CUDA reference, blue = this machine

The cell below sweeps **{sc['axis']}** and times each point on whatever backend
is live, then overlays:

* 🟩 **green** — the frozen reference measured on **{GPU}** (`cuda_reference.json`),
* 🟦 **blue** — what *this* machine computes now (torch-native on Aurora XPU / CUDA / CPU),
{orange.strip()}

The grid and the knobs at the top of the cell are **tunable** — they default to
the reference grid so blue lines up with green; widen them to push the frontier.
""")


def _sweep_tail(grid_var, loop_var, ref_key, title, sycl_supported):
    """The blue(/orange) sweep + plot, shared by the kernel/feature cells."""
    if sycl_supported:
        return f"""
# BLUE = torch-native baseline: force the SYCL fast-path OFF so this curve is
# the eager wavefront (on XPU, SYCL auto-engages by default, which would make
# blue == orange). A no-op where SYCL is absent.
nb.enable_sycl(False)
times = [time_one({loop_var}) for {loop_var} in {grid_var}]

# ORANGE = SYCL fast-path. Drawn ONLY when a SYCL build is available; run the
# same sweep with the fast-path on, then restore the torch-native default.
sycl_times = None
if nb.sycl_available():
    nb.enable_sycl(True)
    sycl_times = [time_one({loop_var}) for {loop_var} in {grid_var}]
    nb.enable_sycl(False)

nb.scaling_plot({grid_var}, times, "{ref_key}", sycl_seconds=sycl_times,
                title="{title}");
"""
    return f"""
# This kernel has no SYCL fast-path (only SigPDE / GAK / RWS dispatch to
# ksig._sycl), so there is no orange curve -- just green vs blue.
times = [time_one({loop_var}) for {loop_var} in {grid_var}]
nb.scaling_plot({grid_var}, times, "{ref_key}", title="{title}");
"""


def scaling_cell_kernel(stem, ref_key, build_expr, n, d, tunables,
                        sycl_supported=False):
    """Scaling cell for a deterministic kernel (sweeps sequence length L)."""
    grid = REF[ref_key]["scaling"]["x"]
    tun = "\n".join(tunables)
    tail = _sweep_tail("L_GRID", "L", ref_key,
                       f"{ref_key} — wall time vs sequence length",
                       sycl_supported)
    return code(stem, 91, f"""
# --- tunable knobs (default to the CUDA-reference grid) ---------------------
L_GRID = {grid}          # sequence lengths to sweep
N, D   = {n}, {d}                 # fixed batch size / channels
REPS   = 5
{tun}

def time_one(L):
    Xs = nb.simulate(N, L, D, seed=1)
    k = {build_expr}
    return nb.timeit(lambda: k(Xs), reps=REPS, device=ENV["device"])
{tail}""")


def scaling_cell_feature(stem, ref_key, build_expr, L, d, tunables,
                         sycl_supported=False):
    """Scaling cell for a feature kernel (sweeps n_samples N)."""
    grid = REF[ref_key]["scaling"]["x"]
    tun = "\n".join(tunables)
    tail = _sweep_tail("N_GRID", "N", ref_key,
                       f"{ref_key} — wall time vs n_samples",
                       sycl_supported)
    return code(stem, 91, f"""
# --- tunable knobs (default to the CUDA-reference grid) ---------------------
N_GRID = {grid}     # sample counts to sweep (feature methods scale ~linearly)
L, D   = {L}, {d}                  # fixed sequence length / channels
REPS   = 5
{tun}

def time_one(N):
    Xs = nb.simulate(N, L, D, seed=1)
    k = {build_expr}
    k.fit(Xs)
    return nb.timeit(lambda: k(Xs), reps=REPS, device=ENV["device"])
{tail}""")


# ---------------------------------------------------------------------------
# 1-3: deterministic full-rank kernels (elementwise-portable correctness).
# ---------------------------------------------------------------------------
def build_kernel_nb(stem, ref_key, title, intro, sim_expr, build_expr,
                    compute_var, scaling_build, n, d, tunables,
                    sycl_supported=False):
    D = REF[ref_key]["demo"]
    cells = [
        md(stem, 0, f"# {title}\n\n{intro}"),
        md(stem, 1, "## Environment\nDetect the live backend/device and whether "
                    "a SYCL fast-path is available."),
        code(stem, 1, SETUP),
        md(stem, 2, f"## Deterministic input\n`{D['input']}` — a batch of "
                    "integrated random walks (portable: the NumPy RNG is "
                    "bit-identical on every machine, so the values below are "
                    "reproducible on Aurora)."),
        code(stem, 2, f"X = {sim_expr}\nprint('X shape:', X.shape, '| dtype:', X.dtype)"),
        md(stem, 3, f"""
## Compute the kernel

```python
{compute_var} = {build_expr}
K = {compute_var}(X)
print("gram shape :", tuple(K.shape))
print("K[:3,:3]   :\\n", np.round(nb.as_host(K)[:3, :3], 4))
print("diag mean  :", round(float(np.diag(nb.as_host(K)).mean()), 6))
```

**Reference output — {GPU} (CuPy {CUPY_V}), kwargs: `{D['kwargs']}`:**

```text
{ref_block_text(D['gram_shape'], D['gram_block'], D['diag_mean'])}
```

Running the next cell on the target machine should reproduce this to ~1e-8 in
float64 (`diag == 1` exactly when `normalize=True`).
"""),
        code(stem, 3, f"""
{compute_var} = {build_expr}
K = {compute_var}(X)
print("gram shape :", tuple(K.shape))
print("K[:3,:3]   :\\n", np.round(nb.as_host(K)[:3, :3], 4))
print("diag mean  :", round(float(np.diag(nb.as_host(K)).mean()), 6))
"""),
        scaling_md(stem, ref_key, sycl_supported),
        scaling_cell_kernel(stem, ref_key, scaling_build, n, d, tunables,
                            sycl_supported),
    ]
    write(stem, cells)


# ---------------------------------------------------------------------------
# 4-6: randomised feature maps (portable invariants + Monte-Carlo error).
# ---------------------------------------------------------------------------
def build_feature_nb(stem, ref_key, title, intro, sim_expr, build_expr,
                     scaling_build, L, d, tunables, kind,
                     sycl_supported=False):
    D = REF[ref_key]["demo"]
    if kind == "approx":           # rfsf / low-rank: compare to the exact kernel
        conv = "  ".join(f"{c['n_components']}:{c['rel_err']}"
                         for c in D["convergence"])
        ref_text = (f"gram shape   : {tuple(D['gram_shape'])}\n"
                    f"feature shape: {tuple(D['feature_shape'])}\n"
                    f"diag mean    : {D['diag_mean']}\n"
                    f"rel err vs exact: {D['rel_err_vs_exact']}")
        live = f"""
N_COMPONENTS = 100                   # the value the reference was frozen at
k = {build_expr}
k.fit(X)
K = nb.as_host(k(X)); P = nb.as_host(k.transform(X))
exact = ksig.kernels.SignatureKernel(n_levels=4, order=1, normalize=True,
                                     static_kernel=ksig.static.kernels.RBFKernel())
Kex = nb.as_host(exact(X))
rel = float(np.linalg.norm(K - Kex) / np.linalg.norm(Kex))
print("gram shape   :", tuple(K.shape))
print("feature shape:", tuple(P.shape))
print("diag mean    :", round(float(np.diag(K).mean()), 6))
print("rel err vs exact:", round(rel, 4))
"""
        portab = (f"""
**Reference output — {GPU} (CuPy {CUPY_V}), kwargs: `{D['kwargs']}`:**

```text
{ref_text}
```

> ⚠️ **Portability:** these feature maps are **random**. `diag == 1` (from
> `normalize=True`) and the Gram's symmetry are exact on any backend, but the
> *element values* and the exact `rel err` depend on the RNG stream, which
> differs between CuPy and torch. Expect the **same ballpark** rel-err, and the
> same **convergence** — it shrinks as `n_components` grows:
> `{conv}` (n_components:rel_err, on {GPU}).
""")
    else:                           # rws: structural invariants only
        ref_text = (f"gram shape : {tuple(D['gram_shape'])}\n"
                    f"diag mean  : {D['diag_mean']}\n"
                    f"symmetric  : {D['symmetric_max_abs']}\n"
                    f"min eig    : {D['min_eig']}")
        live = """
N_COMPONENTS = 100                   # the value the reference was frozen at
k = {build}
k.fit(X)
K = nb.as_host(k(X))
print("gram shape :", tuple(K.shape))
print("diag mean  :", round(float(np.diag(K).mean()), 6))
print("symmetric  :", round(float(np.abs(K - K.T).max()), 8))
print("min eig    :", round(float(np.linalg.eigvalsh((K + K.T) / 2).min()), 6))
""".replace("{build}", build_expr)
        portab = (f"""
**Reference output — {GPU} (CuPy {CUPY_V}), kwargs: `{D['kwargs']}`:**

```text
{ref_text}
```

> ⚠️ **Portability:** RWS is random + DTW-based. The portable invariants are
> `diag == 1`, symmetry (`max|K - Kᵀ| == 0`) and PSD-ness (`min eig ≥ 0` up to
> rounding). Element values differ across backends by RNG stream.
""")

    cells = [
        md(stem, 0, f"# {title}\n\n{intro}"),
        md(stem, 1, "## Environment\nDetect the live backend/device and whether "
                    "a SYCL fast-path is available."),
        code(stem, 1, SETUP),
        md(stem, 2, f"## Deterministic input\n`{D['input']}` — integrated random "
                    "walks (the NumPy RNG is identical on every machine)."),
        code(stem, 2, f"X = {sim_expr}\nprint('X shape:', X.shape, '| dtype:', X.dtype)"),
        md(stem, 3, f"## Compute the feature map\n{portab}"),
        code(stem, 3, live),
        scaling_md(stem, ref_key, sycl_supported),
        scaling_cell_feature(stem, ref_key, scaling_build, L, d, tunables,
                            sycl_supported),
    ]
    write(stem, cells)


# ===========================================================================
# The six notebooks.
# ===========================================================================
RBF = "ksig.static.kernels.RBFKernel()"

build_kernel_nb(
    "01_signature_kernel", "signature_kernel",
    "Signature Kernel — full-rank, dynamic programming",
    "The exact truncated signature kernel ([Király & Oberhauser, "
    "JMLR 2019](https://jmlr.org/papers/volume20/16-314/16-314.pdf), Alg. 3 & 6): "
    "lift a static kernel on $\\mathbb{R}^d$ to a kernel on sequences. This is the "
    "headline 'plain vanilla' kernel; the `order` knob is the embedding-order cap "
    "(1 = fast first-order path, >1 = higher-order).",
    "nb.simulate(16, 20, 3, seed=0)",
    "ksig.kernels.SignatureKernel(n_levels=4, order=2, normalize=True, "
    f"static_kernel={RBF})",
    "sig",
    "ksig.kernels.SignatureKernel(n_levels=N_LEVELS, order=ORDER, "
    f"normalize=True, static_kernel={RBF})",
    n=32, d=3,
    tunables=["N_LEVELS = 4", "ORDER = 2                         # embedding-order cap (1, 2, 3, ...)"],
)

build_kernel_nb(
    "02_signature_pde_kernel", "signature_pde_kernel",
    "Signature-PDE Kernel",
    "Approximates the *untruncated* signature kernel by solving a Goursat PDE "
    "([Salvi et al., 2021](https://arxiv.org/pdf/2006.14794.pdf)). No truncation "
    "level — `difference` toggles taking increments of the lifted paths.",
    "nb.simulate(16, 20, 3, seed=0)",
    "ksig.kernels.SignaturePDEKernel(difference=True, normalize=True, "
    f"static_kernel={RBF})",
    "pde",
    "ksig.kernels.SignaturePDEKernel(difference=DIFFERENCE, normalize=True, "
    f"static_kernel={RBF})",
    n=32, d=3,
    tunables=["DIFFERENCE = True                 # increments of the lifted path"],
    sycl_supported=True,
)

build_kernel_nb(
    "03_global_alignment_kernel", "global_alignment_kernel",
    "Global Alignment Kernel (GAK)",
    "A similarity that sums over **all** pairwise alignments of two sequences "
    "([Cuturi, 2007](https://members.cbio.mines-paristech.fr/~jvert/publi/pdf/Cuturi2007Kernel.pdf), "
    "eq. 1), computed stably in log-space and normalized. A wider RBF bandwidth "
    "keeps the off-diagonal similarities informative.",
    "nb.simulate(16, 8, 3, seed=0)",
    "ksig.kernels.GlobalAlignmentKernel("
    "static_kernel=ksig.static.kernels.RBFKernel(bandwidth=5.0))",
    "gak",
    "ksig.kernels.GlobalAlignmentKernel("
    "static_kernel=ksig.static.kernels.RBFKernel(bandwidth=BANDWIDTH))",
    n=32, d=3,
    tunables=["BANDWIDTH = 5.0                   # RBF bandwidth of the static kernel"],
    sycl_supported=True,
)

build_feature_nb(
    "04_rfsf_features", "rfsf_features",
    "Random Fourier Signature Features (RFSF)",
    "An unbiased, finite-dimensional approximation to the signature kernel "
    "([Tóth et al., 2023](https://arxiv.org/pdf/2311.12214.pdf), Alg. 2 & 3): "
    "Random Fourier Features for the static kernel + a tensorized random "
    "projection (RFSF-TRP). Scales **linearly** in the number of sequences, so it "
    "is the way to run the signature kernel on large $N$.",
    "nb.simulate(24, 20, 3, seed=0)",
    "ksig.kernels.SignatureFeatures(n_levels=4, order=1, normalize=True, "
    "static_features=ksig.static.features.RandomFourierFeatures("
    "n_components=N_COMPONENTS, random_state=0), "
    "projection=ksig.projections.TensorizedRandomProjection("
    "n_components=N_COMPONENTS, rank=1, random_state=0))",
    "ksig.kernels.SignatureFeatures(n_levels=4, order=1, normalize=True, "
    "static_features=ksig.static.features.RandomFourierFeatures("
    "n_components=N_COMPONENTS, random_state=0), "
    "projection=ksig.projections.TensorizedRandomProjection("
    "n_components=N_COMPONENTS, rank=1, random_state=0))",
    L=50, d=5,
    tunables=["N_COMPONENTS = 100                # more components -> tighter approximation"],
    kind="approx",
)

build_feature_nb(
    "05_low_rank_signature", "low_rank_signature",
    "Low-Rank Signature Features",
    "Signature features via an iterative low-rank approximation of the tensor "
    "outer products ([Király & Oberhauser, JMLR 2019]"
    "(https://jmlr.org/papers/volume20/16-314/16-314.pdf), Alg. 5): a random "
    "projection of the signature, with **no** Random Fourier static features. "
    "`P @ P.T` approximates the full-rank Gram.",
    "nb.simulate(24, 20, 3, seed=0)",
    "ksig.kernels.SignatureFeatures(n_levels=4, order=1, normalize=True, "
    "static_features=None, "
    "projection=ksig.projections.TensorizedRandomProjection("
    "n_components=N_COMPONENTS, rank=1, random_state=0))",
    "ksig.kernels.SignatureFeatures(n_levels=4, order=1, normalize=True, "
    "static_features=None, "
    "projection=ksig.projections.TensorizedRandomProjection("
    "n_components=N_COMPONENTS, rank=1, random_state=0))",
    L=50, d=5,
    tunables=["N_COMPONENTS = 100                # projection width (rank of the approx)"],
    kind="approx",
)

build_feature_nb(
    "06_random_warping_series", "random_warping_series",
    "Random Warping Series (RWS)",
    "DTW-based random features ([Wu et al., 2018]"
    "(https://proceedings.mlr.press/v84/wu18b/wu18b.pdf), Alg. 1): align each "
    "sequence against a bank of random warping series and use the soft-DTW "
    "distances as features. Normalized to unit norm.",
    "nb.simulate(24, 20, 3, seed=0)",
    "ksig.kernels.RandomWarpingSeries(n_components=N_COMPONENTS, stdev=1.0, "
    "max_warp=32, normalize=True, random_state=0)",
    "ksig.kernels.RandomWarpingSeries(n_components=N_COMPONENTS, stdev=1.0, "
    "max_warp=32, normalize=True, random_state=0)",
    L=50, d=5,
    tunables=["N_COMPONENTS = 100                # number of random warping series"],
    kind="rws",
    sycl_supported=True,
)

print("done.")
