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


# Setup for the General-Signature-Kernel notebooks (07 / X): same self-pathing as
# SETUP, but also imports the `_gsk_demo` helper (DGPs, CKA matrix, plots) and
# pins the device to CPU -- these are small *inductive-bias* demos (≤120 paths,
# L=22), not the throughput sweeps of 01–06, so CPU is fast and deterministic.
SETUP_GSK = """
import sys, pathlib
_nbdir = pathlib.Path.cwd()
_root = _nbdir.parent if (_nbdir / "_nbtools.py").exists() else _nbdir
_nbdir = _root / "notebooks"
for _p in (str(_nbdir), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
import ksig
ksig.set_default_device("cpu")          # statistical demos run on CPU (small, reproducible)
import _nbtools as nb
import _gsk_demo as g
%matplotlib inline

ENV = nb.detect_env()
nb.print_env_banner(ENV)
print("GSK demo  | paths/DGP:", g.N_TRAIN, "| length:", g.L, "| channels:", g.D,
      "| truncation N:", g.N_LEVELS, "| fit steps:", g.FIT_STEPS)
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


# ---------------------------------------------------------------------------
# 07 / X: the General Signature Kernel family (inductive-bias demos).
# These are NOT throughput sweeps — they are statistical demos showing which
# kernel reads which structural locus, and whether the learned phi recovers the
# planted order. They share `_gsk_demo` (DGPs, CKA matrix, plots) and have no
# CUDA-reference / green curve (there is nothing to time against).
# ---------------------------------------------------------------------------
def build_gsk_overview_nb():
    """07_general_signature_kernel.ipynb — the six-column GSK family and the
    'data each kernel excels on' CKA confusion matrix."""
    stem = "07_general_signature_kernel"
    cells = [
        md(stem, 0, r"""
# The General Signature Kernel — one object, six kernels

`ksig.generalized.GeneralSignatureKernel` (GSK) is a **single configurable
object** that reproduces six signature-based kernels. Five are the *same*
normalize-once kernel
$$K_\varphi=\sum_{k\ge0}\varphi(k)\,K_k$$
under a choice of **order-weighting** $\varphi(k)$, **truncation depth**, and
**normalization**; the sixth (`sig-EXACT`) is a legacy per-level-normalized
average kept for its inductive bias. $K_k$ is the level-$k$ signature kernel
$\langle S^k(x),S^k(y)\rangle$ — homogeneous of degree $k$ — so *choosing a
kernel is choosing $\varphi$*.

This notebook is the **map of the family**: what each column is, and — the
centerpiece — **which kernel reads which structural locus of a path**, measured
as out-of-sample kernel–target alignment (CKA) on six level-localized
data-generating processes. The companion **`X_learned_kernel.ipynb`** zooms into
the two *learnable* columns (`sig-Wphi`, `sig-PDEphi`) and the order-recovery
question.

Reference: `docs/SIGNATURE_KERNELS.md` (derives all six as specializations of one
skeleton); the matrix below is pinned by
`tests/test_signature_kernel_inductive_bias.py`.
"""),
        md(stem, 1, "## Environment\nDetect the live backend/device. The GSK demos "
                    "run on CPU (small, reproducible); the throughput story for the "
                    "ported kernels is in notebooks `01`–`06`."),
        code(stem, 1, SETUP_GSK),
        md(stem, 2, r"""
## The six columns

| column | `phi` | `truncation` | `normalize` | $\varphi(k)$ | learns? |
|---|---|---|---|---|---|
| `sig-L1` | `level_one` | $N$ | `once` | $e_1$ (level-1 only) | no |
| `sig-TRUNC` | `const` | $N$ | `once` | $1$ (truncated) | no |
| `sig-PDE` | `const` | `None` | `once` | $1$ (Goursat, untruncated) | no |
| `sig-Wphi` | `free` | $N$ | `once` | $\mathrm{softplus}(\theta)\ge0$ | **yes** |
| `sig-PDEphi` | `dilation` | `None` | `once` | $\sum_i w_i\lambda_i^{k}$ | **yes** |
| `sig-EXACT` | `const` | $N$ | `per_level` | $1$, per-level whitened | no |

$\varphi(0)\equiv1$ on every arm (the rank-1 level $K_0=\mathbf{1}\mathbf{1}^\top$
is weighted identically), so cross-column deltas carry no level-0 mismatch. The
cell below builds each fixed-$\varphi$ column and prints its $\varphi$ profile.
"""),
        code(stem, 2, r"""
for name, (phi, trunc, norm) in {
        "sig-L1":    ("level_one", g.N_LEVELS, "once"),
        "sig-TRUNC": ("const",     g.N_LEVELS, "once"),
        "sig-PDE":   ("const",     None,       "once"),
        "sig-EXACT": ("const",     g.N_LEVELS, "per_level"),
}.items():
    k = g.gsk(phi, trunc, normalize=norm, bw=1.0)
    print(f"{name:10s}  phi(0:N) = {np.round(k.phi_profile(g.N_LEVELS), 3)}")
print("\n(sig-Wphi / sig-PDEphi learn phi from data -> see X_learned_kernel.ipynb)")
"""),
        md(stem, 3, r"""
## The data each kernel excels on

Six **level-localized** DGPs, each planting a binary-class signal at one
structural locus of the path (so the *matched* kernel should win):

| DGP | what is planted | who should read it |
|---|---|---|
| `D_disp` | level-1 net displacement | order-blind suffices (no order gain) |
| `D_area` | level-2 signed area (**order**) | every order-aware kernel; blind reps at chance |
| `D_peak` | level-2 peak, levels 1 & 3 noise | only free weights (`sig-Wphi`) can peak |
| `D_scale` | level $\ge$ 3 tail amplitude | clamp chain `Wphi > PDE > PDEphi` |
| `D_lowsig` | level-2 signal + tail noise | soft-decay `PDEphi > PDE` |
| `D_perlevel` | level-2 under wide per-path scale | only per-level whitening (`sig-EXACT`) |

First, a look at the cleanest one — `D_area`: a closed loop whose **time
direction** (CW vs CCW) is the class. The endpoint and the set of points are
identical across classes; only the **signed area** (a level-2, order-only
statistic) differs.
"""),
        code(stem, 3, r"""
Xtr, ytr, _, _ = g.split(g.make_area, seed=list(g.DATASETS).index("D_area"))
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for cls, a in zip((0, 1), ax):
    for i in np.where(ytr == cls)[0][:6]:
        a.plot(Xtr[i, :, 0], Xtr[i, :, 1], alpha=0.7)
        a.plot(*Xtr[i, 0], "ko", ms=3)
    a.set_title(f"class {cls}  ({'CCW' if cls == 0 else 'CW'} loop)"); a.set_aspect("equal")
fig.suptitle("D_area — class = loop orientation (a level-2 / order-only signal)")
plt.tight_layout(); plt.show()
"""),
        md(stem, 4, r"""
## The confusion matrix — who reads what

For each DGP we fit the two learnable kernels on the **train** split, then score
**every** kernel out-of-sample by CKA on the **test** split (centered alignment
of the Gram with $yy^\top$ — the same objective `fit_phi` maximizes). The
per-row winner is boxed.

**Reference (CPU, `GeneralSignatureKernel`, 2026-06-11) — the predicted diagonal:**

```text
dataset      sig-L1  sig-TRUNC  sig-PDE  sig-Wphi  sig-PDEphi  sig-EXACT  pooled  kme
D_disp       +0.224  +0.240     +0.241   +0.248    +0.249      +0.229     +0.295  +0.278
D_area       +0.004  +0.875     +0.836   +0.915    +0.823      -0.008     +0.010  +0.009
D_peak       +0.015  +0.088     +0.085   +0.182    +0.047      +0.134     +0.008  +0.013
D_scale      +0.031  +0.255     +0.198   +0.800    +0.082      +0.079     +0.011  +0.125
D_lowsig     -0.005  +0.202     +0.379   +0.219    +0.433      -0.012     +0.013  +0.019
D_perlevel   +0.015  +0.052     +0.043   +0.098    +0.044      +0.171     +0.013  +0.009
```

The next cell recomputes this live (~30 s on CPU). Values move by a few points
with the backend RNG, but the **winner of each row is stable**.
"""),
        code(stem, 4, r"""
M = g.confusion_matrix()                 # {DGP: {kernel: out-of-sample CKA}}, ~30 s
g.plot_cka_heatmap(M); plt.show()
"""),
        md(stem, 5, r"""
## Reading the matrix

* **`D_disp` (level 1).** Everything ties, and the order-blind `pooled-RBF` is
  *best* — a straight displacement needs no order, and the signature family
  manufactures no spurious order gain.
* **`D_area` (order).** The order-aware kernels jump to $0.82$–$0.92$ while every
  order-**blind** representation (`sig-L1`, `pooled-RBF`, `kme-RBF`) sits at
  chance. This is the signature kernel's reason to exist.
* **`D_peak`.** Signal at level 2 with noise at levels 1 **and** 3: the optimal
  $\varphi$ is a **non-monotone peak**. Only `sig-Wphi`'s free weights can
  represent it — the dilation cone and the uniform/per-level kernels cannot.
* **`D_scale` (tail).** The class lives in the level-$\ge$3 tail; free weights
  reach it, uniform `sig-PDE` partly, and the $\lambda_{\max}=0.5$ dilation cone
  (`sig-PDEphi`, forced *decaying*) cannot — the clamp chain `Wphi > PDE > PDEphi`.
* **`D_lowsig`.** Signal at level 2, noise in the tail: `sig-PDEphi`'s soft-decay
  $\varphi$ down-weights the tail and **beats uniform `sig-PDE`**.
* **`D_perlevel`.** A class-independent level-1 drift dominates the global norm;
  only `sig-EXACT`'s per-path, **per-level** whitening recovers the level-2 sign.

Two of these winners — `sig-Wphi` on `D_peak`/`D_scale` and `sig-PDEphi` on
`D_lowsig` — are kernels that **learned** their $\varphi$ from the data. That is
the subject of **`X_learned_kernel.ipynb`**: can the kernel *recover the order* a
process plants?
"""),
    ]
    write(stem, cells)


def build_learned_kernel_nb():
    """X_learned_kernel.ipynb — the flagship: the learnable GSK, order recovery,
    and the free-weights-vs-dilation-cone structural contrast."""
    stem = "X_learned_kernel"
    cells = [
        md(stem, 0, r"""
# The learnable General Signature Kernel — recovering the order

Two of the six GSK columns **learn** their order-weighting $\varphi(k)$ from
labelled data instead of fixing it:

* **`sig-Wphi`** (truncated, *free weights*): $\varphi(0)=1$ pinned,
  $\varphi(1{:}N)=\mathrm{softplus}(\theta)\ge0$ — an **arbitrary** nonnegative,
  possibly **non-monotone** level profile.
* **`sig-PDEphi`** (untruncated, *dilation mixture*):
  $\varphi(k)=\sum_{i=1}^m w_i\lambda_i^{k}$ with $w=\mathrm{softmax}(\cdot)\ge0$,
  $\lambda_i\in(0,\lambda_{\max})$. By the **dilation identity** (Cass–Lyons–Xu),
  a geometric mixture of dilated signature kernels *is* an order-weighting — but
  a **completely monotone** (decaying) one.

Both are **two-phase**: `fit_phi(Xtr, ytr)` learns $\varphi$ by maximizing
centered kernel–target alignment (CKA) on the **training** set; the object is
then frozen and behaves as an ordinary precomputed PSD kernel ($\varphi\ge0\Rightarrow$
conic sum of PSD level kernels). Design: `docs/SIGNATURE_KERNELS.md` §0.3–0.7.

**The question this notebook answers:** if a data-generating process plants its
class signal at a *known* signature order $k^\star$, does the learned
$\varphi(k)$ **peak at $k^\star$**? And what can each of the two learners
represent?
"""),
        md(stem, 1, "## Environment"),
        code(stem, 1, SETUP_GSK),
        md(stem, 2, r"""
## Planted-order data

`g.make_planted(level, n, seed)` confines the class signal to one signature
**level** and makes everything else class-independent:

* **level 1** — a class-dependent straight **drift** (net displacement, zero area);
* **level 2** — a class-**oriented** closed loop (signed area, zero displacement);
* **level 3** — a class-**time-reversed** figure-eight (zero area; energy in the
  odd $\ge$3 tail) over a class-independent micro-loop.

These are the level-localized primitives of the inductive-bias suite, retuned so
the matched order is unambiguous at truncation $N=3$.
"""),
        code(stem, 2, r"""
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for lvl, a in zip((1, 2, 3), ax):
    X, y = g.make_planted(lvl, 60, seed=lvl)
    for cls, c in zip((0, 1), ("tab:blue", "tab:red")):
        for i in np.where(y == cls)[0][:4]:
            a.plot(X[i, :, 0], X[i, :, 1], c=c, alpha=0.6)
    a.set_title(f"planted level {lvl}"); a.set_aspect("equal")
fig.suptitle("Planted-order DGPs  (blue = class 0, red = class 1)")
plt.tight_layout(); plt.show()
"""),
        md(stem, 3, r"""
## Order recovery — does $\varphi$ peak at the planted level?

For each planted level $k^\star\in\{1,2,3\}$ we fit **`sig-Wphi`** (free weights,
$N=3$) on the train split and read off the learned $\varphi(1{:}N)$. The dashed
line marks $k^\star$; recovery means the **argmax lands on it**.
"""),
        code(stem, 3, r"""
import torch
profiles, recovered = {}, {}
for lvl in (1, 2, 3):
    Xtr, ytr, _, _ = g.split(lambda n, s: g.make_planted(lvl, n, s), seed=lvl)
    bw = g.median_bw(Xtr)
    torch.manual_seed(0)
    k = g.gsk("free", g.N_LEVELS, bw=bw).fit_phi(Xtr, ytr, steps=g.FIT_STEPS)
    phi = np.asarray(k.phi_profile())
    profiles[f"planted $k^\\star$={lvl}"] = phi
    recovered[lvl] = 1 + int(np.argmax(phi[1:]))          # argmax over levels 1..N
    print(f"planted level {lvl}:  phi = {np.round(phi, 3)}   ->  peak at level {recovered[lvl]}"
          f"   {'(recovered)' if recovered[lvl] == lvl else '(MISS)'}")

g.plot_phi(profiles, planted={f"k{l}": l for l in (1, 2, 3)},
           title="sig-Wphi learns phi peaked at the planted order")
plt.show()
"""),
        md(stem, 4, r"""
## What each learner can represent — free weights vs the dilation cone

`D_peak` plants the signal at **level 2** with **noise at levels 1 and 3**, so
the ideal $\varphi$ is a **non-monotone peak** (suppress 1 and 3, keep 2). We fit
both learnable kernels and compare their learned $\varphi$:

* **`sig-Wphi`** (free weights) *can* peak — $\varphi(2)>\varphi(1)$.
* **`sig-PDEphi`** is a sum of $\lambda_i^k$ with $\lambda_i,w_i\ge0$, i.e. a
  **completely monotone** (Stieltjes) sequence — it is **structurally forbidden**
  from peaking and must be non-increasing.

This is the theory linchpin behind `D_peak`'s CKA gap, checked here directly on
the **coefficients** (no data-calibration dependence).
"""),
        code(stem, 4, r"""
import torch
Xtr, ytr, _, _ = g.split(g.make_peak, seed=list(g.DATASETS).index("D_peak"))
bw = g.median_bw(Xtr)
torch.manual_seed(0)
wphi = g.gsk("free", g.N_LEVELS, bw=bw).fit_phi(Xtr, ytr, steps=g.FIT_STEPS)
torch.manual_seed(0)
pdephi = g.gsk("dilation", None, bw=bw).fit_phi(Xtr, ytr, steps=g.FIT_STEPS)

w = np.asarray(wphi.phi_profile(g.N_LEVELS))
p = np.asarray(pdephi.phi_profile(g.N_LEVELS))
print(f"sig-Wphi    phi = {np.round(w, 3)}   peak above level 1? {bool(w[2] > w[1])}")
print(f"sig-PDEphi  phi = {np.round(p, 3)}   monotone non-increasing? {bool(np.all(np.diff(p) <= 1e-6))}")

g.plot_phi({"sig-Wphi (free, can peak)": w, "sig-PDEphi (dilation cone, monotone)": p},
           planted=2, title="D_peak — free weights peak at level 2; the dilation cone cannot")
plt.show()
"""),
        md(stem, 5, r"""
## The payoff — learning $\varphi$ buys alignment on the matched data

Finally, the out-of-sample CKA of the two learners against the fixed-$\varphi$
baselines and an order-blind reference, on the DGPs each learner is built for:

* **`D_peak`** (non-monotone) — `sig-Wphi` should win outright.
* **`D_lowsig`** (tail noise) — `sig-PDEphi`'s soft-decay should beat uniform `sig-PDE`.
* **`D_scale`** (tail signal) — the clamp chain `sig-Wphi > sig-PDE > sig-PDEphi`.
"""),
        code(stem, 5, r"""
cols = ["sig-L1", "sig-TRUNC", "sig-PDE", "sig-Wphi", "sig-PDEphi"]
reg = {k: v for k, v in g.kernel_registry().items() if k in cols}
M = g.confusion_matrix(datasets=["D_peak", "D_lowsig", "D_scale"], kernels=reg)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
xpos = np.arange(len(M)); w = 0.16
for j, name in enumerate(cols):
    ax.bar(xpos + (j - 2) * w, [M[d][name] for d in M], w, label=name)
ax.axhline(g.CHANCE, ls="--", color="grey", alpha=0.6, label=f"chance ({g.CHANCE})")
ax.set_xticks(xpos); ax.set_xticklabels(list(M)); ax.set_ylabel("out-of-sample CKA")
ax.set_title("Learned phi (sig-Wphi / sig-PDEphi) vs fixed baselines")
ax.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.show()
"""),
        md(stem, 6, r"""
## Takeaways

* **The learned $\varphi$ recovers the planted order.** On a signal confined to
  level $k^\star$, `sig-Wphi`'s free weights peak at $k^\star$ for $k^\star=1,2,3$.
* **Representation is a design choice.** Free weights (`sig-Wphi`) span arbitrary
  nonnegative — including non-monotone — profiles; the dilation mixture
  (`sig-PDEphi`) is **completely monotone** by construction, trading expressivity
  for an *untruncated* kernel and a soft, decaying tail.
* **Each wins where its bias matches the data** (`D_peak` → `sig-Wphi`,
  `D_lowsig` → `sig-PDEphi`), and the $\lambda_{\max}$ clamp caps the dilation
  cone away from tail-*amplifying* signals (`D_scale`).
* **PSD throughout** ($\varphi\ge0$), so a fitted kernel drops straight into any
  precomputed-kernel SVM/GP. See `07_general_signature_kernel.ipynb` for the full
  family map and `docs/SIGNATURE_KERNELS.md` for the derivations.
"""),
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

# General Signature Kernel family (inductive-bias / learnable demos).
build_gsk_overview_nb()
build_learned_kernel_nb()

print("done.")
