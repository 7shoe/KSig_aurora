# `./notebooks` — visual demos & scaling playground

A **playful, visual companion to `tests/` and `monitoring/`.** Where the test
suite proves correctness with `assert`s and the monitoring harness records
numbers, these six notebooks let you *see* — after the port is adapted to
Aurora — both:

* **(a) correctness** — each notebook computes a kernel on simple deterministic
  data and shows, in the markdown cell directly above the code, the **reference
  output measured on NVIDIA CUDA**. Run the cell on the target machine and eyeball
  that the live output matches.
* **(b) scaling / throughput** — a tunable sweep at the bottom plots wall time vs
  problem size, overlaying the CUDA reference against what *this* machine does.

One notebook per significant feature:

| Notebook | Feature | Class |
|---|---|---|
| `01_signature_kernel.ipynb` | Signature kernel (full-rank DP, "plain vanilla") | `SignatureKernel` |
| `02_signature_pde_kernel.ipynb` | Signature-PDE kernel (Goursat PDE) | `SignaturePDEKernel` |
| `03_global_alignment_kernel.ipynb` | Global Alignment Kernel (log-space) | `GlobalAlignmentKernel` |
| `04_rfsf_features.ipynb` | Random Fourier Signature Features (RFSF-TRP) | `SignatureFeatures` + `RandomFourierFeatures` |
| `05_low_rank_signature.ipynb` | Low-rank signature features | `SignatureFeatures` (projection only) |
| `06_random_warping_series.ipynb` | Random Warping Series (DTW features) | `RandomWarpingSeries` |

## The colour contract

Every scaling plot draws up to three curves:

* 🟩 **green — CUDA reference.** Baked into `cuda_reference.json`, *measured on an
  NVIDIA H100 NVL* (CuPy). It does **not** require CUDA to display — it's just
  data — so it shows on Aurora as the line to beat / match.
* 🟦 **blue — this machine.** Computed live by the notebook on whatever backend is
  loaded (torch-native on Aurora XPU, or CUDA / CPU / MPS in dev).
* 🟧 **orange — SYCL fast-path.** Drawn **only when** a `ksig._sycl` build is
  present (`nb.sycl_available()` is `True`). The code exposes a `--sycl`-style
  toggle (`nb.enable_sycl`); the notebook runs the sweep a second time through it
  and adds the orange curve. **If SYCL is never adopted** (per the acceptance
  gate in [`tests/TEST_PLAN.md`](../tests/TEST_PLAN.md) §12 — accepted only if
  both correct *and* measurably faster), the extension is simply absent and **no
  orange curve ever appears.** Nothing to configure.

## Running on Aurora (after the port is adapted)

No CuPy/Numba needed — the green reference is frozen data; you only need the
torch-native `ksig`, plus `matplotlib` and a Jupyter kernel.

```bash
# interactive
jupyter lab          # then open 01_signature_kernel.ipynb and Run All

# or headless, execute every notebook in place:
jupyter nbconvert --to notebook --execute --inplace notebooks/0*_*.ipynb
```

Each notebook is self-pathing: it adds the repo root to `sys.path`, so the
in-repo `ksig` imports without a `pip install` (the same reason the tests use
`pythonpath = .`). The first cell prints a one-line banner — backend, device,
and whether SYCL is available.

## Tuning

The scaling cell starts with a small knobs block that **defaults to the CUDA
reference grid** (so blue lines up with green). Widen it to push the frontier:

* `01` exposes `ORDER` (the embedding-order cap) and `N_LEVELS`;
* `02` exposes `DIFFERENCE`; `03` exposes `BANDWIDTH`;
* `04`–`06` expose `N_COMPONENTS` (more components → tighter approximation, the
  convergence is printed in the correctness cell).

`REPS` controls timing robustness (median over reps after a warm-up).

## Regenerating (maintainers, on a CUDA box)

The notebooks are built from one place so they stay consistent:

```bash
python notebooks/_gen_reference.py      # re-measure the CUDA reference  -> cuda_reference.json
python notebooks/_build_notebooks.py    # rebuild 0X_*.ipynb from the reference + templates
```

`_nbtools.py` holds the shared helpers (env/SYCL detection, deterministic data,
timing, the green/blue/orange plot). `_gen_reference.py` must run where CuPy +
CUDA work; `_build_notebooks.py` runs anywhere.

> **One thing for the port author:** wire the real SYCL switch into
> `nb.enable_sycl()` / `nb.sycl_available()` in `_nbtools.py` (the same toggle
> the `--sycl` flag flips). Until then they return `False` and the notebooks
> quietly stay green+blue.
