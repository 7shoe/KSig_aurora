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
| `07_general_signature_kernel.ipynb` | General Signature Kernel — the six-column family + "data each excels on" CKA matrix | `GeneralSignatureKernel` |
| `X_learned_kernel.ipynb` | **Learnable** General Signature Kernel — order recovery & free-weights-vs-dilation-cone | `GeneralSignatureKernel` (`sig-Wphi`, `sig-PDEphi`) |

## The two flavours of notebook

`01`–`06` are **throughput / correctness** demos for the ported kernels: they
compute a kernel on deterministic data, eyeball it against the frozen CUDA
reference, and sweep wall time (the green/blue/orange contract below).

`07` and `X_learned_kernel` are **inductive-bias** demos for the
`GeneralSignatureKernel` family (`ksig/generalized.py`, `docs/SIGNATURE_KERNELS.md`).
They answer a *statistical* question rather than a timing one — **which kernel
reads which structural locus of a path** — so they have **no green curve** (there
is nothing to time against) and run on CPU (≤120 short paths). They share
`_gsk_demo.py` (the level-localized data-generating processes, the CKA confusion
matrix, and the φ/heatmap plots), the way `01`–`06` share `_nbtools.py`.

* **`07`** maps the family: the six columns
  (`sig-L1`, `sig-TRUNC`, `sig-PDE`, `sig-Wphi`, `sig-PDEphi`, `sig-EXACT`) as
  configurations of one object, and the **DGP × kernel CKA heatmap** showing each
  kernel winning on the data its order-weighting φ is built for (e.g. order-blind
  references at chance on the signed-area `D_area`; only free weights peaking on
  `D_peak`; the `Wphi > PDE > PDEphi` clamp chain on the tail signal `D_scale`).
* **`X_learned_kernel`** is the flagship for the **learnable** kernel: it plants a
  class signal at a known signature order $k^\star\in\{1,2,3\}$ and shows the
  learned φ(k) **peaks at $k^\star$** (order recovery), then contrasts what each
  learner can represent — `sig-Wphi`'s free, possibly non-monotone weights vs
  `sig-PDEphi`'s completely-monotone dilation cone — on the level-2 peak `D_peak`.

The CKA matrix these reproduce is pinned by
`tests/test_signature_kernel_inductive_bias.py`.

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
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

The General-Signature-Kernel notebooks (`07`, `X_learned_kernel`) need only
`ksig` + `matplotlib` — no CUDA reference — and run on CPU in ~1 min each.

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
python notebooks/_build_notebooks.py    # rebuild 01-07 + X_learned_kernel from templates
```

`_build_notebooks.py` writes **un-executed** notebooks; execute them with the
`nbconvert` line above to embed outputs (the shipped notebooks are executed).
`_nbtools.py` holds the shared helpers for `01`–`06` (env/SYCL detection,
deterministic data, timing, the green/blue/orange plot); `_gsk_demo.py` holds the
shared helpers for `07`/`X_learned_kernel` (the level-localized DGPs, the CKA
confusion matrix, the φ/heatmap plots). `_gen_reference.py` must run where CuPy +
CUDA work; `_build_notebooks.py` runs anywhere.

> **One thing for the port author:** wire the real SYCL switch into
> `nb.enable_sycl()` / `nb.sycl_available()` in `_nbtools.py` (the same toggle
> the `--sycl` flag flips). Until then they return `False` and the notebooks
> quietly stay green+blue.
