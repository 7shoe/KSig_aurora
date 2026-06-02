# `ksig._sycl` — native SYCL fast-path for the DP kernels (Aurora XPU)

Optional, profile-gated acceleration for the three dynamic-programming kernels
(SigPDE, GAK, RWS/DTW). The portable **torch wavefront** in
`ksig/algorithms.py` is always the canonical path and the numerical oracle;
this extension is dispatched only when **(a)** the input is an XPU tensor and
**(b)** the extension built (`ksig._sycl.loader.available()`), via
`ksig.algorithms._try_sycl`. Anything else falls through to the torch wavefront.

## Why it can win (TORCH_PORT §10.1)
1. **Launch overhead** — the torch wavefront does `l_X+l_Y-1` eager kernel
   launches; the SYCL kernel does the whole DP in **one launch**, holding the
   trailing three antidiagonals in device local memory (`local_accessor`) with
   `group_barrier` between sweeps (one work-group per sequence pair).
2. **Memory** — a future *fused* variant (`sig_pde_rbf`, §10.5) computes the
   static-kernel entry inside the DP, never materializing the
   `[n_X,n_Y,l_X,l_Y]` tensor → O(N²L²) → O(N²)+O(N·L·d). The kernels here are
   the **non-fused** 1:1 ports first (lowest risk).

## What's here
- `pde_kernels.sycl` — the three non-fused kernels (three rolling
  `local_accessor`s with role-rotation; SigPDE / GAK-log / RWS-DTW).
- `bindings.cpp` — pybind11 module + **all torch/ATen glue** (tensor reshaping,
  driver transforms, `AT_DISPATCH`); exposes `sig_pde`, `gak_log`, `rws_dtw`.
  The `.sycl` TU is kept ATen-free and takes raw device pointers.
- `loader.py` — JIT build via `torch.utils.cpp_extension.load` (`.sycl` files in
  `sources`; also applies the build workarounds noted under Status).

## Building & running — needs an Aurora **compute node** (not a login/UAN node)

A login/UAN node has **no XPU device** (`torch.xpu.is_available()` is `False`),
so the extension neither builds nor runs there. Grab an interactive compute node
with a GPU, e.g.:

```bash
# Interactive single-node job on Aurora (adjust account/queue/walltime):
qsub -I -l select=1 -l walltime=01:00:00 -q debug -A <ALLOCATION>

# On the compute node, load the toolchain and confirm the device + fp64:
module load frameworks                 # native-XPU torch
bash scripts/probe_hardware.sh         # reports XPU count, sycl-ls, fp64
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"

# Build + validate against the torch wavefront (the oracle):
KSIG_SYCL_VERBOSE=1 python -c "from ksig._sycl import loader; print('built:', loader.available())"
pytest -m "xpu and sycl" tests/test_sycl.py -q
```

Do **not** pass `-fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80`:
those are for Polaris (NVIDIA), a different ALCF machine. On Aurora the default
SYCL target is Intel/Level-Zero; `loader.py` uses plain `-fsycl`.

## Acceptance gate (TORCH_PORT §12)
A SYCL kernel becomes the dispatched path only if **both** hold:
1. **Correct** — passes `tests/test_sycl.py` (agrees with the torch wavefront
   within the f64 band, incl. `l_Y=1`/unequal-length and stream-ordering).
2. **Beneficial** — `monitoring/` shows a measurable speedup *or* memory
   reduction over the torch-XPU baseline on ≥ the `medium` tier, reproducibly.

If correct but not beneficial → keep torch canonical (simpler, already correct).
If beneficial but not correct → reject (a fast wrong answer is worthless).

## Status
**Builds + validates on Aurora** (2026-06-02; `frameworks/2025.3.1`, torch
`2.8.0a0`, icpx 2025.3.2): `pytest -m "xpu and sycl"` → 13 passed, and ~80–130×
faster than the wavefront at the sizes tried. Build workarounds applied in
`loader.py`/`bindings.cpp`/`pde_kernels.sycl` for this stack (see `docs/SYCL_HANDOFF.md`
§4): `load()` takes `.sycl` in `sources` (no `sycl_sources` kwarg); kernels are
explicitly named; the `.sycl` TU is ATen-free; `loader._fix_sycl_host_flags`
strips the mis-tokenizing `-DPYBIND11_*` defines from the SYCL host pass and
forces `-fPIC`. Still TODO: benchmark at scale via `monitoring/`, then the fused
`sig_pde_rbf` (§10.5) memory win.
