# Stage 1 — freezing the golden ground truth

`make_golden.py` runs the **legacy CuPy/Numba `ksig`** over the canonical fixture
matrix (`tests/fixtures/matrix.py`) and writes frozen oracle outputs to
`tests/golden/`. It must run on an **NVIDIA box with CuPy + Numba** (the oracle
stack). The committed artifacts let the torch port be validated on Aurora/XPU
**without CuPy present**.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tests.freeze.make_golden --out tests/golden
```

## What it produces (per case)
- `tests/golden/<case_id>.npz` — the golden arrays (float64). Sequence kernels
  store `gram`, `diag`, and (when applicable) `xy`; static kernels store `value`.
- `tests/golden/<case_id>.json` — provenance sidecar: entry point, kwargs,
  input specs + shapes, tolerance class, **golden source**, **legacy status**,
  **cross-check vs the independent NumPy oracle** (max abs/rel), an
  `ill_conditioned` flag + `recommended_rtol`, value stats, and the full
  versions block (GPU name, cupy/numba/cuda).
- `tests/golden/INDEX.json` — manifest over all artifacts.

## What makes it trustworthy
Every value is **cross-checked against an independent brute-force NumPy oracle**
(`tests/oracles/`). The recorded `cross_check_vs_oracle_max_abs` proves two
independent computations agree (typically ≤ 1e-12). This is *not* "the old code
agrees with itself".

## Known legacy bugs handled automatically
The freeze run on this repo found and routed around real legacy defects:
- **`Matern12` / `Matern32`** — broken by the spurious `self` in
  `ksig.utils.euclid_dist`; the legacy call raises. Golden is sourced from the
  **oracle** (`golden_source = "numpy_oracle"`, `legacy_status = "broken"`).
- **SigPDE on `L=1` with `difference=True`** — the Numba kernel launches a
  0-thread grid (`cuLaunchKernel` fails). Auto-detected via try/except and
  sourced from the oracle (correct value `K = 1`).

These cases are `xfail` against the legacy backend in the Stage-2 suite and
**required to pass on the torch port** — so the suite actively drives the fixes.

## Idempotence
Same fixtures → byte-identical `.json` except `created_utc`. Re-running
regenerates the full set deterministically (all randomness via
`numpy.random.default_rng(seed)`).
