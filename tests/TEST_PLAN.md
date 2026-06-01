# KSig-Aurora — Test & Monitoring Plan

> **Audience:** the coding agent porting KSig from **CuPy + Numba-CUDA** to **PyTorch-native (+ optional SYCL)** per [`docs/TORCH_PORT.md`](../docs/TORCH_PORT.md).
> **Mandate:** define a `tests/` suite (and a sibling `monitoring/` harness) that catches *correctness*, *runtime*, and *hardware-utilization* regressions **early and at the level of individual input shapes**, across a CUDA→XPU/SYCL migration in which the **reference (oracle) values are frozen on NVIDIA/CuPy/legacy-NumPy** but the **port runs on Aurora Intel XPUs (oneAPI)** that may round floats differently.
> **Status:** this is the *plan*. No test code is written yet. §13 gives the file-by-file build order.

---

## 1. Why two stages

The single hardest fact about this migration: **the golden values and the implementation under test never run on the same hardware/stack.**

- **Oracle stack:** NVIDIA GPU, CuPy ≥12.2, Numba-CUDA, NumPy 1.24.4, float64. This is the *current* `ksig` (the code in this repo before the port).
- **Port stack:** Aurora Intel Data Center GPU Max 1550 (Ponte Vecchio), PyTorch XPU build (Intel oneAPI), optional `ksig._sycl` `icpx -fsycl` extension. Possibly also CPU and Apple MPS for dev.

We therefore **decouple "produce the truth" from "check the port"**:

- **Stage 1 — Freeze (run once, on a known-good CUDA box).** Execute the legacy CuPy/Numba implementation over a *deterministic fixture matrix* and persist **frozen oracle outputs** to `tests/golden/*.npz` plus a per-artifact `*.json` sidecar (full provenance). Committed (or DVC/LFS-tracked) so the port can be validated **without CuPy present.**
- **Stage 2 — Compare (run continuously, on Aurora/XPU + CPU + MPS).** Run the torch-native (and, when built, SYCL) implementations over the *same fixtures*, load the golden `.npz`, and assert agreement with `numpy.testing.assert_allclose` / `torch.testing.assert_close` under a **tolerance class** chosen per algorithm family.

**Oracle-of-last-resort:** for the three DP kernels (SigPDE, GAK, RWS/DTW), also implement **brute-force NumPy DP oracles** straight from the recurrences in `docs/TORCH_PORT.md §4`. These are hardware-independent and let us test the DP kernels even where no `.npz` exists or where we suspect the CuPy oracle itself drifted. **Write these oracles first** (before touching `algorithms.py`).

---

## 2. Directory layout

```
tests/
  TEST_PLAN.md               ← this document
  conftest.py                ← device/dtype fixtures, marker registration, tol policy, golden loader
  tolerances.py              ← TolClass enum + (rtol, atol) tables keyed by (family, dtype, device)
  fixtures/
    __init__.py
    generators.py            ← deterministic input builders (see §6); NO randomness outside seeded RNG
    matrix.py                ← the canonical fixture grid (param ids) shared by freeze + compare
    biological.py            ← one-hot DNA/RNA/protein sequence builders + cumsum-over-seq transform
  oracles/
    __init__.py
    dp_numpy.py              ← brute-force NumPy DP for SigPDE / GAK / RWS-DTW (hardware-independent)
    signature_numpy.py       ← brute-force truncated signature kernel / features (small n_levels)
  golden/                    ← Stage-1 frozen artifacts (committed via LFS/DVC)
    <fn>__<tag>__<hash>.npz
    <fn>__<tag>__<hash>.json
    INDEX.json               ← manifest: every artifact + its fixture id + provenance digest
  freeze/
    make_golden.py           ← Stage-1 entry point (imports legacy CuPy ksig, writes golden/)
    README.md                ← how/where to run on the CUDA box
  unit/
    test_utils_tensor_algebra.py     ← multi_cumsum, matrix_mult, squared_euclid_dist, outer_prod, …
    test_static_kernels.py
    test_static_features.py          ← RFF, RFF1D, Nystroem
    test_projections.py
    test_preprocessing.py            ← SequenceTabulator, SequenceAugmentor
  golden/                            (data only — see above)
  test_algorithms_dp.py              ← SigPDE / GAK / RWS vs numpy-DP oracle AND vs golden
  test_signature_full.py            ← SignatureKernel (order 1 & higher, normalized/diag)
  test_signature_lowrank.py         ← low-rank signature features
  test_signature_features.py        ← SignatureFeatures (RFSF-TRP / RFSF-DP)
  test_kernels_public.py            ← public class contracts (call/fit/transform, symmetry, diag)
  test_models.py                    ← PrecomputedKernelSVC / PrecomputedFeatureLinSVC (sklearn fallback)
  test_device_equivalence.py        ← cpu vs xpu/cuda/mps cross-device agreement
  test_sycl.py                      ← SYCL path vs torch wavefront (marker: sycl)

monitoring/                          ← performance, NOT correctness (kept out of the pytest gate)
  run_benchmarks.py                  ← scalable grid over (n,L,d,n_levels,order,n_components,batch,dtype)
  grids.py                           ← benchmark grid definitions (small/medium/stress tiers)
  record.py                          ← JSONL/CSV writers + schema
  probes.py                          ← timing, peak-mem, device-util, CPU-fallback detector
  results/                           ← *.jsonl / *.csv artifacts (git-ignored, or LFS)
  README.md
```

> The `tests/golden/` directory appears once (data). Test modules that compare against it import the `golden(fn, tag)` loader from `conftest.py`.

---

## 3. Stage 1 — freezing the golden oracle (`tests/freeze/make_golden.py`)

Runs **only** on the NVIDIA/CuPy box. For every `(entry_point, fixture)` pair in the canonical matrix (§6):

1. Build the input with the **seeded** generator (the *same* generator the port will call in Stage 2 — fixtures are shared code, not duplicated).
2. Call the **legacy** ksig entry point on GPU; `cupy.asnumpy(...)` the result to host, **cast to float64**.
3. Persist:
   - **`<fn>__<tag>__<hash>.npz`** — `np.savez_compressed(out=..., [aux arrays])`. `<hash>` is a digest of the fixture spec (so a fixture change can't silently reuse a stale oracle).
   - **`<fn>__<tag>__<hash>.json`** — metadata sidecar:
     ```json
     {
       "function": "ksig.kernels.SignatureKernel.__call__",
       "semantic_tags": ["full_rank", "order1", "normalized", "gram"],
       "fixture_id": "gaussian__n8_L20_d5",
       "seed": 0,
       "dtype_in": "float64",
       "dtype_out": "float64",
       "input_shapes": {"X": [8, 20, 5]},
       "kwargs": {"n_levels": 4, "order": 1, "normalize": true},
       "device": "cuda:0",
       "gpu_name": "NVIDIA A100-SXM4-40GB",
       "versions": {"python": "...", "numpy": "1.24.4", "cupy": "12.2.0",
                    "numba": "...", "cuda": "12.3", "ksig": "<git-sha>"},
       "tolerance_class": "DP_CUMSUM",
       "value_stats": {"min": ..., "max": ..., "mean": ..., "n_nonfinite": 0},
       "created_utc": "2026-..."
     }
     ```
4. Append the artifact to `golden/INDEX.json` (the manifest the compare stage walks).

**Determinism rules for Stage 1** (so re-freezing is reproducible):
- All randomness via `numpy.random.default_rng(seed)` *fed into* both stacks (see RNG caveat §5). **Never** persist raw CuPy-`RandomState` draws as oracle — those streams are not reproducible on torch.
- Save **float64** regardless of the compute dtype, so float32 ports compare against a higher-precision truth.
- `make_golden.py` must be **idempotent**: same fixtures → byte-identical `.json` provenance except `created_utc`.

---

## 4. Stage 2 — comparing the port (the pytest suite)

On Aurora/XPU (and CPU/MPS in dev): the test loads the golden `.npz`, runs the port over the identical fixture, and compares.

```python
exp = golden("SignatureKernel.__call__", "gaussian__n8_L20_d5__order1_norm")   # -> np.ndarray f64
got = sig_kernel(X).cpu().numpy()                                              # port output
assert_allclose_tol(got, exp, family="DP_CUMSUM", dtype=X.dtype, device=dev)   # tol from §5
```

- Comparison uses `numpy.testing.assert_allclose` for ndarray vs ndarray, and `torch.testing.assert_close` when both sides are tensors (keeps device/dtype in the message).
- **Missing golden ⇒ `pytest.skip`** with a clear reason (so the port is still developable on a box that never saw the `.npz`), *except* for DP kernels which fall back to the **numpy-DP oracle** instead of skipping.
- Every failure message must name the **fixture id** and the **first-divergent index** (see §9 "pinpointing") so the agent learns *which input shape* broke, not merely *that* something broke.

---

## 5. Tolerance policy (`tests/tolerances.py`)

Tolerances are chosen by **algorithm family**, not per-test, so the policy is auditable in one table. Family is recorded in each golden sidecar (`tolerance_class`).

| Family | Routines | float64 (CUDA/XPU/CPU) | float32 / MPS |
|---|---|---|---|
| `EXACT_ALGEBRA` | `matrix_mult`, `outer_prod`, `squared_norm`, `matrix_diag`, linear/poly static kernels | `rtol=1e-10, atol=1e-12` | `rtol=1e-5, atol=1e-6` |
| `DP_CUMSUM` | `multi_cumsum`, full/low-rank signature kernels & features, SigPDE, GAK | `rtol=1e-8, atol=1e-10` | `rtol=1e-3, atol=1e-4` |
| `DTW_EXACT` | RWS / DTW (min,+ over integers/exact) | `rtol=1e-12, atol=0` | `rtol=1e-5, atol=1e-5` |
| `RANDOM_FEATURE` | RFF, RFF1D, Nystroem, all `projections.*`, RWS feature map | **Monte-Carlo band**, not elementwise (§5.1) | same, looser N |
| `E2E_SCORE` | `PrecomputedKernelSVC` / `PrecomputedFeatureLinSVC` accuracy | exact-match labels / `|Δacc| ≤ 0.0` on fixed split | `|Δacc| ≤ 0.01` |
| `SYCL` | `_sycl` kernels vs torch wavefront | agreement within the **f64 band of its family** | n/a (XPU only) |

**Rationale for the XPU loosening:** oneAPI fp64 is IEEE-754 and should match CUDA fp64 to the `DP_CUMSUM` band; fp32 fused-multiply-add ordering and transcendental (`exp`, `sqrt`) implementations differ between vendors, so **float32 on XPU/MPS gets the loose column by design**, and strict-f64 assertions are `skipif`'d on MPS (no native fp64). Document any test that needed a looser-than-table tolerance with an inline comment + the observed max-abs-diff.

### 5.1 RNG caveat (critical — do not compare raw random matrices)

CuPy `RandomState` and `torch.Generator` produce **different streams for the same seed**. For every randomized routine (RFF, Nystroem landmark draw, all projections, RWS warping series), **never** assert that two random weight matrices match. Instead:

- **(a) Inject-the-weights:** generate the random matrix with `numpy.random.default_rng(seed)` and inject it into *both* the legacy and port code paths via a test hook, then compare the **deterministic math** that consumes it under the family tolerance. *(Preferred — turns a random test into an exact one.)*
- **(b) Compare-the-estimand:** assert the **kernel/Gram matrix the features approximate** matches the *exact* kernel within a Monte-Carlo band (`‖K_approx − K_exact‖ / ‖K_exact‖ ≤ ε(n_components)`), with `ε` shrinking as `n_components` grows. Use this where injection is impractical.

---

## 6. Fixtures — the canonical input matrix (`tests/fixtures/`)

One shared generator module builds every input; freeze and compare import the *same* functions so the bytes line up. Each fixture has a stable `id` (used in golden filenames and pytest param ids).

**Input archetypes** (the `what-shape-broke` axis):

| Archetype | Builder | Purpose / what it stresses |
|---|---|---|
| Dense Gaussian | `gaussian(n, L, d, seed)` | nominal correctness |
| Degenerate constant | `constant(n, L, d, c)` | zero-variance → normalization `/0`, `robust_sqrt` clamp, GAK degeneracy |
| Near-zero | `near_zero(scale=1e-12)` | `_EPS` clamps, `robust_nonzero`, catastrophic cancellation in `squared_euclid_dist` |
| Large-magnitude | `large(scale=1e6)` | overflow in `exp` (RBF/GAK log-space), float32 range |
| NaN-bearing ragged | `ragged_with_nan(...)` | **only** through preprocessing paths that support NaN filtering (`SequenceTabulator`) |
| Variable-length list | `ragged_lengths([L1,L2,...])` | tabulation/interpolation; `l_Y=1` edge for DTW |
| Flattened 2-D + `n_features` | `flat_2d(n, L*d, n_features=d)` | the explicit-`n_features` reshape path |
| Biological one-hot | `onehot_dna / onehot_protein` | one-hot {A,C,G,T} or 20-aa, then **cumsum over the sequence axis** (the bio embedding) |
| Single-sequence | `gaussian(1, L, d)` | broadcast / batch-of-1 corner |
| Single-timestep | `gaussian(n, 1, d)` | `L=1` recurrence base case |

**Dimension sweep** (the `which-dimension-broke` axis) — small by default so the suite is fast; `slow`/`stress` markers unlock the big end:

```
n_samples    ∈ {1, 2, 8, 33}          # incl. odd to catch even-only assumptions
seq_len  L   ∈ {1, 2, 20, 101}        # incl. L=1 base case, odd length
n_features d ∈ {1, 3, 5, 16}          # incl. d=1
n_levels     ∈ {1, 2, 4, 5}           # signature truncation
order        ∈ {1, 2, 3}              # 1 = first-order fast path, >1 = higher-order
n_components ∈ {16, 100}              # RFF / projections
batch_size   ∈ {None, 4}             # tiling vs whole-matrix
dtype        ∈ {float64, float32}     # (+ float16 only under `stress`)
```

The cross-product is pruned per entry point (e.g. `order` only applies to signature kernels) by `fixtures/matrix.py`, which yields `(entry_point, kwargs, fixture_id)` tuples consumed by both `make_golden.py` and the parametrized tests.

---

## 7. Per-module test specifications

Each subsection lists **functions/classes under test → what each test asserts**. The "Common assertions" of §9 apply on top of the specifics here.

### 7.1 Low-level tensor algebra — `ksig/utils.py` (`tests/unit/test_utils_tensor_algebra.py`)

These are the highest-leverage tests: every kernel is built from them, and they are where CuPy→torch axis/pad/dtype mismatches surface first.

- **`multi_cumsum(M, exclusive, axis)`** — the single most port-sensitive primitive (CuPy `cp.pad` ↔ torch pad ordering, multi-axis loop, `exclusive` slice-then-prepad).
  - inclusive vs `exclusive=True` against a hand-written numpy `cumsum`+pad oracle;
  - **single axis and multi-axis** (`axis=[-1]`, `axis=[-2,-1]`); negative-axis normalization;
  - `exclusive` prepends a zero and drops the last element → assert **output shape equals input shape** and `out[...,0]==0`;
  - `L=1` (base case), `d=1`; float32 & float64.
- **`matrix_mult(X, Y, transpose_X, transpose_Y)`** — all four transpose combos vs `np.einsum`/`@`; batched (`[b,m,k]`); `Y=None` self-product symmetry; non-square.
- **`squared_euclid_dist(X, Y)`** — vs `scipy`/brute `(x-y)^2`; **non-negativity** (clamp ≥0 after cancellation); **zero diagonal** when `Y=None`; near-zero & large-magnitude inputs (cancellation stress); symmetry of the self-distance.
- **`outer_prod(X, Y)`** — shape `[..., d1*d2]`, value vs `np.outer` reshaped, batch dims preserved.
- **`squared_norm`, `matrix_diag`, `robust_sqrt`, `robust_nonzero`, `euclid_dist`** — value + the `_EPS` clamp behavior at exactly 0 and just below `_EPS`.
- **`draw_rademacher_matrix`, `draw_bernoulli_matrix`** — entries ∈ {−1,+1}/{0,1}, shape, requested `prob` within a binomial band (statistical, not exact), seeded determinism (inject-weights pattern §5.1).
- **`subsample_outer_prod`, `compute_count_sketch`, `convolve_fft`** — shape + value vs numpy oracle; `convolve_fft` especially (FFT backend differs CuPy↔torch) gets `DP_CUMSUM` tolerance and a real-output assertion.

### 7.2 Static kernels — `ksig/static/kernels.py` (`tests/unit/test_static_kernels.py`)

`Linear, Polynomial, RBF, Matern12/32/52, RationalQuadratic` — each via `_K` (Gram) and `_Kdiag`:
- Gram **symmetry** (`K(X)==K(X).T`) and **PSD-ish** (eigenvalues ≥ −tol);
- `_Kdiag(X)` equals `diag(_K(X))`;
- RBF/Matern/RatQuad diagonal **== 1** (or the stationary self-value);
- value vs golden / closed-form numpy; `Linear`/`Polynomial` get `EXACT_ALGEBRA`, RBF/Matern get `DP_CUMSUM` (because of `exp`);
- bandwidth/`sigma` and degree params swept; large-magnitude inputs to probe `exp` overflow handling.

### 7.3 Static features — `ksig/static/features.py` (`tests/unit/test_static_features.py`)

`RandomFourierFeatures`, `RandomFourierFeatures1D`, `NystroemFeatures`:
- `fit` then `transform` shape `[n, n_components]`, finite, dtype preserved;
- **estimand test** (§5.1b): `Φ(X) Φ(Y)^T ≈ RBFKernel(X, Y)` within the Monte-Carlo band; band tightens as `n_components` ↑;
- **inject-weights test** (§5.1a): with a numpy-seeded frequency matrix injected, `transform` matches golden exactly under `RANDOM_FEATURE`-but-exact;
- Nystroem: landmark count ≤ n, reconstruction of the Gram on the landmark set;
- seeded determinism: two `transform`s with the same generator are identical.

### 7.4 Projections — `ksig/projections.py` (`tests/unit/test_projections.py`)

`Gaussian, Subsampling, VerySparse, TensorSketch, TensorizedRandom, Diagonal`:
- `fit/transform` shape & dtype; output dimensionality matches `n_components`;
- **norm/inner-product preservation** within the JL band (`E[⟨Px,Py⟩] ≈ ⟨x,y⟩`) — estimand test;
- `TensorSketch`/`TensorizedRandom` (used by RFSF-TRP) get a dedicated test that the **tensored** projection of an outer product matches the projected tensor (the algebraic identity the method relies on);
- sparse projections: assert the realized sparsity matches the configured density (statistical band);
- inject-weights determinism for each.

### 7.5 Signature algorithms — `ksig/algorithms.py` (`tests/test_algorithms_dp.py`, `test_signature_full.py`, `test_signature_lowrank.py`)

This is the **highest-risk** module (the three Numba kernels become torch wavefronts / SYCL). Every routine is tested **twice**: against the **numpy-DP/brute oracle** (`tests/oracles/`) *and* against the frozen golden `.npz`.

- **`signature_kern` / `signature_kern_first_order` / `signature_kern_higher_order`** (full rank):
  - vs brute-force truncated-signature numpy oracle for tiny `(n_levels ≤ 3, L ≤ 5, d ≤ 3)`;
  - `order=1` fast path **vs** `order>1` higher-order path must agree where they should coincide;
  - monotone level structure, symmetry of the Gram, finite, `n_levels=1` base case.
- **`signature_kern_low_rank` & first/higher-order low-rank** (`test_signature_lowrank.py`):
  - low-rank features `P` satisfy `P P^T ≈` the full-rank Gram within `DP_CUMSUM` band (the whole point of low rank);
  - rank/`n_components` sweep shows convergence to full rank as rank ↑;
  - shape `[n, feature_dim]`, finite.
- **`signature_kern_pde` / `_signature_kern_pde`** (SigPDE):
  - vs **numpy PDE/DP oracle** (`oracles/dp_numpy.py`) on hand-built tiny inputs incl. `L=1`;
  - `difference=True/False` branches;
  - the `(0,0)` boundary condition explicitly checked.
- **`global_align_kern_log` / `_global_align_kern_log`** (GAK, log-space):
  - vs numpy log-space DP oracle; verify the `(0,0)` corner (`up=left=−inf, diag=0 ⇒ logsumexp=0`);
  - degenerate-constant and large-magnitude inputs (log-sum-exp stability);
  - symmetry, finite (no `-inf`/`nan` leaking out).
- **`random_warping_series` / `_random_warping_series_dtw`** (RWS/DTW):
  - vs numpy DTW oracle with **hand-computed** small alignments;
  - the **`l_Y = 1`** edge and **unequal warp lengths** explicitly (named in TORCH_PORT §4 as the scatter-index risk);
  - `DTW_EXACT` tolerance (min,+ is exact) — any nonzero diff is a real bug, not rounding.

### 7.6 Public kernel classes — `ksig/kernels.py` (`tests/test_kernels_public.py`, `test_signature_features.py`)

`SignatureKernel`, `SignaturePDEKernel`, `GlobalAlignmentKernel`, `SignatureFeatures` (RFSF-TRP/RFSF-DP), `RandomWarpingSeries`:
- the **README quick-start contracts**: `K_XX = k(X)` is `(n,n)` symmetric; `k(X, diag=True)` is `(n,)` and equals `diag(k(X))`; `k(X, Y)` is `(n_X, n_Y)`;
- **normalization**: when `normalize=True`, `diag(K)==1` within tolerance; idempotence of re-normalization;
- **fit/transform** for feature kernels: `transform(X) transform(Y)^T == k(X,Y)` within band (the identity printed in the README);
- `SignatureFeatures` with `TensorizedRandomProjection` (TRP) and `DiagonalProjection` (DP) both covered;
- value vs golden for each.

### 7.7 Preprocessing — `ksig/preprocessing.py` (`tests/unit/test_preprocessing.py`)

`SequenceTabulator`, `SequenceAugmentor`:
- **`SequenceTabulator`**: ragged list → uniform `[n, max_len, d]`; **NaN rows filtered** before interp; the `needs_interp` branch (variable length OR NaN OR over-max-len) vs the pass-through branch; `max_len` clipping; CPU(np)/GPU(xp) dispatch parity.
- **`SequenceAugmentor`**: each toggle (`add_time`, `lead_lag`, `basepoint`, `normalize`) independently and combined; output channel count matches the toggles (lead-lag doubles `d`, add_time +1, basepoint +1 timestep); **`normalize` uses population std (`ddof=0`)** — explicitly assert the torch port used `unbiased=False` (a classic CuPy→torch default-flip bug); `max_len` re-interpolation path.
- **biological path**: one-hot DNA/protein → `SequenceAugmentor`/cumsum-over-seq → finite, expected shape, monotone cumulative embedding.

### 7.8 Models — `ksig/models/` (`tests/test_models.py`)

`PrecomputedKernelSVC`, `PrecomputedFeatureLinSVC` (sklearn fallback, since cuML is NVIDIA-only and optional in the port):
- fit/predict on a small fixed labeled set with a fixed train/test split → **accuracy == golden** (`E2E_SCORE`, exact on f64);
- precomputed-kernel path: feeding a golden Gram yields the golden decision values;
- determinism across two fits with the same seed;
- shape/dtype of `decision_function`, label set preserved.

### 7.9 Device equivalence — `tests/test_device_equivalence.py`

Parametrized over `device ∈ {cpu}` always `+ {cuda, xpu, mps}` when available (auto-skip):
- **cpu-f64 vs xpu/cuda-f64**: tight (`DP_CUMSUM` band) for every public entry point;
- **cpu-f64 vs mps-f32**: loose column;
- **batching equivalence**: `batch_size=None` vs `batch_size=4` give the same Gram within `EXACT_ALGEBRA` (tiling must not change the math);
- **no silent CPU fallback**: assert the op actually ran on the requested device (probe in §11) — an XPU op that falls back to CPU is a *monitored* event, and in `xpu`-marked correctness tests it is a **failure**.

### 7.10 SYCL — `tests/test_sycl.py` (marker `sycl`, XPU only)

Skipped unless `ksig._sycl.loader.available()` and `torch.xpu.is_available()` with `aspect::fp64`:
- non-fused SigPDE/GAK/RWS SYCL kernels **vs the torch wavefront** (the wavefront is the numerical oracle here) within the f64 band of each family;
- fused `sig_pde_rbf` vs the non-fused materialized path;
- **torch↔SYCL stream ordering**: interleave SYCL calls with torch ops on the same XPU stream, assert no stale-data race (run N times);
- runs at sequence/batch sizes that OOM the materialized torch path (memory-win existence proof — correctness only; the *speedup* claim lives in `monitoring/`).

---

## 8. Pytest markers (`tests/conftest.py` registers them)

So the agent can run focused subsets during implementation:

| Marker | Selects | Typical use |
|---|---|---|
| `unit` | low-level, no golden needed (oracle/closed-form only) | fast inner loop while porting `utils.py` |
| `golden` | requires a `tests/golden/*.npz` | run where golden is present |
| `xpu` | requires `torch.xpu.is_available()` | Aurora |
| `sycl` | requires built `ksig._sycl` | Aurora, extension built |
| `slow` | large end of the dimension sweep | nightly |
| `stress` | extreme shapes, float16, OOM-probing | pre-release |
| `monitoring` | invokes a `monitoring/` probe as a smoke test (not a perf gate) | sanity that probes import/run |

Examples: `pytest -m "unit and not golden"` (develop with no oracle), `pytest -m "golden and not slow"` (CI), `pytest -m "xpu and sycl"` (Aurora full), `pytest tests/unit/test_utils_tensor_algebra.py -k multi_cumsum` (one primitive).

---

## 9. Common assertions (applied by every test, via shared helpers in `conftest.py`)

Every entry-point test asserts, in this order (cheap structural checks first, so failures are legible):

1. **Shape** — exact expected shape for the given input dims (the `which-dimension-broke` signal).
2. **Dtype** — output dtype == requested compute dtype (no silent upcast to f64 on XPU).
3. **Finite** — `torch.isfinite(out).all()` (no `nan`/`inf` leaking from `exp`, log-space, `/0`).
4. **Symmetry** — for any self-Gram, `‖K − K^T‖ ≤ atol`.
5. **Diagonal normalization** — `normalize=True ⇒ diag ≈ 1`; `diag=True` path equals `diag(full)`.
6. **Device** — output `.device` matches request; no unrequested host round-trip.
7. **Batching equivalence** — tiled == whole, where applicable.
8. **Seeded determinism** — same seed → identical output (inject-weights for random routines).
9. **Numerical agreement** — vs golden and/or oracle, under the §5 family tolerance.

**Pinpointing helper (the "what input dimensions things go wrong" requirement):** the comparison wrapper, on failure, reports
`fixture_id`, `(n, L, d, n_levels, order, dtype, device)`, **max-abs / max-rel diff**, the **flat index and unraveled coordinate of the worst element**, and the **count of elements outside tolerance**. This turns "assert failed" into "diverges first at sample 3, timestep 0, level 4, on float32/XPU only" — which localizes the port bug to a specific shape/dtype/device.

---

## 10. Monitoring strategy (`monitoring/` — separate from the pytest gate)

Performance is **measured, not asserted** in CI (timings are noisy and hardware-bound). The monitoring harness produces artifacts the agent and maintainers read to decide whether the SYCL path earns its keep.

- **Benchmark grids (`monitoring/grids.py`)** over `(n_samples, seq_len, n_features, n_levels, order, n_components, batch_size, dtype)`, in `small` / `medium` / `stress` tiers, for each public kernel + each DP kernel + each backend (`torch-cpu`, `torch-cuda`, `torch-xpu`, `sycl-xpu`).
- **Recorded per run (`monitoring/record.py` → JSONL + CSV)**:
  - wall time (median of K reps, with warmup), and per-phase time where separable;
  - **peak memory** (`torch.{cuda,xpu}.max_memory_allocated`) and host RSS;
  - **throughput** (pairs/s or sequences/s);
  - **device utilization** (sampled via the vendor tool: `nvidia-smi` / `xpu-smi` / Level-Zero sysman) over the run;
  - **fallback-to-CPU events** (probe §11 — any XPU kernel that silently ran on host);
  - **SYCL-vs-torch speedup** and **memory-reduction ratio** on identical fixtures;
  - **failure / blow-up thresholds** — the largest `(n, L)` each backend completes before OOM or a wall-time ceiling; recorded as a frontier, not a crash.
  - full provenance (same `versions` block as the golden sidecar) so numbers are comparable across machines/dates.
- **Schema:** one JSONL row per `(entry_point, backend, fixture, dtype, rep-aggregate)`; CSV mirror for quick plotting. `results/` is git-ignored (or LFS).
- A `monitoring`-marked pytest smoke test imports and runs each probe once on a tiny grid, so the harness can't rot.

> **No silent caps:** if a grid tier is truncated (top-N shapes, sampling, no-retry), the run **logs what it dropped** — a truncated grid must not read as "covered everything."

---

## 11. CPU-fallback & device-utilization probes (`monitoring/probes.py`)

The Intel XPU stack can transparently fall back to CPU for unimplemented ops — which silently destroys both correctness-locality and performance claims. Probes:

- wrap a region and assert (monitoring) / record (benchmark) that allocations landed on the XPU device and that no `aten::..._cpu` fallback fired (via the PyTorch fallback-warning hook / profiler op-device tags);
- sample device utilization and memory at a fixed cadence during a benchmark;
- in **`xpu`-marked correctness tests**, a detected fallback is a hard failure; in `monitoring`, it is a recorded event with the offending op name (feeds the port's "still needs a native kernel" list).

---

## 12. SYCL acceptance criteria (the gate for keeping a `.sycl` kernel)

A SYCL kernel is **accepted into the canonical path only if both** hold; otherwise the **torch-native path stays canonical** and the SYCL source is kept behind the `sycl` marker as experimental:

1. **Correctness:** passes `tests/test_sycl.py` — agrees with the torch wavefront (and thus golden) within the f64 band of its family, including the `l_Y=1`/unequal-length and stream-ordering cases.
2. **Benefit:** `monitoring/` shows a **measurable speedup OR memory reduction** over the torch-native XPU baseline on Aurora, on at least the `medium` grid tier, reproducibly (median over reps, beyond noise).

If correctness holds but there is no measurable benefit → **do not** dispatch to SYCL (torch-native is simpler and already correct). If benefit holds but correctness fails → **reject** (a fast wrong answer is worthless). This keeps the torch wavefront as the always-available fallback *and* the numerical oracle.

---

## 13. Build order (write the suite in this sequence)

1. `tests/conftest.py` + `tolerances.py` + `fixtures/` — the scaffolding everything imports. **No production code needed yet.**
2. `tests/oracles/dp_numpy.py` + `signature_numpy.py` — brute-force NumPy oracles **before** porting `algorithms.py` (TORCH_PORT §4 depends on these as ground truth).
3. `tests/freeze/make_golden.py` — run once on the CUDA box; commit `tests/golden/`.
4. `tests/unit/test_utils_tensor_algebra.py` — port `utils.py`, get these green first (everything else builds on them).
5. `unit/test_static_kernels.py` → `test_projections.py` → `test_static_features.py` → `unit/test_preprocessing.py`.
6. `test_algorithms_dp.py`, `test_signature_full.py`, `test_signature_lowrank.py` — the high-risk DP ports, validated vs oracle **and** golden.
7. `test_kernels_public.py`, `test_signature_features.py`, `test_models.py`.
8. `test_device_equivalence.py` — once ≥2 backends run.
9. `monitoring/` — stand up grids + probes once torch-native is correct.
10. `test_sycl.py` + SYCL acceptance loop (§12) — last, on Aurora with the extension built.

---

## 14. How to run

```bash
# Develop a primitive with no golden present:
pytest -m "unit and not golden" tests/unit/test_utils_tensor_algebra.py -k multi_cumsum

# Full CI on a box with golden committed:
pytest -m "golden and not slow"

# On Aurora, torch-native XPU:
pytest -m "xpu and not sycl"

# On Aurora, with the SYCL extension built:
pytest -m "xpu and sycl"

# Stage 1 (CUDA box, once):
python tests/freeze/make_golden.py --out tests/golden

# Performance (never in the correctness gate):
python monitoring/run_benchmarks.py --tier medium --backends torch-xpu,sycl-xpu --out monitoring/results
```
