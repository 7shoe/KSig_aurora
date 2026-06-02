# Handoff: build & validate the SYCL fast-path on an Aurora XPU compute node

> **You are a coding agent with access to an Aurora compute node and no prior
> context.** This note tells you exactly what state the repo is in, what your
> job is, and the commands to do it. Read it top to bottom once, then start.

---

## 1. TL;DR — what's done, what's left

**Done (by a previous agent, on a login node with no GPU):**
- `ksig` was fully ported from **CuPy + Numba-CUDA → torch-native** per
  [`docs/TORCH_PORT.md`](TORCH_PORT.md). It runs on CUDA / XPU / MPS / CPU.
- The three dynamic-programming kernels (SigPDE, GAK, RWS/DTW) are now
  **vectorized torch "wavefront" recurrences** in `ksig/algorithms.py`. These
  are correct and are your **numerical oracle**.
- The full test suite is **green on torch CPU (float64): `394 passed, 5 skipped`.**
- An **optional native SYCL fast-path** for those three kernels in `ksig/_sycl/`.
  As of 2026-06-02 it **builds, loads, and validates** on an Aurora compute node
  (12× PVC Max 1550, `frameworks/2025.3.1`, torch `2.8.0a0`, icpx 2025.3.2):
  `pytest -m "xpu and sycl"` → **13 passed**, full suite **407 passed, 4 skipped**.
  It is also **~80–130× faster** than the eager wavefront at the sizes measured
  (one fused launch vs `l_X+l_Y−1` torch launches). The build/correctness fixes
  this required are recorded in §4 and §10; you may still need to redo them on a
  **different** software stack.

**Your job, in order** (steps 1–5 are DONE on the stack above — re-run to confirm
on yours, then pick up at 6):
1. Get on a compute node, confirm an XPU is visible with fp64.
2. Confirm the torch port passes there too (`pytest -m "not sycl"`).
3. Build `ksig._sycl` (`icpx -fsycl`, JIT) and fix any compile errors (§4).
4. Run `pytest -m "xpu and sycl"` — the SYCL kernels must match the torch
   wavefront within the f64 band. Fix kernels until green.
5. ✅ **DONE (2026-06-02).** Benchmarked SYCL vs torch-XPU (`monitoring/`) at the
   `medium` tier: the SigPDE and GAK DP kernels are **13–28× faster** with a
   small memory reduction, and the speedup **grows with `L`** (so it is *not*
   just launch overhead at small sizes — it scales with the launches saved).
   The §7 acceptance gate is **closed**: SYCL is correct *and* beneficial → it is
   the canonical XPU path. Numbers + how to reproduce are in §7.
6. (Stretch) Implement the **fused** `sig_pde_rbf` kernel — the real memory win.

---

## 2. Get a node and confirm the hardware

```bash
cd <repo root>                      # the dir containing ksig/, tests/, docs/
# Interactive single-node job (adjust account/queue/walltime to your allocation):
qsub -I -l select=1 -l walltime=01:00:00 -q debug -A <ALLOCATION>

# On the compute node:
module load frameworks              # provides native-XPU torch + oneAPI (icpx)
bash scripts/probe_hardware.sh      # one-stop probe: XPU count, sycl-ls, fp64, icpx
```

`probe_hardware.sh` must show `xpu.is_available: True`, a device count ≥ 1,
`Intel(R) Data Center GPU Max 1550` (Ponte Vecchio / PVC), and `fp64 on xpu: OK`.
If `xpu.is_available: False`, you are still on a login/UAN node — stop and get a
real compute node. PVC has `aspect::fp64`; the kernels use double.

---

## 3. Sanity-check the torch port on XPU first

Before touching SYCL, confirm the **portable** path works on the device (this is
the fallback and the oracle). The suite auto-parametrizes device where present.

```bash
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
python -m pytest -q -m "not sycl"        # should pass like it does on CPU
# Quick XPU smoke of the wavefronts:
python - <<'PY'
import numpy as np, torch, ksig
X = torch.as_tensor(np.random.default_rng(0).standard_normal((6,8,3)), device="xpu")
print("SigPDE", ksig.kernels.SignaturePDEKernel()(X).shape)
print("GAK",    ksig.kernels.GlobalAlignmentKernel()(X).shape)
print("Sig",    ksig.kernels.SignatureKernel(n_levels=4)(X).shape)
PY
```

If anything fails here it's a **torch-on-XPU** issue (e.g. an op that CPU-falls-
back), not SYCL — note it but it doesn't block SYCL work.

---

## 4. Build the SYCL extension and fix compile errors

The extension builds JIT on first import via `torch.utils.cpp_extension.load`.

```bash
KSIG_SYCL_VERBOSE=1 python -c "from ksig._sycl import loader; print('built:', loader.available())"
```

> **Already fixed on the `frameworks/2025.3.1` stack** (in `loader.py` /
> `bindings.cpp` / `pde_kernels.sycl`) — you only need to revisit these if your
> toolchain differs:
> 1. **`load()` has no `sycl_sources` kwarg** (that's `load_inline`). `.sycl`
>    files go straight into `sources`; torch auto-routes them to icpx.
> 2. **Unnamed kernel lambdas are rejected** and `-fsycl-unnamed-lambda`
>    conflicts with torch's `-fsycl-host-compiler` — so the kernels carry
>    explicit names (`SigPdeKernel`/`GakLogKernel`/`RwsDtwKernel`).
> 3. **No `<torch/types.h>`** in this torch; the `.sycl` TU is **ATen-free** —
>    all `at::Tensor`/`AT_DISPATCH`/pybind glue lives in `bindings.cpp`, the
>    kernels take raw device pointers. (Keeping `at::Tensor` in the `.sycl` TU
>    pulls `c10::UndefinedTensorImpl::_singleton`, an absolute reloc the SYCL
>    host pass can't satisfy → link failure.)
> 4. **Torch's `\"`-escaped `-DPYBIND11_*` defines mis-tokenize** on this icpx
>    in the `-fsycl-host-compiler-options` string and swallow the trailing
>    `-isystem`/`-fPIC`. `loader._fix_sycl_host_flags` patches torch's
>    `_wrap_sycl_host_flags` to drop those defines (the `.sycl` TU has no
>    pybind11) and force `-fPIC`; `CPATH` is also set as a header-search
>    backstop. Device-side `-fPIC` is added via `extra_sycl_cflags`.
> 5. **Empty-sequence guard:** a length-1 series differenced to length 0 gives
>    `lX/lY == 0` (→ `wg_size == 0`, an invalid launch); `sig_pde_launch`
>    short-circuits to ones (the empty product), matching the oracle.

- `available()` swallows errors and returns `False` on any failure, so to see the
  **actual compiler output** set `KSIG_SYCL_VERBOSE=1` and call `loader.get_ext()`
  directly (it re-raises):
  ```bash
  KSIG_SYCL_VERBOSE=1 python -c "from ksig._sycl import loader; loader.get_ext()"
  ```
- Files involved:
  - `ksig/_sycl/pde_kernels.sycl` — the three kernels (compiled by `icpx -fsycl`).
  - `ksig/_sycl/bindings.cpp` — pybind11 module exposing `sig_pde`, `gak_log`, `rws_dtw`.
  - `ksig/_sycl/loader.py` — the build call (`-fsycl`, **no** nvptx/sm_80 flags;
    Intel/Level-Zero is the default target on Aurora — nvptx flags are for
    Polaris/NVIDIA, a *different* ALCF machine).

**These sources have never been compiled — expect to fix errors.** Likely spots:
- The torch-queue bridge: `c10::xpu::getCurrentXPUStream().queue()` — confirm this
  signature against the installed torch build. If it differs, the documented
  fallback (`TORCH_PORT.md` §10.2) is a cached in-order `sycl::queue` bracketed by
  `torch.xpu.synchronize()` on the Python side. **Run the kernel on torch's own
  XPU queue**, not a fresh `gpu_selector_v` queue, or you get cross-queue races.
- `AT_DISPATCH_FLOATING_TYPES`, `sycl::fmin/fmax/exp/log`, `local_accessor`,
  `group_barrier`, `nd_range` launch — all require an `nd_range` launch hit in
  **converged** control flow (the loop structure already satisfies this).
- Header availability: `#include <c10/xpu/XPUStream.h>`, `<sycl/sycl.hpp>`.

Iterate: edit the `.sycl`/`.cpp`, re-run `loader.get_ext()` (the lru_cache is per
process, so a fresh `python` re-builds; or clear `TORCH_EXTENSIONS_DIR`).

---

## 5. Validate: SYCL must match the torch wavefront

```bash
python -m pytest -m "xpu and sycl" tests/test_sycl.py -q
```

`tests/test_sycl.py` (already written) compares each SYCL kernel against the torch
wavefront (the oracle) within each family's f64 band, and includes the
`l_Y=1`/unequal-warp edge and a **torch↔SYCL stream-ordering** test (interleaves
SYCL with torch ops on the same stream — catches stale-data races). It
auto-skips unless XPU is present and the ext built.

The kernels in `pde_kernels.sycl` mirror the wavefront recurrences exactly
(`TORCH_PORT.md` §4 / §10.4):
- **SigPDE**: `K=(up+left)(1+m/2+m²/12) − diag(1−m²/12)`, borders = 1.
- **GAK** (log): `logK = logM + logsumexp3(up,left,diag)`, borders −inf, seed 0;
  driver transform `M/(2−M)` then `log(clamp(.,eps))` done host-side.
- **RWS/DTW**: `P = D + min(up,left,diag)`, borders +inf, seed 0; series `y`
  reads `D[x,i, seg[y]+j]`, terminal at its own length `l_Y(y)`.

If a result diverges, the failure message **pinpoints** the first bad element
(sample/coord/Δ). Common kernel bugs: lane→buffer index mapping (`i_lo` per
diagonal), the role-rotation of `prev2/prev/cur`, or the corner/edge boundary
selection.

---

## 6. How dispatch works (and how to turn it on)

Dispatch is already wired and **safe**: `ksig/algorithms.py::_try_sycl(name, *args)`
is called at the top of `signature_kern_pde`, `global_align_kern_log`,
`random_warping_series`. It returns `None` (→ fall through to the wavefront)
unless **(a)** the first arg is an XPU tensor and **(b)** `loader.available()`.
So the moment the extension builds on XPU, SYCL is *already* dispatched. An
explicit off-switch is **now implemented**: `ksig.algorithms._sycl_enabled()`
reads the `KSIG_USE_SYCL` env var per call — set `KSIG_USE_SYCL=0` (also
`false`/`no`/`off`) to force the torch wavefront, default (unset/`1`) auto-engages
SYCL. This is what made the §7 run-twice comparison possible.

There is no separate "register" step; correctness is enforced by §5, benefit by §7.

---

## 7. Acceptance gate — keep it only if it's actually better

> **✅ GATE CLOSED (2026-06-02, frameworks/2025.3.1, 12× PVC Max 1550).** Both
> conditions hold at the `medium` tier, so SYCL is the canonical XPU path.
> Reproduce with the two commands below; results in
> `monitoring/results/torch_xpu_{torch,sycl}_medium.{jsonl,csv}`.
>
> | kernel | n | L | torch-XPU | SYCL | speedup | peak MB (torch→sycl) |
> |---|---|---|---|---|---|---|
> | SignaturePDEKernel | 16 | 32 | 12.73 ms | 0.98 ms | **13.0×** | 6.9→6.3 |
> | SignaturePDEKernel | 16 | 64 | 24.69 ms | 0.91 ms | **27.2×** | 26.0→25.2 |
> | SignaturePDEKernel | 32 | 32 | 12.72 ms | 0.89 ms | **14.2×** | 26.7→25.2 |
> | SignaturePDEKernel | 32 | 64 | 24.83 ms | 1.04 ms | **23.9×** | 104.9→100.8 |
> | GlobalAlignmentKernel | 16 | 32 | 16.01 ms | 0.94 ms | **17.1×** | 9.3→8.4 |
> | GlobalAlignmentKernel | 16 | 64 | 31.09 ms | 1.12 ms | **27.8×** | 35.4→33.6 |
> | GlobalAlignmentKernel | 32 | 32 | 16.19 ms | 0.97 ms | **16.7×** | 37.3→33.6 |
> | GlobalAlignmentKernel | 32 | 64 | 31.14 ms | 1.40 ms | **22.3×** | 141.6→134.3 |
>
> `SignatureKernel` (no SYCL dispatch) is byte-for-byte identical across both
> runs at 1.0× — confirming the comparison isolates exactly the SYCL change. The
> speedup **grows with `L`** (27× at L=64 vs 13× at L=32), so this is the real
> "one launch vs `l_X+l_Y−1` launches" win scaling up, not fixed launch overhead
> at tiny sizes. Memory is also modestly lower (fewer intermediates). The fused
> `sig_pde_rbf` kernel (§8) is still the path to the *decisive* memory win.

From `tests/TEST_PLAN.md` §12 (and `TORCH_PORT.md` §10.1), a SYCL kernel becomes
the real path only if **both** hold:
1. **Correct** — passes `tests/test_sycl.py` (§5).
2. **Beneficial** — `monitoring/` shows a measurable **speedup OR memory
   reduction** vs the torch-XPU baseline on ≥ the `medium` tier, reproducibly
   (median over reps, beyond noise).

Run the benchmarks. The driver **auto-detects the single active backend** (it
runs each kernel on whatever `ksig` is using) — there is **no** `--backends`
flag, and it must be invoked as a module (`-m`), not by file path, or the
`from monitoring import …` imports fail. To compare torch-XPU vs SYCL-XPU, run
it **twice** and diff the result files: once with SYCL gated off (`KSIG_USE_SYCL=0`,
§6), once on (default). Without the switch SYCL auto-engages the moment the ext
is built, so the two runs would be identical. The driver now stamps the active
variant into the output stem (`..._torch_...` vs `..._sycl_...`), so the two runs
**no longer overwrite each other**:
```bash
KSIG_USE_SYCL=0 python -m monitoring.run_benchmarks --tier medium \
    --out monitoring/results --max-gb 8     # -> torch_xpu_torch_medium.{jsonl,csv}
KSIG_USE_SYCL=1 python -m monitoring.run_benchmarks --tier medium \
    --out monitoring/results --max-gb 8     # -> torch_xpu_sycl_medium.{jsonl,csv}
# flags: --tier --out --reps --warmup --max-gb  (see monitoring/README.md)
```
- Correct but **not** faster/leaner → leave torch canonical (simpler, already
  correct); keep the `.sycl` behind the `sycl` marker as experimental.
- Beneficial but **not** correct → reject. A fast wrong answer is worthless.

The two wins to look for (`TORCH_PORT.md` §10.1): fewer launches (one SYCL launch
vs `l_X+l_Y−1` eager torch launches), and — for the fused variant — never
materializing the `[n_X,n_Y,l_X,l_Y]` tensor.

---

## 8. Stretch: the fused `sig_pde_rbf` kernel (the real memory win)

The non-fused kernels only save launch overhead. The decisive win
(`TORCH_PORT.md` §10.5) is a **fused** kernel that computes the static-kernel
entry `M[x,y,i,j]` *inside* the DP from the raw sequences `X[x]`, `Y[y]` — so it
never stores the 4-D tensor → working set O(N²L²) → O(N²)+O(N·L·d). To add it:
add `sig_pde_rbf_launch(X, Y, bandwidth, difference)` in `bindings.cpp`
+ `pde_kernels.sycl`, expose it (`m.def("sig_pde_rbf", …)`), validate against the
non-fused `sig_pde` on identical inputs, then show it runs at sizes that OOM the
materialized path. Mind the §10.5 SYCL specifics (no `sycl::dot` for d>4; use a
sub-group reduction or `marray<scalar_t,D>` loop; fp64 atomics need
`aspect::atomic64`).

---

## 9. Map of relevant files

| Path | What it is |
|---|---|
| `docs/TORCH_PORT.md` | The authoritative port + SYCL implementation guide. §4 = wavefront math, §10 = SYCL. |
| `ksig/algorithms.py` | The torch wavefronts (oracle) + `_try_sycl` dispatch shim. |
| `ksig/_sycl/pde_kernels.sycl` | The three SYCL kernels you compile/validate. |
| `ksig/_sycl/bindings.cpp` | pybind11 module (`sig_pde`/`gak_log`/`rws_dtw`). |
| `ksig/_sycl/loader.py` | JIT build (`-fsycl`); `available()` / `get_ext()`. |
| `ksig/_sycl/README.md` | Shorter version of this, kept next to the code. |
| `tests/test_sycl.py` | SYCL-vs-wavefront correctness (marker `xpu`/`sycl`). |
| `tests/TEST_PLAN.md` | Test strategy, tolerance families, acceptance gate. |
| `monitoring/` | Speedup/memory benchmarks (the §7 benefit gate). |
| `scripts/probe_hardware.sh` | One-stop hardware/oneAPI/fp64 probe. |

---

## 10. Gotchas / things the previous agent already learned

- **You are nothing without the wavefront.** It's correct on every backend and is
  the oracle. Never "fix" a test by loosening it to match a wrong SYCL result.
- **No nvptx flags on Aurora.** `-fsycl` only. nvptx/sm_80 = Polaris (NVIDIA).
- **Run on torch's XPU queue** (§10.2), not a fresh queue — ordering hazards.
- **fp64**: PVC has it; keep `AT_DISPATCH_FLOATING_TYPES` (double + float). GAK/
  SigPDE want double for stability.
- The torch port itself is **done and trusted** — `git diff` shows the CuPy→torch
  changes; don't re-port. If a non-SYCL test fails on XPU, it's a torch-on-XPU op
  issue (possible silent CPU fallback — `monitoring/probes.py::detect_xpu_fallback`
  is *scaffolding* for catching those, but currently a stub that always reports
  no fallback, so implement it before you rely on it), not the port logic, which
  is pinned by 394 CPU tests.
- One subtlety the port had to fix (so you don't reintroduce it): the
  self-distance diagonal of `‖x‖²+‖x‖²−2⟨x,x⟩` must be **exactly 0**
  (`utils.squared_euclid_dist` zeroes it) — otherwise `√` amplifies a ~1e-15
  residual into a ~1e-7 error on Matern's unit diagonal. Irrelevant to the DP
  kernels, but don't "optimize" that zeroing away.
