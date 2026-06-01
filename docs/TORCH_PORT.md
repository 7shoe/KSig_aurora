# KSig → PyTorch Port + Aurora-XPU SYCL Fast-Path: Implementation Plan

> **Audience:** a coding agent with access to this repository (`/Users/carlo/Projects/KSig`) and this document.
> **Mandate:** translate KSig from **CuPy + Numba-CUDA** to a **PyTorch-native** library that runs on NVIDIA CUDA, Intel XPU (Aurora), and Apple MPS, then add an **optional native SYCL fast-path** for the three dynamic-programming (DP) kernels on Aurora.
> **Decisions already made by the user:** (1) **Full replacement** of CuPy/Numba — single torch backend, no dual-backend maintenance; cuML becomes an optional extra. (2) **Full SYCL implementation guide** — including build setup and near-complete `.sycl` sources for the three DP kernels.

---

## 1. Context

KSig is a scikit-learn-compatible library for GPU-accelerated time-series kernels (signature kernel, Signature-PDE, Global Alignment Kernel, Random Warping Series, plus signature features and projections). Today **all GPU work goes through CuPy**, and three core DP algorithms are hand-written **Numba `@cuda.jit`** kernels. Both CuPy and Numba-CUDA are **NVIDIA-only** (CuPy has no SYCL/Metal backend; Numba CUDA is NVIDIA-exclusive). This hard-locks the library to NVIDIA hardware and makes it unrunnable on **Aurora's Intel Max 1550 (Ponte Vecchio) GPUs** and on **Apple Silicon (MPS)**.

**Why CuPy/Numba were originally chosen** (so the porter understands the design intent):
- The codebase is written in **NumPy idioms** (`cp.diff`, `cp.cumsum`, `cp.einsum`, `cp.pad`, `cupy.random.RandomState`, `...`-broadcasting) with sklearn `fit`/`transform`/`__call__` conventions. CuPy is a drop-in NumPy-on-GPU, so it was the path of least resistance.
- The three DP recurrences (SigPDE, GAK, RWS/DTW) are **not** expressible as clean dense tensor ops; they were written as custom CUDA kernels (one block per sequence pair, threads cooperating along an antidiagonal via shared memory). **CuPy ↔ Numba-CUDA interoperate zero-copy** (both speak `__cuda_array_interface__`), so CuPy was the natural array type to pair with them.
- The models layer uses **cuML** (`cuml.svm.LinearSVC`), which is CUDA/CuPy-only.

**Intended outcome:** one torch codebase that runs on CUDA, XPU, and MPS; the three DP kernels rewritten as **vectorized torch "wavefront" recurrences** (portable *and* generally better GPU utilization than the current per-pair-block kernels); and an **optional SYCL fast-path** on Aurora that can additionally **fuse the static-kernel evaluation into the DP** to defeat the O(N²L²) memory bottleneck.

---

## 2. Scope & target layout

**Files to change** (all under `ksig/`): `utils.py`, `algorithms.py`, `kernels.py`, `projections.py`, `preprocessing.py`, `static/kernels.py`, `static/features.py`, `models/pre_base.py`, `models/pre_svc.py`, `models/pre_lin_svc.py`, `__init__.py`, plus `setup.py`.

**New files:**
- `ksig/torch_backend.py` — device/dtype/RNG policy (imported by every module).
- `ksig/_sycl/` — optional SYCL fast-path: `pde_kernels.sycl`, `bindings.cpp`, `loader.py`.
- `tests/` — numerical-equivalence suite.
- `docs/TORCH_PORT.md` — final home of this plan in the repo (copy this document there on first commit).

**Principle:** do **not** alias `cp = torch`. Torch's API diverges from CuPy in enough places (`dim` vs `axis`, no `asnumpy`, no `RandomState`, pad ordering, `std` ddof default, no `interp`/`apply_along_axis`) that a thin shim is fragile. Do **explicit per-call translation** per the table in §5.

---

## 3. `ksig/torch_backend.py` (NEW — implement first)

Centralizes everything device/dtype/RNG. Every other module imports from here.

```python
"""Device, dtype, and RNG policy for the torch backend."""
import numpy as np, torch
from numbers import Integral
from typing import Optional, Union

_EPS = 1e-12

def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None: return torch.device(prefer)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")  # Aurora
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

_DEFAULT_DEVICE = get_device()
def current_device() -> torch.device: return _DEFAULT_DEVICE
def set_default_device(dev) -> None:
    global _DEFAULT_DEVICE; _DEFAULT_DEVICE = torch.device(dev)

def supports_float64(device=None) -> bool:
    device = device or current_device()
    return device.type != "mps"             # MPS has no float64

def default_float_dtype(device=None) -> torch.dtype:
    device = device or current_device()
    return torch.float64 if supports_float64(device) else torch.float32

def eps_for(dtype) -> float:
    return 1e-12 if dtype == torch.float64 else 1e-7   # 1e-12 underflows float32

def as_tensor(x, dtype=None, device=None) -> torch.Tensor:
    device = device or current_device()
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if (dtype is not None or x.device != device) else x
    return torch.as_tensor(np.asarray(x),
                           dtype=dtype or default_float_dtype(device), device=device)

def as_index(x, device=None) -> torch.Tensor:           # integer index arrays stay long
    device = device or current_device()
    if isinstance(x, torch.Tensor): return x.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)

def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

class TorchRandomState:
    """Drop-in for cupy.random.RandomState used by KSig (normal/uniform/randint/choice)."""
    def __init__(self, seed=None, device=None):
        self.device = device or current_device()
        # MPS cannot host a Generator: generate on CPU then move.
        self._gen_device = "cpu" if self.device.type == "mps" else self.device
        self.generator = torch.Generator(device=self._gen_device)
        if seed is not None: self.generator.manual_seed(int(seed))
    def _move(self, t): return t.to(self.device)
    def normal(self, size, dtype=None):
        dt = dtype or default_float_dtype(self.device)
        return self._move(torch.randn(*size, generator=self.generator,
                                      device=self._gen_device, dtype=dt))
    def uniform(self, size, low=0., high=1., dtype=None):
        dt = dtype or default_float_dtype(self.device)
        u = torch.rand(*size, generator=self.generator, device=self._gen_device, dtype=dt)
        return self._move(low + (high - low) * u)
    def randint(self, low, high=None, size=None):
        if high is None: low, high = 0, low
        return self._move(torch.randint(low, high, tuple(size), generator=self.generator,
                                        device=self._gen_device))
    def choice(self, n, size, replace=False):
        assert not replace, "KSig only uses replace=False"
        return self._move(torch.randperm(n, generator=self.generator,
                                         device=self._gen_device)[:size])

RandomStateOrSeed = Union[Integral, TorchRandomState]
def check_random_state(rs=None) -> TorchRandomState:
    if rs is None or isinstance(rs, Integral): return TorchRandomState(rs)
    if isinstance(rs, TorchRandomState): return rs
    raise ValueError(f"{rs} cannot seed a TorchRandomState")
```

**XPU note:** on Aurora, ensure torch sees the GPU. Recent ALCF modules ship native-XPU torch (no IPEX needed); on older stacks `import intel_extension_for_pytorch` may be required — guard it in `__init__.py` with a `try/except`.

`utils.check_random_state` and `utils.RandomStateOrSeed` should re-export these (keep `utils` as the public import site the rest of the code already uses).

---

## 4. The three DP kernels → vectorized torch wavefront recurrences

This is the highest-risk, highest-value part. All three Numba kernels (`algorithms.py:344`, `:453`, `:566`) share one skeleton: one CUDA block per `(idx_X, idx_Y)` pair, threads sweep a single **antidiagonal**, a rolling buffer of `3·min(l_X,l_Y)` holds (two-ago, one-ago, current) diagonals; `num_iter = l_X+l_Y-1` antidiagonals; result is the `(l_X-1, l_Y-1)` cell. The `I==0/I==1` index branching and `3*k±…` arithmetic exist only to pack storage into shared memory — **irrelevant in torch.**

### 4.1 Recommended layout: full padded DP table (Option B)

Allocate a padded table `H` of shape `[n_X, n_Y, l_X+1, l_Y+1]`; data cell `(i,j)` lives at `H[...,i+1,j+1]`. The input `M`/`logM` is **already** `[n_X,n_Y,l_X,l_Y]`, so `H` is the same memory order — no asymptotic blow-up. Neighbors become trivial slices; boundary conditions become padded borders. (Option A, a rolling 3-diagonal buffer of `[n_X,n_Y,min(l_X,l_Y)]`, saves a factor ~`min(l)/(l_X·l_Y)` of working set but reintroduces the realignment indexing; implement only as an optional memory-saver after Option B is correct.)

The antidiagonal sweep stays a Python `for it in range(l_X+l_Y-1)` loop (hundreds of iterations); each iteration is **one vectorized op** over the whole batch and the whole current antidiagonal.

**Shared antidiagonal index helper:**
```python
def _antidiag_indices(it, l_X, l_Y, device):
    i_lo = max(0, it - (l_Y - 1)); i_hi = min(it, l_X - 1)
    i = torch.arange(i_lo, i_hi + 1, device=device)   # [d]
    j = it - i                                         # [d]
    return i, j
```

### 4.2 SigPDE — `signature_kern_pde` (from `_signature_kern_pde`, L379–L385)

**Recurrence** (with `m = M[x,y,i,j]`):
```
K(i,j) = (K(i-1,j) + K(i,j-1)) · (1 + ½·m + m²/12)  −  K(i-1,j-1) · (1 − m²/12)
```
**Boundaries:** `K(i-1,j)=1` if `i==0`; `K(i,j-1)=1` if `j==0`; `K(i-1,j-1)=1` if `i==0 or j==0`.
**Encoding:** init `H[...,0,:] = 1` and `H[...,:,0] = 1`.
```python
def signature_kern_pde(M, difference=True):
    is_diag = (M.ndim == 3)
    if is_diag: M = M[:, None]
    if difference: M = torch.diff(torch.diff(M, dim=-2), dim=-1)
    nX, nY, lX, lY = M.shape; dev = M.device
    H = torch.ones((nX, nY, lX + 1, lY + 1), dtype=M.dtype, device=dev)
    for it in range(lX + lY - 1):
        i, j = _antidiag_indices(it, lX, lY, dev)
        up   = H[:, :, i,     j + 1]
        left = H[:, :, i + 1, j]
        diag = H[:, :, i,     j]
        m    = M[:, :, i, j]
        H[:, :, i + 1, j + 1] = (up + left) * (1 + 0.5*m + m*m/12) - diag * (1 - m*m/12)
    K = H[:, :, lX, lY]
    return K.squeeze(1) if is_diag else K
```
**Numerical:** pure polynomial; error accumulates multiplicatively over `lX+lY` steps → keep **float64** where available. No `_EPS`. Preserve `M.dtype` exactly.

### 4.3 GAK log-space — `global_align_kern_log` (from `_global_align_kern_log`, L488–L497, L531–L534)

**Driver transforms:** `M = M / (2 - M)`; `logM = torch.log(torch.clamp(M, min=_EPS))`.
**Recurrence:** `logK(i,j) = logM(i,j) + logsumexp( logK(i-1,j), logK(i,j-1), logK(i-1,j-1) )`.
**Boundaries:** up `=-inf` if `i==0`; left `=-inf` if `j==0`; diag `=0` if `i==0 and j==0`, else `-inf` on an edge.
**Encoding:** fill all borders with `-inf`, then set the single seed `H[...,0,0] = 0`.
```python
def global_align_kern_log(M):
    is_diag = (M.ndim == 3)
    if is_diag: M = M[:, None]
    M = M / (2. - M)
    logM = torch.log(torch.clamp(M, min=eps_for(M.dtype)))
    nX, nY, lX, lY = M.shape; dev = M.device
    NEG = torch.finfo(M.dtype).min          # use a large negative; -inf also works
    H = torch.full((nX, nY, lX + 1, lY + 1), float("-inf"), dtype=M.dtype, device=dev)
    H[:, :, 0, 0] = 0.
    for it in range(lX + lY - 1):
        i, j = _antidiag_indices(it, lX, lY, dev)
        up   = H[:, :, i,     j + 1]
        left = H[:, :, i + 1, j]
        diag = H[:, :, i,     j]
        H[:, :, i + 1, j + 1] = logM[:, :, i, j] + torch.logsumexp(
            torch.stack([up, left, diag], dim=0), dim=0)
    logK = H[:, :, lX, lY]
    return logK.squeeze(1) if is_diag else logK
```
Verify corner: at `(0,0)`, up=left=−inf, diag=0 ⇒ `logsumexp=0` ⇒ `logM(0,0)` ✓. `torch.logsumexp` does the max-shift internally and is `-inf`-safe. Exponentiation + log-space normalization stay in `kernels.py::SignatureKernel._K` (the `is_log_space` branch). **Strongly prefer float64;** on MPS expect divergence (document looser tolerance). Match `_EPS` clamp exactly.

### 4.4 RWS / DTW — `random_warping_series` (from `_random_warping_series_dtw`, L603–L608)

**Recurrence:** `P(i,j) = D(i,j) + min( P(i-1,j), P(i,j-1), P(i-1,j-1) )`.
**Boundaries:** up `=+inf` if `i==0`; left `=+inf` if `j==0`; diag `=0` if `i==0 and j==0`, else `+inf`.
**Variable warp lengths:** `D` is `[n_X, l_X, Σ l_Y]`; series `y` occupies columns `[warp_segments[y], warp_segments[y+1])`. **Pad-and-gather** strategy (recommended): build `Dpad` of `[n_X, n_Y, l_X, l_Y_max]` (valid columns filled, rest arbitrary-finite), run the identical antidiagonal DP with `min`/`+inf`, then read each series at its **true** terminal column `l_Y(y)`. Right-padding never corrupts valid cells because the DP only reads smaller indices.
```python
def random_warping_series(D, warp_lens):
    nX, lX = D.shape[:2]; nY = warp_lens.shape[0]; dev = D.device
    seg = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                     torch.cumsum(warp_lens.to(torch.long), 0)])     # [nY+1]
    lY = (seg[1:] - seg[:-1])                                         # [nY]
    lY_max = int(lY.max())
    # scatter D -> Dpad[x, y, i, 0:lY(y)]
    Dpad = torch.full((nX, nY, lX, lY_max), float("inf"), dtype=D.dtype, device=dev)
    for y in range(nY):                       # nY ~ n_components; loop is fine, or vectorize via index map
        Dpad[:, y, :, :lY[y]] = D[:, :, seg[y]:seg[y+1]]
    H = torch.full((nX, nY, lX + 1, lY_max + 1), float("inf"), dtype=D.dtype, device=dev)
    H[:, :, 0, 0] = 0.
    for it in range(lX + lY_max - 1):
        i, j = _antidiag_indices(it, lX, lY_max, dev)
        up   = H[:, :, i,     j + 1]
        left = H[:, :, i + 1, j]
        diag = H[:, :, i,     j]
        H[:, :, i + 1, j + 1] = Dpad[:, :, i, j] + torch.minimum(torch.minimum(up, left), diag)
    # gather each series at its own terminal column
    term = lY.view(1, nY, 1, 1).expand(nX, nY, 1, 1)
    P = H[:, :, lX, :].gather(-1, term.squeeze(-1)).squeeze(-1)       # [nX, nY]
    return P
```
(If the per-`y` scatter loop is hot, replace with a precomputed column-index map via `torch.arange`/`repeat_interleave` from `seg`.) **Numerical:** `min`/`+` is exact → tightest tolerance. Only risk is `seg`/scatter index arithmetic — test against a hand-computed DTW including `l_Y=1` and unequal lengths. Use `+inf` of `D.dtype`.

### 4.5 Factoring

A single private `_wavefront(...)` parameterized by border-init, corner-seed, and a `combine(up,left,diag,payload)` callable is acceptable, but the three combines differ enough (polynomial vs logsumexp vs min) that three concrete functions sharing `_antidiag_indices` is cleaner. Remove all `numba`/`cuda` imports and the `MaxSharedMemoryPerBlock` logic from `algorithms.py`.

### 4.6 Why this is also *faster* (efficiency goal)

The current CUDA kernel launches one block per `(i,j)` pair with only `min(l_X,l_Y)` threads — for short sequences (the common case) this underutilizes the GPU and serializes the antidiagonal anyway. The vectorized version parallelizes across the **entire** `n_X·n_Y` batch **and** the antidiagonal in each step. Same number of sequential steps, far more work each. On XPU, `torch.compile` can further fuse the per-step elementwise math.

---

## 5. Mechanical array-API translation (cupy → torch)

Applies to `utils.py`, `static/*`, `projections.py`, `preprocessing.py`, `kernels.py`, and the full/low-rank parts of `algorithms.py`.

| CuPy | torch | Notes |
|---|---|---|
| `cp.asarray(x)` | `as_tensor(x)` | set device+dtype |
| `cp.asnumpy(x)` | `to_numpy(x)` | |
| `cp.ndarray` (alias) | `torch.Tensor` | `ArrayOnGPU = torch.Tensor` |
| `cp.diff/cumsum(…, axis=a)` | `torch.diff/cumsum(…, dim=a)` | |
| `cp.einsum(s, …)` | `torch.einsum(s, …)` | identical strings |
| `cp.pad(M, pads)` | `torch.nn.functional.pad` | **reverse-axis flat tuple** — see §5.1 |
| `cp.fft.fft/ifft(…, axis=-1)`, `cp.real` | `torch.fft.fft/ifft(…, dim=-1)`, `.real` | MPS may CPU-fallback |
| `cp.where(cond, a, b)` | `torch.where(cond, a, b)` | |
| `cp.where(cond)[0]` (1-arg) | `torch.nonzero(cond, as_tuple=True)[0]` | `projections.py:340` |
| `cp.take(X, idx, axis=a)` | `torch.index_select(X, a, idx)` | `idx` must be `long`; positive dim |
| `cp.maximum(a, eps)` | `torch.clamp(a, min=eps)` | use `eps_for(dtype)` for f32 safety |
| `cp.maximum(a, b)` (tensors) | `torch.maximum(a, b)` | |
| `cp.power(a, p)` | `torch.pow(a, p)` or `a**p` | |
| `cp.sqrt/square/abs/exp/log/sin/cos` | `torch.sqrt/square/abs/exp/log/sin/cos` | |
| `cp.ones/zeros/empty/full(shape, dtype=)` | add `device=` | `cp.full((n,),1)` → `torch.full((n,),1.0,dtype=float,device=)` (cupy made int!) |
| `cp.zeros_like/ones_like` | `torch.zeros_like/ones_like` | |
| `cp.copy(x)` | `x.clone()` | |
| `cp.concatenate(seq, axis=a)` | `torch.cat(seq, dim=a)` | |
| `cp.stack(seq, axis=a)` | `torch.stack(seq, dim=a)` | |
| `x.reshape(...)`, `cp.squeeze(x, axis=a)` | `x.reshape(...)`, `x.squeeze(a)` | |
| `cp.arange/linspace(...)` | add `device=` | |
| `cp.tile(x, reps)` | `torch.tile(x, reps)` | |
| `cp.repeat(x, k, axis=a)` | `torch.repeat_interleave(x, k, dim=a)` | **not** `torch.repeat` (preprocessing lead-lag) |
| `cp.linalg.eigh(A)` | `torch.linalg.eigh(A)` | ascending eigvals, matches; `features.py:256` |
| `cp.std(x, axis)` | `torch.std(x, dim, unbiased=False)` | **ddof FLAG** — cupy is population; torch defaults unbiased |
| `cp.mean/sum/max/min` | `torch.mean/sum/max/min` | `torch.max(t)` → scalar tensor |
| `cp.any/all/isnan` | `torch.any/all/isnan` | |
| `cp.pi` | `math.pi` | `features.py:377` |
| `cp.isscalar(axis)` | `np.isscalar(axis)` | axis is a py int/list |
| `cp.cuda.device…MaxSharedMemoryPerBlock` | **delete** | only for CUDA shared-mem choice |
| `cupy.random.RandomState` | `TorchRandomState` | §3 |
| `.normal/.uniform/.randint/.choice` | see `TorchRandomState` | `choice` → `randperm[:size]` |
| **`cp.interp`** | **no equivalent** | §5.2 |
| **`cp.apply_along_axis`** | **no equivalent** | §5.2 |

### 5.1 `multi_cumsum` pad (utils.py:63)
torch `pad` wants a flat tuple in **reverse** axis order:
```python
pad_flat = []
for ax in reversed(range(ndim)):
    pad_flat += [1, 0] if ax in axis else [0, 0]
M = torch.nn.functional.pad(M, pad_flat)
```
Rest of `multi_cumsum` (per-axis `cumsum`, exclusive slice) maps directly; keep `np.isscalar(axis)`.

### 5.2 No-equivalent functions (preprocessing.py only)
- `cp.interp` (SequenceTabulator:59, SequenceAugmentor:134) and `cp.apply_along_axis` (:135). Pragmatic port: **interpolate on CPU with `np.interp`/`np.apply_along_axis`, then `as_tensor` back** (the tabulator already operates per-sample in Python comprehensions over possibly-ragged lists). Higher-effort device-native option: a `_interp1d(x, xp, fp)` via `torch.searchsorted` + linear blend, applied with `torch.vmap`. Recommend the CPU-numpy path first; document the choice.
- Resolve `xp = cp if isinstance(.., ArrayOnGPU) else np` branches to: operate in numpy if input is numpy, else torch via `isinstance(x, torch.Tensor)`.

---

## 6. `algorithms.py` full/low-rank functions (non-CUDA)

`signature_kern_first_order` / `signature_kern_higher_order` / `signature_kern_*_low_rank`: pure array ops — translate `cp.diff→torch.diff`, `cp.copy→.clone`, `cp.empty→torch.empty(..., device=)`, `cp.sum(axis=)→torch.sum(dim=)`, `cp.stack/concatenate→torch.stack/cat`, `cp.ones→torch.ones(..., device=)`. `multi_cumsum` comes from `utils`. The higher-order `R_next = torch.empty((d,d)+M.shape, …)` etc. are direct. No algorithmic change.

---

## 7. Static kernels, features, projections

- **`static/kernels.py`:** translate `Kernel.__call__` (`as_tensor`/`to_numpy`), each `_K`/`_Kdiag` (`clamp(min=eps)`, `torch.pow`, `torch.sqrt`, `torch.full((X.shape[0],),1.0,dtype=…,device=…)`). `squared_euclid_dist`/`matrix_mult` come from `utils`.
- **`static/features.py`:** `RandomFourierFeatures`, `RandomFourierFeatures1D` (RNG `normal`/`uniform`, `torch.cat`, `math.pi`, `/sqrt(n_components)`), `NystroemFeatures` (`torch.linalg.eigh`, `choice→randperm`, `robust_nonzero` mask). `KernelFeatures.transform` uses `as_tensor`/`to_numpy`.
- **`projections.py`:** base `transform` (`as_tensor`/`to_numpy`), every `_make_projection_components` (RNG), `_project`/`_project_outer_prod` (`index_select`, `nonzero`, `einsum`, `clone`). `TensorSketch` uses `utils.convolve_fft` (`torch.fft`). `compute_count_sketch` einsum unchanged.

---

## 8. Models layer (cuML → optional, sklearn fallback)

- **`pre_lin_svc.py`:** make the cuML import lazy/optional:
  ```python
  try:
      from cuml.svm import LinearSVC as LinearSVCOnGPU; _HAS_CUML = True
  except ImportError:
      LinearSVCOnGPU = None; _HAS_CUML = False
  ```
  In `_get_svc_model`, if `on_gpu and not _HAS_CUML` → **warn + fall back to `sklearn.svm.LinearSVC`** so the same script runs on XPU/MPS/CPU. cuML consumes cupy/numba arrays, not torch tensors — so on non-cuML backends always hand sklearn a **numpy** feature matrix via `to_numpy`. Replace `ArrayOnGPU`/`ArrayOnCPU` checks with `isinstance(x, torch.Tensor)` / `np.ndarray`.
- **`pre_svc.py`:** `SVC(kernel='precomputed')` already wants a numpy kernel matrix — just swap the two `cp.asnumpy` (L91, L100) for `to_numpy`; drop the cupy import.
- **`pre_base.py`:** drop cupy; `cp.concatenate(feature_lst)→torch.cat(feature_lst, dim=0)`; the `cp.asnumpy`/`ArrayOnGPU` guards (L200–203, L219–221, L279–281) → `isinstance(_, torch.Tensor)` + `to_numpy`. `prod_mat`/`kernel_mat` stay numpy host arrays (feed sklearn). `GridSearchCV`/`StratifiedKFold` unchanged.

---

## 9. Packaging & `__init__.py`

- **`setup.py`:** replace `'cupy>=12.2.0'` with `'torch>=2.5'`; relax the hard `numpy==1.24.4` pin (let torch pull a compatible numpy, e.g. `'numpy>=1.24'`); move cuML to `extras_require={'cuda-svm': ['cuml']}`; update keywords (`cupy`→`torch`). Optionally `extras_require={'xpu-sycl': []}` documented in README.
- **`ksig/__init__.py`:** add `from . import torch_backend` and optionally re-export `set_default_device`. Guard an optional `import intel_extension_for_pytorch` in a `try/except` for older Aurora stacks.

---

## 10. Aurora-XPU native SYCL fast-path (full implementation guide)

> **Target hardware/software (authoritative):** Aurora = **Intel Data Center GPU Max 1550 (Ponte Vecchio / PVC)** on the **Level Zero** backend, programmed with **Intel oneAPI DPC++ (`icpx -fsycl`)**. The default Aurora environment already loads `oneapi/eng-compiler/...` + `intel_compute_runtime` + `mpich`. Build SYCL with plain `-fsycl` (the default SYCL target is Intel/Level-Zero — **do NOT** use the `-fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80` flags; those are for *Polaris* (NVIDIA A100), a different ALCF machine). CMake: `find_package(IntelSYCL REQUIRED)` (see `$CMPLR_ROOT/lib/cmake/IntelSYCL/IntelSYCLConfig.cmake`). PVC has `aspect::fp64`; query `aspect::atomic64` before using fp64 atomics.

### 10.1 When is native SYCL actually worth it?

The portable torch wavefront (§4) runs correctly on XPU and is usually adequate (it dispatches through oneDNN/oneMKL; `torch.compile`+Triton also works on PVC). Native SYCL is justified by **two specific wins**:

1. **Launch overhead:** the torch wavefront does `l_X+l_Y-1` eager kernel launches. A single SYCL kernel does the whole DP in **one launch** with `n_X·n_Y` work-groups (one per sequence pair, mirroring the Numba design), holding the trailing antidiagonals in **device local memory** (`local_accessor`) with `group_barrier` between sweeps — exactly the pattern PVC's local memory (≥ guaranteed 32 KB, larger in practice) + many Xe-cores reward.
2. **Memory — the decisive argument:** the README calls out the O(N²·L²) bottleneck. Both the CuPy original *and* the torch wavefront **materialize** `M = [n_X,n_Y,l_X,l_Y]`. A **fused** SYCL kernel computes the static-kernel entry `M[x,y,i,j]` (the increment `⟨ẋ_i, ẏ_j⟩` / RBF distance) **on the fly inside the DP**, reading only the two raw sequences `X[x]`, `Y[y]` — so it **never stores the 4-D tensor**. Working memory drops from O(N²L²) to O(N²) (result) + O(N·L·d) (inputs), enabling far larger problems on one GPU. This is what dedicated SigPDE libraries do.

**Recommendation (staged):** ship the portable torch wavefront first; add the SYCL fast-path as an **optional, profile-gated** path dispatched when `device.type == "xpu"` and the extension built. Implement the **non-fused** kernels first (1:1 port of the Numba kernels — lowest risk), then the **fused** RBF/linear variant (§10.5) as the high-value follow-up. The torch wavefront stays as the always-available fallback **and** the numerical oracle.

### 10.2 The torch ↔ SYCL bridge (critical correctness point)

A torch **XPU tensor's `tensor.data_ptr<scalar_t>()` is a USM device pointer** (`malloc_device`-class) — pass it straight into a SYCL kernel as a raw pointer (the USM pointer model is 1:1 with how CuPy arrays were used). **The kernel MUST run on torch's own XPU queue**, not a fresh `sycl::queue{gpu_selector_v}`, or you get cross-queue ordering hazards against the surrounding torch ops. Obtain it from the current XPU stream:

```cpp
#include <c10/xpu/XPUStream.h>
sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();   // confirm signature vs installed torch
```
If a given torch build exposes this differently, the documented fallback is: create one `sycl::queue{xpu_device, {property::queue::in_order{}}}` cached per device, and bracket SYCL calls with `torch.xpu.synchronize()` on the Python side so torch and the SYCL queue can't race. Prefer the torch-queue route.

### 10.3 Build setup (`ksig/_sycl/loader.py` + `bindings.cpp`)

JIT build via `torch.utils.cpp_extension.load(..., sycl_sources=[...])` (or AOT `setup.py` with `SyclExtension`/`find_package(IntelSYCL)`). `.sycl` files are compiled by `icpx`; bindings live in a `.cpp` that declares C++ launchers (no bindings are generated for `.sycl` directly).

```python
# ksig/_sycl/loader.py
import os, functools, torch
from torch.utils.cpp_extension import load

@functools.lru_cache(maxsize=1)
def get_ext():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("SYCL fast-path requires an available XPU device.")
    here = os.path.dirname(__file__)
    return load(
        name="ksig_sycl",
        sources=[os.path.join(here, "bindings.cpp"),
                 os.path.join(here, "pde_kernels.sycl")],
        with_sycl=True,                       # detect .sycl -> compile with icpx
        extra_sycl_cflags=["-O3", "-fsycl"],  # NO nvptx targets on Aurora (Intel/L0 default)
        extra_cflags=["-O3"],
        verbose=False,
    )

def available():
    try: get_ext(); return True
    except Exception: return False
```
```cpp
// ksig/_sycl/bindings.cpp
#include <torch/extension.h>
at::Tensor sig_pde_launch(const at::Tensor& M, bool difference);
at::Tensor gak_log_launch(const at::Tensor& M);
at::Tensor rws_dtw_launch(const at::Tensor& D, const at::Tensor& warp_lens);
at::Tensor sig_pde_rbf_launch(const at::Tensor& X, const at::Tensor& Y,
                              double bandwidth, bool difference);  // fused, §10.5
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sig_pde", &sig_pde_launch);
    m.def("gak_log", &gak_log_launch);
    m.def("rws_dtw", &rws_dtw_launch);
    m.def("sig_pde_rbf", &sig_pde_rbf_launch);
}
```
Dispatch shim in `algorithms.py`:
```python
def signature_kern_pde(M, difference=True):
    if M.device.type == "xpu":
        try:
            from ._sycl.loader import get_ext, available
            if available(): return get_ext().sig_pde(M.contiguous(), difference)
        except Exception: pass    # fall through to torch wavefront (§4.2)
    ... # torch wavefront
```

### 10.4 `ksig/_sycl/pde_kernels.sycl` — non-fused SigPDE (three rolling `local_accessor`s, role-rotation)

**Refinement over the Numba layout:** instead of the packed `3*k±offset` single buffer (the source of the `I==0/I==1` index gymnastics), keep **three separate `local_accessor`s** `prev2`/`prev`/`cur` indexed by the diagonal lane, and **rotate their roles** each sweep. This is the canonical SYCL antidiagonal idiom and removes the fragile offset arithmetic. Shown for SigPDE; **GAK** swaps the combine to `logM + logsumexp3(...)` with `-inf` borders / `0` corner-seed and the driver transform `M/(2-M)` then `log(clamp(·,_EPS))`; **RWS** swaps to `D + min3(...)` with `+inf` borders / `0` seed, indexes `D` by `warp_segments[y]`, and reads the terminal at the per-series length `lY(y)` (see §4.3/§4.4).

```cpp
#include <torch/types.h>
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
using namespace sycl;

template <typename scalar_t>
static void sig_pde_impl(queue& q, const scalar_t* M, scalar_t* K,
                         int64_t nX, int64_t nY, int64_t lX, int64_t lY) {
  const int64_t D = lX + lY - 1;                 // number of antidiagonals
  const int64_t Ld = std::min(lX, lY) + 1;       // max lane count on any diagonal (+pad)
  const int wg = (int)std::min<int64_t>(std::min(lX, lY), 1024);
  q.submit([&](handler& h) {
    // three diagonals in local memory; size Ld each. ALL must fit local_mem_size.
    local_accessor<scalar_t,1> prev2(range<1>(Ld), h);  // i+j = d-2
    local_accessor<scalar_t,1> prev (range<1>(Ld), h);  // i+j = d-1
    local_accessor<scalar_t,1> cur  (range<1>(Ld), h);  // i+j = d
    h.parallel_for(
      nd_range<3>(range<3>(nX, nY, wg), range<3>(1, 1, wg)),
      [=](nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {   // PVC: 16 or 32
        const int64_t x = it.get_global_id(0), y = it.get_global_id(1);
        const int     lid = (int)it.get_local_id(2);
        const int64_t base = (x * nY + y) * lX * lY;
        auto g = it.get_group();
        // local_accessor is uninitialized; zeroing not required because boundary
        // reads are guarded by (i==0)/(j==0) below.
        for (int64_t d = 0; d < D; ++d) {
          const int64_t i_lo = std::max<int64_t>(0, d - (lY - 1));
          const int64_t i_hi = std::min<int64_t>(d, lX - 1);
          for (int64_t i = i_lo + lid; i <= i_hi; i += wg) {   // lane owns cells on diag
            const int64_t j = d - i;
            const int     a = (int)(i - i_lo);                  // lane index on THIS diag
            const scalar_t m = M[base + i * lY + j];
            // neighbors: up=(i-1,j) & left=(i,j-1) live on diag d-1 (=prev);
            //            diag=(i-1,j-1) lives on diag d-2 (=prev2).
            // Map cell->lane on prev/prev2 via that diagonal's own i_lo.
            const int64_t i_lo1 = std::max<int64_t>(0, (d-1) - (lY-1));   // prev start row
            const int64_t i_lo2 = std::max<int64_t>(0, (d-2) - (lY-1));   // prev2 start row
            const scalar_t K01 = (i==0)        ? (scalar_t)1 : prev [(i-1) - i_lo1]; // up
            const scalar_t K10 = (j==0)        ? (scalar_t)1 : prev [ i    - i_lo1]; // left
            const scalar_t K00 = (i==0||j==0)  ? (scalar_t)1 : prev2[(i-1) - i_lo2]; // diag
            cur[a] = (K01 + K10) * (1 + (scalar_t)0.5*m + m*m/(scalar_t)12)
                   - K00 * (1 - m*m/(scalar_t)12);
          }
          group_barrier(g);                       // converged: every lane reaches it
          // rotate roles prev2 <- prev <- cur (cycle the three accessors)
          for (int64_t a = lid; a < Ld; a += wg) { prev2[a] = prev[a]; prev[a] = cur[a]; }
          group_barrier(g);
        }
        // terminal cell (lX-1,lY-1) is on the last diagonal; after the final rotate it
        // sits in prev at lane ((lX-1) - i_loLast). Compute once on the group leader.
        if (g.leader()) {
          const int64_t i_loLast = std::max<int64_t>(0, (D-1) - (lY-1));
          K[x * nY + y] = prev[(lX-1) - i_loLast];
        }
      });
  });
  // do NOT q.wait() if running on torch's stream — let torch order it; sync in Python.
}

at::Tensor sig_pde_launch(const at::Tensor& M_in, bool difference) {
  TORCH_CHECK(M_in.device().is_xpu(), "sig_pde expects an XPU tensor");
  TORCH_CHECK(c10::xpu::getCurrentXPUStream().queue().get_device().has(aspect::fp64)
              || M_in.scalar_type() == at::kFloat, "fp64 path needs aspect::fp64");
  at::Tensor M = M_in;
  bool is_diag = (M.dim() == 3);
  if (is_diag) M = M.unsqueeze(1);
  if (difference) M = at::diff(at::diff(M, 1, -2), 1, -1);
  M = M.contiguous();
  const int64_t nX=M.size(0), nY=M.size(1), lX=M.size(2), lY=M.size(3);
  at::Tensor K = at::empty({nX, nY}, M.options());
  queue& q = c10::xpu::getCurrentXPUStream().queue();
  AT_DISPATCH_FLOATING_TYPES(M.scalar_type(), "sig_pde", [&]{
    sig_pde_impl<scalar_t>(q, M.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                           nX, nY, lX, lY);
  });
  return is_diag ? K.squeeze(1) : K;
}
```
**Notes / gotchas (from the SYCL-2020 device-code rules):**
- `group_barrier`, `local_accessor`, sub-groups, and all group algorithms **require an `nd_range` launch** and must be hit in **converged** control flow (a barrier inside a divergent branch is UB). The `for (i = i_lo+lid; …; i += wg)` + barrier-outside structure above satisfies this.
- **No recursion / virtual calls / function pointers / exceptions / RTTI / `new`/`delete`** in device code; static-duration vars must be `const`/`constexpr`.
- **Local-memory budget:** `3 * Ld * sizeof(scalar_t)` must fit `info::device::local_mem_size`. For sequences too long, **tile** with `async_work_group_copy` carrying tile-boundary diagonals through a `malloc_device` scratch (§10.5), or fall back to the torch wavefront (mirror the Numba `shared_mem > max_shared_mem` guard).
- `wg ≤ info::device::max_work_group_size`; pin sub-group size with `[[sycl::reqd_sub_group_size(32)]]` (PVC supports 16/32 — query `info::device::sub_group_sizes`). These are *optional features*: an unsupported value throws `errc::kernel_not_supported`.
- **fp64** is available on PVC (`aspect::fp64`); keep `AT_DISPATCH_FLOATING_TYPES` (double + float). GAK/SigPDE want double for stability — fine here (unlike MPS).
- Use `sycl::fma`, `sycl::exp`, `sycl::fmax`, `sycl::isinf`/`isnan` from `sycl::`. Reserve `sycl::native::*` (faster, less accurate, float-only) for RFF feature maps — **never** the PDE/GAK solve.

### 10.5 Fused variant `sig_pde_rbf_launch` (breaks the O(N²L²) memory wall)

Takes raw sequences `X:[nX,lX,d]`, `Y:[nY,lY,d]` and a `bandwidth`; computes the increment `m = M[x,y,i,j]` **inside** the DP loop instead of reading a materialized 4-D tensor. For SigPDE the cell needs the lifted-increment kernel value at `(i,j)`; for an RBF static kernel that is a function of `‖x_i − y_j‖²` (with `difference` taking second differences of the kernel grid — implement the increment form used by the existing `_compute_embedding`/`signature_kern_pde` pipeline so values match the non-fused path exactly).

Key SYCL specifics for the d-dimensional inner work:
- **Do NOT use `sycl::dot`** — it is geometric and limited to **2–4 dims**. For the d-dim increment dot/distance, either loop with `sycl::fma` over `d`, or use a **sub-group reduction** `reduce_over_group(it.get_sub_group(), partial, plus<>())` (SYCL 2020 group reductions accept only `sycl::plus/multiplies/minimum/maximum<>`). All group/sub-group functions are barriers — keep them converged.
- Hold the per-lane increment vector in **`marray<scalar_t, D>`** (fixed-size, `std::array` layout, device-copyable); reserve `vec`/swizzles for ≤16-wide SIMD only.
- **Tiling long sequences:** stage `X[x]`/`Y[y]` row-blocks from global → local with `it.async_work_group_copy(local_ptr, global_ptr, len)` + `it.wait_for(e)`; carry tile-boundary diagonals through a `malloc_device` scratch between tiles.
- **Accumulator option:** if a variant sums partial contributions into `K[x,y]`, use `atomic_ref<scalar_t, memory_order::relaxed, memory_scope::device>::fetch_add` — but **fp64 atomics need `aspect::atomic64`**; if absent, reduce per-group in local memory and have `g.leader()` write once (the SigPDE/GAK/RWS kernels already write once per group, so atomics are only needed for a future reduction-style variant).
- Use an **in-order queue** (`property::queue::in_order`) when chaining stages without manual events; torch's stream queue is already ordered.

Validate `sig_pde_rbf` against `sig_pde` (non-fused) on identical inputs within the f64 band, then confirm it runs at `(nX·nY·lX·lY)` sizes that OOM the materialized path.

### 10.6 Alternative programming models (noted, not recommended for the DP core)
- **OpenMP target offload** is available on Aurora and is a fine model for the *embarrassingly-parallel* parts, but it maps awkwardly to the antidiagonal wavefront (no clean equivalent of `local_accessor` + `group_barrier` rolling buffers). Keep the DP core in SYCL.
- **oneMath** (`oneapi::math::blas::*gemm`) is available, but the dense BLAS/elementwise layer already goes through torch's oneMKL/oneDNN backend on XPU — no hand-written GEMM needed.

### 10.7 Optional CUDA parity
The same algorithm has a CUDA twin (the existing Numba kernels). If NVIDIA peak performance matters later, provide it as a `.cu` `CUDAExtension` dispatched on `device.type=="cuda"`. Out of scope for the initial port (the torch wavefront already covers CUDA correctly).

---

## 11. Numerical-equivalence test plan (`tests/`)

**Strategy:** on a machine with the original CuPy/cuML install, freeze reference outputs to `.npz` (numpy, **float64**) for fixed seeds and small inputs, per public entry point; the torch port is tested against these `.npz` on any backend (decouples tests from needing CuPy present). Where CuPy is unavailable, validate the three DP kernels against **brute-force numpy DP oracles** written directly from the recurrences in §4 — implement these oracles *first*.

**Public surface to cover:** all `static.kernels` (`Linear/Polynomial/RBF/Matern12/32/52/RationalQuadratic`, `_K`+`_Kdiag`); `static.features` (`RandomFourierFeatures`, `…1D`, `Nystroem`); `projections` (`Gaussian/Subsampling/VerySparse/TensorSketch/TensorizedRandom/Diagonal`); `SignatureKernel` (order 1 & higher, normalized/unnormalized); `SignaturePDEKernel`; `GlobalAlignmentKernel`; `RandomWarpingSeries` (incl. `l_Y=1` and unequal warp lengths); `SignatureFeatures`; `preprocessing` (ragged/NaN/over-max-len interp paths); `models` (sklearn-fallback fit/predict accuracy).

**Tolerances:**
- float64 (CUDA/XPU/CPU): algebraic kernels `rtol=1e-10, atol=1e-12`; SigPDE/GAK `rtol=1e-8, atol=1e-10`; DTW exact `rtol=1e-12`.
- float32 / MPS: DP kernels & RFF `rtol=1e-3, atol=1e-4`; pure algebra `rtol=1e-5`. `skipif`/loosen strict-f64 assertions on MPS by design.
- SYCL path: assert agreement with the torch wavefront within the f64 band (it computes the same recurrence).

**RNG caveat:** CuPy `RandomState` and `torch.Generator` produce **different** streams for the same seed. Do **not** compare raw random matrices. Either (a) inject a numpy-seeded weight matrix into both code paths via a test hook and compare the *math*, or (b) compare the *kernel/Gram matrix* the randomized features approximate against the exact kernel within a Monte-Carlo band.

**Device matrix:** parametrize `device ∈ {cpu}` always + `{cuda, xpu, mps}` when available (auto-skip). Assert cross-device agreement (cpu-f64 vs cuda/xpu-f64 tight; cpu-f64 vs mps-f32 loose).

---

## 12. File-by-file checklist (implementation order)

1. **`ksig/torch_backend.py`** (NEW) — §3. Unit-test device/dtype/RNG in isolation.
2. **`ksig/utils.py`** — array-API + `multi_cumsum` pad (§5.1) + re-export `check_random_state`/`RandomStateOrSeed`/`ArrayOnGPU=torch.Tensor`.
3. **`ksig/static/kernels.py`** — §7.
4. **`ksig/projections.py`** — §7.
5. **`ksig/static/features.py`** — §7.
6. **`ksig/algorithms.py`** — (a) full/low-rank functions (§6); (b) **replace the 3 Numba kernels with the torch wavefronts** (§4); add XPU SYCL dispatch shims (§10.2). Validate against numpy DP oracles. *Highest risk.*
7. **`ksig/preprocessing.py`** — §5.2 (interp), `repeat_interleave`, `std(unbiased=False)`.
8. **`ksig/kernels.py`** — all public kernel classes; `torch.full((n,),1.0,…)`, `clamp`, `exp`, `mean`; `RandomWarpingSeries._make_feature_components` (`std(unbiased=False)` FLAG, `randint`, `normal(dtype=X.dtype)`, `int(sum(...))`).
9. **`ksig/models/pre_base.py`** → **`pre_svc.py`** → **`pre_lin_svc.py`** — §8.
10. **`setup.py`**, **`ksig/__init__.py`** — §9.
11. **`tests/`** (NEW) — §11; write the brute-force DP oracles before step 6b if possible.
12. **`ksig/_sycl/`** (NEW, optional) — §10.3/10.4 non-fused kernels, then the fused RBF-SigPDE variant; gate behind `device.type=="xpu"` + `available()`.
13. **`docs/TORCH_PORT.md`** — commit this plan into the repo.

---

## 13. Verification (end-to-end)

1. **Smoke test, CPU:** run the README quick-start (`SignatureKernel` on random `X`) with `set_default_device('cpu')`; assert shapes `(10,10)`, `(10,)`, `(10,8)` and finite values.
2. **Oracle test:** `pytest tests/` — SigPDE/GAK/RWS torch wavefronts vs numpy DP oracles (tiny hand-built inputs), plus all public classes vs frozen `.npz` references.
3. **Device sweep:** rerun the suite on each available backend (`cpu`, and `cuda`/`xpu`/`mps` where present) via device parametrization; confirm cross-device tolerances.
4. **Aurora SYCL:** on an Aurora compute node with the default `oneapi/eng-compiler` + `intel_compute_runtime` modules loaded (and the XPU torch build), confirm `torch.xpu.is_available()` and `aspect::fp64`; build `ksig._sycl` (`loader.available()` → True, compiled with `icpx -fsycl`, **no** nvptx/sm_80 flags); run SigPDE/GAK/RWS and assert agreement with the torch wavefront within the f64 band; benchmark vs the wavefront on a large `n_X·n_Y` kernel matrix to confirm the launch/memory win; then validate the fused `sig_pde_rbf` against the non-fused path and confirm it runs at sequence/batch sizes that OOM the materialized path. Verify torch↔SYCL ordering by interleaving SYCL calls with torch ops on the same XPU stream (no stale-data races).
5. **Models:** fit/predict `PrecomputedKernelSVC` and `PrecomputedFeatureLinSVC` (sklearn fallback) on a small labeled set; assert accuracy matches the reference.
