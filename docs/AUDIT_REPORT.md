# Paper-to-Code Audit Report

Audit date: 2026-06-11

Scope: `docs/SIGNATURE_KERNELS.md`, `ksig/generalized.py`, `ksig/algorithms.py`,
`ksig/kernels.py`, `ksig/static/features.py`, `ksig/projections.py`,
`ksig/utils.py`, `ksig/torch_backend.py`, the paper-derived specs under
`paper_specs/`, and the original KSig reference in
`paper_specs/KSig_original_repo/ksig.txt`.

## 1. Executive Summary

Overall verdict: **MIXED**.

The torch port is close to the original KSig implementation for the legacy dense
truncated recursion, legacy per-level normalization, random Fourier features,
and the second-order antidiagonal PDE stencil. The strongest matches are:

- `ksig/algorithms.py:58-96` mirrors original KSig
  `signature_kern_first_order` at
  `paper_specs/KSig_original_repo/ksig.txt:2929-2967`.
- `ksig/algorithms.py:414-454` mirrors the original second-order PDE recurrence
  at `paper_specs/KSig_original_repo/ksig.txt:3226-3328`.
- `ksig/kernels.py:181-215` intentionally preserves legacy
  per-level-normalize-then-average behavior from
  `paper_specs/KSig_original_repo/ksig.txt:2477-2511`.
- `ksig/static/features.py:289-340` preserves the original RFF map shape and
  scaling from `paper_specs/KSig_original_repo/ksig.txt:5073-5124`.

The highest-risk discrepancies are:

- `truncation=0` / `n_levels=0` is mathematically wrong: the first-order
  recursion still includes level 1 and the weighted wrapper broadcasts
  `phi(0)` onto both level 0 and level 1.
- The `GeneralSignatureKernel` facade drops `diag=True` for delegated legacy
  engines (`sig-PDE`, `sig-EXACT`).
- The generalized kernels force NumPy conversion and float32 in `_static_block`
  and `_diag_block`, and use `torch.no_grad()` in evaluation, so they do not
  satisfy the paper/test-contract differentiability and dtype-preservation
  requirements with respect to input paths.
- Random-feature level independence is not guaranteed. `SignatureFeatures`
  clones static feature/projection estimators using the same parameter set,
  so integer or default RNG seeds can produce identical weights across levels,
  violating the RFSF unbiasedness contract.
- Several paper-4 algorithms are not implemented as specified:
  simultaneous low-rank Gram construction and string-kernel sequentialization
  are absent.

The implementation is **paper-inspired with faithful ports for a subset**:
the KSig port is largely faithful to the original reference, but the broader
paper suite is only partially implemented and the learnable/general facade has
important edge-case and autograd/API gaps. Current tests are insufficient for a
paper-contract claim. They cover many port-level recurrences and the new facade,
but they do not cover `truncation=0`, generalized input gradients, RFSF
unbiasedness/independence, very sparse projection sparsity, paper-4 low-rank
matrix construction, or string-kernel sequentialization.

## 2. Repository/Code Map

| Algorithm / concept | Paper spec source | Claimed implementation | Actual code locations | Confidence |
|---|---|---|---|---|
| Truncated signature kernel, first order | `paper_1/algorithm_cards/truncated_signature_kernel.md`; `paper_3/algorithm_cards/truncated_signature_kernel.md`; KSig reference | Legacy KSig port plus `WeightedSignatureKernel` wrapper | `ksig/algorithms.py:33-96`, `ksig/kernels.py:98-215`, `ksig/generalized.py:126-215` | High for `N>=1`; low for `N=0` |
| Higher-order dense sequential/signature kernel | `paper_4/algorithm_card/higher_order_sequential_kernel.md` | `order>1` branch of dense recurrence | `ksig/algorithms.py:99-152`; `ksig/kernels.py:101-164` | Medium |
| Goursat PDE untruncated kernel | `paper_1/algorithm_cards/untruncated_signature_pde_kernel.md`; `paper_2/algorithm_cards/base_signature_kernel_pde.md`; KSig reference | Legacy `SignaturePDEKernel` and generalized wavefront | `ksig/algorithms.py:414-454`, `ksig/kernels.py:219-257`, `ksig/generalized.py:85-101` | High for KSig stencil; medium for paper PDE contract |
| General signature facade | `docs/SIGNATURE_KERNELS.md` | Six kernel configurations under `(phi, truncation, normalization)` | `ksig/generalized.py:330-433` | Medium |
| Truncated learned `phi` | `paper_2/algorithm_cards/truncated_phi_signature_kernel.md` | Nonnegative fitted level weights, `phi(0)=1` pinned | `ksig/generalized.py:126-215` | Medium |
| Dilation-mixture learned `phi` | `paper_2/algorithm_cards/randomised_phi_kernel_quadrature.md`; `integral_transform_phi_kernel.md` | Softmax weights over positive lambda grid | `ksig/generalized.py:219-326` | Medium |
| Fourier/integral transform phi kernels | `paper_2/algorithm_cards/fourier_phi_kernel.md`; `integral_transform_phi_kernel.md` | Not present as complex/measure transform APIs | No dedicated implementation found | Low / not implemented |
| RFF static kernel | `paper_3/algorithm_cards/rff_static_kernel.md` | RFF feature map for RBF | `ksig/static/features.py:289-340`, `345-398` | High for base RFF; low for gradient to bandwidth |
| RFSF / DP / TRP maps | `paper_3/algorithm_cards/rfsf_map.md`, `rfsf_dp_map.md`, `rfsf_trp_map.md` | Generic `SignatureFeatures` plus projection classes | `ksig/kernels.py:264-360`, `ksig/projections.py:468-678` | Low to medium |
| Paper-4 low-rank pair kernel | `paper_4/algorithm_card/lowrank_sequential_kernel.md` | Low-rank signature features exist, but not Algorithm 5 factors | `ksig/algorithms.py:159-353`, `ksig/kernels.py:264-360` | Low |
| Simultaneous low-rank Gram matrix | `paper_4/algorithm_card/simultaneous_lowrank_kernel_matrix.md` | No dedicated low-rank Gram factor API found | No implementation found | High confidence not implemented |
| String-kernel sequentialization | `paper_4/algorithm_card/string_kernel_sequentialization.md` | No string kernel API found | No implementation found | High confidence not implemented |
| Cumulative-sum primitive | `paper_4/algorithm_card/cumulative_sum_subroutine.md` | Inclusive cumsum plus exclusive shift for strict order | `ksig/utils.py:45-83` | High |
| SYCL fast path | `docs/TORCH_PORT.md`, `docs/SYCL_HANDOFF.md` | Optional XPU native dispatch | `ksig/algorithms.py:373-407`, `ksig/_sycl/*` | Not runtime-tested here |

## 3. Paper-to-Code Traceability Table

| Claim | Source | Expected behavior | Actual code behavior | Status | Evidence | Notes |
|---|---|---|---|---|---|---|
| Truncation includes level 0 through level N | `paper_2/algorithm_cards/truncated_phi_signature_kernel.md:30-40`; `paper_3/implementation_contracts.md:5-8` | `N=0` returns `phi(0)` only | `signature_kern_first_order` always seeds `[K0, K1]` before looping; `WeightedSignatureKernel` broadcasts length-1 `phi` over both | FAIL | `ksig/algorithms.py:83-96`; `ksig/generalized.py:136-166`, `200-214`; diagnostic in section 5 | Confirmed edge-case bug |
| For `N>=1`, first-order recursion uses second-differenced static Gram and exclusive cumsums | KSig reference; paper 4 cumsum card | `R_k = M * cumsum_exclusive(R_{k-1})`, strict in both axes | Matches reference | PASS | `ksig/algorithms.py:74-96`; original `ksig.txt:2945-2967`; `ksig/utils.py:45-83` | Port-faithful |
| Legacy per-level normalization is not normalize-once | `docs/SIGNATURE_KERNELS.md:88-101` | Each level normalized by its own diagonal, then averaged | Code does this when `return_levels=True` and `K.ndim==3` | PASS | `ksig/kernels.py:191-215`; original `ksig.txt:2487-2511` | This is `sig-EXACT`, not a global `phi` kernel |
| `sig-EXACT` is outside normalized global-`phi` family | `docs/SIGNATURE_KERNELS.md:217-224` | Treat as legacy normalization probe | Code delegates to `SignatureKernel(normalize=True)` | PASS | `ksig/generalized.py:379-385`; `ksig/kernels.py:212-215` | Documentation later contradicts itself; see F9 |
| `sig-PDE` is untruncated and PDE-based | `docs/SIGNATURE_KERNELS.md:151-170`; paper 1 PDE card | No truncation parameter; Goursat boundary 1 | Delegates to `SignaturePDEKernel` and `signature_kern_pde` | PASS | `ksig/generalized.py:386-391`; `ksig/kernels.py:219-257`; `ksig/algorithms.py:414-454` | Stencil is second-order KSig stencil, not the first-order explicit stencil in one card |
| PDE boundaries are 1 | Paper 1 and paper 2 PDE cards | First row/column equal 1 | Padded `H` initialized to ones | PASS | `ksig/algorithms.py:443-452`; `ksig/generalized.py:91-101` | Length-1 behavior covered in tests |
| PDE wavefront is autograd-safe for learned lambda | `docs/SIGNATURE_KERNELS.md:82-85` | Out-of-place recurrence for `learn_lam=True` | `sigpde_wavefront` uses `index_put` into a new tensor each antidiagonal | PASS for lambda graph | `ksig/generalized.py:85-101`, `245-255`, `289-296` | Input-path graph is still broken by `_static_block` |
| General facade passes the KSig call contract including `diag` | `ksig/generalized.py` docstring lines 38-39 | `__call__(diag=True)` returns diagonal | Delegated legacy paths ignore `diag` | FAIL | `ksig/generalized.py:430-433` | Affects `sig-PDE` and `sig-EXACT` |
| Generalized kernels preserve dtype/device and gradients | Paper 3 contract 11; paper 2 PDE gradient tests | No NumPy hops or forced dtype on differentiable values | `_static_block` and `_diag_block` use `np.asarray` and force `torch.float32`; `__call__` uses `torch.no_grad()` | FAIL | `ksig/generalized.py:51-68`, `200-215`, `305-326`; `paper_3/implementation_contracts.md:49-51` | High risk for learned/differentiable workflows |
| `phi(0)` is pinned for learned truncated `phi` | `docs/SIGNATURE_KERNELS.md:103-112` | `phi(0)=1`, no gradient | Code constructs `cat([1, softplus(theta[1:])])` | PASS | `ksig/generalized.py:150-156`, `184-193` | `freeze_phi0` option is unused; see F4 |
| Nonnegative learned `phi` preserves PSD | Paper 2 truncated phi card | `phi(k)>=0` | softplus for free weights; softmax for dilation weights; fixed phi checked nonnegative | PASS for parameterization | `ksig/generalized.py:147-156`, `248-255` | Numerical solver instability can still break values |
| `learn_lam=False` learns only weights on fixed lambda grid | `docs/SIGNATURE_KERNELS.md:191-204` | Precompute node Grams, optimize softmax weights | Code does this under `torch.no_grad()` | PASS | `ksig/generalized.py:257-288` | Stability warning return is ignored during fit |
| RBF-only untruncated guard | `docs/SIGNATURE_KERNELS.md:254-255`, `279-281` | Reject linear static for untruncated kernels | Facade rejects `truncation=None and static=="linear"` | PASS | `ksig/generalized.py:360-365` | Plain `SignaturePDEKernel` still accepts arbitrary static kernels if used directly |
| RFF map has dimension `2*d_tilde` and `1/sqrt(d_tilde)` scaling | Paper 3 RFF card | `[cos, sin]/sqrt(d_tilde)` | Code uses `[sin, cos]/sqrt(n_components)` | PASS for kernel, AMBIGUOUS for feature ordering | `ksig/static/features.py:319-340`; original `ksig.txt:5103-5124` | Order is swapped but inner product is identical |
| RFF weights are fixed after fit | Paper 3 RFF card | Sample once and reuse | Code stores `random_weights_` at fit | PASS | `ksig/static/features.py:310-340` | No gradient to bandwidth after fit |
| RFSF uses independent RFF weights per level | Paper 3 contracts 4, 8, 9 | `W^(1)..W^(M)` independent | `SignatureFeatures` clones estimators with identical constructor params; integer/default RNGs can repeat streams | FAIL / HIGH RISK | `ksig/kernels.py:302-327`; `ksig/torch_backend.py:132-139`; diagnostic in section 5 | Breaks unbiasedness |
| RFSF/TRP projections are standard normal, not spectral | Paper 3 contract 9 | TRP matrices `N(0,I)` | `TensorizedRandomProjection` draws normal components | PASS with seed caveat | `ksig/projections.py:541-585` | Scaling/independence need explicit tests |
| Very sparse projection is sparse | Projection docs/tests plan | Bernoulli support times Rademacher signs | Bernoulli matrix is overwritten by dense Rademacher matrix | FAIL | `ksig/projections.py:336-347` | Confirmed static bug |
| Paper-4 simultaneous low-rank Gram factor is implemented | Paper 4 Algorithm 6 | Return factor `U` with `K=U U^T` in `O(N L rho M)` | No such API found | NOT IMPLEMENTED | `paper_4/algorithm_card/simultaneous_lowrank_kernel_matrix.md:21-44`; code search | Generic feature maps are different |
| String-kernel sequentialization is implemented | Paper 4 string card | String inputs and gap-decay lambda | No string kernel API found | NOT IMPLEMENTED | `paper_4/algorithm_card/string_kernel_sequentialization.md:25-43`; code search | No direct claim in design doc |
| Existing tests cover paper contracts | User brief | Tests for invariants and random-feature contracts | `pytest` unavailable in environment; static test tree lacks many listed planned files | UNTESTED | `tests/TEST_PLAN.md:221-268`; actual `find tests` output | See section 5 |

## 4. Kernel-by-Kernel Audit

#### sig-L1

- Claimed object: Level-1-only kernel, `phi=e_1`, normalize-once.
- Actual object: `WeightedSignatureKernel` with `phi_fixed=[0,1,0,...]`.
- Dispatch path: `GeneralSignatureKernel.__init__` lines
  `397-405` after `phi=="level_one"`.
- Normalization: Once after level selection via `_normalize` at
  `ksig/generalized.py:213-214`.
- Level-zero handling: Correct for `truncation>=1`: `phi(0)=0`.
  Broken if `truncation=0`, because `phi_fixed[1]=1` would index out of range.
- Truncation semantics: Requires `truncation` at least 1 but does not validate it.
- PSD validity: PSD for valid `truncation>=1`; cosine normalization is diagonal
  congruence when diagonals are positive.
- Differentiability: Evaluation is inside `torch.no_grad()` and `_static_block`
  converts through NumPy, so no input gradients.
- Main risks: Missing validation for `truncation>=1`; no input-gradient support.
- Verdict: **MOSTLY PASS** for normal use, **FAIL** for edge-case validation and
  differentiability.

#### sig-TRUNC-phi1

- Claimed object: `sum_{k=0}^N K_k`, normalize-once.
- Actual object: `WeightedSignatureKernel` with `phi_fixed=ones(N+1)`.
- Dispatch path: `ksig/generalized.py:397-405`.
- Normalization: Once after summing level Grams.
- Level-zero handling: Correct for `N>=1`; incorrect for `N=0` because the
  underlying algorithm still returns `K1`.
- Truncation semantics: Code's public parameter behaves like paper truncation
  depth for `N>=1`, but not for `N=0`.
- PSD validity: PSD for nonnegative levels under normal numerical conditions.
- Differentiability: No input gradients through generalized wrapper.
- Main risks: Off-by-one edge case; dtype forced to float32; no input gradients.
- Verdict: **MIXED**.

#### sig-PDE

- Claimed object: Untruncated ordinary signature kernel via Goursat PDE,
  normalize-once.
- Actual object: `ksig.kernels.SignaturePDEKernel` with RBF static kernel.
- Dispatch path: `ksig/generalized.py:386-391` to
  `ksig/kernels.py:219-257` to `ksig/algorithms.py:414-454`.
- Normalization: Once by legacy `SignatureKernel._K`; no level axis is present.
- Level-zero handling: Boundary table initialized to one.
- Truncation semantics: No truncation parameter.
- PSD validity: Mathematically expected; numerically unguarded for unstable cell
  magnitudes.
- Differentiability: The legacy torch recurrence is differentiable in principle,
  but `GeneralSignatureKernel.__call__` does not pass `diag`, and runtime testing
  was blocked in this environment.
- Main risks: No stability gate on plain `sig-PDE`; direct `SignaturePDEKernel`
  accepts non-RBF statics; `diag=True` facade bug.
- Verdict: **MOSTLY PASS** as a KSig port, **MIXED** against full PDE contract.

#### sig-Wphi

- Claimed object: Truncated `K_phi=sum phi(k)K_k`, `phi(0)=1`,
  `phi(k>=1)=softplus(theta)`, normalize-once.
- Actual object: Matches for `N>=1`.
- Dispatch path: `ksig/generalized.py:397-405`, fitted by
  `ksig/generalized.py:408-417` and `169-194`.
- Normalization: Once after mixing.
- Level-zero handling: Pinned to one in fitted mode.
- Truncation semantics: Same `N=0` bug as `sig-TRUNC-phi1`.
- PSD validity: Nonnegative `phi` preserves PSD if level kernels are PSD.
- Differentiability: Gradients flow to `theta`, not to input paths.
- Main risks: `freeze_phi0` argument ignored; input-gradient and dtype issues;
  no protection from degenerate level diagonals beyond clamp.
- Verdict: **MIXED**.

#### sig-PDEphi

- Claimed object: Untruncated dilation mixture
  `sum_i w_i SigPDE(lambda_i M)`, with `phi(k)=sum_i w_i lambda_i^k`.
- Actual object: Matches the positive real dilation-mixture subset.
- Dispatch path: `ksig/generalized.py:392-395` to
  `LearnedPhiSignaturePDEKernel`.
- Normalization: Once after node mixture.
- Level-zero handling: `sum_i w_i=1` by softmax.
- Truncation semantics: Untruncated only.
- PSD validity: Conic mixture when nodes are finite and solver is stable.
- Differentiability: Weight and optional lambda gradients are available through
  `sigpde_wavefront`; input gradients are broken by `_static_block`.
- Main risks: Stability warning is ignored in `fit_phi`; `auto_clamp=True`
  changes evaluation lambdas but `phi()` still reports unclamped nodes; no
  complex/Fourier transform support.
- Verdict: **MIXED**.

#### sig-EXACT

- Claimed object: Legacy per-level-normalized average, not a global normalize-once
  `K_phi`.
- Actual object: Delegates to `SignatureKernel(normalize=True)`.
- Dispatch path: `ksig/generalized.py:379-385` to `ksig/kernels.py:181-215`.
- Normalization: Per-level whitening followed by mean over level axis.
- Level-zero handling: Included by the legacy `return_levels=True` path for
  `n_levels>=1`.
- Truncation semantics: Same `n_levels=0` unsupported/misleading edge as legacy.
- PSD validity: Average of normalized PSD level kernels, assuming nonzero
  diagonals and numerical clamps.
- Differentiability: Legacy path should be differentiable if called directly;
  facade ignores `diag=True`.
- Main risks: Documentation contradicts itself by sometimes saying all six
  kernels are the same global `K_phi` object; degenerate levels can dominate
  after per-level whitening.
- Verdict: **MOSTLY PASS** as legacy KSig behavior, **AMBIGUOUS** as a member of
  the claimed common skeleton.

## 5. Numerical and Mathematical Invariants

### Commands and Environment

Requested inventory commands were run:

- `pwd`: `/home/siebenschuh/Projects/KSig_aurora`
- `find paper_specs -maxdepth 4 -type f | sort`
- `find docs -maxdepth 2 -type f | sort`
- `find ksig -maxdepth 3 -type f | sort`
- `find tests -maxdepth 3 -type f | sort`

Existing tests could not be run in this environment:

- `pytest -q`: `/bin/bash: pytest: command not found`
- `python -m pytest -q`: `/bin/bash: python: command not found`
- `python3 -m pytest -q`: `/usr/bin/python3: No module named pytest`

Interpreter checks:

- `/usr/bin/python3` is Python 3.6.15 with NumPy 1.17.3, no torch, no sklearn,
  and cannot parse the repo because `from __future__ import annotations` is not
  available in Python 3.6.
- `/usr/bin/python3.10` is Python 3.10.10 with torch 2.8.0+cu128, but no NumPy,
  no sklearn, no pytest. Importing `ksig` fails because NumPy is absent.
- `setup.py:22-31` requires `numpy>=1.24`, `scikit-learn>=1.3.2`,
  `torch>=2.5`, and Python `>=3.9`, so neither available interpreter is a
  complete supported test environment.

### Actual Diagnostics Run

Torch-only recurrence diagnostic, copying the code structure from
`ksig/algorithms.py:58-96`, on a single increment `M=[[[[2.0]]]]`:

```text
N=0: levels_shape=(2, 1, 1) levels=[1.0, 2.0] sum=[3.0]
N=1: levels_shape=(2, 1, 1) levels=[1.0, 2.0] sum=[3.0]
N=2: levels_shape=(3, 1, 1) levels=[1.0, 2.0, 0.0] sum=[3.0]
N=3: levels_shape=(4, 1, 1) levels=[1.0, 2.0, 0.0, 0.0] sum=[3.0]
weighted_N0_broadcast_sum= [3.0]
```

Expected for `N=0` under the paper contracts: level 0 only, value `1.0`
or `phi(0)`. Actual: level 1 is included and the wrapper would sum to `3.0`.

Torch-only RNG diagnostic for independent level weights:

```text
seed None identical_two_new_generators True
seed 0 identical_two_new_generators True
seed 123 identical_two_new_generators True
shared_generator_sequential_identical False
```

This matters because `SignatureFeatures._make_feature_components` constructs
level copies from the same estimator params (`ksig/kernels.py:307-327`), while
`TorchRandomState` creates a fresh `torch.Generator` and only manually seeds it
when an integer seed is supplied (`ksig/torch_backend.py:132-139`). Fresh
generators with the same seed, including the default generator state, produce
identical streams. RFSF paper contracts require independent `W^(m)` by level.

### Invariant Assessment

- Constant path: likely passes for legacy dense/PDE paths when `N>=1`;
  generalized `N=0` fails by construction.
- One-step linear path: first-order `N>=1` matches expected `1 + M`; `N=0`
  fails.
- Level-zero-only behavior: **confirmed fail** for direct algorithm and
  generalized weighted wrapper.
- Symmetry: expected for dense and PDE Grams; covered in parts of the test tree,
  not run here.
- Positive diagonal: expected under PSD kernels; per-level degenerate diagonals
  are floored and can inject noise.
- PSD Gram matrix: likely for nonnegative finite weights; numerical PDE
  instability and sign-changing/unimplemented Fourier weights are not guarded.
- CPU/GPU or CPU/SYCL consistency: not tested here; tests exist for SYCL but
  require proper environment.
- dtype consistency: generalized code forces float32 in `_static_block` and
  `_diag_block`; legacy code uses `torch_backend.as_tensor` defaults.
- Gradient flow with respect to input paths: fails in generalized wrappers due
  to NumPy conversion and `torch.no_grad()`.
- Gradient flow with respect to `phi`: present for `WeightedSignatureKernel` fit.
- Gradient flow through PDE wavefront when `learn_lam=True`: present with
  respect to lambda/weights, not input paths.
- Batching consistency: legacy dense/PDE code is batched; no full runtime check.
- Padding/masking behavior: no variable-length mask found in these kernels;
  preprocessing may tabulate elsewhere, not audited deeply.
- RFF approximation sanity: base RFF map shape/scaling looks correct; RFSF
  independence is high risk.
- Normalization equivalence/non-equivalence: tests document once vs per-level
  difference (`tests/test_general_signature.py:39-56`), but not run here.

## 6. Discrepancies and Bugs

### Finding F1: `truncation=0` Includes Level 1

- Severity: High
- Status: Confirmed
- Location: `ksig/algorithms.py:83-96`; `ksig/generalized.py:136-166`,
  `200-214`
- Paper/spec expectation: `N=0` returns only the level-0 term:
  `paper_2/algorithm_cards/truncated_phi_signature_kernel.md:37-40` and
  `paper_3/implementation_contracts.md:5-8`.
- Actual code behavior: `signature_kern_first_order(..., n_levels=0,
  return_levels=True)` returns `[K0, K1]`. The weighted wrapper with a length-1
  `phi` broadcasts `phi(0)` over both levels.
- Why it matters: This is an off-by-one semantic error exactly at the
  level-zero paper contract. It can silently corrupt edge-case tests and any
  caller using `truncation=0` as a constant baseline.
- Minimal reproducer:
  `GeneralSignatureKernel(phi="const", truncation=0)` on a one-increment path
  should return an all-ones normalized Gram, but the underlying level stack
  contains level 1 too.
- Suggested fix: Special-case `n_levels == 0` in `signature_kern*` to return
  only `K0`; or reject zero at the facade and document that `truncation` must be
  positive. If paper-faithful semantics are desired, support zero.
- Suggested regression test:
  `test_truncated_phi_N0_returns_phi0_only`.

### Finding F2: Facade Drops `diag=True` for Delegated Engines

- Severity: Medium
- Status: Confirmed
- Location: `ksig/generalized.py:430-433`
- Paper/spec expectation: The KSig call contract includes `diag`.
- Actual code behavior: For `_kind == "ksig"`, the facade calls
  `self._engine(X, Y, return_on_gpu=return_on_gpu)` and never forwards `diag`.
- Why it matters: `GeneralSignatureKernel(phi="const", truncation=None,
  diag=True)` and `normalize="per_level"` return full Gram blocks instead of
  diagonals. This is an API false claim and can create expensive accidental
  `O(n^2)` calls.
- Minimal reproducer: `GeneralSignatureKernel(phi="const", truncation=None)(X,
  diag=True).shape` should be `(n,)`, but the delegated call path requests a full
  matrix.
- Suggested fix: Pass `diag=diag` in the `_kind=="ksig"` branch.
- Suggested regression test: `test_general_signature_delegated_diag_shape`.

### Finding F3: Generalized Kernels Break Input Gradients and Dtype Preservation

- Severity: High
- Status: Confirmed
- Location: `ksig/generalized.py:51-68`, `200-215`, `305-326`
- Paper/spec expectation: No `.detach()`, `.item()`, NumPy conversion, or
  in-place mutation on differentiable values unless justified
  (`paper_3/implementation_contracts.md:49-51`); PDE cards request gradient
  tests.
- Actual code behavior: `_static_block` and `_diag_block` do
  `torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)`, forcing
  host NumPy materialization and float32. Evaluation is wrapped in
  `torch.no_grad()`.
- Why it matters: The generalized facade cannot be used as a differentiable
  torch module with respect to input paths or bandwidth. It also discards
  float64 inputs, which the PDE specs recommend for stability.
- Minimal reproducer: Pass a torch tensor with `requires_grad=True` into
  `WeightedSignatureKernel`; the static block converts through NumPy and the
  returned kernel is outside the input graph.
- Suggested fix: Use `torch_backend.as_tensor` or direct tensor-preserving
  conversion, preserve dtype unless explicitly requested, and make `no_grad`
  optional for evaluation.
- Suggested regression test: `test_generalized_weighted_input_gradient` and
  `test_generalized_preserves_float64`.

### Finding F4: `freeze_phi0` Is a Dead Constructor Argument

- Severity: Medium
- Status: Confirmed
- Location: `ksig/generalized.py:351-353`, `374-405`
- Paper/spec expectation: The user-facing constructor exposes `freeze_phi0`,
  so it should either control behavior or be absent.
- Actual code behavior: `freeze_phi0` is accepted but never read. `phi(0)` is
  always pinned for free weights and structural for dilation.
- Why it matters: A caller can pass `freeze_phi0=False` and receive no error and
  no behavior change.
- Minimal reproducer: Construct `GeneralSignatureKernel(phi="free",
  freeze_phi0=False)` and inspect `phi_profile()[0]` after fitting; it remains
  pinned.
- Suggested fix: Remove the argument or implement a documented alternate
  parameterization.
- Suggested regression test: `test_freeze_phi0_false_rejected_or_effective`.

### Finding F5: PDE Stability Gates Are Incomplete

- Severity: High
- Status: Likely
- Location: `ksig/generalized.py:112-122`, `257-288`, `317-325`;
  `ksig/generalized.py:386-391`; `ksig/algorithms.py:414-454`
- Paper/spec expectation: Stability and finite-value behavior should be
  guarded; paper cards warn about large-scale paths and float32 accumulation.
- Actual code behavior: `LearnedPhiSignaturePDEKernel.fit_phi` calls
  `_stability_ok` but ignores the return and still optimizes on the computed
  Grams. Plain `sig-PDE` has no stability guard. Evaluation for `sig-PDEphi`
  returns NaN blocks only in some paths.
- Why it matters: Divergent PDE solves can silently enter training or
  downstream SVMs. Plain untruncated PDE is the highest-exposure path and has no
  guard.
- Minimal reproducer: Use small bandwidth or high-energy paths so
  `max(abs(M))` is large; `fit_phi` will warn but continue.
- Suggested fix: Centralize a finite/stability policy for all PDE paths:
  raise, auto-clamp, or return a structured failure. Use the gate during fit,
  not only evaluation.
- Suggested regression test: `test_sigpde_stability_guard_plain_and_learned`.

### Finding F6: RFSF Level Independence Can Be Violated by RNG Cloning

- Severity: High
- Status: Confirmed risk
- Location: `ksig/kernels.py:302-327`; `ksig/torch_backend.py:132-139`
- Paper/spec expectation: RFSF, DP, and TRP require independent RFF weights per
  level (`paper_3/implementation_contracts.md:17-23`).
- Actual code behavior: `SignatureFeatures` rebuilds each static feature copy
  from identical constructor params. With `random_state=None` or an integer,
  fresh `torch.Generator()` instances produce identical streams; the diagnostic
  confirmed this.
- Why it matters: Sharing `W` across levels makes the RFSF estimator biased,
  exactly one of the paper's highest-risk silent bugs.
- Minimal reproducer: Use `SignatureFeatures(static_features=RandomFourierFeatures(random_state=0),
  indep_feat=True)`, then compare `static_features_[0].random_weights_` and
  `static_features_[1].random_weights_` after fit.
- Suggested fix: Use one shared RNG object advanced across clones, or spawn
  child seeds deterministically (`seed+i`) and document the policy.
- Suggested regression test: `test_rfsf_independent_rff_weights_per_level`.

### Finding F7: `VerySparseRandomProjection` Is Not Sparse

- Severity: Medium
- Status: Confirmed
- Location: `ksig/projections.py:336-347`
- Paper/spec expectation: Very sparse projection should use a Bernoulli support
  and Rademacher signs.
- Actual code behavior: `components_full` is first assigned the Bernoulli
  matrix, then overwritten by a dense Rademacher matrix. The sparsity mask is
  lost while sparse scaling remains.
- Why it matters: The projection distribution and scaling are wrong, giving a
  dense signed projection mis-scaled as sparse.
- Minimal reproducer: Fit `VerySparseRandomProjection` and inspect the fraction
  of zeros in `components_`; it will be zero or near zero.
- Suggested fix: Multiply the Bernoulli support by the Rademacher sign matrix
  instead of overwriting it.
- Suggested regression test: `test_very_sparse_projection_realized_sparsity`.

### Finding F8: Paper-4 Low-Rank Matrix and String Algorithms Are Absent

- Severity: Medium
- Status: Confirmed
- Location: No implementation found for
  `paper_4/algorithm_card/simultaneous_lowrank_kernel_matrix.md` or
  `string_kernel_sequentialization.md`
- Paper/spec expectation: Algorithm 6 returns a low-rank factor `U` for the
  whole Gram matrix; string sequentialization supports gap decay and equality
  kernels.
- Actual code behavior: The repo has generic low-rank signature feature maps,
  but no simultaneous low-rank Gram factor API and no string kernel API.
- Why it matters: The paper-4 audit scope cannot be marked implemented.
- Minimal reproducer: Code search for string/gap-decay APIs and simultaneous
  low-rank factor output returns no candidates.
- Suggested fix: Document as not implemented, or add explicit APIs and tests.
- Suggested regression test: `test_simultaneous_lowrank_gram_matches_dense` and
  `test_string_kernel_bruteforce_short_strings`.

### Finding F9: `docs/SIGNATURE_KERNELS.md` Contradicts Itself and References a Missing File

- Severity: Medium
- Status: Confirmed
- Location: `docs/SIGNATURE_KERNELS.md:3-5`, `217-224`, `301-303`,
  `305-314`
- Paper/spec expectation: Documentation should state precisely which kernels
  are members of the normalize-once global `K_phi` family.
- Actual code behavior/documentation: Lines 3-5 and 305-314 say every kernel is
  the same object under `(phi, truncation, normalization)`, while lines 217-224
  correctly say `sig-EXACT` is not a member of the global `phi` family. The doc
  also references `LEARNABLE_PHI_SIGNATURE_KERNELS.md`, but no such file exists
  in this checkout.
- Why it matters: This is the central conceptual distinction of the facade.
  Users can incorrectly interpret `sig-EXACT` as a weighted signature kernel.
- Minimal reproducer: Read the cited lines and run `find . -iname '*LEARNABLE*'`.
- Suggested fix: Rephrase the skeleton claim: five kernels are normalize-once
  `K_phi` members, while `sig-EXACT` is a legacy per-level normalization probe.
  Add or remove the missing learnable-phi doc reference.
- Suggested regression test: Documentation lint or link-check test.

## 7. Ambiguities Requiring Author Decision

### Ambiguity A1: Which PDE Stencil Is the Contract?

- Where ambiguity appears: Paper 1 card gives a first-order explicit recurrence
  (`paper_1/algorithm_cards/untruncated_signature_pde_kernel.md:20-26`), while
  KSig and the port use the second-order correction
  `1 + m/2 + m^2/12`.
- Possible interpretations: The implementation is a faithful KSig PDE port, or
  it is intended to match the paper card's explicit scheme.
- Current code choice: KSig second-order stencil.
- Consequence: Tests should compare to the original KSig recurrence and analytic
  limits, not to the first-order recurrence cell-by-cell.
- Recommended author decision: Declare the second-order KSig stencil as the
  implementation contract and update paper-card cross-references accordingly.

### Ambiguity A2: Should `truncation=0` Be Public?

- Where ambiguity appears: Paper contracts require `N=0`, but legacy KSig
  validates `n_levels` as positive in `SignatureBase`.
- Possible interpretations: `truncation` is mathematical depth and may be zero;
  or it is a legacy positive "number of nonzero levels" parameter.
- Current code choice: The generalized wrapper accepts zero but mishandles it.
- Consequence: Silent wrong edge behavior.
- Recommended author decision: Either implement paper-faithful zero or reject it
  with `ValueError`.

### Ambiguity A3: Is Input Differentiability a Supported API?

- Where ambiguity appears: Paper/test contracts ask for gradients w.r.t. paths,
  while generalized wrappers are sklearn-style precomputed kernels.
- Possible interpretations: Only `phi`/lambda fitting needs gradients, or full
  differentiability should be preserved.
- Current code choice: `phi`/lambda gradients only; input gradients broken.
- Consequence: Paper gradient tests would fail for generalized wrappers.
- Recommended author decision: Document sklearn-only no-input-gradient behavior,
  or refactor generalized kernels to be differentiable torch modules.

### Ambiguity A4: Random Feature APIs Versus Paper-3 Named Algorithms

- Where ambiguity appears: `SignatureFeatures` plus projections can approximate
  signature features, but there are no explicit `RFSF`, `RFSF-DP`, or
  `RFSF-TRP` wrappers enforcing paper constraints.
- Possible interpretations: Generic components are sufficient, or named paper
  algorithms should be first-class.
- Current code choice: Generic components.
- Consequence: Users can build biased/non-paper configurations easily.
- Recommended author decision: Add named constructors that enforce independent
  RFF copies, DP/TRP dimensions, and projection distributions.

### Ambiguity A5: Normalization Naming

- Where ambiguity appears: `normalize="once"` and `normalize="per_level"` are
  both in the facade, but only one is a global normalized `K_phi`.
- Possible interpretations: Normalization is part of the common skeleton; or
  `per_level` is a legacy exception.
- Current code choice: `per_level` is a legacy exception.
- Consequence: Documentation can overstate commonality.
- Recommended author decision: Treat `sig-EXACT` as explicitly outside the
  normalize-once GSK family in all summaries.

## 8. Test Recommendations

### `tests/test_paper_contracts_truncated_signature.py`

- Test name: `test_truncation_zero_level_zero_only`
- Purpose: Enforce `N=0 => K0` for dense and weighted paths.
- Expected result: Unnormalized value is `phi(0)`; normalized diagonal is one.
- Minimal input: Two length-2 one-dimensional paths with nonzero increment.
- Failure mode caught: F1 off-by-one and broadcasting bug.

- Test name: `test_level_increment_difference_exact`
- Purpose: Check `K^{<=N} - K^{<=N-1} == K_N`.
- Expected result: Exact within dtype tolerance.
- Minimal input: Small RBF and linear static Grams.
- Failure mode caught: Level indexing and extraction errors.

- Test name: `test_general_signature_delegated_diag_shape`
- Purpose: Ensure `diag=True` reaches delegated legacy engines.
- Expected result: Shape `(n,)` for all six facade configurations.
- Minimal input: `n=3`, `L=4`, `d=2`.
- Failure mode caught: F2.

### `tests/test_paper_contracts_pde_signature.py`

- Test name: `test_sigpde_constant_path_is_one`
- Purpose: Goursat boundary and zero-increment behavior.
- Expected result: Full Gram of ones when either argument is constant.
- Minimal input: Constant `X`, random `Y`.
- Failure mode caught: Boundary/difference mistakes.

- Test name: `test_sigpde_stability_guard_plain_and_learned`
- Purpose: Uniform finite-value policy for `sig-PDE` and `sig-PDEphi`.
- Expected result: Raise, warn-and-NaN, or auto-clamp consistently.
- Minimal input: Small bandwidth/high-energy path.
- Failure mode caught: F5.

- Test name: `test_sigpde_wavefront_lambda_gradcheck`
- Purpose: Ensure `learn_lam=True` wavefront gradients are finite.
- Expected result: Finite lambda gradients and finite-difference agreement.
- Minimal input: `n=3`, `L=4`, `m_nodes=2`.
- Failure mode caught: Autograd breaks in wavefront.

### `tests/test_paper_contracts_learned_phi.py`

- Test name: `test_freeze_phi0_false_rejected_or_effective`
- Purpose: Remove dead API behavior.
- Expected result: Either `ValueError` or learned `phi(0)` changes.
- Minimal input: Small classification fit.
- Failure mode caught: F4.

- Test name: `test_generalized_weighted_input_gradient`
- Purpose: Decide and enforce differentiability contract.
- Expected result: Either finite `X.grad` or documented `NotImplementedError`.
- Minimal input: Torch tensor paths with `requires_grad=True`.
- Failure mode caught: F3.

- Test name: `test_auto_clamp_phi_profile_matches_evaluation_nodes`
- Purpose: Ensure reported `phi` matches effective lambdas.
- Expected result: `phi_profile` uses clamped lambdas when `auto_clamp=True`,
  or docs state otherwise.
- Minimal input: Path triggering auto-clamp.
- Failure mode caught: Misreported `phi`.

### `tests/test_paper_contracts_rff.py`

- Test name: `test_rff_self_kernel_one_and_shape`
- Purpose: Paper-3 RFF map sanity.
- Expected result: Feature dimension `2*d_tilde`, self-inner product one.
- Minimal input: Random `N x d`.
- Failure mode caught: RFF scaling or dimension errors.

- Test name: `test_rfsf_independent_rff_weights_per_level`
- Purpose: Enforce independent `W^(m)`.
- Expected result: Level weight matrices are not identical for int and default
  seeds, while runs remain reproducible.
- Minimal input: `SignatureFeatures` with RFF static features.
- Failure mode caught: F6.

- Test name: `test_very_sparse_projection_realized_sparsity`
- Purpose: Validate sparse projection distribution.
- Expected result: Zero fraction near `1 - prob_nonzero`.
- Minimal input: `n_features=1000`, `n_components=100`.
- Failure mode caught: F7.

### `tests/test_paper_contracts_sequential.py`

- Test name: `test_higher_order_D1_reduces_to_first_order`
- Purpose: Paper-4 Algorithm 4 anchor.
- Expected result: `order=1` and first-order dense recurrence match.
- Minimal input: Small second-differenced Gram.
- Failure mode caught: Higher-order repeat-count mistakes.

- Test name: `test_simultaneous_lowrank_gram_matches_dense`
- Purpose: Add/validate Algorithm 6 if implemented.
- Expected result: `U @ U.T` equals dense `SignatureKernel` for linear inputs.
- Minimal input: `N=4`, `L=5`, `d=2`, `M=3`.
- Failure mode caught: Missing or wrong simultaneous low-rank implementation.

- Test name: `test_string_kernel_bruteforce_short_strings`
- Purpose: Add/validate string sequentialization if implemented.
- Expected result: DP equals explicit strict-subsequence sum.
- Minimal input: Strings `"aba"`, `"baa"`, lambda values `{1.0, 0.5}`.
- Failure mode caught: Missing gap-decay/span/strictness behavior.

## 9. Suggested Patches

| Patch | File | Function | Change | Rationale | Risk |
|---|---|---|---|---|---|
| P1 | `ksig/algorithms.py` | `signature_kern_first_order`, `signature_kern_higher_order` | Add explicit `n_levels==0` branch returning only `K0`; validate negative values | Fix paper-level truncation semantics | Low if tested |
| P2 | `ksig/generalized.py` | `WeightedSignatureKernel.__init__`, `_levels`, `__call__` | Validate `n_levels>=0`; assert `len(phi)==level_count`; avoid broadcasting mismatches | Prevent silent level-weight bugs | Low |
| P3 | `ksig/generalized.py` | `GeneralSignatureKernel.__call__` | Forward `diag=diag` to delegated `_kind=="ksig"` engines | Restore KSig call contract | Low |
| P4 | `ksig/generalized.py` | `_static_block`, `_diag_block` | Replace `np.asarray` conversion with tensor-preserving conversion; preserve dtype/device | Satisfy dtype/autograd contracts | Medium |
| P5 | `ksig/generalized.py` | `__call__`, `fit_phi` methods | Make `torch.no_grad()` optional or only wrap public NumPy-returning evaluation | Allow differentiable use when requested | Medium |
| P6 | `ksig/generalized.py` | `GeneralSignatureKernel.__init__` | Remove `freeze_phi0` or implement it | Eliminate dead API | Low |
| P7 | `ksig/generalized.py`, `ksig/algorithms.py` | PDE call paths | Apply a shared stability/finite policy to plain PDE and learned PDE, including fitting | Prevent silent divergence | Medium |
| P8 | `ksig/kernels.py`, `ksig/torch_backend.py` | `SignatureFeatures._make_feature_components`; RNG helpers | Spawn independent child seeds or advance a shared RNG for per-level feature/projection clones | Restore RFSF unbiasedness | Medium |
| P9 | `ksig/projections.py` | `VerySparseRandomProjection._make_projection_components` | Use `bernoulli * rademacher` instead of overwriting support | Fix sparse distribution | Low |
| P10 | `docs/SIGNATURE_KERNELS.md` | Summary and skeleton sections | State that `sig-EXACT` is a legacy exception outside normalize-once global `K_phi`; fix missing learnable-phi doc link | Prevent conceptual misuse | Low |
| P11 | New module or docs | Paper-4 algorithms | Either implement Algorithm 6/string APIs or mark them explicitly not implemented | Align scope with paper specs | Medium |

## 10. Final Verdict

The repository contains a credible torch/SYCL-oriented port of the original KSig
dense signature and PDE kernels for ordinary `N>=1` use, but it is not yet a
paper-faithful implementation of all four paper specifications. The safest
classification is **MIXED**: core ported recurrences mostly pass static review,
while edge-case truncation, facade API behavior, differentiability, random
feature independence, sparse projection correctness, and paper-4 coverage are
high-risk or incomplete.

## Priority Action List

1. Fix or reject `truncation=0`; add level-zero contract tests.
2. Forward `diag` through `GeneralSignatureKernel` delegated engines.
3. Remove NumPy/float32 coercion from generalized tensor paths or document them
   as non-differentiable sklearn-style kernels.
4. Fix random-feature RNG cloning and `VerySparseRandomProjection`.
5. Add PDE stability policy tests for both plain and learned PDE kernels.
6. Correct `docs/SIGNATURE_KERNELS.md` so `sig-EXACT` is consistently described
   as outside the normalize-once global-`phi` family.
7. Mark paper-4 simultaneous low-rank and string algorithms as not implemented,
   or add first-class implementations and tests.
