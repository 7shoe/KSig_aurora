# Audit Response and Remediation Plan

Response date: 2026-06-11
Revision: rev-2 (2026-06-11) — incorporates external reviewer feedback on rev-1
and records the fixes that have since landed.
Responding to: `docs/AUDIT_REPORT.md` (external paper-to-code audit, 2026-06-11)
Author: KSig_aurora maintainers

## Changelog (rev-2)

The rev-1 response was reviewed; rev-2 tightens it on five points and lands the
low-risk statistical fixes:

- **F6 softened and corrected.** Rev-1 called integer seeds "definitively
  broken." That is wrong, and we now have the repro to prove it (§1, F6): with
  the current code the per-level clones *share one advancing generator*, so the
  weight matrices are in fact independent. The real defect is that this
  independence is *accidental and fragile*, not that it is currently violated.
  Reframed accordingly; the fix (explicit child seeds) is unchanged and now
  implemented.
- **F5 split by kernel.** Plain `sig-PDE` is no longer auto-clamped — clamping
  rescales the driver and silently changes the kernel object the user asked for.
  Policy is now: warn on risky `max|M|`, run the *intended* solve, then
  assert/report finite results. Auto-clamp/NaN stays only for `sig-PDEphi`,
  where λ is already a fitted design variable.
- **F3 return contract made explicit.** Differentiable evaluation cannot return
  NumPy. `differentiable=True` therefore requires a torch-returning path
  (`return_on_gpu=True`, or a new `return_torch=True`); the default sklearn-style
  path returns detached NumPy.
- **F8 de-overclaimed.** We no longer say Algorithm 6 is "largely already
  available." A feature matrix `U` with Gram `U Uᵀ` is *not* the paper-4
  simultaneous low-rank DP; it is at most a dense Gram factor.
- **Consistency fixes.** Reconciled the findings count (§ headline) and corrected
  the `test_freeze_phi0_false_rejected_or_effective` citation (F4, not D-A4).

**Landed in rev-2 (code + tests green, 432 passed):** F1, F2, F3-dtype, F6, F7,
plus the dtype-preserving block conversion. Remaining (F3-grad API, F5 policy,
F4, F9 docs, F8 scope labels) stay sequenced in §3.

## 0. Purpose and method

This document is our point-by-point response to the external audit. For every
finding we did three things before writing a word here:

1. **Re-derived the expected behavior from the primary sources** the auditors
   cited — `paper_specs/paper_3/implementation_contracts.md`,
   `paper_specs/paper_2/algorithm_cards/truncated_phi_signature_kernel.md`, and
   the original reference `paper_specs/KSig_original_repo/ksig.txt` — rather than
   taking the audit's restatement on faith.
2. **Re-read the cited code** at the exact line ranges and reproduced the logic
   by hand (the auditors could not execute the suite; see §6).
3. **Classified our position** as one of: **ACCEPT** (confirmed, will fix),
   **ACCEPT-WITH-NUANCE** (confirmed, but the fix or the framing differs from the
   audit's suggestion), or **DISPUTE** (we believe the current behavior is
   correct or is a deliberate, documented contract).

Headline position: **we agree with the audit's overall "MIXED" verdict.** The
ported recurrences are faithful; the regressions are concentrated in (a) the
level-0 edge case, (b) the learnable/general facade's API and dtype/grad
behavior, and (c) one random-feature correctness bug plus one fragile-but-not-
yet-wrong RNG invariant. Of the nine findings, **four are unambiguous live bugs**
(F1 level-0, F2 dropped `diag`, the F3 float32 coercion, F7 sparse projection)
and **seven are release blockers** (those four plus F5 stability policy, F6 RNG
robustness, F9 docs). None of the findings invalidate the core `N>=1` dense /
PDE / RFF ports, which the audit itself rates "High" confidence and which our
golden tests already lock.

We treat the audit as a release blocker list. §1 dispositions each finding, §2
resolves the five author-decision ambiguities, §3 is the consolidated change
plan with sequencing, §4 is the test plan we will add, §5 covers the "frozen
engine" tension that two fixes touch, and §6 addresses the environment gap that
prevented the auditors from running anything.

---

## 1. Finding-by-finding disposition

### F1 — `truncation=0` / `n_levels=0` includes level 1 — **ACCEPT (High)**

**Independent verification.** Confirmed by hand-trace of
`ksig/algorithms.py:83-96`. With `return_levels=True` the code unconditionally
sets `K = [K0, sum(M)]` (i.e. `[K0, K1]`) *before* the `for i in range(1,
n_levels)` loop, and that loop does not execute for `n_levels=0`. So
`signature_kern_first_order(M, 0, return_levels=True)` returns a 2-element stack
`[K0, K1]`, not `[K0]`. The same is true of `signature_kern_higher_order`
(`:126-129`). This is a genuine paper violation, not a style preference:
`paper_specs/paper_3/implementation_contracts.md:7` (Contract 1) states
"`M=0` returns only the constant 1," and
`paper_specs/paper_2/.../truncated_phi_signature_kernel.md:39` states
"`N == 0 ⇒ output phi(0)` (level-0 term only)." Pitfall list line 58 of that card
names this exact off-by-one.

**Secondary defect (the broadcast).** In `WeightedSignatureKernel` the `const`
column builds `phi_fixed = np.ones(truncation+1)`, so for `truncation=0` `phi`
has length 1 while `_levels()` returns a length-2 stack. `phi[:, None, None] *
Kl` then broadcasts the single weight across *both* levels and `.sum(0)` adds
level 1 in — the wrapper silently returns `K0 + K1` weighted by `phi(0)`. We
confirmed the length-N+1-vs-N+1 match is exact for every `N>=1` (the loop
contributes `N-1` extra levels on top of the seeded 2, giving `N+1`), so this is
strictly an `N=0` bug; the golden tests at `N=1,4` are unaffected.

**Fix (see P1, P2).** Special-case `n_levels == 0` in `signature_kern_first_order`
and `signature_kern_higher_order` to return only `K0` (the paper-faithful
choice; see decision D-A2 in §2). Add a defensive length assertion in
`WeightedSignatureKernel._levels`/`__call__` so a `len(phi) != level_count`
mismatch raises instead of broadcasting. This also future-proofs against any
caller-supplied `phi_fixed` of the wrong length.

### F2 — Facade drops `diag=True` for delegated KSig engines — **ACCEPT (Medium)**

**Independent verification.** Confirmed at `ksig/generalized.py:430-433`. The
`self._kind == "ksig"` branch calls `self._engine(X, Y,
return_on_gpu=return_on_gpu)` with no `diag`, while the `"gen"` branch one line
later *does* forward `diag`. So `sig-PDE` and `sig-EXACT` (the two `"ksig"`
columns) silently compute a full `O(n^2)` Gram and return the wrong shape when
the caller asks for the diagonal. Both delegated engines
(`ksig.kernels.SignaturePDEKernel`, `ksig.kernels.SignatureKernel`) accept a
`diag` argument, so the fix is a one-line forward. The class docstring
(`:38-39`) explicitly promises the full `__call__(X, Y, diag, return_on_gpu)`
contract, so this is a contract violation, not an undocumented limitation.

**Fix (P3).** Forward `diag=diag` in the `"ksig"` branch. Add the regression
test that asserts shape `(n,)` for all six facade columns under `diag=True`.

### F3 — Generalized kernels break input gradients and force float32 — **ACCEPT-WITH-NUANCE (High)**

**Independent verification.** Confirmed at `ksig/generalized.py:51-56`
(`_static_block`), `63-68` (`_diag_block`), and the `torch.no_grad()` wrappers at
`:201`, `:306`. Both block builders do `torch.as_tensor(np.asarray(X),
dtype=torch.float32, device=dev)`, which (a) forces a host round-trip through
NumPy, severing any incoming autograd graph and any non-CPU device placement,
and (b) hard-casts to float32.

**Where we agree with the audit, emphatically:** the **float32 coercion is an
outright bug independent of the gradient question.** The module's *own* stability
guard advises "Lower lam_max or use float64" (`:120`) and the design notes say
the Goursat path "is stable only for `|lam*m|<~1`" — yet `_static_block`
discards any float64 input the user supplies for exactly that stability margin.
We will preserve input dtype and device unconditionally.

**Where we add nuance (the gradient half):** the module is *designed* as a set of
precomputed, sklearn-style kernels — the class docstring says after `fit_phi`
"they are ordinary precomputed kernels (drop into gak_gram + SVC)". So
end-to-end input-path differentiability was never a design goal of this facade,
and the audit itself raises this as open question A3. The paper-2 truncated-φ
card (test 7) does request a finite-difference-consistent `X`-gradient, and
paper-3 Contract 11 forbids NumPy hops on differentiable values — but Contract 11
is scoped to the **RFSF/RFSF-DP/RFSF-TRP feature maps** (`ksig/kernels.py
SignatureFeatures`, which already operate directly on torch tensors with no
NumPy hop), not to the paper-2 GSK facade.

**Decision (resolves A3; see D-A3 in §2).** Two parts, now with an explicit
return contract.

*(1) dtype/device preservation — done.* Both block builders no longer hop through
`np.asarray` or force float32. `_as_block_tensor` adopts a torch input on its own
device at its own dtype (and keeps its autograd graph), and adopts a numpy/list
input at its native float dtype (`generalized.py`, P4). A float64 path in now
yields a float64 kernel out, as the Goursat stability notes recommend. Locked by
`test_facade_preserves_float64`.

*(2) differentiable evaluation — explicit return contract (reviewer's
sharpening).* The subtlety the reviewer caught: **`differentiable=True` is
incompatible with the default NumPy return.** `.cpu().numpy()` detaches the graph,
so a flag that "keeps the graph" while the method still returns NumPy would be a
contradiction. The contract is therefore:

| call | returns | gradient to inputs |
|---|---|---|
| default (sklearn path) | detached NumPy | no (by design) |
| `return_on_gpu=True` / `return_torch=True` | torch tensor | only if `differentiable=True` |
| `differentiable=True` **without** a torch-returning flag | **rejected** (`ValueError`) | n/a |

So `differentiable=True` *requires* opting into a torch return; it never silently
returns NumPy. `torch.no_grad()` becomes opt-out (wrapping only the NumPy-return
path). The default remains a precomputed sklearn kernel, and we **document** that
φ/λ fitting — not input-path autograd — is the supported learning surface
(P5). (P5 is sequenced for the API change; P4/dtype is already in.)

### F4 — `freeze_phi0` is a dead constructor argument — **ACCEPT-WITH-NUANCE (Medium)**

**Independent verification.** Confirmed: `freeze_phi0=True` is accepted at
`ksig/generalized.py:353` and never read anywhere in the class. φ(0) is pinned
structurally — `WeightedSignatureKernel._phi_vec` always prepends `ones(1)`
(`:155`), and the dilation mixture has φ(0)=Σwᵢ=1 by softmax construction.

**Nuance — do not "implement" it.** The audit offers "remove or implement."
Implementing `freeze_phi0=False` (i.e. learning φ(0)) would actively break the
central design invariant the module is built on: "φ(0)=1 on EVERY signature arm
… so Delta(learn-phi) carries no level-0 mismatch" (`:32-33`). A learnable φ(0)
re-introduces exactly the cross-arm level-0 mismatch the facade exists to
eliminate. **We will remove the parameter** (preferred), or, if backward-compat
of the constructor signature matters, retain it and raise `ValueError` on
`freeze_phi0=False` with a message pointing at the invariant. Either way the dead
no-op behavior is eliminated (P6).

### F5 — PDE stability gates are incomplete — **ACCEPT (High)**

**Independent verification.** Confirmed in two places. (1)
`LearnedPhiSignaturePDEKernel.fit_phi` calls `_stability_ok(M, self.lam_max,
"fit_phi")` at `:271` purely for its warning side-effect — the boolean return is
discarded and optimization proceeds on possibly-divergent Grams. (2) Plain
`sig-PDE` delegates to `ksig.kernels.SignaturePDEKernel` (`:386-391`), which has
no stability guard at all — and the audit correctly notes this is the
highest-exposure path (untruncated, default column). The evaluation paths *do*
NaN-out (`:312`, `:321`), so the inconsistency is specifically: eval guards, fit
warns-but-continues, plain-PDE does neither.

**Fix (P7) — policy differs by kernel (reviewer's sharpening).** A single
finite/stability *check* (`lam_max * max|m|`) is shared, but the *action* on
violation is deliberately not uniform, because "auto-clamp λ" silently rescales
the PDE driver and hands the user a different kernel object than the one they
constructed:

- **Plain `sig-PDE` (untruncated, default column): never auto-rescale.** There is
  no λ design variable to clamp here — λ≡1 is the kernel's definition. The policy
  is: (a) **warn** when `max|M|` enters the risky Goursat regime, (b) run the
  **intended** solve unchanged, then (c) **assert/report finite** results before
  they can enter a downstream SVM. We surface a structured failure (or raise,
  configurable) on non-finite output rather than quietly substituting a stabler
  but different kernel.
- **Learned `sig-PDEphi`: clamp/NaN is appropriate.** Here λ is already a fitted
  design variable, so the existing `auto_clamp` mechanism (`:234-239`) and the
  NaN-block-on-divergence path are legitimate — clamping moves within the
  model's own parameter space. We additionally (i) *gate the fit* on the bound
  (the rev-1 bug: `fit_phi` discarded `_stability_ok`'s return at `:271` and
  optimized on divergent Grams), and (ii) fix the reporting mismatch by having
  `phi()`/`phi_profile()` report the **effective** (clamped) λ rather than the
  requested nodes.

The shared invariant across both is therefore "a divergent solve can never
silently enter training or a downstream SVM," but only the kernel that *has* a
tunable λ is allowed to change λ to get there.

### F6 — RFSF level independence rests on a fragile RNG invariant — **ACCEPT-WITH-NUANCE (High-risk, not a live bug)**

**Correction to rev-1.** Rev-1 called the integer-seed case "definitively
broken." That was an over-claim made without a runnable repro, and it is wrong.
The reviewer flagged it, and we have now traced *and executed* the actual path:

1. `KernelFeatures.__init__` / `RandomProjection.__init__` call
   `utils.check_random_state(random_state)` **immediately**
   (`static/features.py:43`, `projections.py:46`), so `self.random_state` is a
   live `TorchRandomState` object, not a raw seed.
2. sklearn's `get_params()` returns that **same live object**, and
   `check_random_state` passes a `TorchRandomState` straight through
   (`torch_backend.py`), so every level clone shares **one** generator.
3. A shared generator *advances* across the sequential `.fit()` calls, so each
   level draws a fresh, independent slice of the stream.

Executed on this build (torch from `activate_ddp_venv.sh`):

```text
SignatureFeatures(n_levels=3, static=RandomFourierFeatures(random_state=0))
shared random_state object across clones: True
W[0] vs W[1] identical: False
W[0] vs W[2] identical: False
W[1] vs W[2] identical: False
```

So the per-level weights **are** independent today, for both integer and `None`
seeds — Contract 4 is *currently satisfied*. The audit's torch-only diagnostic
("two fresh `Generator()` collide") was measuring a scenario the code never
takes, because the object is shared rather than re-seeded.

**Why it is still a finding (and still High-risk).** The independence is an
*accident of object aliasing*, not a designed property:

- It breaks the instant anything makes `get_params()` return a raw seed (a
  reasonable, arguably more-correct sklearn change), or deep-copies the estimator
  (sklearn `clone()` does exactly this), or constructs the clones in parallel
  instead of sequentially. Any of those re-seeds identical streams ⇒ the biased
  `W^(m)` the paper warns about.
- It also makes "reproducibility" subtle: re-`fit`ing the same object advances
  the shared stream, so weights differ run-to-run within a process.

The safer wording the reviewer suggested is exactly right: **high-risk; pin it
with a test; make independence explicit so a future clone-by-seed cannot
reintroduce the bias.**

**Fix (P8) — implemented.** We now spawn an explicit, deterministic per-level
child seed (drawn from the parent state) for each static-feature copy *and* each
projection copy (`kernels.py` `_spawn_child_seeds`). Independence is now a
property of the design, not of object aliasing; a seeded parent stays
reproducible, an unseeded parent matches sklearn's `random_state=None`
semantics. Locked by `test_rfsf_independent_weights_per_level` (int/`None`
seeds) and `test_rfsf_reproducible_for_integer_seed` (§4). The named
RFSF/RFSF-DP/RFSF-TRP constructors (D-A4) remain the recommended surface so users
cannot assemble a configuration that bypasses this.

### F7 — `VerySparseRandomProjection` is not sparse — **ACCEPT (Medium)**

**Independent verification.** Confirmed at `ksig/projections.py:336-346`. Line
336 draws the Bernoulli support into `components_full`, line 339 forces one
nonzero, then **line 340-341 overwrites `components_full` entirely with a dense
Rademacher matrix.** The sparsity mask is destroyed. Downstream, line 342-344
computes `sampled_idx_` via `robust_nonzero` over the now-dense ±1 matrix —
which has no zeros — so *every* column is selected and no subsampling occurs,
while `scaling_` (line 347) is still the sparse `sqrt(1/(prob_nonzero *
n_components))`. Net effect: a **dense signed projection mis-scaled as if
sparse** — both the distribution and the variance are wrong.

**Fix (P9).** Multiply the Bernoulli support by the Rademacher signs
(`support * signs`) instead of overwriting, so realized density ≈ `prob_nonzero`
and the existing sparse scaling becomes correct again. Add the realized-sparsity
regression test.

### F8 — Paper-4 simultaneous low-rank Gram and string kernels absent — **ACCEPT-WITH-NUANCE (Medium)**

**Independent verification.** Confirmed by code search: there is no
implementation of `paper_4/.../simultaneous_lowrank_kernel_matrix.md`
(Algorithm 6) nor `paper_4/.../string_kernel_sequentialization.md`. The repo
*does* ship generic low-rank signature **feature maps** (`signature_kern_low_rank`
in `ksig/algorithms.py:159-353`, `SignatureFeatures` in `ksig/kernels.py`), which
are a different object from Algorithm 6's simultaneous low-rank Gram **factor**.

**Nuance — corrected, do not overclaim (reviewer's point).** Rev-1 said
Algorithm 6 was "largely already available." That overclaims. What we actually
ship is a low-rank signature **feature map**: it produces per-batch features `U`
whose `U @ U.T` is *a* dense Gram factorization. That is **not** the paper-4
simultaneous low-rank construction — Algorithm 6 is a joint, simultaneous
low-rank DP with its own `O(N L ρ M)` complexity and joint-factorization
semantics across the whole Gram, none of which a per-batch feature map provides.
The honest statement is: *some feature-map factorizations can serve as a dense
Gram factor, but this is not yet Algorithm 6.* We may still expose a thin,
explicitly-named "dense Gram factor" API (returning `U` and `U Uᵀ`, validated
against the dense `SignatureKernel` for linear inputs) **as long as it is not
labeled Algorithm 6**. The **string-kernel sequentialization** (gap-decay λ,
equality kernel, strict-subsequence DP) is genuinely absent and is a larger build
with no existing scaffolding.

**Decision (resolves A4 scope; see D-A4 in §2).** For the release we will
**explicitly document both as out-of-scope / not-implemented** in `README.md` and
the docs scope section, and have any partial stub raise `NotImplementedError`
rather than silently returning a different object — so no user can mistake a
generic feature map for Algorithm 6. We will additionally *attempt* the thin
simultaneous-low-rank wrapper if it lands without destabilizing the release; the
string kernel is deferred to a follow-up with a tracked issue. Honest scope
labeling is the release-blocking part; the implementation is a stretch goal.

### F9 — `docs/SIGNATURE_KERNELS.md` self-contradicts and links a missing file — **ACCEPT (Medium)**

**Independent verification.** Both halves confirmed. (1) Self-contradiction:
line 4 says the six kernels "are all the **same object**" and the Summary
(line ~307) says "Every kernel here is the General Signature Kernel
`K_φ=Σ φ(k)K_k`" while listing `sig-EXACT` among them — yet §6 line 220 correctly
says `sig-EXACT` is "Not a member of the global-φ family." (2) Missing file:
`grep` confirms `LEARNABLE_PHI_SIGNATURE_KERNELS.md` is referenced **four times**
(`docs/SIGNATURE_KERNELS.md:17, 189, 215, 301`) and does not exist anywhere in
the checkout.

**Fix (P10).** Rewrite the skeleton/summary so the membership claim is precise:
**five** columns (`sig-L1`, `sig-TRUNC-φ1`, `sig-PDE`, `sig-Wphi`, `sig-PDEphi`)
are normalize-once members of the global-φ family `K_φ`, and `sig-EXACT` is a
**legacy per-level-normalize-then-average probe outside** that family (this
matches §6 of the same doc and the audit's A5). Either author the missing
`LEARNABLE_PHI_SIGNATURE_KERNELS.md` design note or remove the four references
and fold the needed content into `SIGNATURE_KERNELS.md`. Add a docs link-check /
lint step to CI so a dangling internal reference fails the build (cheap, prevents
recurrence).

---

## 2. Author decisions on the audit's open ambiguities (§7 of the report)

| ID | Ambiguity | **Our decision** | Rationale |
|---|---|---|---|
| **D-A1** | Which PDE stencil is the contract? | **Declare the second-order KSig stencil `1 + m/2 + m²/12` the implementation contract.** *We DISPUTE treating this as a defect.* | It is a faithful port of the original reference (`ksig.txt`, second-order PDE recurrence) and is documented in `TORCH_PORT.md`. The paper-1 card's first-order explicit scheme is a *different discretization of the same PDE*; both converge. Action is documentation-only: cross-reference the card and state that cell-by-cell tests must target the KSig recurrence and analytic limits, not the first-order stencil. |
| **D-A2** | Should `truncation=0` be public? | **Yes — implement paper-faithful `N=0 ⇒ phi(0)` only.** | Contract 1 and the truncated-φ card both define `N=0` as the level-0 constant. Making it correct is strictly better than rejecting it and costs one branch (P1). The facade will still validate `truncation >= 0` and reject negatives. |
| **D-A3** | Is input differentiability a supported API? | **Default: no — detached NumPy (precomputed sklearn kernel). Opt-in: yes, but only via a torch-returning path: `differentiable=True` *requires* `return_on_gpu=True`/`return_torch=True`, else it raises.** | A `differentiable=True` that still returned NumPy would be self-contradictory (`.numpy()` detaches). Gating it behind a torch return makes the contract honest while keeping the sklearn default. The float32 coercion is fixed unconditionally regardless of this toggle. |
| **D-A4** | Generic random-feature components vs. named paper-3/paper-4 algorithms | **Add named, constraint-enforcing constructors** for the random-feature family (independent per-level `W`, DP/TRP dimensions, correct projection distributions); **document paper-4 Algorithm 6 / string kernels as not-implemented** for this release. | Named constructors prevent users from silently assembling a biased (F6) or mis-distributed (F7) configuration; explicit not-implemented labels prevent over-claiming paper-4 scope. |
| **D-A5** | Normalization naming (`once` vs `per_level`) | **`sig-EXACT` (`per_level`) is explicitly outside the normalize-once `K_φ` family in all docs and summaries.** | Resolves F9; aligns the prose with the code, which already restricts `per_level` to the single `phi="const"` truncated column (`generalized.py:370-372`). |

---

## 3. Consolidated change plan (release-blocking unless noted)

Mapped to the audit's suggested patch IDs P1–P11. Ordered by risk-to-correctness
and dependency. Status as of rev-2: **✅ landed** (code + test, suite green) /
**◻ sequenced**.

**Tier 1 — correctness bugs, must fix before release**

1. ✅ **P1 — `n_levels == 0` branch** in `signature_kern_first_order` /
   `signature_kern_higher_order`: return only `K0`; reject negative `n_levels`.
   (Fixes F1 root cause.) *Tests: `test_truncation_zero_is_level_zero_only`,
   `test_negative_n_levels_rejected`.*
2. ◻ **P2 — length validation** in `WeightedSignatureKernel`: assert
   `len(phi) == level_count`; reject negative `truncation` at the facade.
   (The wrapper already validates `phi_fixed` length at `:144`; the remaining
   work is the facade-level negative-`truncation` guard.)
3. ✅ **P3 — forward `diag`** in `GeneralSignatureKernel.__call__` `"ksig"` branch.
   (Fixes F2.) *Test: `test_delegated_diag_shape` (sig-PDE, sig-EXACT).*
4. ✅ **P9 — `VerySparseRandomProjection`**: `support * signs`. (Fixes F7.)
   *Test: `test_very_sparse_projection_realized_sparsity`.*
5. ✅ **P8 — per-level RNG independence** via explicit child seeds for
   static-feature and projection clones. (Hardens F6 — independence is now
   by-design, see the corrected F6 above.) *Tests:
   `test_rfsf_independent_weights_per_level`,
   `test_rfsf_reproducible_for_integer_seed`.*

**Tier 2 — contract/dtype/stability, must fix before release**

6. ✅ **P4 — dtype/device preservation** in `_static_block` / `_diag_block` via
   `_as_block_tensor` (no `np.asarray` host hop, no forced float32). (Fixes the
   unconditional half of F3.) *Test: `test_facade_preserves_float64`.*
7. ◻ **P5 — opt-in differentiable evaluation** with the explicit return contract
   (`differentiable=True` requires a torch return, else raises; `no_grad`
   becomes opt-out). (Resolves the F3/A3 gradient half.)
8. ◻ **P7 — PDE stability policy** — *plain `sig-PDE`: warn + run intended solve
   + assert finite (no rescale); `sig-PDEphi`: gate fit + clamp/NaN +
   `phi()`/`phi_profile()` report effective λ.* (Fixes F5; see the revised F5.)
9. ◻ **P6 — remove (or hard-reject) `freeze_phi0`.** (Fixes F4.)

**Tier 3 — docs/scope, must ship accurate**

10. **P10 — `docs/SIGNATURE_KERNELS.md` correction** (five members + `sig-EXACT`
    exception; resolve the missing-file references) + CI link-check. (Fixes F9.)
11. **P11 — paper-4 scope labeling**: document Algorithm 6 / string kernels as
    not-implemented; `NotImplementedError` on any stub. *Stretch:* thin
    simultaneous-low-rank Gram wrapper validated against dense. (Addresses F8.)

**Documentation-only (no code change)**

12. **D-A1** — declare and cross-reference the second-order KSig stencil as the
    PDE contract.

---

## 4. Test plan (new files / cases we will add)

We adopt the audit's §8 test matrix essentially verbatim, because each test
pins one regression above. Concretely:

- `tests/test_paper_contracts_truncated_signature.py`
  - `test_truncation_zero_level_zero_only` — `N=0` ⇒ unnormalized `phi(0)`,
    normalized diagonal `1`, for **both** the dense algorithm and the weighted
    wrapper. (F1)
  - `test_level_increment_difference_exact` — `K^{<=N} - K^{<=N-1} == K_N`. (level indexing)
  - `test_general_signature_delegated_diag_shape` — `diag=True` ⇒ shape `(n,)`
    for all six facade columns. (F2)
  - `test_weighted_phi_length_mismatch_raises` — mismatched `phi_fixed`/level
    count raises, never broadcasts. (F1 secondary)
- `tests/test_paper_contracts_pde_signature.py`
  - `test_sigpde_constant_path_is_one` (boundary). 
  - `test_sigpde_stability_guard_plain_and_learned` — one consistent policy for
    `sig-PDE` and `sig-PDEphi`. (F5)
  - `test_sigpde_wavefront_lambda_gradcheck` — finite λ-gradients,
    finite-difference agreement. 
- `tests/test_paper_contracts_learned_phi.py`
  - `test_freeze_phi0_false_rejected_or_effective` — `False` raises (per F4; the
    φ(0)=1 invariant). *(D-A4 is unrelated — it scopes the random-feature /
    paper-4 work, not freeze_phi0.)*
  - `test_generalized_weighted_input_gradient` — with `differentiable=True`,
    finite `X.grad`; default path documented non-differentiable. (F3/A3)
  - `test_generalized_preserves_float64` — float64 in ⇒ float64 out. (F3)
  - `test_auto_clamp_phi_profile_matches_evaluation_nodes` — reported φ uses
    effective λ. (F5 reporting)
- `tests/test_paper_contracts_rff.py`
  - `test_rff_self_kernel_one_and_shape` — feature dim `2*d_tilde`, self-inner
    product ≈ 1. (Contract 5)
  - `test_rfsf_independent_rff_weights_per_level` — level `W^(m)` differ for both
    integer and default seeds, runs remain reproducible. (F6)
  - `test_very_sparse_projection_realized_sparsity` — zero-fraction ≈
    `1 - prob_nonzero`. (F7)
- `tests/test_paper_contracts_sequential.py`
  - `test_higher_order_D1_reduces_to_first_order` (paper-4 anchor).
  - `test_simultaneous_lowrank_gram_matches_dense` — only if P11 stretch lands;
    otherwise an `xfail`/skip annotated "not implemented this release". (F8)
  - `test_string_kernel_bruteforce_short_strings` — `xfail`/skip, "not
    implemented this release". (F8)

We will also add the **CI docs link-check** (F9) and keep the existing golden
suite green — none of P1–P10 change `N>=1` numerics, so the golden `.npz`
fixtures (`tests/golden/`) must continue to match bit-for-bit; any golden drift
is a red flag that a fix overreached.

---

## 5. The "frozen engine" tension (important for the reviewer)

Two fixes touch code the module marks as **FROZEN / "must not be rewritten"**
(`ksig/generalized.py:6`): F1's `signature_kern` (imported, legacy) and F3's
`_static_block` / `_diag_block`. We flag this deliberately:

- **F1 / `signature_kern`:** the legacy `SignatureBase` validates `n_levels >=
  1`, so the legacy `SignatureKernel`/golden path never reaches `n_levels=0`; the
  `N=0` branch is reachable *only* through the new generalized facade. The P1
  branch is therefore additive (a new guarded early-return) and cannot alter any
  existing `N>=1` result — we verified the level-count arithmetic stays `N+1` for
  all `N>=1`. The golden suite is the safety net.
- **F3 / static blocks:** "frozen" was intended to mean "do not change the
  *recurrence math*," not "preserve the float32 host hop." Dtype/device
  preservation changes representation, not values (float64 is a strict
  precision increase; CPU↔device placement is invariant). The landed change keeps
  the RBF math character-identical and only swaps the conversion call for
  `_as_block_tensor`; the smoke test in `generalized.py:__main__`, the new
  `test_facade_preserves_float64`, and the unchanged golden suite guard this.

If the reviewer prefers we leave the frozen engines untouched, the alternative
for F1 is to handle `N=0` entirely inside `WeightedSignatureKernel` (slice the
returned stack to `[K0]`), and for F3 to wrap conversions in a new helper rather
than editing the existing ones. We can ship either; our default is the minimal
in-place change above.

---

## 6. On the environment gap (audit §5)

The auditors could not run `pytest` (no interpreter had numpy+torch+sklearn
simultaneously: `/usr/bin/python3` is 3.6 with numpy-only; `/usr/bin/python3.10`
has torch but no numpy). We acknowledge this and note it does **not** invalidate
their static findings — we reproduced every confirmed finding by hand-trace
above.

**rev-2 update.** For this revision we *did* run the full suite in the supported
environment (`source .../activate_ddp_venv.sh`): **432 passed, 5 skipped, 1
xfailed** with the landed fixes in place. Crucially, executing the F6 RNG path
(rather than reasoning from structure) is what corrected the rev-1 over-claim:
the diagnostic showed the level weights are *independent* under the current
object-aliasing, not identical — see the revised F6. This is the concrete lesson
of the environment gap: a structural argument ("same constructor params ⇒ same
stream") looked airtight but missed that `get_params()` shares a live, advancing
generator. The remaining sequenced items (P5, P6, P7, P10) get the same
execute-then-claim treatment before they are marked done.

For the release we will document the **supported test environment** explicitly
(matching `setup.py`: Python ≥3.9, numpy ≥1.24, scikit-learn ≥1.3.2, torch ≥2.5)
and the exact Aurora activation (`activate_ddp_venv.sh`) used for XPU/SYCL
validation, plus a CPU-only fallback environment file so a reviewer with neither
GPU nor the Aurora stack can run the full numerical suite. The SYCL fast-path
(`ksig/_sycl`) remains optional and falls through to the torch wavefront oracle
when unavailable, so test correctness never depends on it.

---

## 7. Summary of our position

- **Landed in rev-2 (code + test, suite green):** F1 (level-0), F2 (`diag`
  forward), F3-dtype (float64 preserved), F6 (per-level RNG independence now
  by-design), F7 (sparse projection).
- **Accept, sequenced before release:** F3-grad (explicit differentiable return
  contract, P5), F5 (PDE policy — *no rescale* for plain `sig-PDE`, clamp/NaN
  only for `sig-PDEphi`, P7), F4 (`freeze_phi0` removed/hard-rejected, P6),
  F9 (docs correction + link-check, P10).
- **Accept with a chosen framing:** F4 (remove, do not implement), F8 (document
  not-implemented; a *dense Gram factor* helper is a stretch — explicitly **not**
  labeled Algorithm 6).
- **Dispute as a deliberate contract, document only:** A1 (second-order KSig
  stencil is the contract).
- **Author decisions recorded:** D-A1…D-A5 in §2.
- **Reframed after running the code:** F6 is a fragility/robustness fix, not a
  live-bug fix — the integer-seed estimator was never biased in the current
  build.

We believe that after the sequenced items the repository satisfies the paper
contracts the audit checked: level-0 semantics (Contract 1), the full `diag` call
contract, dtype preservation and an explicit differentiability policy, RFSF
per-level independence by design (Contract 4), correct sparse-projection
distribution, a stability policy that never silently substitutes a different
kernel, and accurate, internally-consistent documentation with honest paper-4
scope. The faithful `N>=1` dense / PDE / RFF ports — the core of the library —
are unaffected and remain locked by the golden suite (432 passed in rev-2).
