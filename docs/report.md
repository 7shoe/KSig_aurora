# KSig_aurora — Audit Remediation & Test-Hardening Report

Date: 2026-06-11 · Status: rev-2 landed · Suite: **449 passed, 5 skipped, 6 xfailed**

This report summarizes the work done in response to the external paper-to-code
audit (`docs/AUDIT_REPORT.md`) and the reviewer feedback on our first response
(`docs/AUDIT_RESPONSE.md`). It covers (1) the statistical/correctness fixes that
landed, (2) the layered test suite that pins them, and (3) a runnable usage
example. Items still sequenced for a follow-up are listed at the end.

---

## 1. What changed and why

The audit's verdict was **MIXED**: the core `N≥1` dense / PDE / RFF ports are
faithful (and locked by the golden suite), but a cluster of edge-case, API, and
random-feature issues needed fixing. Of nine findings, **four were unambiguous
live bugs** and **seven are release blockers**. Five fixes landed in rev-2; the
rest are sequenced with tracking tests.

### Landed (code + test, suite green)

| ID | Issue | Fix | Where | Test |
|----|-------|-----|-------|------|
| **F1** | `truncation=0` folded level 1 into the level-0 constant | `n_levels==0` returns only `K₀`; negative `n_levels` rejected | `ksig/algorithms.py` `signature_kern_first_order` / `_higher_order` | `paper_contracts/test_truncated_signature.py` |
| **F2** | Facade dropped `diag=True` for delegated `sig-PDE`/`sig-EXACT` | forward `diag=diag` to the ksig engine | `ksig/generalized.py` `GeneralSignatureKernel.__call__` | `paper_contracts/test_general_facade.py` |
| **F3 (dtype)** | `_static_block`/`_diag_block` forced a NumPy host hop + float32 | `_as_block_tensor`: preserve dtype, device, and autograd graph | `ksig/generalized.py` | `paper_contracts/test_general_facade.py` |
| **F6** | RFSF per-level weight independence held only by RNG object-aliasing | explicit deterministic per-level **child seeds** for static-feature *and* projection clones | `ksig/kernels.py` `_spawn_child_seeds`, `SignatureFeatures._make_feature_components` | `paper_contracts/test_random_features.py` |
| **F7** | `VerySparseRandomProjection` overwrote its Bernoulli mask with a dense Rademacher matrix (dense, mis-scaled) | `support * signs` → realized density ≈ `prob_nonzero` | `ksig/projections.py` | `paper_contracts/test_random_features.py` |

### Reviewer-driven corrections to the response

- **F6 reframed.** The first response called integer seeds "definitively
  broken." Running the actual path disproved that: the level clones *share one
  advancing generator* (because `get_params()` returns the live
  `TorchRandomState` object), so the weights were already independent. The real
  defect is **fragility** — the invariant breaks under `sklearn.clone()`, a
  raw-seed `get_params`, or parallel construction. The fix makes independence a
  property of the design rather than an accident.
- **F5 split by kernel.** Plain `sig-PDE` is **never** auto-clamped — clamping
  silently rescales the PDE driver and hands back a different kernel. Policy:
  warn on risky `max|M|`, run the intended solve, assert finite. Auto-clamp/NaN
  stays only for `sig-PDEphi`, where λ is a fitted design variable.
- **F3 return contract.** `differentiable=True` cannot return NumPy
  (`.numpy()` detaches). It therefore *requires* a torch-returning path
  (`return_on_gpu=True` / `return_torch=True`), else it raises; the default
  sklearn path returns detached NumPy.
- **F8 de-overclaimed.** A feature matrix `U` with Gram `U Uᵀ` is at most a dense
  Gram factor — **not** paper-4 Algorithm 6 (simultaneous low-rank DP). Labeled
  accordingly.

Full point-by-point dispositions: `docs/AUDIT_RESPONSE.md` (rev-2).

---

## 2. Test suite: `tests/paper_contracts/`

The remediations are pinned by a small, layered package — **sentinel coverage**
(a few representative kernels per invariant), not a parameter cross-product; the
full numerical matrix stays in the golden/oracle suites. Statistical checks are
tiny, seeded, and tolerance-loose; the one Monte-Carlo smoke is marked
`@pytest.mark.random_feature` so it is deselectable and cannot flake a core gate.

| File | Pins |
|------|------|
| `test_truncated_signature.py` | F1 level-0 contract; level-increment identity `K^{≤N} − K^{≤N−1} == K_N` |
| `test_general_facade.py` | F2 delegated `diag` (shape + value vs full diagonal); F3 dtype preservation (both directions); Gram symmetry |
| `test_pde_signature.py` | Goursat boundary / constant-path; F5 policy (plain-PDE warn-not-rescale, sig-PDEphi clamp/reporting) |
| `test_random_features.py` | F6 per-level independence + reproducibility; F7 realized sparsity; RFF self-kernel = 1; MC convergence smoke |
| `test_docs_scope.py` | paper-4 Algorithm 6 / string kernels (not implemented); internal docs link-check |

**`xfail` is the live to-do list.** Five tests are `xfail` for not-yet-landed
work; each carries its reason and will *xpass → promote to required-pass* when the
fix lands (repo convention — see `tests/conftest.py` suite notes):

```
plain sig-PDE stability gate (F5/P7)        auto_clamp φ-reporting (F5/P7)
paper-4 Algorithm 6 (F8)                     string-kernel sequentialization (F8)
docs link: LEARNABLE_PHI_SIGNATURE_KERNELS.md (F9/P10)
```

### Running

```bash
source /home/siebenschuh/Projects/Aurora_HPC/environment/activate_ddp_venv.sh   # torch/oneAPI on Aurora

pytest tests/paper_contracts/ -q          # the contract layer (fast)
pytest tests/ -q                          # full suite incl. golden/oracle
pytest -m "not random_feature" -q         # skip the Monte-Carlo smoke
pytest tests/paper_contracts/ -rx         # show the xfail reasons (the to-do list)
```

---

## 3. Usage example

The six kernels are one object, `GeneralSignatureKernel`, selected by
`(phi, truncation, normalize)`. After an optional `fit_phi`, every column is an
ordinary precomputed kernel: call it to get a Gram, drop it into an SVM.

```python
import numpy as np
import ksig
ksig.set_default_device("cpu")
from ksig.generalized import GeneralSignatureKernel

# n sequences of length L in d dimensions  ->  [n, L, d]
rng = np.random.default_rng(0)
n, L, d = 64, 24, 6
y = rng.integers(0, 2, n)
X = np.cumsum(rng.standard_normal((n, L, d)), 1) / np.sqrt(L)
X = X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), 1e-6, None)

# --- the six columns (phi, truncation, normalize) ---
configs = {
    "sig-L1":       dict(phi="level_one", truncation=3),               # level-1 only
    "sig-TRUNC-φ1": dict(phi="const",     truncation=3),               # depth-3, phi≡1
    "sig-PDE":      dict(phi="const",     truncation=None),            # untruncated Goursat
    "sig-Wphi":     dict(phi="free",      truncation=4),               # learned level weights
    "sig-PDEphi":   dict(phi="dilation",  truncation=None),            # learned dilation mixture
    "sig-EXACT":    dict(phi="const",     truncation=3, normalize="per_level"),  # legacy probe
}
for name, cfg in configs.items():
    k = GeneralSignatureKernel(bw=1.0, **cfg)
    if cfg["phi"] in ("free", "dilation"):
        k.fit_phi(X, y, steps=60)          # learn phi by kernel-target alignment
    K = np.asarray(k(X, X))                # full Gram  [n, n], unit diagonal
    diag = np.asarray(k(X, diag=True))     # cheap diagonal  [n]  (F2: now correct)
    print(f"{name:12s} Gram{K.shape}  phi(0)={k.phi_profile()[0]:.2f}")

# dtype is preserved end to end (F3): float64 in -> float64 out
K64 = GeneralSignatureKernel(phi="const", truncation=3)(X.astype(np.float64),
                                                        return_on_gpu=True)
assert K64.dtype.__str__() == "torch.float64"
```

Drop into scikit-learn with a precomputed kernel:

```python
from sklearn.svm import SVC
k = GeneralSignatureKernel(phi="free", truncation=4, bw=1.0).fit_phi(X, y, steps=120)
Ktr = np.asarray(k(X, X))                                  # train Gram
SVC(kernel="precomputed").fit(Ktr, y)                      # ... .predict(np.asarray(k(Xte, X)))
```

### Random-feature (RFSF) maps

Per-level weights are now independent **by design** and reproducible under a
fixed integer seed (F6):

```python
from ksig.kernels import SignatureFeatures
from ksig.static.features import RandomFourierFeatures

sf = SignatureFeatures(
    n_levels=4,
    static_features=RandomFourierFeatures(n_components=256, random_state=0),
).fit(X)
# sf.static_features_[m].random_weights_ are independent across levels m,
# and identical across two runs with random_state=0.
```

---

## 4. Sequenced (not yet landed)

Tracked by the `xfail` tests above; each flips to a required-pass when done.

- **P5** — opt-in differentiable evaluation (the F3 return contract: torch-return
  required, `no_grad` opt-out).
- **P7** — PDE stability policy: plain `sig-PDE` warn + assert-finite (no
  rescale); `sig-PDEphi` gate-the-fit + report effective (clamped) λ.
- **P6** — remove / hard-reject the dead `freeze_phi0` argument (F4).
- **P10** — author or fold in `LEARNABLE_PHI_SIGNATURE_KERNELS.md` so the docs
  link-check goes green (F9).
- **P11 / F8** — explicitly-labeled dense Gram-factor helper (not Algorithm 6);
  string-kernel sequentialization deferred.

See `docs/AUDIT_RESPONSE.md` §3 for the full sequencing and `docs/AUDIT_REPORT.md`
for the original findings.
