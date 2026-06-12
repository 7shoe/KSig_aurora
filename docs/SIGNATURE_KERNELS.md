# Signature Kernels in KSig_aurora — a Reference

`GeneralSignatureKernel` (`ksig/generalized.py:GeneralSignatureKernel`) exposes six
signature-based kernels. **Five** are the **same object** — the normalize-once General Signature
Kernel — under three choices: an order-weighting $\varphi$, a truncation depth, and a normalization
convention. The sixth, **`sig-EXACT`**, is a legacy per-level-normalized average that sits *outside*
that family (§0.6, §6); it is documented here for completeness, not as a $\varphi$ specialization.
This document derives each as a specialization of one skeleton (or, for `sig-EXACT`, as the legacy
exception) and states, for each, what it computes, how, and what (if anything) is iteratively
estimated. Every algorithm and nontrivial formula cites the code
(`file:symbol`); where a textbook identity and the implementation disagree, the implementation is
described and the discrepancy flagged.

**Provenance classes.** The kernels fall into three origins, marked per section and collected in
the References. **(P) Port** — an existing KSig algorithm (cupy/CUDA-native) ported to torch/SYCL
in this repo; see `docs/TORCH_PORT.md`. **(L) Literature** — a published construction implemented
here (the Goursat-PDE signature kernel). **(X) Extension** — original work by the author: the
*learnable* General Signature Kernel, which lives inside the General-Signature-Kernel framework but
makes the order-weighting $\varphi$ trainable; design rationale in
`LEARNABLE_PHI_SIGNATURE_KERNELS.md` and §0.3–0.7 here.

Notation: a path $X:[0,T]\to\mathbb R^{d}$, observed as a length-$L$ sequence $x=(x_1,\dots,x_L)$,
$x_t\in\mathbb R^{d}$. Its signature is $S(X)=(1,S^1,S^2,\dots)\in T((\mathbb R^{d}))$ with
$$S^k=\int_{0<u_1<\dots<u_k<T}\mathrm dX_{u_1}\otimes\cdots\otimes\mathrm dX_{u_k}\in(\mathbb R^{d})^{\otimes k}.$$
A static kernel $\kappa:\mathbb R^{d}\times\mathbb R^{d}\to\mathbb R$ with RKHS $\mathcal H_\kappa$
lifts states; Grams live in $\mathbb R^{n\times n}$. $\odot$ is the Hadamard product and
$\oslash$ elementwise division.

---

## 0. The common skeleton

### 0.1 Signature
$S^k$ decays factorially, $\lVert S^k\rVert\le \lVert X\rVert_{1\text{-var}}^k/k!$. Chen’s identity
makes $S$ multiplicative over concatenation; on paths augmented with a monotone coordinate (a
basepoint/time) $S$ is faithful (injective up to tree-like equivalence). These are the classical
rough-path facts; they are used below only to justify that $\{S^k\}$ is a graded, order-aware
feature set.

### 0.2 The kernel trick and the per-level recursion
The signature kernel never forms $S(X)$. Working in $\mathcal H_\kappa$, one needs only the
**second-differenced increment Gram** of the lifted states. For two sequences $x,y$,
$$M_{ab}=\Delta\Delta\,[\kappa(x_a,y_b)]
       =\kappa(x_{a+1},y_{b+1})-\kappa(x_{a+1},y_b)-\kappa(x_a,y_{b+1})+\kappa(x_a,y_b),$$
implemented as `_static_block` (the RBF Gram $\kappa(x_a,y_b)=\exp(-\lVert x_a-y_b\rVert^2/2\sigma^2)$,
`ksig/generalized.py:_static_block`) followed by `_second_diff` (a double `torch.diff`,
`ksig/generalized.py:_second_diff`). For a batch, $M\in\mathbb R^{n_X\times n_Y\times(L-1)\times(L-1)}$.

The per-level Grams are then the Király–Oberhauser first-order recursion
(`ksig/algorithms.py:signature_kern_first_order`):
$$K_0=\mathbf 1,\qquad R^{(1)}=M,\qquad
R^{(k)}=M\odot \operatorname{cumsum}^{\mathrm{excl}}\!\big(R^{(k-1)}\big),\qquad
K_k=\textstyle\sum_{a,b}R^{(k)}_{ab},$$
where $\operatorname{cumsum}^{\mathrm{excl}}$ is the exclusive cumulative sum over both time axes
(`multi_cumsum`, used at `ksig/algorithms.py:signature_kern_first_order` line `R = M * multi_cumsum(R, exclusive=True, ...)`). $K_k\in\mathbb R^{n_X\times n_Y}$ is the level-$k$ signature kernel
$\langle S^k(x),S^k(y)\rangle_{\mathcal H_\kappa}$.

**Load-bearing fact (homogeneity).** $R^{(k)}$ contains exactly $k$ factors of $M$, so
$$K_k(\lambda M)=\lambda^{k}\,K_k(M)\qquad\text{for any scalar }\lambda. \tag{H}$$
Everything below is a corollary of (H).

### 0.3 The General Signature Kernel
For an order-weighting $\varphi:\mathbb N\to\mathbb R_{\ge0}$ (Cass–Lyons–Xu),
$$K_\varphi=\sum_{k\ge0}\varphi(k)\,K_k .$$
$\varphi\equiv1$ is the ordinary signature kernel. Truncated at depth $N$ this is a finite sum over
the cached levels (`ksig/generalized.py:WeightedSignatureKernel._levels`, `.__call__`); untruncated
it is computed by the PDE of §0.5.

### 0.4 The dilation identity
Define the dilated kernel $K^{(\lambda)}:=\mathrm{SigPDE}(\lambda M)$. By (H),
$K^{(\lambda)}=\sum_k\lambda^k K_k$, hence
$$\sum_i w_i\,K^{(\lambda_i)}=\sum_k\Big(\underbrace{\textstyle\sum_i w_i\lambda_i^{k}}_{=\;\varphi(k)}\Big)K_k
\;\Longleftrightarrow\;\varphi(k)=\sum_i w_i\lambda_i^{k}. \tag{D}$$
So a geometric **mixture of dilations** *is* an order-weighting. The code realizes the left side
directly: $K=\sum_i w_i\,\texttt{sigpde\_wavefront}(\lambda_i M)$
(`ksig/generalized.py:LearnedPhiSignaturePDEKernel._combine`), and reports the right side as
$\varphi(k)=\sum_i w_i\lambda_i^k$ (`ksig/generalized.py:LearnedPhiSignaturePDEKernel.phi`).

### 0.5 The Goursat PDE (untruncated engine)
The untruncated kernel solves the Goursat problem $\partial_s\partial_t K=\langle\dot x_s,\dot y_t\rangle\,K$,
$K(\cdot,0)=K(0,\cdot)=1$. KSig_aurora discretizes it with a second-order scheme on the
antidiagonals: with $m=M_{ab}$ and padded boundary $H\equiv1$,
$$H_{i+1,j+1}=(H_{i+1,j}+H_{i,j+1})\Big(1+\tfrac m2+\tfrac{m^2}{12}\Big)-H_{i,j}\Big(1-\tfrac{m^2}{12}\Big),
\qquad K=H_{L,L}. \tag{G}$$
Two implementations of (G) coexist and are bit-identical in the forward pass:
`ksig/algorithms.py:signature_kern_pde` (in-place, used by `SignaturePDEKernel`) and
`ksig/generalized.py:sigpde_wavefront` (out-of-place via `index_put`, **autograd-safe**, used by the
learned dilation kernel). The PDE identity (G)$\Leftrightarrow$untruncated signature kernel is the
textbook (Salvi et al.) result; the code implements only the recurrence.

### 0.6 The two normalization conventions
Both produce a unit diagonal but are different objects.

- **Normalize-once** (the GSK convention), $d=\operatorname{diag}K_\varphi$:
  $$\hat K_\varphi=K_\varphi\oslash\sqrt{d\,d^{\top}},\qquad
  \texttt{\_normalize}(K,d_X,d_Y)=K\oslash\sqrt{d_X d_Y^{\top}}$$
  (`ksig/generalized.py:_normalize`, with a $10^{-12}$ floor). Applied **once, after** mixing.
- **Per-level-then-average** (`ksig/kernels.py:SignatureKernel._K`): each level is whitened by its
  *own* per-pair diagonal and the whitened levels are averaged,
  $$\hat K_{\mathrm{EX}}=\frac1{N+1}\sum_{k=0}^N \frac{K_k}{\sqrt{(K_k)_{xx}(K_k)_{yy}}}.$$
  In code this is the division `K / (K_Xd_sqrt[...,:,None]*K_Xd_sqrt[...,None,:])` over a tensor that
  still carries the level axis, followed by `torch.mean(K, dim=0)`
  (`ksig/kernels.py:SignatureKernel._K`). This is a per-pair, **per-level** whitening and is **not**
  expressible as any global $\varphi$ in $\hat K_\varphi$.

### 0.7 The learned-$\varphi$ fitting contract
Learned kernels are two-phase: `fit_phi` learns $\varphi$ on the **train** set, then the object is
frozen and behaves as an ordinary precomputed kernel. The objective is centered kernel–target
alignment against the rank-1 label kernel $yy^\top$,
$$\mathcal L(\varphi)=-\frac{\langle H\hat K_\varphi H,\;H\,yy^\top H\rangle}{\lVert H\hat K_\varphi H\rVert\,\lVert H\,yy^\top H\rVert},
\qquad H=I-\tfrac1n\mathbf 1\mathbf1^\top$$
(`ksig/generalized.py:_cka_loss`), with $y\mapsto 2y-1$ for classification and $y\mapsto y-\bar y$
for regression (`ksig/generalized.py:_target`). Invariants enforced in code: $\varphi(0)\equiv1$
(pinned, never learned), $\varphi\ge0$ (so $K_\varphi$ is a conic combination of PSD level kernels,
hence **PSD**), and normalization applied **once after** mixing. Optimization is Adam.

---

## 1. `sig-L1` — the level-1 kernel

1. **Object.** $\varphi=e_1$ (i.e. $\varphi(0)=0,\varphi(1)=1$), depth $N$, normalize-once. A
   single-level specialization of §0.3.
2. **How computed.** `GeneralSignatureKernel(phi="level_one")` builds a
   `WeightedSignatureKernel` with the literal vector `phi_fixed = e_1`
   (`ksig/generalized.py:GeneralSignatureKernel.__init__`); the kernel is
   $\hat K=K_1\oslash\sqrt{(K_1)_{xx}(K_1)_{yy}}$ with $K_1$ from §0.2. The literal-vector path is
   needed because $\varphi(0)=0$ exactly, which `softplus` cannot produce.
3. **Iteratively estimated.** Nothing. $\sigma$ (RBF bandwidth) is a caller-supplied median
   heuristic, not learned by the kernel.
4. **Validity.** $K_1$ is a Gram in $\mathcal H_\kappa$ (PSD); the cosine normalization is a
   diagonal congruence and preserves PSD.
5. **Inductive bias.** Sees only net displacement $\propto S^1=X_T-X_0$; order-blind. Isolated by
   `D_disp` (it suffices) and `D_area` (it is at chance — area is invisible) in
   `tests/test_signature_kernel_inductive_bias.py`.
6. **Provenance.** **(P/X)** — a one-level configuration in the author’s GSK façade over the
   **KSig-ported** truncated recursion (Király & Oberhauser 2019; `docs/TORCH_PORT.md`, §0.2).

## 2. `sig-TRUNC-φ1` — the truncated ordinary kernel

1. **Object.** $\varphi\equiv1$, depth $N\ge0$, normalize-once: $\hat K=\big(\sum_{k=0}^N K_k\big)\oslash\sqrt{d d^\top}$. The correct $\varphi\equiv1$ baseline. **Level-0 (`truncation=0`)** returns the level-0 constant only — i.e. an all-ones normalized Gram, not $K_0+K_1$ (audit F1, now fixed; negative depth is rejected).
2. **How computed.** `GeneralSignatureKernel(phi="const", truncation=N)` →
   `WeightedSignatureKernel` with `phi_fixed = (1,\dots,1)`; levels from
   `signature_kern_first_order`, summed and normalized once
   (`ksig/generalized.py:WeightedSignatureKernel.__call__`).
3. **Iteratively estimated.** Nothing (a constant $\varphi$ cancels in normalize-once, so the
   result is independent of the constant’s value).
4. **Validity.** Conic ($\varphi\ge0$) sum of PSD level kernels → PSD; complexity $O(n^2L^2N)$.
5. **Inductive bias.** The full depth-$N$ discrete signature, all orders weighted equally.
6. **Provenance.** **(P)** — port of KSig’s truncated kernel-trick signature kernel
   (Király & Oberhauser 2019), cupy/CUDA → torch/SYCL (`ksig/algorithms.py:signature_kern_first_order`,
   `docs/TORCH_PORT.md`). The φ≡1, normalize-once *framing* as the baseline GSK member is the
   author’s.

## 3. `sig-PDE` — the untruncated ordinary kernel

1. **Object.** $\varphi\equiv1$, **untruncated**, normalize-once.
2. **How computed.** `GeneralSignatureKernel(phi="const", truncation=None)` →
   `ksig/kernels.py:SignaturePDEKernel`, which runs the Goursat recurrence (G)
   (`ksig/algorithms.py:signature_kern_pde`) and normalizes once
   (`SignatureKernel._K`, single-solve branch — no level axis, so the per-level average does not
   fire).
3. **Iteratively estimated.** Nothing.
4. **Validity.** PSD by construction (a genuine signature kernel); the wavefront is $O(n^2L^2)$.
   Requires the RBF static kernel ($|m|\le2$); the scheme is stable only for $|m|\lesssim1$ per cell.
5. **Inductive bias.** All orders to $\infty$ — sensitive to the high-level tail; cf. `D_scale`,
   `D_lowsig`.
   **Discrepancy (flagged):** `sig-PDE` is **not** the $N\to\infty$ limit of `sig-TRUNC-φ1`. The
   truncated discrete kernel is Cauchy in $N$ but converges to its own discrete-untruncated limit,
   which differs from the continuous Goursat solve by $\sim0.04$ (off-diagonal abs-mean on the
   diagnostic paths) — the discrete-tensor vs continuous-PDE discretization gap. So a `sig-PDE − sig-TRUNC-φ1` contrast is truncation **plus** solver, not a pure depth-tail.
6. **Provenance.** **(L)** — implementation of the signature kernel as the solution of a Goursat PDE
   (Salvi, Cass, Foster, Lyons & Yang 2021), with the wavefront discretization (G) ported to
   torch/SYCL (`ksig/algorithms.py:signature_kern_pde`; `docs/TORCH_PORT.md` §4.2).

## 4. `sig-Wphi` — truncated, learned level weights

1. **Object.** $\varphi(0)=1$ pinned, $\varphi(1{:}N)=\operatorname{softplus}(\theta)\ge0$, depth
   $N$, normalize-once: $\hat K_\varphi=\big(\sum_k\varphi(k)K_k\big)\oslash\sqrt{dd^\top}$.
2. **How computed.** `WeightedSignatureKernel`: cache $K_0,\dots,K_N$ once via
   `signature_kern_first_order`, then $\sum_k\varphi(k)K_k$ normalized once
   (`ksig/generalized.py:WeightedSignatureKernel.__call__`, `._levels`).
3. **Iteratively estimated.** $\theta\in\mathbb R^{N}$ by Adam on $\mathcal L$ of §0.7
   (`fit_phi`); $\varphi=\operatorname{cat}(1,\operatorname{softplus}(\theta))$, so $\theta_0$
   carries no gradient and $\varphi(0)\equiv1$. The level Grams are constants in $\theta$, so the
   **unnormalized** alignment is convex over $\varphi\ge0$ (the normalized objective used by default
   is not — see Caveats).
4. **Validity.** $\varphi\ge0$ → conic combination of PSD level kernels → PSD.
5. **Inductive bias.** Arbitrary nonnegative — and possibly **non-monotone** — level profile;
   uniquely able to peak (suppress levels 1,3 and keep 2). Isolated by `D_peak`.
6. **Provenance.** **(X)** — author’s extension: the order-weighting $\varphi$ of the General
   Signature Kernel (Cass, Lyons & Xu 2021) made **learnable** by kernel-target alignment, over the
   KSig-ported level Grams. Design note `LEARNABLE_PHI_SIGNATURE_KERNELS.md`; §0.3, §0.7 here.

## 5. `sig-PDEphi` — untruncated, learned dilation mixture

1. **Object.** $\varphi(k)=\sum_{i=1}^m w_i\lambda_i^{k}$ via (D), **untruncated**, normalize-once;
   $\varphi(0)=\sum_i w_i=1$ holds automatically (softmax).
2. **How computed.** $K=\sum_i w_i\,\texttt{sigpde\_wavefront}(\lambda_i M)$ normalized once
   (`ksig/generalized.py:LearnedPhiSignaturePDEKernel._combine`), with
   $w=\operatorname{softmax}(\texttt{raw\_w})$, $\lambda_i=\lambda_{\max}\sigma(\texttt{raw\_l})\in(0,\lambda_{\max})$,
   $\lambda_{\max}=0.5$ (`__init__`). `sigpde_wavefront` is the out-of-place autograd-safe form of (G).
3. **Iteratively estimated.** The mixture, by `fit_phi`:
   - **default `learn_lam=False`** — freeze the $\lambda$-grid, precompute the $m$ per-node Grams
     $\{\texttt{sigpde\_wavefront}(\lambda_i M)\}$ once, and learn only $w=\operatorname{softmax}(\texttt{raw\_w})$.
     This is a **convex** quadrature-weight problem (no backprop through the wavefront).
   - **`learn_lam=True`** — also learn the node positions $\lambda_i$ by backprop through the
     autograd-safe wavefront (optionally gradient-checkpointed).
4. **Validity.** Each $\texttt{sigpde\_wavefront}(\lambda_i M)$ is PSD; $w\ge0$ → conic combination
   → PSD. Cost $O(n^2L^2)$ per node $\times\,m$. A stability gate NaNs the Gram (with a warning,
   never raising) when $\lambda_{\max}\max|m|>1$ (`ksig/generalized.py:_stability_ok`).
5. **Inductive bias.** A **decaying** (completely monotone) level profile — soft tail suppression;
   `D_lowsig` (beats uniform PDE on tail noise) and `D_scale` (cannot up-weight a tail signal — the
   clamp limit).
6. **Provenance.** **(X)** — author’s extension: a *learnable* dilation-mixture realization of the
   General Signature Kernel via identity (D) (Cass, Lyons & Xu 2021, dilation/Cor. 2.10), built over
   the Goursat-PDE engine (Salvi et al. 2021) with an **out-of-place, autograd-safe** reimplementation
   of the ported solver (`ksig/generalized.py:sigpde_wavefront`). Design note
   `LEARNABLE_PHI_SIGNATURE_KERNELS.md`; §0.4–0.5, §0.7 here.

## 6. `sig-EXACT` — per-level-normalized average

1. **Object.** $\varphi\equiv1$, depth $N$, **per-level** normalization:
   $\hat K_{\mathrm{EX}}=\frac1{N+1}\sum_{k}K_k\oslash\sqrt{(K_k)_{xx}(K_k)_{yy}}$ (§0.6). Not a
   member of the global-$\varphi$ family.
2. **How computed.** `GeneralSignatureKernel(phi="const", normalize="per_level")` delegates to
   `ksig/kernels.py:SignatureKernel(normalize=True)`: levels from `signature_kern`, per-level whiten,
   then `torch.mean(dim=0)` (`SignatureKernel._K`).
3. **Iteratively estimated.** Nothing.
4. **Validity.** Each whitened level kernel is PSD (diagonal congruence of a Gram); their average is
   PSD.
5. **Inductive bias.** Per-path, per-level whitening — recovers a level’s *direction* across wide
   per-path magnitude. Isolated by `D_perlevel`; see the degenerate-level caveat below.
6. **Provenance.** **(P)** — port of KSig’s `SignatureKernel(normalize=True)` and its native
   per-level-normalize-then-average convention (Király & Oberhauser 2019), cupy/CUDA → torch/SYCL
   (`ksig/kernels.py:SignatureKernel._K`; `docs/TORCH_PORT.md`).

---

## 7. Config → kernel map

`GeneralSignatureKernel(phi, truncation, normalize, static, bw, m_nodes, lam_max, …)`
(`ksig/generalized.py:GeneralSignatureKernel.__init__`) dispatches the six columns; no engine is
reimplemented in the façade.

| column | `phi` | `truncation` | `normalize` | engine (delegated) | origin |
|---|---|---|---|---|---|
| `sig-L1` | `level_one` | $N$ | `once` | `WeightedSignatureKernel` (`phi_fixed=e_1`) | P/X |
| `sig-TRUNC-φ1` | `const` | $N$ | `once` | `WeightedSignatureKernel` (`phi_fixed=\mathbf1`) | P |
| `sig-PDE` | `const` | `None` | `once` | `ksig.kernels.SignaturePDEKernel` | L |
| `sig-Wphi` | `free` | $N$ | `once` | `WeightedSignatureKernel` (fitted) | X |
| `sig-PDEphi` | `dilation` | `None` | `once` | `LearnedPhiSignaturePDEKernel` (fitted) | X |
| `sig-EXACT` | `const` | $N$ | `per_level` | `ksig.kernels.SignatureKernel(normalize=True)` | P |

Origin key: **P** ported from KSig (torch/SYCL); **L** literature implementation; **X** author’s
learnable extension.

Guard: `truncation=None` with `static="linear"` is rejected — the Goursat scheme needs bounded
increments (`ksig/generalized.py:GeneralSignatureKernel.__init__`).

---

## 8. Caveats (each: what, and why)

- **The dilation cone is completely monotone.** $\varphi(k)=\sum_i w_i\lambda_i^k$ with
  $w,\lambda\ge0$ is a Stieltjes/completely-monotone sequence: it can represent any **monotone**
  $\varphi$ but **not** a non-monotone peak — free weights (`sig-Wphi`) can. Demonstrated end-to-end
  on `D_peak` and at the coefficient level by `test_dilation_cone_monotone_vs_free_nonmonotone`
  (`tests/test_signature_kernel_inductive_bias.py`).
- **The stability clamp restricts `sig-PDEphi` to decaying $\varphi$.** With $\lambda_{\max}=0.5$
  and $|m|\le2$ for RBF, $|\lambda m|\le1$, so $\lambda_i<1$ and $\varphi(k)$ necessarily decays —
  it cannot up-weight the high-level tail (the `D_scale` clamp chain $\text{Wphi}>\text{PDE}>\text{PDEphi}$).
- **Per-level normalization is fragile to degenerate levels.** If a level’s self-magnitude is
  $\approx0$ (e.g. closed loops give $\langle S^1,S^1\rangle\approx0$), the $10^{-12}$-floored
  whitening turns that level into high-variance noise that dominates the uniform average. Observed
  directly: on `D_area` the level-2 sub-kernel aligns at $0.92$ while the averaged `sig-EXACT` sits
  at chance (`test_perlevel_path_whitens_each_level`).
- **PSD is proven; characteristicness is conjectured.** PSD holds for every configuration by conic
  combination (above). Characteristicness is plausible for the untruncated kernel with strictly
  positive $\varphi$ and a characteristic static ($\sigma$-RBF), but is **not** proven in this
  repository; any truncated kernel is non-characteristic by truncation alone (finitely many tensor
  levels), independent of whether $\varphi$ zeros a level.
- **Untruncated kernels require the RBF static.** The Goursat scheme (G) needs bounded increments
  ($|\lambda m|\lesssim1$); unbounded linear increments diverge, so `sig-PDE`/`sig-PDEphi` are
  RBF-only (enforced by the §7 guard).

---

## 9. Implementation status (audit remediation)

Following the paper-to-code audit (`docs/AUDIT_REPORT.md` → `docs/AUDIT_RESPONSE.md`), these
call-contract guarantees are now **implemented and regression-tested**
(`tests/paper_contracts/`, summarized in `docs/report.md`):

- **Level-0 contract (F1).** `truncation=0` returns the level-0 constant only (all-ones normalized
  Gram); negative depth is rejected. Affects `sig-TRUNC-φ1`, `sig-Wphi`, `sig-L1`.
- **`diag=True` everywhere (F2).** The delegated `sig-PDE` and `sig-EXACT` engines now honor
  `diag=True` and return shape $(n,)$ — they no longer compute the full $O(n^2)$ Gram.
- **Dtype/device preservation (F3).** A float64 input yields a float64 kernel (and learning,
  `fit_phi`, runs in float64); torch inputs keep their device and autograd graph. No silent float32
  host hop.
- **RFSF per-level independence (F6)** and **sparse-projection distribution (F7)** are fixed in the
  random-feature maps (`ksig/kernels.py`, `ksig/projections.py`); see `docs/report.md`.

Still sequenced (tracked as `xfail` until landed): the PDE stability policy (plain `sig-PDE` warns
but is never auto-clamped; `sig-PDEphi` reports the effective clamped λ), the differentiable-input
return contract, removal of the dead `freeze_phi0` argument, and the
`LEARNABLE_PHI_SIGNATURE_KERNELS.md` design note referenced below.

---

## References

Implementations (P / L) and the framework the extensions (X) build on:

- **Király & Oberhauser (2019)**, *Kernels for sequentially ordered data*, JMLR — the kernel-trick
  truncated signature kernel and the first-order level recursion (§0.2); ported in
  `ksig/algorithms.py:signature_kern_first_order`, `ksig/kernels.py:SignatureKernel`. Backs
  `sig-TRUNC-φ1`, `sig-EXACT`, `sig-L1`.
- **Salvi, Cass, Foster, Lyons & Yang (2021)**, *The Signature Kernel is the solution of a Goursat
  PDE*, SIAM J. Math. Data Sci. — the untruncated kernel as PDE (G), §0.5; implemented in
  `ksig/algorithms.py:signature_kern_pde`. Backs `sig-PDE`, and the engine reused by `sig-PDEphi`.
- **Cass, Lyons & Xu (2021)**, *General/weighted signature kernels* — the order-weighting
  $K_\varphi=\sum_k\varphi(k)K_k$ and the dilation identity (D), Cor. 2.10 (§0.3–0.4). The framework
  the author’s learnable kernels specialize.
- **Port:** `docs/TORCH_PORT.md` — the cupy/CUDA → torch/SYCL port (device/dtype policy, the
  antidiagonal wavefronts) underlying every (P)/(L) kernel above.
- **Extension (X):** `LEARNABLE_PHI_SIGNATURE_KERNELS.md` — design and theory for the *learnable*
  General Signature Kernel (`sig-Wphi`, `sig-PDEphi`): the two-phase `fit_phi`, the convex
  weights-only default, the PSD/normalization invariants, and the stability clamp.

## Summary

**Five** of the six kernels are members of the normalize-once General Signature Kernel
$K_\varphi=\sum_k\varphi(k)K_k$, under a choice of order-weighting $\varphi$, truncation depth, and
normalization: `sig-L1` ($\varphi=e_1$), `sig-TRUNC-φ1` and `sig-PDE` ($\varphi\equiv1$, truncated
vs Goursat-untruncated) — **fixed** — and `sig-Wphi` (free $\varphi\ge0$) and `sig-PDEphi` (a
geometric mixture $\varphi(k)=\sum_i w_i\lambda_i^k$ via the dilation identity) — **learned** by
aligning the Gram to $yy^\top$ on the training set. The sixth, **`sig-EXACT`**, is a *legacy
per-level-normalize-then-average probe* that is **outside** the normalize-once $K_\varphi$ family
(§0.6, §6) — it is kept for its per-level-whitening inductive bias, not as a $\varphi$ choice.
For the five members, choosing a kernel is choosing $(\varphi,\text{truncation},\text{normalization})$
and learning a kernel is learning $\varphi$ — under homogeneity (H), nothing else changes.
