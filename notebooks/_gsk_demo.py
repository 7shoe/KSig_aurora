"""Shared helpers for the General-Signature-Kernel demo notebooks (``07_*`` and
``X_learned_kernel``).

This is the statistical-demo companion to :mod:`_nbtools` (which serves the six
throughput/correctness notebooks ``01``–``06``). Where ``_nbtools`` is about
*wall time* and *bit-for-bit* CUDA-reference agreement, this module is about
*inductive bias*: it ships the level-localized data-generating processes (DGPs)
that plant a class signal at one structural locus of a path, plus the metrics
and plots that show **which signature kernel reads that locus** — and, for the
learnable kernels, whether the learned order-weighting ``phi(k)`` **recovers the
planted order**.

Design mirrors ``_nbtools``: NumPy is the only hard import; ``torch`` / ``ksig``
/ ``matplotlib`` are imported lazily inside the functions that need them, so
importing this module never fails on a box that is missing one of them.

The DGPs and the kernel registry are a portable lift of
``tests/test_signature_kernel_inductive_bias.py`` (the regression that pins the
CKA confusion matrix), so a notebook reproduces the numbers the test asserts.

Glossary of the six configurations (all one object,
``ksig.generalized.GeneralSignatureKernel``, by configuration — see
``docs/SIGNATURE_KERNELS.md``):

==============  ==================================================  ===========
column          configuration                                       learns phi?
==============  ==================================================  ===========
``sig-L1``      ``phi='level_one', truncation=N``  (phi = e_1)       no
``sig-TRUNC``   ``phi='const',     truncation=N``  (phi == 1)        no
``sig-PDE``     ``phi='const',     truncation=None`` (Goursat)       no
``sig-Wphi``    ``phi='free',      truncation=N``  (free weights)    YES
``sig-PDEphi``  ``phi='dilation',  truncation=None`` (dilation mix)  YES
``sig-EXACT``   ``phi='const', N, normalize='per_level'``            no
==============  ==================================================  ===========
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Configuration (kept in lock-step with the inductive-bias regression test so a
# notebook reproduces the asserted CKA matrix).
# ---------------------------------------------------------------------------
L, D = 22, 2                       # path length / channels for the DGPs
N_LEVELS, M_NODES, LAM_MAX = 3, 6, 0.5   # truncation depth; dilation grid; stability clamp
FIT_STEPS = 200                    # Adam steps for the learned phi
N_TRAIN = N_TEST = 120
CHANCE = 0.07                      # CKA below this == order/level-blind (no signal read)


# ---------------------------------------------------------------------------
# Path primitives. A DGP returns trajectories [n, L, d]; the kernel takes the
# (second-differenced) increments internally, so we hand it positions.
# ---------------------------------------------------------------------------
def _path(incr):
    """increments [n, L-1, d] -> path [n, L, d] starting at the origin."""
    z = np.zeros((incr.shape[0], 1, incr.shape[2]), np.float32)
    return np.concatenate([z, np.cumsum(incr, 1)], 1).astype(np.float32)


def _closed_loop(radius):
    """An exactly closed circle (t=2pi coincides with t=0): zero net displacement,
    nonzero signed area -> a pure LEVEL-2 carrier."""
    t = np.linspace(0, 2 * np.pi, L)
    return np.stack([radius * np.cos(t), radius * np.sin(t)], 1).astype(np.float32)


def _microloop(eps, cls):
    """Confined LEVEL-2 primitive: a small oriented loop, signed area = +-pi*eps^2
    with sign set by ``cls``. Small eps keeps level-3 leakage (~eps^3) negligible."""
    t = np.linspace(0, 2 * np.pi, L)
    o = 2 * cls - 1
    return (eps * np.stack([np.cos(t), o * np.sin(t)], 1)).astype(np.float32)


def _figure8(amp, freq, phase):
    """TAIL primitive: a high-frequency figure-eight (1:2 Lissajous). ZERO net
    signed area (invisible to level 2); its energy lives in level >= 3."""
    t = np.linspace(0, 2 * np.pi, L)
    return (amp * np.stack([np.sin(freq * t + phase),
                            0.5 * np.sin(2 * freq * t + phase)], 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# The six inductive-bias DGPs. Each plants class signal at ONE structural locus;
# the matched kernel should attain the highest out-of-sample CKA. Verbatim port
# of the regression test so the notebook == the asserted matrix.
# ---------------------------------------------------------------------------
def make_disp(n, seed):
    """LEVEL 1. Class = net displacement along axis 0 (a straight drift adds NO
    area), so level>=2 statistics are class-independent noise."""
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n)
    incr = rng.normal(0, 0.4, (n, L - 1, D)).astype(np.float32)
    incr[:, :, 0] += (2 * y - 1)[:, None] * 1.6 / (L - 1)
    return _path(incr), y


def make_area(n, seed):
    """LEVEL 2 (ORDER), isolated. Closed loop; class 1 = time-REVERSED loop ->
    flips the signed area while preserving the endpoint and the point multiset.
    Order-blind reps (level-1 / set / mean) are class-invariant by construction."""
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n)
    X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        loop = _closed_loop(0.5 + rng.random())
        if y[i] == 1:
            loop = loop[::-1].copy()
        X[i] = loop + rng.normal(0, 0.5, (1, D)).astype(np.float32)
    return X, y


def make_peak(n, seed):
    """PEAK at level 2. Level-2 area = class (signal); level 1 = large class-indep
    drift (noise); level 3 = class-indep high-freq wiggle (noise). The optimal phi
    suppresses 1 & 3 and keeps 2 -> a NON-MONOTONE peak: representable by free
    weights, forbidden to the completely-monotone dilation cone."""
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n)
    t = np.linspace(0, 2 * np.pi, L); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        loop = _closed_loop(0.6)
        if y[i] == 1:
            loop = loop[::-1].copy()
        disp = rng.normal(0, 1.0, D).astype(np.float32)
        loop = loop + np.linspace(0, 1, L)[:, None].astype(np.float32) * disp[None, :] * 3.0
        wig = np.zeros((L, D), np.float32)
        for f, p in zip((7, 9), rng.uniform(0, 6, 2)):
            wig[:, 0] += 0.25 * np.sin(f * t + p); wig[:, 1] += 0.25 * np.cos(f * t + p)
        X[i] = loop + wig
    return X, y


def make_scale(n, seed):
    """TAIL SIGNAL + dilation-clamp limit. A class-INDEP micro-loop + drift fixes
    the low levels; a figure-eight (energy in level>=3) has its AMPLITUDE set by
    the class, so the signal lives in the tail. Free weights up-weight the
    boundary level; the lam_max=0.5 dilation cone is forced decaying and cannot
    reach the tail -> the clamp chain  Wphi > PDE > PDEphi."""
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        base = _microloop(0.7, int(rng.integers(0, 2))) + \
            np.linspace(0, 1, L)[:, None].astype(np.float32) * rng.normal(0, 0.6, (1, D)).astype(np.float32)
        X[i] = base + _figure8(0.4 + 0.6 * y[i], 11, float(rng.uniform(0, 6)))
    return X, y


def make_lowsig(n, seed):
    """LOW-LEVEL signal + TAIL NOISE. Level-2 area = class; a class-INDEP high-freq
    wiggle injects noise into HIGH levels. PDEphi's soft-decay down-weights the
    tail noise and beats uniform untruncated PDE."""
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n)
    t = np.linspace(0, 2 * np.pi, L); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        loop = _closed_loop(0.8)
        if y[i] == 1:
            loop = loop[::-1].copy()
        wig = np.zeros((L, D), np.float32)
        for f in (12, 19, 27):
            p0, p1 = rng.uniform(0, 6, 2)
            wig[:, 0] += 0.5 * np.sin(f * t + p0); wig[:, 1] += 0.5 * np.cos(f * t + p1)
        X[i] = loop + wig
    return X, y


def make_perlevel(n, seed):
    """PER-LEVEL NORMALIZATION. Class = confined level-2 micro-loop area sign. Each
    path is scaled by a WIDE per-path magnitude AND carries a LARGE class-INDEP
    level-1 drift, so the GLOBAL self-norm is dominated by per-path-random level-1
    energy. Only per-path PER-LEVEL whitening (sig-EXACT) recovers the area sign."""
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        mag = float(np.exp(rng.normal(0, 1.3)))
        drift = np.linspace(0, 1, L)[:, None].astype(np.float32) * \
            rng.normal(0, 1.0, (1, D)).astype(np.float32) * 3.0
        X[i] = mag * (_microloop(0.7, int(y[i])) + drift)
    return X, y


DATASETS = {"D_disp": make_disp, "D_area": make_area, "D_peak": make_peak,
            "D_scale": make_scale, "D_lowsig": make_lowsig, "D_perlevel": make_perlevel}

# One-line "what is planted, and who should win" for each DGP (notebook captions).
DATASET_DOC = {
    "D_disp":     "level-1 net displacement     -> order-blind suffices (no order gain)",
    "D_area":     "level-2 signed area (order)  -> every order-aware kernel; blind reps at chance",
    "D_peak":     "level-2 peak, levels 1&3 noise -> only free weights (sig-Wphi) can peak",
    "D_scale":    "level>=3 tail amplitude      -> clamp chain  Wphi > PDE > PDEphi",
    "D_lowsig":   "level-2 signal + tail noise  -> soft-decay PDEphi > uniform PDE",
    "D_perlevel": "level-2 under per-path scale -> only per-level whitening (sig-EXACT)",
}


# ---------------------------------------------------------------------------
# Planted-ORDER DGPs for the learnable notebook: class signal localized at a
# CHOSEN signature level, noise elsewhere -> the learned phi should peak there.
# ---------------------------------------------------------------------------
def make_planted(level, n, seed):
    """Class signal confined to signature ``level`` (1, 2 or 3); everything else is
    class-independent. Used to test whether the learned ``phi(k)`` recovers the
    order. Tuned so the free-weight peak lands exactly on ``level`` at
    ``truncation=N_LEVELS``.

    * level 1 -- a class-dependent straight drift (net displacement; zero area).
    * level 2 -- a class-oriented closed loop (signed area; zero displacement).
    * level 3 -- a class-time-reversed figure-eight (zero area; energy in the
      odd >=3 tail), over a class-independent micro-loop.
    """
    if level not in (1, 2, 3):
        raise ValueError("planted level must be 1, 2 or 3")
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        if level == 1:
            incr = rng.normal(0, 0.4, (L - 1, D)).astype(np.float32)
            incr[:, 0] += (2 * y[i] - 1) * 1.6 / (L - 1)
            X[i] = _path(incr[None])[0]
        elif level == 2:
            loop = _closed_loop(0.6)
            if y[i] == 1:
                loop = loop[::-1].copy()
            X[i] = loop + rng.normal(0, 0.5, (1, D)).astype(np.float32)
        else:  # level == 3
            base = _microloop(0.5, int(rng.integers(0, 2)))
            f8 = _figure8(1.0, 9, float(rng.uniform(0, 6)))
            if y[i] == 1:
                f8 = f8[::-1].copy()
            X[i] = base + f8
    return X, y


def split(maker, seed, n_train=N_TRAIN, n_test=N_TEST, **kw):
    """(Xtr, ytr, Xte, yte) train/test split from a DGP, disjoint seeds."""
    Xtr, ytr = maker(n_train, seed, **kw) if kw else maker(n_train, seed)
    Xte, yte = maker(n_test, seed + 1000, **kw) if kw else maker(n_test, seed + 1000)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Metrics and order-blind reference kernels.
# ---------------------------------------------------------------------------
def median_bw(X, n_sub=1500, seed=0):
    """The median-distance heuristic for the RBF static bandwidth."""
    pts = X.reshape(-1, X.shape[-1]); rng = np.random.default_rng(seed)
    s = pts[rng.choice(len(pts), min(n_sub, len(pts)), replace=False)]
    d2 = ((s[:, None] - s[None]) ** 2).sum(-1)
    return float(np.sqrt(np.median(d2[np.triu_indices(len(s), 1)]) + 1e-12))


def cka(K, y):
    """Centered kernel-target alignment of K with the rank-1 label kernel; in
    [-1, 1]. This is the same objective ``fit_phi`` maximizes (``_cka_loss``), so
    it is the natural read-out of "does the kernel see the class"."""
    K = np.asarray(K, float); n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H
    yc = y.astype(float) - y.mean(); Yc = np.outer(yc, yc)
    return float((Kc * Yc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-12))


def pooled_rbf_gram(X, bw):
    """Order-blind reference: RBF on the time-pooled mean state (sees only the
    centroid, no order)."""
    v = X.mean(1); d2 = ((v[:, None] - v[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * bw * bw))


def kme_rbf_gram(X, bw):
    """Order-blind reference: kernel mean embedding (RBF averaged over the L x L
    static block -- the set of states, order discarded)."""
    n = len(X); K = np.empty((n, n))
    for i in range(n):
        d2 = ((X[i][None, :, None, :] - X[:, None, :, :]) ** 2).sum(-1)
        K[i] = np.exp(-d2 / (2 * bw * bw)).mean(axis=(1, 2))
    return K


# ---------------------------------------------------------------------------
# The GeneralSignatureKernel registry (the six columns + two order-blind refs).
# ---------------------------------------------------------------------------
def gsk(phi, truncation, normalize="once", bw=1.0, m_nodes=M_NODES,
        lam_max=LAM_MAX, dev="cpu"):
    """Construct one GeneralSignatureKernel column (lazy ksig import)."""
    from ksig.generalized import GeneralSignatureKernel
    return GeneralSignatureKernel(phi=phi, truncation=truncation, normalize=normalize,
                                  static="rbf", bw=bw, m_nodes=m_nodes, lam_max=lam_max,
                                  dev=dev)


def kernel_registry(n_levels=N_LEVELS):
    """Name -> spec for the six GSK columns plus the two order-blind references.
    A 'sig' spec carries a ``build(bw)`` and a ``learned`` flag; a 'ref' spec a
    ``gram(X, bw)``."""
    def _sig(phi, trunc, norm):
        return lambda bw: gsk(phi, trunc, norm, bw=bw)
    return {
        "sig-L1":     dict(kind="sig", learned=False, build=_sig("level_one", n_levels, "once")),
        "sig-TRUNC":  dict(kind="sig", learned=False, build=_sig("const",     n_levels, "once")),
        "sig-PDE":    dict(kind="sig", learned=False, build=_sig("const",     None,     "once")),
        "sig-Wphi":   dict(kind="sig", learned=True,  build=_sig("free",      n_levels, "once")),
        "sig-PDEphi": dict(kind="sig", learned=True,  build=_sig("dilation",  None,     "once")),
        "sig-EXACT":  dict(kind="sig", learned=False, build=_sig("const",     n_levels, "per_level")),
        "pooled-RBF": dict(kind="ref", learned=False, gram=pooled_rbf_gram),
        "kme-RBF":    dict(kind="ref", learned=False, gram=kme_rbf_gram),
    }


def _score(spec, Xtr, ytr, Xte, yte, bw, steps=FIT_STEPS):
    """Out-of-sample CKA of one kernel: learned columns fit phi on TRAIN only,
    then score on TEST (the two-phase contract)."""
    if spec["kind"] == "ref":
        return cka(spec["gram"](Xte, bw), yte)
    try:
        import torch
        torch.manual_seed(0)
    except Exception:
        pass
    ker = spec["build"](bw)
    if spec["learned"]:
        ker.fit_phi(Xtr, ytr, task="classification", steps=steps)
    return cka(np.asarray(ker(Xte)), yte)


def confusion_matrix(datasets=None, kernels=None, steps=FIT_STEPS, progress=True):
    """Out-of-sample CKA of every kernel on every DGP -> ``{ds: {kernel: cka}}``.

    Reproduces the matrix asserted in
    ``tests/test_signature_kernel_inductive_bias.py``. Pass ``progress=True`` to
    print each row as it finishes (the full 6x8 grid is ~30 s on CPU)."""
    datasets = datasets or list(DATASETS)
    kernels = kernels or kernel_registry()
    out = {}
    for ds in datasets:
        Xtr, ytr, Xte, yte = split(DATASETS[ds], seed=list(DATASETS).index(ds))
        bw = median_bw(Xtr)
        out[ds] = {name: _score(spec, Xtr, ytr, Xte, yte, bw, steps)
                   for name, spec in kernels.items()}
        if progress:
            row = "  ".join(f"{k}={v:+.3f}" for k, v in out[ds].items())
            print(f"[{ds}] {row}")
    return out


def winners(matrix, margin=0.04):
    """For each DGP, the kernel(s) within ``margin`` of the best CKA -> the
    'who reads this locus' annotation for the heatmap."""
    out = {}
    for ds, row in matrix.items():
        best = max(row.values())
        out[ds] = [k for k, v in row.items() if v >= best - margin]
    return out


# ---------------------------------------------------------------------------
# Plotting (lazy matplotlib).
# ---------------------------------------------------------------------------
def plot_cka_heatmap(matrix, title="out-of-sample CKA  (kernel reads the planted locus)",
                     ax=None, annotate=True):
    """Render the DGP x kernel CKA matrix as a heatmap, bordering the per-row
    winner. Green = strong alignment, white = chance."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    rows = list(matrix); cols = list(matrix[rows[0]])
    M = np.array([[matrix[r][c] for c in cols] for r in rows])
    if ax is None:
        _, ax = plt.subplots(figsize=(1.05 * len(cols) + 1.5, 0.6 * len(rows) + 1.5))
    norm = TwoSlopeNorm(vmin=min(-0.05, M.min()), vcenter=CHANCE, vmax=max(0.5, M.max()))
    im = ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    win = winners(matrix)
    for i, r in enumerate(rows):
        best_j = int(np.argmax(M[i]))
        for j, c in enumerate(cols):
            if annotate:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="black")
            if c in win[r]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="black", lw=2))
    ax.set_title(title, fontsize=10)
    ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="CKA")
    ax.figure.tight_layout()
    return ax


def plot_phi(profiles, planted=None, ax=None,
             title="learned order-weighting  phi(k)"):
    """Bar/line plot of one or more ``phi(k)`` profiles (``{label: array}``).
    If ``planted`` (a dict label->level or a single int) is given, mark the
    planted order so 'recovery' is visible at a glance."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, phi in profiles.items():
        phi = np.asarray(phi)
        ks = np.arange(len(phi))
        ax.plot(ks, phi, "-o", label=label)
    if planted is not None:
        if isinstance(planted, dict):
            for lvl in sorted(set(planted.values())):
                ax.axvline(lvl, ls="--", color="grey", alpha=0.6)
        else:
            ax.axvline(planted, ls="--", color="grey", alpha=0.6, label=f"planted level {planted}")
    ax.set_xlabel("signature level $k$"); ax.set_ylabel(r"$\varphi(k)$")
    ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    return ax
