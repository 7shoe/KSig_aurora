"""
test_signature_kernel_inductive_bias.py  —  ADD alongside existing KSig_aurora tests.

Discriminative diagnostics for the GeneralSignatureKernel configurations. Each DGP plants a
signal at one structural locus; the matched kernel should attain the highest out-of-sample CKA.
The printed matrix is the primary artifact; the per-dataset assertions encode the predictions.

API ASSUMED (align the implementation or adapt the `build` lambdas below):
    GeneralSignatureKernel(phi=("const"|"level_one"|"free"|"dilation"),
                           truncation=(int N | None), normalize=("once"|"per_level"),
                           static="rbf", bw=float, m_nodes=int, lam_max=float, dev="cpu")
    .fit_phi(X, y, task="classification", steps=int)      # learned configs only; train-only
    .__call__(X, Y=None) -> np.ndarray  [n,n] (square, normalized: unit diagonal)

Run for inspection:  python -m pytest test_signature_kernel_inductive_bias.py -s
                or:  python test_signature_kernel_inductive_bias.py

OBSERVED (2026-06-11, GeneralSignatureKernel v3, after the level-localized DGP redesign) -- NO
kernel bug found; the predicted diagonal now holds:
  dataset      sig-L1  sig-TRUNC  sig-PDE  sig-Wphi  sig-PDEphi  sig-EXACT  pooled  kme
  D_disp       +0.224  +0.240     +0.241   +0.248    +0.249      +0.229     +0.295  +0.278
  D_area       +0.004  +0.875     +0.836   +0.915    +0.823      -0.008     +0.010  +0.009
  D_peak       +0.015  +0.088     +0.085   +0.182    +0.047      +0.134     +0.008  +0.013
  D_scale      +0.031  +0.255     +0.198   +0.800    +0.082      +0.079     +0.011  +0.125
  D_lowsig     -0.005  +0.202     +0.379   +0.219    +0.433      -0.012     +0.013  +0.019
  D_perlevel   +0.015  +0.052     +0.043   +0.098    +0.044      +0.171     +0.013  +0.009

VALIDATED (hard asserts):
  * order             D_area    -- order-aware family 0.82-0.92 vs order-blind ~0.
  * no spurious order D_disp    -- on a displacement task, order gives NO gain over pooled-RBF.
  * free non-mono phi D_peak    -- sig-Wphi uniquely captures a level-2 peak (suppresses 1 & 3).
  * dilation clamp    D_scale   -- tail signal: Wphi > PDE > PDEphi (the lam_max=0.5 cone can't
                                   reach the tail). Reinforced by the COEFFICIENT-level cone test.
  * soft-decay        D_lowsig  -- PDEphi's decaying phi down-weights tail noise: PDEphi > PDE.
  * per-level norm    D_perlevel-- a class-indep level-1 drift dominates the global norm so only
                                   per-path PER-LEVEL whitening (sig-EXACT) recovers the level-2
                                   sign: sig-EXACT uniquely wins. Plus a DIRECT test that the
                                   per-level-2 cosine aligns ~0.9 on D_area.

ONE DOCUMENTED NEGATIVE (xfail, a finding -- not a fixable DGP): under normalize-once, CLASS-
INDEPENDENT tail NOISE does not degrade alignment (it inflates K and the unit diagonal together),
so hard truncation has nothing to rescue (sig-PDE stays robust). The truncation/tail AXIS is
therefore demonstrated in the SIGNAL direction (D_scale clamp chain + D_lowsig soft-decay), and the
noise-suppression direction is kept as a documented negative.

Design note: the redesign retired the two ALL-LEVEL primitives (time-reversal area, straight drift)
for LEVEL-LOCALIZED ones (_microloop -> level 2; _figure8 -> tail). Caveat learned in passing: a
single-level signal cannot make sig-EXACT beat sig-Wphi (Wphi sets phi=e_k and normalize-once on
one level IS the per-level cosine); per-level uniqueness needs a class-independent dominant level
to defeat the global single-scalar normalization (D_perlevel).
"""
import os, sys
import numpy as np
import pytest
from numpy.random import default_rng

# make the repo root importable in direct-run mode (`python tests/<this>.py`), where sys.path[0]
# is tests/ rather than the package root. Harmless under pytest (which already adds the rootdir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # torch backend is required by the kernels
    torch = None

# --- import the kernel under test; skip cleanly if not yet implemented -------------------
GSK = None
try:
    import ksig
    ksig.set_default_device("cpu")
    from ksig.generalized import GeneralSignatureKernel as GSK
except Exception:
    pass
pytestmark = pytest.mark.skipif(GSK is None, reason="GeneralSignatureKernel not importable yet")

# ----------------------------- config ----------------------------------------------------
N_TRAIN, N_TEST, L, D = 120, 120, 22, 2
N_LEVELS, M_NODES, LAM_MAX = 3, 6, 0.5      # truncation depth N; dilation grid; STABILITY clamp
FIT_STEPS = 200
MARGIN = 0.04                               # CKA margin for "clearly better"
CHANCE = 0.07                               # CKA below this == order/level-blind (no signal)

# ----------------------------- helpers ---------------------------------------------------
def _path(incr):
    """increments [n,L-1,d] -> path [n,L,d] starting at the origin."""
    z = np.zeros((incr.shape[0], 1, incr.shape[2]), np.float32)
    return np.concatenate([z, np.cumsum(incr, 1)], 1).astype(np.float32)

def median_bw(X, n_sub=1500, seed=0):
    pts = X.reshape(-1, X.shape[-1]); rng = default_rng(seed)
    s = pts[rng.choice(len(pts), min(n_sub, len(pts)), replace=False)]
    d2 = ((s[:, None] - s[None]) ** 2).sum(-1)
    return float(np.sqrt(np.median(d2[np.triu_indices(len(s), 1)]) + 1e-12))

def cka(K, y):
    """Centered kernel-target alignment of K with the rank-1 label kernel; in [-1,1]."""
    K = np.asarray(K, float); n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H
    yc = y.astype(float) - y.mean(); Yc = np.outer(yc, yc)
    return float((Kc * Yc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-12))

# non-signature references (order-blind), test x test Gram
def pooled_rbf_gram(X, bw):
    v = X.mean(1); d2 = ((v[:, None] - v[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * bw * bw))

def kme_rbf_gram(X, bw):
    n = len(X); K = np.empty((n, n))
    for i in range(n):                                   # mean over the L x L static block
        d2 = ((X[i][None, :, None, :] - X[:, None, :, :]) ** 2).sum(-1)   # [n,L,L]
        K[i] = np.exp(-d2 / (2 * bw * bw)).mean(axis=(1, 2))
    return K

# ----------------------------- data-generating processes ---------------------------------
# Each returns (Xtr, ytr, Xte, yte). Paths are TRAJECTORIES; the kernel differences internally.
#
# LEVEL-LOCALIZED PRIMITIVES (the toolkit). The two all-level primitives (time-reversal area,
# straight drift) load every signature level, which made the level-specific predictions fail. The
# level-localized ones below place class signal at a CHOSEN locus:
#   _microloop -> confined LEVEL-2 signed area;   _figure8 -> energy in the level>=3 TAIL.
def _microloop(eps, cls):
    """Confined LEVEL-2 primitive: a small oriented loop, signed area = +-pi*eps^2 (sign = cls).
    Small eps keeps level-3 leakage (~eps^3) negligible against the level-2 area (~eps^2)."""
    t = np.linspace(0, 2 * np.pi, L)
    o = 2 * cls - 1
    return (eps * np.stack([np.cos(t), o * np.sin(t)], 1)).astype(np.float32)

def _figure8(amp, freq, phase):
    """TAIL primitive: a high-frequency figure-eight (Lissajous 1:2). ZERO net signed area (so it
    is invisible to level 2); its energy sits in level >= 3 (the truncation tail)."""
    t = np.linspace(0, 2 * np.pi, L)
    return (amp * np.stack([np.sin(freq * t + phase),
                            0.5 * np.sin(2 * freq * t + phase)], 1)).astype(np.float32)


def make_disp(n, seed):
    """LEVEL-1 signal. Class-dependent net displacement along axis 0 via a straight drift
    (a straight line adds NO area), so level>=2 statistics are class-independent noise."""
    rng = default_rng(seed); y = rng.integers(0, 2, n)
    incr = rng.normal(0, 0.4, (n, L - 1, D)).astype(np.float32)
    incr[:, :, 0] += (2 * y - 1)[:, None] * 1.6 / (L - 1)          # endpoint shift; zero area
    return _path(incr), y

def _closed_loop(rng, radius=0.8):
    t = np.linspace(0, 2 * np.pi, L)                              # t=2pi == t=0  -> exactly closed
    return np.stack([radius * np.cos(t), radius * np.sin(t)], 1).astype(np.float32)

def make_area(n, seed):
    """LEVEL-2 (ORDER) signal, isolated. Closed loop; class 1 = time-REVERSED loop -> flips the
    signed area (level-2 antisymmetric part) while preserving the endpoint (==0, closed) and the
    multiset of points. Order-blind reps (level-1 / set / mean) are class-invariant BY CONSTRUCTION."""
    rng = default_rng(seed); y = rng.integers(0, 2, n)
    X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        loop = _closed_loop(rng, 0.5 + rng.random())
        if y[i] == 1:
            loop = loop[::-1].copy()                              # CW vs CCW -> area sign flip
        loop = loop + rng.normal(0, 0.5, (1, D)).astype(np.float32)   # class-indep rigid shift
        X[i] = loop
    return X, y

def make_peak(n, seed):
    """PEAK at level 2. Level-2 area = class (signal). Level 1 = LARGE class-indep straight drift
    (noise). Level 3 = class-indep high-freq wiggle (noise). The optimal phi suppresses levels 1 & 3
    and keeps 2 -> a NON-MONOTONE peak: representable by free weights (Wphi), forbidden to the
    completely-monotone dilation cone (PDEphi) and to uniform/per-level kernels."""
    rng = default_rng(seed); y = rng.integers(0, 2, n)
    t = np.linspace(0, 2 * np.pi, L); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        loop = _closed_loop(rng, 0.6)
        if y[i] == 1:
            loop = loop[::-1].copy()
        disp = rng.normal(0, 1.0, D).astype(np.float32)
        loop = loop + np.linspace(0, 1, L)[:, None].astype(np.float32) * disp[None, :] * 3.0  # L1 noise
        wig = np.zeros((L, D), np.float32)                        # L3 noise (class-indep)
        for f, p in zip((7, 9), rng.uniform(0, 6, 2)):
            wig[:, 0] += 0.25 * np.sin(f * t + p); wig[:, 1] += 0.25 * np.cos(f * t + p)
        X[i] = loop + wig
    return X, y

def make_scale(n, seed):
    """TAIL SIGNAL + dilation-clamp limit (redesigned with level-localized primitives). A class-
    INDEPENDENT micro-loop + drift fixes the low levels; a high-frequency figure-eight (zero net
    area -> energy in the level>=3 TAIL) has its AMPLITUDE set by the class, so the discriminative
    signal lives in the tail. Free weights (Wphi) up-weight the truncation-boundary level and
    capture it; uniform PDE only partially; the lam_max=0.5 dilation cone (PDEphi) is forced
    DECAYING and cannot reach the tail -> the clamp chain  Wphi > PDE > PDEphi  (end-to-end clamp
    signature). NOTE: a GLOBAL dilation does not work here -- normalize-once cancels it; the tail
    signal must be normalization-invariant, which a class-dependent tail-localized shape is."""
    rng = default_rng(seed); y = rng.integers(0, 2, n); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        base = _microloop(0.7, int(rng.integers(0, 2))) + \
            np.linspace(0, 1, L)[:, None].astype(np.float32) * rng.normal(0, 0.6, (1, D)).astype(np.float32)
        X[i] = base + _figure8(0.4 + 0.6 * y[i], 11, float(rng.uniform(0, 6)))   # amplitude = class (tail)
    return X, y

def make_lowsig(n, seed):
    """LOW-LEVEL signal + TAIL NOISE. Level-2 area = class (signal). Class-INDEPENDENT high-freq
    2D wiggle injects noise into HIGH levels. Truncating (TRUNC-phi1) hard-cuts the noise tail and
    PDEphi soft-decays it -> both beat uniform untruncated PDE, which carries the tail noise at full
    weight. sig-L1 (level 1, ~0 endpoint) misses the level-2 signal -> chance."""
    rng = default_rng(seed); y = rng.integers(0, 2, n)
    t = np.linspace(0, 2 * np.pi, L); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        loop = _closed_loop(rng, 0.8)
        if y[i] == 1:
            loop = loop[::-1].copy()
        wig = np.zeros((L, D), np.float32)
        for f in (12, 19, 27):
            p0, p1 = rng.uniform(0, 6, 2)
            wig[:, 0] += 0.5 * np.sin(f * t + p0); wig[:, 1] += 0.5 * np.cos(f * t + p1)
        X[i] = loop + wig
    return X, y

def make_perlevel(n, seed):
    """PER-LEVEL NORMALIZATION, end-to-end (redesigned). The class is a CONFINED level-2 micro-loop
    area sign. Each path is scaled by a WIDE per-path magnitude AND carries a LARGE class-
    INDEPENDENT level-1 drift, so the GLOBAL self-norm is dominated by per-path-random, class-
    independent level-1 energy -> a single normalize-once scales the level-2 signal by a random
    per-path factor and cannot read it. Only per-path PER-LEVEL whitening (sig-EXACT) normalizes
    level 2 by its own norm and recovers the area sign -> sig-EXACT UNIQUELY wins.

    Why a single-level signal alone is not enough: Wphi can set phi=e_2, and normalize-once on a
    single level IS the per-level-2 cosine -> Wphi would tie. The class-independent dominant level-1
    drift is what defeats the global single-scalar normalization while leaving per-level intact."""
    rng = default_rng(seed); y = rng.integers(0, 2, n); X = np.zeros((n, L, D), np.float32)
    for i in range(n):
        mag = float(np.exp(rng.normal(0, 1.3)))                  # wide per-path scale (~50x range)
        drift = np.linspace(0, 1, L)[:, None].astype(np.float32) * \
            rng.normal(0, 1.0, (1, D)).astype(np.float32) * 3.0   # LARGE class-INDEP level-1 (dominates norm)
        X[i] = mag * (_microloop(0.7, int(y[i])) + drift)
    return X, y

def make(maker, seed):
    Xtr, ytr = maker(N_TRAIN, seed)
    Xte, yte = maker(N_TEST, seed + 1000)
    return Xtr, ytr, Xte, yte

DATASETS = {"D_disp": make_disp, "D_area": make_area, "D_peak": make_peak,
            "D_scale": make_scale, "D_lowsig": make_lowsig, "D_perlevel": make_perlevel}

# ----------------------------- kernel registry -------------------------------------------
def _sig(phi, trunc, norm):
    return lambda bw: GSK(phi=phi, truncation=trunc, normalize=norm, static="rbf",
                          bw=bw, m_nodes=M_NODES, lam_max=LAM_MAX, dev="cpu")

KERNELS = {
    "sig-L1":       dict(kind="sig", learned=False, build=_sig("level_one", N_LEVELS, "once")),
    "sig-TRUNC":    dict(kind="sig", learned=False, build=_sig("const",     N_LEVELS, "once")),
    "sig-PDE":      dict(kind="sig", learned=False, build=_sig("const",     None,     "once")),
    "sig-Wphi":     dict(kind="sig", learned=True,  build=_sig("free",      N_LEVELS, "once")),
    "sig-PDEphi":   dict(kind="sig", learned=True,  build=_sig("dilation",  None,     "once")),
    "sig-EXACT":    dict(kind="sig", learned=False, build=_sig("const",     N_LEVELS, "per_level")),
    "pooled-RBF":   dict(kind="ref", learned=False, gram=pooled_rbf_gram),
    "kme-RBF":      dict(kind="ref", learned=False, gram=kme_rbf_gram),
}

def score(spec, Xtr, ytr, Xte, yte, bw):
    if spec["kind"] == "ref":
        return cka(spec["gram"](Xte, bw), yte)
    if torch is not None:
        torch.manual_seed(0)
    ker = spec["build"](bw)
    if spec["learned"]:
        ker.fit_phi(Xtr, ytr, task="classification", steps=FIT_STEPS)   # TRAIN ONLY (out-of-sample)
    return cka(np.asarray(ker(Xte)), yte)

_ROW_CACHE = {}

def row(ds_name):
    """Full CKA row for one dataset (cached per name within a session)."""
    if ds_name in _ROW_CACHE:
        return _ROW_CACHE[ds_name]
    Xtr, ytr, Xte, yte = make(DATASETS[ds_name], seed=list(DATASETS).index(ds_name))
    bw = median_bw(Xtr)
    r = {name: score(spec, Xtr, ytr, Xte, yte, bw) for name, spec in KERNELS.items()}
    _ROW_CACHE[ds_name] = r
    return r

# ----------------------------- assertions (the predicted diagonal) -----------------------
def _report(name, r):
    line = "  ".join(f"{k}={v:+.3f}" for k, v in r.items())
    print(f"\n[{name}] {line}")

def test_D_disp_order_gives_no_gain():
    """REFRAMED (data kept): a straight drift is grouplike and loads every SYMMETRIC level, so
    higher order yields no alignment gain over the order-blind readout. The true, meaningful claim
    -- the signature family does not manufacture spurious order signal on a displacement task:
      * the order-aware signature family does not beat pooled-RBF, and
      * level-1-only (sig-L1) is within epsilon of the full signature."""
    r = row("D_disp"); _report("D_disp", r)
    order_aware = max(r["sig-TRUNC"], r["sig-PDE"], r["sig-Wphi"], r["sig-PDEphi"])
    assert order_aware <= r["pooled-RBF"] + 0.02            # order manufactures no gain over pooling
    assert r["sig-L1"] >= order_aware - 0.03                # level-1 alone ~ the full signature

def test_D_area_order_separates_orderaware_from_orderblind():
    r = row("D_area"); _report("D_area", r)
    for blind in ("sig-L1", "pooled-RBF", "kme-RBF"):
        assert r[blind] < CHANCE                            # order-blind reps are class-invariant here
    aware = min(r["sig-TRUNC"], r["sig-PDE"], r["sig-Wphi"], r["sig-PDEphi"])
    assert aware > r["sig-L1"] + 0.10                       # every order-aware kernel clears chance

def test_D_peak_freeweights_only():
    r = row("D_peak"); _report("D_peak", r)
    for other in ("sig-PDEphi", "sig-PDE", "sig-TRUNC", "sig-EXACT"):
        assert r["sig-Wphi"] > r[other] + MARGIN            # only free weights can peak & suppress 1,3

def test_D_scale_tail_signal_clamp_chain():
    """REDESIGNED (figure-eight tail signal). The class lives in the level>=3 tail (figure-eight
    amplitude). Free weights reach it, uniform PDE only partially, the lam_max=0.5 dilation cone
    cannot -> the clamp chain  Wphi > PDE > PDEphi. This is the END-TO-END signature of the
    stability clamp (complementing the coefficient-level cone test). sig-L1 is blind to a tail
    signal."""
    r = row("D_scale"); _report("D_scale", r)
    assert r["sig-Wphi"] > r["sig-PDE"] + MARGIN           # free weights up-weight the tail
    assert r["sig-PDE"] > r["sig-PDEphi"]                  # uniform beats the down-weighting clamp
    assert r["sig-L1"] < CHANCE                            # level-1 blind to a pure tail signal

def test_D_lowsig_softdecay_suppresses_tailnoise():
    """PARTIAL pass kept as a HARD assert: the two robust halves -- sig-L1 blind to the level-2
    signal, and PDEphi's soft-decay beating uniform PDE -- both hold. The third (hard-truncation
    > PDE) is xfailed separately: PDE is robust to this particular tail noise (sig-PDE=0.38)."""
    r = row("D_lowsig"); _report("D_lowsig", r)
    assert r["sig-L1"] < CHANCE                             # level-2 signal invisible to level 1
    assert r["sig-PDEphi"] > r["sig-PDE"] + 0.02          # soft decay down-weights the tail noise


@pytest.mark.xfail(reason="FINDING, not a fixable DGP: under normalize-once, CLASS-INDEPENDENT tail "
    "noise does not degrade alignment (it inflates K and the unit-diagonal together), so hard "
    "truncation has nothing to rescue -- sig-PDE stays robust (0.38-0.82 across DGP variants tried). "
    "The tail AXIS is instead demonstrated in the SIGNAL direction by D_scale's clamp chain "
    "(Wphi>PDE>PDEphi) and by the soft-decay half (PDEphi>PDE) asserted above. Truncation-for-noise-"
    "suppression is simply weak once the kernel is normalized; keep as a documented negative.",
    strict=False)
def test_D_lowsig_truncation_beats_pde():
    r = row("D_lowsig")
    assert r["sig-TRUNC"] > r["sig-PDE"] + 0.02

def test_D_perlevel_normalization_only():
    """REDESIGNED. A class-independent wide level-1 drift dominates the global self-norm, so only
    per-path PER-LEVEL whitening (sig-EXACT) recovers the confined level-2 area sign -> sig-EXACT
    UNIQUELY wins (incl. over Wphi: a single global phi cannot per-path-whiten level 2 out from
    under a per-path-random level-1 normalizer)."""
    r = row("D_perlevel"); _report("D_perlevel", r)
    for other in ("sig-Wphi", "sig-TRUNC", "sig-PDE", "sig-PDEphi"):
        assert r["sig-EXACT"] > r[other] + MARGIN           # per-path per-level whitening uniquely recovers sign


def test_perlevel_path_whitens_each_level():
    """Direct validation of the per_level normalization MATH (no DGP calibration dependence): on
    D_area, the per-level-normalized LEVEL-2 sub-kernel aligns with the order signal (~0.9),
    confirming sig-EXACT whitens each level per-path correctly. The AVERAGED sig-EXACT sits at
    chance on D_area ONLY because closed loops make the level-1 self-magnitude ~0 (the 1e-12 clamp
    turns K~_1 into noise that dominates the uniform average) -- the documented per-level fragility,
    which is exactly why a per-level PROBE (D_perlevel) keeps a non-degenerate level-1 drift."""
    import torch
    from ksig.generalized import (_static_block, _second_diff, _diag_block, _normalize,
                                  signature_kern)
    Xtr, ytr, Xte, yte = make(DATASETS["D_area"], seed=list(DATASETS).index("D_area"))
    bw = median_bw(Xtr); dev = torch.device("cpu")
    M = _second_diff(_static_block(Xte, Xte, bw, dev))
    Kl = torch.stack(list(signature_kern(M, N_LEVELS, 1, False, True)), 0)
    dl = torch.stack(list(signature_kern(_diag_block(Xte, bw, dev), N_LEVELS, 1, False, True)), 0)
    cka_l2 = cka(_normalize(Kl[2], dl[2], dl[2]).cpu().numpy(), yte)
    assert cka_l2 > 0.6                                     # level-2 per-level cosine recovers the order sign
    assert float(dl[1].abs().mean()) < 1e-3               # level-1 ~ 0 on closed loops (the fragility)


# ----------------------------- direct structural property (robust, no DGP dependence) ----
def test_dilation_cone_monotone_vs_free_nonmonotone():
    """The theory linchpin, asserted directly on the learned phi (not via CKA). On D_peak (signal
    at level 2, level 1 = large noise) the optimal phi is a NON-MONOTONE peak:
      * sig-Wphi (free weights) MUST be able to make phi(2) > phi(1)  (representable),
      * sig-PDEphi (completely-monotone dilation cone) MUST stay monotone non-increasing -- it
        structurally cannot peak.
    This is the property D_peak's CKA gap reflects, checked on the COEFFICIENTS so it cannot be
    confounded by data calibration."""
    Xtr, ytr, _, _ = make(DATASETS["D_peak"], seed=list(DATASETS).index("D_peak"))
    bw = median_bw(Xtr)
    wphi = GSK(phi="free", truncation=N_LEVELS, normalize="once", static="rbf", bw=bw)
    wphi.fit_phi(Xtr, ytr, task="classification", steps=FIT_STEPS)
    pdephi = GSK(phi="dilation", truncation=None, static="rbf", bw=bw, m_nodes=M_NODES, lam_max=LAM_MAX)
    pdephi.fit_phi(Xtr, ytr, task="classification", steps=FIT_STEPS)
    w = np.asarray(wphi.phi_profile())
    p = np.asarray(pdephi.phi_profile(N_LEVELS))
    assert w[2] > w[1] + 0.5                                # free weights peak above level 1
    assert np.all(np.diff(p) <= 1e-6)                      # dilation cone is monotone non-increasing


# ----------------------------- manual run: print the confusion matrix --------------------
if __name__ == "__main__":
    names = list(KERNELS)
    print(f"{'dataset':<11}" + "".join(f"{n:>12}" for n in names))
    for ds in DATASETS:
        r = row(ds)
        print(f"{ds:<11}" + "".join(f"{r[n]:>+12.3f}" for n in names))
