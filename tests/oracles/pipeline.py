"""Public-level NumPy oracles: full ``_K`` / ``_Kdiag`` pipelines for the three
sequence kernels and the static kernels, returning the same ``{gram, diag, xy}``
contract the shared runner emits.  This lets the freeze stage cross-check *every*
golden value (including normalization / exp / averaging) against independent
math, and lets Stage-2 tests fall back to the oracle when no ``.npz`` exists.
"""
from __future__ import annotations

import numpy as np

from . import dp_numpy as dp
from . import signature_numpy as sn
from . import static_numpy as st

_EPS = 1e-12


# -----------------------------------------------------------------------------
# Sequence kernels.
# -----------------------------------------------------------------------------
def signature_outputs(X, Y, kwargs):
    nl = kwargs.get("n_levels", 4)
    order = kwargs.get("order", 1)
    diff = kwargs.get("difference", True)
    norm = kwargs.get("normalize", True)
    static = kwargs.get("static_kernel", "rbf")
    out = {
        "gram": sn.signature_kernel(X, None, nl, order, diff, norm, static),
        "diag": sn.signature_kernel(X, None, nl, order, diff, norm, static,
                                    diag=True),
    }
    if Y is not None:
        out["xy"] = sn.signature_kernel(X, Y, nl, order, diff, norm, static)
    return out


def _normalize_real(K, dX, dY):
    sX = np.maximum(np.sqrt(np.maximum(dX, 0.0)), _EPS)
    sY = np.maximum(np.sqrt(np.maximum(dY, 0.0)), _EPS)
    return K / (sX[:, None] * sY[None, :])


def sigpde_outputs(X, Y, kwargs):
    diff = kwargs.get("difference", True)
    norm = kwargs.get("normalize", True)
    static = kwargs.get("static_kernel", "rbf")

    def raw_gram(A, B):
        M = sn.static_gram(A, B, static)
        return dp.sig_pde_gram(M, difference=diff)

    def raw_diag(A):
        M = sn.static_gram(A, None, static, diag=True)   # [n, lX, lX]
        return dp.sig_pde_gram(M, difference=diff)

    G = raw_gram(X, None)
    if norm:
        d = np.diag(G).copy()
        G = _normalize_real(G, d, d)
    out = {"gram": G}
    out["diag"] = np.ones(X.shape[0]) if norm else raw_diag(X)
    if Y is not None:
        Gxy = raw_gram(X, Y)
        if norm:
            Gxy = _normalize_real(Gxy, raw_diag(X), raw_diag(Y))
        out["xy"] = Gxy
    return out


def gak_outputs(X, Y, kwargs):
    static = kwargs.get("static_kernel", "rbf")

    def raw_gram(A, B):
        M = sn.static_gram(A, B, static)
        return dp.gak_log_gram(M)                # log-space [nA, nB]

    def raw_diag(A):
        M = sn.static_gram(A, None, static, diag=True)
        return dp.gak_log_gram(M)                # log-space [n]

    logK = raw_gram(X, None)
    dX = np.diag(logK).copy()
    logK = logK - 0.5 * (dX[:, None] + dX[None, :])
    out = {"gram": np.exp(logK), "diag": np.ones(X.shape[0])}
    if Y is not None:
        logKxy = raw_gram(X, Y)
        logKxy = logKxy - 0.5 * (raw_diag(X)[:, None] + raw_diag(Y)[None, :])
        out["xy"] = np.exp(logKxy)
    return out


# -----------------------------------------------------------------------------
# Static kernels.
# -----------------------------------------------------------------------------
def static_outputs(entry, kwargs, X, Y):
    kw = dict(kwargs)
    name = kw.pop("kernel")
    if entry == "static_diag":
        return {"value": st.diag(name, X, **kw)}
    return {"value": st.gram(name, X, Y, **kw)}


# -----------------------------------------------------------------------------
# Dispatch.
# -----------------------------------------------------------------------------
def oracle_output(entry, kwargs, inputs):
    """Return the oracle ground truth in the runner's output shape.

    ``inputs`` is the resolved ``{"X": ndarray, "Y": ndarray?}`` dict.
    Returns a dict (sequence kernels: gram/diag/xy; static: value) plus a flag
    ``__legacy_broken__`` when the legacy code is known-wrong for this case.
    """
    X = inputs["X"]
    Y = inputs.get("Y")
    if entry == "SignatureKernel":
        return signature_outputs(X, Y, kwargs), False
    if entry == "SignaturePDEKernel":
        return sigpde_outputs(X, Y, kwargs), False
    if entry == "GlobalAlignmentKernel":
        return gak_outputs(X, Y, kwargs), False
    if entry in ("static", "static_diag"):
        name = kwargs["kernel"]
        broken = st.KERNELS[name][1]
        return static_outputs(entry, kwargs, X, Y), broken
    raise KeyError(entry)
