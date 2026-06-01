"""The canonical fixture grid — the single source of truth shared by the freeze
(Stage 1) and compare (Stage 2) stages.

Each :class:`Case` fully specifies one ground-truth artifact:

* ``case_id``     — stable, filesystem-safe id (golden filename stem).
* ``entry``       — key into :mod:`tests.fixtures.runners` (which callable).
* ``kwargs``      — constructor kwargs for the kernel/estimator.
* ``inputs``      — ``{"X": (builder, params, label), "Y": ...}`` specs.
* ``family``      — tolerance class (see :mod:`tests.tolerances`).
* ``tags``        — semantic tags recorded in the provenance sidecar.

MEMORY DISCIPLINE
-----------------
The signature / DP kernels materialize an ``[n_X, n_Y, l_X, l_Y]`` tensor and
the ``order>1`` path additionally carries an ``[order, order]`` factor, so the
working set is ``O(n^2 L^2 order^2)``.  The default tier below is deliberately
tiny (``n <= 4``, ``L <= 8``) so the whole golden set fits in a few MB and never
threatens GPU memory on a shared box.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# An input spec is (builder_name, params, label):
#   builder_name + params  -> resolved into a numpy array by build_inputs()
#   label                  -> human-readable token used only in the case_id
InputSpec = Tuple[str, Dict[str, Any], str]


@dataclass(frozen=True)
class Case:
    case_id: str
    entry: str
    kwargs: Dict[str, Any]
    inputs: Dict[str, InputSpec]
    family: str
    tags: Tuple[str, ...] = ()

    def digest(self) -> str:
        payload = {
            "entry": self.entry,
            "kwargs": self.kwargs,
            "inputs": {k: (b, p) for k, (b, p, _) in self.inputs.items()},
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]


def build_inputs(inputs: Dict[str, InputSpec]) -> Dict[str, Any]:
    """Resolve input specs into concrete numpy arrays (host float64)."""
    from . import generators as g
    from . import biological as bio

    registry = {
        "gaussian": g.gaussian, "constant": g.constant, "near_zero": g.near_zero,
        "large": g.large, "ramp": g.ramp, "flat_2d": g.flat_2d,
        "onehot_dna": lambda **kw: bio.cumsum_embed(bio.onehot_dna(**kw)),
        "onehot_protein": lambda **kw: bio.cumsum_embed(bio.onehot_protein(**kw)),
    }
    return {k: registry[b](**p) for k, (b, p, _) in inputs.items()}


# -----------------------------------------------------------------------------
# Reusable input specs (builder, params, label).
# -----------------------------------------------------------------------------
NOMINAL = ("gaussian", dict(n=4, L=8, d=3, seed=0), "gaussN4L8d3")
SMALL   = ("gaussian", dict(n=2, L=5, d=2, seed=1), "gaussN2L5d2")
ONE     = ("gaussian", dict(n=1, L=6, d=3, seed=2), "gaussN1L6d3")     # batch-of-1
BASE_L1 = ("gaussian", dict(n=3, L=1, d=3, seed=3), "gaussN3L1d3")     # L=1 base case
DIM1    = ("gaussian", dict(n=3, L=7, d=1, seed=4), "gaussN3L7d1")     # d=1
CONST   = ("constant", dict(n=4, L=8, d=3, c=0.7),  "const07N4L8d3")   # zero-variance
RAMP    = ("ramp",     dict(n=4, L=8, d=3, seed=0), "rampN4L8d3")
BIO     = ("onehot_dna", dict(n=4, L=8, seed=0),    "bioDNAcumN4L8")
Y_SEQ   = ("gaussian", dict(n=3, L=8, d=3, seed=10), "gaussYN3L8d3")

FLAT    = ("flat_2d", dict(n=5, L=4, d=3, seed=0), "flatN5F12")
FLAT_Y  = ("flat_2d", dict(n=3, L=4, d=3, seed=7), "flatYN3F12")
LARGE2D = ("large",   dict(n=4, L=1, d=6, scale=1e3, seed=0), "largeN4L1d6")


def _case(entry, kwargs, inputs, family, tags) -> Case:
    label = "_".join(spec[2] for spec in inputs.values())
    kbits = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    base = f"{entry}__{label}__{kbits}".replace(" ", "")
    tmp = Case(base, entry, kwargs, inputs, family, tuple(tags))
    return Case(f"{base}__{tmp.digest()}", entry, kwargs, inputs, family,
                tuple(tags))


# -----------------------------------------------------------------------------
# Sequence-kernel cases.
# -----------------------------------------------------------------------------
def _seq_cases() -> List[Case]:
    cases: List[Case] = []
    for static in ("rbf", "linear"):
        for order in (1, 2, 3):
            for normalize in (True, False):
                kw = dict(n_levels=4, order=order, normalize=normalize,
                          static_kernel=static)
                tags = ("signature", f"order{order}",
                        "norm" if normalize else "unnorm", static)
                for xf in (NOMINAL, CONST, RAMP, BIO, ONE, BASE_L1, DIM1):
                    cases.append(_case("SignatureKernel", kw, {"X": xf},
                                       "DP_CUMSUM", tags))
                cases.append(_case("SignatureKernel", kw,
                                   {"X": NOMINAL, "Y": Y_SEQ}, "DP_CUMSUM",
                                   tags + ("xy",)))
    for n_levels in (1, 2, 5):
        kw = dict(n_levels=n_levels, order=1, normalize=False,
                  static_kernel="rbf")
        cases.append(_case("SignatureKernel", kw, {"X": NOMINAL}, "DP_CUMSUM",
                           ("signature", f"nlevels{n_levels}")))

    for difference in (True, False):
        for normalize in (True, False):
            kw = dict(difference=difference, normalize=normalize,
                      static_kernel="rbf")
            tags = ("sigpde", "diff" if difference else "nodiff",
                    "norm" if normalize else "unnorm")
            for xf in (NOMINAL, CONST, RAMP, BASE_L1):
                cases.append(_case("SignaturePDEKernel", kw, {"X": xf},
                                   "DP_CUMSUM", tags))
            cases.append(_case("SignaturePDEKernel", kw,
                               {"X": NOMINAL, "Y": Y_SEQ}, "DP_CUMSUM",
                               tags + ("xy",)))

    for xf in (NOMINAL, CONST, RAMP, BASE_L1, ONE):
        cases.append(_case("GlobalAlignmentKernel", dict(static_kernel="rbf"),
                           {"X": xf}, "DP_CUMSUM", ("gak", "logspace")))
    cases.append(_case("GlobalAlignmentKernel", dict(static_kernel="rbf"),
                       {"X": NOMINAL, "Y": Y_SEQ}, "DP_CUMSUM", ("gak", "xy")))
    return cases


# -----------------------------------------------------------------------------
# Static-kernel cases.
# -----------------------------------------------------------------------------
def _static_cases() -> List[Case]:
    cases: List[Case] = []
    specs = [
        ("LinearKernel", {}, "EXACT_ALGEBRA"),
        ("PolynomialKernel", dict(degree=3), "EXACT_ALGEBRA"),
        ("RBFKernel", {}, "DP_CUMSUM"),
        ("Matern12Kernel", {}, "DP_CUMSUM"),
        ("Matern32Kernel", {}, "DP_CUMSUM"),
        ("Matern52Kernel", {}, "DP_CUMSUM"),
        ("RationalQuadraticKernel", {}, "DP_CUMSUM"),
    ]
    for kernel, kwargs, family in specs:
        kw = dict(kernel=kernel, **kwargs)
        cases.append(_case("static", kw, {"X": FLAT}, family,
                           ("static", "gram", kernel)))
        cases.append(_case("static", kw, {"X": FLAT, "Y": FLAT_Y}, family,
                           ("static", "xy", kernel)))
        cases.append(_case("static_diag", kw, {"X": FLAT}, family,
                           ("static", "diag", kernel)))
        cases.append(_case("static", kw, {"X": LARGE2D}, family,
                           ("static", "large", kernel)))
    return cases


def all_cases() -> List[Case]:
    """The full memory-bounded golden set (sequence + static kernels)."""
    return _seq_cases() + _static_cases()


if __name__ == "__main__":
    import sys

    cs = all_cases()
    args = sys.argv[1:]

    # `--list [SUBSTR]` enumerates the case_ids -- i.e. the exact `-k` tokens the
    # compare/oracle tests are parametrized by -- optionally filtered by a
    # case-insensitive substring.  Needs only NumPy, so it runs on the port box
    # (no CuPy/ksig).  With no args: a per-entry count summary.
    if args and args[0] == "--list":
        needle = args[1].lower() if len(args) > 1 else ""
        hits = [c.case_id for c in cs if needle in c.case_id.lower()]
        for cid in hits:
            print(cid)
        tail = f" matching {needle!r}" if needle else ""
        print(f"# {len(hits)}/{len(cs)} cases{tail}", file=sys.stderr)
    else:
        print(f"{len(cs)} cases total")
        from collections import Counter
        by_entry = Counter(c.entry for c in cs)
        for k, v in by_entry.items():
            print(f"  {k:24s} {v}")
