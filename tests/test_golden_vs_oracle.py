"""Validate the *ground truth itself*: every frozen golden artifact must agree
with the independent brute-force NumPy oracle.

This is the non-tautological check.  The golden was frozen from the legacy CuPy
code; here we recompute the same quantity from first principles (a different
implementation) and require agreement.  Two independent computations agreeing is
what makes the golden trustworthy — not "the old code agrees with itself".

Runs anywhere (pure NumPy, no ksig backend, no GPU).  Marked ``oracle``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness import assert_close, load_golden
from tests.fixtures import matrix as fxmatrix
from tests.oracles import pipeline as oracle

CASES = fxmatrix.all_cases()
# Only cases that were sourced FROM legacy can be cross-checked against the
# oracle here (oracle-sourced ones are the oracle, so the check is trivial; we
# still verify they load and are finite).
IDS = [c.case_id for c in CASES]


@pytest.mark.oracle
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_golden_matches_oracle(case):
    g = load_golden(case.case_id)
    if g is None:
        pytest.skip(f"no golden for {case.case_id}")

    inputs = fxmatrix.build_inputs(case.inputs)
    oracle_out, _ = oracle.oracle_output(case.entry, case.kwargs, inputs)
    oracle_d = oracle_out if isinstance(oracle_out, dict) else {"value": oracle_out}

    # Loosen for artifacts the freeze flagged ill-conditioned.
    rtol_override = (g.meta.get("recommended_rtol")
                     if g.meta.get("ill_conditioned") else None)

    for key in g.keys:
        exp = oracle_d[key]
        got = g[key]
        assert np.isfinite(got).all(), f"non-finite golden {case.case_id}[{key}]"
        assert_close(got, exp, family=case.family, case_id=case.case_id,
                     note=f"golden[{key}] vs oracle",
                     rtol_override=rtol_override)
