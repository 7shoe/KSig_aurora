"""Stage 2 — compare the implementation under test against the frozen golden.

This is the test the **torch port** will run on Aurora/XPU/CPU: load the golden
``.npz``, run the *current* ``ksig`` over the identical fixture, assert agreement
under the family tolerance with a pinpointing failure message.

Run today against the legacy CuPy ``ksig`` it serves as a self-consistency
check.  The two kernels the legacy code gets wrong (``Matern12``/``Matern32`` —
the ``euclid_dist`` bug) are ``xfail`` on the legacy backend and **required to
pass on the port**, so the suite actively drives the bug-fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness import BACKEND, DEVICE, IS_LEGACY, assert_close, load_golden
from tests.fixtures import matrix as fxmatrix
from tests.fixtures import runners

ksig = pytest.importorskip("ksig")

CASES = fxmatrix.all_cases()
IDS = [c.case_id for c in CASES]


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_port_matches_golden(case):
    g = load_golden(case.case_id)
    if g is None:
        pytest.skip(f"no golden for {case.case_id}")

    legacy_broken = (g.meta.get("legacy_status") == "broken")
    if IS_LEGACY and legacy_broken:
        pytest.xfail(f"legacy ksig is broken here ({g.meta.get('legacy_error')}); "
                     "the torch port must fix this and pass.")

    inputs = fxmatrix.build_inputs(case.inputs)
    out = runners.run(ksig, case.entry, inputs, case.kwargs)
    got = out if isinstance(out, dict) else {"value": out}

    rtol_override = (g.meta.get("recommended_rtol")
                     if g.meta.get("ill_conditioned") else None)

    for key in g.keys:
        assert key in got, f"runner did not produce '{key}' for {case.entry}"
        arr = np.asarray(got[key])
        assert np.isfinite(arr).all(), f"non-finite output {case.case_id}[{key}]"
        assert_close(arr, g[key], family=case.family, dtype="float64",
                     device=DEVICE, case_id=case.case_id,
                     note=f"{BACKEND}:{key}", rtol_override=rtol_override)


@pytest.mark.golden
def test_backend_detected():
    # A guard so a misconfigured environment (neither cupy nor torch ksig) is
    # loud rather than silently skipping everything.
    assert BACKEND in ("legacy_cupy", "torch"), (
        f"could not detect ksig backend (got {BACKEND!r}); "
        "ArrayOnGPU is neither cupy.ndarray nor torch.Tensor")
