"""Pytest scaffolding: markers + fixture wrappers around :mod:`tests.harness`.

The reusable logic (backend detection, golden loader, pinpointing comparison)
lives in :mod:`tests.harness` so test files can use stable absolute imports.
"""
from __future__ import annotations

import pytest

# Re-export so `from tests.conftest import ...` and `from tests.harness import
# ...` are interchangeable.
from tests.harness import (  # noqa: F401
    BACKEND, DEVICE, IS_LEGACY, Golden, assert_close, detect_backend,
    load_golden,
)


def pytest_configure(config):
    for name, desc in [
        ("unit", "low-level, oracle/closed-form only (no golden needed)"),
        ("golden", "requires a tests/golden/*.npz artifact"),
        ("oracle", "validates ground truth: legacy/port vs NumPy oracle"),
        ("xpu", "requires torch.xpu.is_available()"),
        ("sycl", "requires a built ksig._sycl extension"),
        ("slow", "large end of the dimension sweep"),
        ("stress", "extreme shapes / float16 / OOM-probing"),
        ("monitoring", "smoke-runs a monitoring probe (not a perf gate)"),
        ("random_feature", "Monte-Carlo random-feature smoke (seeded, loose "
                           "tolerance; separable so it can't flake core gates)"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")

    # Silence Numba's benign "low occupancy" GPU perf warnings: our fixtures are
    # intentionally tiny (n<=4, L<=8) so grids are small by design. Registered
    # here (not in pytest.ini) and guarded by the import, so the suite still
    # loads on the torch port where numba is absent.
    try:
        import numba.core.errors  # noqa: F401

        config.addinivalue_line(
            "filterwarnings",
            "ignore::numba.core.errors.NumbaPerformanceWarning",
        )
    except Exception:
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Explain the non-pass results in one glance, so a green run reads as a
    clean success instead of a wall of yellow.

    Silent when there is a real failure/error: pytest's own report must stay
    loud there. Otherwise prints one bullet per category (xfailed / xpassed /
    skipped) plus a note that the benign Numba perf warnings were suppressed.
    """
    tr = terminalreporter
    count = lambda key: len(tr.stats.get(key, []))  # noqa: E731

    if count("failed") or count("error"):
        return  # real problems -- don't sugar-coat; leave pytest's report alone
    xfailed, xpassed, skipped = count("xfailed"), count("xpassed"), count("skipped")
    if not (xfailed or xpassed or skipped):
        return

    tr.section("suite notes", sep="-", blue=True)
    if xfailed:
        tr.write_line(
            f"* {xfailed} xfailed = KNOWN gaps pinned on purpose, two kinds: "
            "(a) legacy ksig bugs the torch port must FIX (euclid_dist breaks "
            "Matern12/32; SigPDE(difference=True) can't launch on L=1); and "
            "(b) paper-contract remediations not yet landed "
            "(tests/paper_contracts: plain sig-PDE stability gate, auto_clamp phi "
            "reporting, paper-4 Algorithm 6 / string kernels, docs link F9). "
            "Each carries its reason; when the fix lands it flips to xpassed -> "
            "promote to a required-pass assertion."
        )
    if xpassed:
        tr.write_line(
            f"* {xpassed} xpassed = the backend FIXED a case the legacy code got "
            "wrong. Promote these from xfail to a required-pass assertion."
        )
    if skipped:
        tr.write_line(
            f"* {skipped} skipped = cases N/A to this fixture (e.g. multi-axis "
            "tensor algebra needs >=2 dims)."
        )
    if IS_LEGACY:
        tr.write_line(
            "* Numba 'low occupancy' GPU perf warnings suppressed = benign; "
            "fixtures are intentionally tiny (n<=4, L<=8)."
        )


@pytest.fixture
def golden():
    return load_golden


@pytest.fixture
def close():
    return assert_close
