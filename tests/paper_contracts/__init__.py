"""Paper-contract tests: one behavior per test, tiny deterministic inputs.

These pin the statistical/API contracts the paper-to-code audit
(``docs/AUDIT_REPORT.md`` / ``docs/AUDIT_RESPONSE.md``) flagged, layered as:

- ``test_truncated_signature``  -- level-0 / level-increment semantics (F1).
- ``test_general_facade``       -- delegated ``diag`` + dtype preservation (F2/F3).
- ``test_pde_signature``        -- Goursat boundary, finite-output / clamp policy (F5).
- ``test_random_features``      -- per-level RNG independence, sparsity, RFF sanity (F6/F7).
- ``test_docs_scope``           -- paper-4 not-implemented markers + docs link-check (F8/F9).

Sentinel coverage only: a few representative kernels per invariant. Full
parameter-matrix coverage lives in the golden/oracle suites, not here. Anything
heavier than a micro-check is seeded, tolerance-loose, and marked
``random_feature``/``slow`` so it can never flake a core gate.

xfail here follows the repo convention (see ``conftest.pytest_terminal_summary``):
a known gap the remediation must close. When the fix lands the test xpasses ->
promote it to a required-pass assertion.
"""
