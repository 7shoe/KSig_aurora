"""Scope honesty (F8) and docs integrity (F9).

- Paper-4 Algorithm 6 (simultaneous low-rank Gram) and string-kernel
  sequentialization are NOT implemented this release. They are pinned as `xfail`
  (try-to-use -> ImportError); if a real implementation lands, the test xpasses
  -> promote it to a required-pass numerical check (audit's
  `test_simultaneous_lowrank_gram_matches_dense` /
  `test_string_kernel_bruteforce_short_strings`).
- A docs link-check guards internal `*.md` references. New dangling links fail
  loudly NOW; the one KNOWN dangling reference (F9) is tracked as `xfail` until
  P10 removes/creates it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

# Known-missing internal doc, pending P10 (audit F9). Any OTHER dangling link is
# a real failure, not an expected one.
KNOWN_MISSING = {"LEARNABLE_PHI_SIGNATURE_KERNELS.md"}


# ---------------------------------------------------------------- F8: paper-4 not implemented
@pytest.mark.xfail(reason="paper-4 Algorithm 6 (simultaneous low-rank Gram) not "
                          "implemented this release (audit F8).", strict=False)
def test_simultaneous_lowrank_gram_matches_dense():
    """When Algorithm 6 lands: U @ U.T must equal the dense SignatureKernel for
    linear inputs. Until then, importing the API raises -> xfail."""
    from ksig.kernels import SimultaneousLowRankSignatureKernel  # noqa: F401
    raise AssertionError("implement the U@U.T-vs-dense check once the API exists")


@pytest.mark.xfail(reason="paper-4 string-kernel sequentialization not implemented "
                          "this release (audit F8).", strict=False)
def test_string_kernel_bruteforce_short_strings():
    """When the string kernel lands: gap-decay DP must equal the brute-force
    strict-subsequence sum. Until then, importing the API raises -> xfail."""
    from ksig.kernels import StringKernel  # noqa: F401
    raise AssertionError("implement the DP-vs-bruteforce check once the API exists")


# ---------------------------------------------------------------- F9: docs link-check
_MD_TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.md")


def _existing_md_basenames():
    return {p.name for p in REPO_ROOT.rglob("*.md")}


def _dangling_md_references():
    """Set of referenced `*.md` basenames that do not exist anywhere in the repo."""
    existing = _existing_md_basenames()
    dangling = set()
    for md in DOCS_DIR.rglob("*.md"):
        text = md.read_text(encoding="utf-8", errors="ignore")
        for tok in _MD_TOKEN.findall(text):
            if tok.startswith(("http://", "https://")):
                continue
            name = Path(tok).name
            if name not in existing:
                dangling.add(name)
    return dangling


def test_no_unexpected_dangling_doc_links():
    """No NEW dangling internal `*.md` links creep in (the one known gap, F9, is
    excluded and tracked separately below)."""
    unexpected = _dangling_md_references() - KNOWN_MISSING
    assert not unexpected, f"new dangling doc references: {sorted(unexpected)}"


@pytest.mark.xfail(reason="F9/P10 not landed: docs/SIGNATURE_KERNELS.md still "
                          "references the missing LEARNABLE_PHI_SIGNATURE_KERNELS.md.",
                   strict=False)
def test_known_missing_doc_is_resolved():
    """The known dangling reference is eventually removed or the file authored."""
    still_missing = _dangling_md_references() & KNOWN_MISSING
    assert not still_missing, f"still dangling: {sorted(still_missing)}"
