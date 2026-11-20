"""docs/threat_model.rst names every registered scheme, with its status where it is not reviewed, and the README and
SECURITY.md link to it (R-1.0, T-4)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pyfe4ai.core import registry

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "threat_model.rst"
LINK = "https://spire-studio.github.io/pyfe4ai/docs/threat_model.html"

pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="the sources of the documentation are not in this tree")


def _text() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())


def test_every_registered_scheme_is_named():
    text = _text()
    missing = [f"{s.family}/{s.variant}" for s in registry.SCHEMES if f"``{s.family}/{s.variant}``" not in text]
    assert not missing, missing


def test_every_scheme_that_is_not_reviewed_is_marked_with_its_status():
    text = _text()
    for s in registry.SCHEMES:
        if s.status.status != registry.REVIEWED:
            assert f"``{s.family}/{s.variant}`` (*{s.status.status}*" in text, (s.family, s.variant, s.status.status)


def test_no_reviewed_scheme_is_marked_otherwise():
    text = _text()
    for s in registry.SCHEMES:
        if s.status.status == registry.REVIEWED:
            assert not re.search(re.escape(f"``{s.family}/{s.variant}`` (*") + r"(experimental|unreviewed|known-insecure)",
                                 text), (s.family, s.variant)


def test_the_page_is_in_the_documentation_and_linked():
    assert re.search(r"^\s+threat_model$", (ROOT / "docs" / "index.rst").read_text(encoding="utf-8"), re.M)
    for name in ("README.md", "SECURITY.md"):
        path = ROOT / name
        if path.exists():
            assert LINK in path.read_text(encoding="utf-8"), name
