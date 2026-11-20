"""docs/audit.rst places every registered scheme in one scope group that fits its tier, and the README and SECURITY.md
link to it instead of announcing it (plan 11, E2)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pyfe4ai.core import registry, stability

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "audit.rst"
LINK = "https://spire-studio.github.io/pyfe4ai/docs/audit.html"
GROUPS = ("A1", "A2", "A3", "A, formats and correctness only", "B (optional, later)", "Not in scope")

pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="the sources of the documentation are not in this tree")


def _groups() -> dict:
    section = PAGE.read_text(encoding="utf-8").split("\nScope\n-----\n", 1)[1].split("\nBesides the schemes:", 1)[0]
    found = {}
    for item in re.split(r"\n\* ", section)[1:]:
        # the group's list: the ``ids`` right after the label, up to the first sentence of explanation
        label = re.match(r"\*\*(.+?)\*\* -- ((?:``[a-z]+/[a-z0-9_]+``,?\s*)+)", item)
        assert label, item[:60]
        found[label.group(1)] = re.findall(r"``([a-z]+/[a-z0-9_]+)``", label.group(2))
    return found


def test_the_groups_are_the_expected_ones():
    assert tuple(_groups()) == GROUPS


def test_every_registered_scheme_is_in_exactly_one_group():
    placed = [scheme for ids in _groups().values() for scheme in ids]
    registered = [f"{s.family}/{s.variant}" for s in registry.SCHEMES]
    assert sorted(placed) == sorted(registered)


def test_each_group_fits_the_tiers():
    groups = _groups()
    by_id = {f"{s.family}/{s.variant}": s for s in registry.SCHEMES}
    for label, ids in groups.items():
        for scheme in ids:
            tier, refused = stability.tier(scheme), bool(by_id[scheme].refused)
            if label in ("A1", "A2", "A3"):
                assert (tier, refused) == (stability.STABLE, False), (label, scheme)
            elif label.startswith("A,"):
                assert (tier, refused) == (stability.STABLE, True), (label, scheme)
            elif label.startswith("B"):
                assert tier == stability.EXPERIMENTAL_TIER, (label, scheme)
            else:
                assert tier == stability.KNOWN_INSECURE_TIER, (label, scheme)


def test_the_readme_and_security_md_link_to_the_package():
    assert re.search(r"^\s+audit$", (ROOT / "docs" / "index.rst").read_text(encoding="utf-8"), re.M)
    for name in ("README.md", "SECURITY.md"):
        path = ROOT / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            assert LINK in text and "planned for the release" not in text, name
