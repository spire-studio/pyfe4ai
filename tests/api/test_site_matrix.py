"""The project page (site/index.html) shows what the registry says: its scheme
matrix is the rendering of tools/gen_site_matrix.py, and its counts of
schemes, families, variants and review states are the registry's."""

import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest

import pyfe4ai
from pyfe4ai.core import registry

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "site" / "index.html"
TOOL = ROOT / "tools" / "gen_site_matrix.py"

# the source distribution carries the tests but neither the page nor the tool
pytestmark = pytest.mark.skipif(not (PAGE.exists() and TOOL.exists()),
                                reason="the project page is not part of this source tree")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("gen_site_matrix", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


def _schemes():
    return [pyfe4ai.get_scheme(i) for i in pyfe4ai.list_schemes()]


def test_every_registered_scheme_has_one_cell(tool):
    placed = [s.id for schemes in tool.cells().values() for s in schemes]
    assert sorted(placed) == sorted(pyfe4ai.list_schemes())


def test_the_page_matrix_is_the_rendered_matrix(tool, page):
    """Regenerate it with ``python tools/gen_site_matrix.py --write``
    rather than editing the page."""
    assert tool.page_block(page) == tool.block()


def test_the_page_counts_are_the_registry_counts(page):
    schemes = _schemes()
    expected = {
        "schemes": len(schemes),
        **Counter(s.family for s in schemes),
        "threshold": len(pyfe4ai.list_schemes(capabilities="threshold")),
        "decentralized": len(pyfe4ai.list_schemes(capabilities="decentralized")),
        "function-hiding": sum("fh_" in s.variant for s in schemes),
        "refused": sum(bool(s.refused) for s in schemes),
    }
    shown = {k: int(v) for k, v in re.findall(r'data-count="([\w-]+)">(\d+)<', page)}
    assert shown == expected


def test_the_page_review_status_is_the_registry_status(page):
    counts = Counter(s.status.status for s in _schemes())
    expected = {status: counts.get(status, 0) for status in registry.STATUSES}
    listed = {k: int(v) for k, v in re.findall(r'data-status="([\w-]+)">(\d+)<', page)}
    assert listed == expected
    bar = {k: int(v) for k, v in re.findall(r'class="seg (\w+)" style="--n:(\d+)"', page)}
    marks = {"rev": "reviewed", "exp": "experimental", "unr": "unreviewed", "ins": "known-insecure"}
    assert {marks[k]: v for k, v in bar.items()} == {k: v for k, v in expected.items() if v}
    label = re.search(r"Of (\d+) registered schemes: (\d+) reviewed, (\d+) experimental, (\d+) unreviewed, "
                      r"(\d+) known-insecure", page)
    assert label is not None
    assert [int(g) for g in label.groups()] == [sum(expected.values())] + [expected[s] for s in registry.STATUSES]
