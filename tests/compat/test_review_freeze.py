"""Review of the 1.0 freeze line (feat/1.0-freeze d29a918): the checks of the freeze spec the compat tests miss.

See records/2026-11-perf/sessions/v10-freeze/review_report.md of the project's records. These tests pass on
d29a918 and fail under the mutations the report describes (M2: a functional-key value written in another encoding
that the reader still accepts; M3: a stable scheme's fixture dropped together with its manifest entry; M4: a
reviewed scheme moved to experimental).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from pyfe4ai.core import stability
from tests.compat import recipes

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RELEASES = sorted(p for p in FIXTURES.iterdir() if p.is_dir()) if FIXTURES.is_dir() else []
FILES = [f for d in RELEASES for f in sorted(d.glob("*__*.json"))]


def _param(path: Path):
    scheme = path.stem.replace("__", "/")
    return pytest.param(path, id="{}:{}".format(path.parent.name, scheme),
                        marks=pytest.mark.skipif(not recipes.available(scheme), reason="pairing backend"))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


_DEC = re.compile(r"-?[0-9]+")
_HEX0X = re.compile(r"-?0x[0-9a-f]+")


def _signature(value):
    """The encoding of a value, not its content: types, and for strings the alphabet of a number."""
    if isinstance(value, dict):
        return ("dict", tuple(sorted((k, _signature(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return ("list", frozenset(_signature(v) for v in value))
    if isinstance(value, bool) or value is None:
        return type(value).__name__
    if isinstance(value, (int, float, bytes)):
        return type(value).__name__
    if isinstance(value, str):
        if _DEC.fullmatch(value):
            return "str:decimal"
        if _HEX0X.fullmatch(value):
            return "str:0x-hex"
        return "str"
    return type(value).__name__


def _encoding_changes(old, new, where=""):
    """Fields of *old* whose encoding differs in *new* (fields new adds are allowed, R2)."""
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k, v in old.items():
            if k in new:
                out += _encoding_changes(v, new[k], "{}.{}".format(where, k))
        return out
    if _signature(old) != _signature(new) and not (isinstance(old, dict) or isinstance(new, dict)):
        out.append(where)
    return out


@pytest.mark.parametrize("path", [_param(f) for f in FILES])
def test_r2_no_field_changed_its_encoding(path):
    """R2: existing fields are not removed, renamed **or re-encoded**. The compat tests compare field names only;
    a value written in another encoding that the reader still accepts passes them (mutation M2)."""
    fx = recipes.load(path)
    fresh = recipes.generate(fx["scheme"])
    changed = []
    for role, record in fx["records"].items():
        if role in fresh["records"]:
            changed += _encoding_changes(record, fresh["records"][role], role)
    assert not changed, "{}: fields re-encoded: {}".format(fx["scheme"], changed)


def test_every_stable_scheme_has_a_fixture_in_some_release():
    """Sec. 8.4 item 5: every stable scheme has a fixture (in the set of the release it became stable in). The
    release test's subset check filters by the manifest first and cannot fail (mutation M3)."""
    on_disk = {f.stem.replace("__", "/") for d in RELEASES for f in d.glob("*__*.json")}
    missing = sorted(set(stability.stable_schemes()) - on_disk)
    assert not missing, "stable schemes without a fixture: {}".format(missing)


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_a_scheme_with_a_fixture_never_left_the_stable_tier(path):
    """Sec. 2.4: stable -> experimental is not allowed (stable -> known-insecure is, by the security exception).
    The release test checks the tier stored in the frozen file, which never changes (mutation M4)."""
    scheme = recipes.load(path)["scheme"]
    assert stability.tier(scheme) in (stability.STABLE, stability.KNOWN_INSECURE_TIER), \
        "{} has a fixture but is now {}".format(scheme, stability.tier(scheme))
