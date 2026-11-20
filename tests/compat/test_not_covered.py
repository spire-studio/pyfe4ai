"""The "Not covered" list of docs/compatibility.rst names every package that the promise leaves out.

Every subpackage of ``pyfe4ai`` and of ``pyfe4ai.schemes`` has a place here: covered by the promise, internal, or
experimental. A new subpackage fails until it is placed; an internal or experimental one must be named under "Not
covered", and so must the package of a scheme family whose schemes are all experimental (from the registry). The
packages that ``import pyfe4ai`` loads are checked too, since a reader sees them without importing them (as
``pyfe4ai.circuits``).
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import pyfe4ai
import pyfe4ai.schemes
from pyfe4ai.core import registry

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "compatibility.rst"

pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="the sources of the documentation are not in this tree")

COVERED, INTERNAL, EXPERIMENTAL = "covered", "internal", "experimental"

#: The subpackages of pyfe4ai and what the promise says of them (core is covered in part: its record classes and
#: the formats of experimental parts are named under "Not covered" one by one).
TOP = {
    "core": COVERED,
    "schemes": COVERED,
    "ml": COVERED,
    "policy": COVERED,
    "backend": INTERNAL,
    "primitives": INTERNAL,
    "ac": EXPERIMENTAL,
    "circuits": EXPERIMENTAL,
}

#: The subpackages of pyfe4ai.schemes: a scheme family's package takes the tier of its schemes (below).
SCHEMES = {
    "common": INTERNAL,
    "threshold": INTERNAL,
}


def _not_covered() -> str:
    text = PAGE.read_text(encoding="utf-8")
    return " ".join(text.split("Not covered\n-----------", 1)[1].split())


def _subpackages(package) -> set:
    return {info.name for info in pkgutil.iter_modules(package.__path__) if info.ispkg}


def _family_places() -> dict:
    statuses = {}
    for scheme in registry.SCHEMES:
        statuses.setdefault(scheme.family, set()).add(scheme.status.status)
    return {family: EXPERIMENTAL if found <= {registry.EXPERIMENTAL, registry.UNREVIEWED} else COVERED
            for family, found in statuses.items()}


def test_every_subpackage_has_a_place():
    assert _subpackages(pyfe4ai) == set(TOP)
    assert _subpackages(pyfe4ai.schemes) == set(SCHEMES) | set(_family_places())


def test_the_packages_outside_the_promise_are_named_under_not_covered():
    section = _not_covered()
    places = {**{f"pyfe4ai.{name}": place for name, place in TOP.items()},
              **{f"pyfe4ai.schemes.{name}": place for name, place in {**SCHEMES, **_family_places()}.items()}}
    missing = [name for name, place in places.items() if place != COVERED and name not in section]
    assert not missing, missing
    # a covered package is not listed as left out
    listed = [name for name, place in places.items() if place == COVERED and f"``{name}``" in section]
    assert not listed, listed


def test_what_import_pyfe4ai_loads_is_covered_or_named():
    loaded = subprocess.run(
        [sys.executable, "-c", "import sys, pyfe4ai; print(' '.join(sorted({m.split('.')[1] for m in sys.modules "
                               "if m.startswith('pyfe4ai.')})))"],
        capture_output=True, text=True, check=True, cwd=ROOT).stdout.split()
    assert set(loaded) <= set(TOP), sorted(set(loaded) - set(TOP))
    section = _not_covered()
    for name in loaded:
        if TOP[name] != COVERED:
            assert f"pyfe4ai.{name}" in section, name
