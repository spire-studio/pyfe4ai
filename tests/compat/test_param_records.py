"""R2 and R5 for the parameter records the current version writes.

The compatibility promise lists parameter records among the records whose fields are never removed, renamed or
re-encoded (docs/compatibility.rst, *Serialized records*), and a 1.x key generator must reuse what it cached.
``test_no_field_was_dropped_or_re_encoded`` compares the fixture's records of every role with fresh ones, but not
its ``param_records``; ``test_the_parameter_records_are_reused`` sets up on the fixture's parameter records, never
on those the current version writes. So a writer that re-encodes a field of a parameter record (a decimal string
as 0x-hex) or writes a value it then does not match on reload (a rounded sigma) passes the whole test suite.

Each of these tests fails under one of those two mutations of ``sife/ring_lwe``'s parameter writer.
"""

from __future__ import annotations

import json

import pytest

from tests.compat import recipes
from tests.compat.test_fixtures import PARAMS


def _json(text: str):
    return json.loads(text)


@pytest.mark.parametrize("path", PARAMS)
def test_parameter_records_keep_their_fields_and_encodings(path):
    fx = recipes.load(path)
    if not fx.get("param_records"):
        pytest.skip("no parameter record in this fixture")
    fresh = recipes.generate(fx["scheme"])
    problems = []
    for part_old, part_new, where in [(fx, fresh, "")] + (
            [(fx["ndarray"], fresh.get("ndarray", {}), "ndarray ")] if "ndarray" in fx else []):
        for rel, text in part_old.get("param_records", {}).items():
            new = part_new.get("param_records", {}).get(rel)
            if new is None:
                problems.append("{}{}: no longer written".format(where, rel))
                continue
            dropped, changed = recipes.r2_differences(_json(text), _json(new), rel)
            problems += ["{}{}: dropped".format(where, d) for d in dropped]
            problems += ["{}{}: re-encoded".format(where, c) for c in changed]
    assert not problems, "{}: {}".format(fx["scheme"], problems[:10])


@pytest.mark.parametrize("path", PARAMS)
def test_the_parameter_records_written_now_are_reused(path):
    fx = recipes.load(path)
    fresh = recipes.generate(fx["scheme"])
    if not fresh.get("param_records"):
        pytest.skip("no parameter record for this scheme")
    part = dict(fresh, config=fx["config"])
    before, after, warned = recipes.reuse_param_records(part)
    assert after == before, "{}: a parameter record this version wrote was rewritten".format(fx["scheme"])
    assert not warned, warned
