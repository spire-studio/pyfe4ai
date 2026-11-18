"""The 1.x compatibility promise, checked against frozen record fixtures (roadmap v1.0/02, Sec. 3 and 8).

``tests/compat/fixtures/<version>/`` holds the records one release wrote for every stable scheme
(:mod:`tests.compat.gen_fixtures`). Every later version must

* read them and compute the same results (R1): the stored ciphertexts decrypt under the stored keys, a fresh
  ciphertext under the stored encryption keys decrypts with the stored functional key, stored partial
  decryptions combine, stored key shares serve, and the stored gradients (the ML adapter's wire form of the
  ndarray layers) decrypt;
* keep their fields (R2): a record written now has every field the fixture's record of the same role has;
* reuse their parameter records without regenerating them (R5);
* keep the version markers (R4): every marker of the snapshot, with its value.

The fixtures are data: these tests never rewrite them. ``0.5.0-proto`` is the prototype set written by 0.5.0
while the policy was drafted; the frozen set of the release is ``1.0.0``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from pyfe4ai.core import stability
from tests.compat import recipes
from tests.compat.gen_fixtures import marker_snapshot

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RELEASES = sorted(p for p in FIXTURES.iterdir() if p.is_dir()) if FIXTURES.is_dir() else []
FILES = [f for d in RELEASES for f in sorted(d.glob("*__*.json"))]
NO_PAIRING = "needs the native extension (the BLS12-381 backend) for the pairing fixtures"


def _param(path: Path):
    scheme = path.stem.replace("__", "/")
    return pytest.param(path, id="{}:{}".format(path.parent.name, scheme),
                        marks=pytest.mark.skipif(not recipes.available(scheme), reason=NO_PAIRING))


PARAMS = [_param(f) for f in FILES]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("path", PARAMS)
def test_the_records_still_decrypt(path):
    fx = recipes.load(path)
    assert fx["fixture_format"] == recipes.FIXTURE_FORMAT
    assert recipes.replay(fx) == recipes.expected_replay(fx)


@pytest.mark.parametrize("path", PARAMS)
def test_the_ndarray_records_still_decrypt(path):
    fx = recipes.load(path)
    if "ndarray" not in fx:
        pytest.skip("no ndarray step in this recipe")
    assert recipes.replay_ndarray(fx["scheme"], fx["ndarray"]) == fx["ndarray"]["expected"]


@pytest.mark.parametrize("path", PARAMS)
def test_the_parameter_records_are_reused(path):
    fx = recipes.load(path)
    for part in [fx] + ([dict(fx["ndarray"], scheme=fx["scheme"])] if "ndarray" in fx else []):
        before, after, warned = recipes.reuse_param_records(part)
        assert after == before, "the parameter record was rewritten"
        assert not warned


@pytest.mark.parametrize("path", PARAMS)
def test_no_field_was_dropped(path):
    fx = recipes.load(path)
    fresh = recipes.generate(fx["scheme"])
    pairs = [(fx["records"], fresh["records"], "")]
    if "ndarray" in fx:
        pairs.append((fx["ndarray"]["records"], fresh.get("ndarray", {}).get("records", {}), "ndarray "))
    for old, new, where in pairs:
        for role, record in old.items():
            missing = recipes.record_keys(record) - recipes.record_keys(new.get(role, {}))
            assert not missing, "{}: fields {} of the {}{} record are no longer written".format(
                fx["scheme"], sorted(missing), where, role)


@pytest.mark.parametrize("release", RELEASES, ids=[d.name for d in RELEASES])
def test_the_version_markers_are_kept(release):
    frozen = recipes.load(release / "markers.json")
    now = marker_snapshot()
    gone = sorted(set(frozen) - set(now))
    changed = sorted(name for name in set(frozen) & set(now) if recipes.encode(now[name]) != recipes.encode(frozen[name])
                     and not _extends(frozen[name], now[name]))
    assert not gone and not changed, "markers removed {}, changed {}".format(gone, changed)


def _extends(old, new) -> bool:
    """A table of ids may gain entries; an id or a value may not change."""
    if isinstance(old, dict) and isinstance(new, dict):
        return all(k in new and recipes.encode(new[k]) == recipes.encode(v) for k, v in old.items())
    if isinstance(old, tuple) and isinstance(new, tuple):
        return new[: len(old)] == old
    return False


@pytest.mark.parametrize("release", RELEASES, ids=[d.name for d in RELEASES])
def test_every_stable_scheme_of_the_release_has_its_fixture(release):
    manifest = recipes.load(release / "manifest.json")
    on_disk = {f.stem.replace("__", "/") for f in release.glob("*__*.json")}
    assert manifest["failed"] == {}
    assert set(manifest["schemes"]) == on_disk
    # a scheme stable now and at the release has its fixture (schemes promoted later get theirs in a later set)
    assert not ({s for s in stability.stable_schemes() if s in set(manifest["schemes"]) | on_disk} - on_disk)
    for path in release.glob("*__*.json"):
        assert recipes.load(path)["tier"] == stability.STABLE


def test_the_tiers_follow_the_registry():
    import pyfe4ai

    stability._check_tables()
    for scheme in pyfe4ai.list_schemes():
        status = pyfe4ai.get_scheme(scheme).status.status
        assert stability.tier(scheme) == stability.TIER_OF_STATUS[status]
    assert stability.stable_schemes() == pyfe4ai.list_schemes(status="reviewed")
    assert stability.config_tier("sife/fullysec_ring_lwe", {"ndarray_layout": "packed"}) == "experimental"
    assert stability.config_tier("sife/fullysec_ring_lwe", {"ndarray_layout": "elementwise"}) == "stable"
    assert stability.config_tier("mcfe/kp_ab_pairing") == "known-insecure"


def test_the_policy_compiler_marker_is_listed_with_the_others():
    from pyfe4ai.core import versions
    from pyfe4ai.policy import encoding

    assert versions.POLICY_COMPILER == encoding.POLICY_COMPILER
