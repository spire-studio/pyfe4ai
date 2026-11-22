"""The 1.x compatibility promise, checked against frozen record fixtures (``docs/compatibility.rst``).

``tests/compat/fixtures/<version>/`` holds the records one release wrote for every stable scheme
(:mod:`tests.compat.gen_fixtures`). Every later version must

* read them and compute the same results (R1): the stored ciphertexts decrypt under the stored keys, a fresh
  ciphertext under the stored encryption keys decrypts with the stored functional key, stored partial
  decryptions combine, stored key shares serve, and the stored gradients (the ML adapter's wire form of the
  ndarray layers) decrypt;
* keep their fields and their encodings (R2): a record written now has every field the fixture's record of the
  same role has, at every depth, in the same encoding;
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

#: The stable schemes of each fixture set, frozen when the set was written. The registry may change later -- a
#: scheme may become known-insecure (the security exception), never experimental, and is never removed -- and a
#: later set adds the schemes that became stable since.
STABLE_AT = {
    "0.5.0-proto": (
        "sife/ddh", "sife/lwe", "sife/paillier", "sife/damgard", "sife/ring_lwe", "sife/fullysec_ring_lwe",
        "sife/fullysec_lwe", "sife/ddh_dynamic", "sife/fh_ipe_pairing", "sife/part_fh_ipe_pairing",
        "sife/cp_ab_pairing", "sife/kp_ab_pairing", "mife/ddh", "mife/lwe", "mife/paillier", "mife/damgard",
        "mife/ring_lwe", "mife/fullysec_lwe", "mife/fh_ipe_pairing", "mife/fh_multi_ipe_pairing",
        "mife/ddh_threshold", "mcfe/ddh", "mcfe/lwe", "mcfe/paillier", "mcfe/damgard", "mcfe/ring_lwe",
        "mcfe/fullysec_lwe", "mcfe/fh_multi_ipe_pairing", "mcfe/fh_multi_ipe_pairing_decentralized",
        "mcfe/ddh_threshold", "mcfe/lwe_decentralized", "mcfe/ring_lwe_decentralized",
        "mcfe/paillier_decentralized", "quadratic/quad", "quadratic/sgp", "quadratic/miqfe",
    ),
    # the frozen set of the release (written by 80aae14, the 1.0.0 RC, on the native backend): the same 36 schemes
    "1.0.0": (
        "sife/ddh", "sife/lwe", "sife/paillier", "sife/damgard", "sife/ring_lwe", "sife/fullysec_ring_lwe",
        "sife/fullysec_lwe", "sife/ddh_dynamic", "sife/fh_ipe_pairing", "sife/part_fh_ipe_pairing",
        "sife/cp_ab_pairing", "sife/kp_ab_pairing", "mife/ddh", "mife/lwe", "mife/paillier", "mife/damgard",
        "mife/ring_lwe", "mife/fullysec_lwe", "mife/fh_ipe_pairing", "mife/fh_multi_ipe_pairing",
        "mife/ddh_threshold", "mcfe/ddh", "mcfe/lwe", "mcfe/paillier", "mcfe/damgard", "mcfe/ring_lwe",
        "mcfe/fullysec_lwe", "mcfe/fh_multi_ipe_pairing", "mcfe/fh_multi_ipe_pairing_decentralized",
        "mcfe/ddh_threshold", "mcfe/lwe_decentralized", "mcfe/ring_lwe_decentralized",
        "mcfe/paillier_decentralized", "quadratic/quad", "quadratic/sgp", "quadratic/miqfe",
    ),
}



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
def test_no_field_was_dropped_or_re_encoded(path):
    """R2 at every depth: a record written now has every field of the fixture's record of the same role, nested
    fields included, each in the same encoding (type, and for strings the form of the number or text); fields may
    be added (review of the freeze line, RF-1 and RF-3)."""
    fx = recipes.load(path)
    fresh = recipes.generate(fx["scheme"])
    pairs = [(fx["records"], fresh["records"], "")]
    if "ndarray" in fx:
        pairs.append((fx["ndarray"]["records"], fresh.get("ndarray", {}).get("records", {}), "ndarray "))
    problems = []
    for old, new, where in pairs:
        for role, record in old.items():
            if role not in new:
                problems.append("{}{}: no longer written".format(where, role))
                continue
            dropped, changed = recipes.r2_differences(record, new[role], role)
            problems += ["{}{}: dropped".format(where, d) for d in dropped]
            problems += ["{}{}: re-encoded".format(where, c) for c in changed]
    assert not problems, "{}: {}".format(fx["scheme"], problems[:10])


#: Markers of experimental parts (bcfe, the threshold key issuance, the identity directory): the snapshot lists them,
#: but the promise covers the records of stable schemes only (docs/compatibility.rst, *Serialized records*), so they
#: may change; a change updates this list, not the fixture.
EXPERIMENTAL_MARKERS = frozenset({
    "BCFE_GC", "BCFE_GC_HASH", "BCFE_KEM", "BCFE_MARKERS", "BCFE_VERSION",
    "CP_AB_KEYISSUE_PROTOCOL", "CP_AB_KEYREQUEST_VERSION", "CP_AB_KEYSERVERS_VERSION", "CP_AB_KEYSERVER_SHARE_VERSION",
    "CP_AB_PARTIAL_KEY_VERSION", "IDENTITY_DIRECTORY_VERSION",
})


@pytest.mark.parametrize("release", RELEASES, ids=[d.name for d in RELEASES])
def test_the_version_markers_are_kept(release):
    frozen = {name: value for name, value in recipes.load(release / "markers.json").items()
              if name not in EXPERIMENTAL_MARKERS}
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
def test_the_set_holds_a_fixture_of_every_scheme_stable_at_its_release(release):
    """The fixtures on disk, the set's manifest and the frozen list of the set (STABLE_AT) name the same schemes,
    each written as stable (review of the freeze line, RF-2: the check used to compare the manifest with itself)."""
    assert release.name in STABLE_AT, "fixture set {} has no frozen list in STABLE_AT".format(release.name)
    manifest = recipes.load(release / "manifest.json")
    on_disk = {f.stem.replace("__", "/") for f in release.glob("*__*.json")}
    assert manifest["failed"] == {}
    assert on_disk == set(manifest["schemes"]) == set(STABLE_AT[release.name])
    for path in release.glob("*__*.json"):
        assert recipes.load(path)["tier"] == stability.STABLE


def test_no_scheme_of_a_fixture_set_left_the_stable_tier():
    """Sec. 2.4 against the live registry: a scheme that was stable when a set was written is still registered and
    is stable, or known-insecure by the security exception; never experimental (RF-2, RF-4)."""
    import pyfe4ai

    registered = set(pyfe4ai.list_schemes())
    for release, schemes in STABLE_AT.items():
        for scheme in schemes:
            assert scheme in registered, "{} (stable in {}) is no longer registered".format(scheme, release)
            assert stability.tier(scheme) in (stability.STABLE, stability.KNOWN_INSECURE_TIER), \
                "{} (stable in {}) is now {}".format(scheme, release, stability.tier(scheme))


def test_every_stable_scheme_has_a_fixture_set():
    """Every scheme stable now is in the frozen list of some set: a scheme promoted to stable gets its fixtures in
    the version it is promoted in (Sec. 8.4 item 5)."""
    frozen = set().union(*STABLE_AT.values())
    missing = sorted(set(stability.stable_schemes()) - frozen)
    assert not missing, "stable schemes without a fixture set: {}".format(missing)


def test_the_r2_comparison_itself():
    """recipes.r2_differences on made-up records: nested drops and re-encodings are found, additions are not."""
    old = {"pp": {"q": "12345", "g": {"x": "0a1b", "y": [1, 2]}}, "ct": [{"c": "17", "tag": "QUJDREVGR0hJSktMTU5PUA=="}]}
    assert recipes.r2_differences(old, old) == ([], [])
    added = {"pp": {"q": "12345", "g": {"x": "0a1b", "y": [1, 2], "z": 3}, "new": 1},
             "ct": [{"c": "17", "tag": "QUJDREVGR0hJSktMTU5PUA==", "more": "x"}]}
    assert recipes.r2_differences(old, added) == ([], [])
    nested_drop = {"pp": {"q": "12345", "g": {"y": [1, 2]}}, "ct": [{"c": "17", "tag": "QUJDREVGR0hJSktMTU5PUA=="}]}
    assert recipes.r2_differences(old, nested_drop) == (["pp.g.x"], [])
    re_encoded = {"pp": {"q": "0x3039", "g": {"x": "0a1b", "y": ["1", "2"]}},
                  "ct": [{"c": 17, "tag": "QUJDREVGR0hJSktMTU5PUA=="}]}
    assert recipes.r2_differences(old, re_encoded) == ([], ["pp.q", "pp.g.y[]", "ct[0].c"])
    assert recipes.encoding_of("ab12") == "str:hex" and recipes.encoding_of("12") == "str:decimal"


def test_every_experimental_part_the_registry_names_is_in_the_stability_tables():
    """What the registry's text marks experimental on a stable scheme is in pyfe4ai.core.stability (freeze spec
    Sec. 2.2; review INT-1): a capability whose description says "experimental" is in EXPERIMENTAL_CAPABILITIES of
    every stable scheme that has it, and an option value a stable scheme's note calls experimental ('Experimental:
    option = "value"') is in EXPERIMENTAL_OPTIONS. Experimental schemes need no entry (bcfe's bounded_keys)."""
    import re

    from pyfe4ai.core import registry

    marked_caps = {cap for cap, text in registry.CAPABILITIES.items() if "experimental" in text.lower()}
    assert "distributed_keygen" in marked_caps
    for scheme in registry.SCHEMES:
        if stability.tier(scheme.id) != stability.STABLE:
            continue
        for cap in scheme.capabilities & marked_caps:
            assert cap in stability.EXPERIMENTAL_CAPABILITIES.get(scheme.id, frozenset()), (scheme.id, cap)
        for option, value in re.findall(r'Experimental: (\w+) = "([^"]+)"', scheme.status.note):
            assert value in stability.EXPERIMENTAL_OPTIONS.get(scheme.id, {}).get(option, frozenset()), \
                (scheme.id, option, value)
    # and nothing is listed that the registry does not mark
    for scheme_id, caps in stability.EXPERIMENTAL_CAPABILITIES.items():
        assert set(caps) <= marked_caps, (scheme_id, caps)


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
