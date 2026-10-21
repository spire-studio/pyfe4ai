"""The version and compatibility decisions of :mod:`pyfe4ai.core.versions`:
what an absent marker means, and which artifacts are accepted or refused."""

from __future__ import annotations

import pickle

import pytest

from pyfe4ai.core import versions
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FESecurityWarning, FEValidationError
from pyfe4ai.primitives.hashing import LABEL_HASH_CODES
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH
from pyfe4ai.primitives.sampling import FLOODING_SAMPLER
from pyfe4ai.core import config


def test_absent_markers_mean_the_derivations_of_earlier_versions():
    assert versions.pp_label_hash({}) == versions.key_label_hash({}) == LEGACY_LABEL_HASH == "shake256-wide"
    assert versions.ciphertext_label_hash({}) == LEGACY_LABEL_HASH
    assert versions.pp_ct_format({}) == "json"
    assert versions.pp_flooding({}) == versions.LEGACY_FLOODING == "gaussian"
    assert versions.LEGACY_SHARING == "shamir"
    assert versions.key_share_sharing_id({}) is None
    assert versions.key_share_part({}) is None
    assert versions.label_hash_code(LEGACY_LABEL_HASH) is None
    assert versions.label_hash_code("shake128-rejection") == LABEL_HASH_CODES["shake128-rejection"] == 2
    assert versions.RECORD_ABSENT == {"ring_modulus": "prime", "ndarray_layout": "elementwise"}


def test_unknown_markers_are_refused():
    with pytest.raises(FEValidationError, match="unknown label hash"):
        versions.pp_label_hash({"label_hash": "sha1"})
    with pytest.raises(FEValidationError, match="unknown ciphertext format"):
        versions.pp_ct_format({"ct_format": "xml"})
    assert versions.ciphertext_label_hash({"label_hash": 7}) is None
    assert versions.ciphertext_label_hash({"label_hash": True}) is None  # a bool is not a code
    with pytest.raises(FEValidationError, match="unknown flooding"):
        versions.check_key_flooding("laplace", {}, ("uniform", "gaussian"))


def test_label_hash_checks():
    versions.check_key_label_hash({}, LEGACY_LABEL_HASH)
    with pytest.raises(FESchemeError, match="^the key share was derived with another label hash"):
        versions.check_key_label_hash({}, "shake128-rejection", "the key share was")
    with pytest.raises(FEKeyError, match="issued for label hash 'shake256-wide'"):
        versions.check_encryption_key_label_hash({}, "shake128-rejection")
    assert versions.check_encryption_key_label_hash({"label_hash": "shake128-rejection"},
                                                    "shake128-rejection") == "shake128-rejection"
    versions.check_ciphertext_label_hashes([{"label_hash": 2}], "shake128-rejection")
    with pytest.raises(FESchemeError, match="a ciphertext of c1 was produced"):
        versions.check_ciphertext_label_hashes([{}], "shake128-rejection", "c1")
    # dense MCFE-LWE: an element without a code takes its layer's, a single ciphertext v0.3.0's
    versions.check_lwe_ciphertext_label_hash({}, "c1", "shake128-rejection", absent=2)
    with pytest.raises(FESchemeError, match="the ciphertext of c1"):
        versions.check_lwe_ciphertext_label_hash({}, "c1", "shake128-rejection", absent=0)
    with pytest.raises(FESchemeError, match="a ciphertext layer of c1"):
        versions.check_lwe_layer_label_hash(None, "c1", "shake128-rejection")


def test_flooding_derivation_and_distribution():
    good = {"noise_version": versions.FLOODING_NOISE_VERSION, "noise_sampler": FLOODING_SAMPLER}
    versions.check_flooding_version(good)
    with pytest.raises(FEKeyError, match="flooding derivation 1"):
        versions.check_flooding_version({"noise_sampler": FLOODING_SAMPLER})
    with pytest.raises(FEKeyError, match="flooding sampler"):
        versions.check_flooding_version({"noise_version": versions.FLOODING_NOISE_VERSION})
    versions.check_key_flooding("gaussian", {}, ("uniform", "gaussian"))
    with pytest.raises(FEKeyError, match="another flooding"):
        versions.check_key_flooding("uniform", {}, ("uniform", "gaussian"))


def test_construction_ids_and_sharings():
    assert config.KL == "kl" and config.PER_SAMPLE == "per-sample"
    assert versions.tmcfe_ring_lwe_construction_id("shamir", "gaussian", "kl") == versions.TMCFE_RING_LWE_CONSTRUCTION_KL
    assert versions.tmcfe_ring_lwe_construction_id("replicated", "gaussian", "kl") is None
    assert versions.tmcfe_ring_lwe_construction_id("replicated", "uniform", "per-sample") == \
        "bmms21-ibmcfe-replicated-smudging-uniform/v1"
    shamir_pp = {}  # versions <= 0.3.0: no sharing, no construction id
    versions.check_served_sharing(shamir_pp, {}, versions.TMCFE_LWE_CONSTRUCTIONS)
    with pytest.raises(FESchemeError, match="must record threshold_construction"):
        versions.check_served_sharing({"sharing": "replicated"}, {"sharing": "replicated"},
                                      versions.TMCFE_LWE_CONSTRUCTIONS)
    with pytest.raises(FESchemeError, match="not of the sharing"):
        versions.check_served_sharing({"sharing": "replicated"}, {}, versions.TMCFE_LWE_CONSTRUCTIONS)
    with pytest.raises(FESchemeError, match="unknown threshold construction"):
        versions.check_served_sharing({"threshold_construction": "later/v9"}, {}, versions.TMCFE_LWE_CONSTRUCTIONS)
    versions.check_tmife_lwe_construction({"threshold_construction": versions.TMIFE_LWE_CONSTRUCTION_KL}, kl=True)
    with pytest.raises(FESchemeError, match="not that of their accounting"):
        versions.check_tmife_lwe_construction({"threshold_construction": versions.TMIFE_LWE_CONSTRUCTION}, kl=True)
    with pytest.raises(FESchemeError, match="not that of their flooding"):
        versions.check_tmcfe_ring_lwe_construction(
            {"threshold_construction": versions.TMCFE_RING_LWE_CONSTRUCTION_IDS[("shamir", "uniform")]},
            "shamir", "gaussian", "per-sample")


def test_legacy_shamir_policy():
    assert versions.legacy_shamir_policy({}) == "refuse"
    with pytest.raises(FEValidationError):
        versions.legacy_shamir_policy({"legacy_shamir": "allow"})
    versions.check_legacy_shamir("sid_0", 3, 3, "refuse", None)  # t = s: served
    with pytest.raises(FESchemeError, match="legacy_shamir='warn' serves it"):
        versions.check_legacy_shamir("sid_0", 2, 3, "refuse", None)
    with pytest.raises(FESchemeError):
        versions.check_legacy_shamir("sid_0", 2, 3, "warn", "standard")
    with pytest.warns(FESecurityWarning, match="t = 2 < s = 3"):
        versions.check_legacy_shamir("sid_0", 2, 3, "warn", "toy")


def test_sharing_ids():
    with pytest.raises(FEKeyError, match="sharing_id is a string"):
        versions.key_share_sharing_id({"sharing_id": 5})
    versions.check_partials_sharing([{"sharing_id": "a"}, {"sharing_id": "a"}])
    versions.check_partials_sharing([{}, {"sharing_id": None}])  # partials without one combine with each other
    with pytest.raises(versions.MixedSharingsError, match="fetch the label parts again"):
        versions.check_partials_sharing([{"sharing_id": "a"}, {}])
    with pytest.raises(FESchemeError, match="malformed partial decryption"):
        versions.check_partials_sharing([{"sharing_id": 1}])
    assert issubclass(versions.MixedSharingsError, FESchemeError)


def test_records_match_field_by_field():
    assert versions.record_matches({"construction": "x/v1", "n": 3}, {"construction": "x/v1", "n": 3})
    assert not versions.record_matches({"construction": "x/v0", "n": 3}, {"construction": "x/v1", "n": 3})
    assert versions.record_matches({}, {"ring_modulus": "prime", "ndarray_layout": "elementwise", "q_factors": None})
    assert not versions.record_matches({}, {"ring_modulus": "rns"})

    class KeyGenerator:
        _holds_dcr_modulus = True
        security_profile = "standard"

    assert not versions.caches_record(KeyGenerator())
    KeyGenerator.security_profile = "toy"
    assert versions.caches_record(KeyGenerator())


class _State:
    pass


def test_state_versions():
    obj = _State()
    versions.check_state(obj, versions.versioned_state({"a": 1}, "v"), "v")
    assert obj.a == 1
    with pytest.raises(FESchemeError, match="this _State was pickled by an incompatible version"):
        versions.check_state(_State(), {"a": 1}, "v")
    assert pickle.loads(pickle.dumps(versions.MixedSharingsError(versions.MIXED_SHARINGS))).args == (
        versions.MIXED_SHARINGS,)


def test_declared_absences_are_the_ones_versions_applies():
    """Every field a record declares with a meaning when absent
    (``Field(absent=...)``) has the default that core.versions (or the
    scheme, through core.config) applies to it."""
    import importlib
    import inspect

    from pyfe4ai.core.formats import lwe, ring_lwe
    from pyfe4ai.core.formats.record import Record, _REQUIRED

    seen = set()
    for name in ("lwe", "ring_lwe", "dl", "pairing", "threshold"):
        module = importlib.import_module("pyfe4ai.core.formats." + name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not (issubclass(cls, Record) and cls.__module__ == module.__name__):
                continue
            for field_name, field in cls._fields.items():
                if field.absent is _REQUIRED:
                    continue
                seen.add(field_name)
                absent = field.absent
                if field_name == "label_hash" and cls in (lwe.MCFELWECiphertext, ring_lwe.RingLWECiphertext):
                    assert absent is None, cls  # code 0 alone, the layer's code in a layer (see the doc)
                elif field_name == "label_hash" and cls is ring_lwe.RingLWELayer:
                    assert absent == 0 and versions.ciphertext_label_hash({}) == LEGACY_LABEL_HASH
                elif field_name == "label_hash":
                    assert absent == versions.key_label_hash({}) == versions.pp_label_hash({}) == LEGACY_LABEL_HASH
                elif field_name == "sharing":
                    assert absent == versions.LEGACY_SHARING
                elif field_name == "flooding":
                    assert absent == versions.LEGACY_FLOODING == versions.pp_flooding({})
                elif field_name == "ct_format":
                    assert absent == versions.pp_ct_format({})
                elif field_name in ("ring_modulus", "ndarray_layout"):
                    assert absent == versions.RECORD_ABSENT[field_name]
                elif field_name == "flooding_accounting":
                    assert absent == config.PER_SAMPLE
                elif field_name == "noise_version":
                    for dk in ({}, {"noise_version": absent}):
                        with pytest.raises(FEKeyError, match="flooding derivation 1"):
                            versions.check_flooding_version(dk)
                elif field_name == "sharing_id":
                    assert absent is None and versions.key_share_sharing_id({}) is None
                elif field_name == "part":
                    assert absent is None and versions.key_share_part({}) is None
                else:
                    assert field_name in ("threshold_construction", "q_factors", "max_partial_decryptions"), field_name
                    assert absent is None
    assert {"label_hash", "sharing", "flooding", "noise_version", "sharing_id"} <= seen
