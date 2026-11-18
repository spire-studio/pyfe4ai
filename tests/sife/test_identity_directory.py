"""The identity directory over sife/fullysec_ring_lwe (pyfe4ai.ac, experimental)."""

import copy
import json
import os

import numpy as np
import pytest

from pyfe4ai.ac import IdentityDirectory, IdentityKeyCentre
from pyfe4ai.ac import identity_directory as idd
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.versions import IDENTITY_DIRECTORY_VERSION

CFG = {"eta": 3, "bound_x": 4, "bound_y": 4}


@pytest.fixture(autouse=True)
def _param_cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


@pytest.fixture
def centre():
    kc = IdentityKeyCentre(dict(CFG))
    for identity in ("bob", "alice", "Alice"):
        kc.add_identity(identity)
    return kc


def _json(obj):
    return json.loads(json.dumps(obj))


# -- correctness ----------------------------------------------------------------


def test_each_identity_decrypts_its_own_ciphertexts(centre):
    record = _json(centre.directory())
    d = IdentityDirectory(record, expected_digest=record["digest"])
    assert d.identities() == ["Alice", "alice", "bob"]
    for identity, x, y in (("alice", [1, 0, 4], [1, 2, 3]), ("bob", [-4, 4, 2], [4, -1, 0]),
                           ("Alice", [3, 3, 3], [-4, -4, 4])):
        dk = _json(centre.get_decryption_keys(identity, y))
        ct = _json(d.encrypt(identity, x))
        assert ct["version"] == IDENTITY_DIRECTORY_VERSION and ct["identity"] == identity
        assert d.decrypt(ct, dk, y) == sum(a * b for a, b in zip(x, y))


@pytest.mark.parametrize("layout", ["elementwise", "packed"])
def test_ndarrays(layout):
    kc = IdentityKeyCentre({"eta": 1, "bound_x": 100, "bound_y": 1, "ndarray_layout": layout})
    kc.add_identity("alice")
    d = IdentityDirectory(kc.directory())
    dk = kc.get_decryption_keys("alice", [1])
    arrays = [np.array([[1, -2], [3, 100]]), np.arange(-40, 41).reshape(9, 9)]
    layers = d.encrypt_lst_ndarray("alice", arrays)
    for got, want in zip(d.decrypt_lst_ndarray_ct(layers, dk, [1]), arrays):
        assert np.array_equal(got, want)
    kc.add_identity("bob")
    d = IdentityDirectory(kc.directory())
    with pytest.raises(FEKeyError):
        d.decrypt_lst_ndarray_ct(layers, kc.get_decryption_keys("bob", [1]), [1])


def test_standard_profile_packed_smoke():
    kc = IdentityKeyCentre({"eta": 1, "bound_x": 100, "bound_y": 1, "ndarray_layout": "packed",
                            "security_profile": "standard"})
    kc.add_identity("alice")
    d = IdentityDirectory(kc.directory())
    assert d.public_parameters["security_profile"] == "standard"
    layers = d.encrypt_lst_ndarray("alice", [np.array([5, -7, 0])])
    got = d.decrypt_lst_ndarray_ct(layers, kc.get_decryption_keys("alice", [1]), [1])
    assert np.array_equal(got[0], np.array([5, -7, 0]))


# -- the instances ------------------------------------------------------------------


def test_instances_share_the_parameters_and_have_their_own_keys(centre):
    record = centre.directory()
    pks = [e["pk"] for e in record["entries"]]
    assert len({json.dumps(pk) for pk in pks}) == len(pks)
    assert centre._instances["alice"].a is centre._instances["bob"].a
    assert centre._instances["alice"].msk is not centre._instances["bob"].msk


def test_a_key_of_one_identity_is_refused_on_another_identitys_ciphertext(centre):
    d = IdentityDirectory(centre.directory())
    ct = d.encrypt("bob", [1, 2, 3])
    dk = centre.get_decryption_keys("alice", [1, 1, 1])
    with pytest.raises(FEKeyError, match="another identity"):
        d.decrypt(ct, dk, [1, 1, 1])
    # relabelling the ciphertext does not help: its entry is bob's
    forged = dict(ct, identity="alice")
    with pytest.raises(FEKeyError, match="another entry"):
        d.decrypt(forged, dk, [1, 1, 1])


def test_case_matters(centre):
    d = IdentityDirectory(centre.directory())
    with pytest.raises(FEKeyError):
        d.decrypt(d.encrypt("Alice", [1, 1, 1]), centre.get_decryption_keys("alice", [1, 1, 1]), [1, 1, 1])


# -- bindings ----------------------------------------------------------------------


def test_expected_digest(centre):
    record = centre.directory()
    IdentityDirectory(record, expected_digest=record["digest"])
    other = IdentityKeyCentre(dict(CFG))
    other.add_identity("alice")
    with pytest.raises(FEValidationError, match="expected"):
        IdentityDirectory(other.directory(), expected_digest=record["digest"])


@pytest.mark.parametrize("tamper", ["pk", "entry", "identity", "pp", "pp_digest", "digest", "order",
                                    "duplicate", "version", "scheme", "missing"])
def test_inconsistent_directories_are_refused(centre, tamper):
    record = copy.deepcopy(centre.directory())
    entries = record["entries"]
    if tamper == "pk":
        entries[1]["pk"] = entries[2]["pk"]
    elif tamper == "entry":
        entries[1]["entry"] = entries[2]["entry"]
    elif tamper == "identity":
        entries[1]["identity"] = "carol"
    elif tamper == "pp":
        record["pp"]["sigma3"] = record["pp"]["sigma3"] * 2
    elif tamper == "pp_digest":
        record["pp_digest"] = "00" * 32
    elif tamper == "digest":
        record["digest"] = "00" * 32
    elif tamper == "order":
        entries.reverse()
    elif tamper == "duplicate":
        entries.append(copy.deepcopy(entries[0]))
    elif tamper == "version":
        record["version"] = "fullysec-ring-lwe-identity-directory/v0"
    elif tamper == "scheme":
        record["scheme"] = "sife/ring_lwe"
    elif tamper == "missing":
        del record["pp_digest"]
    with pytest.raises(FEValidationError):
        IdentityDirectory(record)


def test_a_swapped_key_entry_with_recomputed_digests_changes_the_directory_digest(centre):
    """An attacker who replaces bob's pk and recomputes every digest gets a
    consistent directory, which expected_digest then refuses."""
    record = copy.deepcopy(centre.directory())
    attacker = IdentityKeyCentre(dict(CFG))
    attacker.add_identity("bob")
    e = record["entries"][2]
    assert e["identity"] == "bob"
    e["pk"] = attacker.directory()["entries"][0]["pk"]
    e["entry"] = idd.entry_digest(record["pp_digest"], "bob", e["pk"])
    record["digest"] = idd.directory_digest(record["pp_digest"], record["entries"])
    IdentityDirectory(record)
    with pytest.raises(FEValidationError):
        IdentityDirectory(record, expected_digest=centre.directory()["digest"])


def test_keys_and_ciphertexts_name_their_entry(centre):
    d = IdentityDirectory(centre.directory())
    ct = d.encrypt("alice", [1, 2, 3])
    dk = centre.get_decryption_keys("alice", [1, 0, 0])
    for bad_ct, bad_dk in ((dict(ct, entry="00" * 32), dk), (ct, dict(dk, entry="00" * 32)),
                           (dict(ct, entry=None), dk)):
        with pytest.raises(FEKeyError):
            d.decrypt(bad_ct, bad_dk, [1, 0, 0])
    for bad_ct, bad_dk in ((dict(ct, version="x"), dk), (ct, dict(dk, version="x")), (ct["ct"], dk)):
        with pytest.raises(FEValidationError):
            d.decrypt(bad_ct, bad_dk, [1, 0, 0])
    with pytest.raises(FEValidationError):
        d.decrypt({k: v for k, v in ct.items() if k != "ct"}, dk, [1, 0, 0])


def test_a_ciphertext_of_another_directory_is_refused(centre):
    other = IdentityKeyCentre(dict(CFG))
    other.add_identity("alice")
    ct = IdentityDirectory(other.directory()).encrypt("alice", [1, 2, 3])
    with pytest.raises(FEKeyError, match="another entry"):
        IdentityDirectory(centre.directory()).decrypt(ct, centre.get_decryption_keys("alice", [1, 1, 1]),
                                                      [1, 1, 1])


def test_unknown_identity(centre):
    d = IdentityDirectory(centre.directory())
    with pytest.raises(FEKeyError):
        d.encrypt("carol", [1, 2, 3])
    with pytest.raises(FEKeyError):
        centre.get_decryption_keys("carol", [1, 2, 3])


# -- growth ------------------------------------------------------------------------


def test_adding_identities_keeps_earlier_ciphertexts_and_keys(centre):
    before = centre.directory()
    d0 = IdentityDirectory(before)
    ct = d0.encrypt("alice", [4, 4, 4])
    dk = centre.get_decryption_keys("alice", [1, 1, 1])
    centre.add_identity("carol")
    after = centre.directory()
    assert after["digest"] != before["digest"]
    d1 = IdentityDirectory(after)
    assert d1.entry_digest("alice") == d0.entry_digest("alice")
    assert d1.decrypt(ct, dk, [1, 1, 1]) == 12


def test_a_new_key_centre_for_the_same_identity_is_a_new_entry(centre):
    again = IdentityKeyCentre(dict(CFG))
    again.add_identity("alice")
    d_old, d_new = IdentityDirectory(centre.directory()), IdentityDirectory(again.directory())
    assert d_old.entry_digest("alice") != d_new.entry_digest("alice")
    with pytest.raises(FEKeyError):
        d_new.decrypt(d_new.encrypt("alice", [1, 1, 1]), centre.get_decryption_keys("alice", [1, 1, 1]), [1, 1, 1])


# -- identities --------------------------------------------------------------------


@pytest.mark.parametrize("identity", ["", None, 5, b"alice", "a" * 257, "al\x00ice", "ali​ce",
                                      "é", "tab\there"])
def test_invalid_identities(identity):
    kc = IdentityKeyCentre(dict(CFG))
    with pytest.raises(FEValidationError):
        kc.add_identity(identity)


def test_valid_identities():
    kc = IdentityKeyCentre(dict(CFG))
    for identity in ("a" * 256, "é", "alice@example.org", "用户 1"):
        kc.add_identity(identity)
    assert IdentityDirectory(kc.directory()).identities() == sorted(kc.identities())


def test_an_identity_is_added_once(centre):
    with pytest.raises(FESchemeError):
        centre.add_identity("alice")


def test_weights_are_checked(centre):
    with pytest.raises(FEValidationError):
        centre.get_decryption_keys("alice", [5, 0, 0])
    with pytest.raises(FEValidationError):
        centre.get_decryption_keys("alice", [1, 0])


# -- records -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["IdentityDirectoryRecord", "IdentityEntry", "IdentityCiphertext",
                                  "IdentityLayers", "IdentityFunctionalKey"])
def test_records_document_their_fields(name):
    from pyfe4ai.core.formats import identity

    cls = getattr(identity, name)
    assert cls.FIELDS and len(set(cls.FIELDS)) == len(cls.FIELDS)
    assert "Fields, in wire order" in cls.__doc__
    assert all(cls._fields[f].doc for f in cls.FIELDS), name


def test_objects_are_their_records_in_wire_order(centre):
    from pyfe4ai.core.formats import identity

    record = centre.directory()
    d = IdentityDirectory(record)
    pairs = [(record, identity.IdentityDirectoryRecord), (record["entries"][0], identity.IdentityEntry),
             (d.encrypt("alice", [1, 2, 3]), identity.IdentityCiphertext),
             (centre.get_decryption_keys("alice", [1, 1, 1]), identity.IdentityFunctionalKey)]
    for obj, cls in pairs:
        assert list(obj) == list(cls.FIELDS)
        assert cls.from_dict(obj).to_dict() is obj


def test_digests_survive_a_json_round_trip(centre):
    record = centre.directory()
    again = _json(record)
    assert idd.pp_digest(again["pp"]) == record["pp_digest"]
    assert IdentityDirectory(again, expected_digest=record["digest"]).digest == record["digest"]


def test_malformed_fields_are_refused_with_the_library_errors(centre):
    """R-0.7 I-4: a lone surrogate in a digest field is a mismatch (FEKeyError), not a UnicodeEncodeError."""
    d = IdentityDirectory(centre.directory())
    ct = d.encrypt("alice", [1, 2, 3])
    dk = centre.get_decryption_keys("alice", [1, 0, 0])
    with pytest.raises(FEKeyError):
        d.decrypt(dict(ct, entry="\ud800"), dk, [1, 0, 0])
    with pytest.raises(FEValidationError):
        d.decrypt(dict(ct, identity="\ud800"), dict(dk, identity="\ud800"), [1, 0, 0])


def test_unassigned_code_points_are_not_added():
    kc = IdentityKeyCentre(dict(CFG))
    with pytest.raises(FEValidationError, match="unassigned"):
        kc.add_identity("x͸")
