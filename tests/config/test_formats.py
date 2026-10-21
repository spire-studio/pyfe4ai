"""The serialized-object records (:mod:`pyfe4ai.core.formats`): the
:class:`Record` view semantics, and for every scheme family that the
objects its schemes write are the records' wire dicts -- declared fields
only, in the declared order -- and come back unchanged through JSON."""

import importlib
import inspect
import json
import os
import random

import pytest

from pyfe4ai.core.formats.record import ABSENT
from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record

_MODULES = ["record", "lwe", "ring_lwe", "dl", "pairing", "threshold"]


def _records():
    seen = {}
    for name in _MODULES:
        module = importlib.import_module("pyfe4ai.core.formats." + name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Record) and cls is not Record and cls.__module__ == module.__name__:
                seen[cls.__name__] = cls
    return seen


RECORDS = _records()


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(str(tmp_path), "config"))


def _has_bytes(obj) -> bool:
    if isinstance(obj, (bytes, bytearray)):
        return True
    if isinstance(obj, dict):
        return any(_has_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_bytes(v) for v in obj)
    return False


def check(obj, cls):
    """*obj* is a wire dict of *cls*: the record is the dict itself, its keys
    are declared fields in the declared order, and JSON gives it back (binary
    objects travel through their codecs instead)."""
    assert isinstance(obj, dict), (cls.__name__, type(obj))
    assert cls.from_dict(obj).to_dict() is obj
    undeclared = [key for key in obj if key not in cls.FIELDS]
    assert not undeclared, "{} has undeclared fields {}".format(cls.__name__, undeclared)
    positions = [cls.FIELDS.index(key) for key in obj]
    assert positions == sorted(positions), "{} out of order: {}".format(cls.__name__, list(obj))
    if not _has_bytes(obj):
        back = json.loads(json.dumps(obj))
        assert back == _as_json(obj) and list(back) == list(obj), "{} changes through JSON".format(cls.__name__)


def _as_json(obj):
    """*obj* as JSON gives it back: tuples are lists (nothing else may change)."""
    if isinstance(obj, dict):
        return {k: _as_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_json(v) for v in obj]
    return obj


def record_file(kg):
    with open(os.path.join(kg.config_folder, kg._scheme_type, kg._param_file_name()), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Record


class _Point(Record):
    """A test record."""

    x = Field("int")
    y = Field("int")
    tag = Field("optional", absent=None)


class _Point3(_Point):
    """A test record whose field goes between its base's."""

    ORDER = ("x", "y", "z", "tag")
    z = Field("int")


def test_a_record_is_its_dict():
    d = {"x": 1, "y": 2, "extra": 3}
    p = _Point.from_dict(d)
    assert p.to_dict() is d and p.x == 1 and p.y == 2
    p.x = 5
    assert d["x"] == 5
    del p.y
    assert d == {"x": 5, "extra": 3}


def test_reads_are_the_dict_reads():
    p = _Point.from_dict({"x": 1})
    with pytest.raises(KeyError):
        p.y
    with pytest.raises(TypeError):
        _Point.from_dict(None).x
    with pytest.raises(TypeError):
        _Point.from_dict(["x"]).x
    with pytest.raises(AttributeError):
        _Point.from_dict([1]).get("x")
    assert p.get("y") is None and p.get("y", 7) == 7 and p.get("tag", "t") == "t"
    assert "x" in p and "y" not in p
    with pytest.raises(AttributeError, match="no field 'extra'"):
        p.get("extra")
    with pytest.raises(AttributeError):
        "extra" in p


def test_new_writes_the_declared_order():
    assert list(_Point.new(y=2, x=1).to_dict()) == ["x", "y"]
    assert _Point.new(x=1, y=2, tag=ABSENT).to_dict() == {"x": 1, "y": 2}
    assert list(_Point3.new(tag="t", z=3, y=2, x=1).to_dict()) == ["x", "y", "z", "tag"]
    assert _Point3.FIELDS == ("x", "y", "z", "tag")
    with pytest.raises(TypeError, match="no fields"):
        _Point.new(x=1, w=2)


def test_extend_update_and_pop_are_the_dict_operations():
    base = {"y": 2, "x": 1}
    p = _Point.extend(base, tag="t", x=9)
    assert p.to_dict() == dict(base, tag="t", x=9) and list(p.to_dict()) == ["y", "x", "tag"]
    assert base == {"y": 2, "x": 1}
    p.update(y=3, tag=ABSENT)
    assert p.to_dict() == {"y": 3, "x": 9, "tag": "t"}
    with pytest.raises(TypeError):
        p.update(w=1)
    assert p.pop("tag") == "t" and p.pop("tag", 0) == 0
    with pytest.raises(AttributeError):
        p.pop("w")


def test_equality_and_repr():
    assert _Point.from_dict({"x": 1}) == _Point.from_dict({"x": 1})
    assert _Point.from_dict({"x": 1}) != _Point3.from_dict({"x": 1})
    assert repr(_Point.from_dict({"x": 1, "y": 2})) == "_Point('x', 'y')"
    with pytest.raises(TypeError):
        hash(_Point.from_dict({}))


def test_the_check_refuses_what_json_changes():
    check({"x": 1, "y": "2"}, _Point)
    with pytest.raises(AssertionError, match="changes through JSON"):
        check({"x": {1: 2}, "y": "2"}, _Point)  # int keys come back as strings
    with pytest.raises(AssertionError, match="out of order"):
        check({"y": 1, "x": 2}, _Point)
    with pytest.raises(AssertionError, match="undeclared"):
        check({"x": 1, "w": 2}, _Point)


def test_order_must_name_every_field():
    with pytest.raises(TypeError, match="ORDER"):
        type("Bad", (_Point,), {"ORDER": ("x", "y"), "z": Field()})


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_every_record_documents_its_fields(name):
    cls = RECORDS[name]
    assert cls.FIELDS and len(set(cls.FIELDS)) == len(cls.FIELDS)
    assert "Fields, in wire order" in cls.__doc__
    assert all(cls._fields[f].doc for f in cls.FIELDS), name


# ---------------------------------------------------------------------------
# the objects of each family

from pyfe4ai.core.formats import dl, lwe, pairing, ring_lwe, threshold  # noqa: E402
from pyfe4ai.primitives.pairing.pairing_backend import pairing_available  # noqa: E402
from tests.pairing_group import GROUP, GROUP_MARKS, needs_group  # noqa: E402


def _sife_ddh():
    from pyfe4ai.schemes.sife.ddh import SIFEDDH, SIFEDDHKeyGenerator

    kg = SIFEDDHKeyGenerator({"sec_param": 128, "eta": 3})
    kg.setup()
    pp, sk = kg.get_public_parameters(), kg.get_private_keys()
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1, 2, 3]})
    ct = SIFEDDH({"keys": {"pp": pp, "sk": sk}}).encrypt([1, 2, 3])
    assert SIFEDDH({"keys": {"pp": pp}}).decrypt(json.loads(json.dumps(ct)), dk, [1, 2, 3]) == 14
    return [(record_file(kg), dl.DDHRecord), (record_file(kg)["group"], dl.DLGroupRecord),
            (pp, dl.SIFEDDHParameters), (sk, dl.SIFEDDHEncryptionKey), (dk, dl.SIFEDDHFunctionalKey),
            (ct, dl.SIFEDDHCiphertext)]


def _mife_lwe():
    from pyfe4ai.schemes.mife.lwe import MIFELWE, MIFELWEKeyGenerator

    nids = ["nid_0", "nid_1"]
    kg = MIFELWEKeyGenerator({"sec_param": 64, "eta": 2, "n": 2, "lst_nid": nids, "lwe_n": 16,
                              "bound_x": 6, "bound_y": 6})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1, 2] for nid in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    sks = {nid: kg.get_private_keys(nid) for nid in nids}
    cts = {nid: MIFELWE({"id": nid, "keys": {"pp": pp, "sk": sks[nid]}}).encrypt([1, 1]) for nid in nids}
    assert MIFELWE({"keys": {"pp": pp}}).decrypt(json.loads(json.dumps(cts)), dk, y) == 6
    return [(record_file(kg), lwe.ABDP15Record), (pp, lwe.MIFELWEParameters), (dk, lwe.MIFELWEFunctionalKey)] + [
        (sks[nid], lwe.MIFELWEEncryptionKey) for nid in nids] + [(cts[nid], lwe.LWECiphertext) for nid in nids]


def _mcfe_paillier():
    from pyfe4ai.schemes.mcfe.paillier import MCFEPaillier, MCFEPaillierKeyGenerator

    nids = ["nid_0", "nid_1"]
    kg = MCFEPaillierKeyGenerator({"sec_param": 64, "eta": 1, "n": 2, "lst_nid": nids, "bit_length": 64,
                                   "bound_x": 10, "bound_y": 10})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [3] for nid in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    sk = kg.get_private_keys("nid_0")
    cts = {nid: MCFEPaillier({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([1], "l")
           for nid in nids}
    crypto = MCFEPaillier({"keys": {"pp": pp}})
    assert crypto.decrypt(json.loads(json.dumps(cts)), dk, y, "l") == 6
    prepared = crypto.prepare_key(dk, "l", [(2,)])
    assert list(prepared) == list(dl.MCFEPaillierFunctionalKey.FIELDS)  # key_terms: in process only, not JSON
    return [(record_file(kg), dl.MCFEPaillierRecord), (pp, dl.MCFEPaillierParameters), (sk, dl.MCFEDDHEncryptionKey),
            (dk, dl.MCFEPaillierFunctionalKey), (cts["nid_0"], dl.MCFEDDHCiphertext)]


def _tmcfe_ddh():
    from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDH, ThresholdMCFEDDHKeyGenerator

    nids, sids = ["nid_0", "nid_1"], ["sid_0", "sid_1"]
    kg = ThresholdMCFEDDHKeyGenerator({"sec_param": 128, "lst_nid": nids, "lst_sid": sids, "eta": 1, "n": 2,
                                    "s": 2, "t": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in nids}
    cts = {nid: ThresholdMCFEDDH({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([1], "l")
           for nid in nids}
    dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in sids}
    partials = {sid: ThresholdMCFEDDH({"id": sid, "keys": {"pp": pp}}).share_decrypt(cts, y, dks[sid], sids, "l")
                for sid in sids}
    assert ThresholdMCFEDDH({"id": "c", "keys": {"pp": pp}}).combine_decrypt(json.loads(json.dumps(partials))) == 2
    return [(record_file(kg), dl.DDHRecord), (pp, dl.TMCFEDDHParameters),
            (kg.get_private_keys("nid_0"), dl.MCFEDDHEncryptionKey), (dks["sid_0"], threshold.TMCFEDDHKeyShare),
            (cts["nid_0"], dl.MCFEDDHCiphertext), (partials["sid_0"], threshold.TMCFEDDHPartial)]


def _dmcfe_ddh():
    from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFEDDH, DecentralizedMCFEDDHKeyGenerator

    nids = ["nid_0", "nid_1"]
    kg = DecentralizedMCFEDDHKeyGenerator({"sec_param": 128, "eta": 1, "n": 2, "lst_nid": nids})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in nids}
    clients = {nid: DecentralizedMCFEDDH({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    shares = {nid: c.derive_function_decryption_key_share(y) for nid, c in clients.items()}
    combiner = DecentralizedMCFEDDH({"id": "c", "keys": {"pp": pp}})
    dk = combiner.combine_function_decryption_key_share(json.loads(json.dumps(shares)))
    cts = {nid: c.encrypt([2], "l") for nid, c in clients.items()}
    assert combiner.decrypt(cts, dk, y, "l") == 4
    return [(record_file(kg), dl.DDHRecord), (pp, dl.MCFEDDHParameters),
            (kg.get_private_keys("nid_0"), dl.MCFEDDHEncryptionKey), (shares["nid_0"], dl.MCFE2KeyShare),
            (dk, dl.MCFEDDHFunctionalKey), (cts["nid_0"], dl.MCFEDDHCiphertext)]


def _mcfe_ring_lwe(**options):
    from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator

    nids = ["nid_0", "nid_1"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": nids, "ring_n": 16, "bound_x": 4, "bound_y": 4}
    kg = MCFERingLWEKeyGenerator(dict(cfg, **options))
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "label": "l"})
    cts = {nid: MCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([[1, 2]], "l")
           for nid in nids}
    out = [(record_file(kg), ring_lwe.RingLWERecord), (pp, ring_lwe.MCFERingLWEParameters),
           (kg.get_private_keys("nid_0"), ring_lwe.MCFERingLWEEncryptionKey), (dk, ring_lwe.MCFERingLWEFunctionalKey)]
    if isinstance(cts["nid_0"], dict):
        cts = json.loads(json.dumps(cts))
        out.append((cts["nid_0"], ring_lwe.RingLWECiphertext))
    assert MCFERingLWE({"keys": {"pp": pp}}).decrypt(cts, dk, y, "l") == [2, 4]
    return out


def _tmcfe_ring_lwe(**options):
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator

    nids, sids = ["nid_0", "nid_1"], ["sid_0", "sid_1", "sid_2"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": nids, "lst_sid": sids, "ring_n": 16,
           "bound_x": 4, "bound_y": 4}
    kg = ThresholdMCFERingLWEKeyGenerator(dict(cfg, **options))
    kg.setup()
    pp = kg.get_public_parameters()
    creds = {"fusion_weight": {nid: [1] for nid in nids}, "label": "l"}
    cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(
        [[1, 2]], "l") for nid in nids}
    out = [(record_file(kg), ring_lwe.ThresholdRingLWERecord), (pp, ring_lwe.ThresholdRingLWEParameters),
           (kg.get_private_keys("nid_0"), ring_lwe.ThresholdRingLWEEncryptionKey)]
    enrolled, partials = ["sid_0", "sid_1"], {}
    for sid in enrolled:
        server = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp}})
        dk = kg.get_decryption_keys(sid, credentials=creds)
        if not server.has_weight_key_share(dk):
            weight = kg.get_weight_key_share(sid, credentials=creds)
            out.append((weight, threshold.TMCFERingLWEWeightShare))
            server.add_weight_key_share(weight)
            out.append((dk, threshold.TMCFERingLWELabelShare))
        else:
            out.append((dk, threshold.TMCFERingLWEKeyShare))
        partials[sid] = server.share_decrypt(cts, creds, dk, enrolled)
        if isinstance(partials[sid], dict):
            out.append((partials[sid], threshold.TMCFERingLWEPartial))
        for piece in threshold.ThresholdKeyShare(dk).get("pieces") or []:
            out.append((piece, threshold.KeyPiece))
    if isinstance(cts["nid_0"], dict):
        out.append((cts["nid_0"], ring_lwe.RingLWECiphertext))
    assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(partials) == [2, 4]
    return out


def _tmcfe_fh():
    if not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP))
    from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import (
        ThresholdMCFEFHMultiIPEPairing, ThresholdMCFEFHMultiIPEPairingKeyGenerator)

    nids, sids = ["nid_0", "nid_1"], ["sid_0", "sid_1"]
    kg = ThresholdMCFEFHMultiIPEPairingKeyGenerator({"sec_param": 64, "eta": 1, "n": 2, "lst_nid": nids, "lst_sid": sids,
                                              "s": 2, "t": 2, "bound_x": 4, "bound_y": 4,
                                              "pairing_group_param": GROUP})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in nids}
    sk = kg.get_private_keys("nid_0")
    cts = {nid: ThresholdMCFEFHMultiIPEPairing({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(
        [2], "l") for nid in nids}
    dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in sids}
    partials = {sid: ThresholdMCFEFHMultiIPEPairing({"id": sid, "keys": {"pp": pp}}).share_decrypt(cts, y, dks[sid], sids,
                                                                                               "l")
                for sid in sids}
    assert ThresholdMCFEFHMultiIPEPairing({"id": "c", "keys": {"pp": pp}}).combine_decrypt(partials, "l") == 4
    return [(pp, pairing.TMCFEFHParameters), (sk, pairing.FHClientKey), (sk["imsk"], pairing.FHMasterKey),
            (dks["sid_0"], threshold.TMCFEFHKeyShare), (cts["nid_0"], pairing.FHCiphertext),
            (partials["sid_0"], threshold.TMCFEFHPartial)]


def _dmcfe_fh():
    if not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP))
    from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import (
        DecentralizedMCFEFHMultiIPEPairing, DecentralizedMCFEFHMultiIPEPairingKeyGenerator)

    nids = ["nid_0", "nid_1"]
    kg = DecentralizedMCFEFHMultiIPEPairingKeyGenerator({"sec_param": 64, "eta": 1, "n": 2, "lst_nid": nids,
                                                  "bound_x": 4, "bound_y": 4, "pairing_group_param": GROUP})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: DecentralizedMCFEFHMultiIPEPairing({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    shares = {nid: c.derive_function_decryption_key_share({nid: [1] for nid in nids}, "f") for nid, c in
              clients.items()}
    dk = clients["nid_0"].combine_function_decryption_key_share(shares)
    cts = {nid: c.encrypt([2], "l") for nid, c in clients.items()}
    assert clients["nid_0"].decrypt(cts, dk) == 4
    sk = kg.get_private_keys("nid_0")
    return [(pp, pairing.FHParameters), (sk, pairing.FHClientKey), (sk["imsk"], pairing.FHMasterKey),
            (shares["nid_0"], pairing.FHKeyEntry), (dk, pairing.MCFEFHFunctionalKey),
            (dk["dk"]["nid_0"], pairing.FHKeyEntry), (cts["nid_0"], pairing.FHCiphertext)]


def _quad():
    if not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP))
    from pyfe4ai.schemes.quadratic.quad import QuadraticQuad, QuadraticQuadKeyGenerator

    kg = QuadraticQuadKeyGenerator({"sec_param": 64, "n": 2, "m": 2, "bound": 3, "pairing_group_param": GROUP})
    kg.setup()
    pp = kg.get_public_parameters()
    dk = kg.get_decryption_keys("s", credentials={"function_matrix": [[1, 0], [0, 1]]})
    ct = QuadraticQuad({"keys": {"pp": pp}}).encrypt({"x": [1, 2], "y": [2, 1]})
    assert QuadraticQuad({"keys": {"pp": json.loads(json.dumps(pp))}}).decrypt(json.loads(json.dumps(ct)), dk) == 4
    return [(pp, pairing.QuadParameters), (pp["ipfe"], pairing.QuadIPFEPublicKey), (dk, pairing.QuadFunctionalKey),
            (ct, pairing.QuadCiphertext)]


FAMILIES = {
    "sife-ddh": _sife_ddh,
    "mife-lwe": _mife_lwe,
    "mcfe-paillier": _mcfe_paillier,
    "tmcfe-ddh": _tmcfe_ddh,
    "dmcfe-ddh": _dmcfe_ddh,
    "mcfe-ring-lwe-json": lambda: _mcfe_ring_lwe(ct_format="json"),
    "mcfe-ring-lwe-binary": lambda: _mcfe_ring_lwe(ct_format="binary"),
    "tmcfe-ring-lwe-defaults": _tmcfe_ring_lwe,
    "tmcfe-ring-lwe-legacy": lambda: _tmcfe_ring_lwe(ct_format="json", ring_modulus="prime", key_shares="full"),
    "tmcfe-fh": _tmcfe_fh,
    "dmcfe-fh": _dmcfe_fh,
    "quad": _quad,
}


#: The families on the pairing group of the tests (native with BLS12-381: see tests/pairing_group.py).
PAIRING_FAMILIES = {"tmcfe-fh", "dmcfe-fh", "quad"}


@pytest.mark.parametrize("family", [pytest.param(f, marks=GROUP_MARKS if f in PAIRING_FAMILIES else ())
                                    for f in sorted(FAMILIES)])
def test_scheme_objects_are_their_records(family):
    random.seed(0)
    objects = FAMILIES[family]()
    assert objects
    for obj, cls in objects:
        check(obj, cls)


def test_standard_profile_objects_are_their_records():
    from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator

    nids = ["nid_0", "nid_1"]
    kg = MCFERingLWEKeyGenerator({"eta": 1, "n": 2, "lst_nid": nids, "bound_x": 4, "bound_y": 4,
                                  "security_profile": "standard"})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "label": "l"})
    cts = {nid: MCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([[1]], "l")
           for nid in nids}
    assert MCFERingLWE({"keys": {"pp": pp}}).decrypt(cts, dk, y, "l") == [2]
    for obj, cls in [(record_file(kg), ring_lwe.RingLWERecord), (pp, ring_lwe.MCFERingLWEParameters),
                     (dk, ring_lwe.MCFERingLWEFunctionalKey)]:
        check(obj, cls)


def test_threshold_binary_ciphertexts_round_trip():
    from pyfe4ai.core.formats import binary

    q, n, k = 2 ** 77 + 0x4001, 16, 3
    rng = random.Random(2)
    ct0 = [[rng.randrange(q) for _ in range(k)] for _ in range(2)]
    ct1 = [rng.randrange(q) for _ in range(n)]
    data = binary.encode_threshold_ciphertext(ct0, ct1, k, q, n)
    assert binary.threshold_ciphertext_header(data, q, 2, n) == k
    assert binary.decode_threshold_ciphertext(data, q, 2, n) == (k, ct0, ct1)
    assert binary.decode_values(binary.encode_values(ct1, q), q, n) == ct1


# ---------------------------------------------------------------------------
# malformed input: the exceptions of earlier versions, in the same order


@needs_group
def test_dmcfe_fh_combines_malformed_share_sets_as_before():
    if not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP))
    from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import (
        DecentralizedMCFEFHMultiIPEPairing, DecentralizedMCFEFHMultiIPEPairingKeyGenerator)

    kg = DecentralizedMCFEFHMultiIPEPairingKeyGenerator({"eta": 1, "n": 2, "pairing_group_param": GROUP})
    kg.setup()
    pp = kg.get_public_parameters()
    with pytest.raises(IndexError):  # not a StopIteration leaking out
        DecentralizedMCFEFHMultiIPEPairing({"id": "c", "keys": {"pp": dict(pp, lst_nid=[])}}).combine_function_decryption_key_share({})
    with pytest.raises(AttributeError, match="'values'"):
        DecentralizedMCFEFHMultiIPEPairing({"id": "c", "keys": {"pp": pp}}).combine_function_decryption_key_share(
            list(pp["lst_nid"]))


def test_mife_ddh_encryption_reads_the_parameters_before_the_key():
    from pyfe4ai.schemes.mife.ddh import MIFEDDH, MIFEDDHKeyGenerator

    kg = MIFEDDHKeyGenerator({"sec_param": 128, "eta": 1, "n": 2})
    kg.setup()
    pp = {k: v for k, v in kg.get_public_parameters().items() if k != "g"}
    sk = {k: v for k, v in kg.get_private_keys("nid_0").items() if k != "h"}
    with pytest.raises(KeyError, match="'g'"):
        MIFEDDH({"id": "nid_0", "keys": {"pp": pp, "sk": sk}}).encrypt([1])
