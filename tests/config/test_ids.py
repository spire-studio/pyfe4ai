"""Server and client ids (:mod:`pyfe4ai.core.ids`).

An id is a ``str`` or an ``int`` (not a ``bool``). A key generator refuses a
``lst_sid`` or ``lst_nid`` whose ids cannot key a dict (an unhashable entry,
or a value that is not a sized collection) with
:class:`~pyfe4ai.core.exceptions.FEValidationError` and a constant message,
where it raised a bare ``TypeError`` (``unhashable type``); messages that list
ids order ids of both types by ``repr``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.ids import DUPLICATE_IDS, ID_LISTS, MIXED_IDS, id_list, is_id

needs_bls = pytest.mark.skipif(not native_backend.available(), reason="BLS12-381 needs the native extension")

THRESHOLD = pyfe4ai.list_schemes(capabilities="threshold")
#: every scheme whose key generator takes ``lst_nid``
WITH_CLIENTS = [s for s in pyfe4ai.list_schemes() if s.split("/")[0] in ("mife", "mcfe")] + ["quadratic/miqfe"]


def _marks(scheme):
    return pytest.param(scheme, marks=needs_bls) if "pairing" in scheme else scheme


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _config(scheme, **ids):
    group = any(token in scheme for token in ("ddh", "damgard", "paillier"))
    config = {"eta": 1, "n": 2, "lst_nid": ["c0", "c1"], "s": 2, "t": 2, "lst_sid": ["s0", "s1"], "ring_n": 16,
              "lwe_n": 16, "sec_param": 128 if group else 64, "bound_x": 4, "bound_y": 4, "bound_u": 2,
              "label_modulus": 8}
    config.update(ids)
    return config


def test_ids_are_str_or_int():
    assert all(is_id(v) for v in ("s0", "", 0, 1, -3, 2 ** 70))
    assert not any(is_id(v) for v in (True, False, 1.0, None, b"s0", ("s", 0), ["s0"], {"s": 0}))


def test_messages_order_ids_of_both_types():
    assert id_list({"b", "a"}) == ["a", "b"] and id_list([3, 1]) == [1, 3]
    assert id_list(["s0", 1]) == ["s0", 1]  # by repr: "'s0'" < "1"
    assert id_list([None, 1]) == [1, None]


@pytest.mark.parametrize("ids", [[["s", 0], ["s", 1]], [{"s": 0}, {"s": 1}], 5, iter(["s0", "s1"])],
                         ids=["lists", "dicts", "an int", "an iterator"])
@pytest.mark.parametrize("scheme", [_marks(s) for s in THRESHOLD])
def test_server_ids_that_cannot_key_a_dict_are_refused_at_setup(scheme, ids):
    with pytest.raises(FEValidationError) as info:
        kg, _ = pyfe4ai.create(scheme, _config(scheme, lst_sid=ids))
        kg.setup()
    assert str(info.value) == ID_LISTS["lst_sid"]


@pytest.mark.parametrize("ids", [[["c", 0], ["c", 1]], [{"c": 0}, {"c": 1}], iter(["c0", "c1"])],
                         ids=["lists", "dicts", "an iterator"])
@pytest.mark.parametrize("scheme", [_marks(s) for s in WITH_CLIENTS])
def test_client_ids_that_cannot_key_a_dict_are_refused_at_setup(scheme, ids):
    with pytest.raises(FEValidationError) as info:
        kg, _ = pyfe4ai.create(scheme, _config(scheme, lst_nid=ids))
        kg.setup()
    assert str(info.value) == ID_LISTS["lst_nid"]


@pytest.mark.parametrize("scheme", [_marks(s) for s in THRESHOLD])
def test_repeated_ids_keep_their_message(scheme):
    with pytest.raises(FEValidationError) as info:
        pyfe4ai.create(scheme, _config(scheme, lst_sid=["s0", "s0"]))[0].setup()
    assert str(info.value) == DUPLICATE_IDS["lst_sid"]
    with pytest.raises(FEValidationError) as info:
        pyfe4ai.create(scheme, _config(scheme, lst_nid=[1, 1]))[0].setup()
    assert str(info.value) == DUPLICATE_IDS["lst_nid"]


def _mife_round(scheme, nids):
    kg, crypto = pyfe4ai.create(scheme, _config(scheme, n=3, lst_nid=nids, bound_x=4, bound_y=4))
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {nid: crypto({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([i + 1])
           for i, nid in enumerate(nids)}
    y = {nid: [1] for nid in nids}
    return crypto({"id": "agg", "keys": {"pp": pp}}).decrypt(cts, kg.get_decryption_keys("agg", credentials={
        "fusion_weight": y}), y)


def _threshold_ddh_round(scheme, nids):
    kg, crypto = pyfe4ai.create(scheme, _config(scheme, n=3, lst_nid=nids, s=2, t=2, lst_sid=["s0", "s1"]))
    kg.setup()
    pp = kg.get_public_parameters()
    mife = scheme.startswith("mife")
    clients = {nid: crypto({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in nids}
    cts = {nid: c.encrypt([i + 1]) if mife else c.encrypt([i + 1], "L") for i, (nid, c) in enumerate(clients.items())}
    y = {nid: [1] for nid in nids}
    parts = {}
    for sid in ("s0", "s1"):
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y} if mife else {"fusion_weight": y, "label": "L"})
        server = crypto({"id": sid, "keys": {"pp": pp, "dk": dk}})
        parts[sid] = server.share_decrypt(cts, y, dk, ["s0", "s1"]) if mife else server.share_decrypt(
            cts, y, dk, ["s0", "s1"], "L")
    return crypto({"id": "agg", "keys": {"pp": pp}}).combine_decrypt(parts)


def _dmcfe_ddh_round(scheme, nids):
    kg, crypto = pyfe4ai.create(scheme, _config(scheme, n=3, lst_nid=nids))
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: crypto({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in nids}
    cts = {nid: c.encrypt([i + 1], "L") for i, (nid, c) in enumerate(clients.items())}
    y = {nid: [1] for nid in nids}
    agg = crypto({"id": "agg", "keys": {"pp": pp}})
    dk = agg.combine_function_decryption_key_share({nid: c.derive_function_decryption_key_share(y)
                                                    for nid, c in clients.items()})
    return agg.decrypt(cts, dk, y, "L")


def _ml_round(scheme, nids):
    from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, aggregate_gradients, encrypt_gradient

    family, variant = scheme.split("/")
    wrapper = FESchemeWrapper(family, variant, _config(scheme, n=3, lst_nid=nids, bound_x=50, bound_y=2), precision=1)
    sent = {nid: json.loads(json.dumps(encrypt_gradient(np.array([0.5 * (i + 1)]), wrapper, nid=nid, label="r")
                                       .to_dict())) for i, nid in enumerate(nids)}
    received = {nid: EncryptedGradient.from_dict(d) for nid, d in sent.items()}
    return float(aggregate_gradients(received, wrapper, {nid: 2 for nid in nids}, label="r")[0])


ROUNDS = [(s, _mife_round, 6) for s in ("mife/ddh", "mife/ddh_ristretto255")]
ROUNDS += [pytest.param(s, _mife_round, 6, marks=needs_bls) for s in ("mife/fh_ipe_pairing", "mife/fh_multi_ipe_pairing")]
ROUNDS += [(s, _threshold_ddh_round, 6) for s in ("mife/ddh_threshold", "mcfe/ddh_threshold")]
ROUNDS += [("mcfe/ddh_decentralized", _dmcfe_ddh_round, 6)]
ROUNDS += [(s, _ml_round, 6.0) for s in ("mcfe/ddh", "mcfe/ddh_ristretto255")]


@pytest.mark.parametrize("nids", [[0, 1, 2], [1, 2, 3]], ids=["from 0", "from 1"])
@pytest.mark.parametrize("scheme,run,expected", ROUNDS)
def test_integer_client_ids_in_the_ddh_and_pairing_schemes(scheme, run, expected, nids):
    # these schemes took client ids to be strings ("fusion weight of " + nid, isinstance(nid, str) when
    # issuing an encryption key): integer client ids failed with TypeError or got no key
    assert run(scheme, nids) == expected


#: lists of values that are not ids, which the key generators accepted (and which worked in memory in some
#: schemes, but not in partial decryptions, the wire records or JSON)
OTHER_IDS = {"bytes": lambda p: [(p + "0").encode(), (p + "1").encode()], "floats": lambda p: [1.0, 2.0],
             "bools": lambda p: [True, False], "None": lambda p: [None, 1], "tuples": lambda p: [(p, 0), (p, 1)],
             "numpy ints": lambda p: list(np.arange(2)), "a string": lambda p: "ab",
             "a set": lambda p: {p + "0", p + "1"}, "a dict": lambda p: {p + "0": 0, p + "1": 1}}
#: lists that mix str and int ids (JSON names 1 and "1" alike; sorting them fails)
MIXED = {"str and int": lambda p: [p + "0", 1], "1 and '1'": lambda p: [1, "1"]}


@pytest.mark.parametrize("ids", sorted(OTHER_IDS) + sorted(MIXED))
@pytest.mark.parametrize("scheme", [_marks(s) for s in THRESHOLD])
def test_server_ids_of_other_types_are_refused_at_setup(scheme, ids):
    with pytest.raises(FEValidationError) as info:
        pyfe4ai.create(scheme, _config(scheme, lst_sid=(OTHER_IDS.get(ids) or MIXED[ids])("s")))[0].setup()
    assert str(info.value) == (MIXED_IDS if ids in MIXED else ID_LISTS)["lst_sid"]


@pytest.mark.parametrize("ids", sorted(OTHER_IDS) + sorted(MIXED))
@pytest.mark.parametrize("scheme", [_marks(s) for s in WITH_CLIENTS])
def test_client_ids_of_other_types_are_refused_at_setup(scheme, ids):
    with pytest.raises(FEValidationError) as info:
        pyfe4ai.create(scheme, _config(scheme, lst_nid=(OTHER_IDS.get(ids) or MIXED[ids])("c")))[0].setup()
    assert str(info.value) == (MIXED_IDS if ids in MIXED else ID_LISTS)["lst_nid"]


@pytest.mark.parametrize("ids", ["floats", "None", "str and int"])
def test_the_ml_adapter_refuses_them_when_it_is_built(ids):
    from pyfe4ai.ml import FESchemeWrapper

    with pytest.raises(FEValidationError):
        FESchemeWrapper("mcfe", "ring_lwe", _config("mcfe/ring_lwe", lst_nid=(OTHER_IDS.get(ids) or MIXED[ids])("c")))
    with pytest.raises(FEValidationError):
        FESchemeWrapper("mcfe", "ring_lwe_threshold", _config("mcfe/ring_lwe_threshold",
                                                              lst_sid=(OTHER_IDS.get(ids) or MIXED[ids])("s")))
