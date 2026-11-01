"""The multi-client key-policy attribute-based IPFE (NPP22 Sec. 5.2 Fig. 3): what it computes, setup, key generation,
the records and the cost of a decryption.

This file is what the scheme does when it is used as it is meant to be; test_kp_ab_pairing_hardening.py is what it
refuses. A key whose policy the attributes of the ciphertexts do not satisfy gives a ``FEKeyError``; malformed input,
ciphertexts that are not those of every client under one tag and one set of attributes, or objects of another setup a
``FEValidationError``; a key and ciphertexts that are well formed but were not made for each other a ``ValueError`` from
the discrete log, which is not a ``FEValidationError``.
"""

import json
import logging
import os
import threading
import warnings

import gmpy2 as gp
import numpy as np
import pytest

import pyfe4ai
from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.mcfe.conftest import ATTRS, EXPECTED, LST, POLICY, X, Y
from tests.mcfe.conftest import creds, encrypt_all, refuse_to_compute, through_json
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError, FEWarning
from pyfe4ai.core.formats.pairing import MCKPABCiphertext, MCKPABClientKey, MCKPABFunctionalKey
from pyfe4ai.core.formats.pairing import MCKPABParameters, MCKPABRecord
from pyfe4ai.policy import ATTR_HASH, POLICY_COMPILER, attribute_set_digest
from pyfe4ai.primitives.pairing import pairing_backend
from pyfe4ai.primitives.pairing.dpvs import vector_pair
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup, deserialize_many, hash_to_g1, pair
from pyfe4ai.schemes.common import ac_codec
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.kp_ab import TAG_DST, TAG_HASH, attribute_vector, tag_points
from pyfe4ai.schemes.mcfe import kp_ab_pairing
from pyfe4ai.schemes.mcfe.kp_ab_pairing import MCFEKPABPairing, MCFEKPABPairingKeyGenerator
from pyfe4ai.schemes.mcfe.labels import ElementLabel, encode_label

PP_DOMAIN = b"pyfe4ai/mcfe/kp-ab/pp-binding/v1|"
SCHEME_ID = "mcfe/kp_ab_pairing"

pytestmark = pytest.mark.usefixtures("own_parameter_cache")


# ---------------------------------------------------------------------------
# what the scheme computes


POLICIES = [
    ("and(purpose:research, region:EU)", ["purpose:research", "region:EU"]),
    ("or(region:EU, region:US)", ["region:US"]),
    ("2of(purpose:research, region:EU, org:A)", ["org:A", "region:EU"]),            # weights other than 1
    ("or(and(region:EU, org:A), and(region:EU, org:B))", ["org:B", "region:EU"]),   # repeated attribute
]


@pytest.mark.parametrize("policy,attributes", POLICIES)
def test_aggregates(policy, attributes, make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=attributes)
    dk = kg.get_decryption_keys("agg", credentials=creds(policy))
    assert agg.decrypt(cts, dk) == EXPECTED


def test_every_client_can_aggregate(mc_fixed):
    _, clients, _, cts, dk = mc_fixed
    assert all(crypto.decrypt(cts, dk) == EXPECTED for crypto in clients.values())


def test_attribute_sets_in_any_form(make_mc):           # Review Focus 1
    kg, clients, agg = make_mc()
    spellings = {"nid_0": {"region:EU", "purpose:research"},
                 "nid_1": ["region:EU", "purpose:research", "region:EU"],
                 "nid_2": ["purpose:research", "region:EU"]}
    cts = {nid: clients[nid].encrypt([X[nid]], "round-1", attributes=spellings[nid]) for nid in LST}
    assert {ct["attributes_digest"] for ct in cts.values()} == {attribute_set_digest(ATTRS)}
    assert all(ct["attributes"] == ATTRS for ct in cts.values())
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds())) == EXPECTED


def test_a_zero_weight_client_takes_part(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients)
    y = dict(Y, nid_1=0)
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds(y=y))) == X["nid_0"] * 2 + X["nid_2"] * -4
    zero = {nid: 0 for nid in LST}
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds(y=zero))) == 0


def test_one_key_serves_every_tag(make_mc):
    kg, clients, agg = make_mc()
    dk = kg.get_decryption_keys("agg", credentials=creds())
    for label, x in (("round-1", X), ("round-2", {"nid_0": -1, "nid_1": 2, "nid_2": 3})):
        cts = encrypt_all(clients, x=x, label=label)
        assert agg.decrypt(cts, dk) == sum(x[nid] * Y[nid] for nid in LST)
        assert agg.decrypt(cts, dk, label=label) == sum(x[nid] * Y[nid] for nid in LST)


@pytest.mark.parametrize("policy,attributes", [("and(purpose:research, region:EU)", ["region:EU"]),
                                               ("2of(a:1, b:1, c:1)", ["c:1"]),
                                               ("a:1", [])])
def test_unsatisfied_is_refused_before_any_pairing(policy, attributes, make_mc, monkeypatch):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=attributes)
    dk = kg.get_decryption_keys("agg", credentials=creds(policy))
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match="do not satisfy"):
        agg.decrypt(cts, dk)


def test_extreme_inner_products(make_mc):
    kg, clients, agg = make_mc(bound_x=10, bound_y=10)
    for sign in (1, -1):
        x = {nid: sign * 10 for nid in LST}
        dk = kg.get_decryption_keys("agg", credentials=creds(y={nid: 10 for nid in LST}))
        assert agg.decrypt(encrypt_all(clients, x=x), dk) == sign * 300


def test_other_numbers_of_clients(make_mc):
    for n in (1, 2, 5):
        kg, clients, agg = make_mc(n=n)
        x = {nid: i + 1 for i, nid in enumerate(kg.lst_nid)}
        y = {nid: 3 - i for i, nid in enumerate(kg.lst_nid)}
        cts = encrypt_all(clients, x=x)
        dk = kg.get_decryption_keys("agg", credentials=creds(y=y))
        assert agg.decrypt(cts, dk) == sum(x[nid] * y[nid] for nid in x), n


def test_one_root_for_all_clients(make_mc):             # decision 1 of this plan
    kg, _, _ = make_mc()
    group = PairingGroup(GROUP)
    order = int(group.order())
    ek = kg.get_private_keys("nid_0")
    f = [deserialize_many(group, row) for row in ek["f"]]
    dk = kg.get_decryption_keys("agg", credentials=creds("region:EU"))
    c = attribute_vector(f, "region:EU", 123456789, 1, order, group)
    paired = [vector_pair(c, deserialize_many(group, dk["k_rows"][nid][0])) for nid in LST]
    # [a_j z]_T for the one a_j and the one z of the key, whoever the client is
    assert paired[0] == paired[1] == paired[2]
    assert paired[0] != vector_pair(c, deserialize_many(group, dk["k_rows"]["nid_0"][0])) ** 0
    # the vectors differ: every client and every row has its own pi
    assert len({tuple(dk["k_rows"][nid][0]) for nid in LST}) == 3


def test_a_row_takes_a_fresh_pi_for_each_client_and_each_occurrence(make_mc):
    kg, _, _ = make_mc()
    dk = kg.get_decryption_keys("agg", credentials=creds("or(and(region:EU, org:A), and(region:EU, org:B))"))
    rows = [tuple(row) for nid in LST for row in dk["k_rows"][nid]]
    assert len(rows) == 4 * 3 and len(set(rows)) == len(rows)


def test_two_keys_use_fresh_randomness(make_mc):
    kg, _, _ = make_mc()
    first = kg.get_decryption_keys("agg", credentials=creds())
    second = kg.get_decryption_keys("agg", credentials=creds())
    assert first["policy"] == second["policy"] and first["y"] == second["y"]
    assert all(first["k_ipfe"][nid] != second["k_ipfe"][nid] for nid in LST)


def test_keys_of_one_master_key_all_open_one_set_of_ciphertexts(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients)
    for policy, y in (("region:EU", Y), ("or(region:EU, org:A)", Y), (POLICY, {nid: 1 for nid in LST})):
        dk = kg.get_decryption_keys("agg", credentials=creds(policy, y))
        assert agg.decrypt(cts, dk) == sum(X[nid] * y[nid] for nid in LST)


# ---------------------------------------------------------------------------
# setup and the keys


def test_bases_are_dual(make_mc):
    kg, _, _ = make_mc()
    group = PairingGroup(GROUP)
    g1 = group.deserialize(kg.get_public_parameters()["g1"].encode("utf-8"))
    g2 = group.deserialize(kg.get_public_parameters()["g2"].encode("utf-8"))
    gt = pair(g1, g2)
    f_star = kg.msk["f_star"]
    for nid in LST:
        ek = kg.get_private_keys(nid)
        f = [deserialize_many(group, row) for row in ek["f"]]
        h3 = deserialize_many(group, ek["h3"])
        for i in range(3):
            for j in range(3):
                assert vector_pair(f[i], f_star[j]) == (gt if i == j else gt ** 0), (nid, i, j)
        h_star = kg.msk["h_star"][nid]
        assert [vector_pair(h3, vector) for vector in h_star] == [gt ** 0, gt ** 0, gt]


def test_master_key_holds_group_elements(make_mc):      # D-2
    kg, _, _ = make_mc()
    assert set(kg.msk) == {"s", "u", "f_star", "h_star"}
    assert set(kg.msk["s"]) == set(kg.msk["u"]) == set(kg.msk["h_star"]) == set(LST)
    assert len(kg.msk["f_star"]) == 3 and all(len(row) == 8 for row in kg.msk["f_star"])
    assert all(element.type == G2 for row in kg.msk["f_star"] for element in row)
    for nid in LST:
        assert len(kg.msk["h_star"][nid]) == 3 and all(len(row) == 4 for row in kg.msk["h_star"][nid])
        assert all(element.type == G2 for row in kg.msk["h_star"][nid] for element in row)
    order = int(PairingGroup(GROUP).order())
    for value in [*kg.msk["s"].values(), *kg.msk["u"].values()]:
        assert 0 < int(value) < order                   # drawn from Z_q^*


def reachable_scalars(root):
    """Every int, ``mpz`` and decimal string reachable from the attributes of *root* through dicts and sequences."""
    found, seen, todo = set(), set(), [vars(root)]
    while todo:
        item = todo.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, dict):
            todo.extend(item.keys())
            todo.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            todo.extend(item)
        elif isinstance(item, (int, type(gp.mpz(0)))):
            found.add(int(item))
        elif isinstance(item, str) and item.lstrip("-").isdigit():
            found.add(int(item))
    return found


def test_setup_keeps_no_basis(make_mc, monkeypatch):
    drawn, matrices = [], []
    random_nonzero_zr, random_dual_bases = kp_ab_pairing.random_nonzero_zr, kp_ab_pairing.random_dual_bases

    def spy_zr(group):
        value = random_nonzero_zr(group)
        drawn.append(int(value))
        return value

    def spy_bases(*args, **kwargs):
        pair_of_matrices = random_dual_bases(*args, **kwargs)
        matrices.extend(pair_of_matrices)
        return pair_of_matrices

    monkeypatch.setattr(kp_ab_pairing, "random_nonzero_zr", spy_zr)
    monkeypatch.setattr(kp_ab_pairing, "random_dual_bases", spy_bases)
    kg, _, _ = make_mc()
    assert [len(m) for m in matrices] == [8, 8] + [4, 4] * 3        # F, F*, then H_i, H_i* of each client
    secrets = {int(value) for value in [*kg.msk["s"].values(), *kg.msk["u"].values()]}
    assert set(drawn) == secrets and len(drawn) == 2 * 3            # S and U, and no mu, no z
    kept = reachable_scalars(kg)
    assert secrets <= kept
    entries = {int(entry) for matrix in matrices for row in matrix for entry in row}
    assert not entries & kept                                       # the exponents of the bases are gone


def test_the_shares_of_one_add_up_to_one(mc_fixed):
    # sum_i p_i = 1 makes the mask of the clients cancel: prod_i e(c_i,ipfe, h*_i,1) = e(omega, g2), and the same
    # with omega' and h*_i,2. omega and omega' are the points of the tag, which the test hashes for itself.
    kg, _, _, cts, _ = mc_fixed
    group = PairingGroup(GROUP)
    pp = kg.get_public_parameters()
    g2 = group.deserialize(pp["g2"].encode("utf-8"))
    omega, omega_prime = tag_points(group, SCHEME_ID, pp["pp_binding"], encode_label("round-1"),
                                    attribute_set_digest(ATTRS))
    first, second = pair(omega, g2) ** 0, pair(omega_prime, g2) ** 0
    for nid in LST:
        c_ipfe = deserialize_many(group, cts[nid]["c_ipfe"])
        first *= vector_pair(c_ipfe, kg.msk["h_star"][nid][0])
        second *= vector_pair(c_ipfe, kg.msk["h_star"][nid][1])
    assert first == pair(omega, g2) and second == pair(omega_prime, g2)


def test_the_masked_plaintext_is_in_the_exponent(mc_fixed):
    # t_i = omega s_i + omega' u_i + x_i: e(t_i, g2) = e(omega, g2)^s_i e(omega', g2)^u_i e(g1, g2)^x_i
    kg, _, _, cts, _ = mc_fixed
    group = PairingGroup(GROUP)
    order = int(group.order())
    pp = kg.get_public_parameters()
    g1, g2 = (group.deserialize(pp[name].encode("utf-8")) for name in ("g1", "g2"))
    omega, omega_prime = tag_points(group, SCHEME_ID, pp["pp_binding"], encode_label("round-1"),
                                    attribute_set_digest(ATTRS))
    for nid in LST:
        t = group.deserialize(cts[nid]["t"].encode("utf-8"))
        assert pair(t, g2) == (pair(omega, g2) ** int(kg.msk["s"][nid]) * pair(omega_prime, g2) ** int(kg.msk["u"][nid])
                               * pair(g1, g2) ** (X[nid] % order))


def test_client_key_fields(make_mc):                    # nothing of another client in ek_i
    kg, _, _ = make_mc()
    keys = {nid: kg.get_private_keys(nid) for nid in LST}
    g1_tag = "{}:".format(G1)
    for nid, ek in keys.items():
        assert list(ek) == list(MCKPABClientKey.FIELDS)
        assert ek["nid"] == nid and ek["pp_binding"] == kg.get_public_parameters()["pp_binding"]
        assert len(ek["p_h1"]) == len(ek["p_h2"]) == len(ek["h3"]) == 4 and [len(row) for row in ek["f"]] == [8] * 3
        assert all(value.isdigit() for value in [ek["s"], ek["u"], *ek["p_h1"], *ek["p_h2"]])
        assert all(item.startswith(g1_tag) for item in [*ek["h3"], *(item for row in ek["f"] for item in row)])
        assert ek["s"] == kg.msk["s"][nid] and ek["u"] == kg.msk["u"][nid]
    own = {nid: {*keys[nid]["p_h1"], *keys[nid]["p_h2"], *keys[nid]["h3"], keys[nid]["s"], keys[nid]["u"]}
           for nid in LST}
    for a in LST:
        for b in LST:
            if a != b:
                assert not own[a] & own[b]                      # no value of the client b in the key of a
    assert keys["nid_0"]["f"] == keys["nid_1"]["f"] == keys["nid_2"]["f"]       # F is shared by all clients


def test_a_client_key_is_a_copy(make_mc):
    kg, _, _ = make_mc()
    ek = kg.get_private_keys("nid_1")
    ek["s"] = "0"
    ek["h3"].clear()
    assert kg.get_private_keys("nid_1")["s"] == kg.msk["s"]["nid_1"] and len(kg.get_private_keys("nid_1")["h3"]) == 4


def test_unknown_clients_have_no_key(make_mc):
    kg, _, _ = make_mc()
    assert [kg.get_private_keys(nid) for nid in ("nobody", 1, None, ["nid_0"], 1.0, True)] == [None] * 6
    kg, _, _ = make_mc(lst_nid=[1, 2, 3])
    assert kg.get_private_keys(1.0) is None and kg.get_private_keys(True) is None and kg.get_private_keys(4) is None
    assert kg.get_private_keys(np.int64(2)) == kg.get_private_keys(2)


def test_public_parameters_layout(make_mc):
    kg, _, _ = make_mc()
    pp = kg.get_public_parameters()
    assert list(pp) == list(MCKPABParameters.FIELDS)
    assert (pp["n"], pp["lst_nid"], pp["eta"]) == (3, LST, 1)
    assert (pp["bound_x"], pp["bound_y"], pp["max_rows"], pp["max_attributes"]) == ("10", "10", 64, 64)
    assert (pp["policy_compiler"], pp["attr_hash"], pp["tag_hash"]) == (POLICY_COMPILER, ATTR_HASH, TAG_HASH)
    assert pp["pp_binding"] == pp_digest(pp, PP_DOMAIN)
    assert not {"s", "u", "p_h1", "f", "h3"} & set(pp)


def test_public_parameters_hold_no_secret(make_mc):
    kg, _, _ = make_mc()
    secrets = reachable_scalars(kg) - {3, 10, 64}
    text = json.dumps(kg.get_public_parameters())
    assert not [value for value in secrets if value > 1000 and str(value) in text]


def test_setups_differ(make_mc):
    first, _, _ = make_mc()
    second, _, _ = make_mc()
    assert first.get_public_parameters()["pp_binding"] != second.get_public_parameters()["pp_binding"]


def test_integer_client_ids(make_mc):
    kg, clients, agg = make_mc(lst_nid=[10, 20, 30])
    x, y = {10: 1, 20: 2, 30: 3}, {10: 4, 20: -5, 30: 6}
    cts = encrypt_all(clients, x=x)
    dk = kg.get_decryption_keys("agg", credentials=creds(y=y))
    assert agg.decrypt(cts, dk) == 4 - 10 + 18
    assert list(dk["y"]) == [10, 20, 30] and cts[20]["nid"] == 20 and kg.get_private_keys(20)["nid"] == 20
    assert agg.decrypt({np.int64(nid): ct for nid, ct in cts.items()}, dk) == 4 - 10 + 18     # numpy ids are ints
    assert agg.decrypt({nid: dict(ct, nid=np.int64(nid)) for nid, ct in cts.items()}, dk) == 4 - 10 + 18
    assert agg.decrypt(cts, dk, fusion_weight={np.int64(nid): [v] for nid, v in y.items()}) == 4 - 10 + 18
    # JSON turns the keys of the dicts into strings: refused before any pairing, naming the clients
    with pytest.raises(FEValidationError, match="10"):
        agg.decrypt(cts, through_json(dk))


def test_objects_survive_json(mc_fixed):
    kg, clients, _, cts, dk = mc_fixed
    pp = through_json(kg.get_public_parameters())
    ek = through_json(kg.get_private_keys("nid_2"))
    agg = MCFEKPABPairing({"keys": {"pp": pp}})
    assert agg.decrypt(through_json(cts), through_json(dk)) == EXPECTED
    client = MCFEKPABPairing({"id": "nid_2", "keys": {"pp": pp, "sk": ek}})
    assert client.encrypt([X["nid_2"]], "round-1", attributes=ATTRS)["t"] == cts["nid_2"]["t"]   # t is deterministic


def test_numpy_integers_as_plaintext_and_weights(make_mc):
    kg, clients, agg = make_mc()
    cts = {nid: clients[nid].encrypt([np.int64(X[nid])], "round-1", attributes=ATTRS) for nid in LST}
    dk = kg.get_decryption_keys("agg", credentials=creds(y={nid: np.int32(Y[nid]) for nid in LST}))
    assert agg.decrypt(cts, dk) == EXPECTED


# ---------------------------------------------------------------------------
# the records


def test_functional_key_layout(mc_fixed):
    _, _, _, _, dk = mc_fixed
    group = PairingGroup(GROUP)
    assert list(dk) == list(MCKPABFunctionalKey.FIELDS)
    assert dk["policy_compiler"] == POLICY_COMPILER
    assert dk["y"] == {nid: str(Y[nid]) for nid in LST}
    assert set(dk["k_rows"]) == set(dk["k_ipfe"]) == set(LST)
    for nid in LST:
        assert len(dk["k_rows"][nid]) == 2 and all(len(row) == 8 for row in dk["k_rows"][nid])     # two leaves
        assert len(dk["k_ipfe"][nid]) == 4
        for item in [*dk["k_ipfe"][nid], *(item for row in dk["k_rows"][nid] for item in row)]:
            assert group.deserialize(item.encode("utf-8")).type == G2


def test_ciphertext_layout(mc_fixed):
    kg, _, _, cts, _ = mc_fixed
    group = PairingGroup(GROUP)
    for nid, ct in cts.items():
        assert list(ct) == list(MCKPABCiphertext.FIELDS)
        assert ct["nid"] == nid and ct["label"] == encode_label("round-1") and ct["attributes"] == ATTRS
        assert ct["attributes_digest"] == attribute_set_digest(ATTRS)
        assert (ct["policy_compiler"], ct["attr_hash"], ct["tag_hash"]) == (POLICY_COMPILER, ATTR_HASH, TAG_HASH)
        assert ct["pp_binding"] == kg.get_public_parameters()["pp_binding"]
        assert len(ct["c_attr"]) == 2 and all(len(row) == 8 for row in ct["c_attr"]) and len(ct["c_ipfe"]) == 4
        assert group.deserialize(ct["t"].encode("utf-8")).type == G1
        assert all(group.deserialize(item.encode("utf-8")).type == G1
                   for item in [*ct["c_ipfe"], *(item for row in ct["c_attr"] for item in row)])


def test_a_client_encrypts_once_under_a_tag_for_t_is_determined(mc_fixed):
    # why the scheme asks for one ciphertext per client and tag: t has no randomness of its own, so two ciphertexts of
    # one client under one tag differ in t by the difference of the plaintexts, in the exponent
    kg, clients, _, cts, _ = mc_fixed
    group = PairingGroup(GROUP)
    again = clients["nid_0"].encrypt([X["nid_0"]], "round-1", attributes=ATTRS)
    assert again["t"] == cts["nid_0"]["t"]
    assert again["c_ipfe"] != cts["nid_0"]["c_ipfe"] and again["c_attr"] != cts["nid_0"]["c_attr"]
    other = clients["nid_0"].encrypt([X["nid_0"] + 3], "round-1", attributes=ATTRS)
    t, t_other = (group.deserialize(ct["t"].encode("utf-8")) for ct in (cts["nid_0"], other))
    g1 = group.deserialize(kg.get_public_parameters()["g1"].encode("utf-8"))
    assert t_other == t * g1 ** 3


def test_tags_and_attributes_give_other_ciphertexts(mc_fixed):
    _, clients, _, cts, _ = mc_fixed
    other_tag = clients["nid_0"].encrypt([X["nid_0"]], "round-2", attributes=ATTRS)
    other_set = clients["nid_0"].encrypt([X["nid_0"]], "round-1", attributes=ATTRS + ["org:A"])
    assert other_tag["t"] != cts["nid_0"]["t"] and other_set["t"] != cts["nid_0"]["t"]
    assert other_tag["label"] != cts["nid_0"]["label"]
    assert other_set["attributes_digest"] != cts["nid_0"]["attributes_digest"]
    element = clients["nid_0"].encrypt([1], ElementLabel("round-1", 0, (0,)), attributes=ATTRS)
    assert element["label"] == encode_label(ElementLabel("round-1", 0, (0,))) != cts["nid_0"]["label"]


def test_multi_pair_is_called_once_with_n_times_the_pairs_of_a_client_and_one_more(mc_fixed, monkeypatch):
    kg, clients, agg, cts, dk = mc_fixed
    calls = []
    real = pairing_backend.multi_pair

    def spy(lhs, rhs):
        calls.append(len(lhs))
        return real(lhs, rhs)

    monkeypatch.setattr(kp_ab_pairing, "multi_pair", spy)
    assert agg.decrypt(cts, dk) == EXPECTED
    assert calls == [3 * (8 * 2 + 4) + 1]                           # two used rows
    one = kg.get_decryption_keys("agg", credentials=creds("or(region:EU, org:A)"))
    assert agg.decrypt(cts, one) == EXPECTED
    assert calls[-1] == 3 * (8 * 1 + 4) + 1                         # the first satisfied child only


def test_the_generator_pair_is_computed_once(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    calls = []
    real = pairing_backend.pair
    monkeypatch.setattr(kp_ab_pairing, "pair", lambda a, b: calls.append(1) or real(a, b))
    for _ in range(3):
        assert agg.decrypt(cts, dk) == EXPECTED
    assert calls == [1]


def test_encrypt_decodes_the_key_once(make_mc, monkeypatch):
    kg, clients, _ = make_mc()
    calls = []
    real = kp_ab_pairing._elements
    monkeypatch.setattr(kp_ab_pairing, "_elements", lambda *args: calls.append(args[3]) or real(*args))
    for _ in range(3):
        clients["nid_0"].encrypt([1], "round-1", attributes=ATTRS)
    assert sorted(calls) == ["f", "f", "f", "g1", "h3"]


def test_ndarray_helpers_are_not_implemented(make_mc):
    _, _, agg = make_mc()
    for name in ("encrypt_lst_ndarray", "compute_lst_ndarray_ct", "decrypt_lst_ndarray_ct"):
        with pytest.raises(NotImplementedError):
            getattr(agg, name)([])


# ---------------------------------------------------------------------------
# tag points


TAG_ARGS = (SCHEME_ID, "00" * 32, encode_label("round-1"), "11" * 32)


def serialised(group, args):
    return tuple(group.serialize(point).decode("utf-8") for point in tag_points(group, *args))


def test_tag_points_depend_on_every_input():
    group = PairingGroup(GROUP)
    reference = serialised(group, TAG_ARGS)
    assert serialised(group, TAG_ARGS) == reference and reference[0] != reference[1]
    changes = [(0, "mcfe/kp_ab_pairing_per_client"), (1, "01" * 32), (2, encode_label("round-2")), (3, "12" * 32),
               (3, None)]
    for index, value in changes:
        args = list(TAG_ARGS)
        args[index] = value
        points = serialised(group, args)
        assert points[0] != reference[0] and points[1] != reference[1], (index, value)
    # the digest None is its own input, and the pair of inputs is not blurred by moving text from one to the next
    assert serialised(group, TAG_ARGS[:3] + (None,)) != serialised(group, TAG_ARGS[:3] + ("null",))
    assert serialised(group, ("a", "bc", "d", "e")) != serialised(group, ("ab", "c", "d", "e"))


def test_tag_points_vector():
    group = PairingGroup(GROUP)
    if group.groupType() != "BLS12-381":
        pytest.skip("the known answer is for BLS12-381")
    message = ('["mcfe/kp_ab_pairing","' + "00" * 32 + '","[\\"label\\", \\"round-1\\"]","' + "11" * 32 + '"]').encode()
    assert [group.serialize(hash_to_g1(group, message + suffix, b"PYFE4AI-V05-MCFE-KPAB-TAG")).decode()
            for suffix in (b"|0", b"|1")] == list(serialised(group, TAG_ARGS))
    assert serialised(group, TAG_ARGS) == (
        "1:ke/QHmt3cw/kFi3FR+ceZbM/dqXgMxJx+DLeXk1VWjxMgfcYLY0DZPegbLPYZmKL",
        "1:qj2qUm0xYD3fp5eAsBJnO4WoMt5gv3AYeQHE5LR2c6ULsUfel+dXwEaPdbdjyeMq",
    )
    assert serialised(group, TAG_ARGS[:3] + (None,)) == (                  # the digest None is JSON null
        "1:jkLuj2Us3j+KRZjhQ1OpKIGr9KwGRQT1TGJFC1NTeGEKGyqzFDfQN+ffKFU40GYP",
        "1:hXWrEzlsKCDxa9w3+cfd2nXvqUkN/MAzslgtNqzZK0SuX19F5aeM0e7NbRIFU86E",
    )


def test_the_hash_domain_is_the_documented_one():
    assert TAG_HASH == "PYFE4AI-V05-MCFE-KPAB-TAG" and TAG_DST == b"PYFE4AI-V05-MCFE-KPAB-TAG"
    assert (POLICY_COMPILER, ATTR_HASH) == ("c1", "shake256-64")


# ---------------------------------------------------------------------------
# the configuration and its record


def test_limits_have_defaults(own_parameter_cache):
    kg = MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": GROUP})
    assert (kg.max_rows, kg.max_attributes, kg.eta) == (64, 64, 1)
    assert kg.lst_nid == LST


def record_file(cache):
    return os.path.join(cache, "authority", CryptoCONST.TYPE_MCFE_KP_AB, "param.json")


def test_the_record_holds_the_configuration_only(make_mc, tmp_path):
    cache = str(tmp_path)
    make_mc(cache_dir=cache)
    with open(record_file(cache), encoding="utf-8") as f:
        record = json.load(f)
    assert list(record) == list(MCKPABRecord.FIELDS)
    assert (record["n"], record["lst_nid"], record["bound_x"]) == (3, LST, "10")
    assert [record[name] for name in ("policy_compiler", "attr_hash", "tag_hash")] == [
        POLICY_COMPILER, ATTR_HASH, TAG_HASH]


def test_the_scheme_says_its_record_is_the_configuration():
    assert MCFEKPABPairingKeyGenerator._config_only_record is True
    assert MCFEKPABPairingKeyGenerator._scheme_type == "MCFE_KP_AB"


@pytest.mark.parametrize("override", [{"n": 4}, {"lst_nid": ["a", "b", "c"]}, {"max_attributes": 8}, {"bound_x": 11}])
def test_a_new_configuration_replaces_the_record_without_a_warning(make_mc, tmp_path, recwarn, caplog, override):
    cache = str(tmp_path)
    make_mc(cache_dir=cache)
    with caplog.at_level(logging.INFO, logger="pyfe4ai.core.base"):
        make_mc(cache_dir=cache, **override)
    assert not [w for w in recwarn if issubclass(w.category, FEWarning)], [str(w.message) for w in recwarn]
    assert [r.levelno for r in caplog.records if "does not match" in r.getMessage()] == [logging.INFO]


def test_same_configuration_keeps_the_cached_record(make_mc, tmp_path):
    cache = str(tmp_path)
    make_mc(cache_dir=cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FEWarning)
        make_mc(cache_dir=cache)


def test_create_and_list():
    kg, crypto_cls = pyfe4ai.create(SCHEME_ID, n=2, pairing_group_param=GROUP)
    assert (type(kg).__name__, crypto_cls.__name__) == ("MCFEKPABPairingKeyGenerator", "MCFEKPABPairing")
    assert kp_ab_pairing._SCHEME_ID == SCHEME_ID == pyfe4ai.get_scheme(SCHEME_ID).id   # hashed into the tag points
    # `in`, not `==`: each attribute-based scheme that is added joins the lists
    assert SCHEME_ID in pyfe4ai.list_schemes("mcfe", capabilities=["labels", "access_control"])
    assert {"sife/cp_ab_pairing", "sife/kp_ab_pairing", SCHEME_ID} <= set(
        pyfe4ai.list_schemes(capabilities="access_control"))
    assert pyfe4ai.get_scheme(SCHEME_ID).capabilities == {"access_control", "labels", "pairing"}
    assert SCHEME_ID not in pyfe4ai.list_schemes("sife", capabilities="access_control")
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: crypto_cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in kg.lst_nid}
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", {"nid_0": 2, "nid_1": -1}))
    cts = {"nid_0": clients["nid_0"].encrypt([3], "round", attributes=["org:A"]),
           "nid_1": clients["nid_1"].encrypt([4], "round", attributes=["org:A"])}
    assert crypto_cls({"keys": {"pp": pp}}).decrypt(cts, dk) == 2


# ---------------------------------------------------------------------------
# the helpers that the crypto objects share


def test_check_dlog_limit_names_its_first_factor():
    ac_codec.check_dlog_limit(2 ** 24, 2 ** 12, 2 ** 12)
    with pytest.raises(FEValidationError, match=r"^eta \* bound_x"):
        ac_codec.check_dlog_limit(2 ** 25, 2 ** 12, 2 ** 12)
    with pytest.raises(FEValidationError, match=r"^n \* bound_x"):
        ac_codec.check_dlog_limit(2 ** 25, 2 ** 12, 2 ** 12, "n")


def test_copy_parameters():
    original = {"a": [1, 2], "b": {"c": 3}}
    copied = ac_codec.copy_parameters(original)
    assert copied == original and copied["a"] is not original["a"] and copied["b"] is not original["b"]
    with pytest.raises(FEValidationError, match="client key must be JSON data"):
        ac_codec.copy_parameters({"f": threading.Lock()}, "client key")
    with pytest.raises(FEValidationError, match="public parameters must be JSON data"):
        ac_codec.copy_parameters({"f": threading.Lock()})


def test_load_group():
    group = ac_codec.load_group(GROUP)
    assert int(group.order()) == int(PairingGroup(GROUP).order())
    for name in (5, "nothing", None, ["BLS12-381"]):
        with pytest.raises(FEValidationError, match="pairing_group_param"):
            ac_codec.load_group(name)
    if pairing_backend.pairing_available("SS512"):
        with pytest.raises(FEValidationError, match="symmetric"):
            ac_codec.load_group("SS512")


def test_check_marker():
    ac_codec.check_marker("c1", "c1", "the policy_compiler of the ciphertext")
    with pytest.raises(FEValidationError, match=r"^the policy_compiler of the ciphertext is 'c2', not this version's"):
        ac_codec.check_marker("c2", "c1", "the policy_compiler of the ciphertext")
    with pytest.raises(FEValidationError, match="x" * 39 + r"\.\.\."):
        ac_codec.check_marker("x" * 500, "c1", "a marker")


def test_scalar():
    order = 101
    assert [ac_codec.scalar(text, "s", order) for text in ("0", "7", "100")] == [0, 7, 100]
    for bad in ("101", "-1", "07", "", "1e2", " 1", "1 ", 5, None, b"1", "9" * 1000, "٣"):
        with pytest.raises(FEValidationError, match="s must hold"):
            ac_codec.scalar(bad, "s", order)


