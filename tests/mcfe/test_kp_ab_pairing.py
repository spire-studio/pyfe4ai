"""The multi-client key-policy attribute-based IPFE (NPP22 Sec. 5.2 Fig. 3): setup, key generation, encryption and
decryption, and what the decryption refuses before it pairs anything.

A key whose policy the attributes of the ciphertexts do not satisfy gives a ``FEKeyError``; malformed input, ciphertexts
that are not those of every client under one tag and one set of attributes, or objects of another setup a
``FEValidationError``; a key and ciphertexts that are well formed but were not made for each other a ``ValueError`` from
the discrete log, which is not a ``FEValidationError``.
"""

import copy
import json
import logging
import os
import threading
import warnings

import gmpy2 as gp
import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FEValidationError, FEWarning
from pyfe4ai.core.formats.pairing import MCKPABCiphertext, MCKPABClientKey, MCKPABFunctionalKey
from pyfe4ai.core.formats.pairing import MCKPABParameters, MCKPABRecord
from pyfe4ai.policy import ATTR_HASH, POLICY_COMPILER, attribute_set_digest, canonical_encoding, parse
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

LST = ["nid_0", "nid_1", "nid_2"]
X = {"nid_0": 5, "nid_1": -3, "nid_2": 7}
Y = {"nid_0": 2, "nid_1": 1, "nid_2": -4}
EXPECTED = sum(X[nid] * Y[nid] for nid in LST)
ATTRS = ["purpose:research", "region:EU"]
POLICY = "and(purpose:research, region:EU)"

pytestmark = pytest.mark.usefixtures("own_parameter_cache")


def creds(policy=POLICY, y=Y):
    return {"fusion_weight": {nid: [value] for nid, value in y.items()}, "policy": policy}


def encrypt_all(clients, x=X, label="round-1", attributes=ATTRS):
    return {nid: clients[nid].encrypt([x[nid]], label, attributes=attributes) for nid in x}


def through_json(value):
    return json.loads(json.dumps(value))


def refuse_to_compute(monkeypatch):
    """Make the decoding of a group element, a pairing and a discrete log fail the test."""
    def never(*args, **kwargs):
        pytest.fail("a group element was decoded, or a pairing or a discrete log was computed")

    for name in ("_elements", "multi_pair", "bounded_discrete_log_gt"):
        monkeypatch.setattr(kp_ab_pairing, name, never)


@pytest.fixture
def mc_fixed(make_mc):
    """A key generator, the clients, an aggregator, the ciphertexts of ``X`` under ``round-1`` for ``ATTRS``, a key for
    ``POLICY`` and ``Y``."""
    kg, clients, agg = make_mc()
    return kg, clients, agg, encrypt_all(clients), kg.get_decryption_keys("agg", credentials=creds())


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


def test_every_client_is_needed(make_mc, monkeypatch):  # Review Focus 2
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients)
    dk = kg.get_decryption_keys("agg", credentials=creds(y=dict(Y, nid_0=0)))      # nid_0 has the weight 0
    refuse_to_compute(monkeypatch)
    for missing in LST:
        with pytest.raises(FEValidationError, match=missing):
            agg.decrypt({nid: ct for nid, ct in cts.items() if nid != missing}, dk)
    with pytest.raises(FEValidationError, match="nid_1.*nid_2|nid_2.*nid_1"):
        agg.decrypt({"nid_0": cts["nid_0"]}, dk)
    with pytest.raises(FEValidationError, match="nid_0"):
        agg.decrypt({}, dk)
    with pytest.raises(FEValidationError, match="nid_9"):                           # a ciphertext of no client
        agg.decrypt(dict(cts, nid_9=cts["nid_0"]), dk)


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
    monkeypatch.setattr(kp_ab_pairing, "_elements", pytest.fail)
    with pytest.raises(FEKeyError, match="do not satisfy"):
        agg.decrypt(cts, dk)


def test_a_key_of_another_setup_does_not_open(make_mc):
    kg, clients, agg = make_mc()
    other_kg, _, _ = make_mc()
    with pytest.raises(FEValidationError, match="pp_binding"):
        agg.decrypt(encrypt_all(clients), other_kg.get_decryption_keys("agg", credentials=creds()))


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


def test_keys_before_setup(own_parameter_cache):
    kg = MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": GROUP})
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_private_keys("nid_0")
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_decryption_keys("agg", credentials=creds())


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


@pytest.mark.parametrize("args", [(1, "a", "b", "c"), ("a", b"b", "c", "d"), ("a", "b", None, "d"),
                                  ("a", "b", "c", 5), ("a", "b", "c", b"d")])
def test_tag_points_refuses_what_is_no_text(args):
    with pytest.raises(FEValidationError):
        tag_points(PairingGroup(GROUP), *args)


# ---------------------------------------------------------------------------
# the configuration and its record


def test_clients_are_needed(own_parameter_cache):
    with pytest.raises(FEConfigError, match="n"):
        MCFEKPABPairingKeyGenerator({"pairing_group_param": GROUP})


@pytest.mark.parametrize("n", [0, -1, True, 2.0, "3"])
def test_n_must_be_a_positive_int(n, own_parameter_cache):
    with pytest.raises(FEValidationError, match="n must be a positive int"):
        MCFEKPABPairingKeyGenerator({"n": n, "pairing_group_param": GROUP})


@pytest.mark.parametrize("lst_nid", [["a", "b"], ["a", "b", "a"], ["a", "b", 1], "abc", [None, None, None]])
def test_the_client_ids_must_be_n_distinct_ids(lst_nid, own_parameter_cache):
    with pytest.raises(FEValidationError, match="lst_nid"):
        MCFEKPABPairingKeyGenerator({"n": 3, "lst_nid": lst_nid, "pairing_group_param": GROUP})


@pytest.mark.parametrize("option", ["bound_x", "bound_y", "max_rows", "max_attributes"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "5", None])
def test_limits_must_be_positive_ints(option, value, own_parameter_cache):
    with pytest.raises(FEValidationError, match=option):
        MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": GROUP, option: value})


def test_limits_have_defaults(own_parameter_cache):
    kg = MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": GROUP})
    assert (kg.max_rows, kg.max_attributes, kg.eta) == (64, 64, 1)
    assert kg.lst_nid == LST


@pytest.mark.parametrize("eta", [2, 0, True, 1.0, "1", None])
def test_eta_is_one(eta, own_parameter_cache):
    with pytest.raises(FEValidationError, match="eta must be 1"):
        MCFEKPABPairingKeyGenerator({"n": 3, "eta": eta, "pairing_group_param": GROUP})
    assert MCFEKPABPairingKeyGenerator({"n": 3, "eta": 1, "pairing_group_param": GROUP}).eta == 1


def test_dlog_limit(own_parameter_cache):
    config = {"n": 3, "bound_x": 2 ** 24, "bound_y": 2 ** 24, "pairing_group_param": GROUP}
    with pytest.raises(FEValidationError, match=r"n \* bound_x \* bound_y.*2\*\*48"):
        MCFEKPABPairingKeyGenerator(config)
    assert MCFEKPABPairingKeyGenerator(dict(config, n=1)).n == 1            # 2**48 itself is allowed


def test_a_symmetric_group_is_refused(own_parameter_cache):
    if not pairing_backend.pairing_available("SS512"):
        pytest.skip("no charm: SS512 is not built")
    with pytest.raises(FEValidationError, match="symmetric"):
        MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": "SS512"})


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


# ---------------------------------------------------------------------------
# key requests and encryption that are refused


@pytest.mark.parametrize("credentials", [
    None, {}, {"policy": POLICY}, {"fusion_weight": {nid: [1] for nid in LST}},
    {"fusion_weight": {nid: [1] for nid in LST[:2]}, "policy": POLICY},                     # a client missing
    {"fusion_weight": {**{nid: [1] for nid in LST}, "nid_9": [1]}, "policy": POLICY},       # a client too many
    {"fusion_weight": {**{nid: [1] for nid in LST}, 1: [1]}, "policy": POLICY},
    {"fusion_weight": {**{nid: [1] for nid in LST[:2]}, True: [1]}, "policy": POLICY},      # True is no client id
    {"fusion_weight": {nid: [11] for nid in LST}, "policy": POLICY},                        # above bound_y
    {"fusion_weight": {nid: [1, 1] for nid in LST}, "policy": POLICY},                      # eta = 1
    {"fusion_weight": {nid: [] for nid in LST}, "policy": POLICY},
    {"fusion_weight": {nid: 1 for nid in LST}, "policy": POLICY},
    {"fusion_weight": {nid: [1.5] for nid in LST}, "policy": POLICY},
    {"fusion_weight": [1, 1, 1], "policy": POLICY},
    {"fusion_weight": {nid: [1] for nid in LST}, "policy": None},
    {"fusion_weight": {nid: [1] for nid in LST}, "policy": "or(region:EU,"},
    {"fusion_weight": {nid: [1] for nid in LST}, "attributes": ["region:EU"]},
    ["fusion_weight", "policy"],
])
def test_key_requests_are_checked(make_mc, credentials):
    kg, _, _ = make_mc()
    with pytest.raises((FEKeyError, FEValidationError)):
        kg.get_decryption_keys("agg", credentials=credentials)


def test_a_policy_of_more_rows_than_the_limit_is_refused(make_mc):
    kg, _, _ = make_mc(max_rows=2)
    with pytest.raises(FEValidationError, match="max_rows|rows"):
        kg.get_decryption_keys("agg", credentials=creds("and(a:1, b:1, c:1)"))
    assert kg.get_decryption_keys("agg", credentials=creds("and(a:1, b:1)"))


def test_encrypt_without_a_key_is_a_key_error(make_mc):
    _, _, agg = make_mc()
    with pytest.raises(FEKeyError, match="sk"):
        agg.encrypt([1], "round-1", attributes=ATTRS)


@pytest.mark.parametrize("plaintext", [[11], [-11], [], [1, 2], 5, "1", [1.5], [True], None])
def test_encrypt_checks_the_plaintext(plaintext, make_mc):
    _, clients, _ = make_mc()
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt(plaintext, "round-1", attributes=ATTRS)


@pytest.mark.parametrize("label", [None, 5, b"round-1", ["round-1"], ("round-1", 0)])
def test_encrypt_checks_the_label(label, make_mc):
    _, clients, _ = make_mc()
    with pytest.raises(FEValidationError, match="label"):
        clients["nid_0"].encrypt([1], label, attributes=ATTRS)


@pytest.mark.parametrize("attributes", [None, "region:EU", 5, ["region"], [5], ["region:EU", None]])
def test_encrypt_checks_the_attributes(attributes, make_mc):
    _, clients, _ = make_mc()
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1], "round-1", attributes=attributes)


def test_encrypt_limits_the_attributes(make_mc):
    _, clients, _ = make_mc(max_attributes=2)
    with pytest.raises(FEValidationError, match="max_attributes"):
        clients["nid_0"].encrypt([1], "round-1", attributes=["a:1", "b:1", "c:1"])
    assert clients["nid_0"].encrypt([1], "round-1", attributes=["a:1", "b:1", "a:1"])


def test_a_refused_encryption_computes_nothing(make_mc, monkeypatch):
    _, clients, _ = make_mc()

    def never(*args, **kwargs):
        pytest.fail("a point was hashed or a group element was combined for input that is refused")

    monkeypatch.setattr(kp_ab_pairing, "tag_points", never)
    monkeypatch.setattr(kp_ab_pairing, "multiexp_many", never)
    for plaintext, label, attributes in (([99], "round-1", ATTRS), ([1], 5, ATTRS), ([1], "round-1", None)):
        with pytest.raises(FEValidationError):
            clients["nid_0"].encrypt(plaintext, label, attributes=attributes)


# ---------------------------------------------------------------------------
# the crypto object: public parameters and encryption keys


def rebound(pp, **changes):
    """*pp* with *changes* and the ``pp_binding`` that goes with them."""
    changed = dict(copy.deepcopy(pp), **changes)
    changed["pp_binding"] = pp_digest(changed, PP_DOMAIN)
    return changed


def test_crypto_needs_public_parameters():
    with pytest.raises(FEKeyError, match="pp"):
        MCFEKPABPairing({"keys": {"other": 1}})
    with pytest.raises(FEKeyError):
        MCFEKPABPairing({"keys": {}})


def test_tampered_pp(make_mc):
    kg, _, _ = make_mc()
    pp = kg.get_public_parameters()
    tampered = dict(pp, bound_x="1000")
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": tampered}})
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": dict(pp, lst_nid=["nid_0", "nid_2", "nid_1"])}})   # the order fixes p_i


@pytest.mark.parametrize("field", ["policy_compiler", "attr_hash", "tag_hash"])
def test_crypto_refuses_an_unknown_version_marker(field, make_mc):
    kg, _, _ = make_mc()
    with pytest.raises(FEValidationError, match=field):
        MCFEKPABPairing({"keys": {"pp": rebound(kg.get_public_parameters(), **{field: "v9"})}})


@pytest.mark.parametrize("changes,match", [
    ({"eta": 2}, "eta"), ({"eta": True}, "eta"), ({"n": 4}, "lst_nid|clients"), ({"n": "3"}, "n must be"),
    ({"lst_nid": ["nid_0", "nid_1"]}, "lst_nid"), ({"lst_nid": "abc"}, "lst_nid"),
    ({"lst_nid": ["nid_0", "nid_1", "nid_1"]}, "lst_nid"), ({"lst_nid": ["nid_0", "nid_1", 2]}, "lst_nid"),
    ({"bound_x": "1" * 17}, "bound_x"), ({"bound_y": 3}, "bound_y"), ({"max_rows": 0}, "max_rows"),
    ({"max_attributes": True}, "max_attributes"), ({"bound_x": str(2 ** 40), "bound_y": str(2 ** 40)}, "2\\*\\*48"),
    ({"pairing_group_param": 5}, "pairing_group_param"), ({"pairing_group_param": "nothing"}, "pairing group"),
])
def test_crypto_refuses_parameters_that_are_not_these(changes, match, make_mc):
    kg, _, _ = make_mc()
    with pytest.raises(FEValidationError, match=match):
        MCFEKPABPairing({"keys": {"pp": rebound(kg.get_public_parameters(), **changes)}})


@pytest.mark.parametrize("field", MCKPABParameters.FIELDS)
def test_crypto_refuses_public_parameters_without_a_field(field, make_mc):
    kg, _, _ = make_mc()
    pp = dict(kg.get_public_parameters())
    del pp[field]
    with pytest.raises(FEValidationError, match=field):
        MCFEKPABPairing({"keys": {"pp": pp}})


def test_parameters_changed_after_construction_do_not_reach_the_object(make_mc):
    kg, clients, agg = make_mc()
    pp = copy.deepcopy(kg.get_public_parameters())
    sk = kg.get_private_keys("nid_0")
    client = MCFEKPABPairing({"id": "nid_0", "keys": {"pp": pp, "sk": sk}})
    pp["bound_x"], pp["lst_nid"], sk["s"] = "1", ["x"], "0"
    sk["p_h1"][0] = "0"
    cts = encrypt_all(clients)
    cts["nid_0"] = client.encrypt([X["nid_0"]], "round-1", attributes=ATTRS)
    assert client.bound_x == 10 and client.lst_nid == LST
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds())) == EXPECTED


def test_parameters_that_cannot_be_copied_are_refused(make_mc):
    kg, _, _ = make_mc()
    pp = dict(kg.get_public_parameters(), extra=threading.Lock())
    with pytest.raises(FEValidationError, match="public parameters"):
        MCFEKPABPairing({"keys": {"pp": pp}})


def test_a_client_key_of_another_setup_is_refused(make_mc):
    kg, _, _ = make_mc()
    other, _, _ = make_mc()
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": other.get_private_keys("nid_0")}})


def test_the_id_of_a_client_must_be_the_one_of_its_key(make_mc):
    kg, _, _ = make_mc()
    pp = kg.get_public_parameters()
    with pytest.raises(FEValidationError, match="nid_1"):
        MCFEKPABPairing({"id": "nid_1", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    assert MCFEKPABPairing({"keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}}).encrypt(
        [1], "t", attributes=[])["nid"] == "nid_0"


@pytest.mark.parametrize("field,value,match", [
    ("nid", "stranger", "no client"), ("nid", ["nid_0"], "no client"), ("nid", True, "no client"),
    ("s", "-1", "s must hold"), ("s", "01", "s must hold"), ("s", "x", "s must hold"), ("s", 5, "s must hold"),
    ("s", "9" * 200, "s must hold"), ("u", "", "u must hold"),
    ("p_h1", ["1", "2", "3"], "p_h1"), ("p_h1", "1234", "p_h1"), ("p_h1", ["1", "2", "3", "x"], "p_h1"),
    ("p_h2", ["1", "2", "3", "4", "5"], "p_h2"), ("p_h2", [1, 2, 3, 4], "p_h2"),
    ("h3", ["1:x"] * 3, "h3"), ("f", [["1:x"] * 8] * 2, "f"), ("f", [["1:x"] * 8, ["1:x"] * 8, ["1:x"] * 7], "f"),
])
def test_crypto_refuses_a_malformed_client_key(field, value, match, make_mc):
    kg, _, _ = make_mc()
    sk = kg.get_private_keys("nid_0")
    sk[field] = value
    with pytest.raises(FEValidationError, match=match):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


def test_a_scalar_must_be_below_the_group_order(make_mc):
    kg, _, _ = make_mc()
    order = int(PairingGroup(GROUP).order())
    sk = kg.get_private_keys("nid_0")
    sk["s"] = str(order)
    with pytest.raises(FEValidationError, match="below the group order"):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})
    sk["s"] = str(order - 1)
    MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


@pytest.mark.parametrize("sk", [None, [], "key", 5])
def test_the_client_key_must_be_a_dict(sk, make_mc):
    kg, _, _ = make_mc()
    with pytest.raises((FEKeyError, FEValidationError)):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


@pytest.mark.parametrize("field", MCKPABClientKey.FIELDS)
def test_crypto_refuses_a_client_key_without_a_field(field, make_mc):
    kg, _, _ = make_mc()
    sk = kg.get_private_keys("nid_0")
    del sk[field]
    with pytest.raises(FEValidationError, match=field):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


def test_a_tampered_element_of_the_client_key_is_refused_when_it_is_read(make_mc):
    kg, _, _ = make_mc()
    sk = kg.get_private_keys("nid_0")
    sk["h3"][0] = kg.get_decryption_keys("agg", credentials=creds())["k_ipfe"]["nid_0"][0]       # an element of G2
    client = MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})
    with pytest.raises(FEValidationError, match="h3"):
        client.encrypt([1], "round-1", attributes=ATTRS)


# ---------------------------------------------------------------------------
# decryption: what is refused before anything is paired


def test_a_ciphertext_in_the_slot_of_another_client_is_refused(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    refuse_to_compute(monkeypatch)
    swapped = dict(cts, nid_0=cts["nid_1"], nid_1=cts["nid_0"])
    with pytest.raises(FEValidationError, match="nid_0.*nid_1"):
        agg.decrypt(swapped, dk)


@pytest.mark.parametrize("owner,field", [("nid_0", "label"), ("nid_1", "label"), ("nid_2", "label"),
                                         ("nid_0", "attributes_digest"), ("nid_2", "attributes_digest")])
def test_ciphertexts_that_differ_in_the_tag_or_the_attributes_are_refused(owner, field, mc_fixed, monkeypatch):
    _, clients, agg, cts, dk = mc_fixed
    if field == "label":
        cts[owner] = clients[owner].encrypt([X[owner]], "round-2", attributes=ATTRS)
    else:
        cts[owner] = clients[owner].encrypt([X[owner]], "round-1", attributes=ATTRS + ["org:A"])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="differ in their {}".format(field)):
        agg.decrypt(cts, dk)


def test_attributes_that_are_not_those_of_their_digest_are_refused(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    cts["nid_1"]["attributes"] = ATTRS + ["zzz:1"]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="attributes_digest of the ciphertext of nid_1"):
        agg.decrypt(cts, dk)
    cts["nid_1"]["attributes"] = ["region:EU", "purpose:research"]                  # out of order
    with pytest.raises(FEValidationError, match="sorted"):
        agg.decrypt(cts, dk)


def test_the_label_argument_must_be_the_one_of_the_ciphertexts(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    assert agg.decrypt(cts, dk, label="round-1") == EXPECTED
    assert agg.decrypt(cts, dk, label=None) == EXPECTED
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt(cts, dk, label="round-2")
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt(cts, dk, label=5)
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt(cts, dk, label=ElementLabel("round-1", 0, (0,)))


def test_the_fusion_weight_must_be_the_weights_of_the_key(mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    assert agg.decrypt(cts, dk, fusion_weight={nid: [Y[nid]] for nid in LST}) == EXPECTED
    for wrong in ({**{nid: [Y[nid]] for nid in LST}, "nid_2": [-3]},
                  {nid: [Y[nid]] for nid in LST[:2]},
                  {**{nid: [Y[nid]] for nid in LST}, "nid_9": [1]},
                  {nid: [Y[nid], 0] for nid in LST},
                  {nid: Y[nid] for nid in LST},
                  [Y[nid] for nid in LST], "fusion"):
        with pytest.raises(FEValidationError, match="fusion"):
            agg.decrypt(cts, dk, fusion_weight=wrong)


def test_a_client_that_hashed_other_points_than_it_says_gives_no_value(mc_fixed):       # Review Focus 3
    # nid_1 hashed "round-2" and says "round-1": every field agrees, and there is no value, not a wrong one
    _, clients, agg, cts, dk = mc_fixed
    hashed_other = clients["nid_1"].encrypt([X["nid_1"]], "round-2", attributes=ATTRS)
    cts["nid_1"] = dict(hashed_other, label=cts["nid_0"]["label"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)


def test_a_client_that_hashed_another_set_of_attributes_than_it_says_gives_no_value(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=["a:1", "b:1"])
    cts["nid_1"] = clients["nid_1"].encrypt([X["nid_1"]], "round-1", attributes=["a:1", "c:1"])
    cts["nid_1"]["attributes"] = ["a:1", "b:1"]                         # the row of a:1 is where the key reads it
    cts["nid_1"]["attributes_digest"] = attribute_set_digest(["a:1", "b:1"])
    dk = kg.get_decryption_keys("agg", credentials=creds("a:1"))
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)


def test_the_label_and_the_digest_of_a_ciphertext_are_claims(mc_fixed):
    # the points of the tag are inside t and c_ipfe; the fields are checked against each other and cannot be checked
    # against the points without the master key. Ciphertexts are not authenticated (module docstring).
    _, _, agg, cts, dk = mc_fixed
    for ct in cts.values():
        ct["label"] = encode_label("round-2")
    assert agg.decrypt(cts, dk) == EXPECTED
    assert agg.decrypt(cts, dk, label="round-2") == EXPECTED


def test_a_ciphertext_of_another_client_with_the_right_nid_gives_no_value(mc_fixed):
    _, clients, agg, cts, dk = mc_fixed
    forged = dict(cts["nid_1"], nid="nid_0")
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(dict(cts, nid_0=forged), dk)
    assert not isinstance(error.value, FEValidationError)


def test_decrypt_checks_the_bindings_before_anything_else(mc_fixed, make_mc, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    other_kg, other_clients, _ = make_mc()
    other_cts = encrypt_all(other_clients)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="ciphertext of nid_1.*pp_binding"):
        agg.decrypt(dict(cts, nid_1=other_cts["nid_1"]), dk)
    with pytest.raises(FEValidationError, match="functional key.*pp_binding"):
        agg.decrypt(cts, other_kg.get_decryption_keys("agg", credentials=creds()))
    no_binding = {nid: {k: v for k, v in ct.items() if k != "pp_binding"} for nid, ct in cts.items()}
    with pytest.raises(FEValidationError, match="no pp_binding"):
        agg.decrypt(no_binding, dk)


@pytest.mark.parametrize("field", MCKPABCiphertext.FIELDS)
def test_decrypt_refuses_a_ciphertext_without_a_field(field, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    del cts["nid_1"][field]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field", MCKPABFunctionalKey.FIELDS)
def test_decrypt_refuses_a_key_without_a_field(field, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    del dk[field]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,record", [("policy_compiler", "key"), ("policy_compiler", "ct"),
                                          ("attr_hash", "ct"), ("tag_hash", "ct")])
def test_decrypt_refuses_an_unknown_version_marker(field, record, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    (dk if record == "key" else cts["nid_2"])[field] = "v9"
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("bad", [None, [], "cts", 5, [("nid_0", {})]])
def test_decrypt_refuses_what_is_no_dict_of_ciphertexts(bad, mc_fixed):
    _, _, agg, _, dk = mc_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt(bad, dk)


@pytest.mark.parametrize("bad", [None, [], "key", 5])
def test_decrypt_refuses_what_is_no_key(bad, mc_fixed):
    _, _, agg, cts, _ = mc_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt(cts, bad)


@pytest.mark.parametrize("nid_key,match", [(1.5, "no client id"), (None, "no client id"), (True, "no client id"),
                                           (("nid_0",), "no client id"), (1, "not a client")])
def test_decrypt_refuses_keys_that_are_no_clients(nid_key, match, mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    with pytest.raises(FEValidationError, match=match):
        agg.decrypt({nid_key: cts["nid_0"], "nid_1": cts["nid_1"], "nid_2": cts["nid_2"]}, dk)


@pytest.mark.parametrize("field,shape", [
    ("c_attr", []), ("c_attr", [["1:x"] * 8]), ("c_attr", [["1:x"] * 7, ["1:x"] * 8]), ("c_attr", "x"),
    ("c_ipfe", ["1:x"] * 3), ("c_ipfe", "1:x"), ("t", ["1:x"]), ("t", 5), ("t", None),
])
def test_decrypt_refuses_a_ciphertext_of_the_wrong_shape(field, shape, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    cts["nid_2"][field] = shape
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,change", [
    ("y", lambda value: {k: v for k, v in value.items() if k != "nid_1"}),
    ("y", lambda value: dict(value, nid_9="1")),
    ("y", lambda value: dict(value, nid_1="11")),               # above bound_y
    ("y", lambda value: dict(value, nid_1=3)),
    ("y", lambda value: dict(value, nid_1="01")),
    ("y", lambda value: dict(value, nid_1="1" * 17)),
    ("y", lambda value: [value["nid_0"]]),
    ("k_rows", lambda value: {k: v for k, v in value.items() if k != "nid_2"}),
    ("k_rows", lambda value: dict(value, nid_2=value["nid_2"][:1])),
    ("k_rows", lambda value: dict(value, nid_2=[value["nid_2"][0], value["nid_2"][1][:7]])),
    ("k_rows", lambda value: dict(value, nid_2="rows")),
    ("k_ipfe", lambda value: dict(value, nid_0=value["nid_0"][:3])),
    ("k_ipfe", lambda value: {k: v for k, v in value.items() if k != "nid_0"}),
    ("k_ipfe", lambda value: [value["nid_0"]]),
])
def test_decrypt_refuses_a_key_of_the_wrong_shape(field, change, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    dk[field] = change(dk[field])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


def test_decrypt_decodes_every_vector_of_the_ciphertext_also_the_unused(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=ATTRS + ["org:A"])            # org:A is no row of the policy
    dk = kg.get_decryption_keys("agg", credentials=creds())
    assert agg.decrypt(cts, dk) == EXPECTED
    cts["nid_1"]["c_attr"][0][3] = "1:not an element"
    with pytest.raises(FEValidationError, match="c_attr"):
        agg.decrypt(cts, dk)
    cts["nid_1"]["c_attr"][0][3] = dk["k_ipfe"]["nid_0"][0]             # an element of G2
    with pytest.raises(FEValidationError, match="c_attr"):
        agg.decrypt(cts, dk)


def test_a_malformed_element_in_an_unused_key_row_is_not_decoded(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=["region:EU"])
    dk = kg.get_decryption_keys("agg", credentials=creds("or(region:EU, org:A)"))      # the first leaf is used
    dk["k_rows"]["nid_1"][1][2] = "2:not an element"
    assert agg.decrypt(cts, dk) == EXPECTED
    dk["k_rows"]["nid_1"][0][2] = "2:not an element"
    with pytest.raises(FEValidationError, match="k_rows"):
        agg.decrypt(cts, dk)


def test_decrypt_refuses_elements_of_the_wrong_group(mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    keys = copy.deepcopy(dk)
    keys["k_ipfe"]["nid_0"][1] = cts["nid_0"]["c_ipfe"][0]                          # G1 where G2 is due
    with pytest.raises(FEValidationError, match="k_ipfe"):
        agg.decrypt(cts, keys)
    cts["nid_0"]["t"] = dk["k_ipfe"]["nid_0"][0]                                    # G2 where G1 is due
    with pytest.raises(FEValidationError, match="t of nid_0"):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("text,reason", [
    ("and(", "parse|policy|canonical"), ("", "parse|policy|canonical"), ("a" * 100000, "characters|limit"),
    (canonical_encoding(parse("org:A")), "digest"),
])
def test_hostile_policy(text, reason, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    dk["policy"] = text
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=reason):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("policy", [None, 5, ["region:EU"], b"region:EU"])
def test_decrypt_refuses_a_policy_that_is_no_text(policy, mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    dk["policy"] = policy
    with pytest.raises(FEValidationError, match="policy"):
        agg.decrypt(cts, dk)


def test_a_key_with_a_swapped_ipfe_part_gives_no_value(mc_fixed):
    # the ipfe part of another client's slot: well formed, not for this client
    _, _, agg, cts, dk = mc_fixed
    dk["k_ipfe"]["nid_0"], dk["k_ipfe"]["nid_1"] = dk["k_ipfe"]["nid_1"], dk["k_ipfe"]["nid_0"]
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)


def test_a_key_with_the_weights_of_another_key_gives_no_value(mc_fixed):
    kg, _, agg, cts, dk = mc_fixed
    other = kg.get_decryption_keys("agg", credentials=creds(y=dict(Y, nid_1=5)))
    spliced = dict(dk, y=other["y"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)


def test_collusion_splice_of_rows_of_two_keys(mc_fixed):
    # the second row of another key for the same policy and weights: its share and its z are not those of this key
    kg, _, agg, cts, dk = mc_fixed
    donor = kg.get_decryption_keys("agg", credentials=creds())
    spliced = copy.deepcopy(dk)
    for nid in LST:
        spliced["k_rows"][nid][1] = donor["k_rows"][nid][1]
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)
    assert agg.decrypt(cts, donor) == EXPECTED


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
