"""The multi-client key-policy attribute-based IPFE with an attribute set for each client (NPP22 Sec. 5.4): what it
computes, setup and key generation, the fifth coordinate and the splice of two keys under one tag, the attack that
the fifth coordinate does not stop (two tags, two keys; the test is named for it), the records, what it refuses, the
helpers for arrays, and its status.

The scheme is known-insecure. App. B.5 of the paper states Theorem 25 for it and sketches the proof as changes to that
of Thm. 14; Thm. 25 as stated fails against the attack above, which these tests reproduce. The other tests check that
the scheme computes what Sec. 5.4 writes, that the objects have the shape the module documents, that the splice of
Sec. 5.4 (one tag) does not work on the keys, and the refusals it shares with ``mcfe/kp_ab_pairing`` (which has tests
of its own). They are no evidence of security. The scheme is kept for functionality and performance experiments and as
the record of the attack.

A key whose policy the attributes of one client do not satisfy gives a ``FEKeyError`` that names the client; malformed
input, ciphertexts that are not those of every client under one tag, or objects of another setup a
``FEValidationError``; a key and ciphertexts that are well formed but were not made for each other a ``ValueError`` from
the discrete log, which is not a ``FEValidationError``.
"""

import copy
import json
import logging
import os
import re
import warnings

import gmpy2 as gp
import numpy as np
import pytest

import pyfe4ai
from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.mcfe.conftest import EXPECTED, JUNK_G2, LST, X, Y
from tests.mcfe.conftest import aggregate, combined, creds, key_part, masked_plaintext, random_arrays
from tests.mcfe.conftest import refuse_to_compute, through_json
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FESecurityWarning, FEValidationError
from pyfe4ai.core.exceptions import FEWarning
from pyfe4ai.core.formats.pairing import MCKPABParameters, MCKPABRecord
from pyfe4ai.core.formats.pairing import PCKPABCiphertext, PCKPABClientKey, PCKPABFunctionalKey
from pyfe4ai.policy import ATTR_HASH, POLICY_COMPILER
from pyfe4ai.primitives.pairing import pairing_backend
from pyfe4ai.primitives.pairing.dpvs import vector_pair
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, ZR, PairingGroup, deserialize_many, hash_to_g1, pair
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt, random_nonzero_zr
from pyfe4ai.schemes.common import ac_codec
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.kp_ab import MAX_LABEL_CHARS, TAG_DST, TAG_HASH, attribute_vector, tag_points
from pyfe4ai.schemes.mcfe import kp_ab_pairing, kp_ab_pairing_per_client as per_client
from pyfe4ai.schemes.mcfe.kp_ab_pairing import MCFEKPABPairing
from pyfe4ai.schemes.mcfe.kp_ab_pairing_per_client import MCFEKPABPairingPerClient
from pyfe4ai.schemes.mcfe.kp_ab_pairing_per_client import MCFEKPABPairingPerClientKeyGenerator, _zero_sum_coordinates
from pyfe4ai.schemes.mcfe.labels import ElementLabel, element_label, encode_label

PP_DOMAIN = b"pyfe4ai/mcfe/kp-ab-per-client/pp-binding/v1|"
SCHEME_ID = "mcfe/kp_ab_pairing_per_client"

#: The policy of ``pc_fixed``, and the set of attributes of each client: three different ones, each satisfying it.
POLICY = "or(region:EU, region:US)"
SETS = {"nid_0": {"region:EU"}, "nid_1": {"region:US"}, "nid_2": {"region:EU", "org:A"}}

#: The scheme is known-insecure, so every key generator and crypto object of these tests warns; the test of the warning
#: records it (``pytest.warns`` sees it whatever the filters).
pytestmark = [
    pytest.mark.usefixtures("own_parameter_cache"),
    pytest.mark.filterwarnings("ignore:mcfe/kp_ab_pairing_per_client is known to be insecure:UserWarning"),
]


@pytest.fixture
def make_pc(own_parameter_cache):
    """``make_pc(n=3, **overrides)``: a key generator after ``setup()``, the crypto object of each client (by client
    id, with its encryption key) and an aggregator, which holds the public parameters only."""
    def build(n=3, **overrides):
        config = {"n": n, "bound_x": 10, "bound_y": 10, "pairing_group_param": GROUP}
        config.update(overrides)
        kg = MCFEKPABPairingPerClientKeyGenerator(config)
        kg.setup()
        pp = kg.get_public_parameters()
        clients = {nid: MCFEKPABPairingPerClient({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
                   for nid in kg.lst_nid}
        return kg, clients, MCFEKPABPairingPerClient({"keys": {"pp": pp}})
    return build


def encrypt_each(clients, sets=SETS, x=X, label="round-1"):
    """One ciphertext of ``x[nid]`` for every client of *sets*, under one tag and the client's own set."""
    return {nid: clients[nid].encrypt([x[nid]], label, attributes=sets[nid]) for nid in sets}


def encrypt_arrays_each(clients, arrays, sets=SETS, label="round-1"):
    """The ciphertext arrays of every client of *arrays*, each for the client's own set."""
    return {nid: clients[nid].encrypt_lst_ndarray(arrays[nid], label=label, attributes=sets[nid]) for nid in arrays}


@pytest.fixture
def pc_fixed(make_pc):
    """A key generator, the clients, an aggregator, the ciphertexts of ``X`` under ``round-1`` for ``SETS``, and a key
    for ``POLICY`` and ``Y``."""
    kg, clients, agg = make_pc()
    return kg, clients, agg, encrypt_each(clients), kg.get_decryption_keys("agg", credentials=creds(POLICY))


# ---------------------------------------------------------------------------
# what the scheme computes


def test_clients_with_their_own_attributes(make_pc):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients)
    assert [ct["attributes"] for ct in cts.values()] == [["region:EU"], ["region:US"], ["org:A", "region:EU"]]
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY))
    assert agg.decrypt(cts, dk) == EXPECTED


PER_CLIENT = [
    # a set of a different size for each client, all of which satisfy the policy
    ("and(purpose:research, region:EU)",
     [["purpose:research", "region:EU"], ["org:A", "purpose:research", "region:EU"],
      ["org:B", "purpose:research", "region:EU"]]),
    # each client satisfies it by another leaf
    ("or(region:EU, region:US)", [["region:EU"], ["region:US"], ["region:EU", "region:US"]]),
    # weights other than 1, and another pair of leaves for each client
    ("2of(purpose:research, region:EU, org:A)",
     [["org:A", "region:EU"], ["org:A", "purpose:research"], ["purpose:research", "region:EU"]]),
    # a repeated attribute in the policy
    ("or(and(region:EU, org:A), and(region:EU, org:B))",
     [["org:A", "region:EU"], ["org:B", "region:EU"], ["org:A", "org:B", "region:EU"]]),
]


@pytest.mark.parametrize("policy,sets", PER_CLIENT)
def test_aggregates(policy, sets, make_pc):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients, dict(zip(LST, sets)))
    dk = kg.get_decryption_keys("agg", credentials=creds(policy))
    assert agg.decrypt(cts, dk) == EXPECTED


def test_one_set_for_all_clients_works_too(make_pc):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients, {nid: ["purpose:research", "region:EU"] for nid in LST})
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds("and(purpose:research, region:EU)"))) \
        == EXPECTED


def test_every_client_can_aggregate(pc_fixed):
    _, clients, _, cts, dk = pc_fixed
    assert all(crypto.decrypt(cts, dk) == EXPECTED for crypto in clients.values())


def test_attribute_sets_in_any_form(make_pc):           # Review Focus 1
    kg, clients, agg = make_pc()
    spellings = {"nid_0": {"region:EU", "purpose:research"},
                 "nid_1": ["region:EU", "purpose:research", "region:EU"],
                 "nid_2": ["purpose:research", "region:EU"]}
    cts = encrypt_each(clients, spellings)
    assert all(ct["attributes"] == ["purpose:research", "region:EU"] for ct in cts.values())
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds("and(purpose:research, region:EU)"))) \
        == EXPECTED


def test_a_zero_weight_client_takes_part(make_pc):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients)
    y = dict(Y, nid_1=0)
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds(POLICY, y))) \
        == X["nid_0"] * 2 + X["nid_2"] * -4
    zero = {nid: 0 for nid in LST}
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds(POLICY, zero))) == 0


def test_one_key_serves_every_tag(make_pc):
    kg, clients, agg = make_pc()
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY))
    for label, x in (("round-1", X), ("round-2", {"nid_0": -1, "nid_1": 2, "nid_2": 3})):
        cts = encrypt_each(clients, x=x, label=label)
        assert agg.decrypt(cts, dk) == sum(x[nid] * Y[nid] for nid in LST)
        assert agg.decrypt(cts, dk, label=label) == sum(x[nid] * Y[nid] for nid in LST)


def test_extreme_inner_products(make_pc):
    kg, clients, agg = make_pc(bound_x=10, bound_y=10)
    for sign in (1, -1):
        x = {nid: sign * 10 for nid in LST}
        dk = kg.get_decryption_keys("agg", credentials=creds(POLICY, {nid: 10 for nid in LST}))
        assert agg.decrypt(encrypt_each(clients, x=x), dk) == sign * 300


def test_other_numbers_of_clients(make_pc):
    for n in (1, 2, 5):
        kg, clients, agg = make_pc(n=n)
        x = {nid: i + 1 for i, nid in enumerate(kg.lst_nid)}
        y = {nid: 3 - i for i, nid in enumerate(kg.lst_nid)}
        sets = {nid: ["org:A"] if i % 2 == 0 else ["org:B", "org:C"] for i, nid in enumerate(kg.lst_nid)}
        dk = kg.get_decryption_keys("agg", credentials=creds("or(org:A, org:B)", y))
        assert agg.decrypt(encrypt_each(clients, sets, x), dk) == sum(x[nid] * y[nid] for nid in x), n


def test_keys_of_one_master_key_all_open_one_set_of_ciphertexts(make_pc):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients)
    for policy, y in ((POLICY, Y), ("or(region:EU, org:A, region:US)", Y), ("or(region:US, region:EU)", Y),
                      (POLICY, {nid: 1 for nid in LST})):
        dk = kg.get_decryption_keys("agg", credentials=creds(policy, y))
        assert agg.decrypt(cts, dk) == sum(X[nid] * y[nid] for nid in LST)


def test_a_policy_that_only_some_clients_satisfy_is_refused_naming_the_first_that_does_not(make_pc):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients)
    dk = kg.get_decryption_keys("agg", credentials=creds("region:EU"))
    with pytest.raises(FEKeyError, match="ciphertext of nid_1"):
        agg.decrypt(cts, dk)


def test_each_client_has_its_own_root(make_pc):         # decision 1 of this plan, for the per-client scheme
    kg, _, _ = make_pc()
    group = PairingGroup(GROUP)
    order = int(group.order())
    ek = kg.get_private_keys("nid_0")
    f = [deserialize_many(group, row) for row in ek["f"]]
    dk = kg.get_decryption_keys("agg", credentials=creds("region:EU"))
    c = attribute_vector(f, "region:EU", 123456789, 1, order, group)
    paired = [vector_pair(c, deserialize_many(group, dk["k_rows"][nid][0])) for nid in LST]
    # [a_i,0 z]_T: the root a_i,0 of each client times the z of the key: three different values, where
    # mcfe/kp_ab_pairing has one root, and so one value, for all clients
    assert len({group.serialize(value) for value in paired}) == 3
    # and the root of a client sits in its ipfe part too: e(h_i,3, k*_i,ipfe) = [a_i,0 z]_T
    for nid, value in zip(LST, paired):
        h3 = deserialize_many(group, kg.get_private_keys(nid)["h3"])
        assert vector_pair(h3, deserialize_many(group, dk["k_ipfe"][nid])) == value
    # the vectors differ as well: every client and every row has its own pi
    assert len({tuple(dk["k_rows"][nid][0]) for nid in LST}) == 3


def test_a_row_takes_a_fresh_pi_for_each_client_and_each_occurrence(make_pc):
    kg, _, _ = make_pc()
    dk = kg.get_decryption_keys("agg", credentials=creds("or(and(region:EU, org:A), and(region:EU, org:B))"))
    rows = [tuple(row) for nid in LST for row in dk["k_rows"][nid]]
    assert len(rows) == 4 * 3 and len(set(rows)) == len(rows)


def test_two_keys_use_fresh_randomness(make_pc):
    kg, _, _ = make_pc()
    first = kg.get_decryption_keys("agg", credentials=creds())
    second = kg.get_decryption_keys("agg", credentials=creds())
    assert first["policy"] == second["policy"] and first["y"] == second["y"]
    assert all(first["k_ipfe"][nid] != second["k_ipfe"][nid] for nid in LST)
    assert all(first["k_rows"][nid] != second["k_rows"][nid] for nid in LST)


# ---------------------------------------------------------------------------
# setup and the keys


def test_bases_are_dual(make_pc):
    kg, _, _ = make_pc()
    group = PairingGroup(GROUP)
    g1 = group.deserialize(kg.get_public_parameters()["g1"].encode("utf-8"))
    g2 = group.deserialize(kg.get_public_parameters()["g2"].encode("utf-8"))
    gt = pair(g1, g2)
    f_star = kg.msk["f_star"]
    for nid in LST:
        ek = kg.get_private_keys(nid)
        f = [deserialize_many(group, row) for row in ek["f"]]
        h3 = deserialize_many(group, ek["h3"])
        theta_h5 = deserialize_many(group, ek["theta_h5"])
        theta = int(kg.msk["theta"][nid])
        for i in range(3):
            for j in range(3):
                assert vector_pair(f[i], f_star[j]) == (gt if i == j else gt ** 0), (nid, i, j)
        h_star = kg.msk["h_star"]                       # h*_1, h*_2, h*_3 and h*_5 of each client
        assert len(h_star[nid]) == 4
        assert [vector_pair(h3, vector) for vector in h_star[nid]] == [gt ** 0, gt ** 0, gt, gt ** 0]
        # theta h_i,5 against h*_i,1, h*_i,2, h*_i,3 and h*_i,5: theta where the coordinate is the fifth
        assert [vector_pair(theta_h5, vector) for vector in h_star[nid]] == [gt ** 0, gt ** 0, gt ** 0, gt ** theta]


def test_master_key_holds_group_elements(make_pc):      # D-2 and decision 2 of this plan
    kg, _, _ = make_pc()
    assert set(kg.msk) == {"s", "u", "theta", "f_star", "h_star"}
    assert set(kg.msk["s"]) == set(kg.msk["u"]) == set(kg.msk["theta"]) == set(kg.msk["h_star"]) == set(LST)
    assert len(kg.msk["f_star"]) == 3 and all(len(row) == 8 for row in kg.msk["f_star"])
    assert all(element.type == G2 for row in kg.msk["f_star"] for element in row)
    for nid in LST:         # h*_i,1, h*_i,2, h*_i,3 as in the paper's list, and h*_i,5, which the list leaves out
        assert len(kg.msk["h_star"][nid]) == 4 and all(len(row) == 5 for row in kg.msk["h_star"][nid])
        assert all(element.type == G2 for row in kg.msk["h_star"][nid] for element in row)
    order = int(PairingGroup(GROUP).order())
    for value in [*kg.msk["s"].values(), *kg.msk["u"].values(), *kg.msk["theta"].values()]:
        assert 0 < int(value) < order                   # drawn from Z_q^*: the last d_i can be solved for


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


def test_setup_keeps_no_basis(make_pc, monkeypatch):
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
    monkeypatch.setattr(per_client, "random_nonzero_zr", spy_zr)
    monkeypatch.setattr(kp_ab_pairing, "random_dual_bases", spy_bases)
    kg, _, _ = make_pc()
    assert [len(m) for m in matrices] == [8, 8] + [5, 5] * 3        # F, F*, then H_i, H_i* of each client
    secrets = {int(value) for name in ("s", "u", "theta") for value in kg.msk[name].values()}
    assert set(drawn) == secrets and len(drawn) == 3 * 3            # S, U and the thetas, and no mu, no z
    kept = reachable_scalars(kg)
    assert secrets <= kept
    entries = {int(entry) for matrix in matrices for row in matrix for entry in row}
    assert not entries & kept                                       # the exponents of the bases are gone


def test_the_shares_of_one_add_up_to_one(pc_fixed):
    # sum_i p_i = 1 makes the mask of the clients cancel: prod_i e(c_i,ipfe, h*_i,1) = e(omega, g2), and the same with
    # omega' and h*_i,2, the fifth coordinate of c_i,ipfe being orthogonal to both. omega and omega' are the points of
    # the tag, which the test hashes for itself.
    kg, _, _, cts, _ = pc_fixed
    group = PairingGroup(GROUP)
    pp = kg.get_public_parameters()
    g2 = group.deserialize(pp["g2"].encode("utf-8"))
    omega, omega_prime = tag_points(group, SCHEME_ID, pp["pp_binding"], encode_label("round-1"), None)
    first, second = pair(omega, g2) ** 0, pair(omega_prime, g2) ** 0
    for nid in LST:
        c_ipfe = deserialize_many(group, cts[nid]["c_ipfe"])
        first *= vector_pair(c_ipfe, kg.msk["h_star"][nid][0])
        second *= vector_pair(c_ipfe, kg.msk["h_star"][nid][1])
    assert first == pair(omega, g2) and second == pair(omega_prime, g2)


def test_the_ciphertext_has_theta_in_its_fifth_coordinate(pc_fixed):
    # c_i,ipfe = (omega p_i, omega' p_i, psi_i, 0, theta_i)_H_i: against h*_i,5 it gives theta_i, against h*_i,3 the
    # psi_i of the ciphertext, which also stands in c_attr
    kg, _, _, cts, _ = pc_fixed
    group = PairingGroup(GROUP)
    pp = kg.get_public_parameters()
    gt = pair(group.deserialize(pp["g1"].encode("utf-8")), group.deserialize(pp["g2"].encode("utf-8")))
    for nid in LST:
        c_ipfe = deserialize_many(group, cts[nid]["c_ipfe"])
        assert len(c_ipfe) == 5
        assert vector_pair(c_ipfe, kg.msk["h_star"][nid][3]) == gt ** int(kg.msk["theta"][nid])


def test_the_masked_plaintext_is_in_the_exponent(pc_fixed):
    # t_i = omega s_i + omega' u_i + x_i: e(t_i, g2) = e(omega, g2)^s_i e(omega', g2)^u_i e(g1, g2)^x_i, with the
    # points of the tag alone: the attributes of the client are not hashed
    kg, _, _, cts, _ = pc_fixed
    group = PairingGroup(GROUP)
    order = int(group.order())
    pp = kg.get_public_parameters()
    g1, g2 = (group.deserialize(pp[name].encode("utf-8")) for name in ("g1", "g2"))
    omega, omega_prime = tag_points(group, SCHEME_ID, pp["pp_binding"], encode_label("round-1"), None)
    for nid in LST:
        t = group.deserialize(cts[nid]["t"].encode("utf-8"))
        assert pair(t, g2) == (pair(omega, g2) ** int(kg.msk["s"][nid]) * pair(omega_prime, g2) ** int(kg.msk["u"][nid])
                               * pair(g1, g2) ** (X[nid] % order))


def test_the_tag_does_not_bind_the_attribute_set(make_pc):
    kg, clients, _ = make_pc()
    one = clients["nid_0"].encrypt([4], "round-1", attributes=["region:EU"])
    other = clients["nid_0"].encrypt([4], "round-1", attributes=["org:A", "region:US"])
    again = clients["nid_0"].encrypt([4], "round-2", attributes=["region:EU"])
    assert one["t"] == other["t"] != again["t"]             # t is deterministic, and the tag alone sets it
    assert one["c_attr"] != other["c_attr"] and one["c_ipfe"] != other["c_ipfe"]        # the rest is fresh


def test_a_second_ciphertext_under_a_tag_gives_away_the_difference(make_pc, make_mc):
    # Why a client must encrypt once under a tag, and here the only safeguard: the points of the tag do not depend on
    # the attributes, so any two ciphertexts of one client under one tag, for whatever sets, have t_1 / t_2 =
    # g1^(x_1 - x_2). In mcfe/kp_ab_pairing the points depend on the set, and this ratio holds for one set only.
    kg, clients, _ = make_pc()
    group = PairingGroup(GROUP)
    order = int(group.order())
    g1 = group.deserialize(kg.get_public_parameters()["g1"].encode("utf-8"))
    (x_1, x_2) = (7, -2)
    ciphertexts = [clients["nid_0"].encrypt([x], "round-1", attributes=attributes)
                   for x, attributes in ((x_1, ["region:EU"]), (x_2, ["org:A", "region:US"]))]
    t_1, t_2 = (group.deserialize(ct["t"].encode("utf-8")) for ct in ciphertexts)
    assert t_1 / t_2 == g1 ** ((x_1 - x_2) % order)

    kg_n2, clients_n2, _ = make_mc()
    g1_n2 = group.deserialize(kg_n2.get_public_parameters()["g1"].encode("utf-8"))
    ciphertexts = [clients_n2["nid_0"].encrypt([x], "round-1", attributes=attributes)
                   for x, attributes in ((x_1, ["region:EU"]), (x_2, ["region:US"]))]
    t_1, t_2 = (group.deserialize(ct["t"].encode("utf-8")) for ct in ciphertexts)
    assert t_1 / t_2 != g1_n2 ** ((x_1 - x_2) % order)
    ciphertexts = [clients_n2["nid_0"].encrypt([x], "round-1", attributes=["region:EU"]) for x in (x_1, x_2)]
    t_1, t_2 = (group.deserialize(ct["t"].encode("utf-8")) for ct in ciphertexts)
    assert t_1 / t_2 == g1_n2 ** ((x_1 - x_2) % order)


def test_client_key_fields(make_pc):                    # nothing of another client in ek_i
    kg, _, _ = make_pc()
    keys = {nid: kg.get_private_keys(nid) for nid in LST}
    g1_tag = "{}:".format(G1)
    for nid, ek in keys.items():
        assert list(ek) == list(PCKPABClientKey.FIELDS)
        assert ek["nid"] == nid and ek["pp_binding"] == kg.get_public_parameters()["pp_binding"]
        assert len(ek["p_h1"]) == len(ek["p_h2"]) == len(ek["h3"]) == len(ek["theta_h5"]) == 5
        assert [len(row) for row in ek["f"]] == [8] * 3
        assert all(value.isdigit() for value in [ek["s"], ek["u"], *ek["p_h1"], *ek["p_h2"]])
        assert all(item.startswith(g1_tag) for item in [*ek["h3"], *ek["theta_h5"],
                                                         *(item for row in ek["f"] for item in row)])
        assert ek["s"] == kg.msk["s"][nid] and ek["u"] == kg.msk["u"][nid]
        assert kg.msk["theta"][nid] not in ek.values()          # theta_i h_i,5 is in the key, theta_i is not
    own = {nid: {*keys[nid]["p_h1"], *keys[nid]["p_h2"], *keys[nid]["h3"], *keys[nid]["theta_h5"],
                 keys[nid]["s"], keys[nid]["u"]} for nid in LST}
    for a in LST:
        for b in LST:
            if a != b:
                assert not own[a] & own[b]                      # no value of the client b in the key of a
    assert keys["nid_0"]["f"] == keys["nid_1"]["f"] == keys["nid_2"]["f"]       # F is shared by all clients


def test_a_client_key_is_a_copy(make_pc):
    kg, _, _ = make_pc()
    ek = kg.get_private_keys("nid_1")
    ek["s"] = "0"
    ek["theta_h5"].clear()
    assert kg.get_private_keys("nid_1")["s"] == kg.msk["s"]["nid_1"]
    assert len(kg.get_private_keys("nid_1")["theta_h5"]) == 5


def test_unknown_clients_have_no_key(make_pc):
    kg, _, _ = make_pc()
    assert [kg.get_private_keys(nid) for nid in ("nobody", 1, None, ["nid_0"], 1.0, True)] == [None] * 6
    kg, _, _ = make_pc(lst_nid=[1, 2, 3])
    assert kg.get_private_keys(1.0) is None and kg.get_private_keys(True) is None and kg.get_private_keys(4) is None
    assert kg.get_private_keys(np.int64(2)) == kg.get_private_keys(2)


def test_public_parameters_layout(make_pc):
    kg, _, _ = make_pc()
    pp = kg.get_public_parameters()
    assert list(pp) == list(MCKPABParameters.FIELDS)        # the parameters of mcfe/kp_ab_pairing, unchanged
    assert (pp["n"], pp["lst_nid"], pp["eta"]) == (3, LST, 1)
    assert (pp["bound_x"], pp["bound_y"], pp["max_rows"], pp["max_attributes"]) == ("10", "10", 64, 64)
    assert (pp["policy_compiler"], pp["attr_hash"], pp["tag_hash"]) == (POLICY_COMPILER, ATTR_HASH, TAG_HASH)
    assert pp["pp_binding"] == pp_digest(pp, PP_DOMAIN)
    assert pp["pp_binding"] != pp_digest(pp, b"pyfe4ai/mcfe/kp-ab/pp-binding/v1|")      # not the other scheme's
    assert not {"s", "u", "theta", "p_h1", "f", "h3", "theta_h5"} & set(pp)


def test_public_parameters_hold_no_secret(make_pc):
    kg, _, _ = make_pc()
    secrets = reachable_scalars(kg) - {3, 10, 64}
    text = json.dumps(kg.get_public_parameters())
    assert not [value for value in secrets if value > 1000 and str(value) in text]


def test_setups_differ(make_pc):
    first, _, _ = make_pc()
    second, _, _ = make_pc()
    assert first.get_public_parameters()["pp_binding"] != second.get_public_parameters()["pp_binding"]


def test_integer_client_ids(make_pc):
    kg, clients, agg = make_pc(lst_nid=[10, 20, 30])
    x, y = {10: 1, 20: 2, 30: 3}, {10: 4, 20: -5, 30: 6}
    sets = {10: ["region:EU"], 20: ["region:US"], 30: ["region:EU", "org:A"]}
    cts = encrypt_each(clients, sets, x)
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY, y))
    assert agg.decrypt(cts, dk) == 4 - 10 + 18
    assert list(dk["y"]) == [10, 20, 30] and cts[20]["nid"] == 20 and kg.get_private_keys(20)["nid"] == 20
    assert agg.decrypt({np.int64(nid): ct for nid, ct in cts.items()}, dk) == 4 - 10 + 18     # numpy ids are ints
    # JSON turns the keys of the dicts into strings: refused before any pairing, naming the clients
    with pytest.raises(FEValidationError, match="10"):
        agg.decrypt(cts, through_json(dk))


def test_objects_survive_json(pc_fixed):
    kg, clients, _, cts, dk = pc_fixed
    pp = through_json(kg.get_public_parameters())
    ek = through_json(kg.get_private_keys("nid_2"))
    agg = MCFEKPABPairingPerClient({"keys": {"pp": pp}})
    assert agg.decrypt(through_json(cts), through_json(dk)) == EXPECTED
    client = MCFEKPABPairingPerClient({"id": "nid_2", "keys": {"pp": pp, "sk": ek}})
    assert client.encrypt([X["nid_2"]], "round-1", attributes=SETS["nid_2"])["t"] == cts["nid_2"]["t"]


# ---------------------------------------------------------------------------
# the fifth coordinate: the d_i of a key, and the splice of two keys


def test_zero_sum_of_d():
    class Drawn:
        """A group whose random elements of Z_q are the given ints, in order."""
        def __init__(self, *values):
            self.values = iter(values)

        def random(self, kind):
            return next(self.values)

    order = 101
    # the first n - 1 are the draws, the last solves sum theta_i d_i = 0: 3*10 + 5*20 + 7*d = 0 mod 101
    d = _zero_sum_coordinates([3, 5, 7], order, Drawn(10, 20))
    assert d[:2] == [10, 20] and 0 <= d[2] < order
    assert (3 * d[0] + 5 * d[1] + 7 * d[2]) % order == 0
    assert _zero_sum_coordinates([9], order, Drawn()) == [0]           # one client: theta d = 0 leaves only d = 0
    assert _zero_sum_coordinates([3, 5], order, Drawn(0)) == [0, 0]
    # the real group, many sizes and thetas
    group = PairingGroup(GROUP)
    q = int(group.order())
    for n in (1, 2, 3, 7, 16):
        thetas = [int(random_nonzero_zr(group)) for _ in range(n)]
        d = _zero_sum_coordinates(thetas, q, group)
        assert len(d) == n and all(0 <= value < q for value in d)
        assert sum(theta * value for theta, value in zip(thetas, d)) % q == 0
        if n > 1:
            assert any(d) and d != _zero_sum_coordinates(thetas, q, group)      # fresh for each key


def test_zero_sum_of_d_needs_nonzero_thetas():
    group = PairingGroup(GROUP)
    q = int(group.order())
    for thetas in ([1, 0, 3], [0], [q, 2], []):
        with pytest.raises(FEValidationError, match="theta"):
            _zero_sum_coordinates(thetas, q, group)


def test_the_fifth_coordinates_of_a_key_cancel(make_pc):
    # theta_i h_i,5 is in the key of client i: e(theta_i h_i,5, k*_i,ipfe) = [theta_i d_i]_T, whatever the others
    kg, _, _ = make_pc()
    group = PairingGroup(GROUP)
    pp = kg.get_public_parameters()
    gt = pair(group.deserialize(pp["g1"].encode("utf-8")), group.deserialize(pp["g2"].encode("utf-8")))
    theta_h5 = {nid: deserialize_many(group, kg.get_private_keys(nid)["theta_h5"]) for nid in LST}
    keys = [kg.get_decryption_keys("agg", credentials=creds()) for _ in range(2)]
    terms = [{nid: vector_pair(theta_h5[nid], deserialize_many(group, dk["k_ipfe"][nid])) for nid in LST}
             for dk in keys]
    for one in terms:
        total = gt ** 0
        for nid in LST:
            total *= one[nid]
        assert total == gt ** 0                                 # sum theta_i d_i = 0
        assert all(one[nid] != gt ** 0 for nid in LST)          # and none of them is 0
        assert one["nid_0"] * one["nid_1"] != gt ** 0           # not for a subset of the clients
    assert all(terms[0][nid] != terms[1][nid] for nid in LST)   # independent for each key


def test_splice_of_two_keys_fails(make_pc):
    # NPP22 Sec. 5.4, under ONE tag. Two clients encrypt for the sets S_1 = {p} and S_2 = {q}; the key A is for the
    # policy p (client 1 satisfies it, client 2 does not) and the key B for q, both for the same y. Client 1 takes its
    # part from A and client 2 from B, each of which opens its own policy. Without the fifth coordinate the two parts
    # would add up to <x, y>: the paper's mix-and-match attack. With it they leave -theta_1 d_A,1 - theta_2 d_B,2.
    # That term does not depend on the tag: see test_two_tags_and_two_keys_read_the_difference_of_two_rounds.
    kg, clients, agg = make_pc(n=2)
    group = PairingGroup(GROUP)
    order = int(group.order())
    nids = kg.lst_nid
    pp = kg.get_public_parameters()
    g1, g2 = (group.deserialize(pp[name].encode("utf-8")) for name in ("g1", "g2"))
    gt = pair(g1, g2)
    x, y = {"nid_0": 5, "nid_1": -3}, {"nid_0": 2, "nid_1": 3}
    target = gt ** (sum(x[nid] * y[nid] for nid in nids) % order)
    sets = {"nid_0": ["org:p"], "nid_1": ["org:q"]}
    cts = encrypt_each(clients, sets, x)

    def masked(nid):
        """e(t_i, [y_i]_2): the masked plaintext against the weight of the client."""
        return pair(group.deserialize(cts[nid]["t"].encode("utf-8")), g2 ** (y[nid] % order))

    def total(parts):
        value = gt ** 0
        for nid, part in zip(nids, parts):
            value = value * part * masked(nid)
        return value

    # the positive control: one key for a policy that both clients satisfy, each by its own attribute. The sum of the
    # parts is <x, y>, so the arithmetic of this test is that of the scheme
    both = "or(org:p, org:q)"
    dk = kg.get_decryption_keys("agg", credentials=creds(both, y))
    assert agg.decrypt(cts, dk) == sum(x[nid] * y[nid] for nid in nids)
    assert total([key_part(group, cts[nid], dk, nid, both) for nid in nids]) == target

    key_a = kg.get_decryption_keys("agg", credentials=creds("org:p", y))
    key_b = kg.get_decryption_keys("agg", credentials=creds("org:q", y))
    spliced = total([key_part(group, cts["nid_0"], key_a, "nid_0", "org:p"),
                     key_part(group, cts["nid_1"], key_b, "nid_1", "org:q")])
    assert spliced != target

    # and what is in the way is exactly the fifth coordinate: [theta_i d_i]_T of the two parts
    theta_h5 = {nid: deserialize_many(group, kg.get_private_keys(nid)["theta_h5"]) for nid in nids}
    obstacle = (vector_pair(theta_h5["nid_0"], deserialize_many(group, key_a["k_ipfe"]["nid_0"]))
                * vector_pair(theta_h5["nid_1"], deserialize_many(group, key_b["k_ipfe"]["nid_1"])))
    assert obstacle != gt ** 0
    assert spliced * obstacle == target


@pytest.mark.parametrize("rounds", ["two labels", "two elements of one array"])
def test_two_tags_and_two_keys_read_the_difference_of_two_rounds(rounds, make_pc):
    """THE ATTACK THAT THE FIFTH COORDINATE DOES NOT STOP (why the scheme is known-insecure).

    ``theta_i d_{K,i}`` is a function of the key ``K`` and the client, not of the tag. So for two keys A and B of one
    ``y``, where the set of each client satisfies one of them in both rounds, the part ``V_i(tag)`` of a client (its
    attribute rows, minus ``c_ipfe x k_ipfe``, plus ``e(t_i, [y_i]_2)``) has the same obstacle ``theta_i d_{K,i}``
    under every tag, and ``sum_i (V_i(tag) - V_i(tag'))`` is ``[<x_tag - x_tag', y>]_T``. Neither key opens either round
    (``FEKeyError``). Remark 10 and the game of Fig. 2 let the adversary encrypt under a tag other than the challenge
    tag, and Def. 9 calls it admissible (``Rel = 0`` for both keys and both rounds, so its condition 3 does not hold),
    so Thm. 25 as stated fails. The two tags are two labels, or two elements of one array, which the helpers encrypt
    under a tag each. The parts are computed from the fields of the records alone.
    """
    kg, clients, agg = make_pc(n=2)
    group = PairingGroup(GROUP)
    order = int(group.order())
    nids = kg.lst_nid
    pp = kg.get_public_parameters()
    g1, g2 = (group.deserialize(pp[name].encode("utf-8")) for name in ("g1", "g2"))
    gt = pair(g1, g2)
    sets = {"nid_0": ["org:p"], "nid_1": ["org:q"]}
    y = {"nid_0": 2, "nid_1": 3}
    x_1, x_2 = {"nid_0": 5, "nid_1": -3}, {"nid_0": 1, "nid_1": 7}      # the two rounds' plaintexts
    if rounds == "two labels":
        cts_1 = encrypt_each(clients, sets, x_1, "round-1")
        cts_2 = encrypt_each(clients, sets, x_2, "round-2")
    else:
        arrays = {nid: [np.array([x_1[nid], x_2[nid]])] for nid in nids}
        elements = encrypt_arrays_each(clients, arrays, sets)
        cts_1 = {nid: elements[nid][0][0] for nid in nids}
        cts_2 = {nid: elements[nid][0][1] for nid in nids}

    key_a = kg.get_decryption_keys("agg", credentials=creds("org:p", y))
    key_b = kg.get_decryption_keys("agg", credentials=creds("org:q", y))
    # each key refuses both rounds through the scheme: client 1 does not satisfy A, client 0 does not satisfy B
    for cts in (cts_1, cts_2):
        for dk in (key_a, key_b):
            with pytest.raises(FEKeyError, match="do not satisfy the policy"):
                agg.decrypt(cts, dk)

    parts = {"nid_0": (key_a, "org:p"), "nid_1": (key_b, "org:q")}      # each client: the key that it satisfies

    def spliced(cts):
        """``prod_i V_i``: the parts of the clients, each from its own key, with the masked plaintexts."""
        value = gt ** 0
        for nid in nids:
            dk, policy = parts[nid]
            part = key_part(group, cts[nid], dk, nid, policy)
            value = value * part * masked_plaintext(group, cts[nid], y[nid], g2)
        return value

    inner_1 = sum(x_1[nid] * y[nid] for nid in nids)
    inner_2 = sum(x_2[nid] * y[nid] for nid in nids)
    bound = len(nids) * 10 * 10
    # under one tag the splice gives no value (the obstacle theta_0 d_A,0 + theta_1 d_B,1 is a random exponent) ...
    for round_value, inner in ((spliced(cts_1), inner_1), (spliced(cts_2), inner_2)):
        assert round_value != gt ** (inner % order)
        with pytest.raises(ValueError, match="discrete log"):
            bounded_discrete_log_gt(group, gt, round_value, bound)
    # ... and under two tags the obstacle is the same, so the quotient is the difference of the two inner products
    quotient = spliced(cts_1) / spliced(cts_2)
    assert quotient == gt ** ((inner_1 - inner_2) % order)
    assert bounded_discrete_log_gt(group, gt, quotient, bound) == inner_1 - inner_2 == -22
    assert inner_1 - inner_2 not in (inner_1, inner_2, 0)


def test_splice_of_keys_of_one_function_fails(pc_fixed):
    # Not only keys of two policies: under one tag the fifth coordinate stops the splice of two keys for the same policy
    # and the same y as well (in mcfe/kp_ab_pairing that splice gives the value of the function, no more). Client 0
    # takes both its parts from the first key, the others theirs from the second. Across tags it stops nothing: see
    # test_two_tags_and_two_keys_read_the_difference_of_two_rounds.
    kg, _, agg, cts, dk = pc_fixed
    other = kg.get_decryption_keys("agg", credentials=creds(POLICY))
    assert other["k_rows"] != dk["k_rows"] and other["k_ipfe"] != dk["k_ipfe"]
    assert agg.decrypt(cts, other) == EXPECTED
    spliced = copy.deepcopy(other)
    spliced["k_rows"]["nid_0"] = copy.deepcopy(dk["k_rows"]["nid_0"])
    spliced["k_ipfe"]["nid_0"] = copy.deepcopy(dk["k_ipfe"]["nid_0"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)
    assert agg.decrypt(cts, dk) == EXPECTED


def test_a_client_must_not_mix_the_rows_and_the_ipfe_part_of_two_keys(pc_fixed):
    kg, _, agg, cts, dk = pc_fixed
    other = kg.get_decryption_keys("agg", credentials=creds(POLICY))
    mixed = copy.deepcopy(other)
    mixed["k_ipfe"]["nid_1"] = copy.deepcopy(dk["k_ipfe"]["nid_1"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, mixed)
    assert not isinstance(error.value, FEValidationError)


def test_collusion_splice_of_rows_of_two_keys(pc_fixed):
    kg, _, agg, cts, dk = pc_fixed
    donor = kg.get_decryption_keys("agg", credentials=creds("or(region:EU, region:US)"))
    spliced = copy.deepcopy(dk)
    spliced["k_rows"]["nid_2"] = donor["k_rows"]["nid_2"]
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)
    assert agg.decrypt(cts, donor) == EXPECTED


def test_key_splice(pc_fixed):
    # two keys for one policy and different weights; client 0 takes its parts of the first, the others theirs of the
    # second, and y is the first key's: <S, y> and <U, y> in the parts of the clients are not those of one y
    kg, _, agg, cts, dk = pc_fixed
    other_y = {"nid_0": 1, "nid_1": 2, "nid_2": 3}
    other = kg.get_decryption_keys("agg", credentials=creds(POLICY, other_y))
    assert other["y"] != dk["y"] and other["policy"] == dk["policy"]
    spliced = copy.deepcopy(other)
    spliced["y"] = copy.deepcopy(dk["y"])
    spliced["k_rows"]["nid_0"] = copy.deepcopy(dk["k_rows"]["nid_0"])
    spliced["k_ipfe"]["nid_0"] = copy.deepcopy(dk["k_ipfe"]["nid_0"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)
    assert agg.decrypt(cts, dk) == EXPECTED
    assert agg.decrypt(cts, other) == sum(X[nid] * other_y[nid] for nid in LST)


def test_a_ciphertext_of_another_client_with_the_right_nid_gives_no_value(pc_fixed):
    _, _, agg, cts, dk = pc_fixed
    forged = dict(cts, nid_1=dict(cts["nid_2"], nid="nid_1"))
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(forged, dk)
    assert not isinstance(error.value, FEValidationError)


def test_a_client_that_hashed_other_points_than_it_says_gives_no_value(pc_fixed):       # Review Focus 3
    _, clients, agg, cts, dk = pc_fixed
    cts["nid_1"] = clients["nid_1"].encrypt([X["nid_1"]], "round-2", attributes=SETS["nid_1"])
    cts["nid_1"]["label"] = cts["nid_0"]["label"]           # it says round-1, the points are those of round-2
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)
    cts["nid_1"]["label"] = clients["nid_1"].encrypt([1], "round-2", attributes=[])["label"]
    with pytest.raises(FEValidationError, match="differ in their label"):
        agg.decrypt(cts, dk)


def test_the_label_of_a_ciphertext_is_a_claim(pc_fixed):
    # all clients change the label alike: the value is still the one of round-1, since the label is not what the
    # ciphertext is made of (the same as in mcfe/kp_ab_pairing)
    _, _, agg, cts, dk = pc_fixed
    relabelled = {nid: dict(ct, label=encode_label("round-2")) for nid, ct in cts.items()}
    assert agg.decrypt(relabelled, dk) == EXPECTED
    with pytest.raises(FEValidationError, match="claim the label"):
        agg.decrypt(cts, dk, label="round-2")


def test_attributes_changed_by_one_client_give_no_value_and_never_a_wrong_one(make_pc):
    # The list of attributes says which vector of c_attr pairs with which row of the key. A client whose list names
    # another label that the policy reads gets a value that does not decrypt; a label that no row reads can be renamed
    # without effect (as in mcfe/kp_ab_pairing).
    kg, clients, agg = make_pc()
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY))
    cts = encrypt_each(clients, {nid: ["region:EU"] for nid in LST})
    assert agg.decrypt(cts, dk) == EXPECTED
    renamed = dict(cts, nid_1=dict(cts["nid_1"], attributes=["region:US"]))
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(renamed, dk)
    assert not isinstance(error.value, FEValidationError)
    cts = encrypt_each(clients, {nid: ["org:A", "region:EU"] for nid in LST})
    assert agg.decrypt(dict(cts, nid_1=dict(cts["nid_1"], attributes=["org:B", "region:EU"])), dk) == EXPECTED


# ---------------------------------------------------------------------------
# the one client whose attributes do not satisfy the policy


def test_one_unsatisfied_client_refuses(make_pc, monkeypatch):
    kg, clients, agg = make_pc()
    sets = dict(SETS, nid_2=["org:A"])                      # satisfies nothing
    cts = encrypt_each(clients, sets)
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY))
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match=r"ciphertext of nid_2 do not satisfy the policy") as error:
        agg.decrypt(cts, dk)
    assert "org:A" in str(error.value)
    # the first of several is named
    cts = encrypt_each(clients, dict(SETS, nid_1=[], nid_2=["org:A"]))
    with pytest.raises(FEKeyError, match=r"ciphertext of nid_1 do not satisfy"):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("policy,sets,named", [
    ("and(purpose:research, region:EU)", [["purpose:research", "region:EU"], ["region:EU"], ["purpose:research"]],
     "nid_1"),
    ("2of(a:1, b:1, c:1)", [["a:1", "b:1"], ["a:1", "c:1"], ["c:1"]], "nid_2"),
    ("a:1", [[], ["a:1"], ["a:1"]], "nid_0"),
])
def test_an_unsatisfied_client_is_refused_before_any_pairing(policy, sets, named, make_pc, monkeypatch):
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients, dict(zip(LST, sets)))
    dk = kg.get_decryption_keys("agg", credentials=creds(policy))
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match="ciphertext of {} do not satisfy".format(named)):
        agg.decrypt(cts, dk)


# ---------------------------------------------------------------------------
# the key's rows differ between clients: which of them a decryption reads


def test_each_client_reads_the_rows_its_own_attributes_use(make_pc):
    kg, clients, agg = make_pc(n=2)
    cts = encrypt_each(clients, {"nid_0": ["a:1"], "nid_1": ["b:1"]}, {"nid_0": 3, "nid_1": 4})
    dk = kg.get_decryption_keys("agg", credentials=creds("or(a:1, b:1)", {"nid_0": 2, "nid_1": 5}))
    assert agg.decrypt(cts, dk) == 3 * 2 + 4 * 5
    # client 0 reads row 0 only and client 1 row 1 only: a row nobody reads is not decoded, one that is read is
    dk["k_rows"]["nid_0"][1][2] = JUNK_G2
    dk["k_rows"]["nid_1"][0][2] = JUNK_G2
    assert agg.decrypt(cts, dk) == 3 * 2 + 4 * 5
    dk["k_rows"]["nid_1"][1][2] = JUNK_G2
    with pytest.raises(FEValidationError, match="k_rows of nid_1"):
        agg.decrypt(cts, dk)


def test_multi_pair_is_called_once_with_the_pairs_of_each_client_and_one_more(make_pc, monkeypatch):
    kg, clients, agg = make_pc()
    calls = []
    real = pairing_backend.multi_pair

    def spy(lhs, rhs):
        calls.append(len(lhs))
        return real(lhs, rhs)

    monkeypatch.setattr(kp_ab_pairing, "multi_pair", spy)
    sets = {"nid_0": ["a:1", "b:1"], "nid_1": ["c:1", "d:1", "e:1"], "nid_2": ["a:1", "b:1", "c:1"]}
    dk = kg.get_decryption_keys("agg", credentials=creds("or(and(a:1, b:1), c:1)"))
    assert agg.decrypt(encrypt_each(clients, sets), dk) == EXPECTED
    # the rows of the first satisfied child: two for clients 0 and 2, one for client 1; five pairs of c_ipfe and k_ipfe
    assert calls == [(8 * 2 + 5) + (8 * 1 + 5) + (8 * 2 + 5) + 1]


def test_the_generator_pair_is_computed_once(pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    calls = []
    real = pairing_backend.pair
    from pyfe4ai.schemes.common import ac_decoded
    monkeypatch.setattr(ac_decoded, "pair", lambda a, b: calls.append(1) or real(a, b))
    for _ in range(3):
        assert agg.decrypt(cts, dk) == EXPECTED
    assert calls == [1]


def test_encrypt_decodes_the_key_once(make_pc, monkeypatch):
    kg, clients, _ = make_pc()
    calls = []
    from pyfe4ai.schemes.common import ac_decoded
    real = ac_decoded.read_elements
    monkeypatch.setattr(ac_decoded, "read_elements", lambda *args: calls.append(args[3]) or real(*args))
    for _ in range(3):
        clients["nid_0"].encrypt([1], "round-1", attributes=["org:A"])
    assert sorted(calls) == ["f", "f", "f", "g1", "h3", "theta_h5"]


# ---------------------------------------------------------------------------
# the records


def test_functional_key_layout(pc_fixed):
    _, _, _, _, dk = pc_fixed
    group = PairingGroup(GROUP)
    assert list(dk) == list(PCKPABFunctionalKey.FIELDS)
    assert dk["policy_compiler"] == POLICY_COMPILER
    assert dk["y"] == {nid: str(Y[nid]) for nid in LST}
    assert set(dk["k_rows"]) == set(dk["k_ipfe"]) == set(LST)
    for nid in LST:
        assert len(dk["k_rows"][nid]) == 2 and all(len(row) == 8 for row in dk["k_rows"][nid])     # two leaves
        assert len(dk["k_ipfe"][nid]) == 5
        for item in [*dk["k_ipfe"][nid], *(item for row in dk["k_rows"][nid] for item in row)]:
            assert group.deserialize(item.encode("utf-8")).type == G2


def test_ciphertext_layout(pc_fixed):
    kg, _, _, cts, _ = pc_fixed
    group = PairingGroup(GROUP)
    for nid, ct in cts.items():
        assert list(ct) == list(PCKPABCiphertext.FIELDS)
        assert "attributes_digest" not in ct                # the set of a client is not a claim that the others check
        assert ct["nid"] == nid and ct["label"] == encode_label("round-1")
        assert ct["attributes"] == sorted(SETS[nid])
        assert (ct["policy_compiler"], ct["attr_hash"], ct["tag_hash"]) == (POLICY_COMPILER, ATTR_HASH, TAG_HASH)
        assert ct["pp_binding"] == kg.get_public_parameters()["pp_binding"]
        assert len(ct["c_attr"]) == len(SETS[nid]) and all(len(row) == 8 for row in ct["c_attr"])
        assert len(ct["c_ipfe"]) == 5 and isinstance(ct["t"], str)
        for item in [*ct["t"].split(), *ct["c_ipfe"], *(item for row in ct["c_attr"] for item in row)]:
            assert group.deserialize(item.encode("utf-8")).type == G1


def test_the_records_are_those_of_the_other_scheme_where_they_do_not_differ():
    assert PCKPABClientKey.FIELDS == ("nid", "s", "u", "p_h1", "p_h2", "h3", "theta_h5", "f", "pp_binding")
    assert PCKPABCiphertext.FIELDS == ("nid", "label", "attributes", "policy_compiler", "attr_hash", "tag_hash",
                                       "c_attr", "t", "c_ipfe", "pp_binding")
    assert PCKPABFunctionalKey.FIELDS == ("policy", "policy_digest", "policy_compiler", "y", "k_rows", "k_ipfe",
                                          "pp_binding")


# ---------------------------------------------------------------------------
# the other scheme's objects, and the ones of other setups


def test_the_objects_of_the_other_multi_client_scheme_are_refused(make_pc, make_mc):
    kg, clients, agg = make_pc()
    kg_n2, clients_n2, _ = make_mc()
    pp, pp_n2 = kg.get_public_parameters(), kg_n2.get_public_parameters()
    # the public parameters of one are not the other's: the digest is taken under a domain of its own
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairingPerClient({"keys": {"pp": pp_n2}})
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": pp}})
    # and so are the keys and the ciphertexts: a different binding, and, with the binding forged, a different shape
    cts, dk = encrypt_each(clients), kg.get_decryption_keys("agg", credentials=creds(POLICY))
    cts_n2 = {nid: clients_n2[nid].encrypt([X[nid]], "round-1", attributes=["region:EU"]) for nid in LST}
    dk_n2 = kg_n2.get_decryption_keys("agg", credentials=creds("region:EU"))
    with pytest.raises(FEValidationError, match="pp_binding"):
        agg.decrypt(cts_n2, dk)
    with pytest.raises(FEValidationError, match="pp_binding"):
        agg.decrypt(cts, dk_n2)
    for ct in cts_n2.values():
        ct["pp_binding"] = pp["pp_binding"]
    dk_n2["pp_binding"] = pp["pp_binding"]
    with pytest.raises(FEValidationError, match="c_ipfe"):
        agg.decrypt(cts_n2, dk)
    with pytest.raises(FEValidationError, match="k_ipfe"):
        agg.decrypt(cts, dk_n2)


def test_a_key_of_another_setup_does_not_open(make_pc):
    kg, clients, agg = make_pc()
    other_kg, _, _ = make_pc()
    with pytest.raises(FEValidationError, match="pp_binding"):
        agg.decrypt(encrypt_each(clients), other_kg.get_decryption_keys("agg", credentials=creds(POLICY)))


def test_decrypt_checks_the_bindings_before_anything_else(pc_fixed, make_pc, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    other_kg, other_clients, _ = make_pc()
    other_cts = encrypt_each(other_clients)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="ciphertext of nid_1.*pp_binding"):
        agg.decrypt(dict(cts, nid_1=other_cts["nid_1"]), dk)
    with pytest.raises(FEValidationError, match="functional key.*pp_binding"):
        agg.decrypt(cts, other_kg.get_decryption_keys("agg", credentials=creds(POLICY)))
    no_binding = {nid: {k: v for k, v in ct.items() if k != "pp_binding"} for nid, ct in cts.items()}
    with pytest.raises(FEValidationError, match="no pp_binding"):
        agg.decrypt(no_binding, dk)


# ---------------------------------------------------------------------------
# what is refused before anything is paired


def test_every_client_is_needed(make_pc, monkeypatch):  # Review Focus 2
    kg, clients, agg = make_pc()
    cts = encrypt_each(clients)
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY, dict(Y, nid_0=0)))      # nid_0 has the weight 0
    refuse_to_compute(monkeypatch)
    for missing in LST:
        with pytest.raises(FEValidationError, match=missing):
            agg.decrypt({nid: ct for nid, ct in cts.items() if nid != missing}, dk)
    with pytest.raises(FEValidationError, match="nid_1.*nid_2|nid_2.*nid_1"):
        agg.decrypt({"nid_0": cts["nid_0"]}, dk)
    with pytest.raises(FEValidationError, match="nid_9"):                           # a ciphertext of no client
        agg.decrypt(dict(cts, nid_9=cts["nid_0"]), dk)


def test_a_ciphertext_in_the_slot_of_another_client_is_refused(pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    refuse_to_compute(monkeypatch)
    swapped = dict(cts, nid_0=cts["nid_1"], nid_1=cts["nid_0"])
    with pytest.raises(FEValidationError, match="nid_0.*nid_1"):
        agg.decrypt(swapped, dk)


@pytest.mark.parametrize("owner", LST)
def test_ciphertexts_that_differ_in_the_tag_are_refused(owner, pc_fixed, monkeypatch):
    _, clients, agg, cts, dk = pc_fixed
    cts[owner] = clients[owner].encrypt([X[owner]], "round-2", attributes=SETS[owner])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="differ in their label"):
        agg.decrypt(cts, dk)


def test_the_label_argument_must_be_the_one_of_the_ciphertexts(pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    assert agg.decrypt(cts, dk, label="round-1") == EXPECTED
    assert agg.decrypt(cts, dk, label=None) == EXPECTED
    refuse_to_compute(monkeypatch)
    for label in ("round-2", 5, ElementLabel("round-1", 0, (0,))):
        with pytest.raises(FEValidationError, match="label"):
            agg.decrypt(cts, dk, label=label)


def test_the_fusion_weight_must_be_the_weights_of_the_key(pc_fixed):
    _, _, agg, cts, dk = pc_fixed
    assert agg.decrypt(cts, dk, fusion_weight={nid: [Y[nid]] for nid in LST}) == EXPECTED
    for wrong in ({**{nid: [Y[nid]] for nid in LST}, "nid_2": [-3]},
                  {nid: [Y[nid]] for nid in LST[:2]},
                  {**{nid: [Y[nid]] for nid in LST}, "nid_9": [1]},
                  {nid: [Y[nid], 0] for nid in LST},
                  {nid: Y[nid] for nid in LST},
                  [Y[nid] for nid in LST], "fusion"):
        with pytest.raises(FEValidationError, match="fusion"):
            agg.decrypt(cts, dk, fusion_weight=wrong)


@pytest.mark.parametrize("field,value,match", [
    ("attributes", "region:EU", "attributes must be a list"),               # a str would be read a character at a time
    ("attributes", ("region:EU",), "attributes must be a list"),
    ("attributes", None, "attributes must be a list"),
    ("attributes", [5, 6], "label"),
    ("attributes", ["region:US", None], "label"),
    ("attributes", ["region:US", "region:US"], "sorted and hold each label once"),
    ("attributes", ["region:US", "org:A"], "sorted and hold each label once"),
    ("attributes", ["a:%03d" % i for i in range(65)], "max_attributes = 64"),
    ("attributes", ["a:" + "x" * MAX_LABEL_CHARS], "MAX_LABEL_CHARS"),
    ("label", 5, "label of the ciphertext of nid_1 must be a str"),
    ("label", None, "label of the ciphertext of nid_1 must be a str"),
    ("nid", None, "client id"), ("nid", 1.5, "client id"), ("nid", True, "client id"), ("nid", ["nid_1"], "client id"),
    ("nid", np.array([0, 1]), "client id"), ("nid", "nid_2", "nid_1.*nid_2"), ("nid", 7, "nid_1.*7"),
])
def test_hostile_ciphertexts(field, value, match, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    cts["nid_1"][field] = value
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=match):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("record,field", [
    ("ct", "policy_compiler"), ("ct", "attr_hash"), ("ct", "tag_hash"), ("ct", "pp_binding"),
    ("key", "policy_compiler"), ("key", "pp_binding"), ("key", "policy_digest"),
])
def test_an_array_in_a_field_is_refused_and_not_compared(record, field, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    (dk if record == "key" else cts["nid_1"])[field] = np.array([0, 1])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field", PCKPABCiphertext.FIELDS)
def test_decrypt_refuses_a_ciphertext_without_a_field(field, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    del cts["nid_1"][field]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field", PCKPABFunctionalKey.FIELDS)
def test_decrypt_refuses_a_key_without_a_field(field, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    del dk[field]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,record", [("policy_compiler", "key"), ("policy_compiler", "ct"),
                                          ("attr_hash", "ct"), ("tag_hash", "ct")])
def test_decrypt_refuses_an_unknown_version_marker(field, record, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    (dk if record == "key" else cts["nid_2"])[field] = "v9"
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,shape", [
    ("c_attr", []), ("c_attr", [["1:x"] * 8]), ("c_attr", [["1:x"] * 7, ["1:x"] * 8]), ("c_attr", "x"),
    ("c_ipfe", ["1:x"] * 4), ("c_ipfe", ["1:x"] * 6), ("c_ipfe", "1:x"), ("t", ["1:x"]), ("t", 5), ("t", None),
])
def test_decrypt_refuses_a_ciphertext_of_the_wrong_shape(field, shape, pc_fixed, monkeypatch):
    # four elements of c_ipfe are the ciphertext of mcfe/kp_ab_pairing
    _, _, agg, cts, dk = pc_fixed
    cts["nid_2"][field] = shape
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,change", [
    ("y", lambda value: {k: v for k, v in value.items() if k != "nid_1"}),
    ("y", lambda value: dict(value, nid_9="1")),
    ("k_rows", lambda value: {k: v for k, v in value.items() if k != "nid_2"}),
    ("k_rows", lambda value: dict(value, nid_2=value["nid_2"][:1])),
    ("k_rows", lambda value: dict(value, nid_2=[value["nid_2"][0], value["nid_2"][1][:7]])),
    ("k_ipfe", lambda value: dict(value, nid_0=value["nid_0"][:4])),          # the four elements of the other scheme
    ("k_ipfe", lambda value: dict(value, nid_0=value["nid_0"] + value["nid_0"][:1])),
    ("k_ipfe", lambda value: {k: v for k, v in value.items() if k != "nid_0"}),
])
def test_decrypt_refuses_a_key_of_the_wrong_shape(field, change, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    dk[field] = change(dk[field])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


def test_decrypt_decodes_every_vector_of_the_ciphertext_also_the_unused(make_pc):
    kg, clients, agg = make_pc()
    sets = {"nid_0": ["region:EU", "org:B"], "nid_1": ["region:US"], "nid_2": ["org:A", "region:EU"]}
    cts = encrypt_each(clients, sets)
    dk = kg.get_decryption_keys("agg", credentials=creds(POLICY))                    # org:A and org:B are no row
    assert agg.decrypt(cts, dk) == EXPECTED
    cts["nid_2"]["c_attr"][0][3] = "1:not an element"
    with pytest.raises(FEValidationError, match="c_attr of nid_2"):
        agg.decrypt(cts, dk)
    cts["nid_2"]["c_attr"][0][3] = dk["k_ipfe"]["nid_0"][0]                          # an element of G2
    with pytest.raises(FEValidationError, match="c_attr of nid_2"):
        agg.decrypt(cts, dk)


def test_decrypt_refuses_elements_of_the_wrong_group(pc_fixed):
    _, _, agg, cts, dk = pc_fixed
    keys = copy.deepcopy(dk)
    keys["k_ipfe"]["nid_0"][1] = cts["nid_0"]["c_ipfe"][0]                          # G1 where G2 is due
    with pytest.raises(FEValidationError, match="k_ipfe"):
        agg.decrypt(cts, keys)
    cts["nid_0"]["t"] = dk["k_ipfe"]["nid_0"][0]                                    # G2 where G1 is due
    with pytest.raises(FEValidationError, match="t of nid_0"):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("text,reason", [
    ("and(", "parse|policy|canonical"), ("", "parse|policy|canonical"), ("a" * 100000, "characters|limit"),
])
def test_hostile_policy(text, reason, pc_fixed, monkeypatch):
    _, _, agg, cts, dk = pc_fixed
    dk["policy"] = text
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=reason):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("bad", [None, [], "cts", 5, [("nid_0", {})]])
def test_decrypt_refuses_what_is_no_dict_of_ciphertexts(bad, pc_fixed):
    _, _, agg, _, dk = pc_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt(bad, dk)


@pytest.mark.parametrize("bad", [None, [], "key", 5])
def test_decrypt_refuses_what_is_no_key(bad, pc_fixed):
    _, _, agg, cts, _ = pc_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt(cts, bad)


# ---------------------------------------------------------------------------
# encryption: the client key and what a client is asked to encrypt


def test_encrypt_without_a_key_is_a_key_error(make_pc):
    _, _, agg = make_pc()
    with pytest.raises(FEKeyError, match="sk"):
        agg.encrypt([1], "round-1", attributes=["org:A"])


@pytest.mark.parametrize("attributes", [None, "region:EU", 5, ["region"], [5], ["region:EU", None]])
def test_encrypt_checks_the_attributes(attributes, make_pc):
    _, clients, _ = make_pc()
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1], "round-1", attributes=attributes)


def test_a_refused_encryption_computes_nothing(make_pc, monkeypatch):
    _, clients, _ = make_pc()
    monkeypatch.setattr(kp_ab_pairing, "multiexp_many", lambda *a, **k: pytest.fail("computed for refused input"))
    monkeypatch.setattr(kp_ab_pairing, "attribute_vector", lambda *a, **k: pytest.fail("computed for refused input"))
    for plaintext in ([11], [-11], [], [1, 2], 5, None):
        with pytest.raises(FEValidationError):
            clients["nid_0"].encrypt(plaintext, "round-1", attributes=["org:A"])
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1], 5, attributes=["org:A"])


#: The random scalar of an encryption whose randomness a test pins: ``psi`` and every ``sigma``.
S = 1234567891011


def pin_randomness(crypto, monkeypatch):
    """Make every scalar that the group of *crypto* draws ``S``."""
    group = crypto.group
    monkeypatch.setattr(group, "random", lambda *args, **kwargs: group.init(ZR, S))


def test_a_negative_plaintext_is_not_a_long_exponent_of_encrypt(make_pc, monkeypatch):
    """The encryption of ``mcfe/kp_ab_pairing``, which this scheme takes: ``x % order`` would be a long exponent."""
    kg, clients, _ = make_pc()
    ek = kg.get_private_keys("nid_0")
    exponents, real = [], kp_ab_pairing.multiexp_many

    def recorded(rows, exps, group):
        exponents.extend(exps)
        return real(rows, exps, group)

    monkeypatch.setattr(kp_ab_pairing, "multiexp_many", recorded)
    pin_randomness(clients["nid_0"], monkeypatch)
    masks = [[int(a), int(b), S] for a, b in zip(ek["p_h1"], ek["p_h2"])]
    for x in (3, -3, 0, -10):
        del exponents[:]
        clients["nid_0"].encrypt([x], "round-1", attributes=["org:A"])
        assert exponents == [[int(ek["s"]), int(ek["u"]), abs(x)], *masks]


@pytest.mark.parametrize("x", [0, 1, -1, 10, -10], ids=["zero", "one", "minus-one", "bound", "minus-bound"])
def test_encrypt_gives_the_ciphertext_of_the_exponent_mod_the_order(make_pc, monkeypatch, x):
    """``|x|`` on the inverse of ``g1`` and ``x % order`` on ``g1`` are one element, so the ciphertext keeps its
    bytes: that made by hand with ``g1 ** (x % order)``, under the same randomness."""
    kg, clients, _ = make_pc()
    pin_randomness(clients["nid_0"], monkeypatch)
    ct = clients["nid_0"].encrypt([x], "round-1", attributes=["org:A", "region:EU"])
    group = PairingGroup(GROUP)
    order = int(group.order())
    pp, ek = kg.get_public_parameters(), kg.get_private_keys("nid_0")
    omega, omega_prime = tag_points(group, SCHEME_ID, pp["pp_binding"], encode_label("round-1"), None)
    g1 = group.deserialize(pp["g1"].encode("utf-8"))
    f = [deserialize_many(group, row) for row in ek["f"]]
    t = omega ** int(ek["s"]) * omega_prime ** int(ek["u"]) * g1 ** (x % order)
    c_ipfe = [omega ** int(a) * omega_prime ** int(b) * h ** S * fifth for a, b, h, fifth in zip(
        ek["p_h1"], ek["p_h2"], deserialize_many(group, ek["h3"]), deserialize_many(group, ek["theta_h5"]))]
    assert ct == dict(
        ct,
        c_attr=[ac_codec.serialize_vec(group, attribute_vector(f, name, S, S, order, group))
                for name in ["org:A", "region:EU"]],
        t=group.serialize(t).decode("utf-8"),
        c_ipfe=ac_codec.serialize_vec(group, c_ipfe),
    )


def test_the_client_key_must_have_the_fifth_coordinate(make_pc):
    kg, _, _ = make_pc()
    pp = kg.get_public_parameters()
    ek = kg.get_private_keys("nid_0")
    del ek["theta_h5"]
    with pytest.raises(FEValidationError, match="theta_h5"):
        MCFEKPABPairingPerClient({"id": "nid_0", "keys": {"pp": pp, "sk": ek}})
    h3 = kg.get_private_keys("nid_0")["h3"]
    for wrong in (h3[:4], h3 + h3[:1], "x", None):          # four elements: the key of the other scheme
        ek = kg.get_private_keys("nid_0")
        ek["theta_h5"] = wrong
        with pytest.raises(FEValidationError, match="theta_h5"):
            MCFEKPABPairingPerClient({"id": "nid_0", "keys": {"pp": pp, "sk": ek}})
    for field in ("p_h1", "p_h2", "h3"):
        ek = kg.get_private_keys("nid_0")
        ek[field] = ek[field][:4]
        with pytest.raises(FEValidationError, match=field):
            MCFEKPABPairingPerClient({"id": "nid_0", "keys": {"pp": pp, "sk": ek}})


def test_a_tampered_element_of_theta_h5_is_refused_when_it_is_read(make_pc):
    kg, _, _ = make_pc()
    pp = kg.get_public_parameters()
    g2_element = kg.get_decryption_keys("agg", credentials=creds())["k_ipfe"]["nid_0"][0]
    for bad in ("1:not an element", g2_element):            # no element of G1, and an element of G2
        ek = kg.get_private_keys("nid_0")
        ek["theta_h5"][2] = bad
        client = MCFEKPABPairingPerClient({"id": "nid_0", "keys": {"pp": pp, "sk": ek}})
        with pytest.raises(FEValidationError, match="theta_h5"):
            client.encrypt([1], "round-1", attributes=["org:A"])


def test_a_client_key_of_the_other_scheme_is_refused(make_pc, make_mc):
    kg, _, _ = make_pc()
    kg_n2, _, _ = make_mc()
    pp, ek = kg.get_public_parameters(), kg_n2.get_private_keys("nid_0")
    with pytest.raises(FEValidationError, match="theta_h5|pp_binding"):
        MCFEKPABPairingPerClient({"id": "nid_0", "keys": {"pp": pp, "sk": ek}})


def test_the_id_of_a_client_must_be_the_one_of_its_key(make_pc):
    kg, _, _ = make_pc()
    pp = kg.get_public_parameters()
    with pytest.raises(FEValidationError, match="nid_1"):
        MCFEKPABPairingPerClient({"id": "nid_1", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})


def test_crypto_needs_public_parameters():
    with pytest.raises(FEKeyError, match="pp"):
        MCFEKPABPairingPerClient({"keys": {"other": 1}})
    with pytest.raises(FEKeyError):
        MCFEKPABPairingPerClient({"keys": {}})


def test_tampered_pp(make_pc):
    kg, _, _ = make_pc()
    pp = kg.get_public_parameters()
    for changes in ({"bound_x": "11"}, {"max_attributes": 65}, {"tag_hash": "other"}, {"n": 4}):
        with pytest.raises(FEValidationError):
            MCFEKPABPairingPerClient({"keys": {"pp": dict(pp, **changes)}})


# ---------------------------------------------------------------------------
# the key generator


def test_clients_are_needed(own_parameter_cache):
    with pytest.raises(FEConfigError, match="need n"):
        MCFEKPABPairingPerClientKeyGenerator({"pairing_group_param": GROUP})


@pytest.mark.parametrize("eta", [2, 0, True, 1.0, "1", None])
def test_eta_is_one(eta, own_parameter_cache):
    with pytest.raises(FEValidationError, match="eta"):
        MCFEKPABPairingPerClientKeyGenerator({"n": 2, "eta": eta, "pairing_group_param": GROUP})


def test_dlog_limit(own_parameter_cache):
    with pytest.raises(FEValidationError, match="n \\* bound_x \\* bound_y"):
        MCFEKPABPairingPerClientKeyGenerator(
            {"n": 5, "bound_x": 2 ** 24, "bound_y": 2 ** 24, "pairing_group_param": GROUP})


def test_keys_before_setup(own_parameter_cache):
    kg = MCFEKPABPairingPerClientKeyGenerator({"n": 2, "pairing_group_param": GROUP})
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_decryption_keys("agg", credentials=creds(y={"nid_0": 1, "nid_1": 1}))
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_private_keys("nid_0")


@pytest.mark.parametrize("credentials", [
    None, {}, {"policy": "a:1"}, {"fusion_weight": {"nid_0": [1], "nid_1": [1], "nid_2": [1]}},
    {"fusion_weight": {"nid_0": [1], "nid_1": [1]}, "policy": "a:1"},
    {"fusion_weight": {"nid_0": [11], "nid_1": [1], "nid_2": [1]}, "policy": "a:1"},
    {"fusion_weight": {"nid_0": [1], "nid_1": [1], "nid_2": [1]}, "policy": "and("},
    {"fusion_weight": {"nid_0": [1], "nid_1": [1], "nid_2": [1]}, "policy": 5},
])
def test_key_requests_are_checked(credentials, make_pc):
    kg, _, _ = make_pc()
    with pytest.raises((FEKeyError, FEValidationError)):
        kg.get_decryption_keys("agg", credentials=credentials)


def test_a_policy_of_more_rows_than_the_limit_is_refused(make_pc):
    kg, _, _ = make_pc(max_rows=2)
    with pytest.raises(FEValidationError, match="rows"):
        kg.get_decryption_keys("agg", credentials=creds("and(a:1, b:1, c:1)"))


def test_the_parameter_record_has_a_folder_of_its_own(make_pc, tmp_path):
    assert CryptoCONST.TYPE_MCFE_KP_AB_PC != CryptoCONST.TYPE_MCFE_KP_AB
    kg, _, _ = make_pc(cache_dir=str(tmp_path))
    with open(os.path.join(kg.config_folder, CryptoCONST.TYPE_MCFE_KP_AB_PC, "param.json"), encoding="utf-8") as f:
        record = json.load(f)
    assert list(record) == list(MCKPABRecord.FIELDS)                # the configuration only


@pytest.mark.parametrize("override", [{"n": 4}, {"lst_nid": ["a", "b", "c"]}, {"max_attributes": 8}, {"bound_x": 11}])
def test_a_new_configuration_replaces_the_record_without_a_warning(make_pc, tmp_path, recwarn, caplog, override):
    cache = str(tmp_path)
    make_pc(cache_dir=cache)
    with caplog.at_level(logging.INFO, logger="pyfe4ai.core.base"):
        make_pc(cache_dir=cache, **override)
    assert not [w for w in recwarn if issubclass(w.category, FEWarning)], [str(w.message) for w in recwarn]
    assert [r.levelno for r in caplog.records if "does not match" in r.getMessage()] == [logging.INFO]


def test_same_configuration_keeps_the_cached_record(make_pc, tmp_path):
    cache = str(tmp_path)
    make_pc(cache_dir=cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FEWarning)
        make_pc(cache_dir=cache)


# ---------------------------------------------------------------------------
# registry


def test_status_is_known_insecure():
    scheme = pyfe4ai.get_scheme(SCHEME_ID)
    assert (scheme.status.status, scheme.status.fidelity) == ("known-insecure", "adapted")
    assert scheme.status.reference == "NPP22 Sec. 5.4"
    issue, note = scheme.status.issue, scheme.status.note
    # the attack in one sentence: two tags, two keys of one y that open neither (and whose condition is that the set of
    # each client satisfies one of the keys under both tags), the difference of the inner products
    assert "\n" not in issue and not issue.endswith(".")
    for words in ("fifth coordinate", "does not depend on the tag", "two keys",
                  "the set of each client satisfies one of the keys", "under both tags", "difference", "Thm. 25"):
        assert words in issue, words
    assert scheme.refused == "known to be insecure: " + issue      # the standard profile refuses it through `refused`
    assert "h*_{i,5}" in note and "stops" not in note and "splic" not in note       # the note no longer claims it
    assert scheme.capabilities == {"access_control", "labels", "pairing"}
    for status, listed in (("known-insecure", True), ("experimental", False), ("unreviewed", False)):
        assert (SCHEME_ID in pyfe4ai.list_schemes(status=status)) is listed, status


def test_the_key_generator_and_the_crypto_object_warn_of_the_issue(make_pc):
    issue = pyfe4ai.get_scheme(SCHEME_ID).status.issue
    message = "mcfe/kp_ab_pairing_per_client is known to be insecure: " + issue
    config = {"n": 2, "bound_x": 10, "bound_y": 10, "pairing_group_param": GROUP}
    with pytest.warns(FESecurityWarning) as caught:
        kg = MCFEKPABPairingPerClientKeyGenerator(config)
    assert [str(w.message).startswith(message) for w in caught if "known to be insecure" in str(w.message)] == [True]
    kg.setup()
    with pytest.warns(FESecurityWarning) as caught:
        MCFEKPABPairingPerClient({"keys": {"pp": kg.get_public_parameters()}})
    assert [str(w.message).startswith(message) for w in caught if "known to be insecure" in str(w.message)] == [True]
    with pytest.warns(FESecurityWarning, match="known to be insecure"):
        pyfe4ai.create(SCHEME_ID, **config)


def test_the_standard_profile_refuses_the_scheme(own_parameter_cache):
    issue = pyfe4ai.get_scheme(SCHEME_ID).status.issue
    for build in (lambda: MCFEKPABPairingPerClientKeyGenerator({"n": 2, "security_profile": "standard"}),
                  lambda: pyfe4ai.create(SCHEME_ID, n=2, security_profile="standard")):
        expected = "no 'standard' parameter set for {}: known to be insecure: ".format(SCHEME_ID)
        expected += re.escape(issue[:50])
        with pytest.raises(FEValidationError, match=expected):
            build()


def test_create_and_list():
    kg, crypto_cls = pyfe4ai.create(SCHEME_ID, n=2, pairing_group_param=GROUP)
    assert (type(kg).__name__, crypto_cls.__name__) == ("MCFEKPABPairingPerClientKeyGenerator",
                                                        "MCFEKPABPairingPerClient")
    assert per_client._SCHEME_ID == SCHEME_ID == pyfe4ai.get_scheme(SCHEME_ID).id       # hashed into the tag points
    assert SCHEME_ID in pyfe4ai.list_schemes("mcfe", capabilities=["labels", "access_control"])
    assert SCHEME_ID not in pyfe4ai.list_schemes("sife", capabilities="access_control")
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: crypto_cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in kg.lst_nid}
    dk = kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B)", {"nid_0": 2, "nid_1": -1}))
    cts = {"nid_0": clients["nid_0"].encrypt([3], "round", attributes=["org:A"]),
           "nid_1": clients["nid_1"].encrypt([4], "round", attributes=["org:B"])}
    assert crypto_cls({"keys": {"pp": pp}}).decrypt(cts, dk) == 2


def test_the_domains_are_the_documented_ones():
    assert per_client._PP_DOMAIN == PP_DOMAIN and per_client._SCHEME_ID == SCHEME_ID
    assert per_client.H_DIM == 5 and kp_ab_pairing.H_DIM == 4
    assert MCFEKPABPairingPerClient.H_DIM == 5 and MCFEKPABPairing.H_DIM == 4
    assert (POLICY_COMPILER, ATTR_HASH, TAG_HASH) == ("c1", "shake256-64", "PYFE4AI-V05-MCFE-KPAB-TAG")


# ---------------------------------------------------------------------------
# tag points


TAG_ARGS = (SCHEME_ID, "00" * 32, encode_label("round-1"), None)


def serialised(group, args):
    return tuple(group.serialize(point).decode("utf-8") for point in tag_points(group, *args))


def test_tag_points_vector():
    # the tag alone is hashed: the digest of the attributes is null in the message
    group = PairingGroup(GROUP)
    if group.groupType() != "BLS12-381":
        pytest.skip("the known answer is for BLS12-381")
    message = ('["mcfe/kp_ab_pairing_per_client","' + "00" * 32 + '","[\\"label\\", \\"round-1\\"]",null]').encode()
    assert [group.serialize(hash_to_g1(group, message + suffix, TAG_DST)).decode()
            for suffix in (b"|0", b"|1")] == list(serialised(group, TAG_ARGS))
    assert serialised(group, TAG_ARGS) == (
        "1:g7g2cRhQWMh4prZ+zpF+5sKHLVDLk1HjP+KDsrxDgpjt56Nql6dQP3pu5A4f1WE7",
        "1:i2jmDXZuC/Q0mr7+DRq+QnBWG4FNgTxbBs3MTZUhxGoIqwVW77JUGjKXY7faO5Zw",
    )


def test_the_tag_points_are_not_those_of_the_other_scheme():
    group = PairingGroup(GROUP)
    other = (kp_ab_pairing._SCHEME_ID, "00" * 32, encode_label("round-1"), None)
    assert serialised(group, TAG_ARGS)[0] != serialised(group, other)[0]


# ---------------------------------------------------------------------------
# arrays: each element under a tag of its own, each client for a set of its own, all decrypted in one batch


@pytest.fixture
def array_fixed(make_pc):
    """A key generator, the clients, an aggregator, the arrays of every client, their ciphertexts for ``SETS`` and a
    key for ``POLICY``."""
    kg, clients, agg = make_pc()
    arrays = random_arrays()
    cts = encrypt_arrays_each(clients, arrays)
    return kg, clients, agg, arrays, cts, kg.get_decryption_keys("agg", credentials=creds(POLICY))


def test_ndarray_with_attribute_sets_of_their_own(array_fixed):
    _, _, agg, arrays, cts, dk = array_fixed
    got = aggregate(agg, cts, dk)
    assert [layer.shape for layer in got] == [(2, 3), (4,)]
    for layer, expected in zip(got, combined(arrays)):
        assert layer.dtype == float and (layer == expected).all()
    for l, layer in enumerate(got):                                         # and what decrypt gives for the element
        for i, value in np.ndenumerate(layer):
            ct = {nid: cts[nid][l][i] for nid in LST}
            assert value == agg.decrypt(ct, dk, label=element_label("round-1", l, i))


@pytest.mark.parametrize("policy,sets", PER_CLIENT)
def test_ndarray_matches_the_plaintext(policy, sets, make_pc):
    kg, clients, agg = make_pc()
    arrays = random_arrays()
    cts = encrypt_arrays_each(clients, arrays, dict(zip(LST, sets)))
    got = aggregate(agg, cts, kg.get_decryption_keys("agg", credentials=creds(policy)))
    assert all((layer == expected).all() for layer, expected in zip(got, combined(arrays)))


def test_compute_is_decrypt(array_fixed):
    _, _, agg, arrays, cts, dk = array_fixed
    got = agg.compute_lst_ndarray_ct(cts, dk=dk, fusion_weight={nid: [Y[nid]] for nid in LST}, label="round-1")
    assert all((layer == expected).all() for layer, expected in zip(got, combined(arrays)))


def test_encrypt_lst_ndarray_reads_the_attributes_once(make_pc):
    # attributes given as an iterator are read once for the whole array, not once for each element
    kg, clients, agg = make_pc()
    arrays = random_arrays()
    cts = {nid: clients[nid].encrypt_lst_ndarray(arrays[nid], label="round-1", attributes=iter(sorted(SETS[nid])))
           for nid in LST}
    assert all(ct["attributes"] == sorted(SETS["nid_2"]) for layer in cts["nid_2"] for ct in layer.flat)
    got = aggregate(agg, cts, kg.get_decryption_keys("agg", credentials=creds(POLICY)))
    assert all((layer == expected).all() for layer, expected in zip(got, combined(arrays)))


def test_ndarray_elements_are_under_their_own_tags_and_not_their_sets(array_fixed):
    kg, _, _, arrays, cts, _ = array_fixed
    group = PairingGroup(GROUP)
    pp = kg.get_public_parameters()
    for nid in LST:
        tags = set()
        for l, layer in enumerate(cts[nid]):
            for i, ct in np.ndenumerate(layer):
                assert ct["label"] == encode_label(ElementLabel("round-1", l, i))
                tags.add(tag_points(group, SCHEME_ID, pp["pp_binding"], ct["label"], None)[0])
        assert len({group.serialize(point) for point in tags}) == 6 + 4


def spied(monkeypatch, module, name):
    """The list of the arguments of the calls to ``module.name``, which still does what it did."""
    calls, real = [], getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


@pytest.mark.parametrize("batch", [None, 4, 1])
def test_ndarray_pairs_all_elements_in_slices_whatever_the_size_of_each_clients_set(batch, make_pc, monkeypatch):
    # The clients' sets have 2, 3 and 3 labels and use 2, 1 and 2 rows of the policy, so that the ciphertexts and the
    # rows of the key differ in length between clients: every element holds 22 + 30 + 30 decoded G1 elements and is
    # paired over 21 + 13 + 21 + 1 points of the prepared key. A slice holds as many elements as the budget of decoded
    # elements allows: all ten by default, batch with the largest budget that holds batch elements of 22 + 30 + 30
    # decoded elements each (and more elements of any smaller count).
    if batch is not None:
        monkeypatch.setattr(kp_ab_pairing, "_SLICE_DECODED_ELEMENTS", (batch + 1) * (22 + 30 + 30) - 1)
    step = kp_ab_pairing._slice_size(22 + 30 + 30)
    assert step == (kp_ab_pairing._SLICE_DECODED_ELEMENTS // (22 + 30 + 30) if batch is None else batch)
    sizes = [min(step, 10 - start) for start in range(0, 10, step)]
    kg, clients, agg = make_pc()
    sets = {"nid_0": ["a:1", "b:1"], "nid_1": ["c:1", "d:1", "e:1"], "nid_2": ["a:1", "b:1", "c:1"]}
    arrays = random_arrays()
    cts = encrypt_arrays_each(clients, arrays, sets)
    dk = kg.get_decryption_keys("agg", credentials=creds("or(and(a:1, b:1), c:1)"))
    aggregate(agg, cts, dk)                                 # the generators are decoded at their first use
    prepared = spied(monkeypatch, kp_ab_pairing, "prepare_g2")
    products = spied(monkeypatch, kp_ab_pairing, "multi_pair_prepared")
    single = spied(monkeypatch, kp_ab_pairing, "multi_pair")
    decoded = spied(monkeypatch, ac_codec, "deserialize_many")
    dlogs = spied(monkeypatch, kp_ab_pairing, "bounded_discrete_log_gt")
    for _ in range(2):
        got = aggregate(agg, cts, dk)
        assert all((layer == expected).all() for layer, expected in zip(got, combined(arrays)))
    per_element = (8 * 2 + 5) + (8 * 1 + 5) + (8 * 2 + 5) + 1
    assert [len(args[0]) for args in prepared] == [per_element] * 2
    assert [[len(row) for row in args[0]] for args in products] == [[per_element] * size for size in sizes] * 2
    assert single == [] and len(dlogs) == 20
    assert [len(args[1]) for args in decoded] == [
        (8 * 2 + 5) + (8 * 1 + 5) + (8 * 2 + 5), *(size * ((8 * 2 + 6) + (8 * 3 + 6) + (8 * 3 + 6)) for size in sizes)
    ] * 2


@pytest.mark.parametrize("batch", [1, 3, 4, 10, 11])
def test_ndarray_does_not_depend_on_the_slices(batch, make_pc, monkeypatch):
    kg, clients, agg = make_pc()
    arrays = random_arrays()
    sets = {"nid_0": ["org:A", "region:EU"], "nid_1": ["region:US", "org:A"], "nid_2": ["region:EU", "region:US"]}
    cts = encrypt_arrays_each(clients, arrays, sets)        # weights other than 1, other rows for each client
    dk = kg.get_decryption_keys("agg", credentials=creds("2of(region:EU, region:US, org:A)"))
    monkeypatch.setattr(kp_ab_pairing, "_SLICE_DECODED_ELEMENTS", batch * 3 * (8 * 2 + 6))       # batch elements
    got = aggregate(agg, cts, dk)
    assert [layer.shape for layer in got] == [(2, 3), (4,)]
    assert all((layer == expected).all() for layer, expected in zip(got, combined(arrays)))


def test_ndarray_refuses_an_unsatisfied_client_before_it_pairs_anything(array_fixed, monkeypatch):
    kg, clients, agg, arrays, _, dk = array_fixed
    sets = dict(SETS, nid_1=["org:A"])
    cts = encrypt_arrays_each(clients, arrays, sets)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match="ciphertext of nid_1 do not satisfy"):
        aggregate(agg, cts, dk)


def test_ndarray_swapped_elements_do_not_decrypt(array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    layer = cts["nid_1"][0]
    layer[0, 0], layer[0, 1] = layer[0, 1], layer[0, 0]
    with monkeypatch.context() as patched:                  # the label fields give them away, before any pairing
        refuse_to_compute(patched)
        with pytest.raises(FEValidationError, match=r"layer 0, element \(0, 0\).*differ in their label"):
            aggregate(agg, cts, dk)
    for index in ((0, 0), (0, 1)):                          # with the label fields forged too: no value
        layer[index] = dict(layer[index], label=encode_label(ElementLabel("round-1", 0, index)))
    with pytest.raises(FESchemeError, match=r"element \(0, 0\) of layer 0 does not decrypt"):
        aggregate(agg, cts, dk)


def test_one_attribute_set_per_client_and_array(array_fixed, monkeypatch):
    _, clients, agg, arrays, cts, dk = array_fixed
    # element (1, 2) of layer 0 of client 1 is for another set of attributes, which satisfies the policy as well: the
    # sets of the other clients are theirs, and the elements of one client are for one set
    other_set = clients["nid_1"].encrypt([int(arrays["nid_1"][0][1, 2])], ElementLabel("round-1", 0, (1, 2)),
                                         attributes=["region:US", "org:B"])
    cts["nid_1"][0][1, 2] = other_set
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=r"layer 0, element \(1, 2\).*nid_1.*one set of attributes"):
        aggregate(agg, cts, dk)
    with pytest.raises(FEValidationError, match=r"layer 0, element \(1, 2\).*nid_1.*one set of attributes"):
        agg.check_ndarray_ct(cts, label="round-1")


def test_check_ndarray_ct_follows(array_fixed, monkeypatch):
    _, clients, agg, arrays, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    assert agg.check_ndarray_ct(cts, label="round-1") is None           # every client for a set of its own
    with pytest.raises(FEValidationError, match="string label"):
        agg.check_ndarray_ct(cts, label=None)
    with pytest.raises(FEValidationError, match=r"layer 0, element \(0, 0\).*label"):
        agg.check_ndarray_ct(cts, label="round-2")
    with pytest.raises(FEValidationError, match="nid_2"):
        agg.check_ndarray_ct({nid: cts[nid] for nid in LST[:2]}, label="round-1")
    bad = copy.deepcopy(cts)
    bad["nid_2"][1][3] = dict(bad["nid_2"][1][3], c_ipfe=bad["nid_2"][1][3]["c_ipfe"][:4])
    with pytest.raises(FEValidationError, match=r"layer 1, element \(3,\).*c_ipfe of nid_2"):
        agg.check_ndarray_ct(bad, label="round-1")
    with pytest.raises(FEValidationError, match=r"layer 1, element \(3,\).*c_ipfe of nid_2"):
        aggregate(agg, bad, dk)                             # the decryption refuses it the same way


@pytest.mark.parametrize("change,match", [
    (lambda cts: cts["nid_1"].pop(), "layers"),
    (lambda cts: cts.__setitem__("nid_1", [cts["nid_1"][0], cts["nid_1"][1][:2]]), "shape"),
    (lambda cts: cts.__setitem__("nid_2", "arrays"), "list of arrays"),
])
def test_ndarray_clients_send_the_same_layers(change, match, array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    change(cts)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=match):
        aggregate(agg, cts, dk)


def test_ndarray_with_no_elements(pc_fixed, monkeypatch):
    _, clients, agg, _, dk = pc_fixed
    arrays = {nid: [np.zeros((0, 3), dtype=np.int64)] for nid in LST}
    cts = encrypt_arrays_each(clients, arrays)
    refuse_to_compute(monkeypatch)
    got = aggregate(agg, cts, dk)
    assert len(got) == 1 and got[0].shape == (0, 3) and got[0].dtype == float
    assert agg.decrypt_lst_ndarray_ct({nid: [] for nid in LST}, dk=dk, label="round-1") == []
