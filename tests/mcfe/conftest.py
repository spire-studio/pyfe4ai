"""Fixtures and helpers shared by the tests of the multi-client key-policy attribute-based IPFE
(test_kp_ab_pairing.py, test_kp_ab_pairing_hardening.py).

``own_parameter_cache`` is not autouse, so that the other tests of ``tests/mcfe`` keep the parameter cache of the run:
the modules that need it enable it with ``pytestmark``, and ``make_mc`` uses it. The scheme is imported when a fixture
or a helper runs, as the test modules skip themselves before they import it where no pairing backend implements the
group.
"""

import json

import numpy as np
import pytest

from tests.pairing_group import GROUP, needs_extension

#: The clients of ``mc_fixed``, what they encrypt, the weights of its key and their inner product, the attributes of
#: its ciphertexts and the policy of its key.
LST = ["nid_0", "nid_1", "nid_2"]
X = {"nid_0": 5, "nid_1": -3, "nid_2": 7}
Y = {"nid_0": 2, "nid_1": 1, "nid_2": -4}
EXPECTED = sum(X[nid] * Y[nid] for nid in LST)
ATTRS = ["purpose:research", "region:EU"]
POLICY = "and(purpose:research, region:EU)"
#: A text in the place of an element of G2 that no element serialises to. On a group of the Rust extension (BLS12-381)
#: it carries the type tag of G2, so that the deserializer refuses it; charm may never return from a G2-tagged string
#: that it cannot read (see ``docs/security.rst``: keys for the charm groups come from a trusted source), so on a charm
#: group the text has no tag and the check of the type refuses it before charm reads it. The tests that put it in a
#: row of a key check which rows are decoded, and what they check does not depend on which of the two refuses it.
JUNK_G2 = "2:not an element" if needs_extension() else "not an element"


def creds(policy=POLICY, y=Y):
    """The credentials of a key request: the weights ``y`` by client, and the policy."""
    return {"fusion_weight": {nid: [value] for nid, value in y.items()}, "policy": policy}


def encrypt_all(clients, x=X, label="round-1", attributes=ATTRS):
    """One ciphertext of ``x[nid]`` for every client of *x*, under one tag and one set of attributes."""
    return {nid: clients[nid].encrypt([x[nid]], label, attributes=attributes) for nid in x}


def through_json(value):
    return json.loads(json.dumps(value))


def random_arrays(shapes=((2, 3), (4,)), seed=1, bound=10, lst=LST):
    """The plaintext of every client of *lst* for the ndarray helpers: one integer array of each of *shapes*, with
    entries in ``[-bound, bound]``."""
    rng = np.random.default_rng(seed)
    return {nid: [rng.integers(-bound, bound + 1, size=shape) for shape in shapes] for nid in lst}


def encrypt_arrays(clients, arrays, label="round-1", attributes=ATTRS):
    """The ciphertext arrays of every client of *arrays*, under one tag and one set of attributes."""
    return {nid: clients[nid].encrypt_lst_ndarray(arrays[nid], label=label, attributes=attributes) for nid in arrays}


def combined(arrays, y=Y):
    """``sum_i y_i x_i``, layer by layer: what the aggregation of the arrays of *arrays* under the weights *y* is."""
    layers = len(next(iter(arrays.values())))
    return [sum(arrays[nid][layer] * y[nid] for nid in arrays) for layer in range(layers)]


def aggregate(agg, cts, dk, y=Y, label="round-1"):
    """The decryption of the ciphertext arrays *cts* with the weights *y* given as the caller of the ML layer does."""
    return agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight={nid: [y[nid]] for nid in y}, label=label)


def key_part(group, ct, dk, nid, policy):
    """``prod_j e(c_j, k*_j)^w_j / e(c_ipfe, k*_ipfe)``: the part of the client *nid* in a decryption with the key *dk*
    for *policy*, from the fields of the records, a span program of :mod:`pyfe4ai.policy` and the pairing alone (the
    scheme under test is not called). It holds for both multi-client schemes: the vectors of ``c_ipfe`` and ``k_ipfe``
    are as long as the scheme's."""
    from pyfe4ai.policy import compile_formula, parse
    from pyfe4ai.primitives.pairing.dpvs import vector_pair
    from pyfe4ai.primitives.pairing.pairing_backend import deserialize_many

    order = int(group.order())
    msp = compile_formula(parse(policy), order)
    weights = msp.reconstruction(ct["attributes"])
    assert weights is not None, "the client's attributes do not satisfy the policy"
    part = None
    for row, weight in weights.items():
        at = ct["attributes"].index(msp.row_labels[row])
        term = vector_pair(deserialize_many(group, ct["c_attr"][at]),
                           deserialize_many(group, dk["k_rows"][nid][row])) ** (weight % order)
        part = term if part is None else part * term
    ipfe = vector_pair(deserialize_many(group, ct["c_ipfe"]), deserialize_many(group, dk["k_ipfe"][nid]))
    return part / ipfe


def masked_plaintext(group, ct, y_i, g2):
    """``e(t_i, [y_i]_2)``: the masked plaintext of the ciphertext *ct* against the weight *y_i* of its client."""
    from pyfe4ai.primitives.pairing.pairing_backend import pair

    return pair(group.deserialize(ct["t"].encode("utf-8")), g2 ** (y_i % int(group.order())))


def refuse_to_compute(monkeypatch):
    """Make the decoding of a group element, a pairing (single, or over prepared points) and a discrete log fail the
    test."""
    from pyfe4ai.schemes.mcfe import kp_ab_pairing

    def never(*args, **kwargs):
        pytest.fail("a group element was decoded, or a pairing or a discrete log was computed")

    for name in ("_elements", "multi_pair", "prepare_g2", "multi_pair_prepared", "bounded_discrete_log_gt"):
        monkeypatch.setattr(kp_ab_pairing, name, never)


@pytest.fixture
def own_parameter_cache(tmp_path, monkeypatch):
    """Tests with other limits would otherwise replace each other's parameter record."""
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


@pytest.fixture
def make_mc(own_parameter_cache):
    """``make_mc(n=3, **overrides)``: a key generator after ``setup()``, the crypto object of each client (by client
    id, with its encryption key) and an aggregator, which holds the public parameters only."""
    def build(n=3, **overrides):
        from pyfe4ai.schemes.mcfe.kp_ab_pairing import MCFEKPABPairing, MCFEKPABPairingKeyGenerator

        config = {"n": n, "bound_x": 10, "bound_y": 10, "pairing_group_param": GROUP}
        config.update(overrides)
        kg = MCFEKPABPairingKeyGenerator(config)
        kg.setup()
        pp = kg.get_public_parameters()
        clients = {nid: MCFEKPABPairing({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
                   for nid in kg.lst_nid}
        return kg, clients, MCFEKPABPairing({"keys": {"pp": pp}})
    return build


@pytest.fixture
def mc_fixed(make_mc):
    """A key generator, the clients, an aggregator, the ciphertexts of ``X`` under ``round-1`` for ``ATTRS``, a key for
    ``POLICY`` and ``Y``."""
    kg, clients, agg = make_mc()
    return kg, clients, agg, encrypt_all(clients), kg.get_decryption_keys("agg", credentials=creds())
