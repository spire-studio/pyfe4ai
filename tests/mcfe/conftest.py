"""Fixtures and helpers shared by the tests of the multi-client key-policy attribute-based IPFE
(test_kp_ab_pairing.py, test_kp_ab_pairing_hardening.py).

``own_parameter_cache`` is not autouse, so that the other tests of ``tests/mcfe`` keep the parameter cache of the run:
the modules that need it enable it with ``pytestmark``, and ``make_mc`` uses it. The scheme is imported when a fixture
or a helper runs, as the test modules skip themselves before they import it where no pairing backend implements the
group.
"""

import json

import pytest

from tests.pairing_group import GROUP

#: The clients of ``mc_fixed``, what they encrypt, the weights of its key and their inner product, the attributes of
#: its ciphertexts and the policy of its key.
LST = ["nid_0", "nid_1", "nid_2"]
X = {"nid_0": 5, "nid_1": -3, "nid_2": 7}
Y = {"nid_0": 2, "nid_1": 1, "nid_2": -4}
EXPECTED = sum(X[nid] * Y[nid] for nid in LST)
ATTRS = ["purpose:research", "region:EU"]
POLICY = "and(purpose:research, region:EU)"


def creds(policy=POLICY, y=Y):
    """The credentials of a key request: the weights ``y`` by client, and the policy."""
    return {"fusion_weight": {nid: [value] for nid, value in y.items()}, "policy": policy}


def encrypt_all(clients, x=X, label="round-1", attributes=ATTRS):
    """One ciphertext of ``x[nid]`` for every client of *x*, under one tag and one set of attributes."""
    return {nid: clients[nid].encrypt([x[nid]], label, attributes=attributes) for nid in x}


def through_json(value):
    return json.loads(json.dumps(value))


def refuse_to_compute(monkeypatch):
    """Make the decoding of a group element, a pairing and a discrete log fail the test."""
    from pyfe4ai.schemes.mcfe import kp_ab_pairing

    def never(*args, **kwargs):
        pytest.fail("a group element was decoded, or a pairing or a discrete log was computed")

    for name in ("_elements", "multi_pair", "bounded_discrete_log_gt"):
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
