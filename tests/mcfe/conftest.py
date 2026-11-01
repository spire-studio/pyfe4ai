"""Fixtures shared by the tests of the multi-client key-policy attribute-based IPFE (test_kp_ab_pairing.py).

``own_parameter_cache`` is not autouse, so that the other tests of ``tests/mcfe`` keep the parameter cache of the run:
the modules that need it enable it with ``pytestmark``, and ``make_mc`` uses it. The scheme is imported when a fixture
runs, as the test modules skip themselves before they import it where no pairing backend implements the group.
"""

import pytest

from tests.pairing_group import GROUP


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
