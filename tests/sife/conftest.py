"""Fixtures shared by the tests of the attribute-based IPFE (test_cp_ab_pairing.py, test_cp_ab_pairing_hardening.py,
test_kp_ab_pairing.py).

``own_parameter_cache`` is not autouse, so that the other tests of ``tests/sife`` keep the parameter cache of the run:
the modules enable it with ``pytestmark``. The scheme is imported when a fixture runs, as the test modules skip
themselves before they import it where no pairing backend implements the group.
"""

import pytest

from tests.pairing_group import GROUP

UNIVERSE = ["org:A", "org:B", "org:C", "role:auditor"]
ETA = 5


@pytest.fixture
def own_parameter_cache(tmp_path, monkeypatch):
    """Tests with other limits would otherwise replace each other's parameter record, with a warning each."""
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


@pytest.fixture
def make():
    """``make(**overrides)``: a key generator after ``setup()`` and a crypto object on its public parameters."""
    def build(**overrides):
        from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing, SIFECPABPairingKeyGenerator

        config = {"eta": ETA, "bound_x": 10, "bound_y": 10, "universe": UNIVERSE, "pairing_group_param": GROUP}
        config.update(overrides)
        kg = SIFECPABPairingKeyGenerator(config)
        kg.setup()
        return kg, SIFECPABPairing({"keys": {"pp": kg.get_public_parameters()}})
    return build


@pytest.fixture
def make_kp():
    """``make_kp(**overrides)``: a key-policy key generator after ``setup()`` and a crypto object on its public
    parameters."""
    def build(**overrides):
        from pyfe4ai.schemes.sife.kp_ab_pairing import SIFEKPABPairing, SIFEKPABPairingKeyGenerator

        config = {"eta": ETA, "bound_x": 10, "bound_y": 10, "pairing_group_param": GROUP}
        config.update(overrides)
        kg = SIFEKPABPairingKeyGenerator(config)
        kg.setup()
        return kg, SIFEKPABPairing({"keys": {"pp": kg.get_public_parameters()}})
    return build


#: What ``kp_fixed`` encrypts, the policy of its key and the weights of the key, and the attributes of its ciphertext.
KP_X = [3, -1, 0, 7, 2]
KP_Y = [1, 2, 3, -4, 5]
KP_POLICY = "and(org:A, role:auditor)"
KP_HELD = ["org:A", "org:B", "role:auditor"]        # org:B is no row of KP_POLICY


@pytest.fixture
def kp_fixed(make_kp):
    """A key-policy key generator, a crypto object, a ciphertext of ``KP_X`` for ``KP_HELD``, a key for ``KP_POLICY``
    and ``KP_Y`` that opens it, and their inner product."""
    kg, crypto = make_kp()
    ct = crypto.encrypt(KP_X, attributes=KP_HELD)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": KP_Y, "policy": KP_POLICY})
    return kg, crypto, ct, dk, sum(a * b for a, b in zip(KP_X, KP_Y))


@pytest.fixture
def fixed(make):
    """A key generator, a crypto object, a ciphertext of ``[3, -1, 0, 7, 2]`` under
    ``and(org:A, role:auditor)`` and a key that opens it, with their inner product."""
    kg, crypto = make()
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]
    ct = crypto.encrypt(x, policy="and(org:A, role:auditor)")
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": ["org:A", "role:auditor"]})
    return kg, crypto, ct, dk, sum(a * b for a, b in zip(x, y))
