"""AGT22 (TCC 2022) Sec. 5: multi-input FE for quadratic functions (cross-user terms)."""

import random

import pytest

from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.schemes.quadratic.miqfe import MultiInputQuadratic  # noqa: E402
from pyfe4ai.schemes.quadratic.miqfe import MultiInputQuadraticKeyGenerator  # noqa: E402
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError  # noqa: E402

NIDS = ["nid_0", "nid_1"]


@pytest.fixture(scope="module")
def small():
    """n = 2 users with m = 1 integer each."""
    kg = MultiInputQuadraticKeyGenerator(
        {"n": 2, "m": 1, "lst_nid": NIDS, "bound_x": 4, "bound_c": 3, "pairing_group_param": GROUP}
    )
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: MultiInputQuadratic({"keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    return kg, pp, clients


def _dec(pp, cts, dk):
    return MultiInputQuadratic({"keys": {"pp": pp}}).decrypt(cts, dk)


def test_cross_term_and_extremes(small):
    kg, pp, clients = small
    dk = kg.get_decryption_keys(None, credentials={"coefficients": [[0, 3], [0, 0]]})
    cts = {"nid_0": clients["nid_0"].encrypt([4]), "nid_1": clients["nid_1"].encrypt([-4])}
    assert _dec(pp, cts, dk) == 3 * 4 * -4
    dk2 = kg.get_decryption_keys(None, credentials={"coefficients": [[0, -2], [0, 0]]})
    assert _dec(pp, cts, dk2) == -2 * 4 * -4
    # MIFE: a user's other ciphertext combines with the same key
    cts["nid_1"] = clients["nid_1"].encrypt([1])
    assert _dec(pp, cts, dk) == 12


def test_user_keys_hold_master_ciphertexts_only(small):
    kg, pp, _ = small
    ek = kg.get_private_keys("nid_0")
    assert set(ek) == {"mct1", "mct0"}
    assert len(ek["mct1"]) == 1 and len(ek["mct0"]) == 4 * 1 + 2 * 2 + 17
    assert set(pp) == {"pairing_group_param", "g1", "g2", "n", "m", "mddh_k", "lst_nid",
                       "bound_x", "bound_c", "sec_param"}


def test_function_class_and_inputs_validated(small):
    kg, pp, clients = small
    with pytest.raises(FEValidationError, match="i < k"):
        kg.get_decryption_keys(None, credentials={"coefficients": [[1, 0], [0, 0]]})
    with pytest.raises(FEValidationError, match="i < k"):
        kg.get_decryption_keys(None, credentials={"coefficients": [[0, 0], [2, 0]]})
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1.5])
    dk = kg.get_decryption_keys(None, credentials={"coefficients": [[0, 1], [0, 0]]})
    with pytest.raises(FESchemeError):
        _dec(pp, {"nid_0": clients["nid_0"].encrypt([1])}, dk)


def test_parameter_guards():
    with pytest.raises(FEValidationError):
        MultiInputQuadraticKeyGenerator({"n": 1, "m": 1, "lst_nid": ["a"], "pairing_group_param": GROUP})
    with pytest.raises(FEValidationError):
        MultiInputQuadraticKeyGenerator({"n": 2, "m": 1, "lst_nid": NIDS, "mddh_k": 1, "pairing_group_param": GROUP})


def test_random_cross_terms_two_entries():
    m, n, X, C = 2, 2, 5, 3
    kg = MultiInputQuadraticKeyGenerator(
        {"n": n, "m": m, "lst_nid": NIDS, "bound_x": X, "bound_c": C, "pairing_group_param": GROUP}
    )
    kg.setup()
    pp = kg.get_public_parameters()
    x = {nid: [random.randint(-X, X) for _ in range(m)] for nid in NIDS}
    cts = {
        nid: MultiInputQuadratic({"keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
        for nid in NIDS
    }
    coeffs = [[0] * (n * m) for _ in range(n * m)]
    for mu in range(m):
        for nu in range(m, 2 * m):
            coeffs[mu][nu] = random.randint(-C, C)
    xs = x["nid_0"] + x["nid_1"]
    expected = sum(coeffs[a][b] * xs[a] * xs[b] for a in range(n * m) for b in range(n * m))
    dk = kg.get_decryption_keys(None, credentials={"coefficients": coeffs})
    assert _dec(pp, cts, dk) == expected


def test_three_users_cross_blocks():
    """With n = 3 the blocks (0,1), (0,2), (1,2) are allowed: an i/k swap would show."""
    nids = ["u0", "u1", "u2"]
    kg = MultiInputQuadraticKeyGenerator(
        {"n": 3, "m": 1, "lst_nid": nids, "bound_x": 6, "bound_c": 4, "pairing_group_param": GROUP}
    )
    kg.setup()
    pp = kg.get_public_parameters()
    x = {"u0": [5], "u1": [-3], "u2": [6]}
    cts = {nid: MultiInputQuadratic({"keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
           for nid in nids}
    coeffs = [[0, 2, -4], [0, 0, 3], [0, 0, 0]]
    dk = kg.get_decryption_keys(None, credentials={"coefficients": coeffs})
    assert _dec(pp, cts, dk) == 2 * 5 * -3 - 4 * 5 * 6 + 3 * -3 * 6
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(None, credentials={"coefficients": [[0, 0, 0], [1, 0, 0], [0, 0, 0]]})
