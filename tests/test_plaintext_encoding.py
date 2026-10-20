"""LWE plaintext encodings must be linear: floor(q/p) x mod q.

ABDP15 needs the linear ciphertext homomorphism of its PKE and BMMS21's last
game needs floor(q/K) (x0 + u alpha); trunc(x q / p) differs from
floor(q/p) x by a small, non-linear term."""

import gmpy2 as gp
import pytest

from pyfe4ai.primitives.lattice.lwe_utils import center_lwe_vector
from pyfe4ai.primitives.lattice.ring_lwe_utils import center_matrix

Q = gp.next_prime(gp.mpz(2) ** 70)
P = gp.mpz(98)  # q mod p != 0, so the two encodings differ


@pytest.mark.parametrize("x", [-4, -1, 1, 3, 4])
def test_lwe_vector_encoding_is_floor_q_over_p_times_x(x):
    assert center_lwe_vector([x], P, Q) == [Q // P * x % Q]


def test_lwe_vector_encoding_is_linear():
    a, b = center_lwe_vector([3, -4], P, Q)
    assert (a + b) % Q == center_lwe_vector([-1], P, Q)[0]


def test_ring_matrix_encoding_is_floor_q_over_p_times_x():
    rows = center_matrix([[3, -4], [4, 0]], P, Q, 4)
    assert rows == [[Q // P * 3 % Q, Q // P * -4 % Q, 0, 0], [Q // P * 4 % Q, 0, 0, 0]]


# -- real-valued plaintexts of the ndarray helpers ----------------------------

import numpy as np

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai._validation import scale_to_int


def test_scaling_rounds_instead_of_truncating():
    # 0.57 * 100 = 56.99999999999999: truncation toward zero gave 56
    assert scale_to_int(np.array([0.57, -0.57, 0.29]), 2).tolist() == [57, -57, 29]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_scaling_refuses_non_finite_values(bad):
    # astype(int) turned them into -2^63, which some helpers then encrypted
    with pytest.raises(FEValidationError, match="NaN or infinit"):
        scale_to_int(np.array([1.0, bad]), 2)


def test_scaling_keeps_integers_exact_and_refuses_overflow():
    assert scale_to_int(np.array([2 ** 50, -3]), 3).tolist() == [2 ** 50 * 1000, -3000]
    with pytest.raises(FEValidationError, match="too large"):
        scale_to_int(np.array([1e300]), 0)
    with pytest.raises(FEValidationError, match="too large"):
        scale_to_int(np.array([2 ** 62]), 1)


def test_ndarray_helpers_round_and_refuse_nan(tmp_path, monkeypatch):
    from pyfe4ai.schemes.mcfe.ddh import MCFE, MCFEKeyGenerator

    monkeypatch.chdir(tmp_path)
    nids = ["nid_0", "nid_1"]
    kg = MCFEKeyGenerator({"sec_param": 128, "eta": 1, "n": 2, "lst_nid": nids})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: MCFE({"id": nid, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    cts = {nid: c.encrypt_lst_ndarray([np.array([0.57, -0.29])], label="r") for nid, c in clients.items()}
    dct_y = {nid: [1] for nid in nids}
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": dct_y})
    out = MCFE({"id": "sid", "precision": 2, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(cts, dk, dct_y, "r")
    assert np.allclose(out[0], [1.14, -0.58])
    with pytest.raises(FEValidationError, match="NaN"):
        clients["nid_0"].encrypt_lst_ndarray([np.array([np.nan])], label="r2")


def test_scaling_refuses_a_negative_precision():
    with pytest.raises(FEValidationError, match="precision"):
        scale_to_int(np.array([1.5]), -1)
