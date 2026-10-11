"""LWE plaintext encodings must be linear: floor(q/p) x mod q.

ABDP15 needs the linear ciphertext homomorphism of its PKE and BMMS21's last
game needs floor(q/K) (x0 + u alpha); trunc(x q / p) differs from
floor(q/p) x by a small, non-linear term."""

import gmpy2 as gp
import pytest

from pyfe4ai.utils.lwe_utils import center_lwe_vector
from pyfe4ai.utils.ring_lwe_utils import center_matrix

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
