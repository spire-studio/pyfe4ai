"""Tests for NTT-based polynomial multiplication in ring_lwe_utils."""

import gmpy2 as gp
import pytest

from pyfe4ai.utils.ring_lwe_utils import (
    _find_primitive_root,
    ntt_forward,
    ntt_inverse,
    poly_mul_negacyclic,
    poly_mul_ntt,
)


# NTT-friendly prime: q ≡ 1 (mod 2n) so a primitive 2n-th root of unity exists.
# For n=8:  2n=16, need q ≡ 1 (mod 16).  q=17 works (17-1=16).
# For n=16: 2n=32, need q ≡ 1 (mod 32).  q=97 works (97-1=96, 96/32=3).
# For n=64: 2n=128, need q ≡ 1 (mod 128). q=257 works (257-1=256, 256/128=2).

Q_17 = gp.mpz(17)
Q_97 = gp.mpz(97)
Q_257 = gp.mpz(257)
# Larger NTT-friendly prime for n=256: q=7681 (7680 = 256*30, and 7680/512=15).
Q_7681 = gp.mpz(7681)


class TestFindPrimitiveRoot:
    def test_root_exists_n8(self):
        root = _find_primitive_root(Q_17, 16)
        assert root is not None
        assert gp.powmod(root, 16, Q_17) == 1
        assert gp.powmod(root, 8, Q_17) != 1

    def test_root_exists_n16(self):
        root = _find_primitive_root(Q_97, 32)
        assert root is not None
        assert gp.powmod(root, 32, Q_97) == 1
        assert gp.powmod(root, 16, Q_97) != 1

    def test_root_exists_n64(self):
        root = _find_primitive_root(Q_257, 128)
        assert root is not None
        assert gp.powmod(root, 128, Q_257) == 1
        assert gp.powmod(root, 64, Q_257) != 1

    def test_no_root(self):
        # q=13, order=16.  13-1=12, 12 % 16 != 0 → no root
        assert _find_primitive_root(gp.mpz(13), 16) is None


class TestNTTRoundTrip:
    """Verify that forward + inverse NTT is identity."""

    @pytest.mark.parametrize("n, q", [(8, Q_17), (16, Q_97), (64, Q_257)])
    def test_roundtrip(self, n, q):
        root = _find_primitive_root(q, 2 * n)
        assert root is not None
        a = [gp.mpz(i + 1) for i in range(n)]
        a_ntt = ntt_forward(a, q, root)
        a_back = ntt_inverse(a_ntt, q, root)
        assert a_back == [(x % q) for x in a]

    def test_roundtrip_zeros(self):
        n, q = 8, Q_17
        root = _find_primitive_root(q, 16)
        a = [gp.mpz(0)] * n
        assert ntt_inverse(ntt_forward(a, q, root), q, root) == a


class TestPolyMulNTTCorrectness:
    """Verify NTT multiplication matches naive negacyclic multiplication."""

    @pytest.mark.parametrize("n, q", [(8, Q_17), (16, Q_97), (64, Q_257)])
    def test_matches_naive(self, n, q):
        root = _find_primitive_root(q, 2 * n)
        assert root is not None
        a = [gp.mpz((3 * i + 1) % int(q)) for i in range(n)]
        b = [gp.mpz((7 * i + 2) % int(q)) for i in range(n)]
        expected = poly_mul_negacyclic(a, b, q)
        result = poly_mul_ntt(a, b, q, root)
        assert result == expected

    def test_multiply_by_one(self):
        """Multiplying by polynomial 1 (= [1, 0, 0, ...]) should be identity."""
        n, q = 8, Q_17
        root = _find_primitive_root(q, 16)
        a = [gp.mpz(i) for i in range(n)]
        one = [gp.mpz(1)] + [gp.mpz(0)] * (n - 1)
        result = poly_mul_ntt(a, one, q, root)
        assert result == [(x % q) for x in a]

    def test_multiply_by_x(self):
        """Multiplying by X (= [0, 1, 0, ...]) in X^n+1 ring negates the last coeff."""
        n, q = 8, Q_17
        root = _find_primitive_root(q, 16)
        a = [gp.mpz(i + 1) for i in range(n)]
        x_poly = [gp.mpz(0), gp.mpz(1)] + [gp.mpz(0)] * (n - 2)
        result_ntt = poly_mul_ntt(a, x_poly, q, root)
        result_naive = poly_mul_negacyclic(a, x_poly, q)
        assert result_ntt == result_naive

    def test_random_large(self):
        """Test with n=256 and a larger prime."""
        n = 256
        q = Q_7681
        root = _find_primitive_root(q, 2 * n)
        assert root is not None
        import random
        rng = random.Random(42)
        a = [gp.mpz(rng.randint(0, int(q) - 1)) for _ in range(n)]
        b = [gp.mpz(rng.randint(0, int(q) - 1)) for _ in range(n)]
        expected = poly_mul_negacyclic(a, b, q)
        result = poly_mul_ntt(a, b, q, root)
        assert result == expected

    def test_commutativity(self):
        """NTT mul should be commutative: a*b == b*a."""
        n, q = 16, Q_97
        root = _find_primitive_root(q, 32)
        a = [gp.mpz((5 * i) % int(q)) for i in range(n)]
        b = [gp.mpz((11 * i + 3) % int(q)) for i in range(n)]
        assert poly_mul_ntt(a, b, q, root) == poly_mul_ntt(b, a, q, root)
