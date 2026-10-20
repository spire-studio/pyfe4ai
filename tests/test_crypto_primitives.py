"""Hashing into Z_q^length and batched fixed-base exponentiation."""

import hashlib
import random

import gmpy2 as gp
import pytest

from pyfe4ai.primitives.hashing import hash_to_zq_vector
from pyfe4ai.primitives.groups.modular_utils import pow_signed, pow_signed_many


def test_hash_to_zq_vector_reduces_one_shake256_stream():
    q = gp.next_prime(gp.mpz(2) ** 130)
    v, domain, length = "label", "domain", 5
    n_bytes = (int(q).bit_length() + 128 + 7) // 8
    material = "|".join([domain, "zq-vector", str(len(v)), v]).encode("utf-8")
    stream = hashlib.shake_256(material).digest(length * n_bytes)
    expected = [
        int.from_bytes(stream[k * n_bytes:(k + 1) * n_bytes], "big") % q for k in range(length)
    ]
    assert hash_to_zq_vector(v, q, length, domain=domain) == expected


def test_hash_to_zq_vector_separates_domains_and_inputs():
    q = gp.next_prime(gp.mpz(2) ** 64)
    base = hash_to_zq_vector("a", q, 4, domain="d")
    assert hash_to_zq_vector("a", q, 4, domain="e") != base
    assert hash_to_zq_vector("b", q, 4, domain="d") != base
    assert all(0 <= h < q for h in base)


@pytest.mark.parametrize("bits", [64, 700])
def test_pow_signed_many_matches_pow_signed(bits):
    rng = random.Random(bits)
    modulus = gp.mpz(rng.getrandbits(bits) | 1) ** 2
    base = gp.mpz(rng.getrandbits(bits)) | 1
    while gp.gcd(base, modulus) != 1:
        base += 2
    exponents = [0, 1, -1, 2 ** bits, -(2 ** bits) + 1] + [
        rng.randint(-(2 ** (2 * bits)), 2 ** (2 * bits)) for _ in range(30)
    ]
    assert pow_signed_many(base, exponents, modulus) == [
        pow_signed(base, e, modulus) for e in exponents
    ]


def test_pow_signed_many_of_nothing():
    assert pow_signed_many(gp.mpz(3), [], gp.mpz(35)) == []


@pytest.mark.parametrize("count", [1, 20])
def test_pow_signed_many_fails_like_pow_signed_for_a_non_invertible_base(count):
    with pytest.raises(ZeroDivisionError):
        pow_signed(gp.mpz(6), gp.mpz(-5), gp.mpz(9))
    with pytest.raises(ZeroDivisionError):
        pow_signed_many(gp.mpz(6), [-5] * count, gp.mpz(9))
