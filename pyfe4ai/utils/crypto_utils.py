"""Core cryptographic primitives: safe random generation, group/prime generation, and hashing."""

import random
import hashlib

import gmpy2 as gp

_CSPRNG = random.SystemRandom()


def _random(maximum: gp.mpz, bits: int = 256):
    """generate random integer with specified bits

    Args:
        maximum (gmpy2.mpz): the maximum threshold of generated integer
        bits (int, optional): the specified bit size of generated integer.
            Defaults to 256.

    Returns:
        gmpy2.mpz: the generated random big integer
    """
    r = gp.mpz(_CSPRNG.getrandbits(bits))
    while r > maximum:
        r = gp.mpz(_CSPRNG.getrandbits(bits))
    return r


def _random_prime(bits):
    r = gp.mpz(_CSPRNG.getrandbits(bits))
    r = gp.bit_set(r, bits - 1)
    return gp.next_prime(r)


def generate_group_primes(bits: int) -> tuple:
    """
    Generates two safe prime numbers with the restriction:p=2q+1.

    Args:
        bits (int): security parameters

    Returns:
        tuple: p, q
    """
    while True:
        p_prime = _random_prime(bits - 1)
        p = p_prime * 2 + 1
        if gp.is_prime(p) and gp.is_prime(p_prime):
            break
    return p, p_prime


def _param_generator(bits: int, r: int = 2):
    """Perform the _param_generator operation.

        Args:
            bits: Security parameter (bit length).
            r: Cofactor (default 2).
    """
    while True:
        q = _random_prime(bits - 1)
        p = q * 2 + 1
        if gp.is_prime(p) and gp.is_prime(q):
            break
    return p, q, r


def _g_generator(bits, p, r):
    """Perform the _g_generator operation.

        Args:
            bits: Security parameter (bit length).
            p: Prime modulus.
            r: Cofactor (default 2).
    """
    while True:
        h = _random(p, bits)
        g = gp.powmod(h, r, p)
        if not g == 1:
            break
    return g


def group_generator_threshold_fe(bits: int, r: int = 2):

    """Generate a safe prime and generator for threshold FE.

        Args:
            bits: Security parameter (bit length).
            r: Cofactor (default 2).
    """
    p, _, r = _param_generator(bits, r=r)
    g = _g_generator(bits, p, r)
    return p, g


def group_generator_paillier(bits: int) -> tuple:
    """
     Generates an integer group (p,q,n),
     where n = pq, and p, q are prime numbers.

    Args:
        bits (int): the length of the prime number.

    Returns:
        tuple: p, q, n
    """
    while True:
        p, q = _random_prime(bits), _random_prime(bits)
        if gp.is_prime(p) and gp.is_prime(q) and gp.gcd(p * q, (p - 1) * (q - 1)) == 1:
            break
    n = p * q
    return p, q, n


def group_generator_fe(bits: int, r: int = 2) -> tuple:
    """Generate an RSA modulus and generator for FE schemes.

        Args:
            bits: Security parameter (bit length).
            r: Cofactor (default 2).
    """
    p, q, r = _param_generator(bits, r=r)
    g = _g_generator(bits, p, r)
    return p, q, r, g


def md5_hash(v: str, p: gp.mpz) -> gp.mpz:
    """Compute the MD5 hash of a string and return the result modulo p.

        Args:
            v: Input string.
            p: Prime modulus.
    """
    return gp.mpz(hashlib.md5(v.encode("utf-8")).hexdigest(), 16) % p
