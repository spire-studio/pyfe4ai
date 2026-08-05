"""
Shared threshold secret-sharing utilities for threshold FE schemes.

Provides polynomial evaluation, Shamir secret sharing, Lagrange
interpolation and modular centering used identically by both the
LWE and Ring-LWE threshold variants.
"""

from __future__ import annotations

import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()


def eval_poly(coeffs: list[gp.mpz], x_value: int, modulus: gp.mpz) -> gp.mpz:
    """Evaluate a polynomial at *x_value* modulo *modulus*.

    Args:
        coeffs: Polynomial coefficient list.
        x_value: Evaluation point.
        modulus: Modulus for the arithmetic operation.
    """
    acc = gp.mpz(0)
    x = gp.mpz(x_value)
    power = gp.mpz(1)
    for coeff in coeffs:
        acc = (acc + coeff * power) % modulus
        power = (power * x) % modulus
    return acc


def share_secret(
    secret: gp.mpz, threshold: int, share_points: list[int], modulus: gp.mpz
) -> dict[int, gp.mpz]:
    """Create Shamir secret shares of *secret*.

    Args:
        secret: The secret to share.
        threshold: Minimum number of shares needed for reconstruction.
        share_points: Evaluation points for the shares.
        modulus: Modulus for the arithmetic operation.
    """
    coeffs = [gp.mpz(secret) % modulus] + [
        gp.mpz(_CSPRNG.randrange(0, int(modulus)))
        for _ in range(1, threshold)
    ]
    return {x: eval_poly(coeffs, x, modulus) for x in share_points}


def lagrange_at_zero(shares: dict[int, gp.mpz], modulus: gp.mpz) -> gp.mpz:
    """Reconstruct the secret (polynomial value at 0) from *shares*.

    Args:
        shares: List of ``(x, y)`` share pairs.
        modulus: Modulus for the arithmetic operation.
    """
    result = gp.mpz(0)
    xs = list(shares.keys())
    for x_j in xs:
        num = gp.mpz(1)
        den = gp.mpz(1)
        for x_m in xs:
            if x_m == x_j:
                continue
            num = (num * gp.mpz(x_m)) % modulus
            den = (den * gp.mpz(x_m - x_j)) % modulus
        lagrange = (num * gp.invert(den % modulus, modulus)) % modulus
        result = (result + gp.mpz(shares[x_j]) * lagrange) % modulus
    return result


def center_mod(value: gp.mpz, modulus: gp.mpz) -> gp.mpz:
    """Center *value* into the range ``(-modulus/2, modulus/2]``.

    Args:
        value: Input value.
        modulus: Modulus for the arithmetic operation.
    """
    value %= modulus
    half = modulus // 2
    if value > half:
        value -= modulus
    return value
