"""Modular arithmetic helpers for signed-exponent modular exponentiation."""

from __future__ import annotations

import gmpy2 as gp


def pow_signed(base: gp.mpz, exponent: gp.mpz, modulus: gp.mpz) -> gp.mpz:
    """Compute modular exponentiation with signed exponent support.

        Args:
            base: Base for modular exponentiation.
            exponent: Exponent value.
            modulus: Modulus for the arithmetic operation.
    """
    exp = gp.mpz(exponent)
    if exp >= 0:
        return gp.powmod(base, exp, modulus)
    inv = gp.invert(base, modulus)
    if inv == 0:
        raise ValueError("base is not invertible modulo modulus")
    return gp.powmod(inv, -exp, modulus)
