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


#: Digit size (bits) of the fixed-base method of :func:`pow_signed_many`.
FIXED_BASE_WINDOW = 6
#: Fewer exponents than this are raised one by one with :func:`pow_signed`.
FIXED_BASE_MIN_COUNT = 16


def pow_signed_many(base: gp.mpz, exponents, modulus: gp.mpz) -> list:
    """Return ``[base^e mod modulus for e in exponents]``; ``e`` may be negative.

    For many exponents of one base (an MCFE2 encryption of a key contribution
    raises one label hash to every coordinate of the key) the powers
    ``base^(2^(w i))`` are computed once; each exponentiation then multiplies
    the powers selected by its base-``2^w`` digits into ``2^w - 1`` buckets
    and combines them (Brickell, Gordon, McCurley, Wilson, EUROCRYPT 1992,
    with the bucket method), about ``bits(e) / w + 2^(w + 1)`` products
    instead of ``bits(e)`` squarings. A negative exponent inverts the power
    of ``|e|``. The results equal those of :func:`pow_signed`.

    Args:
        base: Base, invertible modulo *modulus* if an exponent is negative.
        exponents: Integer exponents.
        modulus: Modulus.
    """
    base, modulus = gp.mpz(base), gp.mpz(modulus)
    exponents = [gp.mpz(e) for e in exponents]
    if len(exponents) < FIXED_BASE_MIN_COUNT:
        return [pow_signed(base, e, modulus) for e in exponents]
    w = FIXED_BASE_WINDOW
    mask = (1 << w) - 1
    n_digits = (max(abs(e) for e in exponents).bit_length() + w - 1) // w
    powers = [base % modulus]
    for _ in range(1, n_digits):
        powers.append(gp.powmod(powers[-1], 1 << w, modulus))
    out = []
    for e in exponents:
        buckets = [gp.mpz(1)] * (mask + 1)
        rest, i = abs(e), 0
        while rest:
            digit = int(rest & mask)
            if digit:
                buckets[digit] = buckets[digit] * powers[i] % modulus
            rest >>= w
            i += 1
        acc, result = gp.mpz(1), gp.mpz(1) % modulus
        for digit in range(mask, 0, -1):
            acc = acc * buckets[digit] % modulus
            result = result * acc % modulus
        out.append(gp.invert(result, modulus) if e < 0 else result)
    return out


def check_zn2_elements(values, n_square) -> list:
    """Parse ciphertext elements of ``Z_{N^2}`` and reject values outside ``(0, N^2)``."""
    from pyfe4ai.utils.exceptions import FEValidationError

    parsed = [gp.mpz(v) for v in values]
    if any(not 0 < v < n_square for v in parsed):
        raise FEValidationError("ciphertext element outside (0, N^2)")
    return parsed


def decode_paillier(acc, n_mod, bound) -> int | None:
    """Return ``mu`` with ``acc = (1 + N)^mu mod N^2``, centred, or ``None``.

    ``None`` when *acc* is not of that form or ``|mu| > bound``: a decryption
    of a crafted ciphertext must not hand out a key-dependent value (e.g.
    ``ct0 = 1 + N`` turns ``ct0^{-k}`` into ``-k mod N``).
    """
    n_mod = gp.mpz(n_mod)
    if (gp.mpz(acc) - 1) % n_mod != 0:
        return None
    mu = ((gp.mpz(acc) - 1) % (n_mod * n_mod)) // n_mod
    if mu > n_mod // 2:
        mu -= n_mod
    return int(mu) if abs(mu) <= bound else None
