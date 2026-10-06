"""
Shared threshold secret-sharing utilities for threshold FE schemes.

Provides polynomial evaluation, Shamir secret sharing, Lagrange
interpolation and modular centering used identically by both the
LWE and Ring-LWE threshold variants.
"""

from __future__ import annotations

import random

import gmpy2 as gp

from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

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


def lagrange_coefficient_at_zero(
    x_j: int, xs: list[int], modulus: gp.mpz
) -> gp.mpz:
    """Return the Lagrange basis coefficient of point *x_j* evaluated at 0.

    Computes ``prod_{m != j} x_m / (x_m - x_j)`` modulo *modulus* with a
    modular inverse, so *modulus* must be prime (or at least coprime to every
    difference) and the points in *xs* must be distinct and non-zero.

    Args:
        x_j: Evaluation point whose coefficient is requested.
        xs: All evaluation points taking part in the reconstruction.
        modulus: Modulus for the arithmetic operation.
    """
    num = gp.mpz(1)
    den = gp.mpz(1)
    for x_m in xs:
        if x_m == x_j:
            continue
        num = (num * gp.mpz(x_m)) % modulus
        den = (den * gp.mpz(x_m - x_j)) % modulus
    return (num * gp.invert(den % modulus, modulus)) % modulus


def lagrange_at_zero(shares: dict[int, gp.mpz], modulus: gp.mpz) -> gp.mpz:
    """Reconstruct the secret (polynomial value at 0) from *shares*.

    Args:
        shares: List of ``(x, y)`` share pairs.
        modulus: Modulus for the arithmetic operation.
    """
    result = gp.mpz(0)
    xs = list(shares.keys())
    for x_j in xs:
        lagrange = lagrange_coefficient_at_zero(x_j, xs, modulus)
        result = (result + gp.mpz(shares[x_j]) * lagrange) % modulus
    return result


def share_points(lst_sid: list[str]) -> dict[str, int]:
    """Map each decryption server to its Shamir evaluation point.

    Points start at 1: the polynomial value at 0 is the shared secret itself,
    so no server may ever hold the share at x = 0.

    Args:
        lst_sid: Ordered list of all decryption-server identifiers.
    """
    return {sid: idx + 1 for idx, sid in enumerate(lst_sid)}


def validate_threshold(threshold: int, lst_sid: list[str]) -> None:
    """Reject thresholds outside ``1 <= t <= s``.

    Args:
        threshold: Minimum number of shares needed for reconstruction.
        lst_sid: Ordered list of all decryption-server identifiers.
    """
    if not isinstance(threshold, int) or not 1 <= threshold <= len(lst_sid):
        raise FEValidationError(
            "invalid threshold t={} for s={} decryption servers".format(
                threshold, len(lst_sid)
            )
        )


def validate_enrolled(
    sid: str, lst_sid: list[str], lst_sid_enrolled: list[str], threshold: int
) -> None:
    """Check that *sid* may produce a partial decryption for *lst_sid_enrolled*.

    Args:
        sid: Identifier of the server producing the partial decryption.
        lst_sid: Ordered list of all decryption-server identifiers.
        lst_sid_enrolled: Servers whose shares will be combined.
        threshold: Minimum number of shares needed for reconstruction.
    """
    if len(set(lst_sid_enrolled)) != len(lst_sid_enrolled):
        raise FESchemeError(
            "duplicate server ids in enrolled set: {}".format(lst_sid_enrolled)
        )
    unknown = [s for s in lst_sid_enrolled if s not in lst_sid]
    if unknown:
        raise FESchemeError("unknown server ids in enrolled set: {}".format(unknown))
    if sid not in lst_sid_enrolled:
        raise FESchemeError(
            "server {} is not part of the enrolled set {}".format(
                sid, lst_sid_enrolled
            )
        )
    if len(lst_sid_enrolled) < threshold:
        raise FESchemeError(
            "insufficient threshold shares: {} enrolled, need at least t={}".format(
                len(lst_sid_enrolled), threshold
            )
        )


def validate_partial_decryptions(dict_ct_prime: dict, threshold: int) -> None:
    """Check a set of partial decryptions before combining them.

    Each partial decryption carries the server id that produced it and the
    enrolled set its Lagrange coefficient was computed for; the contributors
    must be exactly that enrolled set and number at least *threshold*.

    Args:
        dict_ct_prime: Mapping of server ids to partial decryptions.
        threshold: Minimum number of shares needed for reconstruction.
    """
    if len(dict_ct_prime) < threshold:
        raise FESchemeError(
            "insufficient threshold shares for decryption: got {}, need at "
            "least t={}".format(len(dict_ct_prime), threshold)
        )
    contributors = set()
    enrolled = None
    for ct_prime in dict_ct_prime.values():
        if "sid" not in ct_prime or "lst_sid_enrolled" not in ct_prime:
            raise FESchemeError(
                "partial decryption is missing its server id / enrolled set"
            )
        contributors.add(ct_prime["sid"])
        _enrolled = set(ct_prime["lst_sid_enrolled"])
        if enrolled is None:
            enrolled = _enrolled
        elif enrolled != _enrolled:
            raise FESchemeError(
                "partial decryptions were computed for different enrolled sets"
            )
    if contributors != enrolled:
        raise FESchemeError(
            "partial decryptions from {} do not match the enrolled set {}".format(
                sorted(contributors), sorted(enrolled)
            )
        )


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
