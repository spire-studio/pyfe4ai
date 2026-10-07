"""LWE parameter derivation, vector centering, and inner-product decoding."""

from __future__ import annotations

import math
import random

import gmpy2 as gp

from pyfe4ai.utils.exceptions import FEValidationError

_CSPRNG = random.SystemRandom()


SIGMA_CDT = math.sqrt(1.0 / (2.0 * math.log(2.0)))


def derive_lwe_parameters(
    max_plain_bound: gp.mpz,
    bound_y: gp.mpz,
    eta: int,
    lwe_n: int,
    extra_dimension: int | None = None,
) -> tuple[gp.mpz, gp.mpz, int, float, gp.mpz]:
    """Derive LWE scheme parameters from plaintext and weight bounds.

        Args:
            max_plain_bound: Maximum plaintext bound.
            bound_y: Bound on weight values.
            eta: Inner-product vector dimension.
            lwe_n: LWE lattice dimension.
            extra_dimension: Extra dimension flag.
    """
    bound_x_bits = max(1, int(max_plain_bound).bit_length())
    bound_y_bits = max(1, int(bound_y).bit_length())
    p_bits = bound_x_bits + bound_y_bits + max(1, int(eta).bit_length()) + 2
    if extra_dimension is not None:
        p_bits += max(1, int(extra_dimension).bit_length())
    p = gp.next_prime(gp.mpz(_CSPRNG.getrandbits(p_bits)) + (1 << (p_bits - 1)))

    val = float(max_plain_bound) * math.sqrt(eta) + 1.0
    x = val * float(p) * float(bound_y)
    x *= 8.0 * lwe_n * math.sqrt(lwe_n + eta + 1.0)
    x *= math.sqrt(x)
    q_bits = max(p_bits + 2, int(gp.mpz(int(x)).bit_length()) + 1)
    q = gp.next_prime(gp.mpz(_CSPRNG.getrandbits(q_bits)) + (1 << (q_bits - 1)))

    m = (lwe_n + eta + 1) * q_bits + 2 * lwe_n + 1
    sigma = 1.0 / (2.0 * math.sqrt(2.0 * eta * m * lwe_n))
    sigma /= float(p)
    sigma /= float(bound_y)
    sigma_q = sigma * float(q)
    l_sigma = gp.mpz(int(sigma_q / SIGMA_CDT) + 1)
    sigma_q = float(l_sigma) * SIGMA_CDT
    return p, q, m, sigma_q, l_sigma


def derive_fullysec_lwe_parameters(
    bound_x: gp.mpz,
    bound_y: gp.mpz,
    eta: int,
    lwe_n: int,
) -> tuple[gp.mpz, gp.mpz, int, float, gp.mpz, float, gp.mpz, float, gp.mpz]:
    """Derive fully-secure LWE parameters with multiple noise levels.

        Args:
            bound_x: Bound on plaintext values.
            bound_y: Bound on weight values.
            eta: Inner-product vector dimension.
            lwe_n: LWE lattice dimension.
    """
    k = gp.mpz(2) * gp.mpz(eta) * gp.mpz(bound_x) * gp.mpz(bound_y)
    k_float = float(k)
    squared = k_float * k_float
    n_float = float(lwe_n)

    n_bits_q = 1
    sigma = 1.0
    sigma1 = 1.0
    sigma2 = 1.0
    l_sigma1 = gp.mpz(1)
    l_sigma2 = gp.mpz(1)
    i = 1
    while True:
        bound_m_float = max(1.0, n_float * float(i))
        log2_m = math.log2(bound_m_float)
        sqrt_n_log_m = math.sqrt(n_float * log2_m)
        max_float = max(squared, bound_m_float)
        sqrt_max = math.sqrt(max_float)

        sigma1 = sqrt_n_log_m * sqrt_max
        l_sigma1 = gp.mpz(int(sigma1 / SIGMA_CDT) + 1)
        sigma1 = float(l_sigma1) * SIGMA_CDT

        mul_val = math.sqrt(n_float) * math.pow(n_float, 3) * math.pow(math.sqrt(log2_m), 5)
        mul_val *= math.sqrt(bound_m_float)
        sigma2 = mul_val * max_float
        l_sigma2 = gp.mpz(int(sigma2 / SIGMA_CDT) + 1)
        sigma2 = float(l_sigma2) * SIGMA_CDT

        bound2 = math.sqrt(sigma1 * sigma1 + sigma2 * sigma2)
        bound2 *= math.sqrt(n_float)
        sigma = 1.0 / squared
        sigma /= bound2
        sigma /= math.log2(n_float)

        sigma_prime = sigma / k_float
        sigma_prime /= math.pow(n_float, 6) * math.pow(float(n_bits_q), 2)
        sigma_prime /= math.pow(math.sqrt(math.log2(n_float)), 5)

        bound_for_q = math.sqrt(math.log2(n_float)) / sigma_prime
        next_bits_q = max(2, int(math.floor(math.log2(bound_for_q))) + 2)
        if next_bits_q < i:
            break
        n_bits_q = next_bits_q
        i = next_bits_q + 1

    q = gp.next_prime(gp.mpz(_CSPRNG.getrandbits(n_bits_q)) + gp.mpz(1 << (n_bits_q - 1)))
    m = max(eta + 2, int(1.01 * n_float * float(n_bits_q)))
    q_float = float(q)
    sigma_q = sigma * q_float
    l_sigma_q = gp.mpz(int(sigma_q / SIGMA_CDT) + 1)
    sigma_q = float(l_sigma_q) * SIGMA_CDT
    return k, q, m, sigma_q, l_sigma_q, sigma1, l_sigma1, sigma2, l_sigma2


def center_lwe_vector(vector: list[int] | list[gp.mpz], p: gp.mpz, q: gp.mpz) -> list[gp.mpz]:
    """Scale and center a vector for LWE message extraction.

        Args:
            vector: Input vector.
            p: Prime modulus.
            q: Modulus (or second prime).
    """
    ret = []
    for x in vector:
        t = gp.mpz(x) * q
        t = gp.t_div(t, p)
        t %= q
        ret.append(t)
    return ret


def decode_lwe_inner_product(value: gp.mpz, p: gp.mpz, q: gp.mpz) -> int:
    """Extract a plaintext value from an LWE inner-product ciphertext.

        Args:
            value: Input value.
            p: Prime modulus.
            q: Modulus (or second prime).
    """
    d = gp.mpz(value) % q
    half_q = q // 2
    if d > half_q:
        d -= q
    d = d * p
    d = (d + half_q) // q
    return int(d)


def label_scalar_from_hash(hash_value: gp.mpz, label_modulus: int) -> gp.mpz:
    """Convert a hash value to a non-zero centered label scalar.

    The scalar lies in ``[-(M - 1 - M // 2), M // 2]`` excluding 0, for
    ``M = label_modulus``. Zero is excluded because a zero scalar removes the
    per-client label mask ``u * l`` altogether, letting any functional-key
    holder decode each client's plaintext.

        Args:
            hash_value: Hash digest value.
            label_modulus: Modulus for centering the label scalar (>= 2).

        Raises:
            FEValidationError: If ``label_modulus < 2``.
    """
    label_modulus = int(label_modulus)
    if label_modulus < 2:
        raise FEValidationError(
            "label_modulus must be >= 2, got {}".format(label_modulus)
        )
    raw = int(hash_value % gp.mpz(label_modulus - 1)) + 1
    half = label_modulus // 2
    if raw > half:
        raw -= label_modulus
    return gp.mpz(raw)


def decode_fullysec_lwe_inner_product(value: gp.mpz, k: gp.mpz, q: gp.mpz) -> int:
    """Extract a plaintext from a fully-secure LWE ciphertext.

        Args:
            value: Input value.
            k: Inner-product bound.
            q: Modulus (or second prime).
    """
    d = gp.mpz(value) % q
    half_q = q // 2
    if d > half_q:
        d -= q
    q_div_k = q // k
    q_div_2k = q // (k * 2)
    return int((d + q_div_2k) // q_div_k)
