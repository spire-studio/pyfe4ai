"""Random sampling utilities: uniform vectors/matrices and discrete Gaussians."""

from __future__ import annotations

import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()


def random_below(maximum: gp.mpz) -> gp.mpz:
    upper = int(max(2, int(maximum)))
    return gp.mpz(_CSPRNG.randrange(1, upper))


def rand_uniform_matrix(rows: int, cols: int, modulus: gp.mpz) -> list[list[gp.mpz]]:
    """Generate a random matrix with entries uniform in [0, modulus).

        Args:
            rows: Number of rows.
            cols: Number of columns.
            modulus: Modulus for the arithmetic operation.
    """
    q_int = int(modulus)
    return [[gp.mpz(_CSPRNG.randrange(q_int)) for _ in range(cols)] for _ in range(rows)]


def rand_uniform_vector(length: int, modulus: gp.mpz) -> list[gp.mpz]:
    """Generate a random vector with entries uniform in [0, modulus).

        Args:
            length: Length of the vector.
            modulus: Modulus for the arithmetic operation.
    """
    q_int = int(modulus)
    return [gp.mpz(_CSPRNG.randrange(q_int)) for _ in range(length)]


def rand_bit_vector(length: int) -> list[gp.mpz]:
    return [gp.mpz(_CSPRNG.randrange(2)) for _ in range(length)]


def discrete_gaussian_matrix(rows: int, cols: int, sigma: float) -> list[list[gp.mpz]]:
    """Sample a matrix from the (rounded) discrete Gaussian distribution.

    Samples are drawn from a CSPRNG (``random.SystemRandom``, i.e. OS entropy)
    and rounded to the nearest integer, so this sampler is suitable for LWE
    noise and secret-key material. Note it is a rounded *continuous* Gaussian
    rather than a true discrete-Gaussian sampler, and it is not constant-time;
    this is adequate for a research prototype but not for production
    side-channel resistance.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            sigma: Standard deviation for the Gaussian distribution.
    """
    return [
        [gp.mpz(int(round(_CSPRNG.gauss(0.0, sigma)))) for _ in range(cols)]
        for _ in range(rows)
    ]
