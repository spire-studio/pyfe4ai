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


def discrete_gaussian_bigint(sigma, count: int) -> list[gp.mpz]:
    """Sample *count* integers from a rounded Gaussian of arbitrary width.

    Unlike :func:`discrete_gaussian_matrix`, *sigma* may exceed the range of
    a double (e.g. the ``sqrt(kappa) * N^{5/2}`` widths of the DCR schemes):
    a Box-Muller sample is computed with ``gmpy2.mpfr`` at
    ``bits(sigma) + 128`` bits of precision from CSPRNG uniforms and rounded
    to the nearest integer. As with :func:`discrete_gaussian_matrix` this is
    a rounded continuous Gaussian, not an exact discrete-Gaussian sampler,
    and it is not constant-time.

        Args:
            sigma: Standard deviation (int, mpz, float or mpfr), > 0.
            count: Number of samples.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    prec = int(gp.ceil(gp.mpfr(sigma))).bit_length() + 128
    samples = []
    with gp.context(gp.get_context(), precision=prec):
        two_pi = 2 * gp.const_pi()
        sig = gp.mpfr(sigma)
        scale = gp.mpfr(2) ** prec
        for _ in range(count):
            u1 = (gp.mpfr(_CSPRNG.getrandbits(prec)) + 1) / scale  # (0, 1]
            u2 = gp.mpfr(_CSPRNG.getrandbits(prec)) / scale  # [0, 1)
            z = gp.sqrt(-2 * gp.log(u1)) * gp.cos(two_pi * u2)
            samples.append(gp.mpz(gp.rint(z * sig)))
    return samples
