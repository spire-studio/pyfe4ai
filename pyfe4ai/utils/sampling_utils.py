"""Random sampling utilities: uniform vectors/matrices and discrete Gaussians."""

from __future__ import annotations

import hashlib
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


# Above this width a double-precision sample no longer resolves the integers
# (53-bit mantissa, samples up to ~2^4 sigma): its low bits would be constant.
_DOUBLE_SAFE_SIGMA = 2 ** 40


def discrete_gaussian_matrix(rows: int, cols: int, sigma: float) -> list[list[gp.mpz]]:
    """Sample a matrix from the (rounded) discrete Gaussian distribution.

    Samples are drawn from a CSPRNG (``random.SystemRandom``, i.e. OS entropy)
    and rounded to the nearest integer, so this sampler is suitable for LWE
    noise and secret-key material. Widths above ``2^40`` are sampled with
    :func:`discrete_gaussian_bigint` (a double would leave the low bits of
    every sample zero). Note it is a rounded *continuous* Gaussian rather
    than a true discrete-Gaussian sampler, and it is not constant-time; this
    is adequate for a research prototype but not for production side-channel
    resistance.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            sigma: Standard deviation for the Gaussian distribution.
    """
    if sigma > _DOUBLE_SAFE_SIGMA:
        flat = discrete_gaussian_bigint(sigma, rows * cols)
        return [flat[i * cols:(i + 1) * cols] for i in range(rows)]
    return [
        [gp.mpz(int(round(_CSPRNG.gauss(0.0, sigma)))) for _ in range(cols)]
        for _ in range(rows)
    ]


class XofRandom:
    """Deterministic bit source: SHAKE-256 keyed by a secret seed.

    Provides ``getrandbits`` for derandomised sampling (the same seed always
    yields the same bits). The seed must be secret and unpredictable.

        Args:
            seed: Secret seed bytes (at least 32 bytes of entropy).
    """

    def __init__(self, seed: bytes) -> None:
        self._seed = len(seed).to_bytes(8, "big") + seed
        self._counter = 0

    def getrandbits(self, k: int) -> int:
        n_bytes = (k + 7) // 8
        block = hashlib.shake_256(self._seed + self._counter.to_bytes(8, "big")).digest(n_bytes)
        self._counter += 1
        return int.from_bytes(block, "big") >> (8 * n_bytes - k)


def discrete_gaussian_bigint(sigma, count: int, rng=None) -> list[gp.mpz]:
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
            rng: Bit source with ``getrandbits`` (default: the OS CSPRNG).
            An :class:`XofRandom` gives derandomised sampling.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    prec = int(gp.ceil(gp.mpfr(sigma))).bit_length() + 128
    rng = rng or _CSPRNG
    samples = []
    with gp.context(gp.get_context(), precision=prec):
        two_pi = 2 * gp.const_pi()
        sig = gp.mpfr(sigma)
        scale = gp.mpfr(2) ** prec
        for _ in range(count):
            u1 = (gp.mpfr(rng.getrandbits(prec)) + 1) / scale  # (0, 1]
            u2 = gp.mpfr(rng.getrandbits(prec)) / scale  # [0, 1)
            z = gp.sqrt(-2 * gp.log(u1)) * gp.cos(two_pi * u2)
            samples.append(gp.mpz(gp.rint(z * sig)))
    return samples
