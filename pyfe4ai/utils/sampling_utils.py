"""Random sampling utilities: uniform vectors/matrices and discrete Gaussians."""

from __future__ import annotations

import hashlib
import os
import random
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import NamedTuple

import gmpy2 as gp
import numpy as np

from pyfe4ai.utils.exceptions import FEValidationError

_CSPRNG = random.SystemRandom()

#: Gaussian samplers: ``"rounded"`` (the default; a rounded continuous
#: Gaussian) and ``"facct"`` (exact ``D_{Z,sigma}``, see
#: :func:`discrete_gaussian_facct`). Select one per call (``sampler=``), per
#: context (:func:`gaussian_sampler`), per process (:func:`set_gaussian_sampler`)
#: or with the ``PYFE4AI_GAUSSIAN_SAMPLER`` environment variable.
GAUSSIAN_SAMPLERS = ("rounded", "facct")
_SAMPLER_ENV = "PYFE4AI_GAUSSIAN_SAMPLER"
_default_sampler = os.environ.get(_SAMPLER_ENV, "rounded")
_sampler_override: ContextVar[str | None] = ContextVar("pyfe4ai_gaussian_sampler", default=None)


def _check_sampler(name) -> str:
    if name not in GAUSSIAN_SAMPLERS:
        raise FEValidationError(
            "unknown Gaussian sampler {!r}; choose one of {}".format(name, ", ".join(GAUSSIAN_SAMPLERS))
        )
    return name


def get_gaussian_sampler() -> str:
    """Name of the Gaussian sampler in effect in this context."""
    override = _sampler_override.get()
    return _check_sampler(_default_sampler if override is None else override)


def set_gaussian_sampler(name: str) -> None:
    """Set the process-wide default Gaussian sampler (``"rounded"`` or ``"facct"``).

    Parties that derive noise from a shared seed (the flooding noise of the
    threshold schemes, drawn from an :class:`XofRandom`) must use the same
    sampler, or they derive different noise.
    """
    global _default_sampler
    _default_sampler = _check_sampler(name)


@contextmanager
def gaussian_sampler(name: str):
    """Use the Gaussian sampler *name* inside a ``with`` block.

    The choice is a context variable: threads started inside the block use
    the process-wide default.
    """
    token = _sampler_override.set(_check_sampler(name))
    try:
        yield
    finally:
        _sampler_override.reset(token)


def _resolve_sampler(sampler) -> str:
    return get_gaussian_sampler() if sampler is None else _check_sampler(sampler)


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


def rand_bit_matrix(rows: int, cols: int) -> np.ndarray:
    """``rows x cols`` uniform bits (``uint8``) from the OS CSPRNG."""
    count = rows * cols
    bits = np.unpackbits(np.frombuffer(os.urandom((count + 7) // 8), dtype=np.uint8))
    return bits[:count].reshape(rows, cols)


# Above this width a double-precision sample no longer resolves the integers
# (53-bit mantissa, samples up to ~2^4 sigma): its low bits would be constant.
_DOUBLE_SAFE_SIGMA = 2 ** 40


def discrete_gaussian_matrix(
    rows: int, cols: int, sigma: float, sampler: str | None = None
) -> list[list[gp.mpz]]:
    """Sample a matrix from the (rounded) discrete Gaussian distribution.

    With the default sampler ``"rounded"``, samples are drawn from a CSPRNG
    (``random.SystemRandom``, i.e. OS entropy) and rounded to the nearest
    integer, so this sampler is suitable for LWE noise and secret-key
    material. Widths above ``2^40`` are sampled with
    :func:`discrete_gaussian_bigint` (a double would leave the low bits of
    every sample zero). Note it is a rounded *continuous* Gaussian rather
    than a true discrete-Gaussian sampler, and it is not constant-time; this
    is adequate for a research prototype but not for production side-channel
    resistance. With ``sampler="facct"`` the samples come from
    :func:`discrete_gaussian_facct` (exact ``D_{Z,sigma}``).

    Args:
        rows: Number of rows.
        cols: Number of columns.
        sigma: Standard deviation for the Gaussian distribution.
        sampler: ``"rounded"`` or ``"facct"``; ``None`` takes
            :func:`get_gaussian_sampler`.
    """
    if _resolve_sampler(sampler) == "facct":
        flat = discrete_gaussian_facct(sigma, rows * cols)
        return [flat[i * cols:(i + 1) * cols] for i in range(rows)]
    if sigma > _DOUBLE_SAFE_SIGMA:
        flat = discrete_gaussian_bigint(sigma, rows * cols, sampler="rounded")
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


def discrete_gaussian_bigint(
    sigma, count: int, rng=None, sampler: str | None = None, precision: int | None = None
) -> list[gp.mpz]:
    """Sample *count* integers from a rounded Gaussian of arbitrary width.

    Unlike :func:`discrete_gaussian_matrix`, *sigma* may exceed the range of
    a double (e.g. the ``sqrt(kappa) * N^{5/2}`` widths of the DCR schemes):
    a Box-Muller sample is computed with ``gmpy2.mpfr`` at
    ``bits(sigma) + 128`` bits of precision from CSPRNG uniforms and rounded
    to the nearest integer. As with :func:`discrete_gaussian_matrix` this is
    a rounded continuous Gaussian, not an exact discrete-Gaussian sampler,
    and it is not constant-time. With ``sampler="facct"`` the samples come
    from :func:`discrete_gaussian_facct` (exact ``D_{Z,sigma}``) with the
    same *rng*.

    Args:
        sigma: Standard deviation (int, mpz, float or mpfr), > 0.
        count: Number of samples.
        rng: Bit source with ``getrandbits`` (default: the OS CSPRNG).
            An :class:`XofRandom` gives derandomised sampling.
        sampler: ``"rounded"`` or ``"facct"``; ``None`` takes
            :func:`get_gaussian_sampler`.
        precision: Bits of the rounded sampler, at least (and by default)
            ``bits(sigma) + 128``; FACCT has a fixed precision and ignores it.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if _resolve_sampler(sampler) == "facct":
        return discrete_gaussian_facct(sigma, count, rng=rng)
    prec = int(gp.ceil(gp.mpfr(sigma))).bit_length() + 128
    if precision is not None:
        if int(precision) < prec:
            raise ValueError("precision must be at least bits(sigma) + 128 = {}".format(prec))
        prec = int(precision)
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


# ---------------------------------------------------------------------------
# Exact discrete Gaussian (FACCT-style double sampling)
# ---------------------------------------------------------------------------

#: Bits of the CDT thresholds and of the acceptance tests of
#: :func:`discrete_gaussian_facct`; each sample is within statistical distance
#: about ``2^-185`` of ``D_{Z,sigma}``.
FACCT_PRECISION = 192
# Entries of the base CDT: the base width is at most sigma_0 = 1/sqrt(2 ln 2),
# whose half-Gaussian puts less than 2^-196 on values >= 14.
_FACCT_CDT_SIZE = 14
# Terms of the base normaliser (the rest is below 2^-2900).
_FACCT_TERMS = _FACCT_CDT_SIZE + 40


class _OSRandomBuffer:
    """``getrandbits`` over ``os.urandom``, read in 64 KiB blocks.

    One system call serves many samples (``random.SystemRandom`` makes one
    per call). Thread-safe; a forked child drops the parent's buffer, so the
    two processes never share bytes. Python cannot wipe memory: consumed
    bytes stay in the buffer until the next refill.
    """

    _BLOCK = 1 << 16

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._lock = threading.Lock()  # new lock: the parent's may be held at fork
        self._buf = b""
        self._pos = 0

    def getrandbits(self, k: int) -> int:
        if k <= 0:
            return 0
        n_bytes = (k + 7) // 8
        with self._lock:
            if self._pos + n_bytes > len(self._buf):
                self._buf = os.urandom(max(self._BLOCK, n_bytes))
                self._pos = 0
            block = self._buf[self._pos:self._pos + n_bytes]
            self._pos += n_bytes
        return int.from_bytes(block, "big") >> (8 * n_bytes - k)


_OS_RANDOM = _OSRandomBuffer()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_OS_RANDOM._reset)


class _FacctParams(NamedTuple):
    k: int  # sigma = k * sigma_base with sigma_base <= sigma_0
    cdt: tuple  # thresholds floor(2^P F(i)) of the base half-Gaussian, i < _FACCT_CDT_SIZE
    neg_c: object  # -1 / (2 sigma^2) as an mpfr
    y_bits: int  # bits of k - 1


def _exact(sigma) -> gp.mpq:
    try:
        return gp.mpq(sigma)
    except TypeError:
        return gp.mpq(float(sigma))  # e.g. numpy.float64


@lru_cache(maxsize=64)
def _facct_params(sigma: gp.mpq) -> _FacctParams:
    """Tables of :func:`discrete_gaussian_facct` for the exact width *sigma*.

    Computed with correctly rounded MPFR arithmetic, so they are the same on
    every platform.
    """
    bits = int(gp.ceil(gp.mpfr(sigma))).bit_length()
    with gp.context(precision=bits + FACCT_PRECISION + 64, round=gp.RoundToNearest):
        sig = gp.mpfr(sigma)
        k = max(1, int(gp.ceil(sig * gp.sqrt(2 * gp.log(2)))))  # k sigma_0 >= sigma
        neg_c = -1 / (2 * sig * sig)
        weights = [gp.exp(neg_c * (x * k) ** 2) for x in range(_FACCT_TERMS)]
        total = sum(weights)
        cdt, acc = [], gp.mpfr(0)
        for w in weights[:_FACCT_CDT_SIZE]:
            acc += w
            cdt.append(int(gp.floor(acc / total * (1 << FACCT_PRECISION))))
    return _FacctParams(k, tuple(cdt), neg_c, (k - 1).bit_length())


def facct_base_width(sigma):
    """``(k, sigma / k)``: the double-sampling factor and the base width used
    by :func:`discrete_gaussian_facct` for *sigma*."""
    k = _facct_params(_exact(sigma)).k
    return k, gp.mpfr(_exact(sigma)) / k


def discrete_gaussian_facct(sigma, count: int, rng=None) -> list[gp.mpz]:
    """Sample *count* integers from the discrete Gaussian ``D_{Z,sigma}``.

    FACCT-style double sampling (Zhao, Steinfeld and Sakzad, "FACCT: FAst,
    Compact, and Constant-Time Discrete Gaussian Sampler over Integers",
    IEEE Trans. Computers 2020, after the BLISS sampler of Ducas et al.,
    CRYPTO 2013; GoFE's ``NormalDoubleConstant`` has the same structure).
    With ``k = ceil(sigma / sigma_0)``, ``sigma_0 = 1/sqrt(2 ln 2)``, and the
    base width ``sigma_b = sigma / k <= sigma_0``, one trial:

    1. ``u <- getrandbits(P)``; ``x`` = number of CDT thresholds ``T_i <= u``
       (the whole table is scanned), so ``x`` follows the half-Gaussian
       ``D_{Z>=0, sigma_b}``;
    2. ``y`` uniform in ``[0, k)``: ``getrandbits(bits(k - 1))`` until below
       ``k`` (no bits when ``k = 1``);
    3. ``v <- getrandbits(P)``; accept ``z = k x + y`` iff
       ``v < floor(2^P exp(-y (y + 2 k x) / (2 sigma^2)))``, so that ``z``
       follows ``D_{Z>=0, sigma}``;
    4. ``s <- getrandbits(1)``; if ``z = 0`` and ``s = 1`` start over,
       else return ``(-1)^s z``.

    ``P`` is :data:`FACCT_PRECISION`. Deviation from FACCT: its base is the
    fixed ``D_{Z>=0, sigma_0}``, which only gives widths ``k sigma_0``; here
    the base table is computed for ``sigma / k`` (and cached), so every
    width is sampled exactly as requested. The tables and the acceptance
    thresholds use correctly rounded MPFR arithmetic: for a given bit source
    the samples are the same on every platform, which makes known-answer
    tests portable to other implementations (the order of ``getrandbits``
    calls above is part of that contract).

    The number of trials does not depend on the returned value, and no step
    branches on secret data except through the comparisons above. CPython
    and big-integer arithmetic still take data-dependent time, so this
    function is *not* constant-time; it is the reference for a native
    implementation that can be. It is also several times slower than the
    rounded sampler in Python.

    Args:
        sigma: Width (int, mpz, float, mpfr or mpq), > 0.
        count: Number of samples.
        rng: Bit source with ``getrandbits`` (default: ``os.urandom`` read in
            blocks). An :class:`XofRandom` gives derandomised sampling.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if count < 0:
        raise ValueError("count must be non-negative")
    params = _facct_params(_exact(sigma))
    rng = rng or _OS_RANDOM
    k, cdt, y_bits, neg_c = params.k, params.cdt, params.y_bits, params.neg_c
    precision = FACCT_PRECISION
    scale = 1 << precision
    samples = []
    with gp.context(precision=precision + 64, round=gp.RoundToNearest):
        while len(samples) < count:
            u = rng.getrandbits(precision)
            x = 0
            for threshold in cdt:
                x += u >= threshold
            y = 0
            if y_bits:
                y = rng.getrandbits(y_bits)
                while y >= k:
                    y = rng.getrandbits(y_bits)
            v = rng.getrandbits(precision)
            if y and v >= int(gp.floor(gp.exp(neg_c * (y * (y + 2 * k * x))) * scale)):
                continue
            z = k * x + y
            negative = rng.getrandbits(1)
            if z == 0 and negative:
                continue
            samples.append(gp.mpz(-z if negative else z))
    return samples
