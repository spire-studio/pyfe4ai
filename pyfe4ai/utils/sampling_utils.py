"""Random sampling utilities: uniform vectors/matrices and discrete Gaussians."""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import NamedTuple

import gmpy2 as gp
import numpy as np

from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

_CSPRNG = random.SystemRandom()
# The OS source of the module; a replaced _CSPRNG (tests) keeps the Python sampler.
_SYSTEM_RANDOM = _CSPRNG

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


#: The sampler the Ring-LWE MIFE / MCFE family, the dense-LWE MCFE key ``t``
#: and SIFE FullySec Ring-LWE use unless told otherwise: the exact
#: ``D_{Z,sigma}`` their proofs assume (the crypto review of October 2026 found
#: the double-precision rounded sampler 2^-25..2^-4 from it per key or
#: ciphertext, against 2^-128 budgets).
PROOF_SAMPLER = "facct"


def scheme_sampler(config: dict | None, pp: dict | None = None, default: str = PROOF_SAMPLER) -> str:
    """The Gaussian sampler of a scheme object: ``config["gaussian_sampler"]``,
    else that of its public parameters *pp*, else the one a
    :func:`gaussian_sampler` block selects, else *default*. The process-wide
    choice (:func:`set_gaussian_sampler`, ``PYFE4AI_GAUSSIAN_SAMPLER``) does
    not apply: these schemes record their sampler in their public parameters.

    Raises:
        FEValidationError: For an unknown sampler.
    """
    name = config.get("gaussian_sampler") if config else None
    if name is None and pp:
        name = pp.get("gaussian_sampler")
    if name is None:
        name = _sampler_override.get()
    return _check_sampler(default if name is None else name)


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


# Guards the counter claims of every XofRandom. Its critical sections make no
# calls, so no signal handler or finalizer can run inside them (CPython runs
# those only at calls and backward jumps) and none of them waits for it; it is
# never held while drawing. A forked child gets a new one, as the parent's may
# be held at the fork.
_XOF_LOCK = threading.Lock()


def _reset_xof_lock() -> None:
    global _XOF_LOCK
    _XOF_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_xof_lock)


class XofRandom:
    """Deterministic bit source: SHAKE-256 keyed by a secret seed.

    Provides ``getrandbits`` for derandomised sampling (the same seed always
    yields the same bits). The seed must be secret and unpredictable. Call
    ``c`` (from 0) hashes ``len(seed) || seed || c``; each call claims its
    ``c`` atomically, so threads sharing an instance never draw the same block
    (which thread gets which block then depends on scheduling). While the
    native sampler draws from an instance (a number of calls known only
    afterwards), other threads' draws from it wait for the end, and a draw
    from the same thread (a signal handler, a finalizer) raises
    ``RuntimeError``.

    Args:
        seed: Secret seed bytes (at least 32 bytes of entropy).
    """

    def __init__(self, seed: bytes) -> None:
        self._seed = len(seed).to_bytes(8, "big") + seed
        self._counter = 0
        self._busy = None  # the thread drawing a number of calls known only afterwards

    def _wait_or_refuse(self, owner, me) -> None:
        if owner == me:
            raise RuntimeError("this XofRandom is being drawn from by the native sampler in this thread")
        time.sleep(0.0002)  # another thread's native draw: wait for its end

    def _claim(self, calls: int) -> int:
        """Reserve the next *calls* counters for the caller; returns the first."""
        me = threading.get_ident()
        while True:
            with _XOF_LOCK:
                owner = self._busy
                start = self._counter
                if owner is None:
                    self._counter = start + calls
            if owner is None:
                return start
            self._wait_or_refuse(owner, me)

    def _claim_dynamic(self, draw):
        """``draw(start) -> (result, calls)`` for a draw whose number of calls
        is known only afterwards (the native FACCT): the instance is marked
        busy -- other threads' draws from it wait, this thread's raise -- while
        *draw* runs without any lock held, then the counter advances by
        *calls*. Returns *result*."""
        me = threading.get_ident()
        while True:
            with _XOF_LOCK:
                owner = self._busy
                start = self._counter
                if owner is None:
                    self._busy = me
            if owner is None:
                break
            self._wait_or_refuse(owner, me)
        calls = 0
        try:
            result, calls = draw(start)
        finally:
            with _XOF_LOCK:
                self._counter = start + calls
                self._busy = None
        return result

    def getrandbits(self, k: int) -> int:
        n_bytes = (k + 7) // 8
        counter = self._claim(1)
        block = hashlib.shake_256(self._seed + counter.to_bytes(8, "big")).digest(n_bytes)
        return int.from_bytes(block, "big") >> (8 * n_bytes - k)


#: Guard bits of the MPFR precision of :func:`discrete_gaussian_bigint`: its
#: uniforms and arithmetic have ``bits(sigma) + BIGINT_GUARD_BITS`` bits. 192
#: covers ``bits(sigma) + stat_param + ceil(log2 N) + 8`` for statistical
#: parameters up to 128 and up to ``2^56`` samples per key: the dozen MPFR
#: roundings of one sample stay below ``2^-(stat_param + log2 N + 8)`` in
#: statistical distance from the ideal rounded Gaussian. (Up to v0.4 it was
#: 128, which the October 2026 crypto review found 2 to 19 bits short of the
#: ``2^-128`` budgets of the DCR keys and masks and of the threshold flooding.)
BIGINT_GUARD_BITS = 192


#: Version of the flooding-noise derivation in the decryption keys of the
#: threshold schemes (tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE): 2 since
#: :data:`BIGINT_GUARD_BITS` is 192 (keys without the field are version 1).
FLOODING_NOISE_VERSION = 2


def check_flooding_version(dk) -> None:
    """Refuse a threshold decryption key of another flooding derivation: it may
    have answered a request with the noise of the earlier derivation, and a
    second noise for the same request would let the answers be averaged.

    Raises:
        FEKeyError: For a key of another version (issue new keys).
    """
    version = dk.get("noise_version", 1) if isinstance(dk, dict) else None
    if version != FLOODING_NOISE_VERSION:
        raise FEKeyError(
            "decryption key of flooding derivation {} (this version derives noise version {}): "
            "issue new decryption keys".format(version, FLOODING_NOISE_VERSION)
        )


def bigint_precision(sigma) -> int:
    """Bits of the uniforms and of the MPFR arithmetic of
    :func:`discrete_gaussian_bigint` for *sigma*: ``bits(ceil(sigma)) + 192``."""
    return int(gp.ceil(gp.mpfr(sigma))).bit_length() + BIGINT_GUARD_BITS


def discrete_gaussian_bigint(sigma, count: int, rng=None, sampler: str | None = None) -> list[gp.mpz]:
    """Sample *count* integers from a rounded Gaussian of arbitrary width.

    Unlike :func:`discrete_gaussian_matrix`, *sigma* may exceed the range of
    a double (e.g. the ``sqrt(kappa) * N^{5/2}`` widths of the DCR schemes):
    a Box-Muller sample is computed with ``gmpy2.mpfr`` at
    ``bits(sigma) + 192`` bits of precision (:data:`BIGINT_GUARD_BITS`) from CSPRNG uniforms and rounded
    to the nearest integer. As with :func:`discrete_gaussian_matrix` this is
    a rounded continuous Gaussian, not an exact discrete-Gaussian sampler,
    and it is not constant-time. With ``sampler="facct"`` the samples come
    from :func:`discrete_gaussian_facct` (exact ``D_{Z,sigma}``) with the
    same *rng*.

    With the native backend (:mod:`pyfe4ai.utils.native_backend`) the
    rounded samples of widths below ``2^448`` are computed by the extension
    when *rng* is an :class:`XofRandom` or ``None``: from the same
    ``getrandbits`` values, it returns the integers this function computes
    (see :func:`~pyfe4ai.utils.native_backend.gaussian_bigint`); with
    ``rng=None`` its bits come from the extension's per-call DRBG.

    Args:
        sigma: Standard deviation (int, mpz, float or mpfr), > 0.
        count: Number of samples.
        rng: Bit source with ``getrandbits`` (default: the OS CSPRNG).
            An :class:`XofRandom` gives derandomised sampling.
        sampler: ``"rounded"`` or ``"facct"``; ``None`` takes
            :func:`get_gaussian_sampler`.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if _resolve_sampler(sampler) == "facct":
        return discrete_gaussian_facct(sigma, count, rng=rng)
    prec = bigint_precision(sigma)
    if count > 0:
        from pyfe4ai.utils import native_backend

        native = native_backend.gaussian_bigint(sigma, count, rng, prec)
        if native is not None:
            return native
    rng = rng or _CSPRNG
    samples = []
    with gp.context(gp.get_context(), precision=prec):
        two_pi = 2 * gp.const_pi()
        sig = gp.mpfr(sigma)
        scale = gp.mpfr(2) ** prec
        for _ in range(count):
            samples.append(_box_muller_rint(rng.getrandbits(prec), rng.getrandbits(prec), two_pi, sig, scale))
    return samples


def _box_muller_rint(b1: int, b2: int, two_pi, sig, scale) -> gp.mpz:
    """One sample of :func:`discrete_gaussian_bigint` from its uniforms
    ``b1, b2 = getrandbits(prec)``, in the caller's context of ``prec`` bits."""
    u1 = (gp.mpfr(b1) + 1) / scale  # (0, 1]
    u2 = gp.mpfr(b2) / scale  # [0, 1)
    z = gp.sqrt(-2 * gp.log(u1)) * gp.cos(two_pi * u2)
    return gp.mpz(gp.rint(z * sig))


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
# The module's OS source; a replaced _OS_RANDOM (tests) keeps the Python sampler.
_OS_SOURCE = _OS_RANDOM
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

    With the native backend (:mod:`pyfe4ai.utils.native_backend`) the
    extension draws the same ``getrandbits`` in the same order and makes the
    same decisions when *rng* is an :class:`XofRandom` or ``None``
    (:func:`~pyfe4ai.utils.native_backend.gaussian_facct`; with ``rng=None``
    its bits come from the per-call DRBG).

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
    if count > 0:
        from pyfe4ai.utils import native_backend

        native = native_backend.gaussian_facct(sigma, count, rng)
        if native is not None:
            return native
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
