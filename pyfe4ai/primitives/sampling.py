"""Random sampling utilities: uniform vectors/matrices and discrete Gaussians."""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import NamedTuple

import gmpy2 as gp
import numpy as np

import warnings

from pyfe4ai.core import forksafe
from pyfe4ai.core.exceptions import FESecurityWarning, FEValidationError

_CSPRNG = random.SystemRandom()
# The OS source of the module; a replaced _CSPRNG (tests) keeps the Python sampler.
_SYSTEM_RANDOM = _CSPRNG

#: The Gaussian samplers, ``"rounded"`` (the library-wide default; a rounded
#: continuous Gaussian) and ``"facct"`` (exact ``D_{Z,sigma}``, see
#: :func:`discrete_gaussian_facct`). Select one per call (``sampler=``), per
#: context (:func:`gaussian_sampler`), per process (:func:`set_gaussian_sampler`)
#: or with the ``PYFE4AI_GAUSSIAN_SAMPLER`` environment variable. Exception:
#: the schemes that record their sampler in their public parameters (see
#: :func:`scheme_sampler`) default to ``"facct"`` and ignore the process-wide
#: choice (ABDP15 only under ``standard``), and the threshold flooding is
#: pinned (:data:`FLOODING_SAMPLER`).
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


#: The default of :func:`scheme_sampler`: the exact ``D_{Z,sigma}`` the proofs of
#: the Ring-LWE MIFE / MCFE family and of the dense-LWE MCFE keys ``t`` assume
#: (the crypto review of October 2026 found the double-precision rounded
#: sampler 2^-25..2^-4 from it per key or ciphertext, against 2^-128 budgets).
#: SIFE FullySec Ring-LWE has the same default (its own ``DEFAULT_SAMPLER``,
#: passed to :func:`scheme_sampler` with ``use_context=False``), and so has the
#: public keys' noise of ABDP15 under ``standard`` (under ``toy`` its default is
#: the library-wide choice, :func:`get_gaussian_sampler`).
PROOF_SAMPLER = "facct"


def scheme_sampler(config: dict | None, pp: dict | None = None, default: str = PROOF_SAMPLER,
                   profile: str | None = None, use_context: bool = True) -> str:
    """The Gaussian sampler of a scheme object whose proofs need ``D_{Z,sigma}``.

    In this order: ``config["gaussian_sampler"]``; else that of its public
    parameters *pp*; else the one a :func:`gaussian_sampler` block selects
    (not under the ``standard`` profile, nor with ``use_context=False``, as
    for SIFE FullySec Ring-LWE); else *default*. The process-wide
    choice (:func:`set_gaussian_sampler`, ``PYFE4AI_GAUSSIAN_SAMPLER``) does
    not apply: these schemes record their sampler in their public parameters.
    *profile* is the key generator's ``security_profile`` (crypto objects take
    that of *pp*).

    Raises:
        FEValidationError: For an unknown sampler, and for an explicit
            ``"rounded"`` under the ``standard`` profile (the double-precision
            rounded sampler does not meet its ``2^-128`` budgets).

    Warns:
        FESecurityWarning: When a crypto object's config selects ``"rounded"``
            although its public parameters record ``"facct"`` (toy profile).
    """
    if profile is None and pp:
        profile = pp.get("security_profile")
    standard = profile == "standard"
    explicit = config.get("gaussian_sampler") if config else None
    recorded = pp.get("gaussian_sampler") if pp else None
    def checked(name) -> str:
        if name not in GAUSSIAN_SAMPLERS:
            raise FEValidationError(
                "gaussian_sampler must be one of {}, got {!r}".format(GAUSSIAN_SAMPLERS, name)
            )
        return name

    if explicit is not None:
        name = checked(explicit)
        if name != PROOF_SAMPLER and standard:
            raise FEValidationError(
                "gaussian_sampler = {!r} under the standard profile: the double-precision rounded sampler is "
                "2^-25..2^-4 from D_{{Z,sigma}} per key or ciphertext, above the profile's 2^-128 budgets "
                "(use 'facct', the default, or the toy profile)".format(name)
            )
        if name != PROOF_SAMPLER and recorded == PROOF_SAMPLER:
            warnings.warn(
                "gaussian_sampler = {!r} overrides the {!r} of the public parameters: these ciphertexts' "
                "noise is not the D_{{Z,sigma}} the proof assumes".format(name, recorded),
                FESecurityWarning, stacklevel=3,
            )
        return name
    if recorded is not None:
        return checked(recorded)
    selected = None if (standard or not use_context) else _sampler_override.get()
    return checked(default if selected is None else selected)


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
# Resolution only: below it the double path keeps the artifacts listed in
# discrete_gaussian_matrix (shared Box-Muller radii, ties to even, the
# 8.57 sigma cut).
_DOUBLE_SAFE_SIGMA = 2 ** 40


def _rounded_gaussians(count: int, sigma: float) -> list:
    """*count* values ``round(N(0, sigma^2))`` from :data:`_CSPRNG`.

    The arithmetic of ``random.gauss`` (Box-Muller: ``cos`` then ``sin`` of one
    pair of uniforms, consumed in that order), but the second value of the
    last pair is dropped, not cached: ``gauss`` keeps it in the generator's
    ``gauss_next``, where a future secret sample stays in memory between
    calls and is drawn twice after ``fork()``.
    """
    rnd = _CSPRNG.random
    out = []
    while len(out) < count:
        x2pi = rnd() * (2.0 * math.pi)
        g2rad = math.sqrt(-2.0 * math.log(1.0 - rnd()))
        out.append(gp.mpz(int(round(math.cos(x2pi) * g2rad * sigma))))
        if len(out) < count:
            out.append(gp.mpz(int(round(math.sin(x2pi) * g2rad * sigma))))
    return out


def discrete_gaussian_matrix(
    rows: int, cols: int, sigma: float, sampler: str | None = None
) -> list[list[gp.mpz]]:
    """Sample a matrix from the (rounded) discrete Gaussian distribution.

    With ``sampler="rounded"``, samples are Box-Muller normals in double
    precision from a CSPRNG (``random.SystemRandom``, i.e. OS entropy; the
    native kernels read their per-call DRBG) scaled by *sigma* and rounded
    to the nearest integer. Widths above ``2^40`` are sampled with
    :func:`discrete_gaussian_bigint` (a double would leave the low bits of
    every sample zero). Besides being a rounded *continuous* Gaussian, the
    double path has artifacts of its own: the two outputs of a pair share one
    radius (rings that show in the tail), exact half-integer ties round to
    even (about ``2^-53.8 sigma`` per sample), and no value exceeds about
    ``8.57 sigma``. Per the crypto review of October 2026 one ciphertext's
    noise is then about 2^-12..2^-4 and a key 2^-25..2^-15 from i.i.d.
    ``D_{Z,sigma}``: fine for toy parameters and benchmarks, not for the
    distributions of the security proofs (which is why the Ring-LWE MIFE /
    MCFE family, the dense-LWE MCFE keys, SIFE FullySec Ring-LWE and, under
    ``standard``, the ABDP15 public keys default to ``"facct"``, see
    :func:`scheme_sampler`; ALS16, the DCR schemes and ABDP15 under ``toy``
    follow the library-wide choice). It is not
    constant-time either. With ``sampler="facct"`` the samples come from
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
    flat = _rounded_gaussians(rows * cols, sigma)
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


# Guards the counter claims of every XofRandom. Its critical sections make no
# calls, so no signal handler or finalizer can run inside them (CPython runs
# those only at calls and backward jumps) and none of them waits for it; it is
# never held while drawing. A forked child gets a new one, as the parent's may
# be held at the fork.
_XOF_LOCK = threading.Lock()
forksafe.renew_lock(sys.modules[__name__], "_XOF_LOCK")


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
#: uniforms and arithmetic have ``bits(sigma) + BIGINT_GUARD_BITS`` bits. 320
#: covers ``bits(sigma) + stat_param + ceil(log2 N) + 8`` for every statistical
#: parameter the key generators accept (at most
#: :data:`~pyfe4ai.core.profiles.MAX_STAT_PARAM` = 256) and up to ``2^56``
#: samples per key: the dozen MPFR roundings of one sample stay below
#: ``2^-(stat_param + log2 N + 8)`` in statistical distance from the ideal
#: rounded Gaussian. (It was 128, which the October 2026 crypto review found
#: 2 to 19 bits short of the ``2^-128`` budgets of the DCR keys and masks and
#: of the threshold flooding; then 192, which covered stat_param <= 128 only.)
BIGINT_GUARD_BITS = 320


#: The sampler of the threshold flooding noise, whatever the process-wide or
#: context choice: the MPFR rounded Gaussian (an ideal rounded Gaussian is a
#: mixture of centred discrete Gaussians, which the flooding argument accepts).
#: Recorded in the decryption keys as ``noise_sampler`` (checked by
#: :func:`~pyfe4ai.core.versions.check_flooding_version`), so that a server
#: never answers one request with noises of two samplers.
FLOODING_SAMPLER = "rounded"


def bigint_precision(sigma) -> int:
    """Bits of the uniforms and of the MPFR arithmetic of
    :func:`discrete_gaussian_bigint` for *sigma*: ``bits(ceil(sigma)) + 320``."""
    return int(gp.ceil(gp.mpfr(sigma))).bit_length() + BIGINT_GUARD_BITS


def discrete_gaussian_bigint(
    sigma, count: int, rng=None, sampler: str | None = None, precision: int | None = None
) -> list[gp.mpz]:
    """Sample *count* integers from a rounded Gaussian of arbitrary width.

    Unlike :func:`discrete_gaussian_matrix`, *sigma* may exceed the range of
    a double (e.g. the ``sqrt(kappa) * N^{5/2}`` widths of the DCR schemes):
    a Box-Muller sample is computed with ``gmpy2.mpfr`` at
    ``bits(sigma) + 320`` bits of precision (:data:`BIGINT_GUARD_BITS`) from CSPRNG uniforms and rounded
    to the nearest integer. As with :func:`discrete_gaussian_matrix` this is
    a rounded continuous Gaussian, not an exact discrete-Gaussian sampler,
    and it is not constant-time. With ``sampler="facct"`` the samples come
    from :func:`discrete_gaussian_facct` (exact ``D_{Z,sigma}``) with the
    same *rng*.

    With the native backend (:mod:`pyfe4ai.backend`) the
    rounded samples of widths below ``2^448`` are computed by the extension
    when *rng* is an :class:`XofRandom` or ``None``: from the same
    ``getrandbits`` values, it returns the integers this function computes
    (see :meth:`~pyfe4ai.backend.native_impl.NativeBackend.gaussian_bigint`); with
    ``rng=None`` its bits come from the extension's per-call DRBG.

    Args:
        sigma: Standard deviation (int, mpz, float or mpfr), > 0.
        count: Number of samples.
        rng: Bit source with ``getrandbits`` (default: the OS CSPRNG).
            An :class:`XofRandom` gives derandomised sampling.
        sampler: ``"rounded"`` or ``"facct"``; ``None`` takes
            :func:`get_gaussian_sampler`.
        precision: Bits of the rounded sampler, at least (and by default)
            :func:`bigint_precision` ``= bits(sigma) + BIGINT_GUARD_BITS``;
            FACCT has a fixed precision and ignores it.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if _resolve_sampler(sampler) == "facct":
        return discrete_gaussian_facct(sigma, count, rng=rng)
    prec = bigint_precision(sigma)
    if precision is not None:
        if int(precision) < prec:
            raise ValueError("precision must be at least bits(sigma) + {} = {}".format(BIGINT_GUARD_BITS, prec))
        prec = int(precision)
    if count > 0:  # the backend's (natively where the extension applies, else _gaussian_bigint_reference)
        from pyfe4ai import backend

        return backend.ops().gaussian_bigint(sigma, count, rng, prec)
    return _gaussian_bigint_reference(sigma, count, rng, prec)


def _gaussian_bigint_reference(sigma, count: int, rng, prec: int) -> list[gp.mpz]:
    """The rounded samples of :func:`discrete_gaussian_bigint` in Python (the reference)."""
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
forksafe.call_in_child(_OS_RANDOM._reset)  # the child neither reuses the parent's bytes nor its lock


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

    With the native backend (:mod:`pyfe4ai.backend`) the
    extension draws the same ``getrandbits`` in the same order and makes the
    same decisions when *rng* is an :class:`XofRandom` or ``None``
    (:meth:`~pyfe4ai.backend.native_impl.NativeBackend.gaussian_facct`; with ``rng=None``
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
    if count > 0:  # the backend's (natively where the extension applies, else _gaussian_facct_reference)
        from pyfe4ai import backend

        return backend.ops().gaussian_facct(sigma, count, rng)
    return _gaussian_facct_reference(sigma, count, rng)


def _gaussian_facct_reference(sigma, count: int, rng) -> list[gp.mpz]:
    """:func:`discrete_gaussian_facct` in Python (the reference)."""
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
