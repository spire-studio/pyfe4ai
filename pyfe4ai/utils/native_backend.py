"""Optional native backend of the Ring-LWE, dense-LWE and ristretto255 schemes.

The extension ``pyfe4ai._rust_native`` (Rust and PyO3, sources in ``rust/``;
built by ``pip install`` when a Rust toolchain is available, or in a source
checkout by ``tools/build_rust.py``) runs the Ring-LWE MIFE / MCFE hot paths
-- NTTs, Gaussian sampling, label pads, the encryption, decryption and
key-derivation loops of the ndarray helpers -- one call per tensor, with the
GIL released and optionally several threads -- and of the threshold MCFE
(``mcfe/ring_lwe_threshold``): the Shamir sharing of the functional key, the
parsing, digests and NTT-domain products of a partial decryption and the
combination. The replicated key sharings of the three LWE threshold layers
(``"replicated"``, the default for ``t < s``, and ``"replicated-seeded"``) are
dealt and added up natively for any odd ``q`` below ``2^253``, with no NTT
ring (:func:`~pyfe4ai.schemes.threshold_utils.share_replicated`,
:func:`~pyfe4ai.schemes.threshold_utils.share_for_enrolled`). It also samples the wide rounded Gaussians of
:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint` (SIFE FullySec,
ALS16 and the flooding noise of the threshold schemes; :func:`gaussian_bigint`)
with the reference's integers, and has the dense-LWE label
sums, the ristretto255 group of the ``*/ddh_ristretto255`` schemes and the
BLS12-381 pairing group (:mod:`pyfe4ai.utils.bls12_381`). The Python implementation stays the
reference and the fallback: the native code computes the same values (same NTT
root and tables, the same Box-Muller arithmetic on the same kind of random
bytes, the same hash-to-``Z_q`` reduction), and ``tests/test_native_backend.py``
checks it against the Python code bit for bit.

Selection, per process (``set_backend``), per context (``backend``) or with
the ``PYFE4AI_BACKEND`` environment variable:

``"auto"`` (default)
    native where the extension is built and the parameters are supported,
    Python otherwise;
``"python"``
    always the Python implementation;
``"native"``
    like ``"auto"``, but an error if the extension is not built.

The native path applies to moduli below ``2^253`` with a ``2 ring_n``-th root
of unity (all moduli the parameter generators derive, including the flooded
moduli of ``mcfe/ring_lwe_threshold``), in 64-bit words below ``2^63``,
128-bit words below ``2^125`` and 256-bit words above -- and to the RNS
moduli of ``ring_modulus="rns"``: ``q = q1 q2`` (MIFE / MCFE Ring-LWE), in two
64-bit residues per coefficient, and the products of up to four primes of
the threshold MCFE, whose 256-bit ring runs one 64-bit NTT per prime --, and samples natively
Gaussians of widths up to ``2^40`` inside its encryption calls (wider ones
come from :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint`)
with the scheme's sampler (its recorded ``gaussian_sampler``, FACCT by
default, see :func:`~pyfe4ai.utils.sampling_utils.scheme_sampler`): the
rounded Gaussian, consuming seven
random bytes per uniform as ``random.SystemRandom.random``, or the exact
FACCT sampler (``sampler="facct"``), which draws ``getrandbits`` in the order
of :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_facct` and makes
its decisions -- inside the encryption calls, and for
:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_facct` itself
(:func:`gaussian_facct`) --: the tables come from Python (:func:`facct_table`,
with ``c = 1 / (2 sigma^2)`` correct to a 128-bit mantissa), the acceptance
test is decided in fixed point with an error below ``2^-110`` (about
``2^-119`` measured), and the tests within ``2^-100`` of the threshold (about
one in ``2^99``) are handed to the Python formula, so for a given bit source
the samples are the reference's (the known-answer tests of
``tests/test_sampling_facct.py`` hold natively). Building a table compares a
few of its fixed-point thresholds with the Python formula and refuses a
constant off by more than about ``2^-102`` there (a 53-bit one is off by
about ``2^-55``).

Random bytes. The extension draws them from a DRBG scoped to one call:
every tensor-level call (``encrypt_tensor``, ``precompute_tensor``,
``encrypt`` for one matrix, and ``shamir_share`` and ``zq_replicated_deal``
for a whole key, which read stream 0 in order and draw each coefficient or
piece value as ``random.SystemRandom.randrange(0, q)`` does) reads one 32-byte seed from the operating system
(``getrandom`` / ``getentropy``, the source of :func:`os.urandom`) and expands
it with ChaCha20 (the ``rand_chacha`` crate), chunk ``c`` of the call reading
stream ``c`` -- no two chunks, hence no two threads, share a stream. The seed,
the generator states and the sample bytes are overwritten when the call
ends; no generator state outlives a call, which limits what a fork or a
virtual-machine snapshot can duplicate to the call in progress, and the
system is asked for 32 bytes per tensor instead of about 86 KB per chunk.
ChaCha20 is a standard stream cipher, not an SP 800-90A DRBG, which matters
only for FIPS validation. The Python reference reads the operating system
for every byte (:func:`os.urandom` semantics); the distribution is the same
either way.

``set_threads(k)`` (or ``PYFE4AI_THREADS``) lets one call use ``k`` threads
over the chunks of a tensor; the default is one, the most
:data:`MAX_THREADS` (``set_threads`` refuses more, the environment variable
is clamped to it, and a value that is not an integer raises ``ValueError``
at import).
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading

import gmpy2 as gp

def _load():
    """The extension module, or ``None`` with the import error."""
    import importlib

    try:
        return importlib.import_module("pyfe4ai._rust_native"), None
    except ImportError as exc:  # pragma: no cover - depends on the build
        return None, exc


_ext, _IMPORT_ERROR = _load()

BACKENDS = ("auto", "python", "native")
_ENV_BACKEND = "PYFE4AI_BACKEND"
_ENV_THREADS = "PYFE4AI_THREADS"

#: Largest Gaussian width sampled natively (that of the double-precision
#: sampler of :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_matrix`).
MAX_NATIVE_SIGMA = 2 ** 40
#: Moduli must be below this bound (four 64-bit words with 3 bits of headroom).
MAX_NATIVE_Q = 2 ** 253


def _check_backend(name) -> str:
    if name not in BACKENDS:
        raise ValueError("unknown backend {!r}; choose one of {}".format(name, ", ".join(BACKENDS)))
    return name


#: Most threads one native call may use (``set_threads`` refuses more;
#: ``PYFE4AI_THREADS`` is clamped to it).
MAX_THREADS = max(64, 4 * (os.cpu_count() or 1))


def _env_threads() -> int:
    """``PYFE4AI_THREADS`` (default 1), clamped to ``1 .. MAX_THREADS``."""
    value = os.environ.get(_ENV_THREADS, "").strip() or "1"
    try:
        threads = int(value)
    except ValueError:
        raise ValueError("{} must be an integer, got {!r}".format(_ENV_THREADS, value)) from None
    return min(max(1, threads), MAX_THREADS)


_state = threading.local()
_default_backend = _check_backend(os.environ.get(_ENV_BACKEND, "auto"))
_threads = _env_threads()


def available() -> bool:
    """Whether the native extension is built and importable."""
    return _ext is not None


def get_backend() -> str:
    """The selected backend (``"auto"``, ``"python"`` or ``"native"``)."""
    return getattr(_state, "backend", None) or _default_backend


def set_backend(name: str) -> None:
    """Select the backend for the process (see the module docstring).

    Raises:
        ValueError: For an unknown name.
        ImportError: For ``"native"`` when the extension is not built.
    """
    global _default_backend
    _check_backend(name)
    if name == "native" and _ext is None:
        raise ImportError("the native extension is not built: {}".format(_IMPORT_ERROR))
    _default_backend = name


@contextlib.contextmanager
def backend(name: str):
    """Use backend *name* inside a ``with`` block (this thread only)."""
    _check_backend(name)
    if name == "native" and _ext is None:
        raise ImportError("the native extension is not built: {}".format(_IMPORT_ERROR))
    previous = getattr(_state, "backend", None)
    _state.backend = name
    try:
        yield
    finally:
        _state.backend = previous


def active_backend() -> str:
    """``"native"`` if the native path is used where it applies, else ``"python"``."""
    return "native" if (_ext is not None and get_backend() != "python") else "python"


def get_threads() -> int:
    """Threads one native call may use (see :func:`set_threads`)."""
    return _threads


def set_threads(threads: int) -> None:
    """Number of threads one native call may use (over the chunks of a tensor),
    at most :data:`MAX_THREADS`."""
    global _threads
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= MAX_THREADS:
        raise ValueError("threads must be an integer between 1 and {}".format(MAX_THREADS))
    _threads = threads


def implementation() -> str | None:
    """``"rust"``, or ``None`` when the extension is not built."""
    return None if _ext is None else "rust"


def ec_ext():
    """The extension's ristretto255 functions, or ``None`` when the Python
    group of :mod:`pyfe4ai.utils.ristretto255` applies."""
    if _ext is None or get_backend() == "python":
        return None
    return _ext


def describe() -> dict:
    """Backend selection and build information, for benchmark records."""
    info = {"available": _ext is not None, "implementation": implementation(), "backend": get_backend(),
            "threads": _threads}
    if _ext is not None:
        info.update(_ext.build_info())
    else:
        info["import_error"] = str(_IMPORT_ERROR)
    return info


@functools.lru_cache(maxsize=64)
def _facct_table(sigma_q):
    """The extension's tables of
    :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_facct` for the exact
    width *sigma_q*: k, the base CDT and ``-neg_c`` (rounded down to a 128-bit
    mantissa) for the fixed-point acceptance test, and the exact threshold of
    the Python reference for the tests it cannot decide (about one in 2^99)."""
    from pyfe4ai.utils import sampling_utils as su

    params = su._facct_params(sigma_q)
    # c = -neg_c exactly: negate the mantissa (negating the mpfr would round it
    # to the context's 53 bits; crypto review 2026-10, finding M4)
    mant, exp = params.neg_c.as_mantissa_exp()
    mant, exp = -int(mant), int(exp)
    shift = mant.bit_length() - 128  # a 128-bit mantissa: truncated, or extended exactly
    mant = mant >> shift if shift > 0 else mant << -shift
    precision, neg_c, k = su.FACCT_PRECISION, params.neg_c, params.k
    scale = 1 << precision

    def exact(x: int, y: int) -> int:
        with gp.context(precision=precision + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(neg_c * (y * (y + 2 * k * x))) * scale))

    return _ext.FacctTable(k, params.y_bits, list(params.cdt), mant, exp + shift, exact)


@functools.lru_cache(maxsize=64)
def _facct_wide_table(sigma_q):
    """``_facct_table`` for a wide width (``2^50 < k < 2^448``): the
    extension's ``FacctWideTable`` (multi-word k, y and samples)."""
    from pyfe4ai.utils import sampling_utils as su

    params = su._facct_params(sigma_q)
    mant, exp = params.neg_c.as_mantissa_exp()  # c = -neg_c exactly (finding M4)
    mant, exp = -int(mant), int(exp)
    shift = mant.bit_length() - 128
    mant = mant >> shift if shift > 0 else mant << -shift
    precision, neg_c, k = su.FACCT_PRECISION, params.neg_c, params.k
    scale = 1 << precision

    def exact(x: int, y: int) -> int:
        with gp.context(precision=precision + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(neg_c * (y * (y + 2 * k * x))) * scale))

    return _ext.FacctWideTable(k, params.y_bits, list(params.cdt), mant, exp + shift, exact)


def facct_table(sigma):
    """Native tables of the exact sampler for *sigma* (see ``_facct_table``)."""
    from pyfe4ai.utils import sampling_utils as su

    return _facct_table(su._exact(sigma))


#: Largest FACCT double-sampling factor k of the native kernel (samples in an i64)
#: and, beyond it, of the wide kernel (k, y and samples in 64-bit words).
MAX_NATIVE_FACCT_K = 2 ** 50
MAX_NATIVE_FACCT_WIDE_K = 2 ** 448 - 1


def gaussian_facct(sigma, count: int, rng):
    """:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_facct` natively;
    ``None`` when the Python code applies.

    The extension draws ``getrandbits`` in the reference's order and makes its
    decisions (fixed-point acceptance tests with the Python threshold for the
    undecided ones; see ``_facct_table``), so for a given bit source the
    samples are the reference's. Bits: from *rng* if it is an
    :class:`~pyfe4ai.utils.sampling_utils.XofRandom` -- sampled with the GIL
    and the XofRandom claim lock held, since the number of calls depends on
    the samples, and its counter then advanced by the calls used -- else, for
    ``rng=None`` with the module's OS source, from the per-call DRBG (stream
    ``c`` for samples ``256 c`` to ``256 c + 255``, on :func:`get_threads`
    threads; or ``_random_override`` in the tests). Widths with ``k <=
    2^50`` (``sigma`` below about ``2^49.8``) use the kernel of
    ``rust/core/src/facct.rs``, wider ones up to ``k < 2^448`` (``sigma``
    below about ``2^447.7``) that of ``facct_wide.rs`` (the same decisions,
    with the error argument in its header); the MPFR context does not matter
    (the reference sets its own).
    """
    if _ext is None or get_backend() == "python" or not hasattr(_ext, "facct_draw"):
        return None
    from pyfe4ai.utils import sampling_utils as su

    if count <= 0 or count > MAX_NATIVE_BIGINT_COUNT:
        return None
    try:
        sigma_q = su._exact(sigma)
        k = su._facct_params(sigma_q).k
    except TypeError:  # an unhashable width: the Python code
        return None
    if k <= MAX_NATIVE_FACCT_K:
        draw_fn, make_table = _ext.facct_draw, _facct_table
    elif k <= MAX_NATIVE_FACCT_WIDE_K and hasattr(_ext, "facct_draw_wide"):
        draw_fn, make_table = _ext.facct_draw_wide, _facct_wide_table
    else:
        return None
    if rng is None:
        if su._OS_RANDOM is not su._OS_SOURCE:
            return None
        kwargs = {"threads": _threads}
        rnd = random_input(512 * count + 4096)  # tests: a byte stream, as _OSRandomBuffer reads the OS
        if rnd is not None:
            kwargs["rnd"] = rnd
        kwargs.update(seed_kwargs())
        values, _ = draw_fn(make_table(sigma_q), count, **kwargs)
    elif type(rng) is su.XofRandom:
        if rng._counter >= 2 ** 63:  # near the 64-bit counters: the reference's behaviour there
            return None
        table = make_table(sigma_q)

        def draw(start):
            values, end = draw_fn(table, count, xof_head=rng._seed, xof_counter=start)
            return values, end - start

        values = rng._claim_dynamic(draw)
    else:
        return None
    return [gp.mpz(v) for v in values]


def noise_kwargs(*sigmas, sampler: str) -> dict:
    """Keyword arguments of the native encryption calls for the scheme's
    *sampler*: the FACCT tables of the widths, or nothing (rounded)."""
    if sampler != "facct":
        return {}
    return {"facct": tuple(facct_table(s) for s in sigmas)}


#: Tests set this to a function ``nbytes -> bytes`` to feed the native sampler
#: the bytes they also feed the Python one; ``None``: the OS source.
_random_override = None


#: Tests set this to a 32-byte seed for the extension's per-call DRBG (see the
#: module docstring); ``None``: a fresh seed from the OS for every call. With
#: it set, every call reads the streams of that one seed (``shamir_share`` and
#: ``zq_replicated_deal`` stream 0), so every call draws the same randomness:
#: tests only, never in use.
_seed_override = None


def seed_kwargs() -> dict:
    """Keyword arguments of the tensor calls: a test-injected DRBG seed, if any."""
    if _seed_override is None:
        return {}
    return {"seed": _seed_override}


def random_input(nbytes: int):
    """Random bytes for a native call: ``None`` (the extension reads the OS
    source itself) unless a test installed ``_random_override``."""
    return None if _random_override is None else _random_override(nbytes)


def gauss_bytes(count: int) -> int:
    """Random bytes the native sampler consumes for *count* samples."""
    return 14 * ((count + 1) // 2)


#: Widths below ``2^MAX_NATIVE_BIGINT_BITS`` (and uniforms of at most
#: ``MAX_NATIVE_BIGINT_PREC`` bits) are sampled natively by :func:`gaussian_bigint`.
MAX_NATIVE_BIGINT_BITS = 448
MAX_NATIVE_BIGINT_PREC = 768
#: Largest exponent (in absolute value) of ``sig = mantissa 2^exponent`` and
#: most samples per call sampled natively (more: the Python code).
MAX_NATIVE_BIGINT_EXP = 2 ** 20
MAX_NATIVE_BIGINT_COUNT = 2 ** 24
#: Tests set this to widen the decision margin of :func:`gaussian_bigint` by
#: ``2^k``, so that many samples take the exact Python formula.
_bigint_widen = 0
_TRAPS = ("trap_underflow", "trap_overflow", "trap_inexact", "trap_invalid", "trap_erange", "trap_divzero")


@functools.lru_cache(maxsize=256)
def _bigint_width(sigma, prec: int):
    """``(sig, mantissa bytes, exponent)``: ``sig = mpfr(sigma)`` at *prec* bits
    (round to nearest; it depends only on the value of *sigma*) as
    :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint` computes it,
    with ``sig = mantissa 2^exponent`` (odd mantissa, little-endian); ``None``
    beyond the native widths."""
    with gp.context(gp.get_context(), precision=prec, round=gp.RoundToNearest):
        sig = gp.mpfr(sigma)
    mant, exp = (int(v) for v in sig.as_mantissa_exp())
    if mant <= 0:
        return None
    zeros = (mant & -mant).bit_length() - 1
    mant, exp = mant >> zeros, exp + zeros
    if (
        mant.bit_length() + exp > MAX_NATIVE_BIGINT_BITS or mant.bit_length() > MAX_NATIVE_BIGINT_PREC
        or abs(exp) > MAX_NATIVE_BIGINT_EXP
    ):
        return None
    return sig, mant.to_bytes((mant.bit_length() + 7) // 8, "little"), exp


def gaussian_bigint(sigma, count: int, rng, prec: int):
    """:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint` with the
    rounded sampler, natively; ``None`` when the Python code applies.

    The extension reads ``b1, b2 = getrandbits(prec)`` per sample as the
    reference does -- from *rng* if it is an
    :class:`~pyfe4ai.utils.sampling_utils.XofRandom` (whose counter advances
    as in Python), else, for ``rng=None``, from its per-call DRBG (stream
    ``c`` for samples ``256 c`` to ``256 c + 255``; or from
    ``_random_override`` in the tests) -- and computes
    ``X = sig sqrt(-2 ln u1) cos(2 pi u2)`` in fixed point with an error
    below ``2^(s + 10 - 64 n)`` (``sig < 2^s``, ``n = ceil((s + 64) / 64)``
    fraction words). The reference's MPFR value is within ``2^-118`` of
    ``X``; whenever ``X`` is farther than ``2^(max(s + 16 - 64 n, -100) + 1)``
    from every half-integer both round to the same integer. The other samples
    (at most about one in ``2^45``, when ``s`` is just below a multiple of 64;
    about one in ``2^62`` at the ``s = 47`` of SIFE FullySec) are computed
    here with the reference formula from
    the same ``b1, b2``. So for a given bit source the samples are the
    reference's. Applies to widths below ``2^448`` (exponents within
    ``2^20``) and up to ``2^24`` samples per call, under an MPFR context like
    the default one (round to nearest, no traps, no subnormals, exponent range
    at least ``2^20``); any other *rng* (a replaced
    ``sampling_utils._CSPRNG`` included) keeps the Python code. The
    ``XofRandom`` calls of the samples are claimed (its counter advanced, as
    ``getrandbits`` claims one) before the extension runs, so a thread drawing
    from the same ``XofRandom`` meanwhile gets other calls.
    """
    if _ext is None or get_backend() == "python" or not hasattr(_ext, "gauss_bigint"):
        return None
    from pyfe4ai.utils import sampling_utils as su

    if rng is None:
        if su._CSPRNG is not su._SYSTEM_RANDOM:
            return None
        xof = None
    elif type(rng) is su.XofRandom:
        xof = rng
    else:
        return None
    if prec > MAX_NATIVE_BIGINT_PREC or count > MAX_NATIVE_BIGINT_COUNT:
        return None
    ctx = gp.get_context()
    if (
        ctx.round != gp.RoundToNearest or any(getattr(ctx, t) for t in _TRAPS) or ctx.subnormalize
        or ctx.emax < MAX_NATIVE_BIGINT_EXP or ctx.emin > -MAX_NATIVE_BIGINT_EXP
    ):
        return None
    try:
        width = _bigint_width(sigma, prec)
    except TypeError:  # an unhashable sigma (gmpy2.xmpz): the same conversion, uncached
        width = _bigint_width.__wrapped__(sigma, prec)
    if width is None:
        return None
    sig, mant, exp = width
    kwargs = {"threads": _threads, "widen": _bigint_widen}
    if xof is not None:
        if xof._counter + 2 * count > 2 ** 64:  # the reference raises OverflowError there
            return None
        # the calls of these samples, claimed as XofRandom.getrandbits claims one
        kwargs.update(xof_head=xof._seed, xof_counter=xof._claim(2 * count))
    else:
        rnd = random_input(2 * count * ((prec + 7) // 8))
        if rnd is not None:
            kwargs["rnd"] = rnd
        kwargs.update(seed_kwargs())
    values, holes = _ext.gauss_bigint(count, prec, mant, exp, **kwargs)
    out = [v if v is None else gp.mpz(v) for v in values]
    if holes:
        from pyfe4ai.utils.sampling_utils import _box_muller_rint

        pad = 8 * ((prec + 7) // 8) - prec
        with gp.context(ctx, precision=prec):
            two_pi = 2 * gp.const_pi()
            scale = gp.mpfr(2) ** prec
            for i, b1, b2 in holes:
                out[i] = _box_muller_rint(int.from_bytes(b1, "big") >> pad, int.from_bytes(b2, "big") >> pad,
                                          two_pi, sig, scale)
    return out


def sampling_ok(*sigmas, sampler: str) -> bool:
    """Whether Gaussians of these widths may be sampled natively with the
    scheme's *sampler*: rounded, or FACCT when the extension has it."""
    if not all(0 < float(s) <= MAX_NATIVE_SIGMA for s in sigmas):
        return False
    return sampler == "rounded" or (sampler == "facct" and hasattr(_ext, "FacctTable"))


class RingContext:
    """Native NTT context of ``Z_q[x] / (x^n + 1)`` (root and tables of
    :func:`~pyfe4ai.utils.ring_lwe_utils.poly_mul`).

    With *factors* (an RNS modulus, see
    :func:`~pyfe4ai.utils.ring_lwe_utils.register_rns_modulus`) the native
    code runs one 64-bit NTT per factor, with the root ``root mod q_i``, and
    converts to and from ``[0, q)`` (Garner) where values cross the
    boundary. Two factors with ``q < 2^124`` (the MIFE / MCFE prototype)
    get the context of their own kernels (``legacy_rns``; packed values are
    128-bit words); other RNS moduli (up to four factors, ``q < 2^253``: the
    threshold MCFE) a ring context whose kernels are those of a prime ring,
    with the values of a ring mod q.
    """

    legacy_rns = False

    def __init__(self, q, ring_n: int, root, factors=None) -> None:
        self.q = gp.mpz(q)
        self.ring_n = ring_n
        if factors is None:
            self.cap = _ext.ctx_new(int(q), ring_n, int(root))
        elif len(factors) == 2 and int(q).bit_length() <= 124 and hasattr(_ext, "rns_ctx_new"):
            self.cap = _ext.rns_ctx_new(
                int(q), ring_n, [int(f) for f in factors], [int(root) % int(f) for f in factors]
            )
            self.legacy_rns = True
        else:
            self.cap = _ext.ctx_new_rns(
                int(q), ring_n, [int(f) for f in factors], [int(root) % int(f) for f in factors]
            )
        info = _ext.ctx_info(self.cap)
        self.word_bits = info["word_bits"]
        self.word_bytes = self.word_bits // 8

    def pack(self, values) -> bytes:
        """Values (ints, mpz or decimal strings) mod q as packed words."""
        return _ext.pack(self.cap, values)

    def unpack(self, data: bytes) -> list:
        return _ext.unpack(self.cap, data)

    def digits(self, data: bytes) -> list:
        return _ext.unpack(self.cap, data, True)

    def ntt_mont(self, values) -> bytes:
        """NTT of *values* in the form the native encryption / decryption keys use."""
        return _ext.ntt_mont(self.cap, self.pack(values))


_contexts: dict = {}
_contexts_lock = threading.Lock()


def _reset_contexts_lock() -> None:
    global _contexts_lock
    _contexts_lock = threading.Lock()  # a forked child: the parent's may be held while a context is built


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_contexts_lock)


def ring_context(q, ring_n: int) -> RingContext | None:
    """The native context for ``(q, ring_n)``, or ``None`` when the Python
    path applies (backend ``"python"``, extension not built, ``q >= 2^253``
    or no ``2 ring_n``-th root of unity)."""
    if _ext is None or get_backend() == "python":
        return None
    from pyfe4ai.utils.ring_lwe_utils import _get_ntt_root, rns_factors

    factors = rns_factors(q)
    key = (int(q), int(ring_n), factors)
    ctx = _contexts.get(key)
    if ctx is not None or key in _contexts:
        return ctx
    ctx = None
    if 2 < key[0] < MAX_NATIVE_Q and (factors is None or hasattr(_ext, "ctx_new_rns")):
        root = _get_ntt_root(gp.mpz(q), ring_n)
        if root is not None:
            try:
                ctx = RingContext(q, ring_n, root, factors)
            except ValueError:
                ctx = None
    with _contexts_lock:
        _contexts.setdefault(key, ctx)
    return _contexts[key]


def ext():
    """The extension module (``None`` if not built)."""
    return _ext
