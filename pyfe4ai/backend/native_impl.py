"""The native implementation of the backend operations: ``pyfe4ai._rust_native``.

:class:`NativeBackend` and :class:`RingContext` (its :class:`~pyfe4ai.backend.interface.Ring`)
subclass the reference (:mod:`pyfe4ai.backend.python_impl`) and run the
extension's kernels where they apply; every input a kernel does not handle
goes to the reference (``super()``), which is what decides "native or
Python" for the given parameters. This is the only module that calls the
extension (the extension object is ``pyfe4ai.backend._ext``, read at
every call so that tests can wrap it), and the only one that assembles a
native call's randomness (:func:`random_input`, :func:`seed_kwargs`), noise
tables (:func:`noise_kwargs`) and threads (:func:`pyfe4ai.backend.get_threads`).

The applicability rules (see :mod:`pyfe4ai.backend`): moduli below
:data:`MAX_NATIVE_Q` with a ``2 ring_n``-th root of unity (and the RNS moduli
of :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.register_rns_modulus`),
Gaussian widths up to :data:`MAX_NATIVE_SIGMA` inside the encryption calls
with the rounded or the FACCT sampler (:func:`sampling_ok`), the rounded
wide Gaussians of :meth:`NativeBackend.gaussian_bigint` and the FACCT
samples of :meth:`NativeBackend.gaussian_facct` within their bounds.
"""

from __future__ import annotations

import functools
import os
import threading

import gmpy2 as gp

from pyfe4ai import backend as _b
from pyfe4ai.backend.python_impl import PythonBackend
from pyfe4ai.backend.python_impl import PythonRing

#: Largest Gaussian width sampled natively (that of the double-precision
#: sampler of :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_matrix`).
MAX_NATIVE_SIGMA = 2 ** 40
#: Moduli must be below this bound (four 64-bit words with 3 bits of headroom).
MAX_NATIVE_Q = 2 ** 253


def random_input(nbytes: int):
    """Random bytes for a native call: ``None`` (the extension reads the OS
    source itself) unless a test installed ``pyfe4ai.backend._random_override``."""
    override = _b._random_override
    return None if override is None else override(nbytes)


def seed_kwargs() -> dict:
    """Keyword arguments of the tensor calls: a test-injected DRBG seed
    (``pyfe4ai.backend._seed_override``), if any."""
    seed = _b._seed_override
    if seed is None:
        return {}
    return {"seed": seed}


def gauss_bytes(count: int) -> int:
    """Random bytes the native sampler consumes for *count* samples."""
    return 14 * ((count + 1) // 2)


# ---------------------------------------------------------------------------- FACCT

@functools.lru_cache(maxsize=64)
def _facct_table(sigma_q):
    """The extension's tables of
    :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct` for the exact
    width *sigma_q*: k, the base CDT and ``-neg_c`` (rounded down to a 128-bit
    mantissa) for the fixed-point acceptance test, and the exact threshold of
    the Python reference for the tests it cannot decide (about one in 2^99)."""
    from pyfe4ai.primitives import sampling as su

    params = su._facct_params(sigma_q)
    # c = -neg_c exactly: negate the mantissa (negating the mpfr would round it
    # to the context's 53 bits)
    mant, exp = params.neg_c.as_mantissa_exp()
    mant, exp = -int(mant), int(exp)
    shift = mant.bit_length() - 128  # a 128-bit mantissa: truncated, or extended exactly
    mant = mant >> shift if shift > 0 else mant << -shift
    precision, neg_c, k = su.FACCT_PRECISION, params.neg_c, params.k
    scale = 1 << precision

    def exact(x: int, y: int) -> int:
        with gp.context(precision=precision + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(neg_c * (y * (y + 2 * k * x))) * scale))

    return _b._ext.FacctTable(k, params.y_bits, list(params.cdt), mant, exp + shift, exact)


@functools.lru_cache(maxsize=64)
def _facct_wide_table(sigma_q):
    """``_facct_table`` for a wide width (``2^50 < k < 2^448``): the
    extension's ``FacctWideTable`` (multi-word k, y and samples)."""
    from pyfe4ai.primitives import sampling as su

    params = su._facct_params(sigma_q)
    mant, exp = params.neg_c.as_mantissa_exp()  # c = -neg_c exactly, as above
    mant, exp = -int(mant), int(exp)
    shift = mant.bit_length() - 128
    mant = mant >> shift if shift > 0 else mant << -shift
    precision, neg_c, k = su.FACCT_PRECISION, params.neg_c, params.k
    scale = 1 << precision

    def exact(x: int, y: int) -> int:
        with gp.context(precision=precision + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(neg_c * (y * (y + 2 * k * x))) * scale))

    return _b._ext.FacctWideTable(k, params.y_bits, list(params.cdt), mant, exp + shift, exact)


def facct_table(sigma):
    """Native tables of the exact sampler for *sigma* (see ``_facct_table``)."""
    from pyfe4ai.primitives import sampling as su

    return _facct_table(su._exact(sigma))


#: Largest FACCT double-sampling factor k of the native kernel (samples in an i64)
#: and, beyond it, of the wide kernel (k, y and samples in 64-bit words).
MAX_NATIVE_FACCT_K = 2 ** 50
MAX_NATIVE_FACCT_WIDE_K = 2 ** 448 - 1


def noise_kwargs(*sigmas, sampler: str) -> dict:
    """Keyword arguments of the native encryption calls for the scheme's
    *sampler*: the FACCT tables of the widths, or nothing (rounded)."""
    if sampler != "facct":
        return {}
    return {"facct": tuple(facct_table(s) for s in sigmas)}


def sampling_ok(*sigmas, sampler: str) -> bool:
    """Whether Gaussians of these widths may be sampled natively with the
    scheme's *sampler*: rounded, or FACCT when the extension has it."""
    if not all(0 < float(s) <= MAX_NATIVE_SIGMA for s in sigmas):
        return False
    return sampler == "rounded" or (sampler == "facct" and hasattr(_b._ext, "FacctTable"))


# ---------------------------------------------------------------------------- wide Gaussians

#: Widths below ``2^MAX_NATIVE_BIGINT_BITS`` (and uniforms of at most
#: ``MAX_NATIVE_BIGINT_PREC`` bits) are sampled natively by :meth:`NativeBackend.gaussian_bigint`.
MAX_NATIVE_BIGINT_BITS = 448
MAX_NATIVE_BIGINT_PREC = 768
#: Largest exponent (in absolute value) of ``sig = mantissa 2^exponent`` and
#: most samples per call sampled natively (more: the Python code).
MAX_NATIVE_BIGINT_EXP = 2 ** 20
MAX_NATIVE_BIGINT_COUNT = 2 ** 24
_TRAPS = ("trap_underflow", "trap_overflow", "trap_inexact", "trap_invalid", "trap_erange", "trap_divzero")


@functools.lru_cache(maxsize=256)
def _bigint_width(sigma, prec: int):
    """``(sig, mantissa bytes, exponent)``: ``sig = mpfr(sigma)`` at *prec* bits
    (round to nearest; it depends only on the value of *sigma*) as
    :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint` computes it,
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


# ---------------------------------------------------------------------------- rings

class RingContext(PythonRing):
    """Native NTT context of ``Z_q[x] / (x^n + 1)`` (root and tables of
    :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.poly_mul`): the
    native :class:`~pyfe4ai.backend.interface.Ring`.

    With *factors* (an RNS modulus, see
    :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.register_rns_modulus`) the native
    code runs one 64-bit NTT per factor, with the root ``root mod q_i``, and
    converts to and from ``[0, q)`` (Garner) where values cross the
    boundary. Two factors with ``q < 2^124`` (the MIFE / MCFE prototype)
    get the context of their own kernels (``legacy_rns``; packed values are
    128-bit words); other RNS moduli (up to four factors, ``q < 2^253``: the
    threshold MCFE) a ring context whose kernels are those of a prime ring,
    with the values of a ring mod q.
    """

    native = True
    legacy_rns = False

    def __init__(self, q, ring_n: int, root, factors=None) -> None:
        super().__init__(q, ring_n)
        ext = _b._ext
        if factors is None:
            self.cap = ext.ctx_new(int(q), ring_n, int(root))
        elif len(factors) == 2 and int(q).bit_length() <= 124 and hasattr(ext, "rns_ctx_new"):
            self.cap = ext.rns_ctx_new(
                int(q), ring_n, [int(f) for f in factors], [int(root) % int(f) for f in factors]
            )
            self.legacy_rns = True
        else:
            self.cap = ext.ctx_new_rns(
                int(q), ring_n, [int(f) for f in factors], [int(root) % int(f) for f in factors]
            )
        info = ext.ctx_info(self.cap)
        self.word_bits = info["word_bits"]
        self.word_bytes = self.word_bits // 8

    def pack(self, values) -> bytes:
        """Values (ints, mpz or decimal strings) mod q as packed words."""
        return _b._ext.pack(self.cap, values)

    def unpack(self, data: bytes) -> list:
        return _b._ext.unpack(self.cap, data)

    def digits(self, data: bytes) -> list:
        return _b._ext.unpack(self.cap, data, True)

    def ntt_mont(self, values) -> bytes:
        """NTT of *values* in the form the native encryption / decryption keys use."""
        return _b._ext.ntt_mont(self.cap, self.pack(values))

    def reference(self) -> PythonRing:
        """The Python ring of the same ``(q, ring_n)``."""
        return _b._PYTHON.ring(self.q, self.ring_n)

    # -- values mod q

    def hash_to_zq(self, material: bytes, count: int, mode: str = "shake256-wide") -> list:
        return [gp.mpz(v) for v in self.unpack(_b._ext.hash_to_zq(self.cap, material, count, **hash_kwargs(mode)))]

    def prepare_rows(self, rows: list):
        """The rows packed (:class:`PackedRow`), each ``ring_n`` words.

        Raises:
            KeyError, TypeError, ValueError: For rows the extension does not read
                (the caller then combines the rows themselves with the reference)."""
        packed = [PackedRow(self.pack(row)) for row in rows]
        if any(len(b) != self.word_bytes * self.ring_n for b in packed):
            raise ValueError("rows of other lengths than ring_n")
        return packed

    def lincomb(self, rows, coeffs: list) -> list:
        rows = [row if type(row) is PackedRow else self.pack(row) for row in rows]
        return _b._ext.lincomb(self.cap, rows, coeffs)

    def label_terms(self, base: list, materials: list, weights: list, eta: int, mode: str,
                    binary: bool = False) -> list:
        if binary and self.legacy_rns:  # the two-prime context has no binary pad terms
            return super().label_terms(base, materials, weights, eta, mode, binary)
        return _b._ext.pad_terms(self.cap, base, materials, weights, eta, _b._threads, **hash_kwargs(mode),
                                 **({"binary": True} if binary else {}))

    # -- Ring-LWE encryption

    def samples_natively(self, *sigmas, sampler: str) -> bool:
        return sampling_ok(*sigmas, sampler=sampler)

    def encryption_key(self, a: list, pk: list, u, delta, hash_mode=None):
        try:
            return _b._ext.enc_key(self.cap, a, pk, u, delta, **({} if hash_mode is None else hash_kwargs(hash_mode)))
        except (TypeError, ValueError, OverflowError):
            return None  # malformed key: the reference reports it

    def encrypt_rows(self, key, reference, w_rows: list, eta: int, sigmas: tuple, sampler: str, binary: bool):
        ring_n = self.ring_n
        if key is None or len(w_rows) != eta or any(len(row) != ring_n for row in w_rows):
            return super().encrypt_rows(key, reference, w_rows, eta, sigmas, sampler, binary)
        sigma2, sigma3 = sigmas
        return _b._ext.encrypt(
            key, w_rows, float(sigma2), float(sigma3), random_input(gauss_bytes(ring_n) * (2 + len(w_rows))), binary,
            **seed_kwargs(), **noise_kwargs(sigma2, sigma3, sampler=sampler),
        )

    def encrypt_tensor(self, key, reference, ints, materials: list, mode: str, sigmas: tuple, sampler: str,
                       binary: bool) -> list:
        if key is None:
            return super().encrypt_tensor(key, reference, ints, materials, mode, sigmas, sampler, binary)
        import numpy as np

        sigma2, sigma3 = sigmas
        return _b._ext.encrypt_tensor(
            key, np.ascontiguousarray(ints, dtype=np.int64), materials, float(sigma2), float(sigma3),
            random_input(3 * gauss_bytes(self.ring_n) * len(materials)), _b._threads, binary, **seed_kwargs(),
            **noise_kwargs(sigma2, sigma3, sampler=sampler),
        )

    def precompute_tensor(self, key, reference, materials: list, mode: str, sigmas: tuple, sampler: str) -> tuple:
        if key is None:
            return super().precompute_tensor(key, reference, materials, mode, sigmas, sampler)
        sigma2, sigma3 = sigmas
        cap = _b._ext.precompute_tensor(
            key, materials, float(sigma2), float(sigma3), random_input(3 * gauss_bytes(self.ring_n) * len(materials)),
            _b._threads, **seed_kwargs(), **noise_kwargs(sigma2, sigma3, sampler=sampler),
        )
        return NativePrecomputation(key, cap), None

    # -- Ring-LWE decryption

    def decryption_key_hat(self, poly):
        try:
            return _b._ext.ntt_mont(self.cap, self.pack(poly))
        except (TypeError, ValueError, OverflowError):
            return None  # a key the extension does not read: the caller's reference steps

    def decrypt_tensor(self, chunks: list, zs: list, ks: list, hats: list, ys: list, delta, bound: int):
        try:
            return _b._ext.decrypt_tensor(self.cap, chunks, zs, ks, hats, ys, delta, bound, _b._threads)
        except NotImplementedError:
            return None


    # -- threshold MCFE over Ring-LWE

    @property
    def threshold_native(self) -> bool:
        return not self.legacy_rns  # the two-prime context has kernels of its own, not these

    def ntt_hat(self, poly):
        """``NTT(poly)`` in the form the native threshold kernels take
        (:meth:`ntt_mont`): values, or the binary words of a bytes object.

        Raises:
            TypeError, ValueError: For values the extension does not read."""
        return self.ntt_mont(bytes(poly) if isinstance(poly, (bytes, bytearray)) else poly)

    def shamir_share(self, values, threshold: int, points: list, binary: bool) -> list:
        from pyfe4ai.core.exceptions import FEKeyError

        width = (int(self.q).bit_length() + 7) // 8
        rnd = random_input(4 * width * len(values) * (threshold - 1) + 64)
        try:
            return _b._ext.shamir_share(self.cap, values, threshold, points, rnd, **seed_kwargs(),
                                        **({"binary": True} if binary else {}))
        except TypeError as exc:  # values that are not ints / decimal strings
            raise FEKeyError("malformed key material") from exc
        except ValueError as exc:
            if rnd is not None and "not enough random bytes" in str(exc):  # a test's random bytes, not the key's
                raise RuntimeError(
                    "native_backend.random_input: the {} test bytes ran out".format(len(rnd))) from exc
            raise FEKeyError("malformed key material") from exc  # values not below q, misaligned words

    def floods_natively(self, width: int) -> bool:
        """Whether the partial-decryption kernels draw the uniform flooding of
        *width* bits themselves (``flood``; the same values as
        :func:`~pyfe4ai.schemes.threshold.roles.uniform_flooding_poly`)."""
        return width < self.q.bit_length()

    def threshold_inputs(self, keys: list, ks: list, rows: list, eta: int):
        return _b._ext.share_inputs(self.cap, keys, ks, rows, eta)

    def threshold_partial(self, packed: list, eta: int, ys: list, sk_hats: list, z_share, delta, multiplier,
                          noise=None, flood=None) -> tuple:
        return _b._ext.share_partial(self.cap, packed, eta, ys, sk_hats, z_share, delta, multiplier,
                                     **_noise_args(noise, flood))

    def threshold_partial_binary(self, cts: list, eta: int, ys: list, sk_hats: list, z_share, delta, multiplier,
                                 noise=None, flood=None) -> tuple:
        return _b._ext.share_partial_bin(self.cap, cts, eta, ys, sk_hats, z_share, delta, multiplier,
                                         **_noise_args(noise, flood))

    def threshold_combine(self, c_y, ws: list, delta, bound, k: int):
        return _b._ext.share_combine(self.cap, c_y, ws, delta, bound, k)

    def threshold_combine_binary(self, c_y: bytes, ws: list, delta, bound, k: int):
        return _b._ext.share_combine_bin(self.cap, c_y, ws, delta, bound, k)


def _noise_args(noise, flood) -> dict:
    """The ``noise`` / ``flood`` keywords of the partial-decryption kernels (as the caller gave them)."""
    out = {}
    if noise is not None:
        out["noise"] = noise
    if flood is not None:
        out["flood"] = flood
    return out


class NativePrecomputation(tuple):
    """The native offline phase of :meth:`RingContext.precompute_tensor`:
    ``(key, capsule)``; the capsule marks every chunk used and wipes it when used
    (``precomputed_info``)."""

    def __new__(cls, key, capsule):
        return super().__new__(cls, (key, capsule))

    def encrypt(self, first: int, ints, binary: bool) -> list:
        """The online phase (:meth:`~pyfe4ai.backend.interface.Ring.encrypt_precomputed`)."""
        import numpy as np

        key, cap = self
        return _b._ext.encrypt_precomputed(key, cap, first, np.ascontiguousarray(ints, dtype=np.int64),
                                          _b._threads, binary)


class PackedRow(bytes):
    """A row packed by :meth:`RingContext.prepare_rows` (the ring's words)."""


def hash_kwargs(mode: str) -> dict:
    """The ``hash_mode`` keyword of the native hash and pad kernels (none for
    the kernels' own ``"shake256-wide"``)."""
    from pyfe4ai.primitives.hashing import LABEL_HASH_CODES, LEGACY_LABEL_HASH

    return {"hash_mode": LABEL_HASH_CODES[mode]} if mode != LEGACY_LABEL_HASH else {}


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
    if _b._ext is None or _b.get_backend() == "python":
        return None
    from pyfe4ai.primitives.lattice.ring_lwe_utils import _get_ntt_root, rns_factors

    factors = rns_factors(q)
    key = (int(q), int(ring_n), factors)
    ctx = _contexts.get(key)
    if ctx is not None or key in _contexts:
        return ctx
    ctx = None
    if 2 < key[0] < MAX_NATIVE_Q and (factors is None or hasattr(_b._ext, "ctx_new_rns")):
        root = _get_ntt_root(gp.mpz(q), ring_n)
        if root is not None:
            try:
                ctx = RingContext(q, ring_n, root, factors)
            except ValueError:
                ctx = None
    with _contexts_lock:
        _contexts.setdefault(key, ctx)
    return _contexts[key]


# ---------------------------------------------------------------------------- ristretto255

_ec_group = None


def _r255():
    """:mod:`pyfe4ai.primitives.groups.ec_group` (its element and scalar encodings)."""
    global _ec_group
    if _ec_group is None:
        from pyfe4ai.primitives.groups import ec_group

        _ec_group = ec_group
    return _ec_group


# ---------------------------------------------------------------------------- BLS12-381

class Bls12381Kernels:
    """The BLS12-381 functions of the extension (arkworks): ``order()``,
    ``random(kind)``, ``init(kind[, value])``, ``generator(kind)``,
    ``deserialize(kind, raw)``, ``deserialize_many(items)``,
    ``hash(kind, msg, dst)``, ``pair``, ``multi_pair``, ``msm``,
    ``pow_many``, ``gt_dlog`` and the element type ``Element``. Like the
    extension module, not picklable."""

    def __init__(self, ext) -> None:
        self._ext = ext  # holds the module: a group holding these is not picklable, as before
        self.order = ext.bls_order
        self.random = ext.bls_random
        self.init = ext.bls_init
        self.generator = ext.bls_generator
        self.deserialize = ext.bls_deserialize
        self.deserialize_many = ext.bls_deserialize_many
        self.hash = ext.bls_hash
        self.pair = ext.bls_pair
        self.multi_pair = ext.bls_multi_pair
        self.msm = ext.bls_msm
        self.pow_many = ext.bls_pow_many
        self.gt_dlog = ext.bls_gt_dlog
        self.Element = ext.Bls12381Element


_bls_kernels = None


def bls12_381_kernels():
    """:class:`Bls12381Kernels` of the extension, or ``None`` if it is not built
    (or built without BLS12-381)."""
    global _bls_kernels
    ext = _b._ext
    if ext is None or not hasattr(ext, "bls_pair"):
        return None
    if _bls_kernels is None or _bls_kernels._ext is not ext:
        _bls_kernels = Bls12381Kernels(ext)
    return _bls_kernels


# ---------------------------------------------------------------------------- the backend

class NativeBackend(PythonBackend):
    """The backend operations on the extension, with the reference
    (:class:`~pyfe4ai.backend.python_impl.PythonBackend`) for what it does not handle."""

    name = "native"
    batched_layers = True

    def ring(self, q, ring_n: int):
        ctx = ring_context(q, ring_n)
        return ctx if ctx is not None else super().ring(q, ring_n)

    # -- lattices and sampling

    def _ntt_ring(self, q, n: int, root):
        """The native context computing the NTT of ``(q, n)`` with *root*
        (the root :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils._get_ntt_root`
        picks), or ``None`` (unsupported modulus, another root)."""
        from pyfe4ai.primitives.lattice.ring_lwe_utils import _get_ntt_root

        if n < 2 or root is None:
            return None
        ctx = ring_context(q, n)
        if ctx is None or gp.mpz(root) != _get_ntt_root(gp.mpz(q), n):
            return None
        return ctx

    def ntt(self, a: list, q, root, inverse: bool = False) -> list:
        ctx = self._ntt_ring(q, len(a), root)
        if ctx is None:
            return super().ntt(a, q, root, inverse)
        return [gp.mpz(v) for v in ctx.unpack(_b._ext.ntt(ctx.cap, ctx.pack(a), inverse))]

    def poly_mul(self, a: list, b: list, q) -> list:
        from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu

        n = len(a)
        root = rlu._get_ntt_root(q, n)
        ctx = self._ntt_ring(q, n, root) if len(b) == n else None
        if ctx is None:
            return rlu._poly_mul_reference(a, b, q, root)
        return [gp.mpz(v) for v in ctx.unpack(_b._ext.poly_mul(ctx.cap, ctx.pack(a), ctx.pack(b)))]

    def gaussian_facct(self, sigma, count: int, rng) -> list:
        """:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct` natively
        where it applies.

        The extension draws ``getrandbits`` in the reference's order and makes its
        decisions (fixed-point acceptance tests with the Python threshold for the
        undecided ones; see ``_facct_table``), so for a given bit source the
        samples are the reference's. Bits: from *rng* if it is an
        :class:`~pyfe4ai.primitives.sampling.XofRandom` -- sampled with the GIL
        and the XofRandom claim lock held, since the number of calls depends on
        the samples, and its counter then advanced by the calls used -- else, for
        ``rng=None`` with the module's OS source, from the per-call DRBG (stream
        ``c`` for samples ``256 c`` to ``256 c + 255``, on
        :func:`~pyfe4ai.backend.get_threads` threads; or ``_random_override`` in
        the tests). Widths with ``k <= 2^50`` (``sigma`` below about
        ``2^49.8``) use the kernel of ``rust/core/src/facct.rs``, wider ones up
        to ``k < 2^448`` (``sigma`` below about ``2^447.7``) that of
        ``facct_wide.rs`` (the same decisions, with the error argument in its
        header); the MPFR context does not matter (the reference sets its own).
        Other inputs: the reference.
        """
        out = self._gaussian_facct(sigma, count, rng)
        return super().gaussian_facct(sigma, count, rng) if out is None else out

    def _gaussian_facct(self, sigma, count: int, rng):
        ext = _b._ext
        if not hasattr(ext, "facct_draw"):
            return None
        from pyfe4ai.primitives import sampling as su

        if count <= 0 or count > MAX_NATIVE_BIGINT_COUNT:
            return None
        try:
            sigma_q = su._exact(sigma)
            k = su._facct_params(sigma_q).k
        except TypeError:  # an unhashable width: the Python code
            return None
        if k <= MAX_NATIVE_FACCT_K:
            draw_fn, make_table = ext.facct_draw, _facct_table
        elif k <= MAX_NATIVE_FACCT_WIDE_K and hasattr(ext, "facct_draw_wide"):
            draw_fn, make_table = ext.facct_draw_wide, _facct_wide_table
        else:
            return None
        if rng is None:
            if su._OS_RANDOM is not su._OS_SOURCE:
                return None
            kwargs = {"threads": _b._threads}
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

    def gaussian_bigint(self, sigma, count: int, rng, prec: int) -> list:
        """:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint` with the
        rounded sampler, natively where it applies.

        The extension reads ``b1, b2 = getrandbits(prec)`` per sample as the
        reference does -- from *rng* if it is an
        :class:`~pyfe4ai.primitives.sampling.XofRandom` (whose counter advances
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
        ``sampling._CSPRNG`` included) keeps the Python code. The
        ``XofRandom`` calls of the samples are claimed (its counter advanced, as
        ``getrandbits`` claims one) before the extension runs, so a thread drawing
        from the same ``XofRandom`` meanwhile gets other calls.
        """
        out = self._gaussian_bigint(sigma, count, rng, prec)
        return super().gaussian_bigint(sigma, count, rng, prec) if out is None else out

    def _gaussian_bigint(self, sigma, count: int, rng, prec: int):
        ext = _b._ext
        if not hasattr(ext, "gauss_bigint"):
            return None
        from pyfe4ai.primitives import sampling as su

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
        kwargs = {"threads": _b._threads, "widen": _b._bigint_widen}
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
        values, holes = ext.gauss_bigint(count, prec, mant, exp, **kwargs)
        out = [v if v is None else gp.mpz(v) for v in values]
        if holes:
            from pyfe4ai.primitives.sampling import _box_muller_rint

            pad = 8 * ((prec + 7) // 8) - prec
            with gp.context(ctx, precision=prec):
                two_pi = 2 * gp.const_pi()
                scale = gp.mpfr(2) ** prec
                for i, b1, b2 in holes:
                    out[i] = _box_muller_rint(int.from_bytes(b1, "big") >> pad, int.from_bytes(b2, "big") >> pad,
                                              two_pi, sig, scale)
        return out

    # -- threshold key sharings over Z_q

    def zq_kernels(self, q) -> bool:
        """Whether the extension does the ``Z_q`` arithmetic of the replicated
        sharings for *q*: built with ``zq_sum``, ``q`` odd in ``[3, 2^253)``
        (no NTT ring needed)."""
        if not hasattr(_b._ext, "zq_sum"):
            return False
        q = int(q)
        return 3 <= q < MAX_NATIVE_Q and bool(q % 2)

    def replicated_deal(self, q, values, n_random: int, binary: bool):
        """The pieces of :func:`~pyfe4ai.schemes.threshold.sharing.share_replicated`
        by the extension -- the same pieces for the same random bytes: each
        random value drawn as ``random.SystemRandom.randrange(0, q)`` draws it,
        from the call's DRBG (or the test bytes of :func:`random_input`; a
        test-installed DRBG seed makes every call read stream 0 of that seed,
        as for the other tensor calls: tests only), in the order of the Python
        code -- or ``None`` when that code applies.

        Raises:
            FEKeyError: For binary values that are not whole words below q.
            RuntimeError: If the test bytes run out (they are bounded; the DRBG
                is not).
        """
        if not self.zq_kernels(q):
            return None
        width = (int(q).bit_length() + 7) // 8
        # test bytes (None otherwise): a draw is rejected with probability below
        # 1/2, so 4 draws per value and 256 more run out with probability below
        # 2^-180 (Hoeffding), also for q just above a power of two
        draws = (len(values) // width if isinstance(values, bytes) else len(values)) * n_random
        rnd = random_input(width * (4 * draws + 256) if draws else 0)
        try:
            return _b._ext.zq_replicated_deal(
                int(q), values, n_random, rnd, **seed_kwargs(), **({"binary": True} if binary else {}))
        except TypeError:  # values other than ints and decimal strings: the Python code reads them
            return None
        except ValueError as exc:
            if binary and "binary values" in str(exc):  # bytes that are not whole words below q
                from pyfe4ai.core.exceptions import FEKeyError

                raise FEKeyError("malformed binary key material") from exc
            if rnd is None:
                raise
            raise RuntimeError(
                "native_backend.random_input: the {} test bytes ran out".format(len(rnd))
            ) from exc

    def zq_sum(self, q, count: int, rows, materials: list, subtract: bool = False, binary: bool = False,
               reference: bool = True):
        if binary:
            if not self.zq_kernels(q):
                return super().zq_sum(q, count, rows, materials, subtract, binary, reference)
            return _b._ext.zq_sum(int(q), count, rows, materials, subtract, True)
        if not self.zq_kernels(q) or not all(len(row) == count for row in rows):
            return super().zq_sum(q, count, rows, materials, subtract, binary, reference)
        try:
            return _b._ext.zq_sum(int(q), count, rows, materials, subtract)
        except TypeError:  # values other than ints and decimal strings: the reference reads them
            return super().zq_sum(q, count, rows, materials, subtract, binary, reference)

    # -- ristretto255 (elements as their 32-byte encodings)

    def r255_add(self, a, b):
        eg = _r255()
        return _b._ext.r255_add(eg._b(a), eg._b(b))

    def r255_sub(self, a, b):
        eg = _r255()
        return _b._ext.r255_sub(eg._b(a), eg._b(b))

    def r255_neg(self, a):
        return _b._ext.r255_neg(_r255()._b(a))

    def r255_mul(self, point, k: int):
        eg = _r255()
        return _b._ext.r255_mul(eg._b(point), eg._k(k))

    def r255_mul_base(self, k: int):
        return _b._ext.r255_mul_base(_r255()._k(k))

    def r255_decode(self, data: bytes):
        if not _b._ext.r255_check(data):
            raise ValueError("invalid encoding")
        return data

    def r255_hash(self, msg: bytes, tag: bytes):
        return _b._ext.r255_hash(msg, tag)

    def r255_dlog(self, point, bound: int):
        return _b._ext.r255_dlog(_r255()._b(point), bound)

    # -- dense LWE and ristretto255 layers (the batched steps)

    def dense_key(self, key_row: list, q):
        """The key row of :meth:`label_dots`: ``(packed, key_row)``, its values
        mod q as little-endian words of ``ceil(bits(q) / 64)`` 64-bit limbs and
        the row itself (plain values: the caller caches and may pickle it)."""
        kl = (int(q).bit_length() + 63) // 64
        return b"".join((int(v) % int(q)).to_bytes(8 * kl, "little") for v in key_row), key_row

    def label_dots(self, materials: list, key, q, dim: int, mode: str) -> list:
        from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH, zq_rejection_parameters

        key, row = key
        if len(row) != dim:  # the kernel takes key rows of the dimension only: the reference
            return super().label_dots(materials, row, q, dim, mode)
        kl = (int(q).bit_length() + 63) // 64
        if mode == LEGACY_LABEL_HASH:
            raw = _b._ext.dense_dots(materials, (int(q).bit_length() + 128 + 7) // 8, dim, key, kl, _b._threads)
        else:  # the accepted candidates instead of the wide blocks, also unreduced
            nb, limit = zq_rejection_parameters(q)
            raw = _b._ext.dense_dots(
                materials, nb, dim, key, kl, _b._threads,
                limit=limit.to_bytes(nb, "big"), rate=168 if mode == "shake128-rejection" else 136,
            )
        step = len(raw) // len(materials)
        return [gp.mpz(int.from_bytes(raw[i:i + step], "little")) % q for i in range(0, len(raw), step)]

    def r255_mcfe_encrypt(self, msgs: list, dst: bytes, values: list, s0: int, s1: int) -> list:
        eg = _r255()
        return _b._ext.r255_mcfe_encrypt(msgs, dst, values, eg._k(s0), eg._k(s1), _b._threads)

    def r255_mcfe_decrypt(self, cts: list, ys: list, d0: int, d1: int, msgs: list, dst: bytes, bound: int):
        eg = _r255()
        return _b._ext.r255_mcfe_decrypt(cts, ys, eg._k(d0), eg._k(d1), msgs, dst, bound, _b._threads)

    # -- BLS12-381

    def bls12_381(self):
        return bls12_381_kernels()
