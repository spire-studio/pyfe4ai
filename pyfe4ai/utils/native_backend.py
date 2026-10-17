"""Optional native backend of the Ring-LWE, dense-LWE and ristretto255 schemes.

The extension ``pyfe4ai._rust_native`` (Rust and PyO3, sources in ``rust/``;
built by ``pip install`` when a Rust toolchain is available, or in a source
checkout by ``tools/build_rust.py``) runs the Ring-LWE MIFE / MCFE hot paths
-- NTTs, Gaussian sampling, label pads, the encryption, decryption and
key-derivation loops of the ndarray helpers -- one call per tensor, with the
GIL released and optionally several threads -- and of the threshold MCFE
(``mcfe/ring_lwe_threshold``): the Shamir sharing of the functional key, the
parsing, digests and NTT-domain products of a partial decryption (its
flooding noise stays Python) and the combination. It also has the dense-LWE label
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
moduli of ``mcfe/ring_lwe_threshold``), in 64-bit words below ``2^62``,
128-bit words below ``2^125`` and 256-bit words above, and samples natively
only for the default rounded Gaussian sampler with widths up to ``2^40``
(wider ones use :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint`
in Python), consuming seven random bytes per uniform as
``random.SystemRandom.random``.

Random bytes. The extension draws them from a DRBG scoped to one call:
every tensor-level call (``encrypt_tensor``, ``precompute_tensor``,
``encrypt`` for one matrix, and ``shamir_share`` for a whole key, which reads
stream 0 in order and draws each coefficient as
``random.SystemRandom.randrange(0, q)`` does) reads one 32-byte seed from the operating system
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
over the chunks of a tensor; the default is one.
"""

from __future__ import annotations

import contextlib
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


_state = threading.local()
_default_backend = _check_backend(os.environ.get(_ENV_BACKEND, "auto"))
_threads = max(1, int(os.environ.get(_ENV_THREADS, "1") or 1))


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
    return _threads


def set_threads(threads: int) -> None:
    """Number of threads one native call may use (over the chunks of a tensor)."""
    global _threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
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


def _sampler_is_rounded() -> bool:
    """The native sampler reproduces the default rounded Gaussian only."""
    from pyfe4ai.utils import sampling_utils

    current = getattr(sampling_utils, "get_gaussian_sampler", None)
    return current is None or current() == "rounded"


#: Tests set this to a function ``nbytes -> bytes`` to feed the native sampler
#: the bytes they also feed the Python one; ``None``: the OS source.
_random_override = None


#: Tests set this to a 32-byte seed for the extension's per-call DRBG (see the
#: module docstring); ``None``: a fresh seed from the OS for every call.
_seed_override = None


def seed_kwargs() -> dict:
    """Keyword arguments of the tensor calls: a test-injected DRBG seed, if any."""
    if _seed_override is None:
        return {}
    return {"seed": _seed_override}


def random_input(nbytes: int):
    """Random bytes for a native call: ``None`` (the extension reads the OS
    source itself) unless a test installed :data:`_random_override`."""
    return None if _random_override is None else _random_override(nbytes)


def gauss_bytes(count: int) -> int:
    """Random bytes the native sampler consumes for *count* samples."""
    return 14 * ((count + 1) // 2)


def sampling_ok(*sigmas) -> bool:
    """Whether Gaussians of these widths may be sampled natively now."""
    return all(0 < float(s) <= MAX_NATIVE_SIGMA for s in sigmas) and _sampler_is_rounded()


class RingContext:
    """Native NTT context of ``Z_q[x] / (x^n + 1)`` (root and tables of
    :func:`~pyfe4ai.utils.ring_lwe_utils.poly_mul`)."""

    def __init__(self, q, ring_n: int, root) -> None:
        self.q = gp.mpz(q)
        self.ring_n = ring_n
        self.cap = _ext.ctx_new(int(q), ring_n, int(root))
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


def ring_context(q, ring_n: int) -> RingContext | None:
    """The native context for ``(q, ring_n)``, or ``None`` when the Python
    path applies (backend ``"python"``, extension not built, ``q >= 2^253``
    or no ``2 ring_n``-th root of unity)."""
    if _ext is None or get_backend() == "python":
        return None
    key = (int(q), int(ring_n))
    ctx = _contexts.get(key)
    if ctx is not None or key in _contexts:
        return ctx
    from pyfe4ai.utils.ring_lwe_utils import _get_ntt_root

    ctx = None
    if 2 < key[0] < MAX_NATIVE_Q:
        root = _get_ntt_root(gp.mpz(q), ring_n)
        if root is not None:
            try:
                ctx = RingContext(q, ring_n, root)
            except ValueError:
                ctx = None
    with _contexts_lock:
        _contexts.setdefault(key, ctx)
    return _contexts[key]


def ext():
    """The extension module (``None`` if not built)."""
    return _ext
