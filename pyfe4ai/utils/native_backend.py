"""Optional native (C++) backend of the Ring-LWE schemes.

The extension ``pyfe4ai._rlwe_native`` (built by ``setup.py`` when a 64-bit
GCC or Clang is available) runs the Ring-LWE MIFE / MCFE hot paths -- NTTs,
Gaussian sampling, label pads, the encryption, decryption and key-derivation
loops of the ndarray helpers -- in C++, one call per tensor, with the GIL
released and optionally several threads. The Python implementation stays the
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

The native path applies to moduli below ``2^125`` with a ``2 ring_n``-th root
of unity (all moduli the parameter generators derive), and samples natively
only for the default rounded Gaussian sampler with widths up to ``2^40``
(wider ones use :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint`
in Python). Random bytes come from the operating system as for
:func:`os.urandom` (``getrandom`` / ``getentropy``), seven per uniform as
``random.SystemRandom.random``.

``set_threads(k)`` (or ``PYFE4AI_THREADS``) lets one call use ``k`` threads
over the chunks of a tensor; the default is one.
"""

from __future__ import annotations

import contextlib
import os
import threading

import gmpy2 as gp

try:  # the optional extension
    from pyfe4ai import _rlwe_native as _ext
except ImportError as _exc:  # pragma: no cover - depends on the build
    _ext = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

BACKENDS = ("auto", "python", "native")
_ENV_BACKEND = "PYFE4AI_BACKEND"
_ENV_THREADS = "PYFE4AI_THREADS"

#: Largest Gaussian width sampled natively (that of the double-precision
#: sampler of :func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_matrix`).
MAX_NATIVE_SIGMA = 2 ** 40
#: Moduli must be below this bound (two 64-bit words with 3 bits of headroom).
MAX_NATIVE_Q = 2 ** 125


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


def describe() -> dict:
    """Backend selection and build information, for benchmark records."""
    info = {"available": _ext is not None, "backend": get_backend(), "threads": _threads}
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
    path applies (backend ``"python"``, extension not built, ``q >= 2^125``
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
