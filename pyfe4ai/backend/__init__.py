"""Backends: the native (Rust) or the Python implementation of the hot operations.

The scheme and primitive layers do not call the native extension
``pyfe4ai._rust_native`` themselves: they call the operations of
:mod:`pyfe4ai.backend.interface` -- NTTs and products, ring contexts and their
linear combinations, label hashes, the Ring-LWE encryption and decryption
kernels, the Gaussian samplers, the dealing and sums of the threshold key
sharings, ristretto255 and the dense-LWE label sums -- on the implementation
:func:`ops` returns (:func:`ring` for the ring of ``(q, ring_n)``). Two
implementations stand behind the interface:

* :mod:`pyfe4ai.backend.python_impl` -- the reference, in Python;
* :mod:`pyfe4ai.backend.native_impl` -- the extension (Rust and PyO3, sources
  in ``rust/``; built by ``pip install`` when a Rust toolchain is available,
  or in a source checkout by ``tools/build_rust.py``), which computes the same
  values (same NTT root and tables, the same samplers on the same kind of
  random bytes, the same hash-to-``Z_q`` reductions) and hands every input it
  does not handle -- moduli of ``2^253`` or more, rings without a ``2
  ring_n``-th root of unity, Gaussian widths above ``2^40`` inside the
  encryption calls, other value types or layouts -- to the reference.
  ``tests/test_backend_interface.py`` runs every operation through both and
  requires the same outputs.

The extension runs the Ring-LWE MIFE / MCFE hot paths -- NTTs, Gaussian
sampling, label pads, the encryption, decryption and key-derivation loops of
the ndarray helpers -- one call per tensor, with the GIL released and
optionally several threads -- and of the threshold MCFE
(``mcfe/ring_lwe_threshold``): the Shamir sharing of the functional key, the
parsing, digests and NTT-domain products of a partial decryption and the
combination. The replicated key sharings of the three LWE threshold layers
(``"replicated"``, the default for ``t < s``, and ``"replicated-seeded"``) are
dealt and added up natively for any odd ``q`` below ``2^253``, with no NTT
ring (:func:`~pyfe4ai.schemes.threshold.sharing.share_replicated`,
:func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled`). It also samples the wide rounded Gaussians of
:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint` (SIFE FullySec,
ALS16 and the flooding noise of the threshold schemes) with the reference's
integers, and has the dense-LWE label sums, the ristretto255 group of the
``*/ddh_ristretto255`` schemes and the BLS12-381 pairing group
(:mod:`pyfe4ai.primitives.pairing.bls12_381`, which has no Python
implementation: both backends use the extension's).

Selection, per process (``set_backend``), per context (``backend``) or with
the ``PYFE4AI_BACKEND`` environment variable:

``"auto"`` (default)
    native where the extension is built and the parameters are supported,
    Python otherwise;
``"python"``
    always the Python implementation;
``"native"``
    like ``"auto"``, but :func:`set_backend` and :func:`backend` raise
    ``ImportError`` if the extension is not built (the environment variable
    alone does not: without the extension the Python implementation runs).

The native path applies to moduli below ``2^253`` with a ``2 ring_n``-th root
of unity (all moduli the parameter generators derive, including the flooded
moduli of ``mcfe/ring_lwe_threshold``), in 64-bit words below ``2^63``,
128-bit words below ``2^125`` and 256-bit words above -- and to the RNS
moduli of ``ring_modulus="rns"``: ``q = q1 q2`` (MIFE / MCFE Ring-LWE), in two
64-bit residues per coefficient, and the products of up to four primes of
the threshold MCFE, whose 256-bit ring runs one 64-bit NTT per prime --, and samples natively
Gaussians of widths up to ``2^40`` inside its encryption calls (wider ones
come from :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint`)
with the scheme's sampler (its recorded ``gaussian_sampler``, FACCT by
default, see :func:`~pyfe4ai.primitives.sampling.scheme_sampler`): the
rounded Gaussian, consuming seven
random bytes per uniform as ``random.SystemRandom.random``, or the exact
FACCT sampler (``sampler="facct"``), which draws ``getrandbits`` in the order
of :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct` and makes
its decisions -- inside the encryption calls, and for
:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct` itself
(:meth:`~pyfe4ai.backend.native_impl.NativeBackend.gaussian_facct`): the
tables come from Python (:func:`~pyfe4ai.backend.native_impl.facct_table`,
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
either way. Tests feed both implementations the same bytes through the
hooks ``_random_override`` (the native calls' random bytes) and
``_seed_override`` (a fixed DRBG seed) of this module.

``set_threads(k)`` (or ``PYFE4AI_THREADS``) lets one call use ``k`` threads
over the chunks of a tensor; the default is one, the most
:data:`MAX_THREADS` (``set_threads`` refuses more, the environment variable
is clamped to it, and a value that is not an integer raises ``ValueError``
at import).
"""

from __future__ import annotations

import contextlib
import os
import threading


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

#: Tests set this to a function ``nbytes -> bytes`` to feed the native calls
#: the bytes they also feed the Python ones; ``None``: the OS source (see
#: :func:`~pyfe4ai.backend.native_impl.random_input`).
_random_override = None

#: Tests set this to a 32-byte seed for the extension's per-call DRBG (see the
#: module docstring); ``None``: a fresh seed from the OS for every call. With
#: it set, every call reads the streams of that one seed (``shamir_share`` and
#: ``zq_replicated_deal`` stream 0), so every call draws the same randomness:
#: tests only, never in use.
_seed_override = None

#: Tests set this to widen the decision margin of the native
#: :meth:`~pyfe4ai.backend.native_impl.NativeBackend.gaussian_bigint` by
#: ``2^k``, so that many samples take the exact Python formula.
_bigint_widen = 0


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


def describe() -> dict:
    """Backend selection and build information, for benchmark records."""
    info = {"available": _ext is not None, "implementation": implementation(), "backend": get_backend(),
            "threads": _threads}
    if _ext is not None:
        info.update(_ext.build_info())
    else:
        info["import_error"] = str(_IMPORT_ERROR)
    return info


def ext():
    """The extension module (``None`` if not built); for the tests of its kernels."""
    return _ext


def ops() -> "interface.Backend":
    """The implementation of the backend operations (:mod:`~pyfe4ai.backend.interface`)
    for this thread's selection: :class:`~pyfe4ai.backend.native_impl.NativeBackend`
    where :func:`active_backend` is ``"native"``, else
    :class:`~pyfe4ai.backend.python_impl.PythonBackend`."""
    if _ext is not None and get_backend() != "python":
        return _NATIVE
    return _PYTHON


def ring(q, ring_n: int) -> "interface.Ring":
    """``ops().ring(q, ring_n)``: the ring ``Z_q[x] / (x^ring_n + 1)`` of the
    selected implementation (natively where the extension supports it)."""
    return ops().ring(q, ring_n)


from pyfe4ai.backend import interface  # noqa: E402
from pyfe4ai.backend import python_impl  # noqa: E402
from pyfe4ai.backend import native_impl  # noqa: E402

_PYTHON = python_impl.PythonBackend()
_NATIVE = native_impl.NativeBackend()

# Transitional, until every scheme module calls the interface: the native
# helpers under the names the scheme modules read here.
from pyfe4ai.backend.native_impl import MAX_NATIVE_BIGINT_BITS  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_BIGINT_COUNT  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_BIGINT_EXP  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_BIGINT_PREC  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_FACCT_K  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_FACCT_WIDE_K  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_Q  # noqa: E402,F401
from pyfe4ai.backend.native_impl import MAX_NATIVE_SIGMA  # noqa: E402,F401
from pyfe4ai.backend.native_impl import RingContext  # noqa: E402,F401
from pyfe4ai.backend.native_impl import _facct_table  # noqa: E402,F401
from pyfe4ai.backend.native_impl import facct_table  # noqa: E402,F401
from pyfe4ai.backend.native_impl import gauss_bytes  # noqa: E402,F401
from pyfe4ai.backend.native_impl import noise_kwargs  # noqa: E402,F401
from pyfe4ai.backend.native_impl import random_input  # noqa: E402,F401
from pyfe4ai.backend.native_impl import ring_context  # noqa: E402,F401
from pyfe4ai.backend.native_impl import sampling_ok  # noqa: E402,F401
from pyfe4ai.backend.native_impl import seed_kwargs  # noqa: E402,F401


def ec_ext():  # transitional
    """The extension's ristretto255 functions, or ``None`` on the Python backend."""
    if _ext is None or get_backend() == "python":
        return None
    return _ext
