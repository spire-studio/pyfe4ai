"""The backend interface (:mod:`pyfe4ai.backend.interface`).

* Only :mod:`pyfe4ai.backend` reaches the native extension: a static scan of
  every other module of the library (imports of ``pyfe4ai._rust_native`` or
  of the implementation modules, attributes of the backend other than
  :func:`~pyfe4ai.backend.ops` and :func:`~pyfe4ai.backend.ring`, native
  internals such as ``.cap`` or ``ext()``).
* Both implementations implement every operation.
* Bit identity, per operation: the same inputs (and random bytes) through the
  Python implementation (under the ``"python"`` selection, so that what it
  calls runs in Python too) and the native one (under ``"native"``), the same
  outputs -- with a spy on the extension that the native kernel did run. The
  native kernels of a different step sequence (:mod:`pyfe4ai.backend.interface`)
  are compared with the scheme's Python steps they replace. The per-module
  tests (``test_native_backend.py``, ``test_native_sife.py``,
  ``test_native_dmcfe.py``, ``test_replicated_native.py``,
  ``test_dense_lwe_native.py``, ``test_ristretto255_native.py``, the FACCT and
  wide-Gaussian tests, ``test_rust_core.py``) cover more parameters at the
  scheme level.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import inspect
import os
import random

import gmpy2 as gp
import numpy as np
import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import interface
from pyfe4ai.backend import native_impl
from pyfe4ai.backend import python_impl
from pyfe4ai.primitives import hashing
from pyfe4ai.primitives import sampling as su
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu

pytestmark = pytest.mark.core

LIB = os.path.dirname(os.path.abspath(pyfe4ai.__file__))
#: What the scheme and primitive layers may read of pyfe4ai.backend.
ALLOWED_BACKEND_ATTRS = {"ops", "ring"}
#: Attributes that only native internals have: never read outside pyfe4ai/backend.
NATIVE_INTERNALS = {"cap", "ext", "_ext", "ec_ext", "ring_context", "RingContext", "noise_kwargs", "random_input",
                    "seed_kwargs", "sampling_ok", "gauss_bytes", "facct_table", "_facct_table", "pack", "unpack"}


# ------------------------------------------------------------------ the static scan

def _library_modules():
    for root, dirs, files in os.walk(LIB):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        if os.path.relpath(root, LIB).split(os.sep)[0] == "backend":
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


def scan(source: str) -> list:
    """The violations of the backend boundary in a module's *source*."""
    tree = ast.parse(source)
    found, aliases = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if "_rust_native" in a.name or a.name.startswith(("pyfe4ai.backend.native_impl",
                                                                  "pyfe4ai.backend.python_impl")):
                    found.append("line {}: import {}".format(node.lineno, a.name))
                elif a.name == "pyfe4ai.backend" and a.asname:
                    aliases.add(a.asname)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                full = mod + "." + a.name
                if "_rust_native" in full or full.startswith(("pyfe4ai.backend.native_impl", "pyfe4ai.backend.python_impl")):
                    found.append("line {}: from {} import {}".format(node.lineno, mod, a.name))
                elif mod == "pyfe4ai" and a.name == "backend":
                    aliases.add(a.asname or a.name)
                elif mod == "pyfe4ai.backend":
                    found.append("line {}: from pyfe4ai.backend import {}".format(node.lineno, a.name))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
            if any(isinstance(arg, ast.Constant) and "_rust_native" in str(arg.value) for arg in node.args):
                found.append("line {}: import_module of the extension".format(node.lineno))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in aliases and node.attr not in ALLOWED_BACKEND_ATTRS:
                found.append("line {}: {}.{}".format(node.lineno, node.value.id, node.attr))
            elif node.attr in NATIVE_INTERNALS:
                found.append("line {}: .{}".format(node.lineno, node.attr))
    return found


def test_only_the_backend_reaches_the_extension():
    bad = {}
    for path in _library_modules():
        with open(path, encoding="utf-8") as f:
            found = scan(f.read())
        if found:
            bad[os.path.relpath(path, LIB)] = found
    assert bad == {}


@pytest.mark.parametrize("snippet", [
    "from pyfe4ai import _rust_native\n",
    "import pyfe4ai._rust_native as n\n",
    "import importlib\nimportlib.import_module('pyfe4ai._rust_native')\n",
    "from pyfe4ai import backend as native_backend\nnative_backend.ext().ntt(1, 2)\n",
    "from pyfe4ai import backend\nbackend.active_backend()\n",
    "from pyfe4ai.backend import native_impl\n",
    "from pyfe4ai.backend.native_impl import ring_context\n",
    "def f(ctx):\n    return ctx.cap\n",
    "def f(b):\n    return b.ring_context(1, 2)\n",
])
def test_the_scan_finds_what_it_forbids(snippet):
    assert scan(snippet)


def test_the_scan_accepts_the_interface():
    assert scan("from pyfe4ai import backend\nfrom pyfe4ai.backend.interface import ReferenceKey\n"
                "r = backend.ring(17, 4)\nbackend.ops().ntt([1], 17, 2)\n") == []


# ------------------------------------------------------------------ conformance

def test_both_implementations_implement_every_operation():
    for impl, iface in ((python_impl.PythonBackend, interface.Backend), (python_impl.PythonRing, interface.Ring)):
        assert not inspect.isabstract(impl), impl
        assert issubclass(impl, iface)
    assert issubclass(native_impl.NativeBackend, python_impl.PythonBackend)
    assert issubclass(native_impl.RingContext, python_impl.PythonRing)
    with native_backend.backend("python"):
        assert native_backend.ops() is native_backend._PYTHON
        ring = native_backend.ring(rlu.next_ntt_prime(gp.mpz(2) ** 40, 16), 16)
        assert type(ring) is python_impl.PythonRing and not ring.native and ring.reference() is ring


def test_selection():
    if native_backend.available():
        with native_backend.backend("auto"):
            assert native_backend.ops() is native_backend._NATIVE and native_backend.ops().batched_layers
        with native_backend.backend("python"):
            assert native_backend.ops() is native_backend._PYTHON and not native_backend.ops().batched_layers
    else:  # pragma: no cover - depends on the build
        assert native_backend.ops() is native_backend._PYTHON


# ------------------------------------------------------------------ bit identity

native = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
PY, NAT = native_backend._PYTHON, native_backend._NATIVE


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


class Spy:
    """The extension, counting calls by name."""

    def __init__(self, ext) -> None:
        self._real, self.calls = ext, collections.Counter()

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr) or isinstance(attr, type):
            return attr

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return attr(*args, **kwargs)

        return counted


@pytest.fixture
def spy(monkeypatch):
    s = Spy(native_backend.ext())
    monkeypatch.setattr(native_backend, "_ext", s)
    return s


def both(fn):
    """``(fn() under "python", fn() under "native")``."""
    with native_backend.backend("python"):
        py = fn(PY)
    with native_backend.backend("native"):
        nat = fn(NAT)
    return py, nat


class Bytes:
    """A deterministic byte stream, handed out in order."""

    def __init__(self, seed: int, size: int = 1 << 22) -> None:
        self.data, self.pos = random.Random(seed).randbytes(size), 0

    def take(self, n: int) -> bytes:
        self.pos += n
        assert self.pos <= len(self.data), "byte stream exhausted"
        return self.data[self.pos - n:self.pos]


class Uniforms(random.Random):
    """``random.SystemRandom.random`` and ``getrandbits`` on a byte stream (7 bytes per uniform)."""

    def __init__(self, stream: Bytes) -> None:
        self.stream = stream
        super().__init__(0)

    def random(self) -> float:
        return (int.from_bytes(self.stream.take(7), "big") >> 3) * 2 ** -53

    def getrandbits(self, k: int) -> int:
        n = (k + 7) // 8
        return int.from_bytes(self.stream.take(n), "big") >> (8 * n - k)


def same_bytes(fn, seed=1, source="_CSPRNG"):
    """*fn* with the Python implementation drawing from a byte stream (the
    sampling module's *source*) and the native one fed the same stream."""
    with native_backend.backend("python"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(su, source, Uniforms(Bytes(seed)))
        py = fn(PY)
    with native_backend.backend("native"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(native_backend, "_random_override", Bytes(seed).take)
        nat = fn(NAT)
    return py, nat


class StreamBits:
    """``_OSRandomBuffer.getrandbits`` over a byte string."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def getrandbits(self, k: int) -> int:
        n = (k + 7) // 8
        self.pos += n
        return int.from_bytes(self.data[self.pos - n:self.pos], "big") >> (8 * n - k)


def same_bits(fn, seed=1):
    """*fn* with the FACCT reference reading ``getrandbits`` of a byte string
    (``sampling._OS_RANDOM``) and the native calls given the same bytes."""
    data = random.Random(seed).randbytes(1 << 22)
    with native_backend.backend("python"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(su, "_OS_RANDOM", StreamBits(data))
        py = fn(PY)
    with native_backend.backend("native"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(native_backend, "_random_override", lambda n: data)  # the native FACCT reads what it needs
        nat = fn(NAT)
    return py, nat


def _q(bits: int, n: int):
    return rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), n)


WORDS = [(40, 16), (61, 64), (100, 32), (126, 16), (200, 64), (253, 32)]  # 64-, 128-, 256-bit words


# -- Backend.ntt, Backend.poly_mul

@native
@pytest.mark.parametrize("bits,n", WORDS)
def test_ntt_and_poly_mul(bits, n, spy):
    q = _q(bits, n)
    root = rlu._get_ntt_root(q, n)
    rng = random.Random(bits)
    a = [gp.mpz(rng.randrange(-2 * int(q), 2 * int(q))) for _ in range(n)]
    b = [gp.mpz(rng.randrange(int(q))) for _ in range(n)]
    for inverse in (False, True):
        py, nat = both(lambda impl: impl.ntt(a, q, root, inverse))
        assert py == nat and all(type(v) is type(gp.mpz(0)) for v in nat)
    py, nat = both(lambda impl: impl.poly_mul(a, b, q))
    assert py == nat
    assert spy.calls["ntt"] == 2 and spy.calls["poly_mul"] == 1
    other = gp.powmod(root, 3, q)  # another root, a modulus beyond 2^253: the reference
    wide = _q(256, n)
    py, nat = both(lambda impl: (impl.ntt(a, q, other), impl.poly_mul(a, b, wide)))
    assert py == nat and spy.calls["ntt"] == 2 and spy.calls["poly_mul"] == 1


# -- Backend.gaussian_bigint, Backend.gaussian_facct

@native
@pytest.mark.parametrize("sigma", [3.2, gp.mpz(2) ** 47 + 12345, gp.mpz(10) ** 60])
def test_gaussian_bigint(sigma, spy):
    prec = su.bigint_precision(sigma)
    xofs = []

    def run(impl):
        xof = su.XofRandom(b"backend-interface/" + str(sigma).encode())
        out = impl.gaussian_bigint(sigma, 300, xof, prec)
        xofs.append(xof._counter)
        return [int(v) for v in out]

    py, nat = both(run)
    assert py == nat and xofs[0] == xofs[1] and spy.calls["gauss_bigint"] == 1
    py, nat = same_bytes(lambda impl: [int(v) for v in impl.gaussian_bigint(sigma, 300, None, prec)])
    assert py == nat and spy.calls["gauss_bigint"] == 2


@native
@pytest.mark.parametrize("sigma", [3.2, 4030.5, 2.0 ** 40, gp.mpz(2) ** 100])
def test_gaussian_facct(sigma, spy):
    def run(impl):
        xof = su.XofRandom(b"backend-interface-facct/" + str(sigma).encode())
        return [int(v) for v in impl.gaussian_facct(sigma, 300, xof)], xof._counter

    py, nat = both(run)
    assert py == nat and spy.calls["facct_draw"] + spy.calls["facct_draw_wide"] == 1
    py, nat = same_bits(lambda impl: [int(v) for v in impl.gaussian_facct(sigma, 300, None)], seed=5)
    assert py == nat and spy.calls["facct_draw"] + spy.calls["facct_draw_wide"] == 2


# -- Ring.hash_to_zq, Ring.lincomb, Ring.prepare_rows, Ring.label_terms

@native
@pytest.mark.parametrize("bits,n", WORDS)
@pytest.mark.parametrize("mode", hashing.LABEL_HASHES)
def test_hash_to_zq(bits, n, mode, spy):
    q = _q(bits, n)
    material = hashing.hash_to_zq_material("label|{}".format(bits), "backend-interface", mode)
    py, nat = both(lambda impl: [int(v) for v in impl.ring(q, n).hash_to_zq(material, 3 * n, mode)])
    assert py == nat == [int(v) for v in hashing.hash_to_zq_vector("label|{}".format(bits), q, 3 * n,
                                                                     "backend-interface", mode)]
    assert spy.calls["hash_to_zq"] == 1


@native
@pytest.mark.parametrize("bits,n", WORDS)
def test_lincomb_and_prepared_rows(bits, n, spy):
    q = _q(bits, n)
    rng = random.Random(n)
    rows = [[rng.randrange(-int(q), 2 * int(q)) for _ in range(n)] for _ in range(4)]
    rows[1] = [gp.digits(gp.mpz(v) % q) for v in rows[1]]  # decimal strings
    coeffs = [3, -2, 0, 7]
    py, nat = both(lambda impl: impl.ring(q, n).lincomb(rows, coeffs))
    assert py == nat and spy.calls["lincomb"] == 1
    py, nat = both(lambda impl: impl.ring(q, n).lincomb(iter(rows), coeffs))  # read row by row
    assert py == nat
    py, nat = both(lambda impl: impl.ring(q, n).lincomb(impl.ring(q, n).prepare_rows(rows), coeffs))
    assert py == nat
    with native_backend.backend("native"):  # rows the extension does not read: not prepared
        with pytest.raises((TypeError, ValueError)):
            NAT.ring(q, n).prepare_rows([rows[0][:-1]])


@native
@pytest.mark.parametrize("eta,binary", [(1, False), (2, False), (1, True), (3, True)])
@pytest.mark.parametrize("bits,n", [(61, 16), (126, 32), (210, 16)])
def test_label_terms(eta, binary, bits, n, spy):
    q = _q(bits, n)
    rng = random.Random(eta + n)
    base = [gp.digits(rng.randrange(int(q))) for _ in range(n)]
    mats = [[hashing.hash_to_zq_material("u{}|l{}".format(i, l), "pad", "shake128-rejection") for i in range(3)]
            for l in range(4)]
    weights = [[rng.randint(-5, 5) for _ in range(eta)] for _ in range(3)]
    py, nat = both(lambda impl: impl.ring(q, n).label_terms(base, mats, weights, eta, "shake128-rejection", binary))
    assert py == nat and spy.calls["pad_terms"] == 1


# -- Ring-LWE encryption: encrypt_rows, encrypt_tensor, precompute_tensor, encrypt_precomputed

def _mcfe(ring_n=32, eta=1, stat_param=None, sampler="rounded", n=3):
    from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    nids = ["c{}".format(i) for i in range(n)]
    cfg = {"n": n, "lst_nid": nids, "eta": eta, "ring_n": ring_n, "sec_param": 64, "bound_x": 50, "bound_y": 2,
           "gaussian_sampler": sampler}
    kg_cls, cls = mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE
    if stat_param is not None:
        kg_cls, cls = tm.ThresholdMCFERingLWEKeyGenerator, tm.ThresholdMCFERingLWE
        cfg.update(s=3, t=2, lst_sid=["s0", "s1", "s2"], stat_param=stat_param, ct_format="json")
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: cls({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return kg, pp, clients, cls


def _key(cl, impl):
    ring = impl.ring(cl.pp["q"], cl.pp["ring_n"])
    return ring, cl._native_encryption_key()


@native
@pytest.mark.parametrize("eta,stat_param,binary", [(1, None, False), (2, None, True), (1, 150, False)])
def test_encrypt_rows(eta, stat_param, binary, spy):
    kg, pp, clients, _ = _mcfe(eta=eta, stat_param=stat_param)
    cl = clients["c0"]
    rows = cl._padded_rows([[random.Random(i).randint(-50, 50) for _ in range(20)] for i in range(eta)])
    sigmas = (pp["sigma2"], pp["sigma3"])

    def run(impl):
        ring, key = _key(cl, impl)
        assert (key is not None) == (impl is NAT)
        return ring.encrypt_rows(key, cl._reference_key, rows, eta, sigmas, "rounded", binary)

    py, nat = same_bytes(run)
    assert py == nat and spy.calls["encrypt"] == 1


@native
def test_encrypt_rows_facct(spy):
    kg, pp, clients, _ = _mcfe(eta=2, sampler="facct")
    cl = clients["c1"]
    rows = cl._padded_rows([[3, -1, 4], [1, 5, -9]])
    sigmas = (pp["sigma2"], pp["sigma3"])

    def run(impl):
        ring, key = _key(cl, impl)
        return ring.encrypt_rows(key, cl._reference_key, rows, 2, sigmas, "facct", False)

    py, nat = same_bits(run, seed=3)
    assert py == nat and spy.calls["encrypt"] == 1


@native
@pytest.mark.parametrize("stat_param,binary", [(None, False), (None, True), (150, False)])
def test_encrypt_tensor_and_precomputed(stat_param, binary, spy):
    from pyfe4ai.schemes.mcfe.labels import chunk_label
    from pyfe4ai.schemes.mcfe.ring_lwe import label_pad_material

    kg, pp, clients, _ = _mcfe(ring_n=16, stat_param=stat_param)
    cl = clients["c2"]
    ints = np.arange(-20, 30, dtype=np.int64) % 41 - 20  # 50 values: 4 chunks
    mode = cl._encryption_label_hash()
    mats = [label_pad_material(cl.sk["u_prime"], chunk_label("L", c), mode) for c in range(4)]
    sigmas = (pp["sigma2"], pp["sigma3"])

    def tensor(impl):
        ring, key = _key(cl, impl)
        return ring.encrypt_tensor(key, cl._reference_key, ints, mats, mode, sigmas, "rounded", binary)

    py, nat = same_bytes(tensor, seed=2)
    assert py == nat and spy.calls["encrypt_tensor"] == 1
    delta = gp.mpz(pp["q"]) // gp.mpz(pp["p"])

    def offline_online(impl):
        ring, key = _key(cl, impl)
        native, rows = ring.precompute_tensor(key, cl._reference_key, mats, mode, sigmas, "rounded")
        assert (native is not None) == (impl is NAT) and (rows is None) == (impl is NAT)
        out = ring.encrypt_precomputed(native, rows, 1, ints[16:], delta, binary)
        if rows is not None:
            assert rows[1:] == [None] * 3 and rows[0] is not None  # used chunks dropped
        return out

    py, nat = same_bytes(offline_online, seed=4)
    assert py == nat and spy.calls["precompute_tensor"] == 1 and spy.calls["encrypt_precomputed"] == 1


# -- Ring-LWE decryption: decryption_key_hat, decrypt_tensor (against the scheme's Python loop)

@native
@pytest.mark.parametrize("stat_param", [None, 150])
def test_decrypt_tensor_against_the_reference_loop(stat_param, spy):
    from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe

    kg, pp, clients, cls = _mcfe(eta=2, stat_param=stat_param)
    weights = {nid: [1, -2] for nid in clients}
    creds = {"fusion_weight": weights, "label": "L"}
    with native_backend.backend("python"):
        dk = mcfe_rlwe.MCFERingLWEKeyGenerator.get_decryption_keys(kg, "a", credentials=creds)
    rng = random.Random(8)
    cts = {nid: cl.encrypt([[rng.randint(-50, 50) for _ in range(9)] for _ in range(2)], "L")
           for nid, cl in clients.items()}
    agg = cls({"id": "agg", "precision": 0, "keys": {"pp": pp}})
    with native_backend.backend("python"):
        assert not PY.ring(pp["q"], pp["ring_n"]).decrypt_tensor([], [], [], [], [], 1, 1)  # no one-call reference
        ref = mcfe_rlwe.MCFERingLWE._decrypt_with_key_term(agg, cts, dk, dk["z"], weights)
    with native_backend.backend("native"):
        ring = NAT.ring(pp["q"], pp["ring_n"])
        hats = [ring.decryption_key_hat(dk["sk_y"][nid]) for nid in clients]
        out = ring.decrypt_tensor(
            [[(cts[nid]["ct0"], cts[nid]["ct1"]) for nid in clients]], [dk["z"]], [9], hats,
            [weights[nid] for nid in clients], int(gp.mpz(pp["q"]) // gp.mpz(pp["p"])),
            int(pp["n"] * pp["eta"] * int(pp["bound_x"]) * int(pp["bound_y"])))
    values, mask = out
    assert mask is None and np.frombuffer(values, dtype=np.int64).tolist() == ref
    assert spy.calls["ntt_mont"] == 3 and spy.calls["decrypt_tensor"] == 1


# -- threshold MCFE: shamir_share, the partial decryptions and their combination

@native
@pytest.mark.parametrize("bits,t,s", [(61, 2, 3), (126, 3, 5), (230, 2, 4)])
def test_shamir_share_against_share_secret(bits, t, s, spy):
    from pyfe4ai.schemes.threshold import sharing

    n = 16
    q = _q(bits, n)
    rng = random.Random(bits)
    values = [rng.randrange(int(q)) for _ in range(2 * n)]
    points = list(range(1, s + 1))
    with native_backend.backend("python"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(sharing, "_CSPRNG", Uniforms(Bytes(6)))
        shares = [sharing.share_secret(v, t, points, q) for v in values]
    py = [[gp.digits(sh[x]) for sh in shares] for x in points]
    with native_backend.backend("native"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(native_backend, "_random_override", Bytes(6).take)
        ring = NAT.ring(q, n)
        assert ring.threshold_native
        nat = ring.shamir_share(values, t, points, False)
    assert [list(row) for row in nat] == py and spy.calls["shamir_share"] == 1


@native
@pytest.mark.parametrize("ct_format", ["json", "binary"])
def test_threshold_partials_and_combination(ct_format, spy):
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    nids, sids = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
    kg = tm.ThresholdMCFERingLWEKeyGenerator({
        "n": 3, "lst_nid": nids, "s": 3, "t": 2, "lst_sid": sids, "eta": 1, "ring_n": 32, "sec_param": 64,
        "bound_x": 4, "bound_y": 4, "ct_format": ct_format, "key_shares": "full"})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    grads = {nid: np.arange(70) % 9 - 4 for nid in nids}
    enc = {nid: cl.encrypt_lst_ndarray([grads[nid]], label="r") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in nids}, "label": "r",
             "n_chunks": clients["c0"].key_chunks(enc["c0"])}
    parts = {}
    for sid in sids[:2]:
        dk = kg.get_decryption_keys(sid, credentials=creds)
        server = tm.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}})
        py, nat = both(lambda impl: server.compute_lst_ndarray_ct(enc, creds, dk, sids[:2]))
        assert py == nat
        parts[sid] = nat
    kernel = "share_partial" if ct_format == "json" else "share_partial_bin"
    assert spy.calls[kernel] == 2 * 3 and spy.calls["shamir_share"] == 0  # replicated sharing (t < s)
    combiner = tm.ThresholdMCFERingLWE({"id": "comb", "precision": 0, "keys": {"pp": pp}})
    py, nat = both(lambda impl: combiner.decrypt_lst_ndarray_ct(parts))
    assert np.array_equal(py[0], nat[0]) and np.array_equal(nat[0], 3 * grads["c0"])
    assert spy.calls["share_combine" if ct_format == "json" else "share_combine_bin"] == 3


# -- threshold sharings over Z_q: replicated_deal, zq_sum

@native
@pytest.mark.parametrize("binary", [False, True])
def test_replicated_deal_against_share_replicated(binary, spy):
    from pyfe4ai.schemes.threshold import sharing

    q = _q(150, 16)
    rng = random.Random(1)
    material = {"sk_y": {"a": [rng.randrange(int(q)) for _ in range(16)]}, "z": [rng.randrange(int(q)) for _ in range(16)]}
    with native_backend.backend("python"), pytest.MonkeyPatch.context() as mp:
        assert PY.replicated_deal(q, [1, 2], 2, False) is None  # the Python steps of share_replicated
        mp.setattr(sharing, "_CSPRNG", Uniforms(Bytes(9)))
        py = sharing.share_replicated(material, 2, [1, 2, 3], q, binary=binary)
    with native_backend.backend("native"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(native_backend, "_random_override", Bytes(9).take)
        nat = sharing.share_replicated(material, 2, [1, 2, 3], q, binary=binary)
    assert py == nat and spy.calls["zq_replicated_deal"] == 1


@native
@pytest.mark.parametrize("bits", [61, 127, 252])
def test_zq_sum(bits, spy):
    q = _q(bits, 16) if bits != 252 else gp.next_prime(gp.mpz(2) ** 251)
    rng = random.Random(bits)
    rows = [[rng.randrange(-int(q), 2 * int(q)) for _ in range(5)] for _ in range(3)]
    rows[0] = [str(v % int(q)) for v in rows[0]]
    mats = [hashing.hash_to_zq_material("seed{}".format(i), "pieces") for i in range(2)]
    for subtract in (False, True):
        py, nat = both(lambda impl: impl.zq_sum(q, 5, rows, mats, subtract))
        assert py == nat
    words = [b"".join((rng.randrange(int(q))).to_bytes((int(q).bit_length() + 7) // 8, "little") for _ in range(5))
             for _ in range(3)]
    py, nat = both(lambda impl: impl.zq_sum(q, 5, words, [], False, True))
    assert py == nat and spy.calls["zq_sum"] == 3
    assert PY.zq_sum(q, 5, rows, [], reference=False) is None
    with native_backend.backend("native"):  # what the extension does not read: the reference, or None
        floats = [[1.5] * 5]
        assert NAT.zq_sum(q, 5, floats, [], reference=False) is None
        assert NAT.zq_sum(q, 5, floats, []) == PY.zq_sum(q, 5, floats, [])


# -- ristretto255

@native
def test_ristretto255(spy):
    from pyfe4ai.primitives.groups import ec_group as eg

    tag = eg.dst("BACKEND-TEST")

    def run(impl):
        p = impl.r255_mul_base(123456789)
        h = impl.r255_hash(b"msg", tag)
        s = impl.r255_add(impl.r255_mul(h, 2 ** 300 + 5), impl.r255_neg(p))
        d = impl.r255_sub(s, h)
        back = impl.r255_decode(bytes.fromhex(eg.to_hex(d)))
        small = impl.r255_mul_base(-4321)
        return [eg.to_hex(x) for x in (p, h, s, d, back)], impl.r255_dlog(small, 5000), impl.r255_dlog(p, 10)

    py, nat = both(run)
    assert py == nat and nat[1] == -4321 and nat[2] is None
    for name in ("r255_mul_base", "r255_hash", "r255_add", "r255_mul", "r255_neg", "r255_sub", "r255_check",
                 "r255_dlog"):
        assert spy.calls[name] >= 1, name
    for impl in (PY, NAT):
        with pytest.raises(ValueError):
            impl.r255_decode(b"\xff" * 32)


@native
def test_ristretto255_mcfe_layers(spy):
    from pyfe4ai.primitives.groups import ec_group as eg

    dst = eg.dst("MCFE-DDH-H(label)")
    msgs = [("lab/{}".format(i)).encode() for i in range(5)]
    values = [3, -2, 0, 7, -50]
    py, nat = both(lambda impl: impl.r255_mcfe_encrypt(msgs, dst, values, 11, 2 ** 200 + 3))
    assert py == nat and spy.calls["r255_mcfe_encrypt"] == 1
    cts = [nat, PY.r255_mcfe_encrypt(msgs, dst, [1] * 5, 5, 6)]
    py, nat = both(lambda impl: impl.r255_mcfe_decrypt(cts, [2, -1], 2 * 11 - 5, 2 * (2 ** 200 + 3) - 6, msgs, dst, 1000))
    assert py == nat and np.frombuffer(nat[0], dtype=np.int64).tolist() == [2 * v - 1 for v in values]
    bad = [list(cts[0]), cts[1]]
    bad[1] = list(bad[1])
    bad[1][2] = "zz" * 32
    for impl, name in ((PY, "python"), (NAT, "native")):
        with native_backend.backend(name), pytest.raises(ValueError) as err:
            impl.r255_mcfe_decrypt(bad, [2, -1], 1, 1, msgs, dst, 1000)
        assert err.value.args[:2] == ("malformed ciphertext", 1)


# -- dense LWE

@native
@pytest.mark.parametrize("mode", hashing.LABEL_HASHES)
@pytest.mark.parametrize("bits", [40, 100, 300])
def test_label_dots(mode, bits, spy):
    q = gp.next_prime(gp.mpz(2) ** bits)
    rng = random.Random(bits)
    row = [gp.mpz(rng.randrange(-int(q), int(q))) for _ in range(24)]
    mats = [hashing.hash_to_zq_material("lab{}".format(i), "dense", mode) for i in range(7)]
    py, nat = both(lambda impl: impl.label_dots(mats, impl.dense_key(row, q), q, 24, mode))
    assert py == nat and spy.calls["dense_dots"] == 1
    py, nat = both(lambda impl: impl.label_dots(mats, impl.dense_key(row[:-1], q), q, 24, mode))  # a short row
    assert py == nat and spy.calls["dense_dots"] == 1


@native
def test_cached_dense_keys_pickle_without_backend_classes():
    """A client pickles its cached dense keys; the pickle names no backend
    class, so versions without :mod:`pyfe4ai.backend.native_impl` load it."""
    import pickle

    from pyfe4ai.schemes.mcfe import lwe as dlwe

    kg = dlwe.MCFELWEKeyGenerator({"n": 2, "lst_nid": ["c0", "c1"]})
    kg.setup()
    cl = dlwe.MCFELWE({"id": "c0", "precision": 0, "keys": {"pp": kg.get_public_parameters(),
                                                            "sk": kg.get_private_keys("c0")}})
    layer = [np.arange(6) % 3 - 1]
    with native_backend.backend("native"):
        before = cl.encrypt_lst_ndarray(layer, label="L")
    data = pickle.dumps(cl)
    assert "_dense_key_native" in cl.__dict__ and b"pyfe4ai.backend" not in data
    with native_backend.backend("native"):
        after = pickle.loads(data).encrypt_lst_ndarray(layer, label="L")
    assert [a.tolist() for a in after] == [b.tolist() for b in before]


# -- BLS12-381

@native
def test_bls12_381_has_one_implementation():
    kernels = NAT.bls12_381()
    assert PY.bls12_381() is kernels and kernels is not None
    g = kernels.generator(1)
    assert kernels.deserialize(1, g.to_bytes()) == g
    assert hashlib.sha256(kernels.hash(1, b"x", b"dst").to_bytes()).hexdigest()  # a G1 element
