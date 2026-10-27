"""Native NTTs of ring_lwe_utils and the SIFE Ring-LWE schemes against the Python code."""

import os
import random

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.sife import fullysec_ring_lwe as fs_rlwe
from pyfe4ai.schemes.sife import ring_lwe as sife_rlwe
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.primitives import sampling as sampling_utils
from tests.native.test_native_backend import ByteStream, StreamRandom, _rounded_sampler  # noqa: F401 (autouse fixture)

pytestmark = pytest.mark.skipif(not native_backend.available(), reason="Rust extension not built")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    with native_backend.backend("auto"):
        yield


def _both(fn):
    with native_backend.backend("python"):
        py = fn()
    with native_backend.backend("native"):
        nat = fn()
    return py, nat


@pytest.mark.parametrize("bits,n", [(30, 16), (62, 64), (63, 64), (77, 256), (124, 64), (212, 32)])
def test_ntt_and_poly_mul_dispatch_identical(bits, n):
    q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), n)
    root = rlu._get_ntt_root(q, n)
    rng = random.Random(bits)
    a = [gp.mpz(rng.randrange(-2 * int(q), 2 * int(q))) for _ in range(n)]  # any sign, unreduced
    b = [gp.mpz(rng.randrange(int(q))) for _ in range(n)]
    for fn in (lambda: rlu.ntt_forward(a, q, root), lambda: rlu.ntt_inverse(a, q, root),
               lambda: rlu.poly_mul(a, b, q), lambda: rlu.RingNTT.get(q, n).forward(b)):
        py, nat = _both(fn)
        assert py == nat and all(type(v) is type(gp.mpz(0)) for v in nat)
    with native_backend.backend("native"):
        assert native_backend.ops()._ntt_ring(q, n, root) is not None
        # another primitive root: the native context does not compute that NTT
        other = gp.powmod(root, 3, q)
        assert native_backend.ops()._ntt_ring(q, n, other) is None
    py, nat = _both(lambda: rlu.ntt_forward(a, q, other))  # Python both times, with that root
    assert py == nat


def _sife(eta=3, ring_n=64):
    kg = sife_rlwe.SIFERingLWEKeyGenerator({"eta": eta, "ring_n": ring_n, "sec_param": 64, "bound_x": 5, "bound_y": 5})
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, sife_rlwe.SIFERingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}}), \
        sife_rlwe.SIFERingLWE({"keys": {"pp": pp}})


@pytest.mark.parametrize("eta,ring_n,k", [(1, 16, 16), (3, 64, 10), (2, 256, 256)])
def test_sife_encrypt_identical(monkeypatch, eta, ring_n, k):
    _, enc, _ = _sife(eta, ring_n)
    rng = random.Random(k)
    x = [[rng.randint(-5, 5) for _ in range(k)] for _ in range(eta)]
    with native_backend.backend("python"):
        monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(k)))
        py = enc.encrypt(x)
    with native_backend.backend("native"):
        assert enc._native_key() is not None
        monkeypatch.setattr(native_backend, "_random_override", ByteStream(k).take)
        nat = enc.encrypt(x)
        monkeypatch.setattr(native_backend, "_random_override", None)
    assert py == nat


def test_sife_decrypt_identical_and_correct():
    kg, enc, dec = _sife(3, 64)
    rng = random.Random(4)
    y = [rng.randint(-5, 5) for _ in range(3)]
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    x = [[rng.randint(-5, 5) for _ in range(64)] for _ in range(3)]
    ct = enc.encrypt(x)
    py, nat = _both(lambda: dec.decrypt(ct, dk, y))
    assert py == nat == [sum(y[i] * x[i][j] for i in range(3)) for j in range(64)]
    q = int(dec.pp["q"])
    bad = dict(ct, ct1=[str(rng.randrange(q)) for _ in range(64)])
    py, nat = _both(lambda: dec.decrypt(bad, dk, y))
    assert py == nat


def test_fullysec_identical_with_native_ntts(monkeypatch):
    # q of 248 bits (256-bit words); sec_param 64 would give q beyond the native 2^253
    kg = fs_rlwe.SIFEFullySecRingLWEKeyGenerator({"eta": 1, "ring_n": 64, "sec_param": 128, "bound_x": 1, "bound_y": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = fs_rlwe.SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dec = fs_rlwe.SIFEFullySecRingLWE({"keys": {"pp": pp}})
    with native_backend.backend("native"):
        assert native_backend.ops()._ntt_ring(gp.mpz(pp["q"]), pp["ring_n"], rlu._get_ntt_root(gp.mpz(pp["q"]), pp["ring_n"]))

    def run():  # the same bits for both runs (FACCT by default for this scheme)
        monkeypatch.setattr(sampling_utils, "_CSPRNG", random.Random(11))
        monkeypatch.setattr(sampling_utils, "_OS_RANDOM", sampling_utils.XofRandom(b"fullysec-native" * 3))
        e = fs_rlwe.SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
        return e.encrypt([-1])

    py, nat = _both(run)
    assert py == nat
    y = [1]
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    py, nat = _both(lambda: dec.decrypt(py, dk, y))
    assert py == nat == -1
    assert enc  # the encryptor of the setup stays usable
