"""Fixes of the October 2026 crypto review beyond tests/review/: the details
of the BLS12-381 and ristretto255 hardening, the thread count, the FACCT
table check, the ML adapter's label reservation and the rounded sampler."""

from __future__ import annotations

import base64
import os
import subprocess
import sys

import numpy as np
import pytest

from pyfe4ai.utils import bls12_381 as B
from pyfe4ai.utils import ec_group, native_backend
from pyfe4ai.utils import sampling_utils as su
from pyfe4ai.utils.exceptions import FEValidationError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
EC_BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
needs_bls = pytest.mark.skipif(not B.available(), reason="Rust extension not built")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def group():
    return B.PairingGroup("BLS12-381")


# ------------------------------------------------------------- BLS12-381

@needs_bls
def test_zr_power_reduces_the_exponent_mod_r_minus_1(group):
    a = group.random(B.ZR)
    one, zero = group.init(B.ZR, 1), group.init(B.ZR, 0)
    assert a ** (R - 1) == one
    assert a ** R == a
    assert a ** (R + 1) == a * a
    assert a ** (2 * R - 1) == a
    assert a ** -1 == one / a
    assert a ** -(R - 1) == one
    assert a ** group.init(B.ZR, 5) == a * a * a * a * a  # a ZR exponent: its representative
    assert group.init(B.ZR, 2) ** 10 == 1024
    assert zero ** 0 == one and zero ** (R - 1) == zero and zero ** 3 == zero
    with pytest.raises(ZeroDivisionError):
        zero ** -1


@needs_bls
def test_only_canonical_strings_deserialise(group):
    z = group.serialize(group.random(B.ZR))
    tag, body = z.split(b":")
    assert body.endswith(b"=")  # 32 bytes: the last digit carries 2 unused bits
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    last = alphabet.index(body[-2:-1])
    other = body[:-2] + alphabet[last ^ 1:(last ^ 1) + 1] + b"="
    assert base64.b64decode(other) == base64.b64decode(body)
    with pytest.raises(ValueError):
        group.deserialize(tag + b":" + other)
    for bad_tag in (b"4", b"00", b"\xd9\xa1", b""):
        with pytest.raises(ValueError):
            group.deserialize(bad_tag + b":" + body)
    with pytest.raises(ValueError):
        group.deserialize_many([z, b"01:" + body])
    assert group.deserialize(z) == group.deserialize(z.decode("ascii"))


@needs_bls
def test_gt_dlog_bounds_and_one_shot_tables(group):
    from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt

    base = B.pair(group.random(B.G1), group.random(B.G2))
    for x in (0, 1, -1, 4321, -50000, 50000):
        target = base ** group.init(B.ZR, x)
        assert B.gt_dlog(base, target, 50000, cache=False) == x
        assert B.gt_dlog(base, target, 50000) == x
    assert B.gt_dlog(base, base ** 50001, 50000, cache=False) is None
    for bound in (B.MAX_DLOG_BOUND + 1, 2 ** 64, -1):
        with pytest.raises(ValueError):
            B.gt_dlog(base, base, bound)
    with pytest.raises(ValueError):
        bounded_discrete_log_gt(group, base, base, 2 ** 63)
    with pytest.raises(ValueError):
        B._require().bls_gt_dlog(base, base, 2 ** 48 + 1)  # the extension's own cap


@needs_bls
def test_dlog_table_caches_are_bounded(group):
    """Many bases (GT) and bounds (ristretto255): at most 16 tables are kept."""
    ext = B._require()
    bases = [B.pair(group.random(B.G1), group.random(B.G2)) for _ in range(20)]
    for b in bases:
        assert B.gt_dlog(b, b ** 7, 2000) == 7
    tables, size = ext.bls_gt_dlog_cache()
    assert tables == 16 and size <= 128 << 20
    assert B.gt_dlog(bases[0], bases[0] ** 3, 2000, cache=False) == 3
    assert ext.bls_gt_dlog_cache() == (tables, size)  # one-shot tables are not kept
    with native_backend.backend("native"):
        for bound in range(1000, 1000 + 20 * 400, 400):  # 20 table sizes
            assert ec_group.dlog(ec_group.mul_base(-5), bound) == -5
    assert ext.r255_dlog_cache()[0] == 16


# ------------------------------------------------------------ ristretto255

@pytest.mark.parametrize("backend", EC_BACKENDS)
def test_ristretto_hex_rule_and_dlog_cap(backend):
    with native_backend.backend(backend):
        h = ec_group.to_hex(ec_group.mul_base(1234))
        assert ec_group.dlog(ec_group.from_hex(h), 2000) == 1234
        assert ec_group.dlog(ec_group.from_hex(h.upper()), 2000) == 1234
        for bad in (h[:-1], h + "0", " " + h[1:], "+" + h[1:], h[:10] + " " + h[11:], h[:-2] + "0x"):
            with pytest.raises(FEValidationError):
                ec_group.from_hex(bad)
        with pytest.raises(ValueError):
            ec_group.dlog(ec_group.from_hex(h), ec_group.MAX_DLOG_BOUND + 1)


def test_python_ristretto_table_is_capped(monkeypatch):
    monkeypatch.setattr(ec_group, "_MAX_BABY", 50)
    ec_group._dlog_table.cache_clear()
    try:
        with native_backend.backend("python"):
            for x in (-12345, 0, 19999):
                assert ec_group.dlog(ec_group.mul_base(x), 20000) == x
        assert ec_group._dlog_table(20000)[1] == 50
    finally:
        ec_group._dlog_table.cache_clear()


# ----------------------------------------------------------------- threads

def test_set_threads_is_capped():
    with pytest.raises(ValueError):
        native_backend.set_threads(native_backend.MAX_THREADS + 1)
    with pytest.raises(ValueError):
        native_backend.set_threads(0)


@pytest.mark.parametrize("value,expect", [("abc", "ValueError"), ("0", "1"), ("10000000", "MAX"), (" 3 ", "3")])
def test_thread_environment_variable(value, expect):
    env = dict(os.environ, PYFE4AI_THREADS=value, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    code = "from pyfe4ai.utils import native_backend as nb; print(nb.get_threads(), nb.MAX_THREADS)"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    if expect == "ValueError":
        assert res.returncode != 0 and "PYFE4AI_THREADS must be an integer" in res.stderr
        return
    threads, cap = res.stdout.split()
    assert threads == (cap if expect == "MAX" else expect)


# ------------------------------------------------------------------- FACCT

@pytest.mark.skipif(not native_backend.available() or not hasattr(native_backend.ext(), "FacctTable"),
                    reason="native FACCT not built")
@pytest.mark.parametrize("sigma", [3.2, 17.3, 437881.7381962938])
def test_facct_table_refuses_an_imprecise_constant(sigma):
    import gmpy2 as gp

    p = su._facct_params(su._exact(sigma))

    def exact(x, y):
        with gp.context(precision=su.FACCT_PRECISION + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(p.neg_c * (y * (y + 2 * p.k * x))) * (1 << su.FACCT_PRECISION)))

    with gp.context(precision=53):  # c rounded to 53 bits: off by about 2^-55
        mant, exp = (-p.neg_c).as_mantissa_exp()
    mant, exp = int(mant), int(exp)
    shift = mant.bit_length() - 128
    with pytest.raises(ValueError, match="disagrees"):
        native_backend.ext().FacctTable(p.k, p.y_bits, list(p.cdt), mant << -shift, exp + shift, exact)
    native_backend._facct_table.cache_clear()
    native_backend.facct_table(sigma)  # the shipped constant passes the check


# -------------------------------------------------------------- ML adapter

def test_failed_encryption_releases_the_label():
    from pyfe4ai.utils.ml_adapter import FESchemeWrapper, encrypt_gradient

    w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": ["c0", "c1", "c2"], "ring_n": 64,
                                             "sec_param": 64, "bound_x": 100, "bound_y": 2}, precision=2)
    with pytest.raises(ValueError):
        encrypt_gradient(np.full(10, 5.0), w, nid="c0", label="r1")  # beyond bound_x: refused
    encrypt_gradient(np.full(10, 0.5), w, nid="c0", label="r1")  # the label was not consumed
    with pytest.raises(ValueError, match="already encrypted"):
        encrypt_gradient(np.full(10, 0.5), w, nid="c0", label="r1")


@pytest.mark.parametrize("clone", ["pickle", "deepcopy", "copy"])
def test_a_copied_wrapper_keeps_the_used_labels(clone):
    import copy
    import pickle

    from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = ["c0", "c1", "c2"]
    w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": nids, "ring_n": 64,
                                             "sec_param": 64, "bound_x": 100, "bound_y": 2}, precision=2)
    cts = {n: encrypt_gradient(np.full(10, 0.5), w, nid=n, label="r1") for n in nids[:2]}
    twin = {"pickle": lambda x: pickle.loads(pickle.dumps(x)), "deepcopy": copy.deepcopy, "copy": copy.copy}[clone](w)
    with pytest.raises(ValueError, match="already encrypted"):
        encrypt_gradient(np.full(10, 0.5), twin, nid="c0", label="r1")
    cts["c2"] = encrypt_gradient(np.full(10, 0.5), w, nid="c2", label="r1")  # the original is unaffected
    assert ("c2", "r1") not in twin._encrypted_labels and twin._label_lock is not w._label_lock
    assert np.allclose(aggregate_gradients(cts, w, {n: 1 for n in nids}, label="r1"), 1.5)
    # after a whole round (native key caches in the key generator) the wrapper still copies
    again = pickle.loads(pickle.dumps(w)) if clone == "pickle" else copy.deepcopy(w)
    cts2 = {n: encrypt_gradient(np.full(10, 0.25), again, nid=n, label="r2") for n in nids}
    assert np.allclose(aggregate_gradients(cts2, again, {n: 1 for n in nids}, label="r2"), 0.75)


@pytest.mark.parametrize("backend", EC_BACKENDS)
def test_ring_lwe_objects_pickle_after_use(backend):
    import pickle

    from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
    from pyfe4ai.schemes.sife import ring_lwe as sife_rlwe

    nids = ["a", "b"]
    with native_backend.backend(backend):
        kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 2, "lst_nid": nids, "eta": 1, "ring_n": 16, "sec_param": 64,
                                                "bound_x": 10, "bound_y": 1})
        kg.setup()
        pp = kg.get_public_parameters()
        enc = {n: mcfe_rlwe.MCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
               for n in nids}
        cts = {n: enc[n].encrypt_lst_ndarray([np.arange(5)], label="L") for n in nids}
        weights = {n: [1] for n in nids}
        kg.get_decryption_keys("s", credentials={"fusion_weight": weights, "label": "L", "n_chunks": 1})
        kg2, enc2 = pickle.loads(pickle.dumps(kg)), pickle.loads(pickle.dumps(enc))
        cts2 = {n: enc2[n].encrypt_lst_ndarray([np.arange(5)], label="M") for n in nids}
        dk = kg2.get_decryption_keys("s", credentials={"fusion_weight": weights, "label": "M", "n_chunks": 1})
        agg = mcfe_rlwe.MCFERingLWE({"id": "s", "precision": 0, "keys": {"pp": pp}})
        out = agg.compute_lst_ndarray_ct(cts2, dk=dk, fusion_weight=weights, label="M")
        assert np.ravel(out[0]).tolist() == [0, 2, 4, 6, 8] and cts
        skg = sife_rlwe.SIFERingLWEKeyGenerator({"eta": 1, "ring_n": 16, "sec_param": 64, "bound_x": 10,
                                                 "bound_y": 1})
        skg.setup()
        s = sife_rlwe.SIFERingLWE({"keys": {"pp": skg.get_public_parameters(), "sk": skg.get_private_keys()},
                                   "precision": 0})
        s.encrypt([[1] * 16])
        pickle.loads(pickle.dumps(s)).encrypt([[1] * 16])


# --------------------------------------------------------- rounded sampler

def test_rounded_sampler_keeps_no_cached_sample():
    for count in (1, 3, 8):
        su.discrete_gaussian_matrix(1, count, 3.2, sampler="rounded")
        assert su._CSPRNG.gauss_next is None


def test_rounded_sampler_is_gauss_on_the_same_uniforms(monkeypatch):
    import random

    monkeypatch.setattr(su, "_CSPRNG", random.Random(42))
    got = [int(v) for row in su.discrete_gaussian_matrix(2, 3, 1000.0, sampler="rounded") for v in row]
    ref = random.Random(42)
    assert got == [int(round(ref.gauss(0.0, 1000.0))) for _ in range(6)]
