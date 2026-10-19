"""The opt-in RNS modulus of the threshold Ring-LWE MCFE (``ring_modulus="rns"``).

The flooded q is a product of NTT primes below 2^62 (four at the 212 to 218
bits of the standard profile); the native backend runs one 64-bit NTT per
prime under the kernels of a prime ring (an RNS backend of the 256-bit ring:
NTT-domain values are residue tuples). Checked: the modulus, its records and
binding, the server bound ``s < min q_i``, and native results equal to the
Python code on the composite q (NTT with the CRT root), for encryption, key
shares, partial decryptions (JSON and binary) and combination.
"""

from __future__ import annotations

import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes import threshold_utils
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.threshold_utils import pp_binding
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils import sampling_utils
from pyfe4ai.utils.exceptions import FEValidationError

NATIVE = native_backend.available() and hasattr(native_backend.ext(), "ctx_new_rns")
needs_native = pytest.mark.skipif(not NATIVE, reason="native extension without RNS rings")
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _rounded_sampler():
    """The native sampler reproduces the rounded Gaussian on given bytes."""
    with sampling_utils.gaussian_sampler("rounded"):
        yield


def _old_two_factor(lower, ring_n, p):
    """The two-factor modulus of earlier versions."""
    q1 = rlu.prev_ntt_prime(gp.isqrt(lower), ring_n)
    q2 = rlu.next_ntt_prime(-(-lower // q1), ring_n)
    while q2 == q1 or gp.gcd((q1 * q2) // gp.mpz(p), q1 * q2) != 1:
        q2 = rlu.next_ntt_prime(q2 + 1, ring_n)
    return q1 * q2, (q1, q2)


@pytest.mark.parametrize("bits", [61, 80, 120, 133, 212, 215, 218])
@pytest.mark.parametrize("ring_n", [16, 8192])
def test_rns_modulus(bits, ring_n):
    lower = (gp.mpz(1) << (bits - 1)) + 12345
    k = rlu.rns_factor_count(lower)
    assert k == max(2, -(-bits // 60))
    q, fs = rlu.rns_ntt_modulus(lower, ring_n, 602, k)
    assert len(fs) == k == len(set(fs)) and list(fs) == sorted(fs)
    assert q == np.prod([int(f) for f in fs], dtype=object) and q >= lower and q.bit_length() == bits
    assert all(gp.is_prime(f) and f % (2 * ring_n) == 1 and int(f).bit_length() <= rlu.RNS_MAX_FACTOR_BITS for f in fs)
    assert gp.gcd(q // 602, q) == 1
    assert rlu.rns_ntt_modulus(lower, ring_n, 602, k) == (q, fs)  # deterministic: records can be checked
    if bits <= 120:
        assert rlu.rns_ntt_modulus(lower, ring_n, 602) == _old_two_factor(lower, ring_n, 602)
    with pytest.raises(FEValidationError):
        rlu.rns_ntt_modulus(lower, ring_n, 602, 1)


@pytest.mark.parametrize("bits", [241, 244, 247, 248])
def test_rns_modulus_up_to_248_bits_has_four_factors(bits):
    """Four factors below 2^62 up to 248 bits (the native ring holds four),
    rather than the five of 60 bits each; one more where the last would exceed
    62 bits (bounds just below 2^248)."""
    for lower in ((gp.mpz(1) << (bits - 1)) + 12345, (gp.mpz(1) << bits) - (gp.mpz(1) << (bits - 8))):
        assert rlu.rns_factor_count(lower) == 4
        q, fs = rlu.rns_ntt_modulus_for(lower, 16, 602)
        assert q >= lower and q == np.prod([int(f) for f in fs], dtype=object)
        assert all(int(f).bit_length() <= rlu.RNS_MAX_FACTOR_BITS for f in fs)
        assert len(fs) == (4 if q.bit_length() <= 248 else 5)
    assert rlu.rns_factor_count(gp.mpz(1) << 248) == 5  # 249 bits: five
    q, fs = rlu.rns_ntt_modulus_for((gp.mpz(1) << 248) - 1, 16, 602)  # a last factor of 63 bits: five instead
    assert len(fs) == 5 and q >= (gp.mpz(1) << 248) - 1


@needs_native
@pytest.mark.parametrize("bits,k,n", [(133, 3, 16), (133, 3, 1024), (215, 4, 16), (215, 4, 8192), (240, 4, 64),
                                      (247, 4, 64)])
def test_ntt_and_products_identical(bits, k, n):
    q, fs = rlu.rns_ntt_modulus((gp.mpz(1) << (bits - 1)) + 7, n, 602, k)
    rlu.register_rns_modulus(q, fs)
    with native_backend.backend("native"):
        ctx = native_backend.ring_context(q, n)
    assert ctx is not None and not ctx.legacy_rns and ctx.word_bits == 256
    assert native_backend.ext().ctx_info(ctx.cap)["rns"] == [int(f) for f in fs]
    root = rlu._get_ntt_root(q, n)
    rng = random.Random(bits * n)
    a = [gp.mpz(rng.randrange(-2 * int(q), 2 * int(q))) for _ in range(n)]
    b = [gp.mpz(rng.randrange(int(q))) for _ in range(n)]
    pa, pb = ctx.pack(a), ctx.pack(b)
    N = native_backend.ext()
    with native_backend.backend("python"):  # the reference: the Python code on the composite q
        fwd = [int(v) for v in rlu.ntt_forward([v % q for v in a], q, root)]
        inv = [int(v) for v in rlu.ntt_inverse(b, q, root)]
        prod = [int(v) for v in rlu.poly_mul(a, b, q)]
    assert ctx.unpack(N.ntt(ctx.cap, pa)) == fwd
    assert ctx.unpack(N.ntt(ctx.cap, pb, True)) == inv
    assert ctx.unpack(N.poly_mul(ctx.cap, pa, pb)) == prod


def _setup(t=2, ct_format="json", stat_param=150, ring_n=32, ring_modulus="rns", **extra):
    config = {"n": 3, "lst_nid": NIDS, "s": 3, "t": t, "lst_sid": SIDS, "eta": 1, "ring_n": ring_n, "sec_param": 64,
              "bound_x": 100, "bound_y": 3, "stat_param": stat_param, "ct_format": ct_format,
              "ring_modulus": ring_modulus, **extra}
    kg = tm.ThresholdMCFERingLWEKeyGenerator(config)
    kg.setup()
    return kg, kg.get_public_parameters()


def _obj(pp, nid, kg=None):
    keys = {"pp": pp} if kg is None else {"pp": pp, "sk": kg.get_private_keys(nid)}
    return tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": keys})


def test_parameters_records_and_binding():
    import json
    import os

    kg, pp = _setup()
    factors = [int(f) for f in pp["q_factors"]]
    assert len(factors) == 4 and int(pp["q"]) == np.prod(factors, dtype=object)
    folder = os.path.join(kg.config_folder, kg._scheme_type)
    with open(os.path.join(folder, "param-rns.json"), encoding="utf-8") as f:
        record = json.load(f)
    assert record["ring_modulus"] == "rns" and record["q_factors"] == pp["q_factors"] and record["q"] == pp["q"]
    again, pp2 = _setup()  # the cached RNS record verifies: same q, factors and a
    assert (pp2["q"], pp2["q_factors"], pp2["A"]) == (pp["q"], pp["q_factors"], pp["A"])
    prime, ppp = _setup(ring_modulus="prime")  # does not take the RNS record (a file of its own)
    assert prime.q_factors is None and "q_factors" not in ppp and ppp["q"] != pp["q"]
    assert int(ppp["q"]).bit_length() == int(pp["q"]).bit_length()
    with open(os.path.join(folder, "param.json"), encoding="utf-8") as f:
        assert "q_factors" not in json.load(f)
    assert _setup(ring_modulus="prime")[1]["A"] == ppp["A"]  # both records kept: loaded, not regenerated
    third, pp3 = _setup()  # nor the RNS configuration the prime record
    assert pp3["q"] == pp["q"] and pp3["q_factors"] == pp["q_factors"]
    # q_factors is bound to the key shares; the digests of prime parameters are unchanged
    assert tm.binding_fields(pp) == tm._BINDING_FIELDS + ("q_factors",) and tm.binding_fields(ppp) == tm._BINDING_FIELDS
    assert kg._binding == pp_binding(pp, tm._BINDING_FIELDS + ("q_factors",), tm._BINDING_DEFAULTS)
    # s < min q_i is checked by key generators and servers
    tm.check_rns_servers(3, factors)
    with pytest.raises(FEValidationError, match="factor"):
        tm.check_rns_servers(5, [5, 7])
    with pytest.raises(FEValidationError, match="factor"):
        _obj(dict(pp, s=min(factors)), "s0")


@pytest.mark.parametrize("ct_format", ["json", "binary"])
@pytest.mark.parametrize("t", [2, 3])
@pytest.mark.parametrize("stat_param", [64, 150])  # 3 factors (133 bits), 4 factors (219 bits)
def test_rounds_decrypt_and_backends_agree(t, ct_format, stat_param):
    kg, pp = _setup(t=t, ct_format=ct_format, stat_param=stat_param)
    rng = np.random.default_rng(t)
    grads = {nid: [rng.integers(-100, 101, 70), rng.integers(-100, 101, (2, 3))] for nid in NIDS}
    weights = {nid: [int(y)] for nid, y in zip(NIDS, (1, -2, 3))}
    with native_backend.backend("python"):
        enc = {nid: _obj(pp, nid, kg).encrypt_lst_ndarray(grads[nid], label="r") for nid in NIDS}
    creds = {"fusion_weight": weights, "label": "r", "n_chunks": tm.ThresholdMCFERingLWE.key_chunks(enc["c0"])}
    dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
    outs = {}
    for name in ["python"] + (["native"] if NATIVE else []):
        with native_backend.backend(name):
            groups = [{sid: _obj(pp, sid).compute_lst_ndarray_ct(enc, creds, dks[sid], list(en)) for sid in en}
                      for en in (SIDS[:t], SIDS[-t:])]
            result = _obj(pp, "comb").decrypt_lst_ndarray_ct(groups)
        outs[name] = groups
        for got, l in zip(result, range(2)):
            assert np.array_equal(got, sum(weights[nid][0] * grads[nid][l] for nid in NIDS))
    if NATIVE:
        assert outs["python"] == outs["native"]


@needs_native
@pytest.mark.parametrize("stat_param", [64, 150])
def test_encryption_and_key_shares_identical(stat_param, monkeypatch):
    from tests.test_native_backend import ByteStream, StreamRandom, _BytesSystemRandom

    kg, pp = _setup(t=3, stat_param=stat_param, sharing="shamir")
    cl = _obj(pp, "c0", kg)
    layers = [np.random.default_rng(1).integers(-100, 101, 70)]

    def both(fn, patch_py, patch_nat):
        with native_backend.backend("python"):
            patch_py()
            py = fn()
        with native_backend.backend("native"):
            patch_nat()
            nat = fn()
            monkeypatch.setattr(native_backend, "_random_override", None)
        return py, nat

    py, nat = both(lambda: cl.encrypt_lst_ndarray(layers, label="r"),
                   lambda: monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(3))),
                   lambda: monkeypatch.setattr(native_backend, "_random_override", ByteStream(3).take))
    assert py == nat
    py, nat = both(lambda: cl.encrypt([[1, -2, 3]], "L"),
                   lambda: monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(4))),
                   lambda: monkeypatch.setattr(native_backend, "_random_override", ByteStream(4).take))
    assert py == nat
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L", "n_chunks": 2}
    key = mcfe_rlwe.MCFERingLWEKeyGenerator.get_decryption_keys(kg, "s0", credentials=creds)
    with native_backend.backend("python"):
        key_py = mcfe_rlwe.MCFERingLWEKeyGenerator.get_decryption_keys(kg, "s0", credentials=creds)
    assert key == key_py
    py, nat = both(lambda: kg._share_key(key, "w"),
                   lambda: monkeypatch.setattr(threshold_utils, "_CSPRNG", _BytesSystemRandom(ByteStream(6))),
                   lambda: monkeypatch.setattr(native_backend, "_random_override", ByteStream(6).take))
    assert py == nat
