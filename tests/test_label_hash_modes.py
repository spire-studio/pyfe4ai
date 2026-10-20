"""The ``label_hash`` modes of the MCFE label hashes (``hashing.LABEL_HASHES``).

* The rejection modes follow their specification (re-implemented here from
  ``hashlib``), are exactly uniform mod q, are separated from each other and
  from the default, and keep known answers.
* The native kernels (``hash_to_zq``, the pads of encryption and key
  derivation, ``dense_dots``) give the Python values bit for bit, for 64-,
  128- and 256-bit words and with rejections actually happening.
* MCFE Ring-LWE (plain, threshold, decentralised) and dense-LWE MCFE decrypt
  in every mode; the default is ``"shake128-rejection"``, public parameters
  and keys without ``label_hash`` (versions <= 0.3.0) are read as
  ``"shake256-wide"``; keys of another mode are refused.
"""

from __future__ import annotations

import hashlib
import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import lwe as mcfe_lwe
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_decentralized as dmcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe_rlwe
from pyfe4ai.primitives import hashing as cu
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError

pytestmark = pytest.mark.core

REJECTION = ("shake256-rejection", "shake128-rejection")
needs_native = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


@pytest.fixture
def native_calls(monkeypatch):
    """Counts the extension's calls by name (to see that a native path ran)."""
    import collections

    real, counts = native_backend._ext, collections.Counter()

    class Spy:
        def __getattr__(self, name):
            fn = getattr(real, name)

            def call(*args, **kwargs):
                counts[(name, kwargs.get("hash_mode", 0))] += 1
                return fn(*args, **kwargs)

            return call if callable(fn) else fn

    monkeypatch.setattr(native_backend, "_ext", Spy())
    return counts


def _spec(v: str, q: int, length: int, domain: str, mode: str) -> list:
    """The rejection modes as specified: XOF over "domain|zq-rej/v1|len|v",
    candidates of ceil(bits(q) / 8) + 1 big-endian bytes, accepted below the
    largest multiple of q that fits, reduced mod q."""
    xof = hashlib.shake_128 if mode == "shake128-rejection" else hashlib.shake_256
    material = "{}|zq-rej/v1|{}|{}".format(domain, len(v), v).encode("utf-8")
    nb = (q.bit_length() + 7) // 8 + 1
    limit = (256 ** nb // q) * q
    stream = xof(material).digest(nb * (2 * length + 64))
    out = []
    for i in range(0, len(stream), nb):
        c = int.from_bytes(stream[i:i + nb], "big")
        if c < limit:
            out.append(c % q)
            if len(out) == length:
                return out
    raise AssertionError("stream too short")


# q with many rejections: 2^(8 nb) mod q close to q
Q_REJECT = 2 ** 15 + 3 * 2 ** 13 + 1  # bits 16, nb 3: 2^24 mod q is large, frequent rejections


@pytest.mark.parametrize("mode", REJECTION)
@pytest.mark.parametrize("q", [Q_REJECT, 2 ** 61 - 1, (1 << 79) + 12345, int(gp.next_prime(1 << 214))])
def test_rejection_modes_follow_the_specification(mode, q):
    for v, dom in (("x", ""), ("u'|label", "pyfe4ai/mcfe/ring_lwe/H(u',label)"), ("é" * 50, "d")):
        got = cu.hash_to_zq_vector(v, q, 2000, domain=dom, mode=mode)
        assert [int(x) for x in got] == _spec(v, q, 2000, dom, mode)


def test_rejection_happens_and_parameters():
    nb, limit = cu.zq_rejection_parameters(Q_REJECT)
    assert nb == 3 and limit % Q_REJECT == 0 and 256 ** 3 - Q_REJECT < limit < 256 ** 3
    stream = hashlib.shake_256(cu.hash_to_zq_material("x", "", "shake256-rejection")).digest(3 * 20000)
    rejected = sum(int.from_bytes(stream[i:i + 3], "big") >= limit for i in range(0, len(stream), 3))
    assert rejected > 5  # the tests above cross rejected candidates
    for bits in (10, 16, 62, 80, 187, 215, 253):
        q = int(gp.next_prime(1 << (bits - 1)))
        nb, limit = cu.zq_rejection_parameters(q)
        assert limit / 256 ** nb > 1 - 2 ** -8


def test_modes_are_separated_and_generic_default_unchanged():
    q = (1 << 79) + 12345
    outs = {m: cu.hash_to_zq_vector("lab", q, 64, domain="d", mode=m) for m in cu.LABEL_HASHES}
    assert len({tuple(v) for v in outs.values()}) == 3
    assert len({cu.hash_to_zq_material("lab", "d", m) for m in cu.LABEL_HASHES}) == 2  # + another XOF
    assert cu.hash_to_zq_material("lab", "d") == b"d|zq-vector|3|lab"
    nb = (q.bit_length() + 128 + 7) // 8
    stream = hashlib.shake_256(b"d|zq-vector|3|lab").digest(64 * nb)
    assert outs["shake256-wide"] == [int.from_bytes(stream[i:i + nb], "big") % q for i in range(0, 64 * nb, nb)]
    with pytest.raises(ValueError):
        cu.hash_to_zq_vector("lab", q, 4, mode="sha3")


# sha256 over the decimal values of 257 values, for v = "kat", domain "pyfe4ai/kat",
# q = next_prime(2^(bits - 1))


def _kat_digest(mode: str, bits: int) -> str:
    q = int(gp.next_prime(1 << (bits - 1)))
    vals = cu.hash_to_zq_vector("kat", q, 257, domain="pyfe4ai/kat", mode=mode)
    return hashlib.sha256(",".join(str(int(v)) for v in vals).encode()).hexdigest()


KAT_VALUES = {
    ("shake256-rejection", 80): "cd4cd0d42c2adf25020a9253b8ae2acd23128994a5c69d5b7b87b119ae1ff214",
    ("shake128-rejection", 80): "203becc62f36e859d19779bca8d2e5c9245d832b4d25fb55538a578f690adad4",
    ("shake256-rejection", 215): "858799896a521ffdbdafd4d3c564630b7e20c40d632419490c86b8eff5cadbe8",
    ("shake128-rejection", 215): "ff75fd955122dace4c3487919d851846012f94835140813c7ea97fe3b1214499",
}


@pytest.mark.parametrize("mode,bits", sorted(KAT_VALUES))
def test_known_answers(mode, bits):
    assert _kat_digest(mode, bits) == KAT_VALUES[(mode, bits)]


@pytest.mark.parametrize("mode", REJECTION)
def test_uniform_mod_q(mode):
    """Chi-square over the residues of a small q (exact uniformity is the
    point of the rejection modes); the threshold has p about 1e-6."""
    q = 257
    vals = cu.hash_to_zq_vector("uniformity", q, 257 * 400, domain="t", mode=mode)
    counts = np.bincount(np.array([int(v) for v in vals]), minlength=q)
    expected = len(vals) / q
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 256 + 5 * (2 * 256) ** 0.5 + 40  # mean 256, sd 22.6


# ------------------------------------------------------------ native kernels

def _ctx(bits: int, n: int):
    q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), n)
    return int(q), native_impl.RingContext(q, n, rlu._get_ntt_root(q, n))


@needs_native
@pytest.mark.parametrize("mode", REJECTION)
@pytest.mark.parametrize("bits", [17, 40, 62, 64, 80, 124, 126, 187, 215, 253])
def test_native_hash_to_zq_identical(mode, bits):
    q, ctx = _ctx(bits, 64)
    N = native_backend.ext()
    code = cu.LABEL_HASH_CODES[mode]
    for v, dom in (("x", ""), ("u'|label", "pyfe4ai/mcfe/ring_lwe/H(u',label)")):
        got = ctx.unpack(N.hash_to_zq(ctx.cap, cu.hash_to_zq_material(v, dom, mode), 3000, hash_mode=code))
        assert got == [int(x) for x in cu.hash_to_zq_vector(v, q, 3000, domain=dom, mode=mode)]
    with pytest.raises(ValueError):
        N.hash_to_zq(ctx.cap, b"x", 4, hash_mode=7)


@needs_native
def test_native_rejects_like_python():
    """A prime q = 1 mod 128 just above Q_REJECT makes rejections frequent
    (nb = 3); the native kernel must skip exactly the same candidates."""
    q = int(rlu.next_ntt_prime(gp.mpz(Q_REJECT), 64))
    nb, limit = cu.zq_rejection_parameters(q)
    assert 1 - limit / 256 ** nb > 5e-4  # about 18 rejections in 20000 candidates
    ctx = native_impl.RingContext(q, 64, rlu._get_ntt_root(q, 64))
    for mode in REJECTION:
        mat = cu.hash_to_zq_material("many", "d", mode)
        got = ctx.unpack(native_backend.ext().hash_to_zq(ctx.cap, mat, 20000, hash_mode=cu.LABEL_HASH_CODES[mode]))
        assert got == [int(x) for x in cu.hash_to_zq_vector("many", q, 20000, domain="d", mode=mode)]


@needs_native
@pytest.mark.parametrize("mode", REJECTION)
@pytest.mark.parametrize("bits", [40, 120, 187, 250])
def test_native_dense_dots_identical(mode, bits):
    q = int(gp.next_prime(1 << (bits - 1)))
    count = 300
    rng = random.Random(bits)
    key = [rng.randrange(q) for _ in range(count)]
    kl = (q.bit_length() + 63) // 64
    labels = ["l{}".format(i) for i in range(5)]
    mats = [cu.hash_to_zq_material(lab, "dom", mode) for lab in labels]
    nb, limit = cu.zq_rejection_parameters(q)
    raw = native_backend.ext().dense_dots(
        mats, nb, count, b"".join(v.to_bytes(8 * kl, "little") for v in key), kl, 2,
        limit=limit.to_bytes(nb, "big"), rate=168 if mode == "shake128-rejection" else 136,
    )
    step = len(raw) // len(labels)
    got = [int.from_bytes(raw[i:i + step], "little") % q for i in range(0, len(raw), step)]
    want = [sum(k * int(h) for k, h in zip(key, cu.hash_to_zq_vector(lab, q, count, "dom", mode))) % q
            for lab in labels]
    assert got == want


# ------------------------------------------------------------ scheme paths

def _rlwe(mode, kg_cls, cls, ring_n=64, **extra):
    nids = ["c0", "c1", "c2"]
    cfg = {"n": 3, "lst_nid": nids, "eta": 1, "ring_n": ring_n, "sec_param": 64, "bound_x": 100, "bound_y": 2,
           **extra}
    if mode is not None:
        cfg["label_hash"] = mode
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: cls({"id": nid, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return kg, pp, clients


@pytest.mark.parametrize("backend", ["python", pytest.param("native", marks=needs_native)])
@pytest.mark.parametrize("mode", cu.LABEL_HASHES)
def test_mcfe_ring_lwe_round_trip(mode, backend):
    with native_backend.backend(backend):
        kg, pp, clients = _rlwe(mode, mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE)
        assert pp["label_hash"] == mode
        rng = np.random.default_rng(3)
        grads = {nid: rng.uniform(-1, 1, 150).round(2) for nid in clients}
        cts = {nid: cl.encrypt_lst_ndarray([g], label="r") for nid, (cl, g) in
               ((nid, (clients[nid], grads[nid])) for nid in clients)}
        weights = {nid: [1] for nid in clients}
        n_chunks = mcfe_rlwe.MCFERingLWE.key_chunks(cts["c0"])
        dk = kg.get_decryption_keys("s", credentials={"fusion_weight": weights, "label": "r", "n_chunks": n_chunks})
        assert dk["label_hash"] == mode
        agg = mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": pp}})
        out = agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=weights, label="r")[0]
        assert np.allclose(out, sum(grads.values()), atol=1e-9)
        # the same key under public parameters of another mode is refused
        other = dict(pp, label_hash="shake128-rejection" if mode != "shake128-rejection" else "shake256-rejection")
        agg2 = mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": other}})
        with pytest.raises(FESchemeError):
            agg2.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=weights, label="r")


@needs_native
@pytest.mark.parametrize("mode", REJECTION)
@pytest.mark.parametrize("ring_n,stat_param", [(16, None), (64, None), (32, 150)])
def test_native_pads_and_keys_identical(mode, ring_n, stat_param, monkeypatch, native_calls):
    """Encryption (same Gaussian bytes for both backends) and key terms: the
    pads of the native kernels are those of the Python code. The rounded
    sampler (selectable under toy), whose 7-byte uniforms both backends read
    from the same bytes here."""
    if stat_param is None:
        kg, pp, clients = _rlwe(mode, mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE, ring_n,
                                gaussian_sampler="rounded")
    else:
        kg, pp, clients = _rlwe(mode, tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator, tmcfe_rlwe.ThresholdMCFERingLWE,
                                ring_n, s=3, t=2, lst_sid=["s0", "s1", "s2"], stat_param=stat_param,
                                gaussian_sampler="rounded")
    creds = {"fusion_weight": {"c0": [1], "c1": [-2], "c2": [2]}, "label": "L", "n_chunks": 3}
    keys = []
    for name in ("python", "native"):
        with native_backend.backend(name):
            keys.append(mcfe_rlwe.MCFERingLWEKeyGenerator.get_decryption_keys(kg, "s", credentials=creds))
    assert keys[0] == keys[1]
    cl = clients["c0"]
    layers = [np.random.default_rng(1).uniform(-1, 1, 3 * ring_n - 5).round(2)]
    data = random.Random(4).randbytes(1 << 20)
    outs = []
    for name in ("python", "native"):
        with native_backend.backend(name):
            pos = [0]

            def take(n):
                pos[0] += n
                return data[pos[0] - n:pos[0]]

            class Src(random.Random):
                def random(self):
                    return (int.from_bytes(take(7), "big") >> 3) * 2 ** -53

            from pyfe4ai.primitives import sampling as sampling_utils
            if name == "python":
                monkeypatch.setattr(sampling_utils, "_CSPRNG", Src(0))
            else:
                monkeypatch.setattr(native_backend, "_random_override", take)
            outs.append(cl.encrypt_lst_ndarray(layers, label="L"))
            monkeypatch.setattr(native_backend, "_random_override", None)
    assert outs[0] == outs[1]
    code = cu.LABEL_HASH_CODES[mode]
    assert native_calls[("pad_terms", code)] == 1 and native_calls[("enc_key", code)] == 1
    assert native_calls[("encrypt_tensor", 0)] == 1  # its pads come from the key's mode


@pytest.mark.parametrize("mode", REJECTION)
def test_threshold_and_decentralised_round_trip(mode):
    kg, pp, clients = _rlwe(mode, tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator, tmcfe_rlwe.ThresholdMCFERingLWE,
                            32, s=3, t=2, lst_sid=["s0", "s1", "s2"], stat_param=40)
    assert pp["label_hash"] == mode
    rng = np.random.default_rng(5)
    grads = {nid: rng.uniform(-1, 1, 40).round(2) for nid in clients}
    cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in clients}
    weights = {nid: [1] for nid in clients}
    creds = {"fusion_weight": weights, "label": "r", "n_chunks": tmcfe_rlwe.ThresholdMCFERingLWE.key_chunks(cts["c0"])}
    enrolled = ["s0", "s2"]
    parts = {}
    for sid in enrolled:
        srv = tmcfe_rlwe.ThresholdMCFERingLWE({"id": sid, "precision": 2, "keys": {"pp": pp}})
        srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials=creds))  # split key shares, the default
        parts[sid] = srv.compute_lst_ndarray_ct(cts, creds, kg.get_decryption_keys(sid, credentials=creds), enrolled)
    comb = tmcfe_rlwe.ThresholdMCFERingLWE({"id": "c", "precision": 2, "keys": {"pp": pp}})
    assert np.allclose(comb.decrypt_lst_ndarray_ct(parts)[0], sum(grads.values()))

    kg, pp, clients = _rlwe(mode, dmcfe_rlwe.DecentralizedMCFERingLWEKeyGenerator,
                            dmcfe_rlwe.DecentralizedMCFERingLWE, 32)
    weights = {nid: [1] for nid in clients}
    cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in clients}
    shares = {nid: cl.derive_function_decryption_key_share(weights, "r", n_chunks=2) for nid, cl in clients.items()}
    agg = dmcfe_rlwe.DecentralizedMCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": pp}})
    dk = agg.combine_function_decryption_key_share(shares)
    assert np.allclose(agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=weights, label="r")[0],
                       sum(grads.values()))
    agg_default = dmcfe_rlwe.DecentralizedMCFERingLWE(
        {"id": "a", "precision": 2, "keys": {"pp": {k: v for k, v in pp.items() if k != "label_hash"}}})
    with pytest.raises(FESchemeError):
        agg_default.combine_function_decryption_key_share(shares)


@pytest.mark.parametrize("backend", ["python", pytest.param("native", marks=needs_native)])
@pytest.mark.parametrize("mode", cu.LABEL_HASHES)
def test_dense_lwe_round_trip_and_backends(mode, backend, native_calls):
    nids = ["c0", "c1"]
    cfg = {"n": 2, "lst_nid": nids, "eta": 1, "bound_x": 50, "bound_y": 2, "label_hash": mode}
    kg = mcfe_lwe.MCFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["label_hash"] == mode
    clients = {nid: mcfe_lwe.MCFELWE({"id": nid, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    rng = np.random.default_rng(2)
    grads = {nid: rng.uniform(-0.5, 0.5, 7).round(2) for nid in nids}
    with native_backend.backend(backend):
        cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in nids}
        weights = {nid: [1] for nid in nids}
        dk = kg.get_decryption_keys("s", credentials={"fusion_weight": weights})
        agg = mcfe_lwe.MCFELWE({"id": "a", "precision": 2, "keys": {"pp": pp}})
        out = agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=weights, label="r")[0]
    assert np.allclose(np.asarray(out, dtype=float), sum(grads.values()))
    # every backend computes the same ciphertexts: c depends only on key, label and x
    with native_backend.backend("python"):
        ref = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in nids}
    assert all(np.array_equal(cts[nid][0], ref[nid][0]) for nid in nids)
    assert (native_calls[("dense_dots", 0)] > 0) == (backend == "native")


def test_unknown_mode_refused():
    nids = ["c0", "c1"]
    with pytest.raises(FEValidationError):
        mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 2, "lst_nid": nids, "ring_n": 16, "label_hash": "md5"})
    with pytest.raises(FEValidationError):
        mcfe_lwe.MCFELWEKeyGenerator({"n": 2, "lst_nid": nids, "label_hash": "md5"})


def test_defaults_and_v030_compatibility():
    """The key generators default to shake128-rejection; public parameters and
    keys without ``label_hash`` (as v0.3.0 wrote them) are read as
    shake256-wide, so their ciphertexts still decrypt, and they do not mix with
    keys of the new default."""
    assert cu.DEFAULT_LABEL_HASH == "shake128-rejection" and cu.LEGACY_LABEL_HASH == "shake256-wide"
    kg, pp, clients = _rlwe(None, mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE)
    assert pp["label_hash"] == "shake128-rejection"
    weights = {nid: [1] for nid in clients}
    creds = {"fusion_weight": weights, "label": "r", "n_chunks": 1}
    new_dk = kg.get_decryption_keys("s", credentials=creds)

    old_kg, old_pp, _ = _rlwe("shake256-wide", mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE)
    v030_pp = {k: v for k, v in old_pp.items() if k != "label_hash"}
    v030 = {nid: mcfe_rlwe.MCFERingLWE({"id": nid, "precision": 2, "keys": {
                "pp": v030_pp, "sk": {k: v for k, v in old_kg.get_private_keys(nid).items() if k != "label_hash"}}})
            for nid in clients}
    assert v030["c0"].label_hash == "shake256-wide"
    grads = {nid: np.random.default_rng(1).uniform(-1, 1, 30).round(2) for nid in clients}
    cts = {nid: v030[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in clients}
    v030_dk = {k: v for k, v in old_kg.get_decryption_keys("s", credentials=creds).items() if k != "label_hash"}
    agg = mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": v030_pp}})
    assert np.allclose(agg.decrypt_lst_ndarray_ct(cts, dk=v030_dk, fusion_weight=weights, label="r")[0],
                       sum(grads.values()))
    with pytest.raises(FESchemeError):  # a key of the new default under v0.3.0 parameters
        agg.decrypt_lst_ndarray_ct(cts, dk=new_dk, fusion_weight=weights, label="r")
    agg_new = mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": pp}})
    with pytest.raises(FESchemeError):  # a v0.3.0 key under the new parameters
        agg_new.decrypt_lst_ndarray_ct(cts, dk=v030_dk, fusion_weight=weights, label="r")

    lkg = mcfe_lwe.MCFELWEKeyGenerator({"n": 2, "lst_nid": ["c0", "c1"], "bound_x": 50, "bound_y": 2})
    assert lkg.get_public_parameters()["label_hash"] == "shake128-rejection"
    lkg.setup()
    lpp = {k: v for k, v in lkg.get_public_parameters().items() if k != "label_hash"}
    assert mcfe_lwe.MCFELWE({"id": "a", "keys": {"pp": lpp}}).label_hash == "shake256-wide"


# ------------------------------------------------------ binding of the mode

def _copy_layers(cts: dict) -> dict:
    return {nid: [dict(layer, chunks=[dict(c) for c in layer["chunks"]]) for layer in v] for nid, v in cts.items()}


def test_ring_mode_bound_to_keys_and_ciphertexts():
    """Encryption keys carry the mode (a client with public parameters of
    another mode refuses to encrypt); every ndarray layer carries its code
    once (its chunks none), single ciphertexts their own; decryption refuses
    layers or ciphertexts of another mode -- e.g. those of a client that
    still runs v0.3.0 -- instead of returning wrong sums."""
    kg, pp, clients = _rlwe(None, mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE)
    sk = kg.get_private_keys("c0")
    assert sk["label_hash"] == pp["label_hash"] == "shake128-rejection"
    for other in ({k: v for k, v in pp.items() if k != "label_hash"}, dict(pp, label_hash="shake256-rejection")):
        stale = mcfe_rlwe.MCFERingLWE({"id": "c0", "precision": 2, "keys": {"pp": other, "sk": sk}})
        for call in (lambda: stale.encrypt_lst_ndarray([np.zeros(5)], label="r"),
                     lambda: stale.encrypt([[1, 2]], "r"),
                     lambda: stale.precompute("r", [(5,)])):
            with pytest.raises(FEKeyError):
                call()
    grads = {nid: np.random.default_rng(2).uniform(-1, 1, 150).round(2) for nid in clients}
    weights = {nid: [1] for nid in clients}
    for backend in ["python"] + (["native"] if native_backend.available() else []):
        with native_backend.backend(backend):
            cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in clients}
            pre = clients["c1"].precompute("r", [grads["c1"].shape])
            cts["c1"] = clients["c1"].encrypt_lst_ndarray([grads["c1"]], label="r", precomputed=pre)
            for nid in clients:
                assert all(layer["label_hash"] == 2 and all("label_hash" not in c for c in layer["chunks"])
                           for layer in cts[nid])
            dk = kg.get_decryption_keys("s", credentials={"fusion_weight": weights, "label": "r",
                                                          "n_chunks": mcfe_rlwe.MCFERingLWE.key_chunks(cts["c0"])})
            agg = mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": pp}})
            assert np.allclose(agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=weights, label="r")[0],
                               sum(grads.values()))
            for tamper in (lambda v: v[0].pop("label_hash"), lambda v: v[0].update(label_hash=1),
                           lambda v: v[0].update(label_hash="2"),
                           lambda v: v[0]["chunks"][1].update(label_hash=1)):  # a chunk of its own other mode
                bad = _copy_layers(cts)
                tamper(bad["c1"])
                with pytest.raises(FESchemeError, match="c1"):
                    agg.decrypt_lst_ndarray_ct(bad, dk=dk, fusion_weight=weights, label="r")
    single = {nid: clients[nid].encrypt([[3, 4]], "s1") for nid in clients}
    assert all(ct["label_hash"] == 2 for ct in single.values())
    w2 = {nid: [1] for nid in clients}
    dk1 = kg.get_decryption_keys("s", credentials={"fusion_weight": w2, "label": "s1"})
    assert agg.decrypt(single, dk1, w2, "s1") == [9, 12]
    single["c2"].pop("label_hash")
    with pytest.raises(FESchemeError, match="c2"):
        agg.decrypt(single, dk1, w2, "s1")


def test_ml_adapter_keeps_the_layer_marker():
    from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = ["c0", "c1"]
    for key, size in ((("mcfe", "ring_lwe"), 70), (("mcfe", "lwe"), 5)):
        w = FESchemeWrapper(*key, {"n": 2, "lst_nid": nids, "eta": 1, "bound_x": 100, "bound_y": 1}, precision=2)
        grads = {nid: np.random.default_rng(5).uniform(-1, 1, size).round(2) for nid in nids}
        enc = {nid: EncryptedGradient.from_dict(encrypt_gradient(g, w, nid=nid, label="r").to_dict())
               for nid, g in grads.items()}
        layer = enc["c0"].ciphertexts[0]
        assert (layer["label_hash"] if isinstance(layer, dict) else layer.flat[0]["label_hash"]) == 2
        assert np.allclose(np.asarray(aggregate_gradients(enc, w, {nid: 1 for nid in nids}, label="r"), dtype=float),
                           sum(grads.values()))


def test_threshold_tags_and_key_shares_bind_the_mode():
    # whole key shares, as v0.3.0's below (a label part of another mode does not complete its weight part)
    kg, pp, clients = _rlwe(None, tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator, tmcfe_rlwe.ThresholdMCFERingLWE,
                            32, s=3, t=2, lst_sid=["s0", "s1", "s2"], stat_param=40, key_shares="full")
    grads = {nid: np.random.default_rng(3).uniform(-1, 1, 40).round(2) for nid in clients}
    cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in clients}
    assert cts["c0"][0]["label_hash"] == 2
    creds = {"fusion_weight": {nid: [1] for nid in clients}, "label": "r",
             "n_chunks": tmcfe_rlwe.ThresholdMCFERingLWE.key_chunks(cts["c0"])}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    assert dk["label_hash"] == "shake128-rejection"
    srv = tmcfe_rlwe.ThresholdMCFERingLWE({"id": "s0", "precision": 2, "keys": {"pp": pp}})
    srv.compute_lst_ndarray_ct(cts, creds, dk, ["s0", "s1"])
    # the layer's code is under the servers' chunk tags: changing or dropping it breaks them
    for tamper in (lambda v: v[0].pop("label_hash"), lambda v: v[0].update(label_hash=1)):
        bad = _copy_layers(cts)
        tamper(bad["c2"])
        with pytest.raises(FESchemeError):
            srv.compute_lst_ndarray_ct(bad, creds, dk, ["s0", "s1"])
    # single ciphertexts carry their own code under their tags
    one = {nid: clients[nid].encrypt([[1, 2]], "one") for nid in clients}
    c1 = {"fusion_weight": {nid: [1] for nid in clients}, "label": "one"}
    dk1 = kg.get_decryption_keys("s0", credentials=c1)
    srv.share_decrypt(one, c1, dk1, ["s0", "s1"])
    one["c1"] = dict(one["c1"], label_hash=1)
    with pytest.raises(FESchemeError):
        srv.share_decrypt(one, c1, dk1, ["s0", "s1"])
    # a key share of another mode (or of v0.3.0, without the field)
    for share in (dict(dk, label_hash="shake256-wide"), {k: v for k, v in dk.items() if k != "label_hash"}):
        with pytest.raises(FESchemeError):
            srv.compute_lst_ndarray_ct(cts, creds, share, ["s0", "s1"])
    # a server holding public parameters of v0.3.0's format
    stale = tmcfe_rlwe.ThresholdMCFERingLWE(
        {"id": "s0", "precision": 2, "keys": {"pp": {k: v for k, v in pp.items() if k != "label_hash"}}})
    with pytest.raises(FESchemeError):
        stale.compute_lst_ndarray_ct(cts, creds, dk, ["s0", "s1"])


@pytest.mark.parametrize("backend", ["python", pytest.param("native", marks=needs_native)])
def test_dense_mode_bound_to_keys_and_ciphertexts(backend):
    nids = ["c0", "c1"]
    kg = mcfe_lwe.MCFELWEKeyGenerator({"n": 2, "lst_nid": nids, "eta": 1, "bound_x": 50, "bound_y": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    sk = kg.get_private_keys("c0")
    assert sk["label_hash"] == pp["label_hash"] == "shake128-rejection"
    stale = mcfe_lwe.MCFELWE({"id": "c0", "precision": 2,
                              "keys": {"pp": {k: v for k, v in pp.items() if k != "label_hash"}, "sk": sk}})
    clients = {nid: mcfe_lwe.MCFELWE({"id": nid, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    grads = {nid: np.random.default_rng(4).uniform(-0.5, 0.5, (2, 3)).round(2) for nid in nids}
    weights = {nid: [1] for nid in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": weights})
    agg = mcfe_lwe.MCFELWE({"id": "a", "precision": 2, "keys": {"pp": pp}})
    with native_backend.backend(backend):
        for call in (lambda: stale.encrypt([1], "x"), lambda: stale.encrypt_lst_ndarray([grads["c0"]], label="r")):
            with pytest.raises(FEKeyError):
                call()
        cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in nids}
        flat = cts["c0"][0].ravel()
        assert flat[0]["label_hash"] == 2 and all(set(ct) == {"c"} for ct in flat[1:])  # once per layer
        out = agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=weights, label="r")[0]
        assert np.allclose(np.asarray(out, dtype=float), sum(grads.values()))
        for tamper in (lambda a: a.__setitem__((0, 0), {"c": a[0, 0]["c"]}),             # as a v0.3.0 client
                       lambda a: a.__setitem__((1, 2), dict(a[1, 2], label_hash=1))):    # an element of its own mode
            bad = {nid: [a.copy() for a in v] for nid, v in cts.items()}
            tamper(bad["c1"][0])
            with pytest.raises(FESchemeError, match="c1"):
                agg.decrypt_lst_ndarray_ct(bad, dk=dk, fusion_weight=weights, label="r")
        single = {nid: clients[nid].encrypt([2], "one") for nid in nids}
        assert all(ct["label_hash"] == 2 for ct in single.values())
        assert agg.decrypt(single, dk, weights, "one") == 4
        single["c0"] = {"c": single["c0"]["c"]}
        with pytest.raises(FESchemeError, match="c0"):
            agg.decrypt(single, dk, weights, "one")
    # all of v0.3.0's format (no field anywhere, shake256-wide) still decrypts
    old = mcfe_lwe.MCFELWEKeyGenerator({"n": 2, "lst_nid": nids, "eta": 1, "bound_x": 50, "bound_y": 2,
                                        "label_hash": "shake256-wide"})
    old.setup()
    opp = {k: v for k, v in old.get_public_parameters().items() if k != "label_hash"}
    ocl = {nid: mcfe_lwe.MCFELWE({"id": nid, "precision": 2, "keys": {
               "pp": opp, "sk": {k: v for k, v in old.get_private_keys(nid).items() if k != "label_hash"}}})
           for nid in nids}
    octs = {nid: ocl[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in nids}
    assert all(set(ct) == {"c"} for ct in octs["c0"][0].ravel())
    oagg = mcfe_lwe.MCFELWE({"id": "a", "precision": 2, "keys": {"pp": opp}})
    oout = oagg.decrypt_lst_ndarray_ct(octs, dk=old.get_decryption_keys("s", credentials={"fusion_weight": weights}),
                                       fusion_weight=weights, label="r")[0]
    assert np.allclose(np.asarray(oout, dtype=float), sum(grads.values()))


def test_dense_threshold_partials_check_the_layer():
    from pyfe4ai.schemes.mcfe import lwe_threshold as tl

    nids, sids = ["c0", "c1"], ["s0", "s1", "s2"]
    kg = tl.ThresholdMCFELWEKeyGenerator({"n": 2, "lst_nid": nids, "eta": 1, "bound_x": 50, "bound_y": 2,
                                          "s": 3, "t": 2, "lst_sid": sids})
    kg.setup()
    pp = kg.get_public_parameters()
    cl = {nid: tl.ThresholdMCFELWE({"id": nid, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
          for nid in nids}
    grads = {nid: np.random.default_rng(6).uniform(-0.5, 0.5, 4).round(2) for nid in nids}
    cts = {nid: cl[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in nids}
    creds = {"fusion_weight": {nid: [1] for nid in nids}, "label": "r"}
    parts = {}
    for sid in sids[:2]:
        srv = tl.ThresholdMCFELWE({"id": sid, "precision": 2, "keys": {"pp": pp}})
        parts[sid] = srv.compute_lst_ndarray_ct(cts, creds, kg.get_decryption_keys(sid, credentials=creds), sids[:2])
    comb = tl.ThresholdMCFELWE({"id": "x", "precision": 2, "keys": {"pp": pp}})
    assert np.allclose(np.asarray(comb.decrypt_lst_ndarray_ct(parts)[0], dtype=float), sum(grads.values()))
    bad = {nid: [a.copy() for a in v] for nid, v in cts.items()}
    bad["c1"][0][0] = {"c": bad["c1"][0][0]["c"]}
    srv = tl.ThresholdMCFELWE({"id": "s0", "precision": 2, "keys": {"pp": pp}})
    with pytest.raises(FESchemeError, match="c1"):
        srv.compute_lst_ndarray_ct(bad, creds, kg.get_decryption_keys("s0", credentials=creds), sids[:2])


@needs_native
def test_native_encryption_key_cache_follows_the_label_hash():
    """The native encryption key cached in a client's key carries the pads'
    label-hash mode: a crypto object of another mode builds its own."""
    kg, pp, clients = _rlwe("shake128-rejection", mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE)
    with native_backend.backend("native"):
        first = clients["c0"]._native_encryption_key()
        assert first is not None and clients["c0"]._native_encryption_key() is first
        other = mcfe_rlwe.MCFERingLWE({"id": "c0", "precision": 2,
                                       "keys": {"pp": dict(pp, label_hash="shake256-wide"), "sk": clients["c0"].sk}})
        assert other._native_encryption_key() is not first

