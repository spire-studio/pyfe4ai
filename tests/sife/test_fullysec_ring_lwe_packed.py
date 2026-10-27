"""Packed layout of the adaptively secure Ring-LWE IPFE (BMMS21 Sec. 5 with Remark 1)."""

import json
import math
import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.sife import fullysec_ring_lwe as fs
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWE, SIFEFullySecRingLWEKeyGenerator
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.primitives import sampling as sampling_utils
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, aggregate_gradients, encrypt_gradient

CFG = {"eta": 3, "bound_x": 4, "bound_y": 4, "ndarray_layout": "packed"}
RECORD = os.path.join("config", "authority", "SIFE_FULLYSEC_RING_LWE", "param.json")


@pytest.fixture(autouse=True)
def _param_cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _keys(cfg):
    kg = SIFEFullySecRingLWEKeyGenerator(dict(cfg))
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}}), pp


@pytest.fixture(scope="module")
def scheme(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("cache"))
        mp.setenv("PYFE4AI_CACHE_DIR", os.path.join(os.getcwd(), "config"))
        return _keys(CFG)


def _products(matrix, y):
    return [sum(row[j] * w for row, w in zip(matrix, y)) for j in range(len(matrix[0]))]


# -- parameters ---------------------------------------------------------------


def test_elementwise_leveraging_is_the_previous_one():
    assert rlu.bmms21_adaptive_leveraging(4, 4, 4096) == (8 * math.log2(9), 1)


@pytest.mark.parametrize("l, X, n", [(1, 100, 4096), (4, 4, 4096), (10, 50, 8192), (40, 2, 32)])
def test_packed_leveraging_ranges_over_the_span_of_the_differences(l, X, n):
    lev, rank = rlu.bmms21_adaptive_leveraging(l, X, n, packed=True)
    assert rank == min(l, n)
    if l == 1:  # V is {0} or Q
        assert lev == 1
    else:  # at most D + 1 dimensions, each subspace spanned by D vectors of [-2X, 2X]^l
        assert lev == pytest.approx(math.log2((rank + 1) * (4 * X + 1) ** (l * rank)))
    # guessing both challenge matrices of k = n vectors would cost (2X + 1)^(2 l n)
    assert lev < 2 * l * n * math.log2(2 * X + 1)


@pytest.mark.parametrize("kappa, l, X, Y, n, m", [
    (128, 10, 50, 50, 8192, 9),
    (128, 4, 4, 4, 4096, 18),
    (128, 1, 100, 1, 4096, 12),
    (128, 1, 4, 4, 32, 6),
    (128, 3, 4, 4, 32, 8),
])
def test_packed_parameters_meet_the_constraints(kappa, l, X, Y, n, m):
    P = rlu.bmms21_adaptive_parameters(kappa, l, X, Y, n, m, packed=True)
    q, p = int(P["q"]), int(P["p"])
    assert q % (2 * n) == 1 and gp.is_prime(q) and p == 2 * l * X * Y + 2
    lev, D = rlu.bmms21_adaptive_leveraging(l, X, n, packed=True)
    # Theorem 2 (k = 2), once per basis vector of the differences: D 2 delta + bad A
    log2_delta = -(kappa + 3 + lev + math.log2(D))
    ln_term = math.log(2 * m * n) + math.log1p(2.0 ** -log2_delta)
    log2_root = 0.5 * math.log2(n * ln_term / math.pi)
    eps = 2 * (math.log2(P["sigma_lhl"]) - log2_root) / math.log2(q) - 4 / m
    bad = (2 * m - math.log2((q ** m - 1) * (q ** m - q)) / math.log2(q))
    log2_bad = n - eps * n * math.log2(q) + n * bad * math.log2(q)
    assert math.log2(D * 2 * 2.0 ** log2_delta + 2.0 ** log2_bad) + lev <= -kappa
    # G3 -> G4: Lemma 1 along a basis B of the differences, ||B|| <= 2X sqrt(l D);
    # for l = 1 the basis [1] of V = Q
    norm = 1 if l == 1 else 2 * X * math.sqrt(l * D)
    assert P["sigma1"] >= math.sqrt(2) * norm * P["sigma_lhl"] * (1 - 1e-12)
    # the differences are rational combinations of B: denominators (minors) below q
    assert l == 1 or D * math.log2(2 * X * math.sqrt(D)) < math.log2(q)
    # Theorem 1 and correctness as for one vector per ciphertext
    C = math.sqrt(kappa) * P["sigma1"]
    eta = math.sqrt((math.log(2 * (n + n * l)) + kappa * math.log(2)) / math.pi)
    ratio = P["sigma"] * n * C * math.sqrt(l + 2) / P["sigma2"]
    assert P["sigma"] * math.sqrt(1 - ratio ** 2) > eta
    assert P["sigma3"] == pytest.approx(math.sqrt(m) * P["sigma2"])
    noise = l * Y * (math.sqrt(kappa) * P["sigma3"] + m * n * kappa * P["sigma1"] * P["sigma2"])
    assert noise < q // (2 * p)


@pytest.mark.parametrize("l, X, Y, expected", [
    (1, 100, 1, (4096, 12, 107)),
    (4, 4, 4, (4096, 18, 111)),
    (10, 50, 50, (8192, 9, 205)),
])
def test_standard_packed_parameters(l, X, Y, expected):
    P = rlu.bmms21_adaptive_choice(128, l, X, Y, "auto", "auto", True, packed=True)
    assert (P["ring_n"], P["m"], int(P["q"]).bit_length()) == expected
    rlu.check_ring_lwe(P["ring_n"], P["q"], P["sigma"])
    if l == 1:  # basis [1]: smaller than one vector per ciphertext, which stays as it was
        E = rlu.bmms21_adaptive_choice(128, l, X, Y, "auto", "auto", True)
        assert (E["ring_n"], E["m"], int(E["q"]).bit_length()) == (4096, 19, 111)
        assert P["sigma1"] < E["sigma1"] and P["m"] < E["m"]


@pytest.mark.parametrize("l, X, Y, expected", [
    (24, 4, 4, (8192, 8, 206)),
    (32, 1, 1, (8192, 8, 190)),
    (64, 1, 1, (16384, 6, 312)),
])
def test_modulus_exceeds_the_minors_of_the_differences(l, X, Y, expected):
    """At n = 4096 these bounds met App. C.1 with a 111-bit q below the minors
    (2X sqrt(D))^D (127, 112 and 256 bits): G5* could not write every
    difference in the basis of their span mod q."""
    P = rlu.bmms21_adaptive_choice(128, l, X, Y, "auto", "auto", True, packed=True)
    assert (P["ring_n"], P["m"], int(P["q"]).bit_length()) == expected
    D = min(l, P["ring_n"])
    assert math.log2(int(P["q"])) > D * math.log2(2 * X * math.sqrt(D)) + 1
    rlu.check_ring_lwe(P["ring_n"], P["q"], P["sigma"])
    # the floor raised q above the correctness minimum of App. C.1 at n = 4096
    assert rlu.bmms21_adaptive_log2_q(128, l, X, Y, 4096, 20) < D * math.log2(2 * X * math.sqrt(D))
    assert rlu.bmms21_adaptive_log2_q(128, l, X, Y, 4096, 20, packed=True) > D * math.log2(2 * X * math.sqrt(D))


# -- records ------------------------------------------------------------------


def test_packed_record_has_its_own_version_and_layout(scheme):
    kg, _, pp = scheme
    assert pp["ndarray_layout"] == "packed"
    SIFEFullySecRingLWEKeyGenerator(dict(CFG))
    record = json.load(open(RECORD))
    assert record["param_version"] == "bmms21-sec5-packed/v2" and record["ndarray_layout"] == "packed"
    expected = rlu.bmms21_adaptive_choice(kg.sec_param, 3, 4, 4, None, "auto", False, packed=True)
    assert (record["m"], record["sigma1"]) == (expected["m"], expected["sigma1"])


def test_switching_the_layout_regenerates_the_parameters():
    elementwise = {k: v for k, v in CFG.items() if k != "ndarray_layout"}
    kg = SIFEFullySecRingLWEKeyGenerator(dict(elementwise))
    record = json.load(open(RECORD))
    assert record["param_version"] == "bmms21-sec5/v2" and "ndarray_layout" not in record
    kg.setup()
    assert "ndarray_layout" not in kg.get_public_parameters()
    packed = SIFEFullySecRingLWEKeyGenerator(dict(CFG))
    assert packed.sigma1 > kg.sigma1  # sqrt(l D) and the larger leveraging
    assert json.load(open(RECORD))["param_version"] == "bmms21-sec5-packed/v2"
    again = SIFEFullySecRingLWEKeyGenerator(dict(elementwise))
    assert json.load(open(RECORD))["param_version"] == "bmms21-sec5/v2" and again.sigma1 == kg.sigma1


def test_a_record_relabelled_as_packed_is_regenerated():
    kg = SIFEFullySecRingLWEKeyGenerator({k: v for k, v in CFG.items() if k != "ndarray_layout"})
    record = json.load(open(RECORD))
    record.update(param_version="bmms21-sec5-packed/v2", ndarray_layout="packed")
    json.dump(record, open(RECORD, "w"))
    packed = SIFEFullySecRingLWEKeyGenerator(dict(CFG))
    assert packed.a_seed != kg.a_seed and packed.sigma1 > kg.sigma1


def test_unknown_layout_is_refused():
    with pytest.raises(FEValidationError, match="ndarray_layout"):
        SIFEFullySecRingLWEKeyGenerator({**CFG, "ndarray_layout": "rows"})


# -- scheme -------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 17, 32])
def test_matrix_round_trip(scheme, k):
    kg, enc, pp = scheme
    dec = SIFEFullySecRingLWE({"keys": {"pp": pp}})
    rng = random.Random(k)
    matrix = [[rng.randint(-4, 4) for _ in range(k)] for _ in range(3)]
    matrix[0][0], matrix[1][0], matrix[2][0] = 4, -4, 4  # an extreme column
    ct = enc.encrypt(matrix)
    assert ct["k"] == k and len(ct["ct0"]) == kg.m and len(ct["ct"]) == 3
    for y in ([4, -4, 4], [1, 0, -2], [0, 0, 0]):
        dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
        assert dec.decrypt(ct, dk, y) == _products(matrix, y)


def test_vectors_still_decrypt_to_an_integer(scheme):
    kg, enc, pp = scheme
    y = [2, -3, 1]
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
    ct = enc.encrypt([4, 4, -4])
    assert "k" not in ct and SIFEFullySecRingLWE({"keys": {"pp": pp}}).decrypt(ct, dk, y) == -8


def test_matrix_plaintexts_are_validated(scheme):
    _, enc, pp = scheme
    n = pp["ring_n"]
    for bad in ([[5], [0], [0]], [[1, 2], [1], [1, 2]], [[1], [1]], [[]] * 3,
                [[0] * (n + 1)] * 3, [[True], [0], [0]]):
        with pytest.raises(FEValidationError):
            enc.encrypt(bad)


def test_elementwise_parameters_refuse_matrices():
    _, enc, pp = _keys({k: v for k, v in CFG.items() if k != "ndarray_layout"})
    with pytest.raises(FEValidationError, match="packed"):
        enc.encrypt([[1, 2], [3, 4], [0, 0]])
    # nor do they decrypt a packed ciphertext
    kg_p, enc_p, pp_p = _keys(CFG)
    ct = enc_p.encrypt([[1, 2], [3, 4], [0, 0]])
    dk = kg_p.get_decryption_keys("sid", credentials={"fusion_weight": [1, 1, 1]})
    with pytest.raises(FEValidationError):
        SIFEFullySecRingLWE({"keys": {"pp": pp}}).decrypt(ct, dk, [1, 1, 1])


@pytest.mark.parametrize("k", [0, 33, True, "2", None])
def test_tampered_batch_lengths_are_refused(scheme, k):
    kg, enc, pp = scheme
    ct = dict(enc.encrypt([[1, 2], [3, 4], [0, 0]]), k=k)
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1, 1, 1]})
    with pytest.raises(FEValidationError):
        SIFEFullySecRingLWE({"keys": {"pp": pp}}).decrypt(ct, dk, [1, 1, 1])


def test_a_wrong_key_does_not_decrypt(scheme):
    kg, enc, pp = scheme
    matrix = [[1] * 32, [2] * 32, [3] * 32]
    ct = enc.encrypt(matrix)
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1, 1, 1]})
    dk = dict(dk, sk_y=[[gp.digits((int(v) + 1) % int(pp["q"])) for v in poly] for poly in dk["sk_y"]])
    # random values: out of the bound l X Y = 48 only with probability 2/98 each
    out = SIFEFullySecRingLWE({"keys": {"pp": pp}}).decrypt(ct, dk, [1, 1, 1])
    assert out is None or out != _products(matrix, [1, 1, 1])


@pytest.mark.parametrize("eta, X, Y, profile, k", [
    (1, 100, 1, "toy", 32), (3, 4, 4, "toy", 32), (2, 7, 3, "toy", 20), (5, 1, 9, "toy", 20),
    # the standard configurations of the parameter table (two columns each: these keys are large)
    (1, 100, 1, "standard", 2), (4, 4, 4, "standard", 2), (10, 50, 50, "standard", 2),
])
def test_packed_decryption_equals_per_element_decryption(eta, X, Y, profile, k):
    """Column j of a packed ciphertext decrypts to what a one-vector ciphertext of
    column j decrypts to -- under packed keys and under elementwise keys -- and
    both are <x_j, y>."""
    base = {"eta": eta, "bound_x": X, "bound_y": Y}
    if profile == "standard":
        base["security_profile"] = "standard"
    kg_p, enc_p, pp_p = _keys({**base, "ndarray_layout": "packed"})
    kg_e, enc_e, pp_e = _keys(base)
    dec_p, dec_e = SIFEFullySecRingLWE({"keys": {"pp": pp_p}}), SIFEFullySecRingLWE({"keys": {"pp": pp_e}})
    rng = random.Random(eta * 1000 + X)
    matrix = [[X] + [rng.randint(-X, X) for _ in range(k - 1)] for _ in range(eta)]  # column 0 extreme
    columns = [[row[j] for row in matrix] for j in range(k)]
    weights = [[Y if i % 2 else -Y for i in range(eta)]]
    if profile == "toy":
        weights += [[Y] * eta, [rng.randint(-Y, Y) for _ in range(eta)]]
    for y in weights:
        dk_p = kg_p.get_decryption_keys("sid", credentials={"fusion_weight": y})
        dk_e = kg_e.get_decryption_keys("sid", credentials={"fusion_weight": y})
        packed = dec_p.decrypt(enc_p.encrypt(matrix), dk_p, y)
        one_by_one_packed_keys = [dec_p.decrypt(enc_p.encrypt(c), dk_p, y) for c in columns]
        one_by_one_elementwise = [dec_e.decrypt(enc_e.encrypt(c), dk_e, y) for c in columns]
        assert packed == one_by_one_packed_keys == one_by_one_elementwise == _products(matrix, y)


# -- ndarray helper (eta = 1) ---------------------------------------------------


@pytest.fixture(scope="module")
def gradient_scheme(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("grad"))
        mp.setenv("PYFE4AI_CACHE_DIR", os.path.join(os.getcwd(), "config"))
        kg = SIFEFullySecRingLWEKeyGenerator({"eta": 1, "bound_x": 100, "bound_y": 1, "ndarray_layout": "packed"})
        kg.setup()
        pp = kg.get_public_parameters()
        enc = SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}, "precision": 2})
        dec = SIFEFullySecRingLWE({"keys": {"pp": pp}, "precision": 2})
        return kg, enc, dec, pp


def test_ndarray_layers_are_packed_in_chunks_of_ring_n(gradient_scheme):
    kg, enc, dec, pp = gradient_scheme
    n = pp["ring_n"]
    rng = np.random.default_rng(3)
    layers = [rng.uniform(-1, 1, size=s).round(2) for s in ((5,), (3, 2 * n), (n + 1,))]
    cts = enc.encrypt_lst_ndarray(layers)
    assert [c["first_chunk"] for c in cts] == [0, 1, 7]
    assert [[ch["k"] for ch in c["chunks"]] for c in cts] == [[5], [n] * 6, [n, 1]]
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1]})
    out = dec.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=[1])
    for got, want in zip(out, layers):
        assert got.shape == want.shape and np.allclose(got, want)
    # the serialised form is the one of the packed MCFE Ring-LWE layers
    grad = EncryptedGradient(ciphertexts=cts, shape=(5,), precision=2, mode="decimal", nid="c")
    back = EncryptedGradient.from_dict(json.loads(json.dumps(grad.to_dict())))
    assert all(np.allclose(a, b) for a, b in
               zip(dec.decrypt_lst_ndarray_ct(back.ciphertexts, dk=dk, fusion_weight=[1]), layers))


def test_ndarray_chunk_failures_are_reported(gradient_scheme):
    kg, enc, dec, pp = gradient_scheme
    cts = enc.encrypt_lst_ndarray([np.zeros(10)])
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1]})
    q = int(pp["q"])
    chunk = cts[0]["chunks"][0]
    chunk["ct"] = [[gp.digits((int(chunk["ct"][0][0]) + q // 2) % q)] + chunk["ct"][0][1:]]  # 0 -> p/2 > X
    with pytest.raises(FESchemeError, match="element"):
        dec.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=[1])
    short = [dict(cts[0], shape=[11])]
    with pytest.raises(FEValidationError, match="shape"):
        dec.decrypt_lst_ndarray_ct(short, dk=dk, fusion_weight=[1])
    with pytest.raises(FEValidationError):
        enc.encrypt_lst_ndarray([np.zeros(0)])


def test_gradient_aggregation_through_the_adapter():
    wrapper = FESchemeWrapper("sife", "fullysec_ring_lwe",
                              {"eta": 1, "bound_x": 100, "bound_y": 1, "ndarray_layout": "packed"},
                              precision=2)
    gradient = np.random.default_rng(5).uniform(-0.5, 0.5, size=(4, 25)).round(2)
    enc = encrypt_gradient(gradient, wrapper, nid="c1")
    assert len(enc.ciphertexts[0]["chunks"]) == math.ceil(gradient.size / wrapper.pp["ring_n"])
    out = aggregate_gradients({"c1": enc}, wrapper, {"c1": [1]})
    assert np.allclose(out, gradient)


# -- Python vs native -----------------------------------------------------------


@pytest.mark.skipif(not native_backend.available(), reason="Rust extension not built")
def test_packed_identical_with_native_ntts(monkeypatch):
    # m = 7: a 127-bit q (D = 2), within the native word sizes
    kg, _, pp = _keys({"eta": 2, "ring_n": 64, "m": 7, "sec_param": 128, "bound_x": 1, "bound_y": 1,
                       "ndarray_layout": "packed"})
    with native_backend.backend("native"):
        assert native_backend.ops()._ntt_ring(gp.mpz(pp["q"]), pp["ring_n"], rlu._get_ntt_root(gp.mpz(pp["q"]), pp["ring_n"]))
    matrix = [[(-1) ** j for j in range(64)], [j % 3 - 1 for j in range(64)]]

    def run(backend):  # the same bits for the FACCT samples of both runs
        monkeypatch.setattr(sampling_utils, "_OS_RANDOM", sampling_utils.XofRandom(b"packed-native" * 3))
        with native_backend.backend(backend):
            return SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}}).encrypt(matrix)

    py, nat = run("python"), run("native")
    assert py == nat
    y = [1, -1]
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    dec = SIFEFullySecRingLWE({"keys": {"pp": pp}})
    with native_backend.backend("python"):
        a = dec.decrypt(py, dk, y)
    with native_backend.backend("native"):
        b = dec.decrypt(py, dk, y)
    assert a == b == _products(matrix, y)


def test_layouts_are_listed():
    assert fs.LAYOUTS == ("elementwise", "packed")


# -- October 2026 review: binding of the layout flag, strict layers, status ------


def test_a_packed_flag_on_elementwise_parameters_is_refused():
    kg, _, pp = _keys({k: v for k, v in CFG.items() if k != "ndarray_layout"})
    relabelled = dict(pp, ndarray_layout="packed")  # sigma1 of one vector per ciphertext
    enc = SIFEFullySecRingLWE({"keys": {"pp": relabelled, "sk": kg.get_private_keys()}})
    with pytest.raises(FEValidationError, match="packed"):
        enc.encrypt([[4] * 32, [-4] * 32, [3] * 32])
    kg_p, enc_p, _ = _keys(CFG)
    ct = enc_p.encrypt([[1], [1], [1]])
    dk = kg_p.get_decryption_keys("sid", credentials={"fusion_weight": [1, 1, 1]})
    with pytest.raises(FEValidationError, match="packed"):  # decryptors check it too
        SIFEFullySecRingLWE({"keys": {"pp": relabelled}}).decrypt(ct, dk, [1, 1, 1])
    for key in ("sigma2", "q"):  # nor a packed pp with one value changed
        kg_p, _, pp_p = _keys(CFG)
        bad = dict(pp_p, **{key: pp_p[key] * 2 if key == "sigma2" else str(int(pp_p[key]) + 2 * pp_p["ring_n"])})
        with pytest.raises(FEValidationError):
            SIFEFullySecRingLWE({"keys": {"pp": bad, "sk": kg_p.get_private_keys()}}).encrypt([[1], [1], [1]])


def test_the_packed_check_follows_changes_of_pp():
    kg, enc, pp = _keys(CFG)
    enc.encrypt([[1], [1], [1]])  # checked and cached
    pp["sigma2"] = pp["sigma2"] / 64  # the same dict, changed after the first use
    with pytest.raises(FEValidationError, match="packed"):
        enc.encrypt([[1], [1], [1]])
    pp["sigma2"] = kg.sigma2
    enc.encrypt([[1], [1], [1]])
    pp["a"] = pp["a"][:-1]  # m - 1 public polynomials
    with pytest.raises(FEValidationError, match="packed"):
        enc.encrypt([[1], [1], [1]])


def test_packed_parameters_survive_a_json_round_trip(scheme):
    kg, _, pp = scheme
    again = json.loads(json.dumps(pp))
    enc = SIFEFullySecRingLWE({"keys": {"pp": again, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1, 1, 1]})
    ct = enc.encrypt([[1, 2], [3, 4], [-4, 0]])
    assert SIFEFullySecRingLWE({"keys": {"pp": again}}).decrypt(ct, dk, [1, 1, 1]) == [0, 6]


@pytest.mark.parametrize("mutate", [
    lambda layer: layer.update(shape=[True, 70]),
    lambda layer: layer.update(shape=["7", 10.9]),
    lambda layer: layer.update(shape=[-7, -10]),  # np.prod gives 70
    lambda layer: layer.update(shape=[2 ** 62, 4], chunks=[]),
    lambda layer: layer.update(shape=[2 ** 62, 2 ** 2]),  # int64 wraps to 0
    lambda layer: layer.update(first_chunk="garbage"),
    lambda layer: layer.update(first_chunk=1),
    lambda layer: layer.update(chunks=layer["chunks"][::-1]),
    lambda layer: layer.update(chunks=layer["chunks"][:1] + layer["chunks"][2:]),
    lambda layer: layer.update(extra=1),
    lambda layer: layer["chunks"][0].update(k=31),
], ids=["bool-dim", "str-float-dims", "negative-dims", "huge-empty", "int64-wrap", "first-chunk-str",
        "first-chunk-offset", "reversed-chunks", "dropped-chunk", "extra-key", "short-chunk"])
def test_malformed_packed_layers_are_refused(gradient_scheme, mutate):
    kg, enc, dec, pp = gradient_scheme
    layers = enc.encrypt_lst_ndarray([(np.arange(70).reshape(7, 10) - 35) / 100])
    assert [c["k"] for c in layers[0]["chunks"]] == [32, 32, 6]
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1]})
    mutate(layers[0])
    with pytest.raises(FEValidationError):
        dec.decrypt_lst_ndarray_ct(layers, dk=dk, fusion_weight=[1])


def test_first_chunk_counts_the_chunks_of_earlier_layers(gradient_scheme):
    kg, enc, dec, pp = gradient_scheme
    layers = enc.encrypt_lst_ndarray([np.zeros(40), np.ones(3) / 100])
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1]})
    assert [c["first_chunk"] for c in layers] == [0, 2]
    assert np.allclose(dec.decrypt_lst_ndarray_ct(layers, dk=dk, fusion_weight=[1])[1], 0.01)
    with pytest.raises(FEValidationError, match="chunk"):
        dec.decrypt_lst_ndarray_ct(layers[1:], dk=dk, fusion_weight=[1])


def test_status_marks_the_packed_layout_experimental():
    from pyfe4ai.core.status import REVIEWED, SCHEME_STATUS

    status = SCHEME_STATUS[("sife", "fullysec_ring_lwe")]
    assert status.status == REVIEWED  # the elementwise layout keeps its status
    assert "experimental" in status.note.lower() and "packed" in status.note


# -- Gaussian sampler (exact FACCT by default for this scheme) -----------------


def _spy(monkeypatch):
    calls = []
    real = sampling_utils.discrete_gaussian_matrix

    def spy(rows, cols, sigma, sampler=None):
        calls.append(sampler)
        return real(rows, cols, sigma, sampler=sampler)

    monkeypatch.setattr(fs, "discrete_gaussian_matrix", spy)
    return calls


@pytest.mark.parametrize("layout", ["elementwise", "packed"])
def test_keys_and_noise_are_exact_gaussians_by_default(monkeypatch, layout):
    calls = _spy(monkeypatch)
    with sampling_utils.gaussian_sampler("rounded"):  # the library-wide choice does not apply
        kg, enc, pp = _keys({**CFG, "ndarray_layout": layout})
        assert pp["gaussian_sampler"] == "facct" and calls == ["facct"] * 3  # s_i, eta = 3
        enc.encrypt([1, 2, 3])
    assert calls[3:] == ["facct"] * 3  # r, f0, f_i


def test_the_rounded_sampler_stays_selectable(monkeypatch):
    calls = _spy(monkeypatch)
    kg, enc, pp = _keys({**CFG, "gaussian_sampler": "rounded"})
    assert pp["gaussian_sampler"] == "rounded" and calls == ["rounded"] * 3
    enc.encrypt([[1], [2], [3]])
    exact = SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}, "gaussian_sampler": "facct"})
    exact.encrypt([[1], [2], [3]])
    assert calls[3:] == ["rounded"] * 3 + ["facct"] * 3
    old_pp = {k: v for k, v in pp.items() if k != "gaussian_sampler"}  # pp of earlier versions
    assert SIFEFullySecRingLWE({"keys": {"pp": old_pp}}).gaussian_sampler == "facct"


def test_unknown_samplers_are_refused(scheme):
    _, _, pp = scheme
    with pytest.raises(FEValidationError, match="gaussian_sampler"):
        SIFEFullySecRingLWEKeyGenerator({**CFG, "gaussian_sampler": "exact"})
    with pytest.raises(FEValidationError, match="gaussian_sampler"):
        SIFEFullySecRingLWE({"keys": {"pp": dict(pp, gaussian_sampler="box-muller")}})

