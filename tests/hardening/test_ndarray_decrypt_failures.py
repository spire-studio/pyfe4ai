"""ndarray helpers: an element that does not decrypt raises FESchemeError.

Element-wise decryption returns ``None`` when the result is outside the
decryption bound or key and ciphertext do not match. The ndarray helpers used
to divide that ``None`` (``TypeError``) or turn it into ``NaN``; they now
raise :class:`FESchemeError` naming the element.
"""

import numpy as np
import os
import pytest

from pyfe4ai.core.exceptions import FESchemeError
from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, compute_linear, encrypt_features
from pyfe4ai.ml import encrypt_gradient

NIDS = ["nid_0", "nid_1"]
_MULTI = {"n": 2, "s": 1, "lst_nid": NIDS, "eta": 1}
LABEL = "round-1"
GRADIENT = np.array([2.0, -1.0, 1.0])  # integers: precision 0 keeps the toy bounds
FAILED = r"element \(1,\) of layer 0 does not decrypt"

AGGREGATION = {
    ("sife", "lwe"): {"sec_param": 64, "lwe_n": 16, "bound_x": 8, "bound_y": 8, "eta": 1},
    ("sife", "paillier"): {"sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10, "eta": 1},
    ("sife", "damgard"): {"sec_param": 64, "modulus_length": 64, "bound": 10, "eta": 1},
    ("sife", "fullysec_lwe"): {"sec_param": 64, "lwe_n": 24, "bound_x": 10, "bound_y": 10, "eta": 1},
    ("sife", "fullysec_ring_lwe"): {"sec_param": 64, "bound_x": 4, "bound_y": 4, "eta": 1},
    ("mcfe", "ddh"): {"sec_param": 128, **_MULTI},
    ("mcfe", "damgard"): {"sec_param": 64, "modulus_length": 64, "bound": 10, **_MULTI},
    ("mcfe", "lwe"): {"sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, **_MULTI},
    ("mcfe", "fullysec_lwe"): {"sec_param": 64, "lwe_n": 24, "bound_x": 8, "bound_y": 8, **_MULTI},
    ("mcfe", "paillier"): {"sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10, **_MULTI},
    ("mcfe", "ring_lwe"): {"sec_param": 64, "ring_n": 16, "bound_x": 2, "bound_y": 2, **_MULTI},
}

MIFEDDH = {
    "ddh": {"sec_param": 128, **_MULTI},
    "damgard": {"sec_param": 64, "modulus_length": 64, "bound": 10, **_MULTI},
    "lwe": {"sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, **_MULTI},
    "fullysec_lwe": {"sec_param": 64, "lwe_n": 24, "bound_x": 8, "bound_y": 8, **_MULTI},
    "paillier": {"sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10, **_MULTI},
}


@pytest.fixture(autouse=True)
def _param_cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _fail_second_element(monkeypatch, cls):
    """Make the element-level decoder of *cls* fail on the second element."""
    if hasattr(cls, "_decrypt_with_key_term"):  # Ring-LWE MCFE: one call per packed chunk
        name = "_decrypt_with_key_term"
        original = getattr(cls, name)
        # the Python path (the native one decrypts a whole layer per call;
        # tests/native/test_native_backend.py covers its failures): the reference ring
        ring = cls._ring
        monkeypatch.setattr(cls, "_ring", lambda self: ring(self).reference())

        def fake(self, *args, **kwargs):
            values = list(original(self, *args, **kwargs))
            values[1] = None
            return values
    else:
        name = "_decode" if cls.__module__.endswith("sife.lwe") else "decrypt"
        original = getattr(cls, name)
        if hasattr(cls, "_native_dots"):  # dense-LWE MCFE: the per-element Python path
            monkeypatch.setattr(cls, "_native_dots", lambda self: False)
        calls = []

        def fake(self, *args, **kwargs):
            calls.append(None)
            return None if len(calls) == 2 else original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, fake)


@pytest.mark.parametrize("key", sorted(AGGREGATION), ids="/".join)
def test_aggregation_reports_the_element_that_does_not_decrypt(key, monkeypatch):
    family, variant = key
    wrapper = FESchemeWrapper(family, variant, AGGREGATION[key], precision=0)
    nids = NIDS if family == "mcfe" else ["nid_default"]
    label = LABEL if family == "mcfe" else None
    encrypted = {nid: encrypt_gradient(GRADIENT, wrapper, nid=nid, label=label) for nid in nids}
    weights = {nid: [1] for nid in nids}
    expected = GRADIENT * len(nids)
    np.testing.assert_allclose(aggregate_gradients(encrypted, wrapper, weights, label=label), expected)

    _fail_second_element(monkeypatch, type(wrapper._get_decryptor("sid_agg")))
    with pytest.raises(FESchemeError, match=FAILED):
        aggregate_gradients(encrypted, wrapper, weights, label=label)


@pytest.mark.parametrize("variant", sorted(MIFEDDH))
def test_mife_ndarray_reports_the_element_that_does_not_decrypt(variant, monkeypatch):
    wrapper = FESchemeWrapper("mife", variant, MIFEDDH[variant], precision=0)
    dct_ct = {nid: wrapper._get_encryptor(nid).encrypt_lst_ndarray([GRADIENT]) for nid in NIDS}
    weights = {nid: [1] for nid in NIDS}
    dk = wrapper.kg.get_decryption_keys("sid_0", credentials={"fusion_weight": weights})
    decryptor = wrapper._get_decryptor("sid_0")
    result = decryptor.decrypt_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=weights)
    np.testing.assert_allclose(result[0], 2 * GRADIENT)

    _fail_second_element(monkeypatch, type(decryptor))
    with pytest.raises(FESchemeError, match=FAILED):
        decryptor.decrypt_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=weights)


def test_sife_paillier_ndarray_checks_ciphertexts_like_decrypt():
    """The ndarray path decoded inline, skipping the range check of decrypt."""
    wrapper = FESchemeWrapper("sife", "paillier", AGGREGATION[("sife", "paillier")], precision=0)
    enc = encrypt_gradient(GRADIENT, wrapper)
    wrong = wrapper.kg.get_decryption_keys("sid_0", credentials={"fusion_weight": [1]})
    wrong = dict(wrong, k=str(int(wrong["k"]) + 1))
    decryptor = wrapper._get_decryptor("sid_0")
    with pytest.raises(FESchemeError, match="element \\(0,\\) of layer 0 does not decrypt"):
        decryptor.decrypt_lst_ndarray_ct(enc.ciphertexts, dk=wrong, fusion_weight=[1])


def test_compute_linear_reports_a_result_that_does_not_decrypt(monkeypatch):
    wrapper = FESchemeWrapper("sife", "damgard", dict(AGGREGATION[("sife", "damgard")], eta=3), precision=0)
    features = encrypt_features(np.array([2.0, 1.0, -1.0]), wrapper)
    assert compute_linear(features, wrapper, np.array([1.0, 1.0, 1.0])) == pytest.approx(2.0)
    monkeypatch.setattr(type(wrapper._get_decryptor("sid_infer")), "decrypt", lambda self, *a, **k: None)
    with pytest.raises(FESchemeError, match="does not decrypt"):
        compute_linear(features, wrapper, np.array([1.0, 1.0, 1.0]))


# -- pairing-based helpers (Docker image) -------------------------------------


def test_pairing_ndarray_helpers_raise_instead_of_nan(monkeypatch):
    from tests.pairing_group import GROUP, skip_unless_available

    skip_unless_available()
    from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEPairing
    from pyfe4ai.schemes.quadratic.quad import QuadraticQuad, QuadraticQuadKeyGenerator
    from pyfe4ai.schemes.quadratic.sgp import QuadraticSGP, QuadraticSGPKeyGenerator

    # MIFE FH-Multi-IPE
    cfg = {"sec_param": 64, "sec_level": 2, "bound_x": 5, "bound_y": 5, "pairing_group_param": GROUP, **_MULTI}
    wrapper = FESchemeWrapper("mife", "fh_multi_ipe_pairing", cfg, precision=0)
    dct_ct = {nid: wrapper._get_encryptor(nid).encrypt_lst_ndarray([GRADIENT]) for nid in NIDS}
    weights = {nid: [1] for nid in NIDS}
    dk = wrapper.kg.get_decryption_keys("sid_0", credentials={"fusion_weight": weights})
    decryptor = wrapper._get_decryptor("sid_0")
    np.testing.assert_allclose(decryptor.decrypt_lst_ndarray_ct(dct_ct, dk=dk)[0], 2 * GRADIENT)
    _fail_second_element(monkeypatch, MIFEFHMultiIPEPairing)
    with pytest.raises(FESchemeError, match=FAILED):
        decryptor.decrypt_lst_ndarray_ct(dct_ct, dk=dk)

    # quadratic Quad and SGP
    x_lst, y_lst = [np.array([1.0, 2.0, -1.0]), np.array([2.0, 1.0, 1.0])], [np.array([1.0, 1.0, 1.0])] * 2
    for kg_cls, crypto_cls, cfg in (
        (QuadraticQuadKeyGenerator, QuadraticQuad, {"n": 2, "m": 2}),
        (QuadraticSGPKeyGenerator, QuadraticSGP, {"n": 2}),
    ):
        kg = kg_cls({"sec_param": 64, "bound": 50, "pairing_group_param": GROUP, **cfg})
        kg.setup()
        pp = kg.get_public_parameters()
        ct = crypto_cls({"keys": {"pp": pp}, "precision": 0}).encrypt_lst_ndarray(x_lst, rhs_lst_ndarray=y_lst)
        dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": [[1, 0], [0, 1]]})
        decryptor = crypto_cls({"keys": {"pp": pp}, "precision": 0})
        np.testing.assert_allclose(decryptor.decrypt_lst_ndarray_ct(ct, dk=dk)[0], [3.0, 3.0, 0.0])
        with monkeypatch.context() as m:
            _fail_second_element(m, crypto_cls)
            with pytest.raises(FESchemeError, match=FAILED):
                decryptor.decrypt_lst_ndarray_ct(ct, dk=dk)
