"""Tests for pyfe4ai.ml — ML framework adapter layer."""

from __future__ import annotations

import numpy as np
import pytest

from pyfe4ai.ml import (
    EncryptedFeatures,
    EncryptedGradient,
    FESchemeWrapper,
    _to_numpy,
    aggregate_gradients,
    compute_linear,
    encrypt_features,
    encrypt_gradient,
)


# ── helpers ────────────────────────────────────────────────────────


class TestToNumpy:
    def test_ndarray_passthrough(self):
        arr = np.array([1, 2, 3])
        result = _to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_list_conversion(self):
        result = _to_numpy([1.0, 2.0])
        assert isinstance(result, np.ndarray)


# ── FESchemeWrapper ────────────────────────────────────────────────


class TestFESchemeWrapper:
    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown scheme"):
            FESchemeWrapper("unknown", "lwe", {})

    def test_sife_lwe_setup(self):
        wrapper = FESchemeWrapper(
            "sife",
            "lwe",
            {"sec_param": 64, "eta": 3, "lwe_n": 16, "bound_x": 20, "bound_y": 20},
        )
        assert wrapper.pp is not None
        assert wrapper.scheme_type == "sife"
        assert wrapper.has_label is False

    def test_mcfe_lwe_setup(self):
        wrapper = FESchemeWrapper(
            "mcfe",
            "lwe",
            {
                "sec_param": 64,
                "eta": 1,
                "n": 2,
                "lst_nid": ["nid_0", "nid_1"],
                "lwe_n": 16,
                "bound_x": 8,
                "bound_y": 8,
                "bound_u": 2,
                "label_modulus": 8,
            },
        )
        assert wrapper.pp is not None
        assert wrapper.has_label is True


# ── Encrypted Inference (SIFE, eta=feature_dim) ───────────────────


class TestEncryptedInference:
    """End-to-end: encrypt_features → compute_linear → scalar result."""

    @pytest.fixture()
    def sife_wrapper(self):
        return FESchemeWrapper(
            "sife",
            "lwe",
            {
                "sec_param": 64,
                "eta": 3,
                "lwe_n": 16,
                "bound_x": 1000,
                "bound_y": 1000,
            },
            precision=2,
            qmode="decimal",
        )

    def test_inner_product(self, sife_wrapper):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        # Expected: 1*4 + 2*5 + 3*6 = 32.0

        enc = encrypt_features(x, sife_wrapper)
        assert isinstance(enc, EncryptedFeatures)
        assert enc.dim == 3

        result = compute_linear(enc, sife_wrapper, w)
        assert abs(result - 32.0) < 0.1

    def test_negative_values(self, sife_wrapper):
        x = np.array([-1.0, 0.5, 2.0])
        w = np.array([2.0, -1.0, 1.0])
        # Expected: -1*2 + 0.5*(-1) + 2*1 = -0.5

        enc = encrypt_features(x, sife_wrapper)
        result = compute_linear(enc, sife_wrapper, w)
        assert abs(result - (-0.5)) < 0.1

    def test_encrypted_features_serialization(self, sife_wrapper):
        x = np.array([1.0, 2.0, 3.0])
        enc = encrypt_features(x, sife_wrapper)
        d = enc.to_dict()
        restored = EncryptedFeatures.from_dict(d)
        assert restored.dim == enc.dim
        assert restored.precision == enc.precision


class TestEncryptedInferenceRingLWE:
    """encrypt_features / compute_linear on SIFE Ring-LWE, whose encrypt takes a
    matrix of eta rows: the feature vector is its one column."""

    def test_inner_product(self):
        wrapper = FESchemeWrapper(
            "sife", "ring_lwe", {"sec_param": 64, "eta": 3, "ring_n": 32, "bound_x": 1000, "bound_y": 1000},
            precision=2, qmode="decimal",
        )
        enc = encrypt_features(np.array([1.0, -2.0, 3.0]), wrapper)
        assert enc.dim == 3 and enc.ciphertext["k"] == 1
        assert abs(compute_linear(enc, wrapper, np.array([4.0, 5.0, -6.0])) - (4.0 - 10.0 - 18.0)) < 0.1


# ── FL Aggregation (SIFE eta=1, single client) ────────────────────


class TestFLAggregationSIFE:
    """SIFE FL aggregation: single client, element-wise encryption."""

    @pytest.fixture()
    def sife_fl_wrapper(self):
        return FESchemeWrapper(
            "sife",
            "lwe",
            {
                "sec_param": 64,
                "eta": 1,
                "lwe_n": 16,
                "bound_x": 50,
                "bound_y": 50,
            },
            precision=1,
        )

    def test_encrypt_decrypt_gradient(self, sife_fl_wrapper):
        gradient = np.array([[1.0, 2.0], [3.0, 4.0]])

        enc = encrypt_gradient(gradient, sife_fl_wrapper)
        assert isinstance(enc, EncryptedGradient)
        assert enc.shape == (2, 2)

        # For SIFE, aggregation with weight=1 should recover original
        result = aggregate_gradients(
            {"nid_default": enc},
            sife_fl_wrapper,
            {"nid_default": [1]},
        )
        np.testing.assert_allclose(result, gradient, atol=0.2)


# ── FL Aggregation (MCFE eta=1, multi-client) ─────────────────────


class TestFLAggregationMCFE:
    """MCFE FL aggregation: multiple clients with labels."""

    @pytest.fixture()
    def mcfe_fl_wrapper(self):
        return FESchemeWrapper(
            "mcfe",
            "lwe",
            {
                "sec_param": 64,
                "eta": 1,
                "n": 2,
                "s": 1,
                "lst_nid": ["nid_0", "nid_1"],
                "lwe_n": 16,
                "bound_x": 50,
                "bound_y": 50,
                "bound_u": 2,
                "label_modulus": 8,
            },
            precision=1,
        )

    def test_mcfe_label_required(self, mcfe_fl_wrapper):
        grad = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="label"):
            encrypt_gradient(grad, mcfe_fl_wrapper, nid="nid_0")

    def test_mcfe_two_client_aggregation(self, mcfe_fl_wrapper):
        label = "round-1"

        g0 = np.array([2.0, 4.0])
        g1 = np.array([1.0, 3.0])

        enc0 = encrypt_gradient(g0, mcfe_fl_wrapper, nid="nid_0", label=label)
        enc1 = encrypt_gradient(g1, mcfe_fl_wrapper, nid="nid_1", label=label)

        # Weighted average: w0=1, w1=1 → result[i] = g0[i]*1 + g1[i]*1
        # Expected: [2+1, 4+3] = [3, 7]
        result = aggregate_gradients(
            {"nid_0": enc0, "nid_1": enc1},
            mcfe_fl_wrapper,
            {"nid_0": [1], "nid_1": [1]},
            label=label,
        )
        np.testing.assert_allclose(result, [3.0, 7.0], atol=0.2)


# ── Multi-party variants are rejected with a clear error ───────────


class TestUnsupportedMultiPartyVariants:
    """Regression (N12): used to fail with an opaque TypeError."""

    NIDS = ["nid_0", "nid_1"]

    def _encrypt_all(self, wrapper, label):
        return {
            nid: encrypt_gradient(np.array([1.0, 2.0]), wrapper, nid=nid, label=label)
            for nid in self.NIDS
        }

    def test_decentralized_rejected(self):
        wrapper = FESchemeWrapper(
            "mcfe",
            "lwe_decentralized",
            {
                "sec_param": 64,
                "eta": 1,
                "n": 2,
                "lst_nid": self.NIDS,
                "lwe_n": 16,
                "bound_x": 6,
                "bound_y": 6,
                "bound_u": 2,
            },
        )
        enc = self._encrypt_all(wrapper, "round-1")
        with pytest.raises(NotImplementedError, match="decentralized"):
            aggregate_gradients(
                enc, wrapper, {nid: 1 for nid in self.NIDS}, label="round-1"
            )
        with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
            wrapper.get_decryption_keys("sid_0", fusion_weight={nid: [1] for nid in self.NIDS})

    def test_threshold_rejected(self):
        wrapper = FESchemeWrapper(
            "mcfe",
            "ddh_threshold",
            {
                "sec_param": 128,
                "eta": 1,
                "n": 2,
                "s": 2,
                "t": 2,
                "lst_nid": self.NIDS,
                "lst_sid": ["sid_0", "sid_1"],
            },
        )
        enc = self._encrypt_all(wrapper, "round-1")
        with pytest.raises(NotImplementedError, match="threshold"):
            aggregate_gradients(
                enc, wrapper, {nid: 1 for nid in self.NIDS}, label="round-1"
            )


# ── EncryptedGradient serialization ───────────────────────────────


class TestEncryptedGradientSerialization:
    def test_round_trip(self):
        shape = (2, 3)
        ary = np.empty(shape, dtype=object)
        for idx, _ in np.ndenumerate(ary):
            ary[idx] = {"ct0": "fake", "ct1": "fake"}

        enc = EncryptedGradient(
            ciphertexts=[ary],
            shape=shape,
            precision=3,
            mode="decimal",
            nid="nid_0",
        )
        d = enc.to_dict()
        restored = EncryptedGradient.from_dict(d)
        assert restored.shape == shape
        assert restored.precision == 3
        assert restored.nid == "nid_0"
        assert restored.ciphertexts[0].shape == shape


@pytest.mark.parametrize("variant", ["ddh", "damgard", "lwe"])
def test_mife_refused_for_gradient_aggregation(variant):
    """MIFE has no labels: one key would let the aggregator combine ciphertexts
    of different elements of a client and learn the differences of its update."""
    import numpy as _np

    from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    cfg = {"n": 2, "lst_nid": ["nid_0", "nid_1"], "eta": 1, "sec_param": 128, "lwe_n": 8,
           "bound_x": 4, "bound_y": 4, "bound": 4}
    wrapper = FESchemeWrapper("mife", variant, cfg)
    with pytest.raises(NotImplementedError, match="MCFE"):
        encrypt_gradient(_np.zeros(2), wrapper, nid="nid_0")
    with pytest.raises(NotImplementedError, match="MCFE"):
        aggregate_gradients({}, wrapper, {"nid_0": 1, "nid_1": 1})


# ── single-trust-domain policy (regressions of the pre-PR review) ──


def _mcfe_wrapper(precision=1):
    return FESchemeWrapper(
        "mcfe", "ddh", {"sec_param": 128, "eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"]},
        precision=precision,
    )


def _sife_wrapper():
    return FESchemeWrapper(
        "sife", "lwe", {"sec_param": 64, "eta": 1, "lwe_n": 16, "bound_x": 50, "bound_y": 50},
        precision=1,
    )


def test_sife_aggregation_refuses_several_clients():
    """It returned the first client's result and ignored the others."""
    wrapper = _sife_wrapper()
    enc = {nid: encrypt_gradient(np.array([1.0]), wrapper, nid=nid) for nid in ("a", "b")}
    with pytest.raises(ValueError, match="one client"):
        aggregate_gradients(enc, wrapper, {"a": [1], "b": [1]})


def test_aggregation_needs_every_client_with_a_nonzero_weight():
    """The wrapper holds the master key: weights (1, 0) or a subset of the
    clients would decrypt a single client's gradient."""
    wrapper = _mcfe_wrapper()
    enc = {nid: encrypt_gradient(np.array([1.0, 2.0]), wrapper, nid=nid, label="r1") for nid in ("nid_0", "nid_1")}
    with pytest.raises(ValueError, match="non-zero"):
        aggregate_gradients(enc, wrapper, {"nid_0": 1, "nid_1": 0}, label="r1")
    with pytest.raises(ValueError, match="every enrolled client"):
        aggregate_gradients({"nid_0": enc["nid_0"]}, wrapper, {"nid_0": 1}, label="r1")
    out = aggregate_gradients(enc, wrapper, {"nid_0": 1, "nid_1": 0}, label="r1", allow_partial=True)
    np.testing.assert_allclose(out, [1.0, 2.0])


def test_labels_are_not_reused():
    wrapper = _mcfe_wrapper()
    encrypt_gradient(np.array([1.0]), wrapper, nid="nid_0", label="r1")
    with pytest.raises(ValueError, match="already"):
        encrypt_gradient(np.array([2.0]), wrapper, nid="nid_0", label="r1")
    encrypt_gradient(np.array([2.0]), wrapper, nid="nid_0", label="r2")


def test_one_aggregation_per_label():
    """Two weight vectors on one round's ciphertexts give two equations in the
    clients' gradients."""
    wrapper = _mcfe_wrapper()
    enc = {nid: encrypt_gradient(np.array([1.0]), wrapper, nid=nid, label="r1") for nid in ("nid_0", "nid_1")}
    first = aggregate_gradients(enc, wrapper, {"nid_0": 1, "nid_1": 1}, label="r1")
    again = aggregate_gradients(enc, wrapper, {"nid_0": 1, "nid_1": 1}, label="r1")
    np.testing.assert_allclose(first, again)
    with pytest.raises(ValueError, match="already aggregated"):
        aggregate_gradients(enc, wrapper, {"nid_0": 1, "nid_1": 2}, label="r1")


def test_fractional_weights_are_rescaled():
    """Weights 0.5 used to give 10^precision times the weighted sum."""
    wrapper = _mcfe_wrapper(precision=1)
    enc = {
        "nid_0": encrypt_gradient(np.array([2.0, 4.0]), wrapper, nid="nid_0", label="r1"),
        "nid_1": encrypt_gradient(np.array([1.0, 3.0]), wrapper, nid="nid_1", label="r1"),
    }
    out = aggregate_gradients(enc, wrapper, {"nid_0": 0.5, "nid_1": 1.5}, label="r1")
    np.testing.assert_allclose(out, [2.5, 6.5])


def test_fractional_weights_need_a_precision():
    wrapper = _mcfe_wrapper(precision=0)
    enc = {nid: encrypt_gradient(np.array([1.0]), wrapper, nid=nid, label="r1") for nid in ("nid_0", "nid_1")}
    with pytest.raises(ValueError, match="precision"):
        aggregate_gradients(enc, wrapper, {"nid_0": 0.5, "nid_1": 1}, label="r1")


@pytest.mark.parametrize("payload, match", [
    ({"shape": [10 ** 6, 10 ** 6], "data": {}}, "too many"),
    ({"shape": [2], "data": {"(5,)": "x", "(0,)": "x"}}, "outside"),
    ({"shape": [2], "data": {"(0,)": "x"}}, "every element"),
    ({"shape": [-1], "data": {}}, "shape"),
])
def test_deserialised_gradients_are_bounded(payload, match):
    d = {"nid": "a", "shape": payload["shape"], "precision": 1, "mode": "decimal", "ciphertexts": [payload]}
    with pytest.raises(ValueError, match=match):
        EncryptedGradient.from_dict(d)


def test_fractional_weights_must_fit_the_precision():
    """(0.5, 0.25) at precision 1 was rounded to (0.5, 0.2 or 0.3) silently."""
    wrapper = _mcfe_wrapper(precision=1)
    enc = {nid: encrypt_gradient(np.array([1.0]), wrapper, nid=nid, label="r1") for nid in ("nid_0", "nid_1")}
    with pytest.raises(ValueError, match="precision"):
        aggregate_gradients(enc, wrapper, {"nid_0": 0.5, "nid_1": 0.25}, label="r1")


# ── packed (Ring-LWE MCFE) gradient serialization ──────────────────


def test_packed_ring_lwe_gradients_round_trip_through_json(tmp_path, monkeypatch):
    """MCFE-Ring-LWE packs a layer into ring_n-coefficient chunks; to_dict used to raise KeyError."""
    import json

    monkeypatch.chdir(tmp_path)  # key generation writes config/ to the working directory
    wrapper = FESchemeWrapper(
        "mcfe", "ring_lwe",
        {"eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"], "ring_n": 16, "sec_param": 64,
         "bound_x": 50, "bound_y": 2},
        precision=1,
    )
    rng = np.random.default_rng(0)
    grads = {nid: rng.integers(-25, 26, size=(3, 7)) / 10 for nid in ("nid_0", "nid_1")}  # 2 chunks
    encrypted = {nid: encrypt_gradient(g, wrapper, nid=nid, label="r1") for nid, g in grads.items()}
    sent = {nid: json.loads(json.dumps(enc.to_dict())) for nid, enc in encrypted.items()}
    received = {nid: EncryptedGradient.from_dict(d) for nid, d in sent.items()}

    assert sent["nid_0"]["ciphertexts"][0]["packed"]["shape"] == [3, 7]
    assert [len(enc.ciphertexts[0]["chunks"]) for enc in received.values()] == [2, 2]
    for nid, enc in received.items():
        assert enc.ciphertexts == encrypted[nid].ciphertexts
        assert enc.shape == (3, 7)
    out = aggregate_gradients(received, wrapper, {"nid_0": 1, "nid_1": 1}, label="r1")
    np.testing.assert_allclose(out, grads["nid_0"] + grads["nid_1"])


@pytest.mark.parametrize("variant, extra", [
    ("ring_lwe_threshold", {"s": 3, "t": 2}),  # chunks also carry server tags
    ("ring_lwe_decentralized", {}),
])
def test_packed_gradients_of_ring_lwe_variants_round_trip(variant, extra, tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    wrapper = FESchemeWrapper(
        "mcfe", variant,
        dict({"eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"], "ring_n": 16, "sec_param": 64,
              "bound_x": 50, "bound_y": 2}, **extra),
        precision=1,
    )
    enc = encrypt_gradient(np.arange(20.0).reshape(4, 5) / 10, wrapper, nid="nid_0", label="r1")
    restored = EncryptedGradient.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert restored.ciphertexts == enc.ciphertexts


def _fake_chunk(k):
    return {"ct0": [["1"]], "ct1": ["2"], "k": k}


def _packed(shape, first, ks):
    return {"shape": shape, "first_chunk": first, "chunks": [_fake_chunk(k) for k in ks]}


def test_packed_layers_of_several_tensors_round_trip():
    import json

    layers = [_packed([2, 3], 0, [4, 2]), _packed([5], 2, [4, 1]), _packed([], 4, [1])]
    enc = EncryptedGradient(ciphertexts=layers, shape=(2, 3), precision=1, mode="decimal", nid="a")
    restored = EncryptedGradient.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert restored.ciphertexts == layers


def test_binary_chunks_are_written_as_base64():
    """Chunk fields holding bytes (binary ciphertexts) are base64 strings named
    in the layer's ``base64`` list, decoded strictly by from_dict."""
    import json

    layer = {"shape": [3], "first_chunk": 0, "label_hash": 2,
             "chunks": [{"ct": b"\x00\x01\xff", "k": 2, "tags": {"s0": "ab"}}, {"ct": bytearray(b"\x07"), "k": 1}]}
    enc = EncryptedGradient(ciphertexts=[layer], shape=(3,), precision=1, mode="decimal", nid="a")
    sent = json.loads(json.dumps(enc.to_dict()))
    assert sent["ciphertexts"][0]["packed"]["base64"] == ["ct"]
    assert sent["ciphertexts"][0]["packed"]["chunks"][0] == {"ct": "AAH/", "k": 2, "tags": {"s0": "ab"}}
    restored = EncryptedGradient.from_dict(sent)
    assert restored.ciphertexts == [dict(layer, chunks=[layer["chunks"][0], {"ct": b"\x07", "k": 1}])]
    assert isinstance(restored.ciphertexts[0]["chunks"][1]["ct"], bytes)
    # a chunk without the binary field of the others
    mixed = dict(layer, chunks=[layer["chunks"][0], {"ct": "x", "k": 1}])
    with pytest.raises(TypeError, match="binary fields"):
        EncryptedGradient(ciphertexts=[mixed], shape=(3,), precision=1, mode="decimal", nid="a").to_dict()
    # not base64, a missing field, a bad field list
    packed = sent["ciphertexts"][0]["packed"]
    for bad, match in (
        (dict(packed, chunks=[dict(packed["chunks"][0], ct="AA=H"), packed["chunks"][1]]), "not base64"),
        (dict(packed, chunks=[dict(packed["chunks"][0], ct="AAé/"), packed["chunks"][1]]), "not base64"),
        (dict(packed, chunks=[{"k": 2}, packed["chunks"][1]]), "base64 string"),
        (dict(packed, base64=[]), "base64"),
        (dict(packed, base64=["k"]), "base64"),
        (dict(packed, base64=["ct", "ct"]), "base64"),
        (dict(packed, base64="ct"), "base64"),
    ):
        with pytest.raises(ValueError, match=match):
            EncryptedGradient.from_dict(dict(sent, ciphertexts=[{"packed": bad}]))


@pytest.mark.parametrize("layers, match", [
    ([{"packed": {"shape": [2], "first_chunk": 0}}], "packed"),
    ([{"packed": dict(_packed([2], 0, [2]), extra=1)}], "packed"),
    ([{"packed": _packed([2], 0, [2]), "shape": [2]}], "packed"),
    ([{"packed": [2, 0, []]}], "packed"),
    ([{"packed": _packed([-1], 0, [2])}], "shape"),
    ([{"packed": _packed([2.0], 0, [2])}], "shape"),
    ([{"packed": _packed([10 ** 6, 10 ** 6], 0, [2])}], "too many"),
    ([{"packed": _packed([2], -1, [2])}], "first chunk"),
    ([{"packed": _packed([2], True, [2])}], "first chunk"),
    ([{"packed": _packed([2], 1, [2])}], "first chunk"),
    ([{"packed": _packed([2], 0, [2])}, {"packed": _packed([2], 0, [2])}], "first chunk"),
    ([{"packed": dict(_packed([2], 0, []), chunks={"0": _fake_chunk(2)})}], "chunks"),
    ([{"packed": _packed([0], 0, [])}], "chunks"),
    ([{"packed": dict(_packed([2], 0, []), chunks=["x"])}], "chunk 0"),
    ([{"packed": dict(_packed([2], 0, []), chunks=[{"ct0": []}])}], "chunk 0"),
    ([{"packed": _packed([2], 0, [0, 2])}], "chunk 0"),
    ([{"packed": _packed([2], 0, ["2"])}], "chunk 0"),
    ([{"packed": _packed([2], 0, [True, True])}], "chunk 0"),
    ([{"packed": _packed([5], 0, [3])}], "cover"),
    ([{"packed": _packed([5], 0, [4, 4])}], "cover"),
    ([{"packed": _packed([5], 0, [2, 3])}], "width"),
    ([{"packed": _packed([4], 0, [4])}, {"packed": _packed([6], 1, [2, 4])}], "width"),
    ([{"packed": _packed([2], 0, [2])}, {"shape": [1], "data": {"(0,)": "x"}}], "mix"),
])
def test_deserialised_packed_layers_are_checked(layers, match):
    d = {"nid": "a", "shape": [2], "precision": 1, "mode": "decimal", "ciphertexts": layers}
    with pytest.raises(ValueError, match=match):
        EncryptedGradient.from_dict(d)
