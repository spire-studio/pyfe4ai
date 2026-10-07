"""Tests for utils.ml_adapter — ML framework adapter layer."""

from __future__ import annotations

import numpy as np
import pytest

from pyfe4ai.utils.ml_adapter import (
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
