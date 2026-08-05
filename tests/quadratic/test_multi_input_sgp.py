"""Correctness tests for multi-input secret-key quadratic FE (MI-SGP).

Requires charm-crypto pairing backend.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.quadratic.multi_input_sgp import (
    MultiInputQuadraticSGP,
    MultiInputQuadraticSGPKeyGenerator,
)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "d": 3,
        "n": 2,
        "lst_nid": ["client_0", "client_1"],
        "s": 1,
        "bound": 5,
        "pairing_group_param": "SS512",
    }


def _quadratic_form(f, x, y):
    return sum(x[i] * f[i][j] * y[j] for i in range(len(x)) for j in range(len(y)))


class TestKeyGenerator:
    def test_setup(self, kg_config):
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()
        pp = kg.get_public_parameters()
        assert pp["d"] == 3
        assert pp["n"] == 2
        assert len(pp["lst_nid"]) == 2

    def test_private_keys_per_client(self, kg_config):
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()
        for nid in kg_config["lst_nid"]:
            sk = kg.get_private_keys(nid)
            assert sk is not None
            assert len(sk["s"]) == 3
            assert len(sk["t"]) == 3

    def test_invalid_nid_returns_none(self, kg_config):
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()
        assert kg.get_private_keys("nonexistent") is None


class TestEntireProcess:
    def test_two_client_aggregation(self, kg_config):
        """sum_i x_i^T F_i y_i across two clients."""
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()

        d = kg_config["d"]
        bound = kg_config["bound"]

        x0 = [random.randint(-bound, bound) for _ in range(d)]
        y0 = [random.randint(-bound, bound) for _ in range(d)]
        f0 = [[random.randint(-bound, bound) for _ in range(d)] for _ in range(d)]

        x1 = [random.randint(-bound, bound) for _ in range(d)]
        y1 = [random.randint(-bound, bound) for _ in range(d)]
        f1 = [[random.randint(-bound, bound) for _ in range(d)] for _ in range(d)]

        expected = _quadratic_form(f0, x0, y0) + _quadratic_form(f1, x1, y1)

        pp = kg.get_public_parameters()

        # Each client encrypts
        enc0 = MultiInputQuadraticSGP(
            {"id": "client_0", "keys": {"pp": pp, "sk": kg.get_private_keys("client_0")}}
        )
        enc1 = MultiInputQuadraticSGP(
            {"id": "client_1", "keys": {"pp": pp, "sk": kg.get_private_keys("client_1")}}
        )
        ct0 = enc0.encrypt({"x": x0, "y": y0})
        ct1 = enc1.encrypt({"x": x1, "y": y1})

        # Decryptor
        dk = kg.get_decryption_keys(
            "sid_0",
            credentials={"function_matrices": {"client_0": f0, "client_1": f1}},
        )
        decryptor = MultiInputQuadraticSGP(
            {"id": "sid_0", "keys": {"pp": pp}}
        )
        computed = decryptor.decrypt({"client_0": ct0, "client_1": ct1}, dk)
        assert computed == expected

    def test_known_values(self, kg_config):
        """Deterministic test: x0=[1,0,0], y0=[1,0,0], F0=I → 1;
        x1=[0,1,0], y1=[0,1,0], F1=I → 1; total = 2."""
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()

        identity_3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        x0, y0 = [1, 0, 0], [1, 0, 0]
        x1, y1 = [0, 1, 0], [0, 1, 0]
        expected = 1 + 1  # = 2

        pp = kg.get_public_parameters()
        enc0 = MultiInputQuadraticSGP(
            {"id": "client_0", "keys": {"pp": pp, "sk": kg.get_private_keys("client_0")}}
        )
        enc1 = MultiInputQuadraticSGP(
            {"id": "client_1", "keys": {"pp": pp, "sk": kg.get_private_keys("client_1")}}
        )
        ct0 = enc0.encrypt({"x": x0, "y": y0})
        ct1 = enc1.encrypt({"x": x1, "y": y1})

        dk = kg.get_decryption_keys(
            "sid_0",
            credentials={"function_matrices": {"client_0": identity_3, "client_1": identity_3}},
        )
        decryptor = MultiInputQuadraticSGP({"id": "sid_0", "keys": {"pp": pp}})
        assert decryptor.decrypt({"client_0": ct0, "client_1": ct1}, dk) == expected


class TestZeroResult:
    def test_zero_quadratic_form(self, kg_config):
        """All-zero matrices → result is 0."""
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()

        d = kg_config["d"]
        zero_f = [[0] * d for _ in range(d)]
        x0, y0, x1, y1 = [1, 2, 3], [4, 5, -1], [3, 0, 0], [0, 1, 0]

        pp = kg.get_public_parameters()
        enc0 = MultiInputQuadraticSGP(
            {"id": "client_0", "keys": {"pp": pp, "sk": kg.get_private_keys("client_0")}}
        )
        enc1 = MultiInputQuadraticSGP(
            {"id": "client_1", "keys": {"pp": pp, "sk": kg.get_private_keys("client_1")}}
        )
        ct0 = enc0.encrypt({"x": x0, "y": y0})
        ct1 = enc1.encrypt({"x": x1, "y": y1})

        dk = kg.get_decryption_keys(
            "sid_0",
            credentials={"function_matrices": {"client_0": zero_f, "client_1": zero_f}},
        )
        decryptor = MultiInputQuadraticSGP({"id": "sid_0", "keys": {"pp": pp}})
        assert decryptor.decrypt({"client_0": ct0, "client_1": ct1}, dk) == 0


class TestNegativeResult:
    def test_negative_quadratic_form(self, kg_config):
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()

        # x0^T F0 y0 = 1*(-3)*1 = -3; x1^T F1 y1 = 1*(-2)*1 = -2; total = -5
        x0, y0, x1, y1 = [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]
        f0 = [[-3, 0, 0], [0, 0, 0], [0, 0, 0]]
        f1 = [[-2, 0, 0], [0, 0, 0], [0, 0, 0]]
        expected = -5

        pp = kg.get_public_parameters()
        enc0 = MultiInputQuadraticSGP(
            {"id": "client_0", "keys": {"pp": pp, "sk": kg.get_private_keys("client_0")}}
        )
        enc1 = MultiInputQuadraticSGP(
            {"id": "client_1", "keys": {"pp": pp, "sk": kg.get_private_keys("client_1")}}
        )
        ct0 = enc0.encrypt({"x": x0, "y": y0})
        ct1 = enc1.encrypt({"x": x1, "y": y1})

        dk = kg.get_decryption_keys(
            "sid_0",
            credentials={"function_matrices": {"client_0": f0, "client_1": f1}},
        )
        decryptor = MultiInputQuadraticSGP({"id": "sid_0", "keys": {"pp": pp}})
        assert decryptor.decrypt({"client_0": ct0, "client_1": ct1}, dk) == expected


class TestValidation:
    def test_missing_clients_in_function_matrices(self, kg_config):
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()
        with pytest.raises(ValueError, match="cover all clients"):
            kg.get_decryption_keys(
                "sid_0",
                credentials={"function_matrices": {"client_0": [[0] * 3] * 3}},
            )

    def test_bound_exceeded(self, kg_config):
        kg = MultiInputQuadraticSGPKeyGenerator(kg_config)
        kg.setup()
        pp = kg.get_public_parameters()
        enc = MultiInputQuadraticSGP(
            {"id": "client_0", "keys": {"pp": pp, "sk": kg.get_private_keys("client_0")}}
        )
        with pytest.raises(ValueError, match="exceeds configured bound"):
            enc.encrypt({"x": [999, 0, 0], "y": [0, 0, 0]})
