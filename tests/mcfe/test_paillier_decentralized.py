import logging
import random

import pytest
import numpy as np

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mcfe.paillier_decentralized import DecentralizedMCFEPaillier
from pyfe4ai.schemes.mcfe.paillier_decentralized import DecentralizedMCFEPaillierKeyGenerator

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bit_length": 64,
        "bound_x": 10,
        "bound_y": 10,
    }


def _setup_and_get_clients(kg_config):
    """Helper: run setup, return (kg, pp, dict of client Crypto instances)."""
    kg = DecentralizedMCFEPaillierKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {}
    for nid in kg_config["lst_nid"]:
        clients[nid] = DecentralizedMCFEPaillier(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
    return kg, pp, clients


class TestKeyGenerator:
    def test_setup_produces_valid_keys(self, kg_config):
        kg = DecentralizedMCFEPaillierKeyGenerator(kg_config)
        kg.setup()

        pp = kg.get_public_parameters()
        assert pp["n"] == 3
        assert pp["eta"] == 2
        assert "lst_nid" in pp

        for nid in kg_config["lst_nid"]:
            sk = kg.get_private_keys(nid)
            assert sk is not None
            assert "pk" in sk and len(sk["pk"]) == 2
            assert "s" in sk and len(sk["s"]) == 2
            assert "u" in sk and len(sk["u"]) == 2
            assert "v" in sk and len(sk["v"]) == 2 * 3  # eta * n

    def test_get_private_keys_invalid_nid(self, kg_config):
        kg = DecentralizedMCFEPaillierKeyGenerator(kg_config)
        kg.setup()
        assert kg.get_private_keys("nonexistent") is None

    def test_get_decryption_keys_returns_not_implemented(self, kg_config):
        kg = DecentralizedMCFEPaillierKeyGenerator(kg_config)
        kg.setup()
        assert kg.get_decryption_keys("sid") is NotImplementedError


class TestEntireProcess:
    def test_basic_inner_product(self, kg_config):
        """End-to-end: encrypt, derive DK shares, combine, decrypt."""
        _, pp, clients = _setup_and_get_clients(kg_config)

        label = "test-label-basic"
        dct_x = {
            nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(2)]
            for nid in kg_config["lst_nid"]
        }
        dct_y = {
            nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(2)]
            for nid in kg_config["lst_nid"]
        }
        expected = sum(
            x_i * y_i
            for nid in kg_config["lst_nid"]
            for x_i, y_i in zip(dct_x[nid], dct_y[nid])
        )

        # Encrypt
        dct_ct = {nid: clients[nid].encrypt(dct_x[nid], label) for nid in kg_config["lst_nid"]}

        # Each client derives DK share
        dct_dk_shares = {
            nid: clients[nid].derive_function_decryption_key_share(dct_y)
            for nid in kg_config["lst_nid"]
        }

        # Combiner aggregates and decrypts
        combiner = DecentralizedMCFEPaillier({"id": "sid_0", "keys": {"pp": pp}})
        dk = combiner.combine_function_decryption_key_share(dct_dk_shares)
        computed = combiner.decrypt(dct_ct, dk, dct_y, label)
        assert computed == expected

    def test_known_values(self, kg_config):
        """Test with deterministic vectors to verify correctness."""
        _, pp, clients = _setup_and_get_clients(kg_config)

        label = "known-label"
        dct_x = {"nid_0": [3, 4], "nid_1": [1, -2], "nid_2": [5, 0]}
        dct_y = {"nid_0": [2, 1], "nid_1": [3, 5], "nid_2": [-1, 2]}
        # <x,y> = 3*2+4*1 + 1*3+(-2)*5 + 5*(-1)+0*2 = 6+4+3-10-5+0 = -2
        expected = -2

        dct_ct = {nid: clients[nid].encrypt(dct_x[nid], label) for nid in kg_config["lst_nid"]}
        dct_dk_shares = {
            nid: clients[nid].derive_function_decryption_key_share(dct_y)
            for nid in kg_config["lst_nid"]
        }
        combiner = DecentralizedMCFEPaillier({"id": "sid_0", "keys": {"pp": pp}})
        dk = combiner.combine_function_decryption_key_share(dct_dk_shares)
        computed = combiner.decrypt(dct_ct, dk, dct_y, label)
        assert computed == expected

    def test_zero_inner_product(self, kg_config):
        """Inner product is exactly zero."""
        _, pp, clients = _setup_and_get_clients(kg_config)

        label = "zero-label"
        dct_x = {"nid_0": [1, 0], "nid_1": [0, 0], "nid_2": [0, 0]}
        dct_y = {"nid_0": [0, 5], "nid_1": [3, 7], "nid_2": [9, 1]}
        expected = 0

        dct_ct = {nid: clients[nid].encrypt(dct_x[nid], label) for nid in kg_config["lst_nid"]}
        dct_dk_shares = {
            nid: clients[nid].derive_function_decryption_key_share(dct_y)
            for nid in kg_config["lst_nid"]
        }
        combiner = DecentralizedMCFEPaillier({"id": "sid_0", "keys": {"pp": pp}})
        dk = combiner.combine_function_decryption_key_share(dct_dk_shares)
        computed = combiner.decrypt(dct_ct, dk, dct_y, label)
        assert computed == expected

    def test_different_labels_produce_different_ciphertexts(self, kg_config):
        """Same plaintext encrypted under different labels gives different ct."""
        _, _, clients = _setup_and_get_clients(kg_config)
        x = [1, 2]
        ct_a = clients["nid_0"].encrypt(x, "label-a")
        ct_b = clients["nid_0"].encrypt(x, "label-b")
        assert ct_a["ct1"] != ct_b["ct1"]

    def test_two_clients(self):
        """Minimal 2-client scenario."""
        cfg = {
            "sec_param": 64,
            "eta": 1,
            "n": 2,
            "s": 1,
            "lst_nid": ["alice", "bob"],
            "bit_length": 64,
            "bound_x": 50,
            "bound_y": 50,
        }
        _, pp, clients = _setup_and_get_clients(cfg)

        label = "2c-test"
        dct_x = {"alice": [7], "bob": [-3]}
        dct_y = {"alice": [4], "bob": [6]}
        expected = 7 * 4 + (-3) * 6  # 10

        dct_ct = {nid: clients[nid].encrypt(dct_x[nid], label) for nid in cfg["lst_nid"]}
        dct_dk_shares = {
            nid: clients[nid].derive_function_decryption_key_share(dct_y)
            for nid in cfg["lst_nid"]
        }
        combiner = DecentralizedMCFEPaillier({"id": "sid", "keys": {"pp": pp}})
        dk = combiner.combine_function_decryption_key_share(dct_dk_shares)
        computed = combiner.decrypt(dct_ct, dk, dct_y, label)
        assert computed == expected


class TestDKShareDerivation:
    def test_share_structure(self, kg_config):
        _, _, clients = _setup_and_get_clients(kg_config)
        dct_y = {"nid_0": [1, 2], "nid_1": [3, 4], "nid_2": [5, 6]}

        share = clients["nid_0"].derive_function_decryption_key_share(dct_y)
        assert "dk0" in share
        assert "dk1" in share

    def test_missing_own_nid_raises(self, kg_config):
        _, _, clients = _setup_and_get_clients(kg_config)
        dct_y = {"nid_1": [1, 2], "nid_2": [3, 4]}  # missing nid_0
        with pytest.raises(ValueError, match="not in fusion_weight"):
            clients["nid_0"].derive_function_decryption_key_share(dct_y)

    def test_wrong_eta_raises(self, kg_config):
        _, _, clients = _setup_and_get_clients(kg_config)
        dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [3]}  # eta=1 but config eta=2
        with pytest.raises(ValueError, match="length mismatch"):
            clients["nid_0"].derive_function_decryption_key_share(dct_y)

    def test_exceeds_bound_y_raises(self, kg_config):
        _, _, clients = _setup_and_get_clients(kg_config)
        dct_y = {"nid_0": [999, 0], "nid_1": [0, 0], "nid_2": [0, 0]}
        with pytest.raises(ValueError, match="exceeds configured bound_y"):
            clients["nid_0"].derive_function_decryption_key_share(dct_y)


class TestEncryptValidation:
    def test_wrong_plaintext_length_raises(self, kg_config):
        _, _, clients = _setup_and_get_clients(kg_config)
        with pytest.raises(ValueError, match="invalid size"):
            clients["nid_0"].encrypt([1, 2, 3], "label")  # eta=2, input length=3

    def test_exceeds_bound_x_raises(self, kg_config):
        _, _, clients = _setup_and_get_clients(kg_config)
        with pytest.raises(ValueError, match="exceeds configured bound_x"):
            clients["nid_0"].encrypt([999, 0], "label")


class TestNdarrayHelpers:
    def test_encrypt_decrypt_ndarray(self, kg_config):
        cfg = dict(kg_config)
        cfg["eta"] = 1
        cfg["bound_x"] = 50

        _, pp, clients = _setup_and_get_clients(cfg)

        precision = 1
        label = "ndarray-label"
        dct_x = {
            nid: [np.random.random_sample((2, 3))]
            for nid in cfg["lst_nid"]
        }
        dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}

        expected = []
        for layer_idx in range(len(dct_x[cfg["lst_nid"][0]])):
            expected.append(
                sum(
                    (
                        (dct_x[nid][layer_idx] * pow(10, precision)).astype(int)
                        / pow(10, precision)
                    )
                    * dct_y[nid][0]
                    for nid in cfg["lst_nid"]
                )
            )

        # Encrypt per client
        dct_ct = {}
        for nid in cfg["lst_nid"]:
            enc = DecentralizedMCFEPaillier(
                {"id": nid, "precision": precision, "keys": {"pp": pp, "sk": clients[nid].sk}}
            )
            dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid], label=label)

        # DK shares
        dct_dk_shares = {
            nid: clients[nid].derive_function_decryption_key_share(dct_y)
            for nid in cfg["lst_nid"]
        }

        combiner = DecentralizedMCFEPaillier(
            {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
        )
        dk = combiner.combine_function_decryption_key_share(dct_dk_shares)
        computed = combiner.compute_lst_ndarray_ct(
            dct_ct, dk=dk, fusion_weight=dct_y, label=label
        )

        for idx in range(len(expected)):
            assert np.allclose(computed[idx], expected[idx], atol=1e-2)
