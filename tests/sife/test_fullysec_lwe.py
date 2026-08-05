import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWE
from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 4,
        "lwe_n": 24,
        "bound_x": 10,
        "bound_y": 10,
    }


def test_key_generator(kg_config):
    kg = SIFEFullySecLWEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]
    assert kg.lwe_n == kg_config["lwe_n"]

    config_folder = os.path.join(
        "config", "authority", CryptoCONST.TYPE_SIFE_FULLYSEC_LWE
    )
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.k) == pp["k"]
    assert gp.digits(kg.q) == pp["q"]
    assert len(pp["A"]) == kg.m
    assert len(pp["A"][0]) == kg.lwe_n

    enc_key = kg.get_private_keys()
    assert "pk" in enc_key
    assert len(enc_key["pk"]) == kg_config["eta"]

    fusion_weight = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "sk_y" in dk
    assert len(dk["sk_y"]) == kg.m


def test_entire_process(kg_config):
    kg = SIFEFullySecLWEKeyGenerator(kg_config)
    kg.setup()

    x = [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(4)]
    y = [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(4)]
    expected_inner_prod = sum(a * b for a, b in zip(x, y))

    pp = kg.get_public_parameters()
    sk_nid = kg.get_private_keys()
    crypto_nid = SIFEFullySecLWE({"id": "nid_default", "keys": {"pp": pp, "sk": sk_nid}})
    ct_nid = crypto_nid.encrypt(x)
    assert isinstance(ct_nid, dict)
    assert "ct0" in ct_nid
    assert "ct1" in ct_nid

    crypto_sid = SIFEFullySecLWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    computed_inner_prod = crypto_sid.decrypt(ct_nid, dk, y)
    assert computed_inner_prod == expected_inner_prod


def test_entire_process_lst_ndarray():
    cfg = {
        "sec_param": 64,
        "eta": 1,
        "lwe_n": 20,
        "bound_x": 40,
        "bound_y": 3,
    }

    kg = SIFEFullySecLWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    fusion_weight = [2]
    lst_x = [np.random.random_sample((2, 3)), np.random.random_sample((2, 2))]
    expected = [
        ((ary * pow(10, precision)).astype(int) / pow(10, precision))
        * fusion_weight[0]
        for ary in lst_x
    ]

    pp = kg.get_public_parameters()
    crypto_nid = SIFEFullySecLWE(
        {
            "id": "nid_default",
            "precision": precision,
            "keys": {"pp": pp, "sk": kg.get_private_keys()},
        }
    )
    crypto_sid = SIFEFullySecLWE(
        {
            "id": "sid_0",
            "precision": precision,
            "keys": {"pp": pp},
        }
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": fusion_weight})
    ct_nid = crypto_nid.encrypt_lst_ndarray(lst_x)
    computed = crypto_sid.compute_lst_ndarray_ct(
        ct_nid, dk=dk, fusion_weight=fusion_weight
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)
