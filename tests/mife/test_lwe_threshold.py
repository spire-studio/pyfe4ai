import os
import random
import logging

import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 1,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
        "lwe_n": 16,
        "bound_x": 6,
        "bound_y": 6,
    }


def test_key_generator(kg_config):
    kg = ThresholdMIFELWEKeyGenerator(kg_config)
    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.t == kg_config["t"]
    assert kg.eta == kg_config["eta"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_TMIFE_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["t"] == kg_config["t"]
    assert pp["s"] == kg_config["s"]
    assert pp["lst_sid"] == kg_config["lst_sid"]

    sk_nid = kg.get_private_keys("nid_0")
    assert "pk" in sk_nid

    fusion_weight = {nid: [1] for nid in kg_config["lst_nid"]}
    dk_sid = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": fusion_weight})
    assert "sk_y" in dk_sid
    assert "sid_index" in dk_sid


def test_entire_process(kg_config):
    kg = ThresholdMIFELWEKeyGenerator(kg_config)
    kg.setup()

    lst_nid = kg_config["lst_nid"]
    lst_sid = kg_config["lst_sid"]
    threshold = kg_config["t"]

    dct_x = {
        nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"])]
        for nid in lst_nid
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"])]
        for nid in lst_nid
    }
    expected = sum(dct_x[nid][0] * dct_y[nid][0] for nid in lst_nid)

    pp = kg.get_public_parameters()
    dct_ct = {}
    encryptors = {}
    for nid in lst_nid:
        enc = ThresholdMIFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = enc.encrypt(dct_x[nid])
        encryptors[nid] = enc

    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dct_ct_prime = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials={"fusion_weight": dct_y})
        dec = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk_sid}})
        dct_ct_prime[sid] = dec.share_decrypt(dct_ct, dct_y, dk_sid, lst_sid_enrolled)

    computed = encryptors[lst_nid[0]].combine_decrypt(dct_ct_prime)
    assert abs(computed - expected) < 1e-6


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["bound_x"] = 50

    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    lst_nid = cfg["lst_nid"]
    lst_sid = cfg["lst_sid"]
    threshold = cfg["t"]

    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
        for nid in lst_nid
    }
    dct_y = {nid: [random.randint(1, 3)] for nid in lst_nid}

    expected = []
    for layer_idx in range(len(dct_x[lst_nid[0]])):
        expected.append(
            sum(
                (
                    (dct_x[nid][layer_idx] * pow(10, precision)).astype(int)
                    / pow(10, precision)
                )
                * dct_y[nid][0]
                for nid in lst_nid
            )
        )

    pp = kg.get_public_parameters()
    dct_ct = {}
    encryptors = {}
    for nid in lst_nid:
        enc = ThresholdMIFELWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid])
        encryptors[nid] = enc

    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dct_ct_prime = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials={"fusion_weight": dct_y})
        dec = ThresholdMIFELWE(
            {"id": sid, "precision": precision, "keys": {"pp": pp, "dk": dk_sid}}
        )
        dct_ct_prime[sid] = dec.compute_lst_ndarray_ct(
            dct_ct, dct_y, dk_sid, lst_sid_enrolled
        )

    computed = encryptors[lst_nid[0]].decrypt_lst_ndarray_ct(dct_ct_prime)
    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)
