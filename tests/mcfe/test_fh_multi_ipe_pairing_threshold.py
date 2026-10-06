import random

import numpy as np
import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 2,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "u_bound": 3,
        "label_modulus": 17,
        "pairing_group_param": "SS512",
    }


def test_entire_process(kg_config):
    label = "tmcfe-fhmultiipe"
    kg = ThresholdMCFEFHMultiIPEKeyGenerator(kg_config)
    kg.setup()

    dct_x = {
        nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(kg_config["eta"])]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(kg_config["eta"])]
        for nid in kg_config["lst_nid"]
    }
    expected = sum(
        x_i * y_i
        for nid in kg_config["lst_nid"]
        for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in kg_config["lst_nid"]:
        enc = ThresholdMCFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = enc.encrypt(dct_x[nid], label)

    credentials = {"fusion_weight": dct_y, "label": label}
    enrolled = random.sample(kg_config["lst_sid"], kg_config["t"])
    dct_prime = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        dec = ThresholdMCFEFHMultiIPE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        dct_prime[sid] = dec.share_decrypt(dct_ct, credentials, dk, enrolled)

    combiner = ThresholdMCFEFHMultiIPE(
        {"id": kg_config["lst_nid"][0], "keys": {"pp": pp, "sk": kg.get_private_keys(kg_config["lst_nid"][0])}}
    )
    computed = combiner.combine_decrypt(dct_prime)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = ThresholdMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "tmcfe-fh-multi-ipe-ndarray"
    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
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

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in cfg["lst_nid"]:
        enc = ThresholdMCFEFHMultiIPE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid], label=label)

    credentials = {"fusion_weight": dct_y, "label": label}
    enrolled = random.sample(cfg["lst_sid"], cfg["t"])
    dct_prime = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        dec = ThresholdMCFEFHMultiIPE(
            {"id": sid, "precision": precision, "keys": {"pp": pp, "dk": dk}}
        )
        dct_prime[sid] = dec.compute_lst_ndarray_ct(dct_ct, credentials, dk, enrolled)

    combiner = ThresholdMCFEFHMultiIPE(
        {
            "id": cfg["lst_nid"][0],
            "precision": precision,
            "keys": {"pp": pp, "sk": kg.get_private_keys(cfg["lst_nid"][0])},
        }
    )
    computed = combiner.decrypt_lst_ndarray_ct(dct_prime)

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)
