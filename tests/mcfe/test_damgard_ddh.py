import logging
import random

import pytest
import numpy as np

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgard
from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgardKeyGenerator

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
        "modulus_length": 64,
        "bound": 10,
    }


def test_entire_process(kg_config):
    kg = MCFEDamgardKeyGenerator(kg_config)
    kg.setup()

    label = "mcfe-damgard-label"
    dct_x = {
        nid: [random.randint(-kg_config["bound"], kg_config["bound"]) for _ in range(2)]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound"], kg_config["bound"]) for _ in range(2)]
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
        crypto_nid = MCFEDamgard(
            {
                "id": nid,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)

    decryptor = MCFEDamgard({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound"] = 50

    kg = MCFEDamgardKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "mcfe-damgard-ndarray"
    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
        for nid in cfg["lst_nid"]
    }
    dct_y = {"nid_0": [2], "nid_1": [1], "nid_2": [-1]}

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
        crypto_nid = MCFEDamgard(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)

    decryptor = MCFEDamgard(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def _round(cfg, dct_x, dct_y, label):
    kg = MCFEDamgardKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_ct = {
        nid: MCFEDamgard(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        ).encrypt(dct_x[nid], label)
        for nid in dct_x
    }
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    return kg, pp, dct_ct, dk


@pytest.mark.parametrize("mddh_k", [1, 2, 3])
def test_mddh_k_instantiations(kg_config, mddh_k):
    """ABMPSW20 Fig. 2 for D_k-MDDH; k = 1 is CDGPP18 (DDH)."""
    cfg = dict(kg_config, mddh_k=mddh_k)
    b = cfg["bound"]
    dct_x = {nid: [b, -b] for nid in cfg["lst_nid"]}
    dct_y = {nid: [b, b] for nid in cfg["lst_nid"]}
    kg, pp, dct_ct, dk = _round(cfg, dct_x, dct_y, "mddh-{}".format(mddh_k))
    assert set(dk) == {"d"} and len(dk["d"]) == mddh_k + 1
    assert all(len(s_j) == mddh_k + 1 for s_j in kg.get_private_keys("nid_0")["s"])
    decryptor = MCFEDamgard({"id": "sid_0", "keys": {"pp": pp}})
    assert decryptor.decrypt(dct_ct, dk, dct_y, "mddh-{}".format(mddh_k)) == 0
    dct_y = {nid: [b, -b] for nid in cfg["lst_nid"]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    expected = 2 * b * b * len(cfg["lst_nid"])
    assert decryptor.decrypt(dct_ct, dk, dct_y, "mddh-{}".format(mddh_k)) == expected


def test_wrong_label_fails(kg_config):
    dct_x = {nid: [1, 2] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1, 1] for nid in kg_config["lst_nid"]}
    _, pp, dct_ct, dk = _round(kg_config, dct_x, dct_y, "label-a")
    decryptor = MCFEDamgard({"id": "sid_0", "keys": {"pp": pp}})
    assert decryptor.decrypt(dct_ct, dk, dct_y, "label-a") == 9
    assert decryptor.decrypt(dct_ct, dk, dct_y, "label-b") is None


def test_ndarray_elements_use_distinct_labels(kg_config):
    cfg = dict(kg_config, eta=1)
    kg = MCFEDamgardKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    client = MCFEDamgard({"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    ct = client.encrypt_lst_ndarray([np.zeros((2, 2))], label="round")
    assert len({c["c"][0] for c in ct[0].flat}) == 4
