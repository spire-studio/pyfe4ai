import random

import numpy as np
import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPEKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 2,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "u_bound": 3,
        "label_modulus": 17,
        "pairing_group_param": "SS512",
    }


def test_entire_process(kg_config):
    label = "dmcfe-fhmultiipe"
    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(kg_config)
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
    dct_dk_shares = {}
    for nid in kg_config["lst_nid"]:
        crypto_nid = DecentralizedMCFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)

    decryptor = DecentralizedMCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "dmcfe-fh-multi-ipe-ndarray"
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
    dct_dk_shares = {}
    for nid in cfg["lst_nid"]:
        crypto_nid = DecentralizedMCFEFHMultiIPE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)

    decryptor = DecentralizedMCFEFHMultiIPE(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def test_get_decryption_keys_raises_not_implemented(kg_config):
    """Regression (N12): the method used to *return* NotImplementedError."""
    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})
