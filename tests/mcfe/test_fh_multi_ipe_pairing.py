import random

import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairing
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairingKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 3,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "u_bound": 3,
        "label_modulus": 17,
        "pairing_group_param": GROUP,
    }


def test_entire_process(kg_config):
    label = "mcfe-fh-multi-ipe"
    kg = MCFEFHMultiIPEPairingKeyGenerator(kg_config)
    kg.setup()

    dct_x = {
        nid: [
            random.randint(-kg_config["bound_x"], kg_config["bound_x"])
            for _ in range(kg_config["eta"])
        ]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [
            random.randint(-kg_config["bound_y"], kg_config["bound_y"])
            for _ in range(kg_config["eta"])
        ]
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
        encryptor = MCFEFHMultiIPEPairing(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid], label)

    decryptor = MCFEFHMultiIPEPairing({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = MCFEFHMultiIPEPairingKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "mcfe-fh-multi-ipe-ndarray"
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
                    np.rint(dct_x[nid][layer_idx] * pow(10, precision))
                    / pow(10, precision)
                )
                * dct_y[nid][0]
                for nid in cfg["lst_nid"]
            )
        )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in cfg["lst_nid"]:
        crypto_nid = MCFEFHMultiIPEPairing(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)

    decryptor = MCFEFHMultiIPEPairing(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def test_symmetric_group_rejected(kg_config):
    pytest.importorskip("charm", reason="SS512 is a charm-crypto group")
    from pyfe4ai.core.exceptions import FEValidationError

    with pytest.raises(FEValidationError):
        MCFEFHMultiIPEPairingKeyGenerator(dict(kg_config, pairing_group_param="SS512"))


def test_key_is_function_hiding_and_tag_bound(kg_config):
    """Keys are randomised (two keys for one y differ) and only decrypt the right tag."""
    cfg = dict(kg_config, eta=1)
    kg = MCFEFHMultiIPEPairingKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_x = {nid: [2] for nid in cfg["lst_nid"]}
    dct_y = {nid: [3] for nid in cfg["lst_nid"]}
    dk1 = kg.get_decryption_keys("sid", credentials={"fusion_weight": dct_y})
    dk2 = kg.get_decryption_keys("sid", credentials={"fusion_weight": dct_y})
    assert dk1["dk"]["nid_0"]["k"] != dk2["dk"]["nid_0"]["k"]
    dct_ct = {
        nid: MCFEFHMultiIPEPairing({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(dct_x[nid], "tag-a")
        for nid in cfg["lst_nid"]
    }
    dec = MCFEFHMultiIPEPairing({"id": "sid", "keys": {"pp": pp}})
    assert dec.decrypt(dct_ct, dk1, dct_y, "tag-a") == 18
    assert dec.decrypt(dct_ct, dk2, dct_y, "tag-a") == 18
    # mixing a ciphertext of another tag breaks the zero-sum mask
    other = MCFEFHMultiIPEPairing({"id": "nid_1", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_1")}}).encrypt([2], "tag-b")
    assert dec.decrypt(dict(dct_ct, nid_1=other), dk1, dct_y, "tag-a") is None
