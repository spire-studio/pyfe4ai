import random

import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.mife.fh_ipe_pairing import MIFEFHIPE
from pyfe4ai.schemes.mife.fh_ipe_pairing import MIFEFHIPEKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 4,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 6,
        "bound_y": 6,
        "pairing_group_param": "SS512",
    }


def test_entire_process(kg_config):
    kg = MIFEFHIPEKeyGenerator(kg_config)
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
        encryptor = MIFEFHIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid])

    decryptor = MIFEFHIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y)
    assert computed == expected
