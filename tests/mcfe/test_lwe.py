import logging
import random

import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mcfe.lwe import MCFELWE
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator

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
        "lwe_n": 16,
        "bound_x": 6,
        "bound_y": 6,
        "bound_u": 2,
        "label_modulus": 8,
    }


def test_entire_process(kg_config):
    kg = MCFELWEKeyGenerator(kg_config)
    kg.setup()

    label = "mcfe-lwe-label"
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

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in kg_config["lst_nid"]:
        crypto_nid = MCFELWE(
            {
                "id": nid,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)

    decryptor = MCFELWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = MCFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "mcfe-lwe-ndarray"
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
        crypto_nid = MCFELWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)

    decryptor = MCFELWE(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


@pytest.mark.parametrize("label_modulus", [2, 3, 8, 17])
def test_label_scalar_is_never_zero(label_modulus):
    """Regression (N4): ~1/label_modulus of all labels used to map to 0.

    A zero label scalar removes the per-client mask u * l entirely, so the
    holder of any functional key could decode every client's plaintext.
    """
    import gmpy2 as gp

    from pyfe4ai.utils.lwe_utils import label_scalar_from_hash

    scalars = {
        int(label_scalar_from_hash(gp.mpz(h), label_modulus))
        for h in range(10 * label_modulus)
    }
    assert 0 not in scalars
    assert len(scalars) == label_modulus - 1
    assert all(abs(s) <= label_modulus // 2 for s in scalars)


def test_labels_that_used_to_map_to_zero_still_decrypt(kg_config):
    import gmpy2 as gp

    from pyfe4ai.utils.crypto_utils import md5_hash

    kg = MCFELWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    p = gp.mpz(pp["p"])
    # labels whose hash is 0 mod label_modulus were mapped to scalar 0 before
    labels = [
        lab
        for lab in ("round-{}".format(i) for i in range(500))
        if int(md5_hash(lab, p)) % kg_config["label_modulus"] == 0
    ][:3]
    assert labels

    lst_nid = kg_config["lst_nid"]
    dct_y = {nid: [1, 2] for nid in lst_nid}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    decryptor = MCFELWE({"id": "sid_0", "keys": {"pp": pp}})
    for label in labels:
        dct_x = {nid: [random.randint(-6, 6) for _ in range(2)] for nid in lst_nid}
        dct_ct = {
            nid: MCFELWE(
                {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
            ).encrypt(dct_x[nid], label)
            for nid in lst_nid
        }
        expected = sum(
            x * y for nid in lst_nid for x, y in zip(dct_x[nid], dct_y[nid])
        )
        assert decryptor.decrypt(dct_ct, dk, dct_y, label) == expected
