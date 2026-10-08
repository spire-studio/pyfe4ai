"""BMMS21 identity-based MCFE over Ring-LWE (keys bound to a label)."""

import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "ring_n": 16,
        "bound_x": 2,
        "bound_y": 2,
    }


def _encrypt_all(kg, pp, dct_x, label):
    return {
        nid: MCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(dct_x[nid], label)
        for nid in dct_x
    }


def _random_round(cfg, cols=4):
    dct_x = {
        nid: [[random.randint(-cfg["bound_x"], cfg["bound_x"]) for _ in range(cols)] for _ in range(cfg["eta"])]
        for nid in cfg["lst_nid"]
    }
    dct_y = {nid: [random.randint(-cfg["bound_y"], cfg["bound_y"]) for _ in range(cfg["eta"])] for nid in cfg["lst_nid"]}
    expected = [
        sum(dct_x[nid][row][col] * dct_y[nid][row] for nid in cfg["lst_nid"] for row in range(cfg["eta"]))
        for col in range(cols)
    ]
    return dct_x, dct_y, expected


def test_key_generator(kg_config):
    kg = MCFERingLWEKeyGenerator(kg_config)
    assert os.path.exists(os.path.join("config", "authority", CryptoCONST.TYPE_MCFE_RING_LWE, "param.json"))
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["eta"] == kg_config["eta"] and pp["ring_n"] == kg_config["ring_n"]
    sk = kg.get_private_keys("nid_0")
    assert set(sk) == {"pk", "u", "u_prime"}
    y = {"nid_0": [1, -1], "nid_1": [0, 1], "nid_2": [1, 1]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y, "label": "l"})
    assert set(dk) == {"sk_y", "z", "z_chunks", "label"} and dk["z_chunks"] == []
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})  # no label


def test_entire_process(kg_config):
    kg = MCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    for label in ["round-1", "round-2"]:
        dct_x, dct_y, expected = _random_round(kg_config)
        dct_ct = _encrypt_all(kg, pp, dct_x, label)
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y, "label": label})
        assert MCFERingLWE({"id": "sid_0", "keys": {"pp": pp}}).decrypt(dct_ct, dk, dct_y, label) == expected


def test_key_only_decrypts_its_label(kg_config):
    kg = MCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_x, dct_y, expected = _random_round(kg_config)
    dct_ct = _encrypt_all(kg, pp, dct_x, "label-b")
    dk_a = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y, "label": "label-a"})
    decryptor = MCFERingLWE({"id": "sid_0", "keys": {"pp": pp}})
    with pytest.raises(FESchemeError):
        decryptor.decrypt(dct_ct, dk_a, dct_y, "label-b")
    # forcing the key onto the other label's ciphertexts gives garbage
    forged = dict(dk_a, label="label-b")
    assert decryptor.decrypt(dct_ct, forged, dct_y, "label-b") != expected


def test_slot_partial_is_padded(kg_config):
    """A slot's Dec1 output encodes x_i + u_i + H(u'_i, l), not x_i."""
    from pyfe4ai.utils.lwe_utils import decode_scaled
    from pyfe4ai.utils.ring_lwe_utils import mat_vec_mul, poly_add, poly_mul, poly_neg, transpose
    import gmpy2 as gp

    kg = MCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_x = {nid: [[1, 1, 1, 1], [0, 0, 0, 0]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1, 0] for nid in kg_config["lst_nid"]}
    dct_ct = _encrypt_all(kg, pp, dct_x, "lbl")
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y, "label": "lbl"})
    q = gp.mpz(pp["q"])
    ct = dct_ct["nid_0"]
    lhs = mat_vec_mul(transpose([[gp.mpz(v) for v in r] for r in ct["ct0"]]), [gp.mpz(1), gp.mpz(0)], q)
    rhs = poly_mul([gp.mpz(v) for v in ct["ct1"]], [gp.mpz(v) for v in dk["sk_y"]["nid_0"]], q)
    slot = [decode_scaled(v, q // gp.mpz(pp["p"]), q, 10 ** 9) for v in poly_add(lhs, poly_neg(rhs, q), q)[:4]]
    assert slot != [1, 1, 1, 1]


def test_entire_process_lst_ndarray():
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"], "ring_n": 16,
           "bound_x": 30, "bound_y": 2}
    precision = 1
    label = "mcfe-ring-lwe-ndarray"
    kg = MCFERingLWEKeyGenerator(cfg)
    kg.setup()
    # 40 and 4 elements: three chunks of 16 coefficients in total
    dct_x = {nid: [np.random.random_sample((5, 8)), np.random.random_sample((2, 2))] for nid in cfg["lst_nid"]}
    dct_y = {"nid_0": [1], "nid_1": [-1]}
    expected = [
        sum(((dct_x[nid][l] * 10 ** precision).astype(int) / 10 ** precision) * dct_y[nid][0] for nid in cfg["lst_nid"])
        for l in range(2)
    ]
    pp = kg.get_public_parameters()
    dct_ct = {
        nid: MCFERingLWE({"id": nid, "precision": precision, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        .encrypt_lst_ndarray(dct_x[nid], label=label)
        for nid in cfg["lst_nid"]
    }
    decryptor = MCFERingLWE({"id": "sid_0", "precision": precision, "keys": {"pp": pp}})
    n_chunks = MCFERingLWE.key_chunks(dct_ct["nid_0"])
    assert n_chunks == 4  # ceil(40/16) + ceil(4/16)
    short = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y, "label": label, "n_chunks": 2})
    with pytest.raises(FEKeyError):
        decryptor.compute_lst_ndarray_ct(dct_ct, dk=short, fusion_weight=dct_y, label=label)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y, "label": label, "n_chunks": n_chunks})
    computed = decryptor.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y, label=label)
    for idx in range(2):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)
