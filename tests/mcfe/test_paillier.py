import logging
import random

import pytest
import numpy as np

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mcfe.paillier import MCFEPaillier
from pyfe4ai.schemes.mcfe.paillier import MCFEPaillierKeyGenerator

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


def test_entire_process(kg_config):
    kg = MCFEPaillierKeyGenerator(kg_config)
    kg.setup()

    label = "mcfe-paillier-label"
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
        crypto_nid = MCFEPaillier(
            {
                "id": nid,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)

    decryptor = MCFEPaillier({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = MCFEPaillierKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "mcfe-paillier-ndarray"
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
        crypto_nid = MCFEPaillier(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)

    decryptor = MCFEPaillier(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def _round(kg_config, label, dct_x, dct_y):
    kg = MCFEPaillierKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_ct = {
        nid: MCFEPaillier(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        ).encrypt(dct_x[nid], label)
        for nid in dct_x
    }
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    return kg, pp, dct_ct, dk


def test_functional_key_is_one_aggregated_integer(kg_config):
    """ABMPSW20 Fig. 3: sk_y = sum y_i s_i; no per-client components."""
    dct_y = {nid: [1, 2] for nid in kg_config["lst_nid"]}
    dct_x = {nid: [1, 1] for nid in kg_config["lst_nid"]}
    _, _, _, dk = _round(kg_config, "agg-key", dct_x, dct_y)
    assert set(dk) == {"k"}
    assert isinstance(dk["k"], str)


def test_secret_keys_follow_the_gaussian_width(kg_config):
    """s_i ~ D_{Z,sigma} with sigma > N^{5/2}: keys are far larger than N^2.

    The previous prototype drew |s_i| <= max(sec_param, bound_y * eta), so
    a key could be found by brute force from public data.
    """
    import gmpy2 as gp

    kg = MCFEPaillierKeyGenerator(kg_config)
    kg.setup()
    n_square = kg.n_mod * kg.n_mod
    keys = [gp.mpz(v) for nid in kg.lst_nid for v in kg.get_private_keys(nid)["s"]]
    assert kg.sigma > n_square * gp.isqrt(kg.n_mod)
    # P(|s| < N^2) is about N^2 / sigma < N^{-1/2}: every key is huge
    assert all(abs(s) > n_square for s in keys)


def test_wrong_label_is_detected(kg_config):
    dct_x = {nid: [3, -2] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1, 1] for nid in kg_config["lst_nid"]}
    _, pp, dct_ct, dk = _round(kg_config, "label-a", dct_x, dct_y)
    decryptor = MCFEPaillier({"id": "sid_0", "keys": {"pp": pp}})
    assert decryptor.decrypt(dct_ct, dk, dct_y, "label-a") == 3
    assert decryptor.decrypt(dct_ct, dk, dct_y, "label-b") is None


def test_ciphertexts_of_different_labels_do_not_mix(kg_config):
    dct_x = {nid: [1, 1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1, 1] for nid in kg_config["lst_nid"]}
    kg, pp, dct_ct, dk = _round(kg_config, "label-a", dct_x, dct_y)
    other = MCFEPaillier(
        {"id": "nid_1", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_1")}}
    ).encrypt([1, 1], "label-b")
    dct_ct["nid_1"] = other
    decryptor = MCFEPaillier({"id": "sid_0", "keys": {"pp": pp}})
    assert decryptor.decrypt(dct_ct, dk, dct_y, "label-a") is None


def test_bounds_are_checked_against_the_modulus(kg_config):
    from pyfe4ai.utils.exceptions import FEValidationError

    cfg = dict(kg_config, bound_x=2 ** 70)
    with pytest.raises(FEValidationError):
        MCFEPaillierKeyGenerator(cfg)


def test_ndarray_elements_use_distinct_labels(kg_config):
    cfg = dict(kg_config, eta=1)
    kg = MCFEPaillierKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    client = MCFEPaillier(
        {"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}}
    )
    ct = client.encrypt_lst_ndarray([np.zeros((2, 2))], label="round")
    # equal plaintexts under one label would give equal ciphertexts
    assert len({c["c"][0] for c in ct[0].flat}) == 4
