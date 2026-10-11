import logging
import random

import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mcfe.lwe import MCFELWE
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.fullysec_lwe import MCFEFullySecLWE
from pyfe4ai.schemes.mcfe.fullysec_lwe import MCFEFullySecLWEKeyGenerator

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


SCHEMES = [
    pytest.param(MCFELWEKeyGenerator, MCFELWE, "selective", id="selective"),
    pytest.param(MCFEFullySecLWEKeyGenerator, MCFEFullySecLWE, "adaptive", id="adaptive"),
]


def _encrypt_all(kg, pp, crypto_cls, dct_x, label):
    return {
        nid: crypto_cls(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        ).encrypt(dct_x[nid], label)
        for nid in dct_x
    }


@pytest.mark.parametrize("kg_cls, crypto_cls, security", SCHEMES)
def test_parameters_meet_appendix_c3(kg_config, kg_cls, crypto_cls, security):
    """ABMPSW20 Appendix C.2/C.3 constraints, signed-message version."""
    import math

    import gmpy2 as gp

    kg = kg_cls(kg_config)
    prm = kg.params
    n = kg.n_slots
    p_range = 2 * kg.bound_x + 1
    kappa = kg.stat_param
    assert prm["K"] == 2 * n * kg.bound_x * kg.bound_y + 1
    assert prm["q0"] > prm["K"] * (n * kg.bound_y + 1)  # correctness
    assert prm["sigma_t"] >= 10 * n * p_range ** 2  # Lemma 34/35
    assert prm["q"] >= prm["q0"] * prm["B"] * 2 ** kappa  # rounding absorbs t.e
    assert gp.is_prime(prm["q"])
    extra = 4 * n * math.log2(p_range) if security == "adaptive" else 0
    assert prm["m0"] * math.log2(4 / 3) >= prm["q"].bit_length() + 2 * kappa + extra
    assert kg.get_public_parameters()["security"] == security


@pytest.mark.parametrize("kg_cls, crypto_cls, security", SCHEMES)
def test_functional_key_is_one_aggregated_vector(kg_config, kg_cls, crypto_cls, security):
    kg = kg_cls(kg_config)
    kg.setup()
    dct_y = {nid: [1, -1] for nid in kg_config["lst_nid"]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    assert set(dk) == {"sk_y"}
    assert len(dk["sk_y"]) == kg.params["n0"] + kg.params["m0"]


@pytest.mark.parametrize("kg_cls, crypto_cls, security", SCHEMES)
def test_extreme_values_and_many_labels(kg_config, kg_cls, crypto_cls, security):
    kg = kg_cls(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    bx, by = kg_config["bound_x"], kg_config["bound_y"]
    lst_nid = kg_config["lst_nid"]
    decryptor = crypto_cls({"id": "sid_0", "keys": {"pp": pp}})
    cases = [
        ({nid: [bx, bx] for nid in lst_nid}, {nid: [by, by] for nid in lst_nid}),
        ({nid: [-bx, -bx] for nid in lst_nid}, {nid: [by, by] for nid in lst_nid}),
        ({nid: [bx, -bx] for nid in lst_nid}, {nid: [-by, by] for nid in lst_nid}),
        ({nid: [0, 0] for nid in lst_nid}, {nid: [by, -by] for nid in lst_nid}),
    ]
    for idx, (dct_x, dct_y) in enumerate(cases * 3):
        label = "round-{}".format(idx)
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
        dct_ct = _encrypt_all(kg, pp, crypto_cls, dct_x, label)
        expected = sum(x * y for nid in lst_nid for x, y in zip(dct_x[nid], dct_y[nid]))
        assert decryptor.decrypt(dct_ct, dk, dct_y, label) == expected


@pytest.mark.parametrize("kg_cls, crypto_cls, security", SCHEMES)
def test_wrong_label_does_not_decrypt(kg_config, kg_cls, crypto_cls, security):
    kg = kg_cls(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    lst_nid = kg_config["lst_nid"]
    dct_x = {nid: [3, -2] for nid in lst_nid}
    dct_y = {nid: [2, 1] for nid in lst_nid}
    expected = 12
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    decryptor = crypto_cls({"id": "sid_0", "keys": {"pp": pp}})
    dct_ct = _encrypt_all(kg, pp, crypto_cls, dct_x, "label-true")
    assert decryptor.decrypt(dct_ct, dk, dct_y, "label-true") == expected
    # a wrong label yields an unrelated value (equal to the result w.p. ~1/K)
    hits = sum(
        decryptor.decrypt(dct_ct, dk, dct_y, "label-wrong-{}".format(i)) == expected
        for i in range(5)
    )
    assert hits <= 1


def test_ndarray_elements_use_distinct_labels(kg_config):
    cfg = dict(kg_config, eta=1)
    kg = MCFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    client = MCFELWE({"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    ct = client.encrypt_lst_ndarray([np.zeros((3, 3))], label="round")
    # equal plaintexts under one label would give equal ciphertexts
    assert len({c["c"][0] for c in ct[0].flat}) > 1
