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
        "eta": 2,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "pairing_group_param": "MNT224",
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


def _setup(cfg):
    kg = ThresholdMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    x = {"nid_0": [1, -2], "nid_1": [0, 3], "nid_2": [2, 1]}
    y = {"nid_0": [2, 1], "nid_1": [-1, 2], "nid_2": [1, -2]}
    dct_ct = {
        nid: ThresholdMCFEFHMultiIPE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(
            x[nid], "round-1"
        )
        for nid in cfg["lst_nid"]
    }
    return kg, pp, y, dct_ct


def _partials(kg, pp, y, dct_ct, enrolled):
    out = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        dec = ThresholdMCFEFHMultiIPE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        out[sid] = dec.share_decrypt(dct_ct, {"fusion_weight": y}, dk, enrolled)
    return out


def test_every_enrolled_set_decrypts(kg_config):
    kg, pp, y, dct_ct = _setup(kg_config)
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    for enrolled in (["sid_0", "sid_1"], ["sid_1", "sid_2"], ["sid_0", "sid_1", "sid_2"]):
        assert combiner.combine_decrypt(_partials(kg, pp, y, dct_ct, enrolled)) == 6


def test_threshold_and_consistency_checks(kg_config):
    from pyfe4ai.utils.exceptions import FESchemeError

    kg, pp, y, dct_ct = _setup(kg_config)
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    parts = _partials(kg, pp, y, dct_ct, ["sid_0", "sid_1"])
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt({"sid_0": parts["sid_0"]})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    dec = ThresholdMCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    assert "weights_id" not in dk and "key_id" in dk  # no digest of y
    with pytest.raises(FESchemeError):
        dec.share_decrypt(dct_ct, {"fusion_weight": y}, dk, ["sid_0"])
    # a server must use its own share (another's has a different Lagrange point)
    other = ThresholdMCFEFHMultiIPE({"id": "sid_1", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError):
        other.share_decrypt(dct_ct, {}, dk, ["sid_0", "sid_1"])
    # servers that decrypted different ciphertexts are not combined
    other = dict(dct_ct)
    other["nid_0"] = dct_ct["nid_1"]
    parts["sid_1"] = _partials(kg, pp, y, other, ["sid_0", "sid_1"])["sid_1"]
    assert combiner.combine_decrypt(parts) is None


def test_single_share_reveals_nothing_on_its_own(kg_config):
    """A partial from fewer than t servers does not decrypt."""
    kg, pp, y, dct_ct = _setup(kg_config)
    parts = _partials(kg, pp, y, dct_ct, ["sid_0", "sid_1"])
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    assert combiner.ctx.solve(combiner.ctx.de(parts["sid_0"]["w"])) is None
