"""Threshold BMMS21 identity-based MCFE over Ring-LWE (smudged partials)."""

import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError


@pytest.fixture
def kg_config():
    return {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2,
            "lst_nid": ["nid_0", "nid_1", "nid_2"], "lst_sid": ["sid_0", "sid_1", "sid_2"],
            "ring_n": 16, "bound_x": 4, "bound_y": 4}


def _round(cfg, dct_x, dct_y, label, enrolled):
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_ct = {
        nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(dct_x[nid], label)
        for nid in dct_x
    }
    credentials = {"fusion_weight": dct_y, "label": label}
    partials = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        partials[sid] = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).share_decrypt(
            dct_ct, credentials, dk, enrolled
        )
    return kg, pp, dct_ct, partials, ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}})


def test_key_shares(kg_config):
    kg = ThresholdMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {n: [1] for n in kg_config["lst_nid"]}, "label": "l"})
    assert set(dk) == {"sk_y", "z", "z_chunks", "label", "noise_key", "noise_version", "noise_sampler", "sid", "lst_sid",
                       "weights_id",
                       "mac_keys"}


@pytest.mark.parametrize("enrolled", [["sid_1", "sid_2"], ["sid_0", "sid_2"], ["sid_0", "sid_1", "sid_2"]])
def test_any_enrolled_set_decrypts_extreme_values(kg_config, enrolled):
    bx, by = kg_config["bound_x"], kg_config["bound_y"]
    for sign in (1, -1):
        dct_x = {nid: [[sign * bx] * 5] for nid in kg_config["lst_nid"]}
        dct_y = {nid: [by] for nid in kg_config["lst_nid"]}
        _, _, _, partials, combiner = _round(kg_config, dct_x, dct_y, "edge", enrolled)
        assert combiner.combine_decrypt(partials) == [sign * 3 * bx * by] * 5


def test_partials_are_flooded_and_derandomised(kg_config):
    dct_x = {nid: [[1, 2]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, combiner = _round(kg_config, dct_x, dct_y, "lbl", enrolled)
    credentials = {"fusion_weight": dct_y, "label": "lbl"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    assert server.share_decrypt(dct_ct, credentials, dk, enrolled) == partials["sid_0"]
    other = server.share_decrypt(dct_ct, credentials, dk, ["sid_0", "sid_2"])
    assert other["w"] != partials["sid_0"]["w"]
    assert combiner.combine_decrypt(partials) == [3, 6]


def test_wrong_label_or_weights_rejected(kg_config):
    dct_x = {nid: [[1]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    kg, pp, dct_ct, partials, combiner = _round(kg_config, dct_x, dct_y, "a", ["sid_0", "sid_1"])
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y, "label": "a"})
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError):
        server.share_decrypt(dct_ct, {"fusion_weight": dct_y, "label": "b"}, dk, ["sid_0", "sid_1"])
    with pytest.raises(FESchemeError):
        server.share_decrypt(dct_ct, {"fusion_weight": {n: [2] for n in dct_y}, "label": "a"}, dk, ["sid_0", "sid_1"])
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt({"sid_0": partials["sid_0"]})


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config, bound_x=50)
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    precision, label = 1, "tmcfe-ring-ndarray"
    lst_nid = cfg["lst_nid"]
    dct_x = {nid: [np.random.random_sample((3, 7)), np.random.random_sample((2,))] for nid in lst_nid}
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}
    expected = [sum((np.rint(dct_x[nid][l] * 10) / 10) * dct_y[nid][0] for nid in lst_nid) for l in range(2)]
    dct_ct = {
        nid: ThresholdMCFERingLWE({"id": nid, "precision": precision, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        .encrypt_lst_ndarray(dct_x[nid], label=label)
        for nid in lst_nid
    }
    n_chunks = ThresholdMCFERingLWE.key_chunks(dct_ct["nid_0"])
    credentials = {"fusion_weight": dct_y, "label": label, "n_chunks": n_chunks}
    enrolled = ["sid_0", "sid_2"]
    partials = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        partials[sid] = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).compute_lst_ndarray_ct(
            dct_ct, credentials, dk, enrolled
        )
    computed = ThresholdMCFERingLWE({"id": "c", "precision": precision, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(partials)
    for l in range(2):
        assert np.allclose(computed[l], expected[l], atol=1e-2)


def test_noise_does_not_depend_on_how_the_request_is_dressed(kg_config):
    """Regression: the flooding noise was seeded with the JSON of the
    ciphertexts, so an extra field drew fresh noise, and averaging such
    partials removes the noise. Re-encoded values (v + q) are now refused
    because the tags cover the ciphertexts as serialised."""
    dct_x = {nid: [[1, 2]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, _ = _round(kg_config, dct_x, dct_y, "lbl", enrolled)
    credentials = {"fusion_weight": dct_y, "label": "lbl"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    q = int(pp["q"])
    first = dct_ct["nid_0"]
    dressed = dict(dct_ct, nid_0=dict(first, padding="1"))
    assert server.share_decrypt(dressed, credentials, dk, enrolled) == partials["sid_0"]
    shifted = dict(dct_ct, nid_0=dict(first, ct1=[str(int(v) + q) for v in first["ct1"]]))
    with pytest.raises(FESchemeError, match="not authenticated"):
        server.share_decrypt(shifted, credentials, dk, enrolled)


def test_servers_refuse_ciphertexts_not_made_by_the_clients(kg_config):
    """Regression: a partial decryption is linear in ct1, so a combiner
    submitting chosen ct1 read the functional-key parts S_i y_i off the
    partials."""
    dct_x = {nid: [[1, 2]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, _ = _round(kg_config, dct_x, dct_y, "lbl", enrolled)
    credentials = {"fusion_weight": dct_y, "label": "lbl"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    first = dct_ct["nid_0"]
    chosen = ["1"] + ["0"] * (len(first["ct1"]) - 1)
    for forged in (
        dict(first, ct1=chosen),
        {k: v for k, v in first.items() if k != "tags"},
        dict(first, tags=dict(first["tags"], sid_0=first["tags"]["sid_1"])),
        dct_ct["nid_1"],
    ):
        with pytest.raises(FESchemeError, match="not authenticated"):
            server.share_decrypt(dict(dct_ct, nid_0=forged), credentials, dk, enrolled)


def test_ciphertexts_of_another_label_are_refused(kg_config):
    dct_x = {nid: [[1]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    kg, pp, old_ct, _, _ = _round(kg_config, dct_x, dct_y, "round-1", ["sid_0", "sid_1"])
    credentials = {"fusion_weight": dct_y, "label": "round-2"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError, match="not authenticated"):
        server.share_decrypt(old_ct, credentials, dk, ["sid_0", "sid_1"])


def test_a_client_cannot_redraw_the_noise_by_changing_ct0(kg_config):
    """The key part depends on ct1 and the key term only."""
    from pyfe4ai.schemes.mcfe.labels import encode_label
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import _TAG_DOMAIN, _tag_payload
    from pyfe4ai.schemes.threshold_utils import tag_ciphertext

    dct_x = {nid: [[1, 2]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, _ = _round(kg_config, dct_x, dct_y, "lbl", enrolled)
    credentials = {"fusion_weight": dct_y, "label": "lbl"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    first = dct_ct["nid_0"]
    changed = dict(first, ct0=[[str((int(v) + 1) % int(pp["q"])) for v in row] for row in first["ct0"]])
    tag_ciphertext(changed, kg.get_private_keys("nid_0")["mac_keys"], _TAG_DOMAIN,
                   _tag_payload(encode_label("lbl"), changed))
    part = server.share_decrypt(dict(dct_ct, nid_0=changed), credentials, dk, enrolled)
    assert part["w"] == partials["sid_0"]["w"] and part["c_y"] != partials["sid_0"]["c_y"]


def test_ndarray_combination_refuses_bad_partials(kg_config):
    """A disagreeing partial crashed with a TypeError (values.extend(None))."""
    cfg = dict(kg_config, bound_x=50)
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    lst_nid = cfg["lst_nid"]
    dct_ct = {
        nid: ThresholdMCFERingLWE({"id": nid, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        .encrypt_lst_ndarray([np.ones(20)], label="l")
        for nid in lst_nid
    }
    credentials = {"fusion_weight": {nid: [1] for nid in lst_nid}, "label": "l",
                   "n_chunks": ThresholdMCFERingLWE.key_chunks(dct_ct["nid_0"])}
    enrolled = ["sid_0", "sid_1"]
    partials = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        partials[sid] = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).compute_lst_ndarray_ct(
            dct_ct, credentials, dk, enrolled)
    combiner = ThresholdMCFERingLWE({"id": "c", "precision": 1, "keys": {"pp": pp}})
    assert np.allclose(combiner.decrypt_lst_ndarray_ct(partials)[0], 3)
    bad = dict(partials["sid_1"][0]["parts"][0], c_y=partials["sid_1"][0]["parts"][0]["w"])
    tampered = dict(partials, sid_1=[dict(partials["sid_1"][0], parts=[bad] + partials["sid_1"][0]["parts"][1:])])
    with pytest.raises(FESchemeError, match="element"):
        combiner.decrypt_lst_ndarray_ct(tampered)
    reshaped = dict(partials, sid_1=[dict(partials["sid_1"][0], shape=[4, 5])])
    with pytest.raises(FESchemeError, match="shape"):
        combiner.decrypt_lst_ndarray_ct(reshaped)


def test_malformed_ciphertexts_are_refused(kg_config):
    dct_x = {nid: [[1]] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, _, _ = _round(kg_config, dct_x, dct_y, "lbl", enrolled)
    credentials = {"fusion_weight": dct_y, "label": "lbl"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    from pyfe4ai.schemes.mcfe.labels import encode_label
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import _TAG_DOMAIN, _tag_payload
    from pyfe4ai.schemes.threshold_utils import tag_ciphertext

    mac_keys = kg.get_private_keys("nid_0")["mac_keys"]
    first = dct_ct["nid_0"]
    for bad in (
        dict(first, ct1=first["ct1"][:-1]),
        dict(first, ct0=first["ct0"] + first["ct0"]),
        dict(first, ct0=[row[:-1] for row in first["ct0"]]),
        dict(first, ct1=["x"] + first["ct1"][1:]),
        dict(first, k=0),
        {"ct1": first["ct1"], "k": first["k"]},
    ):
        # tagged by the (corrupted) client itself, so it reaches the shape check
        tag_ciphertext(bad, mac_keys, _TAG_DOMAIN, _tag_payload(encode_label("lbl"), bad))
        with pytest.raises(FESchemeError, match="malformed"):
            server.share_decrypt(dict(dct_ct, nid_0=bad), credentials, dk, enrolled)
