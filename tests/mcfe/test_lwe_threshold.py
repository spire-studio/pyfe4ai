import random
import logging

import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 1,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
        "lwe_n": 16,
        "bound_x": 6,
        "bound_y": 6,
        "bound_u": 2,
        "label_modulus": 8,
    }


def test_key_generator(kg_config):
    kg = ThresholdMCFELWEKeyGenerator(kg_config)
    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.t == kg_config["t"]
    assert kg.eta == kg_config["eta"]

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["t"] == kg_config["t"]
    assert pp["s"] == kg_config["s"]
    assert pp["lst_sid"] == kg_config["lst_sid"]

    assert int(pp["sigma_f"]) >= 1

    sk_nid = kg.get_private_keys("nid_0")
    assert list(sk_nid) == ["z", "label_hash"]

    fusion_weight = {nid: [1] for nid in kg_config["lst_nid"]}
    dk_sid = kg.get_decryption_keys(
        "sid_0", credentials={"fusion_weight": fusion_weight, "label": "test-label"}
    )
    # one share of the aggregated key, nothing per client
    assert set(dk_sid) == {"sk_y", "sid", "lst_sid", "weights_id", "noise_key"}
    assert len(dk_sid["sk_y"]) == kg.params["n0"] + kg.params["m0"]


def test_entire_process(kg_config):
    kg = ThresholdMCFELWEKeyGenerator(kg_config)
    kg.setup()

    label = "tmcfe-lwe-label"
    lst_nid = kg_config["lst_nid"]
    lst_sid = kg_config["lst_sid"]
    threshold = kg_config["t"]

    dct_x = {
        nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"])]
        for nid in lst_nid
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"])]
        for nid in lst_nid
    }
    expected = sum(dct_x[nid][0] * dct_y[nid][0] for nid in lst_nid)

    pp = kg.get_public_parameters()
    dct_ct = {}
    encryptors = {}
    for nid in lst_nid:
        enc = ThresholdMCFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = enc.encrypt(dct_x[nid], label)
        encryptors[nid] = enc

    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dct_ct_prime = {}
    credentials = {"fusion_weight": dct_y, "label": label}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials=credentials)
        dec = ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dk_sid}})
        dct_ct_prime[sid] = dec.share_decrypt(dct_ct, credentials, dk_sid, lst_sid_enrolled)

    computed = encryptors[lst_nid[0]].combine_decrypt(dct_ct_prime)
    assert abs(computed - expected) < 1e-6


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["bound_x"] = 50

    kg = ThresholdMCFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "tmcfe-lwe-ndarray"
    lst_nid = cfg["lst_nid"]
    lst_sid = cfg["lst_sid"]
    threshold = cfg["t"]

    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
        for nid in lst_nid
    }
    dct_y = {nid: [random.randint(1, 3)] for nid in lst_nid}

    expected = []
    for layer_idx in range(len(dct_x[lst_nid[0]])):
        expected.append(
            sum(
                (
                    np.rint(dct_x[nid][layer_idx] * pow(10, precision))
                    / pow(10, precision)
                )
                * dct_y[nid][0]
                for nid in lst_nid
            )
        )

    pp = kg.get_public_parameters()
    dct_ct = {}
    encryptors = {}
    for nid in lst_nid:
        enc = ThresholdMCFELWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid], label=label)
        encryptors[nid] = enc

    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dct_ct_prime = {}
    credentials = {"fusion_weight": dct_y, "label": label}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials=credentials)
        dec = ThresholdMCFELWE(
            {"id": sid, "precision": precision, "keys": {"pp": pp, "dk": dk_sid}}
        )
        dct_ct_prime[sid] = dec.compute_lst_ndarray_ct(
            dct_ct, credentials, dk_sid, lst_sid_enrolled
        )

    computed = encryptors[lst_nid[0]].decrypt_lst_ndarray_ct(dct_ct_prime)
    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def _round(kg_config, dct_x, dct_y, label, enrolled):
    kg = ThresholdMCFELWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_ct = {
        nid: ThresholdMCFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        ).encrypt(dct_x[nid], label)
        for nid in dct_x
    }
    credentials = {"fusion_weight": dct_y, "label": label}
    partials = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        server = ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        partials[sid] = server.share_decrypt(dct_ct, credentials, dk, enrolled)
    combiner = ThresholdMCFELWE({"id": "nid_0", "keys": {"pp": pp}})
    return kg, pp, dct_ct, partials, combiner


@pytest.mark.parametrize(
    "s, t, enrolled",
    [
        (3, 2, ["sid_1", "sid_2"]),
        (4, 2, ["sid_1", "sid_3"]),
        (4, 3, ["sid_3", "sid_0", "sid_2"]),
        (5, 3, ["sid_0", "sid_1", "sid_2", "sid_3", "sid_4"]),
    ],
)
def test_any_enrolled_set_decrypts_extreme_values(kg_config, s, t, enrolled):
    """Correctness with the full flooding noise of |E| servers at the bounds."""
    cfg = dict(kg_config, s=s, t=t, lst_sid=["sid_{}".format(i) for i in range(s)])
    bx, by = cfg["bound_x"], cfg["bound_y"]
    for sign in (1, -1):
        dct_x = {nid: [sign * bx] for nid in cfg["lst_nid"]}
        dct_y = {nid: [by] for nid in cfg["lst_nid"]}
        _, _, _, partials, combiner = _round(cfg, dct_x, dct_y, "edge-{}".format(sign), enrolled)
        assert combiner.combine_decrypt(partials) == sign * 3 * bx * by


def test_partial_decryptions_are_noise_flooded(kg_config):
    """Partials carry noise, derandomised per (label, enrolled set).

    Without noise a partial is an exact linear function of the key share
    (v0.2.0 known issue); with fresh noise per request a combiner could
    average repeated answers, so identical requests get identical answers.
    """
    import gmpy2 as gp

    from pyfe4ai.schemes.mcfe.lwe import dot_mod
    from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero

    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, combiner = _round(kg_config, dct_x, dct_y, "flood", enrolled)
    credentials = {"fusion_weight": dct_y, "label": "flood"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})

    # identical requests: identical answers (no averaging)
    assert server.share_decrypt(dct_ct, credentials, dk, enrolled) == partials["sid_0"]

    # the answer is not the exact linear value lambda <share, H(l)>
    q = gp.mpz(pp["q"])
    lam = lagrange_coefficient_at_zero(1, [1, 2], q)
    exact = lam * dot_mod([gp.mpz(v) for v in dk["sk_y"]], server._hash_label("flood"), q) % q
    noise = (gp.mpz(partials["sid_0"]["w"]) - exact) % q
    assert noise != 0
    assert min(noise, q - noise) <= 12 * int(pp["sigma_f"])

    # another enrolled set gets independent noise and still decrypts
    other = ["sid_0", "sid_2"]
    w_other = server.share_decrypt(dct_ct, credentials, dk, other)
    lam_o = lagrange_coefficient_at_zero(1, [1, 3], q)
    exact_o = lam_o * dot_mod([gp.mpz(v) for v in dk["sk_y"]], server._hash_label("flood"), q) % q
    assert (gp.mpz(w_other["w"]) - exact_o) % q != noise
    dk2 = kg.get_decryption_keys("sid_2", credentials=credentials)
    server2 = ThresholdMCFELWE({"id": "sid_2", "keys": {"pp": pp, "dk": dk2}})
    partials_o = {"sid_0": w_other, "sid_2": server2.share_decrypt(dct_ct, credentials, dk2, other)}
    assert combiner.combine_decrypt(partials) == 3
    assert combiner.combine_decrypt(partials_o) == 3


def test_server_rejects_other_weights(kg_config):
    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    kg, pp, dct_ct, _, _ = _round(kg_config, dct_x, dct_y, "weights", ["sid_0", "sid_1"])
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    server = ThresholdMCFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    other = {"fusion_weight": {nid: [2] for nid in dct_y}, "label": "weights"}
    with pytest.raises(FESchemeError):
        server.share_decrypt(dct_ct, other, dk, ["sid_0", "sid_1"])


def test_single_server_cannot_decrypt(kg_config):
    dct_x = {nid: [2] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    kg, pp, dct_ct, partials, combiner = _round(
        kg_config, dct_x, dct_y, "lone", ["sid_0", "sid_1"]
    )
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt({"sid_0": partials["sid_0"]})
    credentials = {"fusion_weight": dct_y, "label": "lone"}
    dk = kg.get_decryption_keys("sid_0", credentials=credentials)
    server = ThresholdMCFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError):
        server.share_decrypt(dct_ct, credentials, dk, ["sid_0"])


def test_duplicate_or_mismatched_partials_rejected(kg_config):
    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    _, _, _, partials, combiner = _round(
        kg_config, dct_x, dct_y, "dup", ["sid_0", "sid_1", "sid_2"]
    )
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt(dict(partials, extra=partials["sid_0"]))
    del partials["sid_2"]
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt(partials)


def test_invalid_threshold_rejected(kg_config):
    with pytest.raises(FEValidationError):
        ThresholdMCFELWEKeyGenerator(dict(kg_config, t=4))
