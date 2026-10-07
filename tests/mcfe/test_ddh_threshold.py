import os
import copy
import random
import pytest
import logging
import jsonpickle

import numpy as np
import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFE
from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    sec_param = 128
    n = 3
    s = 2
    config = {
        "sec_param": sec_param,
        "lst_nid": ["nid_{}".format(i) for i in range(n)],
        "lst_sid": ["sid_{}".format(i) for i in range(s)],
        "eta": 1,
        "n": n,
        "s": s,
        "t": 2,
    }
    return config


def test_key_generator(kg_config):
    kg = ThresholdMCFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]
    assert kg.n == kg_config["n"]
    assert kg.t == kg_config["t"]
    assert kg.s == kg_config["s"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_TMCFE)
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    lst_nid_expected = ["nid_{}".format(i) for i in range(kg_config["n"])]
    lst_sid_expected = ["sid_{}".format(i) for i in range(kg_config["s"])]
    assert kg.lst_nid == lst_nid_expected
    assert kg.lst_sid == lst_sid_expected
    assert kg.dict_dk == {}

    # test setup method
    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.p) == pp["p"]
    assert gp.digits(kg.g) == pp["g"]

    # test sk generation
    for nid in lst_nid_expected:
        sk_nid = kg.get_private_keys(nid)
        assert "g_alpha" in sk_nid
        assert "g_alpha_w" in sk_nid
        assert "u" in sk_nid
        if nid == "nid_5":
            logger.debug(sk_nid)

    invalid_nid0 = "id_3"
    invalid_sk0 = kg.get_private_keys(invalid_nid0)
    assert invalid_sk0 == None
    invalid_nid1 = "nid_10"
    invalid_sk1 = kg.get_private_keys(invalid_nid1)
    assert invalid_sk1 == None

    # test dk generation
    label = "test-label"
    fusion_weight = {
        nid: [1 for _ in range(kg_config["eta"])] for nid in lst_nid_expected
    }
    lst_sid = ["sid_{}".format(i) for i in range(kg_config["s"])]
    for sid in lst_sid:
        dk_sid = kg.get_decryption_keys(
            sid, credentials={"fusion_weight": fusion_weight, "label": label}
        )
        assert "v0" in dk_sid
        assert "v1" in dk_sid
        assert isinstance(dk_sid["v1"], dict)
        assert len(dk_sid["v1"]) == kg_config["n"]


def test_entire_process(kg_config):
    kg = ThresholdMCFEKeyGenerator(kg_config)
    kg.setup()

    precision = 3
    label = "test-label"
    lst_nid = kg_config["lst_nid"]
    lst_sid = kg_config["lst_sid"]
    threshold = kg_config["t"]

    # prepare the input
    dict_tmife_n_x = {}
    dict_tmife_n_y = {}
    for nid in lst_nid:
        sign = 1 if random.random() < 0.5 else -1
        dict_tmife_n_x[nid] = [sign * np.random.rand()]
        dict_tmife_n_y[nid] = [random.randint(1, 10)]
    logger.info("x: {}".format(list(dict_tmife_n_x.values())))
    logger.info("y: {}".format(list(dict_tmife_n_y.values())))
    lst_x = list(dict_tmife_n_x.values())
    lst_y = list(dict_tmife_n_y.values())
    expected_inner_prod = sum(lst_x[i][0] * lst_y[i][0] for i in range(len(lst_y)))
    logger.info("<x,y>: {}".format(expected_inner_prod))

    # encryption
    pp = kg.get_public_parameters()
    dict_tmife_n = {}
    dict_tmife_n_ct = {}
    for nid in lst_nid:
        sk_nid = kg.get_private_keys(nid)
        config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": pp, "sk": sk_nid},
        }
        tmife_nid = ThresholdMCFE(config_nid)
        ct_nid = tmife_nid.encrypt(dict_tmife_n_x[nid], label)
        assert isinstance(ct_nid, dict)
        assert "ct0" in ct_nid
        assert "ct1" in ct_nid
        dict_tmife_n_ct[nid] = ct_nid
        dict_tmife_n[nid] = tmife_nid

    # share decryption
    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dict_tmife_s_ct = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(
            sid, credentials={"fusion_weight": dict_tmife_n_y, "label": label}
        )
        config_sid = {
            "id": sid,
            "precision": precision,
            "keys": {"pp": pp, "dk": dk_sid},
        }
        tmife_sid = ThresholdMCFE(config_sid)
        ct_sid = tmife_sid.share_decrypt(
            dict_tmife_n_ct, dict_tmife_n_y, dk_sid, lst_sid_enrolled
        )
        assert isinstance(ct_sid, dict)
        assert "ct0_prime" in ct_sid
        assert "ct1_prime" in ct_sid
        assert "ct2_prime" in ct_sid
        dict_tmife_s_ct[sid] = ct_sid

    # combine decryption
    for nid in lst_nid:
        tmife_nid = dict_tmife_n[nid]
        inner_prod = tmife_nid.combine_decrypt(dict_tmife_s_ct)
        assert abs(inner_prod - expected_inner_prod) < 1e-2
        logger.info("{}-inner prod-{}".format(nid, inner_prod))


def test_entire_process_lst_ndarray(kg_config):
    kg = ThresholdMCFEKeyGenerator(kg_config)
    kg.setup()

    precision = 3
    label = "test-label"
    lst_nid = kg_config["lst_nid"]
    lst_sid = kg_config["lst_sid"]
    threshold = kg_config["t"]

    # prepare the input
    lst_ndarray_layer_shape = [(2, 4), (2, 3, 2), (5, 2)]
    dict_tmife_n_x = {}
    dict_tmife_n_y = {}
    for nid in lst_nid:
        dict_tmife_n_x[nid] = [
            np.random.random_sample(shape) for shape in lst_ndarray_layer_shape
        ]
        dict_tmife_n_y[nid] = [random.randint(1, 10)]
    lst_check_points = [[0, (0, 1)], [1, (1, 1, 1)], [2, (3, 1)]]
    lst_expected_inner_prod = []
    for cp in lst_check_points:
        lst_x = [dict_tmife_n_x[nid][cp[0]][cp[1]] for nid in lst_nid]
        lst_y = list(dict_tmife_n_y.values())
        logger.debug("x[{}][{}]: {}".format(cp[0], cp[1], lst_x))
        logger.debug("y: {}".format(lst_y))
        expected_inner_prod = sum(lst_x[i] * lst_y[i][0] for i in range(len(lst_y)))
        logger.info("<x,y>: {}".format(expected_inner_prod))
        lst_expected_inner_prod.append(expected_inner_prod)

    # encryption
    pp = kg.get_public_parameters()
    dict_tmife_n = {}
    dict_tmife_n_ct = {}
    for nid in lst_nid:
        sk_nid = kg.get_private_keys(nid)
        config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": pp, "sk": sk_nid},
        }
        tmife_nid = ThresholdMCFE(config_nid)
        ct_nid = tmife_nid.encrypt_lst_ndarray(dict_tmife_n_x[nid], label=label)
        assert isinstance(ct_nid, list)
        assert isinstance(ct_nid[0], np.ndarray)
        assert "ct0" in ct_nid[0][(0, 0)]
        assert "ct1" in ct_nid[0][(0, 0)]
        dict_tmife_n_ct[nid] = ct_nid
        dict_tmife_n[nid] = tmife_nid

    # share decryption
    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dict_tmife_s_ct = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(
            sid, credentials={"fusion_weight": dict_tmife_n_y, "label": label}
        )
        config_sid = {
            "id": sid,
            "precision": precision,
            "keys": {"pp": pp, "dk": dk_sid},
        }
        tmife_sid = ThresholdMCFE(config_sid)
        ct_sid = tmife_sid.compute_lst_ndarray_ct(
            dict_tmife_n_ct, dict_tmife_n_y, dk_sid, lst_sid_enrolled
        )
        assert isinstance(ct_sid, list)
        assert isinstance(ct_sid[0], np.ndarray)
        assert "ct0_prime" in ct_sid[0][(0, 0)]
        assert "ct1_prime" in ct_sid[0][(0, 0)]
        assert "ct2_prime" in ct_sid[0][(0, 0)]
        dict_tmife_s_ct[sid] = ct_sid

    # combine decryption
    for nid in lst_nid:
        tmife_nid = dict_tmife_n[nid]
        for i, cp in enumerate(lst_check_points):
            dec_lst_ndarray = tmife_nid.decrypt_lst_ndarray_ct(dict_tmife_s_ct)
            assert isinstance(dec_lst_ndarray, list)
            assert (
                abs(dec_lst_ndarray[cp[0]][cp[1]] - lst_expected_inner_prod[i]) < 1e-2
            )
            logger.info(
                "{}-{}-inner prod-{}".format(nid, cp, dec_lst_ndarray[cp[0]][cp[1]])
            )


def _setup_round(n, s, t, eta, label, precision=0):
    lst_nid = ["nid_{}".format(i) for i in range(n)]
    lst_sid = ["sid_{}".format(i) for i in range(s)]
    kg = ThresholdMCFEKeyGenerator(
        {
            "sec_param": 128,
            "lst_nid": lst_nid,
            "lst_sid": lst_sid,
            "eta": eta,
            "n": n,
            "s": s,
            "t": t,
        }
    )
    kg.setup()
    pp = kg.get_public_parameters()
    dct_x = {nid: [random.randint(-3, 3) for _ in range(eta)] for nid in lst_nid}
    dct_y = {nid: [random.randint(1, 3) for _ in range(eta)] for nid in lst_nid}
    expected = sum(
        x_i * y_i for nid in lst_nid for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )
    clients = {
        nid: ThresholdMCFE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        for nid in lst_nid
    }
    dct_ct = {nid: clients[nid].encrypt(dct_x[nid], label) for nid in lst_nid}
    return kg, pp, clients, dct_ct, dct_y, expected


def _share_decrypt(kg, pp, dct_ct, dct_y, label, lst_sid_enrolled, precision=0):
    shares = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(
            sid, credentials={"fusion_weight": dct_y, "label": label}
        )
        server = ThresholdMCFE(
            {"id": sid, "precision": precision, "keys": {"pp": pp, "dk": dk_sid}}
        )
        shares[sid] = server.share_decrypt(dct_ct, dct_y, dk_sid, lst_sid_enrolled)
    return shares


@pytest.mark.parametrize(
    "n, s, t, eta, enrolled",
    [
        (3, 3, 2, 1, ["sid_1", "sid_2"]),
        (3, 4, 2, 1, ["sid_1", "sid_3"]),
        (3, 4, 3, 1, ["sid_1", "sid_2", "sid_3"]),
        (3, 5, 3, 1, ["sid_4", "sid_0", "sid_2"]),
        (3, 4, 2, 1, ["sid_0", "sid_1", "sid_2", "sid_3"]),
        (2, 3, 2, 2, ["sid_1", "sid_2"]),
        (2, 3, 2, 3, ["sid_2", "sid_1"]),
        (2, 4, 3, 4, ["sid_1", "sid_2", "sid_3"]),
    ],
)
def test_threshold_subsets_and_eta(n, s, t, eta, enrolled):
    """Regression (C2/N3/N5): any >= t subset decrypts, for every eta.

    Covers subsets that exclude sid_0 (whose share used to be the secret
    itself), non-integer Lagrange coefficients (e.g. {sid_1, sid_3}) and
    eta >= 2 (setup used to raise IndexError, combine hard-coded eta = 2).
    """
    label = "test-label-subset"
    kg, pp, clients, dct_ct, dct_y, expected = _setup_round(n, s, t, eta, label)
    shares = _share_decrypt(kg, pp, dct_ct, dct_y, label, enrolled)
    for client in clients.values():
        assert client.combine_decrypt(shares) == expected


def test_single_server_cannot_decrypt():
    """Regression (N3): sid_0 alone must not be able to decrypt when t = 2."""
    label = "test-label-sid0"
    kg, pp, clients, dct_ct, dct_y, expected = _setup_round(3, 3, 2, 1, label)
    combiner = clients["nid_0"]

    # the protocol refuses fewer than t participants on both sides
    dk_sid0 = kg.get_decryption_keys(
        "sid_0", credentials={"fusion_weight": dct_y, "label": label}
    )
    server0 = ThresholdMCFE({"id": "sid_0", "precision": 0, "keys": {"pp": pp, "dk": dk_sid0}})
    with pytest.raises(FESchemeError):
        server0.share_decrypt(dct_ct, dct_y, dk_sid0, ["sid_0"])
    shares = _share_decrypt(kg, pp, dct_ct, dct_y, label, ["sid_0", "sid_1"])
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt({"sid_0": shares["sid_0"]})

    # and the single share is cryptographically useless: even with the
    # count checks bypassed (t forged to 1) it does not reveal the result
    forged_pp = dict(pp, t=1)
    forged_server = ThresholdMCFE(
        {"id": "sid_0", "precision": 0, "keys": {"pp": forged_pp, "dk": dk_sid0}}
    )
    forged_combiner = ThresholdMCFE(
        {
            "id": "nid_0",
            "precision": 0,
            "keys": {"pp": forged_pp, "sk": kg.get_private_keys("nid_0")},
        }
    )
    lone_share = forged_server.share_decrypt(dct_ct, dct_y, dk_sid0, ["sid_0"])
    assert forged_combiner.combine_decrypt({"sid_0": lone_share}) != expected


def test_shares_must_match_enrolled_set():
    label = "test-label-mismatch"
    kg, pp, clients, dct_ct, dct_y, _ = _setup_round(3, 3, 2, 1, label)
    shares = _share_decrypt(
        kg, pp, dct_ct, dct_y, label, ["sid_0", "sid_1", "sid_2"]
    )
    del shares["sid_2"]
    with pytest.raises(FESchemeError):
        clients["nid_0"].combine_decrypt(shares)


def test_invalid_threshold_rejected():
    with pytest.raises(FEValidationError):
        ThresholdMCFEKeyGenerator(
            {
                "sec_param": 128,
                "lst_nid": ["nid_0", "nid_1"],
                "lst_sid": ["sid_0", "sid_1"],
                "eta": 1,
                "n": 2,
                "s": 2,
                "t": 3,
            }
        )


def test_duplicate_partial_decryption_rejected():
    """A share submitted twice under another key used to pass validation
    (contributors were compared as a set) and silently broke decryption."""
    label = "test-label-dup"
    kg, pp, clients, dct_ct, dct_y, _ = _setup_round(3, 3, 2, 1, label)
    shares = _share_decrypt(kg, pp, dct_ct, dct_y, label, ["sid_0", "sid_1"])
    shares["extra"] = shares["sid_0"]
    with pytest.raises(FESchemeError):
        clients["nid_0"].combine_decrypt(shares)
