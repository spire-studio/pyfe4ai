import os
import random
import pytest
import logging

import numpy as np
import gmpy2 as gp

from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFE
from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    sec_param = 256
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
    kg = ThresholdMIFEKeyGenerator(kg_config)

    assert kg.lambd == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]
    assert kg.n == kg_config["n"]
    assert kg.t == kg_config["t"]
    assert kg.s == kg_config["s"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_TMIFE)
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
    credentials = {
        nid: [1 for _ in range(kg_config["eta"])] for nid in lst_nid_expected
    }
    lst_sid = ["sid_{}".format(i) for i in range(kg_config["s"])]
    for sid in lst_sid:
        dk_sid = kg.get_decryption_keys(sid, credentials={"fusion_weight": credentials})
        assert "v0" in dk_sid
        assert "v1" in dk_sid
        assert isinstance(dk_sid["v1"], dict)
        assert len(dk_sid["v1"]) == kg_config["n"]


def test_threshold_mife(kg_config):
    kg = ThresholdMIFEKeyGenerator(kg_config)
    kg.setup()

    precision = 3
    lst_nid = kg_config["lst_nid"]
    lst_sid = kg_config["lst_sid"]
    threshold = kg_config["t"]

    # prepare the input
    dict_tmife_n_x = {}
    dict_tmife_n_y = {}
    for nid in lst_nid:
        dict_tmife_n_x[nid] = [random.random()]
        dict_tmife_n_y[nid] = [random.randint(1, 10)]
    logger.info("x: {}".format(list(dict_tmife_n_x.values())))
    logger.info("y: {}".format(list(dict_tmife_n_y.values())))
    lst_x = list(dict_tmife_n_x.values())
    lst_y = list(dict_tmife_n_y.values())
    expected_inner_prod = sum(lst_x[i][0] * lst_y[i][0] for i in range(len(lst_y)))
    logger.info("<x,y>: {}".format(expected_inner_prod))

    # encryption
    dict_tmife_n = {}
    dict_tmife_n_ct = {}
    for nid in lst_nid:
        pp = kg.get_public_parameters()
        sk_nid = kg.get_private_keys(nid)
        config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": pp, "sk": sk_nid},
        }
        tmife_nid = ThresholdMIFE(config_nid)
        ct_nid = tmife_nid.encrypt(dict_tmife_n_x[nid])
        assert isinstance(ct_nid, dict)
        assert "ct0" in ct_nid
        assert "ct1" in ct_nid
        dict_tmife_n_ct[nid] = ct_nid
        dict_tmife_n[nid] = tmife_nid

    # share decryption
    lst_sid_enrolled = lst_sid.copy()[:4]
    dict_tmife_s_ct = {}
    for sid in lst_sid_enrolled:
        pp = kg.get_public_parameters()
        dk_sid = kg.get_decryption_keys(
            sid, credentials={"fusion_weight": dict_tmife_n_y}
        )
        config_sid = {
            "id": sid,
            "precision": precision,
            "keys": {"pp": pp, "dk": dk_sid},
        }
        tmife_sid = ThresholdMIFE(config_sid)
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
