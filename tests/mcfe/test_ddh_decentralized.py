import os
import random
import pytest
import logging

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFE
from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFEKeyGenerator
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FESchemeError
from tests.log_config import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    sec_param = 128
    n = 3
    s = 1
    _config = {
        "sec_param": sec_param,
        "lst_nid": ["nid_{}".format(i) for i in range(n)],
        "eta": 1,
        "n": n,
        "s": s,
    }
    return _config


def test_key_generator(kg_config):
    kg = DecentralizedMCFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.eta == kg_config["eta"]
    assert kg.lst_nid == kg_config["lst_nid"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_DMCFE)
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    lst_nid_expected = ["nid_{}".format(i) for i in range(kg_config["n"])]
    assert kg.lst_nid == lst_nid_expected

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.p) == pp["p"]
    assert gp.digits(kg.g) == pp["g"]

    for nid in lst_nid_expected:
        sk_nid = kg.get_private_keys(nid)
        assert set(sk_nid) == {"s", "s2", "mask_key"}
        assert len(sk_nid["s2"]) == 2
    # the MCFE2 keys of all clients sum to zero (ABMPSW20 Fig. 14)
    assert all(sum(col) == 0 for col in zip(*kg.msk["s2"].values()))


def test_entire_process(kg_config):
    lst_nid = kg_config["lst_nid"]

    precision = 3
    label = "test-label"

    kg = DecentralizedMCFEKeyGenerator(kg_config)
    kg.setup()

    # prepare input
    dct_x, dct_y = {}, {}
    for nid in lst_nid:
        sign = 1 if random.random() < 0.5 else -1
        dct_x[nid] = [sign * random.randint(1, 10)]
        dct_y[nid] = [random.randint(1, 10)]
    lst_x = list(dct_x.values())
    lst_y = list(dct_y.values())
    expected_inner_prod = sum(lst_x[i][0] * lst_y[i][0] for i in range(len(lst_x)))

    # encryption
    dct_ct = {}
    dct_dk_shares = {}
    _pp = kg.get_public_parameters()
    for nid in lst_nid:
        _sk_nid = kg.get_private_keys(nid)
        _config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": _pp, "sk": _sk_nid},
        }
        crypto_nid = DecentralizedMCFE(_config_nid)
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)
        assert isinstance(dct_ct[nid], dict)
        assert set(dct_ct[nid]) == {"c"}
        assert set(dct_dk_shares[nid]) == {"c", "nid", "weights_id"}

    # decryption
    _config_sid = {
        "id": "sid_0",
        "precision": precision,
        "keys": {"pp": _pp},
    }
    crypto_sid = DecentralizedMCFE(_config_sid)
    dk = crypto_sid.combine_function_decryption_key_share(dct_dk_shares)
    assert set(dk) == {"d"} and len(dk["d"]) == 2

    computed_inner_prod = crypto_sid.decrypt(dct_ct, dk, dct_y, label)
    logger.info(
        "<{},{}> expected inner-prod = {}".format(lst_x, lst_y, expected_inner_prod)
    )
    logger.info(
        "<{},{}> computed inner-prod = {}".format(lst_x, lst_y, computed_inner_prod)
    )
    assert computed_inner_prod == expected_inner_prod


def test_get_decryption_keys_raises_not_implemented(kg_config):
    """Regression (N12): the method used to *return* NotImplementedError."""
    kg = DecentralizedMCFEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})


def _shares(kg, pp, dct_y, lst_nid):
    shares = {}
    for nid in lst_nid:
        client = DecentralizedMCFE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        shares[nid] = client.derive_function_decryption_key_share(dct_y)
    return shares


def test_combined_key_equals_the_central_cdgpp18_key(kg_config):
    """Fig. 14 yields exactly d = sum y_i s_i mod q, the CDGPP18 functional key."""
    cfg = dict(kg_config, eta=2)
    kg = DecentralizedMCFEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    lst_nid = cfg["lst_nid"]
    dct_y = {nid: [random.randint(-5, 5), random.randint(-5, 5)] for nid in lst_nid}
    combiner = DecentralizedMCFE({"id": "agg", "keys": {"pp": pp}})
    dk = combiner.combine_function_decryption_key_share(_shares(kg, pp, dct_y, lst_nid))
    q = kg.q
    expected = [sum(y * s[k] for nid in lst_nid for y, s in zip(dct_y[nid], kg.msk["s"][nid])) % q for k in (0, 1)]
    assert [int(v) for v in dk["d"]] == expected


def test_subset_weights_still_need_every_share(kg_config):
    kg = DecentralizedMCFEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    lst_nid = kg_config["lst_nid"]
    dct_y = {"nid_0": [2], "nid_1": [3]}  # nid_2 absent: contributes zero
    shares = _shares(kg, pp, dct_y, lst_nid)
    combiner = DecentralizedMCFE({"id": "agg", "keys": {"pp": pp}})
    with pytest.raises(FESchemeError):
        combiner.combine_function_decryption_key_share({k: shares[k] for k in ["nid_0", "nid_1"]})
    dk = combiner.combine_function_decryption_key_share(shares)
    dct_ct = {
        nid: DecentralizedMCFE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([4], "round")
        for nid in ["nid_0", "nid_1"]
    }
    assert combiner.decrypt(dct_ct, dk, dct_y, "round") == 20


def test_shares_for_other_weights_do_not_combine(kg_config):
    kg = DecentralizedMCFEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    lst_nid = kg_config["lst_nid"]
    a = _shares(kg, pp, {nid: [1] for nid in lst_nid}, lst_nid)
    b = _shares(kg, pp, {nid: [2] for nid in lst_nid}, lst_nid)
    mixed = dict(a, nid_2=b["nid_2"])
    combiner = DecentralizedMCFE({"id": "agg", "keys": {"pp": pp}})
    with pytest.raises(FESchemeError):
        combiner.combine_function_decryption_key_share(mixed)
    mixed["nid_2"] = dict(b["nid_2"], weights_id=a["nid_0"]["weights_id"])  # forged digest
    with pytest.raises(FESchemeError):
        combiner.combine_function_decryption_key_share(mixed)
