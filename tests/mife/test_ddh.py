import os
import random
import pytest
import logging

import gmpy2 as gp

from pyfe4ai.schemes.mife.ddh import MIFE
from pyfe4ai.schemes.mife.ddh import MIFEKeyGenerator
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture(params=[(3, 1), (2, 2)], ids=["n=3,eta=1", "n=2,eta=2"])
def kg_config(request):
    n, eta = request.param
    return {
        "sec_param": 256,
        "lst_nid": ["nid_{}".format(i) for i in range(n)],
        "eta": eta,
        "n": n,
        "s": 1,
    }


def test_key_generator(kg_config):
    kg = MIFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.eta == kg_config["eta"]
    assert kg.lst_nid == kg_config["lst_nid"]

    config_folder = os.path.join("config", "authority", "MIFE")
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
        assert "w" in sk_nid
        assert "u" in sk_nid

    fusion_weight = {
        nid: [1 for _ in range(kg_config["eta"])] for nid in lst_nid_expected
    }
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "d" in dk
    assert "z" in dk
    assert isinstance(dk["d"], dict)
    assert isinstance(dk["z"], str)


def test_entire_process(kg_config):
    lst_nid = kg_config["lst_nid"]

    precision = 3

    kg = MIFEKeyGenerator(kg_config)
    kg.setup()

    # prepare input: full eta-length vectors so the multi-element encrypt
    # path is exercised (regression for eta >= 2, see C1).
    eta = kg_config["eta"]
    dct_x, dct_y = {}, {}
    for nid in lst_nid:
        sign = 1 if random.random() < 0.5 else -1
        dct_x[nid] = [sign * random.randint(1, 10) for _ in range(eta)]
        dct_y[nid] = [1 for _ in range(eta)]
    lst_x = list(dct_x.values())
    lst_y = list(dct_y.values())
    expected_inner_prod = sum(
        dct_x[nid][k] * dct_y[nid][k] for nid in lst_nid for k in range(eta)
    )

    # encryption
    dct_ct = {}
    _pp = kg.get_public_parameters()
    for nid in lst_nid:
        _sk_nid = kg.get_private_keys(nid)
        _config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": _pp, "sk": _sk_nid},
        }
        crypto_nid = MIFE(_config_nid)
        ct_nid = crypto_nid.encrypt(dct_x[nid])
        assert isinstance(ct_nid, dict)
        assert "t" in ct_nid
        assert "c" in ct_nid
        dct_ct[nid] = ct_nid

    # decryption
    _config_sid = {
        "id": "sid_0",
        "precision": precision,
        "keys": {"pp": _pp},
    }
    crypto_sid = MIFE(_config_sid)
    dk = kg.get_decryption_keys(_config_sid["id"], credentials={"fusion_weight": dct_y})
    computed_inner_prod = crypto_sid.decrypt(dct_ct, dk, dct_y)
    logger.info(
        "<{},{}> expected inner-prod = {}".format(lst_x, lst_y, expected_inner_prod)
    )
    logger.info(
        "<{},{}> computed inner-prod = {}".format(lst_x, lst_y, computed_inner_prod)
    )
    assert computed_inner_prod == expected_inner_prod


def test_client_keys_are_independent():
    """Regression (N6): every client used to receive the same ``w``, so a
    client could remove the mask of another client's ciphertext and learn
    differences of that client's plaintexts."""
    lst_nid = ["nid_0", "nid_1", "nid_2"]
    kg = MIFEKeyGenerator(
        {"sec_param": 256, "lst_nid": lst_nid, "eta": 2, "n": 3, "s": 1}
    )
    kg.setup()
    keys = [kg.get_private_keys(nid)["w"] for nid in lst_nid]
    flat = [e for key in keys for w_i in key for e in w_i]
    assert len(set(flat)) == len(flat)


def test_client_cannot_unmask_other_client():
    """With its own key, client A must not be able to strip the IPFE mask
    of client B's ciphertext (worked with the shared ``w``)."""
    lst_nid = ["nid_0", "nid_1"]
    kg = MIFEKeyGenerator(
        {"sec_param": 256, "lst_nid": lst_nid, "eta": 1, "n": 2, "s": 1}
    )
    kg.setup()
    pp = kg.get_public_parameters()
    p = gp.mpz(pp["p"])
    g = gp.mpz(pp["g"])
    sk_a = kg.get_private_keys("nid_0")
    sk_b = kg.get_private_keys("nid_1")
    ct_b = MIFE({"id": "nid_1", "precision": 3, "keys": {"pp": pp, "sk": sk_b}}).encrypt([5])

    w_a = gp.mpz(sk_a["w"][0][0]) + gp.mpz(sk_a["w"][0][1])
    mask_guess = gp.mpz(1)
    for t in ct_b["t"]:
        mask_guess = mask_guess * gp.powmod(gp.mpz(t), w_a, p) % p
    unmasked = gp.divm(gp.mpz(ct_b["c"][0]), mask_guess, p)
    assert unmasked != gp.powmod(g, 5 + gp.mpz(sk_b["u"][0]), p)
