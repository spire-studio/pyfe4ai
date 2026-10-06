import os
import random
import pytest
import logging

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh import MCFE
from pyfe4ai.schemes.mcfe.ddh import MCFEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
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
    kg = MCFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.eta == kg_config["eta"]
    assert kg.lst_nid == kg_config["lst_nid"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_MCFE)
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
        assert set(sk_nid) == {"s"}
        assert len(sk_nid["s"]) == kg_config["eta"]
        assert all(len(s_j) == 2 for s_j in sk_nid["s"])

    fusion_weight = {
        nid: [1 for _ in range(kg_config["eta"])] for nid in lst_nid_expected
    }
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert set(dk) == {"d"}
    assert isinstance(dk["d"], list) and len(dk["d"]) == 2


def test_entire_process(kg_config):
    lst_nid = kg_config["lst_nid"]

    precision = 3
    label = "test-label"

    kg = MCFEKeyGenerator(kg_config)
    kg.setup()

    # prepare input
    dct_x, dct_y = {}, {}
    for nid in lst_nid:
        sign = 1 if random.random() < 0.5 else -1
        dct_x[nid] = [sign * random.randint(1, 10)]
        dct_y[nid] = [1]
    lst_x = list(dct_x.values())
    lst_y = list(dct_y.values())
    expected_inner_prod = sum(lst_x[i][0] * lst_y[i][0] for i in range(len(lst_x)))

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
        crypto_nid = MCFE(_config_nid)
        ct_nid = crypto_nid.encrypt(dct_x[nid], label)
        assert isinstance(ct_nid, dict)
        assert "c" in ct_nid
        dct_ct[nid] = ct_nid

    # decryption
    _config_sid = {
        "id": "sid_0",
        "precision": precision,
        "keys": {"pp": _pp},
    }
    crypto_sid = MCFE(_config_sid)
    dk = kg.get_decryption_keys(_config_sid["id"], credentials={"fusion_weight": dct_y})
    computed_inner_prod = crypto_sid.decrypt(dct_ct, dk, dct_y, label)
    logger.info(
        "<{},{}> expected inner-prod = {}".format(lst_x, lst_y, expected_inner_prod)
    )
    logger.info(
        "<{},{}> computed inner-prod = {}".format(lst_x, lst_y, computed_inner_prod)
    )
    assert computed_inner_prod == expected_inner_prod


def _run_full_vector_round(n, eta, precision=3, label="test-label-full"):
    lst_nid = ["nid_{}".format(i) for i in range(n)]
    kg = MCFEKeyGenerator(
        {"sec_param": 256, "lst_nid": lst_nid, "eta": eta, "n": n, "s": 1}
    )
    kg.setup()
    pp = kg.get_public_parameters()

    dct_x = {nid: [random.randint(-10, 10) for _ in range(eta)] for nid in lst_nid}
    dct_y = {nid: [random.randint(1, 5) for _ in range(eta)] for nid in lst_nid}
    expected = sum(
        x_i * y_i for nid in lst_nid for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )

    dct_ct = {}
    for nid in lst_nid:
        crypto_nid = MCFE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        assert len(dct_ct[nid]["c"]) == eta

    crypto_sid = MCFE({"id": "sid_0", "precision": precision, "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    return expected, crypto_sid.decrypt(dct_ct, dk, dct_y, label)


@pytest.mark.parametrize("n, eta", [(2, 2), (3, 3), (2, 4)])
def test_entire_process_full_length_vectors(n, eta):
    """Regression (N1): eta-length plaintext vectors must decrypt for eta >= 2.

    Before the fix every slot's ciphertext was masked with the product of all
    ``w[i]`` components, so decryption returned ``None`` whenever more than
    one slot was populated.
    """
    for _ in range(3):
        expected, computed = _run_full_vector_round(n, eta)
        assert computed == expected


def test_client_keys_are_independent():
    """Regression (N6): every client used to receive the same ``w`` / ``g_a``,
    so one client could strip the mask of another client's ciphertext."""
    lst_nid = ["nid_0", "nid_1", "nid_2"]
    kg = MCFEKeyGenerator(
        {"sec_param": 256, "lst_nid": lst_nid, "eta": 2, "n": 3, "s": 1}
    )
    kg.setup()
    keys = [kg.get_private_keys(nid)["s"] for nid in lst_nid]
    flat = [e for key in keys for s_j in key for e in s_j]
    assert len(set(flat)) == len(flat)


def test_decryption_key_is_aggregated():
    """Regression (N2): the functional key used to contain one component per
    client (``d[nid]``), which let its holder strip each client's mask and,
    with two labels, solve for the client's plaintexts. The key must be a
    single aggregated Z_q^2 element that is identical for any split of the
    same weights into clients."""
    lst_nid = ["nid_0", "nid_1"]
    kg = MCFEKeyGenerator(
        {"sec_param": 256, "lst_nid": lst_nid, "eta": 1, "n": 2, "s": 1}
    )
    kg.setup()
    dk = kg.get_decryption_keys(
        "sid_0", credentials={"fusion_weight": {"nid_0": [3], "nid_1": [5]}}
    )
    assert set(dk) == {"d"} and len(dk["d"]) == 2
    s0 = [int(v) for v in kg.get_private_keys("nid_0")["s"][0]]
    s1 = [int(v) for v in kg.get_private_keys("nid_1")["s"][0]]
    q = int(kg.q)
    assert [int(v) for v in dk["d"]] == [(3 * a + 5 * b) % q for a, b in zip(s0, s1)]


def test_label_binding_and_label_mask_is_not_a_public_scalar():
    """Regression (N2): ciphertexts must only decrypt under their own label,
    and the label mask must not cancel by combining two labels' ciphertexts
    with public exponents (the old ``u * MD5(label)`` mask did)."""
    import gmpy2 as gp

    from pyfe4ai.utils.crypto_utils import md5_hash

    lst_nid = ["nid_0", "nid_1"]
    kg = MCFEKeyGenerator(
        {"sec_param": 256, "lst_nid": lst_nid, "eta": 1, "n": 2, "s": 1}
    )
    kg.setup()
    pp = kg.get_public_parameters()
    p = gp.mpz(pp["p"])
    g = gp.mpz(pp["g"])
    clients = {
        nid: MCFE({"id": nid, "precision": 3, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        for nid in lst_nid
    }
    dct_y = {nid: [1] for nid in lst_nid}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    decryptor = MCFE({"id": "sid_0", "precision": 3, "keys": {"pp": pp}})

    ct_a = {nid: clients[nid].encrypt([7], "label-a") for nid in lst_nid}
    assert decryptor.decrypt(ct_a, dk, dct_y, "label-a") == 14
    assert decryptor.decrypt(ct_a, dk, dct_y, "label-b") is None

    # old attack: c_a^{H(b)} / c_b^{H(a)} = g^{x_a H(b) - x_b H(a)} for the
    # client's own masks; with a group-element H(label) this no longer holds
    c_a = gp.mpz(ct_a["nid_0"]["c"][0])
    c_b = gp.mpz(clients["nid_0"].encrypt([7], "label-b")["c"][0])
    h_a, h_b = md5_hash("label-a", p), md5_hash("label-b", p)
    combined = gp.divm(gp.powmod(c_a, h_b, p), gp.powmod(c_b, h_a, p), p)
    assert combined != gp.powmod(g, 7 * (h_b - h_a), p)


def test_unknown_client_in_fusion_weight_rejected():
    from pyfe4ai.utils.exceptions import FEValidationError

    kg = MCFEKeyGenerator(
        {"sec_param": 256, "lst_nid": ["nid_0", "nid_1"], "eta": 1, "n": 2, "s": 1}
    )
    kg.setup()
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(
            "sid_0", credentials={"fusion_weight": {"nid_0": [1], "nid_9": [1]}}
        )
