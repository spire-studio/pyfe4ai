import os
import random
import logging

import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FESchemeError

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
    }


def test_key_generator(kg_config):
    kg = ThresholdMIFELWEKeyGenerator(kg_config)
    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.t == kg_config["t"]
    assert kg.eta == kg_config["eta"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_TMIFE_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["t"] == kg_config["t"]
    assert pp["s"] == kg_config["s"]
    assert pp["lst_sid"] == kg_config["lst_sid"]

    sk_nid = kg.get_private_keys("nid_0")
    assert set(sk_nid) == {"pk", "u", "mac_keys"}
    assert set(sk_nid["mac_keys"]) == set(kg_config["lst_sid"])

    fusion_weight = {nid: [1] for nid in kg_config["lst_nid"]}
    dk_sid = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": fusion_weight})
    assert set(dk_sid) == {"sk_y", "z", "sid", "lst_sid", "weights_id", "noise_key", "mac_keys"}
    assert set(dk_sid["sk_y"]) == set(kg_config["lst_nid"])
    # server k holds client i's key for the tags meant for k, and only those
    assert dk_sid["mac_keys"] == {
        nid: kg.get_private_keys(nid)["mac_keys"]["sid_0"] for nid in kg_config["lst_nid"]
    }


def test_entire_process(kg_config):
    kg = ThresholdMIFELWEKeyGenerator(kg_config)
    kg.setup()

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
        enc = ThresholdMIFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = enc.encrypt(dct_x[nid])
        encryptors[nid] = enc

    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dct_ct_prime = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials={"fusion_weight": dct_y})
        dec = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk_sid}})
        dct_ct_prime[sid] = dec.share_decrypt(dct_ct, dct_y, dk_sid, lst_sid_enrolled)

    computed = encryptors[lst_nid[0]].combine_decrypt(dct_ct_prime)
    assert abs(computed - expected) < 1e-6


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["bound_x"] = 50

    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
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
                    (dct_x[nid][layer_idx] * pow(10, precision)).astype(int)
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
        enc = ThresholdMIFELWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid])
        encryptors[nid] = enc

    lst_sid_enrolled = random.sample(lst_sid, threshold)
    dct_ct_prime = {}
    for sid in lst_sid_enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials={"fusion_weight": dct_y})
        dec = ThresholdMIFELWE(
            {"id": sid, "precision": precision, "keys": {"pp": pp, "dk": dk_sid}}
        )
        dct_ct_prime[sid] = dec.compute_lst_ndarray_ct(
            dct_ct, dct_y, dk_sid, lst_sid_enrolled
        )

    computed = encryptors[lst_nid[0]].decrypt_lst_ndarray_ct(dct_ct_prime)
    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def _round(cfg, dct_x, dct_y, enrolled):
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_ct = {
        nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(dct_x[nid])
        for nid in dct_x
    }
    partials = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": dct_y})
        server = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        partials[sid] = server.share_decrypt(dct_ct, dct_y, dk, enrolled)
    return kg, pp, dct_ct, partials, ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}})


@pytest.mark.parametrize(
    "s, t, enrolled",
    [(3, 2, ["sid_1", "sid_2"]), (4, 3, ["sid_3", "sid_0", "sid_2"]), (4, 2, ["sid_0", "sid_1", "sid_2", "sid_3"])],
)
def test_any_enrolled_set_decrypts_extreme_values(kg_config, s, t, enrolled):
    cfg = dict(kg_config, s=s, t=t, lst_sid=["sid_{}".format(i) for i in range(s)])
    bx, by = cfg["bound_x"], cfg["bound_y"]
    for sign in (1, -1):
        dct_x = {nid: [sign * bx] for nid in cfg["lst_nid"]}
        dct_y = {nid: [by] for nid in cfg["lst_nid"]}
        _, _, _, partials, combiner = _round(cfg, dct_x, dct_y, enrolled)
        assert combiner.combine_decrypt(partials) == sign * 3 * bx * by


def test_partials_are_flooded_and_derandomised(kg_config):
    import gmpy2 as gp

    from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
    from pyfe4ai.utils.matrix_utils import digits_to_vector, vecdot_mod

    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, combiner = _round(kg_config, dct_x, dct_y, enrolled)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    assert server.share_decrypt(dct_ct, dct_y, dk, enrolled) == partials["sid_0"]

    q = gp.mpz(pp["q"])
    delta = q // gp.mpz(pp["p"])
    exact = delta * gp.mpz(dk["z"])
    for nid, ct in dct_ct.items():
        exact += vecdot_mod(digits_to_vector(ct["ct0"]), [gp.mpz(v) for v in dk["sk_y"][nid]], q)
    exact = lagrange_coefficient_at_zero(1, [1, 2], q) * exact % q
    noise = (gp.mpz(partials["sid_0"]["w"]) - exact) % q
    assert noise != 0 and min(noise, q - noise) <= 12 * int(pp["sigma_f"])
    assert combiner.combine_decrypt(partials) == 3


def test_single_server_and_bad_partials_rejected(kg_config):
    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    kg, pp, dct_ct, partials, combiner = _round(kg_config, dct_x, dct_y, ["sid_0", "sid_1", "sid_2"])
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt({"sid_0": partials["sid_0"]})
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt(dict(partials, extra=partials["sid_0"]))
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError):
        server.share_decrypt(dct_ct, {nid: [2] for nid in dct_y}, dk, ["sid_0", "sid_1"])


def test_m_leaves_room_for_the_leftover_hash_lemma_at_the_enlarged_q(kg_config, tmp_path, monkeypatch):
    # ABDP15 needs A^T r close to uniform for a binary r of length m:
    # m >= (lwe_n + eta + 1) log2 q + 2 lwe_n + 1 (the recipe of derive_lwe_parameters),
    # for the q that is actually used, not the one before the flooding enlarged it
    monkeypatch.chdir(tmp_path)
    kg = ThresholdMIFELWEKeyGenerator(kg_config)
    need = (kg.lwe_n + kg.eta + 1) * int(kg.q).bit_length() + 2 * kg.lwe_n + 1
    assert kg.m >= need


def test_servers_refuse_ciphertexts_not_made_by_the_clients(kg_config):
    """Regression: a partial decryption is linear in ``ct0``, so a combiner that
    could submit ``ct0 = c e_j`` would read ``sk_y`` off the partials coordinate
    by coordinate (the flooding hides a fixed noise, not a chosen multiple of
    the key). Servers therefore only accept ciphertexts tagged by the clients."""
    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, combiner = _round(kg_config, dct_x, dct_y, enrolled)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    authentic = dct_ct["nid_0"]
    unit = ["1"] + ["0"] * (len(authentic["ct0"]) - 1)
    forgeries = {
        "chosen ct0, authentic tags": dict(authentic, ct0=unit),
        "chosen ct0, no tags": {"ct0": unit, "ct1": authentic["ct1"]},
        "re-encoded ct1": dict(authentic, ct1=[str(int(v) + int(pp["q"])) for v in authentic["ct1"]]),
        "tag for another server": dict(authentic, tags=dict(authentic["tags"], sid_0=authentic["tags"]["sid_1"])),
        "another client's ciphertext": dct_ct["nid_1"],
    }
    for name, forged in forgeries.items():
        with pytest.raises(FESchemeError, match="not authenticated"):
            server.share_decrypt(dict(dct_ct, nid_0=forged), dct_y, dk, enrolled)
    assert combiner.combine_decrypt(partials) == 3


def test_encryption_needs_the_mac_keys(kg_config):
    from pyfe4ai.utils.exceptions import FEKeyError

    kg = ThresholdMIFELWEKeyGenerator(kg_config)
    kg.setup()
    sk = kg.get_private_keys("nid_0")
    del sk["mac_keys"]
    enc = ThresholdMIFELWE({"id": "nid_0", "keys": {"pp": kg.get_public_parameters(), "sk": sk}})
    with pytest.raises(FEKeyError, match="MAC"):
        enc.encrypt([1])


def test_noise_does_not_depend_on_how_the_request_is_dressed(kg_config):
    """Regression: the flooding noise was seeded with the JSON of the whole
    request, so an extra field or another server's tag drew fresh noise for
    the same ciphertexts, and averaging such partials removes the noise."""
    dct_x = {nid: [1] for nid in kg_config["lst_nid"]}
    dct_y = {nid: [1] for nid in kg_config["lst_nid"]}
    enrolled = ["sid_0", "sid_1"]
    kg, pp, dct_ct, partials, _ = _round(kg_config, dct_x, dct_y, enrolled)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    first = dct_ct["nid_0"]
    for dressed in (
        dict(dct_ct, nid_0=dict(first, padding="1")),
        dict(dct_ct, nid_0=dict(first, tags=dict(first["tags"], sid_2="00"))),
        dict(reversed(list(dct_ct.items()))),
    ):
        assert server.share_decrypt(dressed, dct_y, dk, enrolled) == partials["sid_0"]
