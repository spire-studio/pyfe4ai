import random

import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.mife.fh_ipe_pairing import MIFEFHIPE
from pyfe4ai.schemes.mife.fh_ipe_pairing import MIFEFHIPEKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 4,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 6,
        "bound_y": 6,
        "pairing_group_param": "MNT224",
    }


def test_entire_process(kg_config):
    kg = MIFEFHIPEKeyGenerator(kg_config)
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
        encryptor = MIFEFHIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid])

    decryptor = MIFEFHIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y)
    assert computed == expected


def _round(kg_config, dct_x, dct_y):
    kg = MIFEFHIPEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {
        nid: MIFEFHIPE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(dct_x[nid])
        for nid in kg_config["lst_nid"]
    }
    return kg, pp, cts


def test_slot_results_are_masked_and_keys_do_not_mix(kg_config):
    """v0.2.0 decoded every slot separately (per-slot <x_i, y_i>) and slot keys
    of different functional keys could be recombined."""
    from pyfe4ai.utils.exceptions import FESchemeError
    from pyfe4ai.utils.pairing_backend import GT
    from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt

    nids = kg_config["lst_nid"]
    x = {"nid_0": [5, 0, 0, 0], "nid_1": [-5, 0, 0, 0], "nid_2": [0, 0, 0, 0]}
    y1 = {nid: [1, 0, 0, 0] for nid in nids}
    y2 = {nid: [0, 1, 0, 0] for nid in nids}
    kg, pp, cts = _round(kg_config, x, y1)
    dec = MIFEFHIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk1 = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y1})
    dk2 = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y2})
    assert dec.decrypt(cts, dk1, y1) == 0

    ctx = dec.ctx
    one = ctx.group.init(GT, 1)
    slot0 = ctx.fhipe.decrypt([ctx.de(v) for v in cts["nid_0"]["ct"]], [ctx.de(v) for v in dk1["dk"]["nid_0"]], one)
    with pytest.raises(ValueError):
        bounded_discrete_log_gt(ctx.group, ctx.fhipe.gt, slot0, 1000)

    mixed = {"dk": {"nid_0": dk1["dk"]["nid_0"], "nid_1": dk2["dk"]["nid_1"], "nid_2": dk1["dk"]["nid_2"]},
             "z": dk1["z"], "key_id": dk1["key_id"]}
    assert dec.decrypt(cts, mixed, y1) is None

    with pytest.raises(FESchemeError):
        dec.decrypt({k: v for k, v in cts.items() if k != "nid_2"}, dk1, y1)
    # keys carry no digest of y and are unlinkable; decryption does not need y
    assert set(dk1) == {"dk", "z", "key_id"}
    assert kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y1})["key_id"] != dk1["key_id"]
    assert dec.decrypt(cts, dk1) == 0
    assert "a_out" not in kg.get_private_keys("nid_0")["imsk"]


def test_eta_one_and_zero_weights(kg_config):
    cfg = dict(kg_config, eta=1)
    x = {"nid_0": [3], "nid_1": [-2], "nid_2": [6]}
    y = {"nid_0": [0], "nid_1": [4], "nid_2": [-1]}
    kg, pp, cts = _round(cfg, x, y)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert MIFEFHIPE({"id": "sid_0", "keys": {"pp": pp}}).decrypt(cts, dk, y) == -8 - 6
