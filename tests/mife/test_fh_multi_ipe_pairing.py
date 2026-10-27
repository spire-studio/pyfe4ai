import random

import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEPairing
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEPairingKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 4,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 5,
        "bound_y": 5,
        "pairing_group_param": GROUP,
    }


def test_entire_process(kg_config):
    kg = MIFEFHMultiIPEPairingKeyGenerator(kg_config)
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
    pp["lst_nid"] = kg_config["lst_nid"]
    dct_ct = {}
    for nid in kg_config["lst_nid"]:
        encryptor = MIFEFHMultiIPEPairing(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid])

    decryptor = MIFEFHMultiIPEPairing({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = MIFEFHMultiIPEPairingKeyGenerator(cfg)
    kg.setup()

    precision = 1
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
        crypto_nid = MIFEFHMultiIPEPairing(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid])

    decryptor = MIFEFHMultiIPEPairing(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y)

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def _setup(cfg):
    kg = MIFEFHMultiIPEPairingKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: MIFEFHMultiIPEPairing({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in cfg["lst_nid"]}
    return kg, pp, clients


@pytest.mark.parametrize("k", [1, 2])
def test_slots_masked_keys_unlinkable_and_no_y_needed(kg_config, k):
    from pyfe4ai.core.exceptions import FESchemeError
    from pyfe4ai.primitives.pairing.pairing_backend import GT, pair
    from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt

    cfg = dict(kg_config, sec_level=k, eta=2)
    kg, pp, clients = _setup(cfg)
    x = {"nid_0": [5, 0], "nid_1": [-5, 0], "nid_2": [0, 0]}
    y1 = {nid: [1, 0] for nid in cfg["lst_nid"]}
    y2 = {nid: [0, 1] for nid in cfg["lst_nid"]}
    cts = {nid: c.encrypt(x[nid]) for nid, c in clients.items()}
    dec = MIFEFHMultiIPEPairing({"id": "sid", "keys": {"pp": pp}})
    dk1 = kg.get_decryption_keys("sid", credentials={"fusion_weight": y1})
    dk2 = kg.get_decryption_keys("sid", credentials={"fusion_weight": y2})
    assert dec.decrypt(cts, dk1) == 0
    assert set(dk1) == {"dk"}  # group elements only, nothing derived from y
    assert kg.get_decryption_keys("sid", credentials={"fusion_weight": y1})["dk"] != dk1["dk"]

    g = dec.group
    pub = g.deserialize(pp["pub"].encode())
    slot0 = g.init(GT, 1)
    for c, k_ in zip(cts["nid_0"]["ct"], dk1["dk"]["nid_0"]):
        slot0 *= pair(g.deserialize(c.encode()), g.deserialize(k_.encode()))
    with pytest.raises(ValueError):
        bounded_discrete_log_gt(g, pub, slot0, 1000)

    mixed = {"dk": {"nid_0": dk1["dk"]["nid_0"], "nid_1": dk2["dk"]["nid_1"], "nid_2": dk1["dk"]["nid_2"]}}
    assert dec.decrypt(cts, mixed) is None
    with pytest.raises(FESchemeError):
        dec.decrypt({k_: v for k_, v in cts.items() if k_ != "nid_2"}, dk1)


def test_parameter_guards(kg_config):
    from pyfe4ai.core.exceptions import FEKeyError, FEValidationError

    with pytest.raises(FEKeyError):
        MIFEFHMultiIPEPairingKeyGenerator(dict(kg_config, n=1, lst_nid=["nid_0"], sec_level=1))
    with pytest.raises(FEValidationError):
        MIFEFHMultiIPEPairingKeyGenerator(dict(kg_config, sec_level=0))
    with pytest.raises(FEValidationError):
        MIFEFHMultiIPEPairingKeyGenerator(dict(kg_config, lst_nid=["nid_0", "nid_1"]))
    kg, pp, clients = _setup(kg_config)
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1.5, 0, 0, 0])
