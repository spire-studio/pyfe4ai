import random

import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPEKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "pairing_group_param": GROUP,
    }


def test_entire_process(kg_config):
    label = "dmcfe-fhmultiipe"
    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(kg_config)
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
    dct_dk_shares = {}
    for nid in kg_config["lst_nid"]:
        crypto_nid = DecentralizedMCFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y[nid], "weights-1")

    decryptor = DecentralizedMCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "dmcfe-fh-multi-ipe-ndarray"
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
    dct_dk_shares = {}
    for nid in cfg["lst_nid"]:
        crypto_nid = DecentralizedMCFEFHMultiIPE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y[nid], "weights-1")

    decryptor = DecentralizedMCFEFHMultiIPE(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def test_get_decryption_keys_raises_not_implemented(kg_config):
    """Regression (N12): the method used to *return* NotImplementedError."""
    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})


def _clients(kg, cfg):
    pp = kg.get_public_parameters()
    return pp, {
        nid: DecentralizedMCFEFHMultiIPE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        for nid in cfg["lst_nid"]
    }


def test_key_share_is_deterministic_and_bound(kg_config):
    from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(kg_config)
    kg.setup()
    pp, clients = _clients(kg, kg_config)
    y1 = {"nid_0": [1, 2], "nid_1": [0, 1], "nid_2": [-1, 3]}
    y2 = {"nid_0": [1, 2], "nid_1": [0, 1], "nid_2": [-1, 2]}
    c0 = clients["nid_0"]
    # asking twice yields the same key; the share depends on (tag_f, y_0) only
    # (NPS24 DKeyGen), so the weights of the other clients do not matter ...
    assert c0.derive_function_decryption_key_share(y1, "f") == c0.derive_function_decryption_key_share(y1, "f")
    assert c0.derive_function_decryption_key_share(y1, "f") == c0.derive_function_decryption_key_share([1, 2], "f")
    assert c0.derive_function_decryption_key_share(y2, "f") == c0.derive_function_decryption_key_share(y1, "f")
    # ... while another tag gives a different key
    assert c0.derive_function_decryption_key_share(y1, "f")["k"] != c0.derive_function_decryption_key_share(y1, "g")["k"]
    # NPS24 allows one key per client and tag-f
    with pytest.raises(FESchemeError, match="tag_f"):
        c0.derive_function_decryption_key_share([1, 1], "f")
    with pytest.raises(TypeError):
        c0.derive_function_decryption_key_share(y1)
    with pytest.raises(FEValidationError):
        c0.derive_function_decryption_key_share(y1, "")

    decryptor = DecentralizedMCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp}})
    # shares carry nothing derived from y (function hiding) ...
    share = c0.derive_function_decryption_key_share(y1, "f")
    assert set(share) == {"k", "z", "nid", "tag_f"}
    # ... each client's share encodes its own weights, so the shares of one
    # tag_f give the key of the weights the clients chose
    keys = {nid: c.derive_function_decryption_key_share(y2, "f") for nid, c in clients.items()}
    cts = {nid: c.encrypt([1, 1], "t") for nid, c in clients.items()}
    assert decryptor.decrypt(cts, decryptor.combine_function_decryption_key_share(keys)) == 3 + 1 + 1
    other_tag = dict(keys, nid_2=clients["nid_2"].derive_function_decryption_key_share(y1, "x"))
    with pytest.raises(FESchemeError):
        decryptor.combine_function_decryption_key_share(other_tag)
    partial = {nid: k for nid, k in keys.items() if nid != "nid_1"}
    with pytest.raises(FESchemeError):
        decryptor.combine_function_decryption_key_share(partial)


def test_a_corrupted_client_cannot_test_guesses_of_the_honest_weights():
    """Regression: tag-f was (tag_f, H(y)), so a client's key share depended on
    every client's weights. A corrupted client derived its own share for a
    guess of the honest clients' weights and combined it with their shares:
    decryption succeeded only for the right guess. With NPS24's explicit tag-f
    its share depends on its own weights only, whatever it guesses."""
    import itertools

    cfg = {"n": 3, "lst_nid": ["a", "b", "c"], "eta": 1, "bound_x": 3, "bound_y": 1,
           "pairing_group_param": GROUP}
    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()
    pp, clients = _clients(kg, cfg)
    x = {"a": [2], "b": [-3], "c": [1]}
    cts = {nid: c.encrypt(x[nid], "round-1") for nid, c in clients.items()}
    secret = {"a": [1], "b": [-1], "c": [1]}
    honest = {nid: clients[nid].derive_function_decryption_key_share(secret, "f") for nid in ("a", "b")}
    decryptor = DecentralizedMCFEFHMultiIPE({"id": "d", "keys": {"pp": pp}})
    results = set()
    for ya, yb in itertools.product([-1, 0, 1], repeat=2):
        guess = {"a": [ya], "b": [yb], "c": secret["c"]}
        own = clients["c"].derive_function_decryption_key_share(guess, "f")
        dk = decryptor.combine_function_decryption_key_share(dict(honest, c=own))
        results.add(decryptor.decrypt(cts, dk))
    assert results == {2 + 3 + 1}


def test_wrong_tag_does_not_decrypt(kg_config):
    kg = DecentralizedMCFEFHMultiIPEKeyGenerator(kg_config)
    kg.setup()
    pp, clients = _clients(kg, kg_config)
    x = {"nid_0": [1, -2], "nid_1": [0, 3], "nid_2": [2, 1]}
    y = {"nid_0": [2, 1], "nid_1": [-1, 2], "nid_2": [1, -2]}
    dct_ct = {nid: c.encrypt(x[nid], "round-1") for nid, c in clients.items()}
    decryptor = DecentralizedMCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(
        {nid: c.derive_function_decryption_key_share(y[nid], "f") for nid, c in clients.items()}
    )
    assert decryptor.decrypt(dct_ct, dk, y, "round-1") == (2 - 2) + (0 + 6) + (2 - 2)
    # the decryptor does not need the weights (the key encodes them)
    assert decryptor.decrypt(dct_ct, dk) == 6
    # ciphertexts of different tags do not combine
    dct_ct["nid_1"] = clients["nid_1"].encrypt(x["nid_1"], "round-2")
    assert decryptor.decrypt(dct_ct, dk, y, "round-1") is None


def test_symmetric_group_rejected(kg_config):
    pytest.importorskip("charm", reason="SS512 is a charm-crypto group")
    from pyfe4ai.utils.exceptions import FEValidationError

    with pytest.raises(FEValidationError, match="symmetric"):
        DecentralizedMCFEFHMultiIPEKeyGenerator(dict(kg_config, pairing_group_param="SS512"))
