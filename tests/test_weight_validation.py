"""Weights and plaintexts must be integers: non-integral floats used to be
truncated silently (0.5 -> 0), and ragged Ring-LWE plaintext rows were cut
to the ring dimension."""

import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import (
    DecentralizedMCFERingLWE,
    DecentralizedMCFERingLWEKeyGenerator,
)
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE, MIFERingLWEKeyGenerator
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.utils.exceptions import FEValidationError

NIDS = ["nid_0", "nid_1"]
CFG = {"sec_param": 64, "eta": 2, "n": 2, "lst_nid": NIDS, "ring_n": 16, "bound_x": 4, "bound_y": 4}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_weight_digest_refuses_fractional_weights():
    assert fusion_weight_id({"a": [2.0, 1]}) == fusion_weight_id({"a": [2, 1]})
    with pytest.raises(FEValidationError):
        fusion_weight_id({"a": [1.5, 1]})


@pytest.mark.parametrize("kg_cls", [MIFERingLWEKeyGenerator, MCFERingLWEKeyGenerator])
def test_ring_keys_refuse_fractional_weights(kg_cls):
    kg = kg_cls(dict(CFG))
    kg.setup()
    credentials = {"fusion_weight": {"nid_0": [0.5, 1], "nid_1": [1, 1]}, "label": "l"}
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("sid", credentials=credentials)


def test_ring_decryption_refuses_fractional_weights():
    kg = MIFERingLWEKeyGenerator(dict(CFG))
    kg.setup()
    pp = kg.get_public_parameters()
    y = {"nid_0": [1, 1], "nid_1": [1, 1]}
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
    cts = {nid: MIFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([[1], [2]])
           for nid in NIDS}
    dec = MIFERingLWE({"id": "sid", "keys": {"pp": pp}})
    assert dec.decrypt(cts, dk, y) == [6]
    with pytest.raises(FEValidationError):
        dec.decrypt(cts, dk, {"nid_0": [1.5, 1], "nid_1": [1, 1]})


def test_ring_key_shares_refuse_fractional_weights():
    kg = DecentralizedMCFERingLWEKeyGenerator(dict(CFG))
    kg.setup()
    pp = kg.get_public_parameters()
    client = DecentralizedMCFERingLWE({"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    with pytest.raises(FEValidationError):
        client.derive_function_decryption_key_share({"nid_0": [0.5, 1], "nid_1": [1, 1]}, "l")


@pytest.mark.parametrize("matrix", [
    [[1, 2], [3]],            # ragged rows: the short row was padded, the long one cut
    [[1] * 17, [1] * 17],     # more columns than ring_n
    [[0.5], [1]],             # fractional plaintext
    [[], []],
])
def test_ring_plaintexts_must_be_rectangular_integer_matrices(matrix):
    kg = MCFERingLWEKeyGenerator(dict(CFG))
    kg.setup()
    pp = kg.get_public_parameters()
    client = MCFERingLWE({"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    with pytest.raises(FEValidationError):
        client.encrypt(matrix, "l")
