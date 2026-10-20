"""Weights and plaintexts must be integers: non-integral floats used to be
truncated silently (0.5 -> 0), and ragged Ring-LWE plaintext rows were cut
to the ring dimension."""

import os
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import (
    DecentralizedMCFERingLWE,
    DecentralizedMCFERingLWEKeyGenerator,
)
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE, MIFERingLWEKeyGenerator
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.core.exceptions import FEValidationError

NIDS = ["nid_0", "nid_1"]
CFG = {"sec_param": 64, "eta": 2, "n": 2, "lst_nid": NIDS, "ring_n": 16, "bound_x": 4, "bound_y": 4}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


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


def test_ring_decryption_refuses_malformed_ciphertexts():
    kg = MIFERingLWEKeyGenerator(dict(CFG))
    kg.setup()
    pp = kg.get_public_parameters()
    y = {"nid_0": [1, 1], "nid_1": [1, 1]}
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
    cts = {nid: MIFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([[1], [2]])
           for nid in NIDS}
    dec = MIFERingLWE({"id": "sid", "keys": {"pp": pp}})
    first = cts["nid_0"]
    from pyfe4ai.core.exceptions import FESchemeError

    for bad in (dict(first, ct1=first["ct1"][:-1]), dict(first, ct0=first["ct0"][:1])):
        with pytest.raises(FESchemeError, match="malformed"):
            dec.decrypt(dict(cts, nid_0=bad), dk, y)
    with pytest.raises(FESchemeError, match="malformed"):
        dec.decrypt({nid: dict(ct, k=0) for nid, ct in cts.items()}, dk, y)


def test_ndarray_chunks_must_index_the_key():
    """A negative first_chunk picked chunk keys from the end of the list."""
    import numpy as np

    from pyfe4ai.core.exceptions import FESchemeError

    kg = MCFERingLWEKeyGenerator({**CFG, "eta": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {nid: MCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           .encrypt_lst_ndarray([np.ones(20)], label="l") for nid in NIDS}
    dec = MCFERingLWE({"id": "sid", "keys": {"pp": pp}})
    y = {nid: [1] for nid in NIDS}
    n_chunks = MCFERingLWE.key_chunks(cts["nid_0"])
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y, "label": "l", "n_chunks": n_chunks})
    assert np.allclose(dec.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight=y, label="l")[0], 2)
    for first in (-1, n_chunks, "0"):
        bad = {nid: [dict(layers[0], first_chunk=first)] for nid, layers in cts.items()}
        with pytest.raises(FESchemeError, match="chunk"):
            dec.decrypt_lst_ndarray_ct(bad, dk=dk, fusion_weight=y, label="l")


def test_fractional_inputs_are_refused_everywhere(tmp_path, monkeypatch):
    """SIFE-Paillier truncated [1.7, 2.9] (and a weight 0.5 to 0); the same
    abs(int(v)) checks were in the LWE, Paillier and Damgard MIFE/MCFE schemes."""
    from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
    from pyfe4ai.schemes.mife.damgard_ddh import MIFEDamgard, MIFEDamgardKeyGenerator
    from pyfe4ai.schemes.sife.paillier import SIFEPaillier, SIFEPaillierKeyGenerator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    kg = SIFEPaillierKeyGenerator({"sec_param": 64, "eta": 2, "bit_length": 64, "bound_x": 10, "bound_y": 10})
    kg.setup()
    enc = SIFEPaillier({"keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys()}})
    with pytest.raises(FEValidationError, match="integers"):
        enc.encrypt([1.7, 2.9])
    with pytest.raises(FEValidationError, match="integers"):
        kg.get_decryption_keys("s", credentials={"fusion_weight": [0.5, 1]})
    mk = MCFELWEKeyGenerator({"n": 2, "lst_nid": NIDS, "lwe_n": 8, "bound_x": 4, "bound_y": 4})
    mk.setup()
    with pytest.raises(FEValidationError, match="integers"):
        mk.get_decryption_keys("s", credentials={"fusion_weight": {"nid_0": [0.5], "nid_1": [1]}})
    dk = MIFEDamgardKeyGenerator({"n": 2, "lst_nid": NIDS, "eta": 1, "modulus_length": 64, "bound": 10})
    dk.setup()
    c = MIFEDamgard({"id": "nid_0", "keys": {"pp": dk.get_public_parameters(), "sk": dk.get_private_keys("nid_0")}})
    with pytest.raises(FEValidationError, match="integers"):
        c.encrypt([1.5])


def test_rationals_are_checked_exactly():
    from fractions import Fraction

    from pyfe4ai._validation import int_vector, strict_int

    assert strict_int(Fraction(6, 2)) == 3
    for bad in (Fraction(2 ** 60 + 1, 2), 1 + Fraction(1, 10 ** 20)):
        with pytest.raises(FEValidationError):
            strict_int(bad)
    with pytest.raises(FEValidationError):
        int_vector([Fraction(3, 2)], "x")


def test_mcfe_ddh_refuses_fractional_inputs(tmp_path, monkeypatch):
    """MCFE-DDH used gp.mpz(v), which truncates 2.5 to 2."""
    from pyfe4ai.schemes.mcfe.ddh import MCFE, MCFEKeyGenerator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    kg = MCFEKeyGenerator({"sec_param": 128, "eta": 1, "n": 2, "lst_nid": NIDS})
    kg.setup()
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("s", credentials={"fusion_weight": {"nid_0": [1.5], "nid_1": [1]}})
    c = MCFE({"id": "nid_0", "keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys("nid_0")}})
    with pytest.raises(FEValidationError):
        c.encrypt([2.5], "l")
