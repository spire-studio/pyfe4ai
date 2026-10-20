"""Decentralised key derivation must reveal the functional key and nothing more.

The MCFE2 layer reveals the *integer* sum of the clients' contributions.
CDGPP18 contributions reduced mod q would expose the carry of that sum (a
comparison oracle on the clients' secrets); LWE contributions reduced mod q
would expose, per coordinate, how many y_i t_i are negative."""

import os
import random

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFE, DecentralizedMCFEKeyGenerator
from pyfe4ai.schemes.mcfe.decentralize import combine_contributions
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWE, DecentralizedMCFELWEKeyGenerator

NIDS = ["nid_0", "nid_1", "nid_2"]


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _shares(kg, crypto_cls, weights):
    pp = kg.get_public_parameters()
    clients = {nid: crypto_cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    return pp, {nid: clients[nid].derive_function_decryption_key_share(weights) for nid in NIDS}


def test_ddh_shares_hide_the_carry_of_the_reduced_contributions():
    kg = DecentralizedMCFEKeyGenerator({"sec_param": 64, "n": 3, "lst_nid": NIDS, "eta": 2})
    kg.setup()
    rng = random.Random(4)
    quotients = []
    for _ in range(4):
        weights = {nid: [rng.randint(-9, 9) for _ in range(2)] for nid in NIDS}
        pp, shares = _shares(kg, DecentralizedMCFE, weights)
        q = (gp.mpz(pp["p"]) - 1) // 2
        sums = combine_contributions(pp["n2_mod"], {n: s["c"] for n, s in shares.items()}, NIDS)
        quotients += [int(v // q) for v in sums]
        # repeated derivation for the same weights gives the same share (no averaging)
        assert _shares(kg, DecentralizedMCFE, weights)[1] == shares
    # the bare carry lies in 0..n-1; with masking the quotient is spread over 2^stat_param n
    assert min(quotients) > len(NIDS)


def test_lwe_shares_combine_to_the_integer_key_of_fig_4():
    kg = DecentralizedMCFELWEKeyGenerator({"n": 3, "lst_nid": NIDS, "eta": 2, "lwe_n": 16})
    kg.setup()
    weights = {"nid_0": [3, -4], "nid_1": [-2, 1], "nid_2": [0, 5]}
    pp, shares = _shares(kg, DecentralizedMCFELWE, weights)
    sums = combine_contributions(pp["n2_mod"], {n: s["c"] for n, s in shares.items()}, NIDS)
    exact = [
        sum(y * int(z_j[c]) for nid in NIDS for y, z_j in zip(weights[nid], kg.msk["z"][nid]))
        for c in range(kg.dim)
    ]
    assert sums == exact  # sk_y = sum y_i Z_i over Z, nothing about individual signs
