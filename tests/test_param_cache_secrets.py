"""Parameter caches in ./config must not contain decryption trapdoors.

Versions <= 0.2.0 persisted the Paillier factorization (p, q); with it,
ct^lambda = 1 + lambda x N mod N^2 decrypts SIFE ciphertexts directly and two
ciphertexts of an MCFE slot under two labels reveal both plaintexts.
"""

import json
import os

import pytest

from pyfe4ai.schemes.mcfe.paillier import MCFEPaillierKeyGenerator
from pyfe4ai.schemes.mcfe.paillier_decentralized import DecentralizedMCFEPaillierKeyGenerator
from pyfe4ai.schemes.mife.paillier import MIFEPaillierKeyGenerator
from pyfe4ai.schemes.sife.paillier import SIFEPaillierKeyGenerator

CONFIG = {
    "sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"],
    "bit_length": 64, "bound_x": 10, "bound_y": 10,
}


@pytest.mark.parametrize(
    "kg_cls",
    [
        SIFEPaillierKeyGenerator,
        MIFEPaillierKeyGenerator,
        MCFEPaillierKeyGenerator,
        DecentralizedMCFEPaillierKeyGenerator,
    ],
)
def test_paillier_cache_holds_no_factorization(tmp_path, kg_cls):
    stale = dict(CONFIG, p="11", q="13")  # a v0.2.0-style cache entry
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        folder = tmp_path / "config" / "authority" / kg_cls._scheme_type
        folder.mkdir(parents=True)
        (folder / "param.json").write_text(json.dumps(stale))
        kg = kg_cls(dict(CONFIG))
        param = json.loads((folder / "param.json").read_text())
    finally:
        os.chdir(old_cwd)
    assert "p" not in param and "q" not in param
    assert not hasattr(kg, "p") and not hasattr(kg, "q")
