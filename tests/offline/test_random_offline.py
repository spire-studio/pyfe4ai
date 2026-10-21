"""Offline / online encryption of SIFE-/MIFE-Paillier and MIFE-DDH (output-identical)."""

import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.mife import ddh as mife_ddh
from pyfe4ai.schemes.mife import paillier as mife_paillier
from pyfe4ai.schemes.sife import paillier as sife_paillier
from pyfe4ai.primitives import sampling as sampling_utils
from pyfe4ai.core.exceptions import FEValidationError


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _seed(monkeypatch, seed):
    rng = random.Random(seed)
    monkeypatch.setattr(sampling_utils, "_CSPRNG", rng)
    monkeypatch.setattr(mife_ddh, "_CSPRNG", rng)


def _sife():
    kg = sife_paillier.SIFEPaillierKeyGenerator({"eta": 1, "bit_length": 256, "bound_x": 20, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = sife_paillier.SIFEPaillier({"keys": {"pp": pp, "sk": kg.get_private_keys()}, "precision": 0})
    return kg, enc, sife_paillier.SIFEPaillier({"keys": {"pp": pp}, "precision": 0})


def _mife(mod, kg_name, cls_name, cfg):
    nids = ["a", "b"]
    kg = getattr(mod, kg_name)(dict(cfg, n=2, eta=1, lst_nid=nids))
    kg.setup()
    pp = kg.get_public_parameters()
    cls = getattr(mod, cls_name)
    return kg, {nid: cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}, "precision": 0}) for nid in nids}


CASES = {
    "sife/paillier": lambda: _sife()[1],
    "mife/paillier": lambda: _mife(mife_paillier, "MIFEPaillierKeyGenerator", "MIFEPaillier",
                                   {"bit_length": 256, "bound_x": 20, "bound_y": 3})[1]["a"],
    "mife/ddh": lambda: _mife(mife_ddh, "MIFEDDHKeyGenerator", "MIFEDDH", {"sec_param": 256, "bound_x": 20, "bound_y": 3})[1]["a"],
}


@pytest.mark.parametrize("case", list(CASES))
def test_precomputed_ciphertexts_are_the_plain_ones(monkeypatch, case):
    enc = CASES[case]()
    layers = [np.arange(-6, 6).reshape(3, 4), np.array([7, -7])]
    _seed(monkeypatch, 5)
    plain = enc.encrypt_lst_ndarray(layers)
    _seed(monkeypatch, 5)
    pre = enc.precompute([(3, 4), (2,)])
    online = enc.encrypt_lst_ndarray(layers, precomputed=pre)
    for a, b in zip(plain, online):
        assert a.tolist() == b.tolist()
    with pytest.raises(FEValidationError, match="once"):
        enc.encrypt_lst_ndarray(layers, precomputed=pre)
    with pytest.raises(FEValidationError, match="shapes"):
        enc.encrypt_lst_ndarray([np.arange(3)], precomputed=enc.precompute([(4,)]))


def test_sife_precomputed_decrypts():
    kg, enc, dec = _sife()
    x = np.array([3, -5, 7, 0])
    ct = enc.encrypt_lst_ndarray([x], precomputed=enc.precompute([(4,)]))
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [2]})
    assert np.array_equal(np.asarray(dec.decrypt_lst_ndarray_ct(ct, dk=dk, fusion_weight=[2])[0]), 2 * x)
