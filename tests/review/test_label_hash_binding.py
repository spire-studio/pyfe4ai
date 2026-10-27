"""The label-hash mode is bound to ciphertexts (October 2026 code review).

Public parameters without ``label_hash`` are read as ``"shake256-wide"`` (the
v0.3.0 hash) while the key generators default to ``"shake128-rejection"``.
Functional keys carried the mode and were checked against the decryptor's
parameters, but ciphertexts carried nothing: a client still holding parameters
of the old format (or a different mode) padded with another label hash, and
the aggregate decrypted without any error to wrong values
(``p = 2 n eta X Y + 2``, so almost every garbage value decodes inside the
bound). Ciphertext layers now carry the mode, and a mismatch is refused.
"""

import numpy as np
import os
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as m
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError

NIDS = ["c0", "c1", "c2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def test_a_client_with_another_label_hash_is_detected():
    kg = m.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": NIDS, "eta": 1, "ring_n": 64, "sec_param": 64,
                                    "bound_x": 10 ** 6, "bound_y": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["label_hash"] != "shake256-wide"
    pp_old = {k: v for k, v in pp.items() if k != "label_hash"}  # v0.3.0 format: read as shake256-wide
    clients = {nid: m.MCFERingLWE({"id": nid, "precision": 2,
                                   "keys": {"pp": pp_old if nid == "c0" else pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    rng = np.random.default_rng(0)
    g = {nid: rng.uniform(-1, 1, 200).round(2) for nid in NIDS}
    try:
        enc = {nid: c.encrypt_lst_ndarray([g[nid]], label="r1") for nid, c in clients.items()}
        w = {nid: [1] for nid in NIDS}
        dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": w, "label": "r1",
                                                        "n_chunks": m.MCFERingLWE.key_chunks(enc["c0"])})
        agg = m.MCFERingLWE({"id": "agg", "precision": 2, "keys": {"pp": pp}})
        out = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=w, label="r1")[0]
    except (FEKeyError, FESchemeError, FEValidationError, ValueError):
        return  # detected: the fix
    wrong = int(np.sum(~np.isclose(np.asarray(out, dtype=float), sum(g.values()))))
    pytest.fail("label-hash mismatch not detected: {} of 200 aggregated values silently wrong".format(wrong))
