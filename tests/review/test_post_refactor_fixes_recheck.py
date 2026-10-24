"""Re-check of the PF-1 fix (fix/post-refactor-0.4.0, 4b50be4..e8532e1).

PF-1 residual (low, compatibility): integer server ids now work end to end, but server ids of the other
hashable types that worked on 4d876e2 -- numpy integers (e.g. ``list(np.arange(3))``), tuples, bytes -- are
still accepted by the key generators and refused only when the partial decryptions are combined
(``FEValidationError``, ``core.ids.is_id``). Either refuse them where they are configured (the side branch
fix/post-refactor-0.4.0-strict-ids does) or accept them through the combination too.

The same late refusal remains on both branches when ``lst_sid`` is valid (``[1, 2, 3]``) but the enrolled
list or a server's own id is a numpy integer -- ``list(rng.choice(lst_sid, t, replace=False))`` gives one.
These ids equal and hash like the ints, so every server finds its key share and answers; only the combiner
refuses. Refuse them at the server (``share_decrypt``, or the server object) or accept them through the
combination (numpy integers as ``operator.index``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfe4ai.core.exceptions import FEError
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

NIDS = ["c0", "c1", "c2"]
W = {n: [1] for n in NIDS}
SIDS = {"numpy int64": [np.int64(1), np.int64(2), np.int64(3)], "tuple": [(1,), (2,), (3,)],
        "bytes": [b"s0", b"s1", b"s2"]}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _ring(sids, enrolled=None, server_id=lambda s: s):
    try:
        kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=sids, eta=1, ring_n=32,
                                                      sec_param=64, bound_x=100, bound_y=3, stat_param=64,
                                                      key_shares="full"))
        kg.setup()
    except FEError:
        return None  # refused where configured: consistent
    pp = kg.get_public_parameters()
    cts = {n: tm.ThresholdMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
           .encrypt([[1, 1, 1]], "L") for n in NIDS}
    creds = {"fusion_weight": W, "label": "L"}
    enrolled = sids[:2] if enrolled is None else enrolled
    try:
        parts = {server_id(s): tm.ThresholdMCFERingLWE({"id": server_id(s), "precision": 0, "keys": {"pp": pp}})
                 .share_decrypt(cts, creds, kg.get_decryption_keys(server_id(s), credentials=creds), enrolled)
                 for s in sids[:2]}
    except FEError:
        return None  # refused at the server: consistent
    return tm.ThresholdMCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}}).combine_decrypt(parts)[0]


def _lwe(sids, enrolled=None, server_id=lambda s: s):
    try:
        kg = ThresholdMIFELWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS,
                                           "lst_sid": sids, "lwe_n": 16, "bound_x": 6, "bound_y": 6})
        kg.setup()
    except FEError:
        return None
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([1]) for n in NIDS}
    enrolled = sids[:2] if enrolled is None else enrolled
    try:
        dks = {s: kg.get_decryption_keys(server_id(s), credentials={"fusion_weight": W}) for s in sids[:2]}
        parts = {server_id(s): ThresholdMIFELWE({"id": server_id(s), "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(
            cts, W, dks[s], enrolled) for s in sids[:2]}
    except FEError:
        return None
    return ThresholdMIFELWE({"id": "agg", "keys": {"pp": pp}}).combine_decrypt(parts)


@pytest.mark.parametrize("kind", sorted(SIDS))
@pytest.mark.parametrize("scheme", ["tmcfe-ring-lwe", "tmife-lwe"])
def test_pf1_other_server_id_types_are_refused_at_setup_or_combine(scheme, kind):
    result = (_ring if scheme == "tmcfe-ring-lwe" else _lwe)(SIDS[kind])
    assert result is None or result == 3, result


@pytest.mark.parametrize("where", ["enrolled list", "server object ids"])
@pytest.mark.parametrize("scheme", ["tmcfe-ring-lwe", "tmife-lwe"])
def test_pf1_numpy_int_enrolled_or_server_ids_are_refused_at_the_server_or_combine(scheme, where):
    sids = [1, 2, 3]
    kw = ({"enrolled": [np.int64(s) for s in sids[:2]]} if where == "enrolled list"
          else {"server_id": np.int64})
    result = (_ring if scheme == "tmcfe-ring-lwe" else _lwe)(sids, **kw)
    assert result is None or result == 3, result
