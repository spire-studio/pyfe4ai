"""Partials of two sharings of one label (threshold MCFE with split key
shares; October 2026 code review).

A label asked for again after it left the key generator's cache
(``LABEL_PARTS_KEPT`` = 16 later labels of the same weights) gets a new
sharing of its terms. The re-sharing leaks nothing: for one request an honest
server's answers under the two sharings of a label's terms differ by a sharing
of zero that ``t - 1`` servers compute from their own label parts (checked
numerically). But the partials of the two sharings did not fail closed when
combined -- they decoded, with high probability, to another aggregate inside
the decoding bound, without an error. It happens when the servers of one
round fetch their label parts on both sides of the label's eviction: e.g. one
server fetches round r, the key generator then issues 16 more rounds for the
weights (servers fetching ahead, or keys issued in advance), and another
server fetches round r. Partial decryptions carried nothing that told the
sharings apart; they now carry the sharing's ``sharing_id``, and combiners
raise ``MixedSharingsError``.
"""

from __future__ import annotations

import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.core.exceptions import FEError

NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
WEIGHTS = {nid: [y] for nid, y in zip(NIDS, (1, -2, 3))}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("t,sharing", [(2, "replicated"), (3, "shamir")])
def test_r5c_label_parts_of_one_round_fetched_across_its_eviction(t, sharing):
    kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(n=3, lst_nid=NIDS, s=3, t=t, lst_sid=SIDS, eta=1, ring_n=32,
                                                  sec_param=64, bound_x=100, bound_y=3, stat_param=64,
                                                  sharing=sharing))
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    servers = {}
    for sid in SIDS:
        servers[sid] = tm.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}})
        servers[sid].add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": WEIGHTS}))
    creds = {"fusion_weight": WEIGHTS, "label": "r0"}
    enrolled = SIDS[:t]
    keys = {enrolled[0]: kg.get_decryption_keys(enrolled[0], credentials=creds)}  # the first server fetches round r0
    for r in range(1, tm.LABEL_PARTS_KEPT + 1):  # keys of later rounds of the same weights are issued
        kg.get_decryption_keys(enrolled[0], credentials={"fusion_weight": WEIGHTS, "label": "r%d" % r})
    for sid in enrolled[1:]:  # the other servers fetch round r0 now
        keys[sid] = kg.get_decryption_keys(sid, credentials=creds)
    cts = {nid: cl.encrypt([[1, 2, 3]], "r0") for nid, cl in clients.items()}
    parts = {sid: servers[sid].share_decrypt(cts, creds, keys[sid], enrolled) for sid in enrolled}
    right = [sum(WEIGHTS[nid][0] * v for nid in NIDS) for v in (1, 2, 3)]
    try:
        result = tm.ThresholdMCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}}).combine_decrypt(parts)
    except FEError:
        return  # refused: fails closed
    assert result == right, "partials of two sharings of one label combined silently to {}".format(result)
