"""Caches live in the objects that own them.

Two objects never share cache entries, and a copy or a pickle of an object
starts without the caches that may hold native objects (rebuilt on first
use) and works: the prepared master key of a Ring-LWE key generator, the
encryption keys of the Ring-LWE clients and a threshold server's NTTs of its
key shares. The stores that are state (a threshold server's weight key
shares) are kept; a shallow copy shares them with the original. See
``docs/threading.rst``.
"""

from __future__ import annotations

import copy
import os
import pickle

import numpy as np
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.sife import ring_lwe as sife_rlwe

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
CLONES = {"copy": copy.copy, "deepcopy": copy.deepcopy, "pickle": lambda obj: pickle.loads(pickle.dumps(obj))}
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
WEIGHTS = {nid: [y] for nid, y in zip(NIDS, (1, -2, 3))}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _mcfe():
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": NIDS, "eta": 1, "ring_n": 16, "sec_param": 64,
                                            "bound_x": 10, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: mcfe_rlwe.MCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    return kg, pp, clients


def _mcfe_round(kg, pp, clients, label):
    cts = {nid: cl.encrypt_lst_ndarray([np.arange(5)], label=label) for nid, cl in clients.items()}
    dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": WEIGHTS, "label": label, "n_chunks": 1})
    agg = mcfe_rlwe.MCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}})
    out = agg.compute_lst_ndarray_ct(cts, dk=dk, fusion_weight=WEIGHTS, label=label)
    assert np.ravel(out[0]).tolist() == [2 * v for v in range(5)]  # (1 - 2 + 3) x


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_clients_and_two_key_generators_do_not_share_cache_entries(backend):
    with native_backend.backend(backend):
        kg, pp, clients = _mcfe()
        _mcfe_round(kg, pp, clients, "L")
        other, _, _ = _mcfe()
        other.get_decryption_keys("agg", credentials={"fusion_weight": WEIGHTS, "label": "L", "n_chunks": 1})
    keys = [cl.__dict__["_enc_key"] for cl in clients.values()]
    assert len({id(k) for k in keys}) == len(keys)
    for cl, key in zip(clients.values(), keys):
        assert key["refs"][0] is mcfe_rlwe.MCFERingLWEEncryptionKey(cl.sk).pk  # each its own key
    assert len({tuple(map(int, k["pk"][0])) for k in keys}) == len(keys)
    for gen in (kg, other):
        assert gen.__dict__["_native_msk"][0] is gen.msk
    assert kg.__dict__["_native_msk"][2] is not other.__dict__["_native_msk"][2]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("clone", sorted(CLONES))
def test_copies_and_pickles_start_without_the_key_caches_and_work(backend, clone):
    with native_backend.backend(backend):
        kg, pp, clients = _mcfe()
        _mcfe_round(kg, pp, clients, "L")
        assert "_native_msk" in kg.__dict__ and all("_enc_key" in cl.__dict__ for cl in clients.values())
        kg2 = CLONES[clone](kg)
        clients2 = {nid: CLONES[clone](cl) for nid, cl in clients.items()}
        assert "_native_msk" not in kg2.__dict__
        assert all("_enc_key" not in cl.__dict__ for cl in clients2.values())
        _mcfe_round(kg2, pp, clients2, "M")  # rebuilt on use
        assert "_native_msk" in kg2.__dict__ and kg2.__dict__["_native_msk"] is not kg.__dict__["_native_msk"]
        assert all(cl2.__dict__["_enc_key"] is not cl.__dict__["_enc_key"]
                   for cl, cl2 in zip(clients.values(), clients2.values()))
        _mcfe_round(kg, pp, clients, "N")  # the originals keep theirs

        skg = sife_rlwe.SIFERingLWEKeyGenerator({"eta": 1, "ring_n": 16, "sec_param": 64, "bound_x": 10,
                                                 "bound_y": 1})
        skg.setup()
        s = sife_rlwe.SIFERingLWE({"keys": {"pp": skg.get_public_parameters(), "sk": skg.get_private_keys()},
                                   "precision": 0})
        s.encrypt([[1] * 16])
        assert {"_enc_key", "_nkey"} & set(s.__dict__)
        twin = CLONES[clone](s)
        assert not {"_enc_key", "_nkey"} & set(twin.__dict__)
        dk = skg.get_decryption_keys("sid", credentials={"fusion_weight": [-1]})
        assert twin.decrypt(twin.encrypt([[3] * 16]), dk, [-1]) == s.decrypt(s.encrypt([[3] * 16]), dk, [-1]) == [-3] * 16


# -- threshold MCFE over Ring-LWE: the servers' stores and caches


def _tmcfe():
    kg = tm.ThresholdMCFERingLWEKeyGenerator({"n": 3, "lst_nid": NIDS, "s": 3, "t": 2, "lst_sid": SIDS, "eta": 1,
                                              "ring_n": 16, "sec_param": 64, "bound_x": 100, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0,
                                             "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    servers = {sid: tm.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}}) for sid in SIDS[:2]}
    return kg, pp, clients, servers


def _tmcfe_round(kg, pp, clients, servers, label):
    cts = {nid: cl.encrypt([[1, 2, 3]], label) for nid, cl in clients.items()}
    creds = {"fusion_weight": WEIGHTS, "label": label}
    parts = {}
    for sid, srv in servers.items():
        dk = kg.get_decryption_keys(sid, credentials=creds)
        if not srv.has_weight_key_share(dk):
            srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials=creds))
        parts[sid] = srv.share_decrypt(cts, creds, dk, list(servers))
    comb = tm.ThresholdMCFERingLWE({"id": "comb", "precision": 0, "keys": {"pp": pp}})
    assert comb.combine_decrypt(parts) == [sum(WEIGHTS[n][0] * v for n in NIDS) for v in (1, 2, 3)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_threshold_servers_and_key_generators_keep_their_own_stores(backend):
    with native_backend.backend(backend):
        kg, pp, clients, servers = _tmcfe()
        _tmcfe_round(kg, pp, clients, servers, "L")
        other = _tmcfe()[0]
        other.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    s0, s1 = servers["s0"], servers["s1"]
    names = ["_weight_shares", "_enrolled_share_cache"] + (["_hat_cache"] if backend == "native" else [])
    for name in names:
        assert name in s0.__dict__ and name in s1.__dict__, name
        assert s0.__dict__[name] is not s1.__dict__[name], name
    for srv in (s0, s1):
        assert all(share["sid"] == srv.id for share in srv._weight_shares.values())
        assert all(hit[0]["sid"] == srv.id for hit in srv._enrolled_share_cache.values())
    assert kg.dict_weight_shares is not other.dict_weight_shares
    assert {e["share_set_id"] for e in kg.dict_weight_shares.values()}.isdisjoint(
        e["share_set_id"] for e in other.dict_weight_shares.values())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("clone", sorted(CLONES))
def test_copies_and_pickles_of_threshold_servers(backend, clone):
    with native_backend.backend(backend):
        kg, pp, clients, servers = _tmcfe()
        _tmcfe_round(kg, pp, clients, servers, "L")
        twins = {sid: CLONES[clone](srv) for sid, srv in servers.items()}
        kg2 = CLONES[clone](kg)
        for sid, twin in twins.items():
            assert "_hat_cache" not in twin.__dict__  # NTTs of key shares (native objects): rebuilt on use
            assert twin._weight_shares == servers[sid]._weight_shares  # state: kept
            shared = twin._weight_shares is servers[sid]._weight_shares
            assert shared == (clone == "copy"), "a shallow copy shares the store, a deep copy or pickle has its own"
        assert (kg2.dict_weight_shares is kg.dict_weight_shares) == (clone == "copy")
        _tmcfe_round(kg2, pp, clients, twins, "M")
        _tmcfe_round(kg, pp, clients, servers, "N")
