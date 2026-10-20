"""Defaults of the threshold Ring-LWE MCFE: binary format, RNS modulus, split key shares.

* A key generator without options publishes ``ct_format="binary"`` and an RNS
  ``q`` of at least three factors (native also for the ``toy`` profile's small
  moduli) and issues label parts per round (weight parts once per weights);
  the replicated-seeded sharing issues whole key shares.
* The options of earlier versions stay selectable and their public
  parameters, records, ciphertexts and key shares keep working.
* Whatever does not match fails closed with a message naming the fix.
* Label parts are issued for the weights' current share set only, even when
  another thread evicts it meanwhile; they are kept with that sharing and
  dropped with it, for the last ``LABEL_PARTS_KEPT`` labels; every sharing has
  a ``sharing_id``; the native Shamir dealing tells a test's exhausted random
  bytes from malformed key material.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest

from pyfe4ai.schemes.threshold.flooding_accounting import PartialDecryptionBudget
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError

NATIVE = native_backend.available()
needs_native = pytest.mark.skipif(not NATIVE, reason="native extension not built")
BACKENDS = ["python"] + (["native"] if NATIVE else [])
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
WEIGHTS = {nid: [y] for nid, y in zip(NIDS, (1, -2, 3))}
LEGACY = {"ct_format": "json", "ring_modulus": "prime", "key_shares": "full"}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


@pytest.fixture
def _native():
    with native_backend.backend("native"):
        yield


def _setup(t=2, **extra):
    config = {"n": 3, "lst_nid": NIDS, "s": 3, "t": t, "lst_sid": SIDS, "eta": 1, "ring_n": 16, "sec_param": 64,
              "bound_x": 100, "bound_y": 3, **extra}
    kg = tm.ThresholdMCFERingLWEKeyGenerator(config)
    kg.setup()
    return kg, kg.get_public_parameters()


def _obj(pp, nid, kg=None):
    keys = {"pp": pp} if kg is None else {"pp": pp, "sk": kg.get_private_keys(nid)}
    return tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": keys})


def _server(kg, pp, sid, creds):
    srv = _obj(pp, sid)
    dk = kg.get_decryption_keys(sid, credentials=creds)
    if not srv.has_weight_key_share(dk):
        srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials=creds))
    return srv, dk


def _round(kg, pp, label="r", enrolled=None, servers=None, **cred_extra):
    """A packed round; *servers* ``{sid: (server, dk)}`` (made here if not given)."""
    enrolled = enrolled or SIDS[:pp["t"]]
    rng = np.random.default_rng(len(label))
    grads = {nid: [rng.integers(-100, 101, 21)] for nid in NIDS}
    enc = {nid: _obj(pp, nid, kg).encrypt_lst_ndarray(grads[nid], label=label) for nid in NIDS}
    creds = {"fusion_weight": WEIGHTS, "label": label, "n_chunks": tm.ThresholdMCFERingLWE.key_chunks(enc["c0"]),
             **cred_extra}
    parts = {}
    for sid in enrolled:
        srv, dk = servers[sid] if servers else _server(kg, pp, sid, creds)
        parts[sid] = srv.compute_lst_ndarray_ct(enc, creds, dk, enrolled)
    out = _obj(pp, "comb").decrypt_lst_ndarray_ct(parts)[0]
    assert np.array_equal(out, sum(WEIGHTS[nid][0] * grads[nid][0] for nid in NIDS))
    return enc, creds, parts


# ------------------------------------------------------------------ defaults


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("t,sharing", [(2, "replicated"), (3, "shamir")])
def test_defaults_are_binary_rns_and_split(name, t, sharing):
    kg, pp = _setup(t=t)
    assert (tm.DEFAULT_CT_FORMAT, tm.DEFAULT_RING_MODULUS) == ("binary", "rns")
    assert pp["ct_format"] == "binary" and pp["sharing"] == sharing and kg.key_shares == "split"
    assert len(pp["q_factors"]) >= tm.MIN_RNS_FACTORS and tm.binding_fields(pp)[-2:] == ("ct_format", "q_factors")
    creds = {"fusion_weight": WEIGHTS, "label": "r", "n_chunks": 2}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    assert dk["part"] == "label" and "sk_y" not in dk
    weight = kg.get_weight_key_share("s0", credentials=creds)
    assert weight["part"] == "weights" and weight["share_set_id"] == dk["share_set_id"]
    with native_backend.backend(name):
        enc, _, parts = _round(kg, pp)
    assert isinstance(enc["c0"][0]["chunks"][0]["ct"], bytes)
    assert isinstance(parts["s0"][0]["parts"][0]["w"], bytes)


def test_seeded_sharing_issues_whole_key_shares():
    kg, pp = _setup(t=2, sharing="replicated-seeded")
    assert kg.key_shares == "full" and pp["ct_format"] == "binary"
    dk = kg.get_decryption_keys("s0", credentials={"fusion_weight": WEIGHTS, "label": "r"})
    assert "part" not in dk and dk["sharing"] == "replicated-seeded"
    with pytest.raises(FEValidationError, match="key_shares='full'"):
        _setup(t=2, sharing="replicated-seeded", key_shares="split")
    for bad in ("whole", None, "label"):
        with pytest.raises(FEValidationError, match="key_shares"):
            _setup(key_shares=bad)


def test_full_key_shares_are_selectable_per_generator_and_per_call():
    kg, pp = _setup(key_shares="full")
    creds = {"fusion_weight": WEIGHTS, "label": "r"}
    assert "part" not in kg.get_decryption_keys("s0", credentials=creds)
    assert kg.get_decryption_keys("s0", credentials=dict(creds, key_part="label"))["part"] == "label"
    split, _ = _setup()
    full = split.get_decryption_keys("s0", credentials=dict(creds, key_part="full"))
    assert "part" not in full and "sk_y" in full["pieces"][0]["share"]
    _round(kg, pp)  # whole key shares: no weight part


@needs_native
def test_toy_moduli_get_three_factors_and_run_natively():
    """A flooded q of 115 bits had two factors, the two-prime context of the
    MIFE / MCFE prototype, which has no threshold kernels (the Python path)."""
    kg, pp = _setup(t=2, bound_x=4, bound_y=3, label_modulus=8)
    assert len(pp["q_factors"]) == 3 and int(pp["q"]).bit_length() <= 124
    with native_backend.backend("native"):
        assert tm._native_ring(pp["q"], pp["ring_n"]) is not None
    kg2, pp2 = _setup(t=2, bound_x=4, bound_y=3, ring_modulus="rns", stat_param=40)
    assert len(pp2["q_factors"]) == 3
    from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu

    assert rlu.rns_factor_count(1 << 100) == 2 and rlu.rns_factor_count(1 << 100, 3) == 3
    assert rlu.rns_factor_count(1 << 230, 3) == 4  # more where the bits need them


def test_parameter_records_of_both_moduli_coexist():
    kg, _ = _setup()
    legacy, _ = _setup(**LEGACY)
    folder = os.path.join(kg.config_folder, kg._scheme_type)
    assert sorted(f for f in os.listdir(folder) if f.startswith("param")) == ["param-rns.json", "param.json"]
    with open(os.path.join(folder, "param-rns.json"), encoding="utf-8") as f:
        rns = json.load(f)
    with open(os.path.join(folder, "param.json"), encoding="utf-8") as f:
        prime = json.load(f)
    # the same construction ids; the RNS record names its factors
    assert rns["threshold_construction"] == prime["threshold_construction"]
    assert rns["ring_modulus"] == "rns" and len(rns["q_factors"]) >= 3 and "q_factors" not in prime
    assert _setup()[0].a_seed == kg.a_seed and _setup(**LEGACY)[0].a_seed == legacy.a_seed


def test_the_options_of_earlier_versions_keep_working():
    kg, pp = _setup(**LEGACY)
    assert "ct_format" not in pp and "q_factors" not in pp
    assert tm.binding_fields(pp) == tm._BINDING_FIELDS
    enc, _, parts = _round(kg, pp)
    assert "ct0" in enc["c0"][0]["chunks"][0] and isinstance(parts["s0"][0]["parts"][0]["w"], list)
    # their public parameters, ciphertexts and key shares sent as JSON (as stored by earlier versions)
    pp_j = json.loads(json.dumps(pp))
    enc_j = json.loads(json.dumps(enc))
    creds = {"fusion_weight": WEIGHTS, "label": "r", "n_chunks": tm.ThresholdMCFERingLWE.key_chunks(enc["c0"])}
    dks = {sid: json.loads(json.dumps(kg.get_decryption_keys(sid, credentials=creds))) for sid in SIDS[:2]}
    parts_j = {sid: _obj(pp_j, sid).compute_lst_ndarray_ct(enc_j, creds, dks[sid], SIDS[:2]) for sid in SIDS[:2]}
    assert parts_j == parts


def test_state_version_of_the_defaults():
    """The key generator's state changed (key_shares, label parts kept with their
    share set, sharing ids): pickles of earlier state versions are refused clearly."""
    kg, pp = _setup()
    creds = {"fusion_weight": WEIGHTS, "label": "r"}
    issued = kg.get_decryption_keys("s0", credentials=creds)
    assert tm._STATE_VERSION.endswith("2026-10-3")
    state = dict(kg.__getstate__(), _state_version="pyfe4ai/tmcfe-ring-lwe/state/2026-10-2")
    state.pop("key_shares")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(type(kg), "__getstate__", lambda self: state)
        data = pickle.dumps(kg)
    with pytest.raises(FESchemeError, match="incompatible version of pyfe4ai"):
        pickle.loads(data)
    again = pickle.loads(pickle.dumps(kg))
    assert again.key_shares == "split"
    assert again.get_decryption_keys("s0", credentials=creds) == issued  # kept with its sharing, pickled along


# ---------------------------------------------------------------- fail closed


def test_a_label_part_without_its_weight_part_names_the_fix():
    kg, pp = _setup()
    cts = {nid: _obj(pp, nid, kg).encrypt([[1, 2]], "L") for nid in NIDS}
    creds = {"fusion_weight": WEIGHTS, "label": "L"}
    srv = _obj(pp, "s0")
    dk = kg.get_decryption_keys("s0", credentials=creds)
    assert not srv.has_weight_key_share(dk) and srv.has_weight_key_share(dict(creds))  # a full share needs none
    with pytest.raises(FEKeyError, match=r"add_weight_key_share\(.*get_weight_key_share.*key_part='full'"):
        srv.share_decrypt(cts, creds, dk, SIDS[:2])
    srv.add_weight_key_share(kg.get_weight_key_share("s0", credentials=creds))
    assert srv.has_weight_key_share(dk)
    srv.share_decrypt(cts, creds, dk, SIDS[:2])
    # a weight part alone does not decrypt
    with pytest.raises(FEKeyError, match="weight key share decrypts together"):
        srv.share_decrypt(cts, creds, kg.get_weight_key_share("s0", credentials=creds), SIDS[:2])


def test_formats_and_parameters_do_not_mix():
    kg, pp = _setup()
    old, pp_old = _setup(**LEGACY)
    creds = {"fusion_weight": WEIGHTS, "label": "L"}
    cts = {nid: _obj(pp, nid, kg).encrypt([[1, 2]], "L") for nid in NIDS}
    cts_old = {nid: _obj(pp_old, nid, old).encrypt([[1, 2]], "L") for nid in NIDS}
    srv, dk = _server(kg, pp, "s0", creds)
    srv_old = _obj(pp_old, "s0")
    dk_old = old.get_decryption_keys("s0", credentials=creds)
    # JSON ciphertexts under binary public parameters, and the other way round
    with pytest.raises(FESchemeError, match="is JSON; the public parameters' format is binary"):
        srv.share_decrypt(cts_old, creds, dk, SIDS[:2])
    with pytest.raises(FESchemeError, match="binary"):
        srv_old.share_decrypt(cts, creds, dk_old, SIDS[:2])
    # key shares of the other public parameters (another format, modulus and setup)
    with pytest.raises((FEKeyError, FESchemeError)):
        srv.share_decrypt(cts, creds, dk_old, SIDS[:2])
    with pytest.raises((FEKeyError, FESchemeError)):
        srv_old.share_decrypt(cts_old, creds, dk, SIDS[:2])
    # a client or server asking for the format the public parameters do not have
    with pytest.raises(FEValidationError):
        tm.ThresholdMCFERingLWE({"id": "c0", "ct_format": "json", "keys": {"pp": pp}})
    # partial decryptions of one format at a combiner of the other
    _, _, parts = _round(kg, pp)
    _, _, parts_old = _round(old, pp_old)
    with pytest.raises(FESchemeError, match="format"):
        _obj(pp_old, "comb").decrypt_lst_ndarray_ct(parts)
    with pytest.raises(FESchemeError, match="format"):
        _obj(pp, "comb").decrypt_lst_ndarray_ct(parts_old)


def test_parts_of_other_generators_and_share_sets_are_refused():
    kg, pp = _setup()
    other, _ = _setup()  # the same parameters (record), another setup
    creds = {"fusion_weight": WEIGHTS, "label": "L"}
    cts = {nid: _obj(pp, nid, kg).encrypt([[1, 2]], "L") for nid in NIDS}
    srv = _obj(pp, "s0")
    srv.add_weight_key_share(other.get_weight_key_share("s0", credentials=creds))
    with pytest.raises(FEKeyError):
        srv.share_decrypt(cts, creds, kg.get_decryption_keys("s0", credentials=creds), SIDS[:2])
    # a new setup: new share sets, the kept weight part does not complete the new label parts
    srv.add_weight_key_share(kg.get_weight_key_share("s0", credentials=creds))
    kg.setup()
    pp2 = kg.get_public_parameters()
    cts2 = {nid: _obj(pp2, nid, kg).encrypt([[1, 2]], "L") for nid in NIDS}
    srv2 = _obj(pp2, "s0")
    srv2.add_weight_key_share(other.get_weight_key_share("s0", credentials=creds))
    with pytest.raises(FEKeyError):
        srv2.share_decrypt(cts2, creds, kg.get_decryption_keys("s0", credentials=creds), SIDS[:2])


# ------------------------------------------------- label parts and their sharings


def test_i_b_a_label_part_is_never_issued_for_an_evicted_share_set(monkeypatch):
    """Another thread evicts the weights' sharing while their label part is
    made: the label part is made again for the new share set."""
    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 1)
    kg, pp = _setup()
    other = {nid: [2] for nid in NIDS}
    real, calls = kg._label_part, []

    def racing(*args):
        made = real(*args)
        calls.append(made)
        if len(calls) == 1:  # meanwhile: weights issued elsewhere evict the sharing, and these weights return
            kg.get_weight_key_share("s1", credentials={"fusion_weight": other})
            kg.get_weight_key_share("s1", credentials={"fusion_weight": WEIGHTS})
        return made

    monkeypatch.setattr(kg, "_label_part", racing)
    creds = {"fusion_weight": WEIGHTS, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    current = kg.get_weight_key_share("s0", credentials=creds)
    assert len(calls) == 2 and calls[0]["s0"]["share_set_id"] != dk["share_set_id"]
    assert dk["share_set_id"] == current["share_set_id"]
    # it serves with the weight part handed out now
    cts = {nid: _obj(pp, nid, kg).encrypt([[1, 2, 3]], "L") for nid in NIDS}
    parts = {}
    for sid in SIDS[:2]:
        srv = _obj(pp, sid)
        srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials=creds))
        parts[sid] = srv.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), SIDS[:2])
    assert _obj(pp, "comb").combine_decrypt(parts) == [sum(WEIGHTS[n][0] * v for n in NIDS) for v in (1, 2, 3)]


def test_i_b_an_eviction_during_every_attempt_is_an_error(monkeypatch):
    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 1)
    kg, _ = _setup()
    real = kg._label_part

    def always(*args):
        made = real(*args)
        kg.get_weight_key_share("s1", credentials={"fusion_weight": {nid: [2] for nid in NIDS}})
        return made

    monkeypatch.setattr(kg, "_label_part", always)
    with pytest.raises(FESchemeError, match="WEIGHT_SHARINGS_KEPT"):
        kg.get_decryption_keys("s0", credentials={"fusion_weight": WEIGHTS, "label": "L"})


def test_i_b_concurrent_issuing_with_evictions(monkeypatch):
    """Threads issue label parts for more weights than are kept: every call
    returns a label part of a share set the key generator made (or, under
    eviction at every attempt, fails closed), and the label parts kept with a
    sharing are of that sharing."""
    import concurrent.futures

    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 2)
    kg, _ = _setup()
    weight_sets = [{nid: [k] for nid in NIDS} for k in range(1, 4)]
    made_sets = set()
    real = kg._weight_entry

    def spy(fw):
        entry = real(fw)
        made_sets.add(entry["share_set_id"])
        return entry

    monkeypatch.setattr(kg, "_weight_entry", spy)

    def one(i):
        fw = weight_sets[i % len(weight_sets)]
        try:
            return kg.get_decryption_keys("s%d" % (i % 3), credentials={"fusion_weight": fw, "label": "L%d" % (i % 4)})
        except FESchemeError as exc:
            assert "WEIGHT_SHARINGS_KEPT" in str(exc)
            return None

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(one, range(60)))
    issued = [dk for dk in results if dk is not None]
    assert issued and all(dk["share_set_id"] in made_sets for dk in issued)
    for entry in kg.dict_weight_shares.values():
        assert all(parts[sid]["share_set_id"] == entry["share_set_id"]
                   for parts in entry["label_parts"].values() for sid in SIDS)
    assert len(kg.dict_weight_shares) == 2


def test_i_c_label_parts_go_with_their_sharing(monkeypatch):
    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 1)
    kg, _ = _setup()
    creds = {"fusion_weight": WEIGHTS, "label": "L", "n_chunks": 1}
    first = kg.get_decryption_keys("s0", credentials=creds)
    assert kg.get_decryption_keys("s0", credentials=creds) == first  # kept: the same shares
    entry = next(iter(kg.dict_weight_shares.values()))
    assert len(entry["label_parts"]) == 1 and kg.dict_dk == {}  # dict_dk holds whole key shares only
    kg.get_decryption_keys("s0", credentials=dict(creds, key_part="full"))
    assert len(kg.dict_dk) == 1
    kg.get_weight_key_share("s0", credentials={"fusion_weight": {nid: [2] for nid in NIDS}})  # evicts WEIGHTS
    assert [e["label_parts"] for e in kg.dict_weight_shares.values()] == [{}] and len(kg.dict_dk) == 1
    again = kg.get_decryption_keys("s0", credentials=creds)
    assert again["share_set_id"] != first["share_set_id"]


def test_label_parts_of_the_last_labels_are_kept(monkeypatch):
    """Per sharing of sk_y the label parts of the last
    LABEL_PARTS_KEPT labels are kept (least recently used out); a label asked
    for again later gets a new sharing of its terms, with the same share set
    and noise key."""
    monkeypatch.setattr(tm, "LABEL_PARTS_KEPT", 2)
    kg, _ = _setup()
    creds = {lab: {"fusion_weight": WEIGHTS, "label": lab} for lab in ("L0", "L1", "L2", "L3")}
    first = kg.get_decryption_keys("s0", credentials=creds["L0"])
    first_l1 = kg.get_decryption_keys("s0", credentials=creds["L1"])
    assert kg.get_decryption_keys("s1", credentials=creds["L0"])["pieces"]  # a hit: L0 most recent
    kg.get_decryption_keys("s0", credentials=creds["L2"])  # drops L1, not L0
    entry = next(iter(kg.dict_weight_shares.values()))
    assert [json.loads(i)[1] for i in entry["label_parts"]] == ["L0", "L2"]
    assert kg.get_decryption_keys("s0", credentials=creds["L0"]) == first
    again = kg.get_decryption_keys("s0", credentials=creds["L1"])  # dropped: a new sharing of its terms
    assert len(entry["label_parts"]) == 2
    assert again["pieces"] != first_l1["pieces"]
    assert (again["share_set_id"], again["noise_key"]) == (first_l1["share_set_id"], first_l1["noise_key"])


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("ct_format", ["binary", "json"])
@pytest.mark.parametrize("t", [2, 3])
def test_sharing_ids_keep_two_sharings_of_a_label_apart(monkeypatch, name, ct_format, t):
    """Partials of two sharings of one
    label's terms would decode to a wrong value inside the bound; every
    sharing has a sharing_id, which the partials carry, and combiners refuse
    a mix (MixedSharingsError) instead."""
    monkeypatch.setattr(tm, "LABEL_PARTS_KEPT", 1)
    kg, pp = _setup(t=t, ct_format=ct_format)
    creds = {"fusion_weight": WEIGHTS, "label": "L"}
    enrolled = SIDS[:t]
    servers = {}
    for sid in enrolled:
        servers[sid] = _obj(pp, sid)
        servers[sid].add_weight_key_share(kg.get_weight_key_share(sid, credentials=creds))
    first = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in enrolled}
    assert len({dk["sharing_id"] for dk in first.values()}) == 1
    kg.get_decryption_keys(enrolled[0], credentials={"fusion_weight": WEIGHTS, "label": "M"})  # evicts L's parts
    second = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in enrolled}
    assert len({dk["sharing_id"] for dk in second.values()}) == 1
    assert second[enrolled[0]]["sharing_id"] != first[enrolled[0]]["sharing_id"]
    cts = {nid: _obj(pp, nid, kg).encrypt([[1, 2, 3]], "L") for nid in NIDS}
    right = [sum(WEIGHTS[nid][0] * v for nid in NIDS) for v in (1, 2, 3)]
    comb = _obj(pp, "comb")
    with native_backend.backend(name):
        parts1 = {sid: servers[sid].share_decrypt(cts, creds, first[sid], enrolled) for sid in enrolled}
        parts2 = {sid: servers[sid].share_decrypt(cts, creds, second[sid], enrolled) for sid in enrolled}
        assert {p["sharing_id"] for p in parts1.values()} == {first[enrolled[0]]["sharing_id"]}
        assert comb.combine_decrypt(parts1) == right == comb.combine_decrypt(parts2)
        mixed = dict(parts1, **{enrolled[-1]: parts2[enrolled[-1]]})
        with pytest.raises(tm.MixedSharingsError, match="fetch the label parts again from the key generator"):
            comb.combine_decrypt(mixed)
        with pytest.raises(tm.MixedSharingsError):  # also among several enrolled sets
            comb.combine_decrypt([parts1, mixed])
        # partials without an id (of key shares of earlier versions) combine only among themselves
        bare = {sid: {k: v for k, v in p.items() if k != "sharing_id"} for sid, p in parts1.items()}
        assert comb.combine_decrypt(bare) == right
        with pytest.raises(tm.MixedSharingsError):
            comb.combine_decrypt(dict(bare, **{enrolled[-1]: parts1[enrolled[-1]]}))
        with pytest.raises(FESchemeError, match="sharing_id"):
            comb.combine_decrypt(dict(parts1, **{enrolled[-1]: dict(parts1[enrolled[-1]], sharing_id=5)}))
    # a key share whose sharing_id is not a string is refused by the server
    with pytest.raises(FEKeyError, match="sharing_id"):
        servers[enrolled[0]].share_decrypt(cts, creds, dict(first[enrolled[0]], sharing_id=["x"]), enrolled)


def test_whole_key_shares_carry_a_sharing_id_too():
    kg, pp = _setup(key_shares="full")
    creds = {"fusion_weight": WEIGHTS, "label": "r", "n_chunks": 2}
    dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
    assert len({dk["sharing_id"] for dk in dks.values()}) == 1 and isinstance(dks["s0"]["sharing_id"], str)
    _, _, parts = _round(kg, pp)
    assert all(part["sharing_id"] == dks["s0"]["sharing_id"] for p in parts.values() for part in p[0]["parts"])


@needs_native
@pytest.mark.usefixtures("_native")
def test_i_d_exhausted_test_bytes_are_not_malformed_key_material(monkeypatch):
    kg, _ = _setup(t=3)  # Shamir: the native dealing
    polys = [[1] * 16, [2] * 16]
    points = list(tm.share_points(SIDS).values())
    monkeypatch.setattr(native_backend, "_random_override", lambda n: b"\x01" * 8)
    with pytest.raises(RuntimeError, match="test bytes ran out"):
        kg._share_polys_native(polys, points)
    monkeypatch.setattr(native_backend, "_random_override", None)
    width = (int(kg.q).bit_length() + 7) // 8
    with pytest.raises(FEKeyError, match="malformed key material"):  # binary words not below q
        kg._share_polys_native([int(kg.q).to_bytes(width, "little") * 16], points, binary=True)
    for bad in ([1.5] * 16, ["x"] * 16):
        with pytest.raises(FEKeyError, match="malformed key material"):
            kg._share_polys_native([bad], points)
    assert len(kg._share_polys_native(polys, points)) == 2


def test_kl_accounting_with_split_key_shares():
    kg, pp = _setup(t=3, flooding_accounting="kl", max_partial_decryptions=64)
    assert kg.key_shares == "split" and pp["sharing"] == "shamir"
    budgets = {sid: PartialDecryptionBudget(64) for sid in SIDS}
    creds_probe = {"fusion_weight": WEIGHTS, "label": "r", "n_chunks": 2}
    servers = {}
    for sid in SIDS:
        srv = tm.ThresholdMCFERingLWE({"id": sid, "precision": 0, "partial_budget": budgets[sid], "keys": {"pp": pp}})
        srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials=creds_probe))
        servers[sid] = (srv, kg.get_decryption_keys(sid, credentials=creds_probe))
    _round(kg, pp, enrolled=SIDS, servers=servers)
    assert all(b.used == 2 for b in budgets.values())
