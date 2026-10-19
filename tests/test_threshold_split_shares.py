"""Split key shares of the threshold Ring-LWE MCFE and binary key shares.

The label-independent part (``sk_y``) is issued once per weights
(``get_weight_key_share``, kept by the servers with ``add_weight_key_share``);
per label only the label part (``key_part="label"``: ``z`` and ``z_chunks``,
optionally without ``z``). With binary public parameters every polynomial of
the Shamir and replicated key shares is bytes. Checked: rounds decrypt on both
backends with equal partials, the weight part is reused (and NTT(sk_y) cached
across labels), parts of different weights, share sets, setups or servers are
refused, and binary dealing equals the Python code on the same random bytes.
"""

from __future__ import annotations

import pickle

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes import threshold_utils
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

NATIVE = native_backend.available()
BACKENDS = ["python"] + (["native"] if NATIVE else [])
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
WEIGHTS = {nid: [y] for nid, y in zip(NIDS, (1, -2, 3))}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _setup(t=3, ct_format="binary", **extra):
    config = {"n": 3, "lst_nid": NIDS, "s": 3, "t": t, "lst_sid": SIDS, "eta": 1, "ring_n": 32, "sec_param": 64,
              "bound_x": 100, "bound_y": 3, "stat_param": 64, "ct_format": ct_format, **extra}
    kg = tm.ThresholdMCFERingLWEKeyGenerator(config)
    kg.setup()
    return kg, kg.get_public_parameters()


def _obj(pp, nid, kg=None):
    keys = {"pp": pp} if kg is None else {"pp": pp, "sk": kg.get_private_keys(nid)}
    return tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": keys})


def _servers(kg, pp, sids=SIDS):
    out = {}
    for sid in sids:
        out[sid] = _obj(pp, sid)
        out[sid].add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": WEIGHTS}))
    return out


@pytest.mark.parametrize("ct_format", ["json", "binary"])
@pytest.mark.parametrize("t,sharing", [(3, "shamir"), (2, "replicated"), (3, "replicated")])
@pytest.mark.parametrize("label_term", [True, False])
def test_split_rounds_decrypt_and_backends_agree(t, sharing, ct_format, label_term):
    kg, pp = _setup(t=t, ct_format=ct_format, sharing=sharing)
    clients = {nid: _obj(pp, nid, kg) for nid in NIDS}
    weight = kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    assert weight["part"] == "weights" and "z" not in weight and "z_chunks" not in weight
    assert kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS}) == weight  # issued once
    results = {}
    for r in range(2):
        label = "round-{}".format(r)
        rng = np.random.default_rng(r)
        grads = {nid: [rng.integers(-100, 101, 70)] for nid in NIDS}
        enc = {nid: clients[nid].encrypt_lst_ndarray(grads[nid], label=label) for nid in NIDS}
        creds = {"fusion_weight": WEIGHTS, "label": label, "n_chunks": clients["c0"].key_chunks(enc["c0"]),
                 "key_part": "label", "label_term": label_term}
        dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
        held = dks["s0"]["pieces"][0]["share"] if "pieces" in dks["s0"] else dks["s0"]
        assert dks["s0"]["part"] == "label" and dks["s0"]["share_set_id"] == weight["share_set_id"]
        assert "sk_y" not in held and ("z" in held) == label_term and len(held["z_chunks"]) == 3
        poly = held["z_chunks"][0]
        assert isinstance(poly, bytes) == (ct_format == "binary")
        if ct_format == "binary":
            assert len(poly) == pp["ring_n"] * ((int(pp["q"]).bit_length() + 7) // 8)
        expected = sum(WEIGHTS[nid][0] * grads[nid][0] for nid in NIDS)
        for name in BACKENDS:
            with native_backend.backend(name):
                srv = _servers(kg, pp)
                groups = [{sid: srv[sid].compute_lst_ndarray_ct(enc, creds, dks[sid], list(en)) for sid in en}
                          for en in (SIDS[:t], SIDS[-t:])]
                out = srv["s0"].decrypt_lst_ndarray_ct(groups)
            assert np.array_equal(out[0], expected)
            results[(r, name)] = groups
        if NATIVE:
            assert results[(r, "python")] == results[(r, "native")]


def test_split_parts_are_smaller_and_a_weight_part_is_reused(monkeypatch):
    kg, pp = _setup(t=3)
    full = kg.get_decryption_keys("s0", credentials={"fusion_weight": WEIGHTS, "label": "L", "n_chunks": 2,
                                                     "key_part": "full"})
    weight = kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    label = kg.get_decryption_keys("s0", credentials={"fusion_weight": WEIGHTS, "label": "L", "n_chunks": 2,
                                                      "key_part": "label", "label_term": False})
    def size(o):  # bytes of the key-share polynomials
        if isinstance(o, dict):
            return sum(size(v) for v in o.values())
        if isinstance(o, list):
            return sum(size(v) for v in o)
        return len(o) if isinstance(o, bytes) else 0

    poly = pp["ring_n"] * ((int(pp["q"]).bit_length() + 7) // 8)
    assert (size(full), size(weight), size(label)) == (6 * poly, 3 * poly, 2 * poly)
    assert len(pickle.dumps(label)) < len(pickle.dumps(full))
    # the same noise key as the full key share (per server, weights and label)
    assert label["noise_key"] == full["noise_key"]
    # a new setup: new shares and share set (the old weight part does not complete the new label parts)
    old = weight["share_set_id"]
    kg.setup()
    assert kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})["share_set_id"] != old


@pytest.mark.skipif(not NATIVE, reason="native extension not built")
def test_weight_part_ntt_is_cached_across_labels(monkeypatch):
    import collections

    kg, pp = _setup(t=3)
    clients = {nid: _obj(pp, nid, kg) for nid in NIDS}
    real, counts = native_backend._ext, collections.Counter()

    class Spy:
        def __getattr__(self, name):
            fn = getattr(real, name)

            def call(*args, **kwargs):
                counts[name] += 1
                return fn(*args, **kwargs)

            return call if callable(fn) else fn

    with native_backend.backend("native"):
        srv = _servers(kg, pp)
        monkeypatch.setattr(native_backend, "_ext", Spy())
        per_round = []
        for r in range(3):
            label = "r{}".format(r)
            enc = {nid: cl.encrypt_lst_ndarray([np.arange(40) % 7], label=label) for nid, cl in clients.items()}
            creds = {"fusion_weight": WEIGHTS, "label": label, "n_chunks": 2, "key_part": "label", "label_term": False}
            dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
            before = counts["ntt_mont"]
            parts = {sid: srv[sid].compute_lst_ndarray_ct(enc, creds, dks[sid], SIDS) for sid in SIDS}
            per_round.append(counts["ntt_mont"] - before)
            assert np.array_equal(srv["s0"].decrypt_lst_ndarray_ct(parts)[0], 2 * (np.arange(40) % 7))
    assert per_round == [9, 0, 0]  # 3 servers x 3 clients once, then from the cache


def test_split_parts_are_refused_when_they_do_not_belong_together():
    kg, pp = _setup(t=3)
    clients = {nid: _obj(pp, nid, kg) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": WEIGHTS, "label": "L", "key_part": "label"}
    label = kg.get_decryption_keys("s0", credentials=creds)
    srv = _obj(pp, "s0")
    with pytest.raises(FEKeyError, match="add_weight_key_share"):
        srv.share_decrypt(cts, creds, label, SIDS)
    weight = kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    with pytest.raises(FEKeyError, match="not a weight key share"):
        srv.add_weight_key_share(kg.get_weight_key_share("s1", credentials={"fusion_weight": WEIGHTS}))
    with pytest.raises(FEKeyError, match="not a weight key share"):
        srv.add_weight_key_share(label)
    srv.add_weight_key_share(weight)
    parts = {"s0": srv.share_decrypt(cts, creds, label, SIDS)}
    with pytest.raises(FEKeyError, match="together with a label"):
        srv.share_decrypt(cts, creds, weight, SIDS)
    # a weight part whose fields differ from the label part's
    for field, value in [("pp_binding", "00" * 32), ("noise_version", 1), ("flooding", "gaussian"),
                         ("mac_keys", {nid: "00" * 32 for nid in NIDS})]:
        other = _obj(pp, "s0")
        other.add_weight_key_share(dict(weight, **{field: value}))
        with pytest.raises(FEKeyError, match="belong together"):
            other.share_decrypt(cts, creds, label, SIDS)
    # a label part of another share set (another setup, same weights): no weight part for it
    kg2, _ = _setup(t=3)
    label2 = kg2.get_decryption_keys("s0", credentials=creds)
    with pytest.raises(FEKeyError, match="add_weight_key_share"):
        srv.share_decrypt(cts, creds, label2, SIDS)
    # without the label's own term: chunks only
    no_z = kg.get_decryption_keys("s0", credentials=dict(creds, label_term=False, n_chunks=1))
    with pytest.raises(FEKeyError, match="label itself"):
        srv.share_decrypt(cts, creds, no_z, SIDS)
    for name, value in [("key_part", "both"), ("label_term", False)]:
        with pytest.raises(FEValidationError):
            kg.get_decryption_keys("s0", credentials=dict(creds, **{name: value}))
    kgs, _ = _setup(t=2, sharing="replicated-seeded")
    with pytest.raises(FEValidationError, match="split"):
        kgs.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    for sid in SIDS[1:]:
        s = _obj(pp, sid)
        s.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": WEIGHTS}))
        parts[sid] = s.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), SIDS)
    assert srv.combine_decrypt(parts) == [sum(WEIGHTS[nid][0] * v for nid in NIDS) for v in (1, 2, 3)]


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("sharing,t", [("shamir", 3), ("replicated", 2)])
def test_malformed_binary_key_shares_are_refused(name, sharing, t):
    kg, pp = _setup(t=t, sharing=sharing)
    clients = {nid: _obj(pp, nid, kg) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": WEIGHTS, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=dict(creds, key_part="full"))  # sk_y and z in one share
    q = int(pp["q"])
    wb = (q.bit_length() + 7) // 8

    def bad_poly(poly):
        return q.to_bytes(wb, "little") + poly[wb:]  # a value q: not below q

    if sharing == "shamir":
        bads = [dict(dk, z=bad_poly(dk["z"])), dict(dk, sk_y=dict(dk["sk_y"], c1=bad_poly(dk["sk_y"]["c1"])))]
    else:
        pieces = [dict(p, share=dict(p["share"], z=bad_poly(p["share"]["z"]))) for p in dk["pieces"]]
        bads = [dict(dk, pieces=pieces)]
    with native_backend.backend(name):
        for bad in bads:
            with pytest.raises((FEKeyError, FESchemeError), match="malformed binary key"):
                _obj(pp, "s0").share_decrypt(cts, creds, bad, SIDS[:t])


@pytest.mark.skipif(not NATIVE, reason="native extension not built")
@pytest.mark.parametrize("sharing,t", [("shamir", 3), ("replicated", 2)])
def test_binary_dealing_is_the_python_dealing(sharing, t, monkeypatch):
    from tests.test_native_backend import ByteStream, _BytesSystemRandom

    kg, pp = _setup(t=t, sharing=sharing)
    q = gp.mpz(pp["q"])
    rng = np.random.default_rng(3)
    material = {"sk_y": {nid: [int(v) for v in rng.integers(0, 2 ** 62, 32)] for nid in NIDS},
                "z": [int(v) for v in rng.integers(0, 2 ** 62, 32)],
                "z_chunks": [[int(v) for v in rng.integers(0, 2 ** 62, 32)] for _ in range(2)]}
    binary_terms = threshold_utils.binary_material({"z": material["z"], "z_chunks": material["z_chunks"]}, q)
    for mat in (material, binary_terms):
        with native_backend.backend("python"):
            monkeypatch.setattr(threshold_utils, "_CSPRNG", _BytesSystemRandom(ByteStream(8)))
            py = kg._share_material(mat)
        with native_backend.backend("native"):
            monkeypatch.setattr(native_backend, "_random_override", ByteStream(8).take)
            nat = kg._share_material(mat)
            monkeypatch.setattr(native_backend, "_random_override", None)
        assert py == nat
        # binary shares are the binary form of the decimal ones
        kgj, _ = _setup(t=t, sharing=sharing, ct_format="json")
        with native_backend.backend("python"):
            monkeypatch.setattr(threshold_utils, "_CSPRNG", _BytesSystemRandom(ByteStream(8)))
            dec = kgj._share_material({k: (v if mat is material else material[k]) for k, v in mat.items()})
        assert _binary_form(dec, q) == py


def _binary_form(obj, q):
    """Decimal key shares with their polynomials as binary words."""
    if isinstance(obj, dict):
        return {k: _binary_form(v, q) for k, v in obj.items()}
    if isinstance(obj, list) and obj and all(isinstance(v, str) for v in obj):
        return rlu.encode_values([int(v) for v in obj], q)
    if isinstance(obj, list):
        return [_binary_form(v, q) for v in obj]
    return obj


def test_binary_partial_digests_known_answer():
    """Pins the v2 tag message and the request digests (format-2 derivations)."""
    q = (1 << 70) + 0x2_0001
    data = rlu.encode_threshold_ciphertext([[1, 2, 3, 4]], [5, 6, 7, 8], 2, q, 4)
    ct = {"ct": data, "k": 2, "label_hash": 2}
    digests = tm.binary_ciphertext_digests(ct, q, 1, 4)
    message = tm.binary_tag_message(b"\x11" * 32, "lab", ct, digests)
    import hashlib

    assert hashlib.sha256(message).hexdigest() == hashlib.sha256(
        b"pyfe4ai/tmcfe-ring-lwe/ciphertext-tag/v2-binary|" + b"\x11" * 32 + b"\x00\x00\x00\x03lab\x02"
        + b"\x00\x00\x00\x02" + digests[1] + digests[2]).hexdigest()
    assert digests[1] == hashlib.sha256(b"pyfe4ai/tmcfe-ring-lwe/ct0/v2|" + data[:14 + 2 * 9]).digest()
    assert digests[2] == hashlib.sha256(b"pyfe4ai/tmcfe-ring-lwe/ct1/v2|" + data[14 + 2 * 9:]).digest()
    ct_digest, key_inputs = tm.binary_request_digests({"b": digests, "a": digests})
    name = lambda s: len(s).to_bytes(4, "big") + s
    assert key_inputs == hashlib.sha256(b"pyfe4ai/tmcfe-ring-lwe/key-inputs/v2|" + name(b'"a"') + digests[2]
                                        + name(b'"b"') + digests[2]).hexdigest()
    assert ct_digest == hashlib.sha256(
        b"pyfe4ai/tmcfe-ring-lwe/ct-digest/v2|" + b"".join(name(n) + (2).to_bytes(4, "big") + digests[1] + digests[2]
                                                          for n in (b'"a"', b'"b"'))).hexdigest()


@pytest.mark.parametrize("split", [False, True])
def test_binary_replicated_shares_with_a_server_assigned_no_piece(split):
    """All three servers enrolled at t = 2: one is assigned no piece (its sum
    is 0, in the binary layout of its pieces); the result decrypts and both
    backends agree."""
    kg, pp = _setup(t=2, sharing="replicated")
    clients = {nid: _obj(pp, nid, kg) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": WEIGHTS, "label": "L", "key_part": "label" if split else "full"}
    dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
    out = {}
    for name in BACKENDS:
        with native_backend.backend(name):
            srv = _servers(kg, pp) if split else {sid: _obj(pp, sid) for sid in SIDS}
            parts = {sid: srv[sid].share_decrypt(cts, creds, dks[sid], SIDS) for sid in SIDS}
            assert srv["s0"].combine_decrypt(parts) == [sum(WEIGHTS[nid][0] * v for nid in NIDS) for v in (1, 2, 3)]
        out[name] = parts
    if NATIVE:
        assert out["python"] == out["native"]


def test_the_generators_weight_store_is_bounded(monkeypatch):
    """The key generator keeps the sharings of sk_y for the last
    WEIGHT_SHARINGS_KEPT weights; weights issued again after that get a new
    share set, which servers holding the old weight part refuse."""
    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 1)
    kg, pp = _setup(t=3)
    other = {nid: [2] for nid in NIDS}
    first = kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    kg.get_weight_key_share("s0", credentials={"fusion_weight": other})
    assert len(kg.dict_weight_shares) == 1
    again = kg.get_weight_key_share("s0", credentials={"fusion_weight": WEIGHTS})
    assert again["share_set_id"] != first["share_set_id"]
    clients = {nid: _obj(pp, nid, kg) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": WEIGHTS, "label": "L", "key_part": "label"}
    srv = _obj(pp, "s0")
    srv.add_weight_key_share(first)
    with pytest.raises(FEKeyError, match="add_weight_key_share"):
        srv.share_decrypt(cts, creds, kg.get_decryption_keys("s0", credentials=creds), SIDS)
