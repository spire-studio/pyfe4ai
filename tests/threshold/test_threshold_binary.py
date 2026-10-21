"""Binary ciphertexts of the threshold Ring-LWE MCFE (``ct_format="binary"``).

Format 2 (ct0 truncated to the batch length, every value strictly below q),
tags v2 over the public parameters' digest and the bytes' digests, binary
partial decryptions of the first ``k`` coefficients, and the uniform
flooding noise drawn natively (also on the JSON path). Both backends give the
same bytes; JSON public parameters, ciphertexts and partials are unchanged;
any mix of formats is refused.
"""

from __future__ import annotations

import hashlib
import json

import gmpy2 as gp
import numpy as np
import os
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.threshold import binding as threshold_binding
from pyfe4ai.schemes.threshold import codec as threshold_codec
from pyfe4ai.schemes.threshold import roles as threshold_roles
from pyfe4ai.schemes.mcfe.labels import chunk_label, encode_label
from pyfe4ai.schemes.threshold.binding import pp_binding, tag_ciphertext, tag_ciphertext_message
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.core.formats import binary as fmt_binary
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _setup(t=2, ct_format="binary", flooding="uniform", ring_n=32, eta=1, **extra):
    """Whole key shares (split ones: test_threshold_split_shares.py); *ct_format*
    ``None`` leaves it to the default (binary)."""
    config = {"n": 3, "lst_nid": NIDS, "s": 3, "t": t, "lst_sid": SIDS, "eta": eta, "ring_n": ring_n,
              "sec_param": 64, "bound_x": 100, "bound_y": 3, "stat_param": 64, "flooding": flooding,
              "key_shares": "full", **extra}
    if ct_format is not None:
        config["ct_format"] = ct_format
    kg = tm.ThresholdMCFERingLWEKeyGenerator(config)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    return kg, pp, clients


def _obj(pp, sid):
    return tm.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}})


def _round(kg, pp, clients, rng, sizes=((70,), (2, 3))):
    grads = {nid: [rng.integers(-100, 101, s) for s in sizes] for nid in NIDS}
    weights = {nid: [int(y)] for nid, y in zip(NIDS, (1, -2, 3))}
    enc = {nid: clients[nid].encrypt_lst_ndarray(grads[nid], label="r") for nid in NIDS}
    creds = {"fusion_weight": weights, "label": "r", "n_chunks": clients["c0"].key_chunks(enc["c0"])}
    dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
    expected = [sum(weights[nid][0] * grads[nid][l] for nid in NIDS) for l in range(len(sizes))]
    return enc, creds, dks, expected


# ------------------------------------------------------------------ format 2


def test_format_2_known_answer():
    q = (1 << 70) + 0x2_0001  # 9-byte values
    ct0 = [[1, 2, 3, 4], [q - 1, 0, 7, 8]]
    ct1 = [5, 6, 7, q - 2]
    data = fmt_binary.encode_threshold_ciphertext(ct0, ct1, 2, q, 4)
    assert data[:14] == b"PFRL" + bytes([2, 9, 2, 0, 2, 0]) + (2).to_bytes(4, "little")
    assert len(data) == 14 + (2 * 2 + 4) * 9
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(
        data[:14] + b"".join(v.to_bytes(9, "little") for v in [1, 2, q - 1, 0, 5, 6, 7, q - 2])).hexdigest()
    assert fmt_binary.decode_threshold_ciphertext(data, q, 2, 4) == (2, [[1, 2], [q - 1, 0]], ct1)
    # format 1 -> format 2 is the same bytes
    f1 = fmt_binary.encode_binary_ciphertext(ct0, ct1, q, 4)
    assert fmt_binary.threshold_ciphertext_from_format1(f1, 2, q, 2, 4) == data


def test_format_2_refuses_non_canonical_and_malformed_bytes():
    q = (1 << 70) + 0x2_0001
    data = bytearray(fmt_binary.encode_threshold_ciphertext([[1, 2, 3, 4]], [5, 6, 7, 8], 3, q, 4))
    for pos in range(14, len(data), 9):  # any value replaced by q (a second encoding of 0)
        bad = bytes(data[:pos]) + q.to_bytes(9, "little") + bytes(data[pos + 9:])
        with pytest.raises(ValueError):
            fmt_binary.decode_threshold_ciphertext(bad, q, 1, 4)
    for pos, value in [(4, 1), (5, 10), (6, 2), (8, 3), (9, 1), (10, 0), (10, 5)]:
        bad = bytearray(data)
        bad[pos] = value
        with pytest.raises(ValueError):
            fmt_binary.decode_threshold_ciphertext(bytes(bad), q, 1, 4)
    for bad in (bytes(data[:-1]), bytes(data) + b"\0", b"", "text"):
        with pytest.raises(ValueError):
            fmt_binary.decode_threshold_ciphertext(bad, q, 1, 4)
    f1 = fmt_binary.encode_binary_ciphertext([[1, 2, 3, 4]], [5, 6, 7, 8], q, 4)
    with pytest.raises(ValueError):  # format 1 is not format 2
        fmt_binary.decode_threshold_ciphertext(f1, q, 1, 4)
    with pytest.raises(ValueError):
        fmt_binary.decode_values(q.to_bytes(9, "little"), q, 1)


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
@pytest.mark.parametrize("bits_q", [60, 120, 215])
def test_native_uniform_flooding_is_the_python_noise(bits_q):
    q = rlu.next_ntt_prime(gp.mpz(1) << (bits_q - 1), 16)
    with native_backend.backend("native"):  # also when the suite runs on the Python backend
        ctx = native_impl.ring_context(q, 16)
    for bits in [b for b in (1, 7, 8, 63, 64, bits_q - 2, bits_q - 1) if b < bits_q]:
        for seed in (b"", b"k" * 40, bytes(range(200))):
            material = threshold_roles.uniform_flooding_material(seed)
            want = [v % int(q) for v in threshold_roles.uniform_flooding_poly(bits, 37, seed)]
            assert native_backend.ext().uniform_flooding(ctx.cap, bits, material, 37) == want
            # a prefix of the stream: k values are the first k of n
            assert threshold_roles.uniform_flooding_poly(bits, 5, seed) == threshold_roles.uniform_flooding_poly(bits, 37, seed)[:5]
    with pytest.raises(ValueError):
        native_backend.ext().uniform_flooding(ctx.cap, q.bit_length(), threshold_roles.uniform_flooding_material(b""), 3)


# ----------------------------------------------------------- whole rounds


@pytest.mark.parametrize("flooding", ["uniform", "gaussian"])
@pytest.mark.parametrize("t", [2, 3])
def test_binary_rounds_decrypt_and_backends_agree(t, flooding):
    kg, pp, clients = _setup(t=t, flooding=flooding)
    assert pp["ct_format"] == "binary"
    enc, creds, dks, expected = _round(kg, pp, clients, np.random.default_rng(t))
    ring_n, wb = pp["ring_n"], (int(pp["q"]).bit_length() + 7) // 8
    chunks = enc["c0"][0]["chunks"]
    assert [c["k"] for c in chunks] == [32, 32, 6]
    assert [len(c["ct"]) for c in chunks] == [14 + (k + ring_n) * wb for k in (32, 32, 6)]
    outs = {}
    for name in BACKENDS:
        with native_backend.backend(name):
            groups = [{sid: _obj(pp, sid).compute_lst_ndarray_ct(enc, creds, dks[sid], list(en)) for sid in en}
                      for en in (SIDS[:t], SIDS[-t:])]
            result = _obj(pp, "comb").decrypt_lst_ndarray_ct(groups)
        outs[name] = groups
        for got, want in zip(result, expected):
            assert np.array_equal(got, want)
    part = outs["python"][0][SIDS[0]][0]["parts"][2]
    assert isinstance(part["w"], bytes) and len(part["w"]) == len(part["c_y"]) == 6 * wb and part["k"] == 6
    if len(outs) == 2:
        assert outs["python"] == outs["native"]
    # one single ciphertext per client (eta rows)
    single = {nid: cl.encrypt([[1, -2, 3]], "L") for nid, cl in clients.items()}
    c2 = {"fusion_weight": creds["fusion_weight"], "label": "L"}
    dk = {sid: kg.get_decryption_keys(sid, credentials=c2) for sid in SIDS}
    for name in BACKENDS:
        with native_backend.backend(name):
            parts = {sid: _obj(pp, sid).share_decrypt(single, c2, dk[sid], SIDS[:t]) for sid in SIDS[:t]}
            assert _obj(pp, "comb").combine_decrypt(parts) == [2, -4, 6]


def test_binary_keeps_the_noise_of_one_key_part():
    """The same request answered twice gets the same partial; JSON public
    parameters of the same deployment would give another derivation, but a key
    share is bound to one format (pp_binding)."""
    import copy

    kg, pp, clients = _setup(t=3)
    enc, creds, dks, _ = _round(kg, pp, clients, np.random.default_rng(1))
    first = _obj(pp, "s0").compute_lst_ndarray_ct(enc, creds, dks["s0"], SIDS)
    again = _obj(pp, "s0").compute_lst_ndarray_ct(copy.deepcopy(enc), creds, copy.deepcopy(dks["s0"]), SIDS)
    assert first == again
    # a re-sharing of the same key part (another n_chunks) answers with the same noise:
    # the partials differ only by the shares, whose difference t - 1 servers can compute
    more = dict(creds, n_chunks=creds["n_chunks"] + 1)
    dk2 = kg.get_decryption_keys("s0", credentials=more)
    assert dk2["noise_key"] == dks["s0"]["noise_key"]


def test_json_public_parameters_are_unchanged():
    """With the options of versions <= 0.3.0 (JSON, a prime q) the public
    parameters and their binding are those of these versions."""
    kg, pp, clients = _setup(ct_format="json", ring_modulus="prime")
    assert "ct_format" not in pp
    assert threshold_binding.tmcfe_ring_lwe_binding_fields(pp) == threshold_binding.TMCFE_RING_LWE_BINDING_FIELDS
    assert kg._binding == pp_binding(pp, threshold_binding.TMCFE_RING_LWE_BINDING_FIELDS, threshold_binding.TMCFE_RING_LWE_BINDING_DEFAULTS)
    enc, creds, dks, expected = _round(kg, pp, clients, np.random.default_rng(2))
    assert "ct0" in enc["c0"][0]["chunks"][0]
    outs = []
    for name in BACKENDS:
        with native_backend.backend(name):
            parts = {sid: _obj(pp, sid).compute_lst_ndarray_ct(enc, creds, dks[sid], SIDS[:2]) for sid in SIDS[:2]}
            outs.append(parts)
            result = _obj(pp, "comb").decrypt_lst_ndarray_ct(parts)
        assert all(np.array_equal(a, b) for a, b in zip(result, expected))
        assert isinstance(parts["s0"][0]["parts"][0]["w"], list) and len(parts["s0"][0]["parts"][0]["w"]) == 32
    assert outs[0] == outs[-1]


# ------------------------------------------------------------------ refusals


def _retag_binary(kg, pp, nid, label, ct, layer=None, pp_digest=None):
    coded = threshold_codec._chunk_with_code(ct, layer) if layer is not None else ct
    digests = threshold_codec.binary_ciphertext_digests(coded, pp["q"], pp["eta"], pp["ring_n"])
    digest = bytes.fromhex(pp_digest or pp_binding(pp, threshold_binding.tmcfe_ring_lwe_binding_fields(pp), threshold_binding.TMCFE_RING_LWE_BINDING_DEFAULTS))
    return tag_ciphertext_message(dict(ct), kg.get_private_keys(nid)["mac_keys"],
                                  threshold_codec.binary_tag_message(digest, encode_label(label), coded, digests))


@pytest.mark.parametrize("name", BACKENDS)
def test_binary_refuses_non_canonical_values_even_when_tagged(name):
    """A client holding the MAC keys re-encodes a value v as v + q: refused,
    else the same key part would be answered with a second noise."""
    kg, pp, clients = _setup(t=3)
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    q = int(pp["q"])
    wb = (q.bit_length() + 7) // 8
    data = cts["c1"]["ct"]
    pos = 14 + 3 * wb + 5 * wb  # a ct1 value
    v = int.from_bytes(data[pos:pos + wb], "little")
    if v + q >= 1 << (8 * wb):
        pytest.skip("no second encoding of this value in wb bytes")
    bad = data[:pos] + (v + q).to_bytes(wb, "little") + data[pos + wb:]
    tampered = dict(cts, c1=_retag_binary(kg, pp, "c1", "L", dict(cts["c1"], ct=bad)))
    with native_backend.backend(name), pytest.raises(FESchemeError, match="malformed ciphertext of c1"):
        _obj(pp, "s0").share_decrypt(tampered, creds, dk, SIDS)


def test_mixed_formats_are_refused():
    kg, pp, clients = _setup(t=3)
    kgj, ppj, clientsj = _setup(t=3, ct_format="json")
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    ctsj = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clientsj.items()}
    dk, dkj = kg.get_decryption_keys("s0", credentials=creds), kgj.get_decryption_keys("s0", credentials=creds)
    srv, srvj = _obj(pp, "s0"), _obj(ppj, "s0")
    q, eta, ring_n = int(pp["q"]), pp["eta"], pp["ring_n"]
    # a JSON ciphertext under binary public parameters, tagged with the JSON payload
    k, ct0, ct1 = fmt_binary.decode_threshold_ciphertext(cts["c1"]["ct"], q, eta, ring_n)
    as_json = {"ct0": [[str(v) for v in row] + ["0"] * (ring_n - k) for row in ct0], "ct1": [str(v) for v in ct1],
               "k": k, "label_hash": cts["c1"]["label_hash"]}
    as_json = tag_ciphertext(as_json, kg.get_private_keys("c1")["mac_keys"], threshold_codec._TAG_DOMAIN,
                             threshold_codec._tag_payload(encode_label("L"), as_json))
    with pytest.raises(FESchemeError, match="is JSON; the public parameters' format is binary"):
        srv.share_decrypt(dict(cts, c1=as_json), creds, dk, SIDS)
    # format 1 under binary public parameters, tagged with a v2 message: refused
    f1 = fmt_binary.encode_binary_ciphertext([row + [0] * (ring_n - k) for row in ct0], ct1, q, ring_n)
    with pytest.raises(FESchemeError, match="not authenticated"):
        srv.share_decrypt(dict(cts, c1=dict(cts["c1"], ct=f1)), creds, dk, SIDS)
    # a JSON (v1) tag on a format-2 ciphertext
    v1 = tag_ciphertext(dict(cts["c1"]), kg.get_private_keys("c1")["mac_keys"], threshold_codec._TAG_DOMAIN,
                        threshold_codec._tag_payload(encode_label("L"), cts["c1"]))
    with pytest.raises(FESchemeError, match="not authenticated"):
        srv.share_decrypt(dict(cts, c1=v1), creds, dk, SIDS)
    # binary ciphertexts (format 1 or 2) under JSON public parameters
    for data in (cts["c1"]["ct"], f1):
        bad = tag_ciphertext({"ct": data, "k": k, "label_hash": ctsj["c1"]["label_hash"]},
                             kgj.get_private_keys("c1")["mac_keys"], threshold_codec._TAG_DOMAIN,
                             threshold_codec._tag_payload(encode_label("L"), {"ct": data, "k": k}))
        with pytest.raises(FESchemeError, match="binary"):
            srvj.share_decrypt(dict(ctsj, c1=bad), creds, dkj, SIDS)
    # a client object asking for the other format
    with pytest.raises(FEValidationError):
        tm.ThresholdMCFERingLWE({"id": "c0", "ct_format": "json", "keys": {"pp": pp}})
    with pytest.raises(FEValidationError):
        tm.ThresholdMCFERingLWE({"id": "c0", "ct_format": "binary", "keys": {"pp": ppj}})
    with pytest.raises(FEValidationError):
        tm.ThresholdMCFERingLWE({"id": "c0", "keys": {"pp": dict(pp, ct_format="xml")}})
    with pytest.raises(FEValidationError):
        _setup(ct_format="xml")


def test_tags_bind_the_parameters_label_code_and_k():
    kg, pp, clients = _setup(t=3)
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    srv = _obj(pp, "s0")
    assert srv.share_decrypt(cts, creds, dk, SIDS)["k"] == 3
    other = pp_binding(dict(pp, stat_param=65), threshold_binding.tmcfe_ring_lwe_binding_fields(pp), threshold_binding.TMCFE_RING_LWE_BINDING_DEFAULTS)
    cases = [
        _retag_binary(kg, pp, "c1", "L", cts["c1"], pp_digest=other),     # other parameters' digest
        _retag_binary(kg, pp, "c1", "M", cts["c1"]),                      # another label
        dict(_retag_binary(kg, pp, "c1", "L", cts["c1"]), k=2),           # dict k not the header's
    ]
    coded = dict(cts["c1"], label_hash=0)
    cases.append(dict(_retag_binary(kg, pp, "c1", "L", coded), label_hash=cts["c1"]["label_hash"]))  # other code
    for bad in cases:
        with pytest.raises(FESchemeError, match="not authenticated"):
            srv.share_decrypt(dict(cts, c1=bad), creds, dk, SIDS)
    # ndarray chunks: the tag covers the chunk label (index) and the layer's code
    enc, creds2, dks, _ = _round(kg, pp, clients, np.random.default_rng(5))
    layer = enc["c1"][0]
    swapped = [dict(layer, chunks=[layer["chunks"][1], layer["chunks"][0]] + layer["chunks"][2:])] + enc["c1"][1:]
    with pytest.raises(FESchemeError, match="not authenticated"):
        _obj(pp, "s0").compute_lst_ndarray_ct(dict(enc, c1=swapped), creds2, dks["s0"], SIDS)


def test_key_shares_are_bound_to_one_format():
    """A key share of binary public parameters is refused by a server of the
    same parameters without ct_format, and the other way round."""
    kg, pp, clients = _setup(t=3)
    kgj, ppj, _ = _setup(t=3, ct_format="json")
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    dk, dkj = kg.get_decryption_keys("s0", credentials=creds), kgj.get_decryption_keys("s0", credentials=creds)
    assert dk["pp_binding"] != pp_binding(pp, threshold_binding.TMCFE_RING_LWE_BINDING_FIELDS, threshold_binding.TMCFE_RING_LWE_BINDING_DEFAULTS)
    with pytest.raises(FEKeyError, match="other public parameters"):
        _obj(pp, "s0").share_decrypt(cts, creds, dict(dk, pp_binding=pp_binding(
            pp, threshold_binding.TMCFE_RING_LWE_BINDING_FIELDS, threshold_binding.TMCFE_RING_LWE_BINDING_DEFAULTS)), SIDS)
    stripped = {k: v for k, v in pp.items() if k != "ct_format"}
    retagged = {nid: _retag_binary(kg, stripped, nid, "L", ct) for nid, ct in cts.items()}
    with pytest.raises(FESchemeError):  # JSON parameters: binary ciphertexts refused before the key share
        _obj(stripped, "s0").share_decrypt(retagged, creds, dk, SIDS)
    del dkj  # its JSON binding never matches binary parameters: covered by the first refusal


@pytest.mark.parametrize("name", BACKENDS)
def test_partials_must_be_of_the_parameters_format(name):
    kg, pp, clients = _setup(t=3)
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    with native_backend.backend(name):
        parts = {sid: _obj(pp, sid).share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), SIDS)
                 for sid in SIDS}
        comb = _obj(pp, "comb")
        assert comb.combine_decrypt(parts) == [3, 6, 9]
        q = int(pp["q"])
        wb = (q.bit_length() + 7) // 8
        for bad in ({"w": list(parts["s0"]["w"])}, {"w": parts["s0"]["w"][:-1]},
                    {"w": q.to_bytes(wb, "little") + parts["s0"]["w"][wb:]}):
            with pytest.raises(FESchemeError):
                comb.combine_decrypt(dict(parts, s0=dict(parts["s0"], **bad)))
        kgj, ppj, clientsj = _setup(t=3, ct_format="json")
        ctsj = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clientsj.items()}
        partsj = {sid: _obj(ppj, sid).share_decrypt(ctsj, creds, kgj.get_decryption_keys(sid, credentials=creds), SIDS)
                  for sid in SIDS}
        with pytest.raises(FESchemeError, match="format"):
            _obj(ppj, "comb").combine_decrypt(parts)
        with pytest.raises(FESchemeError, match="format"):
            comb.combine_decrypt(partsj)
