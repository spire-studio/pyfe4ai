"""Invariants of the threshold MCFE's binary format, RNS ring and split key
shares (October 2026 code review; these tests pass).

Kept with the review tests so that later changes keep them:

* format 2 is refused behind a valid tag for every non-canonical or malformed
  encoding (a corrupted client holds its MAC keys), on both backends, prime
  and RNS: values ``>= q`` in ``ct0`` and ``ct1``, trailing or missing bytes,
  header fields, ``k`` of the header and of the dict;
* one key part, one noise: answers for a shorter ``k`` are prefixes, another
  ``ct0`` leaves ``w`` unchanged, the backends agree;
* tags: v1 (JSON) tags, tags of another server or client are refused;
* the RNS ring on edge values (0, q - 1, the primes and their neighbours,
  limb boundaries) against the Python code on the composite q -- the
  reference computed on the Python backend (tests/test_threshold_rns.py
  computes it natively under the default backend);
* RNS with the sharings and floodings its tests do not combine
  (replicated-seeded, KL, Gaussian replicated, split shares).
"""

from __future__ import annotations

import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.threshold.kl import PartialDecryptionBudget
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.threshold import binding as threshold_binding
from pyfe4ai.schemes.threshold import codec as threshold_codec
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.threshold.binding import pp_binding, tag_ciphertext, tag_ciphertext_message
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.core.exceptions import FESchemeError

NATIVE = native_backend.available()
BACKENDS = ["python"] + (["native"] if NATIVE else [])
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _setup(t=3, ring_modulus="prime", eta=2, ring_n=32, **extra):
    cfg = dict(n=3, lst_nid=NIDS, s=3, t=t, lst_sid=SIDS, eta=eta, ring_n=ring_n, sec_param=64, bound_x=100,
               bound_y=3, stat_param=64, ct_format="binary", ring_modulus=ring_modulus, **extra)
    cfg.setdefault("key_shares", "full")  # the default when these tests were written (split: key_part="label")
    kg = tm.ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    return kg, pp, clients


def _srv(pp, sid, **extra):
    return tm.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}, **extra})


def _tag_raw(kg, pp, nid, ct, k):
    """Tag the bytes of ct["ct"] whatever they are (a client holding its MAC keys)."""
    import hashlib

    wb = (int(pp["q"]).bit_length() + 7) // 8
    data = bytes(ct["ct"])
    split = 14 + pp["eta"] * k * wb
    d0 = hashlib.sha256(threshold_codec._CT0_DOMAIN + data[:split]).digest()
    d1 = hashlib.sha256(threshold_codec._CT1_DOMAIN + data[split:]).digest()
    digest = bytes.fromhex(pp_binding(pp, threshold_binding.tmcfe_ring_lwe_binding_fields(pp), threshold_binding.TMCFE_RING_LWE_BINDING_DEFAULTS))
    msg = threshold_codec.binary_tag_message(digest, encode_label("L"), ct, (k, d0, d1))
    return tag_ciphertext_message(dict(ct), kg.get_private_keys(nid)["mac_keys"], msg)


def _mutations(pp, ct):
    q = int(pp["q"])
    wb = (q.bit_length() + 7) // 8
    eta, n, k = pp["eta"], pp["ring_n"], ct["k"]
    data = bytes(ct["ct"])
    out = []
    for p in (0, eta * k - 1, eta * k, eta * k + n - 1):  # first / last of ct0, first / last of ct1
        off = 14 + p * wb
        v = int.from_bytes(data[off:off + wb], "little")
        for nv in (v + q, q):
            if nv < 1 << (8 * wb):
                out.append((data[:off] + nv.to_bytes(wb, "little") + data[off + wb:], k))
    out += [(data + b"\x00", k), (data[:-1], k), (data[:-wb], k)]
    for i, b in ((4, 1), (4, 3), (5, data[5] + 1), (7, 1), (8, data[8] + 1), (9, 1)):
        out.append((data[:i] + bytes([b]) + data[i + 1:], k))
    for k2 in (0, k - 1, k + 1, n + 1):
        out.append((data[:10] + k2.to_bytes(4, "little") + data[14:], k2))
    out.append((data, k + 1))
    return out


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("modulus", ["prime", "rns"])
def test_format_2_refused_behind_a_valid_tag(name, modulus):
    kg, pp, clients = _setup(ring_modulus=modulus)
    cts = {nid: cl.encrypt([[1, 2, 3, 4], [5, 6, 7, 8]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1, 2] for nid in NIDS}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    with native_backend.backend(name):
        _srv(pp, "s0").share_decrypt(cts, creds, dk, SIDS)
        for data, k in _mutations(pp, cts["c1"]):
            bad = _tag_raw(kg, pp, "c1", dict(cts["c1"], ct=data, k=k), k)
            with pytest.raises(FESchemeError):
                _srv(pp, "s0").share_decrypt(dict(cts, c1=bad), creds, dk, SIDS)


def _recut(pp, ct, k2):
    wb = (int(pp["q"]).bit_length() + 7) // 8
    eta, k = pp["eta"], ct["k"]
    body = bytes(ct["ct"])[14:]
    rows = [body[r * k * wb:(r * k + k2) * wb] for r in range(eta)]
    return dict(ct, ct=bytes(ct["ct"])[:10] + k2.to_bytes(4, "little") + b"".join(rows) + body[eta * k * wb:], k=k2)


@pytest.mark.parametrize("modulus", ["prime", "rns"])
@pytest.mark.parametrize("flooding,t", [("uniform", 2), ("uniform", 3), ("gaussian", 3)])
def test_one_key_part_one_noise(modulus, flooding, t):
    kg, pp, clients = _setup(t=t, ring_modulus=modulus, flooding=flooding)
    cts = {nid: cl.encrypt([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1, 2] for nid in NIDS}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    wb = (int(pp["q"]).bit_length() + 7) // 8
    answers = []
    for name in BACKENDS:
        with native_backend.backend(name):
            full = _srv(pp, "s0").share_decrypt(cts, creds, dk, SIDS[:t])
            answers.append(full)
            for k2 in (1, 4):
                short = {nid: _tag_raw(kg, pp, nid, _recut(pp, ct, k2), k2) for nid, ct in cts.items()}
                part = _srv(pp, "s0").share_decrypt(short, creds, dk, SIDS[:t])
                assert part["w"] == full["w"][:k2 * wb] and part["c_y"] == full["c_y"][:k2 * wb]
            data = bytearray(cts["c1"]["ct"])
            data[14:14 + wb] = (5).to_bytes(wb, "little")  # another ct0 value
            other = dict(cts, c1=_tag_raw(kg, pp, "c1", dict(cts["c1"], ct=bytes(data)), cts["c1"]["k"]))
            part = _srv(pp, "s0").share_decrypt(other, creds, dk, SIDS[:t])
            assert part["w"] == full["w"] and part["c_y"] != full["c_y"]
            # tags of another server, swapped clients, a v1 (JSON) tag
            dk1 = kg.get_decryption_keys("s1", credentials=creds)
            swapped = {nid: dict(ct, tags=dict(ct["tags"], s1=ct["tags"]["s0"])) for nid, ct in cts.items()}
            with pytest.raises(FESchemeError):
                _srv(pp, "s1").share_decrypt(swapped, creds, dk1, SIDS[:t])
            with pytest.raises(FESchemeError):
                _srv(pp, "s0").share_decrypt(dict(cts, c0=cts["c1"], c1=cts["c0"]), creds, dk, SIDS[:t])
            v1 = tag_ciphertext(dict(cts["c1"]), kg.get_private_keys("c1")["mac_keys"], threshold_codec._TAG_DOMAIN,
                                threshold_codec._tag_payload(encode_label("L"), cts["c1"]))
            with pytest.raises(FESchemeError):
                _srv(pp, "s0").share_decrypt(dict(cts, c1=v1), creds, dk, SIDS[:t])
    assert all(a == answers[0] for a in answers)


def _edges(q, fs):
    q = int(q)
    vals = {0, 1, 2, q - 1, q - 2, q // 2}
    prod = 1
    for p in map(int, fs):
        prod *= p
        vals |= {p - 1, p, p + 1, q - p, prod % q, (prod - 1) % q}
    for k in range(1, 4):
        vals |= {(1 << (64 * k)) - 1, 1 << (64 * k)}
    return sorted(v % q for v in vals)


@pytest.mark.skipif(not NATIVE, reason="native extension not built")
@pytest.mark.parametrize("bits,k,n", [(133, 3, 16), (180, 3, 64), (215, 4, 16), (215, 4, 256), (247, 4, 16)])
def test_rns_ring_on_edge_values_against_python(bits, k, n):
    q, fs = rlu.rns_ntt_modulus((gp.mpz(1) << (bits - 1)) + 7, n, 602, k)
    rlu.register_rns_modulus(q, fs)
    with native_backend.backend("native"):
        ctx = native_impl.ring_context(q, n)
    assert ctx is not None and not ctx.legacy_rns
    root = rlu._get_ntt_root(q, n)
    edges = _edges(q, fs)
    rng = random.Random(bits * n)
    N = native_backend.ext()
    for trial in range(4):
        a = [gp.mpz(edges[(trial * 7 + i) % len(edges)]) for i in range(n)]
        b = [gp.mpz(edges[(trial * 5 + 3 * i) % len(edges)]) if i % 3 else gp.mpz(rng.randrange(int(q)))
             for i in range(n)]
        with native_backend.backend("python"):  # the reference: the Python code on the composite q
            fa, ib, ab = rlu.ntt_forward(list(a), q, root), rlu.ntt_inverse(list(b), q, root), rlu.poly_mul(a, b, q)
        assert ctx.unpack(N.ntt(ctx.cap, ctx.pack(a))) == [int(v) for v in fa]
        assert ctx.unpack(N.ntt(ctx.cap, ctx.pack(b), True)) == [int(v) for v in ib]
        assert ctx.unpack(N.poly_mul(ctx.cap, ctx.pack(a), ctx.pack(b))) == [int(v) for v in ab]
        assert ctx.unpack(N.ntt(ctx.cap, N.ntt(ctx.cap, ctx.pack(b), True))) == [int(v) for v in b]


@pytest.mark.parametrize("ct_format", ["json", "binary"])
@pytest.mark.parametrize("case", [
    dict(t=2, sharing="replicated-seeded"),
    dict(t=2, sharing="replicated", flooding="gaussian"),
    dict(t=3, sharing="shamir", flooding="gaussian", flooding_accounting="kl", max_partial_decryptions=1 << 20),
    dict(t=3, sharing="shamir", split=True),
    dict(t=2, sharing="replicated", split=True),
], ids=["seeded", "gaussian-replicated", "kl", "split-shamir", "split-replicated"])
def test_rns_with_other_sharings_and_floodings(case, ct_format):
    case = dict(case)
    split = case.pop("split", False)
    t = case["t"]
    cfg = dict(n=3, lst_nid=NIDS, s=3, lst_sid=SIDS, eta=1, ring_n=32, sec_param=64, bound_x=100, bound_y=3,
               stat_param=150, ct_format=ct_format, ring_modulus="rns", **case)
    kg = tm.ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    assert len(pp["q_factors"]) >= 3
    w = {nid: [y] for nid, y in zip(NIDS, (1, -2, 3))}
    g = {nid: [np.arange(70) % 11 - 5 + i] for i, nid in enumerate(NIDS)}
    cl = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
          for nid in NIDS}
    enc = {nid: cl[nid].encrypt_lst_ndarray(g[nid], label="r") for nid in NIDS}
    creds = {"fusion_weight": w, "label": "r", "n_chunks": cl["c0"].key_chunks(enc["c0"])}
    creds["key_part"] = "label" if split else "full"
    dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in SIDS}
    outs = []
    for name in BACKENDS:
        with native_backend.backend(name):
            parts = {}
            for sid in SIDS[:t]:
                extra = {"partial_budget": PartialDecryptionBudget(1 << 20)} if "flooding_accounting" in case else {}
                srv = _srv(pp, sid, **extra)
                if split:
                    srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": w}))
                parts[sid] = srv.compute_lst_ndarray_ct(enc, creds, dks[sid], SIDS[:t])
            res = _srv(pp, "agg").decrypt_lst_ndarray_ct(parts)[0]
            assert np.array_equal(res, sum(w[nid][0] * g[nid][0] for nid in NIDS))
            outs.append(parts)
    assert all(o == outs[0] for o in outs)
