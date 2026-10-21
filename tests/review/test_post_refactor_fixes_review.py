"""Review of the post-refactor fix batch (fix/post-refactor-0.4.0, 4d876e2..4b50be4).

PF-1 (low, compatibility; new in this batch -- I3-a b425c81 and I3-c 66f1e07): server and client ids are
documented as "distinct ids" with no type, and on 4d876e2 integer ids work end to end -- the threshold
schemes deal, answer and combine with ``lst_sid = [1, 2, 3]``, and the ML adapter round-trips an
``EncryptedGradient`` of ``lst_nid = [0, 1, 2]`` through JSON. On 4b50be4 the key generators and the
wrapper still accept these ids, but every threshold combination then raises ``FEValidationError``
("sid and lst_sid_enrolled must be server ids (strings)") and ``EncryptedGradient.from_dict`` raises
``ValueError`` ("nid is a string"): a deployment with integer ids breaks only at the last step.

These tests accept either consistent fix: refuse non-string ids where they are configured (the key
generator or wrapper raises), or accept them in the partial-decryption and wire checks too.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyfe4ai.core.exceptions import FEError

NIDS = ["c0", "c1", "c2"]
INT_SIDS = [1, 2, 3]
W = {n: [1] for n in NIDS}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _lwe(kind):
    from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
    from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

    kg_cls, crypto = (ThresholdMIFELWEKeyGenerator, ThresholdMIFELWE) if kind == "mife" else (
        ThresholdMCFELWEKeyGenerator, ThresholdMCFELWE)
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": INT_SIDS, "lwe_n": 16,
           "bound_x": 6, "bound_y": 6}
    if kind == "mcfe":
        cfg.update(bound_u=2, label_modulus=8)
    try:
        kg = kg_cls(cfg)
        kg.setup()
    except FEError:
        return None  # refused where configured: consistent
    pp = kg.get_public_parameters()
    clients = {n: crypto({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in NIDS}
    cts = {n: (c.encrypt([1]) if kind == "mife" else c.encrypt([1], "L")) for n, c in clients.items()}
    creds = {"fusion_weight": W} if kind == "mife" else {"fusion_weight": W, "label": "L"}
    arg = W if kind == "mife" else creds
    enrolled = INT_SIDS[:2]
    dks = {s: kg.get_decryption_keys(s, credentials=creds) for s in enrolled}
    parts = {s: crypto({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(cts, arg, dks[s], enrolled)
             for s in enrolled}
    return crypto({"id": "agg", "keys": {"pp": pp}}).combine_decrypt(parts)


def _ring():
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    try:
        kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=INT_SIDS, eta=1, ring_n=32,
                                                      sec_param=64, bound_x=100, bound_y=3, stat_param=64))
        kg.setup()
    except FEError:
        return None
    pp = kg.get_public_parameters()
    cts = {n: tm.ThresholdMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
           .encrypt([[1, 1, 1]], "L") for n in NIDS}
    creds = {"fusion_weight": W, "label": "L"}
    enrolled = INT_SIDS[:2]
    parts = {}
    for s in enrolled:
        server = tm.ThresholdMCFERingLWE({"id": s, "precision": 0, "keys": {"pp": pp}})
        server.add_weight_key_share(kg.get_weight_key_share(s, credentials={"fusion_weight": W}))
        parts[s] = server.share_decrypt(cts, creds, kg.get_decryption_keys(s, credentials=creds), enrolled)
    out = tm.ThresholdMCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}}).combine_decrypt(parts)
    return out[0]


def _ddh(kind):
    if kind == "mife":
        from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDH as crypto
        from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDHKeyGenerator as kg_cls
    else:
        from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDH as crypto
        from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDHKeyGenerator as kg_cls
    try:
        kg = kg_cls({"sec_param": 128, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": INT_SIDS})
        kg.setup()
    except FEError:
        return None
    pp = kg.get_public_parameters()
    clients = {n: crypto({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in NIDS}
    cts = {n: (c.encrypt([1]) if kind == "mife" else c.encrypt([1], "L")) for n, c in clients.items()}
    creds = {"fusion_weight": W} if kind == "mife" else {"fusion_weight": W, "label": "L"}
    enrolled = INT_SIDS[:2]
    parts = {}
    for s in enrolled:
        dk = kg.get_decryption_keys(s, credentials=creds)
        server = crypto({"id": s, "keys": {"pp": pp, "dk": dk}})
        parts[s] = (server.share_decrypt(cts, W, dk, enrolled) if kind == "mife"
                    else server.share_decrypt(cts, W, dk, enrolled, "L"))
    return crypto({"id": "agg", "keys": {"pp": pp}}).combine_decrypt(parts)


@pytest.mark.parametrize("scheme", ["tmife-lwe", "tmcfe-lwe", "tmcfe-ring-lwe", "tmife-ddh", "tmcfe-ddh"])
def test_pf1_integer_server_ids_are_refused_at_setup_or_combine(scheme):
    run = {"tmife-lwe": lambda: _lwe("mife"), "tmcfe-lwe": lambda: _lwe("mcfe"), "tmcfe-ring-lwe": _ring,
           "tmife-ddh": lambda: _ddh("mife"), "tmcfe-ddh": lambda: _ddh("mcfe")}[scheme]
    result = run()  # None: the key generator refused the ids
    assert result is None or result == 3, result


def test_pf1_integer_client_ids_round_trip_or_are_refused():
    from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = [0, 1, 2]
    try:
        wrapper = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": nids, "ring_n": 16,
                                                       "sec_param": 64, "bound_x": 50, "bound_y": 2}, precision=1)
        grads = {n: np.round(np.linspace(-1, 1, 6).reshape(2, 3), 1) for n in nids}
        sent = {n: encrypt_gradient(grads[n], wrapper, nid=n, label="a") for n in nids}
    except (FEError, ValueError, TypeError):
        return  # refused where configured: consistent
    wire = {n: json.loads(json.dumps(e.to_dict())) for n, e in sent.items()}
    received = {n: EncryptedGradient.from_dict(d) for n, d in wire.items()}
    out = aggregate_gradients(received, wrapper, {n: 1 for n in nids}, label="a")
    np.testing.assert_allclose(out, sum(grads.values()), atol=1e-9)
