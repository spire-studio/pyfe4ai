"""Numpy integer server and client ids after setup.

A key generator takes ``lst_sid`` / ``lst_nid`` as lists of ``str`` or of
``int`` (:func:`~pyfe4ai.core.ids.check_id_list`), but ids that come later --
the enrolled servers (``list(rng.choice(lst_sid, t, replace=False))``), a
server object's own ``id``, the server named at issuance, the servers of the
ML adapter, the client of a gradient -- are often numpy integers. They equal
and hash like the ints, so every server found its key share and answered, and
only the combiner refused the partial decryptions (``FEValidationError``);
v0.3.0 combined them. A numpy integer (``numpy.integer``; ``numpy.bool_`` is
none) is now read as the ``int`` it equals wherever an id enters after setup
(:func:`~pyfe4ai.core.ids.as_id`), the partial decryptions and key shares
carry plain ints, and other values (floats, ``bool``, ``None``) are refused
by the server rather than by the combiner.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.ids import as_id, is_id
from pyfe4ai.schemes.threshold.roles import SERVER_IDS, validate_partial_decryptions

needs_bls = pytest.mark.skipif(not native_backend.available(), reason="BLS12-381 needs the native extension")
NIDS = ["c0", "c1", "c2"]
SIDS = [1, 2, 3]
W = {n: [1] for n in NIDS}
SCHEMES = ["tmife-lwe", "tmcfe-lwe", "tmcfe-ring-lwe", "tmife-ddh", "tmcfe-ddh",
           pytest.param("tmcfe-fh-pairing", marks=needs_bls)]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def test_numpy_integers_are_read_as_ints():
    for value in (np.int64(2), np.int32(2), np.uint8(2), np.intc(2)):
        assert type(as_id(value)) is int and as_id(value) == 2 and is_id(as_id(value))
    for value in (np.bool_(True), np.float64(2.0), 2.0, True, None, b"2", (2,), "2", 2):
        assert as_id(value) is value
    assert not any(is_id(as_id(v)) for v in (np.bool_(True), np.float64(2.0), None))


def _round(kind, issue=lambda s: s, enrolled=lambda e: e, server=lambda s: s, keys=lambda s: s):
    """One round of *kind* with lst_sid = [1, 2, 3], the first two servers enrolled: the result, the partial
    decryptions and the key shares."""
    E = SIDS[:2]
    if kind == "tmcfe-ring-lwe":
        from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

        kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=SIDS, eta=1, ring_n=16,
                                                      sec_param=64, bound_x=10, bound_y=3))
        kg.setup()
        pp = kg.get_public_parameters()
        cts = {n: tm.ThresholdMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
               .encrypt([[1]], "L") for n in NIDS}
        creds = {"fusion_weight": W, "label": "L"}
        parts, dks = {}, {}
        for s in E:
            srv = tm.ThresholdMCFERingLWE({"id": server(s), "precision": 0, "keys": {"pp": pp}})
            srv.add_weight_key_share(kg.get_weight_key_share(issue(s), credentials={"fusion_weight": W}))
            dks[s] = kg.get_decryption_keys(issue(s), credentials=creds)
            parts[keys(s)] = srv.share_decrypt(cts, creds, dks[s], enrolled(E))
        out = tm.ThresholdMCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}}).combine_decrypt(parts)
        return out[0], parts, dks
    mife = kind.startswith("tmife")
    if kind in ("tmife-lwe", "tmcfe-lwe"):
        from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
        from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

        K, C = (ThresholdMIFELWEKeyGenerator, ThresholdMIFELWE) if mife else (ThresholdMCFELWEKeyGenerator,
                                                                             ThresholdMCFELWE)
        cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS, "lwe_n": 16,
               "bound_x": 6, "bound_y": 6}
        if not mife:
            cfg.update(bound_u=2, label_modulus=8)
    elif kind in ("tmife-ddh", "tmcfe-ddh"):
        if mife:
            from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDH as C
            from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDHKeyGenerator as K
        else:
            from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDH as C
            from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDHKeyGenerator as K
        cfg = {"sec_param": 128, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS}
    else:
        from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEPairing as C
        from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import (
            ThresholdMCFEFHMultiIPEPairingKeyGenerator as K)
        cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS, "bound_x": 4,
               "bound_y": 4}
    kg = K(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: C({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in NIDS}
    cts = {n: c.encrypt([1]) if mife else c.encrypt([1], "L") for n, c in clients.items()}
    creds = {"fusion_weight": W} if mife else {"fusion_weight": W, "label": "L"}
    dks = {s: kg.get_decryption_keys(issue(s), credentials=creds) for s in E}
    parts = {}
    for s in E:
        srv = C({"id": server(s), "keys": {"pp": pp, "dk": dks[s]}})
        if kind in ("tmife-lwe", "tmcfe-lwe"):
            parts[keys(s)] = srv.share_decrypt(cts, W if mife else creds, dks[s], enrolled(E))
        elif kind == "tmife-ddh":
            parts[keys(s)] = srv.share_decrypt(cts, W, dks[s], enrolled(E))
        elif kind == "tmcfe-ddh":
            parts[keys(s)] = srv.share_decrypt(cts, W, dks[s], enrolled(E), "L")
        else:
            parts[keys(s)] = srv.share_decrypt(cts, creds, dks[s], enrolled(E), "L")
    return C({"id": "agg", "keys": {"pp": pp}}).combine_decrypt(parts), parts, dks


NUMPY = {
    "enrolled list": {"enrolled": lambda e: [np.int64(s) for s in e]},
    "enrolled ndarray": {"enrolled": np.array},
    "enrolled uint8": {"enrolled": lambda e: [np.uint8(s) for s in e]},
    "server object ids": {"server": np.int64},
    "issuance": {"issue": np.int64},
    "all": {"issue": np.int64, "enrolled": lambda e: [np.int64(s) for s in e], "server": np.int64, "keys": np.int64},
}


@pytest.mark.parametrize("where", sorted(NUMPY))
@pytest.mark.parametrize("kind", SCHEMES)
def test_numpy_integer_ids_after_setup_combine_and_travel_as_ints(kind, where):
    result, parts, dks = _round(kind, **NUMPY[where])
    assert result == 3
    for part in parts.values():  # what a server writes: plain ints
        assert type(part["sid"]) is int and all(type(s) is int for s in part["lst_sid_enrolled"])
    for dk in dks.values():  # what the key generator issues: plain ints
        assert type(dk["sid"]) is int
    if kind not in ("tmcfe-ring-lwe", "tmcfe-fh-pairing"):  # the others hold bytes / group elements
        assert all(json.loads(json.dumps(dk))["sid"] in SIDS for dk in dks.values())


OTHER = {
    "enrolled floats": {"enrolled": lambda e: [float(s) for s in e]},
    "enrolled numpy floats": {"enrolled": lambda e: [np.float64(s) for s in e]},
    "enrolled numpy bools": {"enrolled": lambda e: [np.bool_(True), np.int64(2)]},
    "enrolled None": {"enrolled": lambda e: [e[0], None]},
    "enrolled not a list": {"enrolled": lambda e: 5},
    "server object id a float": {"server": float},
}


@pytest.mark.parametrize("where", sorted(OTHER))
@pytest.mark.parametrize("kind", SCHEMES)
def test_other_values_are_refused_by_the_server(kind, where):
    # refused by share_decrypt, before the combiner (a float equals and hashes like the int it stands for)
    with pytest.raises(FEValidationError) as info:
        _round(kind, **OTHER[where])
    assert str(info.value) == SERVER_IDS


KEY_GENERATORS = {
    "tmife-lwe": ("mife/lwe_threshold", {"lwe_n": 16, "bound_x": 6, "bound_y": 6, "sec_param": 64}),
    "tmcfe-lwe": ("mcfe/lwe_threshold", {"lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2, "label_modulus": 8,
                                         "sec_param": 64}),
    "tmcfe-ring-lwe": ("mcfe/ring_lwe_threshold", {"ring_n": 16, "bound_x": 10, "bound_y": 3, "sec_param": 64}),
    "tmife-ddh": ("mife/ddh_threshold", {"sec_param": 128}),
    "tmcfe-ddh": ("mcfe/ddh_threshold", {"sec_param": 128}),
    "tmcfe-fh-pairing": ("mcfe/fh_multi_ipe_pairing_threshold", {"bound_x": 4, "bound_y": 4, "sec_param": 64}),
}


@pytest.mark.parametrize("value", [1.0, np.float64(1), np.bool_(True), True], ids=["float", "numpy float", "numpy bool",
                                                                                 "bool"])
@pytest.mark.parametrize("kind", SCHEMES)
def test_issuing_for_a_value_that_is_not_an_id_gives_no_key(kind, value):
    # as for an unknown server: None and an FEWarning (each of these equals server 1)
    import pyfe4ai
    from pyfe4ai.core.exceptions import FEWarning

    scheme, extra = KEY_GENERATORS[kind]
    kg, _ = pyfe4ai.create(scheme, dict(extra, eta=1, n=3, s=3, t=2, lst_nid=NIDS, lst_sid=SIDS))
    kg.setup()
    creds = {"fusion_weight": W} if kind.startswith("tmife") else {"fusion_weight": W, "label": "L"}
    with pytest.warns(FEWarning, match="invalid sid"):
        assert kg.get_decryption_keys(value, credentials=creds) is None
    assert kg.get_decryption_keys(np.int64(1), credentials=creds)["sid"] == 1
    if kind == "tmcfe-ring-lwe":
        with pytest.warns(FEWarning, match="invalid sid"):
            assert kg.get_weight_key_share(value, credentials={"fusion_weight": W}) is None


def test_partials_written_with_numpy_ids_combine():
    # partial decryptions of objects made before (or kept in memory with numpy ids) are read with as_id
    parts = {np.int64(s): {"w": "1", "sid": np.int64(s), "lst_sid_enrolled": [np.int64(1), np.int64(2)]}
             for s in (1, 2)}
    validate_partial_decryptions(parts, 2)
    with pytest.raises(FEValidationError):
        validate_partial_decryptions({s: dict(p, sid=float(p["sid"])) for s, p in parts.items()}, 2)
    with pytest.raises(FEValidationError):
        validate_partial_decryptions({s: dict(p, lst_sid_enrolled=[np.bool_(True), 2]) for s, p in parts.items()}, 2)


# -- the ML adapter


def _served(servers=lambda e: e, server=lambda s: s):
    from pyfe4ai.ml import (FESchemeWrapper, ThresholdPartial, combine_partial_decryptions, encrypt_gradient,
                            partial_decrypt_gradients, prepare_aggregation)

    w = FESchemeWrapper("mcfe", "ring_lwe_threshold", {"eta": 1, "n": 3, "lst_nid": NIDS, "s": 3, "t": 2,
                                                     "lst_sid": SIDS, "ring_n": 16, "sec_param": 64, "bound_x": 50,
                                                     "bound_y": 2}, precision=1)
    enc = {n: encrypt_gradient(np.array([0.5, -0.5]), w, nid=n, label="a") for n in NIDS}
    prepared = prepare_aggregation(w, {n: 1 for n in NIDS}, shape=(2,), label="a", servers=servers(SIDS[:2]))
    partials = {server(s): partial_decrypt_gradients(enc, w, prepared, server=server(s)) for s in SIDS[:2]}
    sent = {s: ThresholdPartial.from_dict(json.loads(json.dumps(p.to_dict()))) for s, p in partials.items()}
    return combine_partial_decryptions(sent, w, prepared), prepared, partials


def test_ml_adapter_numpy_integer_servers():
    out, prepared, partials = _served(servers=lambda e: [np.int64(s) for s in e], server=np.int64)
    np.testing.assert_allclose(out, [1.5, -1.5])
    assert all(type(s) is int for s in prepared.servers) and all(type(p.sid) is int for p in partials.values())


def test_ml_adapter_other_servers_are_refused_before_answering():
    from pyfe4ai.ml import FESchemeWrapper, encrypt_gradient, partial_decrypt_gradients, prepare_aggregation

    w = FESchemeWrapper("mcfe", "ring_lwe_threshold", {"eta": 1, "n": 3, "lst_nid": NIDS, "s": 3, "t": 2,
                                                     "lst_sid": SIDS, "ring_n": 16, "sec_param": 64, "bound_x": 50,
                                                     "bound_y": 2}, precision=1)
    enc = {n: encrypt_gradient(np.array([0.5, -0.5]), w, nid=n, label="a") for n in NIDS}
    for servers in ([1.0, 2.0], [np.float64(1), 2], [np.bool_(True), 2]):  # each equals [1, 2]
        with pytest.raises(ValueError, match="servers must be"):
            prepare_aggregation(w, {n: 1 for n in NIDS}, shape=(2,), label="a", servers=servers)
    prepared = prepare_aggregation(w, {n: 1 for n in NIDS}, shape=(2,), label="a", servers=[1, 2])
    for server in (1.0, np.float64(1), True):
        with pytest.raises(ValueError, match="is not enrolled"):
            partial_decrypt_gradients(enc, w, prepared, server=server)


@pytest.mark.parametrize("variant", ["lwe", "ring_lwe", "ddh", "paillier"])
def test_ml_adapter_numpy_integer_client_ids(variant):
    from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = [0, 1, 2]
    w = FESchemeWrapper("mcfe", variant, {"eta": 1, "n": 3, "lst_nid": nids, "ring_n": 16, "lwe_n": 8, "bound_x": 50,
                                          "bound_y": 2, "sec_param": 128 if variant in ("ddh", "paillier") else 64},
                        precision=1)
    enc = {n: encrypt_gradient(np.array([0.5, -0.5]), w, nid=np.int64(n), label="a") for n in nids}
    assert all(type(e.nid) is int for e in enc.values())
    sent = {n: EncryptedGradient.from_dict(json.loads(json.dumps(e.to_dict()))) for n, e in enc.items()}
    np.testing.assert_allclose(aggregate_gradients(sent, w, {n: 1 for n in nids}, label="a"), [1.5, -1.5])
    in_memory = dict(enc[0].to_dict(), nid=np.int64(0))  # a dict made in memory
    assert type(EncryptedGradient.from_dict(in_memory).nid) is int
    with pytest.raises(ValueError):
        EncryptedGradient.from_dict(dict(in_memory, nid=np.float64(0)))
