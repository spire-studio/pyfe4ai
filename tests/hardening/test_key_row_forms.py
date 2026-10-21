"""Ring-LWE keys whose rows are tuples or numpy arrays.

The checks of the rows of a Ring-LWE encryption key (``pk``, ``u``; ``u``
and ``s`` of a decentralized client) and of a decentralized key share
(``sk_y``, ``z``, ``z_chunks``) took lists and tuples only, but a key whose
rows are numpy arrays -- a list of 1-D arrays, or one 2-D array -- encrypted
and combined correctly before them, in both backends. Any 1-D sequence of
the right length is a row: a list, a tuple or a 1-D numpy array, in a list,
a tuple or a numpy array; other values and other lengths are still refused
with :class:`~pyfe4ai.core.exceptions.FEKeyError`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEKeyError
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe
from pyfe4ai.schemes.mcfe import ring_lwe_decentralized as dmcfe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe
from pyfe4ai.schemes.mife import ring_lwe as mife
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWE, SIFEFullySecRingLWEKeyGenerator
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE, SIFERingLWEKeyGenerator

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS = ["c0", "c1", "c2"]
ETA = 2

#: name -> the rows of a key field in another form
FORMS = {
    "tuple of lists": lambda rows: tuple(rows),
    "list of tuples": lambda rows: [tuple(r) for r in rows],
    "list of numpy str arrays": lambda rows: [np.array(r) for r in rows],
    "list of numpy object arrays": lambda rows: [np.array([int(v) for v in r], dtype=object) for r in rows],
    "numpy 2-D object array": lambda rows: np.array([[int(v) for v in r] for r in rows], dtype=object),
    "numpy 2-D str array": lambda rows: np.array([[str(v) for v in r] for r in rows]),
}
#: name -> (the rows of a key field made malformed in a numpy form, the message's start)
MALFORMED = {
    "numpy 2-D array one row short": (lambda rows: np.array([[int(v) for v in r] for r in rows[:-1]], dtype=object),
                                      "the encryption key has fewer than eta rows"),
    "numpy row one value short": (lambda rows: [np.array(r[:-1]) for r in rows], "the encryption key has a row"),
    "numpy 2-D row": (lambda rows: [np.array([r]) for r in rows], "the encryption key has a row"),
    "numpy scalar row": (lambda rows: [np.array(r[0]) for r in rows], "the encryption key has a row"),
}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


SCHEMES = {
    "mcfe/ring_lwe": (mcfe.MCFERingLWEKeyGenerator, mcfe.MCFERingLWE, {}),
    "mcfe/ring_lwe_threshold": (tmcfe.ThresholdMCFERingLWEKeyGenerator, tmcfe.ThresholdMCFERingLWE,
                                {"s": 2, "t": 2, "lst_sid": ["s0", "s1"]}),
    "mife/ring_lwe": (mife.MIFERingLWEKeyGenerator, mife.MIFERingLWE, {}),
}


def _mixed_round(scheme, field, change):
    """The sum of every client's ``[[1, 2, 3]] * ETA`` under weights 1, client c0's key changed by *change*."""
    kg_cls, crypto, extra = SCHEMES[scheme]
    kg = kg_cls(dict({"sec_param": 64, "eta": ETA, "n": 3, "lst_nid": NIDS, "ring_n": 16, "bound_x": 50,
                      "bound_y": 2}, **extra))
    kg.setup()
    pp = kg.get_public_parameters()
    sks = {n: kg.get_private_keys(n) for n in NIDS}
    sks["c0"] = dict(sks["c0"], **{field: change(sks["c0"][field])})
    weights = {n: [1] * ETA for n in NIDS}
    mife_scheme = scheme.startswith("mife")
    clients = {n: crypto({"id": n, "precision": 0, "keys": {"pp": pp, "sk": sks[n]}}) for n in NIDS}
    cts = {n: c.encrypt([[1, 2, 3]] * ETA) if mife_scheme else c.encrypt([[1, 2, 3]] * ETA, "L")
           for n, c in clients.items()}
    agg = crypto({"id": "agg", "precision": 0, "keys": {"pp": pp}})
    if mife_scheme:
        return agg.decrypt(cts, kg.get_decryption_keys("x", credentials={"fusion_weight": weights}), weights)
    creds = {"fusion_weight": weights, "label": "L"}
    if scheme == "mcfe/ring_lwe":
        return agg.decrypt(cts, kg.get_decryption_keys("x", credentials=creds), weights, "L")
    parts = {}
    for sid in ("s0", "s1"):
        server = crypto({"id": sid, "precision": 0, "keys": {"pp": pp}})
        server.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": weights}))
        parts[sid] = server.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), ["s0", "s1"])
    return agg.combine_decrypt(parts)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("field", ["pk", "u"])
@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_mife_and_mcfe_keys_with_rows_in_other_forms(scheme, field, form, backend):
    with native_backend.backend(backend):
        assert _mixed_round(scheme, field, FORMS[form]) == [3 * ETA, 6 * ETA, 9 * ETA]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(MALFORMED))
@pytest.mark.parametrize("field", ["pk", "u"])
def test_malformed_rows_in_numpy_forms_are_refused(field, case, backend):
    change, message = MALFORMED[case]
    with native_backend.backend(backend), pytest.raises(FEKeyError, match=message):
        _mixed_round("mcfe/ring_lwe", field, change)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("form", sorted(FORMS))
def test_sife_keys_with_rows_in_other_forms(form, backend):
    with native_backend.backend(backend):
        kg = SIFERingLWEKeyGenerator({"sec_param": 64, "eta": ETA, "ring_n": 16, "bound_x": 50, "bound_y": 2})
        kg.setup()
        pp, sk = kg.get_public_parameters(), kg.get_private_keys()
        dk = kg.get_decryption_keys("x", credentials={"fusion_weight": [1] * ETA})
        ct = SIFERingLWE({"precision": 0, "keys": {"pp": pp, "sk": dict(sk, pk=FORMS[form](sk["pk"]))}}).encrypt(
            [[1, 2, 3]] * ETA)
        assert SIFERingLWE({"precision": 0, "keys": {"pp": pp}}).decrypt(ct, dk, [1] * ETA) == [ETA, 2 * ETA, 3 * ETA]
        kg = SIFEFullySecRingLWEKeyGenerator({"sec_param": 64, "eta": ETA, "ring_n": 16, "bound_x": 5, "bound_y": 2})
        kg.setup()
        pp, sk = kg.get_public_parameters(), kg.get_private_keys()
        dk = kg.get_decryption_keys("x", credentials={"fusion_weight": [1] * ETA})
        ct = SIFEFullySecRingLWE({"precision": 0, "keys": {"pp": pp, "sk": dict(sk, pk=FORMS[form](sk["pk"]))}}
                                 ).encrypt([1] * ETA)
        assert SIFEFullySecRingLWE({"precision": 0, "keys": {"pp": pp}}).decrypt(ct, dk, [1] * ETA) == ETA


def _dmcfe():
    kg = dmcfe.DecentralizedMCFERingLWEKeyGenerator({"sec_param": 64, "eta": ETA, "n": 3, "lst_nid": NIDS,
                                                     "ring_n": 16, "bound_x": 50, "bound_y": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    sks = {n: kg.get_private_keys(n) for n in NIDS}
    clients = {n: dmcfe.DecentralizedMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": sks[n]}})
               for n in NIDS}
    cts = {n: clients[n].encrypt([[1, 2, 3]] * ETA, "L") for n in NIDS}
    return pp, sks, clients, {n: [1] * ETA for n in NIDS}, cts


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("field", ["u", "s"])
def test_decentralized_keys_with_rows_in_other_forms(field, form, backend):
    with native_backend.backend(backend):
        pp, sks, clients, weights, cts = _dmcfe()
        c0 = dmcfe.DecentralizedMCFERingLWE({"id": "c0", "precision": 0, "keys": {
            "pp": pp, "sk": dict(sks["c0"], **{field: FORMS[form](sks["c0"][field])})}})
        shares = {n: (c0 if n == "c0" else clients[n]).derive_function_decryption_key_share(weights, "L") for n in NIDS}
        key = clients["c1"].combine_function_decryption_key_share(shares)
        assert clients["c1"].decrypt(cts, key, weights, "L") == [3 * ETA, 6 * ETA, 9 * ETA]


POLY_FORMS = {
    "tuple": tuple,
    "numpy str array": np.array,
    "numpy object array": lambda poly: np.array([int(v) for v in poly], dtype=object),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("form", sorted(POLY_FORMS))
@pytest.mark.parametrize("field", ["sk_y", "z", "z_chunks"])
def test_decentralized_key_shares_with_values_in_other_forms(field, form, backend):
    with native_backend.backend(backend):
        pp, sks, clients, weights, cts = _dmcfe()
        shares = {n: clients[n].derive_function_decryption_key_share(weights, "L", n_chunks=1) for n in NIDS}
        change = POLY_FORMS[form]
        share = dict(shares["c0"])
        share[field] = [change(p) for p in share[field]] if field == "z_chunks" else change(share[field])
        key = clients["c1"].combine_function_decryption_key_share(dict(shares, c0=share))
        assert clients["c1"].decrypt(cts, key, weights, "L") == [3 * ETA, 6 * ETA, 9 * ETA]
        bad = dict(share, **{field: [np.array([p]) for p in share[field]] if field == "z_chunks"
                             else np.array([share[field]])})  # a 2-D array is no polynomial
        with pytest.raises(FEKeyError, match="malformed key share"):
            clients["c1"].combine_function_decryption_key_share(dict(shares, c0=bad))
