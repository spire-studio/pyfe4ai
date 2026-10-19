"""Re-check 2026-10 of the tMCFE review fixes (perf/tmcfe-threshold 4de73ce..02ae981).

L1, L2, I2 (as reported), I3 and the info items are fixed on 02ae981 (see
docs/review/crypto_review_2026-10_tmcfe_recheck.md). Fail on 02ae981:

* L2b (low, latent): the native Shamir dealing (``_share_polys_native``) of
  all-bytes key material does not check each polynomial's length: bytes
  polynomials that are not whole words but add up to whole words are dealt
  shifted by a byte (shares of other values, no error); other malformed
  bytes raise ``ValueError``. The Python dealing refuses all of them
  (``FEKeyError``). No caller passes such material today;
* I2b (info, mostly as on b51989f): malformed value-list key shares are
  served or fail with other exceptions -- a Shamir ``z`` shorter than
  ``ring_n`` gives a shorter partial (an empty ``z`` an empty result), float
  values are truncated in the Python reads and sums (another result), wrong
  lengths and bad values raise ``IndexError`` / ``ZeroDivisionError`` /
  ``TypeError`` / ``ValueError``;
* I10 (info): a label part cached by the key generator (``dict_dk``) keeps the
  share set of its first issuance after the weights' sharing was evicted
  (``WEIGHT_SHARINGS_KEPT``): the generator then hands out weight parts of the
  new share set only, so servers holding the current weight part can never
  serve that label part (fail closed; availability);
* I11 (info): objects pickled by b51989f (no ``ct_format`` / ``threshold_format``
  / ``_pp_snapshot``) do not work after loading: ``AttributeError`` since
  5af955b, ``FESchemeError("the public parameters changed ...")`` since 02ae981
  for clients and servers, although nothing changed. Whether old pickles must
  load is a compatibility decision; the test encodes "they keep working".
"""

from __future__ import annotations

import pickle

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils.exceptions import FEError, FEKeyError

NATIVE = native_backend.available()
BACKENDS = ["python"] + (["native"] if NATIVE else [])
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
N = 32


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _setup(t=3, sharing="shamir", ct_format="json", **extra):
    cfg = dict(n=3, lst_nid=NIDS, s=3, t=t, lst_sid=SIDS, eta=1, ring_n=N, sec_param=64, bound_x=100, bound_y=3,
               stat_param=64, ct_format=ct_format, sharing=sharing, **extra)
    kg = tm.ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    return kg, kg.get_public_parameters()


def _obj(pp, nid, sk=None):
    keys = {"pp": pp} if sk is None else {"pp": pp, "sk": sk}
    return tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": keys})


def _bytes_material(q, case):
    vals = {nid: [17 * i + 3 + 100 * k for i in range(N)] for k, nid in enumerate(NIDS)}
    blobs = {nid: rlu.encode_values(v, q) for nid, v in vals.items()}
    wb = (int(q).bit_length() + 7) // 8
    if case == "misaligned":  # c0 a byte longer, c1 a byte shorter: whole words in total
        sk_y = {"c0": blobs["c0"] + b"\x00", "c1": blobs["c1"][:-1], "c2": blobs["c2"]}
    elif case == "not-whole-words":
        sk_y = {"c0": blobs["c0"] + b"\x01", "c1": blobs["c1"], "c2": blobs["c2"]}
    else:  # a value >= q
        sk_y = {"c0": blobs["c0"][:-wb] + b"\xff" * wb, "c1": blobs["c1"], "c2": blobs["c2"]}
    return {"sk_y": sk_y, "z": blobs["c2"], "z_chunks": []}


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("ct_format", ["json", "binary"])
@pytest.mark.parametrize("case", ["misaligned", "not-whole-words", "value-ge-q"])
def test_l2b_shamir_dealing_checks_each_binary_polynomial(name, ct_format, case):
    kg, pp = _setup(ct_format=ct_format)
    material = _bytes_material(gp.mpz(pp["q"]), case)
    with native_backend.backend(name):
        with pytest.raises(FEKeyError):
            kg._share_material(material)


def _short_z(dk, q):
    return dict(dk, z=dk["z"][:-1])


def _empty_z(dk, q):
    return dict(dk, z=[])


def _float_sk(dk, q):
    return dict(dk, sk_y=dict(dk["sk_y"], c0=[float(v) for v in dk["sk_y"]["c0"]]))


def _short_sk(dk, q):
    return dict(dk, sk_y=dict(dk["sk_y"], c0=dk["sk_y"]["c0"][:-1]))


def _empty_sk(dk, q):
    return dict(dk, sk_y=dict(dk["sk_y"], c0=[]))


def _none_value(dk, q):
    return dict(dk, z=[None] + list(dk["z"][1:]))


def _bad_digits(dk, q):
    return dict(dk, z=["abc"] + list(dk["z"][1:]))


def _float_piece(dk, q):
    pieces = [dict(p, share=dict(p["share"], sk_y=dict(p["share"]["sk_y"], c0=[float(v) for v in p["share"]["sk_y"]["c0"]])))
              for p in dk["pieces"]]
    return dict(dk, pieces=pieces)


def _long_values_z(dk, q):  # binary public parameters: a value list one longer than ring_n
    wb = (int(q).bit_length() + 7) // 8
    return dict(dk, z=[gp.digits(v) for v in rlu.decode_values(dk["z"], q, len(dk["z"]) // wb)] + ["0"])


def _nested_sk(dk, q):
    wb = (int(q).bit_length() + 7) // 8
    return dict(dk, sk_y=dict(dk["sk_y"], c0=[rlu.decode_values(dk["sk_y"]["c0"], q, len(dk["sk_y"]["c0"]) // wb)]))


I2B_CASES = [
    # (id, sharing, t, ct_format, backends, mutation) -- served or a non-library exception on 02ae981
    ("shamir z of n-1 values (served)", "shamir", 3, "json", BACKENDS, _short_z),
    ("shamir empty z (served, empty result)", "shamir", 3, "json", BACKENDS, _empty_z),
    ("shamir float sk_y (served truncated)", "shamir", 3, "json", ["python"], _float_sk),
    ("shamir float sk_y (TypeError)", "shamir", 3, "json", [b for b in BACKENDS if b == "native"], _float_sk),
    ("shamir sk_y of n-1 values (IndexError)", "shamir", 3, "json", BACKENDS, _short_sk),
    ("shamir empty sk_y (ZeroDivisionError)", "shamir", 3, "json", BACKENDS, _empty_sk),
    ("shamir None value (TypeError)", "shamir", 3, "json", [b for b in BACKENDS if b == "native"], _none_value),
    ("shamir non-decimal string (ValueError)", "shamir", 3, "json", [b for b in BACKENDS if b == "native"], _bad_digits),
    ("replicated float piece (served truncated)", "replicated", 2, "json", BACKENDS, _float_piece),
    ("binary shamir z value list n+1 (ValueError)", "shamir", 3, "binary", [b for b in BACKENDS if b == "native"], _long_values_z),
    ("binary shamir nested sk_y (TypeError)", "shamir", 3, "binary", [b for b in BACKENDS if b == "native"], _nested_sk),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in I2B_CASES])
def test_i2b_malformed_value_list_key_shares_are_refused(case):
    _, sharing, t, ct_format, backends, mutate = case
    if not backends:
        pytest.skip("native extension not built")
    kg, pp = _setup(t=t, sharing=sharing, ct_format=ct_format)
    clients = {nid: _obj(pp, nid, kg.get_private_keys(nid)) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    bad = mutate(kg.get_decryption_keys("s0", credentials=creds), gp.mpz(pp["q"]))
    for name in backends:
        with native_backend.backend(name):
            try:
                _obj(pp, "s0").share_decrypt(cts, creds, bad, SIDS[:t])
            except FEError:
                continue
            except Exception as exc:  # noqa: BLE001
                pytest.fail("{}: {}: {}".format(name, type(exc).__name__, exc))
            pytest.fail("{}: a malformed key share was served".format(name))


@pytest.mark.parametrize("sharing,t", [("shamir", 3), ("replicated", 2)])
def test_i10_a_cached_label_part_is_served_after_its_sharing_was_evicted(monkeypatch, sharing, t):
    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 1)
    kg, pp = _setup(t=t, sharing=sharing, ct_format="binary")
    wa, wb = {nid: [1] for nid in NIDS}, {nid: [2] for nid in NIDS}
    creds = {"fusion_weight": wa, "label": "L", "key_part": "label"}
    for sid in SIDS:
        kg.get_decryption_keys(sid, credentials=creds)  # cached (dict_dk) with the first share set
    kg.get_weight_key_share("s0", credentials={"fusion_weight": wb})  # evicts the sharing of wa
    clients = {nid: _obj(pp, nid, kg.get_private_keys(nid)) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    parts = {}
    for sid in SIDS[:t]:
        srv = _obj(pp, sid)
        srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": wa}))  # what it hands out now
        parts[sid] = srv.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), SIDS[:t])
    assert _obj(pp, "agg").combine_decrypt(parts) == [3, 6, 9]


def _as_b51989f(obj, missing):
    """*obj* as a pickle of b51989f loads: without the attributes added since."""
    state = dict(obj.__getstate__() or obj.__dict__)
    for name in missing:
        state.pop(name, None)
    old = object.__new__(type(obj))
    old.__dict__.update(state)
    return pickle.loads(pickle.dumps(old))


def test_i11_objects_pickled_by_b51989f_keep_working():
    kg, pp = _setup(t=2, sharing="replicated")
    client = _obj(pp, "c0", kg.get_private_keys("c0"))
    server = _obj(pp, "s0")
    old_kg = _as_b51989f(kg, ("ct_format", "dict_weight_shares"))
    old_client = _as_b51989f(client, ("_binary_ciphertexts", "_pp_snapshot", "threshold_format"))
    old_server = _as_b51989f(server, ("_binary_ciphertexts", "_pp_snapshot", "threshold_format"))
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    cts = {nid: _obj(pp, nid, kg.get_private_keys(nid)).encrypt([[1, 2, 3]], "L") for nid in NIDS}
    old_client.encrypt([[1, 2, 3]], "M")
    old_server.share_decrypt(cts, creds, kg.get_decryption_keys("s0", credentials=creds), SIDS[:2])
    old_kg.get_decryption_keys("s0", credentials={"fusion_weight": {nid: [1] for nid in NIDS}, "label": "N"})


def test_i10_i11_controls_pass_on_the_fix():
    """Controls: without eviction a label part and the current weight part serve; fresh objects work."""
    kg, pp = _setup(t=3, ct_format="binary")
    w = {nid: [1] for nid in NIDS}
    creds = {"fusion_weight": w, "label": "L", "key_part": "label"}
    clients = {nid: _obj(pp, nid, kg.get_private_keys(nid)) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    parts = {}
    for sid in SIDS:
        srv = _obj(pp, sid)
        srv.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": w}))
        parts[sid] = srv.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), SIDS)
    assert _obj(pp, "agg").combine_decrypt(parts) == [3, 6, 9]
    fresh = pickle.loads(pickle.dumps(_obj(pp, "c0", kg.get_private_keys("c0"))))
    fresh.encrypt([[1, 2, 3]], "M")
