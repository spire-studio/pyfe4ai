"""Review 2026-10 (perf/tmcfe-threshold b51989f..0716163): binary, RNS and split tMCFE.

Format 2, tags v2, pp_binding, the RNS ring, split and binary key shares hold
as specified (see test_tmcfe_threshold_invariants.py for the checks that
pass). Fail on 0716163:

* L2 (latent): the native Shamir dealing of ``_share_material`` reads key
  material that mixes bytes polynomials with value lists byte by byte (each
  byte a coefficient): the shares are of other values, without an error. The
  Python dealing reads the mix right. No caller passes such a mix today
  (full keys are all values, label parts all bytes, weight parts all values);
* I2: replicated pieces mixing bytes and decimal polynomials, and key-share
  polynomials given as ``memoryview``, are refused with ``ValueError`` /
  ``TypeError`` / ``IndexError`` from the parsers instead of ``FEKeyError``
  ("malformed binary key share");
* I3: a server fixes its ciphertext format when it is created, while it binds
  key shares and seeds the noise with the digest of its public parameters as
  they are at each call (``self.pp`` is the caller's dict): after an
  in-place ``pp["ct_format"] = "binary"`` it serves JSON ciphertexts with a
  binary key share, with the JSON derivation of the noise inputs, so one key
  part gets two noises (format 2 of the same values gets the other).
"""

from __future__ import annotations

import gmpy2 as gp
import pytest

from pyfe4ai.schemes import threshold_utils as tu
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.threshold_utils import pp_binding, tag_ciphertext_message
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError

NATIVE = native_backend.available()
BACKENDS = ["python"] + (["native"] if NATIVE else [])
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
CREDS = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _setup(t, sharing, ct_format="binary", **extra):
    cfg = dict(n=3, lst_nid=NIDS, s=3, t=t, lst_sid=SIDS, eta=1, ring_n=32, sec_param=64, bound_x=100, bound_y=3,
               stat_param=64, ct_format=ct_format, sharing=sharing, **extra)
    # the defaults when these tests were written: a prime q, whole key shares (split ones: key_part="label")
    cfg.setdefault("ring_modulus", "prime")
    cfg.setdefault("key_shares", "full")
    kg = tm.ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    return kg, kg.get_public_parameters()


def _obj(pp, nid, sk=None):
    keys = {"pp": pp} if sk is None else {"pp": pp, "sk": sk}
    return tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": keys})


def _values(poly, q):
    if isinstance(poly, bytes):
        return rlu.decode_values(poly, q, len(poly) // ((int(q).bit_length() + 7) // 8))
    return [int(v) for v in poly]


@pytest.mark.skipif(not NATIVE, reason="native extension not built")
def test_l2_native_shamir_dealing_of_mixed_material():
    kg, pp = _setup(3, "shamir")
    q = gp.mpz(pp["q"])
    n = pp["ring_n"]
    z = [3 * i + 1 for i in range(n)]
    material = {"sk_y": {nid: [str(i + 7 * j) for i in range(n)] for j, nid in enumerate(NIDS)},
                "z": rlu.encode_values(z, q), "z_chunks": []}
    points = tu.share_points(SIDS)
    xs = list(points.values())
    for name in BACKENDS:
        with native_backend.backend(name):
            shares = kg._share_material(material)
        got = [0] * n
        for sid in SIDS:
            lam = tu.lagrange_coefficient_at_zero(points[sid], xs, q)
            vals = _values(shares[sid]["z"], q)
            assert len(vals) == n, "{}: z shared as {} values, not {}".format(name, len(vals), n)
            got = [(a + lam * v) % q for a, v in zip(got, vals)]
        assert got == z, name


def _mixed_dk(kg):
    dk = kg.get_decryption_keys("s0", credentials=CREDS)
    q = gp.mpz(kg.get_public_parameters()["q"])

    def dec(blob):
        return [gp.digits(v) for v in _values(blob, q)]

    pieces = [dict(p, share=dict(p["share"], sk_y={k: dec(v) for k, v in p["share"]["sk_y"].items()}))
              for p in dk["pieces"]]
    return dk, dict(dk, pieces=pieces)


def _memoryview_dk(dk):
    def mv(obj, key=None):
        if isinstance(obj, bytes) and key not in ("noise_key",):
            return memoryview(obj)
        if isinstance(obj, dict):
            return {k: (v if k in ("mac_keys", "pp_binding") else mv(v, k)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [mv(v) for v in obj]
        return obj

    return mv(dk)


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("case", ["mixed-pieces", "memoryview-replicated", "memoryview-shamir"])
def test_i2_malformed_key_material_is_a_key_error(name, case):
    t, sharing = (3, "shamir") if case == "memoryview-shamir" else (2, "replicated")
    kg, pp = _setup(t, sharing)
    clients = {nid: _obj(pp, nid, kg.get_private_keys(nid)) for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    if case == "mixed-pieces":
        _, bad = _mixed_dk(kg)
    else:
        bad = _memoryview_dk(kg.get_decryption_keys("s0", credentials=CREDS))
    with native_backend.backend(name):
        try:
            _obj(pp, "s0").share_decrypt(cts, CREDS, bad, SIDS[:t])
        except (FEKeyError, FESchemeError):
            return
        except Exception as exc:  # noqa: BLE001
            pytest.fail("{}: {}: {}".format(case, type(exc).__name__, exc))
        pytest.fail("{}: served".format(case))


def test_i3_server_format_follows_the_parameters_it_binds():
    kg, pp_bin = _setup(3, "shamir")
    pp_json = {k: v for k, v in pp_bin.items() if k != "ct_format"}
    sks = {nid: kg.get_private_keys(nid) for nid in NIDS}
    # JSON ciphertexts with v1 tags under this setup's MAC keys (a client holding them can make these)
    json_cts = {nid: _obj(pp_json, nid, sks[nid]).encrypt([[1, 2, 3]], "L") for nid in NIDS}
    dk = kg.get_decryption_keys("s0", credentials=CREDS)
    live = dict(pp_json)
    stale = _obj(live, "s0")
    live["ct_format"] = "binary"  # the caller's dict, changed in place after the server was made
    try:
        part = stale.share_decrypt(json_cts, CREDS, dk, SIDS)
    except (FEKeyError, FESchemeError):
        return  # refused: one derivation per binding
    # served: then format 2 of the same values must get the same noise (it does not)
    q, eta, n = int(pp_bin["q"]), pp_bin["eta"], pp_bin["ring_n"]
    digest = bytes.fromhex(pp_binding(pp_bin, tm.binding_fields(pp_bin), tm._BINDING_DEFAULTS))
    bin_cts = {}
    for nid, ct in json_cts.items():
        k = ct["k"]
        data = rlu.encode_threshold_ciphertext([[int(v) for v in row] for row in ct["ct0"]],
                                               [int(v) for v in ct["ct1"]], k, q, n)
        b = {"ct": data, "k": k, "label_hash": ct["label_hash"]}
        msg = tm.binary_tag_message(digest, encode_label("L"), b, tm.binary_ciphertext_digests(b, q, eta, n))
        bin_cts[nid] = tag_ciphertext_message(b, sks[nid]["mac_keys"], msg)
    good = _obj(pp_bin, "s0").share_decrypt(bin_cts, CREDS, dk, SIDS)
    k = good["k"]
    assert [int(v) for v in part["w"][:k]] == rlu.decode_values(good["w"], q, k), (
        "one key part (same ct1 values, same key share, same pp_binding) answered with two noises")
