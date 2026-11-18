"""Recipes of the record fixtures: what the 1.x compatibility promise is checked against.

A recipe runs one scheme through its public API once -- set-up, keys, encryption, the threshold or decentralised
steps, decryption -- and keeps every record that comes out (public parameters, encryption keys, functional keys,
key shares, ciphertexts, partial decryptions) together with the parameter records of its cache, its inputs and the
results. :func:`generate` makes such a fixture; :func:`replay` reads one with the code under test, as a later
version would, and returns what that code computes from it. ``tests/compat/test_fixtures.py`` compares the two.

The fixtures are frozen once written (``docs/compatibility``, roadmap ``v1.0/02`` Sec. 8): the structure of a
fixture is fixed by the recipe (scheme set, inputs, record roles), its random bytes are those of the run that wrote
it. Only the public API is used, so that a later version reads the records exactly as an application would.
"""

from __future__ import annotations

import base64
import json
import tempfile
import warnings
from pathlib import Path

import pyfe4ai
from pyfe4ai.core import config as configs

#: Version of the fixture file layout (not of pyfe4ai).
FIXTURE_FORMAT = 1

# ---------------------------------------------------------------------------
# the file encoding: JSON with tags for what JSON lacks


def encode(value):
    """*value* as JSON data: ``bytes`` as ``{"$bytes": base64}``, tuples as ``{"$tuple": [...]}``, a dict with a
    key that is not a ``str`` as ``{"$dict": [[key, value], ...]}``; everything else as is."""
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, tuple):
        return {"$tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: encode(v) for k, v in value.items()}
        return {"$dict": [[encode(k), encode(v)] for k, v in value.items()]}
    if value is None or isinstance(value, (bool, str, float)):
        return value
    if isinstance(value, int):
        return int(value)
    if type(value).__name__ == "mpz":  # gmpy2, written as the int it is
        return int(value)
    raise TypeError("no fixture encoding for {}".format(type(value).__name__))


def decode(value):
    """The inverse of :func:`encode`."""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$bytes"}:
            return base64.b64decode(value["$bytes"])
        if set(value) == {"$tuple"}:
            return tuple(decode(v) for v in value["$tuple"])
        if set(value) == {"$dict"}:
            return {_hashable(decode(k)): decode(v) for k, v in value["$dict"]}
        return {k: decode(v) for k, v in value.items()}
    return value


def _hashable(key):
    return tuple(_hashable(k) for k in key) if isinstance(key, (list, tuple)) else key


def dump(fixture: dict, path: Path) -> None:
    path.write_text(json.dumps(encode(fixture), indent=1, sort_keys=False) + "\n")


def load(path: Path) -> dict:
    return decode(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# inputs: small, fixed, inside every toy bound

LABEL = "compat-label-1"
NIDS = ["nid_0", "nid_1"]
SIDS = ["sid_0", "sid_1", "sid_2"]
SIFE_X, SIFE_Y, SIFE_X2 = [3, 1], [2, 5], [1, 4]
MULTI_X = {"nid_0": [3, 1], "nid_1": [2, 2]}
MULTI_Y = {"nid_0": [2, 5], "nid_1": [1, 1]}
MULTI_X2 = {"nid_0": [1, 4], "nid_1": [0, 3]}
QUAD_X, QUAD_Y = [1, 2], [2, 1]
QUAD_F = [[1, 0], [2, 1]]
UNIVERSE = ["org:A", "role:auditor", "role:regulator"]
CP_POLICY, CP_ATTRIBUTES = "and(org:A, or(role:auditor, role:regulator))", ["org:A", "role:auditor"]
KP_ATTRIBUTES, KP_POLICY = ["purpose:research", "region:EU"], "and(purpose:research, region:EU)"


def ip(x, y) -> int:
    return sum(a * b for a, b in zip(x, y))


def multi_ip(x: dict, y: dict) -> int:
    return sum(ip(x[n], y[n]) for n in x)


#: Options of the fixture configs, cut to what each scheme declares (toy profile, two slots, small bounds).
BASE_CONFIG = {"eta": 2, "n": 2, "s": 1, "lst_nid": NIDS, "bound_x": 8, "bound_y": 8, "bound": 5,
               "universe": UNIVERSE}
#: Per-scheme overrides.
CONFIG = {
    "mife/ddh_threshold": {"n": 2, "s": 3, "t": 2, "lst_sid": SIDS},
    "mcfe/ddh_threshold": {"n": 2, "s": 3, "t": 2, "lst_sid": SIDS},
    "quadratic/quad": {"n": 2, "m": 2},
    # ALS16 publishes its m x lwe_n matrix (m = 4 lwe_n log2 q): the smallest toy dimension keeps the file small
    "sife/fullysec_lwe": {"lwe_n": 4},
    "mife/fullysec_lwe": {"lwe_n": 4},
    "quadratic/sgp": {"n": 2},
    "quadratic/miqfe": {"n": 2, "m": 1, "bound_x": 4, "bound_c": 3},
}
#: MIQFE: one integer per client and the coefficients of the cross term x_0 x_1.
MIQFE_X, MIQFE_X2, MIQFE_C = {"nid_0": [2], "nid_1": [3]}, {"nid_0": [1], "nid_1": [2]}, [[0, 3], [0, 0]]
#: PartFH-IPE: the coefficients of the public encryption (in the span of ``M``; ``subspace_dim`` = 1).
PART_T, PART_T2 = [1], [-1]


def config_of(scheme: str, cache_dir: str) -> dict:
    declared = configs.config_class(tuple(scheme.split("/"))).OPTIONS
    cfg = {k: v for k, v in dict(BASE_CONFIG, **CONFIG.get(scheme, {})).items() if k in declared}
    cfg["cache_dir"] = cache_dir
    return cfg


def kind_of(scheme: str) -> str:
    """The recipe a scheme follows."""
    entry = pyfe4ai.get_scheme(scheme)
    family, variant = scheme.split("/")
    if scheme in ("sife/cp_ab_pairing", "sife/kp_ab_pairing", "sife/part_fh_ipe_pairing"):
        return scheme.split("/")[1]
    if family == "quadratic":
        return "miqfe" if variant == "miqfe" else "quadratic"
    if entry.has("threshold"):
        return "threshold"
    if entry.has("decentralized"):
        return "decentralized"
    return family


def _matrix(scheme: str) -> bool:
    """Whether encrypt takes a matrix (one row per vector: the Ring-LWE schemes)."""
    return scheme.split("/")[1].startswith("ring_lwe")


def _pt(scheme, vec):
    """The plaintext of one vector: a column of ``eta`` rows for the Ring-LWE schemes."""
    return [[v] for v in vec] if _matrix(scheme) else vec


def _cache_files(cache_dir: str) -> dict:
    root = Path(cache_dir)
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# generate


def generate(scheme: str) -> dict:
    """Run *scheme* once and return its fixture (see the module docstring)."""
    with tempfile.TemporaryDirectory() as cache_dir:
        cfg = config_of(scheme, cache_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kg, crypto = pyfe4ai.create(scheme, cfg)
            kg.setup()
            records, inputs, expected = _GENERATE[kind_of(scheme)](scheme, kg, crypto)
        params = _cache_files(cache_dir)
    cfg.pop("cache_dir")
    return {
        "fixture_format": FIXTURE_FORMAT, "pyfe4ai_version": pyfe4ai.__version__, "scheme": scheme,
        "recipe": kind_of(scheme), "config": cfg, "inputs": inputs, "records": records, "expected": expected,
        "param_records": params,
    }


def _gen_sife(scheme, kg, C):
    pp, sk = kg.get_public_parameters(), kg.get_private_keys()
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": SIFE_Y})
    ct = C({"id": "nid_default", "keys": {"pp": pp, "sk": sk}}).encrypt(_pt(scheme, SIFE_X))
    result = _scalar(C({"id": "sid_0", "keys": {"pp": pp}}).decrypt(ct, dk, SIFE_Y))
    _check(result, ip(SIFE_X, SIFE_Y), scheme)
    return ({"pp": pp, "sk": sk, "dk": dk, "ct": ct}, {"x": SIFE_X, "y": SIFE_Y, "x2": SIFE_X2},
            {"decrypt": result, "decrypt_fresh": ip(SIFE_X2, SIFE_Y)})


def _gen_ab(scheme, kg, C):
    cp = scheme.endswith("cp_ab_pairing")
    pp = kg.get_public_parameters()
    cred = {"fusion_weight": SIFE_Y, **({"attributes": CP_ATTRIBUTES} if cp else {"policy": KP_POLICY})}
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    crypto = C({"keys": {"pp": pp}})
    ct = crypto.encrypt(SIFE_X, policy=CP_POLICY) if cp else crypto.encrypt(SIFE_X, attributes=KP_ATTRIBUTES)
    result = crypto.decrypt(ct, dk)
    _check(result, ip(SIFE_X, SIFE_Y), scheme)
    return ({"pp": pp, "dk": dk, "ct": ct}, {"x": SIFE_X, "y": SIFE_Y, "x2": SIFE_X2},
            {"decrypt": result, "decrypt_fresh": ip(SIFE_X2, SIFE_Y)})


def _encryptors(scheme, kg, C, pp):
    sks = {nid: kg.get_private_keys(nid) for nid in NIDS}
    return sks, {nid: C({"id": nid, "keys": {"pp": pp, "sk": sks[nid]}}) for nid in NIDS}


def _gen_multi(scheme, kg, C, labelled: bool):
    pp = kg.get_public_parameters()
    sks, enc = _encryptors(scheme, kg, C, pp)
    lab = (LABEL,) if labelled else ()
    cts = {nid: enc[nid].encrypt(_pt(scheme, MULTI_X[nid]), *lab) for nid in NIDS}
    # the identity-based MCFE over Ring-LWE issues a key per label
    cred = {"fusion_weight": MULTI_Y, **({"label": LABEL} if scheme == "mcfe/ring_lwe" else {})}
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    result = _scalar(C({"id": "sid_0", "keys": {"pp": pp}}).decrypt(cts, dk, MULTI_Y, *lab))
    _check(result, multi_ip(MULTI_X, MULTI_Y), scheme)
    return ({"pp": pp, "sk": sks, "dk": dk, "ct": cts}, {"x": MULTI_X, "y": MULTI_Y, "x2": MULTI_X2, "label": LABEL},
            {"decrypt": result, "decrypt_fresh": multi_ip(MULTI_X2, MULTI_Y)})


def _derive(scheme, crypto, nid):
    if scheme.endswith("fh_multi_ipe_pairing_decentralized"):
        return crypto.derive_function_decryption_key_share(MULTI_Y[nid], "compat-weights")
    if scheme.endswith("ring_lwe_decentralized"):
        return crypto.derive_function_decryption_key_share(MULTI_Y, LABEL)
    return crypto.derive_function_decryption_key_share(MULTI_Y)


def _gen_decentralized(scheme, kg, C):
    pp = kg.get_public_parameters()
    sks, enc = _encryptors(scheme, kg, C, pp)
    cts = {nid: enc[nid].encrypt(_pt(scheme, MULTI_X[nid]), LABEL) for nid in NIDS}
    shares = {nid: _derive(scheme, enc[nid], nid) for nid in NIDS}
    dec = C({"id": "sid_0", "keys": {"pp": pp}})
    dk = dec.combine_function_decryption_key_share(shares)
    result = _scalar(dec.decrypt(cts, dk, MULTI_Y, LABEL))
    _check(result, multi_ip(MULTI_X, MULTI_Y), scheme)
    return ({"pp": pp, "sk": sks, "dk_shares": shares, "dk": dk, "ct": cts},
            {"x": MULTI_X, "y": MULTI_Y, "x2": MULTI_X2, "label": LABEL},
            {"decrypt": result, "decrypt_fresh": multi_ip(MULTI_X2, MULTI_Y)})


def _labelled(scheme):
    return scheme.startswith("mcfe/")


def _share(scheme, C, pp, sid, cts, dk, servers):
    server = C({"id": sid, "keys": {"pp": pp, "dk": dk}})
    lab = (LABEL,) if _labelled(scheme) else ()
    return server.share_decrypt(cts, MULTI_Y, dk, servers, *lab)


def _combine(scheme, client, partials):
    return client.combine_decrypt(partials, LABEL) if _labelled(scheme) else client.combine_decrypt(partials)


def _gen_threshold(scheme, kg, C):
    pp = kg.get_public_parameters()
    sks, enc = _encryptors(scheme, kg, C, pp)
    lab = (LABEL,) if _labelled(scheme) else ()
    cts = {nid: enc[nid].encrypt(MULTI_X[nid], *lab) for nid in NIDS}
    cred = {"fusion_weight": MULTI_Y, **({"label": LABEL} if lab else {})}
    servers = SIDS[1:]
    dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in servers}
    partials = {sid: _share(scheme, C, pp, sid, cts, dks[sid], servers) for sid in servers}
    result = _combine(scheme, enc[NIDS[0]], partials)
    _check(result, multi_ip(MULTI_X, MULTI_Y), scheme)
    return ({"pp": pp, "sk": sks, "dk": dks, "ct": cts, "partials": partials},
            {"x": MULTI_X, "y": MULTI_Y, "x2": MULTI_X2, "label": LABEL, "servers": servers},
            {"decrypt": result, "decrypt_fresh": multi_ip(MULTI_X2, MULTI_Y)})


def _gen_quadratic(scheme, kg, C):
    pp = kg.get_public_parameters()
    ct = C({"id": "nid_default", "keys": {"pp": pp}}).encrypt({"x": QUAD_X, "y": QUAD_Y})
    dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": QUAD_F})
    result = C({"id": "sid_0", "keys": {"pp": pp}}).decrypt(ct, dk)
    want = sum(QUAD_X[i] * QUAD_F[i][j] * QUAD_Y[j] for i in range(2) for j in range(2))
    _check(result, want, scheme)
    return {"pp": pp, "dk": dk, "ct": ct}, {"x": QUAD_X, "y": QUAD_Y, "f": QUAD_F}, {"decrypt": result}


def _span(pp, t):
    m = [[int(v) for v in row] for row in pp["m"]]
    return [sum(m[i][j] * t[j] for j in range(len(t))) for i in range(len(m))]


def _gen_part_fh(scheme, kg, C):
    pp = kg.get_public_parameters()
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": SIFE_Y})
    ct_secret = kg.sec_encrypt(SIFE_X)
    ct_public = C({"id": "nid_default", "keys": {"pp": pp}}).encrypt(PART_T)
    dec = C({"id": "sid_0", "keys": {"pp": pp}})
    result, result_public = dec.decrypt(ct_secret, dk), dec.decrypt(ct_public, dk)
    _check(result, ip(SIFE_X, SIFE_Y), scheme)
    _check(result_public, ip(_span(pp, PART_T), SIFE_Y), scheme)
    return ({"pp": pp, "dk": dk, "ct": ct_secret, "ct_public": ct_public},
            {"x": SIFE_X, "y": SIFE_Y, "t": PART_T, "t2": PART_T2},
            {"decrypt": result, "decrypt_public": result_public, "decrypt_fresh": ip(_span(pp, PART_T2), SIFE_Y)})


def _miqfe(x, c):
    return sum(c[i][j] * x[NIDS[i]][0] * x[NIDS[j]][0] for i in range(2) for j in range(2))


def _gen_miqfe(scheme, kg, C):
    pp = kg.get_public_parameters()
    sks = {nid: kg.get_private_keys(nid) for nid in NIDS}
    cts = {nid: C({"keys": {"pp": pp, "sk": sks[nid]}}).encrypt(MIQFE_X[nid]) for nid in NIDS}
    dk = kg.get_decryption_keys(None, credentials={"coefficients": MIQFE_C})
    result = C({"keys": {"pp": pp}}).decrypt(cts, dk)
    _check(result, _miqfe(MIQFE_X, MIQFE_C), scheme)
    return ({"pp": pp, "sk": sks, "dk": dk, "ct": cts}, {"x": MIQFE_X, "x2": MIQFE_X2, "c": MIQFE_C},
            {"decrypt": result, "decrypt_fresh": _miqfe(MIQFE_X2, MIQFE_C)})


def _check(result, want, scheme):
    got = result[0] if isinstance(result, list) and len(result) == 1 else result
    if got != want:
        raise AssertionError("{}: decrypted {!r}, expected {!r}".format(scheme, result, want))


_GENERATE = {
    "sife": _gen_sife,
    "cp_ab_pairing": _gen_ab,
    "kp_ab_pairing": _gen_ab,
    "mife": lambda s, kg, C: _gen_multi(s, kg, C, labelled=False),
    "mcfe": lambda s, kg, C: _gen_multi(s, kg, C, labelled=True),
    "decentralized": _gen_decentralized,
    "threshold": _gen_threshold,
    "quadratic": _gen_quadratic,
    "part_fh_ipe_pairing": _gen_part_fh,
    "miqfe": _gen_miqfe,
}


# ---------------------------------------------------------------------------
# replay: the reading half, with the code under test


def replay(fx: dict) -> dict:
    """What the code under test computes from the records of fixture *fx*: ``decrypt`` (the stored ciphertexts
    under the stored keys) and, where the scheme has encryption keys or public encryption, ``decrypt_fresh`` (a
    fresh ciphertext of ``x2`` under the stored keys, decrypted with the stored functional key)."""
    scheme, rec, inp = fx["scheme"], fx["records"], fx["inputs"]
    C = pyfe4ai.get_scheme(scheme).crypto
    kind = fx["recipe"]
    pp = rec["pp"]
    out = {}
    if kind == "sife":
        dec = C({"id": "sid_0", "keys": {"pp": pp}})
        out["decrypt"] = _scalar(dec.decrypt(rec["ct"], rec["dk"], inp["y"]))
        ct2 = C({"id": "nid_default", "keys": {"pp": pp, "sk": rec["sk"]}}).encrypt(_pt(scheme, inp["x2"]))
        out["decrypt_fresh"] = _scalar(dec.decrypt(ct2, rec["dk"], inp["y"]))
    elif kind in ("cp_ab_pairing", "kp_ab_pairing"):
        crypto = C({"keys": {"pp": pp}})
        out["decrypt"] = crypto.decrypt(rec["ct"], rec["dk"])
        ct2 = (crypto.encrypt(inp["x2"], policy=CP_POLICY) if kind == "cp_ab_pairing"
               else crypto.encrypt(inp["x2"], attributes=KP_ATTRIBUTES))
        out["decrypt_fresh"] = crypto.decrypt(ct2, rec["dk"])
    elif kind in ("mife", "mcfe", "decentralized"):
        lab = (inp["label"],) if kind != "mife" else ()
        dec = C({"id": "sid_0", "keys": {"pp": pp}})
        dk = rec["dk"]
        if kind == "decentralized":
            dk = dec.combine_function_decryption_key_share(rec["dk_shares"])
        out["decrypt"] = _scalar(dec.decrypt(rec["ct"], dk, inp["y"], *lab))
        cts2 = {nid: C({"id": nid, "keys": {"pp": pp, "sk": rec["sk"][nid]}}).encrypt(_pt(scheme, inp["x2"][nid]), *lab)
                for nid in rec["sk"]}
        out["decrypt_fresh"] = _scalar(dec.decrypt(cts2, rec["dk"], inp["y"], *lab))
    elif kind == "threshold":
        servers = inp["servers"]
        lab = (inp["label"],) if _labelled(scheme) else ()
        client = C({"id": NIDS[0], "keys": {"pp": pp, "sk": rec["sk"][NIDS[0]]}})
        out["combine_stored"] = _combine(scheme, client, rec["partials"])
        partials = {sid: _share(scheme, C, pp, sid, rec["ct"], rec["dk"][sid], servers) for sid in servers}
        out["decrypt"] = _combine(scheme, client, partials)
        cts2 = {nid: C({"id": nid, "keys": {"pp": pp, "sk": rec["sk"][nid]}}).encrypt(inp["x2"][nid], *lab)
                for nid in rec["sk"]}
        partials2 = {sid: _share(scheme, C, pp, sid, cts2, rec["dk"][sid], servers) for sid in servers}
        out["decrypt_fresh"] = _combine(scheme, client, partials2)
    elif kind == "quadratic":
        out["decrypt"] = C({"id": "sid_0", "keys": {"pp": pp}}).decrypt(rec["ct"], rec["dk"])
    elif kind == "part_fh_ipe_pairing":
        dec = C({"id": "sid_0", "keys": {"pp": pp}})
        out["decrypt"] = dec.decrypt(rec["ct"], rec["dk"])
        out["decrypt_public"] = dec.decrypt(rec["ct_public"], rec["dk"])
        ct2 = C({"id": "nid_default", "keys": {"pp": pp}}).encrypt(inp["t2"])
        out["decrypt_fresh"] = dec.decrypt(ct2, rec["dk"])
    elif kind == "miqfe":
        out["decrypt"] = C({"keys": {"pp": pp}}).decrypt(rec["ct"], rec["dk"])
        cts2 = {nid: C({"keys": {"pp": pp, "sk": rec["sk"][nid]}}).encrypt(inp["x2"][nid]) for nid in rec["sk"]}
        out["decrypt_fresh"] = C({"keys": {"pp": pp}}).decrypt(cts2, rec["dk"])
    else:
        raise ValueError("no replay for recipe {!r}".format(kind))
    return out


def _scalar(result):
    return result[0] if isinstance(result, list) and len(result) == 1 else result


def expected_replay(fx: dict) -> dict:
    """What :func:`replay` must return for *fx*."""
    exp = dict(fx["expected"])
    if fx["recipe"] == "threshold":
        exp["combine_stored"] = exp["decrypt"]
    return exp


def reuse_param_records(fx: dict) -> tuple[dict, dict, list]:
    """Put the fixture's parameter records in a fresh cache, set up a key generator of the fixture's config on it,
    and return the cache files before and after, and the warnings of the set-up."""
    with tempfile.TemporaryDirectory() as cache_dir:
        root = Path(cache_dir)
        for rel, text in fx["param_records"].items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        before = _cache_files(cache_dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kg, _ = pyfe4ai.create(fx["scheme"], dict(fx["config"], cache_dir=cache_dir))
            kg.setup()
        after = _cache_files(cache_dir)
    return before, after, [str(w.message) for w in caught if issubclass(w.category, pyfe4ai.FEWarning)
                           and "native extension" not in str(w.message)]


def record_keys(value) -> set:
    """The top-level field names of a record, or of the records of a ``{id: record}`` map (first entry)."""
    if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()) and \
            all(isinstance(k, str) and (k.startswith("nid_") or k.startswith("sid_")) for k in value):
        return record_keys(next(iter(value.values())))
    return set(value) if isinstance(value, dict) else set()


def available(scheme: str) -> bool:
    """Whether the scheme runs here: the pairing schemes need a backend for the default pairing group (BLS12-381,
    from the native extension), which the fixtures are written in."""
    if pyfe4ai.get_scheme(scheme).kind != "pairing":
        return True
    from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, pairing_available

    return pairing_available(DEFAULT_PAIRING_GROUP)
