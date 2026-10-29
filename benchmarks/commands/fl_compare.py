"""Federated gradient aggregation per scheme family, for before/after comparisons.

Every client encrypts a gradient of ``--elements`` values in ``[-1, 1]`` with
``--precision`` decimal digits through :func:`~pyfe4ai.ml.encrypt_gradient`;
the aggregator derives the functional key for the round label and decrypts the
weighted sum (``--weights``, one integer per client, default all 1; ``bound_y``
is the largest absolute weight) with :func:`~pyfe4ai.ml.aggregate_gradients`. Each
round uses a fresh label; round 0 warms the caches and is not counted. The
result is checked against the plaintext weighted sum.

Reported per family (medians over the counted rounds):

* ``client_ms``: one client's ``encrypt_gradient``;
* ``aggregator_ms``: ``aggregate_gradients`` (key derivation + decryption),
  with ``dkgen_ms`` the key derivation inside it;
* ``ct_bytes_per_element``: one client's ciphertexts as JSON (bytes values
  of the binary ciphertext format, ``--ct-format binary``, counted at their
  length), per element.

The threshold family ``ring_lwe_threshold`` (``mcfe/ring_lwe_threshold``,
``--servers`` servers, threshold ``--threshold``) runs on the scheme classes,
for the time of each party (:func:`~pyfe4ai.ml.aggregate_gradients`
runs the same steps in one process): ``client_ms`` is one client's
``encrypt_lst_ndarray``; ``dkgen_ms`` the key shares of the ``t`` enrolled
servers; ``partial_ms`` one server's ``compute_lst_ndarray_ct``;
``combine_ms`` the combiner's ``decrypt_lst_ndarray_ct``; ``aggregator_ms``
their sum over the ``t`` servers on one machine. It runs with the scheme's
defaults -- binary ciphertexts, an RNS modulus and split key shares: the
weight part issued once before the rounds (``weight_shares_ms``), per round
only the label part (``label_term=False``) -- unless ``--ct-format json``,
``--ring-modulus prime`` or ``--key-shares full`` select the options of
earlier versions.
``--flooding-accounting kl --max-partial-decryptions Q`` runs it with the
opt-in KL accounting of its flooding noise (see
:mod:`pyfe4ai.schemes.threshold.kl`; it needs ``--threshold`` equal
to ``--servers``), one budget per server.

The schemes write their parameter cache to ``<workdir>/config`` (as
``PYFE4AI_CACHE_DIR``); the run happens in ``--workdir`` (default: a
temporary directory) so a cache can be reused across runs. Usage::

    python benchmarks/run.py fl-compare --families ring_lwe --elements 10000 --json out.json
    python benchmarks/run.py fl-compare --families ring_lwe --backend native --threads 4
    python benchmarks/run.py fl-compare --families ring_lwe --ct-format binary --weights 1,2,3
    python benchmarks/run.py fl-compare --families ring_lwe lwe --label-hash shake128-rejection
    python benchmarks/run.py fl-compare --families ring_lwe_threshold  # binary, RNS, split key shares
    python benchmarks/run.py fl-compare --families ring_lwe_threshold --ct-format json --ring-modulus prime \
        --key-shares full  # the options of earlier versions
    python benchmarks/run.py fl-compare --families ring_lwe --ring-modulus rns  # q = q1 q2 (opt-in)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from unittest import mock

import numpy as np

#: family -> (scheme, keygen config without bounds, default element count); the schemes of pyfe4ai.core.registry
FAMILIES = {
    "ring_lwe": (("mcfe", "ring_lwe"), {"eta": 1}, 10_000),
    "ring_lwe_threshold": (("mcfe", "ring_lwe_threshold"), {"eta": 1}, 10_000),
    "ring_lwe_decentralized": (("mcfe", "ring_lwe_decentralized"), {"eta": 1}, 10_000),
    "lwe": (("mcfe", "lwe"), {"eta": 1}, 200),
    "ddh": (("mcfe", "ddh"), {"eta": 1}, 20),
    "ddh_r255": (("mcfe", "ddh_ristretto255"), {"eta": 1}, 20),
    "damgard": (("mcfe", "damgard"), {"eta": 1}, 20),
    "paillier": (("mcfe", "paillier"), {"eta": 1}, 20),
    "pairing": (("mcfe", "fh_multi_ipe_pairing"), {"eta": 1}, 5),
}
#: Families with an offline phase (``--offline``): precompute_gradient / prepare_aggregation.
OFFLINE_FAMILIES = ("ring_lwe", "ddh", "damgard", "paillier")


def _bounds(family: str, bound_x: int, bound_y: int = 1) -> dict:
    if family in ("damgard",):
        return {"bound": max(bound_x, bound_y)}
    return {"bound_x": bound_x, "bound_y": bound_y}


def _env() -> dict:
    import gmpy2

    info = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gmpy2": gmpy2.version(),
        "numpy": np.__version__,
        "cpu": _cpu_name(),
        "benchmark_campaign_running": _campaign_running(),
    }
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        info["git_commit"] = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        pass
    try:
        from pyfe4ai import backend as native_backend

        info["native"] = native_backend.describe()
    except ImportError:
        info["native"] = None
    return info


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=False).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith(("model name", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _campaign_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "run_benchmarks.py"], capture_output=True, text=True, check=False)
        return bool(out.stdout.strip())
    except OSError:
        return False


def _set_backend(name: str, threads: int) -> str:
    try:
        from pyfe4ai import backend as native_backend
    except ImportError:
        if name not in ("python", "auto"):
            raise SystemExit("this version has no native backend")
        return "python"
    native_backend.set_backend(name)
    native_backend.set_threads(threads)
    return native_backend.active_backend()


def _ct_bytes(layers: list) -> int:
    """Size of one client's ciphertexts: JSON, with any bytes values (binary
    ciphertext format) counted at their length, as in a binary container."""
    raw = []

    def default(obj):
        if isinstance(obj, (bytes, bytearray)):
            raw.append(len(obj))
            return ""
        return str(obj)

    total = 0
    for layer in layers:
        items = layer.ravel().tolist() if isinstance(layer, np.ndarray) else [layer]
        total += sum(len(json.dumps(ct, default=default).encode()) for ct in items)
    return total + sum(raw)


def _scheme_params(pp: dict) -> dict:
    out = {}
    for key in ("q", "p", "ring_n", "lwe_n", "n0", "m0"):
        if key in pp:
            try:
                out[key + ("_bits" if key in ("q", "p") else "")] = (
                    int(pp[key]).bit_length() if key in ("q", "p") else int(pp[key])
                )
            except (TypeError, ValueError):
                pass
    if "q_factors" in pp:  # an RNS modulus
        out["q_factor_bits"] = [int(f).bit_length() for f in pp["q_factors"]]
    return out


def measure(family: str, elements: int, clients: int, rounds: int, precision: int, bound_x: int,
            profile: str, ct_format: str = "json", offline: bool = False, extra: dict | None = None,
            weights: list | None = None) -> dict:
    from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    if offline:  # not in versions before the native backend
        from pyfe4ai.ml import precompute_gradient, prepare_aggregation

    key, base_cfg, _ = FAMILIES[family]
    nids = ["c{}".format(i) for i in range(clients)]
    weights = list(weights or [1] * clients)
    bounds = _bounds(family, bound_x, max(abs(w) for w in weights))
    cfg = dict(base_cfg, n=clients, lst_nid=nids, security_profile=profile, **bounds, **(extra or {}))
    row = {"family": family, "scheme": "/".join(key), "profile": profile, "elements": elements,
           "clients": clients, "rounds": rounds, "precision": precision, "bound_x": bound_x,
           "weights": weights, "ct_format": ct_format, **(extra or {})}
    t0 = time.perf_counter()
    options = {"crypto_options": {"ct_format": ct_format}} if ct_format != "json" else {}
    wrapper = FESchemeWrapper(*key, cfg, precision=precision, **options)
    row["setup_s"] = time.perf_counter() - t0
    row.update(_scheme_params(wrapper.pp))

    dk_times = []
    derive = wrapper.kg.get_decryption_keys

    def timed_derive(*a, **kw):
        t = time.perf_counter()
        out = derive(*a, **kw)
        dk_times.append(time.perf_counter() - t)
        return out

    wrapper.kg.get_decryption_keys = timed_derive
    rng = np.random.default_rng(7)
    lim = bound_x / 10 ** precision
    enc_t, agg_t, dk_t, enc_off, agg_off = [], [], [], [], []
    ct_size = None
    fusion = dict(zip(nids, weights))
    for r in range(rounds + 1):
        label = "round-{}".format(r)
        grads = {nid: rng.uniform(-lim, lim, elements).round(precision) for nid in nids}
        pre, prepared = {}, None
        if offline:  # before the round: client masks, then the aggregator's key
            t0 = time.perf_counter()
            pre = {nid: precompute_gradient((elements,), wrapper, nid=nid, label=label) for nid in nids}
            t1 = time.perf_counter()
            prepared = prepare_aggregation(wrapper, fusion, shape=(elements,), label=label)
            t2 = time.perf_counter()
            if r:
                enc_off.append((t1 - t0) / clients)
                agg_off.append(t2 - t1)
        dk_times.clear()
        t0 = time.perf_counter()
        enc = {nid: encrypt_gradient(g, wrapper, nid=nid, label=label, **({"precomputed": pre[nid]} if pre else {}))
               for nid, g in grads.items()}
        t1 = time.perf_counter()
        if offline:
            total = aggregate_gradients(enc, wrapper, label=label, prepared=prepared)
        else:
            total = aggregate_gradients(enc, wrapper, fusion, label=label)
        t2 = time.perf_counter()
        expected = sum(w * grads[nid] for nid, w in zip(nids, weights))
        if not np.allclose(np.asarray(total, dtype=float), expected, atol=10 ** -precision / 2):
            raise AssertionError("{}: wrong aggregate".format(family))
        if r == 0:
            ct_size = _ct_bytes(next(iter(enc.values())).ciphertexts) / elements
            continue
        enc_t.append((t1 - t0) / clients)
        agg_t.append(t2 - t1)
        dk_t.append(sum(dk_times))
    med = statistics.median
    row.update(
        client_ms=med(enc_t) * 1e3,
        aggregator_ms=med(agg_t) * 1e3,
        dkgen_ms=med(dk_t) * 1e3,
        decrypt_ms=med([a - d for a, d in zip(agg_t, dk_t)]) * 1e3,
        client_us_per_element=med(enc_t) * 1e6 / elements,
        aggregator_us_per_element=med(agg_t) * 1e6 / elements,
        client_ms_min=min(enc_t) * 1e3,
        aggregator_ms_min=min(agg_t) * 1e3,
        ct_bytes_per_element=round(ct_size, 1),
        offline=offline,
    )
    if offline:
        row.update(client_offline_ms=med(enc_off) * 1e3, aggregator_offline_ms=med(agg_off) * 1e3)
    return row


def measure_threshold(family: str, elements: int, clients: int, rounds: int, precision: int, bound_x: int,
                      profile: str, servers: int, threshold: int, extra: dict | None = None,
                      weights: list | None = None) -> dict:
    """:func:`measure` for a threshold scheme, on its classes (see the module docstring)."""
    from pyfe4ai.core.registry import get_scheme

    key, base_cfg, _ = FAMILIES[family]
    scheme = get_scheme(key)
    crypto_cls = scheme.crypto
    nids = ["c{}".format(i) for i in range(clients)]
    sids = ["s{}".format(i) for i in range(servers)]
    weights = list(weights or [1] * clients)
    bounds = _bounds(family, bound_x, max(abs(w) for w in weights))
    cfg = dict(base_cfg, n=clients, lst_nid=nids, s=servers, t=threshold, lst_sid=sids,
               security_profile=profile, **bounds, **(extra or {}))
    row = {"family": family, "scheme": "/".join(key), "profile": profile, "elements": elements,
           "clients": clients, "servers": servers, "threshold": threshold, "rounds": rounds,
           "precision": precision, "bound_x": bound_x, "weights": weights, **(extra or {})}
    t0 = time.perf_counter()
    kg = scheme.key_generator(cfg)
    kg.setup()
    row["setup_s"] = time.perf_counter() - t0
    pp = kg.get_public_parameters()
    split = getattr(kg, "key_shares", "full") == "split"
    row.update(ct_format=pp.get("ct_format", "json"), ring_modulus="rns" if pp.get("q_factors") else "prime",
               key_shares="split" if split else "full", split_key_shares=split)
    row.update(_scheme_params(pp))
    cls = {nid: crypto_cls({"id": nid, "precision": precision, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           for nid in nids}
    enrolled = sids[:threshold]
    budget = {}
    if pp.get("flooding_accounting") == "kl":  # one budget per server, kept by the deployment
        from pyfe4ai.schemes.threshold.kl import PartialDecryptionBudget

        budget = {sid: {"partial_budget": PartialDecryptionBudget(pp["max_partial_decryptions"])} for sid in enrolled}
    srv = {sid: crypto_cls({"id": sid, "precision": precision, "keys": {"pp": pp}, **budget.get(sid, {})})
           for sid in enrolled}
    combiner = crypto_cls({"id": "combiner", "precision": precision, "keys": {"pp": pp}})
    fusion = {nid: [w] for nid, w in zip(nids, weights)}
    if split:  # the label-independent part of the key shares, once
        t0 = time.perf_counter()
        for sid in enrolled:
            srv[sid].add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": fusion}))
        row["weight_shares_ms"] = (time.perf_counter() - t0) * 1e3
    rng = np.random.default_rng(7)
    lim = bound_x / 10 ** precision
    enc_t, dk_t, part_t, comb_t, agg_t = [], [], [], [], []
    ct_size = None
    for r in range(rounds + 1):
        label = "round-{}".format(r)
        grads = {nid: rng.uniform(-lim, lim, elements).round(precision) for nid in nids}
        t0 = time.perf_counter()
        enc = {nid: cls[nid].encrypt_lst_ndarray([g], label=label) for nid, g in grads.items()}
        t1 = time.perf_counter()
        creds = {"fusion_weight": fusion, "label": label, "n_chunks": cls[nids[0]].key_chunks(enc[nids[0]])}
        if split:  # the label part (the default key_part): chunk keys only
            creds.update(label_term=False)
        dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in enrolled}
        t2 = time.perf_counter()
        parts, one = {}, []
        for sid in enrolled:
            ts = time.perf_counter()
            parts[sid] = srv[sid].compute_lst_ndarray_ct(enc, creds, dks[sid], enrolled)
            one.append(time.perf_counter() - ts)
        t3 = time.perf_counter()
        total = combiner.decrypt_lst_ndarray_ct(parts)[0]
        t4 = time.perf_counter()
        expected = sum(w * grads[nid] for nid, w in zip(nids, weights))
        if not np.allclose(np.asarray(total, dtype=float), expected, atol=10 ** -precision / 2):
            raise AssertionError("{}: wrong aggregate".format(family))
        if r == 0:
            ct_size = _ct_bytes(enc[nids[0]]) / elements
            continue
        enc_t.append((t1 - t0) / clients)
        dk_t.append(t2 - t1)
        part_t.append(statistics.median(one))
        comb_t.append(t4 - t3)
        agg_t.append(t4 - t1)
    med = statistics.median
    row.update(
        client_ms=med(enc_t) * 1e3,
        aggregator_ms=med(agg_t) * 1e3,
        dkgen_ms=med(dk_t) * 1e3,
        partial_ms=med(part_t) * 1e3,
        combine_ms=med(comb_t) * 1e3,
        decrypt_ms=med([a - d for a, d in zip(agg_t, dk_t)]) * 1e3,
        client_us_per_element=med(enc_t) * 1e6 / elements,
        aggregator_us_per_element=med(agg_t) * 1e6 / elements,
        client_ms_min=min(enc_t) * 1e3,
        aggregator_ms_min=min(agg_t) * 1e3,
        ct_bytes_per_element=round(ct_size, 1),
        offline=False,
    )
    return row


def measure_decentralized(family: str, elements: int, clients: int, rounds: int, precision: int, bound_x: int,
                          profile: str, ct_format: str = "json", extra: dict | None = None,
                          weights: list | None = None) -> dict:
    """:func:`measure` for a decentralized scheme: every client derives a key
    share for the round, the aggregator combines them and decrypts the sum."""
    from pyfe4ai.core.registry import get_scheme

    key, base_cfg, _ = FAMILIES[family]
    scheme = get_scheme(key)
    crypto_cls = scheme.crypto
    nids = ["c{}".format(i) for i in range(clients)]
    weights = list(weights or [1] * clients)
    bounds = _bounds(family, bound_x, max(abs(w) for w in weights))
    cfg = dict(base_cfg, n=clients, lst_nid=nids, security_profile=profile, **bounds, **(extra or {}))
    row = {"family": family, "scheme": "/".join(key), "profile": profile, "elements": elements,
           "clients": clients, "rounds": rounds, "precision": precision, "bound_x": bound_x, "weights": weights,
           "ct_format": ct_format, **(extra or {})}
    t0 = time.perf_counter()
    kg = scheme.key_generator(cfg)
    kg.setup()
    row["setup_s"] = time.perf_counter() - t0
    pp = kg.get_public_parameters()
    row.update(_scheme_params(pp))
    cls = {nid: crypto_cls({"id": nid, "precision": precision, "ct_format": ct_format,
                            "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in nids}
    agg = crypto_cls({"id": "agg", "precision": precision, "ct_format": ct_format, "keys": {"pp": pp}})
    fusion = {nid: [w] for nid, w in zip(nids, weights)}
    rng = np.random.default_rng(7)
    lim = bound_x / 10 ** precision
    enc_t, share_t, comb_t, dec_t = [], [], [], []
    ct_size = None
    for r in range(rounds + 1):
        label = "round-{}".format(r)
        grads = {nid: rng.uniform(-lim, lim, elements).round(precision) for nid in nids}
        t0 = time.perf_counter()
        enc = {nid: cls[nid].encrypt_lst_ndarray([g], label=label) for nid, g in grads.items()}
        t1 = time.perf_counter()
        n_chunks = crypto_cls.key_chunks(enc[nids[0]])
        shares = {nid: cls[nid].derive_function_decryption_key_share(fusion, label, n_chunks) for nid in nids}
        t2 = time.perf_counter()
        dk = agg.combine_function_decryption_key_share(shares)
        t3 = time.perf_counter()
        total = agg.compute_lst_ndarray_ct(enc, dk=dk, fusion_weight=fusion, label=label)[0]
        t4 = time.perf_counter()
        expected = sum(w * grads[nid] for nid, w in zip(nids, weights))
        if not np.allclose(np.asarray(total, dtype=float), expected, atol=10 ** -precision / 2):
            raise AssertionError("{}: wrong aggregate".format(family))
        if r == 0:
            ct_size = _ct_bytes(enc[nids[0]]) / elements
            continue
        enc_t.append((t1 - t0) / clients)
        share_t.append((t2 - t1) / clients)
        comb_t.append(t3 - t2)
        dec_t.append(t4 - t3)
    med = statistics.median
    row.update(
        client_ms=med(enc_t) * 1e3,
        key_share_ms=med(share_t) * 1e3,
        combine_ms=med(comb_t) * 1e3,
        aggregator_ms=(med(comb_t) + med(dec_t)) * 1e3,
        dkgen_ms=med(comb_t) * 1e3,
        decrypt_ms=med(dec_t) * 1e3,
        client_us_per_element=med(enc_t) * 1e6 / elements,
        aggregator_us_per_element=(med(comb_t) + med(dec_t)) * 1e6 / elements,
        ct_bytes_per_element=round(ct_size, 1),
        offline=False,
    )
    return row


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--families", nargs="+", default=["ring_lwe"], choices=sorted(FAMILIES))
    parser.add_argument("--elements", type=int, default=None, help="per family default if omitted")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--servers", type=int, default=3, help="threshold family: servers s")
    parser.add_argument("--threshold", type=int, default=2, help="threshold family: threshold t")
    parser.add_argument("--flooding-accounting", default=None, choices=["per-sample", "kl"],
                        help="threshold family: accounting of the flooding noise (default: the scheme's)")
    parser.add_argument("--max-partial-decryptions", type=int, default=None,
                        help="threshold family: partial decryptions per server under --flooding-accounting kl")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--weights", default=None,
                        help="comma-separated integer weights, one per client (default: all 1)")
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--bound-x", type=int, default=None, help="default 10^precision (values in [-1, 1])")
    parser.add_argument("--profile", default="standard", choices=["standard", "toy"])
    parser.add_argument("--backend", default="auto", choices=["auto", "python", "native"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ct-format", default=None, choices=["json", "binary"],
                        help="ciphertext format of the Ring-LWE schemes (default: json; ring_lwe_threshold: "
                             "the scheme's, binary)")
    parser.add_argument("--offline", action="store_true",
                        help="Ring-LWE, DDH, Damgard and Paillier MCFE: precompute client masks and the "
                             "aggregator's key before each round; client_ms / aggregator_ms are then the online phase")
    parser.add_argument("--label-hash", default=None,
                        help="label_hash of the MCFE Ring-LWE / dense-LWE families (default: the scheme's)")
    parser.add_argument("--flooding", default=None, help="flooding of ring_lwe_threshold (default: the scheme's)")
    parser.add_argument("--ring-modulus", default=None, choices=["prime", "rns"],
                        help="ring_lwe / ring_lwe_decentralized: q an NTT prime (default) or a product of two "
                             "(opt-in); ring_lwe_threshold: the scheme's (rns, three or four factors)")
    parser.add_argument("--key-shares", default=None, choices=["split", "full"],
                        help="ring_lwe_threshold: weight key shares once and label parts per round (the "
                             "scheme's default for the Shamir and replicated sharings) or whole key shares")
    parser.add_argument("--split-key-shares", action="store_const", const="split", dest="key_shares",
                        help="the same as --key-shares split")
    parser.add_argument("--workdir", default=None, help="directory holding the parameter cache config/ (default: temporary)")
    parser.add_argument("--json", default=None, help="append the results to this JSON file")
    parser.add_argument("--tag", default="", help="free-form label stored with every row")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    bound_x = args.bound_x or 10 ** args.precision
    weights = [int(w) for w in args.weights.split(",")] if args.weights else None
    if weights is not None and len(weights) != args.clients:
        parser.error("--weights needs one weight per client ({} clients)".format(args.clients))

    backend = _set_backend(args.backend, args.threads)
    env = _env()
    rows = []
    cwd = os.getcwd()
    tmp = None
    if args.workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = tmp.name
    else:
        workdir = args.workdir
        os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    cache = mock.patch.dict(os.environ, {"PYFE4AI_CACHE_DIR": os.path.join(os.getcwd(), "config")})
    cache.start()
    try:
        for family in args.families:
            elements = args.elements or FAMILIES[family][2]
            extra = {}
            if args.label_hash and family in ("ring_lwe", "ring_lwe_threshold", "ring_lwe_decentralized", "lwe"):
                extra["label_hash"] = args.label_hash
            if args.flooding and family == "ring_lwe_threshold":
                extra["flooding"] = args.flooding
            if args.ring_modulus == "rns" and family in ("ring_lwe", "ring_lwe_decentralized"):
                extra["ring_modulus"] = args.ring_modulus  # rows and configs of the default stay as before
            if family == "ring_lwe_threshold":  # options of the key generator (the scheme's defaults if not given)
                for name, value in (("ring_modulus", args.ring_modulus), ("ct_format", args.ct_format),
                                    ("key_shares", args.key_shares)):
                    if value is not None:
                        extra[name] = value
            if family.endswith("_threshold"):
                if args.flooding_accounting is not None:
                    extra["flooding_accounting"] = args.flooding_accounting
                if args.max_partial_decryptions is not None:
                    extra["max_partial_decryptions"] = args.max_partial_decryptions
            fmt = (args.ct_format or "json") if family in ("ring_lwe", "ring_lwe_decentralized") else "json"
            if family.endswith("_decentralized"):
                row = measure_decentralized(family, elements, args.clients, args.rounds, args.precision, bound_x,
                                            args.profile, fmt, extra, weights)
            elif family.endswith("_threshold"):
                row = measure_threshold(family, elements, args.clients, args.rounds, args.precision, bound_x,
                                        args.profile, args.servers, args.threshold, extra, weights)
            else:
                row = measure(family, elements, args.clients, args.rounds, args.precision, bound_x, args.profile,
                              fmt, args.offline and family in OFFLINE_FAMILIES, extra, weights)
            row.update(backend=backend, threads=args.threads, tag=args.tag)
            rows.append(row)
            print("{family:9s} {backend:6s} t={threads} {elements:6d} el  client {client_ms:9.2f} ms  "
                  "aggregator {aggregator_ms:9.2f} ms (dkgen {dkgen_ms:7.2f})  ct {ct_bytes_per_element} "
                  "B/el {ct_format}".format(**row)
                  + ("  [offline: client {client_offline_ms:.2f} ms, aggregator {aggregator_offline_ms:.2f} ms]"
                     .format(**row) if row.get("offline") else "")
                  + ("  [partial {partial_ms:.2f} ms per server, combine {combine_ms:.2f} ms]".format(**row)
                     if "partial_ms" in row else "")
                  + ("  [key share {key_share_ms:.2f} ms per client, combine {combine_ms:.2f} ms]".format(**row)
                     if "key_share_ms" in row else ""), flush=True)
    finally:
        cache.stop()
        os.chdir(cwd)
        if tmp is not None:
            tmp.cleanup()
    if args.json:
        doc = {"runs": []}
        if os.path.exists(args.json):
            with open(args.json, encoding="utf-8") as f:
                doc = json.load(f)
        doc["runs"].append({"env": env, "args": vars(args), "rows": rows,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=str)

