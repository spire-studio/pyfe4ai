"""Federated gradient aggregation with the threshold and decentralized MCFE schemes.

``run.py fl-aggregation`` covers the registry entries whose aggregation
runs through one :class:`FESchemeWrapper`; the multi-party MCFE schemes timed
here need several decryption servers (threshold) or the clients' key shares
(decentralized), so this script drives their scheme classes directly. Every
client of the scheme's test configuration (see ``run.py fl-coverage``)
encrypts a gradient of ``--elements`` values in ``[-1, 1]`` (quantised with
``--precision`` decimal digits) with ``encrypt_lst_ndarray()`` under a new
label per round, and the aggregate with weight 1 for every client is checked
against the plaintext sum. Per round:

* threshold (``s = 3`` servers, threshold ``t = 2``, the first ``t`` servers
  enrolled): ``keygen`` is the ``t`` servers' ``get_decryption_keys()``,
  ``partial`` one server's ``compute_lst_ndarray_ct()`` and ``combine`` the
  combiner's ``decrypt_lst_ndarray_ct()``. Key shares that do not depend on
  the label (DDH, LWE, FH) are cached by the key generator per weight
  vector, so after the first round their ``keygen`` is a lookup; Ring-LWE
  keys are bound to the label and derived every round. With split key shares
  (the default of the threshold Ring-LWE MCFE) the servers keep their weight
  part, derived once per setup (``weight_keygen``), and ``keygen`` is the
  label part;
* decentralized: ``keygen`` is every client's
  ``derive_function_decryption_key_share()`` (FH under a fresh function tag
  per round) plus the combiner's ``combine_function_decryption_key_share()``,
  and ``combine`` the final ``compute_lst_ndarray_ct()``.

Key generator construction and setup (parameter generation included) are
timed once per scheme, in a fresh temporary directory; building the
clients', servers' and combiner's crypto objects is not timed.

Usage::

    python benchmarks/run.py fl-multiparty --elements 1000 --security-profile standard
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import numpy as np

from _common import compact_bytes
from _common import fresh_cache
from _common import keygen_config
from _common import parse_args
from _common import prepare_benchmark
from _common import size_bytes
from _common import stats
from _common import with_clients
from commands.fl_coverage import CONFIGS
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.registry import get_scheme

#: Multi-party MCFE registry entries (threshold first, then decentralized).
SCHEMES = [
    ("mcfe", "ddh_threshold"), ("mcfe", "lwe_threshold"), ("mcfe", "ring_lwe_threshold"),
    ("mcfe", "fh_multi_ipe_pairing_threshold"),
    ("mcfe", "ddh_decentralized"), ("mcfe", "lwe_decentralized"), ("mcfe", "ring_lwe_decentralized"),
    ("mcfe", "fh_multi_ipe_pairing_decentralized"), ("mcfe", "paillier_decentralized"),
]

HEADER = ["scheme", "security_profile", "status", "elements", "clients", "servers", "rounds", "setup_sec",
          "encrypt_mean_sec_per_client", "weight_keygen_sec", "keygen_mean_sec", "partial_mean_sec_per_server",
          "combine_mean_sec", "round_std_sec", "ct_bytes_per_element", "ct_compact_bytes_per_element", "note"]

#: Threshold schemes whose partial decryption and combination take the label
#: as an argument (the LWE ones read it from the credentials / partials).
_LABEL_ARG = {"ddh_threshold", "fh_multi_ipe_pairing_threshold"}


def _config(key, precision: int) -> dict:
    cfg = dict(CONFIGS[key], eta=1)
    bound = 10 ** precision
    for name in ("bound_x", "bound"):
        if name in cfg:
            cfg[name] = max(int(cfg[name]), bound)
    if "bound_y" in cfg:
        cfg["bound_y"] = max(int(cfg["bound_y"]), 2)
    return cfg


def _ct_bytes(layers: list) -> int:
    """JSON size of one client's ciphertexts (per element, or packed layers)."""
    total = 0
    for layer in layers:
        if isinstance(layer, np.ndarray):
            total += sum(size_bytes(ct) for ct in layer.ravel())
        else:
            total += size_bytes(layer)
    return total


def _classes(key) -> tuple:
    scheme = get_scheme(key)
    return scheme.key_generator, scheme.crypto


def _threshold_round(variant, kg, crypto, combiner, dct_ct, weights, label, enrolled, servers=None) -> tuple:
    """Key shares, partial decryptions and combination of one threshold round
    (*servers*: the servers that hold their weight key shares, if split)."""
    credentials = {"fusion_weight": weights, "label": label}
    if variant == "ring_lwe_threshold":  # the key share must cover the packed chunks
        credentials["n_chunks"] = combiner.key_chunks(next(iter(dct_ct.values())))
    if servers:  # split key shares: the label part only
        credentials["label_term"] = False
    label_arg = (label,) if variant in _LABEL_ARG else ()
    # the DDH servers check the bare weight mapping, the others the credentials
    server_credentials = weights if variant == "ddh_threshold" else credentials
    keygen, partial_times, partials = 0.0, [], {}
    for sid in enrolled:
        t0 = time.perf_counter()
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        keygen += time.perf_counter() - t0
        server = servers[sid] if servers else crypto(sid, dk=dk)
        t0 = time.perf_counter()
        partials[sid] = server.compute_lst_ndarray_ct(dct_ct, server_credentials, dk, enrolled, *label_arg)
        partial_times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    result = combiner.decrypt_lst_ndarray_ct(partials, *label_arg)
    return result, keygen, partial_times, time.perf_counter() - t0


def _decentralized_round(variant, clients, combiner, dct_ct, weights, label) -> tuple:
    """Clients' key shares, their combination and the decryption of one round."""
    if variant == "ring_lwe_decentralized":  # identity-based: key for (label, chunks)
        n_chunks = combiner.key_chunks(next(iter(dct_ct.values())))
        args = (weights, label, n_chunks)
    elif variant == "fh_multi_ipe_pairing_decentralized":  # one function tag per key
        args = (weights, "fedavg-" + label)
    else:
        args = (weights,)
    t0 = time.perf_counter()
    shares = {nid: c.derive_function_decryption_key_share(*args) for nid, c in clients.items()}
    dk = combiner.combine_function_decryption_key_share(shares)
    t1 = time.perf_counter()
    result = combiner.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=weights, label=label)
    return result, t1 - t0, [], time.perf_counter() - t1


def measure(key, elements: int, precision: int, rounds: int, profile: str, clients: int | None = None) -> dict:
    variant = key[1]
    threshold = variant.endswith("_threshold")
    cfg = with_clients(_config(key, precision), clients)
    nids = cfg["lst_nid"]
    enrolled = cfg["lst_sid"][: cfg["t"]] if threshold else []
    row = {"scheme": "/".join(key), "security_profile": profile, "elements": elements,
           "clients": len(nids), "servers": len(enrolled) if threshold else "", "rounds": rounds}
    rng = np.random.default_rng(7)
    with fresh_cache():
        try:
            kg_cls, crypto_cls = _classes(key)
            t0 = time.perf_counter()
            kg = kg_cls(keygen_config(cfg))
            kg.setup()
        except FEValidationError as exc:
            if profile == "toy" or "parameter set" not in str(exc):
                raise
            return dict(row, status="refused", note=str(exc).splitlines()[0][:200])
        except ImportError as exc:  # pairing-based scheme without a backend for its group
            if "charm" not in str(exc) and "pairing" not in str(exc):
                raise
            return dict(row, status="unverified", note=str(exc).splitlines()[0][:200])
        setup = time.perf_counter() - t0
        pp = kg.get_public_parameters()

        def crypto(actor_id, **keys):
            return crypto_cls({"id": actor_id, "precision": precision, "keys": dict(keys, pp=pp)})

        clients = {nid: crypto(nid, sk=kg.get_private_keys(nid)) for nid in nids}
        combiner = crypto("combiner")
        weights = {nid: [1] for nid in nids}
        servers, weight_keygen = None, ""
        if threshold and getattr(kg, "key_shares", "full") == "split":
            servers = {sid: crypto(sid) for sid in enrolled}
            t0 = time.perf_counter()
            for sid in enrolled:
                servers[sid].add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": weights}))
            weight_keygen = time.perf_counter() - t0
        enc_times, keygen_times, partial_times, combine_times, totals, ct_bytes = [], [], [], [], [], 0
        for r in range(rounds):
            label = "round-{}".format(r)
            grads = {nid: rng.uniform(-1, 1, elements).round(precision) for nid in nids}
            t0 = time.perf_counter()
            dct_ct = {nid: clients[nid].encrypt_lst_ndarray([g], label=label) for nid, g in grads.items()}
            enc = time.perf_counter() - t0
            if threshold:
                result, keygen, partials, combine = _threshold_round(
                    variant, kg, crypto, combiner, dct_ct, weights, label, enrolled, servers)
            else:
                result, keygen, partials, combine = _decentralized_round(
                    variant, clients, combiner, dct_ct, weights, label)
            if not np.allclose(np.asarray(result[0], dtype=float), sum(grads.values()), atol=10 ** -precision):
                raise AssertionError("{}: wrong aggregate".format(row["scheme"]))
            enc_times.append(enc / len(nids))
            keygen_times.append(keygen)
            if partials:
                partial_times.append(sum(partials) / len(partials))
            combine_times.append(combine)
            totals.append(enc + keygen + sum(partials) + combine)
            ct_bytes = _ct_bytes(dct_ct[nids[0]]) / elements
            ct_compact = compact_bytes(dct_ct[nids[0]]) / elements
    return dict(
        row, status="ok", setup_sec=setup,
        encrypt_mean_sec_per_client=stats(enc_times)[0], weight_keygen_sec=weight_keygen,
        keygen_mean_sec=stats(keygen_times)[0],
        partial_mean_sec_per_server=stats(partial_times)[0] if threshold else "",
        combine_mean_sec=stats(combine_times)[0], round_std_sec=stats(totals)[2],
        ct_bytes_per_element=round(ct_bytes, 1),
        ct_compact_bytes_per_element=round(ct_compact, 1), note="",
    )


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--elements", type=int, default=1000)
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=None,
                        help="number of clients of the multi-client schemes (default: the test configuration's three)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--schemes", nargs="*", help="family/variant entries (default: all nine)")
    args = parse_args(parser, argv)
    prepare_benchmark(args.seed)
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    keys = [k for k in SCHEMES if not args.schemes or "/".join(k) in args.schemes]
    rows = [measure(k, args.elements, args.precision, args.rounds, args.security_profile, args.clients) for k in keys]
    if args.output == "json":
        print(json.dumps(rows, indent=2))
        return
    print(",".join(HEADER))
    for r in rows:
        print(",".join(
            "{:.4f}".format(r[h]) if isinstance(r.get(h), float) else str(r.get(h, "")).replace(",", ";")
            for h in HEADER
        ))

