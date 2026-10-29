"""Federated gradient aggregation through the ML adapter.

For every registry entry whose FL path is usable (see ``run.py fl-coverage``)
this script builds a :class:`FESchemeWrapper`, lets each client encrypt a
gradient of ``--elements`` values in ``[-1, 1]`` (quantised with
``--precision`` decimal digits) with ``encrypt_gradient()``, and aggregates
them with weight 1 with ``aggregate_gradients()``, checking the result
against the plaintext sum; the functional-key derivation inside it is timed
separately (``dkgen``), the rest is decryption. Each round uses a
new label. Wrapper construction (parameter generation and setup) is timed
once, in a fresh temporary directory.

Usage::

    python benchmarks/run.py fl-aggregation --elements 1000 --security-profile standard
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
from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

#: Registry entries whose gradient aggregation runs through one wrapper.
SCHEMES = [
    ("sife", "lwe"), ("sife", "paillier"), ("sife", "damgard"), ("sife", "fullysec_ring_lwe"),
    ("sife", "fullysec_lwe"),
    ("mcfe", "ddh"), ("mcfe", "ddh_ristretto255"), ("mcfe", "damgard"), ("mcfe", "paillier"), ("mcfe", "lwe"),
    ("mcfe", "fullysec_lwe"), ("mcfe", "ring_lwe"), ("mcfe", "fh_multi_ipe_pairing"),
    # the wrapper also runs the s decryption servers of the threshold Ring-LWE MCFE
    ("mcfe", "ring_lwe_threshold"),
]

HEADER = ["scheme", "security_profile", "status", "elements", "clients", "rounds", "setup_sec",
          "encrypt_mean_sec_per_client", "aggregate_mean_sec", "dkgen_mean_sec", "decrypt_mean_sec",
          "round_std_sec", "ct_bytes_per_element", "ct_compact_bytes_per_element", "note"]


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


def measure(key, elements: int, precision: int, rounds: int, profile: str, clients: int | None = None) -> dict:
    family, variant = key
    row = {"scheme": "/".join(key), "security_profile": profile, "elements": elements, "rounds": rounds}
    cfg = with_clients(_config(key, precision), clients)
    nids = cfg["lst_nid"] if family == "mcfe" else ["nid_default"]
    rng = np.random.default_rng(7)
    with fresh_cache():
        t0 = time.perf_counter()
        try:
            wrapper = FESchemeWrapper(family, variant, keygen_config(cfg), precision=precision)
        except FEValidationError as exc:
            if profile == "toy" or "parameter set" not in str(exc):
                raise
            return dict(row, status="refused", clients=len(nids), note=str(exc).splitlines()[0][:200])
        setup = time.perf_counter() - t0
        # time the functional-key derivation inside aggregate_gradients
        dk_times = []
        derive = wrapper.kg.get_decryption_keys

        def timed_derive(*a, **kw):
            t = time.perf_counter()
            out = derive(*a, **kw)
            dk_times.append(time.perf_counter() - t)
            return out

        wrapper.kg.get_decryption_keys = timed_derive
        enc_times, agg_times, totals, ct_bytes = [], [], [], 0
        for r in range(rounds):
            label = "round-{}".format(r) if family == "mcfe" else None
            grads = {nid: rng.uniform(-1, 1, elements).round(precision) for nid in nids}
            t0 = time.perf_counter()
            enc = {nid: encrypt_gradient(g, wrapper, nid=nid, label=label) for nid, g in grads.items()}
            t1 = time.perf_counter()
            result = aggregate_gradients(enc, wrapper, {nid: 1 for nid in nids}, label=label)
            t2 = time.perf_counter()
            if not np.allclose(np.asarray(result, dtype=float), sum(grads.values()), atol=10 ** -precision):
                raise AssertionError("{}: wrong aggregate".format(row["scheme"]))
            enc_times.append((t1 - t0) / len(nids))
            agg_times.append(t2 - t1)
            totals.append(t2 - t0)
            ct_bytes = _ct_bytes(next(iter(enc.values())).ciphertexts) / elements
            ct_compact = compact_bytes(next(iter(enc.values())).ciphertexts) / elements
    return dict(
        row, status="ok", clients=len(nids), setup_sec=setup,
        encrypt_mean_sec_per_client=stats(enc_times)[0], aggregate_mean_sec=stats(agg_times)[0],
        dkgen_mean_sec=stats(dk_times)[0],
        decrypt_mean_sec=stats([a - d for a, d in zip(agg_times, dk_times)])[0],
        round_std_sec=stats(totals)[2], ct_bytes_per_element=round(ct_bytes, 1),
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
    parser.add_argument("--schemes", nargs="*", help="family/variant entries (default: all usable)")
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

