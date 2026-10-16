"""Federated gradient aggregation per scheme family, for before/after comparisons.

Every client encrypts a gradient of ``--elements`` values in ``[-1, 1]`` with
``--precision`` decimal digits through :func:`~pyfe4ai.utils.ml_adapter.encrypt_gradient`;
the aggregator derives the functional key for the round label and decrypts the
sum (weight 1) with :func:`~pyfe4ai.utils.ml_adapter.aggregate_gradients`. Each
round uses a fresh label; round 0 warms the caches and is not counted. The
result is checked against the plaintext sum.

Reported per family (medians over the counted rounds):

* ``client_ms``: one client's ``encrypt_gradient``;
* ``aggregator_ms``: ``aggregate_gradients`` (key derivation + decryption),
  with ``dkgen_ms`` the key derivation inside it;
* ``ct_bytes_per_element``: one client's ciphertexts as JSON (bytes values
  of the binary ciphertext format, ``--ct-format binary``, counted at their
  length), per element.

The schemes write their parameter cache to ``./config``; the run happens in
``--workdir`` (default: a temporary directory) so a cache can be reused across
runs. Usage::

    python benchmarks/bench_fl_compare.py --families ring_lwe --elements 10000 --json out.json
    python benchmarks/bench_fl_compare.py --families ring_lwe --backend native --threads 4
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

import numpy as np

#: family -> (registry key, keygen config without bounds, default element count)
FAMILIES = {
    "ring_lwe": (("mcfe", "ring_lwe"), {"eta": 1}, 10_000),
    "lwe": (("mcfe", "lwe"), {"eta": 1}, 200),
    "ddh": (("mcfe", "ddh"), {"eta": 1}, 20),
    "damgard": (("mcfe", "damgard"), {"eta": 1}, 20),
    "paillier": (("mcfe", "paillier"), {"eta": 1}, 20),
    "pairing": (("mcfe", "fh_multi_ipe_pairing"), {"eta": 1}, 5),
}


def _bounds(family: str, bound_x: int) -> dict:
    if family in ("damgard",):
        return {"bound": bound_x}
    return {"bound_x": bound_x, "bound_y": 1}


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
        from pyfe4ai.utils import native_backend

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
        from pyfe4ai.utils import native_backend
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


def _scheme_params(wrapper) -> dict:
    pp = wrapper.pp
    out = {}
    for key in ("q", "p", "ring_n", "lwe_n", "n0", "m0"):
        if key in pp:
            try:
                out[key + ("_bits" if key in ("q", "p") else "")] = (
                    int(pp[key]).bit_length() if key in ("q", "p") else int(pp[key])
                )
            except (TypeError, ValueError):
                pass
    return out


def measure(family: str, elements: int, clients: int, rounds: int, precision: int, bound_x: int,
            profile: str, ct_format: str = "json", offline: bool = False) -> dict:
    from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    if offline:  # not in versions before the native backend
        from pyfe4ai.utils.ml_adapter import precompute_gradient, prepare_aggregation

    key, base_cfg, _ = FAMILIES[family]
    nids = ["c{}".format(i) for i in range(clients)]
    cfg = dict(base_cfg, n=clients, lst_nid=nids, security_profile=profile, **_bounds(family, bound_x))
    row = {"family": family, "scheme": "/".join(key), "profile": profile, "elements": elements,
           "clients": clients, "rounds": rounds, "precision": precision, "bound_x": bound_x,
           "ct_format": ct_format}
    t0 = time.perf_counter()
    options = {"crypto_options": {"ct_format": ct_format}} if ct_format != "json" else {}
    wrapper = FESchemeWrapper(*key, cfg, precision=precision, **options)
    row["setup_s"] = time.perf_counter() - t0
    row.update(_scheme_params(wrapper))

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
    weights = {nid: 1 for nid in nids}
    for r in range(rounds + 1):
        label = "round-{}".format(r)
        grads = {nid: rng.uniform(-lim, lim, elements).round(precision) for nid in nids}
        pre, prepared = {}, None
        if offline:  # before the round: client masks, then the aggregator's key
            t0 = time.perf_counter()
            pre = {nid: precompute_gradient((elements,), wrapper, nid=nid, label=label) for nid in nids}
            t1 = time.perf_counter()
            prepared = prepare_aggregation(wrapper, weights, shape=(elements,), label=label)
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
            total = aggregate_gradients(enc, wrapper, weights, label=label)
        t2 = time.perf_counter()
        if not np.allclose(np.asarray(total, dtype=float), sum(grads.values()), atol=10 ** -precision / 2):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", nargs="+", default=["ring_lwe"], choices=sorted(FAMILIES))
    parser.add_argument("--elements", type=int, default=None, help="per family default if omitted")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--bound-x", type=int, default=None, help="default 10^precision (values in [-1, 1])")
    parser.add_argument("--profile", default="standard", choices=["standard", "toy"])
    parser.add_argument("--backend", default="auto", choices=["auto", "python", "native"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ct-format", default="json", choices=["json", "binary"],
                        help="ciphertext format of the Ring-LWE schemes")
    parser.add_argument("--offline", action="store_true",
                        help="Ring-LWE MCFE: precompute client masks and the aggregator's key before each "
                             "round; client_ms / aggregator_ms are then the online phase")
    parser.add_argument("--workdir", default=None, help="directory holding ./config (default: temporary)")
    parser.add_argument("--json", default=None, help="append the results to this JSON file")
    parser.add_argument("--tag", default="", help="free-form label stored with every row")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    bound_x = args.bound_x or 10 ** args.precision

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
    try:
        for family in args.families:
            elements = args.elements or FAMILIES[family][2]
            fmt = args.ct_format if family == "ring_lwe" else "json"
            row = measure(family, elements, args.clients, args.rounds, args.precision, bound_x, args.profile, fmt,
                          args.offline and family == "ring_lwe")
            row.update(backend=backend, threads=args.threads, tag=args.tag)
            rows.append(row)
            print("{family:9s} {backend:6s} t={threads} {elements:6d} el  client {client_ms:9.2f} ms  "
                  "aggregator {aggregator_ms:9.2f} ms (dkgen {dkgen_ms:7.2f})  ct {ct_bytes_per_element} "
                  "B/el {ct_format}".format(**row)
                  + ("  [offline: client {client_offline_ms:.2f} ms, aggregator {aggregator_offline_ms:.2f} ms]"
                     .format(**row) if row.get("offline") else ""), flush=True)
    finally:
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


if __name__ == "__main__":
    main()
