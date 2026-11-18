"""Sizes and times of one slot of ``bcfe/gc_ristretto255`` (0.6 P1), per circuit.

For each library circuit and data-wire mode: the gate statistics of
``guard(C)``, the sizes of a slot's public-key table and of one ciphertext
component (raw bytes, before any record encoding), and the median time of
the backend operations a slot needs -- ``bcfe_new_slot`` (issuing, h1),
``bcfe_encrypt_slot`` (garbling plus label encryption, per slot) and
``bcfe_decrypt_slot`` -- at the requested thread counts. One JSON line per
(circuit, mode, threads). A ciphertext to k slots is k components plus a
16-byte nonce; encryption time scales with k.

Usage::

    python benchmarks/run.py bcfe [--runs 5] [--threads 1,4] [--circuits lr16,lr8,score16]
        [--modes hardwired,labels] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time

from pyfe4ai import backend
from pyfe4ai import circuits

#: name -> (builder, keyword arguments); the sizes of 01_schemes_overview section 4.
CIRCUITS = {
    "lr16": ("linear_threshold", {"n": 100, "bw": 16, "bx": 16}),
    "lr8": ("linear_threshold", {"n": 100, "bw": 8, "bx": 8}),
    "lr16x8": ("linear_threshold", {"n": 100, "bw": 16, "bx": 8}),
    "score16": ("linear_score", {"n": 100, "bw": 16, "bx": 16}),
    "mlp64": ("mlp", {"n": 64, "hidden": 16, "bw": 8, "bx": 8, "bh": 8, "shift": 8}),
}


def _pack(bits) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        out[i >> 3] |= (b & 1) << (i & 7)
    return bytes(out)


def _median_time(fn, runs: int):
    times, result = [], None
    for _ in range(runs):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), result


def measure(name: str, mode: str, threads: int, runs: int, rng: random.Random) -> dict:
    builder, kwargs = CIRCUITS[name]
    pc = getattr(circuits, builder)(**kwargs)
    guarded = pc.circuit.guard()
    enc = guarded.encode()
    m = pc.circuit.m
    n = pc.circuit.n
    stats = guarded.stats(mode)
    ops = backend.ops()
    backend.set_threads(threads)
    setup_digest = bytes(rng.randrange(256) for _ in range(32))
    pp_digest = bytes(rng.randrange(256) for _ in range(32))
    w = _pack([rng.randrange(2) for _ in range(m)])
    x = _pack([rng.randrange(2) for _ in range(n)])
    t_new, slot = _median_time(lambda: ops.bcfe_new_slot(setup_digest, 0, w, m=m), runs)
    nonce = bytes(16)
    t_enc, comp = _median_time(
        lambda: ops.bcfe_encrypt_slot(enc, mode, pp_digest, nonce, 0, x, slot["pks"]), runs)
    t_dec, _ = _median_time(
        lambda: ops.bcfe_decrypt_slot(enc, mode, pp_digest, nonce, 0, comp["R"], comp["label_cts"], comp["tables"],
                                      comp["decode"], comp["data_active"], slot["scalars"], w), runs)
    component = sum(len(comp[k]) for k in ("R", "label_cts", "tables", "decode", "data_active"))
    return {
        "circuit": name, "builder": builder, "params": kwargs, "mode": mode, "threads": threads,
        "backend": backend.implementation() or "python", "m": m, "n": n, "gates": guarded.n_gates, **stats,
        "pk_table_bytes": len(slot["pks"]), "key_scalars_bytes": len(slot["scalars"]),
        "ct_component_bytes": component, "tables_bytes": len(comp["tables"]),
        "label_cts_bytes": len(comp["label_cts"]), "runs": runs,
        "new_slot_median_s": t_new, "encrypt_slot_median_s": t_enc, "decrypt_slot_median_s": t_dec,
    }


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--circuits", default="lr16,lr8,score16")
    parser.add_argument("--modes", default="hardwired,labels")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="append the JSON lines to this file too")
    args = parser.parse_args(argv)
    header = {"host": platform.node(), "machine": platform.machine(), "python": sys.version.split()[0],
              "backend": backend.implementation() or "python", "load": os.getloadavg()}
    print(json.dumps({"header": header}))
    rng = random.Random(args.seed)
    rows = []
    for name in args.circuits.split(","):
        for mode in args.modes.split(","):
            for threads in (int(t) for t in args.threads.split(",")):
                row = measure(name, mode, threads, args.runs, rng)
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"header": header}) + "\n")
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
