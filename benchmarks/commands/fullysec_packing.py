"""Elementwise vs packed layout of ``sife/fullysec_ring_lwe`` (standard profile).

For each configuration: the parameters of both layouts, the size of one
ciphertext (JSON, as ``json.dumps`` writes the dict, and the raw coefficients
at ``ceil(bits(q) / 8)`` bytes each) and the median encryption and decryption
time of one ciphertext, per vector. The runs of the two layouts alternate, so
that a load from other processes affects both alike.

Usage::

    python benchmarks/run.py fullysec-packing [--runs 3] [--configs 1,4,10]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time
from unittest import mock

from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWE, SIFEFullySecRingLWEKeyGenerator
from pyfe4ai import backend as native_backend

CONFIGS = {1: (1, 100, 1), 4: (4, 4, 4), 10: (10, 50, 50)}


def _scheme(eta, x, y, layout):
    kg = SIFEFullySecRingLWEKeyGenerator({"eta": eta, "bound_x": x, "bound_y": y,
                                          "security_profile": "standard", "ndarray_layout": layout})
    t = time.perf_counter()
    kg.setup()
    setup = time.perf_counter() - t
    pp = kg.get_public_parameters()
    return kg, pp, SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}}), setup


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--configs", default="1,4,10")
    args = parser.parse_args(argv)
    print("backend:", native_backend.implementation() or "python", "| load:", os.getloadavg())
    rng = random.Random(1)
    for key in (int(c) for c in args.configs.split(",")):
        eta, bx, by = CONFIGS[key]
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.dict(os.environ, {"PYFE4AI_CACHE_DIR": os.path.join(tmp, "config")}):
                    schemes = {layout: _scheme(eta, bx, by, layout) for layout in ("elementwise", "packed")}
            finally:
                os.chdir(cwd)
        y = [rng.randint(-by, by) for _ in range(eta)]
        times = {layout: {"enc": [], "dec": []} for layout in schemes}
        sizes = {}
        for _ in range(args.runs):
            for layout, (kg, pp, enc, _) in schemes.items():
                n = pp["ring_n"]
                k = 1 if layout == "elementwise" else n
                pt = [[rng.randint(-bx, bx) for _ in range(k)] for _ in range(eta)]
                plain = [row[0] for row in pt] if layout == "elementwise" else pt
                t = time.perf_counter()
                ct = enc.encrypt(plain)
                times[layout]["enc"].append(time.perf_counter() - t)
                dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
                dec = SIFEFullySecRingLWE({"keys": {"pp": pp}})
                t = time.perf_counter()
                out = dec.decrypt(ct, dk, y)
                times[layout]["dec"].append(time.perf_counter() - t)
                want = [sum(r[j] * w for r, w in zip(pt, y)) for j in range(k)]
                assert out == (want[0] if layout == "elementwise" else want)
                wb = (int(pp["q"]).bit_length() + 7) // 8
                sizes[layout] = (k, len(json.dumps(ct)), (pp["m"] + eta) * n * wb)
        print("\neta {}, X {}, Y {}".format(eta, bx, by))
        print("| layout | n | m | log2 q | setup s | vectors / ct | JSON B / vector | raw B / vector "
              "| enc ms / ct | enc ms / vector | dec ms / ct | dec ms / vector |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for layout, (kg, pp, _, setup) in schemes.items():
            k, js, raw = sizes[layout]
            e = statistics.median(times[layout]["enc"]) * 1e3
            d = statistics.median(times[layout]["dec"]) * 1e3
            print("| {} | {} | {} | {} | {:.1f} | {} | {:,.0f} | {:,.0f} | {:.0f} | {:.3f} | {:.0f} | {:.3f} |".format(
                layout, pp["ring_n"], pp["m"], int(pp["q"]).bit_length(), setup, k, js / k, raw / k,
                e, e / k, d, d / k))

