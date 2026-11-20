"""Timing of the native bcfe paths that ``rust/core/examples/ct_dudect`` cannot reach.

The label KEM of ``bcfe/gc_ristretto255`` (the slot tables chosen by ``w``, the label encryption, the decryption with
``w``) lives in the PyO3 module ``rust/native/src/bcfe.rs``, so it is timed here from Python: one call per
measurement, ``time.perf_counter_ns`` around it, the classes in random order, Welch's t on the raw times and on the
times cropped at a few percentiles (as dudect). Both classes get inputs prepared in the same way, each a distinct
object (class 0 copies of one secret, class 1 fresh ones), so that neither reuses a hot buffer. Every target has an
A/A twin (``*_null``: class 1 in both), whose spread is the resolution of the method on the machine; a target is
read against its twin. Run on a quiet machine with the native extension built:

    python tools/ct_bcfe.py TARGET [MEASUREMENTS] [--seed S]

Targets (circuit ``linear_threshold(4, 4, 4)`` under ``guard``: m = 24 key bits, n = 16 data bits, mode hardwired):

* ``new_slot_w``: ``bcfe_new_slot`` (h1), the randomness fixed, ``w`` the secret;
* ``place_slot_w``: ``bcfe_place_slot`` (h2), the holder's keys and ``k`` fixed, ``w`` the secret;
* ``encrypt_slot_x``: ``bcfe_encrypt_slot``, the slot and the nonce fixed, the seed fresh (OS), ``x`` the secret;
* ``decrypt_slot_w``: ``bcfe_decrypt_slot`` of a fresh ciphertext, the holder's slot (its ``w`` and scalars) the
  secret: one slot for class 0, a fresh one per input for class 1.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the tree, as tools/gen_site_matrix.py

from pyfe4ai import backend  # noqa: E402
from pyfe4ai.circuits import library  # noqa: E402

M, N = 24, 16
CIRCUIT = library.linear_threshold(4, 4, 4).circuit.guard().encode()
SETUP, PP, NONCE, SLOT = b"\x11" * 32, b"\x22" * 32, b"\x33" * 16, 7
POOL = 400  # inputs prepared per class (cycled)


def _ext():
    ext = backend.ext()
    if ext is None or not hasattr(ext, "bcfe_new_slot"):
        sys.exit("the native extension with bcfe is not built")
    return ext


def _bits(nbits: int, rng: random.Random) -> bytes:
    return rng.getrandbits(nbits).to_bytes((nbits + 7) // 8, "little")


def _copy(b: bytes) -> bytes:
    return bytes(bytearray(b))  # a distinct object with the same value


def _slot(ext, w: bytes, rng: random.Random):
    rnd = rng.randbytes(64 * (M + 1) + 16)
    scalars, pks, k = ext.bcfe_new_slot(SETUP, SLOT, M, w, rnd)
    return scalars, pks, k


def _holder_pks(pks: bytes, w: bytes) -> bytes:
    bits = int.from_bytes(w, "little")
    sel = [pks[32 * (2 * j + ((bits >> j) & 1)):32 * (2 * j + ((bits >> j) & 1)) + 32] for j in range(M)]
    return b"".join(sel) + pks[32 * 2 * M:32 * (2 * M + 1)]


def prepare(target: str, rng: random.Random):
    """(call, inputs of class 0, inputs of class 1): call(input) runs the measured operation."""
    ext = _ext()
    null = target.endswith("_null")
    base = target[:-5] if null else target
    fixed_w, fixed_x = _bits(M, rng), _bits(N, rng)

    def secrets(nbits, fixed):
        zero = [_copy(fixed) for _ in range(POOL)] if not null else [_bits(nbits, rng) for _ in range(POOL)]
        return zero, [_bits(nbits, rng) for _ in range(POOL)]

    if base == "new_slot_w":
        rnd = rng.randbytes(64 * (M + 1) + 16)
        c0, c1 = secrets(M, fixed_w)
        return (lambda w: ext.bcfe_new_slot(SETUP, SLOT, M, w, rnd)), c0, c1
    if base == "place_slot_w":
        _, pks, k = _slot(ext, fixed_w, rng)
        holder = _holder_pks(pks, fixed_w)
        c0, c1 = secrets(M, fixed_w)
        return (lambda w: ext.bcfe_place_slot(SETUP, SLOT, M, w, holder, k)), c0, c1
    if base == "encrypt_slot_x":
        _, pks, _ = _slot(ext, fixed_w, rng)
        c0, c1 = secrets(N, fixed_x)
        return (lambda x: ext.bcfe_encrypt_slot(CIRCUIT, 0, PP, NONCE, SLOT, x, pks)), c0, c1
    if base == "decrypt_slot_w":
        def ciphertext(scalars, pks, w):
            r, cts, tables, decode, active = ext.bcfe_encrypt_slot(CIRCUIT, 0, PP, NONCE, SLOT, _bits(N, rng), pks)
            return (r, cts, tables, decode, active, scalars, w)

        fixed = _slot(ext, fixed_w, rng)
        c0 = []
        for _ in range(POOL):
            if null:
                w = _bits(M, rng)
                scalars, pks, _ = _slot(ext, w, rng)
            else:
                (scalars, pks, _), w = fixed, fixed_w
            c0.append(ciphertext(_copy(scalars), pks, _copy(w)))
        c1 = []
        for _ in range(POOL):
            w = _bits(M, rng)
            scalars, pks, _ = _slot(ext, w, rng)
            c1.append(ciphertext(scalars, pks, w))
        return (lambda a: ext.bcfe_decrypt_slot(CIRCUIT, 0, PP, NONCE, SLOT, *a)), c0, c1
    raise SystemExit(f"unknown target {target}")


def welch(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a, ma), statistics.variance(b, mb)
    se = math.sqrt(va / len(a) + vb / len(b))
    return ma, mb, ma - mb, se, (ma - mb) / se if se else 0.0


def report(times, label):
    t0, t1 = times
    ma, mb, diff, se, t = welch(t0, t1)
    rel, half = diff / mb, 1.96 * se / mb
    crop = []
    for q in (0.5, 0.9, 0.99):
        cut = sorted(t0 + t1)[int(q * (len(t0) + len(t1))) - 1]
        a, b = [v for v in t0 if v <= cut], [v for v in t1 if v <= cut]
        crop.append(abs(welch(a, b)[4]) if len(a) > 2 and len(b) > 2 else 0.0)
    print(f"{label:>10}  mean {ma:12.1f} / {mb:12.1f} ns  diff {diff:+9.1f} ns  se {se:7.1f} ns  "
          f"rel {100 * rel:+.4f} % [{100 * (rel - half):+.4f}, {100 * (rel + half):+.4f}]  t_raw {t:+6.2f}  "
          f"max|t_crop| {max(crop):6.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("measurements", nargs="?", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big"))
    call, c0, c1 = prepare(args.target, rng)
    for inputs in (c0, c1):  # warm both
        for a in inputs[:20]:
            call(a)
    print(f"target {args.target} measurements {args.measurements} (m = {M}, n = {N}, pool {POOL} per class)")
    times = ([], [])
    clock = time.perf_counter_ns
    step = max(1, args.measurements // 10)
    for i in range(args.measurements):
        cls = rng.getrandbits(1)
        a = (c0 if cls == 0 else c1)[rng.randrange(POOL)]
        start = clock()
        call(a)
        times[cls].append(clock() - start)
        if (i + 1) % step == 0:
            report(times, str(i + 1))


if __name__ == "__main__":
    main()
