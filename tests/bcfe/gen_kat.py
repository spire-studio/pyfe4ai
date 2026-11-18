"""Generate the known-answer tests of the half-gates garbling (``tests/bcfe/kat/gc_v1.json``).

::

    python tests/bcfe/gen_kat.py --out tests/bcfe/kat/gc_v1.json

The answers come from the Python reference :mod:`pyfe4ai.primitives.garbling`
(half gates of Zahur, Rosulek and Evans, EUROCRYPT 2015, Fig. 2, with the
gate hash, seed expansion and encodings of the 0.6 P1 plan, section 3) with
the seed passed explicitly. The output is deterministic: every seed, salt,
input bit and random circuit is read from SHAKE-256 of a domain-separation
string and the case's name (:func:`derive`), so a second run writes the same
bytes; ``tests/bcfe/test_gc_kat.py`` checks that, the answers against both
backends, and the library circuits' digests against :mod:`pyfe4ai.circuits`.

Cases (each garbles ``guard(C)``, the g bit of SS10 as key wire ``m``):

* ``hand-*``: tiny circuits written gate by gate, every gate kind pair in
  both modes (the suffix ``-ll``, ``-lk``, ... names the input kinds in mode
  ``"hardwired"``; in mode ``"labels"`` every wire is labelled), with the
  tables in full and the output for every key ``w`` (``output_by_w``);
* ``lib-*``: library circuits, given by their function, parameters and
  digest (a change of the builder changes the digest);
* ``random-2000-*``: one random circuit of 2,000 gates per mode, given by its
  encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if not __package__:  # run as a script: the repository's package before an installed one
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyfe4ai import circuits  # noqa: E402
from pyfe4ai.circuits import Circuit  # noqa: E402
from pyfe4ai.primitives import garbling as gc  # noqa: E402

#: The KAT file this script writes by default (relative to the repository).
DEFAULT_OUT = Path(__file__).resolve().parent / "kat" / "gc_v1.json"
#: The specification the answers follow.
SPEC = "04_p1_plan section 3, 2026-11-19 rev. 2 (salted seed expansion)"

#: Domain-separation strings of the KAT's own derivations (not of the scheme).
KAT_SEED_DST = b"PYFE4AI-V06-BCFE-KAT-SEED"
KAT_SALT_DST = b"PYFE4AI-V06-BCFE-KAT-SALT"
KAT_X_DST = b"PYFE4AI-V06-BCFE-KAT-X"
KAT_W_DST = b"PYFE4AI-V06-BCFE-KAT-W"
KAT_CIRCUIT_DST = b"PYFE4AI-V06-BCFE-KAT-CIRCUIT"

MODES = (gc.MODE_HARDWIRED, gc.MODE_LABELS)

#: The one-gate circuits: m = 2 key wires (0, 1), n = 2 data wires (2, 3),
#: the gate writes wire 4, the only output. Named by the kinds of the inputs
#: in mode "hardwired" (key wires L, data wires K).
ONE_GATE = (
    ("and-ll", ("AND", 0, 1)),
    ("and-lk", ("AND", 0, 2)),
    ("and-kl", ("AND", 3, 1)),
    ("and-kk", ("AND", 2, 3)),
    ("xor-ll", ("XOR", 0, 1)),
    ("xor-lk", ("XOR", 1, 3)),
    ("xor-kl", ("XOR", 2, 0)),
    ("xor-kk", ("XOR", 2, 3)),
    ("inv-l", ("INV", 1, 0)),
    ("inv-k", ("INV", 3, 0)),
)

#: Every gate kind pair in one circuit, then gates on their outputs (m = n = 2).
ALL_KINDS = Circuit(2, 2, [
    ("AND", 0, 1), ("AND", 0, 2), ("AND", 3, 1), ("AND", 2, 3),  # wires 4..7: LL, LK, KL, KK
    ("XOR", 0, 1), ("XOR", 1, 3), ("XOR", 2, 0), ("XOR", 2, 3),  # wires 8..11
    ("INV", 0, 0), ("INV", 3, 0),                                # wires 12, 13: L, K
    ("AND", 4, 7), ("AND", 11, 13), ("XOR", 15, 12), ("AND", 14, 9),  # wires 14..17: LK, KK, KL, LL
], list(range(4, 18)) + [0, 2])

#: ``w + x`` on two bits each (ripple carry; three output bits, low first).
ADDER2 = Circuit(2, 2, [
    ("XOR", 0, 2), ("AND", 0, 2),   # 4: s0, 5: c0
    ("XOR", 1, 3), ("XOR", 6, 5),   # 6: t = w1 ^ x1, 7: s1
    ("AND", 1, 3), ("AND", 5, 6),   # 8, 9
    ("XOR", 8, 9),                  # 10: c1
], [4, 7, 10])

#: The library circuits of the KAT: (case stem, function, keyword arguments, modes).
LIBRARY = (
    ("lib-linear-threshold-n4-bw4-bx4", "linear_threshold", {"n": 4, "bw": 4, "bx": 4}, MODES),
    ("lib-linear-score-n3-bw5-bx3-s2", "linear_score", {"n": 3, "bw": 5, "bx": 3, "shift": 2}, MODES),
    ("lib-mlp-n2-h2-bw3-bx3-bh3-s1", "mlp", {"n": 2, "hidden": 2, "bw": 3, "bx": 3, "bh": 3, "shift": 1},
     (gc.MODE_HARDWIRED,)),
)

#: Gates of ``guard(C)`` in the random cases.
RANDOM_GATES = 2000


def derive(dst: bytes, name: str, nbytes: int) -> bytes:
    """``SHAKE-256(dst || name)[0:nbytes]`` (*name* in UTF-8)."""
    return hashlib.shake_256(dst + name.encode("utf-8")).digest(nbytes)


def derive_bits(dst: bytes, name: str, count: int) -> list:
    """The first *count* bits of ``SHAKE-256(dst || name)``, least significant bit of each byte first."""
    data = derive(dst, name, (count + 7) // 8)
    return [(data[i >> 3] >> (i & 7)) & 1 for i in range(count)]


def case_seed(name: str) -> bytes:
    return derive(KAT_SEED_DST, name, gc.SEED_BYTES)


def case_salt(name: str) -> bytes:
    return derive(KAT_SALT_DST, name, gc.SALT_BYTES)


def case_x(name: str, n: int) -> list:
    return derive_bits(KAT_X_DST, name, n)


def case_w(name: str, m: int) -> list:
    """The ``m + 1`` key bits of ``guard(C)``: ``m`` derived bits, then ``g = 0``."""
    return derive_bits(KAT_W_DST, name, m) + [0]


class ShakeStream:
    """Integers below a bound from ``SHAKE-256(KAT-CIRCUIT DST || name)``:
    little-endian u32 words in order, rejection sampled (a word ``v`` is
    used when ``v < 2^32 - 2^32 mod bound``, as ``v mod bound``)."""

    def __init__(self, name: str, nbytes: int = 1 << 18) -> None:
        self._data = derive(KAT_CIRCUIT_DST, name, nbytes)
        self._pos = 0

    def below(self, bound: int) -> int:
        if not 1 <= bound <= 1 << 32:
            raise ValueError("bound out of range")
        limit = (1 << 32) - (1 << 32) % bound
        while True:
            if self._pos + 4 > len(self._data):
                raise RuntimeError("the KAT's random stream is exhausted")
            v = int.from_bytes(self._data[self._pos:self._pos + 4], "little")
            self._pos += 4
            if v < limit:
                return v % bound


def random_circuit(name: str, total_gates: int = RANDOM_GATES) -> Circuit:
    """A random circuit ``C`` whose ``guard(C)`` has *total_gates* gates, from the stream of *name*.

    Reads, in order: ``m = 1 + below(32)``, ``n = below(48)``, ``n_out = 1 +
    below(8)``; then per gate the op (``below(20)``: 0-7 XOR, 8-16 AND, 17-19
    INV; INV while only one wire exists), the input ``a``, and for XOR and
    AND the input ``b`` (drawn again while ``b = a``). An input is
    (``below(10)``) 0-2: a wire that is known to the garbler in mode
    ``"hardwired"`` (if one exists; ``below(count)`` of them in order), 3-5: one
    of the last 16 wires (``o - 1 - below(min(o, 16))``), else any wire
    (``below(o)``). Then the ``n_out`` output wires, ``below(m + n + gates)``
    each. ``guard(C)`` adds ``1 + n_out`` gates.
    """
    rng = ShakeStream(name)
    m, n, n_out = 1 + rng.below(32), rng.below(48), 1 + rng.below(8)
    count = total_gates - 1 - n_out
    known = list(range(m, m + n))  # wires of kind K in mode "hardwired"
    is_k = [False] * m + [True] * n

    def pick(o: int) -> int:
        s = rng.below(10)
        if s < 3 and known:
            return known[rng.below(len(known))]
        if s < 6:
            return o - 1 - rng.below(min(o, 16))
        return rng.below(o)

    gates = []
    for i in range(count):
        o = m + n + i
        r = rng.below(20)
        op = gc.OP_XOR if r < 8 else gc.OP_AND if r < 17 else gc.OP_INV
        if o == 1:
            op = gc.OP_INV
        a = pick(o)
        b = 0
        if op != gc.OP_INV:
            b = pick(o)
            while b == a:
                b = pick(o)
        gates.append((op, a, b))
        k = is_k[a] if op == gc.OP_INV else is_k[a] and is_k[b]
        is_k.append(k)
        if k:
            known.append(o)
    outputs = [rng.below(m + n + count) for _ in range(n_out)]
    return Circuit(m, n, gates, outputs)


def library_circuit(function: str, params: dict) -> Circuit:
    """The circuit ``C`` (without the g bit) of ``pyfe4ai.circuits.<function>(**params)``."""
    return getattr(circuits, function)(**params).circuit


def _bits(bits) -> str:
    return "".join(str(b) for b in bits)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def active_key_labels(garbling: dict, w) -> bytes:
    """``key0[j] xor w_j Delta`` for every key wire."""
    delta = int.from_bytes(garbling["delta"], "little")
    key0 = garbling["key0"]
    out = bytearray()
    for j, bit in enumerate(w):
        label = int.from_bytes(key0[16 * j:16 * j + 16], "little") ^ (-bit & delta)
        out += label.to_bytes(16, "little")
    return bytes(out)


def make_case(name: str, mode: str, unguarded: Circuit, circuit_info: dict, full: bool) -> dict:
    """One KAT case: garble ``guard(unguarded)`` with the case's seed and salt
    for its ``x``, evaluate it with the labels of its ``w``. *full*: also
    the key labels, data labels and tables in full and the output for every
    ``w`` (the tiny circuits)."""
    garbled = unguarded.guard()
    enc = garbled.encode()
    seed, salt = case_seed(name), case_salt(name)
    x = case_x(name, garbled.n)
    w = case_w(name, unguarded.m)
    res = gc.garble(enc, mode, salt, x, seed)
    out = gc.evaluate(enc, mode, salt, res["tables"], res["decode"], active_key_labels(res, w), res["data_active"])
    out_bits = gc.unpack_bits(out, garbled.n_out)
    info = dict(circuit_info)
    if "encoding" in info:
        info["encoding"] = enc.hex()
    info.update({"digest": garbled.digest().hex(), "m": garbled.m, "n": garbled.n, "n_gates": garbled.n_gates,
                 "n_out": garbled.n_out})
    expected = {"delta": res["delta"].hex()}
    if full:
        expected.update({"key0": res["key0"].hex(), "data_active": res["data_active"].hex(),
                         "tables": res["tables"].hex()})
    expected.update({
        "key0_sha256": _sha256(res["key0"]),
        "data_active_sha256": _sha256(res["data_active"]),
        "tables_sha256": _sha256(res["tables"]),
        "decode": res["decode"].hex(),
        "table_entries": len(res["tables"]) // gc.LABEL_BYTES,
        "output": _bits(out_bits),
        "plain_output": _bits(garbled.evaluate_plain(w, x)),
    })
    if full:
        by_w = []
        for v in range(1 << garbled.m):
            wv = [(v >> j) & 1 for j in range(garbled.m)]
            y = gc.evaluate(enc, mode, salt, res["tables"], res["decode"], active_key_labels(res, wv),
                            res["data_active"])
            by_w.append(_bits(gc.unpack_bits(y, garbled.n_out)))
        expected["output_by_w"] = by_w
    return {
        "name": name,
        "mode": mode,
        "circuit": info,
        "salt": salt.hex(),
        "seed": seed.hex(),
        "x": _bits(x),
        "x_packed": gc.pack_bits(x, garbled.n).hex(),
        "w": _bits(w),
        "expected": expected,
    }


def cases() -> list:
    out = []
    hand = [(stem, Circuit(2, 2, [gate], [4])) for stem, gate in ONE_GATE]
    hand += [("all-kinds", ALL_KINDS), ("adder2", ADDER2)]
    for stem, c in hand:
        for mode in MODES:
            info = {"source": "hand", "unguarded_encoding": c.encode().hex(),
                    "unguarded_digest": c.digest().hex(), "encoding": None}
            out.append(make_case("hand-{}-{}".format(stem, mode), mode, c, info, full=True))
    for stem, function, params, modes in LIBRARY:
        c = library_circuit(function, params)
        for mode in modes:
            info = {"source": "library", "function": function, "params": dict(params),
                    "unguarded_digest": c.digest().hex()}
            out.append(make_case("{}-{}".format(stem, mode), mode, c, info, full=False))
    for mode in MODES:
        name = "random-{}-{}".format(RANDOM_GATES, mode)
        c = random_circuit(name)
        info = {"source": "random", "generator": "gen_kat.random_circuit({!r})".format(name),
                "encoding": None}
        out.append(make_case(name, mode, c, info, full=False))
    return out


def build_kat() -> dict:
    """The whole KAT document."""
    return {
        "kat": "pyfe4ai bcfe half-gates garbling, known answers",
        "version": 1,
        "spec": SPEC,
        "generator": "python tests/bcfe/gen_kat.py --out tests/bcfe/kat/gc_v1.json",
        "construction": "half gates (Zahur, Rosulek, Evans, EUROCRYPT 2015, Fig. 2) on CIRC v1 circuits, "
                        "free XOR, point-and-permute; reference pyfe4ai.primitives.garbling",
        "dst": {
            "gate_hash": gc.GATE_HASH_DST.decode(),
            "garbling_prg": gc.PRG_DST.decode(),
            "circuit_digest": circuits.CIRCUIT_DIGEST_DOMAIN.decode(),
            "kat_seed": KAT_SEED_DST.decode(),
            "kat_salt": KAT_SALT_DST.decode(),
            "kat_x": KAT_X_DST.decode(),
            "kat_w": KAT_W_DST.decode(),
            "kat_circuit": KAT_CIRCUIT_DST.decode(),
        },
        "derivation": {
            "seed": "SHAKE-256(dst.kat_seed || name)[0:32], name in UTF-8",
            "salt": "SHAKE-256(dst.kat_salt || name)[0:56] (opaque to the garbling; "
                    "the scheme's salt is pp_digest || ct_nonce || u64(slot_id))",
            "x": "the first n bits of SHAKE-256(dst.kat_x || name), LSB of each byte first",
            "w": "the first m bits of SHAKE-256(dst.kat_w || name) (m: key bits of C), then g = 0",
            "random circuits": "gen_kat.random_circuit(name): SHAKE-256(dst.kat_circuit || name) read as "
                               "u32 LE words, rejection sampled; the encoding is stored",
        },
        "conventions": {
            "circuit": "every case garbles guard(C) (SS10 g bit = key wire m of C); circuit.encoding / "
                       "circuit.digest are those of guard(C), unguarded_* those of C (what pp binds); "
                       "library circuits are pyfe4ai.circuits.<function>(**params).circuit",
            "bits": "x, w and outputs are strings of 0/1, character i = wire / output i; w has the m + 1 key "
                    "bits of guard(C), g last",
            "packed": "x_packed and decode: bits packed LSB first (spec 3.7)",
            "labels": "16-byte strings; key_active[j] = key0[j] xor w_j * delta; in mode labels data_active "
                      "is returned by the garbling, in mode hardwired it is empty",
            "seed expansion": "SHAKE-256(dst.garbling_prg || salt(56) || seed(32)) read as delta (then "
                              "delta[0] |= 1), key0 per key wire, the data wires' zero labels (mode labels)",
            "sha256": "SHA-256 of the raw bytes of key0, data_active, tables",
            "output": "evaluation of the garbling with the labels of w, decoded; plain_output: guard(C)(w, x) "
                      "in the clear; output_by_w[v]: the output for the key bits of the integer v (LSB = "
                      "wire 0), same garbling",
        },
        "cases": cases(),
    }


def dumps(kat: dict) -> str:
    return json.dumps(kat, indent=1) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the JSON file to write")
    args = parser.parse_args(argv)
    text = dumps(build_kat())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print("wrote {} ({} bytes)".format(args.out, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
