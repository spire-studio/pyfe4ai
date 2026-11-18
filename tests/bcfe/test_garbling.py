"""The half-gates garbling reference (:mod:`pyfe4ai.primitives.garbling`) and
the backend operations ``gc_garble`` / ``gc_evaluate`` of the Python backend.

The circuits are encoded by hand (``CIRC v1``) with the helpers below, so that
these tests do not depend on :mod:`pyfe4ai.circuits`; ``guard`` is the
transformation of the specification (key wire ``m`` is the g bit, every
output is ANDed with ``INV(g)``), which makes every output a labelled wire.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import struct

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.backend import interface
from pyfe4ai.primitives import garbling as gc

XOR, AND, INV = gc.OP_XOR, gc.OP_AND, gc.OP_INV
SALT = bytes(range(56))
SEED = bytes(range(100, 132))


# ------------------------------------------------------------------ helpers

def encode(m, n, gates, outputs, magic=b"PFE6CIRC", version=1, counts=None) -> bytes:
    """``CIRC v1`` of the gates ``(op, a, b)`` (*counts* overrides ``(m, n, n_gates, n_out)``)."""
    m_, n_, g_, o_ = counts or (m, n, len(gates), len(outputs))
    out = struct.pack("<8sBIIII", magic, version, m_, n_, g_, o_)
    out += b"".join(struct.pack("<BII", op, a, b) for op, a, b in gates)
    return out + struct.pack("<{}I".format(len(outputs)), *outputs)


def guard(m, n, gates, outputs):
    """The g-bit transformation: ``(m + 1, n, gates', outputs')``."""
    remap = lambda v: v if v < m else v + 1  # noqa: E731
    new = [(op, remap(a), 0 if op == INV else remap(b)) for op, a, b in gates]
    ng = m + 1 + n + len(new)
    new.append((INV, m, 0))
    outs = []
    for o in outputs:
        outs.append(m + 1 + n + len(new))
        new.append((AND, remap(o), ng))
    return m + 1, n, new, outs


def plain(gates, outputs, w, x) -> list:
    """The circuit evaluated in the clear."""
    v = list(w) + list(x)
    for op, a, b in gates:
        v.append(v[a] ^ v[b] if op == XOR else v[a] & v[b] if op == AND else 1 - v[a])
    return [v[o] for o in outputs]


def random_circuit(rng, m, n, count):
    gates = []
    for i in range(count):
        o = m + n + i
        op = rng.choice((XOR, AND, AND, INV)) if o > 1 else INV
        a = rng.randrange(o)
        b = 0
        if op != INV:  # two distinct inputs: CIRC v1 refuses XOR(a, a) and AND(a, a)
            b = rng.randrange(o - 1)
            b += b >= a
        gates.append((op, a, b))
    outputs = [rng.randrange(m + n + count) for _ in range(rng.randint(1, 4))]
    return gates, outputs


def as_int(label: bytes) -> int:
    return int.from_bytes(label, "little")


def as_bytes(value: int) -> bytes:
    return value.to_bytes(16, "little")


def split(data: bytes) -> list:
    return [data[i:i + 16] for i in range(0, len(data), 16)]


def active_keys(result: dict, w) -> bytes:
    """The active labels of the key bits *w* (``W0 xor w_j Delta``)."""
    delta = as_int(result["delta"])
    return b"".join(as_bytes(as_int(k) ^ (delta if b else 0)) for k, b in zip(split(result["key0"]), w))


def garble_and_evaluate(circuit, mode, w, x, salt=SALT, seed=SEED) -> tuple:
    """``(output bits, garbling)`` of garbling *circuit* for *x* and evaluating it with the labels of *w*."""
    res = gc.garble(circuit, mode, salt, x, seed)
    y = gc.evaluate(circuit, mode, salt, res["tables"], res["decode"], active_keys(res, w), res["data_active"])
    return gc.unpack_bits(y, gc.parse_circuit(circuit).n_out), res


def gate_classes(parsed, mode) -> list:
    """``"XOR-LK"``, ``"AND-KK"``, ``"INV-L"``, ... per gate."""
    kinds = gc.wire_kinds(parsed, mode)
    name = {gc.KIND_L: "L", gc.KIND_K: "K"}
    out = []
    for op, a, b in parsed.gates:
        if op == INV:
            out.append("INV-" + name[kinds[a]])
        else:
            out.append(("XOR-" if op == XOR else "AND-") + name[kinds[a]] + name[kinds[b]])
    return out


# ------------------------------------------------------------------ decoding

def test_parse_round_trip():
    gates = [(XOR, 0, 2), (AND, 1, 3), (INV, 4, 0)]
    data = encode(2, 1, gates, [5, 3])
    for form in (data, bytearray(data), memoryview(data)):
        parsed = gc.parse_circuit(form)
        assert (parsed.m, parsed.n, parsed.gates, parsed.outputs) == (2, 1, tuple(gates), (5, 3))
        assert (parsed.n_gates, parsed.n_out, parsed.n_wires) == (3, 2, 6)
    assert repr(parsed) == "ParsedCircuit(m=2, n=1, n_gates=3, n_out=2)"
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.m = 3


GOOD = encode(2, 1, [(XOR, 0, 2), (AND, 1, 3), (INV, 4, 0)], [5])


@pytest.mark.parametrize("data, match", [
    (b"", "bad magic"),
    (b"PFE6CIRX" + GOOD[8:], "bad magic"),
    (GOOD[:8] + b"\x02" + GOOD[9:], "version 2"),
    (GOOD[:8] + b"\x00" + GOOD[9:], "version 0"),
    (GOOD[:20], "shorter than the 25-byte header"),
    (GOOD[:-1], "the counts give"),
    (GOOD + b"\x00", "the counts give"),
    (encode(2, 1, [(XOR, 0, 2)], [3], counts=(2, 1, 2, 1)), "the counts give"),
    (encode(0, 1, [(INV, 0, 0)], [1]), "m = 0"),
    (encode(2, 1, [(XOR, 0, 2)], []), "n_out = 0"),
    (encode(2 ** 31 - 2, 1, [], [0], counts=(2 ** 31 - 2, 1, 1, 1)), "not below 2"),
    (encode(1, 2 ** 31 - 1, [], [0]), "not below 2"),
    (encode(2 ** 26 - 1, 1, [], [0]), "not below 2\\^26"),
    (encode(1, 2 ** 26, [], [0]), "not below 2\\^26"),
    (encode(2, 1, [(0, 0, 1)], [3]), "unknown op 0"),
    (encode(2, 1, [(4, 0, 1)], [3]), "unknown op 4"),
    (encode(2, 1, [(INV, 0, 1)], [3]), r"\(INV\) has b = 1"),
    (encode(2, 1, [(XOR, 1, 1)], [3]), r"\(XOR\) reads wire 1 twice"),
    (encode(2, 1, [(AND, 0, 0)], [3]), r"\(AND\) reads wire 0 twice"),
    (encode(2, 1, [(XOR, 3, 0)], [3]), "reads wire 3 not below its output wire 3"),
    (encode(2, 1, [(AND, 0, 4)], [3]), "reads wire 4 not below its output wire 3"),
    (encode(2, 1, [(INV, 3, 0)], [3]), "reads wire 3"),
    (encode(2, 1, [(XOR, 0, 1), (AND, 4, 0)], [4]), "gate 1 .* reads wire 4"),
    (encode(2, 1, [(XOR, 0, 1)], [4]), "output 0 is wire 4, out of range"),
    (encode(2, 1, [(XOR, 0, 1)], [3, 2 ** 32 - 1]), "output 1 is wire"),
])
def test_parse_rejects(data, match):
    with pytest.raises(ValueError, match=match):
        gc.parse_circuit(data)


def test_parse_refuses_other_types():
    with pytest.raises(TypeError):
        gc.parse_circuit(GOOD.hex())


def test_wire_kinds():
    parsed = gc.parse_circuit(encode(1, 2, [(XOR, 1, 2), (AND, 0, 3), (INV, 3, 0), (INV, 4, 0)], [4]))
    L, K = gc.KIND_L, gc.KIND_K
    assert gc.wire_kinds(parsed, "hardwired") == bytes([L, K, K, K, L, K, L])
    assert gc.wire_kinds(parsed, "labels") == bytes([L] * 7)
    with pytest.raises(ValueError, match="unknown garbling mode"):
        gc.wire_kinds(parsed, "K")


# ------------------------------------------------------------------ hash, seed, bits

def test_gate_hash():
    x = bytes(range(16))
    for idx in (0, 1, 2 ** 64 - 1):
        expected = hashlib.shake_128(b"PFE6GC02" + SALT + idx.to_bytes(8, "little") + x).digest(16)
        assert gc.gate_hash(SALT, idx, x) == expected
    assert gc.gate_hash(SALT, 0, x) != gc.gate_hash(SALT, 1, x) != gc.gate_hash(SALT[::-1], 1, x)
    for salt in (SALT[:-1], SALT + b"\x00", b""):
        with pytest.raises(ValueError, match="salt must be 56 bytes"):
            gc.gate_hash(salt, 0, x)
    for label in (x[:-1], x + b"\x00"):
        with pytest.raises(ValueError, match="label must be 16 bytes"):
            gc.gate_hash(SALT, 0, label)
    for idx in (-1, 2 ** 64, True, 1.0):
        with pytest.raises(ValueError, match="index"):
            gc.gate_hash(SALT, idx, x)


def test_expand_seed():
    delta, key0, data0 = gc.expand_seed(SALT, SEED, 3, 2)
    stream = hashlib.shake_256(b"PYFE4AI-V06-BCFE-GBPRG" + SALT + SEED).digest(16 * 6)
    assert delta == bytes([stream[0] | 1]) + stream[1:16] and delta[0] & 1
    assert key0 == split(stream[16:64]) and data0 == split(stream[64:96])
    assert gc.expand_seed(SALT, SEED, 3, 0) == (delta, key0, [])
    # the salt is in the expansion: another instance, unrelated Delta and labels from the same seed
    other = gc.expand_seed(SALT[:-1] + bytes([SALT[-1] ^ 1]), SEED, 3, 2)
    assert other[0] != delta and set(other[1]).isdisjoint(key0) and set(other[2]).isdisjoint(data0)
    for seed in (SEED[:-1], SEED + b"\x00"):
        with pytest.raises(ValueError, match="seed must be 32 bytes"):
            gc.expand_seed(SALT, seed, 1, 0)
    for salt in (SALT[:-1], SALT + b"\x00"):
        with pytest.raises(ValueError, match="salt must be 56 bytes"):
            gc.expand_seed(salt, SEED, 1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        gc.expand_seed(SALT, SEED, -1, 0)


def test_pack_and_unpack_bits():
    bits = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
    assert gc.pack_bits(bits, 11) == b"\x81\x05"
    assert gc.unpack_bits(b"\x81\x05", 11) == bits
    assert gc.pack_bits([True, False], 2) == b"\x01"
    assert gc.pack_bits([], 0) == b"" and gc.unpack_bits(b"", 0) == []
    assert gc.unpack_bits(b"\xff", 8) == [1] * 8
    with pytest.raises(ValueError, match="nonzero padding"):
        gc.unpack_bits(b"\x81\x0d", 11)
    with pytest.raises(ValueError, match="take 2 bytes"):
        gc.unpack_bits(b"\x81", 11)
    with pytest.raises(ValueError, match="take 1 bytes"):
        gc.unpack_bits(b"\x01\x00", 3)
    with pytest.raises(ValueError, match="expected 3 bits"):
        gc.pack_bits([1, 0], 3)
    with pytest.raises(ValueError, match="bit 1 is not 0 or 1"):
        gc.pack_bits([1, 2], 2)


# ------------------------------------------------------------------ correctness

@pytest.mark.parametrize("mode", ["hardwired", "labels"])
def test_random_circuits_evaluate_to_the_plain_circuit(mode):
    rng = random.Random(20261119 + len(mode))
    seen = set()
    for trial in range(600):
        m0, n = rng.randint(1, 4), rng.randint(0, 5)
        gates0, outputs0 = random_circuit(rng, m0, n, rng.randint(0, 60))
        m, n, gates, outputs = guard(m0, n, gates0, outputs0)
        circuit = encode(m, n, gates, outputs)
        parsed = gc.parse_circuit(circuit)
        classes = gate_classes(parsed, mode)
        seen.update(classes)
        w = [rng.randrange(2) for _ in range(m0)] + [int(rng.random() < 0.25)]
        x = [rng.randrange(2) for _ in range(n)]
        salt, seed = rng.randbytes(56), rng.randbytes(32)
        y, res = garble_and_evaluate(circuit, mode, w, x if trial % 2 else gc.pack_bits(x, n), salt, seed)
        expected = plain(gates, outputs, w, x)
        assert y == expected
        assert y == (plain(gates0, outputs0, w[:-1], x) if w[-1] == 0 else [0] * len(outputs0))
        entries = 2 * classes.count("AND-LL") + classes.count("AND-LK") + classes.count("AND-KL")
        assert len(res["tables"]) == 16 * entries == 16 * gc.table_entries(parsed, mode)
        assert len(res["key0"]) == 16 * m and len(res["delta"]) == 16 and res["delta"][0] & 1
        assert len(res["data_active"]) == (16 * n if mode == "labels" else 0)
    every = {"XOR-LL", "XOR-LK", "XOR-KL", "XOR-KK", "AND-LL", "AND-LK", "AND-KL", "AND-KK", "INV-L", "INV-K"}
    assert seen == (every if mode == "hardwired" else {c for c in every if "K" not in c})


def test_every_gate_kind_on_every_input():
    # hardwired: key wires 0, 1 (L), data wires 2, 3 (K)
    gates0 = [(XOR, 0, 1), (XOR, 0, 2), (XOR, 3, 1), (XOR, 2, 3),   # 4..7: LL LK KL KK
              (AND, 0, 1), (AND, 1, 3), (AND, 2, 0), (AND, 2, 3),   # 8..11: LL LK KL KK
              (INV, 0, 0), (INV, 2, 0),                             # 12, 13: L K
              (AND, 7, 9), (XOR, 11, 13), (AND, 4, 15), (AND, 16, 6)]  # 14..17 on derived wires
    outputs0 = list(range(4, 18))
    m, n, gates, outputs = guard(2, 2, gates0, outputs0)
    circuit = encode(m, n, gates, outputs)
    assert set(gate_classes(gc.parse_circuit(circuit), "hardwired")[:10]) == {
        "XOR-LL", "XOR-LK", "XOR-KL", "XOR-KK", "AND-LL", "AND-LK", "AND-KL", "AND-KK", "INV-L", "INV-K"}
    for mode in ("hardwired", "labels"):
        for bits in range(32):
            w = [bits & 1, bits >> 1 & 1, bits >> 4 & 1]
            x = [bits >> 2 & 1, bits >> 3 & 1]
            y, _ = garble_and_evaluate(circuit, mode, w, x, seed=bytes([bits]) * 32)
            assert y == plain(gates, outputs, w, x)


def test_hash_indices_follow_next_index():
    """XOR and INV take no index; AND(L, L) takes j = idx, j' = idx + 1; a half-gate the next one."""
    # key wires 0, 1; data wire 2 (K): XOR(0, 1) -> 3, AND(0, 1) -> 4, INV(4) -> 5, AND(2, 5) -> 6
    circuit = encode(2, 1, [(XOR, 0, 1), (AND, 0, 1), (INV, 4, 0), (AND, 2, 5)], [6])
    for x0 in (0, 1):
        res = gc.garble(circuit, "hardwired", SALT, [x0], SEED)
        delta_b, key0, _ = gc.expand_seed(SALT, SEED, 2, 0)
        assert res["delta"] == delta_b and res["key0"] == b"".join(key0) and res["data_active"] == b""
        delta = as_int(delta_b)
        a0, b0 = as_int(key0[0]), as_int(key0[1])

        def h(label, idx):
            return as_int(gc.gate_hash(SALT, idx, as_bytes(label)))

        t_g = h(a0, 0) ^ h(a0 ^ delta, 0) ^ (delta if b0 & 1 else 0)
        t_e = h(b0, 1) ^ h(b0 ^ delta, 1) ^ a0
        w4 = (h(a0, 0) ^ (t_g if a0 & 1 else 0)) ^ (h(b0, 1) ^ (t_e ^ a0 if b0 & 1 else 0))
        w5 = w4 ^ delta
        t = h(w5, 2) ^ h(w5 ^ delta, 2) ^ (delta if x0 else 0)
        w6 = h(w5, 2) ^ (t if w5 & 1 else 0)
        assert split(res["tables"]) == [as_bytes(t_g), as_bytes(t_e), as_bytes(t)]
        assert res["decode"] == bytes([w6 & 1])

    # a half-gate first: AND(0, 2) takes 0, the AND(L, L) after it 1 and 2
    circuit = encode(2, 1, [(AND, 0, 2), (AND, 1, 3)], [4])
    res = gc.garble(circuit, "hardwired", SALT, [1], SEED)
    delta_b, key0, _ = gc.expand_seed(SALT, SEED, 2, 0)
    delta, a0, b0 = as_int(delta_b), as_int(key0[0]), as_int(key0[1])

    def h(label, idx):
        return as_int(gc.gate_hash(SALT, idx, as_bytes(label)))

    t = h(a0, 0) ^ h(a0 ^ delta, 0) ^ delta
    w3 = h(a0, 0) ^ (t if a0 & 1 else 0)
    t_g = h(b0, 1) ^ h(b0 ^ delta, 1) ^ (delta if w3 & 1 else 0)
    t_e = h(w3, 2) ^ h(w3 ^ delta, 2) ^ b0
    assert split(res["tables"]) == [as_bytes(t), as_bytes(t_g), as_bytes(t_e)]


def test_decoding_bits_are_the_permute_bits_of_the_outputs():
    circuit = encode(3, 0, [], [0, 2, 1, 2])  # outputs are key wires: no gate, no table
    res = gc.garble(circuit, "hardwired", SALT, b"", SEED)
    key0 = split(res["key0"])
    assert res["tables"] == b""
    assert gc.unpack_bits(res["decode"], 4) == [key0[0][0] & 1, key0[2][0] & 1, key0[1][0] & 1, key0[2][0] & 1]
    for w in ([0, 0, 0], [1, 0, 1], [0, 1, 1]):
        out = gc.evaluate(circuit, "hardwired", SALT, b"", res["decode"], active_keys(res, w), b"")
        assert gc.unpack_bits(out, 4) == [w[0], w[2], w[1], w[2]]


def test_labels_mode_hands_out_the_active_data_labels():
    m, n, gates, outputs = guard(2, 5, [(AND, 0, 2), (XOR, 1, 3)], [7, 8, 6])
    circuit = encode(m, n, gates, outputs)
    x = [1, 0, 1, 1, 0]
    res = gc.garble(circuit, "labels", SALT, x, SEED)
    delta_b, key0, data0 = gc.expand_seed(SALT, SEED, m, n)
    delta = as_int(delta_b)
    assert res["delta"] == delta_b and res["key0"] == b"".join(key0)
    assert split(res["data_active"]) == [as_bytes(as_int(w0) ^ (delta if b else 0)) for w0, b in zip(data0, x)]
    hard = gc.garble(circuit, "hardwired", SALT, x, SEED)  # the key labels come before the data labels
    assert hard["delta"] == res["delta"] and hard["key0"] == res["key0"] and hard["data_active"] == b""


def test_seed_and_salt():
    m, n, gates, outputs = guard(3, 3, [(AND, 0, 1), (AND, 2, 3), (XOR, 6, 7), (AND, 8, 4)], [9, 6])
    circuit = encode(m, n, gates, outputs)
    x = [1, 0, 1]
    first = gc.garble(circuit, "hardwired", SALT, x, SEED)
    assert gc.garble(circuit, "hardwired", SALT, bytearray(gc.pack_bits(x, n)), SEED) == first
    other_seed = gc.garble(circuit, "hardwired", SALT, x, SEED[::-1])
    assert other_seed["delta"] != first["delta"] and other_seed["key0"] != first["key0"]
    assert other_seed["tables"] != first["tables"]
    salt2 = SALT[:48] + bytes([SALT[48] ^ 1]) + SALT[49:]  # another slot id (the low byte of u64(slot_id))
    other_salt = gc.garble(circuit, "hardwired", salt2, x, SEED)
    # the salt is in the seed expansion too: a seed reused for another slot shares no label
    assert other_salt["delta"] != first["delta"]
    assert set(split(other_salt["key0"])).isdisjoint(split(first["key0"]))
    entries = list(zip(split(first["tables"]), split(other_salt["tables"])))
    assert len(entries) == gc.table_entries(gc.parse_circuit(circuit), "hardwired") > 4
    assert all(e1 != e2 for e1, e2 in entries)
    w = [1, 1, 0, 0]
    y = gc.evaluate(circuit, "hardwired", salt2, other_salt["tables"], other_salt["decode"],
                    active_keys(other_salt, w), b"")
    assert gc.unpack_bits(y, 2) == plain(gates, outputs, w, x)


def test_tables_do_not_depend_on_x():
    m, n, gates, outputs = guard(2, 3, [(AND, 0, 2), (AND, 3, 4), (XOR, 5, 6), (AND, 7, 1)], [8, 6])
    circuit = encode(m, n, gates, outputs)
    sizes = {tuple(len(v) for v in gc.garble(circuit, "hardwired", SALT, list(map(int, format(v, "03b"))),
                                                SEED).values()) for v in range(8)}
    assert len(sizes) == 1


def test_outputs_must_be_labelled_wires():
    circuit = encode(1, 2, [(AND, 1, 2), (XOR, 0, 3)], [3, 4])  # wire 3 = AND(K, K)
    for call in (lambda: gc.garble(circuit, "hardwired", SALT, [1, 1], SEED),
                 lambda: gc.evaluate(circuit, "hardwired", SALT, b"", b"\x00", bytes(16), b"")):
        with pytest.raises(ValueError, match="output 0 .* not a labelled wire"):
            call()
    y, _ = garble_and_evaluate(circuit, "labels", [1], [1, 1])
    assert y == [1, 0]


# ------------------------------------------------------------------ argument errors

CIRCUIT = encode(*guard(2, 3, [(AND, 0, 2), (AND, 1, 3), (XOR, 5, 6)], [7]))
M, N = 3, 3


@pytest.mark.parametrize("change, error, match", [
    ({"mode": "K"}, ValueError, "unknown garbling mode"),
    ({"salt": SALT[:-1]}, ValueError, "salt must be 56 bytes"),
    ({"salt": SALT + b"\x00"}, ValueError, "salt must be 56 bytes"),
    ({"salt": SALT.hex()}, TypeError, "salt must be bytes"),
    ({"seed": SEED[:-1]}, ValueError, "seed must be 32 bytes"),
    ({"seed": SEED + b"\x00"}, ValueError, "seed must be 32 bytes"),
    ({"x": b"\x01\x00"}, ValueError, "take 1 bytes"),
    ({"x": b"\x09"}, ValueError, "nonzero padding"),
    ({"x": [1, 0]}, ValueError, "expected 3 bits"),
    ({"x": [1, 0, 2]}, ValueError, "not 0 or 1"),
    ({"circuit": CIRCUIT[:-1]}, ValueError, "the counts give"),
])
def test_garble_errors(change, error, match):
    args = {"circuit": CIRCUIT, "mode": "hardwired", "salt": SALT, "x": b"\x05", "seed": SEED}
    args.update(change)
    with pytest.raises(error, match=match):
        gc.garble(args["circuit"], args["mode"], args["salt"], args["x"], args["seed"])


@pytest.mark.parametrize("mode", ["hardwired", "labels"])
def test_evaluate_errors(mode):
    res = gc.garble(CIRCUIT, mode, SALT, b"\x05", SEED)
    good = {"circuit": CIRCUIT, "mode": mode, "salt": SALT, "tables": res["tables"], "decode": res["decode"],
            "key_active": active_keys(res, [1, 0, 0]), "data_active": res["data_active"]}

    def run(**change):
        args = dict(good, **change)
        return gc.evaluate(args["circuit"], args["mode"], args["salt"], args["tables"], args["decode"],
                           args["key_active"], args["data_active"])

    _, _, gates, outputs = guard(2, 3, [(AND, 0, 2), (AND, 1, 3), (XOR, 5, 6)], [7])
    assert gc.unpack_bits(run(), 1) == plain(gates, outputs, [1, 0, 0], [1, 0, 1]) == [1]
    with pytest.raises(ValueError, match="tables: expected"):
        run(tables=res["tables"][:-16])
    with pytest.raises(ValueError, match="tables: expected"):
        run(tables=res["tables"] + bytes(16))
    with pytest.raises(ValueError, match="tables: expected"):
        run(tables=res["tables"][:-1])
    with pytest.raises(ValueError, match="key_active: expected 3 entries"):
        run(key_active=good["key_active"][:-16])
    with pytest.raises(ValueError, match="key_active"):
        run(key_active=good["key_active"] + b"\x00")
    with pytest.raises(ValueError, match="data_active: expected"):
        run(data_active=bytes(16) if mode == "hardwired" else res["data_active"][:-16])
    with pytest.raises(ValueError, match="take 1 bytes"):
        run(decode=b"")
    with pytest.raises(ValueError, match="nonzero padding"):
        run(decode=bytes([res["decode"][0] | 2]))
    with pytest.raises(ValueError, match="salt must be 56 bytes"):
        run(salt=SALT[:-1])
    with pytest.raises(TypeError, match="tables must be bytes"):
        run(tables=list(res["tables"]))


# ------------------------------------------------------------------ the backend operations

def test_the_interface_declares_the_operations():
    assert {"gc_garble", "gc_evaluate"} <= interface.Backend.__abstractmethods__


def test_backend_garbles_with_one_fresh_seed_per_call(monkeypatch):
    drawn = []

    def take(nbytes):
        drawn.append(nbytes)
        return SEED

    with native_backend.backend("python"):
        ops = native_backend.ops()
        monkeypatch.setattr(native_backend, "_random_override", take)
        for mode in ("hardwired", "labels"):
            res = ops.gc_garble(CIRCUIT, mode, SALT, b"\x05")
            assert res == gc.garble(CIRCUIT, mode, SALT, b"\x05", SEED)
            key_active = active_keys(res, [1, 1, 0])
            out = ops.gc_evaluate(CIRCUIT, mode, SALT, res["tables"], res["decode"], key_active, res["data_active"])
            assert out == gc.evaluate(CIRCUIT, mode, SALT, res["tables"], res["decode"], key_active,
                                      res["data_active"])
        assert drawn == [32, 32]
        monkeypatch.setattr(native_backend, "_random_override", lambda nbytes: SEED[:nbytes - 1])
        with pytest.raises(ValueError, match="seed must be 32 bytes"):
            ops.gc_garble(CIRCUIT, "hardwired", SALT, b"\x05")
        monkeypatch.setattr(native_backend, "_random_override", None)
        first, second = (ops.gc_garble(CIRCUIT, "hardwired", SALT, b"\x05") for _ in range(2))
        assert first["delta"] != second["delta"] and first["key0"] != second["key0"]
        assert first["tables"] != second["tables"]
        with pytest.raises(TypeError, match="packed data bits"):
            ops.gc_garble(CIRCUIT, "hardwired", SALT, [1, 0, 1])
