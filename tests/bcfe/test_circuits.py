"""pyfe4ai.circuits: the canonical encoding ``CIRC v1`` and its rejection rules,
the digest, the guard bit, wire kinds and costs, the builder's folding and
word arithmetic, the layouts and the library circuits (0.6 P1 plan,
sections 3.1-3.3)."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import struct

import numpy as np
import pytest

from pyfe4ai.circuits import (
    AND,
    INV,
    XOR,
    Builder,
    Circuit,
    DataLayout,
    Field,
    KeyLayout,
    Layout,
    OutputLayout,
    ParamCircuit,
    argmax_scores,
    linear_acc_bits,
    linear_score,
    linear_threshold,
    mlp,
)
from pyfe4ai.core.exceptions import FEValidationError

# -- helpers


def _bits(value: int, width: int) -> list:
    return [(value >> i) & 1 for i in range(width)]


def _signed(bits) -> int:
    v = sum(b << i for i, b in enumerate(bits))
    return v - (1 << len(bits)) if bits[-1] else v


def _unsigned(bits) -> int:
    return sum(b << i for i, b in enumerate(bits))


def _wrap(value: int, width: int) -> int:
    """*value* as a signed *width*-bit number (two's complement wrap)."""
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def _range(width: int) -> tuple:
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _samples(width: int, rng: random.Random, count: int = 24) -> list:
    """Boundary values of a signed *width*-bit word, then random ones."""
    lo, hi = _range(width)
    edge = sorted({lo, lo + 1, -1, 0, 1, hi - 1, hi} & set(range(lo, hi + 1)))
    return edge + [rng.randint(lo, hi) for _ in range(count)]


class Compiled:
    """A builder circuit over signed key words and data words that returns one
    word; constant result bits are kept aside (a circuit output cannot be
    constant)."""

    def __init__(self, key_widths, data_widths, fn):
        self.key_widths, self.data_widths = list(key_widths), list(data_widths)
        b = Builder(max(1, sum(self.key_widths)), sum(self.data_widths))
        keys, pos = [], 0
        for w in self.key_widths:
            keys.append(b.key_word(pos, w))
            pos += w
        datas, pos = [], 0
        for w in self.data_widths:
            datas.append(b.data_word(pos, w))
            pos += w
        word = fn(b, keys, datas)
        if isinstance(word, int):
            word = [word]
        self.width = len(word)
        self.consts, self.wires = {}, []
        for i, h in enumerate(word):
            if b.is_const(h):
                self.consts[i] = b.const_value(h)
            else:
                self.wires.append(i)
                b.output(h)
        assert self.wires, "every result bit is constant"
        self.circuit = b.build()

    def run(self, key_rows, data_rows) -> list:
        """The result bits for each input row (rows of signed values)."""
        w_rows = [sum((_bits(v, w) for v, w in zip(row, self.key_widths)), []) or [0] for row in key_rows]
        x_rows = [sum((_bits(v, w) for v, w in zip(row, self.data_widths)), []) for row in data_rows]
        outs = self.circuit.evaluate_plain_batch(w_rows, x_rows)
        results = []
        for j in range(len(key_rows)):
            bits = [0] * self.width
            for i, v in self.consts.items():
                bits[i] = v
            for k, i in enumerate(self.wires):
                bits[i] = outs[k][j]
            results.append(bits)
        return results


def _pairs(wa: int, wb: int, rng: random.Random) -> list:
    """All pairs of boundary values plus random pairs."""
    sa, sb = _samples(wa, rng, 0), _samples(wb, rng, 0)
    pairs = [(a, b) for a in sa for b in sb]
    la, ha = _range(wa)
    lb, hb = _range(wb)
    pairs += [(rng.randint(la, ha), rng.randint(lb, hb)) for _ in range(60)]
    return pairs


WIDTHS = [(1, 1), (1, 5), (5, 1), (2, 3), (4, 4), (7, 3), (3, 8), (8, 8)]

# -- CIRC v1: encoding, digest, rejection rules

#: m = 2, n = 1: g3 = XOR(w0, x0), g4 = AND(g3, w1), g5 = INV(g4); outputs (g5, g3).
TINY = (2, 1, [(XOR, 0, 2), (AND, 3, 1), (INV, 4, 0)], [5, 3])
#: Its CIRC v1 encoding, written out field by field from the plan's section 3.1.
TINY_ENCODING = bytes.fromhex(
    "5046453643495243" "01" "02000000" "01000000" "03000000" "02000000"
    "01" "00000000" "02000000"
    "02" "03000000" "01000000"
    "03" "04000000" "00000000"
    "05000000" "03000000")
#: hashlib.shake_256(b"PYFE4AI-V06-BCFE-CIRCUIT" + TINY_ENCODING).hexdigest(32), computed
#: once (2026-11-19) and pinned so that a change of the encoding or the domain shows.
TINY_DIGEST = "59926c9131ba56048c7c0ad9b1702e2ca3c829f9313b318ce69891c74d44ca98"


def test_tiny_circuit_encoding_and_digest_are_pinned():
    c = Circuit(*TINY)
    assert c.encode() == TINY_ENCODING
    assert c.digest().hex() == TINY_DIGEST
    assert c.digest() == hashlib.shake_256(b"PYFE4AI-V06-BCFE-CIRCUIT" + TINY_ENCODING).digest(32)
    assert (c.m, c.n, c.n_gates, c.n_out, c.n_wires) == (2, 1, 3, 2, 6)
    # the op names are accepted too and give the same circuit
    assert Circuit(2, 1, [("XOR", 0, 2), ("AND", 3, 1), ("INV", 4, 0)], [5, 3]) == c


def test_tiny_circuit_evaluates():
    c = Circuit(*TINY)
    for w0 in (0, 1):
        for w1 in (0, 1):
            for x0 in (0, 1):
                g3 = w0 ^ x0
                assert c.evaluate_plain([w0, w1], [x0]) == [1 - (g3 & w1), g3]


def _random_circuit(rng: random.Random, m: int, n: int, n_gates: int, n_out: int) -> Circuit:
    gates = []
    for i in range(n_gates):
        out = m + n + i
        op = rng.choice((XOR, AND, INV)) if out >= 2 else INV
        a = rng.randrange(out)
        if op == INV:
            gates.append((op, a, 0))
        else:
            b = rng.randrange(out - 1)
            gates.append((op, a, b if b < a else b + 1))
    return Circuit(m, n, gates, [rng.randrange(m + n + n_gates) for _ in range(n_out)])


def test_encode_decode_round_trip():
    rng = random.Random(7)
    circuits = [Circuit(*TINY), linear_threshold(3, 4, 3).circuit, Circuit(1, 0, [], [0])]
    circuits += [_random_circuit(rng, rng.randint(1, 5), rng.randint(0, 5), rng.randint(0, 40), rng.randint(1, 4))
                 for _ in range(30)]
    for c in circuits:
        data = c.encode()
        d = Circuit.decode(data)
        assert d == c and d.encode() == data and d.digest() == c.digest()
        assert (d.m, d.n, d.gates, d.outputs) == (c.m, c.n, c.gates, c.outputs)
        assert Circuit.decode(bytearray(data)) == c
        assert pickle.loads(pickle.dumps(c)) == c
        assert hash(d) == hash(c)


def _encode_raw(m, n, gates, outputs, magic=b"PFE6CIRC", version=1, n_gates=None, n_out=None) -> bytes:
    out = magic + struct.pack("<BIIII", version, m, n, len(gates) if n_gates is None else n_gates,
                              len(outputs) if n_out is None else n_out)
    for op, a, b in gates:
        out += struct.pack("<BII", op, a, b)
    return out + b"".join(struct.pack("<I", o) for o in outputs)


def test_raw_encoder_of_the_tests_matches():
    assert _encode_raw(*TINY) == TINY_ENCODING


@pytest.mark.parametrize("data,what", [
    (b"PFE6CIRX" + TINY_ENCODING[8:], "magic"),
    (TINY_ENCODING[:8] + b"\x02" + TINY_ENCODING[9:], "version"),
    (TINY_ENCODING[:8] + b"\x00" + TINY_ENCODING[9:], "version"),
    (b"", "short"),
    (TINY_ENCODING[:24], "short"),
    (TINY_ENCODING[:-1], "length"),
    (TINY_ENCODING + b"\x00", "length"),
    (TINY_ENCODING[:-4], "length"),
    (_encode_raw(2, 1, [(XOR, 0, 2)], [3], n_gates=2), "length"),
    (_encode_raw(2, 1, [(XOR, 0, 2)], [3], n_out=2), "length"),
    (_encode_raw(2, 1, [(XOR, 0, 2)], [3], n_gates=0xFFFFFFFF), "length"),
    (_encode_raw(2, 1, [(0, 0, 2)], [3]), "op"),
    (_encode_raw(2, 1, [(4, 0, 2)], [3]), "op"),
    (_encode_raw(2, 1, [(255, 0, 2)], [3]), "op"),
    (_encode_raw(2, 1, [(INV, 0, 1)], [3]), "INV b"),
    (_encode_raw(2, 1, [(INV, 2, 2)], [3]), "INV b"),
    (_encode_raw(2, 1, [(XOR, 3, 0)], [3]), "a not below"),
    (_encode_raw(2, 1, [(AND, 0, 3)], [3]), "b not below"),
    (_encode_raw(2, 1, [(AND, 0, 4)], [3]), "b not below"),
    (_encode_raw(2, 1, [(INV, 3, 0)], [3]), "INV a not below"),
    (_encode_raw(2, 1, [(XOR, 0, 2), (AND, 3, 4)], [4]), "b not below (second gate)"),
    (_encode_raw(2, 1, [(XOR, 1, 1)], [3]), "XOR a == b"),
    (_encode_raw(2, 1, [(AND, 2, 2)], [3]), "AND a == b"),
    (_encode_raw(2, 1, [(XOR, 0, 2)], [4]), "output out of range"),
    (_encode_raw(2, 1, [(XOR, 0, 2)], [3, 0xFFFFFFFF]), "output out of range"),
    (_encode_raw(2, 0, [], [2]), "output out of range"),
    (_encode_raw(0, 1, [], [0]), "m = 0"),
    (_encode_raw(0, 2, [(XOR, 0, 1)], [2]), "m = 0"),
    (_encode_raw(2, 1, [(XOR, 0, 2)], []), "n_out = 0"),
    (_encode_raw((1 << 31) - 1, 1, [], [0]), "2^31 wires"),
    (_encode_raw(1 << 30, 1 << 30, [], [0]), "2^31 wires"),
    (_encode_raw(0xFFFFFFFF, 0xFFFFFFFF, [], [0]), "2^31 wires"),
])
def test_decode_rejects(data, what):
    with pytest.raises(ValueError):
        Circuit.decode(data)


def test_decode_rejects_non_bytes():
    for data in (TINY_ENCODING.hex(), None, list(TINY_ENCODING)):
        with pytest.raises(ValueError):
            Circuit.decode(data)


def test_the_largest_valid_wire_count_is_accepted():
    # m + n + n_gates = 2^31 - 1 decodes (only the header counts, no gate)
    c = Circuit.decode(_encode_raw((1 << 31) - 2, 1, [], [5]))
    assert c.n_wires == (1 << 31) - 1


@pytest.mark.parametrize("args", [
    (0, 1, [], [0]),
    (2, 1, [(XOR, 0, 2)], []),
    (2, 1, [(XOR, 0, 0)], [3]),
    (2, 1, [(AND, 3, 0)], [3]),
    (2, 1, [(INV, 0, 1)], [3]),
    (2, 1, [(7, 0, 1)], [3]),
    (2, 1, [("NAND", 0, 1)], [3]),
    (2, 1, [(XOR, 0, 2)], [4]),
    (2, 1, [(XOR, 0)], [3]),
    (2, 1, [(XOR, -1, 0)], [3]),
    (2, 1, [(XOR, True, 0)], [3]),
    (-1, 1, [], [0]),
    (2.0, 1, [], [0]),
])
def test_constructor_applies_the_same_rules(args):
    with pytest.raises(ValueError):
        Circuit(*args)


def test_circuit_is_immutable():
    c = Circuit(*TINY)
    with pytest.raises(AttributeError):
        c._m = 3
    with pytest.raises(AttributeError):
        c.m = 3
    assert isinstance(c.gates, tuple) and isinstance(c.outputs, tuple)


# -- guard


def test_guard_rewrites_as_specified():
    g = Circuit(*TINY).guard()
    # old wires: 0, 1 stay; x0 = 2 -> 3; g3 -> 4, g4 -> 5, g5 -> 6; g = 2, ng = 7
    assert (g.m, g.n, g.n_out) == (3, 1, 2)
    assert g.gates == ((XOR, 0, 3), (AND, 4, 1), (INV, 5, 0), (INV, 2, 0), (AND, 6, 7), (AND, 4, 7))
    assert g.outputs == (8, 9)


def test_guard_of_outputs_on_input_wires():
    c = Circuit(2, 2, [], [1, 2, 0, 3])  # outputs: key 1, data 0, key 0, data 1
    g = c.guard()
    # g = 2, the data wires move to 3, 4; ng = m' + n + 0 = 5
    assert g.gates == ((INV, 2, 0), (AND, 1, 5), (AND, 3, 5), (AND, 0, 5), (AND, 4, 5))
    assert g.outputs == (6, 7, 8, 9)


def test_guard_semantics_on_random_inputs():
    rng = random.Random(11)
    for c in [Circuit(*TINY), linear_score(3, 4, 3, shift=1).circuit] + [
            _random_circuit(rng, rng.randint(1, 6), rng.randint(0, 6), rng.randint(1, 60), rng.randint(1, 5))
            for _ in range(20)]:
        g = c.guard()
        assert (g.m, g.n, g.n_out, g.n_gates) == (c.m + 1, c.n, c.n_out, c.n_gates + 1 + c.n_out)
        ws = [[rng.randint(0, 1) for _ in range(c.m)] for _ in range(40)]
        xs = [[rng.randint(0, 1) for _ in range(c.n)] for _ in range(40)]
        plain = c.evaluate_plain_batch(ws, xs)
        assert g.evaluate_plain_batch([w + [0] for w in ws], xs) == plain
        assert g.evaluate_plain_batch([w + [1] for w in ws], xs) == [[0] * 40 for _ in range(c.n_out)]
        # every output of the guarded circuit is labelled in both modes
        for mode in ("hardwired", "labels"):
            kinds = g.kinds(mode)
            assert all(kinds[o] == "L" for o in g.outputs)


# -- kinds and stats


#: m = 1, n = 2: g3 = XOR(x0, x1), g4 = AND(g3, w0), g5 = INV(x0), g6 = AND(g5, x1),
#: g7 = AND(g4, g6), g8 = XOR(g7, w0); output g8.
MIXED = (1, 2, [(XOR, 1, 2), (AND, 3, 0), (INV, 1, 0), (AND, 5, 2), (AND, 4, 6), (XOR, 7, 0)], [8])


def test_kinds_and_stats_of_a_known_circuit():
    c = Circuit(*MIXED)
    assert c.kinds("hardwired") == ("L", "K", "K", "K", "L", "K", "K", "L", "L")
    assert c.kinds("labels") == ("L",) * 9
    assert c.stats("hardwired") == {"and_ll": 0, "and_lk": 2, "free": 1, "garbler_only": 3, "table_entries": 2}
    assert c.stats("labels") == {"and_ll": 3, "and_lk": 0, "free": 3, "garbler_only": 0, "table_entries": 6}
    assert c.stats() == c.stats("hardwired")
    t = Circuit(*TINY)
    assert t.stats("hardwired") == {"and_ll": 1, "and_lk": 0, "free": 2, "garbler_only": 0, "table_entries": 2}
    # INV's b = 0 is not an input: INV of a K wire stays K although wire 0 is L
    assert Circuit(1, 1, [(INV, 1, 0)], [0]).kinds("hardwired") == ("L", "K", "K")
    with pytest.raises(FEValidationError):
        c.kinds("hardware")
    with pytest.raises(FEValidationError):
        c.stats("label")


def _kinds_reference(c: Circuit, mode: str) -> tuple:
    kinds = ["L"] * c.m + ["K" if mode == "hardwired" else "L"] * c.n
    for op, a, b in c.gates:
        ins = [kinds[a]] if op == INV else [kinds[a], kinds[b]]
        kinds.append("L" if "L" in ins else "K")
    return tuple(kinds)


def test_kinds_are_structural_and_do_not_depend_on_x():
    """I12: the kinds and the table size are a function of the encoding and the
    mode; the builder never sees a value, so building twice gives the same
    circuit, and the kinds equal an independent recomputation from the gates."""
    for mul in ("booth", "shift_add"):
        c1 = linear_threshold(4, 5, 4, mul=mul).circuit
        c2 = linear_threshold(4, 5, 4, mul=mul).circuit
        assert c1.encode() == c2.encode()
        g = c1.guard()
        d = Circuit.decode(g.encode())
        for mode in ("hardwired", "labels"):
            assert g.kinds(mode) == d.kinds(mode) == _kinds_reference(g, mode)
            assert g.stats(mode) == d.stats(mode)
            s = g.stats(mode)
            assert s["table_entries"] == 2 * s["and_ll"] + s["and_lk"]
            assert s["and_ll"] + s["and_lk"] + s["free"] + s["garbler_only"] == g.n_gates
        # data wires are garbler-known in hardwired mode: the Booth flags cost nothing there
        hw = g.stats("hardwired")
        assert hw["and_lk"] > 0
        if mul == "booth":
            assert hw["garbler_only"] > 0


# -- builder: folding


def test_builder_folds_constants_and_trivial_gates():
    b = Builder(2, 1)
    w0, w1, x0 = b.key(0), b.key(1), b.data(0)
    Z, O = b.zero, b.one
    assert (Z, O) == (Builder.ZERO, Builder.ONE) == (b.const(0), b.const(1))
    assert b.is_const(Z) and b.is_const(O) and not b.is_const(w0)
    assert (b.const_value(Z), b.const_value(O), b.const_value(w0)) == (0, 1, None)
    assert b.xor(w0, Z) == w0 and b.xor(Z, w0) == w0
    assert b.and_(w0, O) == w0 and b.and_(O, w0) == w0
    assert b.and_(w0, Z) == Z and b.and_(Z, w0) == Z
    assert b.xor(O, O) == Z and b.xor(O, Z) == O and b.and_(O, O) == O
    assert b.inv(Z) == O and b.inv(O) == Z
    assert b.xor(w0, w0) == Z
    assert b.and_(w0, w0) == w0
    assert b.n_gates == 0
    nw0 = b.inv(w0)
    assert b.n_gates == 1
    assert b.inv(nw0) == w0 and b.inv(w0) == nw0
    assert b.xor(w0, O) == nw0
    assert b.xor(w0, nw0) == O and b.xor(nw0, w0) == O
    assert b.and_(w0, nw0) == Z and b.and_(nw0, w0) == Z
    assert b.n_gates == 1
    # structural hashing: the same gate is made once, inputs in either order
    g = b.xor(w0, x0)
    assert b.xor(x0, w0) == g
    a = b.and_(w1, x0)
    assert b.and_(x0, w1) == a
    assert b.n_gates == 3
    assert b.or_(w0, Z) == w0 and b.or_(w0, O) == O
    assert b.mux(O, w0, w1) == w0 and b.mux(Z, w0, w1) == w1 and b.mux(x0, w0, w0) == w0


def test_builder_refuses_constant_outputs_and_foreign_handles():
    b = Builder(1, 1)
    for h in (b.zero, b.one, b.xor(b.key(0), b.key(0))):
        with pytest.raises(FEValidationError):
            b.output(h)
    with pytest.raises(FEValidationError):
        b.build()  # no output yet
    for bad in (5, -3, 1.0, True, None, "0"):
        with pytest.raises(FEValidationError):
            b.xor(b.key(0), bad)
    for bad in (-1, 1, True):
        with pytest.raises(FEValidationError):
            b.key(bad)
    with pytest.raises(FEValidationError):
        b.data(1)
    with pytest.raises(FEValidationError):
        b.const(2)
    for args in ((0, 0), (1, -1), (True, 0)):
        with pytest.raises(FEValidationError):
            Builder(*args)


def test_build_drops_unused_gates_and_keeps_order():
    b = Builder(2, 1)
    w0, w1, x0 = b.key(0), b.key(1), b.data(0)
    unused = b.and_(w0, w1)
    t = b.xor(w0, x0)
    b.inv(unused)
    o = b.and_(t, w1)
    b.output(o)
    b.output(w1)
    full = b.build(prune=False)
    assert full.n_gates == 4
    pruned = b.build()
    # (the builder writes the inputs of XOR and AND in increasing order)
    assert pruned.gates == ((XOR, 0, 2), (AND, 1, 3))
    assert pruned.outputs == (4, 1)
    for w in ([0, 0], [0, 1], [1, 0], [1, 1]):
        for x in ([0], [1]):
            assert pruned.evaluate_plain(w, x) == full.evaluate_plain(w, x)


# -- builder: word arithmetic against Python integers


@pytest.mark.parametrize("wa,wb", WIDTHS)
def test_add_sub_neg(wa, wb):
    rng = random.Random(wa * 31 + wb)
    width = max(wa, wb)
    for cin_kind in ("zero", "one", "data"):
        def fn(b, keys, datas, cin_kind=cin_kind):
            a, x = b.sign_extend(keys[0], width), b.sign_extend(datas[0], width)
            cin = {"zero": b.zero, "one": b.one, "data": datas[1][0]}[cin_kind]
            return b.add(a, x, cin) + b.sub(a, x) + b.neg(a) + b.neg(x)
        c = Compiled([wa], [wb, 1], fn)
        pairs = _pairs(wa, wb, rng)
        cins = [rng.randint(0, 1) for _ in pairs]
        res = c.run([[a] for a, _ in pairs], [[x, -ci] for (_, x), ci in zip(pairs, cins)])
        for (a, x), ci, bits in zip(pairs, cins, res):
            cv = {"zero": 0, "one": 1, "data": ci}[cin_kind]
            words = [bits[i * width:(i + 1) * width] for i in range(4)]
            assert _signed(words[0]) == _wrap(a + x + cv, width)
            assert _signed(words[1]) == _wrap(a - x, width)
            assert _signed(words[2]) == _wrap(-a, width)
            assert _signed(words[3]) == _wrap(-x, width)


def test_extend_resize_and_shifts():
    rng = random.Random(3)
    for w in (1, 2, 5, 8):
        def fn(b, keys, datas, w=w):
            a = keys[0]
            return (b.sign_extend(a, w + 3) + b.zero_extend(a, w + 2) + b.resize(a, max(1, w - 1))
                    + b.resize(a, w + 1, signed=False) + b.shift_right(a, w - 1) + b.shift_left(a, 1)
                    + b.invert(a))
        c = Compiled([w], [], fn)
        vals = _samples(w, rng)
        for v, bits in zip(vals, c.run([[v] for v in vals], [[] for _ in vals])):
            pos = 0
            parts = []
            for width in (w + 3, w + 2, max(1, w - 1), w + 1, 1, w, w):
                parts.append(bits[pos:pos + width])
                pos += width
            u = v & ((1 << w) - 1)
            assert _signed(parts[0]) == v
            assert _unsigned(parts[1]) == u
            assert _signed(parts[2]) == _wrap(v, max(1, w - 1))
            assert _unsigned(parts[3]) == u
            assert _signed(parts[4]) == v >> (w - 1)
            assert _signed(parts[5]) == _wrap(v * 2, w)
            assert _signed(parts[6]) == _wrap(-v - 1, w)
    b = Builder(4, 0)
    a = b.key_word(0, 4)
    for call in (lambda: b.sign_extend(a, 3), lambda: b.shift_right(a, 4), lambda: b.shift_right(a, -1),
                 lambda: b.add(a, a[:3]), lambda: b.mux_word(a[0], a, a[:2]), lambda: b.resize([], 2),
                 lambda: b.const_word(8, 4), lambda: b.const_word(-1, 4, signed=False)):
        with pytest.raises(FEValidationError):
            call()
    assert b.const_word(-8, 4) == [b.zero, b.zero, b.zero, b.one]
    assert b.const_word(15, 4, signed=False) == [b.one] * 4


@pytest.mark.parametrize("wa,wb", WIDTHS)
def test_multiplications_are_exact_and_agree(wa, wb):
    rng = random.Random(wa * 17 + wb)
    pairs = _pairs(wa, wb, rng)
    exact = wa + wb
    narrow = max(1, exact - 3)
    wide = exact + 2
    results = {}
    for method in ("booth", "shift_add"):
        def fn(b, keys, datas, method=method):
            a, x = keys[0], datas[0]
            return b.mul(a, x, method=method) + b.mul(a, x, narrow, method) + b.mul(a, x, wide, method)
        c = Compiled([wa], [wb], fn)
        res = c.run([[a] for a, _ in pairs], [[x] for _, x in pairs])
        for (a, x), bits in zip(pairs, res):
            assert _signed(bits[:exact]) == a * x
            assert _signed(bits[exact:exact + narrow]) == _wrap(a * x, narrow)
            assert _signed(bits[exact + narrow:]) == a * x
        results[method] = res
    assert results["booth"] == results["shift_add"]
    # the named helpers are the same circuits
    for named, method in (("mul_booth", "booth"), ("mul_shift_add", "shift_add")):
        c1 = Compiled([wa], [wb], lambda b, k, d, named=named: getattr(b, named)(k[0], d[0]))
        c2 = Compiled([wa], [wb], lambda b, k, d, method=method: b.mul(k[0], d[0], method=method))
        assert c1.circuit == c2.circuit


@pytest.mark.parametrize("wa,wb", [(1, 1), (4, 4), (7, 3), (3, 8)])
def test_mac_accumulates_modulo_its_width(wa, wb):
    rng = random.Random(wa + 100 * wb)
    acc_w = wa + wb - 1  # narrower than a product: wraps
    for method in ("booth", "shift_add"):
        c = Compiled([acc_w, wa], [wb], lambda b, k, d, method=method: b.mac(k[0], k[1], d[0], method))
        rows = [(rng.randint(*_range(acc_w)), a, x) for a, x in _pairs(wa, wb, rng)]
        res = c.run([[r[0], r[1]] for r in rows], [[r[2]] for r in rows])
        for (acc, a, x), bits in zip(rows, res):
            assert _signed(bits) == _wrap(acc + a * x, acc_w)
    b = Builder(2, 1)
    with pytest.raises(FEValidationError):
        b.mac([b.key(0)], [b.key(1)], [b.data(0)], "karatsuba")


@pytest.mark.parametrize("method", ["booth", "shift_add"])
def test_dot_and_sum_words(method):
    rng = random.Random(5)
    n, wa, wb = 5, 4, 3
    width = linear_acc_bits(n, wa, wb)

    def fn(b, keys, datas):
        return (b.dot(keys[:n], datas, width, init=keys[n], method=method)
                + b.sum_words(keys[:n] + datas, wa + wb))

    c = Compiled([wa] * n + [wa + wb], [wb] * n, fn)
    rows = []
    for t in range(80):
        if t < 4:  # extremes: every product and the bias at a bound
            a = [_range(wa)[t % 2]] * n
            x = [_range(wb)[t // 2]] * n
            bias = _range(wa + wb)[t % 2]
        else:
            a = [rng.randint(*_range(wa)) for _ in range(n)]
            x = [rng.randint(*_range(wb)) for _ in range(n)]
            bias = rng.randint(*_range(wa + wb))
        rows.append((a, x, bias))
    res = c.run([a + [bias] for a, _, bias in rows], [x for _, x, _ in rows])
    for (a, x, bias), bits in zip(rows, res):
        assert _signed(bits[:width]) == bias + sum(p * q for p, q in zip(a, x))
        assert _signed(bits[width:]) == _wrap(sum(a) + sum(x), wa + wb)
    b = Builder(2, 1)
    with pytest.raises(FEValidationError):
        b.dot([[b.key(0)]], [], 4)
    with pytest.raises(FEValidationError):
        b.dot([], [], 4)
    with pytest.raises(FEValidationError):
        b.sum_words([], 4)


@pytest.mark.parametrize("wa,wb", WIDTHS)
def test_comparisons_relu_and_mux(wa, wb):
    rng = random.Random(wa * 7 + wb)
    width = max(wa, wb)

    def fn(b, keys, datas):
        a, x, s = keys[0], datas[0], datas[1][0]
        return ([b.lt(a, x), b.ge(a, x), b.lt(x, a), b.ge_zero(a), b.ge_zero(x)] + b.relu(a)
                + b.mux_word(s, b.sign_extend(a, width), b.sign_extend(x, width)) + [b.or_(a[0], x[0])])

    c = Compiled([wa], [wb, 1], fn)
    pairs = _pairs(wa, wb, rng)
    sel = [rng.randint(0, 1) for _ in pairs]
    res = c.run([[a] for a, _ in pairs], [[x, -s] for (_, x), s in zip(pairs, sel)])
    for (a, x), s, bits in zip(pairs, sel, res):
        assert bits[:5] == [int(a < x), int(a >= x), int(x < a), int(a >= 0), int(x >= 0)]
        assert _signed(bits[5:5 + wa]) == max(a, 0)
        assert _signed(bits[5 + wa:5 + wa + width]) == (a if s else x)
        assert bits[-1] == (a & 1) | (x & 1)
    # ReLU's top bit folds to the constant 0
    b = Builder(3, 0)
    assert b.relu(b.key_word(0, 3))[-1] == b.zero


@pytest.mark.parametrize("k,width", [(1, 3), (2, 1), (3, 4), (5, 3), (8, 2)])
def test_argmax_gives_the_lowest_index_of_the_maximum(k, width):
    rng = random.Random(k * 13 + width)
    c = Compiled([width] * k, [], lambda b, keys, datas: list(b.argmax(keys)[0]) + b.argmax(keys)[1])
    ibits = max(1, (k - 1).bit_length())
    lo, hi = _range(width)
    rows = [[rng.randint(lo, hi) for _ in range(k)] for _ in range(150)]
    rows += [[hi] * k, [lo] * k]
    res = c.run(rows, [[] for _ in rows])
    for row, bits in zip(rows, res):
        assert _unsigned(bits[:ibits]) == row.index(max(row))
        assert _signed(bits[ibits:]) == max(row)


# -- layouts


def test_layout_bit_order_and_round_trip():
    layout = KeyLayout([Field("weights", 3, 4, True), Field("bias", 1, 6, True), Field("flags", 2, 2, False)])
    assert layout.n_bits == 3 * 4 + 6 + 4 and layout.names == ("weights", "bias", "flags")
    assert (layout.offset("weights"), layout.offset("bias"), layout.offset("flags")) == (0, 12, 18)
    values = {"weights": [-8, 7, -1], "bias": 5, "flags": [3, 0]}
    bits = layout.encode(values)
    assert bits == _bits(-8, 4) + _bits(7, 4) + _bits(-1, 4) + _bits(5, 6) + _bits(3, 2) + _bits(0, 2)
    assert layout.decode(bits) == values
    # NumPy arrays and scalars, tuples, a one-element list for a one-element field
    alt = {"weights": np.array([-8, 7, -1], dtype=np.int16), "bias": [np.int64(5)], "flags": (np.uint8(3), 0)}
    assert layout.encode(alt) == bits
    assert layout.encode({**values, "bias": np.array(5)}) == bits
    for cls in (KeyLayout, DataLayout, OutputLayout):
        lay = cls([Field("v", 2, 3, False)])
        d = lay.to_dict()
        assert json.loads(json.dumps(d)) == d and d["kind"] == cls.KIND
        assert cls.from_dict(d) == lay and Layout.from_dict(d) == lay and type(Layout.from_dict(d)) is cls
        assert pickle.loads(pickle.dumps(lay)) == lay
    assert DataLayout([]).n_bits == 0 and DataLayout([]).encode({}) == []


@pytest.mark.parametrize("values", [
    {"w": [8, 0], "u": 0},          # signed 4-bit max is 7
    {"w": [0, -9], "u": 0},         # signed 4-bit min is -8
    {"w": [0, 0], "u": 16},         # unsigned 4-bit max is 15
    {"w": [0, 0], "u": -1},
    {"w": [0, 0], "u": 12345678},
    {"w": np.array([0, 300]), "u": 0},
])
def test_layout_refuses_out_of_range_values_without_naming_them(values):
    layout = KeyLayout([Field("w", 2, 4, True), Field("u", 1, 4, False)])
    with pytest.raises(FEValidationError) as info:
        layout.encode(values)
    text = str(info.value)
    assert "range" in text
    for secret in ("12345678", "300", "-9", "16"):
        assert secret not in text


@pytest.mark.parametrize("values", [
    {"w": [0, 0]},                               # missing field
    {"w": [0, 0], "u": 0, "z": 1},               # unknown field
    {"w": [0], "u": 0},                          # too short
    {"w": [0, 0, 0], "u": 0},                    # too long
    {"w": 0, "u": 0},                            # scalar for a two-element field
    {"w": [[0, 0]], "u": 0},                     # nested
    {"w": np.zeros((1, 2), dtype=int), "u": 0},  # two-dimensional
    {"w": np.zeros(2), "u": 0},                  # float array
    {"w": [0, 0.0], "u": 0},                     # float element
    {"w": [0, True], "u": 0},                    # bool element
    {"w": [0, np.bool_(True)], "u": 0},
    {"w": [0, "1"], "u": 0},
    {"w": [0, 0], "u": [0, 0]},                  # two values for a one-element field
])
def test_layout_refuses_wrong_names_and_shapes(values):
    layout = KeyLayout([Field("w", 2, 4, True), Field("u", 1, 4, False)])
    with pytest.raises(FEValidationError):
        layout.encode(values)


def test_layout_decode_and_description_rules():
    layout = OutputLayout([Field("score", 1, 5, True)])
    assert layout.decode([1, 1, 1, 1, 1]) == {"score": -1}
    assert layout.decode([True, False, False, False, False]) == {"score": 1}
    for bits in ([0] * 4, [0] * 6, [0, 0, 2, 0, 0], [0, 0, 0.5, 0, 0], b"\x00" * 5, "00000"):
        with pytest.raises(FEValidationError):
            layout.decode(bits)
    with pytest.raises(FEValidationError):
        layout.encode([1])
    for bad in (
        {"kind": "output", "fields": [{"name": "s", "count": 1, "bits": 5}]},
        {"kind": "output", "fields": [{"name": "s", "count": 1, "bits": 5, "signed": True, "x": 1}]},
        {"kind": "output", "fields": [{"name": "s", "count": True, "bits": 5, "signed": True}]},
        {"kind": "output", "fields": [{"name": "s", "count": 1, "bits": 5, "signed": 1}]},
        {"kind": "key", "fields": [{"name": "s", "count": 1, "bits": 5, "signed": True}]},
        {"kind": "secret", "fields": []},
        {"kind": "output", "fields": ()},
        {"kind": "output"},
    ):
        with pytest.raises(FEValidationError):
            OutputLayout.from_dict(bad)
    for args in (("", 1, 1), ("1x", 1, 1), ("a b", 1, 1), ("x" * 65, 1, 1), ("x", 0, 1), ("x", 1, 0),
                 ("x", 1, 1, "yes"), ("x", 1.0, 1)):
        with pytest.raises(FEValidationError):
            Field(*args)
    with pytest.raises(FEValidationError):
        KeyLayout([Field("a", 1, 1), Field("a", 1, 2)])
    with pytest.raises(FEValidationError):
        KeyLayout([])
    with pytest.raises(FEValidationError):
        KeyLayout([("a", 1, 1, True)])
    with pytest.raises(TypeError):
        Layout([Field("a", 1, 1)])
    with pytest.raises(AttributeError):
        layout._fields = ()


# -- library


def _random_linear(rng, n, bw, bx):
    w = [rng.randint(*_range(bw)) for _ in range(n)]
    x = [rng.randint(*_range(bx)) for _ in range(n)]
    bias = rng.randint(*_range(bw + bx))
    return w, x, bias


@pytest.mark.parametrize("mul", ["booth", "shift_add"])
@pytest.mark.parametrize("n,bw,bx", [(1, 1, 1), (1, 3, 2), (3, 4, 3), (6, 5, 4), (5, 2, 7)])
def test_linear_circuits_against_the_formula(n, bw, bx, mul):
    rng = random.Random(n * 1000 + bw * 10 + bx)
    acc_bits = linear_acc_bits(n, bw, bx)
    assert acc_bits == bw + bx + n.bit_length()
    thr = linear_threshold(n, bw, bx, mul=mul)
    shifts = sorted({0, 1, acc_bits - 1})
    scores = {s: linear_score(n, bw, bx, shift=s, mul=mul) for s in shifts}
    cases = [_random_linear(rng, n, bw, bx) for _ in range(30)]
    lo_w, hi_w = _range(bw)
    lo_x, hi_x = _range(bx)
    lo_b, hi_b = _range(bw + bx)
    cases += [([lo_w] * n, [lo_x] * n, hi_b), ([hi_w] * n, [lo_x] * n, lo_b), ([0] * n, [0] * n, 0),
              ([0] * n, [0] * n, -1), ([lo_w] * n, [hi_x] * n, lo_b)]
    for w, x, bias in cases:
        total = sum(a * b for a, b in zip(w, x)) + bias
        wd, xd = {"weights": w, "bias": bias}, {"x": x}
        assert thr.evaluate(wd, xd) == {"decision": int(total >= 0)}
        assert thr.circuit.evaluate_plain(thr.key_layout.encode(wd), thr.data_layout.encode(xd)) == [int(total >= 0)]
        for s, pc in scores.items():
            assert pc.evaluate(wd, xd) == {"score": total >> s}
            assert pc.output_layout.field("score").bits == acc_bits - s
    # NumPy inputs give the same result
    w, x, bias = cases[0]
    assert thr.evaluate({"weights": np.array(w), "bias": np.int32(bias)}, {"x": np.array(x, dtype=np.int8)}) == \
        thr.evaluate({"weights": w, "bias": bias}, {"x": x})


def test_linear_circuits_of_the_plan_size():
    """linear_threshold / linear_score(100, 16, 16): Booth and shift-add agree
    with the formula on random inputs (bit-sliced evaluation, 24 inputs at once)."""
    rng = random.Random(2026)
    n, bw, bx = 100, 16, 16
    cases = [_random_linear(rng, n, bw, bx) for _ in range(22)]
    cases += [([-(1 << 15)] * n, [-(1 << 15)] * n, (1 << 31) - 1), ([-(1 << 15)] * n, [(1 << 15) - 1] * n, -(1 << 31))]
    for mul in ("booth", "shift_add"):
        for pc in (linear_threshold(n, bw, bx, mul=mul), linear_score(n, bw, bx, shift=7, mul=mul)):
            assert pc.params["acc_bits"] == 39
            ws = [pc.key_layout.encode({"weights": w, "bias": b}) for w, _, b in cases]
            xs = [pc.data_layout.encode({"x": x}) for _, x, _ in cases]
            outs = pc.circuit.evaluate_plain_batch(ws, xs)
            for j, (w, x, b) in enumerate(cases):
                total = sum(p * q for p, q in zip(w, x)) + b
                got = pc.output_layout.decode([outs[k][j] for k in range(pc.circuit.n_out)])
                expected = {"decision": int(total >= 0)} if pc.name == "linear_threshold" else {"score": total >> 7}
                assert got == expected


def test_library_layouts_description_and_guard():
    pc = linear_score(4, 6, 5, shift=2)
    assert pc.name == "linear_score"
    assert pc.params == {"n": 4, "bw": 6, "bx": 5, "shift": 2, "mul": "booth", "acc_bits": 14}
    assert pc.key_layout == KeyLayout([Field("weights", 4, 6, True), Field("bias", 1, 11, True)])
    assert pc.data_layout == DataLayout([Field("x", 4, 5, True)])
    assert pc.output_layout == OutputLayout([Field("score", 1, 12, True)])
    assert (pc.circuit.m, pc.circuit.n, pc.circuit.n_out) == (4 * 6 + 11, 20, 12)
    d = pc.describe()
    assert json.loads(pc.describe_json()) == d
    assert pc.describe_json() == json.dumps(d, sort_keys=True, separators=(",", ":"))
    assert d["circuit"]["digest"] == pc.circuit.digest().hex()
    assert KeyLayout.from_dict(d["key_layout"]) == pc.key_layout
    # the library circuits have no guard bit: the guarded circuit has one key bit more
    assert pc.circuit.guard().m == pc.key_layout.n_bits + 1
    p2 = pickle.loads(pickle.dumps(pc))
    assert p2.describe() == d
    params = pc.params
    params["n"] = 99
    assert pc.params["n"] == 4
    with pytest.raises(AttributeError):
        pc._name = "x"
    with pytest.raises(FEValidationError):
        ParamCircuit("x", {}, pc.circuit, pc.key_layout, pc.data_layout, OutputLayout([Field("s", 1, 3)]))
    with pytest.raises(FEValidationError):
        ParamCircuit("x", {"f": object()}, pc.circuit, pc.key_layout, pc.data_layout, pc.output_layout)


@pytest.mark.parametrize("call", [
    lambda: linear_threshold(0, 4, 4),
    lambda: linear_threshold(2, 0, 4),
    lambda: linear_threshold(2, 4, 65),
    lambda: linear_threshold(2, 4, 4, mul="karatsuba"),
    lambda: linear_threshold(2.0, 4, 4),
    lambda: linear_score(2, 4, 4, shift=-1),
    lambda: linear_score(2, 4, 4, shift=linear_acc_bits(2, 4, 4)),
])
def test_library_parameter_checks(call):
    with pytest.raises(FEValidationError):
        call()


def _mlp_formula(w1, b1, w2, b2, x, n, hidden, bh, shift) -> int:
    hs = []
    for k in range(hidden):
        z = b1[k] + sum(w1[k * n + i] * x[i] for i in range(n))
        hs.append(min(max(z, 0) >> shift, (1 << bh) - 1))
    return int(b2 + sum(v * h for v, h in zip(w2, hs)) >= 0)


@pytest.mark.parametrize("mul", ["booth", "shift_add"])
@pytest.mark.parametrize("n,hidden,bw,bx,bh,shift", [(3, 2, 4, 3, 3, 2), (2, 3, 3, 4, 2, 0), (1, 1, 2, 2, 5, 0),
                                                     (4, 2, 3, 3, 1, 4)])
def test_mlp_against_the_formula(n, hidden, bw, bx, bh, shift, mul):
    rng = random.Random(n * 100 + hidden * 10 + bh)
    pc = mlp(n, hidden, bw, bx, bh, shift, mul=mul)
    assert pc.circuit.guard().m == pc.key_layout.n_bits + 1
    lo_w, hi_w = _range(bw)
    cases = []
    for t in range(60):
        w1 = [rng.randint(lo_w, hi_w) for _ in range(hidden * n)]
        b1 = [rng.randint(*_range(bw + bx)) for _ in range(hidden)]
        w2 = [rng.randint(lo_w, hi_w) for _ in range(hidden)]
        # small output biases keep both decisions likely; extremes make the activations saturate
        b2 = rng.randint(-(1 << bh), 1 << bh) if t % 3 else rng.randint(*_range(bw + bh + 1))
        x = [rng.randint(*_range(bx)) for _ in range(n)]
        cases.append((w1, b1, w2, b2, x))
    cases.append(([hi_w] * (hidden * n), [_range(bw + bx)[1]] * hidden, [lo_w] * hidden, _range(bw + bh + 1)[1],
                  [_range(bx)[1]] * n))
    seen = set()
    for w1, b1, w2, b2, x in cases:
        expected = _mlp_formula(w1, b1, w2, b2, x, n, hidden, bh, shift)
        seen.add(expected)
        assert pc.evaluate({"w1": w1, "b1": b1, "w2": w2, "b2": b2}, {"x": x}) == {"decision": expected}
    assert seen == {0, 1}


@pytest.mark.parametrize("mul", ["booth", "shift_add"])
@pytest.mark.parametrize("k,n,bw,bx", [(2, 3, 4, 3), (3, 2, 3, 3), (5, 1, 2, 2), (4, 3, 1, 4)])
def test_argmax_scores_against_the_formula(k, n, bw, bx, mul):
    rng = random.Random(k * 100 + n * 10 + bw)
    pc = argmax_scores(k, n, bw, bx, mul=mul)
    assert pc.output_layout.field("index").bits == max(1, (k - 1).bit_length())
    cases = []
    for t in range(80):
        w = [rng.randint(*_range(bw)) for _ in range(k * n)]
        bias = [rng.randint(*_range(bw + bx)) for _ in range(k)]
        if t % 4 == 0:
            bias = [bias[0]] * k  # ties are likely
            w = w[:n] * k if t % 8 == 0 else w
        cases.append((w, bias, [rng.randint(*_range(bx)) for _ in range(n)]))
    for w, bias, x in cases:
        scores = [bias[j] + sum(w[j * n + i] * x[i] for i in range(n)) for j in range(k)]
        assert pc.evaluate({"weights": w, "bias": bias}, {"x": x}) == {"index": scores.index(max(scores))}


@pytest.mark.parametrize("call", [
    lambda: mlp(2, 2, 3, 3, 2, -1),
    lambda: mlp(2, 2, 3, 3, 2, linear_acc_bits(2, 3, 3) - 1),
    lambda: mlp(2, 0, 3, 3, 2, 0),
    lambda: mlp(2, 2, 3, 3, 0, 0),
    lambda: argmax_scores(1, 2, 3, 3),
    lambda: argmax_scores(2, 2, 3, 3, mul="x"),
])
def test_optional_library_parameter_checks(call):
    with pytest.raises(FEValidationError):
        call()
