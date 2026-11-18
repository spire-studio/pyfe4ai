"""The native half-gates garbling (``gc_garble`` / ``gc_evaluate`` of
pyfe4ai._rust_native, rust/core/src/garble.rs) gives the circuit's values and
the bytes of the specification and of the Python reference
(:mod:`pyfe4ai.primitives.garbling`).

The circuits are random ``CIRC v1`` encodings built here, independently of
the circuit builder; the key holder's active labels are ``key0[i] xor w_i
Delta``. Skipped unless the Rust extension is built.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct

import pytest

from pyfe4ai import backend as native_backend

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(not native_backend.available(), reason="native extension not built"),
]

MODES = ("hardwired", "labels")
CODES = {"hardwired": 0, "labels": 1}
KEYS = ("delta", "key0", "data_active", "tables", "decode")
XOR, AND, INV = 1, 2, 3


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def encode(m: int, n: int, gates: list, outputs: list) -> bytes:
    enc = b"PFE6CIRC" + bytes([1]) + struct.pack("<4I", m, n, len(gates), len(outputs))
    enc += b"".join(struct.pack("<BII", op, a, b) for op, a, b in gates)
    return enc + struct.pack("<%dI" % len(outputs), *outputs)


def random_circuit(rng, m: int, n: int, n_gates: int, n_out: int, mode: str):
    """``(encoding, gates, outputs)`` of a random circuit whose outputs are
    labelled wires in *mode* (then also in mode ``"labels"``); XOR and AND
    gates read two different wires, as ``CIRC v1`` requires."""
    gates, labelled = [], [True] * m + [mode == "labels"] * n
    for i in range(n_gates):
        o = m + n + i
        op = rng.choice((XOR, AND, INV)) if o > 1 else INV
        a = rng.randrange(o)
        b = 0 if op == INV else (a + rng.randrange(1, o)) % o  # XOR and AND read two different wires
        gates.append((op, a, b))
        labelled.append(labelled[a] if op == INV else labelled[a] or labelled[b])
    pool = [v for v, is_l in enumerate(labelled) if is_l]
    outputs = [rng.choice(pool) for _ in range(n_out)]
    return encode(m, n, gates, outputs), gates, outputs


def plain(gates: list, outputs: list, w: list, x: list) -> list:
    v = list(w) + list(x)
    for op, a, b in gates:
        v.append(v[a] ^ v[b] if op == XOR else v[a] & v[b] if op == AND else 1 - v[a])
    return [v[o] for o in outputs]


def pack(bits) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        out[i // 8] |= bit << (i % 8)
    return bytes(out)


def unpack(data: bytes, n: int) -> list:
    return [(data[i // 8] >> (i % 8)) & 1 for i in range(n)]


def label(data: bytes, i: int) -> int:
    return int.from_bytes(data[16 * i:16 * i + 16], "little")


def key_active(g: dict, w: list) -> bytes:
    delta = int.from_bytes(g["delta"], "little")
    return b"".join((label(g["key0"], i) ^ (delta if bit else 0)).to_bytes(16, "little") for i, bit in enumerate(w))


def native_ops():
    from pyfe4ai.backend import native_impl

    ops = native_backend.ops()
    assert isinstance(ops, native_impl.NativeBackend)
    return ops


@pytest.mark.parametrize("mode", MODES)
def test_evaluating_the_garbling_gives_the_circuit_in_the_clear(mode):
    rng = random.Random(1)
    with native_backend.backend("native"):
        ops = native_ops()
        for _ in range(40):
            m, n = rng.randrange(1, 9), rng.randrange(0, 9)
            enc, gates, outputs = random_circuit(rng, m, n, rng.randrange(0, 300), rng.randrange(1, 12), mode)
            salt = rng.randbytes(56)
            for _ in range(3):
                w = [rng.randrange(2) for _ in range(m)]
                x = [rng.randrange(2) for _ in range(n)]
                g = ops.gc_garble(enc, mode, salt, pack(x))
                assert sorted(g) == sorted(KEYS) and all(type(v) is bytes for v in g.values())
                assert len(g["delta"]) == 16 and g["delta"][0] & 1 == 1
                assert len(g["key0"]) == 16 * m
                assert len(g["data_active"]) == (16 * n if mode == "labels" else 0)
                assert len(g["tables"]) % 16 == 0 and len(g["decode"]) == (len(outputs) + 7) // 8
                y = ops.gc_evaluate(enc, mode, salt, g["tables"], g["decode"], key_active(g, w), g["data_active"])
                assert unpack(y, len(outputs)) == plain(gates, outputs, w, x)
                assert y == pack(unpack(y, len(outputs)))  # padding bits zero


def test_the_tables_are_those_of_the_specification():
    """m = 2, n = 1 (hardwired): AND(0, 2) -> 3 is a generator half-gate (index 0),
    AND(3, 1) -> 4 a full AND (indices 1, 2); checked with hashlib alone."""
    enc = encode(2, 1, [(AND, 0, 2), (AND, 3, 1)], [4, 3])
    salt, seed = bytes(range(56)), bytes(range(100, 132))
    ext = native_backend.ext()
    for x in (0, 1):
        g = dict(zip(KEYS, ext.gc_garble(enc, 0, salt, bytes([x]), seed=seed)))
        stream = hashlib.shake_256(b"PYFE4AI-V06-BCFE-GBPRG" + seed).digest(48)
        assert g["delta"] == bytes([stream[0] | 1]) + stream[1:16] and g["key0"] == stream[16:48]
        assert g["data_active"] == b""
        d = int.from_bytes(g["delta"], "little")
        w0, w1 = label(g["key0"], 0), label(g["key0"], 1)

        def h(v, idx):
            msg = b"PFE6GC02" + salt + idx.to_bytes(8, "little") + v.to_bytes(16, "little")
            assert len(msg) == 88
            return int.from_bytes(hashlib.shake_128(msg).digest(16), "little")

        t = h(w0, 0) ^ h(w0 ^ d, 0) ^ (d if x else 0)
        w3 = h(w0, 0) ^ (t if w0 & 1 else 0)
        tg = h(w3, 1) ^ h(w3 ^ d, 1) ^ (d if w1 & 1 else 0)
        te = h(w1, 2) ^ h(w1 ^ d, 2) ^ w3
        w4 = h(w3, 1) ^ (tg if w3 & 1 else 0) ^ h(w1, 2) ^ ((te ^ w3) if w1 & 1 else 0)
        assert g["tables"] == b"".join(v.to_bytes(16, "little") for v in (t, tg, te))
        assert g["decode"] == pack([w4 & 1, w3 & 1])
        for w in ([0, 0], [1, 0], [0, 1], [1, 1]):
            y = ext.gc_evaluate(enc, 0, salt, g["tables"], g["decode"], key_active(g, w), b"")
            assert unpack(y, 2) == [w[0] & x & w[1], w[0] & x]


def test_a_seed_determines_the_garbling(monkeypatch):
    rng = random.Random(2)
    enc, _, _ = random_circuit(rng, 6, 5, 200, 4, "hardwired")
    salt, x = rng.randbytes(56), pack([1, 0, 1, 1, 0])
    seed, other = rng.randbytes(32), rng.randbytes(32)
    ext = native_backend.ext()
    for mode in MODES:
        code = CODES[mode]
        first = ext.gc_garble(enc, code, salt, x, seed=seed)
        assert ext.gc_garble(enc, code, salt, x, seed=seed) == first
        changed = ext.gc_garble(enc, code, salt, x, seed=other)
        assert all(changed[i] != first[i] for i in (0, 1, 3))
        fresh = [ext.gc_garble(enc, code, salt, x) for _ in range(2)]  # seeds from the OS
        assert fresh[0][0] != fresh[1][0] and fresh[0][3] != fresh[1][3]
        # the backend operation reads its seed from the tests' hook: 32 bytes, once
        reads = []
        monkeypatch.setattr(native_backend, "_random_override", lambda k: reads.append(k) or seed)
        with native_backend.backend("native"):
            assert native_ops().gc_garble(enc, mode, salt, x) == dict(zip(KEYS, first))
        monkeypatch.setattr(native_backend, "_random_override", None)
        assert reads == [32]


def _reference():
    try:
        from pyfe4ai.primitives import garbling
    except ImportError:
        pytest.skip("the Python reference pyfe4ai.primitives.garbling is not there yet")
    return garbling


@pytest.mark.parametrize("mode", MODES)
def test_native_is_the_reference_byte_for_byte(mode, monkeypatch):
    garbling = _reference()
    rng = random.Random(3 + CODES[mode])
    ext = native_backend.ext()
    for trial in range(30):
        m, n = rng.randrange(1, 10), rng.randrange(0, 10)
        enc, _, outputs = random_circuit(rng, m, n, rng.randrange(0, 250), rng.randrange(1, 10), mode)
        salt, seed = rng.randbytes(56), rng.randbytes(32)
        x = pack([rng.randrange(2) for _ in range(n)])
        ref = garbling.garble(enc, mode, salt, x, seed)
        assert dict(zip(KEYS, ext.gc_garble(enc, CODES[mode], salt, x, seed=seed))) == ref, trial
        ka = key_active(ref, [rng.randrange(2) for _ in range(m)])
        args = (ref["tables"], ref["decode"], ka, ref["data_active"])
        assert ext.gc_evaluate(enc, CODES[mode], salt, *args) == garbling.evaluate(enc, mode, salt, *args)
        # the two backends' operations on the same test bytes
        got = []
        for name in ("python", "native"):
            monkeypatch.setattr(native_backend, "_random_override", lambda k: seed)
            with native_backend.backend(name):
                got.append(native_backend.ops().gc_garble(enc, mode, salt, x))
            monkeypatch.setattr(native_backend, "_random_override", None)
        assert got[0] == got[1] == ref


def test_bad_inputs_are_refused():
    rng = random.Random(4)
    enc, _, _ = random_circuit(rng, 3, 4, 60, 3, "hardwired")
    salt, x = rng.randbytes(56), pack([1, 0, 0, 1])
    ext = native_backend.ext()
    with native_backend.backend("native"):
        ops = native_ops()
        g = ops.gc_garble(enc, "hardwired", salt, x)
        ka = key_active(g, [1, 1, 0])
        bad_circuits = [
            enc[:-1], enc + b"\0", b"X" + enc[1:], enc[:8] + b"\x02" + enc[9:], enc[:24],
            encode(0, 1, [], [0]), encode(1, 0, [], []), encode(1, 1, [(4, 0, 1)], [2]),
            encode(1, 1, [(INV, 0, 1)], [2]), encode(1, 1, [(AND, 0, 2)], [2]), encode(1, 1, [], [2]),
            encode(1, 1, [(XOR, 0, 0)], [2]), encode(1, 1, [(AND, 1, 1)], [2]),  # a = b
            encode(1, 2, [(XOR, 1, 2)], [3]),  # an output of kind K in mode hardwired
        ]
        for bad in bad_circuits:
            with pytest.raises(ValueError):
                ops.gc_garble(bad, "hardwired", salt, b"\x01")
        for bad_args in [("garbled", salt, x), ("hardwired", salt[:55], x), ("hardwired", salt + b"\0", x),
                         ("hardwired", salt, x + b"\0"), ("hardwired", salt, b""), ("hardwired", salt, b"\x10")]:
            with pytest.raises(ValueError):
                ops.gc_garble(enc, *bad_args)
        for bad_args in [
            (g["tables"][:-16], g["decode"], ka, b""), (g["tables"] + bytes(16), g["decode"], ka, b""),
            (g["tables"][:-1], g["decode"], ka, b""), (g["tables"], b"", ka, b""), (g["tables"], b"\xf8", ka, b""),
            (g["tables"], g["decode"], ka[16:], b""), (g["tables"], g["decode"], ka + bytes(16), b""),
            (g["tables"], g["decode"], ka, bytes(16 * 4)),  # no data labels in mode hardwired
        ]:
            with pytest.raises(ValueError):
                ops.gc_evaluate(enc, "hardwired", salt, *bad_args)
        with pytest.raises(ValueError):
            ops.gc_evaluate(enc, "labels", salt, g["tables"], g["decode"], ka, b"")
        with pytest.raises(TypeError):
            ops.gc_garble(enc.decode("latin-1"), "hardwired", salt, x)
        with pytest.raises(TypeError):
            ops.gc_garble(enc, "hardwired", salt, [1, 0, 0, 1])
        assert ops.gc_garble(bytearray(enc), "hardwired", memoryview(salt), bytearray(x))["key0"]
    for seed in (bytes(31), bytes(33)):
        with pytest.raises(ValueError):
            ext.gc_garble(enc, 0, salt, x, seed=seed)
    with pytest.raises(ValueError):
        ext.gc_garble(enc, 2, salt, x)
    with pytest.raises(ValueError):
        ext.gc_evaluate(enc, 2, salt, g["tables"], g["decode"], ka, b"")
