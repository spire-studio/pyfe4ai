"""The native label KEM of the bounded-key schemes (``bcfe_*`` of
pyfe4ai._rust_native, rust/native/src/bcfe.rs) against the specification of
plan 0.6 P1, sections 3.8 and 3.9, and byte for byte against the Python
reference (:mod:`pyfe4ai.primitives.bcfe_kem`) through both backends on the
same test bytes.

The circuits are random ``CIRC v1`` encodings of ``guard(C)`` built here
(section 3.2), independently of the circuit builder. Skipped unless the Rust
extension is built.
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
XOR, AND, INV = 1, 2, 3
#: The order of ristretto255.
L = 2 ** 252 + 27742317777372353535851937790883648493
IDENTITY = bytes(32)
UNSEL_DST = b"PYFE4AI-V06-BCFE-UNSEL-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_"


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


# ------------------------------------------------------------------ circuits and bits

def encode(m: int, n: int, gates: list, outputs: list) -> bytes:
    enc = b"PFE6CIRC" + bytes([1]) + struct.pack("<4I", m, n, len(gates), len(outputs))
    enc += b"".join(struct.pack("<BII", op, a, b) for op, a, b in gates)
    return enc + struct.pack("<%dI" % len(outputs), *outputs)


def random_c(rng, m: int, n: int, n_gates: int, n_out: int):
    """``(gates, outputs)`` of a random C(w, x); XOR and AND read two different wires."""
    gates = []
    for i in range(n_gates):
        o = m + n + i
        op = rng.choice((XOR, AND, INV)) if o > 1 else INV
        a = rng.randrange(o)
        gates.append((op, a, 0 if op == INV else (a + rng.randrange(1, o)) % o))
    return gates, [rng.randrange(m + n + n_gates) for _ in range(n_out)]


def guard(m: int, n: int, gates: list, outputs: list) -> bytes:
    """The encoding of guard(C) (section 3.2): g is key wire m."""
    def move(v):
        return v if v < m else v + 1

    new = [(op, move(a), 0 if op == INV else move(b)) for op, a, b in gates]
    ng = m + 1 + n + len(new)
    new.append((INV, m, 0))
    new += [(AND, move(o), ng) for o in outputs]
    return encode(m + 1, n, new, [ng + 1 + i for i in range(len(outputs))])


def plain(m: int, gates: list, outputs: list, w: list, x: list) -> list:
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


def chunks(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


def xor(a: bytes, b: bytes) -> bytes:
    return bytes(u ^ v for u, v in zip(a, b))


def setting(rng, m: int, n: int, n_gates: int, n_out: int):
    gates, outputs = random_c(rng, m, n, n_gates, n_out)
    return guard(m, n, gates, outputs), gates, outputs


def native_ops():
    from pyfe4ai.backend import native_impl

    ops = native_backend.ops()
    assert isinstance(ops, native_impl.NativeBackend)
    return ops


def holder_keys(rng, m: int):
    """Random canonical nonzero scalars of the holder (h2) and their public keys."""
    ext = native_backend.ext()
    scalars = [rng.randrange(1, L).to_bytes(32, "little") for _ in range(m + 1)]
    return b"".join(scalars), b"".join(ext.r255_mul_base(s) for s in scalars)


def component(enc: dict) -> tuple:
    return enc["R"], enc["label_cts"], enc["tables"], enc["decode"], enc["data_active"]


# ------------------------------------------------------------------ end to end

@pytest.mark.parametrize("mode", MODES)
def test_h1_slots_decrypt_to_the_circuit(mode):
    rng = random.Random(1)
    with native_backend.backend("native"):
        ops = native_ops()
        for _ in range(12):
            m, n = rng.randrange(1, 12), rng.randrange(0, 10)
            circuit, gates, outputs = setting(rng, m, n, rng.randrange(0, 150), rng.randrange(1, 8))
            setup, pp, nonce, slot = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16), rng.randrange(2 ** 64)
            w = [rng.randrange(2) for _ in range(m)]
            s = ops.bcfe_new_slot(setup, slot, pack(w), m=m)
            assert sorted(s) == ["k", "pks", "scalars"]
            assert (len(s["scalars"]), len(s["pks"]), len(s["k"])) == (32 * (m + 1), 32 * (2 * m + 1), 16)
            assert ops.bcfe_unselected_ok(setup, slot, pack(w), s["k"], s["pks"], m=m) is True
            for _ in range(2):
                x = [rng.randrange(2) for _ in range(n)]
                e = ops.bcfe_encrypt_slot(circuit, mode, pp, nonce, slot, pack(x), s["pks"])
                assert sorted(e) == ["R", "data_active", "decode", "label_cts", "tables"]
                assert len(e["label_cts"]) == 16 * (2 * m + 1)
                y = ops.bcfe_decrypt_slot(circuit, mode, pp, nonce, slot, *component(e), s["scalars"], pack(w))
                assert unpack(y, len(outputs)) == plain(m, gates, outputs, w, x)


@pytest.mark.parametrize("mode", MODES)
def test_h2_slots_decrypt_to_the_circuit_and_check(mode):
    rng = random.Random(2)
    with native_backend.backend("native"):
        ops = native_ops()
        for _ in range(8):
            m, n = rng.randrange(1, 12), rng.randrange(0, 8)
            circuit, gates, outputs = setting(rng, m, n, rng.randrange(0, 120), rng.randrange(1, 6))
            setup, pp, nonce, slot = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16), rng.randrange(2 ** 64)
            w = [rng.randrange(2) for _ in range(m)]
            scalars, mine = holder_keys(rng, m)
            s = ops.bcfe_place_slot(setup, slot, pack(w), mine, m=m)
            table = chunks(s["pks"], 32)
            assert [table[2 * j + w[j]] for j in range(m)] + [table[2 * m]] == chunks(mine, 32)
            assert ops.bcfe_unselected_ok(setup, slot, pack(w), s["k"], s["pks"], m=m) is True
            x = [rng.randrange(2) for _ in range(n)]
            e = ops.bcfe_encrypt_slot(circuit, mode, pp, nonce, slot, pack(x), s["pks"])
            y = ops.bcfe_decrypt_slot(circuit, mode, pp, nonce, slot, *component(e), scalars, pack(w))
            assert unpack(y, len(outputs)) == plain(m, gates, outputs, w, x)


def test_check_auth_refuses_other_tables():
    rng = random.Random(3)
    m = 9
    setup, slot = rng.randbytes(32), 77
    w = [rng.randrange(2) for _ in range(m)]
    _, mine = holder_keys(rng, m)
    with native_backend.backend("native"):
        ops = native_ops()
        s = ops.bcfe_place_slot(setup, slot, pack(w), mine, m=m)
        pks, k = s["pks"], s["k"]

        def ok(pks=pks, k=k, w=w, slot=slot, setup=setup):
            return ops.bcfe_unselected_ok(setup, slot, pack(w), k, pks, m=m)

        assert ok() is True
        for j in range(m):  # an unselected position replaced: refused; a selected one: not this check's business
            unselected = 2 * j + 1 - w[j]
            other = xor(pks[32 * unselected:32 * unselected + 32], b"\x01" + bytes(31))
            assert ok(pks=pks[:32 * unselected] + other + pks[32 * unselected + 32:]) is False
        swapped = b"".join(p for j in range(m) for p in reversed(chunks(pks, 32)[2 * j:2 * j + 2])) + pks[-32:]
        assert ok(pks=swapped) is False  # the holder's keys moved to the unselected positions
        assert ok(k=xor(k, b"\x80" + bytes(15))) is False
        assert ok(slot=slot + 1) is False and ok(setup=xor(setup, b"\x01" + bytes(31))) is False
        assert ok(w=[1 - w[0]] + w[1:]) is False
        assert ok(pks=pks[:-32]) is False and ok(pks=pks + pks[-32:]) is False and ok(pks=b"") is False


def test_other_instances_do_not_decrypt():
    """H_K and the gate hash take (pp_digest, ct_nonce, slot_id): a component
    decrypted with another instance gives unrelated bits."""
    rng = random.Random(4)
    m, n = 6, 4
    gates = [(XOR, i, (i + d) % (m + n)) for d in range(1, 5) for i in range(m + n)]  # 40 XORs of two inputs
    outputs = list(range(m + n, m + n + len(gates)))
    circuit = guard(m, n, gates, outputs)
    setup, pp, nonce, slot = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16), 5
    w, x = [1, 0, 1, 1, 0, 0], [0, 1, 1, 0]
    want = plain(m, gates, outputs, w, x)
    with native_backend.backend("native"):
        ops = native_ops()
        s = ops.bcfe_new_slot(setup, slot, pack(w), m=m)
        e = ops.bcfe_encrypt_slot(circuit, "labels", pp, nonce, slot, pack(x), s["pks"])
        y = ops.bcfe_decrypt_slot(circuit, "labels", pp, nonce, slot, *component(e), s["scalars"], pack(w))
        assert unpack(y, len(outputs)) == want
        others = [(xor(pp, b"\x01" + bytes(31)), nonce, slot), (pp, xor(nonce, b"\x01" + bytes(15)), slot),
                  (pp, nonce, slot + 1)]
        for pp2, nonce2, slot2 in others:
            wrong = ops.bcfe_decrypt_slot(circuit, "labels", pp2, nonce2, slot2, *component(e), s["scalars"], pack(w))
            assert unpack(wrong, len(outputs)) != want  # equal with probability 2^-40


# ------------------------------------------------------------------ the bytes of the specification

def test_new_slot_follows_the_specification():
    rng = random.Random(5)
    ext = native_backend.ext()
    m, setup, slot = 5, rng.randbytes(32), 2 ** 64 - 1
    w = [1, 0, 0, 1, 1]
    rnd = rng.randbytes(64 * (m + 1) + 16)
    scalars, pks, k = ext.bcfe_new_slot(setup, slot, m, pack(w), rnd=rnd)
    want = [(int.from_bytes(rnd[64 * j:64 * j + 64], "little") % L).to_bytes(32, "little") for j in range(m + 1)]
    assert chunks(scalars, 32) == want and k == rnd[-16:]
    table = chunks(pks, 32)
    for j in range(m):
        msg = b"PFE6-UNSEL" + setup + slot.to_bytes(8, "little") + j.to_bytes(4, "little") + k
        assert table[2 * j + w[j]] == ext.r255_mul_base(want[j])
        assert table[2 * j + 1 - w[j]] == ext.r255_hash(msg, UNSEL_DST)
    assert table[2 * m] == ext.r255_mul_base(want[m])
    # the holder's keys of h2 go to the same places
    pks2, k2 = ext.bcfe_place_slot(setup, slot, m, pack(w), b"".join(table[2 * j + w[j]] for j in range(m))
                                   + table[2 * m], rnd=k)
    assert (pks2, k2) == (pks, k)
    # threads do not change the bytes
    assert ext.bcfe_new_slot(setup, slot, m, pack(w), rnd=rnd, threads=4) == (scalars, pks, k)


@pytest.mark.parametrize("mode", MODES)
def test_encrypt_slot_follows_the_specification(mode):
    rng = random.Random(6 + CODES[mode])
    ext = native_backend.ext()
    m, n = 4, 3
    circuit, _, _ = setting(rng, m, n, 40, 3)
    pp, nonce, slot = rng.randbytes(32), rng.randbytes(16), 123456789
    x, seed = pack([1, 0, 1]), rng.randbytes(32)
    _, pks, _ = ext.bcfe_new_slot(rng.randbytes(32), slot, m, pack([0, 1, 1, 0]), rnd=rng.randbytes(64 * 5 + 16))
    big_r, cts, tables, decode, data_active = ext.bcfe_encrypt_slot(circuit, CODES[mode], pp, nonce, slot, x, pks,
                                                                     seed=seed)
    salt = pp + nonce + slot.to_bytes(8, "little")  # the instance salt, also in both seed expansions
    r = int.from_bytes(hashlib.shake_256(b"PYFE4AI-V06-BCFE-EPHEM" + salt + seed).digest(64), "little") % L
    r_enc = r.to_bytes(32, "little")
    assert big_r == ext.r255_mul_base(r_enc)
    delta, key0, g_data, g_tables, g_decode = ext.gc_garble(circuit, CODES[mode], salt, x, seed=seed)
    assert (tables, decode, data_active) == (g_tables, g_decode, g_data)
    labels = chunks(key0, 16)
    table = chunks(pks, 32)
    want = []
    for i in range(2 * m + 1):
        j, b = i // 2, i % 2
        z = ext.r255_mul(table[i], r_enc)
        msg = (b"PYFE4AI-V06-BCFE-KEM" + pp + nonce + slot.to_bytes(8, "little") + j.to_bytes(4, "little")
               + bytes([b]) + big_r + z)
        assert len(msg) == 145
        label = xor(labels[j], delta) if b else labels[j]
        want.append(xor(label, hashlib.shake_256(msg).digest(16)))
    assert chunks(cts, 16) == want
    assert ext.bcfe_encrypt_slot(circuit, CODES[mode], pp, nonce, slot, x, pks, seed=seed, threads=3) == \
        (big_r, cts, tables, decode, data_active)
    assert ext.bcfe_encrypt_slot(circuit, CODES[mode], pp, nonce, slot, x, pks)[0] != big_r  # a fresh seed


# ------------------------------------------------------------------ the reference

def _reference():
    from pyfe4ai.backend import python_impl

    if not hasattr(python_impl.PythonBackend, "bcfe_encrypt_slot"):
        pytest.skip("the Python reference of the bcfe_* operations is not there yet")
    try:
        from pyfe4ai.primitives import bcfe_kem  # noqa: F401
    except ImportError:
        pytest.skip("the Python reference pyfe4ai.primitives.bcfe_kem is not there yet")


def both(monkeypatch, fn, data: bytes):
    """``fn(ops)`` on the Python and on the native backend, each fed *data*
    through the tests' random-bytes hook; the two results."""
    out = []
    for name in ("python", "native"):
        stream = {"pos": 0}

        def take(k, stream=stream):
            stream["pos"] += k
            assert stream["pos"] <= len(data), "test bytes exhausted"
            return data[stream["pos"] - k:stream["pos"]]

        monkeypatch.setattr(native_backend, "_random_override", take)
        try:
            with native_backend.backend(name):
                out.append(fn(native_backend.ops()))
        finally:
            monkeypatch.setattr(native_backend, "_random_override", None)
    return out


@pytest.mark.parametrize("mode", MODES)
def test_native_is_the_reference_byte_for_byte(mode, monkeypatch):
    _reference()
    rng = random.Random(10 + CODES[mode])
    for trial in range(6):
        m, n = rng.randrange(1, 7), rng.randrange(0, 6)
        circuit, _, _ = setting(rng, m, n, rng.randrange(0, 60), rng.randrange(1, 5))
        setup, pp, nonce, slot = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16), rng.randrange(2 ** 64)
        w = pack([rng.randrange(2) for _ in range(m)])
        x = pack([rng.randrange(2) for _ in range(n)])
        # h1
        py, nat = both(monkeypatch, lambda ops: ops.bcfe_new_slot(setup, slot, w, m=m), rng.randbytes(64 * m + 80))
        assert py == nat, trial
        h1 = nat
        # h2
        scalars2, mine = holder_keys(rng, m)
        py, nat = both(monkeypatch, lambda ops: ops.bcfe_place_slot(setup, slot, w, mine, m=m), rng.randbytes(16))
        assert py == nat, trial
        h2 = nat
        for slot_keys, slot_scalars in ((h1, h1["scalars"]), (h2, scalars2)):
            for k, pks, want in ((slot_keys["k"], slot_keys["pks"], True),
                                 (xor(slot_keys["k"], b"\x01" + bytes(15)), slot_keys["pks"], False),
                                 (slot_keys["k"], slot_keys["pks"][:-32], False)):
                py, nat = both(monkeypatch, lambda ops: ops.bcfe_unselected_ok(setup, slot, w, k, pks, m=m), b"")
                assert py is nat is want
            py, nat = both(monkeypatch, lambda ops: ops.bcfe_encrypt_slot(circuit, mode, pp, nonce, slot, x,
                                                                           slot_keys["pks"]), rng.randbytes(32))
            assert py == nat, trial
            comp = component(nat)
            py, nat = both(monkeypatch, lambda ops: ops.bcfe_decrypt_slot(circuit, mode, pp, nonce, slot, *comp,
                                                                           slot_scalars, w), b"")
            assert py == nat, trial
            # another key's scalars: unrelated bits, the same in both
            py, nat = both(monkeypatch, lambda ops: ops.bcfe_decrypt_slot(circuit, mode, pp, nonce, slot, *comp,
                                                                           holder_keys(random.Random(trial), m)[0],
                                                                           w), b"")
            assert py == nat, trial


def test_errors_are_the_references(monkeypatch):
    """The same inputs raise the same exception type in both backends."""
    _reference()
    rng = random.Random(12)
    m, n = 3, 2
    circuit, _, _ = setting(rng, m, n, 20, 2)
    setup, pp, nonce = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16)
    w, x = pack([1, 0, 1]), pack([1, 1])
    _, mine = holder_keys(rng, m)
    with native_backend.backend("native"):
        s = native_ops().bcfe_new_slot(setup, 1, w, m=m)
        e = native_ops().bcfe_encrypt_slot(circuit, "hardwired", pp, nonce, 1, x, s["pks"])
    comp = component(e)
    cases = [
        lambda ops: ops.bcfe_new_slot(setup, 1, w, m=0),
        lambda ops: ops.bcfe_new_slot(setup, -1, w, m=m),
        lambda ops: ops.bcfe_new_slot(setup[:31], 1, w, m=m),
        lambda ops: ops.bcfe_new_slot(setup, 1, w + b"\0", m=m),
        lambda ops: ops.bcfe_new_slot(setup, 1, b"\x08", m=m),
        lambda ops: ops.bcfe_new_slot(setup, 1, [1, 0, 1], m=m),
        lambda ops: ops.bcfe_place_slot(setup, 1, w, mine[:-32], m=m),
        lambda ops: ops.bcfe_place_slot(setup, 1, w, mine[:-32] + IDENTITY, m=m),
        lambda ops: ops.bcfe_place_slot(setup, 1, w, mine[:-32] + b"\xff" * 32, m=m),
        lambda ops: ops.bcfe_place_slot(setup, 2 ** 64, w, mine, m=m),
        lambda ops: ops.bcfe_unselected_ok(setup, 1, w, s["k"][:15], s["pks"], m=m),
        lambda ops: ops.bcfe_unselected_ok(setup, 1, w, s["k"], s["pks"], m=True),
        lambda ops: ops.bcfe_encrypt_slot(encode(2, 0, [(XOR, 0, 1)], [2]), "hardwired", pp, nonce, 1, b"", s["pks"]),
        lambda ops: ops.bcfe_encrypt_slot(circuit, "garbled", pp, nonce, 1, x, s["pks"]),
        lambda ops: ops.bcfe_encrypt_slot(circuit, "hardwired", pp[:31], nonce, 1, x, s["pks"]),
        lambda ops: ops.bcfe_encrypt_slot(circuit, "hardwired", pp, nonce, 1, x, s["pks"][:-32]),
        lambda ops: ops.bcfe_encrypt_slot(circuit, "hardwired", pp, nonce, 1, x, s["pks"][:-32] + IDENTITY),
        lambda ops: ops.bcfe_encrypt_slot(circuit, "hardwired", pp, nonce, 1, x + b"\0", s["pks"]),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, IDENTITY, *comp[1:], s["scalars"], w),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, *comp,
                                          s["scalars"][:-32] + L.to_bytes(32, "little"), w),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, *comp, s["scalars"][:-32], w),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, comp[0], comp[1][:-16], *comp[2:],
                                          s["scalars"], w),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, *comp[:2], comp[2][:-16], *comp[3:],
                                          s["scalars"], w),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, *comp, s["scalars"], w + b"\0"),
        lambda ops: ops.bcfe_decrypt_slot(circuit, "hardwired", pp, nonce, 1, *comp, s["scalars"], w.decode()),
    ]
    for i, case in enumerate(cases):
        raised = []
        for name in ("python", "native"):
            monkeypatch.setattr(native_backend, "_random_override", lambda k: bytes(k))
            try:
                with native_backend.backend(name):
                    case(native_backend.ops())
            except (TypeError, ValueError) as exc:
                raised.append(type(exc))
            else:
                raised.append(None)
            finally:
                monkeypatch.setattr(native_backend, "_random_override", None)
        assert raised[0] is raised[1] is not None, (i, raised)


# ------------------------------------------------------------------ refusals (native)

def test_refusals():
    rng = random.Random(13)
    ext = native_backend.ext()
    m = 3
    circuit, _, _ = setting(rng, m, 2, 20, 2)
    setup, pp, nonce, w = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16), pack([0, 1, 1])
    _, pks, _ = ext.bcfe_new_slot(setup, 0, m, w)
    with native_backend.backend("native"):
        ops = native_ops()
        for bad_m in (0, -1, True, 2 ** 31 - 1, 1.0):
            with pytest.raises(ValueError):
                ops.bcfe_new_slot(setup, 0, w, m=bad_m)
        for bad_slot in (-1, 2 ** 64, False, "1"):
            with pytest.raises(ValueError):
                ops.bcfe_encrypt_slot(circuit, "labels", pp, nonce, bad_slot, b"\x00", pks)
        with pytest.raises(TypeError):
            ops.bcfe_new_slot(setup.hex(), 0, w, m=m)
        unguarded = encode(2, 1, [(AND, 0, 2), (XOR, 3, 1)], [4])
        with pytest.raises(ValueError, match="guard"):
            ops.bcfe_encrypt_slot(unguarded, "labels", pp, nonce, 0, b"\x00", pks[:96])
    # the g bit read by another gate; an output that is not the guard's AND
    for m_prime, gates, outputs in ((2, [(XOR, 0, 1), (INV, 1, 0), (AND, 2, 3)], [4]),
                                    (3, [(XOR, 0, 1), (INV, 2, 0), (AND, 3, 4)], [3])):
        with pytest.raises(ValueError, match="guard"):
            ext.bcfe_encrypt_slot(encode(m_prime, 0, gates, outputs), 0, pp, nonce, 0, b"",
                                  pks[:32 * (2 * m_prime - 1)])
    for bad_rnd in (bytes(64 * 4 + 15), bytes(64 * 4 + 17)):
        with pytest.raises(ValueError):
            ext.bcfe_new_slot(setup, 0, m, w, rnd=bad_rnd)
    with pytest.raises(ValueError, match="0"):  # a drawn scalar 0
        ext.bcfe_new_slot(setup, 0, m, w, rnd=bytes(64 * 4 + 16))
    with pytest.raises(ValueError):
        ext.bcfe_place_slot(setup, 0, m, w, pks[:128], rnd=bytes(15))
    with pytest.raises(ValueError):
        ext.bcfe_encrypt_slot(circuit, 0, pp, nonce, 0, b"\x00", pks, seed=bytes(31))
    with pytest.raises(ValueError):
        ext.bcfe_encrypt_slot(circuit, 2, pp, nonce, 0, b"\x00", pks)


@pytest.mark.parametrize("mode", MODES)
def test_decryption_with_the_table_for_r_is_the_reference(mode):
    """m + 1 >= 64 key wires: the native decryption multiplies R by a
    precomputed table instead of variable-base (from 64 scalars on). Its
    output, with the right key and with another one, is the reference's,
    and the right key gives the circuit's value."""
    _reference()
    from pyfe4ai.primitives import bcfe_kem

    rng = random.Random(30 + CODES[mode])
    ext = native_backend.ext()
    m, n = 70, 4
    gates, outputs = random_c(rng, m, n, 60, 6)
    circuit = guard(m, n, gates, outputs)
    setup, pp, nonce, slot = rng.randbytes(32), rng.randbytes(32), rng.randbytes(16), 9
    w = [rng.randrange(2) for _ in range(m)]
    x = [rng.randrange(2) for _ in range(n)]
    scalars, pks, _ = ext.bcfe_new_slot(setup, slot, m, pack(w), rnd=rng.randbytes(64 * (m + 1) + 16))
    seed = rng.randbytes(32)
    comp = ext.bcfe_encrypt_slot(circuit, CODES[mode], pp, nonce, slot, pack(x), pks, seed=seed)
    assert comp == tuple(bcfe_kem.encrypt_slot(circuit, mode, pp, nonce, slot, pack(x), pks, seed)[k]
                         for k in ("R", "label_cts", "tables", "decode", "data_active"))
    other = b"".join(rng.randrange(1, L).to_bytes(32, "little") for _ in range(m + 1))
    for key, right in ((scalars, True), (other, False)):
        for threads in (1, 4):
            nat = ext.bcfe_decrypt_slot(circuit, CODES[mode], pp, nonce, slot, *comp, key, pack(w), threads=threads)
            assert nat == bcfe_kem.decrypt_slot(circuit, mode, pp, nonce, slot, *comp, key, pack(w))
            if right:
                assert unpack(nat, len(outputs)) == plain(m, gates, outputs, w, x)
