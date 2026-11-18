"""The label KEM and the slot public keys of ``bcfe`` (:mod:`pyfe4ai.primitives.bcfe_kem`)
and the backend operations ``bcfe_*`` of the Python backend.

The circuits are encoded by hand (``CIRC v1``) with the helpers below, so
that these tests do not depend on :mod:`pyfe4ai.circuits`; ``guard`` is the
transformation of plan 0.6 P1 section 3.2. The known-answer values are the
anchors of the cross-language tests: each was computed once with this
module (``python -c`` on the inputs written next to it) and checked against
an independent evaluation of the formula of section 3.8 in the test itself.
"""

from __future__ import annotations

import hashlib
import random
import struct

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.backend import interface
from pyfe4ai.primitives import bcfe_kem as kem
from pyfe4ai.primitives import garbling as gc
from pyfe4ai.primitives.groups import ristretto255 as r255

XOR, AND, INV = gc.OP_XOR, gc.OP_AND, gc.OP_INV
L = r255.L
MODES = ("hardwired", "labels")
OPERATIONS = {"bcfe_new_slot", "bcfe_place_slot", "bcfe_unselected_ok", "bcfe_encrypt_slot", "bcfe_decrypt_slot"}

SETUP = bytes(range(128, 160))
PP = bytes(range(32))
NONCE = bytes(range(32, 48))
SEED = bytes(range(100, 132))


# ------------------------------------------------------------------ helpers

def encode(m, n, gates, outputs) -> bytes:
    """``CIRC v1`` of the gates ``(op, a, b)``."""
    out = struct.pack("<8sBIIII", b"PFE6CIRC", 1, m, n, len(gates), len(outputs))
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
        if op != INV:
            b = rng.randrange(o - 1)
            b += b >= a
        gates.append((op, a, b))
    return gates, [rng.randrange(m + n + count) for _ in range(rng.randint(1, 4))]


class Circuit:
    """``C`` (``m`` key bits, ``n`` data bits) and the encoding of ``guard(C)``."""

    def __init__(self, m, n, gates, outputs) -> None:
        self.m, self.n, self.gates, self.outputs = m, n, gates, outputs
        self.guarded = encode(*guard(m, n, gates, outputs))

    def __call__(self, w, x) -> list:
        return plain(self.gates, self.outputs, w, x)


# key wires 0..2, data wires 3..5; wire 6 = w0 & x0, 7 = w1 ^ x1, 8 = 6 & w2, 9 = 7 ^ x2
SMALL = Circuit(3, 3, [(AND, 0, 3), (XOR, 1, 4), (AND, 6, 2), (XOR, 7, 5)], [8, 9, 6])


def source(seed):
    """A deterministic ``rand(n)``."""
    return random.Random(seed).randbytes


def recording(rand, log: list):
    def take(n):
        log.append(n)
        return rand(n)

    return take


def split(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


def xor(a: bytes, b: bytes) -> bytes:
    return bytes(u ^ v for u, v in zip(a, b))


def point(k: int) -> bytes:
    return r255.encode(r255.mul_base(k))


def true_labels(circuit: bytes, mode: str, slot_id: int, x, seed, pp=PP, nonce=NONCE) -> tuple:
    """``(key0, delta)`` of the garbling an encryption with *seed* makes."""
    res = gc.garble(circuit, mode, kem.instance_salt(pp, nonce, slot_id), x, seed)
    return split(res["key0"], 16), res["delta"]


def active(key0, delta, w) -> list:
    """The active labels of ``w || 0``."""
    return [xor(key0[j], delta) if b else key0[j] for j, b in enumerate(w)] + [key0[len(w)]]


def roundtrip(circuit: Circuit, mode, slot, w, x, slot_id=7, seed=SEED, pp=PP, nonce=NONCE) -> list:
    ct = kem.encrypt_slot(circuit.guarded, mode, pp, nonce, slot_id, x, slot["pks"], seed)
    y = kem.decrypt_slot(circuit.guarded, mode, pp, nonce, slot_id, ct["R"], ct["label_cts"], ct["tables"],
                         ct["decode"], ct["data_active"], slot["scalars"], w)
    return gc.unpack_bits(y, len(circuit.outputs))


# ------------------------------------------------------------------ the hashes of section 3.8

def test_kem_mask_layout_and_known_answer():
    r, z = bytes(range(48, 80)), bytes(range(80, 112))
    slot_id = 0x0102030405060708
    mask = kem.kem_mask(PP, NONCE, slot_id, 5, 1, r, z)
    direct = hashlib.shake_256(b"PYFE4AI-V06-BCFE-KEM" + PP + NONCE + slot_id.to_bytes(8, "little")
                               + (5).to_bytes(4, "little") + b"\x01" + r + z).digest(16)
    assert mask == direct
    # KAT: kem_mask(bytes(range(32)), bytes(range(32, 48)), 0x0102030405060708, 5, 1,
    #               bytes(range(48, 80)), bytes(range(80, 112)))
    assert mask.hex() == "e437c47b4a69c1f123f0048e15845973"
    # KAT: the same with slot_id 0, j = 3, b = 0 (a g bit for m = 3)
    assert kem.kem_mask(PP, NONCE, 0, 3, 0, r, z).hex() == "094538e2253514b6668d475068082518"
    for form in (bytearray, memoryview):
        assert kem.kem_mask(form(PP), form(NONCE), slot_id, 5, 1, form(r), form(z)) == mask


def test_kem_mask_changes_with_every_field():
    base = {"pp_digest": PP, "ct_nonce": NONCE, "slot_id": 9, "j": 4, "b": 0, "R": point(2), "Z": point(3)}
    first = kem.kem_mask(**base)
    flip = lambda data, i: data[:i] + bytes([data[i] ^ 1]) + data[i + 1:]  # noqa: E731
    changes = [("pp_digest", flip(PP, 0)), ("pp_digest", flip(PP, 31)), ("ct_nonce", flip(NONCE, 15)),
               ("slot_id", 8), ("slot_id", 9 + 2 ** 63), ("j", 5), ("j", 4 + 2 ** 31), ("b", 1),
               ("R", point(4)), ("Z", point(5)), ("R", base["Z"]), ("Z", base["R"])]
    masks = {first}
    for name, value in changes:
        masks.add(kem.kem_mask(**dict(base, **{name: value})))
    assert len(masks) == len(changes) + 1


@pytest.mark.parametrize("change, error, match", [
    ({"pp_digest": PP[:-1]}, ValueError, "pp_digest must be 32 bytes"),
    ({"ct_nonce": NONCE + b"\x00"}, ValueError, "ct_nonce must be 16 bytes"),
    ({"R": bytes(31)}, ValueError, "R must be 32 bytes"),
    ({"Z": bytes(33)}, ValueError, "Z must be 32 bytes"),
    ({"slot_id": 2 ** 64}, ValueError, "slot_id"),
    ({"slot_id": -1}, ValueError, "slot_id"),
    ({"slot_id": True}, ValueError, "slot_id"),
    ({"j": 2 ** 32}, ValueError, "j must be"),
    ({"b": 2}, ValueError, "b must be 0 or 1"),
    ({"b": True}, ValueError, "b must be 0 or 1"),
    ({"b": 1.0}, ValueError, "b must be 0 or 1"),
    ({"pp_digest": PP.hex()}, TypeError, "pp_digest must be bytes"),
])
def test_kem_mask_errors(change, error, match):
    args = dict({"pp_digest": PP, "ct_nonce": NONCE, "slot_id": 1, "j": 0, "b": 0, "R": bytes(32), "Z": bytes(32)},
                **change)
    with pytest.raises(error, match=match):
        kem.kem_mask(**args)


def test_unselected_pk_layout_and_known_answer():
    k = bytes(range(200, 216))
    pk = kem.unselected_pk(SETUP, 3, 4, k)
    msg = b"PFE6-UNSEL" + SETUP + (3).to_bytes(8, "little") + (4).to_bytes(4, "little") + k
    dst = b"PYFE4AI-V06-BCFE-UNSEL-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_"
    assert kem.UNSELECTED_DST == dst
    assert pk == r255.encode(r255.hash_to_group(msg, dst))
    # KAT: unselected_pk(bytes(range(128, 160)), 3, 4, bytes(range(200, 216)))
    assert pk.hex() == "3a45412a8637ff51e55b5ccf41308a40e0d0897f12d721a4655eae0b7887c24e"
    # KAT: unselected_pk(bytes(range(128, 160)), 0, 0, bytes(range(200, 216)))
    assert kem.unselected_pk(SETUP, 0, 0, k).hex() == "e622904145d7685f60574878eb5f7a8e14c4aef49d7aa72d0696d0c10223391e"
    assert not r255.equal(r255.decode(pk), r255.IDENTITY)
    others = {kem.unselected_pk(SETUP[::-1], 3, 4, k), kem.unselected_pk(SETUP, 2, 4, k),
              kem.unselected_pk(SETUP, 3, 5, k), kem.unselected_pk(SETUP, 3, 4, k[::-1]), pk}
    assert len(others) == 5
    with pytest.raises(ValueError, match="k must be 16 bytes"):
        kem.unselected_pk(SETUP, 3, 4, k[:-1])
    with pytest.raises(ValueError, match="setup_digest must be 32 bytes"):
        kem.unselected_pk(SETUP + b"\x00", 3, 4, k)
    with pytest.raises(ValueError, match="j must be"):
        kem.unselected_pk(SETUP, 3, -1, k)


def test_setup_digest_layout_and_known_answer():
    cd, nonce = bytes(range(160, 192)), bytes(range(64, 96))
    sid = b"bcfe/gc_ristretto255"
    digest = kem.setup_digest(sid.decode(), 1, cd, "hardwired", "shake128", "ristretto255-hashed-elgamal-v1", 128,
                              nonce)
    direct = hashlib.shake_256(b"PYFE4AI-V06-BCFE-SETUP" + len(sid).to_bytes(4, "little") + sid
                               + (1).to_bytes(4, "little") + cd + b"\x00\x00\x00" + (128).to_bytes(4, "little")
                               + nonce).digest(32)
    assert digest == direct
    # KAT: setup_digest("bcfe/gc_ristretto255", 1, bytes(range(160, 192)), "hardwired", "shake128",
    #                   "ristretto255-hashed-elgamal-v1", 128, bytes(range(64, 96)))
    assert digest.hex() == "ca63f64472cfbaf4e2db3e6c8b64d8383bfed24d13e043fd655654c5d159f21c"
    labels = kem.setup_digest(sid, 1, cd, kem.DATA_WIRES_LABELS, kem.GC_HASH_SHAKE128,
                              kem.KEM_RISTRETTO255_HASHED_ELGAMAL_V1, 128, nonce)
    # KAT: the same with data_wires "labels" (code 1)
    assert labels.hex() == "2aed66895e198e9898d227e3cf0bf417917efe2fdd6a7651bf87725364a6ebed"
    assert (kem.DATA_WIRES_HARDWIRED, kem.DATA_WIRES_LABELS, kem.GC_HASH_SHAKE128,
            kem.KEM_RISTRETTO255_HASHED_ELGAMAL_V1) == (0, 1, 0, 0)
    assert kem.setup_digest(sid, 1, cd, 0, 0, 0, 128, nonce) == digest
    base = (sid, 1, cd, "hardwired", "shake128", "ristretto255-hashed-elgamal-v1", 128, nonce)
    variants = [(b"bcfe/gc_ristretto25",) + base[1:], (sid, 2) + base[2:], base[:2] + (cd[::-1],) + base[3:],
                base[:6] + (192, nonce), base[:7] + (nonce[::-1],)]
    assert len({kem.setup_digest(*v) for v in variants} | {digest, labels}) == len(variants) + 2


@pytest.mark.parametrize("index, value, error, match", [
    (0, 17, TypeError, "scheme_id must be bytes"),
    (1, 2 ** 32, ValueError, "bcfe_version"),
    (2, bytes(31), ValueError, "circuit_digest must be 32 bytes"),
    (3, "K", ValueError, "unknown data_wires"),
    (3, 2, ValueError, "unknown data_wires"),
    (3, True, ValueError, "unknown data_wires"),
    (4, "aes", ValueError, "unknown gc_hash"),
    (4, 1, ValueError, "unknown gc_hash"),
    (5, "ristretto255-hashed-elgamal-v2", ValueError, "unknown kem"),
    (6, -1, ValueError, "label_bits"),
    (7, bytes(33), ValueError, "setup_nonce must be 32 bytes"),
])
def test_setup_digest_errors(index, value, error, match):
    args = ["bcfe/gc_ristretto255", 1, bytes(32), "hardwired", "shake128", "ristretto255-hashed-elgamal-v1", 128,
            bytes(32)]
    args[index] = value
    with pytest.raises(error, match=match):
        kem.setup_digest(*args)


def test_ephemeral_scalar(monkeypatch):
    r = kem.ephemeral_scalar(SEED)
    assert r == int.from_bytes(hashlib.shake_256(b"PYFE4AI-V06-BCFE-EPHEM" + SEED).digest(64), "little") % L
    # KAT: ephemeral_scalar(bytes(range(100, 132)))
    assert r == 0xd03ea23b1c7692e221c15d404f6f9087fad0fe71ce10e550aae69eaf43e295b
    assert 0 < r < L and kem.ephemeral_scalar(SEED[::-1]) != r
    for seed in (SEED[:-1], SEED + b"\x00"):
        with pytest.raises(ValueError, match="seed must be 32 bytes"):
            kem.ephemeral_scalar(seed)

    class Zero:
        def __init__(self, data) -> None:
            pass

        def digest(self, n):
            return L.to_bytes(n, "little")  # l mod l = 0

    monkeypatch.setattr(kem.hashlib, "shake_256", Zero)
    with pytest.raises(ValueError, match="ephemeral scalar is 0"):
        kem.ephemeral_scalar(SEED)


def test_scalar_encoding():
    for s in (0, 1, L - 1, 2 ** 200):
        assert kem.decode_scalar(kem.encode_scalar(s)) == s and len(kem.encode_scalar(s)) == 32
    assert kem.encode_scalar(1) == b"\x01" + bytes(31)
    for s in (L, -1, 2 ** 256, True):
        with pytest.raises(ValueError, match="scalar"):
            kem.encode_scalar(s)
    for data in (L.to_bytes(32, "little"), b"\xff" * 32):
        with pytest.raises(ValueError, match="canonical"):
            kem.decode_scalar(data)
    with pytest.raises(ValueError, match="32 bytes"):
        kem.decode_scalar(bytes(31))


def test_table_index():
    assert [kem.table_index(3, j, b) for j in range(3) for b in (0, 1)] + [kem.table_index(3, 3, 0)] == list(range(7))
    for j, b in ((3, 1), (4, 0), (-1, 0), (0, 2)):
        with pytest.raises(ValueError):
            kem.table_index(3, j, b)
    with pytest.raises(ValueError, match="m must be"):
        kem.table_index(0, 0, 0)


# ------------------------------------------------------------------ issuance

def test_new_slot_reads_its_randomness_in_order():
    w = [1, 0, 1, 1]
    m = len(w)
    data = random.Random(5).randbytes(64 * (m + 1) + 16)
    log = []
    slot = kem.new_slot(SETUP, 11, w, recording(kem.byte_source(data), log))
    assert log == [64] * (m + 1) + [16]
    scalars = [int.from_bytes(c, "little") % L for c in split(data[:64 * (m + 1)], 64)]
    assert slot["scalars"] == b"".join(s.to_bytes(32, "little") for s in scalars)
    assert slot["k"] == data[-16:]
    table = split(slot["pks"], 32)
    assert len(slot["pks"]) == 32 * (2 * m + 1) and len(table) == 2 * m + 1
    for j, bit in enumerate(w):
        assert table[kem.table_index(m, j, bit)] == point(scalars[j])
        assert table[kem.table_index(m, j, 1 - bit)] == kem.unselected_pk(SETUP, 11, j, slot["k"])
    assert table[kem.table_index(m, m, 0)] == point(scalars[m])
    packed = kem.new_slot(SETUP, 11, gc.pack_bits(w, m), kem.byte_source(data), m=m)
    assert packed == slot


def test_new_slot_scalars_are_canonical():
    rand = source(6)
    for m in (1, 2, 9):
        slot = kem.new_slot(SETUP, m, [random.Random(m).randrange(2) for _ in range(m)], rand)
        parts = split(slot["scalars"], 32)
        assert len(parts) == m + 1 and len(slot["k"]) == 16
        for part in parts:
            assert 0 < int.from_bytes(part, "little") < L
    zero = L.to_bytes(64, "little")  # reduces to 0: refused, never redrawn
    with pytest.raises(ValueError, match="secret scalar is 0"):
        kem.new_slot(SETUP, 1, [1], kem.byte_source(zero + bytes(80)))


@pytest.mark.parametrize("args, error, match", [
    ((SETUP[:-1], 1, [1]), ValueError, "setup_digest must be 32 bytes"),
    ((SETUP, 2 ** 64, [1]), ValueError, "slot_id"),
    ((SETUP, 1, []), ValueError, "at least one key bit"),
    ((SETUP, 1, [1, 2]), ValueError, "not 0 or 1"),
    ((SETUP, 1, b"\x01"), ValueError, "packed key bits need m"),
])
def test_new_slot_errors(args, error, match):
    with pytest.raises(error, match=match):
        kem.new_slot(*args, source(1))


def test_new_slot_key_bits():
    with pytest.raises(ValueError, match="expected 3 key bits"):
        kem.new_slot(SETUP, 1, [1, 0], source(1), m=3)
    with pytest.raises(ValueError, match="nonzero padding"):
        kem.new_slot(SETUP, 1, b"\x09", source(1), m=3)
    with pytest.raises(ValueError, match="take 1 bytes"):
        kem.new_slot(SETUP, 1, b"\x01\x00", source(1), m=3)
    for m in (0, -1, 2 ** 31, True):
        with pytest.raises(ValueError, match="m must be"):
            kem.new_slot(SETUP, 1, b"\x01", source(1), m=m)
    with pytest.raises(ValueError, match="must return 64 bytes"):
        kem.new_slot(SETUP, 1, [1], lambda n: bytes(n - 1))


def holder_keys(rng, m) -> tuple:
    """A key holder's ``m + 1`` secret keys and public keys (h2)."""
    scalars = [rng.randrange(1, L) for _ in range(m + 1)]
    return b"".join(s.to_bytes(32, "little") for s in scalars), b"".join(point(s) for s in scalars)


@pytest.mark.parametrize("mode", MODES)
def test_h2_holder_generates_its_keys(mode):
    rng = random.Random(7)
    for w in ([0, 0, 0], [1, 0, 1], [1, 1, 1]):
        scalars, pks = holder_keys(rng, 3)
        log = []
        slot = kem.place_slot(SETUP, 21, w, pks, recording(source(8), log))
        assert log == [16] and set(slot) == {"pks", "k"}
        table, own = split(slot["pks"], 32), split(pks, 32)
        assert [table[kem.table_index(3, j, b)] for j, b in enumerate(w)] + [table[6]] == own
        assert kem.unselected_ok(SETUP, 21, w, slot["k"], slot["pks"])
        for x in ([0, 0, 0], [1, 1, 0], [1, 0, 1]):
            assert roundtrip(SMALL, mode, {"pks": slot["pks"], "scalars": scalars}, w, x, slot_id=21,
                             seed=rng.randbytes(32)) == SMALL(w, x)
        assert kem.place_slot(SETUP, 21, gc.pack_bits(w, 3), pks, source(8), m=3) == slot


def test_place_slot_checks_the_holder_keys():
    """I8: every holder key decodes and is not the identity; nothing is drawn before the check passes."""
    rng = random.Random(9)
    _, pks = holder_keys(rng, 3)
    own = split(pks, 32)
    non_canonical = (2 ** 255 - 19 + 2).to_bytes(32, "little")  # s >= p
    bad_points = [bytes(32), b"\xff" * 32, non_canonical, (1).to_bytes(32, "little")]  # identity, >= p, odd s
    for i in range(4):
        for bad in bad_points:
            log = []
            keys = b"".join(bad if t == i else own[t] for t in range(4))
            with pytest.raises(ValueError, match="holder public key {}".format(i)):
                kem.place_slot(SETUP, 1, [1, 0, 1], keys, recording(source(1), log))
            assert log == []
    for keys in (pks[:-32], pks + own[0], pks[:-1]):
        with pytest.raises(ValueError, match="holder_pks: expected 4 points"):
            kem.place_slot(SETUP, 1, [1, 0, 1], keys, source(1))


# ------------------------------------------------------------------ the h' part of CheckAuth

def test_unselected_ok_accepts_the_honest_tables():
    for m in (1, 3, 8):
        w = [random.Random(m).randrange(2) for _ in range(m)]
        slot = kem.new_slot(SETUP, 30 + m, w, source(m))
        assert kem.unselected_ok(SETUP, 30 + m, w, slot["k"], slot["pks"])
        assert kem.unselected_ok(SETUP, 30 + m, gc.pack_bits(w, m), slot["k"], slot["pks"], m=m)


def test_unselected_ok_rejects_every_change():
    w = [1, 0, 0, 1]
    m = len(w)
    slot = kem.new_slot(SETUP, 40, w, source(40))
    table, k = split(slot["pks"], 32), slot["k"]

    def ok(pks=slot["pks"], key=k, slot_id=40, setup=SETUP, bits=w):
        return kem.unselected_ok(setup, slot_id, bits, key, pks)

    assert ok()
    unselected = [kem.table_index(m, j, 1 - b) for j, b in enumerate(w)]
    for i in unselected:  # any unselected entry replaced
        assert not ok(pks=b"".join(point(99) if t == i else p for t, p in enumerate(table)))
    assert not ok(key=k[::-1]) and not ok(key=bytes(16))
    assert not ok(slot_id=41) and not ok(slot_id=40 + 2 ** 32)
    assert not ok(setup=SETUP[::-1])
    assert not ok(bits=[0, 0, 0, 1])  # another w: the check reads another position
    # the holder's keys moved: the pair of a key bit swapped, the g key put in an unselected position
    for j in range(m):
        swapped = list(table)
        swapped[2 * j], swapped[2 * j + 1] = swapped[2 * j + 1], swapped[2 * j]
        assert not ok(pks=b"".join(swapped))
        moved = list(table)
        moved[unselected[j]], moved[2 * m] = moved[2 * m], moved[unselected[j]]
        assert not ok(pks=b"".join(moved))
    # a table of another length (2m, 2m + 2 entries, one byte more or less)
    for pks in (slot["pks"][:-32], slot["pks"] + table[0], slot["pks"] + b"\x00", slot["pks"][:-1], b""):
        assert not ok(pks=pks)
    with pytest.raises(ValueError, match="k must be 16 bytes"):
        ok(key=k[:-1])
    with pytest.raises(TypeError, match="pks must be bytes"):
        ok(pks=table)


# ------------------------------------------------------------------ encryption and decryption

@pytest.mark.parametrize("mode", MODES)
def test_random_circuits_decrypt_to_the_plain_circuit(mode):
    rng = random.Random(20261119 + len(mode))
    for trial in range(30):
        m, n = rng.randint(1, 4), rng.randint(0, 5)
        gates, outputs = random_circuit(rng, m, n, rng.randint(0, 30))
        circuit = Circuit(m, n, gates, outputs)
        w = [rng.randrange(2) for _ in range(m)]
        x = [rng.randrange(2) for _ in range(n)]
        slot_id = rng.randrange(2 ** 64)
        slot = kem.new_slot(SETUP, slot_id, w if trial % 2 else gc.pack_bits(w, m), rng.randbytes, m=m)
        assert roundtrip(circuit, mode, slot, w if trial % 3 else gc.pack_bits(w, m),
                         x if trial % 2 else gc.pack_bits(x, n), slot_id=slot_id, seed=rng.randbytes(32),
                         pp=rng.randbytes(32), nonce=rng.randbytes(16)) == circuit(w, x)


@pytest.mark.parametrize("mode", MODES)
def test_encrypt_slot_output(mode):
    w, x = [0, 1, 1], [1, 0, 1]
    slot = kem.new_slot(SETUP, 7, w, source(3))
    ct = kem.encrypt_slot(SMALL.guarded, mode, PP, NONCE, 7, x, slot["pks"], SEED)
    assert set(ct) == {"R", "label_cts", "tables", "decode", "data_active"}
    assert ct["R"] == point(kem.ephemeral_scalar(SEED))
    garbled = gc.garble(SMALL.guarded, mode, kem.instance_salt(PP, NONCE, 7), x, SEED)
    for name in ("tables", "decode", "data_active"):
        assert ct[name] == garbled[name]
    assert len(ct["label_cts"]) == 16 * (2 * 3 + 1)
    for secret in [garbled["delta"]] + split(garbled["key0"], 16):  # neither Delta nor a zero label leaves
        assert all(secret not in value for value in ct.values())
    assert kem.encrypt_slot(SMALL.guarded, mode, PP, NONCE, 7, gc.pack_bits(x, 3), slot["pks"], SEED) == ct
    assert kem.recover_labels(PP, NONCE, 7, ct["R"], ct["label_cts"], slot["scalars"], w) == b"".join(
        active(split(garbled["key0"], 16), garbled["delta"], w))


def test_the_g_bit_has_one_position():
    """I3: 2m + 1 label ciphertexts; with every position's secret key known (a
    test-only table), they open to every L_j^b and to L_g^0, never L_g^1."""
    m = 3
    secrets = [5 + i for i in range(2 * m + 1)]
    pks = b"".join(point(s) for s in secrets)
    for mode in MODES:
        ct = kem.encrypt_slot(SMALL.guarded, mode, PP, NONCE, 4, [1, 1, 0], pks, SEED)
        cts = split(ct["label_cts"], 16)
        assert len(cts) == 2 * m + 1
        key0, delta = true_labels(SMALL.guarded, mode, 4, [1, 1, 0], SEED)
        point_r = r255.decode(ct["R"])
        opened = []
        for i, c in enumerate(cts):
            j, b = (i // 2, i % 2) if i < 2 * m else (m, 0)
            z = r255.encode(r255.mul(point_r, secrets[i]))
            opened.append(xor(c, kem.kem_mask(PP, NONCE, 4, j, b, ct["R"], z)))
        expected = [key0[j] if b == 0 else xor(key0[j], delta) for j in range(m) for b in (0, 1)] + [key0[m]]
        assert opened == expected
        assert xor(key0[m], delta) not in opened
    with pytest.raises(ValueError, match="g bit has the one position"):
        kem.table_index(m, m, 1)
    slot = kem.new_slot(SETUP, 4, [1, 1, 0], source(4))
    ct = kem.encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 4, [0, 0, 0], slot["pks"], SEED)
    with pytest.raises(ValueError, match="expected 3 key bits"):  # w has m bits: g is never chosen
        kem.decrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 4, ct["R"], ct["label_cts"], ct["tables"],
                         ct["decode"], ct["data_active"], slot["scalars"], [1, 1, 0, 1])


def test_domain_separation_of_the_label_ciphertexts():
    """Label ciphertexts moved to another position, slot or ciphertext open to
    labels that are neither label of any wire (H_K's metadata, I2)."""
    m, w, x = 3, [1, 0, 1], [0, 1, 1]
    slot = kem.new_slot(SETUP, 7, w, source(10))
    mode = "hardwired"
    ct = kem.encrypt_slot(SMALL.guarded, mode, PP, NONCE, 7, x, slot["pks"], SEED)
    key0, delta = true_labels(SMALL.guarded, mode, 7, x, SEED)
    honest = active(key0, delta, w)
    every_label = set(key0) | {xor(k, delta) for k in key0}

    def opened(label_cts, slot_id=7, pp=PP, nonce=NONCE, big_r=ct["R"]):
        return split(kem.recover_labels(pp, nonce, slot_id, big_r, label_cts, slot["scalars"], w), 16)

    assert opened(ct["label_cts"]) == honest
    cts = split(ct["label_cts"], 16)
    # two positions of the slot swapped: (0, w_0) <-> (2, w_2), and the pair of bit 1
    swapped = list(cts)
    a, b = kem.table_index(m, 0, w[0]), kem.table_index(m, 2, w[2])
    swapped[a], swapped[b] = swapped[b], swapped[a]
    got = opened(b"".join(swapped))
    assert got[0] != honest[0] and got[2] != honest[2] and got[1] == honest[1] and got[3] == honest[3]
    assert got[0] not in every_label and got[2] not in every_label
    swapped = list(cts)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    assert opened(b"".join(swapped))[1] not in every_label
    # the g ciphertext in place of a key bit's
    swapped = list(cts)
    swapped[kem.table_index(m, 1, w[1])] = cts[2 * m]
    assert opened(b"".join(swapped))[1] not in every_label
    # another slot id, pp digest or nonce under the same seed: the same labels and R, other metadata
    for change in ({"slot_id": 8}, {"pp": PP[::-1]}, {"nonce": NONCE[::-1]}):
        args = {"slot_id": 7, "pp": PP, "nonce": NONCE}
        args.update(change)
        other = kem.encrypt_slot(SMALL.guarded, mode, args["pp"], args["nonce"], args["slot_id"], x, slot["pks"],
                                 SEED)
        assert other["R"] == ct["R"]
        same0, same_delta = true_labels(SMALL.guarded, mode, args["slot_id"], x, SEED, args["pp"], args["nonce"])
        assert (same0, same_delta) == (key0, delta)
        assert opened(other["label_cts"], **args) == honest  # opened with its own metadata
        moved = opened(other["label_cts"])  # opened as the component of (pp, nonce, slot 7)
        assert all(label not in every_label for label in moved)
        y = kem.decrypt_slot(SMALL.guarded, mode, PP, NONCE, 7, other["R"], other["label_cts"], ct["tables"],
                             ct["decode"], ct["data_active"], slot["scalars"], w)
        assert len(y) == 1  # garbage bits, no error (one error class at the scheme layer, I10)
    # the label ciphertexts of another ciphertext of the same slot (another seed: another R)
    fresh = kem.encrypt_slot(SMALL.guarded, mode, PP, NONCE, 7, x, slot["pks"], SEED[::-1])
    assert all(label not in every_label for label in opened(fresh["label_cts"]))
    assert all(label not in every_label for label in opened(ct["label_cts"], big_r=fresh["R"]))


def test_the_circuit_must_be_guarded():
    slot = kem.new_slot(SETUP, 1, [1, 0], source(1))
    m, n, gates, outputs = guard(2, 1, [(AND, 0, 2), (XOR, 1, 3)], [4, 3])
    assert m == 3
    good = encode(m, n, gates, outputs)
    kem.encrypt_slot(good, "hardwired", PP, NONCE, 1, [1], slot["pks"], SEED)
    ng = m + n + len(gates) - len(outputs) - 1
    bad = {
        "unguarded": encode(2, 1, [(AND, 0, 2), (XOR, 1, 3)], [4, 3]),
        "outputs reordered": encode(m, n, gates, outputs[::-1]),
        "g read by a gate": encode(m, n, [(AND, 0, 3), (XOR, 2, 4)] + gates[2:], outputs),
        "AND inputs swapped": encode(m, n, gates[:-1] + [(AND, ng, gates[-1][1])], outputs),
        "one key wire": encode(1, 1, [(INV, 0, 0), (AND, 1, 2)], [3]),
    }
    for name, circuit in bad.items():
        with pytest.raises(ValueError, match="not guard"):
            kem.encrypt_slot(circuit, "hardwired", PP, NONCE, 1, [1], slot["pks"], SEED)
        if name != "one key wire":
            with pytest.raises(ValueError, match="not guard"):
                kem.decrypt_slot(circuit, "hardwired", PP, NONCE, 1, bytes(32), b"", b"", b"", b"", b"", [1, 0])


def test_encrypt_slot_errors():
    slot = kem.new_slot(SETUP, 1, [1, 0, 0], source(2))
    table = split(slot["pks"], 32)

    def run(**change):
        args = dict({"circuit": SMALL.guarded, "mode": "hardwired", "pp_digest": PP, "ct_nonce": NONCE,
                     "slot_id": 1, "x_bits": [1, 0, 1], "pks": slot["pks"], "seed": SEED}, **change)
        return kem.encrypt_slot(**args)

    run()
    for i in (0, 3, 6):
        with pytest.raises(ValueError, match="public key {} of the slot is the identity".format(i)):
            run(pks=b"".join(bytes(32) if t == i else p for t, p in enumerate(table)))
        with pytest.raises(ValueError, match="public key {} of the slot is not a canonical".format(i)):
            run(pks=b"".join(b"\xff" * 32 if t == i else p for t, p in enumerate(table)))
    for pks in (slot["pks"][:-32], slot["pks"] + table[0], slot["pks"][:-1]):
        with pytest.raises(ValueError, match="pks: expected 7 points"):
            run(pks=pks)
    cases = [({"mode": "K"}, ValueError, "unknown garbling mode"), ({"pp_digest": PP[:-1]}, ValueError, "pp_digest"),
             ({"ct_nonce": NONCE[:-1]}, ValueError, "ct_nonce"), ({"slot_id": 2 ** 64}, ValueError, "slot_id"),
             ({"seed": SEED[:-1]}, ValueError, "seed must be 32 bytes"), ({"x_bits": [1, 0]}, ValueError, "3 bits"),
             ({"x_bits": b"\x09"}, ValueError, "nonzero padding"), ({"circuit": SMALL.guarded[:-1]}, ValueError,
                                                                    "the counts give"),
             ({"pks": table}, TypeError, "pks must be bytes")]
    for change, error, match in cases:
        with pytest.raises(error, match=match):
            run(**change)


def test_decrypt_slot_errors():
    w, x = [0, 1, 1], [1, 1, 0]
    slot = kem.new_slot(SETUP, 1, w, source(3))
    ct = kem.encrypt_slot(SMALL.guarded, "labels", PP, NONCE, 1, x, slot["pks"], SEED)
    good = {"circuit": SMALL.guarded, "mode": "labels", "pp_digest": PP, "ct_nonce": NONCE, "slot_id": 1,
            "R": ct["R"], "label_cts": ct["label_cts"], "tables": ct["tables"], "decode": ct["decode"],
            "data_active": ct["data_active"], "scalars": slot["scalars"], "w_bits": gc.pack_bits(w, 3)}

    def run(**change):
        return kem.decrypt_slot(**dict(good, **change))

    assert gc.unpack_bits(run(), 3) == SMALL(w, x)
    over = L.to_bytes(32, "little")
    cases = [({"R": ct["R"][:-1]}, "R must be 32 bytes"), ({"R": bytes(32)}, "R is the identity"),
             ({"R": b"\xff" * 32}, "R is not a canonical"), ({"label_cts": ct["label_cts"][:-16]}, "label_cts"),
             ({"label_cts": ct["label_cts"] + bytes(16)}, "label_cts"), ({"scalars": slot["scalars"][:-32]}, "scalars"),
             ({"scalars": slot["scalars"][:32] + over + slot["scalars"][64:]}, "canonical"),
             ({"w_bits": b"\x0b"}, "nonzero padding"), ({"w_bits": b"\x03\x00"}, "take 1 bytes"),
             ({"tables": ct["tables"][:-16]}, "tables: expected"), ({"data_active": b""}, "data_active: expected"),
             ({"decode": b""}, "take 1 bytes"), ({"mode": "K"}, "unknown garbling mode"),
             ({"slot_id": -1}, "slot_id"), ({"pp_digest": PP + b"\x00"}, "pp_digest")]
    for change, match in cases:
        with pytest.raises(ValueError, match=match):
            run(**change)
    with pytest.raises(TypeError, match="scalars must be bytes"):
        run(scalars=list(slot["scalars"]))


# ------------------------------------------------------------------ known answers of a whole slot

KAT_CIRCUIT = encode(*guard(2, 2, [(AND, 0, 2), (XOR, 1, 3), (AND, 4, 5)], [6, 4]))


def kat_digest(*parts) -> str:
    return hashlib.sha256(b"".join(parts)).hexdigest()


def test_known_answers_of_a_slot():
    """Cross-language anchors. Inputs: KAT_CIRCUIT (guard of m = 2, n = 2,
    gates AND(0, 2), XOR(1, 3), AND(4, 5), outputs 6, 4); setup digest
    bytes(range(128, 160)), slot 5, w = [1, 0]; the issuance bytes
    SHAKE-256(b"bcfe-kem-kat") (64 * 3 + 16 bytes); ct encrypted with pp
    bytes(range(32)), nonce bytes(range(32, 48)), x = [1, 1], seed
    bytes(range(100, 132)). Values: SHA-256 of the concatenated outputs."""
    data = hashlib.shake_256(b"bcfe-kem-kat").digest(64 * 3 + 16)
    slot = kem.new_slot(SETUP, 5, [1, 0], kem.byte_source(data))
    assert kat_digest(slot["scalars"], slot["pks"], slot["k"]) == KAT_SLOT
    got = {}
    for mode in MODES:
        ct = kem.encrypt_slot(KAT_CIRCUIT, mode, PP, NONCE, 5, [1, 1], slot["pks"], SEED)
        got[mode] = kat_digest(ct["R"], ct["label_cts"], ct["tables"], ct["decode"], ct["data_active"])
        y = kem.decrypt_slot(KAT_CIRCUIT, mode, PP, NONCE, 5, ct["R"], ct["label_cts"], ct["tables"], ct["decode"],
                             ct["data_active"], slot["scalars"], [1, 0])
        assert gc.unpack_bits(y, 2) == [1, 1]  # (w0 & x0) & (w1 ^ x1), w0 & x0
    assert got == KAT_CT


KAT_SLOT = "1c49af533a862464123d2ec973b9a569a3136c587b86848e477826164a42d8cc"
KAT_CT = {"hardwired": "c71e4733181af291bde1a203815793b9855581ba99960240d5e7aaf73a59c8ab",
          "labels": "378c9bb8f5ddaa9cff7c3bbf1800c2968e7478e47e2ea77f98d94a0d3407c3c8"}


# ------------------------------------------------------------------ the backend operations

def test_the_interface_declares_the_operations():
    assert OPERATIONS <= interface.Backend.__abstractmethods__


@pytest.fixture
def python_ops():
    with native_backend.backend("python"):
        yield native_backend.ops()


def test_backend_operations_draw_the_documented_bytes(python_ops, monkeypatch):
    ops, m, w = python_ops, 3, gc.pack_bits([1, 0, 1], 3)
    drawn = []
    stream = random.Random(12).randbytes

    def take(nbytes):
        drawn.append(nbytes)
        return stream(nbytes)

    monkeypatch.setattr(native_backend, "_random_override", take)
    slot = ops.bcfe_new_slot(SETUP, 3, w, m=m)
    assert drawn == [64 * (m + 1) + 16]
    stream = random.Random(12).randbytes
    assert slot == kem.new_slot(SETUP, 3, w, kem.byte_source(stream(64 * (m + 1) + 16)), m=m)

    drawn.clear()
    stream = random.Random(13).randbytes
    rng = random.Random(14)
    scalars, holder = holder_keys(rng, m)
    placed = ops.bcfe_place_slot(SETUP, 4, w, holder, m=m)
    assert drawn == [16] and placed["k"] == random.Random(13).randbytes(16)
    assert ops.bcfe_unselected_ok(SETUP, 4, w, placed["k"], placed["pks"], m=m) is True
    assert ops.bcfe_unselected_ok(SETUP, 4, w, placed["k"], placed["pks"][:-32], m=m) is False

    drawn.clear()
    stream = random.Random(15).randbytes
    x = gc.pack_bits([0, 1, 1], 3)
    ct = ops.bcfe_encrypt_slot(SMALL.guarded, "labels", PP, NONCE, 4, x, placed["pks"])
    assert drawn == [32]
    assert ct == kem.encrypt_slot(SMALL.guarded, "labels", PP, NONCE, 4, x, placed["pks"],
                                  random.Random(15).randbytes(32))
    drawn.clear()
    y = ops.bcfe_decrypt_slot(SMALL.guarded, "labels", PP, NONCE, 4, ct["R"], ct["label_cts"], ct["tables"],
                              ct["decode"], ct["data_active"], scalars, w)
    assert drawn == [] and gc.unpack_bits(y, 3) == SMALL([1, 0, 1], [0, 1, 1])


def test_backend_encryption_is_deterministic_only_under_the_hook(python_ops, monkeypatch):
    ops, x = python_ops, gc.pack_bits([1, 0, 0], 3)
    slot = kem.new_slot(SETUP, 2, [0, 1, 1], source(16))
    monkeypatch.setattr(native_backend, "_random_override", lambda n: SEED[:n])
    first = ops.bcfe_encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 2, x, slot["pks"])
    assert ops.bcfe_encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 2, x, slot["pks"]) == first
    assert first == kem.encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 2, x, slot["pks"], SEED)
    monkeypatch.setattr(native_backend, "_random_override", None)  # the operating system: fresh every call
    a, b = (ops.bcfe_encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 2, x, slot["pks"]) for _ in range(2))
    assert a["R"] != b["R"] and a["label_cts"] != b["label_cts"] and a["tables"] != b["tables"]
    for ct in (a, b):
        y = ops.bcfe_decrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 2, ct["R"], ct["label_cts"], ct["tables"],
                                  ct["decode"], ct["data_active"], slot["scalars"], gc.pack_bits([0, 1, 1], 3))
        assert gc.unpack_bits(y, 3) == SMALL([0, 1, 1], [1, 0, 0])
    s1, s2 = (ops.bcfe_new_slot(SETUP, 2, b"\x01", m=1) for _ in range(2))
    assert s1["scalars"] != s2["scalars"] and s1["k"] != s2["k"]


def test_backend_operation_errors(python_ops, monkeypatch):
    ops = python_ops
    drawn = []
    monkeypatch.setattr(native_backend, "_random_override", lambda n: drawn.append(n) or bytes(n))
    for m in (0, 2 ** 31, True, "3"):
        with pytest.raises(ValueError, match="m must be"):
            ops.bcfe_new_slot(SETUP, 1, b"\x01", m=m)
    assert drawn == []  # m is checked before the draw
    with pytest.raises(TypeError, match="w must be the packed bits"):
        ops.bcfe_new_slot(SETUP, 1, [1], m=1)
    with pytest.raises(TypeError, match="x must be the packed bits"):
        ops.bcfe_encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 1, [1, 0, 1], bytes(32 * 7))
    with pytest.raises(TypeError):
        ops.bcfe_new_slot(SETUP, 1, b"\x01", 1)  # m is keyword-only
    with pytest.raises(ValueError, match="secret scalar is 0"):  # all-zero test bytes: s = 0
        ops.bcfe_new_slot(SETUP, 1, b"\x01", m=1)
    monkeypatch.setattr(native_backend, "_random_override", lambda n: bytes(n - 1))
    for call in (lambda: ops.bcfe_new_slot(SETUP, 1, b"\x01", m=1),
                 lambda: ops.bcfe_place_slot(SETUP, 1, b"\x01", point(2) + point(3), m=1),
                 lambda: ops.bcfe_encrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 1, b"\x00", bytes(224))):
        with pytest.raises(ValueError, match="random bytes of the call"):
            call()
    monkeypatch.setattr(native_backend, "_random_override", None)
    with pytest.raises(ValueError, match="holder public key 1 is the identity"):
        ops.bcfe_place_slot(SETUP, 1, b"\x01", point(2) + bytes(32), m=1)
    with pytest.raises(ValueError, match="holder_pks: expected 2 points"):
        ops.bcfe_place_slot(SETUP, 1, b"\x01", point(2), m=1)
    with pytest.raises(ValueError, match="k must be 16 bytes"):
        ops.bcfe_unselected_ok(SETUP, 1, b"\x01", bytes(15), bytes(96), m=1)
    with pytest.raises(TypeError, match="w must be the packed bits"):
        ops.bcfe_decrypt_slot(SMALL.guarded, "hardwired", PP, NONCE, 1, point(2), bytes(112), b"", b"", b"",
                              bytes(128), [1, 0, 1])
