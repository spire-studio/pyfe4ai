"""Negative regressions of the bounded-key schemes ``bcfe``: the attacks the
0.6 security argument rules out, run against the primitives.

A key of ``bcfe/*`` is a slot of the construction of Sahai and Seyalioglu,
"Worry-Free Encryption: Functional Encryption with Public Keys" (ACM CCS
2010, SS10): an encryption garbles ``guard(C)`` afresh for every slot with
the half-gates scheme of Zahur, Rosulek and Evans (EUROCRYPT 2015, ZRE15,
Fig. 2) and encrypts the two labels of every key wire under the slot's two
public keys of that position (:mod:`pyfe4ai.primitives.bcfe_kem`). One
garbling must never be opened at two inputs: Gorbunov, Vaikuntanathan and
Wee (CRYPTO 2012, Sec. 5.2) note that two keys for one instance may break it
completely. Two labels of one key wire give ``Delta``, and ``Delta`` with the
tables gives ``x``.

These tests build the situations the requirements of the security argument
(0.6, section 6) forbid -- with test-only helpers written here, never
through the library's issuance -- and show that each is a total break, or
show that the library's own path does not give an attacker what the attack
needs:

* I4 / h' ("one slot, two keys", the attack of the 0.6 prototype):
  :func:`test_one_extra_secret_key_opens_x_from_a_single_half_gate`,
  :func:`test_one_extra_secret_key_opens_every_data_bit_of_a_linear_model`,
  :func:`test_h_prime_slots_have_no_unselected_secret_key`;
* review finding V6-F02 (h2 with a center that did not erase) and h':
  :func:`test_h2_with_a_center_that_kept_its_secret_keys_is_a_total_break`,
  :func:`test_holder_and_center_together_cannot_open_an_unselected_label`;
* I1 (fresh ``Delta`` per (ciphertext, slot)):
  :func:`test_a_seed_shared_by_two_slots_lets_their_holders_open_x`,
  :func:`test_every_component_of_the_library_has_its_own_delta`;
* I2 and I11 (binding of the label ciphertexts and the tables):
  :func:`test_mixed_components_open_to_garbage`;
* I3 (the g bit): :func:`test_the_g_bit_has_one_position_and_forces_zero`;
* I11 and review finding V6-F04 (the instance salt):
  :func:`test_the_instance_salt_separates_garblings_of_one_seed`.

The attacker's tools are written out here: :func:`attacker_view` (the
evaluation, keeping every wire's active label) and :func:`what_delta_opens`
(ZRE15 Fig. 2 read backwards with ``Delta``). The encryptions go through the
active backend's operation ``bcfe_encrypt_slot`` (native by default); where a
test compares with the truth, the seed comes through the tests' hook
``pyfe4ai.backend._random_override`` (I16) and the truth from the reference
:func:`~pyfe4ai.primitives.garbling.garble`.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import random

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.circuits import AND, Circuit, linear_score, linear_threshold
from pyfe4ai.primitives import bcfe_kem as kem
from pyfe4ai.primitives import garbling as gc
from pyfe4ai.primitives.groups import ristretto255 as r255

XOR, INV = gc.OP_XOR, gc.OP_INV
MODES = gc.MODES
ZERO = bytes(16)

SETUP = bytes(range(128, 160))
PP = bytes(range(32))
PP2 = bytes(range(1, 33))
NONCE = bytes(range(32, 48))
NONCE2 = bytes(range(48, 64))

#: ``C(w, x) = w_0 AND x_0``: one AND of a key wire and a data wire (a
#: generator half-gate in mode "hardwired"); its guard has key wires ``w_0``
#: and ``g`` and data wire ``x_0``.
ONE_GATE = Circuit(1, 1, [(AND, 0, 1)], [2])


# ------------------------------------------------------------------ circuits and values

@functools.lru_cache(maxsize=None)
def model(name: str, mul: str):
    """``linear_threshold`` or ``linear_score`` with ``n = 3``, ``bw = bx = 4``."""
    return {"threshold": linear_threshold, "score": linear_score}[name](3, 4, 4, mul=mul)


def model_values(pc, rng) -> tuple:
    """Random ``(w, x)`` of a linear model, as values and as bits."""
    n, bw, bx = pc.params["n"], pc.params["bw"], pc.params["bx"]
    w = {"weights": [rng.randint(-2 ** (bw - 1), 2 ** (bw - 1) - 1) for _ in range(n)],
         "bias": rng.randint(-2 ** (bw + bx - 1), 2 ** (bw + bx - 1) - 1)}
    x = {"x": [rng.randint(-2 ** (bx - 1), 2 ** (bx - 1) - 1) for _ in range(n)]}
    return w, x, pc.key_layout.encode(w), pc.data_layout.encode(x)


def plain_wires(parsed, key_bits, x_bits) -> list:
    """The value of every wire, in the clear."""
    v = list(key_bits) + list(x_bits)
    for op, a, b in parsed.gates:
        v.append(v[a] ^ v[b] if op == XOR else v[a] & v[b] if op == AND else 1 - v[a])
    return v


# ------------------------------------------------------------------ bytes, keys and components

def split(data: bytes, size: int = 16) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


def xor(a: bytes, b: bytes) -> bytes:
    return bytes(u ^ v for u, v in zip(a, b))


def lsb(label: bytes) -> int:
    return label[0] & 1


def point(s: int) -> bytes:
    return r255.encode(r255.mul_base(s))


def draw_scalar(rng) -> int:
    return rng.randrange(1, r255.L)


def encode_scalars(scalars) -> bytes:
    return b"".join(kem.encode_scalar(s) for s in scalars)


def pack(bits) -> bytes:
    return gc.pack_bits(bits, len(bits))


def full_key_slot(m: int, w: list, rng) -> tuple:
    """Test-only: a slot whose ``2m + 1`` positions ALL have a secret key.

    This is the table of the original h1 before h' (the center draws a key
    pair for every position and must erase the unselected secret keys), or
    of h2 with a center that draws the unselected key pairs itself. The
    library never builds such a slot: under h' the unselected public keys
    are hash outputs (I4). Returns ``(secrets, pks, holder)``: the ``2m + 1``
    scalars in table order, the table, and the holder's key -- the ``m + 1``
    scalars of ``(j, w_j)`` and ``(g, 0)``, as ``new_slot`` returns them.
    """
    secrets = [draw_scalar(rng) for _ in range(2 * m + 1)]
    holder = [secrets[kem.table_index(m, j, w[j])] for j in range(m)] + [secrets[2 * m]]
    return secrets, b"".join(point(s) for s in secrets), encode_scalars(holder)


def encrypt(circuit: bytes, mode: str, slot_id: int, x_bits, pks: bytes, seed: bytes | None = None, pp: bytes = PP,
            nonce: bytes = NONCE) -> dict:
    """A slot's component through the active backend's ``bcfe_encrypt_slot``:
    its seed from the operating system, or *seed* through the tests' hook."""
    previous = native_backend._random_override
    if seed is not None:
        native_backend._random_override = lambda nbytes: seed
    try:
        return native_backend.ops().bcfe_encrypt_slot(circuit, mode, pp, nonce, slot_id, pack(x_bits), pks)
    finally:
        native_backend._random_override = previous


def decrypt(circuit: bytes, mode: str, slot_id: int, ct: dict, scalars: bytes, w, pp: bytes = PP,
            nonce: bytes = NONCE) -> list:
    """The holder's decryption through the active backend: the output bits."""
    y = native_backend.ops().bcfe_decrypt_slot(circuit, mode, pp, nonce, slot_id, ct["R"], ct["label_cts"],
                                              ct["tables"], ct["decode"], ct["data_active"], scalars, pack(w))
    return gc.unpack_bits(y, gc.parse_circuit(circuit).n_out)


def truth(circuit: bytes, mode: str, slot_id: int, x_bits, seed: bytes, pp: bytes = PP, nonce: bytes = NONCE) -> tuple:
    """``(key0, delta)`` of the garbling a component with *seed* carries."""
    garbled = gc.garble(circuit, mode, kem.instance_salt(pp, nonce, slot_id), x_bits, seed)
    return split(garbled["key0"]), garbled["delta"]


def active(key0: list, delta: bytes, w) -> list:
    """The true active labels of ``w || 0``."""
    return [xor(key0[j], delta) if bit else key0[j] for j, bit in enumerate(w)] + [key0[len(w)]]


def every_label(key0: list, delta: bytes) -> set:
    """Both labels of every key wire, ``L_g^1`` included."""
    return set(key0) | {xor(label, delta) for label in key0}


def holder_labels(ct: dict, slot_id: int, scalars: bytes, w, pp: bytes = PP, nonce: bytes = NONCE) -> list:
    """The ``m + 1`` labels the holder of *scalars* opens."""
    return split(kem.recover_labels(pp, nonce, slot_id, ct["R"], ct["label_cts"], scalars, w))


def open_position(ct: dict, slot_id: int, j: int, b: int, z: bytes, pp: bytes = PP, nonce: bytes = NONCE) -> bytes:
    """``c[j, b] xor H_K(pp, nonce, slot_id, j, b, R, Z)``: the label at
    position ``(j, b)`` for the right ``Z = s R``, garbage for another."""
    m = (len(ct["label_cts"]) // 16 - 1) // 2
    return xor(split(ct["label_cts"])[kem.table_index(m, j, b)], kem.kem_mask(pp, nonce, slot_id, j, b, ct["R"], z))


def open_with_scalar(ct: dict, slot_id: int, j: int, b: int, s: int, pp: bytes = PP, nonce: bytes = NONCE) -> bytes:
    """Position ``(j, b)`` opened with the scalar *s* (``Z = s R``)."""
    return open_position(ct, slot_id, j, b, r255.encode(r255.mul(r255.decode(ct["R"]), s)), pp, nonce)


# ------------------------------------------------------------------ the attacker's tools

def attacker_view(circuit: bytes, mode: str, salt: bytes, tables: bytes, key_active, data_active: bytes) -> tuple:
    """Evaluate a garbling as an evaluator does (ZRE15 Fig. 2, as
    :func:`~pyfe4ai.primitives.garbling.evaluate`) and keep what it sees.

    Returns ``(parsed, lab, ands)``: the parsed circuit, the active label of
    every wire (``None`` for a garbler-known wire) and every AND gate with a
    table as ``(kind, a, b, idx, entries)`` -- ``"LK"`` with ``a`` the
    labelled and ``b`` the garbler-known input, ``"LL"`` with the inputs in
    their order.
    """
    parsed = gc.parse_circuit(circuit)
    kinds = gc.wire_kinds(parsed, mode)
    entries = split(tables)
    assert len(entries) == gc.table_entries(parsed, mode)
    keys = split(key_active) if isinstance(key_active, bytes) else list(key_active)
    lab = keys + (split(data_active) if mode == gc.MODE_LABELS else [None] * parsed.n)
    ands = []
    ti = idx = 0
    for op, a, b in parsed.gates:
        if op == INV:
            out = lab[a]  # free: the same label (None for a garbler-known wire)
        elif op == XOR:
            out = xor(lab[a], lab[b]) if kinds[a] and kinds[b] else lab[a] if kinds[a] else lab[b]
        elif kinds[a] and kinds[b]:
            wa, wb = lab[a], lab[b]
            t_g, t_e = entries[ti], entries[ti + 1]
            out = xor(xor(gc.gate_hash(salt, idx, wa), t_g if lsb(wa) else ZERO),
                      xor(gc.gate_hash(salt, idx + 1, wb), xor(t_e, wa) if lsb(wb) else ZERO))
            ands.append(("LL", a, b, idx, (t_g, t_e)))
            ti, idx = ti + 2, idx + 2
        elif kinds[a] or kinds[b]:
            l, k = (a, b) if kinds[a] else (b, a)
            out = xor(gc.gate_hash(salt, idx, lab[l]), entries[ti] if lsb(lab[l]) else ZERO)
            ands.append(("LK", l, k, idx, (entries[ti],)))
            ti, idx = ti + 1, idx + 1
        else:
            out = None
        lab.append(out)
    return parsed, lab, ands


def _put(table: dict, key, value) -> None:
    assert table.setdefault(key, value) == value  # every table entry tells the same story about a wire


def what_delta_opens(salt: bytes, lab: list, ands: list, delta: bytes) -> tuple:
    """ZRE15 Fig. 2 read backwards: with ``Delta`` and either label ``A`` of an
    input, ``H(A, j) xor H(A xor Delta, j)`` is known, and every table entry
    gives up its secret:

    * AND(L, K): ``T xor H(A, j) xor H(A xor Delta, j) = v_k Delta``, the
      garbler-known bit ``v_k``;
    * AND(L, L): ``T_G xor H(A, j) xor H(A xor Delta, j) = p_b Delta``, the
      permute bit ``lsb(W0[b])`` of ``b``, and ``T_E xor H(B, j') xor H(B
      xor Delta, j') = W0[a]``, the zero label of ``a``.

    Returns ``(known, permute, zero)``, dicts by wire. Asserts that every
    value is 0 or ``Delta``, which holds for the garbling's own ``Delta``.
    """

    def both(label: bytes, idx: int) -> bytes:
        return xor(gc.gate_hash(salt, idx, label), gc.gate_hash(salt, idx, xor(label, delta)))

    known, permute, zero = {}, {}, {}
    for kind, a, b, idx, entries in ands:
        v = xor(entries[0], both(lab[a], idx))
        assert v in (ZERO, delta)
        _put(known if kind == "LK" else permute, b, int(v == delta))
        if kind == "LL":
            _put(zero, a, xor(entries[1], both(lab[b], idx + 1)))
    return known, permute, zero


def data_bits_from_tables(mode: str, parsed, lab: list, opened: tuple, delta: bytes) -> list:
    """``x`` read off the tables: in mode "hardwired" a data wire is the
    garbler-known input of a half-gate; in mode "labels" its active label is
    compared with its zero label, or its lsb with its permute bit."""
    known, permute, zero = opened
    bits = []
    for v in range(parsed.m, parsed.m + parsed.n):
        if mode == gc.MODE_HARDWIRED:
            bits.append(known[v])
        elif v in zero:
            assert lab[v] in (zero[v], xor(zero[v], delta))
            bits.append(int(lab[v] != zero[v]))
        else:
            bits.append(lsb(lab[v]) ^ permute[v])
    return bits


def any_key(zero_keys: list, g_label: bytes, delta: bytes, key_bits) -> list:
    """The active labels of ``key_bits || g`` for ANY key bits: what ``Delta``
    and one label per key wire give."""
    return [xor(z, delta) if bit else z for z, bit in zip(zero_keys, key_bits)] + [g_label]


def recover_features(pc, decide) -> list:
    """The prototype's attack on ``linear_threshold``: ``x_i`` by binary
    search on ``[x_i >= t] = C((e_i, -t), x)``; ``decide(key_bits)`` is the
    output bit for any key bits."""
    n, bx = pc.params["n"], pc.params["bx"]
    xs = []
    for i in range(n):
        lo, hi = -2 ** (bx - 1), 2 ** (bx - 1) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if decide(pc.key_layout.encode({"weights": [int(k == i) for k in range(n)], "bias": -mid})):
                lo = mid
            else:
                hi = mid - 1
        xs.append(lo)
    return xs


def output_labels(circuit: bytes, mode: str, salt: bytes, tables: bytes, key_active, data_active: bytes) -> list:
    """The active labels of the output wires."""
    parsed, lab, _ = attacker_view(circuit, mode, salt, tables, key_active, data_active)
    return [lab[v] for v in parsed.outputs]


# ------------------------------------------------------------------ one slot, two keys (I4 / h')

@pytest.mark.parametrize("mode", MODES)
def test_one_extra_secret_key_opens_x_from_a_single_half_gate(mode):
    """I4 / h' ("one slot, two keys"): a center that kept ONE unselected
    secret key (the original h1 without erasure) and the holder of the slot
    open both labels of one key position, hence ``Delta = L^0 xor L^1``,
    hence ``x``. Here from the tables of ``C(w, x) = w_0 AND x_0``, also when
    the holder's own output ``C(0, x) = 0`` says nothing about ``x``. Mode
    "hardwired": the generator half-gate's ``T = H(A0, j) xor H(A0 xor
    Delta, j) xor x_0 Delta`` gives ``x_0``; mode "labels": ``T_G`` gives the
    permute bit of the data wire, and with its active label, ``x_0``."""
    circuit = ONE_GATE.guard().encode()
    salt = kem.instance_salt(PP, NONCE, 3)
    rng = random.Random(1)
    for w0, x0 in itertools.product((0, 1), repeat=2):
        secrets, pks, holder = full_key_slot(1, [w0], rng)
        seed = rng.randbytes(32)
        ct = encrypt(circuit, mode, 3, [x0], pks, seed=seed)
        key0, delta = truth(circuit, mode, 3, [x0], seed)
        assert decrypt(circuit, mode, 3, ct, holder, [w0]) == [w0 & x0]  # all the holder may learn
        labels = holder_labels(ct, 3, holder, [w0])
        assert labels == active(key0, delta, [w0])
        kept = secrets[kem.table_index(1, 0, 1 - w0)]  # the one unselected secret key the center kept
        found = xor(labels[0], open_with_scalar(ct, 3, 0, 1 - w0, kept))
        assert found == delta
        parsed, lab, ands = attacker_view(circuit, mode, salt, ct["tables"], labels, ct["data_active"])
        assert ands[0][:3] == (("LK", 0, 2) if mode == gc.MODE_HARDWIRED else ("LL", 0, 2))  # AND(w_0, x_0)
        opened = what_delta_opens(salt, lab, ands, found)
        assert data_bits_from_tables(mode, parsed, lab, opened, found) == [x0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mul", ["shift_add", "booth"])
def test_one_extra_secret_key_opens_every_data_bit_of_a_linear_model(mode, mul):
    """I4 / h' (the attack of the 0.6 prototype) on ``linear_threshold(n=3,
    bw=4, bx=4)``, through the active backend's encryption with a seed from
    the operating system (the attack needs no secret of the encryptor). The
    holder plus ONE kept unselected secret key give ``Delta``; ``Delta``
    gives

    * the secret of every table entry -- every garbler-known bit of a
      half-gate, every permute bit and zero label of an AND(L, L) -- checked
      against the plain values of the wires;
    * every data bit read off the tables wherever a data wire enters an AND
      with a table: all of ``x`` for shift-and-add and in mode "labels"
      (Booth in mode "hardwired" recodes ``x`` with garbler-only gates);
    * ``C(w'', x)`` for every ``w''`` (both labels of every key wire), hence
      every feature ``x_i`` by binary search, in every case.
    """
    pc = model("threshold", mul)
    circuit = pc.circuit.guard().encode()
    m = pc.circuit.m
    rng = random.Random(2)
    _, x_values, w, x = model_values(pc, rng)
    secrets, pks, holder = full_key_slot(m, w, rng)
    ct = encrypt(circuit, mode, 5, x, pks)
    assert decrypt(circuit, mode, 5, ct, holder, w) == pc.circuit.evaluate_plain(w, x)
    labels = holder_labels(ct, 5, holder, w)
    j = rng.randrange(m)  # the one position whose unselected secret key the center kept
    delta = xor(labels[j], open_with_scalar(ct, 5, j, 1 - w[j], secrets[kem.table_index(m, j, 1 - w[j])]))
    assert lsb(delta) == 1

    salt = kem.instance_salt(PP, NONCE, 5)
    parsed, lab, ands = attacker_view(circuit, mode, salt, ct["tables"], labels, ct["data_active"])
    opened = what_delta_opens(salt, lab, ands, delta)
    known, permute, zero = opened
    plain = plain_wires(parsed, w + [0], x)
    zero_label = {v: xor(lab[v], delta) if plain[v] else lab[v] for v in itertools.chain(permute, zero)}
    assert len(known) + len(permute) + len(zero) > 0
    assert all(known[v] == plain[v] for v in known)
    assert all(permute[v] == lsb(zero_label[v]) for v in permute)
    assert all(zero[v] == zero_label[v] for v in zero)
    if mul == "shift_add" or mode == gc.MODE_LABELS:
        assert data_bits_from_tables(mode, parsed, lab, opened, delta) == x

    zero_keys = [xor(labels[i], delta) if w[i] else labels[i] for i in range(m)]

    def decide(key_bits) -> int:
        y = gc.evaluate(circuit, mode, salt, ct["tables"], ct["decode"],
                        b"".join(any_key(zero_keys, labels[m], delta, key_bits)), ct["data_active"])
        return gc.unpack_bits(y, 1)[0]

    assert recover_features(pc, decide) == x_values["x"]


def test_h_prime_slots_have_no_unselected_secret_key():
    """I4 / h' and I13: the library's issuance -- the reference ``new_slot``
    and ``place_slot`` and the active backend's ``bcfe_new_slot`` and
    ``bcfe_place_slot`` -- returns exactly ``m + 1`` scalars (h1) or none
    (h2, the holder's own). Each selected position and the g bit hold ``s B``
    for one of the ``m + 1``; each unselected position holds
    ``unselected_pk(setup, slot, j, k)``, an ``H_P`` output whose discrete
    logarithm nobody knows. The situation of the tests above cannot come
    from a center that fails to erase: there is nothing to erase."""
    pc = model("threshold", "booth")
    m = pc.circuit.m
    rng = random.Random(3)
    w = model_values(pc, rng)[2]
    ops = native_backend.ops()
    holder = [draw_scalar(rng) for _ in range(m + 1)]
    holder_pks = b"".join(point(s) for s in holder)
    slots = {
        "h1 reference": kem.new_slot(SETUP, 11, w, kem.byte_source(rng.randbytes(64 * (m + 1) + 16))),
        "h1 backend": ops.bcfe_new_slot(SETUP, 11, pack(w), m=m),
        "h2 reference": kem.place_slot(SETUP, 11, w, holder_pks, kem.byte_source(rng.randbytes(16))),
        "h2 backend": ops.bcfe_place_slot(SETUP, 11, pack(w), holder_pks, m=m),
    }
    for name, slot in slots.items():
        if name.startswith("h1"):
            assert set(slot) == {"scalars", "pks", "k"}, name
            assert len(slot["scalars"]) == 32 * (m + 1), name
            scalars = [kem.decode_scalar(s) for s in split(slot["scalars"], 32)]
        else:
            assert set(slot) == {"pks", "k"}, name  # the center's whole output: no scalar
            scalars = holder
        assert len(slot["k"]) == 16, name
        table = split(slot["pks"], 32)
        assert len(table) == 2 * m + 1, name
        for j in range(m):
            assert table[kem.table_index(m, j, w[j])] == point(scalars[j]), name
            assert table[kem.table_index(m, j, 1 - w[j])] == kem.unselected_pk(SETUP, 11, j, slot["k"]), name
        assert table[kem.table_index(m, m, 0)] == point(scalars[m]), name
        unselected = {table[kem.table_index(m, j, 1 - w[j])] for j in range(m)}
        assert len(unselected) == m and not unselected & {point(s) for s in scalars}, name
        assert kem.unselected_ok(SETUP, 11, w, slot["k"], slot["pks"]), name
        assert ops.bcfe_unselected_ok(SETUP, 11, pack(w), slot["k"], slot["pks"], m=m), name


# ------------------------------------------------------------------ h2 and the center (V6-F02)

@pytest.mark.parametrize("mode", MODES)
def test_h2_with_a_center_that_kept_its_secret_keys_is_a_total_break(mode):
    """Review finding V6-F02 (I4 holds for h2 as well), the review's
    reproduction: h2 without h' -- the holder generates the ``m + 1``
    selected secret keys, the center generates the ``m`` unselected ones and
    keeps them. The holder alone decrypts ``C(w, x)``; the center alone opens
    the ``m`` labels of the complement of ``w`` and no label of the g bit;
    together they open both labels of every key wire, the XOR is the same
    ``Delta`` at every position, and ``x`` falls out of the tables."""
    pc = model("threshold", "shift_add")
    circuit = pc.circuit.guard().encode()
    m = pc.circuit.m
    rng = random.Random(4)
    _, _, w, x = model_values(pc, rng)
    secrets, pks, holder = full_key_slot(m, w, rng)
    center = [secrets[kem.table_index(m, j, 1 - w[j])] for j in range(m)]
    seed = rng.randbytes(32)
    ct = encrypt(circuit, mode, 9, x, pks, seed=seed)
    key0, delta = truth(circuit, mode, 9, x, seed)
    assert decrypt(circuit, mode, 9, ct, holder, w) == pc.circuit.evaluate_plain(w, x)
    labels = holder_labels(ct, 9, holder, w)
    complement = [open_with_scalar(ct, 9, j, 1 - w[j], center[j]) for j in range(m)]
    assert complement == active(key0, delta, [1 - bit for bit in w])[:m]
    labels_of_all = every_label(key0, delta)
    assert all(open_with_scalar(ct, 9, m, 0, s) not in labels_of_all for s in center)  # no g label
    assert {xor(labels[j], complement[j]) for j in range(m)} == {delta}
    salt = kem.instance_salt(PP, NONCE, 9)
    parsed, lab, ands = attacker_view(circuit, mode, salt, ct["tables"], labels, ct["data_active"])
    assert data_bits_from_tables(mode, parsed, lab, what_delta_opens(salt, lab, ands, delta), delta) == x


@pytest.mark.parametrize("issuance", ["h1", "h2"])
def test_holder_and_center_together_cannot_open_an_unselected_label(issuance):
    """V6-F02 under h' (I4, I13, I14): the coalition of the holder and the
    center has the ``m + 1`` selected secret keys (h1: the center drew them
    and may keep them; h2: the holder's own), ``w`` and the slot's ``k`` --
    the center's only secret of the slot, returned beside the table and no
    scalar. ``k`` recomputes every unselected public key (CheckAuth) but is
    no secret key of any: neither the coalition's scalars nor scalars or
    ``Z`` values derived from ``k`` open ``c[j, 1 - w_j]`` to ``L_j^(1 -
    w_j)``, or to any label of the garbling (compared with the true labels
    of the seeded garbling; the holder's own labels open as they should).
    The coalition keeps one label per key wire and no ``Delta``."""
    pc = model("threshold", "booth")
    circuit = pc.circuit.guard().encode()
    m = pc.circuit.m
    rng = random.Random(5 if issuance == "h1" else 6)
    _, _, w, x = model_values(pc, rng)
    ops = native_backend.ops()
    if issuance == "h1":
        slot = ops.bcfe_new_slot(SETUP, 13, pack(w), m=m)
        assert set(slot) == {"scalars", "pks", "k"}
        coalition = [kem.decode_scalar(s) for s in split(slot["scalars"], 32)]
    else:
        coalition = [draw_scalar(rng) for _ in range(m + 1)]
        slot = ops.bcfe_place_slot(SETUP, 13, pack(w), b"".join(point(s) for s in coalition), m=m)
        assert set(slot) == {"pks", "k"}
    k, table = slot["k"], split(slot["pks"], 32)
    assert ops.bcfe_unselected_ok(SETUP, 13, pack(w), k, slot["pks"], m=m)

    seed = rng.randbytes(32)
    ct = encrypt(circuit, "hardwired", 13, x, slot["pks"], seed=seed)
    key0, delta = truth(circuit, "hardwired", 13, x, seed)
    assert holder_labels(ct, 13, encode_scalars(coalition), w) == active(key0, delta, w)
    labels_of_all = every_label(key0, delta)
    point_r = r255.decode(ct["R"])
    for j in range(m):
        b = 1 - w[j]
        unselected = table[kem.table_index(m, j, b)]
        assert unselected == kem.unselected_pk(SETUP, 13, j, k)
        msg = kem.UNSELECTED_PREFIX + SETUP + (13).to_bytes(8, "little") + j.to_bytes(4, "little") + k
        derived = [int.from_bytes(k, "little"),
                   int.from_bytes(hashlib.sha512(k).digest(), "little") % r255.L,
                   int.from_bytes(hashlib.shake_256(msg).digest(64), "little") % r255.L,
                   int.from_bytes(r255.expand_message_xmd(msg, kem.UNSELECTED_DST, 64), "little") % r255.L]
        assert all(point(s) != unselected for s in coalition + derived)
        zs = [r255.encode(r255.mul(point_r, s)) for s in coalition + derived] + [unselected, ct["R"]]
        assert all(open_position(ct, 13, j, b, z) not in labels_of_all for z in zs)


# ------------------------------------------------------------------ one Delta per (ciphertext, slot) (I1)

@pytest.mark.parametrize("mode", MODES)
def test_a_seed_shared_by_two_slots_lets_their_holders_open_x(mode):
    """I1, with the instance salt in the seed expansion. If the components
    of two slots came from one seed (only through the internal interface:
    the tests' seed hook, I16), they still have unrelated ``Delta`` and zero
    labels and tables with no entry in common: the expansion reads the
    instance salt (pp digest, nonce, slot id) as well as the seed, so the
    two holders' labels share nothing.

    The attack now needs the same seed *and* the same salt: slot B's
    component made with slot A's instance fields (slot id 1, B's table) from
    A's seed -- an encryptor that reuses both -- shares ``Delta`` and every
    zero label with A's. Then two honest holders with their own h' slots
    and ``w_A != w_B``, and no extra key, XOR their labels where ``w_A`` and
    ``w_B`` differ: ``Delta``; their labels coincide where they agree; ``x``
    falls out of the tables: the "one slot, two keys" attack across slots,
    by a coalition of holders the scheme otherwise allows without limit. On
    the library path (a fresh seed per call) the same two holders' labels
    share nothing either: no two XORs agree and no label coincides."""
    pc = model("threshold", "shift_add")
    circuit = pc.circuit.guard().encode()
    m = pc.circuit.m
    rng = random.Random(7)
    _, _, w_a, x = model_values(pc, rng)
    diff = sorted(rng.sample(range(m), 3))
    w_b = [1 - bit if j in diff else bit for j, bit in enumerate(w_a)]
    ops = native_backend.ops()
    slot_a = ops.bcfe_new_slot(SETUP, 1, pack(w_a), m=m)
    slot_b = ops.bcfe_new_slot(SETUP, 2, pack(w_b), m=m)

    seed = rng.randbytes(32)
    # one seed, two slots: two salts, unrelated garblings
    ct_a = encrypt(circuit, mode, 1, x, slot_a["pks"], seed=seed)
    ct_b = encrypt(circuit, mode, 2, x, slot_b["pks"], seed=seed)
    key0, delta = truth(circuit, mode, 1, x, seed)
    key0_b, delta_b = truth(circuit, mode, 2, x, seed)
    assert delta_b != delta and set(key0_b).isdisjoint(key0)
    assert set(split(ct_a["tables"])).isdisjoint(split(ct_b["tables"]))
    labels_a = holder_labels(ct_a, 1, slot_a["scalars"], w_a)
    labels_b = holder_labels(ct_b, 2, slot_b["scalars"], w_b)
    assert len({xor(labels_a[j], labels_b[j]) for j in diff}) == len(diff)
    assert set(labels_a).isdisjoint(labels_b)

    # one seed and one salt: B's component under A's instance fields shares A's garbling
    ct_b = encrypt(circuit, mode, 1, x, slot_b["pks"], seed=seed)
    assert (ct_b["tables"], ct_b["decode"], ct_b["data_active"]) == (ct_a["tables"], ct_a["decode"],
                                                                     ct_a["data_active"])
    labels_b = holder_labels(ct_b, 1, slot_b["scalars"], w_b)
    assert {xor(labels_a[j], labels_b[j]) for j in diff} == {delta}
    assert all(labels_a[j] == labels_b[j] for j in range(m + 1) if j not in diff)
    salt_a = kem.instance_salt(PP, NONCE, 1)
    parsed, lab, ands = attacker_view(circuit, mode, salt_a, ct_a["tables"], labels_a, ct_a["data_active"])
    assert data_bits_from_tables(mode, parsed, lab, what_delta_opens(salt_a, lab, ands, delta), delta) == x

    fresh_a = encrypt(circuit, mode, 1, x, slot_a["pks"])
    fresh_b = encrypt(circuit, mode, 2, x, slot_b["pks"])
    labels_a = holder_labels(fresh_a, 1, slot_a["scalars"], w_a)
    labels_b = holder_labels(fresh_b, 2, slot_b["scalars"], w_b)
    assert len({xor(labels_a[j], labels_b[j]) for j in diff}) == len(diff)
    assert set(labels_a).isdisjoint(labels_b)


def test_every_component_of_the_library_has_its_own_delta():
    """I1 on the library path: the active backend's ``bcfe_encrypt_slot``
    (native by default) draws a fresh seed for every (ciphertext, slot).
    With test-only full-key slots (every position's secret key known) the
    test opens both labels of every key wire of each component: one
    ``Delta`` per component (the XOR is the same at every position), and two
    ciphertexts times two slots give four different ``Delta``, four
    different ``R`` and no zero label in common."""
    pc = model("threshold", "booth")
    circuit = pc.circuit.guard().encode()
    m = pc.circuit.m
    rng = random.Random(8)
    _, _, w, x = model_values(pc, rng)
    slots = {slot_id: full_key_slot(m, w, rng) for slot_id in (1, 2)}
    found = []
    for nonce in (NONCE, NONCE2):
        for slot_id, (secrets, pks, holder) in slots.items():
            ct = encrypt(circuit, "hardwired", slot_id, x, pks, nonce=nonce)
            assert decrypt(circuit, "hardwired", slot_id, ct, holder, w, nonce=nonce) == \
                pc.circuit.evaluate_plain(w, x)
            pairs = [[open_with_scalar(ct, slot_id, j, b, secrets[kem.table_index(m, j, b)], nonce=nonce)
                      for b in (0, 1)] for j in range(m)]
            deltas = {xor(l0, l1) for l0, l1 in pairs}
            assert len(deltas) == 1
            delta = deltas.pop()
            assert lsb(delta) == 1
            zeros = [l0 for l0, _ in pairs] + [open_with_scalar(ct, slot_id, m, 0, secrets[2 * m], nonce=nonce)]
            found.append((delta, ct["R"], set(zeros)))
    assert len({delta for delta, _, _ in found}) == 4
    assert len({big_r for _, big_r, _ in found}) == 4
    assert all(a[2].isdisjoint(b[2]) for a, b in itertools.combinations(found, 2))


# ------------------------------------------------------------------ mixing components (I2, I11)

@pytest.mark.parametrize("mode", MODES)
def test_mixed_components_open_to_garbage(mode):
    """I2 (``H_K`` binds the pp digest, the nonce, the slot, the position,
    the bit and ``R``) and I11 (``H_G`` binds the instance salt): a holder
    that combines parts of different components gets nothing of any
    garbling. Three components of ``linear_score`` (10 output bits) with
    known seeds -- (ciphertext 1, slot A), (ciphertext 1, slot B),
    (ciphertext 2, slot A). The holder of slot A

    * opens the label ciphertexts of another ciphertext or slot (with their
      ``R`` or its own) as its own: none of the labels of any of the three
      garblings;
    * evaluates its true labels on the tables and decoding bits of another
      ciphertext or slot (under either salt): the output labels are neither
      label of any output wire of any of the three garblings, i.e. garbage;
      ``bcfe_decrypt_slot`` returns bits without an error (one error class
      at the scheme layer, I10).
    """
    pc = model("score", "booth")
    circuit = pc.circuit.guard().encode()
    m, n_out = pc.circuit.m, pc.circuit.n_out
    rng = random.Random(9)
    _, _, w_a, x = model_values(pc, rng)
    w_b = model_values(pc, rng)[2]
    ops = native_backend.ops()
    slot_a = ops.bcfe_new_slot(SETUP, 1, pack(w_a), m=m)
    slot_b = ops.bcfe_new_slot(SETUP, 2, pack(w_b), m=m)
    setting = {"1A": (NONCE, 1, slot_a, w_a), "1B": (NONCE, 2, slot_b, w_b), "2A": (NONCE2, 1, slot_a, w_a)}
    cts, salts, labels_of_all, outputs = {}, {}, set(), set()
    for name, (nonce, slot_id, slot, w) in setting.items():
        seed = rng.randbytes(32)
        cts[name] = ct = encrypt(circuit, mode, slot_id, x, slot["pks"], seed=seed, nonce=nonce)
        salts[name] = kem.instance_salt(PP, nonce, slot_id)
        key0, delta = truth(circuit, mode, slot_id, x, seed, nonce=nonce)
        labels_of_all |= every_label(key0, delta)
        honest = holder_labels(ct, slot_id, slot["scalars"], w, nonce=nonce)
        assert honest == active(key0, delta, w)
        out = output_labels(circuit, mode, salts[name], ct["tables"], honest, ct["data_active"])
        outputs |= set(out) | {xor(label, delta) for label in out}
    own = cts["1A"]

    def opened(label_cts: bytes, big_r: bytes, slot_id: int = 1) -> list:
        return split(kem.recover_labels(PP, NONCE, slot_id, big_r, label_cts, slot_a["scalars"], w_a))

    assert set(opened(own["label_cts"], own["R"])) <= labels_of_all
    for other in ("2A", "1B"):
        for label_cts, big_r in ((cts[other]["label_cts"], cts[other]["R"]), (cts[other]["label_cts"], own["R"]),
                                 (own["label_cts"], cts[other]["R"])):
            assert labels_of_all.isdisjoint(opened(label_cts, big_r))
    assert labels_of_all.isdisjoint(opened(cts["1B"]["label_cts"], cts["1B"]["R"], slot_id=2))

    labels = holder_labels(own, 1, slot_a["scalars"], w_a)
    for other in ("2A", "1B"):
        for salt in (salts["1A"], salts[other]):
            out = output_labels(circuit, mode, salt, cts[other]["tables"], labels, own["data_active"])
            assert outputs.isdisjoint(out)
        y = ops.bcfe_decrypt_slot(circuit, mode, PP, NONCE, 1, own["R"], own["label_cts"], cts[other]["tables"],
                                  cts[other]["decode"], own["data_active"], slot_a["scalars"], pack(w_a))
        assert len(gc.unpack_bits(y, n_out)) == n_out


# ------------------------------------------------------------------ the g bit (I3)

@pytest.mark.parametrize("mode", MODES)
def test_the_g_bit_has_one_position_and_forces_zero(mode):
    """I3: a slot has ``2m + 1`` public keys and a component ``2m + 1``
    label ciphertexts; ``c[m, 1]`` does not exist (``table_index`` refuses
    ``(m, 1)``), and with every position's secret key known (a test-only
    full-key slot) the ``2m + 1`` ciphertexts open to both labels of every
    key bit and ``L_g^0``, never ``L_g^1``. Forcing ``g = 1`` anyway
    (``L_g^1 = L_g^0 xor Delta`` from the seeded garbling, test-only) gives
    the all-zero output for every key, including keys with ``C(w'', x) !=
    0``: ``C'(w'' || 1, x) = C(w'', x) AND NOT 1``. This is what the security
    argument uses for a slot whose key the adversary never gets."""
    pc = model("score", "booth")
    circuit = pc.circuit.guard().encode()
    m, n_out = pc.circuit.m, pc.circuit.n_out
    rng = random.Random(10)
    _, _, w, x = model_values(pc, rng)
    slot = native_backend.ops().bcfe_new_slot(SETUP, 4, pack(w), m=m)
    seed = rng.randbytes(32)
    ct = encrypt(circuit, mode, 4, x, slot["pks"], seed=seed)
    assert len(slot["pks"]) == 32 * (2 * m + 1) and len(ct["label_cts"]) == 16 * (2 * m + 1)
    with pytest.raises(ValueError, match="g bit has the one position"):
        kem.table_index(m, m, 1)
    key0, delta = truth(circuit, mode, 4, x, seed)
    salt = kem.instance_salt(PP, NONCE, 4)

    secrets, pks, _ = full_key_slot(m, w, rng)
    full = encrypt(circuit, mode, 4, x, pks, seed=seed)
    every = [open_with_scalar(full, 4, j, b, secrets[kem.table_index(m, j, b)])
             for j in range(m + 1) for b in ((0, 1) if j < m else (0,))]
    assert set(every) == every_label(key0, delta) - {xor(key0[m], delta)}

    def run(labels) -> list:
        y = gc.evaluate(circuit, mode, salt, ct["tables"], ct["decode"], b"".join(labels), ct["data_active"])
        return gc.unpack_bits(y, n_out)

    labels = holder_labels(ct, 4, slot["scalars"], w)
    assert labels == active(key0, delta, w)
    keys = [w] + [model_values(pc, rng)[2] for _ in range(6)]
    results = []
    for key_bits in keys:
        honest = active(key0, delta, key_bits)
        results.append(run(honest))
        assert results[-1] == pc.circuit.evaluate_plain(key_bits, x)
        assert run(honest[:m] + [xor(honest[m], delta)]) == [0] * n_out
    assert any(any(bits) for bits in results)


# ------------------------------------------------------------------ the instance salt (I11, V6-F04)

@pytest.mark.parametrize("mode", MODES)
def test_the_instance_salt_separates_garblings_of_one_seed(mode):
    """I11: the gate hash takes the instance salt (pp digest, ciphertext
    nonce, slot id), and so does the seed expansion. One seed under four
    salts (each field changed in turn) gives four different ``Delta``, zero
    labels with none in common (a seed reused across slots no longer shares
    ``Delta``) and tables with no entry in common, and the tables of one
    instance evaluated with the right labels under another instance's salt
    give output labels of neither value. Each instance has its own gate hash, so a test of a
    guess of ``Delta`` against the tables concerns one instance at a time."""
    pc = model("threshold", "booth")
    circuit = pc.circuit.guard().encode()
    m = pc.circuit.m
    rng = random.Random(11)
    _, _, w, x = model_values(pc, rng)
    seed = rng.randbytes(32)
    salts = {"base": kem.instance_salt(PP, NONCE, 7), "nonce": kem.instance_salt(PP, NONCE2, 7),
             "slot": kem.instance_salt(PP, NONCE, 8), "pp": kem.instance_salt(PP2, NONCE, 7)}
    garbled = {name: gc.garble(circuit, mode, salt, x, seed) for name, salt in salts.items()}
    base = garbled["base"]
    assert gc.garble(circuit, mode, salts["base"], x, seed) == base
    assert len({g["delta"] for g in garbled.values()}) == 4
    for a, b in itertools.combinations(garbled.values(), 2):
        assert set(split(a["key0"])).isdisjoint(split(b["key0"]))
        assert set(split(a["data_active"])).isdisjoint(split(b["data_active"]))
        assert set(split(a["tables"])).isdisjoint(split(b["tables"]))

    delta = base["delta"]
    labels = active(split(base["key0"]), delta, w)
    assert len(labels) == m + 1
    out = output_labels(circuit, mode, salts["base"], base["tables"], labels, base["data_active"])
    pairs = set(out) | {xor(label, delta) for label in out}
    for name in ("nonce", "slot", "pp"):
        assert pairs.isdisjoint(output_labels(circuit, mode, salts[name], base["tables"], labels,
                                              base["data_active"]))
