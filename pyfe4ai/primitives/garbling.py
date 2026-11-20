"""Half-gates garbled circuits: the reference algorithm of the ``bcfe`` schemes.

Garbling and evaluation of a boolean circuit given in its canonical encoding
``CIRC v1`` (XOR, AND and INV gates, key wires carrying the bits of ``w``,
data wires carrying the bits of ``x``), as the bounded-key schemes
``bcfe/*`` use it: the key holder evaluates with the labels of ``w`` and
learns ``C(w, x)``.

The construction is the half-gates garbling of Zahur, Rosulek and Evans,
"Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using
Half Gates" (EUROCRYPT 2015, ZRE15), Fig. 2: free XOR with a global offset
``Delta`` (``lsb(Delta) = 1``), point-and-permute, two table entries per AND
gate of two labelled inputs, the generator half-gate (one entry) for an AND
gate with one input the garbler knows, and output decoding by the permute
bits. The gate hash is ``H_G(salt, idx, X) = SHAKE128("PFE6GC02" || salt ||
u64(idx) || X)[0:16]`` (one Keccak-f permutation, modelled as a random
oracle); the fixed-key AES hashes of Guo, Katz, Wang and Yu (GKWY19) are not
used.

Two kinds of wires (:func:`wire_kinds`): ``L`` wires carry labels, ``K``
wires values the garbler knows. Key wires are always ``L``; data wires are
``K`` in mode ``"hardwired"`` (the garbler folds ``x`` into the tables) and
``L`` in mode ``"labels"`` (the evaluator receives their active labels). All
randomness -- ``Delta`` and the zero labels -- comes from one 32-byte seed
expanded with SHAKE-256 together with the instance salt
(:func:`expand_seed`); :func:`garble` takes the seed, which the backend
operation ``gc_garble`` draws fresh for every call.

What the security argument requires of this module, and how it is met:

* fresh labels -- every label and ``Delta`` is derived from the per-call
  seed, nothing is reused between calls (the caller supplies a fresh seed:
  the backend operation from the operating system); the instance salt is
  expanded with the seed, so a repeated seed under another salt still gives
  unrelated labels;
* hash indices -- the hash index follows ZRE15's NextIndex: a fresh index
  per hash group, two for an AND gate of two labelled inputs, one for a
  generator half-gate, none for the other gates, in gate order;
* instance salt -- the gate hash takes the instance salt (pp digest,
  ciphertext nonce, slot id) as a fixed 56-byte string;
* a structure independent of ``x`` -- the kinds of the wires, which gates
  have tables and their number depend on the circuit and the mode only,
  never on ``x`` (the table contents do); no gate is folded or skipped
  because of ``x``; the garbler's selections by
  ``x`` (values of ``K`` wires, ``val * Delta``) are written as masks, not
  branches;
* injective hash inputs -- the inputs of ``H_G`` have fixed lengths (8 +
  56 + 8 + 16 bytes).

This Python reference makes no constant-time claim (Python integers and
``hashlib`` are not constant time) and cannot wipe its copies of ``Delta``,
the labels or ``x``; the native implementation of the backend does both.
Bit strings are packed least significant bit first (:func:`pack_bits`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct

#: Magic and version of the canonical circuit encoding.
CIRCUIT_MAGIC = b"PFE6CIRC"
CIRCUIT_VERSION = 1
#: Gate operations of ``CIRC v1``.
OP_XOR, OP_AND, OP_INV = 1, 2, 3
#: Bound on ``m + n + n_gates``: 2^26 wires (64 Mi wires, about 1 GiB of native
#: labels; the largest circuit the schemes target, an MLP 784-128-10, is far
#: below), so that a short header cannot make a decoder's caller allocate
#: without bound. Design choice; the alternative was the earlier 2^31 (wire
#: numbers in 31 bits) with the per-wire state sized by the labelled wires only.
MAX_CIRCUIT_WIRES = 2 ** 26
#: The same bound (the name the other modules use).
MAX_WIRES = MAX_CIRCUIT_WIRES

#: The garbling modes (record field ``data_wires``).
MODE_HARDWIRED = "hardwired"
MODE_LABELS = "labels"
MODES = (MODE_HARDWIRED, MODE_LABELS)
#: Wire kinds (:func:`wire_kinds`): ``K`` known to the garbler, ``L`` labelled.
KIND_K, KIND_L = 0, 1

#: Lengths in bytes of a label, of the instance salt and of the seed.
LABEL_BYTES = 16
SALT_BYTES = 56
SEED_BYTES = 32
#: Domain-separation strings of the gate hash and of the seed expansion.
GATE_HASH_DST = b"PFE6GC02"
PRG_DST = b"PYFE4AI-V06-BCFE-GBPRG"

_HEADER = struct.Struct("<8sBIIII")
_GATE = struct.Struct("<BII")
_OP_NAMES = {OP_XOR: "XOR", OP_AND: "AND", OP_INV: "INV"}


@dataclasses.dataclass(frozen=True)
class ParsedCircuit:
    """A decoded ``CIRC v1`` circuit (:func:`parse_circuit`): ``m`` key bits,
    ``n`` data bits, the gates as ``(op, a, b)`` (``b = 0`` for INV; gate
    ``i`` writes wire ``m + n + i``) and the output wires."""

    m: int
    n: int
    gates: tuple
    outputs: tuple

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    @property
    def n_out(self) -> int:
        return len(self.outputs)

    @property
    def n_wires(self) -> int:
        return self.m + self.n + len(self.gates)

    def __repr__(self) -> str:
        return "ParsedCircuit(m={}, n={}, n_gates={}, n_out={})".format(self.m, self.n, self.n_gates, self.n_out)


def _as_bytes(data, what: str) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("{} must be bytes, not {}".format(what, type(data).__name__))
    return bytes(data)


def parse_circuit(data: bytes) -> ParsedCircuit:
    """Decode the canonical encoding ``CIRC v1`` of a circuit.

    Raises:
        TypeError: If *data* is not bytes.
        ValueError: For a wrong magic or version; a length that does not
            match the counts (missing or extra bytes); ``m = 0``, ``n_out =
            0`` or ``m + n + n_gates >= 2^26`` (:data:`MAX_CIRCUIT_WIRES`,
            checked before anything is sized by the counts); an unknown op, an INV gate
            with ``b != 0``, a XOR or AND gate with two equal inputs (the
            builder folds ``XOR(a, a)`` and ``AND(a, a)``), a gate input not
            below the gate's output wire; an output wire out of range.
    """
    data = _as_bytes(data, "circuit")
    if data[:len(CIRCUIT_MAGIC)] != CIRCUIT_MAGIC:
        raise ValueError("circuit: not a CIRC encoding (bad magic)")
    if len(data) > len(CIRCUIT_MAGIC) and data[len(CIRCUIT_MAGIC)] != CIRCUIT_VERSION:
        raise ValueError("circuit: unsupported CIRC version {}".format(data[len(CIRCUIT_MAGIC)]))
    if len(data) < _HEADER.size:
        raise ValueError("circuit: {} bytes, shorter than the {}-byte header".format(len(data), _HEADER.size))
    _, _, m, n, n_gates, n_out = _HEADER.unpack_from(data)
    if m == 0:
        raise ValueError("circuit: m = 0 (a circuit needs at least one key bit)")
    if n_out == 0:
        raise ValueError("circuit: n_out = 0 (a circuit needs at least one output)")
    if m + n + n_gates >= MAX_CIRCUIT_WIRES:
        raise ValueError("circuit: m + n + n_gates = {} is not below 2^26".format(m + n + n_gates))
    expected = _HEADER.size + _GATE.size * n_gates + 4 * n_out
    if len(data) != expected:
        raise ValueError("circuit: {} bytes, the counts give {}".format(len(data), expected))
    body_end = _HEADER.size + _GATE.size * n_gates
    gates = []
    out = m + n
    for i, (op, a, b) in enumerate(_GATE.iter_unpack(data[_HEADER.size:body_end])):
        if op not in _OP_NAMES:
            raise ValueError("circuit: gate {} has an unknown op {}".format(i, op))
        if op == OP_INV and b != 0:
            raise ValueError("circuit: gate {} (INV) has b = {}, must be 0".format(i, b))
        # canonical form, not a security rule: the builder folds XOR(a, a) and AND(a, a); other degenerate gates
        # (XOR(a, INV(a)), duplicated gates) still decode and are garbled safely
        if op != OP_INV and a == b:
            raise ValueError("circuit: gate {} ({}) reads wire {} twice".format(i, _OP_NAMES[op], a))
        if a >= out or (op != OP_INV and b >= out):
            raise ValueError("circuit: gate {} ({}) reads wire {} not below its output wire {}".format(
                i, _OP_NAMES[op], a if a >= out else b, out))
        gates.append((op, a, b))
        out += 1
    outputs = struct.unpack_from("<{}I".format(n_out), data, body_end)
    for k, v in enumerate(outputs):
        if v >= out:
            raise ValueError("circuit: output {} is wire {}, out of range (the circuit has {} wires)".format(
                k, v, out))
    return ParsedCircuit(m, n, tuple(gates), tuple(outputs))


def _check_mode(mode) -> str:
    if mode not in MODES:
        raise ValueError("unknown garbling mode {!r}; choose one of {}".format(mode, ", ".join(MODES)))
    return mode


def wire_kinds(parsed: ParsedCircuit, mode: str) -> bytes:
    """The kind of every wire (:data:`KIND_L` or :data:`KIND_K`, one byte per
    wire): key wires ``L``, data wires ``K`` (``"hardwired"``) or ``L``
    (``"labels"``); a XOR or AND output is ``L`` if an input is, an INV
    output has its input's kind. Depends on the circuit and the mode only.

    Raises:
        ValueError: For an unknown mode.
    """
    _check_mode(mode)
    data_kind = KIND_L if mode == MODE_LABELS else KIND_K
    kinds = bytearray([KIND_L]) * parsed.m + bytearray([data_kind]) * parsed.n
    for op, a, b in parsed.gates:
        kinds.append(kinds[a] if op == OP_INV else kinds[a] | kinds[b])
    return bytes(kinds)


def table_entries(parsed: ParsedCircuit, mode: str) -> int:
    """Number of 16-byte table entries: two per AND gate of two ``L`` inputs,
    one per AND gate of one ``L`` and one ``K`` input."""
    kinds = wire_kinds(parsed, mode)
    count = 0
    for op, a, b in parsed.gates:
        if op == OP_AND:
            count += kinds[a] + kinds[b]
    return count


def _require_label_outputs(parsed: ParsedCircuit, kinds: bytes, mode: str) -> None:
    for k, v in enumerate(parsed.outputs):
        if kinds[v] != KIND_L:
            raise ValueError("output {} (wire {}) is not a labelled wire in mode {!r}: garble guard(C)".format(
                k, v, mode))


def _check_salt(salt) -> bytes:
    salt = _as_bytes(salt, "salt")
    if len(salt) != SALT_BYTES:
        raise ValueError("the salt must be {} bytes, got {}".format(SALT_BYTES, len(salt)))
    return salt


def gate_hash(salt: bytes, idx: int, x: bytes) -> bytes:
    """``H_G(salt, idx, X) = SHAKE128("PFE6GC02" || salt || u64(idx) || X)[0:16]``.

    Raises:
        ValueError: If *salt* is not 56 bytes, *x* not 16 bytes, or *idx*
            not in ``[0, 2^64)``.
    """
    salt = _check_salt(salt)
    x = _as_bytes(x, "x")
    if len(x) != LABEL_BYTES:
        raise ValueError("a label must be {} bytes, got {}".format(LABEL_BYTES, len(x)))
    if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < 2 ** 64:
        raise ValueError("the gate-hash index must be an integer in [0, 2^64)")
    return hashlib.shake_128(GATE_HASH_DST + salt + idx.to_bytes(8, "little") + x).digest(LABEL_BYTES)


def _label_hash(salt: bytes):
    """:func:`gate_hash` on labels as little-endian integers (the loops' form)."""
    prefix = GATE_HASH_DST + salt
    shake = hashlib.shake_128

    def h(label: int, idx: int) -> int:
        return int.from_bytes(shake(prefix + idx.to_bytes(8, "little") + label.to_bytes(16, "little")).digest(16),
                              "little")

    return h


def expand_seed(salt: bytes, seed: bytes, m_prime: int, n_data_labels: int) -> tuple:
    """``Delta`` and the zero labels of a garbling, from its instance salt and 32-byte seed.

    ``SHAKE-256("PYFE4AI-V06-BCFE-GBPRG" || salt(56) || seed(32))`` read in
    order: ``Delta`` (16 bytes, then ``Delta[0] |= 1``), the zero label of
    every key wire ``0 .. m_prime - 1``, then of every data wire
    (``n_data_labels``: ``n`` in mode ``"labels"``, 0 otherwise). Returns
    ``(delta, key0, data0)``: 16 bytes and two lists of 16-byte labels.

    The salt (that of the gate hash, :func:`gate_hash`: ``pp_digest ||
    ct_nonce || u64(slot_id)``) is bound in so that one seed used for two
    (ciphertext, slot) instances still gives unrelated ``Delta`` and labels:
    defence in depth against a repeated seed. Design choice; the alternative
    was the seed alone, relying only on a fresh seed per instance from the
    operating system.

    Raises:
        TypeError: If *salt* or *seed* is not bytes.
        ValueError: If *salt* is not 56 bytes, *seed* not 32 bytes or a
            count is negative.
    """
    salt = _check_salt(salt)
    seed = _as_bytes(seed, "seed")
    if len(seed) != SEED_BYTES:
        raise ValueError("the garbling seed must be {} bytes, got {}".format(SEED_BYTES, len(seed)))
    for name, count in (("m_prime", m_prime), ("n_data_labels", n_data_labels)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("{} must be a non-negative integer".format(name))
    stream = hashlib.shake_256(PRG_DST + salt + seed).digest(LABEL_BYTES * (1 + m_prime + n_data_labels))
    delta = bytes([stream[0] | 1]) + stream[1:LABEL_BYTES]
    labels = [stream[i:i + LABEL_BYTES] for i in range(LABEL_BYTES, len(stream), LABEL_BYTES)]
    return delta, labels[:m_prime], labels[m_prime:]


def pack_bits(bits, n: int) -> bytes:
    """*n* bits (0 / 1) packed least significant bit first: bit ``i`` is bit
    ``i % 8`` of byte ``i // 8``; the unused high bits of the last byte are 0.

    Raises:
        ValueError: If *bits* does not hold exactly *n* values 0 or 1.
    """
    bits = list(bits)
    if len(bits) != n:
        raise ValueError("expected {} bits, got {}".format(n, len(bits)))
    out = bytearray((n + 7) // 8)
    for i, bit in enumerate(bits):
        if not isinstance(bit, int) or bit not in (0, 1):
            raise ValueError("bit {} is not 0 or 1".format(i))
        out[i >> 3] |= bit << (i & 7)
    return bytes(out)


def unpack_bits(data: bytes, n: int) -> list:
    """The *n* bits of :func:`pack_bits`'s encoding, as ints 0 / 1.

    Raises:
        ValueError: If *data* is not ``ceil(n / 8)`` bytes, or an unused high
            bit of the last byte is set.
    """
    data = _as_bytes(data, "packed bits")
    if len(data) != (n + 7) // 8:
        raise ValueError("{} packed bits take {} bytes, got {}".format(n, (n + 7) // 8, len(data)))
    if n % 8 and data[-1] >> (n % 8):
        raise ValueError("nonzero padding bits after bit {}".format(n))
    return [(data[i >> 3] >> (i & 7)) & 1 for i in range(n)]


def _data_bits(x_bits, n: int) -> list:
    if isinstance(x_bits, (bytes, bytearray, memoryview)):
        return unpack_bits(x_bits, n)
    return unpack_bits(pack_bits(x_bits, n), n)


def _labels(data, count: int, what: str) -> list:
    """*count* 16-byte strings of *data* as little-endian integers."""
    data = _as_bytes(data, what)
    if len(data) != LABEL_BYTES * count:
        raise ValueError("{}: expected {} entries of {} bytes ({} bytes), got {} bytes".format(
            what, count, LABEL_BYTES, LABEL_BYTES * count, len(data)))
    return [int.from_bytes(data[i:i + LABEL_BYTES], "little") for i in range(0, len(data), LABEL_BYTES)]


def _join(labels: list) -> bytes:
    return b"".join(v.to_bytes(LABEL_BYTES, "little") for v in labels)


def garble(circuit: bytes, mode: str, salt: bytes, x_bits, seed: bytes) -> dict:
    """Garble the ``CIRC v1`` *circuit* for the data bits *x_bits* (ZRE15 Fig. 2).

    Args:
        circuit: The encoding (usually of ``guard(C)``).
        mode: ``"hardwired"`` (data wires known to the garbler) or ``"labels"``.
        salt: The 56-byte instance salt ``pp_digest || ct_nonce || u64(slot_id)``
            (opaque here).
        x_bits: The ``n`` data bits: a list of 0 / 1, or packed (:func:`pack_bits`).
        seed: 32 bytes, all the randomness of the garbling (:func:`expand_seed`);
            fresh from a CSPRNG for every garbling, fixed only in tests.

    Returns:
        ``{"delta": 16 B, "key0": 16 B x m (the zero labels of the key wires),
        "data_active": 16 B x n (the active data labels ``W0 xor x_i Delta``,
        mode ``"labels"``) or b"", "tables": 16 B x table entries,
        "decode": the n_out decoding bits packed}``. ``delta``, ``key0`` and
        ``data_active`` are secret.

    Raises:
        TypeError: For inputs that are not bytes (*x_bits* may be a list).
        ValueError: For a malformed circuit (:func:`parse_circuit`), an
            unknown mode, a salt of another length than 56 bytes, a seed of
            another length than 32, data bits that are not ``n`` bits, or an
            output wire that is not a labelled wire in this mode.
    """
    parsed = parse_circuit(circuit)
    kinds = wire_kinds(parsed, mode)
    salt = _check_salt(salt)
    m, n = parsed.m, parsed.n
    x = _data_bits(x_bits, n)
    _require_label_outputs(parsed, kinds, mode)
    labels_mode = mode == MODE_LABELS
    delta_bytes, key0, data0 = expand_seed(salt, seed, m, n if labels_mode else 0)
    delta = int.from_bytes(delta_bytes, "little")

    lab = [0] * parsed.n_wires  # W0 of the L wires
    val = [0] * parsed.n_wires  # values of the K wires
    for i, w0 in enumerate(key0):
        lab[i] = int.from_bytes(w0, "little")
    data_active = []
    if labels_mode:
        for i, w0 in enumerate(data0):
            lab[m + i] = int.from_bytes(w0, "little")
            data_active.append(lab[m + i] ^ (-x[i] & delta))
    else:
        val[m:m + n] = x

    h = _label_hash(salt)
    tables = []
    idx = 0
    o = m + n
    for op, a, b in parsed.gates:
        ka, kb = kinds[a], kinds[b]
        if op == OP_XOR:
            if ka and kb:
                lab[o] = lab[a] ^ lab[b]
            elif ka:
                lab[o] = lab[a] ^ (-val[b] & delta)
            elif kb:
                lab[o] = lab[b] ^ (-val[a] & delta)
            else:
                val[o] = val[a] ^ val[b]
        elif op == OP_AND:
            if ka and kb:
                a0, b0 = lab[a], lab[b]
                pa, pb = a0 & 1, b0 & 1
                ha0, ha1 = h(a0, idx), h(a0 ^ delta, idx)
                hb0, hb1 = h(b0, idx + 1), h(b0 ^ delta, idx + 1)
                t_g = ha0 ^ ha1 ^ (-pb & delta)
                w_g = ha0 ^ (-pa & t_g)
                t_e = hb0 ^ hb1 ^ a0
                w_e = hb0 ^ (-pb & (t_e ^ a0))
                tables.append(t_g)
                tables.append(t_e)
                lab[o] = w_g ^ w_e
                idx += 2
            elif ka or kb:
                l, k = (a, b) if ka else (b, a)
                l0 = lab[l]
                h0, h1 = h(l0, idx), h(l0 ^ delta, idx)
                t = h0 ^ h1 ^ (-val[k] & delta)
                tables.append(t)
                lab[o] = h0 ^ (-(l0 & 1) & t)
                idx += 1
            else:
                val[o] = val[a] & val[b]
        else:  # INV
            if ka:
                lab[o] = lab[a] ^ delta
            else:
                val[o] = val[a] ^ 1
        o += 1

    decode = [lab[v] & 1 for v in parsed.outputs]
    return {
        "delta": delta_bytes,
        "key0": b"".join(key0),
        "data_active": _join(data_active),
        "tables": _join(tables),
        "decode": pack_bits(decode, parsed.n_out),
    }


def evaluate(circuit: bytes, mode: str, salt: bytes, tables: bytes, decode: bytes, key_active: bytes,
             data_active: bytes) -> bytes:
    """Evaluate a garbling of :func:`garble` with the active labels of the key
    wires (and, in mode ``"labels"``, of the data wires).

    Args:
        circuit, mode, salt: As given to :func:`garble`.
        tables: The 16-byte table entries.
        decode: The ``n_out`` decoding bits, packed.
        key_active: One 16-byte active label per key wire (``m``).
        data_active: One 16-byte active label per data wire (``n``) in mode
            ``"labels"``; ``b""`` in mode ``"hardwired"``.

    Returns:
        The ``n_out`` output bits, packed (:func:`pack_bits`).

    Raises:
        TypeError: For inputs that are not bytes.
        ValueError: For a malformed circuit, an unknown mode, a salt of
            another length than 56 bytes, an output wire that is not a
            labelled wire, another number of table entries than the circuit
            has, another number of key or data labels than ``m`` / ``n`` (0
            data labels in mode ``"hardwired"``), or decoding bits that are
            not ``n_out`` packed bits.
    """
    parsed = parse_circuit(circuit)
    kinds = wire_kinds(parsed, mode)
    salt = _check_salt(salt)
    _require_label_outputs(parsed, kinds, mode)
    m, n = parsed.m, parsed.n
    entries = _labels(tables, table_entries(parsed, mode), "tables")
    d = unpack_bits(decode, parsed.n_out)
    lab = [0] * parsed.n_wires  # active labels of the L wires
    lab[:m] = _labels(key_active, m, "key_active")
    data = _labels(data_active, n if mode == MODE_LABELS else 0, "data_active")
    lab[m:m + len(data)] = data

    h = _label_hash(salt)
    ti = 0
    idx = 0
    o = m + n
    for op, a, b in parsed.gates:
        ka, kb = kinds[a], kinds[b]
        if op == OP_XOR:
            if ka and kb:
                lab[o] = lab[a] ^ lab[b]
            elif ka:
                lab[o] = lab[a]
            elif kb:
                lab[o] = lab[b]
        elif op == OP_AND:
            if ka and kb:
                wa, wb = lab[a], lab[b]
                t_g, t_e = entries[ti], entries[ti + 1]
                lab[o] = h(wa, idx) ^ (-(wa & 1) & t_g) ^ h(wb, idx + 1) ^ (-(wb & 1) & (t_e ^ wa))
                ti += 2
                idx += 2
            elif ka or kb:
                wl = lab[a] if ka else lab[b]
                lab[o] = h(wl, idx) ^ (-(wl & 1) & entries[ti])
                ti += 1
                idx += 1
        elif ka:  # INV(L); INV(K) and the other K gates are skipped
            lab[o] = lab[a]
        o += 1

    return pack_bits([(lab[v] & 1) ^ d[k] for k, v in enumerate(parsed.outputs)], parsed.n_out)
