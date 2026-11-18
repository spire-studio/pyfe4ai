"""Boolean circuits ``C(w, x)`` of the bounded-key schemes: gate list, canonical
encoding, digest, guard bit, wire kinds and plain evaluation.

A circuit has ``m`` key wires (the bits of the secret parameter ``w`` that the
key holder's key fixes), ``n`` data wires (the bits of the encryptor's ``x``)
and a list of XOR, AND and INV gates in topological order. Wire numbering:

* ``0 .. m-1``: the key wires;
* ``m .. m+n-1``: the data wires;
* ``m+n+i``: the output of gate ``i``.

There are no constant wires: :class:`~pyfe4ai.circuits.builder.Builder` folds
constants away and refuses a constant output.

**Canonical encoding** ``CIRC v1`` (:meth:`Circuit.encode`; integers little
endian)::

    "PFE6CIRC"          8-byte magic
    u8  version = 1
    u32 m               key bits, >= 1
    u32 n               data bits, >= 0
    u32 n_gates
    u32 n_out           >= 1
    n_gates x 9 bytes:  u8 op (1 = XOR, 2 = AND, 3 = INV) || u32 a || u32 b
                        (b = 0 for INV)
    n_out x u32         output wires

:meth:`Circuit.decode` refuses (``ValueError``) a wrong magic or version, a
length that does not match the counts, an unknown op, an INV with ``b != 0``,
an XOR or AND whose two inputs are the same wire (the builder folds those),
a gate input not below the gate's own output wire, an output wire out of
range, ``m = 0``, ``n_out = 0`` and ``m + n + n_gates >= 2^26``
(:data:`MAX_CIRCUIT_WIRES`, checked before anything is sized by the counts;
the constructor, :meth:`Circuit.guard` and the builder refuse the same).
Every field
is fixed, so a circuit has exactly one encoding: ``decode(c.encode()) == c``
and ``decode(data).encode() == data``. The digest (:meth:`Circuit.digest`)
is ``SHAKE-256("PYFE4AI-V06-BCFE-CIRCUIT" || encoding)[0:32]``.

**Guard bit** (:meth:`Circuit.guard`): the extra key position of Sahai and
Seyalioglu, "Worry-free encryption: functional encryption with public keys"
(CCS 2010, SS10, Sec. 3.2): ``C'(w || g, x) = C(w, x) AND NOT g`` on every
output bit, with ``g`` the new key wire ``m``. With ``g = 0`` the circuit
computes ``C``; with ``g = 1`` every output is 0. The schemes garble
``guard(C)`` and bind the digest of ``C`` itself.

**Wire kinds and costs** (:meth:`Circuit.kinds`, :meth:`Circuit.stats`). The
garbling is the half-gates scheme of Zahur, Rosulek and Evans, "Two Halves
Make a Whole" (EUROCRYPT 2015, ZRE15). An ``L`` wire carries labels, a ``K``
wire a value the garbler knows. Key wires are ``L``; data wires are ``K`` in
mode ``"hardwired"`` (the default: the garbler folds ``x`` into the tables)
and ``L`` in mode ``"labels"``. A gate's output is ``L`` if any input is
``L``, else ``K``; INV keeps its input's kind. Costs per gate: XOR and INV
with an ``L`` input are free; AND(L, L) takes two table entries (16 bytes
each); AND(L, K) or AND(K, L) one entry (the generator half-gate); a gate
on ``K`` wires only is computed by the garbler in the clear and leaves
nothing in the ciphertext.

The kinds, the tables and their number depend on the circuit and the mode
only, never on the values of ``w`` or ``x`` (requirement I12 of the 0.6
security argument): :meth:`Circuit.kinds` takes no input values, and
nothing here folds or skips a gate because of a value.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Sequence

from pyfe4ai.core.exceptions import FEValidationError

#: Magic and version of the canonical encoding ``CIRC v1``.
CIRCUIT_MAGIC = b"PFE6CIRC"
CIRCUIT_VERSION = 1
#: Domain of the circuit digest.
CIRCUIT_DIGEST_DOMAIN = b"PYFE4AI-V06-BCFE-CIRCUIT"
#: Gate operations.
XOR, AND, INV = 1, 2, 3
#: Names of the gate operations (for ``repr`` and the constructor).
OP_NAMES = {XOR: "XOR", AND: "AND", INV: "INV"}
#: Wire kinds: labelled, known to the garbler.
WIRE_L, WIRE_K = "L", "K"
#: The two modes of the data wires (the record field ``data_wires``).
MODES = ("hardwired", "labels")
#: ``m + n + n_gates`` stays below this bound: 2^26 wires (64 Mi wires,
#: about 1 GiB of native labels; the largest circuit the schemes target, an
#: MLP 784-128-10, is far below), so that a short header cannot make a
#: decoder's caller allocate without bound. Design choice; the alternative was
#: the earlier 2^31 (wire numbers in 31 bits) with the per-wire state sized by
#: the labelled wires only.
MAX_CIRCUIT_WIRES = 1 << 26
#: The same bound (the name the builder imports).
MAX_WIRES = MAX_CIRCUIT_WIRES

_HEADER = struct.Struct("<8sBIIII")
_GATE = struct.Struct("<BII")
_U32 = struct.Struct("<I")
_OP_BY_NAME = {"XOR": XOR, "AND": AND, "INV": INV}


def _check_mode(mode) -> str:
    if mode not in MODES:
        raise FEValidationError("mode must be one of {}".format(", ".join(repr(m) for m in MODES)))
    return mode


def _check_u32(value, what: str) -> int:
    if type(value) is not int or not 0 <= value < (1 << 32):
        raise ValueError("{} must be an integer in [0, 2^32)".format(what))
    return value


def _validate(m: int, n: int, gates: Sequence[tuple], outputs: Sequence[int]) -> None:
    """The rules of ``CIRC v1`` (``ValueError``), checked by the constructor and by decode."""
    if m == 0:
        raise ValueError("a circuit needs at least one key bit (m >= 1)")
    if len(outputs) == 0:
        raise ValueError("a circuit needs at least one output (n_out >= 1)")
    if m + n + len(gates) >= MAX_WIRES:
        raise ValueError("m + n + n_gates must be below 2^26")
    base = m + n
    for i, (op, a, b) in enumerate(gates):
        out = base + i
        if op == INV:
            if b != 0:
                raise ValueError("gate {}: INV must have b = 0".format(i))
            if a >= out:
                raise ValueError("gate {}: input wire {} is not below the gate's output wire {}".format(i, a, out))
        elif op == XOR or op == AND:
            if a >= out or b >= out:
                raise ValueError("gate {}: an input wire is not below the gate's output wire {}".format(i, out))
            if a == b:
                raise ValueError("gate {}: {} of a wire with itself (the builder folds it)".format(i, OP_NAMES[op]))
        else:
            raise ValueError("gate {}: unknown op {}".format(i, op))
    n_wires = base + len(gates)
    for k, o in enumerate(outputs):
        if o >= n_wires:
            raise ValueError("output {}: wire {} is out of range (the circuit has {} wires)".format(k, o, n_wires))


class Circuit:
    """An immutable boolean circuit over ``m`` key bits and ``n`` data bits.

    Args:
        m: Number of key wires (``>= 1``).
        n: Number of data wires (``>= 0``).
        gates: ``(op, a, b)`` per gate in order; ``op`` is :data:`XOR`,
            :data:`AND`, :data:`INV` (or their names), ``a`` and ``b`` input
            wires below the gate's output wire, ``b = 0`` for INV.
        outputs: The output wires (at least one).

    Raises:
        ValueError: The circuit breaks a rule of ``CIRC v1`` (see the module
            documentation).

    Circuits are public: ``repr`` shows the sizes and the digest. Equality and
    hashing follow the canonical encoding; pickling goes through it.
    """

    __slots__ = ("_m", "_n", "_gates", "_outputs", "_encoding")

    def __init__(self, m: int, n: int, gates: Iterable[Sequence], outputs: Iterable[int]):
        m = _check_u32(m, "m")
        n = _check_u32(n, "n")
        norm = []
        for i, gate in enumerate(gates):
            gate = tuple(gate)
            if len(gate) != 3:
                raise ValueError("gate {}: a gate is (op, a, b)".format(i))
            op, a, b = gate
            if isinstance(op, str):
                if op not in _OP_BY_NAME:
                    raise ValueError("gate {}: unknown op {!r}".format(i, op))
                op = _OP_BY_NAME[op]
            op = _check_u32(op, "gate {} op".format(i))
            norm.append((op, _check_u32(a, "gate {} input a".format(i)), _check_u32(b, "gate {} input b".format(i))))
        outs = tuple(_check_u32(o, "output wire") for o in outputs)
        gates_t = tuple(norm)
        _validate(m, n, gates_t, outs)
        self._init(m, n, gates_t, outs, None)

    def _init(self, m, n, gates, outputs, encoding):
        object.__setattr__(self, "_m", m)
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_gates", gates)
        object.__setattr__(self, "_outputs", outputs)
        object.__setattr__(self, "_encoding", encoding)

    @classmethod
    def _trusted(cls, m: int, n: int, gates: tuple, outputs: tuple) -> "Circuit":
        """A circuit from parts already in canonical form and validated by the caller."""
        self = object.__new__(cls)
        self._init(m, n, gates, outputs, None)
        return self

    def __setattr__(self, name, value):
        raise AttributeError("Circuit is immutable")

    def __delattr__(self, name):
        raise AttributeError("Circuit is immutable")

    def __reduce__(self):
        return (Circuit.decode, (self.encode(),))

    # -- sizes

    @property
    def m(self) -> int:
        """Number of key wires."""
        return self._m

    @property
    def n(self) -> int:
        """Number of data wires."""
        return self._n

    @property
    def n_gates(self) -> int:
        """Number of gates."""
        return len(self._gates)

    @property
    def n_out(self) -> int:
        """Number of outputs."""
        return len(self._outputs)

    @property
    def n_wires(self) -> int:
        """Number of wires, ``m + n + n_gates``."""
        return self._m + self._n + len(self._gates)

    @property
    def gates(self) -> tuple:
        """The gates, ``(op, a, b)`` each (``b = 0`` for INV)."""
        return self._gates

    @property
    def outputs(self) -> tuple:
        """The output wires."""
        return self._outputs

    # -- encoding and digest

    def encode(self) -> bytes:
        """The canonical encoding ``CIRC v1``."""
        enc = self._encoding
        if enc is None:
            n_gates, n_out = len(self._gates), len(self._outputs)
            buf = bytearray(_HEADER.size + _GATE.size * n_gates + _U32.size * n_out)
            _HEADER.pack_into(buf, 0, CIRCUIT_MAGIC, CIRCUIT_VERSION, self._m, self._n, n_gates, n_out)
            put_gate, pos = _GATE.pack_into, _HEADER.size
            for op, a, b in self._gates:
                put_gate(buf, pos, op, a, b)
                pos += _GATE.size
            struct.pack_into("<{}I".format(n_out), buf, pos, *self._outputs)
            enc = bytes(buf)
            object.__setattr__(self, "_encoding", enc)
        return enc

    @classmethod
    def decode(cls, data) -> "Circuit":
        """Read a ``CIRC v1`` encoding; ``ValueError`` on anything else (see the module documentation)."""
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise ValueError("a circuit encoding is bytes")
        if len(data) < _HEADER.size:
            raise ValueError("circuit encoding too short for its header")
        magic, version, m, n, n_gates, n_out = _HEADER.unpack_from(data, 0)
        if magic != CIRCUIT_MAGIC:
            raise ValueError("not a circuit encoding (wrong magic)")
        if version != CIRCUIT_VERSION:
            raise ValueError("unknown circuit encoding version {}".format(version))
        # the counts are checked against the length before anything is allocated
        expected = _HEADER.size + _GATE.size * n_gates + _U32.size * n_out
        if len(data) != expected:
            raise ValueError("circuit encoding of {} bytes, the counts need {}".format(len(data), expected))
        if m == 0:
            raise ValueError("a circuit needs at least one key bit (m >= 1)")
        if n_out == 0:
            raise ValueError("a circuit needs at least one output (n_out >= 1)")
        if m + n + n_gates >= MAX_WIRES:
            raise ValueError("m + n + n_gates must be below 2^26")
        gates = tuple(_GATE.iter_unpack(data[_HEADER.size:_HEADER.size + _GATE.size * n_gates]))
        outputs = struct.unpack_from("<{}I".format(n_out), data, _HEADER.size + _GATE.size * n_gates)
        _validate(m, n, gates, outputs)
        self = object.__new__(cls)
        self._init(m, n, gates, tuple(outputs), data)
        return self

    def digest(self) -> bytes:
        """``SHAKE-256("PYFE4AI-V06-BCFE-CIRCUIT" || encode())``, 32 bytes."""
        return hashlib.shake_256(CIRCUIT_DIGEST_DOMAIN + self.encode()).digest(32)

    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.encode() == other.encode()

    def __hash__(self):
        return hash(self.digest())

    def __repr__(self):
        return "Circuit(m={}, n={}, n_gates={}, n_out={}, digest={}...)".format(
            self._m, self._n, len(self._gates), len(self._outputs), self.digest()[:8].hex())

    # -- the guard bit

    def guard(self) -> "Circuit":
        """``C'(w || g, x) = C(w, x) AND NOT g`` (SS10 Sec. 3.2), as a new ``CIRC v1`` circuit.

        ``m' = m + 1`` and ``g`` is key wire ``m``; an old wire ``v`` becomes
        ``v`` if ``v < m`` and ``v + 1`` otherwise; the old gates follow in
        their order with their inputs renumbered, then ``INV(g)`` (wire
        ``ng``), then ``AND(new(o), ng)`` for every old output ``o`` in order.
        The outputs are those AND gates.
        """
        m = self._m

        def new(v: int) -> int:
            return v if v < m else v + 1

        gates = [(op, new(a), 0 if op == INV else new(b)) for op, a, b in self._gates]
        g = m
        ng = m + 1 + self._n + len(gates)
        gates.append((INV, g, 0))
        outputs = []
        for o in self._outputs:
            outputs.append(m + 1 + self._n + len(gates))
            gates.append((AND, new(o), ng))
        if m + 1 + self._n + len(gates) >= MAX_WIRES:
            raise ValueError("m + n + n_gates must be below 2^26")
        return Circuit._trusted(m + 1, self._n, tuple(gates), tuple(outputs))

    # -- kinds and costs

    def kinds(self, mode: str = "hardwired") -> tuple:
        """The kind of every wire, :data:`WIRE_L` or :data:`WIRE_K`, for *mode*.

        ``"hardwired"``: key wires ``L``, data wires ``K``; ``"labels"``:
        both ``L``. A gate's output is ``L`` if an input is ``L`` (for INV:
        its one input), else ``K``. Depends on the circuit and the mode only.
        """
        _check_mode(mode)
        m, n = self._m, self._n
        is_l = [True] * m + [mode == "labels"] * n
        append = is_l.append
        for op, a, b in self._gates:
            append(is_l[a] if op == INV else (is_l[a] or is_l[b]))
        return tuple(WIRE_L if v else WIRE_K for v in is_l)

    def stats(self, mode: str = "hardwired") -> dict:
        """Gate counts by garbling cost for *mode* (ZRE15 half gates).

        Keys: ``and_ll`` (AND of two ``L`` wires, two table entries each),
        ``and_lk`` (AND of an ``L`` and a ``K`` wire, one entry), ``free``
        (XOR or INV with an ``L`` input), ``garbler_only`` (gates on ``K``
        wires only) and ``table_entries`` = ``2 * and_ll + and_lk``.
        """
        _check_mode(mode)
        m, n = self._m, self._n
        is_l = [True] * m + [mode == "labels"] * n
        append = is_l.append
        and_ll = and_lk = free = garbler_only = 0
        for op, a, b in self._gates:
            la = is_l[a]
            if op == INV:
                if la:
                    free += 1
                else:
                    garbler_only += 1
                append(la)
                continue
            lb = is_l[b]
            if op == AND:
                if la and lb:
                    and_ll += 1
                elif la or lb:
                    and_lk += 1
                else:
                    garbler_only += 1
            elif la or lb:
                free += 1
            else:
                garbler_only += 1
            append(la or lb)
        return {"and_ll": and_ll, "and_lk": and_lk, "free": free, "garbler_only": garbler_only,
                "table_entries": 2 * and_ll + and_lk}

    # -- plain evaluation

    def evaluate_plain(self, w_bits: Sequence[int], x_bits: Sequence[int]) -> list:
        """``C(w, x)`` in the clear: the output bits (0 or 1) for the key bits
        *w_bits* (``m`` of them) and the data bits *x_bits* (``n``). The
        reference of the tests; not constant time."""
        return [row[0] for row in self.evaluate_plain_batch([w_bits], [x_bits])]

    def evaluate_plain_batch(self, w_rows: Sequence[Sequence[int]], x_rows: Sequence[Sequence[int]]) -> list:
        """:meth:`evaluate_plain` on many inputs at once (bit-sliced: one Python
        integer per wire holds the wire's value in every input). Returns, per
        output, the list of its bits over the inputs: ``result[k][j]`` is
        output ``k`` on input ``j``."""
        if len(w_rows) != len(x_rows):
            raise FEValidationError("as many key inputs as data inputs are needed")
        lanes = len(w_rows)
        if lanes == 0:
            return [[] for _ in self._outputs]
        values = self._slice(w_rows, self._m, "key") + self._slice(x_rows, self._n, "data")
        mask = (1 << lanes) - 1
        append = values.append
        for op, a, b in self._gates:
            if op == XOR:
                append(values[a] ^ values[b])
            elif op == AND:
                append(values[a] & values[b])
            else:
                append(values[a] ^ mask)
        return [[(values[o] >> j) & 1 for j in range(lanes)] for o in self._outputs]

    @staticmethod
    def _slice(rows, width: int, what: str) -> list:
        cols = [0] * width
        for j, row in enumerate(rows):
            row = list(row)
            if len(row) != width:
                raise FEValidationError("{} input {} has {} bits, the circuit takes {}".format(
                    what, j, len(row), width))
            for i, bit in enumerate(row):
                if type(bit) is bool:
                    bit = int(bit)
                if bit != 0 and bit != 1:
                    raise FEValidationError("{} input {}: bit {} is not 0 or 1".format(what, j, i))
                if bit:
                    cols[i] |= 1 << j
        return cols
