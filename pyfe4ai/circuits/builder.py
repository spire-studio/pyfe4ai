"""Building circuits: bit operations with constant folding, and word arithmetic.

:class:`Builder` makes a :class:`~pyfe4ai.circuits.circuit.Circuit` gate by
gate. A *handle* is an ``int``: a wire of the circuit under construction
(``>= 0``: :meth:`Builder.key`, :meth:`Builder.data` or the result of an
operation) or one of the two constants :attr:`Builder.ZERO` and
:attr:`Builder.ONE` (negative). Constants exist only in the builder: every
operation folds them away (``x XOR 0 = x``, ``x XOR 1 = NOT x``, ``x AND 0 =
0``, ``x AND 1 = x``, ``NOT 0 = 1``), and so do ``x XOR x = 0``,
``x AND x = x``, ``x XOR NOT x = 1``, ``x AND NOT x = 0`` and
``NOT NOT x = x``. A gate already made with the same inputs is reused
(structural hashing), and :meth:`Builder.build` drops the gates no output
depends on. All of this looks at the structure only, never at a value of
``w`` or ``x``, so the wire kinds of the result depend on the circuit alone,
and so do the layout and the number of a ciphertext's gate tables. A
constant output is refused
(:class:`~pyfe4ai.core.exceptions.FEValidationError`): the encoding has no
constant wires.

**Words** are lists of handles, least significant bit first, read as two's
complement (signed) unless a method says otherwise; arithmetic is modulo
``2^width``. Costs in the half-gates garbling of Zahur, Rosulek and Evans
(EUROCRYPT 2015, ZRE15): XOR and INV are free, an AND of two labelled wires
takes two table entries, an AND of a labelled and a garbler-known wire one.

* :meth:`Builder.add` -- ripple carry, one AND per bit but the top one
  (``s = a ^ b ^ c``, ``c' = ((a ^ c) & (b ^ c)) ^ c``);
* :meth:`Builder.mul_shift_add` -- one row of ANDs per bit of the second
  operand, the sign bit's row subtracted;
* :meth:`Builder.mul_booth` -- radix-4 Booth recoding of the *second*
  operand: half as many rows, each selecting ``0, +-a, +-2a``. When the
  second operand is the data ``x`` and the data wires are garbler-known
  (mode ``"hardwired"``), the digit flags cost nothing and every selection
  is an AND(L, K) generator half-gate (one table entry), which is why the
  library multiplies ``w * x`` with Booth on ``x``. Both multiplications
  give the same value;
* comparisons (:meth:`Builder.lt`, :meth:`Builder.ge`, :meth:`Builder.ge_zero`),
  :meth:`Builder.relu`, :meth:`Builder.mux_word`, :meth:`Builder.argmax`,
  :meth:`Builder.sum_words` (a balanced adder tree) and :meth:`Builder.dot`
  (a sum of exact products; with Booth, the first digit's carries of all
  products are counted once instead of added into each product).
"""

from __future__ import annotations

from typing import Sequence

from pyfe4ai.circuits.circuit import AND, INV, MAX_WIRES, XOR, Circuit
from pyfe4ai.core.exceptions import FEValidationError

#: The two multiplication methods.
MUL_METHODS = ("booth", "shift_add")


def _check_count(value, what: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise FEValidationError("{} must be an integer >= {}".format(what, minimum))
    return value


class Builder:
    """A circuit under construction over *m* key bits and *n* data bits.

    Args:
        m: Number of key wires (``>= 1``).
        n: Number of data wires (``>= 0``).
    """

    #: The constant 0 (a builder-only handle).
    ZERO = -1
    #: The constant 1 (a builder-only handle).
    ONE = -2

    def __init__(self, m: int, n: int):
        self._m = _check_count(m, "m", 1)
        self._n = _check_count(n, "n", 0)
        if m + n >= MAX_WIRES:
            raise FEValidationError("m + n must be below 2^26 (the wire bound of CIRC v1)")
        self._base = m + n
        self._gates = []
        self._outputs = []
        self._known = {}  # (op, a, b) -> wire: structural hashing
        self._negation = {}  # wire -> its INV (both ways)

    # -- handles

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
        """Number of gates made so far (before :meth:`build` drops unused ones)."""
        return len(self._gates)

    @property
    def zero(self) -> int:
        """The constant 0."""
        return self.ZERO

    @property
    def one(self) -> int:
        """The constant 1."""
        return self.ONE

    def key(self, i: int) -> int:
        """Key wire *i* (bit ``i`` of ``w``)."""
        if type(i) is not int or not 0 <= i < self._m:
            raise FEValidationError("key bit index out of range [0, {})".format(self._m))
        return i

    def data(self, i: int) -> int:
        """Data wire *i* (bit ``i`` of ``x``)."""
        if type(i) is not int or not 0 <= i < self._n:
            raise FEValidationError("data bit index out of range [0, {})".format(self._n))
        return self._m + i

    def const(self, bit: int) -> int:
        """The constant *bit* (0 or 1)."""
        if bit is True or bit == 1:
            return self.ONE
        if bit is False or bit == 0:
            return self.ZERO
        raise FEValidationError("a constant is 0 or 1")

    @staticmethod
    def is_const(h: int) -> bool:
        """Whether *h* is one of the two constants."""
        return h < 0

    @staticmethod
    def const_value(h: int):
        """0 or 1 for a constant handle, ``None`` for a wire."""
        return -1 - h if h < 0 else None

    def key_word(self, start: int, width: int) -> list:
        """Key wires ``start .. start + width - 1`` as a word."""
        return [self.key(start + i) for i in range(width)]

    def data_word(self, start: int, width: int) -> list:
        """Data wires ``start .. start + width - 1`` as a word."""
        return [self.data(start + i) for i in range(width)]

    def const_word(self, value: int, width: int, signed: bool = True) -> list:
        """The constant *value* as a *width*-bit word (two's complement if *signed*)."""
        _check_count(width, "width", 1)
        if type(value) is not int:
            raise FEValidationError("a constant word takes an int")
        lo, hi = (-(1 << (width - 1)), 1 << (width - 1)) if signed else (0, 1 << width)
        if not lo <= value < hi:
            raise FEValidationError("constant out of the range of a {}-bit word".format(width))
        return [self.ONE if (value >> i) & 1 else self.ZERO for i in range(width)]

    def _check(self, h) -> int:
        if type(h) is not int or not -2 <= h < self._base + len(self._gates):
            raise FEValidationError("not a handle of this builder")
        return h

    def _gate(self, op: int, a: int, b: int) -> int:
        key = (op, a, b)
        o = self._known.get(key)
        if o is None:
            o = self._base + len(self._gates)
            self._gates.append(key)
            self._known[key] = o
        return o

    # -- bits

    def inv(self, a: int) -> int:
        """``NOT a``."""
        self._check(a)
        if a < 0:
            return self.ZERO if a == self.ONE else self.ONE
        o = self._negation.get(a)
        if o is None:
            o = self._gate(INV, a, 0)
            self._negation[a] = o
            self._negation[o] = a
        return o

    def xor(self, a: int, b: int) -> int:
        """``a XOR b``."""
        self._check(a)
        self._check(b)
        if a < 0:
            return b if a == self.ZERO else self.inv(b)
        if b < 0:
            return a if b == self.ZERO else self.inv(a)
        if a == b:
            return self.ZERO
        if self._negation.get(a) == b:
            return self.ONE
        return self._gate(XOR, a, b) if a < b else self._gate(XOR, b, a)

    def and_(self, a: int, b: int) -> int:
        """``a AND b``."""
        self._check(a)
        self._check(b)
        if a < 0:
            return b if a == self.ONE else self.ZERO
        if b < 0:
            return a if b == self.ONE else self.ZERO
        if a == b:
            return a
        if self._negation.get(a) == b:
            return self.ZERO
        return self._gate(AND, a, b) if a < b else self._gate(AND, b, a)

    def or_(self, a: int, b: int) -> int:
        """``a OR b`` (``NOT(NOT a AND NOT b)``: one AND)."""
        return self.inv(self.and_(self.inv(a), self.inv(b)))

    def mux(self, s: int, a: int, b: int) -> int:
        """``a`` if ``s`` else ``b`` (``b XOR (s AND (a XOR b))``: one AND)."""
        self._check(s)
        self._check(a)
        self._check(b)
        if s < 0:
            return a if s == self.ONE else b
        if a == b:
            return a
        return self.xor(b, self.and_(s, self.xor(a, b)))

    # -- outputs

    def output(self, h: int) -> None:
        """Append *h* to the outputs; a constant is refused."""
        self._check(h)
        if h < 0:
            raise FEValidationError("a circuit output must not be constant (the encoding has no constant wires)")
        self._outputs.append(h)

    def output_word(self, word: Sequence[int]) -> None:
        """:meth:`output` for every bit of *word*, least significant first."""
        for h in word:
            self.output(h)

    def build(self, prune: bool = True) -> Circuit:
        """The circuit with the outputs so far. With *prune* (the default)
        the gates no output depends on are dropped and the rest renumbered,
        in their order. ``FEValidationError`` if it has no output or
        ``m + n + n_gates >= 2^26`` (the wire bound of ``CIRC v1``)."""
        if not self._outputs:
            raise FEValidationError("a circuit needs at least one output")
        base, gates = self._base, self._gates
        if not prune:
            self._check_wires(len(gates))
            return Circuit._trusted(self._m, self._n, tuple(gates), tuple(self._outputs))
        live = bytearray(base + len(gates))
        for o in self._outputs:
            live[o] = 1
        for i in range(len(gates) - 1, -1, -1):
            if live[base + i]:
                op, a, b = gates[i]
                live[a] = 1
                if op != INV:
                    live[b] = 1
        new = list(range(base)) + [0] * len(gates)
        kept = []
        for i, (op, a, b) in enumerate(gates):
            if live[base + i]:
                new[base + i] = base + len(kept)
                kept.append((op, new[a], 0 if op == INV else new[b]))
        self._check_wires(len(kept))
        return Circuit._trusted(self._m, self._n, tuple(kept), tuple(new[o] for o in self._outputs))

    def _check_wires(self, n_gates: int) -> None:
        """``m + n + n_gates`` below :data:`~pyfe4ai.circuits.circuit.MAX_CIRCUIT_WIRES`
        (2^26), as every decoder of ``CIRC v1`` requires, else ``FEValidationError``."""
        if self._base + n_gates >= MAX_WIRES:
            raise FEValidationError("the circuit has {} wires; CIRC v1 allows fewer than 2^26".format(
                self._base + n_gates))

    # -- words

    def _word(self, word, what: str = "word") -> list:
        word = list(word)
        if not word:
            raise FEValidationError("a {} needs at least one bit".format(what))
        for h in word:
            self._check(h)
        return word

    def sign_extend(self, a: Sequence[int], width: int) -> list:
        """*a* repeated in its top bit up to *width* bits (``width >= len(a)``)."""
        a = self._word(a)
        _check_count(width, "width", len(a))
        return a + [a[-1]] * (width - len(a))

    def zero_extend(self, a: Sequence[int], width: int) -> list:
        """*a* padded with zeros up to *width* bits (``width >= len(a)``)."""
        a = self._word(a)
        _check_count(width, "width", len(a))
        return a + [self.ZERO] * (width - len(a))

    def resize(self, a: Sequence[int], width: int, signed: bool = True) -> list:
        """*a* truncated (modulo ``2^width``) or extended (sign or zero) to *width* bits."""
        a = self._word(a)
        _check_count(width, "width", 1)
        if width <= len(a):
            return a[:width]
        return self.sign_extend(a, width) if signed else self.zero_extend(a, width)

    def shift_right(self, a: Sequence[int], k: int) -> list:
        """``floor(a / 2^k)`` for signed *a*: the top ``len(a) - k`` bits (``0 <= k < len(a)``)."""
        a = self._word(a)
        if type(k) is not int or not 0 <= k < len(a):
            raise FEValidationError("shift out of range [0, {})".format(len(a)))
        return a[k:]

    def shift_left(self, a: Sequence[int], k: int) -> list:
        """``a * 2^k`` modulo ``2^len(a)``."""
        a = self._word(a)
        _check_count(k, "shift", 0)
        return ([self.ZERO] * k + a)[:len(a)]

    def invert(self, a: Sequence[int]) -> list:
        """Bitwise NOT (``-a - 1``)."""
        return [self.inv(h) for h in self._word(a)]

    def _add_from(self, acc: list, t: list, start: int, cin: int) -> list:
        """``acc + t + cin * 2^start`` modulo ``2^len(acc)``, where the bits of *t*
        below *start* are zero: one AND per bit from *start*, but the top one."""
        out = list(acc)
        top = len(out) - 1
        c = cin
        for i in range(start, top + 1):
            b = t[i]
            axc = self.xor(out[i], c)
            out[i] = self.xor(axc, b)
            if i < top:
                c = self.xor(self.and_(axc, self.xor(b, c)), c)
        return out

    def add(self, a: Sequence[int], b: Sequence[int], cin: int = ZERO) -> list:
        """``a + b + cin`` modulo ``2^len(a)`` (*a*, *b* of one width; *cin* a
        bit handle, by default the constant 0)."""
        a, b = self._word(a), self._word(b)
        if len(a) != len(b):
            raise FEValidationError("add takes two words of one width")
        self._check(cin)
        return self._add_from(a, b, 0, cin)

    def sub(self, a: Sequence[int], b: Sequence[int]) -> list:
        """``a - b`` modulo ``2^len(a)``."""
        return self.add(a, self.invert(b), self.ONE)

    def neg(self, a: Sequence[int]) -> list:
        """``-a`` modulo ``2^len(a)``."""
        a = self._word(a)
        return self.add([self.ZERO] * len(a), self.invert(a), self.ONE)

    def mac_shift_add(self, acc: Sequence[int], a: Sequence[int], b: Sequence[int]) -> list:
        """``acc + a * b`` modulo ``2^len(acc)`` (signed *a*, *b*) by shift and
        add over the bits of *b*; the row of the sign bit of *b* is subtracted."""
        acc, a, b = self._word(acc, "accumulator"), self._word(a), self._word(b)
        width = len(acc)
        aext = self.resize(a, width)
        for j, bj in enumerate(b):
            if j >= width:
                break
            t = [self.ZERO] * width
            for k in range(j, width):
                t[k] = self.and_(bj, aext[k - j])
            if j + 1 < len(b):
                acc = self._add_from(acc, t, j, self.ZERO)
            else:  # the sign bit of b weighs -2^(len(b) - 1)
                for k in range(j, width):
                    t[k] = self.inv(t[k])
                acc = self._add_from(acc, t, j, self.ONE)
        return acc

    def mac_booth(self, acc: Sequence[int], a: Sequence[int], b: Sequence[int]) -> list:
        """``acc + a * b`` modulo ``2^len(acc)`` (signed *a*, *b*) with *b*
        recoded in radix-4 Booth digits ``d = -2 hi + mid + lo`` in
        ``{-2, ..., 2}``. Per digit: flags ``one = mid ^ lo``, ``two = (hi &
        !mid & !lo) | (!hi & mid & lo)`` and ``neg = hi`` on *b*'s wires, the
        row ``((one & a) ^ (two & 2a)) ^ neg`` and ``neg`` as carry in."""
        return self._mac_booth(acc, a, b, False)[0]

    def _mac_booth(self, acc, a, b, defer_first: bool) -> tuple:
        """``(acc', c)`` with ``acc' + c = acc + a * b`` modulo ``2^len(acc)``.
        With *defer_first* the carry in of the first digit (its ``neg``) is
        returned as ``c`` instead of added: on a zero accumulator that saves
        the whole first addition. Otherwise ``c`` is the constant 0."""
        acc, a, b = self._word(acc, "accumulator"), self._word(a), self._word(b)
        width = len(acc)
        deferred = self.ZERO
        aext = self.resize(a, width)
        bs = b + [b[-1]] if len(b) % 2 else list(b)
        for k in range(len(bs) // 2):
            s = 2 * k
            if s >= width:
                break
            lo = self.ZERO if k == 0 else bs[s - 1]
            mid, hi = bs[s], bs[s + 1]
            one = self.xor(mid, lo)
            nmid, nlo, nhi = self.inv(mid), self.inv(lo), self.inv(hi)
            # the two terms are disjoint, so XOR is OR
            two = self.xor(self.and_(self.and_(hi, nmid), nlo), self.and_(self.and_(nhi, mid), lo))
            neg = hi
            t = [self.ZERO] * width
            for pos in range(s, width):
                p = self.and_(one, aext[pos - s])
                q = self.and_(two, aext[pos - s - 1]) if pos > s else self.ZERO
                t[pos] = self.xor(self.xor(p, q), neg)
            if k == 0 and defer_first:
                deferred = neg
                acc = self._add_from(acc, t, s, self.ZERO)
            else:
                acc = self._add_from(acc, t, s, neg)
        return acc, deferred

    def mac(self, acc: Sequence[int], a: Sequence[int], b: Sequence[int], method: str = "booth") -> list:
        """``acc + a * b`` modulo ``2^len(acc)``; *method* ``"booth"`` (recodes
        *b*) or ``"shift_add"``."""
        if method == "booth":
            return self.mac_booth(acc, a, b)
        if method == "shift_add":
            return self.mac_shift_add(acc, a, b)
        raise FEValidationError("method must be one of {}".format(", ".join(repr(m) for m in MUL_METHODS)))

    def mul_shift_add(self, a: Sequence[int], b: Sequence[int], width: int = None) -> list:
        """``a * b`` modulo ``2^width`` (default ``len(a) + len(b)``: exact) by shift and add."""
        return self.mul(a, b, width, "shift_add")

    def mul_booth(self, a: Sequence[int], b: Sequence[int], width: int = None) -> list:
        """``a * b`` modulo ``2^width`` (default ``len(a) + len(b)``: exact), Booth on *b*."""
        return self.mul(a, b, width, "booth")

    def mul(self, a: Sequence[int], b: Sequence[int], width: int = None, method: str = "booth") -> list:
        """``a * b`` (signed) modulo ``2^width``; the default width
        ``len(a) + len(b)`` holds every product exactly."""
        a, b = self._word(a), self._word(b)
        if width is None:
            width = len(a) + len(b)
        _check_count(width, "width", 1)
        return self.mac([self.ZERO] * width, a, b, method)

    def dot(self, a_words: Sequence[Sequence[int]], b_words: Sequence[Sequence[int]], width: int,
            init: Sequence[int] = None, method: str = "booth") -> list:
        """``init + sum_i a_words[i] * b_words[i]`` (signed) modulo ``2^width``.

        Each product is computed exactly in ``len(a) + len(b)`` bits and the
        terms are added by :meth:`sum_words`. With ``"booth"`` (on the *b*
        words) the carry in of each product's first digit is not added into
        the product (that would be a whole incrementer on a zero accumulator)
        but collected: the carries are counted by an adder tree of their own
        and the count is one more term. When the *b* words are garbler-known
        (data wires in mode ``"hardwired"``) that count costs no table entry.
        """
        a_words, b_words = list(a_words), list(b_words)
        if len(a_words) != len(b_words):
            raise FEValidationError("dot takes as many a words as b words")
        if method not in MUL_METHODS:
            raise FEValidationError("method must be one of {}".format(", ".join(repr(m) for m in MUL_METHODS)))
        _check_count(width, "width", 1)
        terms = [] if init is None else [self._word(init)]
        carries = []
        for a, b in zip(a_words, b_words):
            a, b = self._word(a), self._word(b)
            zero = [self.ZERO] * (len(a) + len(b))
            if method == "booth":
                product, carry = self._mac_booth(zero, a, b, True)
                if carry != self.ZERO:
                    carries.append([carry, self.ZERO])  # 0 or 1 as a signed 2-bit word
            else:
                product = self.mac_shift_add(zero, a, b)
            terms.append(product)
        if carries:
            terms.append(self.sum_words(carries, len(carries).bit_length() + 1))
        if not terms:
            raise FEValidationError("dot needs at least one term")
        return self.sum_words(terms, width)

    def sum_words(self, words: Sequence[Sequence[int]], width: int) -> list:
        """The sum of signed *words* modulo ``2^width``, by a balanced tree of
        adders whose widths grow by one bit per level (capped at *width*). The
        sum is exact when it fits in *width* bits."""
        _check_count(width, "width", 1)
        level = [self._word(w) for w in words]
        if not level:
            raise FEValidationError("sum_words needs at least one word")
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                a, b = level[i], level[i + 1]
                w = min(max(len(a), len(b)) + 1, width)
                nxt.append(self.add(self.resize(a, w), self.resize(b, w)))
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return self.resize(level[0], width)

    # -- comparisons and selections

    def ge_zero(self, a: Sequence[int]) -> int:
        """``[a >= 0]`` for signed *a*."""
        return self.inv(self._word(a)[-1])

    def lt(self, a: Sequence[int], b: Sequence[int]) -> int:
        """``[a < b]`` for signed *a*, *b* (any widths): the sign of ``a - b``
        in one more bit than the wider operand."""
        a, b = self._word(a), self._word(b)
        w = max(len(a), len(b)) + 1
        return self.sub(self.sign_extend(a, w), self.sign_extend(b, w))[-1]

    def ge(self, a: Sequence[int], b: Sequence[int]) -> int:
        """``[a >= b]`` for signed *a*, *b*."""
        return self.inv(self.lt(a, b))

    def relu(self, a: Sequence[int]) -> list:
        """``max(a, 0)`` for signed *a*, in the same width (its top bit is the constant 0)."""
        a = self._word(a)
        keep = self.inv(a[-1])
        return [self.and_(h, keep) for h in a]

    def mux_word(self, s: int, a: Sequence[int], b: Sequence[int]) -> list:
        """*a* if *s* else *b* (words of one width)."""
        a, b = self._word(a), self._word(b)
        if len(a) != len(b):
            raise FEValidationError("mux_word takes two words of one width")
        return [self.mux(s, x, y) for x, y in zip(a, b)]

    def argmax(self, words: Sequence[Sequence[int]]) -> tuple:
        """``(index, maximum)`` of signed *words* by a tournament: *index* is
        the position of the maximum as an unsigned word of
        ``max(1, (k - 1).bit_length())`` bits (binary, not one-hot); on ties
        the lowest position wins. *maximum* has the width of the widest word."""
        words = [self._word(w) for w in words]
        if not words:
            raise FEValidationError("argmax needs at least one word")
        width = max(len(w) for w in words)
        ibits = max(1, (len(words) - 1).bit_length())
        level = [(self.const_word(i, ibits, signed=False), self.sign_extend(w, width)) for i, w in enumerate(words)]
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                (ia, va), (ib, vb) = level[i], level[i + 1]
                left = self.ge(va, vb)  # the left one has the lower positions: it wins ties
                nxt.append((self.mux_word(left, ia, ib), self.mux_word(left, va, vb)))
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return level[0]
