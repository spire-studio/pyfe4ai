"""Library circuits: linear models, a one-hidden-layer network, an argmax.

Every function returns a :class:`ParamCircuit`: the circuit together with
its key, data and output layouts, a name and the public parameters that
built it. The library circuits do *not* contain the guard bit of SS10
(Sahai and Seyalioglu, CCS 2010, Sec. 3.2); the schemes apply
:meth:`~pyfe4ai.circuits.circuit.Circuit.guard` themselves.

Products ``w * x`` multiply a key word by a data word with radix-4 Booth
recoding of the data word (``mul="booth"``, the default; ``"shift_add"``
gives the same values). In mode ``"hardwired"`` the data wires are known to
the garbler, so the Booth digits cost nothing and every partial-product
selection is a generator half-gate of ZRE15 (Zahur, Rosulek and Evans,
EUROCRYPT 2015): one table entry instead of two. The wire kinds, and so the
table layout, depend on the circuit only (requirement I12).

Arithmetic is exact: accumulators are wide enough that no sum overflows
(``acc_bits = bw + bx + ceil(log2(n + 1))`` for a sum of ``n`` products of a
``bw``-bit and a ``bx``-bit signed number and a ``bw + bx``-bit bias). Where
a value is narrowed on purpose (the hidden activations of :func:`mlp`), the
function documents what happens to values that do not fit.
"""

from __future__ import annotations

import json
from typing import Sequence

from pyfe4ai.circuits.builder import MUL_METHODS, Builder
from pyfe4ai.circuits.circuit import Circuit
from pyfe4ai.circuits.layout import DataLayout, Field, KeyLayout, OutputLayout
from pyfe4ai.core.exceptions import FEValidationError

#: Upper bound of the bit widths the library takes (one machine word of
#: weights or data is plenty; it keeps the circuits a sane size).
MAX_WIDTH = 64


class ParamCircuit:
    """A library circuit with its layouts (read-only properties below)."""

    __slots__ = ("_name", "_params", "_circuit", "_key_layout", "_data_layout", "_output_layout")

    def __init__(self, name: str, params: dict, circuit: Circuit, key_layout: KeyLayout, data_layout: DataLayout,
                 output_layout: OutputLayout):
        if not isinstance(circuit, Circuit):
            raise FEValidationError("circuit must be a Circuit")
        if not (isinstance(key_layout, KeyLayout) and isinstance(data_layout, DataLayout)
                and isinstance(output_layout, OutputLayout)):
            raise FEValidationError("the layouts must be a KeyLayout, a DataLayout and an OutputLayout")
        if (key_layout.n_bits, data_layout.n_bits, output_layout.n_bits) != (circuit.m, circuit.n, circuit.n_out):
            raise FEValidationError("the layouts do not match the circuit's key, data and output bits")
        params = dict(params)
        try:
            json.dumps(params, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise FEValidationError("params must be JSON-able") from exc
        for attr, value in (("_name", str(name)), ("_params", params), ("_circuit", circuit),
                            ("_key_layout", key_layout), ("_data_layout", data_layout),
                            ("_output_layout", output_layout)):
            object.__setattr__(self, attr, value)

    def __setattr__(self, name, value):
        raise AttributeError("a ParamCircuit is immutable")

    def __reduce__(self):
        return (ParamCircuit, (self._name, self._params, self._circuit, self._key_layout, self._data_layout,
                               self._output_layout))

    @property
    def name(self) -> str:
        """The library function's name (``"linear_threshold"``, ...)."""
        return self._name

    @property
    def params(self) -> dict:
        """The public parameters (a copy on every access)."""
        return dict(self._params)

    @property
    def circuit(self) -> Circuit:
        """The :class:`~pyfe4ai.circuits.circuit.Circuit` (without the guard bit)."""
        return self._circuit

    @property
    def key_layout(self) -> KeyLayout:
        """The :class:`~pyfe4ai.circuits.layout.KeyLayout` of ``w`` (``m`` bits)."""
        return self._key_layout

    @property
    def data_layout(self) -> DataLayout:
        """The :class:`~pyfe4ai.circuits.layout.DataLayout` of ``x`` (``n`` bits)."""
        return self._data_layout

    @property
    def output_layout(self) -> OutputLayout:
        """The :class:`~pyfe4ai.circuits.layout.OutputLayout` (``n_out`` bits)."""
        return self._output_layout

    def describe(self) -> dict:
        """A JSON-able description: name, parameters, the circuit's sizes and
        digest (hex), and the three layouts. Canonical as JSON with sorted
        keys and compact separators (:meth:`describe_json`)."""
        c = self._circuit
        return {
            "name": self._name,
            "params": dict(self._params),
            "circuit": {"m": c.m, "n": c.n, "n_gates": c.n_gates, "n_out": c.n_out, "digest": c.digest().hex()},
            "key_layout": self._key_layout.to_dict(),
            "data_layout": self._data_layout.to_dict(),
            "output_layout": self._output_layout.to_dict(),
        }

    def describe_json(self) -> str:
        """:meth:`describe` as canonical JSON (sorted keys, no spaces)."""
        return json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))

    def evaluate(self, w: dict, x: dict) -> dict:
        """The output values for the key values *w* and the data values *x*
        (dicts by field name), through the layouts and
        :meth:`~pyfe4ai.circuits.circuit.Circuit.evaluate_plain`: the
        reference of the tests and of the schemes' decryption."""
        bits = self._circuit.evaluate_plain(self._key_layout.encode(w), self._data_layout.encode(x))
        return self._output_layout.decode(bits)

    def __repr__(self):
        return "ParamCircuit({}, {}, {!r})".format(self._name, json.dumps(self._params, sort_keys=True),
                                                    self._circuit)


def _check_width(value, what: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_WIDTH:
        raise FEValidationError("{} must be an integer in [1, {}]".format(what, MAX_WIDTH))
    return value


def _check_n(value, what: str) -> int:
    if type(value) is not int or value < 1:
        raise FEValidationError("{} must be an integer >= 1".format(what))
    return value


def _check_mul(mul) -> str:
    if mul not in MUL_METHODS:
        raise FEValidationError("mul must be one of {}".format(", ".join(repr(m) for m in MUL_METHODS)))
    return mul


def linear_acc_bits(n: int, bw: int, bx: int) -> int:
    """``bw + bx + ceil(log2(n + 1))``: holds ``bias + sum_i w_i x_i`` exactly
    for ``n`` products of signed ``bw``- and ``bx``-bit numbers and a signed
    ``bw + bx``-bit bias (``|sum| <= (n + 2) * 2^(bw + bx - 2) < 2^(acc_bits - 1)``)."""
    return bw + bx + n.bit_length()


def _dot(b: Builder, ws: Sequence[list], bias: list, xs: Sequence[list], width: int, mul: str) -> list:
    """``bias + sum_i ws[i] * xs[i]`` in *width* bits (:meth:`Builder.dot`): exact
    products of ``len(w) + len(x)`` bits (Booth on the data word), then a
    balanced adder tree."""
    return b.dot(ws, xs, width, init=bias, method=mul)


def _linear_layouts(n: int, bw: int, bx: int):
    key = KeyLayout([Field("weights", n, bw, True), Field("bias", 1, bw + bx, True)])
    data = DataLayout([Field("x", n, bx, True)])
    return key, data


def _linear_sum(b: Builder, n: int, bw: int, bx: int, acc_bits: int, mul: str) -> list:
    ws = [b.key_word(i * bw, bw) for i in range(n)]
    bias = b.key_word(n * bw, bw + bx)
    xs = [b.data_word(i * bx, bx) for i in range(n)]
    return _dot(b, ws, bias, xs, acc_bits, mul)


def linear_threshold(n: int, bw: int, bx: int, mul: str = "booth") -> ParamCircuit:
    """``[sum_i w_i x_i + bias >= 0]``: the decision of a logistic regression
    or a linear SVM.

    Key: ``weights`` (``n`` signed ``bw``-bit), ``bias`` (one signed
    ``bw + bx``-bit). Data: ``x`` (``n`` signed ``bx``-bit). Output:
    ``decision`` (one unsigned bit). The sum is exact in
    :func:`linear_acc_bits` bits.
    """
    n, bw, bx, mul = _check_n(n, "n"), _check_width(bw, "bw"), _check_width(bx, "bx"), _check_mul(mul)
    acc_bits = linear_acc_bits(n, bw, bx)
    key, data = _linear_layouts(n, bw, bx)
    out = OutputLayout([Field("decision", 1, 1, False)])
    b = Builder(key.n_bits, data.n_bits)
    acc = _linear_sum(b, n, bw, bx, acc_bits, mul)
    b.output(b.ge_zero(acc))
    params = {"n": n, "bw": bw, "bx": bx, "mul": mul, "acc_bits": acc_bits}
    return ParamCircuit("linear_threshold", params, b.build(), key, data, out)


def linear_score(n: int, bw: int, bx: int, shift: int = 0, mul: str = "booth") -> ParamCircuit:
    """``floor((sum_i w_i x_i + bias) / 2^shift)``: the score of a linear model.

    Same key and data as :func:`linear_threshold`. Output: ``score`` (one
    signed number of ``acc_bits - shift`` bits, ``acc_bits`` from
    :func:`linear_acc_bits`): the exact sum shifted right arithmetically,
    never saturated or wrapped. ``0 <= shift < acc_bits``.
    """
    n, bw, bx, mul = _check_n(n, "n"), _check_width(bw, "bw"), _check_width(bx, "bx"), _check_mul(mul)
    acc_bits = linear_acc_bits(n, bw, bx)
    if type(shift) is not int or not 0 <= shift < acc_bits:
        raise FEValidationError("shift must be an integer in [0, {})".format(acc_bits))
    key, data = _linear_layouts(n, bw, bx)
    out = OutputLayout([Field("score", 1, acc_bits - shift, True)])
    b = Builder(key.n_bits, data.n_bits)
    acc = _linear_sum(b, n, bw, bx, acc_bits, mul)
    b.output_word(b.shift_right(acc, shift))
    params = {"n": n, "bw": bw, "bx": bx, "shift": shift, "mul": mul, "acc_bits": acc_bits}
    return ParamCircuit("linear_score", params, b.build(), key, data, out)


def mlp(n: int, hidden: int, bw: int, bx: int, bh: int, shift: int, mul: str = "booth") -> ParamCircuit:
    """``[sum_k v_k h_k + c >= 0]`` with ``h_k = min(max(z_k, 0) >> shift, 2^bh - 1)``
    and ``z_k = sum_i W_ki x_i + b_k``: a network with one hidden ReLU layer.

    Key: ``w1`` (``hidden * n`` signed ``bw``-bit, row-major: element
    ``k * n + i`` is ``W_ki``), ``b1`` (``hidden`` signed ``bw + bx``-bit),
    ``w2`` (``hidden`` signed ``bw``-bit, the ``v_k``), ``b2`` (one signed
    ``bw + bh + 1``-bit, the ``c``). Data: ``x`` (``n`` signed ``bx``-bit).
    Output: ``decision`` (one unsigned bit).

    Overflow: the sums ``z_k`` (in ``linear_acc_bits(n, bw, bx)`` bits) and
    the output sum (in ``linear_acc_bits(hidden, bw, bh + 1)`` bits) are
    exact. The activations are narrowed to ``bh`` unsigned bits on purpose:
    ``max(z_k, 0)`` is shifted right by *shift* and **saturates** at
    ``2^bh - 1`` (never wraps). ``0 <= shift <= acc_bits - 2``. The second
    layer multiplies two garbled words (``v_k`` and ``h_k``), so its ANDs
    take two table entries in both modes.
    """
    n, hidden = _check_n(n, "n"), _check_n(hidden, "hidden")
    bw, bx, bh = _check_width(bw, "bw"), _check_width(bx, "bx"), _check_width(bh, "bh")
    mul = _check_mul(mul)
    acc1 = linear_acc_bits(n, bw, bx)
    if type(shift) is not int or not 0 <= shift <= acc1 - 2:
        raise FEValidationError("shift must be an integer in [0, {}]".format(acc1 - 2))
    acc2 = linear_acc_bits(hidden, bw, bh + 1)
    key = KeyLayout([Field("w1", hidden * n, bw, True), Field("b1", hidden, bw + bx, True),
                     Field("w2", hidden, bw, True), Field("b2", 1, bw + bh + 1, True)])
    data = DataLayout([Field("x", n, bx, True)])
    out = OutputLayout([Field("decision", 1, 1, False)])
    b = Builder(key.n_bits, data.n_bits)
    o_w1, o_b1, o_w2, o_b2 = (key.offset(f) for f in ("w1", "b1", "w2", "b2"))
    xs = [b.data_word(i * bx, bx) for i in range(n)]
    hs = []
    for k in range(hidden):
        ws = [b.key_word(o_w1 + (k * n + i) * bw, bw) for i in range(n)]
        z = _dot(b, ws, b.key_word(o_b1 + k * (bw + bx), bw + bx), xs, acc1, mul)
        u = b.relu(z)[shift:acc1 - 1]  # max(z, 0) >> shift, unsigned (the top bit of the ReLU is 0)
        if len(u) <= bh:
            h = b.zero_extend(u, bh)
        else:
            over = u[bh]
            for bit in u[bh + 1:]:
                over = b.or_(over, bit)
            h = [b.or_(bit, over) for bit in u[:bh]]
        hs.append(b.zero_extend(h, bh + 1))  # non-negative: a zero sign bit
    vs = [b.key_word(o_w2 + k * bw, bw) for k in range(hidden)]
    y = _dot(b, vs, b.key_word(o_b2, bw + bh + 1), hs, acc2, mul)
    b.output(b.ge_zero(y))
    params = {"n": n, "hidden": hidden, "bw": bw, "bx": bx, "bh": bh, "shift": shift, "mul": mul,
              "acc_bits": acc1, "acc2_bits": acc2}
    return ParamCircuit("mlp", params, b.build(), key, data, out)


def argmax_scores(k: int, n: int, bw: int, bx: int, mul: str = "booth") -> ParamCircuit:
    """The index of the largest of ``k`` linear scores ``s_j = sum_i W_ji x_i + b_j``
    (a linear classifier over ``k`` classes).

    Key: ``weights`` (``k * n`` signed ``bw``-bit, row-major: element
    ``j * n + i`` is ``W_ji``), ``bias`` (``k`` signed ``bw + bx``-bit).
    Data: ``x`` (``n`` signed ``bx``-bit). Output: ``index`` (one unsigned
    number of ``max(1, ceil(log2 k))`` bits, binary): the lowest ``j`` whose
    score is the maximum. The scores are exact (``linear_acc_bits(n, bw,
    bx)`` bits). ``k >= 2``.
    """
    k, n = _check_n(k, "k"), _check_n(n, "n")
    if k < 2:
        raise FEValidationError("k must be at least 2")
    bw, bx, mul = _check_width(bw, "bw"), _check_width(bx, "bx"), _check_mul(mul)
    acc_bits = linear_acc_bits(n, bw, bx)
    ibits = max(1, (k - 1).bit_length())
    key = KeyLayout([Field("weights", k * n, bw, True), Field("bias", k, bw + bx, True)])
    data = DataLayout([Field("x", n, bx, True)])
    out = OutputLayout([Field("index", 1, ibits, False)])
    b = Builder(key.n_bits, data.n_bits)
    xs = [b.data_word(i * bx, bx) for i in range(n)]
    o_bias = key.offset("bias")
    scores = []
    for j in range(k):
        ws = [b.key_word((j * n + i) * bw, bw) for i in range(n)]
        scores.append(_dot(b, ws, b.key_word(o_bias + j * (bw + bx), bw + bx), xs, acc_bits, mul))
    index, _ = b.argmax(scores)
    b.output_word(index)
    params = {"k": k, "n": n, "bw": bw, "bx": bx, "mul": mul, "acc_bits": acc_bits}
    return ParamCircuit("argmax_scores", params, b.build(), key, data, out)
