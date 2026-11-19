"""Layouts: how the values of ``w``, ``x`` and the output map to circuit bits.

A layout is an ordered list of :class:`Field` objects; a field is a named
array of ``count`` integers of ``bits`` bits each, signed (two's complement)
or unsigned. The bits of a layout are the fields in their declared order,
the elements of a field in order, each element least significant bit first:
field ``f`` element ``i`` bit ``j`` is bit ``offset(f) + i * bits + j``.

:class:`KeyLayout` describes ``w`` (the key wires), :class:`DataLayout`
``x`` (the data wires) and :class:`OutputLayout` the output bits.
:meth:`Layout.encode` takes a ``dict`` from field names to values (a Python
``int`` or a NumPy integer for a field of one element, a list, tuple or
one-dimensional NumPy integer array of ``count`` integers otherwise; a
one-element field also takes a one-element sequence) and refuses
(:class:`~pyfe4ai.core.exceptions.FEValidationError`) a missing or unknown
name, a wrong shape, a non-integer (``bool`` and floats included) and a
value outside the field's range: nothing is wrapped or clipped silently. The
messages name the field and the element, never the value (``w`` and ``x``
are secrets). :meth:`Layout.decode` gives back an ``int`` for a field of one
element and a list of ``int`` otherwise.

:meth:`Layout.to_dict` is the canonical JSON-able description
(``{"kind": ..., "fields": [{"name", "count", "bits", "signed"}, ...]}``)
and :meth:`Layout.from_dict` reads exactly that form back.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Sequence

import numpy as np

from pyfe4ai.core.exceptions import FEValidationError

#: A field name is an identifier of at most 64 ASCII characters.
FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_FIELD_KEYS = ("name", "count", "bits", "signed")


def _is_int(value) -> bool:
    return (type(value) is int) or (isinstance(value, np.integer) and not isinstance(value, np.bool_))


@dataclasses.dataclass(frozen=True)
class Field:
    """A named array of *count* integers of *bits* bits, *signed* (two's
    complement) or unsigned."""

    name: str
    count: int
    bits: int
    signed: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not FIELD_NAME.match(self.name):
            raise FEValidationError("a field name is an identifier of at most 64 ASCII characters")
        if type(self.count) is not int or self.count < 1:
            raise FEValidationError("field {!r}: count must be an integer >= 1".format(self.name))
        if type(self.bits) is not int or self.bits < 1:
            raise FEValidationError("field {!r}: bits must be an integer >= 1".format(self.name))
        if type(self.signed) is not bool:
            raise FEValidationError("field {!r}: signed must be a bool".format(self.name))

    @property
    def n_bits(self) -> int:
        """``count * bits``."""
        return self.count * self.bits

    @property
    def min_value(self) -> int:
        """The smallest value of an element."""
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max_value(self) -> int:
        """The largest value of an element."""
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    def to_dict(self) -> dict:
        """``{"name", "count", "bits", "signed"}``."""
        return {"name": self.name, "count": self.count, "bits": self.bits, "signed": self.signed}

    @classmethod
    def from_dict(cls, data) -> "Field":
        """The inverse of :meth:`to_dict` (exactly its keys)."""
        if not isinstance(data, dict) or set(data) != set(_FIELD_KEYS):
            raise FEValidationError("a field is a dict with the keys {}".format(", ".join(_FIELD_KEYS)))
        return cls(data["name"], data["count"], data["bits"], data["signed"])

    def _elements(self, value) -> list:
        """The *count* elements of *value* as Python ints, range-checked."""
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "iu":
                raise FEValidationError("field {!r}: an array must have an integer dtype".format(self.name))
            if value.ndim == 0 and self.count == 1:
                items = [value[()]]
            elif value.ndim == 1:
                items = list(value)
            else:
                raise FEValidationError("field {!r}: an array must be one-dimensional".format(self.name))
        elif isinstance(value, (list, tuple)):
            items = list(value)
        elif self.count == 1:
            items = [value]
        else:
            raise FEValidationError("field {!r}: expected a sequence of {} integers".format(self.name, self.count))
        if len(items) != self.count:
            raise FEValidationError("field {!r}: expected {} elements, got {}".format(
                self.name, self.count, len(items)))
        lo, hi = self.min_value, self.max_value
        out = []
        for i, item in enumerate(items):
            if not _is_int(item):
                raise FEValidationError("field {!r} element {}: not an integer".format(self.name, i))
            item = int(item)
            if not lo <= item <= hi:
                raise FEValidationError("field {!r} element {}: out of the {} {}-bit range".format(
                    self.name, i, "signed" if self.signed else "unsigned", self.bits))
            out.append(item)
        return out


class Layout:
    """An ordered list of fields (see the module documentation). Use
    :class:`KeyLayout`, :class:`DataLayout` or :class:`OutputLayout`."""

    #: ``"key"``, ``"data"`` or ``"output"``: set by the subclasses.
    KIND = None
    #: Whether the layout may have no bits (only the data layout may, ``n = 0``).
    ALLOW_EMPTY = False

    __slots__ = ("_fields",)

    def __init__(self, fields: Iterable[Field]):
        if self.KIND is None:
            raise TypeError("use KeyLayout, DataLayout or OutputLayout")
        fields = tuple(fields)
        names = set()
        for f in fields:
            if not isinstance(f, Field):
                raise FEValidationError("a layout is a sequence of Field objects")
            if f.name in names:
                raise FEValidationError("field {!r} appears twice".format(f.name))
            names.add(f.name)
        if not fields and not self.ALLOW_EMPTY:
            raise FEValidationError("a {} layout needs at least one field".format(self.KIND))
        object.__setattr__(self, "_fields", fields)

    def __setattr__(self, name, value):
        raise AttributeError("a layout is immutable")

    def __reduce__(self):
        return (type(self), (self._fields,))

    @property
    def fields(self) -> tuple:
        """The fields in order."""
        return self._fields

    @property
    def names(self) -> tuple:
        """The field names in order."""
        return tuple(f.name for f in self._fields)

    @property
    def n_bits(self) -> int:
        """Total number of bits."""
        return sum(f.n_bits for f in self._fields)

    def field(self, name: str) -> Field:
        """The field called *name*."""
        for f in self._fields:
            if f.name == name:
                return f
        raise FEValidationError("no field {!r} in the {} layout".format(name, self.KIND))

    def offset(self, name: str) -> int:
        """The first bit of the field called *name*."""
        pos = 0
        for f in self._fields:
            if f.name == name:
                return pos
            pos += f.n_bits
        raise FEValidationError("no field {!r} in the {} layout".format(name, self.KIND))

    def encode(self, values) -> list:
        """The bits (0 or 1) of *values*, a ``dict`` with one entry per field."""
        if not isinstance(values, dict):
            raise FEValidationError("the {} values are a dict from field names to values".format(self.KIND))
        unknown = sorted(str(k) for k in values if k not in self.names)
        if unknown:
            raise FEValidationError("unknown {} fields: {}".format(self.KIND, ", ".join(repr(k) for k in unknown)))
        missing = [n for n in self.names if n not in values]
        if missing:
            raise FEValidationError("missing {} fields: {}".format(self.KIND, ", ".join(repr(k) for k in missing)))
        bits = []
        for f in self._fields:
            mask = (1 << f.bits) - 1
            for v in f._elements(values[f.name]):
                u = v & mask
                bits.extend((u >> j) & 1 for j in range(f.bits))
        return bits

    def decode(self, bits: Sequence[int]) -> dict:
        """The values of the bits *bits* (``n_bits`` of them, each 0 or 1)."""
        if isinstance(bits, (bytes, bytearray, str)):
            raise FEValidationError("decode takes a sequence of bits (0 or 1)")
        bits = list(bits)
        if len(bits) != self.n_bits:
            raise FEValidationError("the {} layout has {} bits, got {}".format(self.KIND, self.n_bits, len(bits)))
        for i, b in enumerate(bits):
            if not (_is_int(b) or type(b) is bool) or b not in (0, 1):
                raise FEValidationError("bit {} is not 0 or 1".format(i))
        out = {}
        pos = 0
        for f in self._fields:
            vals = []
            for _ in range(f.count):
                u = 0
                for j in range(f.bits):
                    u |= int(bits[pos + j]) << j
                pos += f.bits
                if f.signed and u >> (f.bits - 1):
                    u -= 1 << f.bits
                vals.append(u)
            out[f.name] = vals[0] if f.count == 1 else vals
        return out

    def to_dict(self) -> dict:
        """``{"kind": ..., "fields": [...]}`` (JSON-able, canonical)."""
        return {"kind": self.KIND, "fields": [f.to_dict() for f in self._fields]}

    @classmethod
    def from_dict(cls, data) -> "Layout":
        """The inverse of :meth:`to_dict`. On :class:`Layout` itself the kind
        picks the subclass; on a subclass it must be that subclass's kind."""
        if not isinstance(data, dict) or set(data) != {"kind", "fields"}:
            raise FEValidationError("a layout is a dict with the keys kind, fields")
        kinds = {sub.KIND: sub for sub in (KeyLayout, DataLayout, OutputLayout)}
        target = kinds.get(data["kind"]) if isinstance(data["kind"], str) else None
        if target is None or (cls is not Layout and target is not cls):
            raise FEValidationError("unexpected layout kind")
        if not isinstance(data["fields"], list):
            raise FEValidationError("the fields of a layout are a list")
        return target(Field.from_dict(f) for f in data["fields"])

    def __eq__(self, other):
        if not isinstance(other, Layout):
            return NotImplemented
        return type(self) is type(other) and self._fields == other._fields

    def __hash__(self):
        return hash((self.KIND, self._fields))

    def __repr__(self):
        return "{}({})".format(type(self).__name__, ", ".join(
            "{}[{}]:{}{}".format(f.name, f.count, "i" if f.signed else "u", f.bits) for f in self._fields))


class KeyLayout(Layout):
    """The layout of ``w``: the key wires ``0 .. m-1``."""

    KIND = "key"
    __slots__ = ()


class DataLayout(Layout):
    """The layout of ``x``: the data wires ``m .. m+n-1`` (may be empty)."""

    KIND = "data"
    ALLOW_EMPTY = True
    __slots__ = ()


class OutputLayout(Layout):
    """The layout of the output bits."""

    KIND = "output"
    __slots__ = ()
