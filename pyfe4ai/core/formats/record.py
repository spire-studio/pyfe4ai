"""The base of the serialized-object containers: :class:`Record` and :class:`Field`.

A record is a typed view of the dict that is its wire form. It holds that
dict itself -- no copy, no conversion -- so a record and its dict are always
the same bytes on the wire, key order and unknown fields included:
``Record.from_dict(d).to_dict() is d``. Reading a field reads the dict as the
code of earlier versions did:

* ``record.name`` is ``d["name"]`` (``KeyError`` if the field is absent);
* ``record.get("name", default)`` is ``d.get("name", default)`` and
  ``"name" in record`` is ``"name" in d``, for a declared field only (an
  undeclared name raises ``AttributeError``);
* ``record.name = value``, ``record.update(...)`` and ``del record.name``
  write the dict.

So the checks a scheme makes when it reads a key or a ciphertext stay where
they are, with their exceptions; what a record adds is one declaration of
every field, its order on the wire and what its absence means
(``Field(absent=...)``: the derivation of earlier versions that a missing
marker stands for, applied by :mod:`pyfe4ai.core.versions`, never silently
by attribute access). :meth:`Record.new` builds the dict of a new object in
the declared order.
"""

from __future__ import annotations

from typing import Any, ClassVar

#: Marks a field that :meth:`Record.new` leaves out (an optional field
#: whose absence has a meaning, e.g. the JSON format of versions <= 0.3.0).
ABSENT: Any = type("_Absent", (), {"__repr__": lambda self: "ABSENT", "__bool__": lambda self: False})()

_REQUIRED = object()


class Field:
    """A field of a :class:`Record`, by its wire name (the attribute name).

    Args:
        doc: What the field holds (its type and encoding).
        absent: What a missing field means, for fields that versions or
            options leave out (documentation, and the value
            :mod:`pyfe4ai.core.versions` reads for it). Reading a missing
            field raises ``KeyError`` as ``d[name]`` does, whatever it means.
    """

    def __init__(self, doc: str = "", *, absent: Any = _REQUIRED) -> None:
        self.doc = doc
        self.absent = absent
        self.name = ""

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    @property
    def optional(self) -> bool:
        """Whether the field has a meaning when absent."""
        return self.absent is not _REQUIRED

    def __get__(self, record, owner=None):
        if record is None:
            return self
        return record._raw[self.name]

    def __set__(self, record, value) -> None:
        record._raw[self.name] = value

    def __delete__(self, record) -> None:
        del record._raw[self.name]

    def __repr__(self) -> str:
        return "Field({!r})".format(self.name)


class Record:
    """A serialized object: a typed view of its wire dict (see the module docstring).

    Subclasses declare their fields as :class:`Field` attributes, in wire
    order (a base class's fields first), or give that order in ``ORDER``
    when a subclass's fields go between its base's.
    """

    __slots__ = ("_raw",)

    #: Field names in wire order (set for every subclass).
    FIELDS: ClassVar[tuple[str, ...]] = ()
    _fields: ClassVar[dict[str, Field]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Field):
                    fields[name] = value
        order = vars(cls).get("ORDER")
        if order is not None:
            if sorted(order) != sorted(fields):
                raise TypeError("{}.ORDER must name its fields {}".format(cls.__name__, sorted(fields)))
            fields = {name: fields[name] for name in order}
        cls._fields = fields
        cls.FIELDS = tuple(fields)
        if cls.__doc__ and fields:
            cls.__doc__ = cls.__doc__.rstrip() + "\n\n    Fields, in wire order:\n\n" + "".join(
                "    * ``{}``{}{}\n".format(
                    name, " -- " + field.doc if field.doc else "",
                    " (absent: {!r})".format(field.absent) if field.optional and field.absent is not None
                    else " (optional)" if field.optional else "",
                )
                for name, field in fields.items()
            )

    def __init__(self, raw) -> None:
        self._raw = raw

    @classmethod
    def from_dict(cls, data):
        """The record of a wire dict (the dict itself, not a copy; its fields
        are read when used, with the checks of each use)."""
        return cls(data)

    @classmethod
    def new(cls, **values):
        """A new record: a dict of *values* in the declared order; a value
        :data:`ABSENT` leaves its field out.

        Raises:
            TypeError: For a name that is not a field.
        """
        unknown = [name for name in values if name not in cls._fields]
        if unknown:
            raise TypeError("{} has no fields {}".format(cls.__name__, unknown))
        return cls({name: values[name] for name in cls.FIELDS if name in values and values[name] is not ABSENT})

    @classmethod
    def extend(cls, base, **values):
        """``dict(base, **values)`` as a record: a new dict with *base*'s
        fields in their order, then *values* (a field of *base* keeps its
        place, as with a dict).

        Raises:
            TypeError: For a name of *values* that is not a field.
        """
        record = cls(dict(base))
        record.update(**values)
        return record

    def to_dict(self):
        """The wire dict (the record's own, not a copy)."""
        return self._raw

    def _field(self, name: str) -> str:
        if name not in self._fields:
            raise AttributeError("{} has no field {!r}".format(type(self).__name__, name))
        return name

    def get(self, name: str, default=None):
        """``d.get(name, default)`` for the declared field *name*."""
        return self._raw.get(self._field(name), default)

    def update(self, **values) -> None:
        """``d.update(values)`` for declared fields (a field already present
        keeps its place on the wire, as with a dict).

        Raises:
            TypeError: For a name that is not a field.
        """
        unknown = [name for name in values if name not in self._fields]
        if unknown:
            raise TypeError("{} has no fields {}".format(type(self).__name__, unknown))
        self._raw.update({name: value for name, value in values.items() if value is not ABSENT})

    def __contains__(self, name: str) -> bool:
        return self._field(name) in self._raw

    def __eq__(self, other) -> bool:
        if isinstance(other, Record):
            return type(other) is type(self) and other._raw == self._raw
        return NotImplemented

    __hash__ = None

    def __repr__(self) -> str:
        return "{}({})".format(type(self).__name__, ", ".join(repr(k) for k in self._raw) if isinstance(
            self._raw, dict) else type(self._raw).__name__)
