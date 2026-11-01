"""Attribute labels, attribute universes and hashing labels to ``Z_q``.

A label is ``namespace:value``. The namespace is a non-empty string of ASCII
letters, digits, ``_``, ``.`` and ``-``; it ends at the first ``:``. The value
is a non-empty string without whitespace, without control characters (Unicode
category ``Cc``), without invisible format characters such as zero-width and
bidirectional marks (``Cf``), without other code points that render as nothing
(Unicode's Default_Ignorable_Code_Point: variation selectors, Hangul fillers,
the grapheme joiner), without lone surrogates (``Cs``) and without ``,``,
``(``, ``)`` or ``#`` (``#`` is kept for the copies of a label that a compiled
policy numbers). A label must be in Unicode normalization form NFC:
it is checked, never normalized, so the label that is hashed is the one the
caller wrote and two spellings of one text are not silently one attribute. It
must be a ``str`` itself, not an instance of a subclass.

A :class:`Universe` is the set of labels that a scheme setup covers, with the
number of copies of each label that it provides. Its encoding is public
data, so it has one spelling: :meth:`Universe.from_encoding` accepts no other
text than :meth:`Universe.encoding` gives. :func:`attribute_to_zq` hashes a
label to an element of ``Z_q``. :func:`attribute_set_digest` hashes the set of
labels that a key or a ciphertext is made for, however it was written down.
Pure Python: the standard library and :mod:`pyfe4ai.core.exceptions` only.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass

from pyfe4ai.core.exceptions import FEValidationError

_NAMESPACE = re.compile(r"[A-Za-z0-9_.\-]+")

#: Characters a value may not contain besides whitespace and control characters.
_RESERVED = frozenset(",()#")

#: Default_Ignorable_Code_Point of the Unicode character database, which :mod:`unicodedata` does not give: two labels
#: that differ by one of these look the same, whatever the category of the code point.
_IGNORABLE = re.compile(
    "[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e\u2060-\u206f"
    "\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8\U0001bca0-\U0001bca3\U0001d173-\U0001d17a"
    "\U000e0000-\U000e0fff]"
)

_ATTRIBUTE_DOMAIN = b"PYFE4AI-V05-POLICY-ATTR-Zq"
_UNIVERSE_DOMAIN = b"PYFE4AI-V05-POLICY-UNIVERSE"
_ATTRSET_DOMAIN = b"PYFE4AI-V05-POLICY-ATTRSET"

#: The version marker of :func:`attribute_set_digest`: SHAKE256 read to 32 bytes, written as 64 hexadecimal
#: characters. A scheme keeps it in its public parameters and refuses any other value.
ATTR_HASH = "shake256-64"

#: :func:`attribute_to_zq` reads 512 bits of hash, 128 more than the modulus can have.
_MAX_MODULUS_BITS = 384


def clip(text: str, limit: int = 40) -> str:
    """*text* shortened for an error message, so that input of any size is not echoed whole."""
    return text if len(text) <= limit else text[:limit] + "..."


def validate_label(label: str) -> str:
    """Return *label* unchanged if it is a valid attribute label, or raise
    ``FEValidationError`` naming the label and the rule it breaks."""
    # type(label) is str: a str subclass could encode, hash or compare differently from its text
    if type(label) is not str:
        raise FEValidationError("an attribute label must be a str, got {}".format(type(label).__name__))
    namespace, colon, value = label.partition(":")
    if not colon:
        raise FEValidationError("attribute label {} is not of the form namespace:value".format(clip(repr(label))))
    if _NAMESPACE.fullmatch(namespace) is None:
        raise FEValidationError(
            "attribute label {}: the namespace must be one or more of A-Z a-z 0-9 _ . -".format(clip(repr(label)))
        )
    if not value:
        raise FEValidationError("attribute label {}: the value is empty".format(clip(repr(label))))
    for char in value:
        if char.isspace():
            raise FEValidationError("attribute label {}: the value contains whitespace".format(clip(repr(label))))
        if char in _RESERVED:
            raise FEValidationError(
                "attribute label {}: the value contains the reserved character {!r}".format(clip(repr(label)), char)
            )
        category = unicodedata.category(char)
        if category == "Cc":
            raise FEValidationError(
                "attribute label {}: the value contains a control character".format(clip(repr(label)))
            )
        if category == "Cf":
            raise FEValidationError(
                "attribute label {}: the value contains an invisible format character".format(clip(repr(label)))
            )
        if category == "Cs":
            raise FEValidationError(
                "attribute label {}: the value contains a lone surrogate".format(clip(repr(label)))
            )
        if _IGNORABLE.fullmatch(char) is not None:
            raise FEValidationError(
                "attribute label {}: the value contains a character that renders as nothing".format(clip(repr(label)))
            )
    if not unicodedata.is_normalized("NFC", label):
        raise FEValidationError(
            "attribute label {} is not in Unicode normalization form NFC".format(clip(repr(label)))
        )
    return label


def attribute_to_zq(label: str, q: int) -> int:
    """The element of ``Z_q`` that *label* hashes to.

    ``SHAKE256("PYFE4AI-V05-POLICY-ATTR-Zq" || len(b) || b)`` read to 64 bytes,
    with ``b`` the UTF-8 bytes of the label and ``len(b)`` four bytes
    big-endian, taken as a big-endian integer modulo *q*. Within 2^-128 of
    uniform, which is why *q* stops at ``2^384``. *label* must be a valid label
    (:func:`validate_label`) and *q* an ``int`` with ``2 <= q < 2^384``;
    ``FEValidationError`` otherwise.
    """
    validate_label(label)
    # type(q) is int: a bool or an int subclass is not a modulus
    if type(q) is not int:
        raise FEValidationError("the modulus q must be an int, got {}".format(type(q).__name__))
    if q < 2:
        raise FEValidationError("the modulus q must be at least 2, got {}".format(q))
    if q.bit_length() > _MAX_MODULUS_BITS:
        raise FEValidationError(
            "the modulus q must be below 2^{}, got {} bits: 64 bytes of hash do not cover a larger one".format(
                _MAX_MODULUS_BITS, q.bit_length())
        )
    data = label.encode("utf-8")
    digest = hashlib.shake_256(_ATTRIBUTE_DOMAIN + len(data).to_bytes(4, "big") + data).digest(64)
    return int.from_bytes(digest, "big") % q


def _label_list(labels: Iterable[str]) -> list[str]:
    """The labels of the iterable *labels* as a list, each one valid."""
    # a bare str or bytes would be read one character or one byte at a time
    if isinstance(labels, (str, bytes, bytearray)):
        raise FEValidationError(
            "labels must be an iterable of attribute labels, not a bare {}".format(type(labels).__name__)
        )
    try:
        iterator = iter(labels)
    except TypeError as exc:
        raise FEValidationError(
            "labels must be an iterable of attribute labels, got {}".format(type(labels).__name__)
        ) from exc
    items = list(iterator)
    for label in items:
        validate_label(label)
    return items


def attribute_set_encoding(labels: Iterable[str]) -> str:
    """The canonical text of a set of attribute labels: compact JSON array of the distinct labels sorted by
    code point, with non-ASCII characters kept as they are, e.g. ``["a:1","b:1"]``.

    *labels* is an iterable of valid labels (:func:`validate_label`), not a bare ``str`` or ``bytes``, in any order
    and with repeats: a ``set`` and a list that names a label twice give the same text. The empty set is allowed
    and gives ``[]``. ``FEValidationError`` otherwise.
    """
    return json.dumps(sorted(set(_label_list(labels))), separators=(",", ":"), ensure_ascii=False)


def attribute_set_digest(labels: Iterable[str]) -> str:
    """``SHAKE256("PYFE4AI-V05-POLICY-ATTRSET" || encoding)`` read to 32 bytes, in hexadecimal (64 characters),
    with *encoding* the UTF-8 bytes of :func:`attribute_set_encoding` of *labels*; its version marker is
    :data:`ATTR_HASH`.

    Whatever order and repetition the labels were written in, the digest is the same, so that parties that write the
    attributes of one set down differently agree on it.
    """
    return hashlib.shake_256(_ATTRSET_DOMAIN + attribute_set_encoding(labels).encode("utf-8")).hexdigest(32)


@dataclass(frozen=True, init=False)
class Universe:
    """The attribute labels of a setup, with *max_copies* copies of each.

    ``labels`` is the tuple of the distinct labels sorted by Unicode code
    point, whatever order and repetition the constructor was given. A slot is a
    pair (label, copy) with ``1 <= copy <= max_copies``; :meth:`index` numbers
    the slots from 0, the copies of one label together and the labels in
    order. Two universes are equal, and hash alike, when their labels and
    ``max_copies`` are.

    *labels* is an iterable of valid labels (:func:`validate_label`), not a
    bare ``str`` or ``bytes``, with at least one label. *max_copies* is an
    ``int`` of at least 1. ``FEValidationError`` otherwise.
    """

    labels: tuple[str, ...]
    max_copies: int

    def __init__(self, labels: Iterable[str], max_copies: int) -> None:
        # type(max_copies) is int: a bool or an int subclass would be a second spelling of the same limit
        if type(max_copies) is not int:
            raise FEValidationError("max_copies must be an int, got {}".format(type(max_copies).__name__))
        if max_copies < 1:
            raise FEValidationError("max_copies must be at least 1, got {}".format(max_copies))
        items = _label_list(labels)
        if not items:
            raise FEValidationError("a universe needs at least one attribute label")
        object.__setattr__(self, "labels", tuple(sorted(set(items))))
        object.__setattr__(self, "max_copies", max_copies)

    @property
    def size(self) -> int:
        """The number of slots: ``len(labels) * max_copies``."""
        return len(self.labels) * self.max_copies

    def _position(self, label: object) -> int | None:
        """The position of *label* in ``labels``, or ``None`` if it is not there."""
        # type(label) is str, as for every label: a str subclass could compare differently from its text
        if type(label) is str:
            position = bisect_left(self.labels, label)
            if position < len(self.labels) and self.labels[position] == label:
                return position
        return None

    def __contains__(self, label: object) -> bool:
        return self._position(label) is not None

    def index(self, label: str, copy: int) -> int:
        """The number, from 0, of the slot (*label*, *copy*); *copy* counts from 1.

        ``FEValidationError`` if *label* is not in the universe or *copy* is
        not an ``int`` from 1 to ``max_copies``.
        """
        position = self._position(label)
        if position is None:
            raise FEValidationError("attribute label {} is not in the universe".format(clip(repr(label))))
        # type(copy) is int: a bool or a float would be a second spelling of the same copy
        if type(copy) is not int or not 1 <= copy <= self.max_copies:
            raise FEValidationError(
                "copy {} of label {}: the copy must be an int from 1 to {}, the limit max_copies".format(
                    clip(repr(copy)), clip(repr(label)), self.max_copies)
            )
        return position * self.max_copies + copy - 1

    def encoding(self) -> str:
        """The canonical text of the universe: compact JSON ``[max_copies,[label,...]]``,
        the labels sorted by code point, with non-ASCII characters kept as they are."""
        return json.dumps([self.max_copies, list(self.labels)], separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        """``SHAKE256("PYFE4AI-V05-POLICY-UNIVERSE" || encoding)`` read to 32 bytes, in hexadecimal."""
        return hashlib.shake_256(_UNIVERSE_DOMAIN + self.encoding().encode("utf-8")).hexdigest(32)

    @classmethod
    def from_encoding(cls, data: str) -> Universe:
        """The universe whose :meth:`encoding` is *data*.

        *data* is untrusted. It must be a ``str`` and exactly the text that
        :meth:`encoding` gives: any other spelling of the same universe
        (whitespace, labels out of order or repeated, an escape for a
        character that is not escaped) is refused, as is anything that is not
        a valid universe. Everything raises ``FEValidationError``.
        """
        if type(data) is not str:
            raise FEValidationError("a universe encoding must be a str, got {}".format(type(data).__name__))
        try:
            obj = json.loads(data)
        except RecursionError as exc:       # json.loads recurses once per level of nesting
            raise FEValidationError("the universe encoding is not valid JSON: it nests brackets too deeply") from exc
        except ValueError as exc:           # not JSON; also an integer with more digits than int() converts
            raise FEValidationError("the universe encoding is not valid JSON: {}".format(exc)) from exc
        if type(obj) is not list or len(obj) != 2:
            raise FEValidationError("a universe encoding is a JSON array of two elements, [max_copies, [label, ...]]")
        max_copies, labels = obj
        if type(labels) is not list:
            raise FEValidationError(
                "the second element of a universe encoding must be an array of labels, got {}".format(
                    type(labels).__name__)
            )
        universe = cls(labels, max_copies)
        if universe.encoding() != data:
            raise FEValidationError(
                "the universe encoding is valid but not in canonical form: it must be compact JSON with the "
                "labels sorted by code point, without repeats, and non-ASCII characters not escaped"
            )
        return universe
