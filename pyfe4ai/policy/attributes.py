"""Attribute labels.

A label is ``namespace:value``. The namespace is a non-empty string of ASCII
letters, digits, ``_``, ``.`` and ``-``; it ends at the first ``:``. The value
is a non-empty string without whitespace, without control characters (Unicode
category ``Cc``), without lone surrogates (``Cs``) and without ``,``, ``(``,
``)`` or ``#`` (``#`` is kept for the copies of a label that a compiled
policy numbers). A label must be in Unicode normalization form NFC: it is
checked, never normalized, so the label that is hashed is the one the caller
wrote and two spellings of one text are not silently one attribute.
"""

from __future__ import annotations

import re
import unicodedata

from pyfe4ai.core.exceptions import FEValidationError

_NAMESPACE = re.compile(r"[A-Za-z0-9_.\-]+")

#: Characters a value may not contain besides whitespace and control characters.
_RESERVED = frozenset(",()#")


def validate_label(label: str) -> str:
    """Return *label* unchanged if it is a valid attribute label, or raise
    ``FEValidationError`` naming the label and the rule it breaks."""
    if not isinstance(label, str):
        raise FEValidationError("an attribute label must be a str, got {}".format(type(label).__name__))
    namespace, colon, value = label.partition(":")
    if not colon:
        raise FEValidationError("attribute label {!r} is not of the form namespace:value".format(label))
    if _NAMESPACE.fullmatch(namespace) is None:
        raise FEValidationError(
            "attribute label {!r}: the namespace must be one or more of A-Z a-z 0-9 _ . -".format(label)
        )
    if not value:
        raise FEValidationError("attribute label {!r}: the value is empty".format(label))
    for char in value:
        if char.isspace():
            raise FEValidationError("attribute label {!r}: the value contains whitespace".format(label))
        if char in _RESERVED:
            raise FEValidationError(
                "attribute label {!r}: the value contains the reserved character {!r}".format(label, char)
            )
        category = unicodedata.category(char)
        if category == "Cc":
            raise FEValidationError("attribute label {!r}: the value contains a control character".format(label))
        if category == "Cs":
            raise FEValidationError("attribute label {!r}: the value contains a lone surrogate".format(label))
    if not unicodedata.is_normalized("NFC", label):
        raise FEValidationError("attribute label {!r} is not in Unicode normalization form NFC".format(label))
    return label
