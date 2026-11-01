"""Label encoding shared by the labelled multi-client schemes.

MCFE ciphertexts are deterministic per (client slot, label), so a client
must never encrypt twice under the same label. The ndarray helpers encrypt
every array element under its own :class:`ElementLabel`; :func:`encode_label`
maps user labels (strings) and element labels into disjoint domains, so no
user-chosen string can collide with the label of an array element.
"""

from __future__ import annotations

import json
from typing import NamedTuple

from pyfe4ai.core.exceptions import FEValidationError


class ElementLabel(NamedTuple):
    """Label of one ndarray element: the caller's label, the layer and the index."""

    label: str
    layer: int
    index: tuple


def element_label(label: str, layer: int, index) -> ElementLabel:
    """Return the label of element *index* of layer *layer* under *label*."""
    if not isinstance(label, str):
        raise FEValidationError("label must be a string, got {!r}".format(label))
    return ElementLabel(label, int(layer), tuple(int(v) for v in index))


def _check_text(text: str, what: str) -> None:
    """Raise unless *text* is text: it holds no surrogate code point, which is no character and has no UTF-8 encoding.

    ``json.dumps`` writes a character outside the Basic Multilingual Plane as the pair of ``\\u`` escapes of its two
    UTF-16 halves, and writes a pair of lone surrogates as the same two escapes, so without this check two labels
    gave one encoding, and one tag.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FEValidationError("{} must be valid text, but it contains a lone surrogate (U+{:04X}) at position {}"
                                .format(what, ord(text[exc.start]), exc.start)) from None


def encode_label(label) -> str:
    """Return the unambiguous string that is hashed for *label*.

    Args:
        label: A user label (``str``) or an :class:`ElementLabel`.

    Raises:
        FEValidationError: For any other type, for a string that is not valid text (it contains a lone surrogate),
            and for an :class:`ElementLabel` whose layer or index is not made of integers.
    """
    if isinstance(label, ElementLabel):
        if isinstance(label.label, str):
            _check_text(label.label, "the label of an element label")
        try:
            layer, index = int(label.layer), [int(v) for v in label.index]
        except (TypeError, ValueError, OverflowError):
            # not the error of int(): a ValueError out of a decryption is read as "no discrete log"
            raise FEValidationError("the layer and the index of an element label must be integers") from None
        return json.dumps(["element", label.label, layer, index])
    if isinstance(label, str):
        _check_text(label, "label")
        return json.dumps(["label", label])
    raise FEValidationError("label must be a string, got {!r}".format(label))


def chunk_label(label: str, chunk: int) -> ElementLabel:
    """Return the label of batch chunk *chunk* under *label*.

    Used by schemes that pack many values into one ciphertext (one ciphertext
    per chunk and label); encoded in the element domain with layer ``-1``,
    so it never collides with a user label or an element label.
    """
    return element_label(label, -1, (chunk,))
