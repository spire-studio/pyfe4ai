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

from pyfe4ai.utils.exceptions import FEValidationError


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


def encode_label(label) -> str:
    """Return the unambiguous string that is hashed for *label*.

    Args:
        label: A user label (``str``) or an :class:`ElementLabel`.

    Raises:
        FEValidationError: For any other type.
    """
    if isinstance(label, ElementLabel):
        return json.dumps(
            ["element", label.label, int(label.layer), [int(v) for v in label.index]]
        )
    if isinstance(label, str):
        return json.dumps(["label", label])
    raise FEValidationError("label must be a string, got {!r}".format(label))


def chunk_label(label: str, chunk: int) -> ElementLabel:
    """Return the label of batch chunk *chunk* under *label*.

    Used by schemes that pack many values into one ciphertext (one ciphertext
    per chunk and label); encoded in the element domain with layer ``-1``,
    so it never collides with a user label or an element label.
    """
    return element_label(label, -1, (chunk,))
