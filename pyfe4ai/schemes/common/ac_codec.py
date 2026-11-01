"""Reading the untrusted records of the attribute-based pairing schemes.

Public parameters, keys and ciphertexts of these schemes cross trust boundaries
as dicts of strings and lists. The functions here read them: a limit that is a
positive ``int``, a decimal string in its one spelling, a list of the length
that the parameters fix, a list of group elements of the right group in their
one serialised form, and a record view with all its fields. Each raises
``FEValidationError`` naming the field, so that the schemes read their records
alike.
"""

from __future__ import annotations

import re

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.primitives.pairing import bls12_381
from pyfe4ai.primitives.pairing.bls12_381 import MAX_DLOG_BOUND
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup
from pyfe4ai.primitives.pairing.pairing_backend import deserialize_many, serialize_many

_GROUP_NAMES = {G1: "G1", G2: "G2"}

# a decimal string of these objects (bound_x, bound_y, the entries of y) has at most 16 digits, so it stands for
# an integer below 10**16, a little above 2**48; the pattern refuses a longer string before int() reads it
_DECIMAL = re.compile(r"0|-?[1-9][0-9]{0,15}")


def positive_int(value, name: str) -> int:
    # type(value) is int: a bool or a float would be a second spelling of the same limit
    if type(value) is not int or value < 1:
        raise FEValidationError("{} must be a positive int".format(name))
    return value


def check_dlog_limit(eta: int, bound_x: int, bound_y: int) -> None:
    if eta * bound_x * bound_y > MAX_DLOG_BOUND:
        raise FEValidationError(
            "eta * bound_x * bound_y must be at most 2**48, the limit of the discrete log"
        )


def decimal(value, name: str) -> int:
    """The int that *value*, a decimal string in its one spelling, stands for."""
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise FEValidationError("{} must hold decimal strings of integers of at most 16 digits".format(name))
    return int(value)


def check_length(items, length: int, name: str) -> None:
    if type(items) is not list or len(items) != length:
        raise FEValidationError("{} must be a list of {} entries".format(name, length))


def serialize_vec(group: PairingGroup, vec) -> list[str]:
    return [item.decode("utf-8") for item in serialize_many(group, vec)]


def read_elements(group: PairingGroup, items, kind, name: str) -> list:
    """*items* deserialized, all of the group *kind*, or ``FEValidationError`` naming the field *name*."""
    # the type tag before the element: charm reads almost any text as an element of some group, and
    # never returns from some strings tagged G2, which must not reach it from a field of G1 elements
    tag = "{}:".format(kind)
    if type(items) is not list or not all(type(item) is str and item.startswith(tag) for item in items):
        raise FEValidationError("{} must be a list of serialised elements of {}".format(name, _GROUP_NAMES[kind]))
    try:
        elements = deserialize_many(group, items)
    except ValueError as exc:
        raise FEValidationError("{} holds an element that does not deserialize".format(name)) from exc
    # one spelling per element: BLS12-381 has refused any other, charm ignores the bytes after an element
    if not isinstance(group, bls12_381.PairingGroup) and serialize_vec(group, elements) != items:
        raise FEValidationError("{} holds an element that is not in its serialised form".format(name))
    return elements


def read_record(cls, data, name: str):
    """The record view of the dict *data*."""
    if not isinstance(data, dict):
        raise FEValidationError("the {} must be a dict, got {}".format(name, type(data).__name__))
    return cls(data)


def require_fields(record, name: str) -> None:
    for field in record.FIELDS:
        if field not in record:
            raise FEValidationError("the {} has no field {}".format(name, field))
