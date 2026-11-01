"""Reading the untrusted records of the attribute-based pairing schemes.

Public parameters, keys and ciphertexts of these schemes cross trust boundaries
as dicts of strings and lists. The functions here read them: a limit that is a
positive ``int``, a decimal string in its one spelling, a list of the length
that the parameters fix, a list of group elements of the right group in their
one serialised form, and a record view with all its fields. Each raises
``FEValidationError`` naming the field, so that the schemes read their records
alike. The module also holds what the constructors of the crypto objects have in
common: the copy of the public parameters, the pairing group they name and the
version markers they carry.
"""

from __future__ import annotations

import copy
import re

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.attributes import clip
from pyfe4ai.primitives.pairing import bls12_381
from pyfe4ai.primitives.pairing.bls12_381 import MAX_DLOG_BOUND
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup
from pyfe4ai.primitives.pairing.pairing_backend import check_asymmetric, deserialize_many, serialize_many

_GROUP_NAMES = {G1: "G1", G2: "G2"}

# a decimal string of these objects (bound_x, bound_y, the entries of y) has at most 16 digits, so it stands for
# an integer below 10**16, a little above 2**48; the pattern refuses a longer string before int() reads it
_DECIMAL = re.compile(r"0|-?[1-9][0-9]{0,15}")


def positive_int(value, name: str) -> int:
    # type(value) is int: a bool or a float would be a second spelling of the same limit
    if type(value) is not int or value < 1:
        raise FEValidationError("{} must be a positive int".format(name))
    return value


def check_dlog_limit(eta: int, bound_x: int, bound_y: int, first: str = "eta") -> None:
    """Refuse ``eta * bound_x * bound_y`` above ``2**48``, the limit of the discrete log.

    *first* names the first factor in the message: ``"eta"``, or ``"n"`` for a scheme whose clients encrypt one
    scalar each (the number of clients is then passed as *eta*).
    """
    if eta * bound_x * bound_y > MAX_DLOG_BOUND:
        raise FEValidationError(
            "{} * bound_x * bound_y must be at most 2**48, the limit of the discrete log".format(first)
        )


def decimal(value, name: str) -> int:
    """The int that *value*, a decimal string in its one spelling, stands for."""
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise FEValidationError("{} must hold decimal strings of integers of at most 16 digits".format(name))
    return int(value)


_SCALAR = re.compile(r"0|[1-9][0-9]*")


def scalar(value, name: str, order) -> int:
    """The int in ``[0, order)`` that *value*, a decimal string in its one spelling, stands for: a secret exponent.

    The length is checked before the string is converted, so that a long string costs nothing.
    """
    if (type(value) is not str or len(value) > len(str(order)) or _SCALAR.fullmatch(value) is None
            or int(value) >= order):
        raise FEValidationError("{} must hold decimal strings of integers below the group order".format(name))
    return int(value)


def check_length(items, length: int, name: str) -> None:
    if type(items) is not list or len(items) != length:
        raise FEValidationError("{} must be a list of {} entries".format(name, length))


def serialize_vec(group: PairingGroup, vec) -> list[str]:
    return [item.decode("utf-8") for item in serialize_many(group, vec)]


def has_tags(items, kind) -> bool:
    """Whether *items* is a list of strings that all start with the type tag of the group *kind*."""
    tag = "{}:".format(kind)
    return type(items) is list and all(type(item) is str and item.startswith(tag) for item in items)


def check_tags(items, kind, name: str) -> None:
    """``FEValidationError`` naming the field *name* unless :func:`has_tags` holds for *items*."""
    if not has_tags(items, kind):
        raise FEValidationError("{} must be a list of serialised elements of {}".format(name, _GROUP_NAMES[kind]))


def read_elements(group: PairingGroup, items, kind, name: str) -> list:
    """*items* deserialized, all of the group *kind*, or ``FEValidationError`` naming the field *name*."""
    # the type tag before the element: charm reads almost any text as an element of some group, and
    # never returns from some strings tagged G2, which must not reach it from a field of G1 elements
    check_tags(items, kind, name)
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


def copy_parameters(pp, name: str = "public parameters"):
    """A deep copy of *pp*, so that what a constructor checks stays what it uses, whatever the caller's dict does.

    Raises:
        FEValidationError: If *pp* cannot be copied (``deepcopy`` raises what the ``__reduce_ex__`` of any object
            raises). *name* says what *pp* is, in the message.
    """
    try:
        return copy.deepcopy(pp)
    except Exception as exc:
        raise FEValidationError("{} must be JSON data: {}".format(name, exc)) from exc


def load_group(name) -> PairingGroup:
    """The asymmetric pairing group that the parameter ``pairing_group_param`` = *name* names.

    Raises:
        FEValidationError: For a name that is no pairing group, and for a symmetric group.
        ImportError: If no backend is installed for a group that may exist: the installation, not the parameters.
    """
    try:
        group = PairingGroup(name)
    except ImportError:
        raise
    except Exception as exc:    # charm raises AssertionError or its own Error for a name it does not know
        raise FEValidationError("pairing_group_param {} is not a pairing group".format(clip(repr(name)))) from exc
    check_asymmetric(group)
    return group


def check_marker(value, expected: str, name: str) -> None:
    """Refuse a version marker other than the one of this version.

    Args:
        value: The marker in the record.
        expected: The marker of this version.
        name: Whose marker it is, for the message: ``"the policy_compiler of the ciphertext"``.

    Raises:
        FEValidationError: For any other value, of any type (an array is not compared with a string: the comparison
            would be an array, and ``if`` on it a ``ValueError``).
    """
    if type(value) is not str or value != expected:
        raise FEValidationError("{} is {}, not this version's {!r}".format(name, clip(repr(value)), expected))
