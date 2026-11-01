"""The pieces that the key-policy attribute-based schemes of NPP22 share.

| "Multi-Client Functional Encryption with Fine-Grained Access Control"
| By Ky Nguyen, Duong Hieu Phan and David Pointcheval, ASIACRYPT 2022
| (ePrint 2022/215) -- NPP22 Sec. 4 Fig. 1

The schemes use two pairs of dual bases of a dual pairing vector space: ``(F, F*)`` of dimension :data:`F_DIM` for the
attributes and ``(H, H*)`` of dimension :data:`H_DIM` for the inner-product mask. A policy sits in a key as one vector
of ``F*`` per row of its span program, ``(pi j, pi, share z, 0, ..., 0)_F*`` for the label that hashes to ``j``
(:func:`key_row`); a set of attributes sits in a ciphertext as one vector of ``F`` per label,
``(sigma, -j sigma, psi, 0, ..., 0)_F`` (:func:`attribute_vector`). Their pairing is
``sigma pi (j' - j) + psi share z`` in the exponent, which is ``psi share z`` for a row of the label ``j`` and
a masked value for any other. The module also reads the two inputs of the schemes, the policy of a key and the
attributes of a ciphertext, with the checks that both need.

Labels are hashed to ``Z_q`` by :func:`~pyfe4ai.policy.attributes.attribute_to_zq`, so that a label stands for the
attribute ``j in Z_q`` of the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy import MSP, Formula, attribute_to_zq, compile_formula, parse, validate_label
from pyfe4ai.primitives.pairing.dpvs import combine
from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup

#: The dimension of ``F`` and ``F*``: two coordinates for ``(j, 1)``, one for the share and five that stay zero.
F_DIM = 8
#: The dimension of ``H`` and ``H*``: three coordinates for the mask and the share, one that stays zero.
H_DIM = 4
#: The default of ``max_attributes``, the labels a ciphertext may carry.
DEFAULT_MAX_ATTRIBUTES = 64


def read_policy(policy) -> Formula:
    """The formula of *policy*: a :class:`~pyfe4ai.policy.formula.Formula` as it is, a ``str`` through
    :func:`~pyfe4ai.policy.encoding.parse`.

    Raises:
        FEValidationError: For any other type, and for a text that does not parse.
    """
    if isinstance(policy, str):
        return parse(policy)
    if not isinstance(policy, Formula):
        raise FEValidationError("the policy must be a Formula or its text, got {}".format(type(policy).__name__))
    return policy


def compile_policy(formula: Formula, order, max_rows: int) -> MSP:
    """The span program of *formula* over ``Z_order``, with at most *max_rows* rows.

    The leaves are counted before anything is built, so a formula made of shared sub-formulas cannot stand for more
    rows than the limit.

    Raises:
        FEValidationError: For a formula that is invalid or has more than *max_rows* leaves.
    """
    return compile_formula(formula, int(order), max_rows=max_rows)


def held_attributes(attributes, max_attributes: int) -> list[str]:
    """The labels of *attributes*, sorted and each once, for a caller's input.

    *attributes* is any collection of valid labels, in any order and with repeats; a ``set`` and a list that names
    one label twice give the same result. The empty collection is allowed: no policy is satisfied by it.

    Raises:
        FEValidationError: For a bare ``str`` or ``bytes``, which would be read one character at a time, for
            something that is not a collection, for an item that is not a valid label
            (:func:`~pyfe4ai.policy.attributes.validate_label`), and for more than *max_attributes* distinct labels.
    """
    if isinstance(attributes, (str, bytes, bytearray)):
        raise FEValidationError("attributes must be a collection of labels, not a bare {}".format(
            type(attributes).__name__))
    try:
        items = list(attributes)
    except TypeError as exc:
        raise FEValidationError("attributes must be a collection of labels, got {}".format(
            type(attributes).__name__)) from exc
    for label in items:
        validate_label(label)
    labels = sorted(set(items))
    if len(labels) > max_attributes:
        raise FEValidationError("{} attributes are more than the limit max_attributes = {}".format(
            len(labels), max_attributes))
    return labels


def read_attributes(value, max_attributes: int) -> list[str]:
    """The labels of the field ``attributes`` of a ciphertext, which comes from outside.

    The field must be the list that :func:`held_attributes` makes: a ``list`` of valid labels in strictly increasing
    order, so that a set of attributes has one spelling. The length is checked before the labels are.

    Raises:
        FEValidationError: For anything else, and for more than *max_attributes* labels.
    """
    if type(value) is not list:
        raise FEValidationError("attributes must be a list of labels, got {}".format(type(value).__name__))
    if len(value) > max_attributes:
        raise FEValidationError("attributes holds {} labels, more than the limit max_attributes = {}".format(
            len(value), max_attributes))
    for label in value:
        validate_label(label)
    if any(earlier >= later for earlier, later in zip(value, value[1:])):
        raise FEValidationError("attributes must be sorted and hold each label once")
    return list(value)


def key_row(f_star: Sequence[list], label: str, pi, share_z, order, group: PairingGroup) -> list:
    """The vector ``(pi j, pi, share_z, 0, ..., 0)_F*`` of a row of a policy, for the label that hashes to ``j``.

    Args:
        f_star: The first three vectors of ``F*``, each of :data:`F_DIM` elements of ``G2``.
        label: The attribute label of the row.
        pi: The randomness of the row.
        share_z: The share of the row times ``z``, reduced modulo *order* here.
        order: The group order.
        group: Pairing group instance.

    Returns:
        :data:`F_DIM` elements of ``G2``.
    """
    order = int(order)
    j = attribute_to_zq(label, order)
    return combine(f_star, [pi * j % order, pi % order, share_z % order], group)


def attribute_vector(f: Sequence[list], label: str, sigma, psi, order, group: PairingGroup) -> list:
    """The vector ``(sigma, -j sigma, psi, 0, ..., 0)_F`` of an attribute, for the label that hashes to ``j``.

    Args:
        f: The first three vectors of ``F``, each of :data:`F_DIM` elements of ``G1``.
        label: The attribute label.
        sigma: The randomness of the attribute.
        psi: The randomness of the ciphertext, the same for all its attributes.
        order: The group order.
        group: Pairing group instance.

    Returns:
        :data:`F_DIM` elements of ``G1``.
    """
    order = int(order)
    j = attribute_to_zq(label, order)
    return combine(f, [sigma % order, -j * sigma % order, psi % order], group)
