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
attributes of a ciphertext, with the checks that both need, and sets up the pairings of a decryption
(:func:`attribute_pairs`).

Labels are hashed to ``Z_q`` by :func:`~pyfe4ai.policy.attributes.attribute_to_zq`, so that a label stands for the
attribute ``j in Z_q`` of the paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy import MSP, Formula, attribute_to_zq, compile_formula, parse, validate_label
from pyfe4ai.policy.attributes import clip
from pyfe4ai.primitives.pairing.dpvs import combine
from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup, multiexp_many

#: The dimension of ``F`` and ``F*``: two coordinates for ``(j, 1)``, one for the share and five that stay zero.
F_DIM = 8
#: The dimension of ``H`` and ``H*``: three coordinates for the mask and the share, one that stays zero.
H_DIM = 4
#: The default of ``max_attributes``, the labels a ciphertext may carry.
DEFAULT_MAX_ATTRIBUTES = 64
#: The most characters of a label that the schemes of this module take. The key-policy schemes have no universe that
#: bounds a label, and ciphertexts and keys come from outside; this bounds what a decryption reads.
MAX_LABEL_CHARS = 1024
#: What a row of a policy may add to its label in the canonical encoding (gate names, quotes, brackets, commas).
_POLICY_MARGIN_PER_ROW = 400


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


def policy_length_limit(max_rows: int) -> int:
    """The most characters that the canonical encoding of a policy of at most *max_rows* rows may have:
    ``max_rows * (MAX_LABEL_CHARS + 400)``."""
    return max_rows * (MAX_LABEL_CHARS + _POLICY_MARGIN_PER_ROW)


def check_policy_length(policy: str, limit: int) -> None:
    """Refuse a policy text of more than *limit* characters (:func:`policy_length_limit`).

    The limit counts the characters of the canonical encoding. That encoding is JSON, which writes ``"`` and ``\\``
    of a label as two characters each, so a policy over labels made mostly of those can be refused although its labels
    fit: on purpose, because the limit bounds the text that a key carries and a decryption reads.

    Raises:
        FEValidationError: For a longer text.
    """
    if len(policy) > limit:
        raise FEValidationError(
            "the policy has {} characters, more than the limit of {} (max_rows * (MAX_LABEL_CHARS + {}))".format(
                len(policy), limit, _POLICY_MARGIN_PER_ROW)
        )


def compile_policy(formula: Formula, order, max_rows: int) -> MSP:
    """The span program of *formula* over ``Z_order``, with at most *max_rows* rows.

    The leaves are counted before anything is built, so a formula made of shared sub-formulas cannot stand for more
    rows than the limit.

    Raises:
        FEValidationError: For a formula that is invalid or has more than *max_rows* leaves.
    """
    return compile_formula(formula, int(order), max_rows=max_rows)


def check_policy_labels(labels: Sequence[str]) -> None:
    """Refuse a policy with a label of more than :data:`MAX_LABEL_CHARS` characters (*labels*: the labels of its rows).

    A ciphertext carries no such label (:func:`held_attributes`, :func:`read_attributes`), so no ciphertext could
    satisfy its row; a key for the policy is refused when it is issued rather than found useless at the decryption.

    Raises:
        FEValidationError: Naming the first label that is too long and the limit.
    """
    for label in labels:
        if len(label) > MAX_LABEL_CHARS:
            raise FEValidationError(
                "the policy has the label {} of {} characters, more than the limit MAX_LABEL_CHARS = {}: "
                "no ciphertext can hold it".format(clip(repr(label)), len(label), MAX_LABEL_CHARS))


def _check_label(label) -> None:
    """*label* is a valid label (:func:`~pyfe4ai.policy.attributes.validate_label`) of at most
    :data:`MAX_LABEL_CHARS` characters; the length is checked first, so that a huge string is not scanned."""
    if type(label) is str and len(label) > MAX_LABEL_CHARS:
        raise FEValidationError("attribute label {} has {} characters, more than the limit MAX_LABEL_CHARS = {}".format(
            clip(repr(label)), len(label), MAX_LABEL_CHARS))
    validate_label(label)


def held_attributes(attributes, max_attributes: int) -> list[str]:
    """The labels of *attributes*, sorted and each once, for a caller's input.

    *attributes* is any collection of valid labels, in any order and with repeats; a ``set`` and a list that names
    one label twice give the same result. The empty collection is allowed: no policy is satisfied by it.

    Raises:
        FEValidationError: For a bare ``str`` or ``bytes``, which would be read one character at a time, for
            something that is not a collection, for an item that is not a valid label
            (:func:`~pyfe4ai.policy.attributes.validate_label`) or has more than :data:`MAX_LABEL_CHARS` characters,
            and for more than *max_attributes* distinct labels.
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
        _check_label(label)
    labels = sorted(set(items))
    if len(labels) > max_attributes:
        raise FEValidationError("{} attributes are more than the limit max_attributes = {}".format(
            len(labels), max_attributes))
    return labels


def read_attributes(value, max_attributes: int) -> list[str]:
    """The labels of the field ``attributes`` of a ciphertext, which comes from outside.

    The field must be the list that :func:`held_attributes` makes: a ``list`` of valid labels of at most
    :data:`MAX_LABEL_CHARS` characters in strictly increasing order, so that a set of attributes has one spelling.
    The length is checked before the labels are.

    Raises:
        FEValidationError: For anything else, and for more than *max_attributes* labels. The error for a list out of
            order names the first two labels that are.
    """
    if type(value) is not list:
        raise FEValidationError("attributes must be a list of labels, got {}".format(type(value).__name__))
    if len(value) > max_attributes:
        raise FEValidationError("attributes holds {} labels, more than the limit max_attributes = {}".format(
            len(value), max_attributes))
    for label in value:
        _check_label(label)
    for earlier, later in zip(value, value[1:]):
        if earlier >= later:
            raise FEValidationError("attributes must be sorted and hold each label once: {} is followed by {}".format(
                clip(repr(earlier)), clip(repr(later))))
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


def attribute_pairs(weights: Mapping[int, int], rows: Sequence[str], attributes: Sequence[str],
                    c_attr: Sequence[list], k_rows: Mapping[int, list], group: PairingGroup,
                    order) -> tuple[list, list]:
    """The pairings of a decryption that pay the rows of a satisfied policy: the two vectors ``(lhs, rhs)`` for
    ``multi_pair``, :data:`F_DIM` pairs for each used row.

    Row ``j`` of the policy, with the coefficient ``w_j``, is paired with the vector of its label in the
    ciphertext, ``e(c_j^w_j, k*_j)``. The coefficient is a power of the ``G1`` side, so that the ``G2`` elements of
    the key go into the pairing as they are, and a coefficient of 1 costs no exponentiation.

    Args:
        weights: The reconstruction coefficients, from the index of a used row to its coefficient
            (:meth:`~pyfe4ai.policy.msp.MSP.reconstruction`).
        rows: The label of every row of the policy (``msp.row_labels``).
        attributes: The sorted labels of the ciphertext; the position of a label is that of its vector in *c_attr*.
        c_attr: One vector of :data:`F_DIM` ``G1`` elements for each of *attributes*.
        k_rows: The vector of :data:`F_DIM` ``G2`` elements of each used row, by the index of the row.
        group: Pairing group instance.
        order: The group order.

    Returns:
        ``len(weights) * F_DIM`` elements of ``G1`` and as many of ``G2``.

    Raises:
        FEValidationError: For a used row whose label is not in *attributes*, and a vector of another length than
            :data:`F_DIM`.
    """
    order = int(order)
    position = {label: index for index, label in enumerate(attributes)}
    lhs, rhs, powered, exponents = [], [], [], []
    for row, weight in weights.items():
        if rows[row] not in position:
            raise FEValidationError("row {} of the policy names the label {}, which the ciphertext does not carry"
                                    .format(row, clip(repr(rows[row]))))
        c, k = c_attr[position[rows[row]]], k_rows[row]
        if len(c) != F_DIM or len(k) != F_DIM:
            raise FEValidationError("the vectors of row {} must have {} elements".format(row, F_DIM))
        weight %= order
        if weight != 1:
            powered.extend(range(len(lhs), len(lhs) + F_DIM))
            exponents.extend([weight] * F_DIM)
        lhs.extend(c)
        rhs.extend(k)
    if powered:
        raised = multiexp_many(([lhs[slot]] for slot in powered), ([weight] for weight in exponents), group)
        for slot, element in zip(powered, raised):
            lhs[slot] = element
    return lhs, rhs
