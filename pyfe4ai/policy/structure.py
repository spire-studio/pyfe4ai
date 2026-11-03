"""Which sets of holders a policy authorizes, and the Q2 check.

A policy (:mod:`pyfe4ai.policy.formula`) is satisfied by a set of attribute
labels. Given a short list of holders, each holding some labels, a *set of
holders* is a bit set: bit ``i`` stands for holder ``i`` of the list, so the
empty set is ``0`` and the set of all ``n`` holders is ``2**n - 1``. The set is
*qualified* when the labels its members hold together satisfy the formula, and
*unqualified* otherwise. Adding a holder adds labels, so the qualified sets are
closed upwards: they are described by their minimal members, and the
unqualified sets, which are closed downwards, by their maximal members.
:func:`qualified_families` lists both, by evaluating the formula on all
``2**n`` sets of holders, which is why the list is limited to
:data:`MAX_HOLDERS` holders. The empty set is an ordinary set of holders: it is
unqualified, and it is a maximal unqualified set when every single holder is
qualified.

The Q2 condition of Hirt and Maurer asks of a family of sets (here the
unqualified ones, those a policy does not authorize) that no two of its members
together cover all holders. :func:`q2_violation` finds such a pair. It looks at
every pair of members, a member with itself included: a single unqualified set
that is the set of all holders is a violation. Checking the maximal unqualified
sets is enough, since a pair that covers all holders can be enlarged to a pair
of maximal ones. Pure Python: the standard library, :mod:`pyfe4ai.core.exceptions`
and :mod:`pyfe4ai.policy.formula` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.formula import Formula, evaluate, held_labels, validate_formula

#: The greatest number of holders, since the families are found by trying all ``2**n`` sets of holders.
MAX_HOLDERS = 8


def _holder_labels(holders: Sequence[Iterable[str]]) -> list[frozenset[str]]:
    """The labels of each holder as a set, or ``FEValidationError`` if *holders* is not a sequence of one to
    :data:`MAX_HOLDERS` collections of labels (a holder that holds nothing is an empty collection)."""
    # a bare str or bytes is a sequence of one-character holders; any other collection has no order to number by
    if isinstance(holders, (str, bytes, bytearray)) or not isinstance(holders, Sequence):
        raise FEValidationError("holders must be a sequence of collections of labels, got {}".format(
            type(holders).__name__))
    if not holders:
        raise FEValidationError("there must be at least one holder")
    if len(holders) > MAX_HOLDERS:
        raise FEValidationError("at most {} holders are supported, got {}".format(MAX_HOLDERS, len(holders)))
    labels = []
    for index, held in enumerate(holders):
        try:
            labels.append(held_labels(held))
        except FEValidationError as exc:
            raise FEValidationError("holder {}: {}".format(index, exc)) from exc
    return labels


def qualified_families(formula: Formula, holders: Sequence[Iterable[str]]) -> tuple[list[int], list[int]]:
    """The minimal qualified and the maximal unqualified sets of *holders* under *formula*.

    *holders* is a sequence of one to :data:`MAX_HOLDERS` collections of
    attribute labels, holder ``i`` holding the labels of the ``i``-th one (a
    collection may be empty, and holders may hold labels in common). A set of
    holders is the ``int`` whose bit ``i`` is set when holder ``i`` is in it.
    It is qualified when :func:`~pyfe4ai.policy.formula.evaluate` is true on the
    union of the labels of its members.

    Returns ``(minimal, maximal)``, each a list of sets in ascending order:
    the qualified sets from which no holder can be removed without losing the
    qualification, and the unqualified sets to which no holder can be added
    without gaining it. ``minimal`` is empty when no set of holders is
    qualified, and then ``maximal`` is the set of all holders alone.

    *formula* is checked first (:func:`~pyfe4ai.policy.formula.validate_formula`),
    then *holders*; ``FEValidationError`` for either, with the number of the
    offending holder named.
    """
    validate_formula(formula)
    labels = _holder_labels(holders)
    n = len(labels)
    size = 1 << n
    # the labels of each set of holders, built from the set without its lowest member
    union = [frozenset()] * size
    for subset in range(1, size):
        lowest = subset & -subset
        union[subset] = union[subset ^ lowest] | labels[lowest.bit_length() - 1]
    qualified = [evaluate(formula, union[subset]) for subset in range(size)]
    bits = [1 << i for i in range(n)]
    # qualification only grows with the set, so one holder more or less is enough to look at
    minimal = [
        subset for subset in range(size)
        if qualified[subset] and not any(qualified[subset ^ bit] for bit in bits if subset & bit)
    ]
    maximal = [
        subset for subset in range(size)
        if not qualified[subset] and all(qualified[subset | bit] for bit in bits if not subset & bit)
    ]
    return minimal, maximal


def q2_violation(unqualified: Sequence[int], n: int) -> tuple[int, int] | None:
    """Two of the *unqualified* sets of *n* holders that together cover all of them, or ``None``.

    *unqualified* is a sequence of sets of holders (``int`` bit sets, as
    :func:`qualified_families` gives them, from ``0`` to ``2**n - 1``); *n* is
    an ``int`` from 1 to :data:`MAX_HOLDERS`. Every pair is tried, a set with
    itself included, so a single set equal to the set of all holders is
    reported as ``(full, full)``. The answer is the first pair in the order of
    the sequence: ``(a, b)`` with ``a`` at an index not after that of ``b`` and
    ``a | b`` equal to ``2**n - 1``. A sequence without sets has no violation.
    ``FEValidationError`` if *n* or a set is not a plain ``int`` in its range,
    or if *unqualified* is not a sequence (a ``str`` or ``bytes`` is not).
    """
    # type(n) is int: a bool or an int subclass would be a second spelling of the same count
    if type(n) is not int:
        raise FEValidationError("n must be an int, got {}".format(type(n).__name__))
    if not 1 <= n <= MAX_HOLDERS:
        raise FEValidationError("n must be between 1 and {}, got {}".format(MAX_HOLDERS, n))
    if isinstance(unqualified, (str, bytes, bytearray)) or not isinstance(unqualified, Sequence):
        raise FEValidationError("unqualified must be a sequence of sets of holders, got {}".format(
            type(unqualified).__name__))
    full = (1 << n) - 1
    for index, subset in enumerate(unqualified):
        if type(subset) is not int:
            raise FEValidationError("unqualified set {} must be an int, got {}".format(index, type(subset).__name__))
        if not 0 <= subset <= full:
            raise FEValidationError(
                "unqualified set {} is {}, which is not a set of {} holders (0 to {})".format(index, subset, n, full)
            )
    sets = list(unqualified)
    for index, first in enumerate(sets):
        for second in sets[index:]:
            if first | second == full:
                return first, second
    return None
