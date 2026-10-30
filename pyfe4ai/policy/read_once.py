"""Read-once copies of the attributes of a compiled policy.

A label may sit on several rows of a span program (:class:`~pyfe4ai.policy.msp.MSP`),
once for each leaf that carries it. :func:`read_once_rows` gives every such row
its own copy of the label, numbered from 1 in the order of the rows, so that no
(label, copy) pair is used by two rows. A scheme that holds ``max_copies``
copies of every label in its public parameters (:class:`~pyfe4ai.policy.attributes.Universe`)
can then serve any policy in which no label has more than that many leaves.
Pure Python: the standard library, :mod:`pyfe4ai.core.exceptions` and
:mod:`pyfe4ai.policy.msp` only.
"""

from __future__ import annotations

from collections import Counter

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.attributes import clip
from pyfe4ai.policy.msp import MSP


def read_once_rows(msp: MSP, max_copies: int) -> tuple[tuple[str, int], ...]:
    """The (label, copy) pair of every row of *msp*, in row order.

    The copies of one label are numbered 1, 2, ... in the order of its rows.
    ``FEValidationError`` if *msp* is not an :class:`~pyfe4ai.policy.msp.MSP`,
    if *max_copies* is not an ``int`` of at least 1, or if some label is on more
    than *max_copies* rows (the message names the first such label in row order
    and the limit).
    """
    if not isinstance(msp, MSP):
        raise FEValidationError("msp must be an MSP, got {}".format(type(msp).__name__))
    # type(max_copies) is int: a bool or an int subclass would be a second spelling of the same limit
    if type(max_copies) is not int:
        raise FEValidationError("max_copies must be an int, got {}".format(type(max_copies).__name__))
    if max_copies < 1:
        raise FEValidationError("max_copies must be at least 1, got {}".format(max_copies))
    total = Counter(msp.row_labels)
    for label in msp.row_labels:
        if total[label] > max_copies:
            raise FEValidationError(
                "attribute label {} is on {} rows of the policy, more than the limit of {} copies "
                "per label".format(clip(repr(label)), total[label], max_copies)
            )
    used: Counter = Counter()
    rows = []
    for label in msp.row_labels:
        used[label] += 1
        rows.append((label, used[label]))
    return tuple(rows)
