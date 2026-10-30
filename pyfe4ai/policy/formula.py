"""Monotone access formulas over attribute labels.

A formula is a tree of four immutable node types: :class:`Leaf` (an attribute
label, :mod:`pyfe4ai.policy.attributes`), :class:`And`, :class:`Or` and
:class:`Threshold` (``t`` of its children). The constructors check what they
can see locally (the label, the number of children, ``t``);
:func:`validate_formula` checks what only the whole tree shows (its depth and
the labels under one threshold gate). :func:`evaluate` says whether a set of
labels satisfies a formula.

:meth:`Formula.leaves`, :func:`leaf_count`, :func:`evaluate` and
:func:`validate_formula` walk a formula with an explicit stack, not recursion,
so a formula nested deeper than the interpreter's recursion limit is refused by
``validate_formula`` with ``FEValidationError``, not ``RecursionError``. (The
``==``, ``hash`` and ``repr`` that the dataclasses generate do recurse:
validate a formula before comparing or hashing it.) A sub-formula object that
several gates share is visited once by ``leaf_count``, ``evaluate`` and
``validate_formula``, but ``leaves()`` lists its labels once per occurrence: a
formula of unknown size is counted with ``leaf_count`` before it is listed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.attributes import validate_label

#: The default limit on the rows of a compiled policy matrix.
DEFAULT_MAX_ROWS = 64

#: The greatest depth of a formula: the number of gates on its longest path
#: from the root to a leaf (a lone leaf has depth 0).
MAX_DEPTH = 32


class Formula:
    """A node of a monotone access formula: :class:`Leaf`, :class:`And`,
    :class:`Or` or :class:`Threshold`."""

    def leaves(self) -> list[str]:
        """The labels of the leaves in formula order (left to right), a label
        once per occurrence."""
        out: list[str] = []
        stack: list[Formula] = [self]
        while stack:
            node = stack.pop()
            if isinstance(node, Leaf):
                out.append(node.label)
            else:
                stack.extend(reversed(_children(node)))
        return out


@dataclass(frozen=True)
class Leaf(Formula):
    """An attribute label; satisfied by a key that holds it."""

    label: str

    def __post_init__(self) -> None:
        validate_label(self.label)


def _gate_children(gate: str, children: tuple) -> tuple:
    for child in children:
        if not isinstance(child, Formula):
            raise FEValidationError("{} children must be formulas, got {}".format(gate, type(child).__name__))
    return children


def _at_least_two(gate: str, children: tuple) -> tuple:
    children = _gate_children(gate, children)
    if len(children) < 2:
        raise FEValidationError("{} needs at least two children, got {}".format(gate, len(children)))
    return children


@dataclass(frozen=True, init=False)
class And(Formula):
    """Satisfied when all of its (at least two) children are."""

    children: tuple[Formula, ...]

    def __init__(self, *children: Formula) -> None:
        object.__setattr__(self, "children", _at_least_two("And", children))


@dataclass(frozen=True, init=False)
class Or(Formula):
    """Satisfied when at least one of its (at least two) children is."""

    children: tuple[Formula, ...]

    def __init__(self, *children: Formula) -> None:
        object.__setattr__(self, "children", _at_least_two("Or", children))


@dataclass(frozen=True, init=False)
class Threshold(Formula):
    """Satisfied when at least ``t`` of its children are, ``1 <= t <= len(children)``
    (a single child is allowed)."""

    t: int
    children: tuple[Formula, ...]

    def __init__(self, t: int, *children: Formula) -> None:
        # type(t) is int: a bool or an int subclass would be a second spelling of the same threshold
        if type(t) is not int:
            raise FEValidationError("Threshold t must be an int, got {}".format(type(t).__name__))
        children = _gate_children("Threshold", children)
        if not 1 <= t <= len(children):
            raise FEValidationError(
                "Threshold needs 1 <= t <= the number of children, got t={} with {} children".format(t, len(children))
            )
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "children", children)


def _children(node) -> tuple:
    """The children of *node* (none for a leaf), or ``FEValidationError`` if
    *node* is not a node of a formula."""
    if isinstance(node, Leaf):
        return ()
    if isinstance(node, (And, Or, Threshold)):
        return node.children
    raise FEValidationError("expected a formula, got {}".format(type(node).__name__))


def _postorder(root) -> Iterator[Formula]:
    """The distinct node objects under *root* (itself included), each after all
    its children; a node object that occurs more than once is yielded once."""
    done: set[int] = set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in done:
            continue
        if expanded:
            done.add(id(node))
            yield node
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(_children(node)) if id(child) not in done)


def leaf_count(formula: Formula) -> int:
    """The number of leaves of *formula* written out as a tree, a sub-formula
    object that several gates share counted once per occurrence.

    It does not list the leaves and does not recurse: each distinct node
    object is visited once, so a formula that stands for 2**32 leaves through
    32 nested gates of shared children answers at once. Call it before
    :meth:`Formula.leaves` on a formula whose size is not known.
    """
    count: dict[int, int] = {}
    for node in _postorder(formula):
        count[id(node)] = 1 if isinstance(node, Leaf) else sum(count[id(child)] for child in node.children)
    return count[id(formula)]


def held_labels(labels: Iterable[str]) -> frozenset[str]:
    """The attribute *labels* a key holds, as a set: any iterable of hashable
    labels, but not a single ``str`` or ``bytes`` (which would be read as a set
    of one-character labels). Anything else raises ``FEValidationError``."""
    if isinstance(labels, (str, bytes)):
        raise FEValidationError("labels must be an iterable of attribute labels, not a single {}".format(
            type(labels).__name__))
    try:
        return frozenset(labels)
    except TypeError as exc:
        raise FEValidationError("labels must be an iterable of attribute labels") from exc


def evaluate(formula: Formula, labels: Iterable[str]) -> bool:
    """Whether the set of attribute *labels* satisfies *formula*.

    Labels that no leaf names are ignored, so adding labels never turns a
    satisfied formula into an unsatisfied one. *labels* is any iterable of
    labels, not a single ``str`` (:func:`held_labels`).
    """
    held = held_labels(labels)
    value: dict[int, bool] = {}
    for node in _postorder(formula):
        if isinstance(node, Leaf):
            satisfied = node.label in held
        else:
            met = [value[id(child)] for child in node.children]
            if isinstance(node, And):
                satisfied = all(met)
            elif isinstance(node, Or):
                satisfied = any(met)
            else:
                satisfied = sum(met) >= node.t
        value[id(node)] = satisfied
    return value[id(formula)]


def validate_formula(formula: Formula) -> None:
    """Check what only the whole tree shows, or raise ``FEValidationError``:

    * its depth is at most :data:`MAX_DEPTH`;
    * no two children of one :class:`Threshold` gate have a leaf label in
      common, at any depth inside those children (the message names the
      label). A label may repeat under :class:`And` and :class:`Or`, and
      inside one child of a threshold gate.
    """
    height: dict[int, int] = {}
    labels: dict[int, frozenset] = {}
    for node in _postorder(formula):
        key = id(node)
        if isinstance(node, Leaf):
            height[key] = 0
            labels[key] = frozenset((node.label,))
            continue
        children = [id(child) for child in node.children]
        height[key] = 1 + max(height[child] for child in children)
        if height[key] > MAX_DEPTH:
            raise FEValidationError("formula depth exceeds the limit of {} gates on a path".format(MAX_DEPTH))
        if isinstance(node, Threshold):
            seen: set[str] = set()
            for child in children:
                shared = seen.intersection(labels[child])
                if shared:
                    raise FEValidationError(
                        "the children of a Threshold gate share the leaf label {!r}; "
                        "they must use distinct labels".format(min(shared))
                    )
                seen.update(labels[child])
            labels[key] = frozenset(seen)
        else:
            labels[key] = frozenset().union(*(labels[child] for child in children))
