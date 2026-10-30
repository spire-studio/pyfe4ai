"""Span programs whose reconstruction coefficients are all 0 or 1.

A threshold gate is shared by Shamir's polynomial, and reconstructing from it
multiplies shares by Lagrange coefficients, which are field elements and need
inverses (:mod:`pyfe4ai.policy.msp`). A formula with only ``And`` and ``Or``
gates needs neither: the shares of the rows an ``Or`` gate picks are copies of
one value and those of an ``And`` gate add up, so a qualified set reconstructs by
adding the shares of some of its rows, each with the coefficient 1, over any
modulus, a composite one included.

:func:`threshold_to_cnf` rewrites every ``Threshold(t)`` gate of ``m`` children
to conjunctive normal form: an ``And`` of one ``Or`` for each set of ``t - 1``
children, the ``Or`` holding the other ``m - t + 1`` children. Any ``t - 1``
satisfied children miss the clause that leaves out exactly them, and any ``t``
satisfied children meet every clause, so the ``And`` holds exactly when ``t``
children do. The price is the size. The leaves of the rewrite, with a child that
several clauses share counted once per clause, number

* ``size(Leaf) = 1``, and for ``And`` and ``Or`` the sum over the children;
* for ``Threshold(t)`` of ``m`` children, the same sum when ``t == 1`` or
  ``t == m``, and otherwise the sum times ``comb(m - 1, t - 1)``, because every
  child sits in ``comb(m - 1, t - 1)`` clauses (for leaf children this is
  ``comb(m, t - 1) * (m - t + 1)`` leaves).

The size is computed first, from the original formula, and a rewrite that would
pass the row limit is refused before any node of it is built.
:func:`compile_zero_one` hands the rewrite to
:func:`~pyfe4ai.policy.msp.compile_formula`. Pure Python: the standard library,
:mod:`pyfe4ai.core.exceptions` and :mod:`pyfe4ai.policy` only.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.formula import (
    DEFAULT_MAX_ROWS,
    And,
    Formula,
    Leaf,
    Or,
    Threshold,
    _postorder,
    validate_formula,
)
from pyfe4ai.policy.msp import MSP, compile_formula


def _count_text(count: int) -> str:
    """*count* for an error message. The default limit on the digits of ``str(int)``
    would turn a gate with thousands of children into a ``ValueError``, so a count
    beyond 256 bits is given as a power of two it reaches."""
    if count.bit_length() <= 256:
        return str(count)
    return "at least 2**{}".format(count.bit_length() - 1)


def _expanded_rows(formula: Formula) -> int:
    """The number of leaves of ``threshold_to_cnf(formula)`` written out as a
    tree, from the formula alone (see the module docstring). Each distinct node
    object is visited once, so a formula whose shared sub-formulas stand for
    2**32 leaves answers at once."""
    rows: dict[int, int] = {}
    for node in _postorder(formula):
        if isinstance(node, Leaf):
            rows[id(node)] = 1
            continue
        total = sum(rows[id(child)] for child in node.children)
        if isinstance(node, Threshold) and 1 < node.t < len(node.children):
            total *= comb(len(node.children) - 1, node.t - 1)
        rows[id(node)] = total
    return rows[id(formula)]


def _cnf_gate(t: int, children: list[Formula]) -> Formula:
    """``Threshold(t, *children)`` for children that are already free of threshold gates."""
    if len(children) == 1:
        return children[0]
    if t == 1:
        return Or(*children)
    if t == len(children):
        return And(*children)
    # 1 < t < m: every clause keeps m - t + 1 >= 2 children, and there are comb(m, t - 1) >= 3 clauses
    clauses = []
    for left_out in combinations(range(len(children)), t - 1):
        skipped = set(left_out)
        clauses.append(Or(*(child for index, child in enumerate(children) if index not in skipped)))
    return And(*clauses)


def threshold_to_cnf(formula: Formula, *, max_rows: int = DEFAULT_MAX_ROWS) -> Formula:
    """Rewrite every threshold gate of *formula* to ``And`` and ``Or`` gates.

    The result is satisfied by exactly the sets of labels that satisfy
    *formula*, has no :class:`~pyfe4ai.policy.formula.Threshold` gate, and
    keeps the formula's other gates and its leaves. ``Threshold(1)`` of ``m``
    children becomes ``Or``, ``Threshold(m)`` becomes ``And``, and
    ``Threshold(t)`` with ``1 < t < m`` becomes the ``And`` of the ``Or`` of
    the children that remain when ``t - 1`` of them are left out, for each such
    choice, in the order of :func:`itertools.combinations`. The clauses share
    the child objects, and a sub-formula object that several gates share is
    rewritten once, so the result shares it too.

    *max_rows* bounds the number of leaves of the result written out
    (:data:`~pyfe4ai.policy.formula.DEFAULT_MAX_ROWS` by default), and the
    bound is checked before anything is built. Raises ``FEValidationError`` for
    an invalid argument, for a formula that does not pass
    :func:`~pyfe4ai.policy.formula.validate_formula` (the rewrite drops the
    threshold gates and with them the rule on their children's labels, so it is
    checked here), for an expansion of more than *max_rows* leaves (the message
    gives the number and the limit), and for a result nested deeper than
    :data:`~pyfe4ai.policy.formula.MAX_DEPTH`.
    """
    # type(...) is int: a bool is not a limit here, and neither is an int subclass or a float
    if type(max_rows) is not int or max_rows < 1:
        raise FEValidationError("max_rows must be a positive int, got {!r}".format(max_rows))
    if not isinstance(formula, Formula):
        raise FEValidationError("formula must be a Formula, got {}".format(type(formula).__name__))
    validate_formula(formula)
    rows = _expanded_rows(formula)
    if rows > max_rows:
        raise FEValidationError(
            "the formula has {} leaves after its threshold gates are rewritten to conjunctive normal form, "
            "more than the limit of {} rows".format(_count_text(rows), max_rows)
        )
    rewritten: dict[int, Formula] = {}
    for node in _postorder(formula):
        if isinstance(node, Leaf):
            rewritten[id(node)] = node
            continue
        children = [rewritten[id(child)] for child in node.children]
        if isinstance(node, Threshold):
            rewritten[id(node)] = _cnf_gate(node.t, children)
        elif isinstance(node, And):
            rewritten[id(node)] = And(*children)
        else:
            rewritten[id(node)] = Or(*children)
    result = rewritten[id(formula)]
    validate_formula(result)                        # a gate becomes two levels: the rewrite may be too deep
    return result


def compile_zero_one(formula: Formula, modulus: int, *, max_rows: int = DEFAULT_MAX_ROWS) -> MSP:
    """Compile *formula* to a span program whose reconstruction coefficients are all 1.

    The program is that of ``threshold_to_cnf(formula)``
    (:func:`~pyfe4ai.policy.msp.compile_formula`), so its ``formula`` is the
    rewritten formula, its rows are the leaves of that formula and
    ``row_labels == msp.formula.leaves()``, in the order of the rewrite and not
    of *formula*. Every matrix entry is 0, 1 or ``modulus - 1``, and
    :meth:`~pyfe4ai.policy.msp.MSP.reconstruction` returns the coefficient 1 on
    each row it uses. Without threshold gates in the compiled formula no inverse
    is needed, so *modulus* may be composite, such as the modulus of a ring; the
    checks of *modulus* are those of ``compile_formula``. *max_rows* bounds the
    rows of the result, as in :func:`threshold_to_cnf`.
    """
    return compile_formula(threshold_to_cnf(formula, max_rows=max_rows), modulus, max_rows=max_rows)
