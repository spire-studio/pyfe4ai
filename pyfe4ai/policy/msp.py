"""Monotone span programs: access formulas compiled to matrices over ``Z_modulus``.

:func:`compile_formula` turns a formula (:mod:`pyfe4ai.policy.formula`) into an
:class:`MSP`: a matrix with one row per leaf, in formula order, labelled with
the leaf's attribute. A set of labels is *qualified* exactly when the formula
holds, and then the rows its labels select combine, with the coefficients of
:meth:`MSP.reconstruction`, to ``e_1 = (1, 0, ..., 0)``. A secret ``s`` is
dealt as ``M * (s, r_2, ..., r_d)`` with random ``r_i`` (:meth:`MSP.shares`);
the shares of a qualified set, combined with the reconstruction coefficients
``w``, give ``s`` (``sum_j w_j share_j = s``), and for any other set ``e_1`` is
not in the span of its rows (over a prime modulus), so its shares say nothing
about ``s``.

The compiler is nested Shamir sharing at threshold gates with the Lewko-Waters
rules at ``And`` and ``Or`` gates. Every sub-formula compiles to its rows and a
*target column*, the column whose value is the secret that the rows of that
sub-formula share; a gate relates the targets of its children to its own:

* ``Or``: the children share one secret, so their target columns are merged;
* ``And``: the children's secrets are additive shares, ``s = s_1 + ... + s_m``,
  the last ``m - 1`` of them fresh random columns;
* ``Threshold(t)``: the secrets of the ``m`` children are the values at
  ``2, ..., m + 1`` of a random polynomial of degree ``t - 1`` whose value at
  0 is the gate's secret (``t`` fresh columns, one per coefficient).

Column ids are handed out from 1 in the order of the recursion: the children
from left to right, then the gate's own columns. The published matrix keeps
the target column of the whole formula first, then every other column that
has a nonzero entry, in the order of the ids. All entries are Python ints in
``[0, modulus)``. Pure Python: the standard library, :mod:`pyfe4ai.core.exceptions`
and :mod:`pyfe4ai.policy.formula` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import gcd

from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.policy.formula import (
    DEFAULT_MAX_ROWS,
    And,
    Formula,
    Leaf,
    Or,
    Threshold,
    held_labels,
    leaf_count,
    validate_formula,
)

#: A row while the matrix is built: column id -> nonzero entry modulo the modulus.
_SparseRow = dict[int, int]


def _is_int(value: object) -> bool:
    # type(...) is int: a bool is not an integer here, and an int subclass or a float is not a Python int
    return type(value) is int


@dataclass(frozen=True)
class MSP:
    """A monotone span program ``(matrix, row_labels, modulus)`` of *formula*.

    ``matrix[i]`` is row ``i``, the row of the ``i``-th leaf of the formula,
    and ``row_labels[i]`` is that leaf's attribute label. Make one with
    :func:`compile_formula`: the constructor does not check the pieces against
    each other (:meth:`reconstruction` does check the formula and its own result).
    """

    matrix: tuple[tuple[int, ...], ...]
    row_labels: tuple[str, ...]
    modulus: int
    formula: Formula

    @property
    def num_rows(self) -> int:
        return len(self.matrix)

    @property
    def num_cols(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    def shares(self, secret: int, randomness: Sequence[int]) -> list[int]:
        """The shares ``M * (secret, *randomness)`` modulo the modulus, one per
        row. *randomness* holds ``num_cols - 1`` ints."""
        if not _is_int(secret):
            raise FEValidationError("secret must be an int, got {}".format(type(secret).__name__))
        if not isinstance(randomness, Sequence) or isinstance(randomness, (str, bytes, bytearray)):
            raise FEValidationError("randomness must be a sequence of ints, got {}".format(type(randomness).__name__))
        if len(randomness) != self.num_cols - 1:
            raise FEValidationError("randomness needs {} values (one per column after the first), got {}".format(
                self.num_cols - 1, len(randomness)))
        if not all(_is_int(value) for value in randomness):
            raise FEValidationError("randomness must hold ints only")
        vector = (secret, *randomness)
        return [sum(entry * value for entry, value in zip(row, vector)) % self.modulus for row in self.matrix]

    def reconstruction(self, labels: Iterable[str]) -> dict[int, int] | None:
        """The coefficients that reconstruct the secret from the rows of
        *labels*: a dict from row index to a nonzero coefficient modulo the
        modulus, with ``sum(w * matrix[row]) = e_1``, or ``None`` when the
        labels do not satisfy the formula.

        *labels* is any iterable of attribute labels, not a single ``str``;
        labels that no leaf names are ignored. The walk follows the formula:
        an ``Or`` gate uses its first satisfied child, an ``And`` gate all of
        its children, a ``Threshold`` gate its first ``t`` satisfied children
        with their Lagrange coefficients at 0.

        The walk recurses, so the formula is validated first
        (:func:`~pyfe4ai.policy.formula.validate_formula`): a program built by
        hand around a formula that is too deep raises ``FEValidationError``,
        not ``RecursionError``.
        """
        validate_formula(self.formula)
        held = held_labels(labels)
        _, weights = self._solve(self.formula, 0, held)
        if weights is None:
            return None
        total = [0] * self.num_cols
        for row, weight in weights.items():
            for col, entry in enumerate(self.matrix[row]):
                total[col] = (total[col] + weight * entry) % self.modulus
        if total != [1] + [0] * (self.num_cols - 1):
            # the coefficients come from the formula and the matrix from the compiler: they disagree
            raise FESchemeError(
                "the span program does not reconstruct the secret: its matrix does not match its formula")
        return dict(sorted(weights.items()))

    def is_qualified(self, labels: Iterable[str]) -> bool:
        """Whether the rows of *labels* span ``e_1``, that is, whether *labels*
        satisfy the formula."""
        return self.reconstruction(labels) is not None

    def _solve(self, node: Formula, start: int, held: frozenset) -> tuple[int, dict[int, int] | None]:
        """``(leaves under node, coefficients)``: the coefficients reconstruct
        the secret that *node*'s rows share, or are ``None``. *start* is the
        row of the first leaf under *node*. Every child is walked, even after
        the choice is made, because the next sibling's rows start after it."""
        if isinstance(node, Leaf):
            return 1, ({start: 1} if node.label in held else None)
        if not isinstance(node, (And, Or, Threshold)):
            raise FEValidationError("expected a formula, got {}".format(type(node).__name__))
        count = 0
        solutions: list[dict[int, int] | None] = []
        for child in node.children:
            size, solution = self._solve(child, start + count, held)
            count += size
            solutions.append(solution)
        if isinstance(node, Or):
            return count, next((s for s in solutions if s is not None), None)
        if isinstance(node, And):
            if any(s is None for s in solutions):
                return count, None
            return count, {row: weight for s in solutions for row, weight in s.items()}
        chosen = [(index, s) for index, s in enumerate(solutions) if s is not None][:node.t]
        if len(chosen) < node.t:
            return count, None
        # child number index + 1 holds the polynomial's value at the point index + 2
        factors = _lagrange_at_zero([index + 2 for index, _ in chosen], self.modulus)
        return count, {row: factor * weight % self.modulus
                       for factor, (_, s) in zip(factors, chosen) for row, weight in s.items()}


def _lagrange_at_zero(points: Sequence[int], modulus: int) -> list[int]:
    """The coefficients ``prod_{i != j} x_i / (x_i - x_j)`` that give the value
    at 0 of the polynomial through ``(x_j, y_j)`` as ``sum(coef_j * y_j)``.
    The points are distinct units modulo *modulus* (:func:`_check_gate_size`)."""
    out = []
    for j, point in enumerate(points):
        numerator = denominator = 1
        for i, other in enumerate(points):
            if i != j:
                numerator = numerator * other % modulus
                denominator = denominator * (other - point) % modulus
        out.append(numerator * pow(denominator, -1, modulus) % modulus)
    return out


def _check_gate_size(size: int, modulus: int) -> None:
    """A Threshold gate of *size* children evaluates at the points 2..size+1
    and divides by their differences, which are below size. Every integer
    1..size+1 must be a unit modulo *modulus*: for a prime modulus that is
    ``size + 1 < modulus`` (distinct, nonzero points); a composite modulus
    must have no prime factor up to size + 1."""
    for k in range(1, size + 2):
        divisor = gcd(k, modulus)
        if divisor != 1:
            raise FEValidationError(
                "a Threshold gate of size {} needs every integer from 1 to {} to be invertible modulo {}, "
                "but gcd({}, {}) = {}".format(size, size + 1, modulus, k, modulus, divisor))


class _Compiler:
    """Compiles a validated formula to sparse rows; see the module docstring."""

    def __init__(self, modulus: int) -> None:
        self.modulus = modulus
        self._columns = 0

    def _new_column(self) -> int:
        self._columns += 1
        return self._columns

    def compile(self, node: Formula) -> tuple[list[_SparseRow], int]:
        """``(rows, target column)`` of *node*; the rows are fresh dicts that
        the caller may change."""
        if isinstance(node, Leaf):
            column = self._new_column()
            return [{column: 1}], column
        parts = [self.compile(child) for child in node.children]
        if isinstance(node, Or):
            return self._or(parts)
        if isinstance(node, And):
            return self._and(parts)
        return self._threshold(node.t, parts)

    def _or(self, parts: list[tuple[list[_SparseRow], int]]) -> tuple[list[_SparseRow], int]:
        target = parts[0][1]
        rows: list[_SparseRow] = []
        for child_rows, child_target in parts:
            for row in child_rows:
                if child_target != target and child_target in row:
                    row[target] = row.pop(child_target)
            rows.extend(child_rows)
        return rows, target

    def _and(self, parts: list[tuple[list[_SparseRow], int]]) -> tuple[list[_SparseRow], int]:
        targets = [target for _, target in parts]
        new_target = self._new_column()                                # n_0: the secret of the gate
        random_columns = [self._new_column() for _ in targets[1:]]     # n_2 .. n_m: the shares of children 2..m
        rows: list[_SparseRow] = []
        for child_rows, _ in parts:
            for row in child_rows:
                first = row.pop(targets[0], 0)
                out: _SparseRow = {new_target: first} if first else {}
                for column, target in zip(random_columns, targets[1:]):
                    entry = (row.pop(target, 0) - first) % self.modulus
                    if entry:
                        out[column] = entry
                out.update(row)                                        # the columns below the children
                rows.append(out)
        return rows, new_target

    def _threshold(self, t: int, parts: list[tuple[list[_SparseRow], int]]) -> tuple[list[_SparseRow], int]:
        _check_gate_size(len(parts), self.modulus)
        coefficients = [self._new_column() for _ in range(t)]          # l_0 .. l_{t-1}: the polynomial
        rows: list[_SparseRow] = []
        for j, (child_rows, target) in enumerate(parts, start=1):
            point = j + 1
            powers = [pow(point, k, self.modulus) for k in range(t)]
            for row in child_rows:
                entry = row.pop(target, 0)                             # child j's secret is p(point)
                if entry:
                    for column, power in zip(coefficients, powers):
                        # the power of a unit is a unit, so a nonzero entry stays nonzero
                        row[column] = entry * power % self.modulus
            rows.extend(child_rows)
        return rows, coefficients[0]


def compile_formula(formula: Formula, modulus: int, *, max_rows: int = DEFAULT_MAX_ROWS) -> MSP:
    """Compile *formula* to a span program over ``Z_modulus``.

    *modulus* is an int ``>= 2``, normally a prime (the group order); a
    composite modulus is accepted only when every threshold gate can still
    invert its points (:func:`_check_gate_size`). *max_rows* bounds the
    number of rows, which is the number of leaves written out. Raises
    ``FEValidationError`` for an invalid argument, a formula that does not
    pass :func:`~pyfe4ai.policy.formula.validate_formula`, more leaves than
    *max_rows*, or a threshold gate that the modulus cannot take.
    """
    if not _is_int(modulus) or modulus < 2:
        raise FEValidationError("modulus must be an int >= 2, got {!r}".format(modulus))
    if not _is_int(max_rows) or max_rows < 1:
        raise FEValidationError("max_rows must be a positive int, got {!r}".format(max_rows))
    if not isinstance(formula, Formula):
        raise FEValidationError("formula must be a Formula, got {}".format(type(formula).__name__))
    validate_formula(formula)
    # before anything else: a shared sub-formula object can stand for 2**32 leaves
    leaves = leaf_count(formula)
    if leaves > max_rows:
        raise FEValidationError("the formula has {} leaves, more than the limit of {} rows".format(leaves, max_rows))
    rows, target = _Compiler(modulus).compile(formula)
    columns = [target] + sorted({column for row in rows for column in row} - {target})
    matrix = tuple(tuple(row.get(column, 0) for column in columns) for row in rows)
    return MSP(matrix, tuple(formula.leaves()), modulus, formula)
