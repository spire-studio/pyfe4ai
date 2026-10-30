"""Sharings with 0/1 reconstruction weights (pyfe4ai.policy.zero_one).

The tests check the rewrite of threshold gates to conjunctive normal form against matrices and formulas derived by
hand, against the size formula of the expansion, and against the original formula on every subset of the labels of
random formulas; then they check that the compiled span program reconstructs with the coefficient 1 on every row, over
a prime and over composite moduli. Every string is ASCII.
"""

from __future__ import annotations

import itertools
import random
import time
from math import comb

import pytest

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy import zero_one
from pyfe4ai.policy.formula import MAX_DEPTH, And, Formula, Leaf, Or, Threshold, evaluate, leaf_count
from pyfe4ai.policy.msp import MSP, compile_formula
from pyfe4ai.policy.zero_one import compile_zero_one, threshold_to_cnf

#: The order of the BLS12-381 groups, a 255-bit prime.
R = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001

#: A composite modulus: the product of two primes.
COMPOSITE = 1000003 * 1000033

A, B, C, D, E = (Leaf("h:{}".format(c)) for c in "abcde")


def _nodes(formula):
    stack = [formula]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(getattr(node, "children", ()))


def _labels(formula):
    return sorted(set(formula.leaves()))


def _subsets(labels):
    for size in range(len(labels) + 1):
        for subset in itertools.combinations(labels, size):
            yield set(subset)


def _distinct_leaves(count, prefix="x"):
    return [Leaf("{}:{}".format(prefix, i)) for i in range(count)]


# ---------------------------------------------------------------------------
# the rewrite, by hand
# ---------------------------------------------------------------------------

def test_cnf_of_a_threshold_gate():
    a, b, c = Leaf("h:a"), Leaf("h:b"), Leaf("h:c")
    assert threshold_to_cnf(Threshold(2, a, b, c)) == And(Or(b, c), Or(a, c), Or(a, b))
    assert threshold_to_cnf(Threshold(1, a, b)) == Or(a, b)
    assert threshold_to_cnf(Threshold(2, a, b)) == And(a, b)


def test_cnf_of_other_gates_by_hand():
    assert threshold_to_cnf(Threshold(3, A, B, C)) == And(A, B, C)
    assert threshold_to_cnf(Threshold(1, A, B, C)) == Or(A, B, C)
    assert threshold_to_cnf(Threshold(1, A)) == A
    # the clauses leave out every set of t - 1 children, in the order of itertools.combinations
    assert threshold_to_cnf(Threshold(3, A, B, C, D)) == And(
        Or(C, D), Or(B, D), Or(B, C), Or(A, D), Or(A, C), Or(A, B))
    assert threshold_to_cnf(Threshold(2, A, B, C, D)) == And(Or(B, C, D), Or(A, C, D), Or(A, B, D), Or(A, B, C))


def test_cnf_of_a_leaf_is_the_leaf_and_gates_without_thresholds_come_back_equal():
    assert threshold_to_cnf(A) == A
    for formula in (And(A, B), Or(A, B), And(Or(A, B), Or(C, A)), Or(And(A, B), C, And(D, E))):
        assert threshold_to_cnf(formula) == formula


def test_cnf_rewrites_thresholds_below_other_gates():
    assert threshold_to_cnf(And(A, Threshold(2, B, C, D))) == And(A, And(Or(C, D), Or(B, D), Or(B, C)))
    assert threshold_to_cnf(Or(Threshold(2, A, B), Threshold(1, C, D))) == Or(And(A, B), Or(C, D))


def test_cnf_has_no_threshold_gates_and_leaves_the_input_alone():
    formula = Threshold(2, Threshold(2, A, B, C), And(D, Threshold(2, E, Leaf("h:f"), Leaf("h:g"))), Leaf("h:h"))
    before = repr(formula)
    result = threshold_to_cnf(formula)
    assert repr(formula) == before
    assert not any(isinstance(node, Threshold) for node in _nodes(result))
    assert isinstance(result, Formula)


def test_nested_threshold_is_equivalent_and_has_the_expected_size():
    formula = Threshold(2, Threshold(2, A, B, C), D, E)
    result = threshold_to_cnf(formula)
    # the inner gate is 3 clauses of 2 leaves; the outer gate has the three children of sizes 6, 1, 1, and every
    # child sits in comb(2, 1) = 2 of its clauses
    assert (6 + 1 + 1) * comb(3 - 1, 2 - 1) == 16
    assert len(result.leaves()) == leaf_count(result) == 16
    for held in _subsets(_labels(formula)):
        assert evaluate(result, held) is evaluate(formula, held)
    assert not any(isinstance(node, Threshold) for node in _nodes(result))


@pytest.mark.parametrize("m, t", [(3, 2), (4, 2), (4, 3), (5, 2), (5, 3), (5, 4), (6, 3), (7, 3), (7, 5), (8, 4)])
def test_expansion_size_matches_the_formula(m, t):
    formula = Threshold(t, *_distinct_leaves(m))
    result = threshold_to_cnf(formula, max_rows=10 ** 6)
    assert len(result.leaves()) == comb(m, t - 1) * (m - t + 1)
    assert len(result.leaves()) == m * comb(m - 1, t - 1)
    assert len(result.children) == comb(m, t - 1)
    assert all(len(clause.children) == m - t + 1 for clause in result.children)


@pytest.mark.parametrize("m, t", [(3, 1), (3, 3), (6, 1), (6, 6)])
def test_the_edge_thresholds_do_not_expand(m, t):
    result = threshold_to_cnf(Threshold(t, *_distinct_leaves(m)))
    assert len(result.leaves()) == m


EXHAUSTIVE = [
    A,
    Threshold(1, A),
    Threshold(1, A, B),
    Threshold(2, A, B),
    Threshold(2, A, B, C),
    Threshold(3, A, B, C, D),
    And(A, Or(A, B)),                                          # a label on two rows
    Threshold(2, And(A, B), Or(C, D), E),
    Threshold(2, Threshold(2, A, B, C), D, E),
    Threshold(3, A, Or(B, C), And(D, E), Threshold(2, Leaf("h:f"), Leaf("h:g"), Leaf("h:h"))),
    Or(Threshold(2, A, B, C), And(D, E)),
    Threshold(2, Threshold(2, A, B, C), Threshold(2, D, E, Leaf("h:f"))),
]


@pytest.mark.parametrize("formula", EXHAUSTIVE)
def test_cnf_agrees_with_the_formula_on_every_subset_of_the_labels(formula):
    result = threshold_to_cnf(formula)
    assert not any(isinstance(node, Threshold) for node in _nodes(result))
    for held in _subsets(_labels(formula)):
        assert evaluate(result, held) is evaluate(formula, held)


# ---------------------------------------------------------------------------
# the size is checked first
# ---------------------------------------------------------------------------

def test_expansion_size_is_checked_first():
    leaves = [Leaf(f"h:{i}") for i in range(30)]
    started = time.perf_counter()
    with pytest.raises(FEValidationError, match="rows") as info:
        threshold_to_cnf(Threshold(15, *leaves))
    assert time.perf_counter() - started < 1      # C(30, 14) is about 1.45e8
    # the message gives the number of rows that the expansion would have and the limit
    assert str(30 * comb(29, 14)) in str(info.value)
    assert "64" in str(info.value)


def test_no_formula_object_is_built_before_the_size_check_passes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a formula object was built before the size check")

    leaves = _distinct_leaves(30)
    formula = Threshold(15, *leaves)
    monkeypatch.setattr(zero_one, "And", refuse)
    monkeypatch.setattr(zero_one, "Or", refuse)
    with pytest.raises(FEValidationError, match="rows"):
        threshold_to_cnf(formula)
    with pytest.raises(FEValidationError, match="rows"):
        compile_zero_one(formula, R)


def test_the_size_of_a_gate_with_many_children_does_not_overflow_the_message():
    # comb(19999, 9999) has about 6000 digits, more than str() of an int accepts by default
    formula = Threshold(10000, *_distinct_leaves(20000))
    with pytest.raises(FEValidationError, match="rows") as info:
        threshold_to_cnf(formula)
    assert "64" in str(info.value)


def test_the_size_of_a_shared_subformula_object_is_computed_without_expanding_it():
    f = Leaf("a:1")
    for _ in range(MAX_DEPTH):                  # 2**32 leaves when written out
        f = And(f, f)
    started = time.perf_counter()
    with pytest.raises(FEValidationError, match="rows") as info:
        threshold_to_cnf(f)
    assert time.perf_counter() - started < 1
    assert str(2 ** 32) in str(info.value)

    gate = Threshold(2, *_distinct_leaves(3))   # 6 leaves in CNF
    g = gate
    for _ in range(20):
        g = And(g, g)
    with pytest.raises(FEValidationError, match="rows") as info:
        threshold_to_cnf(g)
    assert str(6 * 2 ** 20) in str(info.value)


def test_a_shared_subformula_object_is_rewritten_once_and_shared():
    gate = Threshold(2, *_distinct_leaves(3))
    result = threshold_to_cnf(And(gate, gate))
    assert len(result.leaves()) == 12
    assert result.children[0] is result.children[1]
    # the clauses of one gate share the child objects: a child sits in comb(m - 1, t - 1) = 2 of 3 clauses
    shared = Leaf("s:1")
    result = threshold_to_cnf(Threshold(2, shared, Leaf("s:2"), Leaf("s:3")))
    assert sum(1 for clause in result.children for child in clause.children if child is shared) == 2


def test_row_limit_boundaries():
    formula = Threshold(3, *_distinct_leaves(5))         # comb(5, 2) * 3 = 30 rows
    assert len(threshold_to_cnf(formula, max_rows=30).leaves()) == 30
    with pytest.raises(FEValidationError, match="30") as info:
        threshold_to_cnf(formula, max_rows=29)
    assert "29" in str(info.value)
    assert compile_zero_one(formula, R, max_rows=30).num_rows == 30
    with pytest.raises(FEValidationError, match="29"):
        compile_zero_one(formula, R, max_rows=29)


def test_default_row_limit_is_64():
    # t = 2 of m leaves expands to m * (m - 1) rows: 56 for m = 8, 72 for m = 9
    assert len(threshold_to_cnf(Threshold(2, *_distinct_leaves(8))).leaves()) == 56
    with pytest.raises(FEValidationError, match="72") as info:
        threshold_to_cnf(Threshold(2, *_distinct_leaves(9)))
    assert "64" in str(info.value)
    assert compile_zero_one(And(*_distinct_leaves(64)), R).num_rows == 64
    for build in (lambda ls: Or(*ls), lambda ls: And(*ls), lambda ls: Threshold(1, *ls), lambda ls: Threshold(65, *ls)):
        with pytest.raises(FEValidationError, match="65"):
            threshold_to_cnf(build(_distinct_leaves(65)))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_a_repeated_label_under_a_threshold_gate_is_refused():
    # the rewrite removes the threshold gate and with it the rule, so the input is validated first
    with pytest.raises(FEValidationError, match="a:1"):
        threshold_to_cnf(Threshold(2, Leaf("a:1"), Leaf("a:1")))
    with pytest.raises(FEValidationError, match="h:a"):
        threshold_to_cnf(And(C, Threshold(2, And(A, B), And(A, D))))
    with pytest.raises(FEValidationError, match="a:1"):
        compile_zero_one(Threshold(2, Leaf("a:1"), Leaf("a:1")), R)
    # a label may repeat under And and Or, and inside one child of a threshold gate
    assert len(threshold_to_cnf(And(A, Or(A, B))).leaves()) == 3
    assert threshold_to_cnf(Threshold(2, And(A, A), B)) == And(And(A, A), B)


def _nested_and_around_threshold(depth):
    formula = Threshold(2, *_distinct_leaves(3))
    for i in range(depth):
        formula = And(formula, Leaf("n:{}".format(i)))
    return formula


def test_a_rewrite_that_is_too_deep_is_refused():
    # 31 Ands around the gate: depth 32 as written, 33 after the gate becomes And(Or(..), ..)
    too_deep = _nested_and_around_threshold(MAX_DEPTH - 1)
    with pytest.raises(FEValidationError, match="depth"):
        threshold_to_cnf(too_deep)
    with pytest.raises(FEValidationError, match="depth"):
        compile_zero_one(too_deep, R)
    # 30 Ands: the rewrite has depth 32
    deep_enough = _nested_and_around_threshold(MAX_DEPTH - 2)
    result = threshold_to_cnf(deep_enough)
    around = {"n:{}".format(i) for i in range(MAX_DEPTH - 2)}
    assert evaluate(result, around | {"x:0", "x:1"})
    assert not evaluate(result, around | {"x:0"})
    assert compile_zero_one(deep_enough, R).num_rows == 6 + MAX_DEPTH - 2


def test_a_formula_that_is_too_deep_to_begin_with_is_refused():
    deep = Leaf("a:0")
    for i in range(MAX_DEPTH + 1):
        deep = Or(Leaf("a:{}".format(i + 1)), deep)
    with pytest.raises(FEValidationError, match="depth"):
        threshold_to_cnf(deep)


@pytest.mark.parametrize("max_rows", [0, -1, True, False, 64.0, "64", None])
def test_max_rows_must_be_a_positive_int(max_rows):
    with pytest.raises(FEValidationError, match="max_rows"):
        threshold_to_cnf(A, max_rows=max_rows)
    with pytest.raises(FEValidationError, match="max_rows"):
        compile_zero_one(A, R, max_rows=max_rows)


@pytest.mark.parametrize("formula", [None, "h:a", ["h:a"], 5, Formula(), ("h:a",)])
def test_formula_must_be_a_formula(formula):
    with pytest.raises(FEValidationError):
        threshold_to_cnf(formula)
    with pytest.raises(FEValidationError):
        compile_zero_one(formula, R)


@pytest.mark.parametrize("modulus", [True, False, 7.0, "7", None, 1, 0, -7])
def test_modulus_must_be_an_int_at_least_two(modulus):
    with pytest.raises(FEValidationError, match="modulus"):
        compile_zero_one(A, modulus)


# ---------------------------------------------------------------------------
# the compiled span program
# ---------------------------------------------------------------------------

def test_compile_zero_one_known_matrix():
    msp = compile_zero_one(Threshold(2, A, B, C), R)
    assert isinstance(msp, MSP)
    # And(Or(b, c), Or(a, c), Or(a, b)): the three clauses are shares of one secret, summing to it
    assert [list(row) for row in msp.matrix] == [
        [1, R - 1, R - 1], [1, R - 1, R - 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert msp.row_labels == ("h:b", "h:c", "h:a", "h:c", "h:a", "h:b")
    assert msp.modulus == R
    assert msp.reconstruction({"h:a", "h:b"}) == {0: 1, 2: 1, 4: 1}
    assert msp.reconstruction({"h:b", "h:c"}) == {0: 1, 3: 1, 5: 1}
    assert msp.reconstruction({"h:a"}) is None


def test_the_program_is_that_of_the_rewritten_formula():
    formula = Threshold(2, A, And(B, C), D)
    cnf = threshold_to_cnf(formula)
    msp = compile_zero_one(formula, R)
    assert msp.formula == cnf
    assert msp == compile_formula(cnf, R)
    assert msp.row_labels == tuple(msp.formula.leaves())
    assert msp.row_labels != tuple(formula.leaves())          # the rows follow the rewritten formula
    assert msp.num_rows == len(cnf.leaves())


def test_a_formula_without_threshold_gates_compiles_as_it_is():
    formula = And(Or(A, B), Or(C, D))
    assert compile_zero_one(formula, R) == compile_formula(formula, R)


def _check_zero_one(msp, original, held):
    """What a program with 0/1 weights must do for one set of labels."""
    qualified = evaluate(original, held)
    assert evaluate(msp.formula, held) is qualified
    assert msp.is_qualified(held) is qualified
    weights = msp.reconstruction(held)
    assert (weights is not None) is qualified
    if not qualified:
        return
    modulus = msp.modulus
    assert weights and set(weights.values()) == {1}
    assert all(type(row) is int and msp.row_labels[row] in held for row in weights)
    total = [0] * msp.num_cols
    for row in weights:                                       # every coefficient is 1: the plain sum of the rows
        for col, entry in enumerate(msp.matrix[row]):
            total[col] = (total[col] + entry) % modulus
    assert total == [1] + [0] * (msp.num_cols - 1)


def _random_formula(rng, pool, depth):
    """A random formula over *pool*; the children of a Threshold gate get disjoint parts of the pool."""
    if depth == 0 or len(pool) < 2 or rng.random() < 0.2:
        return Leaf(rng.choice(pool))
    kind = rng.choice(["and", "or", "threshold", "threshold"])
    size = rng.randint(2, 4)
    if kind == "threshold":
        size = min(size, len(pool))
        cuts = [0] + sorted(rng.sample(range(1, len(pool)), size - 1)) + [len(pool)]
        shuffled = rng.sample(pool, len(pool))
        children = [_random_formula(rng, shuffled[a:b], depth - 1) for a, b in zip(cuts, cuts[1:])]
        return Threshold(rng.randint(1, size), *children)
    children = [_random_formula(rng, pool, depth - 1) for _ in range(size)]
    return And(*children) if kind == "and" else Or(*children)


@pytest.mark.parametrize("modulus", [R, COMPOSITE])
def test_random_formulas(modulus):
    # fixed seed; 200 random formulas over at most 7 labels, every subset of the labels for each of them
    rng = random.Random(20261031)
    formulas = 0
    expanded = 0
    gates = set()
    while formulas < 200:
        pool = ["r:{}".format(c) for c in "abcdefg"][:rng.randint(2, 7)]
        formula = _random_formula(rng, pool, 3)
        try:
            msp = compile_zero_one(formula, modulus)         # skip the ones whose expansion passes 64 rows
        except FEValidationError as exc:
            assert "rows" in str(exc)
            continue
        formulas += 1
        gates.update(type(node).__name__ for node in _nodes(formula))
        cnf = msp.formula
        assert cnf == threshold_to_cnf(formula)
        if len(cnf.leaves()) != len(formula.leaves()):
            expanded += 1
        assert all(entry in (0, 1, modulus - 1) for row in msp.matrix for entry in row)
        assert all(type(entry) is int for row in msp.matrix for entry in row)
        for held in _subsets(sorted(pool)):
            assert evaluate(formula, held) is evaluate(cnf, held) is msp.is_qualified(held)
            _check_zero_one(msp, formula, held)
    assert gates == {"Leaf", "And", "Or", "Threshold"}
    assert expanded > 20                                      # the rewrite really changed many of the formulas


@pytest.mark.parametrize("modulus", [2, 3, 4, 6, 30])
@pytest.mark.parametrize("formula", EXHAUSTIVE)
def test_small_and_composite_moduli_take_threshold_gates(formula, modulus):
    # compile_formula needs the points 2..m + 1 of a threshold gate to be units; the CNF has no such need
    msp = compile_zero_one(formula, modulus)
    assert all(entry in (0, 1, modulus - 1) for row in msp.matrix for entry in row)
    for held in _subsets(_labels(formula)):
        _check_zero_one(msp, formula, held)


def test_a_modulus_that_cannot_take_the_threshold_gate_is_still_refused_by_the_direct_compiler():
    with pytest.raises(FEValidationError, match="invertible"):
        compile_formula(Threshold(2, A, B, C), 6)
    assert compile_zero_one(Threshold(2, A, B, C), 6).num_rows == 6
