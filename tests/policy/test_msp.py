"""Compiling formulas to monotone span programs (pyfe4ai.policy.msp).

The tests check the compiler three ways: against matrices derived by hand, against the formula itself on random
formulas (a set of labels is qualified exactly when the formula holds; the reconstruction coefficients really give
e_1; and when the formula does not hold, a Gaussian elimination written here shows that e_1 is not in the span of
the rows that the labels select), and against the shares the matrix deals. Every string is ASCII.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import random
import time

import pytest

from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.policy.formula import MAX_DEPTH, And, Formula, Leaf, Or, Threshold, evaluate
from pyfe4ai.policy.msp import MSP, compile_formula

#: The order of the BLS12-381 groups, a 255-bit prime.
P = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001

A, B, C, D = (Leaf("h:{}".format(c)) for c in "abcd")

#: (formula, matrix): the matrices derived by hand from the rules, entries taken modulo the modulus.
KNOWN = [
    (And(Or(A, B), Or(C, D)), [[1, -1], [1, -1], [0, 1], [0, 1]]),
    (A, [[1]]),
    (Or(A, B), [[1], [1]]),
    (And(A, B), [[1, -1], [0, 1]]),
    (Threshold(2, A, B, C), [[1, 2], [1, 3], [1, 4]]),
    (Or(A, And(B, C)), [[1, 0], [1, -1], [0, 1]]),
    (Threshold(2, A, And(B, C), D), [[1, 0, 2], [1, -1, 3], [0, 1, 0], [1, 0, 4]]),
]
KNOWN_IDS = ["and-or-or", "leaf", "or", "and", "threshold-2-of-3", "or-and", "threshold-with-and"]

#: Formulas on which every subset of the labels is checked.
EXHAUSTIVE = [f for f, _ in KNOWN] + [
    And(A, Or(A, B)),                                      # a label on two rows
    Threshold(2, And(A, B), Or(C, D), Leaf("h:e")),
    Threshold(3, A, Or(B, C), And(D, Leaf("h:e")), Threshold(2, Leaf("h:f"), Leaf("h:g"), Leaf("h:h"))),
    Or(Threshold(2, A, B, C), And(D, Leaf("h:e"))),
    Threshold(1, A, B),
    Threshold(1, A),
    Threshold(2, Threshold(2, A, B, C), Threshold(2, D, Leaf("h:e"), Leaf("h:f"))),
]


def _labels(formula):
    return sorted(set(formula.leaves()))


def _subsets(labels):
    for size in range(len(labels) + 1):
        for subset in itertools.combinations(labels, size):
            yield set(subset)


def _in_rowspace(rows, target, p):
    """Whether *target* is a linear combination of *rows* over GF(p), p prime (Gaussian elimination)."""
    basis = {}                      # pivot column -> row with that pivot equal to 1 and the leftmost nonzero entry

    def reduce(vector):
        vector = [x % p for x in vector]
        for col in range(len(vector)):
            entry = vector[col]
            if entry == 0:
                continue
            if col not in basis:
                return col, vector
            vector = [(x - entry * y) % p for x, y in zip(vector, basis[col])]
        return None, vector

    for row in rows:
        col, vector = reduce(row)
        if col is not None:
            inverse = pow(vector[col], -1, p)
            basis[col] = [(x * inverse) % p for x in vector]
    return reduce(target)[0] is None


def _unit(msp):
    return [1] + [0] * (msp.num_cols - 1)


def _combination(msp, weights):
    """sum of weights[row] * matrix[row] modulo the modulus, as a list."""
    total = [0] * msp.num_cols
    for row, weight in weights.items():
        for col, entry in enumerate(msp.matrix[row]):
            total[col] = (total[col] + weight * entry) % msp.modulus
    return total


def _held_rows(msp, held):
    return [row for row, label in enumerate(msp.row_labels) if label in held]


def _check_labels(msp, formula, held, prime):
    """The three-way agreement for one set of labels; *prime* says whether the modulus is a prime (elimination)."""
    weights = msp.reconstruction(held)
    qualified = evaluate(formula, held)
    assert msp.is_qualified(held) is qualified
    assert (weights is not None) is qualified
    rows = _held_rows(msp, held)
    if qualified:
        assert weights and all(row in rows for row in weights)
        assert all(type(w) is int and 0 < w < msp.modulus for w in weights.values())
        assert _combination(msp, weights) == _unit(msp)
        if prime:
            assert _in_rowspace([msp.matrix[r] for r in rows], _unit(msp), msp.modulus)
    elif prime:
        assert not _in_rowspace([msp.matrix[r] for r in rows], _unit(msp), msp.modulus)


# ---------------------------------------------------------------------------
# known answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("formula, expected", KNOWN, ids=KNOWN_IDS)
def test_known_matrices(formula, expected):
    msp = compile_formula(formula, P)
    assert [list(row) for row in msp.matrix] == [[x % P for x in row] for row in expected]
    assert msp.row_labels == tuple(formula.leaves())
    assert msp.modulus == P
    assert msp.formula is formula
    assert msp.num_rows == len(expected)
    assert msp.num_cols == len(expected[0])


@pytest.mark.parametrize("formula", [f for f, _ in KNOWN], ids=KNOWN_IDS)
def test_matrix_entries_are_plain_ints_in_range(formula):
    msp = compile_formula(formula, P)
    assert isinstance(msp.matrix, tuple) and isinstance(msp.row_labels, tuple)
    for row in msp.matrix:
        assert isinstance(row, tuple) and len(row) == msp.num_cols
        assert all(type(x) is int and 0 <= x < P for x in row)
    assert all(any(row[col] for row in msp.matrix) for col in range(msp.num_cols))     # no empty column


def test_msp_is_frozen():
    msp = compile_formula(And(A, B), P)
    with pytest.raises(dataclasses.FrozenInstanceError):
        msp.modulus = 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        msp.matrix = ()


def test_compilation_is_deterministic():
    f = Threshold(2, A, And(B, C), D)
    first = compile_formula(f, P)
    assert compile_formula(f, P) == first
    assert compile_formula(Threshold(2, A, And(B, C), D), P) == first       # an equal formula built again
    assert hash(compile_formula(f, P)) == hash(first)
    assert compile_formula(f, P).matrix == first.matrix
    assert compile_formula(f, 101) != first                                  # the modulus is part of the program


# ---------------------------------------------------------------------------
# reconstruction against the formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("formula", EXHAUSTIVE)
def test_every_subset_of_labels_agrees_with_the_formula(formula):
    msp = compile_formula(formula, P)
    for held in _subsets(_labels(formula)):
        _check_labels(msp, formula, held, prime=True)


@pytest.mark.parametrize("modulus", [5, 7, 101])
def test_small_prime_moduli_agree_with_the_formula(modulus):
    # modulus 5 is the least prime that takes a threshold gate of three children
    for formula in (Threshold(2, A, B, C), Threshold(2, A, And(B, C), D), Threshold(3, A, B, C), And(Or(A, B), C)):
        msp = compile_formula(formula, modulus)
        for held in _subsets(_labels(formula)):
            _check_labels(msp, formula, held, prime=True)


def test_shares_of_a_set_that_does_not_satisfy_the_formula_do_not_depend_on_the_secret():
    # independent of the elimination above: deal every secret with every randomness over Z_7 and compare the
    # distributions of the shares that a set of labels holds
    modulus = 7
    checked = 0
    for formula in EXHAUSTIVE:
        msp = compile_formula(formula, modulus)
        if msp.num_cols > 4:
            continue
        everything = list(itertools.product(range(modulus), repeat=msp.num_cols - 1))
        for held in _subsets(_labels(formula)):
            rows = _held_rows(msp, held)
            dealt = [collections.Counter(tuple(msp.shares(secret, list(r))[i] for i in rows) for r in everything)
                     for secret in range(modulus)]
            if evaluate(formula, held):
                # the shares fix the secret: no tuple of shares occurs for two secrets
                assert sum(len(d) for d in dealt) == len(set().union(*dealt))
            else:
                # every secret deals the same distribution
                assert all(d == dealt[0] for d in dealt)
            checked += 1
    assert checked > 100


def _random_formula(rng, pool, depth):
    """A random formula over *pool*; the children of a Threshold gate get disjoint parts of the pool."""
    if depth == 0 or len(pool) < 2 or rng.random() < 0.2:
        return Leaf(rng.choice(pool))
    kind = rng.choice(["and", "or", "threshold"])
    size = rng.randint(2, 4)
    if kind == "threshold":
        size = min(size, len(pool))
        cuts = [0] + sorted(rng.sample(range(1, len(pool)), size - 1)) + [len(pool)]
        shuffled = rng.sample(pool, len(pool))
        children = [_random_formula(rng, shuffled[a:b], depth - 1) for a, b in zip(cuts, cuts[1:])]
        return Threshold(rng.randint(1, size), *children)
    children = [_random_formula(rng, pool, depth - 1) for _ in range(size)]
    return And(*children) if kind == "and" else Or(*children)


def test_random_formulas_agree():
    rng = random.Random(20261031)
    pool = ["r:{}".format(c) for c in "abcdef"]
    seen_gates = set()
    for _ in range(1000):
        formula = _random_formula(rng, pool, 3)
        seen_gates.update(type(n).__name__ for n in _nodes(formula))
        msp = compile_formula(formula, P)
        for _ in range(5):
            held = {label for label in pool if rng.random() < 0.5}
            _check_labels(msp, formula, held, prime=True)
    assert seen_gates == {"Leaf", "And", "Or", "Threshold"}


def _nodes(formula):
    stack = [formula]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(getattr(node, "children", ()))


def test_reconstruction_coefficients_by_hand():
    msp = compile_formula(Threshold(2, A, B, C), P)
    # points 2, 3, 4: the Lagrange coefficients at 0 of two of them
    assert msp.reconstruction({"h:a", "h:b"}) == {0: 3, 1: P - 2}
    assert msp.reconstruction({"h:a", "h:c"}) == {0: 2, 2: P - 1}
    assert msp.reconstruction({"h:b", "h:c"}) == {1: 4, 2: P - 3}
    assert msp.reconstruction({"h:a", "h:b", "h:c"}) == {0: 3, 1: P - 2}       # the first two satisfied children
    assert msp.reconstruction({"h:a"}) is None
    assert msp.reconstruction(set()) is None
    and_ = compile_formula(And(A, B), P)
    assert and_.reconstruction({"h:a", "h:b"}) == {0: 1, 1: 1}
    or_ = compile_formula(Or(A, B), P)
    assert or_.reconstruction({"h:b"}) == {1: 1}
    assert or_.reconstruction({"h:a", "h:b"}) == {0: 1}                         # the first satisfied child
    assert compile_formula(A, P).reconstruction(["h:a"]) == {0: 1}


def test_reconstruction_uses_every_row_of_a_repeated_label():
    msp = compile_formula(And(A, Or(A, B)), P)
    assert msp.row_labels == ("h:a", "h:a", "h:b")
    weights = msp.reconstruction({"h:a"})
    assert weights is not None and set(weights) == {0, 1}
    assert _combination(msp, weights) == _unit(msp)
    assert msp.reconstruction({"h:b"}) is None


def test_reconstruction_ignores_labels_the_formula_does_not_name():
    msp = compile_formula(And(A, Or(B, C)), P)
    assert msp.reconstruction({"h:a", "h:c", "z:9", "not a label"}) == msp.reconstruction({"h:a", "h:c"})
    assert msp.reconstruction(iter(["h:a", "h:b"])) is not None                 # any iterable of labels
    assert msp.is_qualified(["h:a", "h:b", "h:b"])


@pytest.mark.parametrize("labels", ["h:a", b"h:a", 5, [["h:a"]], None])
def test_reconstruction_and_is_qualified_refuse_labels_that_are_not_an_iterable_of_labels(labels):
    msp = compile_formula(Or(A, B), P)
    with pytest.raises(FEValidationError):
        msp.reconstruction(labels)
    with pytest.raises(FEValidationError):
        msp.is_qualified(labels)


def test_reconstruction_checks_its_own_result():
    # a matrix that does not belong to its formula must fail loudly, not hand out coefficients
    msp = compile_formula(Threshold(2, A, B, C), P)
    wrong = MSP(matrix=((1, 2), (1, 3), (1, 5)), row_labels=msp.row_labels, modulus=P, formula=msp.formula)
    assert wrong.reconstruction({"h:a", "h:b"}) == {0: 3, 1: P - 2}              # these rows are still right
    with pytest.raises(FESchemeError, match="span program"):
        wrong.reconstruction({"h:a", "h:c"})


# ---------------------------------------------------------------------------
# shares
# ---------------------------------------------------------------------------

def test_shares_by_hand():
    assert compile_formula(And(A, B), P).shares(5, [3]) == [2, 3]
    assert compile_formula(Or(A, B), P).shares(5, []) == [5, 5]
    assert compile_formula(Threshold(2, A, B, C), P).shares(5, [3]) == [11, 14, 17]
    assert compile_formula(And(A, B), P).shares(P - 1, [2]) == [P - 3, 2]
    assert compile_formula(And(A, B), P).shares(-1, [2 + P]) == [P - 3, 2]        # inputs of any size, results reduced
    assert compile_formula(A, P).shares(7, []) == [7]


def test_shares_reconstruct():
    rng = random.Random(7)
    pool = ["r:{}".format(c) for c in "abcdef"]
    for _ in range(200):
        formula = _random_formula(rng, pool, 3)
        msp = compile_formula(formula, P)
        secret = rng.randrange(P)
        shares = msp.shares(secret, [rng.randrange(P) for _ in range(msp.num_cols - 1)])
        assert len(shares) == msp.num_rows and all(type(s) is int and 0 <= s < P for s in shares)
        for _ in range(5):
            held = {label for label in pool if rng.random() < 0.5}
            weights = msp.reconstruction(held)
            if weights is not None:
                assert sum(w * shares[row] for row, w in weights.items()) % P == secret


@pytest.mark.parametrize("randomness", [[], [1, 2], [1, 2, 3, 4]])
def test_shares_refuse_a_wrong_randomness_length(randomness):
    msp = compile_formula(Threshold(2, A, B, C), P)           # two columns: one random value
    with pytest.raises(FEValidationError, match="randomness"):
        msp.shares(5, randomness)
    with pytest.raises(FEValidationError, match="randomness"):
        compile_formula(A, P).shares(5, [1])


@pytest.mark.parametrize("secret, randomness", [
    (1.0, [1]), ("5", [1]), (True, [1]), (None, [1]),
    (5, [1.0]), (5, ["1"]), (5, [True]), (5, "1"), (5, b"\x01"), (5, None), (5, 1),
])
def test_shares_refuse_values_that_are_not_ints(secret, randomness):
    msp = compile_formula(And(A, B), P)
    with pytest.raises(FEValidationError):
        msp.shares(secret, randomness)


# ---------------------------------------------------------------------------
# the row limit
# ---------------------------------------------------------------------------

def test_row_limit():
    leaves = [Leaf("x:{}".format(i)) for i in range(65)]
    msp = compile_formula(Or(*leaves[:64]), P)
    assert msp.num_rows == 64
    for build in (lambda ls: Or(*ls), lambda ls: And(*ls), lambda ls: Threshold(2, *ls)):
        with pytest.raises(FEValidationError, match="64") as info:
            compile_formula(build(leaves), P)
        assert "65" in str(info.value)
    assert compile_formula(And(*leaves[:64]), P).num_rows == 64


def test_row_limit_can_be_set():
    f = And(A, B, C, D)
    assert compile_formula(f, P, max_rows=4).num_rows == 4
    with pytest.raises(FEValidationError, match="3"):
        compile_formula(f, P, max_rows=3)
    assert compile_formula(f, P, max_rows=1000).num_rows == 4


def test_shared_subformula_objects_count_once_per_occurrence():
    f = Leaf("a:1")
    for _ in range(6):                          # 2**6 = 64 leaves written out
        f = And(f, f)
    msp = compile_formula(f, P)
    assert msp.num_rows == 64 and msp.row_labels == ("a:1",) * 64
    assert msp.is_qualified({"a:1"}) and not msp.is_qualified(set())
    g = And(f, f)                               # 128
    with pytest.raises(FEValidationError, match="128"):
        compile_formula(g, P)
    shared = Or(A, B)
    msp = compile_formula(And(shared, shared), P)
    assert msp.row_labels == ("h:a", "h:b", "h:a", "h:b")
    weights = msp.reconstruction({"h:a"})
    assert weights is not None and _combination(msp, weights) == _unit(msp)


def test_row_limit_is_checked_before_the_leaves_are_listed(monkeypatch):
    def refuse(self):
        raise AssertionError("leaves() was called on a formula that was not counted yet")

    f = Leaf("a:1")
    for _ in range(MAX_DEPTH):                  # 2**32 leaves when written out
        f = And(f, f)
    monkeypatch.setattr(Formula, "leaves", refuse)
    started = time.perf_counter()
    with pytest.raises(FEValidationError, match="64"):
        compile_formula(f, P)
    assert time.perf_counter() - started < 1.0


# ---------------------------------------------------------------------------
# moduli and threshold gates
# ---------------------------------------------------------------------------

def test_threshold_points_must_be_distinct():
    f = Threshold(2, A, B, C)
    with pytest.raises(FEValidationError):
        compile_formula(f, 3)               # points 2, 3, 4: 3 is 0 modulo 3
    msp = compile_formula(f, 5)             # points 2, 3, 4 are distinct and nonzero modulo 5
    assert msp.matrix == ((1, 2), (1, 3), (1, 4))
    with pytest.raises(FEValidationError):
        compile_formula(Threshold(2, A, B, C, D), 5)       # points 2..5, and 5 is 0 modulo 5


@pytest.mark.parametrize("modulus", [2, 3, 4, 6, 9, 10, 12, 15, 21])
def test_threshold_gates_refuse_moduli_that_cannot_invert_the_points(modulus):
    f = Threshold(2, A, B, C)               # 1..4 must be invertible: no modulus 2, 3, 4 or any multiple of them
    with pytest.raises(FEValidationError, match=r"size 3.*{}".format(modulus)):
        compile_formula(f, modulus)


def test_composite_moduli_that_can_invert_the_points_work():
    f = Threshold(2, A, B)                   # needs 1, 2, 3 invertible
    msp = compile_formula(f, 35)
    assert msp.matrix == ((1, 2), (1, 3))
    weights = msp.reconstruction({"h:a", "h:b"})
    assert weights == {0: 3, 1: 33}
    shares = msp.shares(17, [9])
    assert shares == [(17 + 18) % 35, (17 + 27) % 35]
    assert sum(w * shares[row] for row, w in weights.items()) % 35 == 17
    assert msp.reconstruction({"h:a"}) is None
    three = compile_formula(Threshold(2, A, B, C), 35)       # 1..4: 5 and 7 do not divide them
    shares = three.shares(11, [20])
    for pair in itertools.combinations(("h:a", "h:b", "h:c"), 2):
        weights = three.reconstruction(set(pair))
        assert sum(w * shares[row] for row, w in weights.items()) % 35 == 11


def test_a_composite_modulus_is_refused_per_gate_and_only_for_thresholds():
    assert compile_formula(And(Or(A, B), C), 15).num_rows == 3          # no Threshold gate: any modulus >= 2
    assert compile_formula(And(A, B), 2).matrix == ((1, 1), (0, 1))
    four = Threshold(2, A, B, C, D)                                    # 1..5
    with pytest.raises(FEValidationError, match=r"size 4.*35"):
        compile_formula(four, 35)
    with pytest.raises(FEValidationError, match=r"size 3.*15"):
        compile_formula(Or(Leaf("x:1"), Threshold(2, A, B, C)), 15)    # a gate below the root
    with pytest.raises(FEValidationError, match=r"size 1.*2"):
        compile_formula(Threshold(1, A), 2)                            # even a gate of one child


def test_composite_modulus_agrees_with_the_formula_wherever_it_compiles():
    for formula in (Threshold(2, A, B), Threshold(2, A, And(B, C)), And(Threshold(2, A, B), Or(C, D))):
        msp = compile_formula(formula, 35)
        for held in _subsets(_labels(formula)):
            _check_labels(msp, formula, held, prime=False)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modulus", [True, False, 7.0, "7", None, 1, 0, -7])
def test_modulus_must_be_an_int_at_least_two(modulus):
    with pytest.raises(FEValidationError, match="modulus"):
        compile_formula(A, modulus)


@pytest.mark.parametrize("max_rows", [0, -1, True, 64.0, "64", None])
def test_max_rows_must_be_a_positive_int(max_rows):
    with pytest.raises(FEValidationError, match="max_rows"):
        compile_formula(A, P, max_rows=max_rows)


@pytest.mark.parametrize("formula", [None, "h:a", ["h:a"], 5, Formula(), ("h:a",)])
def test_formula_must_be_a_formula(formula):
    with pytest.raises(FEValidationError):
        compile_formula(formula, P)


def test_formulas_that_do_not_validate_are_refused():
    with pytest.raises(FEValidationError, match="h:a"):
        compile_formula(Threshold(2, And(A, B), And(A, C)), P)           # one label under two threshold children
    deep = Leaf("a:0")
    for i in range(MAX_DEPTH + 1):
        deep = Or(Leaf("a:{}".format(i + 1)), deep)
    with pytest.raises(FEValidationError, match="depth"):
        compile_formula(deep, P)
