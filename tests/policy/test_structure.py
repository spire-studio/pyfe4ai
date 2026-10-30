"""Qualified and unqualified families of a policy over a list of holders, and the Q2 check (pyfe4ai.policy.structure).

The tests check the families against answers derived by hand, against a brute-force reading of the definition on
random formulas and holder lists (minimal qualified sets, maximal unqualified sets, and that every subset is
qualified exactly when it contains a minimal one), and the Q2 check against a direct search of all pairs. Every
string is ASCII.
"""

from __future__ import annotations

import itertools
import random
import time
from math import comb

import pytest

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.formula import MAX_DEPTH, And, Leaf, Or, Threshold, evaluate
from pyfe4ai.policy.structure import MAX_HOLDERS, q2_violation, qualified_families


class _MyInt(int):
    """An int subclass."""


# ---------------------------------------------------------------------------
# answers derived by hand
# ---------------------------------------------------------------------------

def test_max_holders():
    assert MAX_HOLDERS == 8 and type(MAX_HOLDERS) is int


def test_two_of_three():
    f = Threshold(2, Leaf("h:0"), Leaf("h:1"), Leaf("h:2"))
    minimal, maximal = qualified_families(f, [{"h:0"}, {"h:1"}, {"h:2"}])
    assert minimal == [0b011, 0b101, 0b110] and maximal == [0b001, 0b010, 0b100]
    assert q2_violation(maximal, 3) is None


def test_holders_share_an_attribute():
    f = And(Leaf("org:A"), Leaf("org:B"))
    minimal, _ = qualified_families(f, [{"org:A"}, {"org:A"}, {"org:B"}])
    assert minimal == [0b101, 0b110]


def test_holders_share_an_attribute_maximal_sets():
    f = And(Leaf("org:A"), Leaf("org:B"))
    _, maximal = qualified_families(f, [{"org:A"}, {"org:A"}, {"org:B"}])
    # without holder 2 nobody has org:B; without holders 0 and 1 nobody has org:A
    assert maximal == [0b011, 0b100]


def test_one_holder_with_two_attributes():
    f = And(Leaf("a:1"), Leaf("a:2"))
    minimal, maximal = qualified_families(f, [{"a:1", "a:2"}, {"a:1"}, {"a:2"}])
    assert minimal == [0b001, 0b110]
    assert maximal == [0b010, 0b100]


def test_the_empty_set_is_an_ordinary_subset():
    # every single holder is qualified: the only maximal unqualified set is the empty one
    minimal, maximal = qualified_families(Leaf("h:0"), [{"h:0"}])
    assert minimal == [0b1] and maximal == [0b0]
    minimal, maximal = qualified_families(Or(Leaf("h:0"), Leaf("h:1")), [{"h:0"}, {"h:1"}])
    assert minimal == [0b01, 0b10] and maximal == [0b00]


def test_a_holder_with_no_labels_is_legal_and_in_no_minimal_set():
    minimal, maximal = qualified_families(Leaf("h:0"), [{"h:0"}, set()])
    assert minimal == [0b01]
    assert maximal == [0b10]


def test_a_formula_no_subset_satisfies():
    minimal, maximal = qualified_families(Leaf("h:z"), [{"h:0"}, {"h:1"}, set()])
    assert minimal == [] and maximal == [0b111]
    assert q2_violation(maximal, 3) == (0b111, 0b111)


def test_the_full_set_alone_is_qualified():
    f = And(Leaf("h:0"), Leaf("h:1"), Leaf("h:2"))
    minimal, maximal = qualified_families(f, [{"h:0"}, {"h:1"}, {"h:2"}])
    assert minimal == [0b111]
    assert maximal == [0b011, 0b101, 0b110]


def test_the_families_are_lists_of_plain_ints_in_ascending_order():
    f = Threshold(2, Leaf("h:0"), Leaf("h:1"), Leaf("h:2"), Leaf("h:3"))
    minimal, maximal = qualified_families(f, [{"h:3"}, {"h:2"}, {"h:1"}, {"h:0"}])
    for family in (minimal, maximal):
        assert type(family) is list
        assert all(type(m) is int for m in family)
        assert family == sorted(set(family))
    assert len(minimal) == comb(4, 2) and len(maximal) == comb(4, 1)


def test_holder_labels_may_be_any_iterable():
    f = Threshold(2, Leaf("h:0"), Leaf("h:1"), Leaf("h:2"), Leaf("h:3"), Leaf("h:4"))
    holders = [["h:0"], ("h:1",), frozenset({"h:2"}), (label for label in ["h:3"]), {"h:4"}]
    minimal, maximal = qualified_families(f, holders)
    assert len(minimal) == comb(5, 2) and len(maximal) == 5
    assert qualified_families(f, (["h:0"], ["h:1"], ["h:2"], ["h:3"], ["h:4"])) == (minimal, maximal)


def test_labels_that_no_leaf_names_do_not_matter():
    f = And(Leaf("a:1"), Leaf("a:2"))
    plain = qualified_families(f, [{"a:1"}, {"a:2"}])
    padded = qualified_families(f, [{"a:1", "z:9", "z:8"}, {"a:2", "z:7"}])
    assert plain == padded == ([0b11], [0b01, 0b10])


def test_the_holders_are_not_modified():
    holders = [{"a:1"}, {"a:2"}]
    qualified_families(And(Leaf("a:1"), Leaf("a:2")), holders)
    assert holders == [{"a:1"}, {"a:2"}]


def test_eight_holders():
    pool = ["h:{}".format(i) for i in range(MAX_HOLDERS)]
    f = Threshold(5, *(Leaf(label) for label in pool))
    start = time.perf_counter()
    minimal, maximal = qualified_families(f, [{label} for label in pool])
    assert time.perf_counter() - start < 5
    assert minimal == sorted(sum(1 << i for i in c) for c in itertools.combinations(range(8), 5))
    assert maximal == sorted(sum(1 << i for i in c) for c in itertools.combinations(range(8), 4))
    # two disjoint sets of four holders cover all eight
    assert q2_violation(maximal, MAX_HOLDERS) == (0b00001111, 0b11110000)
    _, maximal = qualified_families(Threshold(4, *(Leaf(label) for label in pool)), [{label} for label in pool])
    assert q2_violation(maximal, MAX_HOLDERS) is None


@pytest.mark.parametrize("n", range(1, 7))
def test_threshold_structures_meet_q2_exactly_when_two_unqualified_sets_cannot_cover(n):
    pool = ["h:{}".format(i) for i in range(n)]
    holders = [{label} for label in pool]
    for t in range(1, n + 1):
        minimal, maximal = qualified_families(Threshold(t, *(Leaf(label) for label in pool)), holders)
        assert len(minimal) == comb(n, t)
        # the maximal unqualified sets have t - 1 holders, and two of them cover n holders iff 2 (t - 1) >= n
        assert (q2_violation(maximal, n) is None) == (2 * (t - 1) < n)


# ---------------------------------------------------------------------------
# the Q2 check
# ---------------------------------------------------------------------------

def test_two_owners_with_their_own_holders_fail_q2():
    # the unqualified sets of a conjunction of two rules are the union of those of each rule
    _, z1 = qualified_families(Leaf("h:0"), [{"h:0"}, {"h:1"}])
    _, z2 = qualified_families(Leaf("h:1"), [{"h:0"}, {"h:1"}])
    assert q2_violation(z1 + z2, 2) in ((0b01, 0b10), (0b10, 0b01))


def test_q2_names_the_first_pair_in_the_order_of_the_sequence():
    assert q2_violation([0b110, 0b001, 0b011, 0b100], 3) == (0b110, 0b001)
    assert q2_violation([0b001, 0b110, 0b011, 0b100], 3) == (0b001, 0b110)
    assert q2_violation([0b100, 0b011], 3) == (0b100, 0b011)


def test_a_single_unqualified_set_equal_to_the_full_set_is_a_violation():
    assert q2_violation([0b111], 3) == (0b111, 0b111)
    # a pair is taken in the order of the sequence, the first set paired with itself and with the sets after it
    assert q2_violation([0b001, 0b111], 3) == (0b001, 0b111)
    assert q2_violation([0b111, 0b001], 3) == (0b111, 0b111)
    assert q2_violation([0b1], 1) == (0b1, 0b1)


def test_nothing_unqualified_is_no_violation():
    assert q2_violation([], 3) is None
    assert q2_violation((), 1) is None


def test_the_empty_set_alone_is_no_violation():
    for n in range(1, MAX_HOLDERS + 1):
        assert q2_violation([0], n) is None


def test_q2_accepts_any_sequence_of_sets():
    assert q2_violation((0b01, 0b10), 2) == (0b01, 0b10)
    assert q2_violation(range(3), 2) == (1, 2)


def test_q2_returns_a_tuple_of_the_given_ints():
    pair = q2_violation([0b0011, 0b1100], 4)
    assert type(pair) is tuple and pair == (0b0011, 0b1100)
    assert all(type(m) is int for m in pair)


def test_q2_agrees_with_a_search_of_all_pairs():
    rng = random.Random(20261031)
    violations = 0
    for _ in range(2000):
        n = rng.randint(1, 5)
        sets = [rng.randrange(1 << n) for _ in range(rng.randint(0, 5))]
        full = (1 << n) - 1
        expected = next(((a, b) for i, a in enumerate(sets) for b in sets[i:] if a | b == full), None)
        assert q2_violation(sets, n) == expected
        violations += expected is not None
    assert 200 < violations < 1800


@pytest.mark.parametrize("n", [0, MAX_HOLDERS + 1, -1, 2 ** 40, True, False, 3.0, "3", None, _MyInt(3)])
def test_q2_refuses_a_bad_number_of_holders(n):
    with pytest.raises(FEValidationError, match=r"\bn\b"):
        q2_violation([0b01], n)


@pytest.mark.parametrize("bad", [-1, 8, 2 ** 40, True, False, 1.0, "1", None, _MyInt(1), b"\x01", (1,)])
def test_q2_refuses_a_set_that_is_not_an_int_in_range(bad):
    with pytest.raises(FEValidationError, match="unqualified"):
        q2_violation([0b001, bad], 3)


def test_q2_names_the_bad_set_and_the_limit():
    with pytest.raises(FEValidationError, match=r"unqualified set 1 is 8, .*3 holders \(0 to 7\)"):
        q2_violation([0b001, 8], 3)
    with pytest.raises(FEValidationError, match=r"between 1 and 8"):
        q2_violation([], 9)


@pytest.mark.parametrize("bad", [None, 5, 1.5, "01", b"\x01\x02", bytearray(b"\x01"), iter([1, 2]), {1, 2}])
def test_q2_refuses_a_collection_that_is_not_a_sequence_of_ints(bad):
    with pytest.raises(FEValidationError, match="unqualified"):
        q2_violation(bad, 3)


def test_q2_refuses_a_bad_set_even_after_a_violation_was_found():
    # the whole input is checked first, so a bad entry after a covering pair is not skipped
    with pytest.raises(FEValidationError, match="unqualified"):
        q2_violation([0b01, 0b10, 8], 2)


# ---------------------------------------------------------------------------
# what the families are given
# ---------------------------------------------------------------------------

def test_too_many_holders():
    with pytest.raises(FEValidationError, match="8"):
        qualified_families(Leaf("h:0"), [{"h:0"}] * 9)


def test_exactly_the_limit_is_accepted():
    minimal, maximal = qualified_families(Leaf("h:0"), [{"h:0"}] * MAX_HOLDERS)
    assert len(minimal) == MAX_HOLDERS and maximal == [0]


def test_no_holders():
    with pytest.raises(FEValidationError, match="at least one holder"):
        qualified_families(Leaf("h:0"), [])
    with pytest.raises(FEValidationError, match="at least one holder"):
        qualified_families(Leaf("h:0"), ())


@pytest.mark.parametrize("holder", ["h:0", b"h:0", "", b""])
def test_a_holder_that_is_a_bare_string_is_refused(holder):
    with pytest.raises(FEValidationError, match="holder 1"):
        qualified_families(Leaf("h:0"), [{"h:0"}, holder])


@pytest.mark.parametrize("holder", [5, None, 1.5, [["h:0"]], [{"h:0"}]])
def test_a_holder_that_is_not_an_iterable_of_labels_is_refused(holder):
    with pytest.raises(FEValidationError, match="holder 0"):
        qualified_families(Leaf("h:0"), [holder])


@pytest.mark.parametrize(
    "holders", [None, 5, "h:0", b"h:0", {frozenset({"h:0"})}, {"h:0": {"h:0"}}, (h for h in [{"h:0"}])]
)
def test_holders_that_are_not_a_sequence_are_refused(holders):
    with pytest.raises(FEValidationError, match="holders"):
        qualified_families(Leaf("h:0"), holders)


def test_the_formula_is_validated_first():
    # two holders' worth of trouble: the formula is the one reported
    shared = Threshold(2, Leaf("h:0"), Leaf("h:0"))
    with pytest.raises(FEValidationError, match="'h:0'"):
        qualified_families(shared, [])
    with pytest.raises(FEValidationError, match="'h:0'"):
        qualified_families(shared, [{"h:0"}] * 9)


def test_a_formula_deeper_than_the_limit_is_refused():
    formula = Leaf("h:0")
    for _ in range(MAX_DEPTH + 1):
        formula = Or(formula, Leaf("h:1"))
    with pytest.raises(FEValidationError, match="depth"):
        qualified_families(formula, [{"h:0"}])


@pytest.mark.parametrize("formula", ["h:0", None, 5, {"h:0"}])
def test_something_that_is_not_a_formula_is_refused(formula):
    with pytest.raises(FEValidationError, match="formula"):
        qualified_families(formula, [{"h:0"}])


# ---------------------------------------------------------------------------
# the definition, read directly, on random formulas and holders
# ---------------------------------------------------------------------------

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


def _qualified(formula, holders, subset):
    held = set()
    for i, labels in enumerate(holders):
        if subset >> i & 1:
            held |= labels
    return evaluate(formula, held)


def test_random_formulas_and_holders_agree_with_the_definition():
    rng = random.Random(20261031)
    pool = ["r:{}".format(c) for c in "abcde"]
    empty_minimal = full_maximal = mixed = 0
    for _ in range(600):
        formula = _random_formula(rng, pool, 3)
        n = rng.randint(1, 6)
        holders = [{label for label in pool if rng.random() < 0.4} for _ in range(n)]
        full = (1 << n) - 1
        minimal, maximal = qualified_families(formula, holders)
        qualified = {s: _qualified(formula, holders, s) for s in range(full + 1)}

        assert minimal == sorted(set(minimal)) and maximal == sorted(set(maximal))
        for s in minimal:
            assert qualified[s]
            assert all(not qualified[s & ~(1 << i)] for i in range(n) if s >> i & 1)
        for s in maximal:
            assert not qualified[s]
            assert all(qualified[s | (1 << i)] for i in range(n) if not s >> i & 1)
        # the families are complete: nothing that has the property is missing
        assert minimal == [
            s for s in range(full + 1)
            if qualified[s] and all(not qualified[s & ~(1 << i)] for i in range(n) if s >> i & 1)
        ]
        assert maximal == [
            s for s in range(full + 1)
            if not qualified[s] and all(qualified[s | (1 << i)] for i in range(n) if not s >> i & 1)
        ]
        # a subset is qualified exactly when it contains a minimal set, unqualified exactly when inside a maximal one
        for s in range(full + 1):
            assert qualified[s] == any(m & s == m for m in minimal)
            assert (not qualified[s]) == any(s & z == s for z in maximal)

        empty_minimal += not minimal
        full_maximal += maximal == [full]
        mixed += bool(minimal) and bool(maximal)
    assert empty_minimal > 5 and full_maximal > 5 and mixed > 200
