"""Monotone access formulas and attribute labels (pyfe4ai.policy.formula, pyfe4ai.policy.attributes).

Every non-ASCII string below is written as an escape, so that this file is pure ASCII: no editor, formatter
or paste step can normalize one spelling of a label into another, and look-alike characters can be told apart.
"""

from __future__ import annotations

import dataclasses
import time
import unicodedata

import pytest

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.attributes import validate_label
from pyfe4ai.policy.formula import (
    MAX_DEPTH,
    And,
    Formula,
    Leaf,
    Or,
    Threshold,
    evaluate,
    held_labels,
    leaf_count,
    validate_formula,
)

DECOMPOSED_E_ACUTE = "e\N{COMBINING ACUTE ACCENT}"             # "e" + U+0301, not in NFC
PRECOMPOSED_E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"    # U+00E9, in NFC


class _MyStr(str):
    """A str subclass."""


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_evaluate_gates():
    f = And(Leaf("org:A"), Threshold(2, Leaf("a:1"), Leaf("a:2"), Leaf("a:3")))
    assert evaluate(f, {"org:A", "a:1", "a:3"})
    assert not evaluate(f, {"org:A", "a:1"})
    assert not evaluate(f, {"a:1", "a:2", "a:3"})


def test_evaluate_or_and_threshold_boundaries():
    f = Or(Leaf("a:1"), And(Leaf("b:1"), Leaf("b:2")))
    assert evaluate(f, ["a:1"])
    assert evaluate(f, ["b:1", "b:2"])
    assert not evaluate(f, ["b:1"])
    assert not evaluate(f, [])
    for t, expected in ((1, True), (2, True), (3, False)):
        g = Threshold(t, Leaf("a:1"), Leaf("a:2"), Leaf("a:3"))
        assert evaluate(g, {"a:1", "a:3"}) is expected


def test_evaluate_is_monotone_and_ignores_other_labels():
    f = And(Leaf("org:A"), Or(Leaf("a:1"), Leaf("a:2")))
    assert evaluate(f, {"org:A", "a:2", "z:9", "not a label"})
    assert evaluate(f, iter(["org:A", "a:1"]))      # any iterable of labels
    assert evaluate(Leaf("a:1"), {"a:1"})
    assert not evaluate(Leaf("a:1"), set())


@pytest.mark.parametrize("labels", ["a:1", b"a:1", 5, [["a:1"]]])
def test_evaluate_refuses_labels_that_are_not_an_iterable_of_labels(labels):
    with pytest.raises(FEValidationError):
        evaluate(Leaf("a:1"), labels)


def test_evaluate_refuses_a_non_formula():
    with pytest.raises(FEValidationError):
        evaluate("a:1", {"a:1"})


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "org", ":A", "org:", "org:A B", "org:A,B",
                                 "org:(A)", "org:A#1", "org:\x07",
                                 "org:" + DECOMPOSED_E_ACUTE,            # not in NFC
                                 "org:\ud800",                           # a lone surrogate (category Cs)
                                 "org:A\N{ZERO WIDTH SPACE}B",           # U+200B, category Cf
                                 "org:A\N{RIGHT-TO-LEFT OVERRIDE}B",     # U+202E, category Cf
                                 "org:A\N{ZERO WIDTH NO-BREAK SPACE}",   # U+FEFF, category Cf
                                 "org:A\N{SOFT HYPHEN}B",                # U+00AD, category Cf
                                 "or g:A",                               # a space in the namespace
                                 "\xf8rg:A",                             # U+00F8 in the namespace
                                 "org\n:A", "org:A\n", "org:A\tB",
                                 "org:A\N{NO-BREAK SPACE}B",             # U+00A0
                                 "org:A\N{EM SPACE}B",                   # U+2003
                                 "org:A\x7f", "org:A\x00"])
def test_bad_labels_are_refused(bad):
    with pytest.raises(FEValidationError):
        validate_label(bad)
    with pytest.raises(FEValidationError):
        Leaf(bad)


@pytest.mark.parametrize("bad", [None, 7, b"org:A", ("org", "A"), ["org:A"], _MyStr("org:A")])
def test_a_label_must_be_a_str(bad):
    # a str subclass is refused too: it could carry a different encoding, hash or equality than its text
    with pytest.raises(FEValidationError, match="str"):
        validate_label(bad)
    with pytest.raises(FEValidationError, match="str"):
        Leaf(bad)


def test_labels_are_kept_as_written():
    # the refused and the accepted spelling of "org:e-acute" are a real pair: NFC maps one onto the other
    assert "org:" + DECOMPOSED_E_ACUTE != "org:" + PRECOMPOSED_E_ACUTE
    assert unicodedata.normalize("NFC", "org:" + DECOMPOSED_E_ACUTE) == "org:" + PRECOMPOSED_E_ACUTE
    assert validate_label("cert:HIPAA-2026") == "cert:HIPAA-2026"
    assert validate_label("org:" + PRECOMPOSED_E_ACUTE) == "org:" + PRECOMPOSED_E_ACUTE
    assert validate_label("Org_1.x-y:Value") == "Org_1.x-y:Value"  # no case folding
    assert validate_label("a:b:c") == "a:b:c"                    # the namespace ends at the first ":"
    assert validate_label("a:b=c/d@e") == "a:b=c/d@e"
    assert Leaf("org:" + PRECOMPOSED_E_ACUTE).label == "org:" + PRECOMPOSED_E_ACUTE


def test_the_refusal_names_the_label():
    with pytest.raises(FEValidationError, match="org:A,B"):
        validate_label("org:A,B")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [lambda: And(Leaf("a:1")), lambda: Or(),
                                  lambda: And(), lambda: Or(Leaf("a:1")),
                                  lambda: Threshold(0, Leaf("a:1"), Leaf("a:2")),
                                  lambda: Threshold(3, Leaf("a:1"), Leaf("a:2")),
                                  lambda: Threshold(-1, Leaf("a:1"), Leaf("a:2")),
                                  lambda: Threshold(1),
                                  lambda: Threshold(True, Leaf("a:1"), Leaf("a:2")),
                                  lambda: Threshold(1.0, Leaf("a:1"), Leaf("a:2")),
                                  lambda: Threshold("1", Leaf("a:1"), Leaf("a:2")),
                                  lambda: Threshold(None, Leaf("a:1"), Leaf("a:2"))])
def test_gate_arity(make):
    with pytest.raises(FEValidationError):
        make()


@pytest.mark.parametrize("make", [lambda: And("a:1", "a:2"),
                                  lambda: Or(Leaf("a:1"), "a:2"),
                                  lambda: Threshold(1, Leaf("a:1"), None),
                                  lambda: And(Leaf("a:1"), [Leaf("a:2")]),
                                  lambda: Threshold(1, "a:1")])
def test_gate_children_must_be_formulas(make):
    with pytest.raises(FEValidationError, match="formula"):
        make()


def test_a_threshold_gate_may_have_one_child():
    f = Threshold(1, Leaf("a:1"))
    validate_formula(f)
    assert evaluate(f, {"a:1"})
    assert not evaluate(f, set())


def test_formulas_are_immutable_values():
    f = And(Leaf("a:1"), Threshold(1, Leaf("a:2"), Leaf("a:3")))
    assert f == And(Leaf("a:1"), Threshold(1, Leaf("a:2"), Leaf("a:3")))
    assert hash(f) == hash(And(Leaf("a:1"), Threshold(1, Leaf("a:2"), Leaf("a:3"))))
    assert f != Or(Leaf("a:1"), Threshold(1, Leaf("a:2"), Leaf("a:3")))
    assert Threshold(1, Leaf("a:1"), Leaf("a:2")) != Threshold(2, Leaf("a:1"), Leaf("a:2"))
    assert isinstance(f.children, tuple)
    assert f.children[1].t == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.children = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        Leaf("a:1").label = "a:2"


def test_leaves_follow_the_formula_order_and_keep_repeats():
    f = Or(And(Leaf("a:1"), Leaf("a:2")), And(Leaf("a:1"), Threshold(1, Leaf("a:3"), Leaf("a:4"))))
    assert f.leaves() == ["a:1", "a:2", "a:1", "a:3", "a:4"]
    assert Leaf("a:1").leaves() == ["a:1"]


# ---------------------------------------------------------------------------
# validate_formula
# ---------------------------------------------------------------------------

def test_threshold_with_a_repeated_leaf_is_refused():
    f = And(Leaf("b:1"), Threshold(2, Leaf("org:A"), Leaf("org:A")))
    with pytest.raises(FEValidationError, match="org:A"):
        validate_formula(f)


def test_a_label_may_repeat_outside_a_threshold():
    validate_formula(Or(And(Leaf("a:1"), Leaf("a:2")), And(Leaf("a:1"), Leaf("a:3"))))
    validate_formula(And(Leaf("a:1"), Threshold(2, Leaf("a:1"), Leaf("a:2"))))
    validate_formula(Threshold(1, And(Leaf("a:1"), Or(Leaf("a:1"), Leaf("a:2")))))   # inside one child


def test_threshold_children_share_no_leaf_at_any_depth():
    def threshold_with(clash):
        return Threshold(2, Leaf("a:1"), And(Leaf("a:2"), Or(Leaf("a:3"), Leaf("a:9"))), clash)

    validate_formula(threshold_with(Leaf("a:4")))
    for clash, label in ((Leaf("a:9"), "a:9"), (And(Leaf("a:5"), Leaf("a:3")), "a:3"),
                         (Or(Leaf("a:1"), Leaf("a:6")), "a:1")):
        with pytest.raises(FEValidationError, match=label):
            validate_formula(threshold_with(clash))
    # two children that are equal subtrees share all their leaves
    with pytest.raises(FEValidationError, match="x:1"):
        validate_formula(Threshold(2, And(Leaf("x:1"), Leaf("x:2")), And(Leaf("x:1"), Leaf("x:2"))))
    # the rule holds at every Threshold gate, however deep
    nested = Or(Leaf("c:1"), And(Leaf("c:2"), Threshold(2, Leaf("c:3"), Or(Leaf("c:4"), Leaf("c:3")))))
    with pytest.raises(FEValidationError, match="c:3"):
        validate_formula(nested)


def test_validate_formula_refuses_a_non_formula():
    with pytest.raises(FEValidationError):
        validate_formula("a:1")


def test_depth_limit():
    f = Leaf("a:1")
    for _ in range(MAX_DEPTH + 1):
        f = And(f, Leaf("a:2"))
    with pytest.raises(FEValidationError, match="depth"):
        validate_formula(f)


def test_depth_limit_is_exact():
    f = Leaf("a:1")
    validate_formula(f)                  # a lone leaf has depth 0
    for _ in range(MAX_DEPTH):
        f = Or(Leaf("a:2"), f)
    validate_formula(f)                  # MAX_DEPTH gates on the longest path
    with pytest.raises(FEValidationError, match="depth"):
        validate_formula(Or(Leaf("a:3"), f))


def test_very_deep_formulas_are_refused_without_recursion():
    f = Leaf("a:1")
    for i in range(20000):
        f = And(f, Leaf("b:{}".format(i)))
    with pytest.raises(FEValidationError, match="depth"):
        validate_formula(f)
    assert len(f.leaves()) == 20001              # leaves() and evaluate() do not recurse either
    assert not evaluate(f, {"a:1"})
    assert evaluate(f, {"a:1"} | {"b:{}".format(i) for i in range(20000)})


def test_shared_subformulas_are_visited_once():
    f = Leaf("a:1")
    for _ in range(MAX_DEPTH):           # 2**32 leaves when written out
        f = And(f, f)
    validate_formula(f)
    assert evaluate(f, {"a:1"})
    assert not evaluate(f, set())
    shared = And(Leaf("a:1"), Leaf("a:2"))
    with pytest.raises(FEValidationError, match="a:1"):
        validate_formula(Threshold(2, shared, shared))


# ---------------------------------------------------------------------------
# leaf_count and held_labels
# ---------------------------------------------------------------------------

def test_leaf_count_is_the_number_of_leaves_written_out():
    a, b, c, d = (Leaf("a:{}".format(i)) for i in range(4))
    shared = Or(a, b)
    for f in (a, And(a, b), Or(a, And(b, c)), Threshold(2, a, And(b, c), d), And(shared, shared),
              Threshold(1, a), Or(a, a, a)):
        assert leaf_count(f) == len(f.leaves())
    assert leaf_count(a) == 1
    assert leaf_count(And(shared, shared)) == 4          # a shared object counts once per occurrence


def test_leaf_count_does_not_list_the_leaves_or_recurse(monkeypatch):
    def refuse(self):
        raise AssertionError("leaves() was called")

    f = Leaf("a:1")
    for _ in range(MAX_DEPTH):           # 2**32 leaves when written out
        f = And(f, f)
    chain = Leaf("a:1")
    for i in range(20000):               # deeper than the interpreter's recursion limit
        chain = And(chain, Leaf("b:{}".format(i)))
    monkeypatch.setattr(Formula, "leaves", refuse)
    started = time.perf_counter()
    assert leaf_count(f) == 2**32
    assert time.perf_counter() - started < 1.0
    assert leaf_count(chain) == 20001


@pytest.mark.parametrize("bad", [None, "a:1", 5, ["a:1"], Formula()])
def test_leaf_count_refuses_what_is_not_a_formula(bad):
    with pytest.raises(FEValidationError):
        leaf_count(bad)


def test_held_labels_is_the_check_evaluate_uses():
    assert held_labels(["a:1", "a:2", "a:1"]) == frozenset({"a:1", "a:2"})
    assert held_labels(iter(("a:1",))) == frozenset({"a:1"})
    assert held_labels(set()) == frozenset()
    for bad in ("a:1", b"a:1", 5, None, [["a:1"]]):
        with pytest.raises(FEValidationError):
            held_labels(bad)
