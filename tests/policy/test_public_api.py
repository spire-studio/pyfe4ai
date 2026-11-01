"""The public names of pyfe4ai.policy.

The tests check that every name in ``__all__`` can be imported from the package, that the list is the agreed one,
and that a policy can be parsed, compiled, reconstructed from and digested using nothing but those names. Every
string is ASCII.
"""

from __future__ import annotations

import random

import pyfe4ai.policy as policy
from pyfe4ai.policy import (
    ATTR_HASH,
    DEFAULT_MAX_ROWS,
    MAX_DEPTH,
    MAX_HOLDERS,
    MSP,
    POLICY_COMPILER,
    And,
    Formula,
    Leaf,
    Or,
    Threshold,
    Universe,
    attribute_set_digest,
    attribute_set_encoding,
    attribute_to_zq,
    canonical_encoding,
    compile_formula,
    compile_zero_one,
    evaluate,
    from_canonical,
    parse,
    policy_digest,
    q2_violation,
    qualified_families,
    read_once_rows,
    threshold_to_cnf,
    validate_formula,
    validate_label,
)

#: The order of the BLS12-381 groups, a 255-bit prime.
P = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001

EXPECTED = [
    "ATTR_HASH", "And", "DEFAULT_MAX_ROWS", "Formula", "Leaf", "MAX_DEPTH", "MAX_HOLDERS", "MSP", "Or",
    "POLICY_COMPILER", "Threshold", "Universe", "attribute_set_digest", "attribute_set_encoding", "attribute_to_zq",
    "canonical_encoding", "compile_formula", "compile_zero_one", "evaluate", "from_canonical", "parse",
    "policy_digest", "q2_violation", "qualified_families", "read_once_rows", "threshold_to_cnf", "validate_formula",
    "validate_label",
]


def test_all_is_the_agreed_list():
    assert sorted(policy.__all__) == sorted(EXPECTED)
    assert len(policy.__all__) == len(set(policy.__all__))


def test_every_name_in_all_is_importable_from_the_package():
    for name in policy.__all__:
        assert hasattr(policy, name), name
    namespace: dict = {}
    exec("from pyfe4ai.policy import *", namespace)
    assert {name for name in namespace if name != "__builtins__"} == set(policy.__all__)


def test_the_constants_have_the_agreed_values():
    assert DEFAULT_MAX_ROWS == 64
    assert MAX_DEPTH == 32
    assert MAX_HOLDERS == 8
    assert POLICY_COMPILER == "c1"
    assert ATTR_HASH == "shake256-64"


def test_parse_compile_reconstruct_and_digest():
    formula = parse("and(org:A, 2of(role:x, role:y, role:z))")
    assert isinstance(formula, Formula)
    assert formula == And(Leaf("org:A"), Threshold(2, Leaf("role:x"), Leaf("role:y"), Leaf("role:z")))
    validate_formula(formula)

    msp = compile_formula(formula, P)
    assert isinstance(msp, MSP) and msp.modulus == P and msp.row_labels == tuple(formula.leaves())

    rng = random.Random(7)
    secret = rng.randrange(P)
    shares = msp.shares(secret, [rng.randrange(P) for _ in range(msp.num_cols - 1)])
    held = {"org:A", "role:x", "role:z"}
    assert evaluate(formula, held)
    weights = msp.reconstruction(held)
    assert weights is not None
    assert sum(weight * shares[row] for row, weight in weights.items()) % P == secret
    assert msp.reconstruction({"org:A", "role:x"}) is None
    assert not evaluate(formula, {"org:A", "role:x"})

    digest = policy_digest(formula)
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert from_canonical(canonical_encoding(formula)) == formula
    assert policy_digest(from_canonical(canonical_encoding(formula))) == digest
    # the digest does not depend on how the text was spaced
    assert policy_digest(parse("and(org:A,2of(role:x,role:y,role:z))")) == digest
    assert policy_digest(Leaf("org:A")) != digest


def test_the_other_names_work_together():
    # labels and universes
    assert validate_label("org:A") == "org:A"
    assert 0 <= attribute_to_zq("org:A", P) < P
    formula = Or(And(Leaf("org:A"), Leaf("org:B")), And(Leaf("org:A"), Leaf("role:x")))
    assert parse("or(and(org:A, org:B), and(org:A, role:x))") == formula
    universe = Universe(formula.leaves(), 2)
    assert "org:A" in universe and universe.size == 6

    # the set of labels a client holds, however it is spelled
    assert attribute_set_encoding({"role:x", "org:A"}) == '["org:A","role:x"]'
    assert attribute_set_digest({"role:x", "org:A"}) == attribute_set_digest(["org:A", "role:x", "org:A"])

    # read-once copies of the rows of a compiled policy
    msp = compile_formula(formula, P)
    assert read_once_rows(msp, universe.max_copies) == (
        ("org:A", 1), ("org:B", 1), ("org:A", 2), ("role:x", 1))

    # a policy with no threshold gate reconstructs with the coefficient 1 on every row
    cnf = threshold_to_cnf(parse("2of(a:1, a:2, a:3)"))
    assert evaluate(cnf, {"a:1", "a:3"}) and not evaluate(cnf, {"a:2"})
    zero_one = compile_zero_one(parse("2of(a:1, a:2, a:3)"), 2 ** 64)
    assert set(zero_one.reconstruction({"a:1", "a:2"}).values()) == {1}

    # who can open it, and whether two groups that cannot can still cover everyone
    minimal, maximal = qualified_families(parse("2of(h:0, h:1, h:2)"), [{"h:0"}, {"h:1"}, {"h:2"}])
    assert minimal == [0b011, 0b101, 0b110] and maximal == [0b001, 0b010, 0b100]
    assert q2_violation(maximal, 3) is None
    assert q2_violation([0b011, 0b100], 3) == (0b011, 0b100)
