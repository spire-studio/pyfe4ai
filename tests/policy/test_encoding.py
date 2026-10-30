"""Text syntax, canonical encoding and digest of policy formulas (pyfe4ai.policy.encoding).

Every non-ASCII string below is written as an escape, so that this file is pure ASCII: no editor, formatter
or paste step can normalize one spelling of a label into another, and look-alike characters can be told apart.
"""

from __future__ import annotations

import hashlib
import itertools
import random

import pytest

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.encoding import POLICY_COMPILER, canonical_encoding, from_canonical, parse, policy_digest
from pyfe4ai.policy.formula import MAX_DEPTH, And, Leaf, Or, Threshold

F = And(Leaf("org:A"), Threshold(2, Leaf("a:1"), Leaf("a:2"), Leaf("a:3")))
TEXT = '["and",["leaf","org:A"],["th",2,["leaf","a:1"],["leaf","a:2"],["leaf","a:3"]]]'

DECOMPOSED_E_ACUTE = "e\N{COMBINING ACUTE ACCENT}"             # "e" + U+0301, not in NFC
PRECOMPOSED_E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"    # U+00E9, in NFC


class _MyStr(str):
    """A str subclass."""


def render(formula) -> str:
    """The text syntax of *formula*, with some whitespace."""
    if isinstance(formula, Leaf):
        return formula.label
    if isinstance(formula, Threshold):
        return "{}of({})".format(formula.t, ",".join(render(child) for child in formula.children))
    name = "and" if isinstance(formula, And) else "or"
    return "{}( {} )".format(name, " , ".join(render(child) for child in formula.children))


def nested(depth: int):
    """A formula of *depth* nested gates (and a lone leaf at the bottom of each)."""
    formula = And(Leaf("a:1"), Leaf("a:2"))
    for _ in range(depth - 1):
        formula = And(formula, Leaf("a:3"))
    return formula


def nested_json(depth: int) -> str:
    """The canonical JSON of ``nested(depth)``, written without building the formula."""
    text = '["and",["leaf","a:1"],["leaf","a:2"]]'
    for _ in range(depth - 1):
        text = '["and",' + text + ',["leaf","a:3"]]'
    return text


def nested_text(depth: int) -> str:
    """The text syntax of ``nested(depth)``, written without building the formula."""
    return "and(" * (depth - 1) + "and(a:1,a:2)" + ",a:3)" * (depth - 1)


def random_formula(rng: random.Random, labels, depth: int):
    """A random formula with fresh labels, so that no threshold gate shares one."""
    if depth == 0 or rng.random() < 0.3:
        label = next(labels)
        return Leaf(rng.choice((label, label + '"\\', label + PRECOMPOSED_E_ACUTE)))
    kind = rng.choice(("and", "or", "th"))
    children = [random_formula(rng, labels, depth - 1) for _ in range(rng.randint(2, 4))]
    if kind == "th":
        return Threshold(rng.randint(1, len(children)), *children)
    return And(*children) if kind == "and" else Or(*children)


# ---------------------------------------------------------------------------
# canonical encoding and digest
# ---------------------------------------------------------------------------

def test_policy_compiler_marker():
    assert POLICY_COMPILER == "c1"


def test_canonical_known_answer():
    assert canonical_encoding(F) == TEXT
    assert policy_digest(F) == "c8b23c796fb31895db7f46b601c9962af3ca580dfb8b231c528cb26da5d72eb3"


def test_digest_is_the_domain_separated_shake256_of_the_utf8_encoding():
    formula = Or(Leaf("org:" + PRECOMPOSED_E_ACUTE), Leaf("a:1"))
    encoding = canonical_encoding(formula)
    assert PRECOMPOSED_E_ACUTE in encoding                      # raw UTF-8, not a \u escape
    expected = hashlib.shake_256(b"PYFE4AI-V05-POLICY" + encoding.encode("utf-8")).hexdigest(32)
    assert policy_digest(formula) == expected
    assert len(expected) == 64 and set(expected) <= set("0123456789abcdef")


def test_canonical_node_shapes_and_child_order():
    assert canonical_encoding(Leaf("a:1")) == '["leaf","a:1"]'
    assert canonical_encoding(Or(Leaf("a:1"), Leaf("a:2"))) == '["or",["leaf","a:1"],["leaf","a:2"]]'
    assert canonical_encoding(Or(Leaf("a:2"), Leaf("a:1"))) == '["or",["leaf","a:2"],["leaf","a:1"]]'
    assert canonical_encoding(Threshold(1, Leaf("a:1"))) == '["th",1,["leaf","a:1"]]'


def test_canonical_escapes_quote_and_backslash():
    assert canonical_encoding(Leaf('a:q"b\\c')) == '["leaf","a:q\\"b\\\\c"]'


def test_digest_tells_formulas_apart():
    digests = {
        policy_digest(f)
        for f in (
            F,
            Leaf("a:1"),
            Leaf("a:2"),
            And(Leaf("a:1"), Leaf("a:2")),
            And(Leaf("a:2"), Leaf("a:1")),
            Or(Leaf("a:1"), Leaf("a:2")),
            Threshold(1, Leaf("a:1"), Leaf("a:2")),
            Threshold(2, Leaf("a:1"), Leaf("a:2")),
        )
    }
    assert len(digests) == 8


def test_encoding_and_digest_validate_the_formula_first():
    bad = Threshold(2, Leaf("a:1"), Leaf("a:1"))
    for function in (canonical_encoding, policy_digest):
        with pytest.raises(FEValidationError, match="a:1"):
            function(bad)
        with pytest.raises(FEValidationError):
            function(None)
        with pytest.raises(FEValidationError):
            function("a:1")


def test_encoding_refuses_a_formula_deeper_than_the_limit():
    deep = Leaf("a:1")
    for _ in range(MAX_DEPTH + 1):
        deep = Or(deep, Leaf("a:2"))
    for function in (canonical_encoding, policy_digest):
        with pytest.raises(FEValidationError, match="depth"):
            function(deep)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

def test_parse_and_round_trip():
    assert parse("and(org:A, 2of(a:1,a:2,  a:3))") == F
    assert from_canonical(canonical_encoding(F)) == F


def test_parse_a_lone_leaf_and_whitespace():
    assert parse("org:A") == Leaf("org:A")
    assert parse("  \t\n org:A \r\n") == Leaf("org:A")
    assert parse(" and(a:1,a:2)") == And(Leaf("a:1"), Leaf("a:2"))
    assert parse("  and( a:1 ,\n\ta:2 ,  or( a:3 , a:4 ) )  ") == And(
        Leaf("a:1"), Leaf("a:2"), Or(Leaf("a:3"), Leaf("a:4"))
    )
    assert parse("1of(a:1)") == Threshold(1, Leaf("a:1"))
    assert parse("10of(" + ",".join("a:{}".format(i) for i in range(10)) + ")") == Threshold(
        10, *[Leaf("a:{}".format(i)) for i in range(10)]
    )


def test_parse_reads_labels_with_special_characters():
    label = "org:" + PRECOMPOSED_E_ACUTE + '"\\[]{}:'
    assert parse("or({}, a:1)".format(label)) == Or(Leaf(label), Leaf("a:1"))


@pytest.mark.parametrize("text", [
    "and(org:A)", "or()", "3of(a:1, a:2)", "0of(a:1)", "and(org:A", "and(org:A, a:1))", "not(org:A)", "",
    "org:" + DECOMPOSED_E_ACUTE,
    "   ", "and(", "and()", "and(a:1,)", "and(,a:1)", "and(a:1,,a:2)", "or(a:1)", "2of(a:1)",
    "a:1 a:2", "and(a:1 a:2)", "and(a:1, a:2) a:3", "and(a:1, a:2)(", "(a:1)", "()", "and((a:1),a:2)",
    "and a:1, a:2", "and (a:1, a:2)", "2 of(a:1, a:2)", "1of (a:1)",
    "nocolon", "a:", ":1", "and(a:1, bad)", "and(a:1, a:2#3)", "and(a:1, a:2\x00)", "and(a:1, a:2\x0b)",
    "and(a:1,\N{NO-BREAK SPACE}a:2)", "and(a:1, a:2\N{ZERO WIDTH SPACE})",
    "2of(a:1, a:1)", "and(a:1, 2of(a:2, and(a:3, a:2), a:4), 2of(a:5, a:5))",
])
def test_parse_errors(text):
    with pytest.raises(FEValidationError, match="position"):
        parse(text)


@pytest.mark.parametrize("text", [
    "02of(a:1,a:2)", "+2of(a:1,a:2)", "-1of(a:1)", "00of(a:1)", "2.0of(a:1,a:2)", "1_0of(a:1,a:2)", "0x2of(a:1,a:2)",
    "\N{ARABIC-INDIC DIGIT TWO}of(a:1,a:2)", "\N{FULLWIDTH DIGIT TWO}of(a:1,a:2)", "\N{SUPERSCRIPT TWO}of(a:1,a:2)",
    "1\N{ARABIC-INDIC DIGIT TWO}of(a:1,a:2)", "2\N{FULLWIDTH DIGIT ZERO}of(a:1,a:2)", "1\N{SUPERSCRIPT TWO}of(a:1)",
    "2OF(a:1,a:2)", "2Of(a:1,a:2)", "2of2(a:1,a:2)", "of(a:1,a:2)",
    "AND(a:1,a:2)", "And(a:1,a:2)", "OR(a:1,a:2)", "Or(a:1,a:2)", "and1(a:1,a:2)", "xand(a:1,a:2)",
])
def test_parse_token_spelling_is_exact(text):
    with pytest.raises(FEValidationError, match="position 0"):
        parse(text)


def test_parse_refuses_non_ascii_digits_that_would_make_a_satisfiable_threshold():
    # int() reads these as digits, and the gate below would be satisfiable: only the spelling is wrong
    children = ",".join("a:{}".format(i) for i in range(10))
    assert parse("10of(" + children + ")") == Threshold(10, *[Leaf("a:{}".format(i)) for i in range(10)])
    for zero in ("\N{ARABIC-INDIC DIGIT ZERO}", "\N{FULLWIDTH DIGIT ZERO}", "\N{DEVANAGARI DIGIT ZERO}"):
        with pytest.raises(FEValidationError, match="position 0"):
            parse("1" + zero + "of(" + children + ")")
    for one in ("\N{ARABIC-INDIC DIGIT ONE}", "\N{FULLWIDTH DIGIT ONE}", "\N{DEVANAGARI DIGIT ONE}"):
        with pytest.raises(FEValidationError, match="position 0"):
            parse(one + "of(a:1)")


def test_parse_huge_threshold_is_a_validation_error():
    for digits in ("9" * 5000, "9" * 100, "1" + "0" * 30):
        with pytest.raises(FEValidationError, match="position 0") as info:
            parse("{}of(a:1,a:2)".format(digits))
        assert len(str(info.value)) < 400                      # the digit string is not echoed


def test_parse_reports_the_position():
    cases = [
        ("", 0),
        ("   ", 3),
        ("and(a:1,)", 8),
        ("and(,a:1)", 4),
        ("and(org:A", 9),
        ("and(org:A, a:1))", 15),
        ("not(org:A)", 0),
        ("  3of(a:1,a:2)", 2),
        ("and(a:1, bad)", 9),
        ("and(a:1, or(a:2))", 9),
        ("and(a:1 a:2)", 8),
        ("and(a:1, a:2) a:3", 14),
        ("2of(a:1, a:1)", 0),
    ]
    for text, position in cases:
        with pytest.raises(FEValidationError, match=r"position {}:".format(position)):
            parse(text)


def test_parse_names_what_is_wrong():
    with pytest.raises(FEValidationError, match="at least two"):
        parse("and(a:1)")
    with pytest.raises(FEValidationError, match="1 <= t"):
        parse("3of(a:1, a:2)")
    with pytest.raises(FEValidationError, match="'bad'"):
        parse("and(a:1, bad)")
    with pytest.raises(FEValidationError, match="'a:1'"):
        parse("2of(a:1, a:1)")
    with pytest.raises(FEValidationError, match="NFC"):
        parse("org:" + DECOMPOSED_E_ACUTE)


@pytest.mark.parametrize("value", [None, b"and(a:1,a:2)", 5, ["a:1"], _MyStr("a:1")])
def test_parse_refuses_a_non_str(value):
    with pytest.raises(FEValidationError, match="str"):
        parse(value)


def test_parse_depth_agrees_with_validate_formula():
    assert parse(nested_text(MAX_DEPTH)) == nested(MAX_DEPTH)
    with pytest.raises(FEValidationError, match="depth") as info:
        parse(nested_text(MAX_DEPTH + 1))
    assert "position {}:".format(len("and(") * MAX_DEPTH) in str(info.value)     # the 33rd gate
    for gate in ("or(", "2of(", "1of("):
        text = gate * (MAX_DEPTH + 5)
        with pytest.raises(FEValidationError, match="depth"):
            parse(text)


def test_parse_hostile_nesting_is_refused():
    with pytest.raises(FEValidationError, match="depth"):
        parse("and(" * 100000)
    with pytest.raises(FEValidationError, match="depth"):
        parse("2of(a:1, " * 100000)
    with pytest.raises(FEValidationError):
        parse("(" * 100000)
    with pytest.raises(FEValidationError):
        parse("and(a:1," * 100000)


def test_parse_wide_gates():
    n = 5000
    text = "or({})".format(",".join("a:{}".format(i) for i in range(n)))
    assert parse(text) == Or(*[Leaf("a:{}".format(i)) for i in range(n)])
    assert parse("{}of({})".format(n, text[3:-1])) == Threshold(n, *[Leaf("a:{}".format(i)) for i in range(n)])


# ---------------------------------------------------------------------------
# from_canonical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data", [
    '["leaf", "a:1"]', '["leaf","\\u0061:1"]', '["th",0,["leaf","a:1"]]',
    '["or",["leaf","a:1"]]', '["x"]', '{}', 'null',
])
def test_non_canonical_is_refused(data):
    with pytest.raises(FEValidationError):
        from_canonical(data)


@pytest.mark.parametrize("data", [
    # not JSON, or JSON with something around the node
    "", " ", "[", '["leaf","a:1"', '["leaf","a:1"]]', '["leaf","a:1"]x', '["leaf","a:1"] ', ' ["leaf","a:1"]',
    '["leaf","a:1"]\n', "\ufeff" + '["leaf","a:1"]', "NaN", "Infinity", "'x'",
    # another spelling of the same node
    '[ "leaf","a:1"]', '["leaf" ,"a:1"]', '["leaf","a\\u003a1"]', '["leaf","a:\\/1"]', '["leaf","a:\\u00e91"]',
    # not a node
    "[]", "{}", "null", "true", "1", '"a:1"', '{"leaf":"a:1"}', '[["leaf","a:1"]]', "[1]", "[null]", "[[]]", "[{}]",
    '["Leaf","a:1"]', '["LEAF","a:1"]', '["and "]', '["x"]', '["th"]', '["or"]', '["and"]', '["and",["leaf","a:1"]]',
    '["leaf"]', '["leaf","a:1","a:2"]', '["leaf",1]', '["leaf",null]', '["leaf",["a:1"]]', '["leaf",["leaf","a:1"]]',
    '["leaf",{"a":1}]', '["leaf",true]',
    '["and",["leaf","a:1"],"a:2"]', '["and",["leaf","a:1"],null]', '["and",["leaf","a:1"],[]]',
    '["or",["leaf","a:1"],["x"]]', '["and",["leaf","a:1"],["leaf","a:2"],["and"]]',
    # a bad threshold
    '["th",1]', '["th"]', '["th",["leaf","a:1"]]', '["th",0,["leaf","a:1"]]', '["th",-1,["leaf","a:1"]]',
    '["th",2,["leaf","a:1"]]', '["th",3,["leaf","a:1"],["leaf","a:2"]]', '["th","2",["leaf","a:1"],["leaf","a:2"]]',
    '["th",null,["leaf","a:1"]]', '["th",[2],["leaf","a:1"],["leaf","a:2"]]',
    '["th",true,["leaf","a:1"]]', '["th",false,["leaf","a:1"]]', '["th",1.0,["leaf","a:1"]]',
    '["th",1e0,["leaf","a:1"]]', '["th",2.0,["leaf","a:1"],["leaf","a:2"]]', '["th",1.5,["leaf","a:1"]]',
    '["th",01,["leaf","a:1"]]', '["th",+1,["leaf","a:1"]]', '["th",1,["leaf","a:1"],["leaf","a:1"]]',
    '["th",-0,["leaf","a:1"]]', '["th",NaN,["leaf","a:1"]]', '["th",Infinity,["leaf","a:1"]]',
    # a bad label
    '["leaf",""]', '["leaf","nocolon"]', '["leaf","a:"]', '["leaf",":1"]', '["leaf","a:1 2"]', '["leaf","a:1,2"]',
    '["leaf","a:1(2"]', '["leaf","a:1#2"]', '["leaf","a:\\u0000"]', '["leaf","a:\\ud800"]',
    '["leaf","a:\\u200b"]', '["leaf","a:\\n"]',
    '["leaf","a:' + DECOMPOSED_E_ACUTE + '"]',
    # a threshold gate whose children share a label
    '["th",2,["leaf","a:1"],["leaf","a:1"]]', '["th",2,["and",["leaf","a:1"],["leaf","a:2"]],["leaf","a:1"]]',
])
def test_from_canonical_refuses(data):
    with pytest.raises(FEValidationError):
        from_canonical(data)


def test_from_canonical_names_what_is_wrong():
    with pytest.raises(FEValidationError, match="NFC"):
        from_canonical('["leaf","a:' + DECOMPOSED_E_ACUTE + '"]')
    with pytest.raises(FEValidationError, match="'a:1'"):
        from_canonical('["th",2,["leaf","a:1"],["leaf","a:1"]]')
    with pytest.raises(FEValidationError, match=r"root\[2\]\[2\]"):
        from_canonical('["and",["leaf","a:1"],["or",["leaf","a:2"],1]]')
    with pytest.raises(FEValidationError, match="canonical"):
        from_canonical('["leaf", "a:1"]')
    with pytest.raises(FEValidationError, match="position 8"):                # first difference from the canonical text
        from_canonical('["leaf",  "a:1"]')
    with pytest.raises(FEValidationError, match="depth"):
        from_canonical(nested_json(MAX_DEPTH + 1))


@pytest.mark.parametrize("value", [
    None, b'["leaf","a:1"]', bytearray(b'["leaf","a:1"]'), 5, ["leaf", "a:1"], _MyStr('["leaf","a:1"]'),
])
def test_from_canonical_refuses_a_non_str(value):
    with pytest.raises(FEValidationError, match="str"):
        from_canonical(value)


def test_huge_integer_is_a_validation_error():
    for digits in ("9" * 5000, "9" * 100, "1" + "0" * 30):
        with pytest.raises(FEValidationError):
            from_canonical('["th",{},["leaf","a:1"]]'.format(digits))
        with pytest.raises(FEValidationError):
            from_canonical('["leaf",{}]'.format(digits))
    with pytest.raises(FEValidationError):
        from_canonical('["th",' + "9" * 100000 + ',["leaf","a:1"]]')


def test_boolean_and_float_thresholds_are_refused():
    for t in ("true", "1.0"):
        with pytest.raises(FEValidationError):
            from_canonical('["th",{},["leaf","a:1"]]'.format(t))
    assert from_canonical('["th",1,["leaf","a:1"]]') == Threshold(1, Leaf("a:1"))


def test_hostile_nesting_is_refused():      # Review Focus 1
    for data in ("[" * 100000, '["and",' * 100000, '{"a":' * 100000, "{" * 100000, '["th",1,' * 100000):
        with pytest.raises(FEValidationError, match="nest"):
            from_canonical(data)
    with pytest.raises(FEValidationError):
        parse("and(" * 100000)


def test_nesting_scan_counts_brackets_and_braces_and_stops_at_twice_the_depth_limit():
    limit = 2 * MAX_DEPTH
    for opener, closer in (("[", "]"), ("{", "}"), ('["and",', "]")):
        data = opener * (limit + 1) + closer * (limit + 1)
        with pytest.raises(FEValidationError, match="nest"):
            from_canonical(data)
    # within the limit the scan passes, and the shape check says what is wrong
    for data in ("[" * limit + "]" * limit, '{"a":' * limit + "1" + "}" * limit):
        with pytest.raises(FEValidationError) as info:
            from_canonical(data)
        assert "nest" not in str(info.value)
    # a mix of both brackets counts together
    with pytest.raises(FEValidationError, match="nest"):
        from_canonical("[{" * MAX_DEPTH + "[")


def test_nesting_scan_skips_the_inside_of_strings():
    brackets = "[" * 100 + "{" * 100
    label = "a:" + brackets
    formula = Or(Leaf(label), Leaf("b:" + "]" * 100))
    data = canonical_encoding(formula)
    assert from_canonical(data) == formula
    # 75 brackets inside one label: more than 2 * MAX_DEPTH, none of them counted
    assert from_canonical('["leaf","a:' + "[" * 75 + '"]') == Leaf("a:" + "[" * 75)


def test_nesting_scan_honours_backslash_escapes():
    # an escaped quote does not end the string, so the brackets after it are still inside it
    label = 'a:"' + "[" * 100
    assert from_canonical(canonical_encoding(Leaf(label))) == Leaf(label)
    label = 'a:\\"' + "[" * 100 + '\\'
    assert from_canonical(canonical_encoding(Leaf(label))) == Leaf(label)
    # an escaped backslash does not escape the quote after it: the string ends there, and what
    # follows is counted
    with pytest.raises(FEValidationError, match="nest"):
        from_canonical('["leaf","a:\\\\"' + "[" * 100)
    with pytest.raises(FEValidationError, match="nest"):
        from_canonical('["leaf","a:\\\\\\\\"' + "[" * 100)
    # the other way round: an escaped quote at the end of a string keeps it open, past the brackets
    with pytest.raises(FEValidationError) as info:
        from_canonical('["leaf","a:\\"' + "[" * 100)
    assert "nest" not in str(info.value)


def test_nesting_scan_handles_unterminated_and_odd_strings():
    for data in ('"', '"\\', '["leaf","a:1', '["leaf","a:\\', '"' * 99999, '"\\"' * 99999, "\\" * 99999):
        with pytest.raises(FEValidationError):
            from_canonical(data)


def test_depth_limit_round_trips_at_the_limit_and_refuses_above():
    deepest = nested(MAX_DEPTH)
    data = canonical_encoding(deepest)
    assert data == nested_json(MAX_DEPTH)
    assert from_canonical(data) == deepest
    assert policy_digest(from_canonical(data)) == policy_digest(deepest)
    with pytest.raises(FEValidationError, match="depth"):
        from_canonical(nested_json(MAX_DEPTH + 1))
    with pytest.raises(FEValidationError):
        from_canonical(nested_json(2 * MAX_DEPTH))


def test_round_trip_of_special_labels():
    labels = ["a:1", "org:" + PRECOMPOSED_E_ACUTE, 'a:q"b', "a:b\\c", "a:\\", 'a:"', "a:/", "a:'", "a:[]{}",
              "a:\N{CJK UNIFIED IDEOGRAPH-4E2D}", "a:\N{GRINNING FACE}", "a:\N{LATIN SMALL LETTER SHARP S}",
              "x.y-z_1:" + "v" * 1000]
    for label in labels:
        assert from_canonical(canonical_encoding(Leaf(label))) == Leaf(label)
        assert parse(label) == Leaf(label)
    formula = Or(*[Leaf(label) for label in labels])
    assert from_canonical(canonical_encoding(formula)) == formula


def test_random_formulas_round_trip():
    rng = random.Random(20261031)
    labels = ("ns:v{}".format(i) for i in itertools.count())
    seen = set()
    for _ in range(300):
        formula = random_formula(rng, labels, 5)
        data = canonical_encoding(formula)
        assert from_canonical(data) == formula
        assert canonical_encoding(from_canonical(data)) == data
        assert parse(render(formula)) == formula
        assert policy_digest(from_canonical(data)) == policy_digest(formula)
        seen.add(policy_digest(formula))
    assert len(seen) == 300


def test_from_canonical_does_not_mutate_or_alias_its_result():
    first, second = from_canonical(TEXT), from_canonical(TEXT)
    assert first == second == F
    assert first is not second
