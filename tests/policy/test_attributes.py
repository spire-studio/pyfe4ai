"""Attribute universes, hashing to Z_q and read-once copies (pyfe4ai.policy.attributes, pyfe4ai.policy.read_once).

The known answers were computed by a separate script that calls hashlib directly. Every non-ASCII string below is
written as an escape, so that this file is pure ASCII: no editor, formatter or paste step can normalize one spelling
of a label into another.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
import sys

import pytest

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.attributes import (
    ATTR_HASH,
    Universe,
    attribute_set_digest,
    attribute_set_encoding,
    attribute_to_zq,
    clip,
    validate_label,
)
from pyfe4ai.policy.formula import And, Leaf, Or, Threshold
from pyfe4ai.policy.msp import compile_formula
from pyfe4ai.policy.read_once import read_once_rows

#: The order of the BLS12-381 groups, a 255-bit prime.
R = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001

E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"                # U+00E9
FULLWIDTH_TILDE = "\N{FULLWIDTH TILDE}"                        # U+FF5E, one UTF-16 unit
DESERET_LONG_I = "\N{DESERET CAPITAL LETTER LONG I}"           # U+10400, two UTF-16 units that sort below U+FF5E
DECOMPOSED_E_ACUTE = "e\N{COMBINING ACUTE ACCENT}"             # "e" + U+0301, not in NFC


#: The limit on the digits of ``int(str)``, 0 for none; ``sys.get_int_max_str_digits`` exists from Python 3.10.7 on.
_LIMIT = getattr(sys, "get_int_max_str_digits", lambda: 0)()


class _MyStr(str):
    """A str subclass."""


class _MyInt(int):
    """An int subclass."""


def reference_hash(label: str, q: int) -> int:
    """The hash of the specification, written out again."""
    data = label.encode("utf-8")
    prefix = b"PYFE4AI-V05-POLICY-ATTR-Zq" + len(data).to_bytes(4, "big")
    return int.from_bytes(hashlib.shake_256(prefix + data).digest(64), "big") % q


# ---------------------------------------------------------------------------
# attribute_to_zq
# ---------------------------------------------------------------------------

def test_attribute_known_answers():
    assert attribute_to_zq("org:A", R) == 0x2c8cb182314dc22be14f4d92c5623cb84ca7a65a5a1a4303a85f4ab5cda78bea
    assert attribute_to_zq("org:A", 101) == 53


def test_attribute_known_answers_non_ascii():
    # the hash is over the UTF-8 bytes of the label, and the length prefix counts bytes, not characters
    assert attribute_to_zq("org:" + E_ACUTE, R) == 0xe16278b48a576f47c3b19f4743c43814489e7f82a054a14a03c05df3e417666
    assert attribute_to_zq("org:" + E_ACUTE, 2**61 - 1) == 1020341863342471049


def test_attribute_matches_the_specification():
    rng = random.Random(4)
    labels = ["org:A", "a:1", "a:11", "x:" + "z" * 300, "org:" + E_ACUTE, "k:" + DESERET_LONG_I + FULLWIDTH_TILDE]
    for label, q in itertools.product(labels, (2, 3, 101, 2**61 - 1, R, rng.getrandbits(255) | 1)):
        assert attribute_to_zq(label, q) == reference_hash(label, q)


def test_attribute_range_and_determinism():
    for q in (2, 3, 7, 101, R):
        values = [attribute_to_zq("a:{}".format(i), q) for i in range(50)]
        assert all(0 <= v < q for v in values)
        assert values == [attribute_to_zq("a:{}".format(i), q) for i in range(50)]
    assert len({attribute_to_zq("a:{}".format(i), R) for i in range(50)}) == 50
    assert attribute_to_zq("a:1", R) != attribute_to_zq("b:1", R)


def test_attribute_refuses_a_bad_label():
    for bad in ("bad", "a:", ":x", "a:b c", "a:b#1", DECOMPOSED_E_ACUTE + ":x", "a:" + DECOMPOSED_E_ACUTE,
                _MyStr("a:1"), b"a:1", None, 7):
        with pytest.raises(FEValidationError):
            attribute_to_zq(bad, R)


def test_attribute_refuses_a_bad_modulus():
    for bad in (1, 0, -1, -101, True, False, 101.0, 2.5, "101", None, _MyInt(101), b"\x65"):
        with pytest.raises(FEValidationError, match="modulus"):
            attribute_to_zq("a:1", bad)
    assert attribute_to_zq("a:1", 2) in (0, 1)


# ---------------------------------------------------------------------------
# attribute_set_encoding, attribute_set_digest
# ---------------------------------------------------------------------------

def test_attribute_set_digest_ignores_order_and_repeats():
    assert attribute_set_digest({"b:1", "a:1"}) == attribute_set_digest(["a:1", "b:1", "a:1"])
    assert attribute_set_encoding(["b:1", "a:1"]) == '["a:1","b:1"]'
    assert attribute_set_digest([]) != attribute_set_digest(["a:1"])


def test_attribute_set_refuses_bad_labels():
    with pytest.raises(FEValidationError):
        attribute_set_digest(["a b"])
    with pytest.raises(FEValidationError):
        attribute_set_digest("a:1")               # a bare string is not a collection of labels


def test_attribute_set_known_answers():
    assert ATTR_HASH == "shake256-64"
    assert attribute_set_encoding([]) == "[]"
    assert attribute_set_encoding(["b:1", "a:1", "b:1"]) == '["a:1","b:1"]'
    assert attribute_set_digest(["a:1", "b:1"]) == hashlib.shake_256(
        b'PYFE4AI-V05-POLICY-ATTRSET["a:1","b:1"]').hexdigest(32)
    assert attribute_set_digest([]) == hashlib.shake_256(b"PYFE4AI-V05-POLICY-ATTRSET[]").hexdigest(32)
    assert attribute_set_digest(["a:1", "b:1"]) == "070190d087bcba266a33ee4804b4bb1ba20441b1c6b20c12fc38bf556fb24085"
    assert attribute_set_digest([]) == "bb381f97eb0917ec23931f052d2f041841f9c767f2bc059918324252f3a5c042"
    assert len(attribute_set_digest(["a:1"])) == 64


def test_attribute_set_encoding_sorts_by_code_point_and_keeps_non_ascii():
    labels = ["k:" + FULLWIDTH_TILDE, "k:" + DESERET_LONG_I, "k:b", "k:" + E_ACUTE]
    text = attribute_set_encoding(labels)
    assert json.loads(text) == sorted(labels)
    assert text == "[" + ",".join('"{}"'.format(label) for label in sorted(labels)) + "]"
    assert E_ACUTE in text                                          # not escaped
    assert attribute_set_digest(labels) == hashlib.shake_256(
        b"PYFE4AI-V05-POLICY-ATTRSET" + text.encode("utf-8")).hexdigest(32)


def test_attribute_set_digest_tells_sets_apart():
    digests = {attribute_set_digest(labels) for labels in ([], ["a:1"], ["a:2"], ["b:1"], ["a:1", "a:2"])}
    assert len(digests) == 5


def test_attribute_set_accepts_any_iterable_of_labels():
    want = attribute_set_digest(["a:1", "b:1"])
    for labels in (("b:1", "a:1"), frozenset({"a:1", "b:1"}), iter(["a:1", "b:1", "a:1"]),
                   (label for label in ("b:1", "a:1")), {"a:1": 0, "b:1": 1}):
        assert attribute_set_digest(labels) == want


def test_attribute_set_refuses_what_is_not_a_collection_of_valid_labels():
    for bad in ("a:1", b"a:1", bytearray(b"a:1"), 7, None):
        with pytest.raises(FEValidationError, match="iterable of attribute labels"):
            attribute_set_encoding(bad)
    for bad in (["a:1", "bad"], ["a:1", 7], ["a:1", None], ["a:1", b"b:1"], [_MyStr("a:1")],
                ["a:" + DECOMPOSED_E_ACUTE]):
        with pytest.raises(FEValidationError):
            attribute_set_digest(bad)
        with pytest.raises(FEValidationError):
            attribute_set_encoding(bad)


def test_attribute_set_error_names_the_label():
    with pytest.raises(FEValidationError, match="b#1"):
        attribute_set_digest(["a:1", "a:b#1"])


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def test_universe_order_and_digest():
    u = Universe(["org:A", "a:1", "org:A"], 4)
    assert u.labels == ("a:1", "org:A") and u.size == 8
    assert u.index("a:1", 1) == 0 and u.index("a:1", 4) == 3 and u.index("org:A", 1) == 4
    assert u.encoding() == '[4,["a:1","org:A"]]'
    assert u.digest() == "befea3f1d6e0de710688dd90b7a575e6607fa62b7909ce354b5a361ba33cbe54"
    assert Universe.from_encoding(u.encoding()) == u


def test_universe_digest_is_over_the_encoding():
    u = Universe(["org:A", "a:1"], 4)
    domain = b"PYFE4AI-V05-POLICY-UNIVERSE"
    assert u.digest() == hashlib.shake_256(domain + u.encoding().encode("utf-8")).hexdigest(32)
    assert len(u.digest()) == 64
    assert u.digest() != Universe(["org:A", "a:1"], 5).digest()
    assert u.digest() != Universe(["org:A", "a:2"], 4).digest()


def test_universe_refuses():
    for bad in (([], 4), (["a:1"], 0), (["bad"], 1)):
        with pytest.raises(FEValidationError):
            Universe(*bad)
    with pytest.raises(FEValidationError, match="org:B"):
        Universe(["a:1"], 1).index("org:B", 1)
    with pytest.raises(FEValidationError):
        Universe(["a:1"], 2).index("a:1", 3)


def test_universe_refuses_labels_that_are_not_a_collection_of_labels():
    # a bare str or bytes would be read one character or one byte at a time
    for bad in ("a:1", "", b"a:1", bytearray(b"a:1"), 5, None):
        with pytest.raises(FEValidationError, match="labels"):
            Universe(bad, 4)
    for bad in ([], (), set(), iter(()), (x for x in ()), {}):
        with pytest.raises(FEValidationError, match="at least one"):
            Universe(bad, 4)
    for bad in ([1], ["a:1", None], [b"a:1"], [("a:1",)], ["a:1", _MyStr("a:2")]):
        with pytest.raises(FEValidationError):
            Universe(bad, 4)
    with pytest.raises(FEValidationError, match="'a:b c'"):
        Universe(["a:1", "a:b c"], 4)
    with pytest.raises(FEValidationError):
        Universe(["a:" + DECOMPOSED_E_ACUTE], 4)


def test_universe_accepts_any_iterable_of_labels_but_not_a_str():
    expected = Universe(["a:1", "b:1", "c:1"], 2)
    assert Universe(("c:1", "a:1", "b:1"), 2) == expected
    assert Universe({"c:1", "a:1", "b:1"}, 2) == expected
    assert Universe((label for label in ("b:1", "c:1", "a:1", "b:1")), 2) == expected
    assert Universe({"c:1": 0, "a:1": 0, "b:1": 0}, 2) == expected


def test_universe_refuses_a_bad_max_copies():
    for bad in (0, -1, -4, True, False, 4.0, 1.5, "4", None, b"\x04", _MyInt(4), [4]):
        with pytest.raises(FEValidationError, match="max_copies"):
            Universe(["a:1"], bad)
    assert Universe(["a:1"], 1).max_copies == 1


def test_universe_equality_and_hash():
    u = Universe(["org:A", "a:1"], 4)
    same = Universe(("a:1", "org:A", "a:1"), 4)
    assert u == same and hash(u) == hash(same) and not u != same
    assert len({u, same}) == 1
    assert {u: "x"}[same] == "x"
    assert u != Universe(["a:1", "org:A"], 5)
    assert u != Universe(["a:1", "org:A", "org:B"], 4)
    assert u != Universe(["a:1"], 4)
    assert u != (4, ("a:1", "org:A")) and u != u.encoding() and u != 4
    assert len({u, Universe(["a:1", "org:A"], 5), Universe(["a:1"], 4)}) == 3


def test_universe_is_immutable():
    u = Universe(["a:1"], 2)
    for name, value in (("labels", ("b:1",)), ("max_copies", 3), ("size", 1), ("extra", 0)):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            setattr(u, name, value)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        del u.max_copies
    assert type(u.labels) is tuple and u.labels == ("a:1",) and u.max_copies == 2


def test_universe_membership_and_size():
    u = Universe(["org:A", "a:1"], 3)
    assert "a:1" in u and "org:A" in u
    assert "a:2" not in u and "org:B" not in u and "" not in u and "a:" not in u and "A:1" not in u
    assert 5 not in u and None not in u and b"a:1" not in u and ["a:1"] not in u and _MyStr("a:1") not in u
    assert u.size == 6 == len(u.labels) * u.max_copies
    assert Universe(["a:1"], 1).size == 1


def test_universe_index_is_a_bijection_onto_the_slots():
    u = Universe(["org:B", "a:1", "org:A", "c:z"], 3)
    seen = {u.index(label, copy): (label, copy) for label in u.labels for copy in range(1, u.max_copies + 1)}
    assert sorted(seen) == list(range(u.size))
    # the copies of one label are consecutive, labels in sorted order
    assert [seen[i] for i in range(u.size)] == [(label, copy) for label in u.labels for copy in (1, 2, 3)]


def test_universe_index_refuses_a_bad_copy():
    u = Universe(["a:1", "b:1"], 4)
    for bad in (0, -1, 5, 100, 2**64):
        with pytest.raises(FEValidationError, match="copy {} .*1 to 4".format(bad)):
            u.index("a:1", bad)
    for bad in (True, False, 1.0, 2.5, "1", None, b"\x01", _MyInt(1), (1,)):
        with pytest.raises(FEValidationError, match="copy"):
            u.index("a:1", bad)
    with pytest.raises(FEValidationError, match="1 to 1"):
        Universe(["a:1"], 1).index("a:1", 2)


def test_universe_index_refuses_an_unknown_label():
    u = Universe(["a:1", "b:1"], 2)
    for bad in ("org:B", "a:2", "", "bad", "a:1 "):
        with pytest.raises(FEValidationError, match="not in the universe"):
            u.index(bad, 1)
    with pytest.raises(FEValidationError, match="org:B"):
        u.index("org:B", 1)
    for bad in (None, 1, b"a:1", ["a:1"]):
        with pytest.raises(FEValidationError):
            u.index(bad, 1)


def test_universe_sorts_by_code_point():
    # UTF-16 order would put U+10400 (a surrogate pair) before U+FF5E; code point order puts it after
    low, high = "a:" + FULLWIDTH_TILDE, "a:" + DESERET_LONG_I
    u = Universe([high, "a:" + E_ACUTE, low, "a:Z", "a:a"], 2)
    assert u.labels == ("a:Z", "a:a", "a:" + E_ACUTE, low, high)
    assert u.index(high, 2) == 9


def test_universe_encoding_keeps_non_ascii_text():
    u = Universe(["org:" + E_ACUTE, "a:" + DESERET_LONG_I + "x"], 2)
    assert u.encoding() == '[2,["a:' + DESERET_LONG_I + 'x","org:' + E_ACUTE + '"]]'
    assert "\\u" not in u.encoding()
    assert Universe.from_encoding(u.encoding()) == u
    assert u.digest() == hashlib.shake_256(
        b"PYFE4AI-V05-POLICY-UNIVERSE" + u.encoding().encode("utf-8")).hexdigest(32)


# ---------------------------------------------------------------------------
# Universe.from_encoding
# ---------------------------------------------------------------------------

def test_from_encoding_accepts_the_canonical_text():
    u = Universe.from_encoding('[4,["a:1","org:A"]]')
    assert u == Universe(["a:1", "org:A"], 4) and u.encoding() == '[4,["a:1","org:A"]]'
    assert Universe.from_encoding('[1,["a:1"]]') == Universe(["a:1"], 1)


REFUSED = [
    ('[4, ["a:1","org:A"]]', "canonical"),                  # whitespace
    ('[4,["a:1", "org:A"]]', "canonical"),
    (' [4,["a:1","org:A"]]', "canonical"),
    ('[4,["a:1","org:A"]] ', "canonical"),
    ('[4,["a:1","org:A"]]\n', "canonical"),
    ('[4,["org:A","a:1"]]', "canonical"),                   # not sorted
    ('[4,["a:1","a:1"]]', "canonical"),                     # a duplicate
    ('[4,["a:1","org:A","a:1"]]', "canonical"),
    ('[4,["\\u0061:1","org:A"]]', "canonical"),             # an escape for a character that is not escaped
    ('[04,["a:1"]]', "JSON"),                               # a leading zero is not JSON
    ('[+4,["a:1"]]', "JSON"),
    ('[4.0,["a:1"]]', "max_copies"),                        # a float, not an int
    ('[4e0,["a:1"]]', "max_copies"),
    ('[true,["a:1"]]', "max_copies"),                       # a bool is not an int
    ('[false,["a:1"]]', "max_copies"),
    ('[null,["a:1"]]', "max_copies"),
    ('["4",["a:1"]]', "max_copies"),
    ('[NaN,["a:1"]]', "max_copies"),
    ('[Infinity,["a:1"]]', "max_copies"),
    ('[0,["a:1"]]', "max_copies"),
    ('[-1,["a:1"]]', "max_copies"),
    ('[4,[]]', "at least one"),
    ('[4,["a:1",2]]', "label"),
    ('[4,[null]]', "label"),
    ('[4,[["a:1"]]]', "label"),
    ('[4,["bad"]]', "bad"),
    ('[4,["a:b c"]]', "whitespace"),
    ('[4,["a:\\ud800"]]', "surrogate"),
    ('[4,"a:1"]', "array"),
    ('[4,{"a:1":0}]', "array"),
    ('[4,null]', "array"),
    ('{}', "array"),
    ('{"max_copies":4,"labels":["a:1"]}', "array"),
    ('[]', "two elements"),
    ('[4]', "two elements"),
    ('[4,["a:1"],0]', "two elements"),
    ('[["a:1"],4]', "array"),
    ('4', "array"),
    ('"[4,[\\"a:1\\"]]"', "array"),
    ('null', "array"),
    ('', "JSON"),
    ('[', "JSON"),
    ('[4,["a:1"]', "JSON"),
    ('[4,["a:1"]]x', "JSON"),
    ('[4,["a:1"]]]', "JSON"),
]


@pytest.mark.parametrize("text,fragment", REFUSED, ids=[repr(text)[:40] for text, _ in REFUSED])
def test_from_encoding_refuses(text, fragment):
    with pytest.raises(FEValidationError, match=fragment):
        Universe.from_encoding(text)


def test_from_encoding_refuses_what_is_not_a_str():
    for bad in (None, 4, b'[4,["a:1"]]', bytearray(b'[4,["a:1"]]'), ["a:1"], (4, ("a:1",)), _MyStr('[4,["a:1"]]')):
        with pytest.raises(FEValidationError, match="str"):
            Universe.from_encoding(bad)


def test_from_encoding_converts_a_recursion_error():
    # json.loads recurses once per level of nesting and raises RecursionError, which is not a ValueError
    for text in ("[" * 100000, "[" * 100000 + "]" * 100000, '[4,' + "[" * 100000, '{"a":' * 100000):
        with pytest.raises(FEValidationError):
            Universe.from_encoding(text)


@pytest.mark.skipif(not _LIMIT, reason="the interpreter has no limit on integer digits")
def test_from_encoding_converts_the_integer_digit_limit():
    # int() refuses more digits than the limit and raises ValueError, which is converted
    with pytest.raises(FEValidationError, match="JSON"):
        Universe.from_encoding("[" + "9" * (_LIMIT + 1) + ',["a:1"]]')


def test_from_encoding_errors_name_the_fault_and_clip_big_input():
    with pytest.raises(FEValidationError, match=r"'a:b c'"):
        Universe.from_encoding('[4,["a:b c"]]')
    text = '[4,["a:1",' + ",".join('"x:{}"'.format(i) for i in range(5000)) + "]]"
    with pytest.raises(FEValidationError) as info:
        Universe.from_encoding(text)             # not sorted, so not canonical
    assert len(str(info.value)) < 200


def test_from_encoding_round_trips_random_universes():
    rng = random.Random(5)
    alphabet = "abxyz019_.-" + E_ACUTE + FULLWIDTH_TILDE + DESERET_LONG_I + '"\\/'
    for _ in range(200):
        labels = [
            rng.choice(("a", "org", "n-1", "x.y")) + ":" + "".join(rng.choices(alphabet, k=rng.randint(1, 6)))
            for _ in range(rng.randint(1, 8))
        ]
        u = Universe(labels, rng.randint(1, 9))
        text = u.encoding()
        back = Universe.from_encoding(text)
        assert back == u and back.encoding() == text and back.digest() == u.digest()
        assert back.labels == tuple(sorted(set(labels)))


def test_from_encoding_refuses_text_with_one_extra_character():
    text = Universe(["a:1", "b:2"], 3).encoding()
    assert Universe.from_encoding(text) == Universe(["a:1", "b:2"], 3)
    # a space, a comma or a newline at any position gives text that is not the encoding of a universe
    for pos in range(len(text) + 1):
        for extra in (" ", ",", "\n"):
            with pytest.raises(FEValidationError):
                Universe.from_encoding(text[:pos] + extra + text[pos:])


# ---------------------------------------------------------------------------
# hostile text in error messages
# ---------------------------------------------------------------------------

HUGE = 100000


def test_clip():
    assert clip("") == "" and clip("abc") == "abc"
    assert clip("x" * 40) == "x" * 40                     # the limit itself is kept whole
    assert clip("x" * 41) == "x" * 40 + "..."
    assert clip("x" * HUGE) == "x" * 40 + "..."
    assert clip("abcdef", 3) == "abc..." and clip("abc", 3) == "abc"


@pytest.mark.parametrize("label", [
    "x" * HUGE,                                  # no colon
    "a b" * HUGE + ":x",                         # the namespace
    "a:" + "x" * HUGE + " ",                     # whitespace in the value
    "a:" + "x" * HUGE + ",",                     # a reserved character
    "a:" + "x" * HUGE + "\x00",                  # a control character
    "a:" + "x" * HUGE + "\u200b",                # an invisible format character
    "a:" + "x" * HUGE + "\ud800",                # a lone surrogate
    "a:" + "x" * HUGE + DECOMPOSED_E_ACUTE,      # not in NFC
    "a:" + "\x00" * HUGE,
], ids=["no-colon", "namespace", "whitespace", "reserved", "control", "format", "surrogate", "nfc", "all-control"])
def test_validate_label_clips_the_label_in_its_message(label):
    with pytest.raises(FEValidationError) as caught:
        validate_label(label)
    assert 0 < len(str(caught.value)) < 300


def test_validate_label_keeps_a_short_label_whole_in_its_message():
    with pytest.raises(FEValidationError, match=r"'org:A B'"):
        validate_label("org:A B")


def test_universe_index_clips_the_label_in_its_message():
    universe = Universe(["a:1"], 2)
    for label in ("x" * HUGE, "a:" + "x" * HUGE, "\x00" * HUGE):
        with pytest.raises(FEValidationError, match="not in the universe") as caught:
            universe.index(label, 1)
        assert len(str(caught.value)) < 300
    with pytest.raises(FEValidationError, match="copy") as caught:
        universe.index("a:1", "9" * HUGE)
    assert len(str(caught.value)) < 300


def test_read_once_rows_clip_the_label_in_their_message():
    label = "a:" + "x" * HUGE
    msp = compile_formula(Or(Leaf(label), Leaf(label), Leaf(label)), R, max_rows=3)
    with pytest.raises(FEValidationError, match="limit of 2") as caught:
        read_once_rows(msp, 2)
    assert len(str(caught.value)) < 300


# ---------------------------------------------------------------------------
# read_once_rows
# ---------------------------------------------------------------------------

def test_read_once_rows():
    f = Or(And(Leaf("a:1"), Leaf("a:2")), And(Leaf("a:1"), Leaf("a:3")))
    msp = compile_formula(f, R)
    assert read_once_rows(msp, 2) == (("a:1", 1), ("a:2", 1), ("a:1", 2), ("a:3", 1))
    with pytest.raises(FEValidationError, match="a:1"):
        read_once_rows(msp, 1)


def test_read_once_rows_error_names_the_label_and_the_limit():
    f = And(Leaf("a:1"), Leaf("b:1"), Leaf("a:1"), Leaf("a:1"), Leaf("b:1"))
    msp = compile_formula(f, R)
    with pytest.raises(FEValidationError, match=r"'a:1'.* 3 .*limit of 2"):
        read_once_rows(msp, 2)
    with pytest.raises(FEValidationError, match=r"'a:1'.*limit of 1"):
        read_once_rows(msp, 1)
    assert read_once_rows(msp, 3) == (("a:1", 1), ("b:1", 1), ("a:1", 2), ("a:1", 3), ("b:1", 2))


def test_read_once_copies_are_distinct_and_numbered_in_row_order():
    # two labels repeat: a:1 three times, b:2 twice; the others once
    f = And(Or(Leaf("a:1"), And(Leaf("b:2"), Leaf("a:1"))),
            Or(Leaf("c:3"), Leaf("b:2")),
            Or(Leaf("a:1"), Leaf("d:4")))
    msp = compile_formula(f, R)
    assert msp.row_labels == ("a:1", "b:2", "a:1", "c:3", "b:2", "a:1", "d:4")
    rows = read_once_rows(msp, 3)
    assert isinstance(rows, tuple) and all(isinstance(item, tuple) and len(item) == 2 for item in rows)
    assert len(rows) == msp.num_rows and [label for label, _ in rows] == list(msp.row_labels)
    assert len(set(rows)) == len(rows)                                         # every (label, copy) pair is distinct
    for label in set(msp.row_labels):
        assert [copy for name, copy in rows if name == label] == list(range(1, msp.row_labels.count(label) + 1))
    assert rows == (("a:1", 1), ("b:2", 1), ("a:1", 2), ("c:3", 1), ("b:2", 2), ("a:1", 3), ("d:4", 1))


def test_read_once_rows_of_a_formula_without_repeats():
    msp = compile_formula(Threshold(2, Leaf("a:1"), Leaf("a:2"), And(Leaf("a:3"), Leaf("b:1"))), R)
    assert read_once_rows(msp, 1) == tuple((label, 1) for label in msp.row_labels)
    assert read_once_rows(compile_formula(Leaf("a:1"), R), 1) == (("a:1", 1),)


def test_read_once_rows_refuse_a_bad_max_copies():
    msp = compile_formula(Or(Leaf("a:1"), Leaf("a:2")), R)
    for bad in (0, -1, True, False, 2.0, 1.5, "2", None, _MyInt(2), [2]):
        with pytest.raises(FEValidationError, match="max_copies"):
            read_once_rows(msp, bad)


def test_read_once_rows_refuse_what_is_not_an_msp():
    for bad in (None, "a:1", ("a:1",), Leaf("a:1"), [("a:1",)]):
        with pytest.raises(FEValidationError, match="MSP"):
            read_once_rows(bad, 2)


def test_read_once_rows_fit_the_universe_of_the_policy():
    rng = random.Random(6)
    labels = ["a:1", "a:2", "a:3", "b:1"]
    for _ in range(40):
        f = rng.choice((And, Or))(*[Leaf(rng.choice(labels)) for _ in range(rng.randint(2, 9))])
        msp = compile_formula(f, R)
        copies = max(msp.row_labels.count(label) for label in msp.row_labels)
        rows = read_once_rows(msp, copies)
        universe = Universe(msp.row_labels, copies)
        # every row names a slot of the universe, and no two rows name the same slot
        slots = [universe.index(label, copy) for label, copy in rows]
        assert len(set(slots)) == len(slots) and all(0 <= slot < universe.size for slot in slots)
        if copies > 1:
            with pytest.raises(FEValidationError):
                read_once_rows(msp, copies - 1)
