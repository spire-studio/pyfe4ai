"""Text syntax, canonical encoding and digest of policy formulas.

**Text syntax** (:func:`parse`)::

    policy := leaf | gate
    gate   := "and(" policy ("," policy)+ ")"
            | "or("  policy ("," policy)+ ")"
            | T "of(" policy ("," policy)* ")"        T = [1-9][0-9]*
    leaf   := an attribute label (:mod:`pyfe4ai.policy.attributes`)

The keywords are lower case, and ``T`` is written in ASCII digits without a
sign or a leading zero. A label runs up to the next ``,``, ``(``, ``)`` or
ASCII whitespace. Space, tab, newline and carriage return are allowed between
tokens and around the whole text, and nowhere else.

**Canonical encoding** (:func:`canonical_encoding`): compact JSON with the
non-ASCII characters of a label kept as they are. A node is ``["leaf", label]``,
``["and", child, ...]``, ``["or", child, ...]`` or ``["th", t, child, ...]``,
the children in the order of the formula. A formula has exactly one canonical
encoding, and :func:`from_canonical` accepts no other text: a policy read out
of a ciphertext has one spelling, so it cannot be reread as another policy or
reformatted into a different digest. The digest (:func:`policy_digest`) is
``SHAKE256("PYFE4AI-V05-POLICY" || encoding)`` of 32 bytes, in hexadecimal.

:func:`from_canonical` reads untrusted input and :func:`parse` reads text, so
neither builds a formula from shared objects, and both refuse a nesting deeper
than the limit before they recurse into it: the parser counts gates as it
goes, and :func:`from_canonical` scans the brackets of the text before it hands
it to ``json.loads``. An error message carries the character position of the
fault when the input is text that :func:`parse` reads (``position N``: the
0-based index of a character in the text), and the place in the tree
(``root[2][1]``: the second element of the third element of the root array)
when it is a JSON document.
"""

from __future__ import annotations

import hashlib
import json
import re

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy.attributes import clip
from pyfe4ai.policy.formula import MAX_DEPTH, And, Formula, Leaf, Or, Threshold, validate_formula

#: The version marker of the policy compiler. A change to the canonical
#: encoding, the digest or the span-program rules of :mod:`pyfe4ai.policy.msp`
#: needs a new marker.
POLICY_COMPILER = "c1"

_DIGEST_DOMAIN = b"PYFE4AI-V05-POLICY"

# longest run of characters that no token ends inside: a label, or the name of a gate before its "("
_WORD = re.compile(r"[^,() \t\n\r]*")
_SPACE = re.compile(r"[ \t\n\r]*")
_THRESHOLD_NAME = re.compile(r"([1-9][0-9]*)of")

# the characters of a JSON text that matter to the nesting scan: outside a string, a quote or a
# bracket; inside one, the quote that ends it or the backslash that escapes the next character
_OUTSIDE_STRING = re.compile(r'["\[\]{}]')
_INSIDE_STRING = re.compile(r'["\\]')


# ---------------------------------------------------------------------------
# canonical encoding and digest
# ---------------------------------------------------------------------------

def _to_json(node: Formula) -> list:
    # at most MAX_DEPTH + 1 levels deep: callers validate the formula first
    if isinstance(node, Leaf):
        return ["leaf", node.label]
    children = [_to_json(child) for child in node.children]
    if isinstance(node, Threshold):
        return ["th", node.t, *children]
    return ["and" if isinstance(node, And) else "or", *children]


def _encode(formula: Formula) -> str:
    return json.dumps(_to_json(formula), separators=(",", ":"), ensure_ascii=False)


def canonical_encoding(formula: Formula) -> str:
    """The canonical JSON text of *formula* (see the module docstring).

    *formula* must pass :func:`~pyfe4ai.policy.formula.validate_formula`,
    which this function calls first; ``FEValidationError`` otherwise. It does
    not bound the size of the text. A formula built from shared sub-formula
    objects can stand for far more leaves than its objects suggest, and the
    text spells every leaf out: a caller that holds a formula from outside
    compiles it first (:func:`~pyfe4ai.policy.msp.compile_formula` counts the
    leaves, with :func:`~pyfe4ai.policy.formula.leaf_count`, before it does
    anything else), or counts them itself, before it encodes or digests it.
    :func:`parse` and :func:`from_canonical` never build such a formula.
    """
    validate_formula(formula)
    return _encode(formula)


def policy_digest(formula: Formula) -> str:
    """The digest of *formula*: 64 hexadecimal characters,
    ``SHAKE256("PYFE4AI-V05-POLICY" || canonical_encoding(formula))`` read to 32 bytes.

    The same rules and the same caveat about shared sub-formulas as
    :func:`canonical_encoding`.
    """
    encoding = canonical_encoding(formula)
    return hashlib.shake_256(_DIGEST_DOMAIN + encoding.encode("utf-8")).hexdigest(32)


# ---------------------------------------------------------------------------
# from_canonical
# ---------------------------------------------------------------------------

def _check_nesting(data: str) -> None:
    """Refuse *data* if its brackets (``[`` and ``{`` alike) nest deeper than
    ``2 * MAX_DEPTH``, in one linear pass that skips the inside of JSON strings
    (a label may contain brackets) and honours their backslash escapes.

    This runs before ``json.loads``, whose scanner recurses once per level. For
    text that ``json.loads`` accepts up to a given point, the number of
    brackets still open there is the number of levels it has recursed into, so
    the limit holds for whatever it reads. Text that is not JSON may be
    counted differently, and is refused by ``json.loads`` after the scan.
    """
    limit = 2 * MAX_DEPTH
    depth = 0
    pos = 0
    while True:
        found = _OUTSIDE_STRING.search(data, pos)
        if found is None:
            return
        pos = found.end()
        char = found.group()
        if char == '"':
            while True:
                found = _INSIDE_STRING.search(data, pos)
                if found is None:               # an unterminated string: json.loads refuses it
                    return
                pos = found.end()
                if found.group() == '"':
                    break
                pos += 1                        # the character a backslash escapes
        elif char in "[{":
            depth += 1
            if depth > limit:
                raise FEValidationError(
                    "the JSON text nests brackets deeper than {} levels (at position {})".format(limit, found.start())
                )
        else:
            depth -= 1


def _where(path: tuple) -> str:
    return "root" + "".join("[{}]".format(index) for index in path)


def _bad(path: tuple, message: str) -> FEValidationError:
    return FEValidationError("node {}: {}".format(_where(path), message))


def _build(path: tuple, build, *args) -> Formula:
    """``build(*args)``, with the place in the tree added to the error of a constructor."""
    try:
        return build(*args)
    except FEValidationError as exc:
        raise _bad(path, str(exc)) from exc


def _node(obj: object, path: tuple) -> Formula:
    """The formula that the parsed JSON value *obj* spells, or ``FEValidationError``.
    Recurses once per level of *obj*, which ``_check_nesting`` has bounded."""
    if type(obj) is not list or not obj:
        raise _bad(path, "expected a non-empty JSON array, got {}".format(
            "an empty array" if obj == [] else type(obj).__name__))
    tag = obj[0]
    if type(tag) is not str or tag not in ("leaf", "and", "or", "th"):
        raise _bad(path, "unknown node tag {}".format(clip(repr(tag))))
    if tag == "leaf":
        if len(obj) != 2 or type(obj[1]) is not str:
            raise _bad(path, 'a leaf is ["leaf", label] with a string label')
        return _build(path, Leaf, obj[1])
    first = 1
    args: tuple = ()
    if tag == "th":
        # type(t) is int: a bool or a float would be a second spelling of the same threshold
        if len(obj) < 2 or type(obj[1]) is not int:
            raise _bad(path, 'a threshold gate is ["th", t, child, ...] with t an integer')
        first, args = 2, (obj[1],)
    children = [_node(child, path + (index,)) for index, child in enumerate(obj[first:], first)]
    return _build(path, {"and": And, "or": Or, "th": Threshold}[tag], *args, *children)


def _first_difference(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def from_canonical(data: str) -> Formula:
    """The formula whose canonical encoding is *data*.

    *data* is untrusted. It must be a ``str`` and exactly the text that
    :func:`canonical_encoding` gives for the formula it spells: any other
    spelling of the same formula (whitespace, a ``\\u`` escape, a different
    number format) is refused, as is anything that is not a valid formula
    (:func:`~pyfe4ai.policy.formula.validate_formula`, depth included).
    Everything raises ``FEValidationError``. Brackets nested deeper than
    ``2 * MAX_DEPTH`` are refused before the text is parsed.
    """
    if type(data) is not str:
        raise FEValidationError("a canonical policy must be a str, got {}".format(type(data).__name__))
    _check_nesting(data)
    try:
        obj = json.loads(data)
    except ValueError as exc:       # not JSON; also an integer with more digits than int() converts
        raise FEValidationError("the policy is not valid JSON: {}".format(exc)) from exc
    formula = _node(obj, ())
    validate_formula(formula)
    canonical = _encode(formula)
    if canonical != data:
        raise FEValidationError(
            "the policy is valid but not in canonical form: it first differs from its canonical encoding "
            "at position {}".format(_first_difference(canonical, data))
        )
    return formula


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive descent over a policy text, at most ``MAX_DEPTH`` gates deep."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    @staticmethod
    def fail(pos: int, message: str) -> FEValidationError:
        return FEValidationError("position {}: {}".format(pos, message))

    def skip_space(self) -> None:
        self.pos = _SPACE.match(self.text, self.pos).end()

    def build(self, pos: int, build, *args) -> Formula:
        """``build(*args)``, with *pos* added to the error of a constructor."""
        try:
            return build(*args)
        except FEValidationError as exc:
            raise self.fail(pos, str(exc)) from exc

    def policy(self, depth: int) -> Formula:
        """A leaf or a gate at the current position, inside *depth* gates."""
        self.skip_space()
        start = self.pos
        word = _WORD.match(self.text, start)
        if word.end() < len(self.text) and self.text[word.end()] == "(":
            return self.gate(word.group(), start, depth)
        if not word.group():
            found = "end of text" if start >= len(self.text) else repr(self.text[start])
            raise self.fail(start, "expected an attribute label or a gate, found {}".format(found))
        self.pos = word.end()
        return self.build(start, Leaf, word.group())

    def gate(self, name: str, start: int, depth: int) -> Formula:
        """A gate called *name* that starts at *start*; the text there is ``name(``."""
        args: tuple = ()
        if name == "and":
            kind = And
        elif name == "or":
            kind = Or
        else:
            threshold = _THRESHOLD_NAME.fullmatch(name)
            if threshold is None:
                raise self.fail(start, "unknown gate {!r}: expected and(, or( or <t>of( with t a positive "
                                       "integer in ASCII digits without a sign or a leading zero".format(clip(name)))
            digits = threshold.group(1)
            # t cannot exceed the number of children, which cannot exceed the length of the text: this
            # keeps int() off a long digit string
            if len(digits) > len(str(len(self.text))):
                raise self.fail(start, "the threshold {} exceeds the number of children that the "
                                       "text can hold".format(clip(digits, 12)))
            kind, args = Threshold, (int(digits),)
        if depth >= MAX_DEPTH:
            raise self.fail(start, "formula depth exceeds the limit of {} gates on a path".format(MAX_DEPTH))
        self.pos = start + len(name) + 1
        children = []
        while True:
            children.append(self.policy(depth + 1))
            self.skip_space()
            if self.pos >= len(self.text):
                raise self.fail(self.pos, "unexpected end of text: the gate at position {} "
                                          "is not closed".format(start))
            char = self.text[self.pos]
            if char == ")":
                self.pos += 1
                return self.build(start, kind, *args, *children)
            if char != ",":
                raise self.fail(self.pos, "expected ',' or ')', found {!r}".format(char))
            self.pos += 1


def parse(text: str) -> Formula:
    """The formula that the policy *text* spells (see the module docstring for the syntax).

    *text* must be a ``str``. Anything that is not a valid policy raises
    ``FEValidationError`` whose message starts with ``position N:``, the
    0-based index of the character at fault: a gate opened at that position
    that has too few children or an impossible threshold, a label that
    :func:`~pyfe4ai.policy.attributes.validate_label` refuses, a nesting
    deeper than ``MAX_DEPTH`` gates, a misplaced token, or text that ends
    early or goes on after the policy. A fault that only the whole formula
    shows (two leaf children of a threshold gate with the same label) is
    reported at the position where the policy starts.
    """
    if type(text) is not str:
        raise FEValidationError("a policy text must be a str, got {}".format(type(text).__name__))
    parser = _Parser(text)
    parser.skip_space()
    start = parser.pos
    formula = parser.policy(0)
    parser.skip_space()
    if parser.pos < len(text):
        raise parser.fail(parser.pos, "unexpected {!r} after the end of the policy".format(text[parser.pos]))
    try:
        validate_formula(formula)
    except FEValidationError as exc:
        raise parser.fail(start, "{} (found when checking the policy as a whole)".format(exc)) from exc
    return formula
