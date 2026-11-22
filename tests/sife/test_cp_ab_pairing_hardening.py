"""Negative tests of the attribute-based IPFE (sife/cp_ab_pairing): keys and ciphertexts that are spliced, swapped,
made under other parameters or hostile, and the limits of the public parameters.

A key whose attributes do not satisfy the policy gives a ``FEKeyError``; malformed input, or input over a limit, a
``FEValidationError``; a key and a ciphertext that are well formed but were not made for each other a ``ValueError``
from the discrete log, which is not a ``FEValidationError``. Where test_cp_ab_pairing.py checks the shape of each
field, these tests are the attacks that leave every field well formed.
"""

import copy
import hashlib
import threading

import numpy as np
import pytest

from tests.pairing_group import skip_unless_available  # noqa: E402

skip_unless_available()

from tests.sife.conftest import ETA  # noqa: E402
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError
from pyfe4ai.policy import Universe, canonical_encoding, parse, policy_digest
from pyfe4ai.schemes.sife import cp_ab_pairing
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing

pytestmark = pytest.mark.usefixtures("own_parameter_cache")

X = [3, -1, 0, 7, 2]
Y = [1, 2, 3, -4, 5]


def key(kg, attributes, y=Y):
    return kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": attributes})


def swap_policy(ct, policy):
    """*ct* with the policy *policy* (text) and its digest in place of its own, as a forger writes them."""
    formula = parse(policy)
    return dict(ct, policy=canonical_encoding(formula), policy_digest=policy_digest(formula))


def digest_of_text(text):
    """The digest that ``policy_digest`` would give *text* if it were a formula, for a hostile text that is not one."""
    return hashlib.shake_256(b"PYFE4AI-V05-POLICY" + text.encode("utf-8")).hexdigest(32)


def policy_length_limit(kg):
    """The longest policy that ``decrypt`` reads, from the public parameters as ``decrypt`` computes it."""
    pp = kg.get_public_parameters()
    return pp["max_rows"] * (max(len(label) for label in Universe.from_encoding(pp["universe"]).labels) + 400)


# ---------------------------------------------------------------------------
# keys and ciphertexts that are put together


@pytest.mark.parametrize("donor", [0, 1], ids=["k1-k0-y-of-key1", "k1-k0-y-of-key2"])
def test_collusion_splice(donor, make):
    kg, crypto = make()
    ct = crypto.encrypt(X, policy="and(org:A, role:auditor)")
    keys = [key(kg, ["org:A"], [1, 2, 3, -4, 5]), key(kg, ["role:auditor"], [2, -1, 0, 3, 1])]
    for alone in keys:
        with pytest.raises(FEKeyError):
            crypto.decrypt(ct, alone)
    # the two keys have disjoint k_attr, so the merge loses nothing; they were made with different randomness r
    assert not set(keys[0]["k_attr"]) & set(keys[1]["k_attr"])
    forged = dict(keys[donor], attributes=["org:A", "role:auditor"], k_attr={**keys[0]["k_attr"], **keys[1]["k_attr"]})
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(ct, forged)
    assert not isinstance(caught.value, FEValidationError)


@pytest.mark.parametrize("policy,swapped,held", [
    ("and(org:A, org:B)", "or(org:A, org:B)", ["org:A"]),
    ("2of(org:A, org:B, org:C)", "or(org:A, org:B, org:C)", ["org:C"]),
], ids=["and-to-or", "threshold-to-or"])
def test_policy_swap(policy, swapped, held, make):
    kg, crypto = make()
    ct = crypto.encrypt(X, policy=policy)
    dk = key(kg, held)
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dk)
    forged = swap_policy(ct, swapped)
    # both rows of c2 are kept: `or` has as many rows as the policy it replaces, so the length check passes
    assert len(forged["c2"]) == len(parse(swapped).leaves())
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(forged, dk)
    assert not isinstance(caught.value, FEValidationError)


def test_reordered_policy_gives_nothing_new(make):
    kg, crypto = make()
    ct = crypto.encrypt(X, policy="and(org:A, role:auditor)")
    reordered = swap_policy(ct, "and(role:auditor, org:A)")
    # The digest binds the spelling of the policy, not its meaning, and reordering the children of an `and` is not
    # refused: the rows are the same up to their labels and the reconstruction weights are equal, so a key that
    # satisfies the policy decrypts as before. That key could decrypt anyway. A key that does not satisfy the
    # policy gets nothing, since the policy is the same function of the attributes in either order.
    holder = key(kg, ["org:A", "role:auditor"])
    assert crypto.decrypt(reordered, holder) == crypto.decrypt(ct, holder)
    with pytest.raises(FEKeyError):
        crypto.decrypt(reordered, key(kg, ["org:A"]))


def test_digest_mismatch(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="policy_digest"):
        crypto.decrypt(dict(ct, policy=canonical_encoding(parse("or(org:A, role:auditor)"))), dk)


def test_unknown_compiler(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="c2"):
        crypto.decrypt(dict(ct, policy_compiler="c2"), dk)


@pytest.mark.parametrize("value", [
    lambda good: np.array([0, 1]),          # `good != array` is an array, and `if` on it a ValueError
    lambda good: np.array([good]),          # and this one would compare equal and be accepted
    lambda good: np.array(good),
    lambda good: [good], lambda good: 5, lambda good: None,
], ids=["array", "array-of-it", "0-d-array", "list", "int", "none"])
@pytest.mark.parametrize("field", ["policy_compiler", "policy_digest"])
def test_a_text_field_of_the_ciphertext_that_is_no_text_is_refused_and_not_compared(field, value, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match=field) as caught:
        crypto.decrypt(dict(ct, **{field: value(ct[field])}), dk)
    assert type(caught.value) is FEValidationError


# ---------------------------------------------------------------------------
# other setups and other parameters


def test_two_setups(make, monkeypatch):
    kg1, crypto1 = make()
    kg2, crypto2 = make()
    assert kg1.get_public_parameters()["pp_binding"] != kg2.get_public_parameters()["pp_binding"]
    ct1, ct2 = crypto1.encrypt(X, policy="org:A"), crypto2.encrypt(X, policy="org:A")
    dk1, dk2 = key(kg1, ["org:A"]), key(kg2, ["org:A"])
    expected = sum(a * b for a, b in zip(X, Y))
    assert crypto1.decrypt(ct1, dk1) == crypto2.decrypt(ct2, dk2) == expected

    def never(*args):
        pytest.fail("multi_pair was called for objects of two setups")

    monkeypatch.setattr(cp_ab_pairing, "multi_pair", never)
    for crypto in (crypto1, crypto2):
        for ct, dk in ((ct1, dk2), (ct2, dk1)):         # the key of one setup with the ciphertext of the other
            with pytest.raises(FEValidationError, match="pp_binding"):
                crypto.decrypt(ct, dk)


def test_tampered_pp(make):
    kg, _ = make()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp["ua"][0] = pp["ua"][1]
    with pytest.raises(FEValidationError, match="pp_binding"):
        SIFECPABPairing({"keys": {"pp": pp}})


@pytest.mark.parametrize("field", ["g1", "pp_binding", "a"])
def test_parameters_changed_after_construction_do_not_reach_the_object(field, make):
    kg, _ = make()
    original = kg.get_public_parameters()
    mine = copy.deepcopy(original)
    crypto = SIFECPABPairing({"keys": {"pp": mine}})
    mine[field] = make()[0].get_public_parameters()[field]
    ct = crypto.encrypt(X, policy="org:A")
    assert ct["pp_binding"] == original["pp_binding"]
    expected = sum(a * b for a, b in zip(X, Y))
    assert SIFECPABPairing({"keys": {"pp": original}}).decrypt(ct, key(kg, ["org:A"])) == expected


def _nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("extra", [_nested(5000), threading.Lock()], ids=["too-deep", "not-data"])
def test_parameters_that_cannot_be_copied_are_refused(extra, make):
    kg, _ = make()
    with pytest.raises(FEValidationError, match="JSON data"):
        SIFECPABPairing({"keys": {"pp": dict(kg.get_public_parameters(), extra=extra)}})


# ---------------------------------------------------------------------------
# hostile policies


@pytest.mark.parametrize("text,reason", [
    (lambda limit: "[" * 100000, "length"),              # over the length limit: refused before from_canonical
    (lambda limit: "[" * (limit + 1), "length"),
    (lambda limit: "[" * (limit - 1), "nesting"),       # just under it: from_canonical's own check of the nesting
    (lambda limit: '["and",' * (limit // 7), "nesting"),
], ids=["100000-brackets", "one-over-the-limit", "just-under-the-limit", "nested-gates"])
def test_hostile_policy(text, reason, fixed):
    kg, crypto, ct, dk, _ = fixed
    limit = policy_length_limit(kg)
    hostile = text(limit)
    assert (len(hostile) > limit) == (reason == "length")
    message = "more than the limit of {}".format(limit) if reason == "length" else "nests brackets"
    with pytest.raises(FEValidationError, match=message):
        crypto.decrypt(dict(ct, policy=hostile, policy_digest=digest_of_text(hostile)), dk)


def test_too_many_rows(fixed):
    _, crypto, ct, dk, _ = fixed
    wide = "or({})".format(", ".join(["org:A"] * 65))
    with pytest.raises(FEValidationError, match="more than the limit of 64 rows"):
        crypto.decrypt(swap_policy(ct, wide), dk)


def test_unknown_label(make):
    kg, crypto = make()
    with pytest.raises(FEValidationError, match="org:Z"):
        crypto.encrypt(X, policy="and(org:A, org:Z)")
    with pytest.raises(FEValidationError, match="org:Z"):
        key(kg, ["org:A", "org:Z"])
    # a forged ciphertext names a label that the universe does not have
    dk = key(kg, ["org:A"])
    with pytest.raises(FEValidationError, match="org:Z"):
        crypto.decrypt(swap_policy(crypto.encrypt(X, policy="org:A"), "org:Z"), dk)


@pytest.mark.parametrize("option,policy,fragment", [
    ("max_copies", "or(and(org:A, org:B), and(org:A, org:C))", "copies per label"),   # org:A on two rows, one copy
    ("max_columns", "and(org:A, org:B)", "max_columns"),                    # two columns, one allowed
], ids=["max_copies", "max_columns"])
def test_limits(option, policy, fragment, make):
    kg, crypto = make(**{option: 1})
    with pytest.raises(FEValidationError, match=fragment):
        crypto.encrypt(X, policy=policy)
    # and a forged ciphertext is refused as well, whatever its fields say
    honest = crypto.encrypt(X, policy="org:A")
    with pytest.raises(FEValidationError, match=fragment):
        crypto.decrypt(swap_policy(honest, policy), key(kg, ["org:A", "org:B", "org:C"]))


@pytest.mark.parametrize("vector", [[11] * ETA, [0] * (ETA - 1) + [-11], [1] * (ETA + 1), [1] * (ETA - 1), []],
                         ids=["above", "below", "too-long", "too-short", "empty"])
def test_bounds(vector, make):
    kg, crypto = make()
    with pytest.raises(FEValidationError):
        crypto.encrypt(vector, policy="org:A")
    with pytest.raises(FEValidationError):
        key(kg, ["org:A"], vector)


# ---------------------------------------------------------------------------
# ciphertexts of the wrong length


@pytest.mark.parametrize("field,shape", [
    ("c1", lambda v: v[:1]), ("c1", lambda v: v + v[:1]),
    ("c2", lambda v: v[:1]), ("c2", lambda v: v + v[:1]), ("c2", lambda v: [v[0], v[1][:1]]),
    ("c3", lambda v: v[:-1]), ("c3", lambda v: v + v[:1]),
], ids=["c1-short", "c1-long", "c2-one-row-short", "c2-one-row-long", "c2-row-short", "c3-one-short", "c3-one-long"])
def test_wrong_lengths(field, shape, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(dict(ct, **{field: shape(ct[field])}), dk)
