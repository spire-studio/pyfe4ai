"""Negative tests of the key-policy attribute-based IPFE (sife/kp_ab_pairing): configurations and key requests that
are refused, keys and ciphertexts that are spliced, swapped, made under other parameters or hostile, and the limits of
the public parameters, of the policies and of the attributes.

A key whose policy the attributes of a ciphertext do not satisfy gives a ``FEKeyError``; malformed input, or input over
a limit, a ``FEValidationError``; a key and a ciphertext that are well formed but were not made for each other a
``ValueError`` from the discrete log, which is not a ``FEValidationError``. Where test_kp_ab_pairing.py checks that the
scheme works, these tests are the attacks and the refusals; the shape of each field of a key and of a ciphertext is
checked here too, and the attacks that leave every field well formed.
"""

import copy
import hashlib
import json
import threading

import numpy as np
import pytest

import pyfe4ai
from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.sife.conftest import ETA, KP_HELD as HELD, KP_POLICY as POLICY, KP_X as X, KP_Y as Y  # noqa: E402
from pyfe4ai.core.exceptions import FEKeyError, FESecurityWarning, FEValidationError
from pyfe4ai.core.formats.pairing import KPABCiphertext, KPABFunctionalKey, KPABParameters
from pyfe4ai.policy import Leaf, Or, canonical_encoding, parse, policy_digest
from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup, deserialize_many, pairing_available
from pyfe4ai.schemes.common import ac_codec, kp_ab
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.kp_ab import (
    MAX_LABEL_CHARS,
    attribute_pairs,
    check_policy_labels,
    check_policy_length,
    compile_policy,
    held_attributes,
    policy_length_limit,
    read_attributes,
    vector_positions,
)
from pyfe4ai.schemes.sife import kp_ab_pairing
from pyfe4ai.schemes.sife.kp_ab_pairing import SIFEKPABPairing, SIFEKPABPairingKeyGenerator

PP_DOMAIN = b"pyfe4ai/sife/kp-ab/pp-binding/v1|"

pytestmark = pytest.mark.usefixtures("own_parameter_cache")


def public(kg):
    """The group, its order and the public bases ``f`` of *kg* as group elements."""
    group = PairingGroup(GROUP)
    return {"group": group, "order": int(group.order()),
            "f": [deserialize_many(group, row) for row in kg.get_public_parameters()["f"]]}


def creds(policy, y=None):
    return {"fusion_weight": y if y is not None else [1] * ETA, "policy": policy}


def inner(x, y):
    return sum(a * b for a, b in zip(x, y))


def label_of(length):
    return "a:" + "x" * (length - 2)


# ---------------------------------------------------------------------------
# the configuration and the key requests that are refused


@pytest.mark.parametrize("option", ["eta", "bound_x", "bound_y", "max_rows", "max_attributes"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "5", None])
def test_limits_must_be_positive_ints(option, value):
    config = {"eta": ETA, "pairing_group_param": GROUP, option: value}
    with pytest.raises(FEValidationError, match=option):
        SIFEKPABPairingKeyGenerator(config)


def test_dlog_limit():
    config = {"eta": 2, "bound_x": 2 ** 24, "bound_y": 2 ** 24, "pairing_group_param": GROUP}
    with pytest.raises(FEValidationError, match=r"2\*\*48"):
        SIFEKPABPairingKeyGenerator(config)
    assert SIFEKPABPairingKeyGenerator(dict(config, eta=1)).eta == 1        # 2**48 itself is allowed


def test_a_symmetric_group_is_refused():
    if not pairing_available("SS512"):
        pytest.skip("no charm: SS512 is not built")
    with pytest.raises(FEValidationError, match="symmetric"):
        SIFEKPABPairingKeyGenerator({"eta": ETA, "pairing_group_param": "SS512"})


@pytest.mark.native     # BLS12-381, the one group that the profile takes, comes from the extension
def test_standard_profile():
    # as sife/cp_ab_pairing: BLS12-381 with a FESecurityWarning about its 120 to 126 bits, not a refusal
    if not pairing_available("BLS12-381"):
        pytest.skip("no backend for pairing group BLS12-381")
    with pytest.warns(FESecurityWarning, match="sife/kp_ab_pairing: BLS12-381 gives about 120 to 126-bit security"):
        kg, crypto_cls = pyfe4ai.create("sife/kp_ab_pairing", eta=2, bound_x=3, bound_y=3, security_profile="standard")
    assert kg.pairing_group_param == "BLS12-381" and kg.security_profile == "standard"
    kg.setup()
    crypto = crypto_cls({"keys": {"pp": kg.get_public_parameters()}})
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1, -2], "policy": "or(org:A, org:B)"})
    assert crypto.decrypt(crypto.encrypt([3, -1], attributes=["org:B"]), dk) == 5
    with pytest.raises(FEValidationError, match="BLS12-381"):                           # no other group
        pyfe4ai.create("sife/kp_ab_pairing", eta=2, pairing_group_param="MNT224", security_profile="standard")


def test_key_before_setup():
    kg = SIFEKPABPairingKeyGenerator({"eta": ETA, "pairing_group_param": GROUP})
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_decryption_keys("s", credentials=creds("org:A"))


@pytest.mark.parametrize("credentials", [
    None, {}, {"policy": "org:A"}, {"fusion_weight": [1] * 5},
    {"fusion_weight": [1] * 4, "policy": "org:A"},
    {"fusion_weight": [99] * 5, "policy": "org:A"},
    {"fusion_weight": [1] * 5, "policy": None},
    {"fusion_weight": [1] * 5, "policy": "org:A and org:B"},
    {"fusion_weight": [1] * 5, "policy": "or(org:A,"},
    {"fusion_weight": [1] * 5, "policy": b"org:A"},
    {"fusion_weight": [1] * 5, "policy": ["org:A"]},
    {"fusion_weight": [1.5] * 5, "policy": "org:A"},
    {"fusion_weight": "11111", "policy": "org:A"},
    {"fusion_weight": [1] * 5, "attributes": ["org:A"]},            # the entry of the ciphertext-policy scheme
    ["fusion_weight", "policy"],
])
def test_key_requests_are_checked(make_kp, credentials):
    kg, _ = make_kp()
    with pytest.raises((FEKeyError, FEValidationError)):
        kg.get_decryption_keys("s", credentials=credentials)


def test_missing_credentials_are_a_key_error_and_bad_ones_a_validation_error(make_kp):
    kg, _ = make_kp()
    for credentials in (None, {}):
        with pytest.raises(FEKeyError, match="credentials"):
            kg.get_decryption_keys("s", credentials=credentials)
    for credentials, match in (({"policy": "org:A"}, "fusion_weight"), ({"fusion_weight": [1] * 5}, "policy"),
                               ({"fusion_weight": [11] * 5, "policy": "org:A"}, "fusion weight"),
                               ({"fusion_weight": [1] * 5, "policy": 7}, "policy")):
        with pytest.raises(FEValidationError, match=match):
            kg.get_decryption_keys("s", credentials=credentials)


def test_a_policy_of_more_rows_than_the_limit_is_refused(make_kp):
    kg, _ = make_kp(max_rows=3)
    kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B, org:C)"))
    with pytest.raises(FEValidationError, match="more than the limit of 3 rows"):
        kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B, org:C, org:D)"))


def test_a_policy_over_the_default_limit_is_refused(make_kp):
    kg, _ = make_kp()
    leaves = ["org:L{}".format(i) for i in range(65)]
    with pytest.raises(FEValidationError, match="64"):
        kg.get_decryption_keys("s", credentials=creds("or({})".format(", ".join(leaves))))


def test_the_key_generator_refuses_a_label_over_the_limit(make_kp):
    # no ciphertext holds a label of more than MAX_LABEL_CHARS characters, so no ciphertext could satisfy its row:
    # the key is refused when it is issued, as a text, as a formula object, in a gate and in a row among others
    kg, crypto = make_kp()
    fits, too_long = label_of(MAX_LABEL_CHARS), label_of(MAX_LABEL_CHARS + 1)
    assert crypto.decrypt(crypto.encrypt(X, attributes=[fits]), kg.get_decryption_keys(
        "s", credentials=creds("or({}, org:A)".format(fits), Y))) == inner(X, Y)
    for policy in (too_long, "or({}, org:A)".format(too_long), "or(org:A, {})".format(too_long),
                   "and(org:A, 2of(org:B, {}, org:C))".format(too_long), Or(Leaf("org:A"), Leaf(too_long))):
        with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024") as caught:
            kg.get_decryption_keys("s", credentials=creds(policy, Y))
        assert "1025 characters" in str(caught.value) and len(str(caught.value)) < 300
    with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024") as caught:
        kg.get_decryption_keys("s", credentials=creds("or(org:A, {})".format(label_of(100000)), Y))
    assert "100000 characters" in str(caught.value) and len(str(caught.value)) < 300


def test_the_key_generator_refuses_a_policy_that_decryption_would_refuse(make_kp):
    # '"' takes two characters in the canonical encoding and one in a label
    label = "a:" + '"' * 800
    assert len(label) <= MAX_LABEL_CHARS
    kg, crypto = make_kp(max_rows=1)
    assert kg.get_decryption_keys("s", credentials=creds("a:1", Y))
    with pytest.raises(FEValidationError, match="characters, more than the limit of 1424"):
        kg.get_decryption_keys("s", credentials=creds(label, Y))
    with pytest.raises(FEValidationError, match="characters, more than the limit of 1424"):
        kg.get_decryption_keys("s", credentials=creds("a:" + '"' * (MAX_LABEL_CHARS - 2), Y))      # the longest label


def test_a_label_of_the_limit_length_fits_every_policy_of_the_limit_rows(make_kp):
    # the 64 rows of the default, each with a label of 1024 characters
    labels = [label_of(MAX_LABEL_CHARS - 5) + "{:05d}".format(i) for i in range(64)]
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("or({})".format(", ".join(labels)), Y))
    assert len(dk["k_rows"]) == 64
    assert len(dk["policy"]) <= crypto.max_policy_length
    assert crypto.decrypt(crypto.encrypt(X, attributes=labels[-1:]), dk) == inner(X, Y)


def test_a_refused_request_does_not_change_the_key_generator(make_kp):
    kg, _ = make_kp()
    before = (dict(kg.msk), json.dumps(kg.get_public_parameters()))
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("s", credentials=creds("or(org:A,"))
    assert (dict(kg.msk), json.dumps(kg.get_public_parameters())) == before


# ---------------------------------------------------------------------------
# the shared readers


@pytest.mark.parametrize("bad", ["a:1", b"a:1", bytearray(b"a:1"), 5, None, ["a"], ["a:1", 7], [b"a:1"], [["a:1"]],
                                 ["a:1 b"], ["a:()"], ["a:\u200b1"]])
def test_held_attributes_refuses_what_is_not_a_collection_of_labels(bad):
    with pytest.raises(FEValidationError):
        held_attributes(bad, 64)


def test_held_attributes_limit():
    assert held_attributes(["a:1", "b:1"], 2) == ["a:1", "b:1"]
    with pytest.raises(FEValidationError, match="max_attributes = 2"):
        held_attributes(["a:1", "b:1", "c:1"], 2)
    assert len(held_attributes(["a:{}".format(i) for i in range(64)], 64)) == 64
    with pytest.raises(FEValidationError, match="64"):
        held_attributes(["a:{}".format(i) for i in range(65)], 64)


@pytest.mark.parametrize("bad", [
    ("a:1", "b:1"), {"a:1"}, "a:1", None, 5,                        # not a list
    ["b:1", "a:1"], ["a:1", "a:1"],                                  # not strictly increasing
    ["a:1", 7], [b"a:1"], ["a"], ["a:1", "b:1 c"],                   # not labels
])
def test_read_attributes_refuses_what_a_ciphertext_does_not_hold(bad):
    with pytest.raises(FEValidationError):
        read_attributes(bad, 64)


def test_read_attributes_limit_is_checked():
    with pytest.raises(FEValidationError, match="max_attributes = 2"):
        read_attributes(["a:1", "b:1", "c:1"], 2)


def test_read_attributes_names_the_labels_that_are_out_of_order():
    with pytest.raises(FEValidationError, match=r"'b:1' is followed by 'a:1'"):
        read_attributes(["a:0", "b:1", "a:1"], 64)
    with pytest.raises(FEValidationError, match=r"'a:1' is followed by 'a:1'"):
        read_attributes(["a:0", "a:1", "a:1"], 64)
    label = "a:" + "x" * 5000
    with pytest.raises(FEValidationError) as caught:
        read_attributes([label + "y", label], 64)
    assert len(str(caught.value)) < 300


def test_labels_have_a_length_limit():
    assert MAX_LABEL_CHARS == 1024
    assert held_attributes([label_of(1024)], 64) == [label_of(1024)]
    assert read_attributes([label_of(1024)], 64) == [label_of(1024)]
    for function in (held_attributes, read_attributes):
        with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024") as caught:
            function(["a:1", label_of(1025)], 64)
        assert "1025 characters" in str(caught.value) and len(str(caught.value)) < 300


def test_the_length_of_a_label_is_checked_before_the_label_is_read():
    # a label of a million characters is refused by its length, whatever else is wrong with it
    for function in (held_attributes, read_attributes):
        with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS"):
            function(["a: " + "x" * 10 ** 6], 64)


def test_policy_length_limit():
    assert policy_length_limit(64) == 64 * (1024 + 400)
    assert policy_length_limit(3) == 3 * 1424
    check_policy_length("x" * 1424, 1424)
    with pytest.raises(FEValidationError, match="1425 characters, more than the limit of 1424"):
        check_policy_length("x" * 1425, 1424)


def test_check_policy_labels():
    check_policy_labels([])
    check_policy_labels(["a:1", label_of(MAX_LABEL_CHARS)])
    with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024") as caught:
        check_policy_labels(["a:1", label_of(MAX_LABEL_CHARS + 1), label_of(5000)])
    assert "1025 characters" in str(caught.value) and len(str(caught.value)) < 300       # the first one, clipped


def test_vector_positions_are_those_of_the_labels_of_the_used_rows_in_the_order_of_the_weights():
    rows = ["b:1", "x:1", "y:1", "a:1"]
    assert vector_positions({0: 1, 3: 5}, rows, ["a:1", "b:1"]) == [1, 0]
    assert vector_positions({3: 5, 0: 1}, rows, ["a:1", "b:1"]) == [0, 1]
    assert vector_positions({1: 1, 2: 1}, rows, ["a:1", "x:1", "y:1"]) == [1, 2]
    assert vector_positions({}, rows, ["a:1"]) == []
    with pytest.raises(FEValidationError, match="row 1 of the policy names the label 'x:1'"):
        vector_positions({0: 1, 1: 1}, rows, ["a:1", "b:1"])


def test_attribute_pairs_refuses_a_label_the_ciphertext_does_not_carry(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    with pytest.raises(FEValidationError, match="b:1"):
        attribute_pairs({1: 1}, ["a:1", "b:1"], ["a:1"], [p["f"][0]], {1: kg.msk["f_star"][0]}, p["group"], p["order"])


@pytest.mark.parametrize("c_length,k_length", [(7, 8), (8, 7), (9, 8)])
def test_attribute_pairs_refuses_vectors_of_another_length(c_length, k_length, make_kp):
    kg, _ = make_kp()
    p = public(kg)
    c = (p["f"][0] * 2)[:c_length]
    k = (kg.msk["f_star"][0] * 2)[:k_length]
    with pytest.raises(FEValidationError, match="8 elements"):
        attribute_pairs({0: 1}, ["a:1"], ["a:1"], [c], {0: k}, p["group"], p["order"])


# ---------------------------------------------------------------------------
# encryption: the input


@pytest.mark.parametrize("vector", [[11] * ETA, [0] * (ETA - 1) + [-11], [1] * (ETA + 1), [1] * (ETA - 1), [], None,
                                    [1.5] * ETA, "12345", [True] * ETA], ids=str)
def test_encrypt_checks_the_plaintext(vector, make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="plaintext"):
        crypto.encrypt(vector, attributes=["org:A"])


@pytest.mark.parametrize("attributes", [None, "org:A", b"org:A", 7, ["org:A", 7], [b"org:A"], ["org"], ["org:A b"],
                                        [["org:A"]], ["org:\u200bA"]], ids=str)
def test_encrypt_checks_the_attributes(attributes, make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="attribute"):
        crypto.encrypt(X, attributes=attributes)


def test_encrypt_without_attributes_is_refused(make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="attributes"):
        crypto.encrypt(X)


def test_encrypt_limits_the_attributes(make_kp):
    _, crypto = make_kp(max_attributes=3)
    labels = ["org:{}".format(i) for i in range(4)]
    assert len(crypto.encrypt(X, attributes=labels[:3])["c_attr"]) == 3
    assert len(crypto.encrypt(X, attributes=labels[:3] * 5)["c_attr"]) == 3         # repeats are not counted
    with pytest.raises(FEValidationError, match="max_attributes = 3"):
        crypto.encrypt(X, attributes=labels)
    _, default = make_kp()                                                          # 64, the default
    with pytest.raises(FEValidationError, match="65 attributes are more than the limit max_attributes = 64"):
        default.encrypt(X, attributes=["org:{:02d}".format(i) for i in range(65)])


def test_encrypt_limits_the_length_of_a_label(make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024"):
        crypto.encrypt(X, attributes=[label_of(1025)])


def test_a_refused_encryption_computes_nothing(make_kp, monkeypatch):
    _, crypto = make_kp()
    monkeypatch.setattr(kp_ab_pairing, "attribute_vector", lambda *a, **k: pytest.fail("computed for refused input"))
    monkeypatch.setattr(kp_ab_pairing, "multiexp_many", lambda *a, **k: pytest.fail("computed for refused input"))
    with pytest.raises(FEValidationError):
        crypto.encrypt(X, attributes="org:A")
    with pytest.raises(FEValidationError):
        crypto.encrypt([11] * ETA, attributes=["org:A"])


# ---------------------------------------------------------------------------
# keys and ciphertexts that are put together, or made under other parameters


def swap_policy(dk, policy):
    """*dk* with the policy *policy* (text) and its digest in place of its own, as a forger writes them."""
    formula = parse(policy)
    return dict(dk, policy=canonical_encoding(formula), policy_digest=policy_digest(formula))


def digest_of_text(text):
    """The digest that ``policy_digest`` would give *text* if it were a formula, for a hostile text that is not one."""
    return hashlib.shake_256(b"PYFE4AI-V05-POLICY" + text.encode("utf-8")).hexdigest(32)


def without(record, field):
    return {name: value for name, value in record.items() if name != field}


@pytest.mark.parametrize("donor", [0, 1], ids=["k-ipfe-of-key1", "k-ipfe-of-key2"])
def test_collusion_splice(donor, make_kp):
    # two keys that each hold one row of a policy are put together into a key for the policy of both: the rows have
    # different a0, so the shares do not add up to the a0 of k_ipfe
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A", "role:auditor"])
    keys = [kg.get_decryption_keys("s", credentials=creds("org:A", Y)),
            kg.get_decryption_keys("s", credentials=creds("role:auditor", Y))]
    for alone in keys:
        assert crypto.decrypt(ct, alone) == inner(X, Y)         # each opens the ciphertext for its own policy
    forged = dict(swap_policy(keys[donor], POLICY), k_rows=[keys[0]["k_rows"][0], keys[1]["k_rows"][0]])
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(ct, forged)
    assert not isinstance(caught.value, FEValidationError)


@pytest.mark.parametrize("donor", [0, 1], ids=["k-ipfe-of-key1", "k-ipfe-of-key2"])
def test_collusion_splice_of_rows_from_keys_that_cannot_open_alone(donor, make_kp):
    # two keys, neither of which opens the ciphertext of {org:A, org:C}; a key for and(org:A, org:C) is put together
    # from the row of org:A of the first and the row of org:C of the second, with the k_ipfe and y of one of them
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A", "org:C"])
    keys = [kg.get_decryption_keys("s", credentials=creds("and(org:A, org:B)", Y)),
            kg.get_decryption_keys("s", credentials=creds("and(org:C, role:auditor)", [2, -1, 0, 3, 1]))]
    for alone in keys:
        with pytest.raises(FEKeyError, match="do not satisfy"):
            crypto.decrypt(ct, alone)
    forged = dict(swap_policy(keys[donor], "and(org:A, org:C)"), k_rows=[keys[0]["k_rows"][0], keys[1]["k_rows"][0]])
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(ct, forged)
    assert not isinstance(caught.value, FEValidationError)


@pytest.mark.parametrize("policy,swapped,held", [
    ("and(org:A, org:B)", "or(org:A, org:B)", ["org:A"]),
    ("2of(org:A, org:B, org:C)", "or(org:A, org:B, org:C)", ["org:C"]),
], ids=["and-to-or", "threshold-to-or"])
def test_policy_swap(policy, swapped, held, make_kp):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=held)
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dk)
    # `or` has as many rows as the policy it replaces, so the length checks pass
    forged = swap_policy(dk, swapped)
    assert len(forged["k_rows"]) == len(parse(swapped).leaves())
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(ct, forged)
    assert not isinstance(caught.value, FEValidationError)


def test_a_ciphertext_of_other_attributes_gives_nothing(make_kp):
    # the attributes of a ciphertext are public data, and the vectors are bound to the labels by their j: the
    # vector of org:B written under the name org:A does not open the row of org:A
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:B"])
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dk)
    with pytest.raises(ValueError, match="discrete log"):
        crypto.decrypt(dict(ct, attributes=["org:A"]), dk)


def test_a_key_of_another_setup_does_not_open_even_when_rebound(make_kp):
    kg, crypto = make_kp()
    other, _ = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A"])
    dk = other.get_decryption_keys("s", credentials=creds("org:A", Y))
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(ct, dk)
    with pytest.raises(ValueError, match="discrete log"):
        crypto.decrypt(ct, dict(dk, pp_binding=ct["pp_binding"]))


def test_two_setups(make_kp, monkeypatch):
    kg1, crypto1 = make_kp()
    kg2, crypto2 = make_kp()
    assert kg1.get_public_parameters()["pp_binding"] != kg2.get_public_parameters()["pp_binding"]
    ct1, ct2 = crypto1.encrypt(X, attributes=["org:A"]), crypto2.encrypt(X, attributes=["org:A"])
    dk1 = kg1.get_decryption_keys("s", credentials=creds("org:A", Y))
    dk2 = kg2.get_decryption_keys("s", credentials=creds("org:A", Y))
    assert crypto1.decrypt(ct1, dk1) == crypto2.decrypt(ct2, dk2) == inner(X, Y)

    def never(*args):
        pytest.fail("multi_pair was called for objects of two setups")

    monkeypatch.setattr(kp_ab_pairing, "multi_pair", never)
    for crypto in (crypto1, crypto2):
        for ct, dk in ((ct1, dk2), (ct2, dk1)):         # the key of one setup with the ciphertext of the other
            with pytest.raises(FEValidationError, match="pp_binding"):
                crypto.decrypt(ct, dk)
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto1.decrypt(ct2, dk2)                       # and both of the other setup


def test_decrypt_checks_the_binding_before_anything_else(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    bad = dict(without(ct, "t"), pp_binding="0" * 64, attr_hash="x", attributes=5)
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(bad, dk)
    bad_key = dict(without(dk, "k_ipfe"), pp_binding="0" * 64, policy="[" * 100000, policy_compiler="c9")
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(ct, bad_key)


def test_decrypt_checks_in_a_fixed_order(make_kp):
    # one defect for each stage, put right one stage after the other: binding, fields, version markers, policy and
    # its digest, lengths, the weights, then whether the attributes satisfy the policy, and only then the elements
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A"])                    # does not satisfy POLICY
    dk = kg.get_decryption_keys("s", credentials=creds(POLICY, Y))
    ct["c_attr"][0][0] = "junk"                                     # the last stage: an element that is no element
    defects = [
        ("pp_binding", lambda c, k: (dict(c, pp_binding="0" * 64), k)),
        ("no field k_ipfe", lambda c, k: (c, without(k, "k_ipfe"))),
        ("attr_hash", lambda c, k: (dict(c, attr_hash="x"), k)),
        ("policy_digest", lambda c, k: (c, dict(k, policy_digest="0" * 64))),
        ("t must be", lambda c, k: (dict(c, t=c["t"][:-1]), k)),
        ("y must hold", lambda c, k: (c, dict(k, y=["x"] * ETA))),
    ]
    for stage, (match, _) in enumerate(defects):
        c, k = ct, dk
        for _, defect in defects[stage:]:
            c, k = defect(c, k)
        with pytest.raises(FEValidationError, match=match):
            crypto.decrypt(c, k)
    with pytest.raises(FEKeyError, match="do not satisfy"):         # the junk element is not read for a key that
        crypto.decrypt(ct, dk)                                       # cannot open the ciphertext


def test_a_failed_discrete_log_is_a_value_error(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    bad = dict(ct, t=list(reversed(ct["t"])))
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(bad, dk)
    assert not isinstance(caught.value, FEValidationError)


def test_a_ciphertext_with_a_changed_vector_gives_no_value(kp_fixed):
    # the vector of org:B is not read by the key and changing it changes nothing; that of org:A is, and gives nothing
    _, crypto, ct, dk, expected = kp_fixed
    other = crypto.encrypt(X, attributes=HELD)
    assert crypto.decrypt(dict(ct, c_attr=[ct["c_attr"][0], other["c_attr"][1], ct["c_attr"][2]]), dk) == expected
    with pytest.raises(ValueError, match="discrete log"):
        crypto.decrypt(dict(ct, c_attr=[other["c_attr"][0], ct["c_attr"][1], ct["c_attr"][2]]), dk)


# ---------------------------------------------------------------------------
# the public parameters


def rebound(pp):
    """*pp* with its ``pp_binding`` recomputed: public parameters that are consistent, but not honest."""
    return dict(pp, pp_binding=pp_digest(pp, PP_DOMAIN))


def tampered(pp, **fields):
    return rebound(dict(pp, **fields))


def test_tampered_pp(make_kp):
    kg, _ = make_kp()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp["t"][0] = pp["t"][1]
    with pytest.raises(FEValidationError, match="pp_binding"):
        SIFEKPABPairing({"keys": {"pp": pp}})


@pytest.mark.parametrize("field", ["g1", "pp_binding", "t"])
def test_parameters_changed_after_construction_do_not_reach_the_object(field, make_kp):
    kg, _ = make_kp()
    original = kg.get_public_parameters()
    mine = copy.deepcopy(original)
    crypto = SIFEKPABPairing({"keys": {"pp": mine}})
    mine[field] = make_kp()[0].get_public_parameters()[field]
    ct = crypto.encrypt(X, attributes=["org:A"])
    assert ct["pp_binding"] == original["pp_binding"]
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)                    # the object whose dict was changed
    assert SIFEKPABPairing({"keys": {"pp": original}}).decrypt(ct, dk) == inner(X, Y)


def _nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("extra", [_nested(5000), threading.Lock()], ids=["too-deep", "not-data"])
def test_parameters_that_cannot_be_copied_are_refused(extra, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="JSON data"):
        SIFEKPABPairing({"keys": {"pp": dict(kg.get_public_parameters(), extra=extra)}})


@pytest.mark.parametrize("field", list(KPABParameters.FIELDS))
def test_crypto_refuses_public_parameters_without_a_field(field, make_kp):
    kg, _ = make_kp()
    pp = without(kg.get_public_parameters(), field)
    with pytest.raises(FEValidationError, match=field):
        SIFEKPABPairing({"keys": {"pp": pp if field == "pp_binding" else rebound(pp)}})


@pytest.mark.parametrize("value", [
    lambda good: np.array([0, 1]), lambda good: np.array([good]), lambda good: np.array(good),
    lambda good: [good], lambda good: 5, lambda good: None,
], ids=["array", "array-of-it", "0-d-array", "list", "int", "none"])
def test_crypto_refuses_parameters_whose_pp_binding_is_no_text(value, make_kp):
    # the digest leaves pp_binding out, so an array there is JSON enough to be compared with it
    kg, _ = make_kp()
    pp = kg.get_public_parameters()
    with pytest.raises(FEValidationError, match="pp_binding") as caught:
        SIFEKPABPairing({"keys": {"pp": dict(pp, pp_binding=value(pp["pp_binding"]))}})
    assert type(caught.value) is FEValidationError


def test_crypto_needs_public_parameters():
    with pytest.raises(FEKeyError):
        SIFEKPABPairing({"keys": {"sk": {}}})
    with pytest.raises(FEValidationError):
        SIFEKPABPairing({"keys": {"pp": "not a dict"}})


@pytest.mark.parametrize("field", ["policy_compiler", "attr_hash"])
def test_crypto_refuses_an_unknown_version_marker(field, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match=field) as caught:
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: "v" * 100000})}})
    assert len(str(caught.value)) < 300


@pytest.mark.parametrize("value", [5, None, ["BLS12-381"], {"name": "BLS12-381"}, True])
def test_crypto_refuses_a_pairing_group_that_is_no_name(value, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="pairing_group_param"):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), pairing_group_param=value)}})


@pytest.mark.parametrize("name", ["nonsense", "", "x" * 100000], ids=["unknown", "empty", "huge"])
def test_crypto_refuses_a_pairing_group_that_does_not_exist(name, make_kp):
    if not pairing_available("MNT224"):
        pytest.skip("no charm: other groups than BLS12-381 are not built")
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="pairing_group_param") as caught:
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), pairing_group_param=name)}})
    assert len(str(caught.value)) < 300


def test_a_symmetric_group_is_refused_by_the_crypto_object(make_kp):
    if not pairing_available("SS512"):
        pytest.skip("no charm: SS512 is not built")
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="symmetric"):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), pairing_group_param="SS512")}})


@pytest.mark.parametrize("field,value", [
    ("h_mask", ["x"] * 3), ("h_mask", ["x"] * 5), ("h_mask", "abcd"), ("h3", ["x"]), ("h3", None),
    ("f", [["x"] * 8] * 2), ("f", [["x"] * 8] * 4), ("f", [["x"] * 7] * 3), ("f", None), ("f", "abc"),
    ("t", ["x"] * (ETA - 1)), ("t", ["x"] * (ETA + 1)), ("t", None),
])
def test_crypto_refuses_public_parameters_of_the_wrong_length(field, value, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="^{} ".format(field)):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: value})}})


@pytest.mark.parametrize("field,value", [
    ("eta", 0), ("eta", "5"), ("eta", True), ("max_rows", -1), ("max_rows", None), ("max_attributes", 0),
    ("max_attributes", 64.0), ("max_attributes", "64"), ("bound_x", "x"), ("bound_x", 10), ("bound_y", "-1"),
    ("bound_y", "007"), ("bound_x", str(2 ** 49)),
])
def test_crypto_refuses_limits_that_are_not_limits(field, value, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: value})}})


def test_a_tampered_element_of_a_read_field_is_refused_when_it_is_read(make_kp):
    kg, _ = make_kp()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp["t"][0] = pp["g2"]          # an element of G2: charm reads almost any text as some element of G1
    crypto = SIFEKPABPairing({"keys": {"pp": rebound(pp)}})
    for _ in range(2):      # a refusal is not kept as a result
        with pytest.raises(FEValidationError, match="^t "):
            crypto.encrypt(X, attributes=["org:A"])


# ---------------------------------------------------------------------------
# decrypt: the key's policy


@pytest.mark.parametrize("text,reason", [
    (lambda limit: "[" * 100000, "length"),              # over the length limit: refused before from_canonical
    (lambda limit: "[" * (limit + 1), "length"),
    (lambda limit: "[" * (limit - 1), "nesting"),       # just under it: from_canonical's own check of the nesting
    (lambda limit: '["and",' * (limit // 7), "nesting"),
], ids=["100000-brackets", "one-over-the-limit", "just-under-the-limit", "nested-gates"])
def test_hostile_policy(text, reason, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    limit = 64 * (MAX_LABEL_CHARS + 400)
    hostile = text(limit)
    assert (len(hostile) > limit) == (reason == "length")
    message = "more than the limit of {}".format(limit) if reason == "length" else "nests brackets"
    with pytest.raises(FEValidationError, match=message):
        crypto.decrypt(ct, dict(dk, policy=hostile, policy_digest=digest_of_text(hostile)))


def test_a_policy_at_the_length_limit_reaches_the_parser(kp_fixed, monkeypatch):
    _, crypto, ct, dk, _ = kp_fixed
    limit = 64 * (MAX_LABEL_CHARS + 400)
    seen = []

    def spy(data):
        seen.append(len(data))
        raise FEValidationError("stop here")

    monkeypatch.setattr(kp_ab_pairing, "from_canonical", spy)
    with pytest.raises(FEValidationError, match="stop here"):
        crypto.decrypt(ct, dict(dk, policy="[" * limit))
    assert seen == [limit]
    with pytest.raises(FEValidationError, match="more than the limit"):
        crypto.decrypt(ct, dict(dk, policy="[" * (limit + 1)))
    assert seen == [limit]                                          # the longer text did not reach it


def test_the_policy_length_limit_follows_max_rows(make_kp):
    kg, crypto = make_kp(max_rows=2)
    limit = 2 * (MAX_LABEL_CHARS + 400)
    assert crypto.max_policy_length == limit
    ct = crypto.encrypt(X, attributes=["org:A"])
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    with pytest.raises(FEValidationError, match="more than the limit of {}".format(limit)):
        crypto.decrypt(ct, dict(dk, policy="[" * (limit + 1)))


@pytest.mark.parametrize("policy", [None, 5, ["and"]])
def test_decrypt_refuses_a_policy_that_is_no_text(policy, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="policy"):
        crypto.decrypt(ct, dict(dk, policy=policy))


def test_digest_mismatch(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="policy_digest"):
        crypto.decrypt(ct, dict(dk, policy=canonical_encoding(parse("or(org:A, role:auditor)"))))


@pytest.mark.parametrize("make", [
    lambda digest: np.array([0, 1]),                    # `digest != array` is an array, and `if` on it a ValueError
    lambda digest: np.array([digest]),                  # and this one would compare equal and be accepted
    lambda digest: np.array(digest),
    lambda digest: [digest], lambda digest: 5, lambda digest: None,
])
def test_a_digest_that_is_no_text_is_refused_and_not_compared(make, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="policy_digest") as caught:
        crypto.decrypt(ct, dict(dk, policy_digest=make(dk["policy_digest"])))
    assert type(caught.value) is FEValidationError


@pytest.mark.parametrize("spelling", [
    lambda text: text.replace(",", ", "),                                           # another spelling of the formula
    lambda text: " " + text,
    lambda text: text + "\n",
])
def test_a_policy_in_another_spelling_is_refused(spelling, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    bad = spelling(dk["policy"])
    assert bad != dk["policy"]
    with pytest.raises(FEValidationError):
        crypto.decrypt(ct, dict(dk, policy=bad, policy_digest=digest_of_text(bad)))


def test_too_many_rows(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    wide = "or({})".format(", ".join(["org:A"] * 65))
    with pytest.raises(FEValidationError, match="more than the limit of 64 rows"):
        crypto.decrypt(ct, swap_policy(dk, wide))


def test_the_rows_limit_of_the_parameters_is_the_one_that_counts(make_kp):
    kg, crypto = make_kp(max_rows=2)
    ct = crypto.encrypt(X, attributes=["org:A"])
    dk = kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B)", Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)
    with pytest.raises(FEValidationError, match="more than the limit of 2 rows"):
        crypto.decrypt(ct, swap_policy(dk, "or(org:A, org:B, org:C)"))
    with pytest.raises(FEValidationError, match="more than the limit of 2 rows"):
        kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B, org:C)", Y))


# ---------------------------------------------------------------------------
# decrypt: the version markers, fields and shapes


@pytest.mark.parametrize("field", list(KPABCiphertext.FIELDS))
def test_decrypt_refuses_a_ciphertext_without_a_field(field, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(without(ct, field), dk)


@pytest.mark.parametrize("field", list(KPABFunctionalKey.FIELDS))
def test_decrypt_refuses_a_key_without_a_field(field, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, without(dk, field))


@pytest.mark.parametrize("ct_bad", ["not a dict", None, ["c_attr"]])
def test_decrypt_refuses_what_is_no_ciphertext(ct_bad, kp_fixed):
    _, crypto, _, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="ciphertext"):
        crypto.decrypt(ct_bad, dk)


@pytest.mark.parametrize("dk_bad", ["not a dict", None, ["k_rows"]])
def test_decrypt_refuses_what_is_no_key(dk_bad, kp_fixed):
    _, crypto, ct, _, _ = kp_fixed
    with pytest.raises(FEValidationError, match="functional key"):
        crypto.decrypt(ct, dk_bad)


@pytest.mark.parametrize("field,record", [("policy_compiler", "ct"), ("attr_hash", "ct"), ("policy_compiler", "dk")])
def test_decrypt_refuses_an_unknown_version_marker(field, record, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field) as caught:
        if record == "ct":
            crypto.decrypt(dict(ct, **{field: "v" * 100000}), dk)
        else:
            crypto.decrypt(ct, dict(dk, **{field: "v" * 100000}))
    assert len(str(caught.value)) < 300


@pytest.mark.parametrize("field,shape", [
    ("attributes", lambda v: "org:A"), ("attributes", lambda v: None), ("attributes", lambda v: tuple(v)),
    ("attributes", lambda v: [5]), ("attributes", lambda v: ["org:A", "org:A"]),
    ("attributes", lambda v: ["role:auditor", "org:A", "org:B"]), ("attributes", lambda v: ["org"]),
    ("attributes", lambda v: v + ["zzz:1"] * 70),
    ("attributes", lambda v: ["a:{:02d}".format(i) for i in range(65)]),         # 65 distinct, sorted, valid
    ("c_attr", lambda v: v[:-1]), ("c_attr", lambda v: v + v[:1]), ("c_attr", lambda v: [v[0], v[1][:7], v[2]]),
    ("c_attr", lambda v: [v[0], v[1] + v[1][:1], v[2]]), ("c_attr", lambda v: [v[0], "ab", v[2]]),
    ("c_attr", lambda v: None), ("c_attr", lambda v: "abc"),
    ("t", lambda v: v[:-1]), ("t", lambda v: v + v[:1]), ("t", lambda v: None), ("t", lambda v: "ab"),
    ("c_ipfe", lambda v: v[:-1]), ("c_ipfe", lambda v: v + v[:1]), ("c_ipfe", lambda v: None),
])
def test_decrypt_refuses_a_ciphertext_of_the_wrong_shape(field, shape, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="attribute" if field == "attributes" else "^{} ".format(field)):
        crypto.decrypt(dict(ct, **{field: shape(ct[field])}), dk)


@pytest.mark.parametrize("field,shape", [
    ("k_rows", lambda v: v[:1]), ("k_rows", lambda v: v + v[:1]), ("k_rows", lambda v: [v[0], v[1][:7]]),
    ("k_rows", lambda v: [v[0], v[1] + v[1][:1]]), ("k_rows", lambda v: [v[0], "ab"]), ("k_rows", lambda v: None),
    ("k_rows", lambda v: {"a": 1}),
    ("k_ipfe", lambda v: v[:-1]), ("k_ipfe", lambda v: v + v[:1]), ("k_ipfe", lambda v: None),
    ("y", lambda v: v[:-1]), ("y", lambda v: v + v[:1]), ("y", lambda v: None), ("y", lambda v: "1"),
    ("y", lambda v: [1] + v[1:]), ("y", lambda v: ["x"] + v[1:]), ("y", lambda v: ["1.5"] + v[1:]),
    ("y", lambda v: ["11"] + v[1:]), ("y", lambda v: ["-11"] + v[1:]), ("y", lambda v: ["007"] + v[1:]),
    ("y", lambda v: ["1" * 5000] + v[1:]), ("y", lambda v: [" 1"] + v[1:]), ("y", lambda v: ["1_0"] + v[1:]),
    ("y", lambda v: ["١"] + v[1:]),
])
def test_decrypt_refuses_a_key_of_the_wrong_shape(field, shape, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="^{} ".format(field)):
        crypto.decrypt(ct, dict(dk, **{field: shape(dk[field])}))


def test_decrypt_refuses_attributes_over_the_limit_before_it_reads_the_labels(make_kp, monkeypatch):
    kg, crypto = make_kp(max_attributes=2)
    ct = crypto.encrypt(X, attributes=["org:A", "org:B"])
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    monkeypatch.setattr(kp_ab, "validate_label", lambda label: pytest.fail("a label was read"))
    with pytest.raises(FEValidationError, match="max_attributes = 2"):
        crypto.decrypt(dict(ct, attributes=["a:1", "b:1", "c:1"]), dk)


def test_decrypt_refuses_a_label_over_the_length_limit(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    long = label_of(MAX_LABEL_CHARS + 1)
    with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024"):
        crypto.decrypt(dict(ct, attributes=["org:A", "org:B", long]), dk)


def test_decrypt_names_the_labels_that_are_out_of_order(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="'role:auditor' is followed by 'org:B'"):
        crypto.decrypt(dict(ct, attributes=["org:A", "role:auditor", "org:B"]), dk)


# ---------------------------------------------------------------------------
# decrypt: the group elements


def test_decrypt_refuses_a_g2_string_in_a_field_of_g1_before_reading_it(kp_fixed, monkeypatch):
    _, crypto, ct, dk, _ = kp_fixed
    read = []
    real = ac_codec.deserialize_many

    def spy(group, items):
        read.append(items)
        return real(group, items)

    monkeypatch.setattr(ac_codec, "deserialize_many", spy)
    with pytest.raises(FEValidationError, match="^t "):
        crypto.decrypt(dict(ct, t=dk["k_ipfe"][:1] + ct["t"][1:]), dk)
    assert not any(dk["k_ipfe"][0] in items for items in read)


@pytest.mark.parametrize("item", ["junk", "", 5, None])
@pytest.mark.parametrize("field", ["t", "c_ipfe"])
def test_decrypt_refuses_undecodable_ciphertext_elements(field, item, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="^{} ".format(field)):
        crypto.decrypt(dict(ct, **{field: [item] + ct[field][1:]}), dk)


@pytest.mark.parametrize("row", [0, 1, 2], ids=["org:A", "org:B-no-row-of-the-key", "role:auditor"])
@pytest.mark.parametrize("item", ["junk", "", 5, None])
def test_decrypt_decodes_every_vector_of_the_ciphertext_also_the_unused(row, item, kp_fixed):
    # R1-F07: a ciphertext is refused whichever key opens it
    _, crypto, ct, dk, _ = kp_fixed
    c_attr = [list(entry) for entry in ct["c_attr"]]
    c_attr[row][3] = item
    with pytest.raises(FEValidationError, match="c_attr"):
        crypto.decrypt(dict(ct, c_attr=c_attr), dk)


def test_decrypt_refuses_an_element_of_g2_among_the_vectors_of_g1(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    c_attr = [list(entry) for entry in ct["c_attr"]]
    c_attr[1][0] = dk["k_ipfe"][0]
    with pytest.raises(FEValidationError, match="c_attr"):
        crypto.decrypt(dict(ct, c_attr=c_attr), dk)


@pytest.mark.parametrize("field,shape", [
    ("k_ipfe", lambda v: ["junk"] + v[1:]),
    ("k_rows", lambda v: [["junk"] + v[0][1:], v[1]]),
    ("k_rows", lambda v: [v[0], v[1][:3] + [5] + v[1][4:]]),
])
def test_decrypt_refuses_undecodable_key_elements(field, shape, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, dict(dk, **{field: shape(dk[field])}))


@pytest.mark.parametrize("policy,held", [
    ("or(org:A, role:auditor)", ["org:A", "role:auditor"]),
    ("2of(org:A, org:B, role:auditor)", ["org:A", "org:B", "role:auditor"]),
], ids=["or", "threshold"])
def test_a_malformed_element_in_an_unused_key_row_is_not_decoded(policy, held, make_kp):
    # the reconstruction uses the rows of some of the satisfied children only (`or`: one of them). Of the key's
    # other rows the lengths are checked and no element is decoded, so an element that is no group element changes
    # nothing there; in a row that is used it is refused. A ciphertext differs: all its vectors are decoded (R1-F07)
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=held)
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
    weights = compile_policy(parse(policy), crypto.order, 64).reconstruction(ct["attributes"])
    used = sorted(weights)
    unused = [row for row in range(len(dk["k_rows"])) if row not in weights]
    assert used and unused
    for row in unused:
        for item in ("junk", "", 5, None, ct["c_ipfe"][0]):    # the last one is an element of G1, not of G2
            k_rows = [list(entry) for entry in dk["k_rows"]]
            k_rows[row][3] = item
            assert crypto.decrypt(ct, dict(dk, k_rows=k_rows)) == inner(X, Y), (row, item)
        shortened = [list(entry) for entry in dk["k_rows"]]
        shortened[row] = shortened[row][:7]                    # its length is checked
        with pytest.raises(FEValidationError, match="^k_rows "):
            crypto.decrypt(ct, dict(dk, k_rows=shortened))
    for row in used:
        k_rows = [list(entry) for entry in dk["k_rows"]]
        k_rows[row][3] = "junk"
        with pytest.raises(FEValidationError, match="k_rows"):
            crypto.decrypt(ct, dict(dk, k_rows=k_rows))


def test_decrypt_refuses_elements_of_the_wrong_group(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="c_ipfe"):
        crypto.decrypt(dict(ct, c_ipfe=dk["k_ipfe"]), dk)
    with pytest.raises(FEValidationError, match="k_ipfe"):
        crypto.decrypt(ct, dict(dk, k_ipfe=ct["c_ipfe"]))
    with pytest.raises(FEValidationError, match="k_rows"):
        crypto.decrypt(ct, dict(dk, k_rows=ct["c_attr"][:2]))


# ---------------------------------------------------------------------------
# hostile text in error messages


HUGE = 100000


def test_huge_attributes_are_clipped_in_the_messages(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    for attributes in (["org:A", "org:" + "x" * HUGE], ["org:{}".format(i) for i in range(HUGE // 10)]):
        with pytest.raises(FEValidationError) as caught:
            crypto.decrypt(dict(ct, attributes=attributes), dk)
        assert len(str(caught.value)) < 300


def test_the_message_of_an_unsatisfied_policy_names_the_attributes_and_is_clipped(kp_fixed):
    _, crypto, _, dk, _ = kp_fixed
    with pytest.raises(FEKeyError, match="org:A"):
        crypto.decrypt(crypto.encrypt(X, attributes=["org:A"]), dk)
    with pytest.raises(FEKeyError) as caught:
        crypto.decrypt(crypto.encrypt(X, attributes=["org:{}".format(i) for i in range(64)]), dk)
    assert len(str(caught.value)) < 300
