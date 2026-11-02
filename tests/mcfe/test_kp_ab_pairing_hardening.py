"""Negative tests of the multi-client key-policy attribute-based IPFE (mcfe/kp_ab_pairing): configurations, key requests
and encryptions that are refused, public parameters and client keys that are not these, ciphertexts and keys that are
put together wrongly, made under other parameters or hostile, and the attacks that leave every field well formed.

A key whose policy the attributes of the ciphertexts do not satisfy gives a ``FEKeyError``; malformed input, ciphertexts
that are not those of every client under one tag and one set of attributes, or objects of another setup a
``FEValidationError``; a key and ciphertexts that are well formed but were not made for each other a ``ValueError`` from
the discrete log, which is not a ``FEValidationError``. Where test_kp_ab_pairing.py checks that the scheme works, these
tests are the refusals and the attacks.

The ``label`` and the ``attributes_digest`` of a ciphertext are claims of its client: the points ``H(tag, S)`` are
inside ``t`` and ``c_ipfe``, and the master key would be needed to check the fields against them. A decryption checks
that the clients agree, so that one client that lies gives no value; all clients that change the label alike still give
the right one; the list of attributes is read by the decryption (it says which vector pairs with which row), so
changing it gives a refusal or no value, never a wrong one
(``test_the_label_and_the_digest_of_a_ciphertext_are_claims``).
"""

import copy
import hashlib
import re
import threading

import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.mcfe.conftest import ATTRS, EXPECTED, JUNK_G2, LST, POLICY, X, Y
from tests.mcfe.conftest import aggregate, combined, creds, encrypt_all, encrypt_arrays, key_part
from tests.mcfe.conftest import masked_plaintext, random_arrays
from tests.mcfe.conftest import refuse_to_compute
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats.pairing import MCKPABCiphertext, MCKPABClientKey, MCKPABFunctionalKey, MCKPABParameters
from pyfe4ai.policy import attribute_set_digest, canonical_encoding, parse
from pyfe4ai.primitives.pairing import pairing_backend
from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup, pair
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.kp_ab import MAX_LABEL_CHARS, tag_points
from pyfe4ai.schemes.mcfe import kp_ab_pairing
from pyfe4ai.schemes.mcfe.kp_ab_pairing import MCFEKPABPairing, MCFEKPABPairingKeyGenerator
from pyfe4ai.schemes.mcfe.labels import ElementLabel, encode_label

PP_DOMAIN = b"pyfe4ai/mcfe/kp-ab/pp-binding/v1|"
SCHEME_ID = "mcfe/kp_ab_pairing"
SURROGATE_PAIR = chr(0xD83D) + chr(0xDE00)      # the two UTF-16 halves of U+1F600, as two lone surrogates

pytestmark = pytest.mark.usefixtures("own_parameter_cache")


# ---------------------------------------------------------------------------
# the configuration and the key requests that are refused


def test_clients_are_needed(own_parameter_cache):
    with pytest.raises(FEConfigError, match="n"):
        MCFEKPABPairingKeyGenerator({"pairing_group_param": GROUP})


@pytest.mark.parametrize("n", [0, -1, True, 2.0, "3"])
def test_n_must_be_a_positive_int(n, own_parameter_cache):
    with pytest.raises(FEValidationError, match="n must be a positive int"):
        MCFEKPABPairingKeyGenerator({"n": n, "pairing_group_param": GROUP})


@pytest.mark.parametrize("lst_nid", [["a", "b"], ["a", "b", "a"], ["a", "b", 1], "abc", [None, None, None]])
def test_the_client_ids_must_be_n_distinct_ids(lst_nid, own_parameter_cache):
    with pytest.raises(FEValidationError, match="lst_nid"):
        MCFEKPABPairingKeyGenerator({"n": 3, "lst_nid": lst_nid, "pairing_group_param": GROUP})


@pytest.mark.parametrize("option", ["bound_x", "bound_y", "max_rows", "max_attributes"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "5", None])
def test_limits_must_be_positive_ints(option, value, own_parameter_cache):
    with pytest.raises(FEValidationError, match=option):
        MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": GROUP, option: value})


@pytest.mark.parametrize("eta", [2, 0, True, 1.0, "1", None])
def test_eta_is_one(eta, own_parameter_cache):
    with pytest.raises(FEValidationError, match="eta must be 1"):
        MCFEKPABPairingKeyGenerator({"n": 3, "eta": eta, "pairing_group_param": GROUP})
    assert MCFEKPABPairingKeyGenerator({"n": 3, "eta": 1, "pairing_group_param": GROUP}).eta == 1


def test_dlog_limit(own_parameter_cache):
    config = {"n": 3, "bound_x": 2 ** 24, "bound_y": 2 ** 24, "pairing_group_param": GROUP}
    with pytest.raises(FEValidationError, match=r"n \* bound_x \* bound_y.*2\*\*48"):
        MCFEKPABPairingKeyGenerator(config)
    assert MCFEKPABPairingKeyGenerator(dict(config, n=1)).n == 1            # 2**48 itself is allowed


def test_a_symmetric_group_is_refused(own_parameter_cache):
    if not pairing_backend.pairing_available("SS512"):
        pytest.skip("no charm: SS512 is not built")
    with pytest.raises(FEValidationError, match="symmetric"):
        MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": "SS512"})


def test_keys_before_setup(own_parameter_cache):
    kg = MCFEKPABPairingKeyGenerator({"n": 3, "pairing_group_param": GROUP})
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_private_keys("nid_0")
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_decryption_keys("agg", credentials=creds())


@pytest.mark.parametrize("credentials", [
    None, {}, {"policy": POLICY}, {"fusion_weight": {nid: [1] for nid in LST}},
    {"fusion_weight": {nid: [1] for nid in LST[:2]}, "policy": POLICY},                     # a client missing
    {"fusion_weight": {**{nid: [1] for nid in LST}, "nid_9": [1]}, "policy": POLICY},       # a client too many
    {"fusion_weight": {**{nid: [1] for nid in LST}, 1: [1]}, "policy": POLICY},
    {"fusion_weight": {**{nid: [1] for nid in LST[:2]}, True: [1]}, "policy": POLICY},      # True is no client id
    {"fusion_weight": {nid: [11] for nid in LST}, "policy": POLICY},                        # above bound_y
    {"fusion_weight": {nid: [1, 1] for nid in LST}, "policy": POLICY},                      # eta = 1
    {"fusion_weight": {nid: [] for nid in LST}, "policy": POLICY},
    {"fusion_weight": {nid: 1 for nid in LST}, "policy": POLICY},
    {"fusion_weight": {nid: [1.5] for nid in LST}, "policy": POLICY},
    {"fusion_weight": [1, 1, 1], "policy": POLICY},
    {"fusion_weight": {nid: [1] for nid in LST}, "policy": None},
    {"fusion_weight": {nid: [1] for nid in LST}, "policy": "or(region:EU,"},
    {"fusion_weight": {nid: [1] for nid in LST}, "attributes": ["region:EU"]},
    ["fusion_weight", "policy"],
])
def test_key_requests_are_checked(make_mc, credentials):
    kg, _, _ = make_mc()
    with pytest.raises((FEKeyError, FEValidationError)):
        kg.get_decryption_keys("agg", credentials=credentials)


def test_a_policy_of_more_rows_than_the_limit_is_refused(make_mc):
    kg, _, _ = make_mc(max_rows=2)
    with pytest.raises(FEValidationError, match="max_rows|rows"):
        kg.get_decryption_keys("agg", credentials=creds("and(a:1, b:1, c:1)"))
    assert kg.get_decryption_keys("agg", credentials=creds("and(a:1, b:1)"))


# ---------------------------------------------------------------------------
# encryption that is refused


def test_encrypt_without_a_key_is_a_key_error(make_mc):
    _, _, agg = make_mc()
    with pytest.raises(FEKeyError, match="sk"):
        agg.encrypt([1], "round-1", attributes=ATTRS)


@pytest.mark.parametrize("plaintext", [[11], [-11], [], [1, 2], 5, "1", [1.5], [True], None])
def test_encrypt_checks_the_plaintext(plaintext, make_mc):
    _, clients, _ = make_mc()
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt(plaintext, "round-1", attributes=ATTRS)


@pytest.mark.parametrize("label", [None, 5, b"round-1", ["round-1"], ("round-1", 0)])
def test_encrypt_checks_the_label(label, make_mc):
    _, clients, _ = make_mc()
    with pytest.raises(FEValidationError, match="label"):
        clients["nid_0"].encrypt([1], label, attributes=ATTRS)


@pytest.mark.parametrize("attributes", [None, "region:EU", 5, ["region"], [5], ["region:EU", None]])
def test_encrypt_checks_the_attributes(attributes, make_mc):
    _, clients, _ = make_mc()
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1], "round-1", attributes=attributes)


def test_encrypt_limits_the_attributes(make_mc):
    _, clients, _ = make_mc(max_attributes=2)
    with pytest.raises(FEValidationError, match="max_attributes"):
        clients["nid_0"].encrypt([1], "round-1", attributes=["a:1", "b:1", "c:1"])
    assert clients["nid_0"].encrypt([1], "round-1", attributes=["a:1", "b:1", "a:1"])


def test_a_refused_encryption_computes_nothing(make_mc, monkeypatch):
    _, clients, _ = make_mc()

    def never(*args, **kwargs):
        pytest.fail("a point was hashed or a group element was combined for input that is refused")

    monkeypatch.setattr(kp_ab_pairing, "tag_points", never)
    monkeypatch.setattr(kp_ab_pairing, "multiexp_many", never)
    for plaintext, label, attributes in (([99], "round-1", ATTRS), ([1], 5, ATTRS), ([1], "round-1", None)):
        with pytest.raises(FEValidationError):
            clients["nid_0"].encrypt(plaintext, label, attributes=attributes)


@pytest.mark.parametrize("args", [(1, "a", "b", "c"), ("a", b"b", "c", "d"), ("a", "b", None, "d"),
                                  ("a", "b", "c", 5), ("a", "b", "c", b"d")])
def test_tag_points_refuses_what_is_no_text(args):
    with pytest.raises(FEValidationError):
        tag_points(PairingGroup(GROUP), *args)


# ---------------------------------------------------------------------------
# the crypto object: public parameters and encryption keys


def rebound(pp, **changes):
    """*pp* with *changes* and the ``pp_binding`` that goes with them."""
    changed = dict(copy.deepcopy(pp), **changes)
    changed["pp_binding"] = pp_digest(changed, PP_DOMAIN)
    return changed


def test_crypto_needs_public_parameters():
    with pytest.raises(FEKeyError, match="pp"):
        MCFEKPABPairing({"keys": {"other": 1}})
    with pytest.raises(FEKeyError):
        MCFEKPABPairing({"keys": {}})


def test_tampered_pp(make_mc):
    kg, _, _ = make_mc()
    pp = kg.get_public_parameters()
    tampered = dict(pp, bound_x="1000")
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": tampered}})
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": dict(pp, lst_nid=["nid_0", "nid_2", "nid_1"])}})   # the order fixes p_i


@pytest.mark.parametrize("value", [
    lambda good: np.array([0, 1]), lambda good: np.array([good]), lambda good: np.array(good),
    lambda good: [good], lambda good: 5, lambda good: None,
], ids=["array", "array-of-it", "0-d-array", "list", "int", "none"])
def test_crypto_refuses_parameters_whose_pp_binding_is_no_text(value, make_mc):
    # the digest leaves pp_binding out, so an array there is JSON enough to be compared with it
    kg, _, _ = make_mc()
    pp = kg.get_public_parameters()
    with pytest.raises(FEValidationError, match="pp_binding") as caught:
        MCFEKPABPairing({"keys": {"pp": dict(pp, pp_binding=value(pp["pp_binding"]))}})
    assert type(caught.value) is FEValidationError


@pytest.mark.parametrize("field", ["policy_compiler", "attr_hash", "tag_hash"])
def test_crypto_refuses_an_unknown_version_marker(field, make_mc):
    kg, _, _ = make_mc()
    with pytest.raises(FEValidationError, match=field):
        MCFEKPABPairing({"keys": {"pp": rebound(kg.get_public_parameters(), **{field: "v9"})}})


@pytest.mark.parametrize("changes,match", [
    ({"eta": 2}, "eta"), ({"eta": True}, "eta"), ({"n": 4}, "lst_nid|clients"), ({"n": "3"}, "n must be"),
    ({"lst_nid": ["nid_0", "nid_1"]}, "lst_nid"), ({"lst_nid": "abc"}, "lst_nid"),
    ({"lst_nid": ["nid_0", "nid_1", "nid_1"]}, "lst_nid"), ({"lst_nid": ["nid_0", "nid_1", 2]}, "lst_nid"),
    ({"bound_x": "1" * 17}, "bound_x"), ({"bound_y": 3}, "bound_y"), ({"max_rows": 0}, "max_rows"),
    ({"max_attributes": True}, "max_attributes"), ({"bound_x": str(2 ** 40), "bound_y": str(2 ** 40)}, "2\\*\\*48"),
    ({"pairing_group_param": 5}, "pairing_group_param"), ({"pairing_group_param": "nothing"}, "pairing group"),
])
def test_crypto_refuses_parameters_that_are_not_these(changes, match, make_mc):
    kg, _, _ = make_mc()
    with pytest.raises(FEValidationError, match=match):
        MCFEKPABPairing({"keys": {"pp": rebound(kg.get_public_parameters(), **changes)}})


@pytest.mark.parametrize("field", MCKPABParameters.FIELDS)
def test_crypto_refuses_public_parameters_without_a_field(field, make_mc):
    kg, _, _ = make_mc()
    pp = dict(kg.get_public_parameters())
    del pp[field]
    with pytest.raises(FEValidationError, match=field):
        MCFEKPABPairing({"keys": {"pp": pp}})


def test_parameters_changed_after_construction_do_not_reach_the_object(make_mc):
    kg, clients, agg = make_mc()
    pp = copy.deepcopy(kg.get_public_parameters())
    sk = kg.get_private_keys("nid_0")
    client = MCFEKPABPairing({"id": "nid_0", "keys": {"pp": pp, "sk": sk}})
    pp["bound_x"], pp["lst_nid"], sk["s"] = "1", ["x"], "0"
    sk["p_h1"][0] = "0"
    cts = encrypt_all(clients)
    cts["nid_0"] = client.encrypt([X["nid_0"]], "round-1", attributes=ATTRS)
    assert client.bound_x == 10 and client.lst_nid == LST
    assert agg.decrypt(cts, kg.get_decryption_keys("agg", credentials=creds())) == EXPECTED


def test_parameters_that_cannot_be_copied_are_refused(make_mc):
    kg, _, _ = make_mc()
    pp = dict(kg.get_public_parameters(), extra=threading.Lock())
    with pytest.raises(FEValidationError, match="public parameters"):
        MCFEKPABPairing({"keys": {"pp": pp}})


def test_a_client_key_of_another_setup_is_refused(make_mc):
    kg, _, _ = make_mc()
    other, _, _ = make_mc()
    with pytest.raises(FEValidationError, match="pp_binding"):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": other.get_private_keys("nid_0")}})


def test_the_id_of_a_client_must_be_the_one_of_its_key(make_mc):
    kg, _, _ = make_mc()
    pp = kg.get_public_parameters()
    with pytest.raises(FEValidationError, match="nid_1"):
        MCFEKPABPairing({"id": "nid_1", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    assert MCFEKPABPairing({"keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}}).encrypt(
        [1], "t", attributes=[])["nid"] == "nid_0"


@pytest.mark.parametrize("field,value,match", [
    ("nid", "stranger", "no client"), ("nid", ["nid_0"], "no client"), ("nid", True, "no client"),
    ("s", "-1", "s must hold"), ("s", "01", "s must hold"), ("s", "x", "s must hold"), ("s", 5, "s must hold"),
    ("s", "9" * 200, "s must hold"), ("u", "", "u must hold"),
    ("p_h1", ["1", "2", "3"], "p_h1"), ("p_h1", "1234", "p_h1"), ("p_h1", ["1", "2", "3", "x"], "p_h1"),
    ("p_h2", ["1", "2", "3", "4", "5"], "p_h2"), ("p_h2", [1, 2, 3, 4], "p_h2"),
    ("h3", ["1:x"] * 3, "h3"), ("f", [["1:x"] * 8] * 2, "f"), ("f", [["1:x"] * 8, ["1:x"] * 8, ["1:x"] * 7], "f"),
])
def test_crypto_refuses_a_malformed_client_key(field, value, match, make_mc):
    kg, _, _ = make_mc()
    sk = kg.get_private_keys("nid_0")
    sk[field] = value
    with pytest.raises(FEValidationError, match=match):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


def test_a_scalar_must_be_below_the_group_order(make_mc):
    kg, _, _ = make_mc()
    order = int(PairingGroup(GROUP).order())
    sk = kg.get_private_keys("nid_0")
    sk["s"] = str(order)
    with pytest.raises(FEValidationError, match="below the group order"):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})
    sk["s"] = str(order - 1)
    MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


@pytest.mark.parametrize("sk", [None, [], "key", 5])
def test_the_client_key_must_be_a_dict(sk, make_mc):
    kg, _, _ = make_mc()
    with pytest.raises((FEKeyError, FEValidationError)):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


@pytest.mark.parametrize("field", MCKPABClientKey.FIELDS)
def test_crypto_refuses_a_client_key_without_a_field(field, make_mc):
    kg, _, _ = make_mc()
    sk = kg.get_private_keys("nid_0")
    del sk[field]
    with pytest.raises(FEValidationError, match=field):
        MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})


def test_a_tampered_element_of_the_client_key_is_refused_when_it_is_read(make_mc):
    kg, _, _ = make_mc()
    sk = kg.get_private_keys("nid_0")
    sk["h3"][0] = kg.get_decryption_keys("agg", credentials=creds())["k_ipfe"]["nid_0"][0]       # an element of G2
    client = MCFEKPABPairing({"keys": {"pp": kg.get_public_parameters(), "sk": sk}})
    with pytest.raises(FEValidationError, match="h3"):
        client.encrypt([1], "round-1", attributes=ATTRS)


# ---------------------------------------------------------------------------
# decryption: the clients, the tag and the attributes


def test_every_client_is_needed(make_mc, monkeypatch):  # Review Focus 2
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients)
    dk = kg.get_decryption_keys("agg", credentials=creds(y=dict(Y, nid_0=0)))      # nid_0 has the weight 0
    refuse_to_compute(monkeypatch)
    for missing in LST:
        with pytest.raises(FEValidationError, match=missing):
            agg.decrypt({nid: ct for nid, ct in cts.items() if nid != missing}, dk)
    with pytest.raises(FEValidationError, match="nid_1.*nid_2|nid_2.*nid_1"):
        agg.decrypt({"nid_0": cts["nid_0"]}, dk)
    with pytest.raises(FEValidationError, match="nid_0"):
        agg.decrypt({}, dk)
    with pytest.raises(FEValidationError, match="nid_9"):                           # a ciphertext of no client
        agg.decrypt(dict(cts, nid_9=cts["nid_0"]), dk)


def test_a_ciphertext_in_the_slot_of_another_client_is_refused(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    refuse_to_compute(monkeypatch)
    swapped = dict(cts, nid_0=cts["nid_1"], nid_1=cts["nid_0"])
    with pytest.raises(FEValidationError, match="nid_0.*nid_1"):
        agg.decrypt(swapped, dk)


@pytest.mark.parametrize("owner,field", [("nid_0", "label"), ("nid_1", "label"), ("nid_2", "label"),
                                         ("nid_0", "attributes_digest"), ("nid_2", "attributes_digest")])
def test_ciphertexts_that_differ_in_the_tag_or_the_attributes_are_refused(owner, field, mc_fixed, monkeypatch):
    _, clients, agg, cts, dk = mc_fixed
    if field == "label":
        cts[owner] = clients[owner].encrypt([X[owner]], "round-2", attributes=ATTRS)
    else:
        cts[owner] = clients[owner].encrypt([X[owner]], "round-1", attributes=ATTRS + ["org:A"])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="differ in their {}".format(field)):
        agg.decrypt(cts, dk)


def test_attributes_that_are_not_those_of_their_digest_are_refused(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    cts["nid_1"]["attributes"] = ATTRS + ["zzz:1"]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="attributes_digest of the ciphertext of nid_1"):
        agg.decrypt(cts, dk)
    cts["nid_1"]["attributes"] = ["region:EU", "purpose:research"]                  # out of order
    with pytest.raises(FEValidationError, match="sorted"):
        agg.decrypt(cts, dk)


def test_the_label_argument_must_be_the_one_of_the_ciphertexts(mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    assert agg.decrypt(cts, dk, label="round-1") == EXPECTED
    assert agg.decrypt(cts, dk, label=None) == EXPECTED
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt(cts, dk, label="round-2")
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt(cts, dk, label=5)
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt(cts, dk, label=ElementLabel("round-1", 0, (0,)))


def test_the_fusion_weight_must_be_the_weights_of_the_key(mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    assert agg.decrypt(cts, dk, fusion_weight={nid: [Y[nid]] for nid in LST}) == EXPECTED
    for wrong in ({**{nid: [Y[nid]] for nid in LST}, "nid_2": [-3]},
                  {nid: [Y[nid]] for nid in LST[:2]},
                  {**{nid: [Y[nid]] for nid in LST}, "nid_9": [1]},
                  {nid: [Y[nid], 0] for nid in LST},
                  {nid: Y[nid] for nid in LST},
                  [Y[nid] for nid in LST], "fusion"):
        with pytest.raises(FEValidationError, match="fusion"):
            agg.decrypt(cts, dk, fusion_weight=wrong)


@pytest.mark.parametrize("nid_key,match", [(1.5, "no client id"), (None, "no client id"), (True, "no client id"),
                                           (("nid_0",), "no client id"), (1, "not a client")])
def test_decrypt_refuses_keys_that_are_no_clients(nid_key, match, mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    with pytest.raises(FEValidationError, match=match):
        agg.decrypt({nid_key: cts["nid_0"], "nid_1": cts["nid_1"], "nid_2": cts["nid_2"]}, dk)


@pytest.mark.parametrize("field,value,match", [
    ("attributes", "region:EU", "attributes must be a list"),               # a str would be read a character at a time
    ("attributes", ("purpose:research", "region:EU"), "attributes must be a list"),
    ("attributes", {"purpose:research", "region:EU"}, "attributes must be a list"),
    ("attributes", None, "attributes must be a list"),
    ("attributes", [5, 6], "label"),
    ("attributes", ["purpose:research", None], "label"),
    ("attributes", ["purpose:research", "purpose:research"], "sorted and hold each label once"),
    ("attributes", ["region:EU", "purpose:research"], "sorted and hold each label once"),
    ("attributes", ["a:%03d" % i for i in range(65)], "max_attributes = 64"),    # over the limit, labels not looked at
    ("attributes", ["a:" + "x" * MAX_LABEL_CHARS], "MAX_LABEL_CHARS"),
    ("attributes", ["purpose:research", ""], "label"),
    ("attributes_digest", 5, "attributes_digest of the ciphertext of nid_1 must be a str"),
    ("attributes_digest", None, "attributes_digest of the ciphertext of nid_1 must be a str"),
    ("attributes_digest", b"00", "attributes_digest of the ciphertext of nid_1 must be a str"),
    ("label", 5, "label of the ciphertext of nid_1 must be a str"),
    ("label", None, "label of the ciphertext of nid_1 must be a str"),
    ("label", ["round-1"], "label of the ciphertext of nid_1 must be a str"),
    ("nid", None, "client id"), ("nid", 1.5, "client id"), ("nid", True, "client id"), ("nid", ["nid_1"], "client id"),
    ("nid", np.array([0, 1]), "client id"), ("nid", np.array(["nid_1", "nid_1"]), "client id"),
    ("nid", "nid_2", "nid_1.*nid_2"), ("nid", 7, "nid_1.*7"),
])
def test_hostile_ciphertexts(field, value, match, mc_fixed, monkeypatch):
    # the attribute list, its length, its types, and the fields that name the client and the tag
    _, _, agg, cts, dk = mc_fixed
    cts["nid_1"][field] = value
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=match):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("record,field", [
    ("ct", "policy_compiler"), ("ct", "attr_hash"), ("ct", "tag_hash"), ("ct", "pp_binding"),
    ("key", "policy_compiler"), ("key", "pp_binding"), ("key", "policy_digest"),
])
def test_an_array_in_a_field_is_refused_and_not_compared(record, field, mc_fixed, monkeypatch):
    # an array compared with a string is an array, and `if` on it raises a ValueError, which would read as the
    # ValueError of "no discrete log"
    _, _, agg, cts, dk = mc_fixed
    (dk if record == "key" else cts["nid_1"])[field] = np.array([0, 1])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("label", [
    ElementLabel("round-1", "x", (0,)), ElementLabel("round-1", None, (0,)), ElementLabel("round-1", 0, (None,)),
    ElementLabel("round-1", 0, 5), ElementLabel("round-1", float("nan"), (0,)),
    SURROGATE_PAIR, chr(0xD83D), ElementLabel(SURROGATE_PAIR, 0, (0,)),
])
def test_a_label_that_cannot_be_encoded_is_a_validation_error_not_a_value_error(label, mc_fixed, monkeypatch):
    # `decrypt` keeps ValueError for "no discrete log"; the label argument and the label of `encrypt` get the same
    # check as any other
    _, clients, agg, cts, dk = mc_fixed
    with pytest.raises(FEValidationError, match="label|layer and the index"):
        clients["nid_0"].encrypt([1], label, attributes=ATTRS)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="layer and the index|surrogate"):
        agg.decrypt(cts, dk, label=label)


# ---------------------------------------------------------------------------
# decryption: what is refused before anything is paired


def test_a_key_of_another_setup_does_not_open(make_mc):
    kg, clients, agg = make_mc()
    other_kg, _, _ = make_mc()
    with pytest.raises(FEValidationError, match="pp_binding"):
        agg.decrypt(encrypt_all(clients), other_kg.get_decryption_keys("agg", credentials=creds()))


def test_decrypt_checks_the_bindings_before_anything_else(mc_fixed, make_mc, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    other_kg, other_clients, _ = make_mc()
    other_cts = encrypt_all(other_clients)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="ciphertext of nid_1.*pp_binding"):
        agg.decrypt(dict(cts, nid_1=other_cts["nid_1"]), dk)
    with pytest.raises(FEValidationError, match="functional key.*pp_binding"):
        agg.decrypt(cts, other_kg.get_decryption_keys("agg", credentials=creds()))
    no_binding = {nid: {k: v for k, v in ct.items() if k != "pp_binding"} for nid, ct in cts.items()}
    with pytest.raises(FEValidationError, match="no pp_binding"):
        agg.decrypt(no_binding, dk)


@pytest.mark.parametrize("field", MCKPABCiphertext.FIELDS)
def test_decrypt_refuses_a_ciphertext_without_a_field(field, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    del cts["nid_1"][field]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field", MCKPABFunctionalKey.FIELDS)
def test_decrypt_refuses_a_key_without_a_field(field, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    del dk[field]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,record", [("policy_compiler", "key"), ("policy_compiler", "ct"),
                                          ("attr_hash", "ct"), ("tag_hash", "ct")])
def test_decrypt_refuses_an_unknown_version_marker(field, record, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    (dk if record == "key" else cts["nid_2"])[field] = "v9"
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("bad", [None, [], "cts", 5, [("nid_0", {})]])
def test_decrypt_refuses_what_is_no_dict_of_ciphertexts(bad, mc_fixed):
    _, _, agg, _, dk = mc_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt(bad, dk)


@pytest.mark.parametrize("bad", [None, [], "key", 5])
def test_decrypt_refuses_what_is_no_key(bad, mc_fixed):
    _, _, agg, cts, _ = mc_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt(cts, bad)


@pytest.mark.parametrize("field,shape", [
    ("c_attr", []), ("c_attr", [["1:x"] * 8]), ("c_attr", [["1:x"] * 7, ["1:x"] * 8]), ("c_attr", "x"),
    ("c_ipfe", ["1:x"] * 3), ("c_ipfe", "1:x"), ("t", ["1:x"]), ("t", 5), ("t", None),
])
def test_decrypt_refuses_a_ciphertext_of_the_wrong_shape(field, shape, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    cts["nid_2"][field] = shape
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("field,change", [
    ("y", lambda value: {k: v for k, v in value.items() if k != "nid_1"}),
    ("y", lambda value: dict(value, nid_9="1")),
    ("y", lambda value: [value["nid_0"]]),
    ("k_rows", lambda value: {k: v for k, v in value.items() if k != "nid_2"}),
    ("k_rows", lambda value: dict(value, nid_2=value["nid_2"][:1])),
    ("k_rows", lambda value: dict(value, nid_2=[value["nid_2"][0], value["nid_2"][1][:7]])),
    ("k_rows", lambda value: dict(value, nid_2="rows")),
    ("k_ipfe", lambda value: dict(value, nid_0=value["nid_0"][:3])),
    ("k_ipfe", lambda value: {k: v for k, v in value.items() if k != "nid_0"}),
    ("k_ipfe", lambda value: [value["nid_0"]]),
])
def test_decrypt_refuses_a_key_of_the_wrong_shape(field, change, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    dk[field] = change(dk[field])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=field):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("bad,match", [
    ("11", "y of nid_2 is 11, beyond the limit bound_y = 10"),
    ("-11", "y of nid_2 is -11, beyond the limit bound_y = 10"),
    (3, "y of nid_2 must hold decimal strings"), ("01", "y of nid_2 must hold decimal strings"),
    ("", "y of nid_2 must hold decimal strings"), ("1" * 17, "y of nid_2 must hold decimal strings"),
    (None, "y of nid_2 must hold decimal strings"), ([1], "y of nid_2 must hold decimal strings"),
])
def test_a_y_that_is_refused_names_its_client(bad, match, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    dk["y"]["nid_2"] = bad
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=match):
        agg.decrypt(cts, dk)


def test_decrypt_decodes_every_vector_of_the_ciphertext_also_the_unused(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=ATTRS + ["org:A"])            # org:A is no row of the policy
    dk = kg.get_decryption_keys("agg", credentials=creds())
    assert agg.decrypt(cts, dk) == EXPECTED
    cts["nid_1"]["c_attr"][0][3] = "1:not an element"
    with pytest.raises(FEValidationError, match="c_attr"):
        agg.decrypt(cts, dk)
    cts["nid_1"]["c_attr"][0][3] = dk["k_ipfe"]["nid_0"][0]             # an element of G2
    with pytest.raises(FEValidationError, match="c_attr"):
        agg.decrypt(cts, dk)


def test_a_malformed_element_in_an_unused_key_row_is_not_decoded(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=["region:EU"])
    dk = kg.get_decryption_keys("agg", credentials=creds("or(region:EU, org:A)"))      # the first leaf is used
    dk["k_rows"]["nid_1"][1][2] = JUNK_G2
    assert agg.decrypt(cts, dk) == EXPECTED
    dk["k_rows"]["nid_1"][0][2] = JUNK_G2
    with pytest.raises(FEValidationError, match="k_rows"):
        agg.decrypt(cts, dk)


def test_decrypt_refuses_elements_of_the_wrong_group(mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    keys = copy.deepcopy(dk)
    keys["k_ipfe"]["nid_0"][1] = cts["nid_0"]["c_ipfe"][0]                          # G1 where G2 is due
    with pytest.raises(FEValidationError, match="k_ipfe"):
        agg.decrypt(cts, keys)
    cts["nid_0"]["t"] = dk["k_ipfe"]["nid_0"][0]                                    # G2 where G1 is due
    with pytest.raises(FEValidationError, match="t of nid_0"):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("text,reason", [
    ("and(", "parse|policy|canonical"), ("", "parse|policy|canonical"), ("a" * 100000, "characters|limit"),
    (canonical_encoding(parse("org:A")), "digest"),
])
def test_hostile_policy(text, reason, mc_fixed, monkeypatch):
    _, _, agg, cts, dk = mc_fixed
    dk["policy"] = text
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=reason):
        agg.decrypt(cts, dk)


@pytest.mark.parametrize("policy", [None, 5, ["region:EU"], b"region:EU"])
def test_decrypt_refuses_a_policy_that_is_no_text(policy, mc_fixed):
    _, _, agg, cts, dk = mc_fixed
    dk["policy"] = policy
    with pytest.raises(FEValidationError, match="policy"):
        agg.decrypt(cts, dk)


# ---------------------------------------------------------------------------
# attacks that leave every field well formed


def test_a_client_that_hashed_other_points_than_it_says_gives_no_value(mc_fixed):       # Review Focus 3
    # nid_1 hashed "round-2" and says "round-1": every field agrees, and there is no value, not a wrong one
    _, clients, agg, cts, dk = mc_fixed
    hashed_other = clients["nid_1"].encrypt([X["nid_1"]], "round-2", attributes=ATTRS)
    cts["nid_1"] = dict(hashed_other, label=cts["nid_0"]["label"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)


def test_a_client_that_hashed_another_set_of_attributes_than_it_says_gives_no_value(make_mc):
    kg, clients, agg = make_mc()
    cts = encrypt_all(clients, attributes=["a:1", "b:1"])
    cts["nid_1"] = clients["nid_1"].encrypt([X["nid_1"]], "round-1", attributes=["a:1", "c:1"])
    cts["nid_1"]["attributes"] = ["a:1", "b:1"]                         # the row of a:1 is where the key reads it
    cts["nid_1"]["attributes_digest"] = attribute_set_digest(["a:1", "b:1"])
    dk = kg.get_decryption_keys("agg", credentials=creds("a:1"))
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)


def test_the_label_and_the_digest_of_a_ciphertext_are_claims(mc_fixed):
    # the points of the tag are inside t and c_ipfe; the fields are checked against each other and cannot be checked
    # against the points without the master key. Ciphertexts are not authenticated (module docstring).
    kg, clients, agg, cts, dk = mc_fixed
    for ct in cts.values():
        ct["label"] = encode_label("round-2")
    assert agg.decrypt(cts, dk) == EXPECTED
    assert agg.decrypt(cts, dk, label="round-2") == EXPECTED

    # the list of attributes is no claim of that kind: it says which vector of c_attr pairs with which row of the key.
    # All clients that change it alike get a refusal or no value, never a wrong one.
    cts = encrypt_all(clients, attributes=["b:1", "c:1"])               # the vector of b:1 first, then that of c:1

    def relabelled(attributes):
        return {nid: dict(ct, attributes=list(attributes), attributes_digest=attribute_set_digest(attributes))
                for nid, ct in cts.items()}

    both = kg.get_decryption_keys("agg", credentials=creds("and(b:1, c:1)"))
    only_b = kg.get_decryption_keys("agg", credentials=creds("b:1"))
    assert agg.decrypt(cts, both) == agg.decrypt(cts, only_b) == EXPECTED
    with pytest.raises(FEKeyError, match="do not satisfy"):             # c:1 renamed: the policy is not satisfied
        agg.decrypt(relabelled(["b:1", "d:1"]), both)
    with pytest.raises(ValueError, match="discrete log") as error:      # b:1 now stands where the vector of c:1 is
        agg.decrypt(relabelled(["a:1", "b:1"]), only_b)
    assert not isinstance(error.value, FEValidationError)
    with pytest.raises(FEValidationError, match="sorted"):              # the order of the list is not a choice either
        agg.decrypt(relabelled(["c:1", "b:1"]), both)
    # what no row reads can be renamed without effect, if it keeps its place in the sorted list (c:1 -> d:1)
    assert agg.decrypt(relabelled(["b:1", "d:1"]), only_b) == EXPECTED


def test_a_ciphertext_of_another_client_with_the_right_nid_gives_no_value(mc_fixed):
    _, clients, agg, cts, dk = mc_fixed
    forged = dict(cts["nid_1"], nid="nid_0")
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(dict(cts, nid_0=forged), dk)
    assert not isinstance(error.value, FEValidationError)


def test_a_key_with_a_swapped_ipfe_part_gives_no_value(mc_fixed):
    # the ipfe part of another client's slot: well formed, not for this client
    _, _, agg, cts, dk = mc_fixed
    dk["k_ipfe"]["nid_0"], dk["k_ipfe"]["nid_1"] = dk["k_ipfe"]["nid_1"], dk["k_ipfe"]["nid_0"]
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, dk)
    assert not isinstance(error.value, FEValidationError)


def test_a_key_with_the_weights_of_another_key_gives_no_value(mc_fixed):
    kg, _, agg, cts, dk = mc_fixed
    other = kg.get_decryption_keys("agg", credentials=creds(y=dict(Y, nid_1=5)))
    spliced = dict(dk, y=other["y"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)


def test_collusion_splice_of_rows_of_two_keys(mc_fixed):
    # the second row of another key for the same policy and weights: its share and its z are not those of this key
    kg, _, agg, cts, dk = mc_fixed
    donor = kg.get_decryption_keys("agg", credentials=creds())
    spliced = copy.deepcopy(dk)
    for nid in LST:
        spliced["k_rows"][nid][1] = donor["k_rows"][nid][1]
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)
    assert agg.decrypt(cts, donor) == EXPECTED


def test_two_tag_attack(make_mc):
    # A scalar h = SHA-256(label) mod q for the tag, with t = g1^(h s + x), would let anyone who holds two ciphertexts
    # of one client cancel the mask s: t_1^h_2 / t_2^h_1 = g1^(x_1 h_2 - x_2 h_1), a group element that depends on
    # the two plaintexts only (a guess at them is checked at once); and the point of the tag, g1^h, would have a
    # discrete log that everybody knows. The tag is hashed to G1 here: neither holds. This shows that this route is
    # closed, it is not a security proof.
    kg, clients, _ = make_mc()
    group = PairingGroup(GROUP)
    order = int(group.order())
    pp = kg.get_public_parameters()
    g1 = group.deserialize(pp["g1"].encode("utf-8"))
    (x_1, label_1), (x_2, label_2) = (5, "round-1"), (-3, "round-2")
    cts = [clients["nid_0"].encrypt([x], label, attributes=ATTRS) for x, label in ((x_1, label_1), (x_2, label_2))]
    t_1, t_2 = (group.deserialize(ct["t"].encode("utf-8")) for ct in cts)
    h_1, h_2 = (int.from_bytes(hashlib.sha256(ct["label"].encode("utf-8")).digest(), "big") % order for ct in cts)

    # the control: the formula holds for the scalar scheme it is meant for (t = g1^(h s + x))
    s = int(kg.msk["s"]["nid_0"])
    scalar_1, scalar_2 = g1 ** ((h_1 * s + x_1) % order), g1 ** ((h_2 * s + x_2) % order)
    assert scalar_1 ** h_2 / scalar_2 ** h_1 == g1 ** ((x_1 * h_2 - x_2 * h_1) % order)

    assert t_1 ** h_2 / t_2 ** h_1 != g1 ** ((x_1 * h_2 - x_2 * h_1) % order)
    for ct, h in zip(cts, (h_1, h_2)):
        omega, omega_prime = tag_points(group, SCHEME_ID, pp["pp_binding"], ct["label"], ct["attributes_digest"])
        assert omega != g1 ** h and omega_prime != g1 ** h


def test_two_tags_and_two_keys_of_different_policies_give_no_value(make_mc):
    # The attack that mcfe/kp_ab_pairing_per_client does not stop (two keys of one y, each opening neither of two
    # rounds, where the set of each client satisfies one of the keys in both rounds: they give the difference of the
    # two rounds' inner products) does not work here. All clients of a round hold one set, so a key that gives one
    # client a part opens the round by itself; the clients that the attack needs, with different sets under one tag,
    # hash other points H(tag, S) each, and the masks of the clients no longer cancel.
    # The parts are computed from the fields of the records, as in the test of that scheme, with a positive control.
    kg, clients, agg = make_mc(n=2)
    group = PairingGroup(GROUP)
    order = int(group.order())
    nids = kg.lst_nid
    pp = kg.get_public_parameters()
    g1, g2 = (group.deserialize(pp[name].encode("utf-8")) for name in ("g1", "g2"))
    gt = pair(g1, g2)
    y = {"nid_0": 2, "nid_1": 3}
    x_1, x_2 = {"nid_0": 5, "nid_1": -3}, {"nid_0": 1, "nid_1": 7}
    bound = len(nids) * 10 * 10

    def total(cts, keys):
        """``prod_i V_i``: each client's part from its own key, plus the masked plaintexts."""
        value = gt ** 0
        for nid in nids:
            dk, policy = keys[nid]
            part = key_part(group, cts[nid], dk, nid, policy)
            value = value * part * masked_plaintext(group, cts[nid], y[nid], g2)
        return value

    def inner(x):
        return sum(x[nid] * y[nid] for nid in nids)

    # the positive control: one set for both clients, one key that they satisfy, one round: the parts add up to <x, y>
    both = "or(org:p, org:q)"
    key_both = kg.get_decryption_keys("agg", credentials=creds(both, y))
    same = {nid: ["org:p", "org:q"] for nid in nids}
    cts = {nid: clients[nid].encrypt([x_1[nid]], "round-1", attributes=same[nid]) for nid in nids}
    assert agg.decrypt(cts, key_both) == inner(x_1)
    assert total(cts, {nid: (key_both, both) for nid in nids}) == gt ** (inner(x_1) % order)

    # the attack: the sets S_0 = {p} and S_1 = {q}, the keys A (p) and B (q), the rounds 1 and 2
    sets = {"nid_0": ["org:p"], "nid_1": ["org:q"]}
    key_a = kg.get_decryption_keys("agg", credentials=creds("org:p", y))
    key_b = kg.get_decryption_keys("agg", credentials=creds("org:q", y))
    keys = {"nid_0": (key_a, "org:p"), "nid_1": (key_b, "org:q")}
    rounds = [{nid: clients[nid].encrypt([x[nid]], tag, attributes=sets[nid]) for nid in nids}
              for x, tag in ((x_1, "round-1"), (x_2, "round-2"))]
    for cts in rounds:                          # the scheme refuses such ciphertexts: the clients disagree on the set
        with pytest.raises(FEValidationError, match="differ in their attributes_digest"):
            agg.decrypt(cts, key_a)
    values = [total(cts, keys) for cts in rounds]
    for value, x in zip(values, (x_1, x_2)):    # one tag: no value
        assert value != gt ** (inner(x) % order)
        with pytest.raises(ValueError, match="discrete log"):
            bounded_discrete_log_gt(group, gt, value, bound)
    quotient = values[0] / values[1]            # two tags: not the difference of the two inner products either
    assert quotient != gt ** ((inner(x_1) - inner(x_2)) % order)
    with pytest.raises(ValueError, match="discrete log"):
        bounded_discrete_log_gt(group, gt, quotient, bound)


def test_corrupted_client_cannot_switch_attributes(make_mc):
    # The honest clients encrypt for {a:1}, the corrupted client 0 for {a:1, b:1}, which satisfies the policy. The
    # clients disagree on the set. If the honest records are forged to say {a:1, b:1}, with the missing vector of
    # b:1 copied from client 0, the vectors do not fit: every client has its own psi, which cancels against its own
    # c_ipfe only, and the honest points of the tag are those of {a:1}.
    kg, clients, agg = make_mc()
    dk = kg.get_decryption_keys("agg", credentials=creds("and(a:1, b:1)"))
    honest, switched = ["a:1"], ["a:1", "b:1"]
    cts = encrypt_all(clients, attributes=honest)
    cts["nid_0"] = clients["nid_0"].encrypt([X["nid_0"]], "round-1", attributes=switched)
    with pytest.raises(FEValidationError, match="differ in their attributes_digest"):
        agg.decrypt(cts, dk)

    forged = {"nid_0": cts["nid_0"]}
    for nid in ("nid_1", "nid_2"):
        forged[nid] = dict(cts[nid], attributes=list(switched), attributes_digest=attribute_set_digest(switched),
                           c_attr=[cts[nid]["c_attr"][0], cts["nid_0"]["c_attr"][1]])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(forged, dk)
    assert not isinstance(error.value, FEValidationError)


def test_key_splice(mc_fixed):
    # two keys for one policy and different weights; client 0 takes its parts of the first, the others theirs of the
    # second, and y is the first key's: <S, y> and <U, y> in the parts of the clients are not those of one y
    kg, _, agg, cts, dk = mc_fixed
    other_y = {"nid_0": 1, "nid_1": 2, "nid_2": 3}
    other = kg.get_decryption_keys("agg", credentials=creds(y=other_y))
    assert other["y"] != dk["y"] and other["policy"] == dk["policy"]
    spliced = copy.deepcopy(other)
    spliced["y"] = copy.deepcopy(dk["y"])
    spliced["k_rows"]["nid_0"] = copy.deepcopy(dk["k_rows"]["nid_0"])
    spliced["k_ipfe"]["nid_0"] = copy.deepcopy(dk["k_ipfe"]["nid_0"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, spliced)
    assert not isinstance(error.value, FEValidationError)
    assert agg.decrypt(cts, dk) == EXPECTED
    assert agg.decrypt(cts, other) == sum(X[nid] * other_y[nid] for nid in LST)


def test_splice_of_keys_of_one_function(mc_fixed):
    # Not an attack. Two keys for the same policy and the same y differ in a0, z, the shares and the pi. The term
    # psi_i a0 z of a client cancels between its own rows and its own ipfe part, so a client may take both parts from
    # either key, and <S, y> and <U, y> are the same in both. The splice gives the one function the keys give, no more.
    kg, _, agg, cts, dk = mc_fixed
    other = kg.get_decryption_keys("agg", credentials=creds())
    assert other["k_rows"] != dk["k_rows"] and other["k_ipfe"] != dk["k_ipfe"]
    spliced = copy.deepcopy(other)
    for nid in ("nid_0", "nid_2"):
        spliced["k_rows"][nid] = copy.deepcopy(dk["k_rows"][nid])
        spliced["k_ipfe"][nid] = copy.deepcopy(dk["k_ipfe"][nid])
    assert agg.decrypt(cts, spliced) == EXPECTED
    # what must not be mixed is the rows and the ipfe part of one client: their a0 z differ
    mixed = copy.deepcopy(other)
    mixed["k_ipfe"]["nid_1"] = copy.deepcopy(dk["k_ipfe"]["nid_1"])
    with pytest.raises(ValueError, match="discrete log") as error:
        agg.decrypt(cts, mixed)
    assert not isinstance(error.value, FEValidationError)


# ---------------------------------------------------------------------------
# arrays: what the ndarray helpers refuse, before anything is paired


@pytest.fixture
def array_fixed(make_mc):
    """A key generator, the clients, an aggregator, the plaintext arrays of the clients (a 2 x 3 array and a vector of
    4 each), their ciphertexts under ``round-1`` for ``ATTRS`` and a key for ``POLICY`` and ``Y``."""
    kg, clients, agg = make_mc()
    arrays = random_arrays()
    return (kg, clients, agg, arrays, encrypt_arrays(clients, arrays),
            kg.get_decryption_keys("agg", credentials=creds()))


def test_ndarray_decryption_needs_a_key_and_a_string_label(array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match="decryption key"):
        agg.decrypt_lst_ndarray_ct(cts, label="round-1")
    for label in (None, 5, b"round-1", ["round-1"], ElementLabel("round-1", 0, (0,))):
        with pytest.raises(FEValidationError, match="label"):
            agg.decrypt_lst_ndarray_ct(cts, dk=dk, label=label)
    with pytest.raises(FEValidationError, match="label"):
        agg.decrypt_lst_ndarray_ct(cts, dk=dk)


def test_ndarray_label_must_be_the_one_of_the_ciphertexts(array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=r"layer 0, element \(0, 0\).*claim the label"):
        aggregate(agg, cts, dk, label="round-2")


def test_ndarray_every_client_is_needed(array_fixed, monkeypatch):           # Review Focus 2
    _, _, agg, _, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    for missing in LST:
        with pytest.raises(FEValidationError, match=missing):
            aggregate(agg, {nid: layers for nid, layers in cts.items() if nid != missing}, dk)
    with pytest.raises(FEValidationError, match="nid_9"):
        aggregate(agg, dict(cts, nid_9=cts["nid_0"]), dk)
    with pytest.raises(FEValidationError, match="nid_0"):
        aggregate(agg, {}, dk)


@pytest.mark.parametrize("bad", [None, [], "cts", 5, [("nid_0", [])]])
def test_ndarray_refuses_what_is_no_dict_of_arrays(bad, array_fixed):
    _, _, agg, _, _, dk = array_fixed
    with pytest.raises(FEValidationError):
        agg.decrypt_lst_ndarray_ct(bad, dk=dk, label="round-1")


@pytest.mark.parametrize("change,match", [
    (lambda layers: layers[:1], "nid_1.*1 layers.*nid_0.*2"),
    (lambda layers: layers + layers, "nid_1.*4 layers.*nid_0.*2"),
    (lambda layers: [layers[0][:1], layers[1]], r"nid_1.*layer 0.*shape \(1, 3\).*\(2, 3\)"),
    (lambda layers: [layers[0], layers[1][:3]], r"nid_1.*layer 1.*shape \(3,\).*\(4,\)"),
    (lambda layers: [layers[0], layers[1].tolist()], r"nid_1.*layer 1.*ndarray"),
    (lambda layers: [layers[0], None], r"nid_1.*layer 1.*ndarray"),
    (lambda layers: "layers", "nid_1.*list of arrays"),
    (lambda layers: None, "nid_1.*list of arrays"),
])
def test_ndarray_clients_send_the_same_layers(change, match, array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    cts["nid_1"] = change(cts["nid_1"])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=match):
        aggregate(agg, cts, dk)


@pytest.mark.parametrize("owners,named", [(["nid_0"], "nid_0"), (["nid_2"], "nid_2"), (LST, "nid_0")],
                         ids=["first client", "last client", "all clients"])
def test_ndarray_refuses_a_layer_that_is_a_scalar(owners, named, array_fixed, monkeypatch):
    # a 0-d array has no shape to go through, and decoded_layer cannot divide it as an array (when all clients
    # sent one, it was a bare AttributeError there)
    _, clients, agg, _, cts, dk = array_fixed
    for owner in owners:            # a scalar layer as a client would have sent it if encrypt had accepted one
        scalar = np.empty((), dtype=object)
        scalar[()] = clients[owner].encrypt([1], ElementLabel("round-1", 1, ()), attributes=ATTRS)
        cts[owner] = [cts[owner][0], scalar]
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=r"{}: layer 1 is a scalar".format(named)):
        aggregate(agg, cts, dk)


@pytest.mark.parametrize("field,value,reason", [
    ("pp_binding", "0" * 64, "pp_binding"), ("policy_compiler", "v9", "policy_compiler"),
    ("attr_hash", "v9", "attr_hash"), ("tag_hash", "v9", "tag_hash"),
    ("t", 5, "t of nid_1"), ("t", ["1:x"], "t of nid_1"), ("c_attr", [], "c_attr of nid_1"),
    ("c_attr", [["1:x"] * 7, ["1:x"] * 8], "c_attr of nid_1"), ("c_ipfe", ["1:x"] * 3, "c_ipfe of nid_1"),
    ("nid", "nid_2", "nid_1"), ("nid", None, "client id"), ("nid", np.array([0, 1]), "client id"),
    ("label", 5, "label of the ciphertext of nid_1"), ("attributes", ["region:EU", "org:A"], "sorted"),
    ("attributes", 5, "attributes"), ("attributes_digest", "00" * 32, "attributes_digest"),
    ("attributes_digest", np.array([0, 1]), "attributes_digest"),
])
def test_ndarray_refuses_a_hostile_element_and_names_it(field, value, reason, array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    cts["nid_1"][1][2] = dict(cts["nid_1"][1][2], **{field: value})
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError) as error:
        aggregate(agg, cts, dk)
    assert "layer 1, element (2,)" in str(error.value) and re.search(reason, str(error.value)), str(error.value)


def test_ndarray_refuses_an_element_without_a_field_or_that_is_no_ciphertext(array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    element = cts["nid_2"][0][1, 1]
    cts["nid_2"][0][1, 1] = {name: value for name, value in element.items() if name != "c_ipfe"}
    with pytest.raises(FEValidationError, match=r"nid_2, layer 0, element \(1, 1\).*c_ipfe"):
        aggregate(agg, cts, dk)
    for bad in (None, "ct", [element], 5):
        cts["nid_2"][0][1, 1] = bad
        with pytest.raises(FEValidationError, match=r"nid_2, layer 0, element \(1, 1\)"):
            aggregate(agg, cts, dk)


def test_ndarray_ciphertext_in_the_slot_of_another_client_is_refused(array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    swapped = dict(cts, nid_0=cts["nid_1"], nid_1=cts["nid_0"])
    with pytest.raises(FEValidationError, match=r"layer 0, element \(0, 0\).*nid_0.*nid_1"):
        aggregate(agg, swapped, dk)


def test_ndarray_clients_agree_on_the_digest_of_each_element(make_mc, monkeypatch):
    kg, clients, agg = make_mc()
    arrays = random_arrays()
    cts = encrypt_arrays(clients, arrays)
    dk = kg.get_decryption_keys("agg", credentials=creds())
    cts["nid_2"][1][3] = clients["nid_2"].encrypt([int(arrays["nid_2"][1][3])], ElementLabel("round-1", 1, (3,)),
                                                  attributes=ATTRS + ["org:A"])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=r"layer 1, element \(3,\).*differ in their attributes_digest"):
        aggregate(agg, cts, dk)


def test_ndarray_unsatisfied_is_refused_before_any_pairing(make_mc, monkeypatch):
    kg, clients, agg = make_mc()
    cts = encrypt_arrays(clients, random_arrays(), attributes=["region:EU"])
    dk = kg.get_decryption_keys("agg", credentials=creds())
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match="do not satisfy"):
        aggregate(agg, cts, dk)


def test_ndarray_fusion_weight_must_be_the_weights_of_the_key(array_fixed, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="the fusion weight of nid_1 is not the y of the functional key"):
        aggregate(agg, cts, dk, y=dict(Y, nid_1=Y["nid_1"] + 1))
    with pytest.raises(FEValidationError, match="the fusion weight"):
        agg.decrypt_lst_ndarray_ct(cts, dk=dk, fusion_weight={"nid_0": [1]}, label="round-1")


def test_ndarray_refuses_a_key_that_is_not_these_parameters_or_not_well_formed(array_fixed, make_mc, monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    other_kg, _, _ = make_mc()
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="functional key.*pp_binding"):
        aggregate(agg, cts, other_kg.get_decryption_keys("agg", credentials=creds()))
    with pytest.raises(FEValidationError, match="policy_digest"):
        aggregate(agg, cts, dict(dk, policy_digest="0" * 64))
    with pytest.raises(FEValidationError, match="k_rows of nid_1"):
        aggregate(agg, cts, dict(dk, k_rows=dict(dk["k_rows"], nid_1=dk["k_rows"]["nid_1"][:1])))
    with pytest.raises(FEValidationError, match="y of nid_2"):
        aggregate(agg, cts, dict(dk, y=dict(dk["y"], nid_2="11")))


def test_ndarray_decodes_every_vector_of_every_element_also_the_unused(make_mc):
    kg, clients, agg = make_mc()
    arrays = random_arrays()
    cts = encrypt_arrays(clients, arrays, attributes=ATTRS + ["org:A"])        # org:A is no row of the policy
    dk = kg.get_decryption_keys("agg", credentials=creds())
    assert all((layer == expected).all() for layer, expected in zip(aggregate(agg, cts, dk), combined(arrays)))
    ct = cts["nid_1"][1][3]
    cts["nid_1"][1][3] = dict(ct, c_attr=[row[:] for row in ct["c_attr"]])
    cts["nid_1"][1][3]["c_attr"][0][5] = "1:not an element"                      # in the vector of org:A, the first
    with pytest.raises(FEValidationError, match=r"c_attr of nid_1, layer 1, element \(3,\)"):
        aggregate(agg, cts, dk)
    cts["nid_1"][1][3]["c_attr"][0][5] = dk["k_ipfe"]["nid_0"][0]                # an element of G2
    with pytest.raises(FEValidationError, match=r"c_attr of nid_1, layer 1, element \(3,\)"):
        aggregate(agg, cts, dk)
    cts["nid_1"][1][3] = dict(ct, t=dk["k_ipfe"]["nid_0"][0])                    # G2 where G1 is due
    with pytest.raises(FEValidationError, match=r"t of nid_1, layer 1, element \(3,\)"):
        aggregate(agg, cts, dk)
    cts["nid_1"][1][3] = dict(ct, c_ipfe=ct["c_ipfe"][:3] + ["1:x"])
    with pytest.raises(FEValidationError, match=r"c_ipfe of nid_1, layer 1, element \(3,\)"):
        aggregate(agg, cts, dk)


def test_ndarray_decodes_the_used_rows_of_the_key_only(make_mc):
    kg, clients, agg = make_mc()
    arrays = random_arrays()
    cts = encrypt_arrays(clients, arrays, attributes=["region:EU"])
    dk = kg.get_decryption_keys("agg", credentials=creds("or(region:EU, org:A)"))      # the first leaf is used
    good = copy.deepcopy(dk)
    dk["k_rows"]["nid_1"][1][2] = JUNK_G2
    assert all((layer == expected).all() for layer, expected in zip(aggregate(agg, cts, dk), combined(arrays)))
    dk["k_rows"]["nid_1"][0][2] = JUNK_G2
    with pytest.raises(FEValidationError, match="k_rows of nid_1"):
        aggregate(agg, cts, dk)
    dk = copy.deepcopy(good)
    dk["k_rows"]["nid_1"][0][2] = cts["nid_0"][0][0, 0]["t"]                            # G1 where G2 is due
    with pytest.raises(FEValidationError, match="k_rows of nid_1"):
        aggregate(agg, cts, dk)
    dk = copy.deepcopy(good)
    dk["k_ipfe"]["nid_0"][1] = cts["nid_0"][0][0, 0]["c_ipfe"][0]
    with pytest.raises(FEValidationError, match="k_ipfe of nid_0"):
        aggregate(agg, cts, dk)


def test_ndarray_with_a_key_of_another_function_gives_no_value(array_fixed):
    kg, _, agg, _, cts, dk = array_fixed
    spliced = dict(dk, y=kg.get_decryption_keys("agg", credentials=creds(y=dict(Y, nid_1=5)))["y"])
    with pytest.raises(FESchemeError, match=r"element \(0, 0\) of layer 0 does not decrypt") as error:
        aggregate(agg, cts, spliced, y={nid: int(spliced["y"][nid]) for nid in LST})
    assert not isinstance(error.value, ValueError)


# ---------------------------------------------------------------------------
# check_ndarray_ct: the checks of a decryption that need neither a key nor a decoded point


def hostile_element(cts, field, value):
    """The ciphertexts *cts* with the element ``(2,)`` of layer 1 of ``nid_1`` changed in one field."""
    cts["nid_1"][1][2] = dict(cts["nid_1"][1][2], **{field: value})


CHECKED_WITHOUT_A_KEY = [
    ("pp_binding", "0" * 64, "pp_binding"), ("policy_compiler", "v9", "policy_compiler"),
    ("attr_hash", "v9", "attr_hash"), ("tag_hash", "v9", "tag_hash"),
    ("t", 5, "t of nid_1"), ("c_attr", [], "c_attr of nid_1"),
    ("c_attr", [["1:x"] * 7, ["1:x"] * 8], "c_attr of nid_1"),
    ("c_ipfe", ["1:x"] * 3, "c_ipfe of nid_1"),
    ("nid", "nid_2", "nid_1"), ("nid", None, "client id"),
    ("label", 5, "label of the ciphertext of nid_1"), ("label", "other", "differ in their label"),
    ("attributes", ["region:EU", "org:A"], "sorted"),
    ("attributes", ["region:EU"], "is not the digest of its attributes"),     # a list the digest does not stand for
    ("attributes_digest", "00" * 32, "attributes_digest"),
]


def test_check_ndarray_ct_accepts_the_ciphertexts_of_a_round_without_a_key(array_fixed, monkeypatch):
    _, _, agg, _, cts, _ = array_fixed
    before = copy.deepcopy(cts)
    refuse_to_compute(monkeypatch)
    assert agg.check_ndarray_ct(cts, label="round-1") is None
    assert all((np.array_equal(cts[nid][layer], before[nid][layer])) for nid in LST for layer in range(2))


@pytest.mark.parametrize("field,value,reason", CHECKED_WITHOUT_A_KEY)
def test_check_ndarray_ct_refuses_what_a_decryption_refuses_and_says_the_same(field, value, reason, array_fixed,
                                                                              monkeypatch):
    _, _, agg, _, cts, dk = array_fixed
    hostile_element(cts, field, value)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError) as checked:
        agg.check_ndarray_ct(cts, label="round-1")
    assert "layer 1, element (2,)" in str(checked.value) and re.search(reason, str(checked.value)), str(checked.value)
    with pytest.raises(FEValidationError) as decrypted:
        aggregate(agg, cts, dk)
    assert str(decrypted.value) == str(checked.value)


@pytest.mark.parametrize("change,match", [
    (lambda layers: layers[:1], "nid_1.*1 layers.*nid_0.*2"),
    (lambda layers: [layers[0][:1], layers[1]], r"nid_1.*layer 0.*shape \(1, 3\).*\(2, 3\)"),
    (lambda layers: [layers[0], layers[1].tolist()], r"nid_1.*layer 1.*ndarray"),
    (lambda layers: None, "nid_1.*list of arrays"),
])
def test_check_ndarray_ct_wants_every_client_to_send_the_same_layers(change, match, array_fixed, monkeypatch):
    _, _, agg, _, cts, _ = array_fixed
    cts["nid_1"] = change(cts["nid_1"])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=match):
        agg.check_ndarray_ct(cts, label="round-1")
    for missing in LST:
        with pytest.raises(FEValidationError, match=missing):
            agg.check_ndarray_ct({nid: layers for nid, layers in cts.items() if nid != missing}, label="round-1")


def test_check_ndarray_ct_checks_the_label_the_elements_claim(array_fixed, monkeypatch):
    _, _, agg, _, cts, _ = array_fixed
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=r"layer 0, element \(0, 0\).*claim the label"):
        agg.check_ndarray_ct(cts, label="round-2")
    for label in (None, 5, b"round-1", ElementLabel("round-1", 0, (0,))):
        with pytest.raises(FEValidationError, match="label"):
            agg.check_ndarray_ct(cts, label=label)
    with pytest.raises(FEValidationError, match="label"):
        agg.check_ndarray_ct(cts)


def test_check_ndarray_ct_checks_that_the_elements_are_for_one_set_of_attributes(make_mc, monkeypatch):
    _, clients, agg = make_mc()
    arrays = random_arrays()
    cts = encrypt_arrays(clients, arrays)
    cts["nid_2"][1][3] = clients["nid_2"].encrypt([int(arrays["nid_2"][1][3])], ElementLabel("round-1", 1, (3,)),
                                                  attributes=ATTRS + ["org:A"])
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match=r"layer 1, element \(3,\).*differ in their attributes_digest"):
        agg.check_ndarray_ct(cts, label="round-1")
    cts = encrypt_arrays(clients, arrays, label="round-2")
    cts["nid_1"][0][1, 1] = clients["nid_1"].encrypt([int(arrays["nid_1"][0][1, 1])],
                                                     ElementLabel("round-2", 0, (1, 1)), attributes=["region:EU"])
    with pytest.raises(FEValidationError, match=r"layer 0, element \(1, 1\).*differ in their attributes_digest"):
        agg.check_ndarray_ct(cts, label="round-2")      # an element that is not the first


def test_check_ndarray_ct_sees_the_type_of_a_group_element_but_decodes_none(array_fixed):
    _, _, agg, _, cts, dk = array_fixed
    wrong_group = dk["k_ipfe"]["nid_0"][0]                  # an element of G2 where G1 is due: the type tag tells
    cts["nid_1"][1][2] = dict(cts["nid_1"][1][2], t=wrong_group)
    with pytest.raises(FEValidationError, match=r"t of nid_1, layer 1, element \(2,\)"):
        agg.check_ndarray_ct(cts, label="round-1")
    # an item with the right tag that is no element is found when it is decoded, which the check does not do
    cts["nid_1"][1][2] = dict(cts["nid_1"][1][2], t="1:not an element")
    assert agg.check_ndarray_ct(cts, label="round-1") is None
    with pytest.raises(FEValidationError, match=r"t of nid_1, layer 1, element \(2,\)"):
        aggregate(agg, cts, dk)


def test_check_ndarray_ct_has_nothing_to_check_in_arrays_without_elements(make_mc):
    _, clients, agg = make_mc()
    cts = encrypt_arrays(clients, {nid: [np.zeros((0, 3), dtype=np.int64)] for nid in LST})
    assert agg.check_ndarray_ct(cts, label="round-1") is None


def test_encrypt_lst_ndarray_needs_the_encryption_key(array_fixed):
    _, _, agg, _, _, _ = array_fixed
    with pytest.raises(FEKeyError, match="encryption key"):
        agg.encrypt_lst_ndarray([np.zeros((2,), dtype=np.int64)], label="round-1", attributes=ATTRS)
    with pytest.raises(FEKeyError, match="encryption key"):
        agg.encrypt_lst_ndarray([], label="round-1", attributes=ATTRS)


@pytest.mark.parametrize("label", [None, 5, b"round-1", ["round-1"], ElementLabel("round-1", 0, (0,))])
def test_encrypt_lst_ndarray_needs_a_string_label(label, array_fixed):
    _, clients, _, _, _, _ = array_fixed
    with pytest.raises(FEValidationError, match="label"):
        clients["nid_0"].encrypt_lst_ndarray([np.ones((2,), dtype=np.int64)], label=label, attributes=ATTRS)
    with pytest.raises(FEValidationError, match="label"):
        clients["nid_0"].encrypt_lst_ndarray([], label=label, attributes=ATTRS)


@pytest.mark.parametrize("attributes", [None, "region:EU", 5, ["region"], [5], ["region:EU", None]])
def test_encrypt_lst_ndarray_checks_the_attributes_also_for_an_empty_array(attributes, array_fixed):
    _, clients, _, _, _, _ = array_fixed
    for layers in ([np.ones((2,), dtype=np.int64)], [np.zeros((0,), dtype=np.int64)], []):
        with pytest.raises(FEValidationError):
            clients["nid_0"].encrypt_lst_ndarray(layers, label="round-1", attributes=attributes)


@pytest.mark.parametrize("layer", [np.array(3), 3, np.int64(2), 1.5, np.float64(0.5)])
def test_encrypt_lst_ndarray_refuses_a_layer_that_is_a_scalar(layer, array_fixed):
    # it used to be accepted, and then the aggregation failed with a bare AttributeError in decoded_layer
    _, clients, _, _, _, _ = array_fixed
    with pytest.raises(FEValidationError, match="layer 1 of the arrays to encrypt is a scalar"):
        clients["nid_0"].encrypt_lst_ndarray([np.ones((2,), dtype=np.int64), layer], label="round-1",
                                             attributes=ATTRS)


@pytest.mark.parametrize("layer", [np.array([11]), np.array([-11]), np.array([10.6]), np.array([np.nan]),
                                   np.array(["a"]), np.array([None])])
def test_encrypt_lst_ndarray_checks_the_plaintext(layer, array_fixed):
    _, clients, _, _, _, _ = array_fixed
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt_lst_ndarray([np.ones((1,), dtype=np.int64), layer], label="round-1",
                                             attributes=ATTRS)


