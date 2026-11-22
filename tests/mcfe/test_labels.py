"""User labels and ndarray element labels must never collide.

MCFE ciphertexts are deterministic per (slot, label); two ciphertexts of one
slot under one label reveal the difference of the plaintexts. Element labels
used to be plain JSON, so encrypt(x, '["r1", 0, [3]]') collided with element
(0, (3,)) of encrypt_lst_ndarray(..., label="r1").
"""

import json

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.labels import ElementLabel, chunk_label, element_label, encode_label
from pyfe4ai.core.exceptions import FEValidationError


def test_encodings_are_disjoint():
    forged = json.dumps(["r1", 0, [3]])
    assert encode_label(forged) != encode_label(element_label("r1", 0, (3,)))
    assert encode_label(json.dumps(["element", "r1", 0, [3]])) != encode_label(
        element_label("r1", 0, (3,))
    )
    with pytest.raises(FEValidationError):
        encode_label(("r1", 0, (3,)))  # a plain tuple is not an ElementLabel
    assert isinstance(element_label("r1", 0, [3]), ElementLabel)


#: The two UTF-16 halves of U+1F600 as two lone surrogates, which is a different string from the character.
SURROGATE_PAIR = chr(0xD83D) + chr(0xDE00)


@pytest.mark.parametrize("label", [SURROGATE_PAIR, chr(0xD83D), chr(0xDE00), "a" + chr(0xDC80) + "b",
                                   chr(0xDFFF) * 3])
def test_a_string_that_is_not_text_is_refused(label):
    # json.dumps writes a character outside the BMP as a pair of \u escapes, and a pair of lone surrogates as the
    # same two escapes: without this check the character and the two lone surrogates gave one encoding, one tag
    assert SURROGATE_PAIR != "\U0001F600" and len(SURROGATE_PAIR) == 2
    assert json.dumps(["label", SURROGATE_PAIR]) == json.dumps(["label", "\U0001F600"])      # why it is refused
    assert encode_label("\U0001F600") == json.dumps(["label", "\U0001F600"])         # the character itself is fine
    with pytest.raises(FEValidationError, match="surrogate"):
        encode_label(label)
    with pytest.raises(FEValidationError, match="surrogate"):
        encode_label(ElementLabel(label, 0, (1,)))
    with pytest.raises(FEValidationError, match="surrogate"):
        encode_label(element_label(label, 0, (1,)))
    with pytest.raises(FEValidationError, match="surrogate"):
        encode_label(chunk_label(label, 2))


def test_no_two_labels_give_one_encoding():
    labels = ["\U0001F600", "\N{LATIN SMALL LETTER E WITH ACUTE}", "e\N{COMBINING ACUTE ACCENT}", "\x00", "a",
              "a\x00", "\N{ZERO WIDTH SPACE}",
              "\U0010ffff", chr(0xD7FF), chr(0xE000)]      # the last two flank the surrogates
    encoded = [encode_label(label) for label in labels]
    assert len(set(encoded)) == len(labels)
    element = [encode_label(ElementLabel(label, 0, (0,))) for label in labels]
    assert len(set(element)) == len(labels) and not set(element) & set(encoded)


@pytest.mark.parametrize("label", [
    ElementLabel("r1", "x", (0,)), ElementLabel("r1", None, (0,)), ElementLabel("r1", 0, (None,)),
    ElementLabel("r1", 0, 5), ElementLabel("r1", 0, ("a",)), ElementLabel("r1", float("nan"), (0,)),
    ElementLabel("r1", float("inf"), (0,)), ElementLabel("r1", 0, (float("nan"),)),
])
def test_a_malformed_element_label_is_refused_with_a_validation_error(label):
    # not the ValueError or TypeError of int(): a decryption reserves ValueError for "no discrete log"
    with pytest.raises(FEValidationError, match="layer and the index"):
        encode_label(label)


def test_a_well_formed_element_label_still_encodes():
    assert encode_label(ElementLabel("r1", 2, (3, 4))) == json.dumps(["element", "r1", 2, [3, 4]])
    assert encode_label(ElementLabel("r1", np.int64(2), [np.int32(3)])) == json.dumps(["element", "r1", 2, [3]])
    assert encode_label(ElementLabel("r1", -1, ())) == json.dumps(["element", "r1", -1, []])


def _schemes():
    from pyfe4ai.schemes.mcfe.ddh import MCFEDDH, MCFEDDHKeyGenerator
    from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgard, MCFEDamgardKeyGenerator
    from pyfe4ai.schemes.mcfe.lwe import MCFELWE, MCFELWEKeyGenerator
    from pyfe4ai.schemes.mcfe.paillier import MCFEPaillier, MCFEPaillierKeyGenerator

    base = {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"]}
    return [
        (MCFEDDHKeyGenerator, MCFEDDH, dict(base, sec_param=128)),
        (MCFEDamgardKeyGenerator, MCFEDamgard, dict(base, modulus_length=64, bound=10)),
        (MCFELWEKeyGenerator, MCFELWE, dict(base, lwe_n=8, bound_x=4, bound_y=4)),
        (MCFEPaillierKeyGenerator, MCFEPaillier, dict(base, bit_length=64, bound_x=10, bound_y=10)),
    ]


@pytest.mark.parametrize("kg_cls, crypto_cls, cfg", _schemes())
def test_user_label_cannot_reproduce_an_element_ciphertext(kg_cls, crypto_cls, cfg):
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    client = crypto_cls({"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    element = client.encrypt_lst_ndarray([np.zeros(5)], label="r1")[0][(3,)]
    scalar = client.encrypt([0], json.dumps(["r1", 0, [3]]))
    assert scalar["c"] != element["c"]
