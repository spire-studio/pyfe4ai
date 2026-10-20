"""User labels and ndarray element labels must never collide.

MCFE ciphertexts are deterministic per (slot, label); two ciphertexts of one
slot under one label reveal the difference of the plaintexts. Element labels
used to be plain JSON, so encrypt(x, '["r1", 0, [3]]') collided with element
(0, (3,)) of encrypt_lst_ndarray(..., label="r1").
"""

import json

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.labels import ElementLabel, element_label, encode_label
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


def _schemes():
    from pyfe4ai.schemes.mcfe.ddh import MCFE, MCFEKeyGenerator
    from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgard, MCFEDamgardKeyGenerator
    from pyfe4ai.schemes.mcfe.lwe import MCFELWE, MCFELWEKeyGenerator
    from pyfe4ai.schemes.mcfe.paillier import MCFEPaillier, MCFEPaillierKeyGenerator

    base = {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"]}
    return [
        (MCFEKeyGenerator, MCFE, dict(base, sec_param=128)),
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
