import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.quadratic.sgp import QuadraticSGP
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGPKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "n": 5,
        "bound": 6,
        "pairing_group_param": "SS512",
    }


def test_key_generator(kg_config):
    kg = QuadraticSGPKeyGenerator(kg_config)
    param_path = os.path.join(
        "config", "authority", CryptoCONST.TYPE_QUADRATIC_SGP, "param.json"
    )
    assert os.path.exists(param_path)

    kg.setup()
    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()
    assert pp["n"] == kg_config["n"]
    assert sk is not None
    assert len(sk["s"]) == kg_config["n"]
    assert len(sk["t"]) == kg_config["n"]


def test_entire_process(kg_config):
    kg = QuadraticSGPKeyGenerator(kg_config)
    kg.setup()

    n = kg_config["n"]
    bound = kg_config["bound"]
    x = [random.randint(-bound, bound) for _ in range(n)]
    y = [random.randint(-bound, bound) for _ in range(n)]
    f_matrix = [
        [random.randint(-bound, bound) for _ in range(n)]
        for _ in range(n)
    ]
    expected = sum(
        x[i] * f_matrix[i][j] * y[j]
        for i in range(n)
        for j in range(n)
    )

    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()
    encryptor = QuadraticSGP({"id": "nid_default", "keys": {"pp": pp, "sk": sk}})
    decryptor = QuadraticSGP({"id": "sid_0", "keys": {"pp": pp}})

    ct = encryptor.encrypt({"x": x, "y": y})
    dk = kg.get_decryption_keys(
        "sid_0", credentials={"function_matrix": f_matrix}
    )
    computed = decryptor.decrypt(ct, dk)
    assert computed == expected


def test_entire_process_lst_ndarray():
    cfg = {
        "sec_param": 64,
        "n": 2,
        "bound": 500,
        "pairing_group_param": "SS512",
    }
    kg = QuadraticSGPKeyGenerator(cfg)
    kg.setup()

    x_lst = [
        np.array([[1.20, -0.50], [0.75, 2.10]]),
        np.array([[0.30, 1.10], [-1.25, 0.40]]),
    ]
    y_lst = [
        np.array([[1.00, 0.50], [0.25, -0.75]]),
        np.array([[0.40, -1.20], [0.60, 0.80]]),
    ]
    f_matrix = [[2, -1], [3, 1]]

    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()
    encryptor = QuadraticSGP(
        {"id": "nid_default", "keys": {"pp": pp, "sk": sk}, "precision": 2}
    )
    decryptor = QuadraticSGP({"id": "sid_0", "keys": {"pp": pp}, "precision": 2})
    dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": f_matrix})

    ct = encryptor.encrypt_lst_ndarray(x_lst, rhs_lst_ndarray=y_lst)
    computed = decryptor.compute_lst_ndarray_ct(ct, dk=dk)[0]
    expected = sum(
        f_matrix[i][j] * x_lst[i] * y_lst[j]
        for i in range(2)
        for j in range(2)
    )
    assert np.allclose(computed, expected, atol=1e-2)
