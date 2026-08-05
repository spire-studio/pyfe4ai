import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.quadratic.quad import QuadraticQuad
from pyfe4ai.schemes.quadratic.quad import QuadraticQuadKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "n": 4,
        "m": 3,
        "bound": 5,
        "pairing_group_param": "SS512",
    }


def test_key_generator(kg_config):
    kg = QuadraticQuadKeyGenerator(kg_config)
    param_path = os.path.join(
        "config", "authority", CryptoCONST.TYPE_QUADRATIC_QUAD, "param.json"
    )
    assert os.path.exists(param_path)

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["n"] == kg_config["n"]
    assert pp["m"] == kg_config["m"]
    assert pp["inner"]["subspace_dim"] == kg_config["n"] * kg_config["m"]


def test_entire_process(kg_config):
    kg = QuadraticQuadKeyGenerator(kg_config)
    kg.setup()

    n = kg_config["n"]
    m = kg_config["m"]
    bound = kg_config["bound"]
    x = [random.randint(-bound, bound) for _ in range(n)]
    y = [random.randint(-bound, bound) for _ in range(m)]
    f_matrix = [
        [random.randint(-bound, bound) for _ in range(m)]
        for _ in range(n)
    ]
    expected = sum(
        x[i] * f_matrix[i][j] * y[j]
        for i in range(n)
        for j in range(m)
    )

    pp = kg.get_public_parameters()
    encryptor = QuadraticQuad({"id": "nid_default", "keys": {"pp": pp}})
    decryptor = QuadraticQuad({"id": "sid_0", "keys": {"pp": pp}})

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
        "m": 3,
        "bound": 500,
        "pairing_group_param": "SS512",
    }
    kg = QuadraticQuadKeyGenerator(cfg)
    kg.setup()

    x_lst = [
        np.array([[1.00, -0.50], [0.75, 1.25]]),
        np.array([[0.30, 0.90], [-1.10, 0.40]]),
    ]
    y_lst = [
        np.array([[0.60, -1.00], [0.25, 0.80]]),
        np.array([[1.20, 0.40], [-0.50, 0.30]]),
        np.array([[0.10, 0.70], [1.50, -0.20]]),
    ]
    f_matrix = [[2, -1, 1], [0, 3, -2]]

    pp = kg.get_public_parameters()
    encryptor = QuadraticQuad({"id": "nid_default", "keys": {"pp": pp}, "precision": 2})
    decryptor = QuadraticQuad({"id": "sid_0", "keys": {"pp": pp}, "precision": 2})
    dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": f_matrix})

    ct = encryptor.encrypt_lst_ndarray(x_lst, rhs_lst_ndarray=y_lst)
    computed = decryptor.compute_lst_ndarray_ct(ct, dk=dk)[0]
    expected = sum(
        f_matrix[i][j] * x_lst[i] * y_lst[j]
        for i in range(2)
        for j in range(3)
    )
    assert np.allclose(computed, expected, atol=1e-2)
