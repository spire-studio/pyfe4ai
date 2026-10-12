import random

import numpy as np
import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.quadratic.quad import QuadraticQuad
from pyfe4ai.schemes.quadratic.quad import QuadraticQuadKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "n": 4,
        "m": 3,
        "bound": 5,
        "pairing_group_param": "MNT224",
    }


def test_key_generator(kg_config):
    """Gay20 Fig. 4: [M]_1 is (2m + 3n) x (m + 2 + 2n) and only in G1."""
    kg = QuadraticQuadKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    n, m = kg_config["n"], kg_config["m"]
    assert pp["n"] == n and pp["m"] == m
    assert len(pp["ipfe"]["m"]) == 2 * m + 3 * n
    assert all(len(row) == m + 2 + 2 * n for row in pp["ipfe"]["m"])
    assert all(isinstance(v, str) and v.startswith("1:") for row in pp["ipfe"]["m"] for v in row)
    assert kg.get_private_keys() is None


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
    # succinct ciphertext: 4n + 2m + 4 elements of G1 and m of G2
    assert len(ct["ct_x"]) + len(ct["ct"]) == 4 * n + 2 * m + 4
    assert len(ct["ct_y"]) == m and all(v.startswith("2:") for v in ct["ct_y"])


def test_extreme_values(kg_config):
    kg = QuadraticQuadKeyGenerator(dict(kg_config, n=2, m=2, bound=3))
    kg.setup()
    pp = kg.get_public_parameters()
    enc = QuadraticQuad({"keys": {"pp": pp}})
    f = [[3, -3], [-3, 3]]
    dk = kg.get_decryption_keys(None, credentials={"function_matrix": f})
    ct = enc.encrypt({"x": [3, -3], "y": [3, -3]})
    assert QuadraticQuad({"keys": {"pp": pp}}).decrypt(ct, dk) == 4 * 27


def test_entire_process_lst_ndarray():
    cfg = {
        "sec_param": 64,
        "n": 2,
        "m": 3,
        "bound": 500,
        "pairing_group_param": "MNT224",
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
