from pyfe4ai.schemes.quadratic.sgp import QuadraticSGP
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGPKeyGenerator


def main() -> None:
    config = {
        "sec_param": 64,
        "n": 3,
        "bound": 5,
        "pairing_group_param": "SS512",
    }

    kg = QuadraticSGPKeyGenerator(config)
    kg.setup()

    x = [1, -2, 3]
    y = [2, 0, -1]
    f_matrix = [
        [1, 0, 2],
        [0, -1, 1],
        [2, 1, 0],
    ]

    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()

    encryptor = QuadraticSGP({"id": "nid_default", "keys": {"pp": pp, "sk": sk}})
    decryptor = QuadraticSGP({"id": "sid_0", "keys": {"pp": pp}})

    ct = encryptor.encrypt({"x": x, "y": y})
    dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": f_matrix})
    result = decryptor.decrypt(ct, dk)

    expected = sum(
        x[i] * f_matrix[i][j] * y[j]
        for i in range(len(x))
        for j in range(len(y))
    )
    print(f"x^T F y = {result}")
    print(f"expected = {expected}")


if __name__ == "__main__":
    main()
