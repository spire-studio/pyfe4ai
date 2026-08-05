"""Minimal end-to-end example for the SIFE/LWE scheme."""

from pyfe4ai.schemes.sife.lwe import SIFELWE
from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator


def main() -> None:
    x = [2, -1, 3]
    y = [4, 5, -2]

    kg = SIFELWEKeyGenerator(
        {
            "sec_param": 64,
            "eta": len(x),
            "lwe_n": 16,
            "bound_x": 8,
            "bound_y": 8,
        }
    )
    kg.setup()

    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

    encryptor = SIFELWE(
        {
            "id": "nid_default",
            "keys": {"pp": pp, "sk": sk},
        }
    )
    decryptor = SIFELWE(
        {
            "id": "sid_0",
            "keys": {"pp": pp},
        }
    )

    ct = encryptor.encrypt(x)
    result = decryptor.decrypt(ct, dk, y)

    print("x =", x)
    print("y =", y)
    print("<x, y> =", result)


if __name__ == "__main__":
    main()
