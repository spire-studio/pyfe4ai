"""Minimal end-to-end example for the SIFE scheme."""

from pyfe4ai.schemes.sife.ddh import SIFE
from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator


def main() -> None:
    x = [2, 1, 3]
    y = [4, 5, 6]

    kg = SIFEKeyGenerator({"sec_param": 128, "eta": len(x)})
    kg.setup()

    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

    encryptor = SIFE(
        {
            "id": "nid_default",
            "precision": 3,
            "keys": {"pp": pp, "sk": sk},
        }
    )
    decryptor = SIFE(
        {
            "id": "sid_0",
            "precision": 3,
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
