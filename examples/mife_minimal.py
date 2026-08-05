"""Minimal end-to-end example for the MIFE scheme."""

from pyfe4ai.schemes.mife.ddh import MIFE
from pyfe4ai.schemes.mife.ddh import MIFEKeyGenerator


def main() -> None:
    x = {
        "nid_0": [2],
        "nid_1": [3],
    }
    y = {
        "nid_0": [4],
        "nid_1": [5],
    }

    kg = MIFEKeyGenerator(
        {
            "sec_param": 128,
            "eta": 1,
            "n": len(x),
            "s": 1,
            "lst_nid": list(x.keys()),
        }
    )
    kg.setup()

    pp = kg.get_public_parameters()
    ciphertexts = {}
    for nid, vec in x.items():
        encryptor = MIFE(
            {
                "id": nid,
                "precision": 3,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        ciphertexts[nid] = encryptor.encrypt(vec)

    decryptor = MIFE(
        {
            "id": "sid_0",
            "precision": 3,
            "keys": {"pp": pp},
        }
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    result = decryptor.decrypt(ciphertexts, dk, y)

    print("x =", x)
    print("y =", y)
    print("sum_i <x_i, y_i> =", result)


if __name__ == "__main__":
    main()
