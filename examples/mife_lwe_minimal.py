"""Minimal end-to-end example for the MIFE/LWE scheme."""

from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator


def main() -> None:
    x = {
        "nid_0": [2],
        "nid_1": [-1],
        "nid_2": [3],
    }
    y = {
        "nid_0": [4],
        "nid_1": [2],
        "nid_2": [-1],
    }

    kg = MIFELWEKeyGenerator(
        {
            "sec_param": 64,
            "eta": 1,
            "n": len(x),
            "s": 1,
            "lst_nid": list(x.keys()),
            "lwe_n": 16,
            "bound_x": 8,
            "bound_y": 8,
        }
    )
    kg.setup()

    pp = kg.get_public_parameters()
    ciphertexts = {}
    for nid in x:
        encryptor = MIFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        ciphertexts[nid] = encryptor.encrypt(x[nid])

    decryptor = MIFELWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    result = decryptor.decrypt(ciphertexts, dk, y)

    print("x =", x)
    print("y =", y)
    print("sum_i <x_i, y_i> =", result)


if __name__ == "__main__":
    main()
