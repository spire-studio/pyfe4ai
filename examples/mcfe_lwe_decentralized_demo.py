"""Decentralized MCFE/LWE end-to-end example."""

from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWE
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWEKeyGenerator


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
    label = "demo-dmcfe-lwe"

    kg = DecentralizedMCFELWEKeyGenerator(
        {
            "sec_param": 64,
            "eta": 1,
            "n": len(x),
            "s": 1,
            "lst_nid": list(x.keys()),
            "lwe_n": 16,
            "bound_x": 8,
            "bound_y": 8,
            "bound_u": 2,
            "label_modulus": 8,
        }
    )
    kg.setup()

    pp = kg.get_public_parameters()
    ciphertexts = {}
    dk_shares = {}
    for nid in x:
        participant = DecentralizedMCFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        ciphertexts[nid] = participant.encrypt(x[nid], label)
        dk_shares[nid] = participant.derive_function_decryption_key_share(y)

    decryptor = DecentralizedMCFELWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(dk_shares)
    result = decryptor.decrypt(ciphertexts, dk, y, label)

    print("label =", label)
    print("x =", x)
    print("y =", y)
    print("sum_i <x_i, y_i> =", result)


if __name__ == "__main__":
    main()
