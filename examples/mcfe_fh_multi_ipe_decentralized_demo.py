from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPEPairing
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPEPairingKeyGenerator


def main() -> None:
    config = {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": 4,
        "bound_y": 4,
    }
    label = "decentralized-demo-label"

    kg = DecentralizedMCFEFHMultiIPEPairingKeyGenerator(config)
    kg.setup()

    dct_x = {
        "nid_0": [1, -2],
        "nid_1": [0, 3],
        "nid_2": [2, 1],
    }
    dct_y = {
        "nid_0": [2, 1],
        "nid_1": [-1, 2],
        "nid_2": [1, -2],
    }

    pp = kg.get_public_parameters()
    dct_ct = {}
    dct_dk_shares = {}
    for nid in config["lst_nid"]:
        crypto_nid = DecentralizedMCFEFHMultiIPEPairing(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        # each client derives its key from its own weights and the function tag
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y[nid], "weights-v1")

    decryptor = DecentralizedMCFEFHMultiIPEPairing({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    result = decryptor.decrypt(dct_ct, dk, dct_y, label)

    expected = sum(
        x_i * y_i
        for nid in config["lst_nid"]
        for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )
    print(f"sum_i <x_i, y_i> = {result}")
    print(f"expected = {expected}")


if __name__ == "__main__":
    main()
