from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPE
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEKeyGenerator


def main() -> None:
    config = {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 3,
        "n": 2,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1"],
        "bound_x": 4,
        "bound_y": 4,
        "pairing_group_param": "SS512",
    }

    kg = MIFEFHMultiIPEKeyGenerator(config)
    kg.setup()

    dct_x = {
        "nid_0": [1, -2, 3],
        "nid_1": [0, 2, -1],
    }
    dct_y = {
        "nid_0": [2, 1, -1],
        "nid_1": [1, -1, 2],
    }

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in config["lst_nid"]:
        encryptor = MIFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid])

    decryptor = MIFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    result = decryptor.decrypt(dct_ct, dk)

    expected = sum(
        x_i * y_i
        for nid in config["lst_nid"]
        for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )
    print(f"sum_i <x_i, y_i> = {result}")
    print(f"expected = {expected}")


if __name__ == "__main__":
    main()
