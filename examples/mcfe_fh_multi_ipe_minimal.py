from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEKeyGenerator


def main() -> None:
    config = {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 2,
        "n": 2,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1"],
        "bound_x": 4,
        "bound_y": 4,
        "u_bound": 3,
        "label_modulus": 17,
        "pairing_group_param": "SS512",
    }
    label = "demo-label"

    kg = MCFEFHMultiIPEKeyGenerator(config)
    kg.setup()

    dct_x = {"nid_0": [1, -2], "nid_1": [0, 3]}
    dct_y = {"nid_0": [2, 1], "nid_1": [-1, 2]}

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in config["lst_nid"]:
        encryptor = MCFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid], label)

    decryptor = MCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
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
