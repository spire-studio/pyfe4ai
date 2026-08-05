import random

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEKeyGenerator


def main() -> None:
    config = {
        "sec_param": 64,
        "sec_level": 2,
        "eta": 2,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "u_bound": 3,
        "label_modulus": 17,
        "pairing_group_param": "SS512",
    }
    label = "threshold-demo-label"

    kg = ThresholdMCFEFHMultiIPEKeyGenerator(config)
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
    for nid in config["lst_nid"]:
        encryptor = ThresholdMCFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid], label)

    credentials = {"fusion_weight": dct_y, "label": label}
    enrolled = random.sample(config["lst_sid"], config["t"])
    dct_prime = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        decryptor = ThresholdMCFEFHMultiIPE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        dct_prime[sid] = decryptor.share_decrypt(dct_ct, credentials, dk, enrolled)

    combiner = ThresholdMCFEFHMultiIPE(
        {"id": config["lst_nid"][0], "keys": {"pp": pp, "sk": kg.get_private_keys(config["lst_nid"][0])}}
    )
    result = combiner.combine_decrypt(dct_prime)

    expected = sum(
        x_i * y_i
        for nid in config["lst_nid"]
        for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )
    print(f"sum_i <x_i, y_i> = {result}")
    print(f"expected = {expected}")


if __name__ == "__main__":
    main()
