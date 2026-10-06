"""Threshold MCFE example: any 2 of 3 decryption servers can decrypt."""

from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFE
from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEKeyGenerator


def main() -> None:
    x = {
        "nid_0": [1],
        "nid_1": [1],
        "nid_2": [1],
    }
    y = {
        "nid_0": [1],
        "nid_1": [1],
        "nid_2": [1],
    }
    label = "demo-label"
    config = {
        "sec_param": 128,
        "eta": 1,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": list(x.keys()),
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
    }

    kg = ThresholdMCFEKeyGenerator(config)
    kg.setup()
    pp = kg.get_public_parameters()

    ciphertexts = {}
    decryptors = {}
    for nid, vec in x.items():
        encryptor = ThresholdMCFE(
            {
                "id": nid,
                "precision": 3,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        ciphertexts[nid] = encryptor.encrypt(vec, label)
        decryptors[nid] = encryptor

    share_ids = ["sid_1", "sid_2"]
    shared_results = {}
    for sid in share_ids:
        dk = kg.get_decryption_keys(
            sid,
            credentials={"fusion_weight": y, "label": label},
        )
        partial = ThresholdMCFE(
            {
                "id": sid,
                "precision": 3,
                "keys": {"pp": pp, "dk": dk},
            }
        )
        shared_results[sid] = partial.share_decrypt(ciphertexts, y, dk, share_ids)

    result = decryptors["nid_0"].combine_decrypt(shared_results)

    print("x =", x)
    print("y =", y)
    print("decryption servers used =", share_ids)
    print("sum_i <x_i, y_i> =", result)


if __name__ == "__main__":
    main()
