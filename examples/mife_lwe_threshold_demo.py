"""Threshold MIFE/LWE example with two enrolled decryption shares."""

from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator


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

    kg = ThresholdMIFELWEKeyGenerator(
        {
            "sec_param": 64,
            "eta": 1,
            "n": len(x),
            "s": 3,
            "t": 2,
            "lst_nid": list(x.keys()),
            "lst_sid": ["sid_0", "sid_1", "sid_2"],
            "lwe_n": 16,
            "bound_x": 8,
            "bound_y": 8,
        }
    )
    kg.setup()

    pp = kg.get_public_parameters()
    ciphertexts = {}
    encryptors = {}
    for nid in x:
        encryptor = ThresholdMIFELWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        ciphertexts[nid] = encryptor.encrypt(x[nid])
        encryptors[nid] = encryptor

    enrolled = ["sid_0", "sid_1"]
    partials = {}
    for sid in enrolled:
        dk_sid = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        partial = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk_sid}})
        partials[sid] = partial.share_decrypt(ciphertexts, y, dk_sid, enrolled)

    result = encryptors["nid_0"].combine_decrypt(partials)

    print("enrolled sids =", enrolled)
    print("x =", x)
    print("y =", y)
    print("sum_i <x_i, y_i> =", result)


if __name__ == "__main__":
    main()
