"""Key-policy attribute-based IPFE (``sife/kp_ab_pairing``, NPP22 Sec. 4).

The ciphertext is made for a set of attribute labels (here a purpose and a
region), and a functional key is for a vector ``y`` and a policy. The key opens
``<x, y>`` when the attributes satisfy its policy and is refused otherwise.
It is a public-key scheme: anyone with the public parameters encrypts, and any
label is an attribute (no list of labels is fixed at setup).
"""

import pyfe4ai


def main() -> None:
    x = [3, -1, 2]
    y = [1, 2, 4]
    expected = sum(x_i * y_i for x_i, y_i in zip(x, y))

    kg, SIFEKPAB = pyfe4ai.create(
        "sife/kp_ab_pairing", {"eta": len(x), "bound_x": 10, "bound_y": 10}
    )
    kg.setup()

    # the public parameters are all that encrypting needs
    crypto = SIFEKPAB({"keys": {"pp": kg.get_public_parameters()}})

    attributes = ["purpose:research", "region:EU"]
    ct = crypto.encrypt(x, attributes=attributes)
    print("x =", x, " y =", y, " plain <x, y> =", expected)
    print("ciphertext attributes:", ct["attributes"])

    for policy in (
        "and(purpose:research, region:EU)",
        "or(region:EU, region:US)",
        "and(purpose:research, region:US)",
    ):
        dk = kg.get_decryption_keys(
            "sid_0", credentials={"fusion_weight": y, "policy": policy}
        )
        try:
            outcome = "<x, y> = {}".format(crypto.decrypt(ct, dk))
        except pyfe4ai.FEKeyError:
            outcome = "refused (the attributes do not satisfy the policy)"
        print("key for {}: {}".format(policy, outcome))


if __name__ == "__main__":
    main()
