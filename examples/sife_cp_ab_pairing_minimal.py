"""Ciphertext-policy attribute-based IPFE (``sife/cp_ab_pairing``, ACGU20).

The ciphertext is made under a policy over attribute labels, and a functional
key is for a vector ``y`` and the set of attributes its holder has. The key
opens ``<x, y>`` when the attributes satisfy the policy and is refused
otherwise. It is a public-key scheme: anyone with the public parameters
encrypts. The labels are fixed at setup (``universe``).

A policy with one leaf, such as ``id:alice``, gives the functionality of
identity-based IPFE: the key of ``id:alice`` opens the ciphertext, and a key of
another identity is refused. (This is a one-leaf policy of the library's
general scheme, not the function encoding of ACGU20 App. C.1.)
"""

import pyfe4ai


def main() -> None:
    x = [3, -1, 2]
    y = [1, 2, 4]
    expected = sum(x_i * y_i for x_i, y_i in zip(x, y))

    config = {
        "eta": len(x),
        "bound_x": 10,
        "bound_y": 10,
        "universe": ["org:A", "role:auditor", "role:regulator", "id:alice", "id:bob"],
    }
    kg, SIFECPAB = pyfe4ai.create("sife/cp_ab_pairing", config)
    kg.setup()

    # the public parameters are all that encrypting needs
    crypto = SIFECPAB({"keys": {"pp": kg.get_public_parameters()}})

    def open_with(ct, attributes):
        """The result of the key for *attributes* on *ct*, or the refusal."""
        dk = kg.get_decryption_keys(
            "sid_0", credentials={"fusion_weight": y, "attributes": attributes}
        )
        try:
            return "<x, y> = {}".format(crypto.decrypt(ct, dk))
        except pyfe4ai.FEKeyError:
            return "refused (the attributes do not satisfy the policy)"

    # --- a policy with and/or gates ---
    policy = "and(org:A, or(role:auditor, role:regulator))"
    ct = crypto.encrypt(x, policy=policy)
    print("x =", x, " y =", y, " plain <x, y> =", expected)
    print("policy:", policy)
    print("key for ['org:A', 'role:regulator']:", open_with(ct, ["org:A", "role:regulator"]))
    print("key for ['org:A']:", open_with(ct, ["org:A"]))
    print("key for ['role:auditor']:", open_with(ct, ["role:auditor"]))

    # --- identity equality: a one-leaf policy ---
    ct = crypto.encrypt(x, policy="id:alice")
    print("policy: id:alice")
    print("key for ['id:alice']:", open_with(ct, ["id:alice"]))
    print("key for ['id:bob']:", open_with(ct, ["id:bob"]))


if __name__ == "__main__":
    main()
