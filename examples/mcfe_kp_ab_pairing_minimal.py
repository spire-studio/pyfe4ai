"""Multi-client FE with access control (``mcfe/kp_ab_pairing``, NPP22 Sec. 5.2).

Each of three clients holds one number and encrypts it under a tag (the round)
and the round's set of attributes; all clients use the same set. A functional
key is for a weight vector and a policy: when the round's attributes satisfy the
policy it gives the weighted sum of the clients' numbers, and when they do not
it is refused. The key authority holds the master key and the
clients' secret keys, so this is a private-key scheme.

A decryption needs the ciphertexts of every client (the shares of the clients'
keys add up to 1, so a client of weight 0 is still needed), and a client
encrypts once per tag.

**Known-insecure** (2026-11): one corrupted client and two keys whose policy
the attributes do not satisfy read an inner product, an honest client's input
among them, so the policy does not hold against a decryptor that colludes
with a client (``docs/security.rst``). The scheme emits a
``FESecurityWarning``, and the ``standard`` profile refuses it. Run this to see
how the scheme is used, not to protect data.
"""

import pyfe4ai


def main() -> None:
    kg, MCFEKPAB = pyfe4ai.create(
        "mcfe/kp_ab_pairing", {"n": 3, "bound_x": 10, "bound_y": 10}
    )
    kg.setup()
    pp = kg.get_public_parameters()

    x = {"nid_0": 5, "nid_1": -3, "nid_2": 7}
    y = {"nid_0": 2, "nid_1": 1, "nid_2": -4}
    expected = sum(x[nid] * y[nid] for nid in x)

    clients = {
        nid: MCFEKPAB({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        for nid in kg.lst_nid
    }
    aggregator = MCFEKPAB({"keys": {"pp": pp}})

    # every client encrypts its number under one tag and one set of attributes
    attributes = ["purpose:research", "region:EU"]
    cts = {
        nid: clients[nid].encrypt([x[nid]], "round-1", attributes=attributes)
        for nid in kg.lst_nid
    }
    print("x =", x, " y =", y, " plain sum_i x_i y_i =", expected)
    print("attributes of round-1:", attributes)

    for policy in (
        "and(purpose:research, region:EU)",
        "or(region:US, region:EU)",
        "and(purpose:research, region:US)",
    ):
        dk = kg.get_decryption_keys(
            "agg",
            credentials={"fusion_weight": {nid: [y[nid]] for nid in y}, "policy": policy},
        )
        try:
            outcome = "sum_i x_i y_i = {}".format(aggregator.decrypt(cts, dk))
        except pyfe4ai.FEKeyError:
            outcome = "refused (the attributes do not satisfy the policy)"
        print("key for {}: {}".format(policy, outcome))


if __name__ == "__main__":
    main()
