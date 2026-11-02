"""Policy-gated federated gradient aggregation with PyFE4AI.

Three FL clients each hold a gradient. They encrypt it for a set of attributes
(``encrypt_gradient(..., attributes=...)``), and the aggregator receives the
weighted sum only for a policy those attributes satisfy
(``aggregate_gradients(..., policy=...)``); for a policy they do not satisfy
the adapter refuses before any key is issued.

This is ``mcfe/kp_ab_pairing`` (NPP22 Sec. 5.2) behind the ML adapter. Every
element of a gradient is a ciphertext of its own, so the scheme is for small
vectors and statistics, up to about 10^3 elements per aggregate (see
``docs/api/ml.rst``), not for the gradients of a large model: use
``ml_fl_aggregation.py`` or the Ring-LWE schemes for those. As in that example,
one wrapper plays every party, for experiments only.
"""

import numpy as np

import pyfe4ai
from pyfe4ai.ml import (
    FESchemeWrapper,
    aggregate_gradients,
    encrypt_gradient,
)


def main() -> None:
    # --- 3 clients, each with a 2-element gradient (one decimal) ---
    gradients = {
        "nid_0": np.array([1.0, 2.0]),
        "nid_1": np.array([3.0, 4.0]),
        "nid_2": np.array([5.0, 6.0]),
    }
    weights = {"nid_0": 1, "nid_1": 2, "nid_2": 3}
    attributes = ["purpose:research", "region:EU"]
    label = "round-1"

    wrapper = FESchemeWrapper(
        "mcfe",
        "kp_ab_pairing",
        keygen_config={
            "n": len(gradients),
            "lst_nid": list(gradients.keys()),
            "bound_x": 1000,
            "bound_y": 10,
        },
        precision=1,
    )

    # --- each client encrypts for the round's set of attributes ---
    encrypted = {}
    for nid, grad in gradients.items():
        encrypted[nid] = encrypt_gradient(
            grad, wrapper, nid=nid, label=label, attributes=attributes,
        )

    expected = sum(w * gradients[nid] for nid, w in weights.items())
    print("weights:", weights)
    print("attributes:", attributes)
    print(f"expected sum: {expected}")

    # --- a policy that the attributes satisfy releases the weighted sum ---
    policy = "and(purpose:research, region:EU)"
    result = aggregate_gradients(
        encrypted, wrapper, weights, label=label, policy=policy,
    )
    print(f"policy {policy}: FE result {result}")

    # --- a policy they do not satisfy is refused before a key is issued ---
    policy = "and(purpose:research, region:US)"
    try:
        aggregate_gradients(
            encrypted, wrapper, weights, label=label, policy=policy,
        )
        print(f"policy {policy}: aggregated")
    except pyfe4ai.FEKeyError:
        print(f"policy {policy}: refused (the attributes do not satisfy the policy)")


if __name__ == "__main__":
    main()
