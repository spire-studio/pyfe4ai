"""Federated learning gradient aggregation with PyFE4AI.

Simulates 3 FL clients, each holding a local gradient vector.
Gradients are encrypted with MCFE so that only the weighted sum
is revealed — individual gradients stay private.

    Client 0    Client 1    Client 2       Aggregator
    --------    --------    --------       ----------
    g0=[1,2]    g1=[3,4]    g2=[5,6]
    enc(g0) ─►  enc(g1) ─►  enc(g2)  ─►   decrypt → w0*g0 + w1*g1 + w2*g2
"""

import numpy as np

from pyfe4ai.utils.ml_adapter import (
    FESchemeWrapper,
    encrypt_gradient,
    aggregate_gradients,
)


def main() -> None:
    # --- 3 clients, each with a 2-element gradient ---
    gradients = {
        "nid_0": np.array([1.0, 2.0]),
        "nid_1": np.array([3.0, 4.0]),
        "nid_2": np.array([5.0, 6.0]),
    }
    # Equal-weight average (weights = 1 for simple sum)
    weights = {"nid_0": 1, "nid_1": 1, "nid_2": 1}
    label = "round-1"

    wrapper = FESchemeWrapper(
        "mcfe",
        "ddh",
        keygen_config={
            "sec_param": 128,
            "eta": 1,
            "n": len(gradients),
            "s": 1,
            "lst_nid": list(gradients.keys()),
        },
        precision=3,
    )

    # --- each client encrypts ---
    encrypted = {}
    for nid, grad in gradients.items():
        encrypted[nid] = encrypt_gradient(
            grad, wrapper, nid=nid, label=label,
        )

    # --- aggregator decrypts the weighted sum ---
    result = aggregate_gradients(
        encrypted, wrapper, weights, label=label,
    )

    expected = sum(w * gradients[nid] for nid, w in weights.items())
    print("gradients:", {k: v.tolist() for k, v in gradients.items()})
    print("weights:  ", weights)
    print(f"expected sum: {expected}")
    print(f"FE result:    {result}")


if __name__ == "__main__":
    main()
