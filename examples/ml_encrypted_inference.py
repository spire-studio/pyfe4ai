"""Privacy-preserving linear model inference with PyFE4AI.

Demonstrates encrypting a feature vector and computing <x, w> without
revealing the raw features to the model server.

    Client                          Server
    ------                          ------
    x = [3.5, 1.2, 0.8]
    enc_x = encrypt(x)
           ──enc_x──►
                                    w = [0.5, -1.0, 2.0]
                                    dk = derive_key(w)
                                    score = decrypt(enc_x, dk, w)
                                    → score ≈ x·w = 2.15
"""

import numpy as np

from pyfe4ai.utils.ml_adapter import FESchemeWrapper, encrypt_features, compute_linear


def main() -> None:
    # --- scheme setup (done once by a trusted authority) ---
    features = np.array([3.5, 1.2, 0.8])
    model_weights = np.array([0.5, -1.0, 2.0])

    wrapper = FESchemeWrapper(
        "sife",
        "ddh",
        keygen_config={"sec_param": 128, "eta": len(features)},
        precision=1,  # scale_factor=10; keep quantized dot-product within dlog bound
    )

    # --- client side: encrypt feature vector ---
    enc_features = encrypt_features(features, wrapper)

    # --- server side: compute inner product on ciphertext ---
    score = compute_linear(enc_features, wrapper, model_weights)

    expected = float(np.dot(features, model_weights))
    print(f"plaintext  <x, w> = {expected}")
    print(f"encrypted  <x, w> = {score}")
    print(f"error             = {abs(score - expected):.6f}")


if __name__ == "__main__":
    main()
