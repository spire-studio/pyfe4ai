"""Encrypted inference with the ML adapter (``eta=feature_dim``, vector mode):
:func:`encrypt_features` and :func:`compute_linear`."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyfe4ai.core.exceptions import FESchemeError
from pyfe4ai.ml.serialization import EncryptedFeatures
from pyfe4ai.ml.wrapper import FESchemeWrapper, _require_single_decryptor, _to_numpy

#: SIFE schemes whose ``encrypt`` takes an ``eta x k`` matrix (``k`` vectors
#: in the ring's coefficients): a feature vector is one column of it.
_MATRIX_PLAINTEXT_SIFE = {("sife", "ring_lwe")}


def encrypt_features(
    features: np.ndarray | Any,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
) -> EncryptedFeatures:
    """Encrypt a feature vector for privacy-preserving inference.

        Args:
            features: Feature vector (numpy array or ``torch.Tensor``),
                flattened and quantized with the wrapper's precision.
            wrapper: FE scheme wrapper instance.
            nid: Node identifier.
    """
    arr = _to_numpy(features).ravel().astype(np.float64)
    qcfg = wrapper.qconfig
    quantized = qcfg.quantize_array(arr).tolist()

    encryptor = wrapper._get_encryptor(nid)
    if (wrapper.scheme_type, wrapper.variant) in _MATRIX_PLAINTEXT_SIFE:
        ct = encryptor.encrypt([[v] for v in quantized])  # one column: the feature vector
    else:
        ct = encryptor.encrypt(quantized)

    return EncryptedFeatures(
        ciphertext=ct,
        dim=len(quantized),
        precision=qcfg.precision,
        mode=qcfg.mode,
    )


def compute_linear(
    encrypted: EncryptedFeatures,
    wrapper: FESchemeWrapper,
    weights: np.ndarray | Any,
    *,
    sid: str = "sid_infer",
) -> float:
    """Compute ⟨features, weights⟩ on encrypted features.

        Args:
            encrypted: Encrypted data container.
            wrapper: FE scheme wrapper instance.
            weights: Weight vector or dict.
            sid: Session / decryption-key identifier.
    """
    _require_single_decryptor(wrapper, "compute_linear")

    w_arr = _to_numpy(weights).ravel().astype(np.float64)
    qcfg = wrapper.qconfig
    q_weights = qcfg.quantize_array(w_arr).tolist()

    dk = wrapper.get_decryption_keys(sid, fusion_weight=q_weights)

    decryptor = wrapper._get_decryptor(sid)
    raw_result = decryptor.decrypt(encrypted.ciphertext, dk, q_weights)
    if isinstance(raw_result, list):  # a matrix ciphertext of one column: its one inner product
        raw_result = raw_result[0] if len(raw_result) == 1 else None
    if raw_result is None:
        raise FESchemeError(
            "the inner product does not decrypt (result outside the decryption bound, "
            "or key and ciphertext do not match)"
        )

    # Both x and w were quantized → result scaled by scale_factor^2
    return qcfg.dequantize(raw_result, order=2)
