"""ML framework adapter layer for PyFE4AI.

Provides two high-level APIs that hide FE complexity from ML engineers:

**FL aggregation** (``eta=1``, ndarray helpers):
    Encrypt per-element gradients from *n* clients, aggregate with weights.

    >>> keygen_cfg = {"eta": 1, "n": 2, "lst_nid": ["c0", "c1"], "bound_x": 1000, "bound_y": 10}
    >>> wrapper = FESchemeWrapper("mcfe", "lwe", keygen_cfg, precision=2)
    >>> enc0 = encrypt_gradient(grad0, wrapper, nid="c0", label="round-1")
    >>> enc1 = encrypt_gradient(grad1, wrapper, nid="c1", label="round-1")
    >>> result = aggregate_gradients(
    ...     {"c0": enc0, "c1": enc1}, wrapper, {"c0": 1, "c1": 2}, label="round-1")

    Extra entries of the crypto objects' config go to ``crypto_options``,
    e.g. ``FESchemeWrapper("mcfe", "ring_lwe", keygen_cfg,
    crypto_options={"ct_format": "binary"})``.

**Encrypted inference** (``eta=feature_dim``, vector mode):
    Encrypt a feature vector, compute ``⟨x, w⟩`` without revealing *x*.

    >>> wrapper = FESchemeWrapper("sife", "lwe", {"eta": 3, "bound_x": 10, "bound_y": 10})
    >>> enc = encrypt_features(feature_vec, wrapper)
    >>> score = compute_linear(enc, wrapper, model_weights)

PyTorch is optional — functions accept both :class:`numpy.ndarray` and
``torch.Tensor`` inputs.

**Threshold MCFE** (``mcfe/ring_lwe_threshold``): the wrapper also runs the
``s`` decryption servers. :func:`prepare_aggregation` (or
:func:`aggregate_gradients` without ``prepared``) gives each enrolled server
its label key share of the round and, once per weights, its weight key share
(the scheme's split key shares, its default); :func:`aggregate_gradients`
then has the servers decrypt partially and combines their answers. The two
steps are also separate functions -- :func:`partial_decrypt_gradients` per
server, :func:`combine_partial_decryptions` at the combiner -- whose
:class:`ThresholdPartial` serialises like :class:`EncryptedGradient`
(binary ciphertexts and partial decryptions, the scheme's default format, as
base64).

**Single trust domain.** A :class:`FESchemeWrapper` runs the key generator
itself, so it holds the master secret key and plays every party (clients,
key authority, decryptor). It is meant for experiments and for code that
already trusts one process with everything; a deployment keeps the master
key away from the aggregator. The wrapper applies a default policy that a
key holder would otherwise have to enforce: :func:`aggregate_gradients`
needs a ciphertext and a non-zero weight for every enrolled client (pass
``allow_partial=True`` to relax this), each client encrypts at most once per
label, and each label is aggregated with one weight vector only. These
checks stop the obvious ways of isolating one client, not all of them: a
single aggregate with very unequal weights (e.g. ``(W, 1)`` with ``W`` above
twice the largest gradient entry) separates two clients' gradients, as with
any inner-product FE; choosing the weights is the key holder's
responsibility, and :meth:`FESchemeWrapper.get_decryption_keys` issues keys
for any weights.


The functions and classes are defined in the submodules ``wrapper``,
``serialization``, ``gradients``, ``threshold`` and ``inference``, and
imported from here; :mod:`pyfe4ai.ml.quantization` holds the float /
integer conversion.
"""

from pyfe4ai.ml.gradients import aggregate_gradients, encrypt_gradient, precompute_gradient, prepare_aggregation
from pyfe4ai.ml.inference import compute_linear, encrypt_features
from pyfe4ai.ml.serialization import EncryptedFeatures, EncryptedGradient
from pyfe4ai.ml.threshold import ThresholdPartial, combine_partial_decryptions, partial_decrypt_gradients
from pyfe4ai.ml.wrapper import FESchemeWrapper, PreparedAggregation, ndarray_to_tensor, tensor_to_ndarray

__all__ = [
    "FESchemeWrapper",
    "PreparedAggregation",
    "EncryptedGradient",
    "EncryptedFeatures",
    "ThresholdPartial",
    "precompute_gradient",
    "encrypt_gradient",
    "prepare_aggregation",
    "aggregate_gradients",
    "partial_decrypt_gradients",
    "combine_partial_decryptions",
    "encrypt_features",
    "compute_linear",
    "tensor_to_ndarray",
    "ndarray_to_tensor",
]
