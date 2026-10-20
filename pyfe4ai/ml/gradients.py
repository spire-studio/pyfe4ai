"""Federated aggregation with the ML adapter (``eta=1``, ndarray helpers):
:func:`encrypt_gradient` per client, :func:`aggregate_gradients` at the
aggregator, and their offline phases :func:`precompute_gradient` and
:func:`prepare_aggregation`."""

from __future__ import annotations

import numbers
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np

from pyfe4ai._validation import int_vector
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.ml.quantization import QuantizationConfig
from pyfe4ai.ml.serialization import EncryptedGradient
from pyfe4ai.ml.threshold import (
    _enrolled_servers,
    _server_key_shares,
    combine_partial_decryptions,
    partial_decrypt_gradients,
)
from pyfe4ai.ml.wrapper import (
    FESchemeWrapper,
    PreparedAggregation,
    _aggregated,
    _require_labelled_aggregation,
    _require_single_decryptor,
    _serves_threshold,
    _to_numpy,
)

def precompute_gradient(
    shape,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
    label: str,
) -> Any:
    """Offline phase of :func:`encrypt_gradient` for a gradient of *shape*.

    For ``mcfe/ring_lwe`` (see
    :meth:`~pyfe4ai.schemes.mcfe.ring_lwe.MCFERingLWE.precompute`): label
    pads, Gaussian samples and NTT products are computed before the gradient
    exists; ``encrypt_gradient(..., precomputed=...)`` then only adds the
    plaintext term. For ``mcfe/ddh``, ``mcfe/damgard`` and ``mcfe/paillier``
    (and their decentralized variants; :mod:`pyfe4ai.schemes.mcfe.offline`)
    the label masks of every element are computed. The result is secret and
    one-time; the ciphertexts are those of the plain call.

    Raises:
        ValueError: If the scheme has no offline phase, or *nid* already
            encrypted under *label*.
    """
    _require_labelled_aggregation(wrapper, "precompute_gradient")
    if not wrapper.has_label or not isinstance(label, str):
        raise ValueError("precomputation needs an MCFE scheme and a string label")
    encryptor = wrapper._get_encryptor(nid)
    precompute = getattr(encryptor, "precompute", None)
    if precompute is None:
        raise ValueError("{}/{} has no offline encryption phase".format(wrapper.scheme_type, wrapper.variant))
    with wrapper._label_lock:
        if (nid, label) in wrapper._encrypted_labels:
            raise ValueError("client {} already encrypted under label {!r}".format(nid, label))
    return precompute(label, [tuple(int(d) for d in np.shape(np.empty(shape)))])


def encrypt_gradient(
    gradient: np.ndarray | Any,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
    label: str | None = None,
    precomputed: Any = None,
) -> EncryptedGradient:
    """Encrypt a gradient array for federated aggregation.

        Args:
            gradient: Gradient array (numpy array or ``torch.Tensor``).
            wrapper: FE scheme wrapper instance.
            nid: Node identifier.
            label: Round label; MCFE allows one ciphertext per client and
                label, so a label this wrapper already used for *nid* is
                refused.
            precomputed: The result of :func:`precompute_gradient` for this
                client, label and shape (online phase).
    """
    _require_labelled_aggregation(wrapper, "encrypt_gradient")
    arr = _to_numpy(gradient).astype(np.float64)
    encryptor = wrapper._get_encryptor(nid)

    kwargs: dict[str, Any] = {}
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for encryption")
        # reserve (nid, label) before encrypting: of two concurrent calls one
        # encrypts, the other is refused; a failed encryption releases it
        with wrapper._label_lock:
            if (nid, label) in wrapper._encrypted_labels:
                raise ValueError(
                    "client {} already encrypted under label {!r}; use a fresh label "
                    "for every round".format(nid, label)
                )
            wrapper._encrypted_labels.add((nid, label))
        kwargs["label"] = label
    if precomputed is not None:
        kwargs["precomputed"] = precomputed

    try:
        cts = encryptor.encrypt_lst_ndarray([arr], **kwargs)
    except BaseException:
        if wrapper.has_label:
            with wrapper._label_lock:
                wrapper._encrypted_labels.discard((nid, label))
        raise

    return EncryptedGradient(
        ciphertexts=cts,
        shape=arr.shape,
        precision=wrapper.qconfig.precision,
        mode=wrapper.qconfig.mode,
        nid=nid,
    )


def _integer_weights(weights: dict, qconfig: QuantizationConfig) -> tuple[dict, int]:
    """Return ``({nid: [int]}, scale)``: the weights as integers, multiplied
    by the quantisation scale if any of them is fractional."""
    lists = {
        nid: [w] if np.ndim(w) == 0 else list(np.ravel(w)) for nid, w in weights.items()
    }
    values = [v for lst in lists.values() for v in lst]
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("weights must be finite")
    try:
        return {nid: int_vector(lst, "weight of {}".format(nid)) for nid, lst in lists.items()}, 1
    except FEValidationError:
        pass  # fractional weights: quantise them below
    scale = qconfig.scale_factor
    if scale == 1:
        raise ValueError("fractional weights need a wrapper precision > 0")
    scaled = {
        nid: [Fraction(v) * scale if isinstance(v, numbers.Rational) else float(v) * scale for v in lst]
        for nid, lst in lists.items()
    }
    # exact for rationals; floats (0.1 * 10 = 1.0000000000000002) get an absolute tolerance
    if any(abs(v - round(v)) > 1e-6 for lst in scaled.values() for v in lst):
        raise ValueError(
            "the weights are not multiples of 1/{} at the wrapper's precision; "
            "raise the precision or round the weights".format(scale)
        )
    return {nid: [int(round(v)) for v in lst] for nid, lst in scaled.items()}, scale


def _aggregation_key(wrapper, weights, label, sid, nids, allow_partial, n_chunks_fn, servers=None):
    """Policy checks of :func:`aggregate_gradients` and the functional key:
    ``(fusion_weight, weight_scale, dk)``; *nids* are the clients whose
    ciphertexts will be aggregated, ``n_chunks_fn()`` the number of packed
    chunks a key must cover (``None``: not a packed scheme). For a threshold
    MCFE run by the wrapper ``dk`` holds the key shares of the enrolled
    *servers* (:func:`_server_key_shares`)."""
    # Normalise weights to {nid: [w]} format for eta=1 schemes
    fusion_weight, weight_scale = _integer_weights(weights, wrapper.qconfig)
    if wrapper.scheme_type == "sife":
        if len(nids) != 1 or set(fusion_weight) != set(nids):
            raise ValueError(
                "single-input FE aggregates one client: pass one ciphertext and its weight"
            )
    elif not allow_partial:
        enrolled = set(getattr(wrapper.kg, "lst_nid", None) or nids)
        if set(nids) != enrolled or set(fusion_weight) != enrolled:
            raise ValueError(
                "aggregation needs a ciphertext and a weight for every enrolled client "
                "(allow_partial=True to override)"
            )
        if any(v == 0 for lst in fusion_weight.values() for v in lst):
            raise ValueError("aggregation needs non-zero weights (allow_partial=True to override)")
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for decryption")
        previous = wrapper._aggregated.setdefault(label, fusion_weight)
        if previous != fusion_weight:
            raise ValueError(
                "label {!r} was already aggregated with other weights".format(label)
            )

    # Build credentials for dk generation
    # SIFE expects fusion_weight as a flat list; MCFE/MIFE expect a dict.
    if wrapper.scheme_type in ("mcfe", "mife"):
        credentials: dict[str, Any] = {"fusion_weight": fusion_weight}
    else:
        # Single-input: take the first (only) client's weight list
        credentials = {"fusion_weight": next(iter(fusion_weight.values()))}
    if wrapper.has_label:
        credentials["label"] = label
    n_chunks = n_chunks_fn()
    if n_chunks is not None:
        # identity-based schemes issue one key term per packed chunk
        credentials["n_chunks"] = n_chunks
    if servers is not None:
        return fusion_weight, weight_scale, _server_key_shares(wrapper, credentials, servers)
    dk = wrapper.kg.get_decryption_keys(sid, credentials=credentials)
    return fusion_weight, weight_scale, dk


def prepare_aggregation(
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list],
    *,
    shape,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
    servers: list | None = None,
) -> PreparedAggregation:
    """Derive the functional key of an aggregation before the ciphertexts arrive.

    The key depends on the weights, the label and (for the packed Ring-LWE
    MCFE) the number of chunks of a gradient of *shape*, not on the
    ciphertexts; :func:`aggregate_gradients` with ``prepared=`` then only
    decrypts. The same policy checks apply, with the clients of *weights* as
    those whose ciphertexts will be aggregated.

    For ``mcfe/ring_lwe_threshold`` the key shares of the enrolled *servers*
    (default: the first ``t`` of the public parameters' ``lst_sid``): their
    label key shares of the round, each server given its weight key share
    once per weights (again where a copy of the wrapper or the server's store
    lacks it when the round is decrypted); *sid* is the combiner. The result
    holds key material (see :class:`PreparedAggregation`).
    """
    _require_labelled_aggregation(wrapper, "prepare_aggregation")
    _require_single_decryptor(wrapper, "prepare_aggregation", served_threshold=True)
    if servers is not None and not _serves_threshold(wrapper):
        raise ValueError("servers are those of a threshold MCFE")
    enrolled = _enrolled_servers(wrapper, servers) if _serves_threshold(wrapper) else None
    decryptor = wrapper._get_decryptor(sid)
    n_chunks = None
    if getattr(decryptor, "key_chunks", None) is not None:
        ring_n = int(wrapper.pp["ring_n"])
        size = int(np.prod(np.shape(np.empty(shape)), dtype=np.int64))
        n_chunks = max(1, -(-size // ring_n))
    fusion_weight, weight_scale, dk = _aggregation_key(
        wrapper, weights, label, sid, list(weights), allow_partial, lambda: n_chunks, enrolled
    )
    prepare_key = getattr(decryptor, "prepare_key", None)
    if prepare_key is not None:  # element-wise MCFE: the label terms of the key too
        dk = prepare_key(dk, label, [tuple(int(d) for d in np.shape(np.empty(shape)))])
    return PreparedAggregation(label, sid, fusion_weight, weight_scale, n_chunks, dk, allow_partial, enrolled)


def aggregate_gradients(
    encrypted: dict[str, EncryptedGradient],
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list] | None = None,
    *,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
    prepared: PreparedAggregation | None = None,
    servers: list | None = None,
) -> np.ndarray:
    """Decrypt and aggregate encrypted gradients from multiple clients.

        Args:
            encrypted: Encrypted data container.
            wrapper: FE scheme wrapper instance.
            weights: Weight per client (a number or a one-element list);
                fractional weights are quantised with the wrapper's
                precision and the result is scaled back.
            sid: Session / decryption-key identifier.
            label: Round label (MCFE); each label is aggregated with one
                weight vector only.
            allow_partial: Allow zero weights and a subset of the enrolled
                clients. Off by default: the wrapper holds the master key,
                and such weights would decrypt a single client's gradient
                (so would very unequal weights; see :mod:`pyfe4ai.ml`).
            prepared: A key from :func:`prepare_aggregation` (online phase);
                *weights*, if given, and *label* must match it.
            servers: ``mcfe/ring_lwe_threshold``: the servers whose partial
                decryptions are combined (default: those of *prepared*, or
                the first ``t`` of ``lst_sid``); the wrapper runs them
                (:func:`partial_decrypt_gradients`,
                :func:`combine_partial_decryptions`) and *sid* is the
                combiner. If their label key shares turn out to be of
                different sharings (``MixedSharingsError``), all servers
                fetch them again and decrypt once more.

        Raises:
            ValueError: If the policy above is violated, or SIFE is given
                more than one client.
    """
    _require_labelled_aggregation(wrapper, "aggregate_gradients")
    _require_single_decryptor(wrapper, "aggregate_gradients", served_threshold=True)
    threshold = _serves_threshold(wrapper)
    if servers is not None and not threshold:
        raise ValueError("servers are those of a threshold MCFE")
    decryptor = None

    def n_chunks_fn():
        # Collect ciphertext arrays per nid
        # decrypt_lst_ndarray_ct expects: list of arrays (one per layer)
        # where each array element is a ciphertext dict
        # For SIFE: dict_ndarray_ct is list of arrays
        # For MCFE/MIFE: dict_ndarray_ct is dict[nid -> list of arrays]
        nonlocal decryptor
        sample_enc = next(iter(encrypted.values()))
        decryptor = wrapper._get_decryptor(prepared.sid if prepared is not None else sid)
        key_chunks = getattr(decryptor, "key_chunks", None)
        return key_chunks(sample_enc.ciphertexts) if key_chunks is not None else None

    if prepared is None:
        if weights is None:
            raise ValueError("need the weights (or a prepared aggregation)")
        enrolled = _enrolled_servers(wrapper, servers) if threshold else None
        fusion_weight, weight_scale, dk = _aggregation_key(
            wrapper, weights, label, sid, list(encrypted), allow_partial, n_chunks_fn, enrolled
        )
        if threshold:
            prepared = PreparedAggregation(label, sid, fusion_weight, weight_scale, n_chunks_fn(), dk, allow_partial,
                                           enrolled)
    else:
        n_chunks = n_chunks_fn()
        if prepared.label != label:
            raise ValueError("the prepared key is for label {!r}".format(prepared.label))
        if weights is not None and _integer_weights(weights, wrapper.qconfig)[0] != prepared.fusion_weight:
            raise ValueError("the prepared key is for other weights")
        if set(encrypted) != set(prepared.fusion_weight):
            raise ValueError("the prepared key is for the clients {}".format(sorted(prepared.fusion_weight)))
        if n_chunks is not None and (prepared.n_chunks or 0) < n_chunks:
            raise ValueError("the prepared key covers fewer chunks than the ciphertexts")
        if threshold and servers is not None and list(servers) != prepared.servers:
            raise ValueError("the prepared key shares are for the servers {}".format(prepared.servers))
        fusion_weight, weight_scale, dk = prepared.fusion_weight, prepared.weight_scale, prepared.dk

    if threshold:  # the servers the wrapper runs, then the combiner
        from pyfe4ai.schemes.threshold.binding import MixedSharingsError

        for attempt in (0, 1):
            partials = {
                server: partial_decrypt_gradients(encrypted, wrapper, prepared, server=server)
                for server in prepared.servers
            }
            try:
                return combine_partial_decryptions(partials, wrapper, prepared)
            except MixedSharingsError:
                if attempt:
                    raise
                # the label's parts were shared again between the servers' fetches: all servers fetch them again
                credentials = {"fusion_weight": prepared.fusion_weight, "label": prepared.label,
                               "n_chunks": prepared.n_chunks}
                prepared = replace(prepared, dk=_server_key_shares(wrapper, credentials, prepared.servers))

    if wrapper.scheme_type in ("mcfe", "mife"):
        # Multi-client: {nid: [array_per_layer]}
        dict_ct: dict[str, list] = {}
        for nid, enc in encrypted.items():
            dict_ct[nid] = enc.ciphertexts

        kwargs: dict[str, Any] = {"dk": dk, "fusion_weight": fusion_weight}
        if wrapper.has_label:
            kwargs["label"] = label

        result_arrays = decryptor.decrypt_lst_ndarray_ct(
            dict_ct, **kwargs
        )
    else:
        # Single-input (SIFE): one client, simple list-of-arrays
        enc = next(iter(encrypted.values()))
        fw_list = next(iter(fusion_weight.values()))
        result_arrays = decryptor.decrypt_lst_ndarray_ct(
            enc.ciphertexts, dk=dk, fusion_weight=fw_list
        )

    return _aggregated(result_arrays, weight_scale)
