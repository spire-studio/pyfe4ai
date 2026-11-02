"""Federated aggregation with the ML adapter (``eta=1``, ndarray helpers):
:func:`encrypt_gradient` per client, :func:`aggregate_gradients` at the
aggregator, and their offline phases :func:`precompute_gradient` and
:func:`prepare_aggregation`. For ``mcfe/kp_ab_pairing`` they take the
``attributes`` of the clients and the ``policy`` of the key."""

from __future__ import annotations

import numbers
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np

from pyfe4ai._validation import int_vector
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError, FEWarning, warn_at_caller
from pyfe4ai.core.ids import as_id, id_list
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
    _check_gradient_owners,
    _client_keyed,
    _require_labelled_aggregation,
    _require_single_decryptor,
    _serves_threshold,
    _to_numpy,
)
from pyfe4ai.policy import Formula, attribute_set_digest, canonical_encoding, evaluate, from_canonical
from pyfe4ai.policy.attributes import clip
from pyfe4ai.policy.formula import leaf_count
from pyfe4ai.schemes.common.kp_ab import read_policy

#: What the aggregator of ``mcfe/kp_ab_pairing`` spends on one thread (see :func:`_warn_policy_cost`): to decode one
#: group element, for one pairing of a product of pairings, and for the discrete logarithm of one element. They come
#: from the aggregation of 1000 elements of 10 clients under a policy of 3 rows, for ciphertexts of 3 attributes, on one
#: thread of an Apple M2 Max (decoding 19.24 s of 290,000 elements, pairings 31.21 s of 281,000, discrete logarithms
#: 0.64 s of 1000), rounded.
_SECONDS_PER_DECODE = 66e-6
_SECONDS_PER_PAIRING = 111e-6
_SECONDS_PER_DLOG = 0.64e-3
#: More elements than this in one aggregation under a policy: a warning of the time it takes.
_WARN_ELEMENTS = 1000
#: A larger bound ``n * bound_x * bound_y`` of the discrete logarithm: a warning of its cost.
_WARN_BOUND = 2 ** 26


def _with_access_control(wrapper: FESchemeWrapper) -> bool:
    """Whether the helpers take ``attributes=`` and ``policy=`` for *wrapper*: for the labelled scheme with access
    control, ``mcfe/kp_ab_pairing``. The single-input attribute-based schemes (``sife/cp_ab_pairing``,
    ``sife/kp_ab_pairing``) have no ndarray helpers and keep failing as they did."""
    return wrapper.access_control and wrapper.has_label


def _access_argument(wrapper: FESchemeWrapper, name: str, value, required: bool = True) -> None:
    """Check that *value*, the ``attributes`` or the ``policy`` argument of a helper, is given for a scheme that takes
    it (unless it is not *required*) and for no other.

    Raises:
        ValueError: If a scheme with access control gets no *value* although it is *required*, or another scheme gets
            one.
    """
    scheme = "{}/{}".format(wrapper.scheme_type, wrapper.variant)
    if _with_access_control(wrapper):
        if required and value is None:
            raise ValueError("'{}' needs {}: pass {}=".format(
                scheme, {"attributes": "the attributes the gradient is encrypted for",
                         "policy": "the policy of the key"}[name], name))
    elif value is not None:
        raise ValueError("{}= is for 'mcfe/kp_ab_pairing', the attribute-based multi-client scheme, not for '{}'"
                         .format(name, scheme))


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
    nid = as_id(nid)  # a numpy integer is the int it equals
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
    attributes=None,
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
                client, label and shape (online phase); ``mcfe/kp_ab_pairing``
                has no offline phase (``ValueError``).
            attributes: ``mcfe/kp_ab_pairing`` only, and required there: the
                attributes (a collection of labels, see
                :mod:`pyfe4ai.policy.attributes`) the gradient is encrypted
                for, the same set for every client of a round.
                :func:`aggregate_gradients` releases the sum only for a
                policy that this set satisfies.

        Raises:
            ValueError: If *attributes* is missing for ``mcfe/kp_ab_pairing``
                or given for another scheme, or *nid* already encrypted under
                *label*.
    """
    _require_labelled_aggregation(wrapper, "encrypt_gradient")
    nid = as_id(nid)  # a numpy integer is the int it equals
    _access_argument(wrapper, "attributes", attributes)
    if precomputed is not None and _with_access_control(wrapper):
        raise ValueError("{}/{} has no offline encryption phase: precomputed= is for the schemes of "
                         "precompute_gradient".format(wrapper.scheme_type, wrapper.variant))
    arr = _to_numpy(gradient).astype(np.float64)
    encryptor = wrapper._get_encryptor(nid)

    kwargs: dict[str, Any] = {}
    if _with_access_control(wrapper):
        kwargs["attributes"] = attributes
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
    by the quantisation scale if any of them is fractional; keyed by client
    ids (numpy integer keys read as ints, other keys refused: ``ValueError``)."""
    weights = _client_keyed(weights, "weights")
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


def _aggregation_policy(wrapper: FESchemeWrapper, policy, prepared: PreparedAggregation | None) -> Formula | None:
    """The policy of an aggregation as a formula: ``None`` for a scheme without one.

    *policy* is a :class:`~pyfe4ai.policy.formula.Formula` or its text. It is required for ``mcfe/kp_ab_pairing``
    unless *prepared* holds the policy of its key, and then it may be left out or must be that policy.

    Raises:
        ValueError: As :func:`_access_argument`, and for a prepared key that is for another policy or for none.
        FEValidationError: For a policy that does not parse or has more leaves than the key generator allows rows.
    """
    _access_argument(wrapper, "policy", policy, required=prepared is None)
    if not _with_access_control(wrapper):
        return None
    formula = None if policy is None else read_policy(policy)
    if formula is not None and leaf_count(formula) > wrapper.kg.max_rows:
        raise FEValidationError("the policy has {} leaves, more than the limit max_rows = {}".format(
            leaf_count(formula), wrapper.kg.max_rows))
    if prepared is None:
        return formula
    if prepared.policy is None:
        raise ValueError("the prepared key has no policy: it is not a key of '{}/{}'".format(
            wrapper.scheme_type, wrapper.variant))
    if formula is not None and canonical_encoding(formula) != prepared.policy:
        raise ValueError("the prepared key is for another policy")
    return from_canonical(prepared.policy)


def _require_every_client(wrapper: FESchemeWrapper, allow_partial: bool, *, ciphertexts=None, weights=None) -> None:
    """Refuse, before a key is issued, an aggregation of ``mcfe/kp_ab_pairing`` that is not for every enrolled client.

    The shares ``p_i`` of the encryption keys add up to 1, so a result opens only with the ciphertexts of all the
    clients (also those of weight 0): ``allow_partial`` is refused, and so is a *ciphertexts* or *weights* dict (by
    client) that leaves a client out or names one that is not enrolled; the message names them.

    Raises:
        ValueError: As above.
    """
    scheme = "{}/{}".format(wrapper.scheme_type, wrapper.variant)
    if allow_partial:
        raise ValueError(
            "allow_partial=True is not supported by '{}': the shares p_1 + ... + p_n of the clients' keys add up to "
            "1, so a result opens only with the ciphertexts of every enrolled client".format(scheme))
    enrolled = list(wrapper.kg.lst_nid)
    if weights is not None:
        weights = _client_keyed(weights, "weights")
    for what, given in (("ciphertexts", ciphertexts), ("weights", weights)):
        if given is None:
            continue
        missing = [nid for nid in enrolled if nid not in given]
        unknown = id_list(nid for nid in given if nid not in enrolled)
        if missing or unknown:
            raise ValueError("the {} must cover every enrolled client of '{}' and no other: missing {}, not a client {}"
                             .format(what, scheme, clip(repr(missing), 120), clip(repr(unknown), 120)))


def _where(nid, layer: int, index: tuple) -> str:
    """The client, layer and element of a ciphertext record, for a message."""
    return "{}, layer {}, element {}".format(clip(repr(nid)), layer, index)


def _ciphertext_records(encrypted: dict):
    """The ciphertext records of the gradients *encrypted* (by client), as ``(nid, layer, index, record)``.

    Raises:
        FEValidationError: For a layer that is no array of ciphertexts (a packed layer, for one).
    """
    for nid, enc in encrypted.items():
        for layer, array in enumerate(enc.ciphertexts):
            if not isinstance(array, np.ndarray):
                raise FEValidationError("the ciphertexts of {}: layer {} is no array of ciphertexts, got {}".format(
                    clip(repr(nid)), layer, type(array).__name__))
            for index, record in np.ndenumerate(array):
                yield nid, layer, index, record


def _common_attributes(encrypted: dict, label) -> list | None:
    """The one set of attributes (a list of labels) that every ciphertext of *encrypted* is for, or ``None`` if there
    is no ciphertext.

    It is read from the ciphertext records: each element carries its ``attributes`` and the ``attributes_digest`` of
    them. The digests of all elements of all clients must be the same, and the attributes of the first element must be
    the ones its digest stands for. :func:`aggregate_gradients` has the scheme check every record before this
    (:func:`_check_ciphertext_arrays`), so that there this is a second look at records that are known to be well formed;
    called alone it sees the digests and the first record only.

    Raises:
        FEValidationError: For a record without a digest, ciphertexts that are for different sets of attributes (the
            message names the first element that differs from the first), and a first element whose attributes are no
            list of labels or are not those of its digest.
    """
    first = digest = None
    for nid, layer, index, record in _ciphertext_records(encrypted):
        found = record.get("attributes_digest") if isinstance(record, dict) else None
        if type(found) is not str:
            raise FEValidationError("the ciphertext of {} has no attributes_digest (a str)".format(
                _where(nid, layer, index)))
        if first is None:
            first, digest = (nid, layer, index, record), found
        elif found != digest:
            raise FEValidationError(
                "the ciphertexts of label {} are not all for one set of attributes: the ciphertext of {} is for "
                "another set than that of {}".format(clip(repr(label)), _where(nid, layer, index),
                                                      _where(*first[:3])))
    if first is None:
        return None
    where, record = _where(*first[:3]), first[3]
    attributes = record.get("attributes")
    try:
        if type(attributes) is not list:
            raise FEValidationError("attributes must be a list of labels, got {}".format(type(attributes).__name__))
        matches = attribute_set_digest(attributes) == digest
    except FEValidationError as exc:
        raise FEValidationError("the attributes of the ciphertext of {}: {}".format(where, exc)) from exc
    if not matches:
        raise FEValidationError("the attributes_digest of the ciphertext of {} is not the digest of its attributes"
                                .format(where))
    return attributes


def _check_ciphertext_arrays(wrapper: FESchemeWrapper, encrypted: dict, label, sid) -> None:
    """Refuse, before a key is issued, ciphertexts that the scheme would refuse before it decodes a point.

    The scheme does it
    (:meth:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing.check_ndarray_ct`, through the decryptor of *sid*): the
    structure of the arrays, the bindings, fields, version markers, clients, tags, digests and lengths of every record,
    each element under its label, one set of attributes for all. Its ``FEValidationError`` (a ``ValueError``) is raised
    as it is, naming the layer and the element (the client and the layer for a wrong shape, the clients and the counts
    for other numbers of layers), like the other refusals of the records here. What it cannot see is in
    :meth:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing.check_ndarray_ct`: the ``label`` and the attributes are
    claims of the clients.

    Raises:
        ValueError: Without a *label* (as :func:`_aggregation_key`).
        FEValidationError: As above.
    """
    if label is None:
        raise ValueError("MCFE schemes require a label for decryption")
    wrapper._get_decryptor(sid).check_ndarray_ct({nid: enc.ciphertexts for nid, enc in encrypted.items()}, label=label)


def _check_policy_aggregation(encrypted: dict, policy, label) -> None:
    """Refuse an aggregation under *policy* before a key is issued or anything is paired: the ciphertexts of
    *encrypted* (by client, the gradients of round *label*) must all be for one set of attributes, and that set must
    satisfy *policy* (a formula or its text).

    The attributes are read from the ciphertext records (:func:`_common_attributes`); neither
    :class:`~pyfe4ai.ml.serialization.EncryptedGradient` nor its wire form carries anything else.

    Raises:
        FEValidationError: For ciphertexts that are for different sets of attributes, or records that are malformed.
        FEKeyError: If the set of attributes does not satisfy the policy.
    """
    attributes = _common_attributes(encrypted, label)
    if attributes is None:
        return          # no element: nothing is decrypted
    if not evaluate(read_policy(policy), attributes):
        raise FEKeyError("the attributes {} of the ciphertexts of label {} do not satisfy the policy of the key"
                         .format(clip(repr(attributes), 120), clip(repr(label))))


def _policy_job(wrapper: FESchemeWrapper, encrypted: dict, formula: Formula) -> tuple:
    """The size of the aggregation under *formula* of the gradients *encrypted*, for :func:`_warn_policy_cost`:
    ``(elements, clients, rows, bound, attributes)``. The ciphertexts have passed :func:`_check_policy_aggregation`.

    ``rows`` is the number of leaves of the policy that the attributes hold: at most that many rows of the policy are
    used for each element (an ``or`` uses one of its branches). ``bound`` is ``n * bound_x * bound_y``: ``bound_x``
    bounds the quantised integers, the gradient entries times ``10 ** precision``. ``attributes`` is the number of
    attributes the ciphertexts are for.
    """
    elements = sum(array.size for array in next(iter(encrypted.values())).ciphertexts)
    record = next((record for *_, record in _ciphertext_records(encrypted)), None)
    held = set(record["attributes"]) if record is not None else set()
    rows = sum(label in held for label in formula.leaves())
    kg = wrapper.kg
    return elements, len(encrypted), rows, kg.n * kg.bound_x * kg.bound_y, len(held)


def _warn_policy_cost(elements: int, clients: int, rows: int, bound: int, attributes: int | None = None) -> None:
    """Warn (:class:`~pyfe4ai.core.exceptions.FEWarning`) when an aggregation under a policy is costly.

    More than 1000 elements: for each element the aggregator decodes ``clients * (8 * attributes + 5)`` group elements
    and pairs ``clients * (8 * rows + 4) + 1``, and takes one discrete logarithm. The warning gives the time that this
    takes on one thread, at :data:`_SECONDS_PER_DECODE`, :data:`_SECONDS_PER_PAIRING` and :data:`_SECONDS_PER_DLOG`;
    the scheme is meant for small vectors and statistics. A bound of the discrete logarithm above ``2**26``: its table
    and its search grow with the square root of the bound. Neither refuses the call.

    Args:
        elements: The number of elements of the gradients.
        clients: The number of clients, all of which take part.
        rows: The number of rows of the policy used for each element.
        bound: ``n * bound_x * bound_y``, the range of the discrete logarithm.
        attributes: The number of attributes the ciphertexts are for; every ciphertext holds a vector of ``G1`` for
            each, which is decoded whether the policy uses it or not. ``None``: as many as *rows*.
    """
    if attributes is None:
        attributes = rows
    if elements > _WARN_ELEMENTS:
        decodes = elements * clients * (8 * attributes + 5)
        pairs = elements * (clients * (8 * rows + 4) + 1)
        seconds = decodes * _SECONDS_PER_DECODE + pairs * _SECONDS_PER_PAIRING + elements * _SECONDS_PER_DLOG
        warn_at_caller(
            "{} elements under a policy: the aggregation of {} clients decodes {} group elements and pairs {}, about "
            "{:.0f} s on one thread (less with more threads; the ciphertexts are for {} attributes, and at most {} "
            "rows of the policy are used for each element). mcfe/kp_ab_pairing is meant for small vectors and "
            "statistics of up to {} elements".format(elements, clients, decodes, pairs, seconds, attributes, rows,
                                                      _WARN_ELEMENTS),
            FEWarning)
    if bound > _WARN_BOUND:
        warn_at_caller(
            "the discrete-logarithm bound n * bound_x * bound_y is {}, above 2^26: the table and the search of the "
            "discrete logarithm of each element grow with its square root (at 2^32 the table takes seconds and each "
            "element tens of milliseconds); lower bound_x or bound_y, which bound the quantised entries and the "
            "weights, or the precision".format(bound),
            FEWarning)


def _aggregation_key(wrapper, weights, label, sid, nids, allow_partial, n_chunks_fn, servers=None, policy=None):
    """Policy checks of :func:`aggregate_gradients` and the functional key:
    ``(fusion_weight, weight_scale, dk)``; *nids* are the clients whose
    ciphertexts will be aggregated, ``n_chunks_fn()`` the number of packed
    chunks a key must cover (``None``: not a packed scheme). For a threshold
    MCFE run by the wrapper ``dk`` holds the key shares of the enrolled
    *servers* (:func:`_server_key_shares`). For ``mcfe/kp_ab_pairing`` the key
    is for the *policy* (a formula)."""
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
            raise ValueError("aggregation needs non-zero weights{}".format(
                "" if policy is not None else " (allow_partial=True to override)"))  # a policy has no override
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for decryption")
        _label_weights(wrapper, label, fusion_weight, tie=False)  # other weights: refused before a key is issued

    # Build credentials for dk generation
    # SIFE expects fusion_weight as a flat list; MCFE/MIFE expect a dict.
    if wrapper.scheme_type in ("mcfe", "mife"):
        credentials: dict[str, Any] = {"fusion_weight": fusion_weight}
    else:
        # Single-input: take the first (only) client's weight list
        credentials = {"fusion_weight": next(iter(fusion_weight.values()))}
    if wrapper.has_label:
        credentials["label"] = label
    if policy is not None:
        credentials["policy"] = policy
    n_chunks = n_chunks_fn()
    if n_chunks is not None:
        # identity-based schemes issue one key term per packed chunk
        credentials["n_chunks"] = n_chunks
    if servers is not None:
        dk = _server_key_shares(wrapper, credentials, servers)
    else:
        dk = wrapper.kg.get_decryption_keys(sid, credentials=credentials)
    if wrapper.has_label:
        _label_weights(wrapper, label, fusion_weight, tie=True)
    return fusion_weight, weight_scale, dk


def _label_weights(wrapper: FESchemeWrapper, label, fusion_weight: dict, *, tie: bool) -> None:
    """Refuse *fusion_weight* for *label* (``ValueError``) if the label is tied to other weights; with *tie*, tie it
    to *fusion_weight* if it is free.

    :func:`_aggregation_key` checks before it asks for a key, and ties once the key is issued: a request that the key
    generator refuses (weights beyond ``bound_y``, a policy it does not take) leaves the label free. Of two calls with
    other weights that pass the check at once, the first to tie keeps the label and the other is refused here, before
    it returns its key; on the threshold path the losing call has already obtained key shares from the servers by then
    (its weight key share is added to the wrapper's server objects where they lack it, in ``_serving``), and these
    shares never reach the caller.
    """
    if tie:
        previous = wrapper._aggregated.setdefault(label, fusion_weight)
    else:
        previous = wrapper._aggregated.get(label, fusion_weight)
    if previous != fusion_weight:
        raise ValueError(
            "label {!r} was already aggregated with other weights".format(label)
        )


def prepare_aggregation(
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list],
    *,
    shape,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
    servers: list | None = None,
    policy=None,
) -> PreparedAggregation:
    """Derive the functional key of an aggregation before the ciphertexts arrive.

    The key depends on the weights, the label and (for the packed Ring-LWE
    MCFE) the number of chunks of a gradient of *shape*, not on the
    ciphertexts; :func:`aggregate_gradients` with ``prepared=`` then only
    decrypts. The same policy checks apply, with the clients of *weights* as
    those whose ciphertexts will be aggregated.

    For ``mcfe/kp_ab_pairing`` the key is for *policy*, a
    :class:`~pyfe4ai.policy.formula.Formula` or its text, which is required
    there and refused for another scheme (``ValueError``); the result keeps its
    canonical encoding in ``policy``. The attributes of the ciphertexts are not
    known yet: they are judged against the policy when the round is
    aggregated. ``allow_partial=True`` is refused for this scheme, and so is
    a *weights* without every enrolled client.

    For ``mcfe/ring_lwe_threshold`` the key shares of the enrolled *servers*
    (default: the first ``t`` of the public parameters' ``lst_sid``): their
    label key shares of the round, each server given its weight key share
    once per weights (again where a copy of the wrapper or the server's store
    lacks it when the round is decrypted); *sid* is the combiner. The result
    holds key material (see :class:`PreparedAggregation`). *weights* is keyed
    by client ids (numpy integer keys are read as ints; another key:
    ``ValueError``).
    """
    _require_labelled_aggregation(wrapper, "prepare_aggregation")
    _require_single_decryptor(wrapper, "prepare_aggregation", served_threshold=True)
    if servers is not None and not _serves_threshold(wrapper):
        raise ValueError("servers are those of a threshold MCFE")
    formula = _aggregation_policy(wrapper, policy, None)
    if formula is not None:
        _require_every_client(wrapper, allow_partial, weights=weights)
    enrolled = _enrolled_servers(wrapper, servers) if _serves_threshold(wrapper) else None
    decryptor = wrapper._get_decryptor(sid)
    n_chunks = None
    if getattr(decryptor, "key_chunks", None) is not None:
        ring_n = int(wrapper.pp["ring_n"])
        size = int(np.prod(np.shape(np.empty(shape)), dtype=np.int64))
        n_chunks = max(1, -(-size // ring_n))
    weights = _client_keyed(weights, "weights")
    fusion_weight, weight_scale, dk = _aggregation_key(
        wrapper, weights, label, sid, list(weights), allow_partial, lambda: n_chunks, enrolled, formula
    )
    prepare_key = getattr(decryptor, "prepare_key", None)
    if prepare_key is not None:  # element-wise MCFE: the label terms of the key too
        dk = prepare_key(dk, label, [tuple(int(d) for d in np.shape(np.empty(shape)))])
    return PreparedAggregation(label, sid, fusion_weight, weight_scale, n_chunks, dk, allow_partial, enrolled,
                               None if formula is None else canonical_encoding(formula))


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
    policy=None,
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
                weight vector only, the first one a key is issued for (a
                request the key generator refuses leaves the label free).
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
            policy: ``mcfe/kp_ab_pairing`` only, and required there (unless
                *prepared* holds it): the policy of the key, a
                :class:`~pyfe4ai.policy.formula.Formula` or its text
                (:func:`~pyfe4ai.policy.encoding.parse`). The attributes the
                clients encrypted for (``encrypt_gradient(...,
                attributes=)``) must be one set that satisfies it. Before a
                key is issued or anything is paired, the aggregation is
                refused (``FEValidationError``, a ``ValueError``) if the
                scheme's checks of the ciphertexts refuse them (malformed
                records, other parameters or label, other shapes, clients
                that differ in their attributes), or (``FEKeyError``) if
                the set does not satisfy the policy, and it warns
                (:class:`~pyfe4ai.core.exceptions.FEWarning`) of a job of
                more than 1000 elements or a discrete-logarithm bound
                ``n * bound_x * bound_y`` above ``2**26``. The scheme needs
                the ciphertexts of every enrolled client, so *allow_partial*
                and a missing client are refused (``ValueError``). The
                ``label`` and the attributes of a ciphertext are claims of its
                client that the checks cannot verify: ciphertexts of an
                earlier round whose label was rewritten (by anyone who holds
                the ciphertexts of that round, with no key) are aggregated,
                silently, to the value of the round they were made in, so
                have the clients sign or MAC the label with the record where
                that matters.

        Raises:
            ValueError: If the policy above is violated, SIFE is given more
                than one client, *encrypted* or *weights* is keyed by a
                value that is no client id (numpy integers are read as ints;
                see :mod:`pyfe4ai.core.ids`), a gradient is filed under
                another client than its ``nid``, *policy* is missing for
                ``mcfe/kp_ab_pairing`` or given for another scheme or other
                than the policy of *prepared*.
            FEKeyError: If the attributes of the clients do not satisfy
                *policy*.
    """
    _require_labelled_aggregation(wrapper, "aggregate_gradients")
    _require_single_decryptor(wrapper, "aggregate_gradients", served_threshold=True)
    encrypted = _client_keyed(encrypted, "encrypted gradients")
    if wrapper.scheme_type != "sife":  # one ciphertext per client: each is decrypted as the client of its key
        _check_gradient_owners(encrypted)
    threshold = _serves_threshold(wrapper)
    if servers is not None and not threshold:
        raise ValueError("servers are those of a threshold MCFE")
    formula = _aggregation_policy(wrapper, policy, prepared)
    if formula is not None:  # refuse before a key is issued or anything is paired
        _require_every_client(wrapper, allow_partial, ciphertexts=encrypted, weights=weights)
        _check_ciphertext_arrays(wrapper, encrypted, label, prepared.sid if prepared is not None else sid)
        _check_policy_aggregation(encrypted, formula, label)
        _warn_policy_cost(*_policy_job(wrapper, encrypted, formula))
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
            wrapper, weights, label, sid, list(encrypted), allow_partial, n_chunks_fn, enrolled, formula
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
            raise ValueError("the prepared key is for the clients {}".format(id_list(prepared.fusion_weight)))
        if n_chunks is not None and (prepared.n_chunks or 0) < n_chunks:
            raise ValueError("the prepared key covers fewer chunks than the ciphertexts")
        if threshold and servers is not None and list(servers) != prepared.servers:
            raise ValueError("the prepared key shares are for the servers {}".format(prepared.servers))
        fusion_weight, weight_scale, dk = prepared.fusion_weight, prepared.weight_scale, prepared.dk

    if threshold:  # the servers the wrapper runs, then the combiner
        from pyfe4ai.core.versions import MixedSharingsError

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
