"""The ML adapter with the attribute-based multi-client scheme (``mcfe/kp_ab_pairing``): policy-gated aggregation.

``encrypt_gradient(..., attributes=)`` encrypts each element for a set of attributes, and
``aggregate_gradients(..., policy=)`` and ``prepare_aggregation(..., policy=)`` issue a key for a policy. The adapter
refuses, before a key is issued and before any pairing, a request whose clients did not all encrypt for one set of
attributes, whose attributes do not satisfy the policy, or that leaves a client out (``allow_partial`` included); it
warns of the cost of a large job.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.mcfe.conftest import refuse_to_compute
from pyfe4ai.core.exceptions import FEKeyError, FESecurityWarning, FEValidationError, FEWarning
from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, PreparedAggregation
from pyfe4ai.ml import aggregate_gradients, encrypt_gradient, precompute_gradient, prepare_aggregation
from pyfe4ai.ml import gradients as ml_gradients
from pyfe4ai.policy import attribute_set_digest, canonical_encoding, parse
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, pairing_available

ROOT = Path(__file__).resolve().parents[2]

NIDS = ["c0", "c1", "c2"]
WEIGHTS = {"c0": 1, "c1": 2, "c2": 3}
ATTRIBUTES = ["purpose:research", "region:EU"]
POLICY = "and(purpose:research, region:EU)"
LABEL = "round-1"
SHAPE = (2, 2)


@pytest.fixture(autouse=True)
def _own_parameter_cache(tmp_path, monkeypatch):
    """The limits of these wrappers differ from those of other tests: a parameter record of their own."""
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


def make_wrapper_of(variant, **overrides):
    config = {"n": 3, "lst_nid": NIDS, "bound_x": 1000, "bound_y": 10, "pairing_group_param": GROUP}
    config.update(overrides)
    return FESchemeWrapper("mcfe", variant, config, precision=1)


def make_wrapper(**overrides):
    return make_wrapper_of("kp_ab_pairing", **overrides)


@pytest.fixture
def wrapper():
    return make_wrapper()


def gradients(seed=0, exact=True):
    """A 2 x 2 gradient of every client: multiples of 0.1 (exact at precision 1), or any value in [-5, 5]."""
    rng = np.random.default_rng(seed)
    values = {nid: rng.uniform(-5, 5, SHAPE) for nid in NIDS}
    return {nid: np.round(value, 1) if exact else value for nid, value in values.items()}


def encrypt_all(wrapper, grads, label=LABEL, attributes=ATTRIBUTES):
    return {nid: encrypt_gradient(grad, wrapper, nid=nid, label=label, attributes=attributes)
            for nid, grad in grads.items()}


def weighted(grads, weights=WEIGHTS):
    return sum(weights[nid] * grads[nid] for nid in weights)


def no_key(wrapper, monkeypatch):
    """Make the issue of a functional key, and any decoding, pairing or discrete log, fail the test."""
    def issued(*args, **kwargs):
        pytest.fail("a functional key was issued")

    monkeypatch.setattr(wrapper.kg, "get_decryption_keys", issued)
    refuse_to_compute(monkeypatch)


# ---------------------------------------------------------------------------
# what the adapter computes


@pytest.mark.parametrize("policy", [POLICY, parse(POLICY)], ids=["text", "formula"])
def test_matches_the_plaintext_aggregate(wrapper, policy):
    grads = gradients()
    encrypted = encrypt_all(wrapper, grads)
    total = aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=policy)
    assert total.shape == SHAPE
    np.testing.assert_allclose(total, weighted(grads), atol=1e-9)


def test_the_error_is_within_the_quantisation_step(wrapper):
    grads = gradients(seed=3, exact=False)
    total = aggregate_gradients(encrypt_all(wrapper, grads), wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    step = 0.5 * 10 ** -1 * sum(WEIGHTS.values())         # each entry is rounded to 0.1, and weighted
    assert np.abs(total - weighted(grads)).max() <= step + 1e-9
    assert np.abs(total - weighted(grads)).max() > 0     # the values were not multiples of 0.1


@pytest.mark.parametrize("policy,attributes", [
    ("or(region:EU, region:US)", ["region:US"]),
    ("2of(purpose:research, region:EU, org:A)", ["org:A", "region:EU"]),
    ("or(and(region:EU, org:A), and(region:EU, org:B))", ["org:B", "region:EU"]),
])
def test_other_policies(wrapper, policy, attributes):
    grads = gradients(seed=1)
    encrypted = encrypt_all(wrapper, grads, attributes=attributes)
    total = aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=policy)
    np.testing.assert_allclose(total, weighted(grads), atol=1e-9)


def test_attribute_sets_written_differently_agree(wrapper):         # Review Focus 1, through the adapter
    grads = gradients()
    spellings = {"c0": {"region:EU", "purpose:research"},
                 "c1": ["region:EU", "purpose:research", "region:EU"],
                 "c2": ("purpose:research", "region:EU")}
    encrypted = {nid: encrypt_gradient(grads[nid], wrapper, nid=nid, label=LABEL, attributes=spellings[nid])
                 for nid in NIDS}
    digests = {ct["attributes_digest"] for enc in encrypted.values() for ct in enc.ciphertexts[0].flat}
    assert digests == {attribute_set_digest(ATTRIBUTES)}
    total = aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(grads), atol=1e-9)


def test_fractional_weights(wrapper):
    grads = gradients()
    weights = {"c0": 0.5, "c1": 0.9, "c2": 1.0}             # 5, 9 and 10 at the precision of 1: within bound_y = 10
    total = aggregate_gradients(encrypt_all(wrapper, grads), wrapper, weights, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(grads, weights), atol=1e-9)


def test_serialised_gradients_round_trip(wrapper):
    grads = gradients()
    sent = [json.loads(json.dumps(enc.to_dict())) for enc in encrypt_all(wrapper, grads).values()]
    received = {wire["nid"]: EncryptedGradient.from_dict(wire) for wire in sent}
    total = aggregate_gradients(received, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(grads), atol=1e-9)


def plain_wrapper():
    """A labelled multi-client scheme without access control."""
    return FESchemeWrapper("mcfe", "ddh", {"sec_param": 128, "eta": 1, "n": 2, "lst_nid": ["c0", "c1"]}, precision=1)


def test_the_wrapper_knows_its_capability():
    assert make_wrapper().access_control is True
    assert plain_wrapper().access_control is False
    assert FESchemeWrapper("sife", "ddh", {"eta": 2}).access_control is False


def test_a_pickled_wrapper_without_the_capability_reads_it_as_absent(wrapper):
    state = wrapper.__getstate__()
    del state["access_control"]                                  # a wrapper pickled before the attribute existed
    old = object.__new__(FESchemeWrapper)
    old.__setstate__(state)
    assert old.access_control is False
    state["access_control"] = True
    new = object.__new__(FESchemeWrapper)
    new.__setstate__(state)
    assert new.access_control is True


# ---------------------------------------------------------------------------
# the attributes and the policy are required where the scheme has them, and refused elsewhere


def test_attributes_are_required(wrapper):
    with pytest.raises(ValueError, match="needs the attributes"):
        encrypt_gradient(gradients()["c0"], wrapper, nid="c0", label=LABEL)
    # the refusal did not use up the label of the client
    encrypt_gradient(gradients()["c0"], wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES)


def test_policy_is_required(wrapper, monkeypatch):
    encrypted = encrypt_all(wrapper, gradients())
    no_key(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="needs the policy"):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL)
    with pytest.raises(ValueError, match="needs the policy"):
        prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL)


def test_a_policy_that_does_not_parse_is_refused(wrapper, monkeypatch):
    encrypted = encrypt_all(wrapper, gradients())
    no_key(wrapper, monkeypatch)
    with pytest.raises(FEValidationError):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy="and(purpose:research")
    with pytest.raises(FEValidationError, match="Formula or its text"):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=["purpose:research"])


def test_other_schemes_refuse_them(monkeypatch):
    plain = plain_wrapper()
    grad = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="attributes= is for"):
        encrypt_gradient(grad, plain, nid="c0", label=LABEL, attributes=ATTRIBUTES)
    encrypted = {nid: encrypt_gradient(grad, plain, nid=nid, label=LABEL) for nid in ("c0", "c1")}   # label not used up

    def issued(*args, **kwargs):
        pytest.fail("a functional key was issued")

    monkeypatch.setattr(plain.kg, "get_decryption_keys", issued)
    with pytest.raises(ValueError, match="policy= is for"):
        aggregate_gradients(encrypted, plain, {"c0": 1, "c1": 1}, label=LABEL, policy=POLICY)
    with pytest.raises(ValueError, match="policy= is for"):
        prepare_aggregation(plain, {"c0": 1, "c1": 1}, shape=(2,), label=LABEL, policy=POLICY)


def test_the_single_input_attribute_schemes_keep_their_refusal():
    """``access_control`` is also the capability of ``sife/cp_ab_pairing`` and ``sife/kp_ab_pairing``, which have no
    ndarray helpers: their path through the adapter is the one it had."""
    cfg = {"sec_param": 64, "bound_x": 8, "bound_y": 8, "pairing_group_param": GROUP}
    wrapper = FESchemeWrapper("sife", "kp_ab_pairing", cfg)
    assert wrapper.access_control is True
    with pytest.raises(NotImplementedError, match="ndarray helper is not implemented"):
        encrypt_gradient(np.array([1.0]), wrapper, nid="nid_default")
    with pytest.raises(ValueError, match="attributes= is for"):
        encrypt_gradient(np.array([1.0]), wrapper, nid="nid_default", attributes=["a:1"])


def test_the_per_client_scheme_is_refused_by_every_entry_point():
    """The per-client scheme of NPP22 Sec. 5.4 lets the clients of one tag hold different sets of attributes, and the
    adapter aggregates one set. The scheme is registered with the capability ``access_control``, and the adapter
    names it in its refusal at the first line of the four aggregation entry points, before it encrypts or issues a
    key."""
    with pytest.warns(FESecurityWarning, match="mcfe/kp_ab_pairing_per_client is known to be insecure"):
        wrapper = make_wrapper_of("kp_ab_pairing_per_client")        # the adapter does not hide the status
    assert wrapper.access_control is True
    grads = gradients()
    match = "kp_ab_pairing_per_client.*different sets of attributes"
    with pytest.raises(NotImplementedError, match=match):
        encrypt_gradient(grads["c0"], wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES)
    with pytest.raises(NotImplementedError, match=match):
        precompute_gradient(SHAPE, wrapper, nid="c0", label=LABEL)
    with pytest.raises(NotImplementedError, match=match):
        prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL, policy=POLICY)
    with pytest.raises(NotImplementedError, match=match):
        aggregate_gradients({}, wrapper, WEIGHTS, label=LABEL, policy=POLICY)


def test_precompute_has_no_offline_phase_here(wrapper):
    with pytest.raises(ValueError, match="no offline encryption phase"):
        precompute_gradient(SHAPE, wrapper, nid="c0", label=LABEL)


def test_a_precomputation_is_refused_for_a_scheme_without_an_offline_phase(wrapper):
    with pytest.raises(ValueError, match="no offline encryption phase"):
        encrypt_gradient(gradients()["c0"], wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES,
                         precomputed=object())
    encrypt_gradient(gradients()["c0"], wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES)   # label not used up


# ---------------------------------------------------------------------------
# one ciphertext of a client under a label (Review Focus 5)


def test_one_label_once(wrapper):
    grad = gradients()["c0"]
    encrypt_gradient(grad, wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES)
    with pytest.raises(ValueError, match="already encrypted under label"):
        encrypt_gradient(grad, wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES)
    with pytest.raises(ValueError, match="already encrypted under label"):             # other attributes do not help
        encrypt_gradient(grad, wrapper, nid="c0", label=LABEL, attributes=["region:US"])
    encrypt_gradient(grad, wrapper, nid="c0", label="round-2", attributes=ATTRIBUTES)   # another label is fine
    encrypt_gradient(grad, wrapper, nid="c1", label=LABEL, attributes=ATTRIBUTES)       # so is another client


def test_a_failed_encryption_releases_its_label(wrapper):
    grad = gradients()["c0"]
    with pytest.raises(FEValidationError):                   # not a label of the scheme: the scheme refuses it
        encrypt_gradient(grad, wrapper, nid="c0", label=LABEL, attributes=["no label like this"])
    encrypt_gradient(grad, wrapper, nid="c0", label=LABEL, attributes=ATTRIBUTES)


# ---------------------------------------------------------------------------
# the checks before a key is issued


def test_unsatisfied_is_refused_before_key_or_pairing(wrapper, monkeypatch):
    encrypted = encrypt_all(wrapper, gradients())
    no_key(wrapper, monkeypatch)
    with pytest.raises(FEKeyError, match="do not satisfy"):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL,
                            policy="and(purpose:research, region:EU, role:auditor)")
    assert LABEL not in wrapper._aggregated                  # the refusal did not tie the label to these weights


def test_differing_attributes_are_refused_first(wrapper, monkeypatch):
    grads = gradients()
    encrypted = {nid: encrypt_gradient(grads[nid], wrapper, nid=nid, label=LABEL,
                                       attributes=["purpose:research"] if nid == "c1" else ATTRIBUTES)
                 for nid in NIDS}
    no_key(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="c1") as raised:
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    assert not isinstance(raised.value, FEKeyError)          # not read as "the attributes do not satisfy"
    assert LABEL not in wrapper._aggregated


def test_allow_partial_is_refused(wrapper, monkeypatch):    # Review Focus 2
    encrypted = encrypt_all(wrapper, gradients())
    no_key(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="allow_partial.*mcfe/kp_ab_pairing.*every enrolled client"):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=POLICY, allow_partial=True)
    with pytest.raises(ValueError, match="allow_partial.*mcfe/kp_ab_pairing.*every enrolled client"):
        prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL, policy=POLICY, allow_partial=True)


def test_missing_client_is_refused(wrapper, monkeypatch):   # Review Focus 2
    grads = gradients()
    encrypted = encrypt_all(wrapper, grads)
    without = {nid: encrypted[nid] for nid in ("c0", "c1")}
    no_key(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="ciphertexts.*missing.*c2"):
        aggregate_gradients(without, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    with pytest.raises(ValueError, match="weights.*missing.*c2"):
        aggregate_gradients(encrypted, wrapper, {"c0": 1, "c1": 2}, label=LABEL, policy=POLICY)
    with pytest.raises(ValueError, match="weights.*missing.*c2"):
        prepare_aggregation(wrapper, {"c0": 1, "c1": 2}, shape=SHAPE, label=LABEL, policy=POLICY)
    # a client of weight 0 that sent nothing is missing all the same
    with pytest.raises(ValueError, match="ciphertexts.*missing.*c2"):
        aggregate_gradients(without, wrapper, {"c0": 1, "c1": 2, "c2": 0}, label=LABEL, policy=POLICY)
    stranger = dataclasses.replace(encrypted["c0"], nid="other")
    with pytest.raises(ValueError, match="not a client.*other"):
        aggregate_gradients(dict(encrypted, other=stranger), wrapper, WEIGHTS, label=LABEL, policy=POLICY)


def test_a_zero_weight_is_not_offered_the_override(wrapper, monkeypatch):
    encrypted = encrypt_all(wrapper, gradients())
    no_key(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="non-zero weights") as raised:
        aggregate_gradients(encrypted, wrapper, {"c0": 1, "c1": 0, "c2": 3}, label=LABEL, policy=POLICY)
    assert "allow_partial" not in str(raised.value)          # this scheme has no such override


# ---------------------------------------------------------------------------
# a request that the scheme would refuse is refused before a key is issued: the scheme's own checks of the
# ciphertexts (MCFEKPABPairing.check_ndarray_ct), not the adapter's copy of them


def spy_on_the_key(wrapper, monkeypatch):
    """The calls to the issue of a functional key (it still issues them)."""
    calls = []
    issue = wrapper.kg.get_decryption_keys

    def spy(*args, **kwargs):
        calls.append(args)
        return issue(*args, **kwargs)

    monkeypatch.setattr(wrapper.kg, "get_decryption_keys", spy)
    return calls


def change_element(client, layer, index, **fields):
    """A change of the gradients that alters the fields of the ciphertext record of one element."""
    def change(encrypted):
        array = encrypted[client].ciphertexts[layer]
        array[index] = dict(array[index], **fields)
    return change


def drop_field(client, layer, index, field):
    def change(encrypted):
        array = encrypted[client].ciphertexts[layer]
        array[index] = {name: value for name, value in array[index].items() if name != field}
    return change


def reshape_layer(encrypted):
    encrypted["c1"].ciphertexts[0] = encrypted["c1"].ciphertexts[0].reshape(4)           # (2, 2) for the others


def add_a_layer(encrypted):
    encrypted["c1"].ciphertexts.append(encrypted["c1"].ciphertexts[0].copy())            # two layers for c1 only


def short_attribute_vectors(encrypted):
    array = encrypted["c2"].ciphertexts[0]
    array[1, 0] = dict(array[1, 0], c_attr=array[1, 0]["c_attr"][:1])


SCHEME_REFUSES = [
    ("a non-first attributes list that its digest does not stand for",
     change_element("c2", 0, (1, 0), attributes=["region:EU"])),
    ("a wrong label field", change_element("c1", 0, (0, 1), label="round-0")),
    ("a swapped nid", change_element("c1", 0, (1, 1), nid="c0")),
    ("a wrong pp_binding", change_element("c1", 0, (0, 0), pp_binding="0" * 64)),
    ("a wrong tag_hash", change_element("c2", 0, (1, 1), tag_hash="v9")),
    ("a missing c_attr", drop_field("c1", 0, (0, 0), "c_attr")),
    ("a short c_attr", short_attribute_vectors),
    ("a malformed t", change_element("c0", 0, (1, 0), t=7)),
    ("a malformed c_ipfe", change_element("c0", 0, (0, 1), c_ipfe=["1:x"] * 3)),
    ("an array of another shape", reshape_layer),
    ("another number of layers", add_a_layer),
]


@pytest.mark.parametrize("change", [change for _, change in SCHEME_REFUSES], ids=[name for name, _ in SCHEME_REFUSES])
def test_what_the_scheme_refuses_is_refused_before_a_key_is_issued(change, wrapper, monkeypatch):
    grads = gradients()
    encrypted = encrypt_all(wrapper, grads)
    tampered = copy.deepcopy(encrypted)
    change(tampered)
    calls = spy_on_the_key(wrapper, monkeypatch)
    with pytest.raises(FEValidationError):
        aggregate_gradients(tampered, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    assert calls == []                                       # no key was issued
    assert LABEL not in wrapper._aggregated                  # and the label is not tied to the weights
    other = {"c0": 2, "c1": 1, "c2": 1}                      # so the round can be run again, with other weights
    total = aggregate_gradients(encrypted, wrapper, other, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(grads, other), atol=1e-9)
    assert len(calls) == 1


def test_ciphertexts_of_a_previous_round_are_refused_before_a_key_is_issued(wrapper, monkeypatch):
    grads = gradients()
    previous = encrypt_all(wrapper, grads, label="round-0")
    encrypted = encrypt_all(wrapper, grads)
    calls = spy_on_the_key(wrapper, monkeypatch)
    with pytest.raises(FEValidationError, match="claim the label"):
        aggregate_gradients(previous, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    assert calls == [] and LABEL not in wrapper._aggregated
    total = aggregate_gradients(encrypted, wrapper, {"c0": 2, "c1": 1, "c2": 1}, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(grads, {"c0": 2, "c1": 1, "c2": 1}), atol=1e-9)


def test_a_prepared_key_is_checked_by_the_scheme_too(wrapper, monkeypatch):
    grads = gradients()
    encrypted = encrypt_all(wrapper, grads)
    prepared = prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL, policy=POLICY)
    tampered = copy.deepcopy(encrypted)
    change_element("c1", 0, (0, 1), pp_binding="0" * 64)(tampered)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEValidationError, match="pp_binding"):
        aggregate_gradients(tampered, wrapper, label=LABEL, prepared=prepared)


def test_a_missing_label_is_refused_as_before(wrapper, monkeypatch):
    encrypted = encrypt_all(wrapper, gradients())
    no_key(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="require a label"):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, policy=POLICY)


def test_what_the_scheme_cannot_see_before_a_decode_is_found_after_the_key(wrapper, monkeypatch):
    """The limit of the gate: an item with the type tag of a group element that is no element is found when its slice
    is decoded, after the key is issued; the call raises and the label stays tied to the weights."""
    encrypted = encrypt_all(wrapper, gradients())
    change_element("c1", 0, (0, 0), t="1:not an element")(encrypted)
    calls = spy_on_the_key(wrapper, monkeypatch)
    with pytest.raises(FEValidationError, match=r"t of c1, layer 0, element \(0, 0\)"):
        aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    assert len(calls) == 1 and LABEL in wrapper._aggregated


def relabel(encrypted, source):
    """The gradients *encrypted*, with the ``label`` field of every element of every client copied from the same
    element of *source* (the gradients of the same shape under another label)."""
    relabelled = copy.deepcopy(encrypted)
    for nid, enc in relabelled.items():
        for layer, array in enumerate(enc.ciphertexts):
            for index in np.ndindex(array.shape):
                array[index] = dict(array[index], label=source[nid].ciphertexts[layer][index]["label"])
    return relabelled


def test_a_relabelled_replay_by_all_clients_decrypts_to_the_sum_of_the_round_the_ciphertexts_were_made_in(
        wrapper, monkeypatch):
    """A stated limit, not a defence. The ``label`` of a ciphertext is a claim of its client: the points it was hashed
    to are inside ``t`` and ``c_ipfe``, and only the master key could check them. The ciphertexts of round-0 with the
    label fields of round-1 pass every check of the scheme and are aggregated under round-1 (one key is issued) to the
    sum of round-0, silently. A caller who must know that ciphertexts belong to a round authenticates them."""
    old, new = gradients(seed=10), gradients(seed=11)
    replayed = encrypt_all(wrapper, old, label="round-0")
    current = encrypt_all(wrapper, new)
    calls = spy_on_the_key(wrapper, monkeypatch)
    total = aggregate_gradients(relabel(replayed, current), wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(old), atol=1e-9)              # the sum of round-0 ...
    assert not np.allclose(total, weighted(new), atol=1e-3)                  # ... under the label of round-1
    assert len(calls) == 1 and LABEL in wrapper._aggregated


def test_a_relabelled_replay_by_one_client_is_refused(wrapper, monkeypatch):
    """The claims are caught where the clients disagree: the last client keeps the label of round-0 and differs from the
    others. (The message names the first client and the first that differs from it, whichever of them lies, so it is
    the refusal and the missing key that this pins.)"""
    old = gradients(seed=10)
    replayed = encrypt_all(wrapper, old, label="round-0")
    current = encrypt_all(wrapper, gradients(seed=11))
    relabelled = relabel(replayed, current)
    relabelled["c2"] = replayed["c2"]                                       # c2 keeps its label of round-0
    calls = spy_on_the_key(wrapper, monkeypatch)
    with pytest.raises(FEValidationError, match="differ in their label"):
        aggregate_gradients(relabelled, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    assert calls == [] and LABEL not in wrapper._aggregated


# ---------------------------------------------------------------------------
# a prepared key keeps its policy


def test_prepared_aggregation_keeps_its_policy(wrapper, monkeypatch):
    grads = gradients()
    encrypted = encrypt_all(wrapper, grads)
    prepared = prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL, policy=POLICY)
    assert isinstance(prepared, PreparedAggregation)
    assert prepared.policy == canonical_encoding(parse(POLICY))
    assert prepared.allow_partial is False
    expected = weighted(grads)
    # the policy may be left out, or given again
    for policy in (None, POLICY, parse(POLICY)):
        total = aggregate_gradients(encrypted, wrapper, label=LABEL, prepared=prepared, policy=policy)
        np.testing.assert_allclose(total, expected, atol=1e-9)
    # another policy is refused, also one that the ciphertexts satisfy, before any pairing
    refuse_to_compute(monkeypatch)
    with pytest.raises(ValueError, match="another policy"):
        aggregate_gradients(encrypted, wrapper, label=LABEL, prepared=prepared, policy="purpose:research")


def test_a_prepared_key_is_checked_against_the_attributes_too(wrapper, monkeypatch):
    encrypted = encrypt_all(wrapper, gradients(), attributes=["region:EU"])
    prepared = prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL, policy=POLICY)
    refuse_to_compute(monkeypatch)
    with pytest.raises(FEKeyError, match="do not satisfy"):
        aggregate_gradients(encrypted, wrapper, label=LABEL, prepared=prepared)


def test_a_prepared_key_without_a_policy_is_refused(wrapper, monkeypatch):
    prepared = prepare_aggregation(wrapper, WEIGHTS, shape=SHAPE, label=LABEL, policy=POLICY)
    keyless = PreparedAggregation(prepared.label, prepared.sid, prepared.fusion_weight, prepared.weight_scale,
                                  prepared.n_chunks, prepared.dk, prepared.allow_partial)
    assert keyless.policy is None
    encrypted = encrypt_all(wrapper, gradients())
    refuse_to_compute(monkeypatch)
    with pytest.raises(ValueError, match="no policy"):
        aggregate_gradients(encrypted, wrapper, label=LABEL, prepared=keyless)


# ---------------------------------------------------------------------------
# _check_policy_aggregation, on records


def element_records(attributes, count=3, digest=None):
    """An object array of *count* ciphertext records, which hold the two fields that the check reads."""
    record = {"attributes": list(attributes),
              "attributes_digest": attribute_set_digest(attributes) if digest is None else digest}
    array = np.empty(count, dtype=object)
    for index in range(count):
        array[index] = dict(record)
    return array


def gradient_of(nid, *layers):
    return EncryptedGradient(list(layers), (len(layers[0]),) if layers else (), 1, "decimal", nid)


def test_check_policy_aggregation_accepts_a_satisfied_policy():
    encrypted = {nid: gradient_of(nid, element_records(ATTRIBUTES)) for nid in NIDS}
    assert ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL) is None
    assert ml_gradients._check_policy_aggregation(encrypted, "or(region:EU, region:US)", LABEL) is None


def test_check_policy_aggregation_refuses_an_unsatisfied_policy():
    encrypted = {nid: gradient_of(nid, element_records(["region:EU"])) for nid in NIDS}
    with pytest.raises(FEKeyError, match="do not satisfy") as raised:
        ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)
    assert "round-1" in str(raised.value)


def test_check_policy_aggregation_reads_every_element_of_every_client():
    other = ["region:EU"]
    for client, layer, position in (("c1", 0, 2), ("c2", 1, 0)):
        layers = {nid: [element_records(ATTRIBUTES), element_records(ATTRIBUTES, count=2)] for nid in NIDS}
        layers[client][layer][position] = element_records(other, count=1)[0]
        encrypted = {nid: gradient_of(nid, *arrays) for nid, arrays in layers.items()}
        with pytest.raises(FEValidationError, match=client) as raised:
            ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)
        assert "layer {}, element ({},)".format(layer, position) in str(raised.value)
        assert not isinstance(raised.value, FEKeyError)


def test_check_policy_aggregation_checks_the_digest_against_the_attributes():
    forged = element_records(ATTRIBUTES, digest=attribute_set_digest(["region:EU"]))
    encrypted = {nid: gradient_of(nid, forged) for nid in NIDS}          # they agree, and the digest is another's
    with pytest.raises(FEValidationError, match="digest"):
        ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)


@pytest.mark.parametrize("damage", [
    lambda record: record.pop("attributes_digest"),
    lambda record: record.update(attributes_digest=7),
    lambda record: record.update(attributes_digest=["x"]),
])
def test_check_policy_aggregation_refuses_a_malformed_digest(damage):
    layers = {nid: element_records(ATTRIBUTES) for nid in NIDS}
    damage(layers["c2"][1])
    encrypted = {nid: gradient_of(nid, array) for nid, array in layers.items()}
    with pytest.raises(FEValidationError, match="c2"):
        ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)


@pytest.mark.parametrize("damage", [
    lambda record: record.pop("attributes"),
    lambda record: record.update(attributes="purpose:research"),
    lambda record: record.update(attributes=[3]),
])
def test_check_policy_aggregation_refuses_malformed_attributes(damage):
    layers = {nid: element_records(ATTRIBUTES) for nid in NIDS}
    damage(layers["c0"][0])                                  # the record whose attributes the policy is judged on
    encrypted = {nid: gradient_of(nid, array) for nid, array in layers.items()}
    with pytest.raises(FEValidationError, match="c0"):
        ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)


def test_check_policy_aggregation_refuses_what_is_no_array_of_records():
    packed = {"shape": [3], "first_chunk": 0, "chunks": []}              # a packed layer, as the Ring-LWE schemes make
    encrypted = {nid: gradient_of(nid, element_records(ATTRIBUTES)) for nid in NIDS}
    encrypted["c1"] = gradient_of("c1", packed)
    with pytest.raises(FEValidationError, match="c1"):
        ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)
    encrypted["c1"] = gradient_of("c1", element_records(ATTRIBUTES))
    encrypted["c1"].ciphertexts[0][0] = None
    with pytest.raises(FEValidationError, match="c1"):
        ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL)


def test_check_policy_aggregation_has_nothing_to_check_without_elements():
    encrypted = {nid: gradient_of(nid, element_records(ATTRIBUTES, count=0)) for nid in NIDS}
    assert ml_gradients._check_policy_aggregation(encrypted, parse(POLICY), LABEL) is None


# ---------------------------------------------------------------------------
# the warnings of a costly job


def test_more_than_1000_elements_warn():
    with pytest.warns(FEWarning, match="1001 elements") as caught:
        ml_gradients._warn_policy_cost(1001, clients=10, rows=3, bound=2 ** 20)
    assert len(caught) == 1


def estimate(elements, clients, rows, attributes):
    """The seconds on one thread: the group elements decoded, the pairings, the discrete logarithms."""
    return (elements * clients * (8 * attributes + 5) * ml_gradients._SECONDS_PER_DECODE
            + elements * (clients * (8 * rows + 4) + 1) * ml_gradients._SECONDS_PER_PAIRING
            + elements * ml_gradients._SECONDS_PER_DLOG)


def test_the_estimate_follows_the_decoding_the_pairings_and_the_logarithms_of_the_job():
    decodes = 1001 * 10 * (8 * 3 + 5)          # per element and client: 8 per attribute, t and the 4 of c_ipfe
    pairs = 1001 * (10 * (8 * 3 + 4) + 1)      # per element, 10 clients: 8 pairs for each of 3 rows and 4 more, and 1
    assert (decodes, pairs) == (290290, 281281)
    seconds = round(estimate(1001, 10, 3, 3))
    text = "decodes {} group elements and pairs {}, about {} s".format(decodes, pairs, seconds)
    with pytest.warns(FEWarning, match=text):
        ml_gradients._warn_policy_cost(1001, clients=10, rows=3, bound=2 ** 20, attributes=3)
    more = round(estimate(1001, 10, 6, 3))
    assert more > seconds                      # more rows of the policy, more pairings
    with pytest.warns(FEWarning, match="about {} s".format(more)):
        ml_gradients._warn_policy_cost(1001, clients=10, rows=6, bound=2 ** 20, attributes=3)


def test_the_estimate_grows_with_the_attributes_of_the_ciphertexts():
    texts = []
    for attributes in (3, 12, 24):
        with pytest.warns(FEWarning) as caught:
            ml_gradients._warn_policy_cost(1001, clients=3, rows=3, bound=2 ** 20, attributes=attributes)
        texts.append(str(caught[0].message))
        assert "about {} s".format(round(estimate(1001, 3, 3, attributes))) in texts[-1]
    seconds = [int(re.search(r"about (\d+) s", text).group(1)) for text in texts]
    assert seconds[0] < seconds[1] < seconds[2]
    # the attributes left out are those of the rows
    with pytest.warns(FEWarning) as left_out:
        ml_gradients._warn_policy_cost(1001, clients=3, rows=3, bound=2 ** 20)
    assert str(left_out[0].message) == texts[0]


def test_the_estimate_matches_the_measured_aggregation():
    """1000 elements of 10 clients under a policy of 3 rows, for ciphertexts of 3 attributes: 51.57 s on one thread,
    the measurement that the constants come from (decoding 19.24 s, pairings 31.21 s, logarithms 0.64 s)."""
    assert estimate(1000, 10, 3, 3) == pytest.approx(51.57, rel=0.05)


def test_a_large_bound_warns():
    with pytest.warns(FEWarning, match="2\\^26") as caught:
        ml_gradients._warn_policy_cost(10, clients=10, rows=3, bound=2 ** 27)
    assert len(caught) == 1


def test_a_large_job_with_a_large_bound_warns_twice():
    with pytest.warns(FEWarning) as caught:
        ml_gradients._warn_policy_cost(2000, clients=10, rows=3, bound=2 ** 30)
    assert len(caught) == 2


def test_small_jobs_do_not_warn(recwarn):
    ml_gradients._warn_policy_cost(1000, clients=10, rows=3, bound=2 ** 26)      # the limits themselves are allowed
    ml_gradients._warn_policy_cost(1, clients=1, rows=1, bound=1)
    assert len(recwarn) == 0


def test_the_warning_is_attributed_to_the_caller():
    with pytest.warns(FEWarning) as caught:
        ml_gradients._warn_policy_cost(1001, clients=10, rows=3, bound=2 ** 20)
    assert caught[0].filename == __file__


def test_a_small_aggregation_does_not_warn(wrapper, recwarn):
    warnings.simplefilter("always")
    grads = gradients()
    aggregate_gradients(encrypt_all(wrapper, grads), wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    assert [str(w.message) for w in recwarn] == []


def test_the_aggregation_passes_the_size_of_the_job(wrapper, monkeypatch):
    seen = []
    monkeypatch.setattr(ml_gradients, "_warn_policy_cost", lambda *args: seen.append(args))
    encrypted = encrypt_all(wrapper, gradients())
    aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    # 4 elements, 3 clients, the 2 rows of the policy that the attributes hold, n * bound_x * bound_y, and the 2
    # attributes of the ciphertexts
    assert seen == [(4, 3, 2, 3 * 1000 * 10, 2)]
    seen.clear()
    aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy="or(region:EU, region:US, org:A)")
    assert seen == [(4, 3, 1, 3 * 1000 * 10, 2)]              # rows the attributes hold: at most these are used


def test_the_aggregation_warns_of_a_large_bound():
    wrapper = make_wrapper(bound_x=2 ** 22)                   # 3 * 2^22 * 10 is above 2^26
    grads = {nid: np.round(g, 1) for nid, g in gradients().items()}
    encrypted = encrypt_all(wrapper, grads)
    with pytest.warns(FEWarning, match="2\\^26"):
        total = aggregate_gradients(encrypted, wrapper, WEIGHTS, label=LABEL, policy=POLICY)
    np.testing.assert_allclose(total, weighted(grads), atol=1e-9)


def test_a_refused_request_does_not_warn(monkeypatch, recwarn):
    wrapper_big = make_wrapper(bound_x=2 ** 22)
    encrypted = encrypt_all(wrapper_big, gradients())
    no_key(wrapper_big, monkeypatch)
    with pytest.raises(FEKeyError):
        aggregate_gradients(encrypted, wrapper_big, WEIGHTS, label=LABEL, policy="role:auditor")
    assert [str(w.message) for w in recwarn if issubclass(w.category, FEWarning)] == []


# ---------------------------------------------------------------------------
# the benchmark that lists the usable schemes


def test_the_module_carries_the_mark_of_the_pairing_tests(request):
    """It needs a pairing backend: ``-m extended`` runs it (tests/conftest.py)."""
    assert request.node.get_closest_marker("extended") is not None
    assert request.node.get_closest_marker("core") is None


def test_the_fl_coverage_benchmark_finds_the_scheme_usable_and_the_per_client_scheme_refused():
    if not pairing_available(DEFAULT_PAIRING_GROUP):         # the benchmark runs the default group, not GROUP
        pytest.skip("no backend for pairing group {}".format(DEFAULT_PAIRING_GROUP))
    path = ROOT / "benchmarks" / "commands" / "fl_coverage.py"
    spec = importlib.util.spec_from_file_location("bench_fl_coverage", path)
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FEWarning)             # the options of its shared configuration that it ignores
        assert coverage.check_entry("mcfe", "kp_ab_pairing") == ("usable", "")
        with pytest.warns(FESecurityWarning, match="known to be insecure"):
            status, detail = coverage.check_entry("mcfe", "kp_ab_pairing_per_client")     # the adapter refuses it
        assert status == "unusable" and "different sets of attributes" in detail
