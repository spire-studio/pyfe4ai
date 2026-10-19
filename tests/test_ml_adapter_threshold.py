"""The ML adapter with the threshold Ring-LWE MCFE (``mcfe/ring_lwe_threshold``).

The wrapper runs the servers: per round their label key shares, once per
weights their weight key shares (the scheme's default split key shares),
partial decryptions and the combination, with the defaults (binary, RNS) and
with the options of earlier versions; offline encryption and prepared keys;
ciphertexts and partial decryptions sent as JSON; threads; the policy checks.
"""

from __future__ import annotations

import concurrent.futures
import json
import pickle

import numpy as np
import pytest

from pyfe4ai.utils import native_backend
from pyfe4ai.utils.ml_adapter import (
    EncryptedGradient,
    FESchemeWrapper,
    ThresholdPartial,
    aggregate_gradients,
    combine_partial_decryptions,
    encrypt_gradient,
    partial_decrypt_gradients,
    precompute_gradient,
    prepare_aggregation,
)

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS = ["nid_0", "nid_1", "nid_2"]
SIDS = ["sid_0", "sid_1", "sid_2"]
WEIGHTS = {"nid_0": 1, "nid_1": 2, "nid_2": 1}
SHAPE = (3, 7)
LEGACY = {"ct_format": "json", "ring_modulus": "prime", "key_shares": "full"}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _wrapper(**extra):
    cfg = {"eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": 16, "sec_param": 64, "bound_x": 50, "bound_y": 2,
           "s": 3, "t": 2, "lst_sid": SIDS, **extra}
    return FESchemeWrapper("mcfe", "ring_lwe_threshold", cfg, precision=1)


def _grads(seed):
    rng = np.random.default_rng(seed)
    return {nid: np.round(rng.uniform(-2, 2, SHAPE), 1) for nid in NIDS}


def _expected(grads, weights=WEIGHTS):
    return sum(weights[nid] * grads[nid] for nid in weights)


def _sent(enc):
    """Encrypted gradients as received over JSON."""
    return {nid: EncryptedGradient.from_dict(json.loads(json.dumps(e.to_dict()))) for nid, e in enc.items()}


def _count_weight_parts(wrapper, monkeypatch):
    calls = []
    real = wrapper.kg.get_weight_key_share

    def spy(sid, **kwargs):
        calls.append(sid)
        return real(sid, **kwargs)

    monkeypatch.setattr(wrapper.kg, "get_weight_key_share", spy)
    return calls


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("phase", ["plain", "prepared", "offline"])
def test_threshold_rounds_with_the_defaults(name, phase, monkeypatch):
    w = _wrapper()
    assert w.pp["ct_format"] == "binary" and len(w.pp["q_factors"]) >= 3 and w.kg.key_shares == "split"
    weight_parts = _count_weight_parts(w, monkeypatch)
    with native_backend.backend(name):
        for r in range(3):
            label = "r%d" % r
            grads = _grads(r)
            prepared = prepare_aggregation(w, WEIGHTS, shape=SHAPE, label=label) if phase != "plain" else None
            pre = {nid: precompute_gradient(SHAPE, w, nid=nid, label=label) for nid in NIDS} if phase == "offline" else {}
            enc = {nid: encrypt_gradient(grads[nid], w, nid=nid, label=label, precomputed=pre.get(nid)) for nid in NIDS}
            assert isinstance(enc["nid_0"].ciphertexts[0]["chunks"][0]["ct"], bytes)
            received = _sent(enc)
            if prepared is None:
                out = aggregate_gradients(received, w, WEIGHTS, label=label)
            else:
                out = aggregate_gradients(received, w, label=label, prepared=prepared)
            np.testing.assert_allclose(out, _expected(grads), atol=1e-9)
    # the weight parts went to the two enrolled servers once, for all rounds
    assert sorted(weight_parts) == ["sid_0", "sid_1"]


@pytest.mark.parametrize("servers", [["sid_1", "sid_2"], SIDS])
def test_partial_decryptions_travel_as_json(servers):
    w = _wrapper()
    grads = _grads(5)
    prepared = prepare_aggregation(w, WEIGHTS, shape=SHAPE, label="r", servers=servers)
    assert prepared.servers == servers and set(prepared.dk) == set(servers)
    received = _sent({nid: encrypt_gradient(grads[nid], w, nid=nid, label="r") for nid in NIDS})
    partials = {}
    for sid in servers:
        part = partial_decrypt_gradients(received, w, prepared, server=sid)
        data = json.loads(json.dumps(part.to_dict()))
        assert data["format"] == "binary" and isinstance(data["layers"][0]["parts"][0]["w"], str)
        partials[sid] = ThresholdPartial.from_dict(data)
        assert partials[sid] == part
    np.testing.assert_allclose(combine_partial_decryptions(partials, w, prepared), _expected(grads), atol=1e-9)


@pytest.mark.parametrize("extra", [LEGACY, {"sharing": "replicated-seeded"}, {"t": 3}],
                         ids=["json-prime-full", "seeded", "shamir"])
def test_threshold_rounds_with_other_options(extra):
    w = _wrapper(**extra)
    grads = _grads(7)
    enc = {nid: encrypt_gradient(grads[nid], w, nid=nid, label="r") for nid in NIDS}
    sent = {nid: json.loads(json.dumps(e.to_dict())) for nid, e in enc.items()}
    assert ("base64" in sent["nid_0"]["ciphertexts"][0]["packed"]) == (extra is not LEGACY)
    prepared = prepare_aggregation(w, WEIGHTS, shape=SHAPE, label="r")
    received = {nid: EncryptedGradient.from_dict(d) for nid, d in sent.items()}
    partials = {sid: ThresholdPartial.from_dict(json.loads(json.dumps(
        partial_decrypt_gradients(received, w, prepared, server=sid).to_dict()))) for sid in prepared.servers}
    assert {p.to_dict()["format"] for p in partials.values()} == {"json" if extra is LEGACY else "binary"}
    np.testing.assert_allclose(combine_partial_decryptions(partials, w, prepared), _expected(grads), atol=1e-9)


def test_threshold_rounds_from_threads():
    """Clients encrypt from threads; rounds of several labels aggregate at once
    on the same server objects."""
    w = _wrapper()
    labels = ["t%d" % i for i in range(4)]
    grads = {label: _grads(i) for i, label in enumerate(labels)}
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        futures = {(label, nid): pool.submit(encrypt_gradient, grads[label][nid], w, nid=nid, label=label)
                   for label in labels for nid in NIDS}
        enc = {label: {nid: futures[(label, nid)].result() for nid in NIDS} for label in labels}
        outs = list(pool.map(lambda label: aggregate_gradients(enc[label], w, WEIGHTS, label=label), labels))
    for label, out in zip(labels, outs):
        np.testing.assert_allclose(out, _expected(grads[label]), atol=1e-9)
    assert len(w._servers) == 2


def test_a_pickled_wrapper_gives_its_servers_the_weight_parts_again(monkeypatch):
    w = _wrapper()
    grads = _grads(1)
    aggregate_gradients({nid: encrypt_gradient(grads[nid], w, nid=nid, label="a") for nid in NIDS}, w, WEIGHTS,
                        label="a")
    copy = pickle.loads(pickle.dumps(w))
    assert copy._servers == {}
    weight_parts = _count_weight_parts(copy, monkeypatch)
    enc = {nid: encrypt_gradient(grads[nid], copy, nid=nid, label="b") for nid in NIDS}
    np.testing.assert_allclose(aggregate_gradients(enc, copy, WEIGHTS, label="b"), _expected(grads), atol=1e-9)
    assert sorted(weight_parts) == ["sid_0", "sid_1"]


def test_threshold_policy_and_errors():
    w = _wrapper()
    grads = _grads(2)
    enc = {nid: encrypt_gradient(grads[nid], w, nid=nid, label="r") for nid in NIDS}
    for servers in (["sid_0"], ["sid_0", "sid_0"], ["sid_0", "sid_9"]):
        with pytest.raises(ValueError, match="servers"):
            aggregate_gradients(enc, w, WEIGHTS, label="r", servers=servers)
    with pytest.raises(ValueError, match="enrolled client"):  # the wrapper's policy as for the other schemes
        aggregate_gradients({n: enc[n] for n in NIDS[:2]}, w, {n: 1 for n in NIDS[:2]}, label="r")
    prepared = prepare_aggregation(w, WEIGHTS, shape=SHAPE, label="r", servers=["sid_0", "sid_2"])
    with pytest.raises(ValueError, match="servers"):
        aggregate_gradients(enc, w, label="r", prepared=prepared, servers=["sid_0", "sid_1"])
    with pytest.raises(ValueError, match="not enrolled"):
        partial_decrypt_gradients(enc, w, prepared, server="sid_1")
    with pytest.raises(ValueError, match="clients"):
        partial_decrypt_gradients({n: enc[n] for n in NIDS[:2]}, w, prepared, server="sid_0")
    parts = {sid: partial_decrypt_gradients(enc, w, prepared, server=sid) for sid in prepared.servers}
    with pytest.raises(ValueError, match="need the partial decryptions"):
        combine_partial_decryptions({"sid_0": parts["sid_0"]}, w, prepared)
    with pytest.raises(ValueError, match="need the partial decryptions"):
        combine_partial_decryptions(dict(parts, sid_2=ThresholdPartial("sid_2", "other", parts["sid_2"].layers)),
                                    w, prepared)
    np.testing.assert_allclose(combine_partial_decryptions(parts, w, prepared), _expected(grads), atol=1e-9)
    with pytest.raises(ValueError, match="other weights"):  # a label is aggregated with one weight vector
        aggregate_gradients(enc, w, {"nid_0": 1, "nid_1": 1, "nid_2": 1}, label="r")
    # flooding_accounting = "kl": a budget per server, which the wrapper does not keep
    kl = _wrapper(t=3, flooding_accounting="kl", max_partial_decryptions=100)
    with pytest.raises(NotImplementedError, match="partial_budget"):
        prepare_aggregation(kl, WEIGHTS, shape=SHAPE, label="r")
    # servers are those of the threshold scheme
    plain = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": 16, "sec_param": 64,
                                                 "bound_x": 50, "bound_y": 2}, precision=1)
    with pytest.raises(ValueError, match="threshold"):
        prepare_aggregation(plain, WEIGHTS, shape=SHAPE, label="r", servers=SIDS)
    with pytest.raises(NotImplementedError, match="ring_lwe_threshold"):
        partial_decrypt_gradients(enc, plain, prepared, server="sid_0")


def test_other_threshold_variants_are_still_refused():
    w = FESchemeWrapper("mcfe", "ddh_threshold", {"sec_param": 128, "eta": 1, "n": 2, "s": 2, "t": 2,
                                                  "lst_nid": NIDS[:2], "lst_sid": SIDS[:2]})
    with pytest.raises(NotImplementedError, match="threshold"):
        prepare_aggregation(w, {"nid_0": 1, "nid_1": 1}, shape=SHAPE, label="r")
    with pytest.raises(NotImplementedError, match="ring_lwe_threshold"):
        combine_partial_decryptions({}, w, None)


def _partial_dict():
    w = _wrapper()
    grads = _grads(3)
    prepared = prepare_aggregation(w, WEIGHTS, shape=SHAPE, label="r")
    enc = {nid: encrypt_gradient(grads[nid], w, nid=nid, label="r") for nid in NIDS}
    return json.loads(json.dumps(partial_decrypt_gradients(enc, w, prepared, server="sid_0").to_dict()))


def _with_part(d, **change):
    part = dict(d["layers"][0]["parts"][0], **change)
    return dict(d, layers=[dict(d["layers"][0], parts=[part] + d["layers"][0]["parts"][1:])])


def test_threshold_partial_layout_is_checked():
    d = _partial_dict()
    assert ThresholdPartial.from_dict(d).to_dict() == d
    bads = [
        (dict(d, extra=1), "keys"),
        (dict(d, format="xml"), "format"),
        (dict(d, sid=1), "sid"),
        (dict(d, layers=[]), "non-empty"),
        (dict(d, layers=[dict(d["layers"][0], extra=1)]), "shape and parts"),
        (dict(d, layers=[dict(d["layers"][0], shape=[-1])]), "shape"),
        (dict(d, layers=[dict(d["layers"][0], parts=[])]), "no parts"),
        (_with_part(d, k=0), "k >= 1"),
        (_with_part(d, k=True), "k >= 1"),
        (_with_part(d, sid="sid_1"), "sid"),
        (_with_part(d, ct_digest=5), "ct_digest"),
        (_with_part(d, lst_sid_enrolled="sid_0"), "lst_sid_enrolled"),
        (_with_part(d, w="not base64!"), "base64"),
        (_with_part(d, c_y=["1", "2"]), "base64"),
        (dict(_with_part(d, w=["1"], c_y=["2"]), format="json"), "lists of decimal strings"),  # others binary
    ]
    for bad, match in bads:
        with pytest.raises(ValueError, match=match):
            ThresholdPartial.from_dict(bad)
    with pytest.raises(TypeError, match="mixing"):
        part = ThresholdPartial.from_dict(d)
        mixed = dict(part.layers[0]["parts"][0], w=["1"], c_y=["2"])
        ThresholdPartial(part.sid, part.label, [dict(part.layers[0], parts=[mixed] + part.layers[0]["parts"][1:])]
                         ).to_dict()
