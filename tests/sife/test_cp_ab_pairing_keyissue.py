"""Threshold key issuance of sife/cp_ab_pairing (experimental): roadmap/v0.7/03 rev. 2, conditions of its Sec. 7."""

import copy
import hashlib
import json
import os
import sys

import pytest

import pyfe4ai
from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError  # noqa: E402
from pyfe4ai.core.formats.pairing import CPABFunctionalKey  # noqa: E402
from pyfe4ai.schemes.sife import cp_ab_pairing_keyissue as ki  # noqa: E402
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing  # noqa: E402
from pyfe4ai.schemes.sife.cp_ab_pairing_keyissue import (  # noqa: E402
    SIFECPABPairingDealer,
    SIFECPABPairingKeyCombiner,
    SIFECPABPairingKeyServer,
)

pytestmark = pytest.mark.usefixtures("own_parameter_cache")

UNIVERSE = ["org:A", "org:B", "org:C", "role:auditor"]
ETA = 3
SERVERS = ["ks1", "ks2", "ks3", "ks4", "ks5"]
X = [3, -1, 7]


def _allow(request, requester, history):
    return True


def _system(t=3, servers=SERVERS, approve=_allow, max_copies=2):
    config = {"eta": ETA, "bound_x": 10, "bound_y": 10, "universe": UNIVERSE, "max_copies": max_copies,
              "pairing_group_param": GROUP}
    dealer = SIFECPABPairingDealer(config, list(servers), t)
    dealer.setup()
    pp, ks = dealer.get_public_parameters(), dealer.key_servers()
    key_servers = {s: SIFECPABPairingKeyServer(pp, ks, dealer.server_share(s), approve) for s in servers}
    return dealer, pp, ks, key_servers, SIFECPABPairingKeyCombiner(pp, ks), SIFECPABPairing({"keys": {"pp": pp}})


@pytest.fixture
def system():
    return _system()


def _issue(key_servers, request, requester=None):
    return [key_servers[s].issue(request, requester or request["sid"]) for s in request["servers"]]


def _resign(request):
    request = copy.deepcopy(request)
    request["digest"] = ki.request_digest(request)
    return request


# -- correctness -------------------------------------------------------------------


@pytest.mark.parametrize("t, servers, chosen", [
    (1, ["a"], ["a"]),
    (2, ["a", "b", "c"], ["a", "c"]),
    (3, SERVERS, ["ks1", "ks3", "ks5"]),
    (3, SERVERS, ["ks2", "ks3", "ks4"]),
    (5, SERVERS, SERVERS),
])
def test_t_partial_keys_give_a_key_of_the_scheme(t, servers, chosen):
    _, pp, ks, key_servers, comb, crypto = _system(t, servers)
    y = [1, -2, 4]
    request = comb.new_request("alice", y, ["org:A", "role:auditor"], chosen)
    dk = comb.combine(request, _issue(key_servers, request))
    assert list(dk) == list(CPABFunctionalKey.FIELDS)
    ct = crypto.encrypt(X, policy="and(org:A, role:auditor)")
    assert crypto.decrypt(ct, dk) == sum(a * b for a, b in zip(X, y))
    with pytest.raises(FEKeyError):
        crypto.decrypt(crypto.encrypt(X, policy="and(org:A, org:B)"), dk)


@pytest.mark.parametrize("policy, held", [
    ("or(and(org:A, org:B), and(org:A, org:C))", ["org:A", "org:C"]),  # copies of org:A
    ("2of(org:A, org:B, role:auditor)", ["org:B", "role:auditor"]),
    ("org:C", ["org:C"]),
])
def test_policies_with_copies(system, policy, held):
    _, _, _, key_servers, comb, crypto = system
    request = comb.new_request("bob", [2, 0, -3], held, ["ks1", "ks2", "ks4"])
    dk = comb.combine(request, _issue(key_servers, request))
    assert crypto.decrypt(crypto.encrypt(X, policy=policy), dk) == 2 * 3 - 3 * 7


def test_json_round_trips(system):
    _, pp, ks, key_servers, comb, crypto = system
    j = lambda o: json.loads(json.dumps(o))  # noqa: E731
    request = j(comb.new_request("alice", [1, 1, 1], ["org:A"], ["ks1", "ks2", "ks3"]))
    partials = [j(p) for p in _issue(key_servers, request)]
    dk = j(SIFECPABPairingKeyCombiner(j(pp), j(ks)).combine(request, partials))
    assert crypto.decrypt(crypto.encrypt(X, policy="org:A"), dk) == sum(X)


# -- determinism, digests, PRF -------------------------------------------------------


def test_a_repeated_request_gets_the_same_partial_key(system):
    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    assert _issue(key_servers, request) == _issue(key_servers, request)
    other = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])  # a new id
    assert other["digest"] != request["digest"]
    assert _issue(key_servers, other)[0]["k1"] != _issue(key_servers, request)[0]["k1"]


def test_the_digest_covers_every_field(system):
    _, _, _, _, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    assert len(request["digest"]) == 2 * ki.DIGEST_BYTES
    changes = {"id": "00" * 16, "sid": "mallory", "y": ["1", "2", "4"], "attributes": ["org:B"],
               "servers": ["ks1", "ks2", "ks4"], "pp_binding": "x", "keyservers": "y", "version": "z"}
    for field, value in changes.items():
        assert ki.request_digest(dict(request, **{field: value})) != request["digest"], field


def test_prf_and_digest_derivation():
    key, digest, index = bytes(range(32)), bytes(range(64)), ki._index(3, 7, 1)
    expected = hashlib.shake_256(len(ki.DST_MASK).to_bytes(2, "big") + ki.DST_MASK + key + digest + index)
    order = 2 ** 255 - 19
    assert ki._prf(key, ki.DST_MASK, digest, index, order) == int.from_bytes(expected.digest(64), "big") % order
    request = {"version": "cp_ab_keyrequest/v1", "id": "00" * 16, "sid": "alice", "y": ["1", "-2"],
               "attributes": ["org:A"], "servers": ["a", "b"], "pp_binding": "p", "keyservers": "k", "digest": ""}
    payload = json.dumps(["cp_ab_keyrequest/v1", "00" * 16, "alice", ["1", "-2"], ["org:A"], ["a", "b"], "p", "k"],
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    dst = ki.DST_REQUEST
    assert ki.request_digest(request) == hashlib.shake_256(len(dst).to_bytes(2, "big") + dst + payload).hexdigest(64)


def test_mask_indices_are_injective_and_carry_the_copy(system):
    _, _, _, key_servers, comb, _ = system
    p = key_servers["ks1"]._p
    slots = p.slots(UNIVERSE)
    indices = [ki._index(ki._K1, 0, t) for t in (0, 1)] + [ki._index(ki._K0, 0, t) for t in (0, 1)]
    indices += [ki._index(ki._KATTR, slot, t) for _, slot in slots for t in (0, 1)]
    assert len(set(indices)) == len(indices)
    assert {name for name, _ in slots} == {"{}#{}".format(l, c) for l in UNIVERSE for c in (1, 2)}


# -- the combiner's checks -----------------------------------------------------------


def _fields(partial):
    out = [("k1", t) for t in (0, 1)] + [("k0", t) for t in (0, 1)]
    return out + [("k_attr", (name, t)) for name in sorted(partial["k_attr"]) for t in (0, 1)]


def test_any_changed_component_fails_the_check(system):
    _, pp, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A", "org:B"], ["ks1", "ks2", "ks3"])
    partials = _issue(key_servers, request)
    for i in range(len(partials)):
        for field, where in _fields(partials[i]):
            bad = copy.deepcopy(partials)
            if field == "k_attr":
                name, t = where
                bad[i]["k_attr"][name][t] = pp["g2"]
            else:
                bad[i][field][where] = pp["g2"]
            with pytest.raises(FEKeyError, match="fails its check"):
                comb.combine(request, bad)


def test_partial_key_sets_are_checked(system):
    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    partials = _issue(key_servers, request)
    other = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    for bad in (partials[:2], partials + partials[:1], [partials[0], partials[0], partials[1]],
                [partials[0], partials[1], _issue(key_servers, other)[2]]):
        with pytest.raises(FEKeyError):
            comb.combine(request, bad)
    missing = copy.deepcopy(partials)
    missing[0]["k_attr"].pop("org:A#2")
    with pytest.raises(FEKeyError, match="every copy"):
        comb.combine(request, missing)


def test_a_partial_key_relabelled_for_another_request_fails_the_check(system):
    """Masks and r are derived from the digest: an honest partial key of one request does not
    complete another, even with the request field rewritten."""
    _, _, _, key_servers, comb, _ = system
    r1 = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    r2 = comb.new_request("alice", [0, 1, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    p1, p2 = _issue(key_servers, r1), _issue(key_servers, r2)
    mixed = [dict(p1[0], request=r2["digest"]), p2[1], p2[2]]
    with pytest.raises(FEKeyError, match="fails its check"):
        comb.combine(r2, mixed)


# -- the servers' checks -------------------------------------------------------------


def test_sid_must_be_the_authenticated_requester(system):
    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError, match="authenticated"):
        key_servers["ks1"].issue(request, authenticated_requester="mallory")


def test_the_approval_rule_sees_y_requester_and_history():
    seen = []

    def approve(request, requester, history):
        seen.append((request["y"], requester, len(history)))
        return request["y"][1] == "0"  # only keys that ignore x_2

    _, _, _, key_servers, comb, _ = _system(approve=approve)
    ok = comb.new_request("alice", [1, 0, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    comb.combine(ok, _issue(key_servers, ok))
    unit = comb.new_request("alice", [0, 1, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError, match="approval rule"):
        key_servers["ks1"].issue(unit, "alice")
    assert seen[0] == (["1", "0", "3"], "alice", 0) and seen[-1][0] == ["0", "1", "0"]
    assert key_servers["ks1"]._history[-1][0] == "alice"


def test_a_server_needs_an_approval_rule(system):
    dealer, pp, ks, _, _, _ = _system(t=1, servers=["a"])
    with pytest.raises(FEConfigError):
        SIFECPABPairingKeyServer(pp, ks, {}, None)


@pytest.mark.parametrize("change", ["duplicate", "short", "unknown", "unsorted", "not_mine"])
def test_server_sets(system, change):
    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    request["servers"] = {"duplicate": ["ks1", "ks1", "ks2"], "short": ["ks1", "ks2"],
                          "unknown": ["ks1", "ks2", "ks9"], "unsorted": ["ks2", "ks1", "ks3"],
                          "not_mine": ["ks2", "ks3", "ks4"]}[change]
    with pytest.raises(FEValidationError):
        key_servers["ks1"].issue(_resign(request), "alice")


@pytest.mark.parametrize("field, value", [
    ("y", ["1", "2"]), ("y", ["1", "2", "11"]), ("y", ["1", "2", "x"]), ("y", ["1", "2", "03"]),
    ("attributes", ["org:Z"]), ("attributes", ["org:B", "org:A"]), ("attributes", ["org:A", "org:A"]),
    ("attributes", []), ("sid", ""), ("id", "00"), ("pp_binding", "00"), ("keyservers", "00"),
    ("version", "cp_ab_keyrequest/v0"),
])
def test_request_inputs(system, field, value):
    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    request[field] = value
    with pytest.raises((FEValidationError, FEKeyError)):
        key_servers["ks1"].issue(_resign(request), request["sid"] or "x")


def test_a_request_must_match_its_digest(system):
    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEValidationError, match="digest"):
        key_servers["ks1"].issue(dict(request, y=["3", "2", "1"]), "alice")


def test_a_digest_used_for_other_content_is_refused(system, monkeypatch):
    """With a broken digest (here: constant), two requests of one digest and different y would give
    k1 = 1, k0 = [U (y1 - y2)]_2; the record digest -> content refuses the second."""
    _, _, _, key_servers, comb, _ = system
    monkeypatch.setattr(ki, "request_digest", lambda request: "ab" * ki.DIGEST_BYTES)
    r1 = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    r2 = comb.new_request("alice", [0, 1, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    assert r1["digest"] == r2["digest"]
    key_servers["ks1"].issue(r1, "alice")
    with pytest.raises(FEKeyError, match="other content"):
        key_servers["ks1"].issue(r2, "alice")


def test_an_id_used_for_other_content_is_refused(system):
    _, _, _, key_servers, comb, _ = system
    r1 = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    key_servers["ks1"].issue(r1, "alice")
    r2 = _resign(dict(r1, y=["0", "1", "0"]))
    with pytest.raises(FEKeyError, match="id"):
        key_servers["ks1"].issue(r2, "alice")


# -- the dealer ----------------------------------------------------------------------


def test_the_dealer_hands_out_each_share_once_and_issues_no_keys(system):
    dealer = system[0]
    assert dealer._pending == {}
    with pytest.raises(FEKeyError):
        dealer.server_share("ks1")
    with pytest.raises(FEKeyError):
        dealer.get_decryption_keys("s", credentials={"fusion_weight": [1, 1, 1], "attributes": ["org:A"]})
    assert dealer._kg.msk is None


def test_each_sharing_has_fresh_prf_keys_and_its_own_id():
    d1, _, ks1, _, _, _ = _system(t=2, servers=["a", "b"])
    d2 = SIFECPABPairingDealer({"eta": ETA, "bound_x": 10, "bound_y": 10, "universe": UNIVERSE, "max_copies": 2,
                                "pairing_group_param": GROUP}, ["a", "b"], 2)
    d2.setup()
    assert d2.key_servers()["sharing_id"] != ks1["sharing_id"]
    s2 = d2.server_share("a")
    assert s2["keyservers"] == ki.keyservers_digest(d2.key_servers())
    assert len(s2["r_key"]) == 64 and set(s2["mask_keys"]) == {"b"}


def test_shares_and_records_must_fit(system):
    dealer, pp, ks, _, _, _ = _system(t=2, servers=["a", "b", "c"])
    d2 = SIFECPABPairingDealer({"eta": ETA, "bound_x": 10, "bound_y": 10, "universe": UNIVERSE, "max_copies": 2,
                                "pairing_group_param": GROUP}, ["a", "b", "c"], 2)
    d2.setup()
    foreign = d2.server_share("a")
    with pytest.raises(FEValidationError):
        SIFECPABPairingKeyServer(pp, ks, foreign, _allow)          # another sharing
    for change in ({"points": [1, 2, 4]}, {"threshold": 4}, {"pp_binding": "00"}, {"sharing": "replicated"},
                   {"servers": ["a", "a", "c"]}, {"version": "cp_ab_keyservers/v0"}):
        with pytest.raises(FEValidationError):
            SIFECPABPairingKeyCombiner(pp, dict(ks, **change))


@pytest.mark.parametrize("servers, t", [([], 1), (["a", "a"], 1), (["a", ""], 1), (["a", "b"], 3), (["a"], 0)])
def test_dealer_arguments(servers, t):
    with pytest.raises(FEValidationError):
        SIFECPABPairingDealer({"eta": ETA, "universe": UNIVERSE, "pairing_group_param": GROUP}, servers, t)


# -- registry and records ------------------------------------------------------------


def test_the_capability_is_registered():
    assert "distributed_keygen" in pyfe4ai.CAPABILITIES
    assert "distributed_keygen" in pyfe4ai.get_scheme("sife/cp_ab_pairing").capabilities


@pytest.mark.parametrize("name", ["CPABKeyServers", "CPABKeyServerShare", "CPABKeyRequest", "CPABPartialKey"])
def test_records_document_their_fields(name):
    from pyfe4ai.core.formats import keyissue

    cls = getattr(keyissue, name)
    assert cls.FIELDS and "Fields, in wire order" in cls.__doc__
    assert all(cls._fields[f].doc for f in cls.FIELDS)


def test_objects_are_their_records(system):
    from pyfe4ai.core.formats import keyissue

    _, _, ks, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    assert list(ks) == list(keyissue.CPABKeyServers.FIELDS)
    assert list(request) == list(keyissue.CPABKeyRequest.FIELDS)
    assert list(_issue(key_servers, request)[0]) == list(keyissue.CPABPartialKey.FIELDS)


# -- re-check (03 rev. 2): repeats, ids per sid, refused requests, the success bit (Lemma 6) --------


def test_a_repeat_is_answered_without_a_new_approval():
    calls = []

    def one_key_per_sid_and_y(request, requester, history):  # a stateful rule
        calls.append(request["id"])
        return all(not (r["sid"] == request["sid"] and r["y"] == request["y"]) for _, r in history)

    _, _, _, key_servers, comb, _ = _system(approve=one_key_per_sid_and_y)
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    first = key_servers["ks1"].issue(request, authenticated_requester="alice")
    again = key_servers["ks1"].issue(copy.deepcopy(request), authenticated_requester="alice")
    assert first == again and calls == [request["id"]] and len(key_servers["ks1"]._history) == 1
    other = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])  # same (sid, y), new id
    with pytest.raises(FEKeyError, match="approval rule"):
        key_servers["ks1"].issue(other, authenticated_requester="alice")


def test_ids_are_kept_per_sid():
    """A corrupted party that sees an honest user's request id cannot block it by using the id first."""
    _, _, _, key_servers, comb, _ = _system()
    honest = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    squat = _resign(dict(comb.new_request("mallory", [0, 0, 1], ["org:B"], ["ks1", "ks2", "ks3"]), id=honest["id"]))
    key_servers["ks1"].issue(squat, authenticated_requester="mallory")
    key_servers["ks1"].issue(honest, authenticated_requester="alice")


def test_a_refused_request_leaves_no_record():
    _, _, _, key_servers, comb, _ = _system(approve=lambda request, requester, history: request["y"][0] == "1")
    refused = comb.new_request("alice", [2, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError):
        key_servers["ks1"].issue(refused, authenticated_requester="alice")
    server = key_servers["ks1"]
    assert refused["digest"] not in server._contents and not server._ids and not server._history
    reused = _resign(dict(refused, y=["1", "0", "0"]))  # the same id, now accepted
    server.issue(reused, authenticated_requester="alice")


def test_another_requester_is_refused_before_the_rule_runs():
    calls = []
    _, _, _, key_servers, comb, _ = _system(approve=lambda request, requester, history: calls.append(1) or True)
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError, match="authenticated"):
        key_servers["ks1"].issue(request, authenticated_requester="mallory")
    assert calls == [] and not key_servers["ks1"]._contents


def _times(group, partial, factors):
    """*partial* with each component multiplied by the matching group element of *factors*."""
    from pyfe4ai.primitives.pairing.pairing_backend import G2, multiexp
    from pyfe4ai.schemes.common.ac_codec import read_elements, serialize_vec

    out = copy.deepcopy(partial)
    for field in ("k1", "k0"):
        elems = read_elements(group, out[field], G2, field)
        out[field] = serialize_vec(group, [multiexp([e, f], [1, 1], group) for e, f in zip(elems, factors[field])])
    for name in out["k_attr"]:
        elems = read_elements(group, out["k_attr"][name], G2, "k_attr")
        out["k_attr"][name] = serialize_vec(group, [multiexp([e, f], [1, 1], group)
                                                    for e, f in zip(elems, factors["k_attr"][name])])
    return out


def test_the_success_bit_is_the_homogeneous_check(system):
    """03 Lemma 6.2: a corrupted server's answer times KeyPart(0, rho) passes (V0), so the honest
    user's combine succeeds, with a key shifted by [b rho]; times a Delta that fails (V0) it fails."""
    import secrets as _secrets

    from pyfe4ai.primitives.pairing.pairing_backend import multiexp

    _, _, _, key_servers, comb, crypto = system
    p = key_servers["ks1"]._p
    group, order = p.group, p.order
    request = comb.new_request("alice", [1, -2, 3], ["org:A", "role:auditor"], ["ks1", "ks2", "ks3"])
    partials = _issue(key_servers, request)
    honest_key = comb.combine(request, partials)
    rho = _secrets.randbelow(order - 1) + 1
    shift = {"k1": [multiexp([e], [rho], group) for e in p.b],
             "k0": [multiexp([e], [rho], group) for e in p.w_b(p.universe.size)],
             "k_attr": {name: [multiexp([e], [rho], group) for e in p.w_b(slot)] for name, slot in p.slots(request["attributes"])}}
    shifted = [_times(group, partials[0], shift)] + partials[1:]
    key = comb.combine(request, shifted)
    assert key["k1"] != honest_key["k1"]
    ct = crypto.encrypt(X, policy="and(org:A, role:auditor)")
    assert crypto.decrypt(ct, key) == crypto.decrypt(ct, honest_key)
    assert crypto.decrypt(ct, key) == sum(a * b for a, b in zip(X, [1, -2, 3]))
    identity = p.identity_g2
    bad = {"k1": [identity, identity], "k0": [p.g2, identity],
           "k_attr": {name: [identity, identity] for name, _ in p.slots(request["attributes"])}}
    with pytest.raises(FEKeyError, match="fails its check"):
        comb.combine(request, [_times(group, partials[0], bad)] + partials[1:])


def test_concurrent_requests_of_one_digest_accept_one_content(system, monkeypatch):
    """04 Sec. 3, row 3 (R-0.7 M1b-9): with a broken (constant) digest, threads sending different contents under
    one digest to one server: exactly one content is accepted, every other thread is refused."""
    import threading

    _, _, _, key_servers, comb, _ = system
    monkeypatch.setattr(ki, "request_digest", lambda request: "cd" * ki.DIGEST_BYTES)
    requests = [comb.new_request("alice", [i, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"]) for i in range(1, 9)]
    barrier = threading.Barrier(len(requests), timeout=5)
    outcomes = []

    def send(request):
        barrier.wait()
        try:
            key_servers["ks1"].issue(request, authenticated_requester="alice")
            outcomes.append(("ok", request["y"][0]))
        except FEKeyError:
            outcomes.append(("refused", request["y"][0]))

    threads = [threading.Thread(target=send, args=(r,)) for r in requests]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    accepted = [y for verdict, y in outcomes if verdict == "ok"]
    assert len(outcomes) == len(requests) and len(accepted) == 1
    assert len(key_servers["ks1"]._contents) == 1 and len(key_servers["ks1"]._history) == 1


@pytest.mark.parametrize("attributes, servers", [(["org:\ud800"], ["ks1", "ks2", "ks3"]),
                                                 (["org:A"], ["ks1", "ks2", "k\ud800"])])
def test_unencodable_labels_and_server_ids_are_validation_errors(system, attributes, servers):
    """R-0.7 re-check RC2-1: a lone surrogate in a label or a server id is an FEValidationError."""
    comb = system[4]
    with pytest.raises(FEValidationError):
        comb.new_request("alice", [1, 0, 0], attributes, servers)


def test_requests_must_be_plain_dicts_that_copy(system):
    """R-0.7 re-check RC2-2, RC2-3: a request that is no plain dict, or does not copy, is an FEValidationError."""
    import threading

    _, _, _, key_servers, comb, _ = system
    request = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])

    class Lying(dict):
        def __len__(self):
            return len(request)

    for bad in (Lying(request, purpose="audit"), dict(request, lock=threading.Lock())):
        with pytest.raises(FEValidationError):
            key_servers["ks1"].issue(bad, authenticated_requester="alice")


# -- the history as read-only views (08_history_copy_followup, recommendation B) ---------------------


def test_the_rule_gets_read_only_views_that_read_like_the_request():
    seen = {}

    def approve(request, requester, history):
        seen["json"] = json.dumps(request, sort_keys=True)
        seen["y_is_list"] = isinstance(request["y"], list) and request["y"] == list(request["y"])
        seen["copy"] = copy.deepcopy(request)
        for attempt in (lambda: request.__setitem__("y", []), lambda: request["y"].append("1"),
                        lambda: request.update(sid="x"), lambda: request["attributes"].sort(),
                        lambda: request.pop("id"), lambda: request["servers"].__setitem__(0, "x")):
            try:
                attempt()
                seen["mutated"] = True
            except TypeError:
                pass
        return True

    _, _, _, key_servers, comb, _ = _system(approve=approve)
    request = comb.new_request("alice", [1, 2, 3], ["org:A"], ["ks1", "ks2", "ks3"])
    key_servers["ks1"].issue(request, authenticated_requester="alice")
    assert seen["json"] == json.dumps(request, sort_keys=True) and seen["y_is_list"]
    assert "mutated" not in seen
    mutable = seen["copy"]
    mutable["y"].append("4")
    assert type(mutable) is dict and type(mutable["y"]) is list
    stored = key_servers["ks1"]._history[0][1]
    assert stored == request and stored["y"] == ["1", "2", "3"]


def test_the_rule_cannot_change_the_stored_history():
    def careless(request, requester, history):
        for _, earlier in history:
            try:
                earlier["sid"] = "mallory"
            except TypeError:
                pass
            try:
                earlier["y"][0] = "9"
            except TypeError:
                pass
        return True

    _, _, _, key_servers, comb, _ = _system(approve=careless)
    requests = [comb.new_request(sid, [i, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
                for i, sid in enumerate(("alice", "bob", "carol"), 1)]
    for r in requests:
        key_servers["ks1"].issue(r, authenticated_requester=r["sid"])
    assert [(who, dict(entry)) for who, entry in key_servers["ks1"]._history] == \
        [(r["sid"], r) for r in requests]


@pytest.mark.parametrize("error", [ValueError("no"), KeyError("y"), RuntimeError, TypeError])
def test_an_exception_of_the_rule_refuses_and_leaves_no_record(error):
    calls = []

    def approve(request, requester, history):
        calls.append(request["sid"])
        if request["sid"] == "mallory":
            raise error
        return True

    _, _, _, key_servers, comb, _ = _system(approve=approve)
    bad = comb.new_request("mallory", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError, match="raised"):
        key_servers["ks1"].issue(bad, authenticated_requester="mallory")
    server = key_servers["ks1"]
    assert not server._contents and not server._ids and not server._history
    good = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    server.issue(good, authenticated_requester="alice")
    assert len(server._history) == 1 and calls == ["mallory", "alice"]


def test_a_history_kept_by_the_rule_does_not_grow():
    kept = []

    def approve(request, requester, history):
        kept.append(history)
        return True

    _, _, _, key_servers, comb, _ = _system(approve=approve)
    for sid in ("alice", "bob", "carol"):
        r = comb.new_request(sid, [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
        key_servers["ks1"].issue(r, authenticated_requester=sid)
    assert [len(h) for h in kept] == [0, 1, 2] and type(kept[0]) is tuple


def test_concurrent_distinct_requests_are_all_accepted_once():
    import threading

    lengths = []

    def approve(request, requester, history):
        lengths.append(len(history))
        return True

    _, _, _, key_servers, comb, _ = _system(approve=approve)
    requests = [comb.new_request("user{}".format(i), [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"]) for i in range(12)]
    barrier = threading.Barrier(len(requests), timeout=5)
    errors = []

    def send(r):
        barrier.wait()
        try:
            key_servers["ks1"].issue(r, authenticated_requester=r["sid"])
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=send, args=(r,)) for r in requests]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    history = key_servers["ks1"]._history
    assert not errors and len(history) == len(requests) and sorted(lengths) == list(range(len(requests)))
    assert sorted(entry["sid"] for _, entry in history) == sorted(r["sid"] for r in requests)


def test_frozen_entries_pickle_to_plain_data():
    import pickle

    frozen = ki._freeze({"a": [1, {"b": ["c"]}], "d": "e"})
    again = pickle.loads(pickle.dumps(frozen))
    assert again == {"a": [1, {"b": ["c"]}], "d": "e"}
    assert type(again) is dict and type(again["a"]) is list and type(again["a"][1]) is dict
    assert copy.copy(frozen) == frozen and type(copy.copy(frozen)) is dict


# -- fork (integrate/1.0 review INT-2, test from review/integrate-1.0 b43d2b6, adapted to the rule chosen) --------


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_int2_a_key_server_copied_into_a_forked_child_refuses_without_hanging():
    """A child forked while another thread of the parent is inside issue() (the approval rule runs under the
    server's lock) must not hang on the copied lock, and its copy refuses to issue (FESchemeError): its records
    would diverge from the parent's. The reviewer's probe asserted that the child issues; the fix chosen refuses
    instead (see roadmap/v0.7/09). A hang ends as SIGALRM."""
    import signal
    import threading

    inside, release = threading.Event(), threading.Event()

    def approve(request, requester, history):
        if request["sid"] == "slow":
            inside.set()
            release.wait(10)
        return True

    _, _, _, key_servers, comb, _ = _system(t=2, servers=["a", "b"], approve=approve)
    server = key_servers["a"]
    slow = threading.Thread(target=lambda: server.issue(comb.new_request("slow", [1, 0, 0], ["org:A"], ["a", "b"]),
                                                        "slow"))
    slow.start()
    try:
        assert inside.wait(5)
        pid = os.fork()
        if pid == 0:
            signal.alarm(5)
            try:
                server.issue(comb.new_request("bob", [1, 2, 3], ["org:A"], ["a", "b"]), "bob")
                os._exit(4)                      # issued: the copy's records would diverge
            except FESchemeError:
                os._exit(0)                      # refused at once
            except BaseException:
                os._exit(3)
        _, status = os.waitpid(pid, 0)
    finally:
        release.set()
        slow.join()
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, (
        status, os.WTERMSIG(status) if os.WIFSIGNALED(status) else None)
    # the parent's server is unaffected
    r = comb.new_request("carol", [1, 0, 0], ["org:A"], ["a", "b"])
    comb.combine(r, [key_servers[s].issue(r, "carol") for s in ("a", "b")])


def test_a_key_server_cannot_be_copied_or_pickled(system):
    import pickle

    server = system[3]["ks1"]
    for attempt in (lambda: copy.copy(server), lambda: copy.deepcopy(server), lambda: pickle.dumps(server)):
        with pytest.raises(TypeError, match="records apart"):
            attempt()
