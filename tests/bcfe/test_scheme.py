"""The bounded-key scheme ``bcfe/gc_ristretto255`` end to end (:mod:`pyfe4ai.schemes.bcfe.gc_ristretto255`):
issuance h1 and h2 with their order of checks, deduplication, the cap, the issuance policy, the slot list and
its versions, the checks of the public parameters, keys and ciphertexts, CheckAuth, the secrets that never reach
the public records, and the reload of an instance. Small circuits keep the Python backend fast."""

from __future__ import annotations

import copy
import json
import os
import pickle
import random

import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai import circuits
from pyfe4ai.core import registry
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats import garbled
from pyfe4ai.policy import parse
from pyfe4ai.schemes.bcfe import BCFEGCRistretto255, BCFEGCRistretto255KeyGenerator
from pyfe4ai.schemes.bcfe import gc_ristretto255 as scheme

SCHEME = "bcfe/gc_ristretto255"
SMALL = circuits.linear_threshold(n=2, bw=2, bx=2)  # m = 8 key bits
W_SMALL = {"weights": [1, -1], "bias": 2}


@pytest.fixture(params=["python", "native"])
def backend_name(request):
    try:
        with native_backend.backend(request.param):
            yield request.param
    except ImportError:
        pytest.skip("the native extension is not built")


def make(tmp_path, circuit=SMALL, **options):
    kg, crypto = pyfe4ai.create(SCHEME, circuit=circuit, cache_dir=str(tmp_path), **options)
    kg.setup()
    return kg, crypto


def random_values(layout, rng):
    out = {}
    for f in layout.fields:
        vals = [rng.randint(f.min_value, f.max_value) for _ in range(f.count)]
        out[f.name] = vals[0] if f.count == 1 else vals
    return out


# ---------------------------------------------------------------------------
# end to end


@pytest.mark.parametrize("data_wires", ["hardwired", "labels"])
@pytest.mark.parametrize("circuit", [circuits.linear_threshold(n=3, bw=4, bx=4),
                                     circuits.linear_score(n=2, bw=3, bx=3, shift=1)], ids=["threshold", "score"])
def test_h1_end_to_end(tmp_path, backend_name, data_wires, circuit):
    rng = random.Random(1)
    kg, BCFE = make(tmp_path, circuit, data_wires=data_wires, max_keys=3)
    assert (type(kg), BCFE) == (BCFEGCRistretto255KeyGenerator, BCFEGCRistretto255)
    ws = [random_values(circuit.key_layout, rng) for _ in range(2)]
    keys = [kg.issue_key(w, requester="holder-{}".format(i)) for i, w in enumerate(ws)]
    pp = kg.get_public_parameters()
    assert pp["slot_list_version"] == 2 and pp["data_wires"] == data_wires
    json.dumps(pp), json.dumps(keys)                                  # JSON-able
    crypto = BCFE({"keys": {"pp": pp}})
    for _ in range(3):
        x = random_values(circuit.data_layout, rng)
        ct = crypto.encrypt(x)
        json.dumps(ct)
        assert len(ct["components"]) == 2
        for w, key in zip(ws, keys):
            assert crypto.decrypt(ct, key) == circuit.evaluate(w, x)


@pytest.mark.parametrize("data_wires", ["hardwired", "labels"])
def test_h2_end_to_end(tmp_path, backend_name, data_wires):
    rng = random.Random(2)
    circuit = circuits.linear_threshold(n=3, bw=4, bx=4)
    kg, BCFE = make(tmp_path, circuit, data_wires=data_wires, key_mode="h2")
    pp = kg.get_public_parameters()
    holder = BCFE.key_holder(pp)
    w = random_values(circuit.key_layout, rng)
    request = holder.request(w)
    json.dumps(request)
    grant = kg.authorize(request, requester="bob", attributes={"org:bank"}, label="credit-v3")
    json.dumps(grant)
    key = holder.finish(grant)
    pp = kg.get_public_parameters()
    assert pp["slots"][0]["label"] == "credit-v3" and key["slot_id"] == pp["slots"][0]["slot_id"]
    crypto = BCFE({"keys": {"pp": pp}})
    for _ in range(3):
        x = random_values(circuit.data_layout, rng)
        assert crypto.decrypt(crypto.encrypt(x), key) == circuit.evaluate(w, x)


def test_create_by_id_and_registry_entry(tmp_path):
    entry = registry.get_scheme(SCHEME)
    assert entry.has(registry.BOUNDED_KEYS) and entry.kind == registry.EC
    assert entry.status.status == registry.EXPERIMENTAL and entry.status.fidelity == registry.ADAPTED
    kg, crypto = pyfe4ai.create(SCHEME, {"circuit": SMALL, "cache_dir": str(tmp_path)})
    assert isinstance(kg, entry.key_generator) and crypto is entry.crypto
    assert pyfe4ai.circuits is circuits


# ---------------------------------------------------------------------------
# deduplication, the cap, the policy


def test_h1_same_requester_and_w_get_the_same_key(tmp_path):
    kg, _ = make(tmp_path)
    key = kg.issue_key(W_SMALL, requester="alice", label="one")
    again = kg.issue_key({"weights": [1, -1], "bias": 2}, requester="alice", label="ignored")
    assert json.dumps(again) == json.dumps(key)                     # byte for byte
    pp = kg.get_public_parameters()
    assert pp["slot_list_version"] == 1 and pp["slots"][0]["label"] == "one"
    other = kg.issue_key(W_SMALL, requester="carol")                   # another requester: a slot of its own
    assert other["slot_id"] != key["slot_id"] and kg.get_public_parameters()["slot_list_version"] == 2
    third = kg.issue_key({"weights": [1, 1], "bias": 2}, requester="alice")  # another w: a new slot
    assert third["slot_id"] not in (key["slot_id"], other["slot_id"])


def test_h2_same_requester_and_w_get_the_same_grant(tmp_path):
    kg, BCFE = make(tmp_path, key_mode="h2")
    holder = BCFE.key_holder(kg.get_public_parameters())
    request = holder.request(W_SMALL)
    grant = kg.authorize(request, requester="alice")
    assert kg.authorize(request, requester="alice") == grant
    with pytest.raises(FEKeyError, match="other public keys"):     # a new request: the slot's keys stay
        kg.authorize(holder.request(W_SMALL), requester="alice")
    other = kg.authorize(holder.request(W_SMALL), requester="carol")
    assert other["slot"]["slot_id"] != grant["slot"]["slot_id"]
    assert kg.get_public_parameters()["slot_list_version"] == 2
    holder.finish(grant), holder.finish(other)


@pytest.mark.parametrize("key_mode", ["h1", "h2"])
def test_the_cap_gives_the_same_error_for_a_hit_and_a_miss(tmp_path, key_mode):
    kg, BCFE = make(tmp_path, max_keys=2, key_mode=key_mode)
    holder = BCFE.key_holder(kg.get_public_parameters())

    def ask(w, requester):
        if key_mode == "h1":
            return kg.issue_key(w, requester=requester)
        return kg.authorize(holder.request(w), requester=requester)

    ask(W_SMALL, "alice")
    ask(W_SMALL, "bob")
    errors = []
    for w, requester in ((W_SMALL, "alice"), ({"weights": [0, 0], "bias": 0}, "dave")):  # hit, then miss
        with pytest.raises(FEKeyError) as info:
            ask(w, requester)
        errors.append(str(info.value))
    assert errors[0] == errors[1] and "max_keys = 2" in errors[0]
    assert kg.get_public_parameters()["slot_list_version"] == 2
    kg.set_max_keys(3)                                                 # raised: one more slot
    ask({"weights": [0, 0], "bias": 0}, "dave")
    with pytest.raises(FEValidationError, match="below the 3 slots"):
        kg.set_max_keys(2)
    assert kg.get_public_parameters()["max_keys"] == 3


def test_the_policy_is_checked_before_any_slot_is_reserved(tmp_path):
    kg, _ = make(tmp_path, issuance_policy=parse("and(org:bank, role:scoring)"))
    with pytest.raises(FEKeyError, match="issuance policy"):
        kg.issue_key(W_SMALL, requester="mallory", attributes={"org:bank"})
    assert kg.get_public_parameters()["slot_list_version"] == 0
    key = kg.issue_key(W_SMALL, requester="alice", attributes=["org:bank", "role:scoring", "org:extra"])
    assert key["slot_id"] == 0                                         # the refused request burned no id
    slot = kg.get_public_parameters()["slots"][0]
    assert slot["policy_digest"] == pyfe4ai.policy.policy_digest(parse("and(org:bank, role:scoring)"))
    assert "attribute_set_digest" not in slot                          # private: the authority's audit
    assert kg._store.private(0)["attribute_set_digest"] == bytes.fromhex(pyfe4ai.policy.attribute_set_digest(
        ["org:bank", "role:scoring", "org:extra"]))


def test_without_kept_keys_a_repeated_request_is_refused(tmp_path):
    kg, _ = make(tmp_path, keep_issued_keys=False)
    kg.issue_key(W_SMALL, requester="alice")
    with pytest.raises(FEKeyError, match="already issued"):
        kg.issue_key(W_SMALL, requester="alice")
    private = kg._store.private(0)
    assert set(private) == {"key_mode", "max_keys", "k", "attribute_set_digest"}  # no scalars, no w kept


def test_issuance_refuses_bad_arguments_before_anything_is_reserved(tmp_path):
    kg, _ = make(tmp_path)
    for w in ({"weights": [1, 9], "bias": 0}, {"weights": [1, 1]}, {"weights": [1, 1], "bias": 0, "x": 1}):
        with pytest.raises(FEValidationError):
            kg.issue_key(w, requester="alice")
    for requester in ("", None, b"alice", 3):
        with pytest.raises(FEValidationError, match="requester"):
            kg.issue_key(W_SMALL, requester=requester)
    with pytest.raises(FEValidationError):
        kg.issue_key(W_SMALL, requester="alice", attributes="org:bank")
    with pytest.raises(FEValidationError):
        kg.issue_key(W_SMALL, requester="alice", label="a\nb")
    assert kg.issue_key(W_SMALL, requester="alice")["slot_id"] == 0
    with pytest.raises(FEKeyError, match="key_mode='h1'"):
        kg.authorize({}, requester="alice")


# ---------------------------------------------------------------------------
# the slot list and the ciphertexts


def test_an_old_ciphertext_and_a_new_key(tmp_path):
    kg, BCFE = make(tmp_path)
    kg.issue_key(W_SMALL, requester="alice", label="first")
    old_ct = BCFE({"keys": {"pp": kg.get_public_parameters()}}).encrypt({"x": [1, 1]})
    new_key = kg.issue_key({"weights": [1, 0], "bias": 0}, requester="bob", label="second")
    pp = kg.get_public_parameters()
    crypto = BCFE({"keys": {"pp": pp}})
    with pytest.raises(FEKeyError, match=r"older than the key \(keys decrypt only ciphertexts made after they were "
                                         r"issued\)"):
        crypto.decrypt(old_ct, new_key)
    first_only = crypto.encrypt({"x": [1, 1]}, slots=["first"])
    with pytest.raises(FEKeyError, match="no component for this slot"):
        crypto.decrypt(first_only, new_key)
    assert crypto.decrypt(crypto.encrypt({"x": [1, 1]}), new_key) == {"decision": 1}


def test_slots_selects_a_subset(tmp_path):
    kg, BCFE = make(tmp_path)
    keys = [kg.issue_key(W_SMALL, requester=r, label=label) for r, label in
            (("a", "group"), ("b", "group"), ("c", "solo"))]
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    ids = lambda ct: [c["slot_id"] for c in ct["components"]]  # noqa: E731
    assert ids(crypto.encrypt({"x": [1, 1]}, slots=["group"])) == [keys[0]["slot_id"], keys[1]["slot_id"]]
    assert ids(crypto.encrypt({"x": [1, 1]}, slots=[keys[2]["slot_id"]])) == [keys[2]["slot_id"]]
    assert ids(crypto.encrypt({"x": [1, 1]}, slots="solo")) == [keys[2]["slot_id"]]
    assert ids(crypto.encrypt({"x": [1, 1]})) == [k["slot_id"] for k in keys]
    for bad in (["nope"], [99], [True], [1.0]):
        with pytest.raises(FEValidationError):
            crypto.encrypt({"x": [1, 1]}, slots=bad)


def test_every_ciphertext_and_component_is_fresh(tmp_path):
    kg, BCFE = make(tmp_path)
    kg.issue_key(W_SMALL, requester="a")
    kg.issue_key(W_SMALL, requester="b")
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    one, two = crypto.encrypt({"x": [1, 1]}), crypto.encrypt({"x": [1, 1]})
    assert one["ct_nonce"] != two["ct_nonce"] and len(bytes.fromhex(one["ct_nonce"])) == 16
    rs = [c["R"] for ct in (one, two) for c in ct["components"]]
    assert len(set(rs)) == 4


def test_a_raised_cap_keeps_older_ciphertexts_decryptable(tmp_path):
    """The decryptor's public parameters may be newer than the encryptor's (another max_keys, more slots): the
    ciphertext's own pp digest goes into the hashes."""
    kg, BCFE = make(tmp_path)
    key = kg.issue_key(W_SMALL, requester="alice")
    ct = BCFE({"keys": {"pp": kg.get_public_parameters()}}).encrypt({"x": [1, -1]})
    kg.set_max_keys(20)
    kg.issue_key(W_SMALL, requester="bob")
    newer = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    assert newer.decrypt(ct, key) == {"decision": 1}


def test_no_slot_no_ciphertext(tmp_path):
    kg, BCFE = make(tmp_path)
    with pytest.raises(FEValidationError, match="no slot"):
        BCFE({"keys": {"pp": kg.get_public_parameters()}}).encrypt({"x": [1, 1]})


# ---------------------------------------------------------------------------
# fail closed: the public parameters, the key, the ciphertext


def _pp_with_two_slots(tmp_path):
    kg, BCFE = make(tmp_path)
    kg.issue_key(W_SMALL, requester="a", label="one")
    key = kg.issue_key({"weights": [0, 1], "bias": 1}, requester="b", label="two")
    return kg, BCFE, kg.get_public_parameters(), key


def _flip_b64(value: str) -> str:
    data = bytearray(garbled.from_b64(value, "field"))
    data[0] ^= 1
    return garbled.b64(bytes(data))


@pytest.mark.parametrize("tamper", [
    "pks", "slot_digest", "pp_digest", "max_keys", "version", "order", "label", "drop_slot", "extra_field",
    "circuit", "nonce", "marker_missing", "marker_unknown", "data_wires",
])
def test_tampered_public_parameters_are_refused(tmp_path, tamper):
    _, BCFE, pp, _ = _pp_with_two_slots(tmp_path)
    bad = copy.deepcopy(pp)
    slot = bad["slots"][1]
    if tamper == "pks":
        slot["pks"] = _flip_b64(slot["pks"])
    elif tamper == "slot_digest":
        slot["slot_digest"] = "00" * 32
    elif tamper == "pp_digest":
        bad["pp_digest"] = "11" * 32
    elif tamper == "max_keys":
        bad["max_keys"] += 1
    elif tamper == "version":
        bad["slot_list_version"] = 1
    elif tamper == "order":
        bad["slots"].reverse()
    elif tamper == "label":
        slot["label"] = "three"
    elif tamper == "drop_slot":
        bad["slots"].pop(0)
        bad["slot_list_version"] = 1
    elif tamper == "extra_field":
        slot["k"] = "00" * 16
    elif tamper == "circuit":
        bad["circuit_encoding"] = garbled.b64(circuits.linear_threshold(n=2, bw=2, bx=3).circuit.encode())
    elif tamper == "nonce":
        bad["setup_nonce"] = "22" * 32
    elif tamper == "marker_missing":
        del bad["gc_hash"]
    elif tamper == "marker_unknown":
        bad["kem"] = "ristretto255-hashed-elgamal-v2"
    elif tamper == "data_wires":
        bad["data_wires"] = "labels"
    with pytest.raises(FEValidationError):
        BCFE({"keys": {"pp": bad}})
    crypto = BCFE({"keys": {"pp": pp}})                               # checked again at every use
    pp.clear()
    pp.update(bad)
    with pytest.raises(FEValidationError):
        crypto.encrypt({"x": [1, 1]})


def test_the_key_and_the_ciphertext_are_checked(tmp_path):
    kg, BCFE, pp, key = _pp_with_two_slots(tmp_path)
    crypto = BCFE({"keys": {"pp": pp}})
    ct = crypto.encrypt({"x": [1, -1]})
    assert crypto.decrypt(ct, key) == {"decision": 1}
    for change, error in (
        (lambda k: k.update(scheme="sife/ddh"), FEKeyError),
        (lambda k: k.pop("bcfe_version"), FEKeyError),
        (lambda k: k.update(bcfe_version=2), FEKeyError),
        (lambda k: k.update(slot_digest="00" * 32), FEKeyError),          # not the slot of the pp
        (lambda k: k.update(circuit_digest="00" * 32), FEKeyError),
        (lambda k: k.update(scalars=k["scalars"][:-4]), FEKeyError),
        (lambda k: k.update(slot_list_version=True), FEKeyError),
    ):
        bad = copy.deepcopy(key)
        change(bad)
        with pytest.raises(error):
            crypto.decrypt(ct, bad)
    other_kg, _ = make(tmp_path / "other")
    with pytest.raises(FEKeyError, match="another setup"):
        crypto.decrypt(ct, other_kg.issue_key(W_SMALL, requester="a"))
    for change in (
        lambda c: c.update(circuit_digest="00" * 32),
        lambda c: c.update(setup_digest="00" * 32),
        lambda c: c.update(pp_digest="00" * 31),
        lambda c: c.pop("slot_list_version"),
        lambda c: c["components"].append(dict(c["components"][0])),    # two components of one slot
        lambda c: c["components"][1].update(R=c["components"][0]["R"]),  # one R twice
    ):
        bad = copy.deepcopy(ct)
        change(bad)
        with pytest.raises(FEValidationError):
            crypto.decrypt(bad, key)


@pytest.mark.parametrize("field", ["R", "label_cts", "tables", "decode", "data_active", "R_point"])
def test_a_component_that_does_not_decrypt_raises_one_error(tmp_path, field):
    _, BCFE, pp, key = _pp_with_two_slots(tmp_path)
    crypto = BCFE({"keys": {"pp": pp}})
    ct = crypto.encrypt({"x": [1, -1]})
    component = next(c for c in ct["components"] if c["slot_id"] == key["slot_id"])
    if field == "R_point":
        component["R"] = "ff" * 32                                     # not a canonical encoding
    elif field == "R":
        component["R"] = component["R"][:-2]
    else:
        component[field] = garbled.b64(garbled.from_b64(component[field], field) + b"\x00" * 16)
    with pytest.raises(FESchemeError) as info:
        crypto.decrypt(ct, key)
    assert str(info.value) == scheme.DECRYPTION_FAILED


# ---------------------------------------------------------------------------
# secrets stay private


def _encodings(secret: bytes) -> list:
    return [secret.hex(), garbled.b64(secret)]


@pytest.mark.parametrize("key_mode", ["h1", "h2"])
def test_public_records_hold_no_slot_secret_scalar_or_dedup_key(tmp_path, key_mode):
    kg, BCFE = make(tmp_path, key_mode=key_mode)
    if key_mode == "h1":
        key = kg.issue_key(W_SMALL, requester="alice")
    else:
        holder = BCFE.key_holder(kg.get_public_parameters())
        grant = kg.authorize(holder.request(W_SMALL), requester="alice")
        key = holder.finish(grant)
    pp = kg.get_public_parameters()
    ct = BCFE({"keys": {"pp": pp}}).encrypt({"x": [1, 1]})
    private = kg._store.private(key["slot_id"])
    setup = scheme._parse_pp(pp).setup
    w_packed = garbled.from_b64(key["w"], "w")
    dedup = garbled.dedup_key(SCHEME, setup.circuit_digest, "alice", w_packed)
    scalars = garbled.from_b64(key["scalars"], "scalars")
    secrets = [private["k"], dedup] + [scalars[i:i + 32] for i in range(0, len(scalars), 32)]
    public = json.dumps(pp) + json.dumps(ct)
    files = {name: open(os.path.join(kg.directory, name), "rb").read()
             for name in (scheme.SETUP_FILE, "slots.json")}
    for secret in secrets:
        for text in _encodings(secret):
            assert text not in public
            assert all(text.encode() not in data for data in files.values())
        assert all(secret not in data for data in files.values())
    assert os.stat(os.path.join(kg.directory, "slots.private.json")).st_mode & 0o077 == 0
    assert os.stat(os.path.join(kg.directory, scheme.AUTHORITY_FILE)).st_mode & 0o077 == 0
    if key_mode == "h1":                                             # the kept key: m + 1 scalars, no other
        assert private["scalars"] == scalars and set(private) == {"key_mode", "max_keys", "k", "attribute_set_digest",
                                                                  "w", "scalars"}
    else:                                                            # the authority holds no scalar of an h2 slot
        assert set(private) == {"key_mode", "max_keys", "k", "attribute_set_digest"}
    authority = open(os.path.join(kg.directory, scheme.AUTHORITY_FILE)).read()
    assert all(text not in authority for secret in secrets for text in _encodings(secret))


def test_the_unselected_keys_are_recomputed_from_k(tmp_path):
    kg, _ = make(tmp_path)
    key = kg.issue_key(W_SMALL, requester="alice")
    pp = kg.get_public_parameters()
    setup = scheme._parse_pp(pp).setup
    k = kg._store.private(key["slot_id"])["k"]
    pks = garbled.from_b64(pp["slots"][0]["pks"], "pks")
    w_packed = garbled.from_b64(key["w"], "w")
    assert native_backend.ops().bcfe_unselected_ok(setup.setup_digest, key["slot_id"], w_packed, k, pks, m=setup.m)
    assert not native_backend.ops().bcfe_unselected_ok(setup.setup_digest, key["slot_id"], w_packed, bytes(16), pks,
                                                       m=setup.m)


# ---------------------------------------------------------------------------
# h2: the key holder and CheckAuth


def _resealed(grant, setup, pks: bytes) -> dict:
    """The grant with another table, its slot digest recomputed (a consistent but malicious authority)."""
    bad = copy.deepcopy(grant)
    slot = bad["slot"]
    slot["pks"] = garbled.b64(pks)
    policy = None if slot["policy_digest"] is None else bytes.fromhex(slot["policy_digest"])
    slot["slot_digest"] = garbled.slot_digest(setup.setup_digest, slot["slot_id"], slot["label"], policy, pks).hex()
    return bad


def test_checkauth_refuses_a_moved_key_and_a_wrong_k(tmp_path, backend_name):
    kg, BCFE = make(tmp_path, key_mode="h2")
    pp = kg.get_public_parameters()
    setup = scheme._parse_pp(pp).setup
    holder = BCFE.key_holder(pp)
    grant = kg.authorize(holder.request(W_SMALL), requester="alice")
    pks = garbled.from_b64(grant["slot"]["pks"], "pks")
    table = [pks[i:i + 32] for i in range(0, len(pks), 32)]
    table[0], table[1] = table[1], table[0]                            # the holder's key of bit 0 moved
    with pytest.raises(FEKeyError, match="CheckAuth failed: the slot of the grant does not hold"):
        holder.finish(_resealed(grant, setup, b"".join(table)))
    with pytest.raises(FEKeyError, match="CheckAuth failed: the unselected positions"):
        holder.finish(dict(grant, k=os.urandom(16).hex()))
    with pytest.raises(FEKeyError, match="slot digest"):                # a table changed without its digest
        holder.finish(dict(grant, slot=dict(grant["slot"], pks=garbled.b64(b"".join(table)))))
    with pytest.raises(FEKeyError, match="pks must be"):                 # 2m + 3 entries: exactly 2m + 1 wanted
        holder.finish(_resealed(grant, setup, pks + pks[:64]))
    key = holder.finish(grant)                                           # the request stayed open
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    assert crypto.decrypt(crypto.encrypt({"x": [1, -1]}), key) == {"decision": 1}
    with pytest.raises(FEKeyError, match="CheckAuth failed"):            # finished: no open request left
        holder.finish(grant)


def test_the_authority_checks_the_request(tmp_path):
    kg, BCFE = make(tmp_path, key_mode="h2")
    holder = BCFE.key_holder(kg.get_public_parameters())
    request = holder.request(W_SMALL)
    pks = garbled.from_b64(request["pks"], "pks")
    for bad in (
        dict(request, pks=garbled.b64(bytes(32) + pks[32:])),           # the identity (I8)
        dict(request, pks=garbled.b64(b"\xff" * 32 + pks[32:])),        # not an encoding
        dict(request, pks=garbled.b64(pks[:-32])),                      # m keys, not m + 1
        dict(request, setup_digest="00" * 32),
        dict(request, w={"weights": [1, 5], "bias": 0}),
        {k: v for k, v in request.items() if k != "bcfe_version"},
    ):
        with pytest.raises(FEValidationError):
            kg.authorize(bad, requester="alice")
    assert kg.get_public_parameters()["slot_list_version"] == 0
    with pytest.raises(FEKeyError, match="key_mode='h2'"):
        kg.issue_key(W_SMALL, requester="alice")


def test_the_key_holder_keeps_its_secrets(tmp_path):
    kg, BCFE = make(tmp_path, key_mode="h2")
    holder = BCFE.key_holder(kg.get_public_parameters())
    holder.request(W_SMALL)
    assert "open requests=1" in repr(holder)
    for dump in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="secret keys"):
            dump(holder)


# ---------------------------------------------------------------------------
# reload, config


def test_an_instance_is_resumed_from_its_directory(tmp_path):
    kg, BCFE = make(tmp_path, max_keys=3, issuance_policy=parse("org:bank"))
    key = kg.issue_key(W_SMALL, requester="alice", attributes={"org:bank"})
    kg.set_max_keys(5)
    pp = kg.get_public_parameters()
    again = BCFEGCRistretto255KeyGenerator.open(kg.directory)
    assert again.get_public_parameters() == pp
    assert (again.max_keys, again.key_mode, again.data_wires) == (5, "h1", "hardwired")
    assert again.issue_key(W_SMALL, requester="alice", attributes={"org:bank"}) == key
    with pytest.raises(FEKeyError, match="issuance policy"):
        again.issue_key(W_SMALL, requester="bob")
    new = again.issue_key({"weights": [0, 1], "bias": 0}, requester="bob", attributes={"org:bank"})
    assert new["slot_id"] == 1 and kg.get_public_parameters() == again.get_public_parameters()
    with pytest.raises(FEConfigError):
        BCFEGCRistretto255KeyGenerator.open(tmp_path)


def test_config_is_checked(tmp_path):
    with pytest.raises(FEConfigError, match="circuit"):
        pyfe4ai.create(SCHEME, cache_dir=str(tmp_path))
    for options in ({"circuit": SMALL.circuit}, {"max_keys": 0}, {"max_keys": True}, {"data_wires": "x"},
                    {"key_mode": "h3"}, {"keep_issued_keys": 1}, {"issuance_policy": "org:bank"},
                    {"gc_hash": "aes"}, {"label_bits": 256}):
        with pytest.raises(FEValidationError):
            pyfe4ai.create(SCHEME, dict({"circuit": SMALL}, **options))
    kg, _ = pyfe4ai.create(SCHEME, circuit=SMALL, security_profile="standard", cache_dir=str(tmp_path))
    kg.setup()
    assert kg.get_public_parameters()["label_bits"] == 128
    with pytest.raises(FEConfigError):
        pyfe4ai.create(SCHEME, circuit=SMALL, security_profile="standard", ring_n=1024)
    with pytest.raises(FESchemeError, match="setup"):
        pyfe4ai.create(SCHEME, circuit=SMALL)[0].issue_key(W_SMALL, requester="a")


def test_the_markers_are_the_versions_module(tmp_path):
    from pyfe4ai.core import versions
    from pyfe4ai.primitives import bcfe_kem

    assert versions.BCFE_KEM in bcfe_kem.KEM_CODES and versions.BCFE_GC_HASH in bcfe_kem.GC_HASH_CODES
    kg, _ = make(tmp_path)
    pp = kg.get_public_parameters()
    assert (pp["bcfe_version"], pp["kem"], pp["gc"], pp["gc_hash"]) == (
        versions.BCFE_VERSION, versions.BCFE_KEM, versions.BCFE_GC, versions.BCFE_GC_HASH)
    assert list(pp) == list(garbled.BCFEPublicParameters.FIELDS)


# ---------------------------------------------------------------------------
# review: no issuance, setup or encryption on the tests' randomness (appended)


def _instances(tmp_path) -> list:
    return sorted(os.listdir(os.path.join(str(tmp_path), "authority", "bcfe")))


def test_the_tests_randomness_hook_stops_every_call_that_draws(tmp_path, backend_name, monkeypatch):
    kg, BCFE = make(tmp_path / "h1")
    kg2, _ = make(tmp_path / "h2", key_mode="h2")
    key = kg.issue_key(W_SMALL, requester="alice")
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    ct = crypto.encrypt({"x": [1, -1]})
    holder = BCFE.key_holder(kg2.get_public_parameters())
    request = holder.request(W_SMALL)
    grant = kg2.authorize(request, requester="alice")
    pending = holder.request({"weights": [0, 1], "bias": 0})
    before = (_instances(tmp_path / "h1"), kg.get_public_parameters(), kg2.get_public_parameters())

    monkeypatch.setattr(native_backend, "_random_override", lambda nbytes: b"\x01" * nbytes)
    assert native_backend.ops().random_override_active()
    calls = {
        "setup": kg.setup,
        "issue_key": lambda: kg.issue_key({"weights": [1, 1], "bias": 0}, requester="bob"),
        "issue_key (a repeat)": lambda: kg.issue_key(W_SMALL, requester="alice"),
        "authorize": lambda: kg2.authorize(pending, requester="bob"),
        "request": lambda: holder.request(W_SMALL),
        "encrypt": lambda: crypto.encrypt({"x": [1, 1]}),
    }
    for name, call in calls.items():
        with pytest.raises(FESchemeError) as info:
            call()
        assert str(info.value) == scheme.TEST_RANDOMNESS_REFUSED, name
    assert "pyfe4ai.backend._random_override is set: bcfe refuses to issue, set up or encrypt with test " \
           "randomness" == scheme.TEST_RANDOMNESS_REFUSED
    # what draws nothing still runs; nothing changed
    assert crypto.decrypt(ct, key) == {"decision": 1}
    assert holder.finish(grant)["slot_id"] == grant["slot"]["slot_id"]
    assert (_instances(tmp_path / "h1"), kg.get_public_parameters(), kg2.get_public_parameters()) == before

    monkeypatch.setattr(native_backend, "_random_override", None)  # cleared: every call works again
    assert not native_backend.ops().random_override_active()
    kg.setup()
    assert len(_instances(tmp_path / "h1")) == 2
    key = kg.issue_key(W_SMALL, requester="alice")
    assert kg2.authorize(pending, requester="bob")["slot"]["slot_id"] == 1
    holder.request(W_SMALL)
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    assert crypto.decrypt(crypto.encrypt({"x": [1, -1]}), key) == {"decision": 1}


# ---------------------------------------------------------------------------
# review: CheckAuth at the scheme layer finds the holder's keys out of place (appended)


def test_checkauth_refuses_a_slot_without_the_holders_keys_at_their_positions(tmp_path, backend_name):
    """The h' part of CheckAuth (``bcfe_unselected_ok``) passes when the authority puts another point at a
    selected position and leaves the unselected ones derived from ``k``; finish() checks the holder's own keys at
    ``(j, w_j)`` and ``(m, 0)`` and refuses. So does it for a slot whose holder keys sit where another function puts
    them."""
    from pyfe4ai.primitives.groups import ec_group as eg

    kg, BCFE = make(tmp_path, key_mode="h2")
    pp = kg.get_public_parameters()
    setup = scheme._parse_pp(pp).setup
    holder = BCFE.key_holder(pp)
    request = holder.request(W_SMALL)
    grant = kg.authorize(request, requester="alice")
    slot_id, k = grant["slot"]["slot_id"], bytes.fromhex(grant["k"])
    pks = garbled.from_b64(grant["slot"]["pks"], "pks")
    bits = SMALL.key_layout.encode(W_SMALL)
    w_packed = scheme.pack_bits(bits, setup.m)
    ops = native_backend.ops()
    other = bytes.fromhex(eg.to_hex(eg.mul_base(12345)))

    def at(table: bytes, index: int, point: bytes) -> bytes:
        return table[:32 * index] + point + table[32 * (index + 1):]

    for index in (2 * 3 + bits[3], 2 * setup.m):                       # a selected position, then (m, 0)
        forged = at(pks, index, other)
        assert ops.bcfe_unselected_ok(setup.setup_digest, slot_id, w_packed, k, forged, m=setup.m)
        with pytest.raises(FEKeyError, match="CheckAuth failed: the slot of the grant does not hold this holder's"):
            holder.finish(_resealed(grant, setup, forged))
    # the holder's keys placed for another function w' (bit 3 flipped), unselected positions derived from k
    flipped = list(bits)
    flipped[3] ^= 1
    placed = ops.bcfe_place_slot(setup.setup_digest, slot_id, scheme.pack_bits(flipped, setup.m),
                                 garbled.from_b64(request["pks"], "pks"), m=setup.m)
    forged = dict(_resealed(grant, setup, placed["pks"]), k=placed["k"].hex())
    with pytest.raises(FEKeyError, match="CheckAuth failed: the slot of the grant does not hold this holder's"):
        holder.finish(forged)
    assert holder.finish(grant)["slot_id"] == slot_id                  # the genuine grant still finishes


# ---------------------------------------------------------------------------
# selections, requester ids and versions refused before use (appended)


def test_a_slot_selection_is_a_label_an_id_or_an_iterable_of_them(tmp_path):
    kg, BCFE = make(tmp_path)
    key = kg.issue_key(W_SMALL, requester="a", label="one")
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    for good in ("one", 0, ["one"], (0,), {0, "one"}, (i for i in [0])):
        assert crypto.decrypt(crypto.encrypt({"x": [1, -1]}, slots=good), key) == {"decision": 1}
    for bad in (b"one", bytearray(b"\x00"), {"one": 1}, 1.0, True, object(), [b"one"], [None], [], memoryview(b"")):
        with pytest.raises(FEValidationError):
            crypto.encrypt({"x": [1, 1]}, slots=bad)


def test_a_requester_id_is_its_utf8_bytes(tmp_path):
    kg, _ = make(tmp_path)
    with pytest.raises(FEValidationError, match="UTF-8"):
        kg.issue_key(W_SMALL, requester="a\udc80")
    nfc, nfd = "é", "é"                                    # one name, two spellings: two requesters
    assert kg.issue_key(W_SMALL, requester=nfc)["slot_id"] != kg.issue_key(W_SMALL, requester=nfd)["slot_id"]


def test_recorded_versions_are_bounded(tmp_path):
    kg, BCFE = make(tmp_path, key_mode="h2")
    holder = BCFE.key_holder(kg.get_public_parameters())
    grant = kg.authorize(holder.request(W_SMALL), requester="a")
    for version in (2 ** 70, 2 ** 63, 2, 0):                          # slot 0: version 1 only
        with pytest.raises(FEKeyError, match="slot_list_version"):
            holder.finish(dict(grant, slot_list_version=version))
    key = holder.finish(grant)
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    ct = crypto.encrypt({"x": [1, -1]})
    for version in (2 ** 70, 2):
        with pytest.raises(FEKeyError, match="slot_list_version"):
            crypto.decrypt(ct, dict(key, slot_list_version=version))
    for version in (2 ** 63, 0):                                      # one component: version >= 1
        with pytest.raises(FEValidationError, match="slot_list_version"):
            crypto.decrypt(dict(ct, slot_list_version=version), key)
    assert crypto.decrypt(ct, key) == {"decision": 1}


def test_set_max_keys_and_the_pp_read_the_cap_under_the_store_lock(tmp_path, monkeypatch):
    kg, _ = make(tmp_path, max_keys=2)
    kg.issue_key(W_SMALL, requester="a")
    seen = []
    real = kg._store.run_locked
    monkeypatch.setattr(kg._store, "run_locked", lambda fn: seen.append(fn) or real(fn))
    kg.set_max_keys(1)
    pp = kg.get_public_parameters()
    assert len(seen) == 2 and pp["max_keys"] == 1 and pp["slot_list_version"] == 1
    with pytest.raises(FEKeyError, match="max_keys = 1"):
        kg.issue_key({"weights": [0, 0], "bias": 0}, requester="b")
    assert kg._store.private(0)["max_keys"] == 2                     # the cap decided at that slot's issue


# ---------------------------------------------------------------------------
# code review 2: the cap's bound, the instance files, open(), the key check, holder keys, concurrency (appended)


def _authority(kg) -> str:
    return os.path.join(kg.directory, scheme.AUTHORITY_FILE)


def _rewrite(path: str, change) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    change(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_max_keys_is_below_2_to_the_63_everywhere(tmp_path):
    for bad in (2 ** 63, 2 ** 64, 0, True):
        with pytest.raises(FEValidationError, match="max_keys"):
            pyfe4ai.create(SCHEME, circuit=SMALL, max_keys=bad, cache_dir=str(tmp_path))
    kg, BCFE = make(tmp_path, max_keys=2 ** 63 - 1)
    key = kg.issue_key(W_SMALL, requester="a")
    assert kg.get_public_parameters()["max_keys"] == 2 ** 63 - 1
    with pytest.raises(FEValidationError, match="max_keys"):
        kg.set_max_keys(2 ** 63)
    _rewrite(_authority(kg), lambda d: d.update(max_keys=2 ** 63))
    for call in (kg.get_public_parameters, lambda: kg.issue_key(W_SMALL, requester="b"),
                 lambda: BCFEGCRistretto255KeyGenerator.open(kg.directory)):
        with pytest.raises(FEConfigError, match="max_keys"):
            call()
    _rewrite(_authority(kg), lambda d: d.update(max_keys=4))
    assert kg.issue_key(W_SMALL, requester="a") == key


@pytest.mark.parametrize("change", [
    "drop:issuance_policy", "drop:key_mode", "drop:keep_issued_keys", "drop:security_profile", "drop:gc_hash",
    "drop:label_bits", "drop:sec_param", "drop:max_keys", "extra", "keep_issued_keys=yes", "key_mode=h3",
    "issuance_policy=5", "issuance_policy=not a policy", "security_profile=strong", "sec_param=True",
    "label_bits=256", "data_wires=labels", "setup_digest", "format_version=True",
])
def test_authority_json_is_parsed_fail_closed(tmp_path, change):
    kg, _ = make(tmp_path, key_mode="h2", keep_issued_keys=False, issuance_policy=parse("org:bank"),
                 security_profile="standard")
    name, _, value = change.partition("=")

    def edit(data):
        if name.startswith("drop:"):
            del data[name[5:]]
        elif name == "extra":
            data["comment"] = "an unknown field"
        elif name == "setup_digest":
            data["setup_digest"] = "00" * 32
        else:
            data[name] = {"yes": "yes", "True": True, "5": 5, "256": 256}.get(value, value)

    _rewrite(_authority(kg), edit)
    with pytest.raises(FEConfigError, match=scheme.AUTHORITY_FILE):
        BCFEGCRistretto255KeyGenerator.open(kg.directory)
    with pytest.raises(FEConfigError):
        kg.get_public_parameters()


def test_the_instance_files_get_the_store_file_checks(tmp_path):
    kg, _ = make(tmp_path)
    path = _authority(kg)
    os.chmod(path, 0o666)
    with pytest.raises(FEConfigError, match="mode 0666"):
        kg.issue_key(W_SMALL, requester="a")
    os.chmod(path, 0o600)
    elsewhere = os.path.join(str(tmp_path), "elsewhere.json")
    os.replace(path, elsewhere)
    os.symlink(elsewhere, path)
    with pytest.raises(FEConfigError, match="symbolic link"):
        kg.get_public_parameters()
    os.unlink(path)
    os.replace(elsewhere, path)
    setup_path = os.path.join(kg.directory, scheme.SETUP_FILE)
    _rewrite(setup_path, lambda d: d.update(comment="an unknown field"))
    with pytest.raises(FEConfigError, match="exactly the fields"):
        BCFEGCRistretto255KeyGenerator.open(kg.directory)


def test_open_checks_the_directory_before_reading_and_never_waits_on_a_fifo(tmp_path):
    kg, _ = make(tmp_path)
    os.chmod(kg.directory, 0o777)
    with pytest.raises(FEConfigError, match="lets other users write"):
        BCFEGCRistretto255KeyGenerator.open(kg.directory)
    os.chmod(kg.directory, 0o700)
    setup_path = os.path.join(kg.directory, scheme.SETUP_FILE)
    os.unlink(setup_path)
    os.mkfifo(setup_path, 0o644)
    with pytest.raises(FEConfigError, match="not a regular file"):
        BCFEGCRistretto255KeyGenerator.open(kg.directory)
    with pytest.raises(FEConfigError, match="does not exist"):
        BCFEGCRistretto255KeyGenerator.open(os.path.join(str(tmp_path), "nowhere"))


def test_a_key_that_does_not_open_its_listed_slot_gives_the_one_decryption_error(tmp_path, backend_name):
    circuit = circuits.linear_score(n=2, bw=3, bx=3)
    kg, BCFE = make(tmp_path, circuit)
    w = {"weights": [3, -2], "bias": 5}
    key = kg.issue_key(w, requester="alice")
    other = kg.issue_key({"weights": [1, 1], "bias": 0}, requester="bob")
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    x = {"x": [1, 2]}
    ct = crypto.encrypt(x)
    assert crypto.decrypt(ct, key) == circuit.evaluate(w, x)
    m = circuit.key_layout.n_bits
    bits = circuit.key_layout.encode(w)
    for flip in (0, m - 1):
        flipped = list(bits)
        flipped[flip] ^= 1
        damaged = dict(key, w=garbled.b64(scheme.pack_bits(flipped, m)))
        with pytest.raises(FESchemeError) as info:
            crypto.decrypt(ct, damaged)
        assert str(info.value) == scheme.DECRYPTION_FAILED
    swapped = dict(key, scalars=other["scalars"], w=other["w"])          # another slot's secrets, this slot's id
    with pytest.raises(FESchemeError) as info:
        crypto.decrypt(ct, swapped)
    assert str(info.value) == scheme.DECRYPTION_FAILED
    scalars = bytearray(garbled.from_b64(key["scalars"], "scalars"))
    scalars[-32:] = (int.from_bytes(scalars[-32:], "little") ^ 1).to_bytes(32, "little")  # the g bit's key
    with pytest.raises(FESchemeError):
        crypto.decrypt(ct, dict(key, scalars=garbled.b64(bytes(scalars))))
    # the verdicts are cached on the object, and left out of its copies and pickles
    calls = []
    real = scheme.eg.mul_base
    try:
        scheme.eg.mul_base = lambda k: calls.append(1) or real(k)
        assert crypto.decrypt(ct, key) == circuit.evaluate(w, x) and calls == []
        clone = pickle.loads(pickle.dumps(crypto))
        assert clone._key_checks == {} and clone.decrypt(ct, key) == circuit.evaluate(w, x)
        assert len(calls) == m + 1
    finally:
        scheme.eg.mul_base = real


def test_a_key_of_a_slot_the_public_parameters_do_not_list_yet_is_not_checked(tmp_path):
    kg, BCFE = make(tmp_path)
    old = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    key = kg.issue_key(W_SMALL, requester="alice")
    ct = BCFE({"keys": {"pp": kg.get_public_parameters()}}).encrypt({"x": [1, -1]})
    assert old.decrypt(ct, key) == {"decision": 1} and old._key_checks == {}


def test_holder_keys_must_be_pairwise_distinct(tmp_path):
    kg, BCFE = make(tmp_path, key_mode="h2")
    holder = BCFE.key_holder(kg.get_public_parameters())
    request = holder.request(W_SMALL)
    pks = garbled.from_b64(request["pks"], "pks")
    for bad in (pks[:32] * (len(pks) // 32), pks[:-32] + pks[:32]):  # one key everywhere; the g bit's = (0, w_0)'s
        with pytest.raises(FEKeyError, match="pairwise distinct"):
            kg.authorize(dict(request, pks=garbled.b64(bad)), requester="alice")
    assert kg.get_public_parameters()["slot_list_version"] == 0
    assert holder.finish(kg.authorize(request, requester="alice"))["slot_id"] == 0


def test_concurrent_finishes_of_one_grant_give_one_key(tmp_path):
    import threading

    kg, BCFE = make(tmp_path, key_mode="h2")
    for _ in range(5):
        holder = BCFE.key_holder(kg.get_public_parameters())
        grant = kg.authorize(holder.request(W_SMALL), requester="alice-{}".format(_))
        results, barrier = [], threading.Barrier(4)

        def finish():
            barrier.wait()
            try:
                results.append(holder.finish(grant)["slot_id"])
            except FEKeyError:
                results.append("refused")

        threads = [threading.Thread(target=finish) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert sorted(results, key=str) == sorted([grant["slot"]["slot_id"]] + ["refused"] * 3, key=str)


def test_records_nested_too_deeply_are_refused(tmp_path):
    kg, BCFE = make(tmp_path)
    pp = kg.get_public_parameters()
    deep = {}
    node = deep
    for _ in range(5000):
        node["a"] = {}
        node = node["a"]
    bad = copy.deepcopy(pp)
    bad["circuit"]["params"] = deep
    # refused either way: as too deep where the interpreter's recursion limit is reached while parsing, or (Python
    # 3.12+, whose limits differ) by the pp digest check
    with pytest.raises(FEValidationError, match="nested too deeply|params|pp_digest"):
        BCFE({"keys": {"pp": bad}})
    with pytest.raises(FEValidationError):
        BCFE.key_holder(bad)


# ---------------------------------------------------------------------------
# code review 3: the key-check cache under threads, the durable instance directory (appended)


def test_the_key_check_cache_is_safe_under_threads(tmp_path, monkeypatch):
    import sys
    import threading

    kg, BCFE = make(tmp_path)
    key = kg.issue_key(W_SMALL, requester="alice")
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    ct = crypto.encrypt({"x": [1, -1]})
    monkeypatch.setattr(scheme, "_KEY_CHECKS_KEPT", 8)                # evictions at every few calls
    rng = random.Random(3)
    damaged = []
    for _ in range(24):                                              # distinct tags, all refused
        scalars = bytearray(garbled.from_b64(key["scalars"], "scalars"))
        scalars[:32] = rng.randrange(1, 2 ** 252).to_bytes(32, "little")
        damaged.append(dict(key, scalars=garbled.b64(bytes(scalars))))
    errors, sizes = [], []
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def run(seed):
        local = random.Random(seed)
        try:
            for i in range(40):
                if i % 4 == 0:
                    assert crypto.decrypt(ct, key) == {"decision": 1}
                else:
                    try:
                        crypto.decrypt(ct, local.choice(damaged))
                        errors.append("a damaged key decrypted")
                    except FESchemeError as exc:
                        if str(exc) != scheme.DECRYPTION_FAILED:
                            errors.append(repr(exc))
                sizes.append(len(crypto._key_checks))
        except Exception as exc:  # noqa: BLE001 -- anything else is the finding
            errors.append(repr(exc))

    try:
        threads = [threading.Thread(target=run, args=(seed,)) for seed in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
    finally:
        sys.setswitchinterval(interval)
    assert errors == [] and max(sizes) <= 8 and len(crypto._key_checks) <= 8


def test_setup_flushes_the_new_instance_directory_entry(tmp_path, monkeypatch):
    flushed = []
    real = scheme.sync_directory
    monkeypatch.setattr(scheme, "sync_directory", lambda d: flushed.append(os.fspath(d)) or real(d))
    kg, _ = make(tmp_path)
    assert flushed == [os.path.dirname(kg.directory)]


def test_too_deep_says_it_may_be_the_callers_stack():
    with pytest.raises(FEValidationError, match=r"nested too deeply \(or the caller's stack is too deep\)"):
        with scheme._not_too_deep("a record"):
            raise RecursionError()
