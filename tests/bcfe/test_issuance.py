"""Issuance of ``bcfe/gc_ristretto255`` at the scheme level (:mod:`pyfe4ai.schemes.bcfe.gc_ristretto255`), beyond
the end-to-end tests of ``test_scheme.py``.

A key is a slot of the single-key scheme of Sahai and Seyalioglu ("Worry-Free Encryption: Functional Encryption
with Public Keys", ACM CCS 2010, SS10 Sec. 3.3), and the security argument of 0.6 needs each slot to hold exactly
one key (I5, I6). These tests check, through ``pyfe4ai.create`` and the key generator, holder and crypto classes:

* an issuance killed (``os._exit`` in a forked child) before the store is called, inside ``build``, after each
  rename into the instance directory, or after the store returned leaves an instance that reopens with
  :meth:`~pyfe4ai.schemes.bcfe.BCFEGCRistretto255KeyGenerator.open` and whose invariants hold: the public
  parameters verify, every listed slot has its private record, no slot id is handed out twice, and a retry of the
  same (requester, w) returns the stored key or grant (or, h1 without kept keys, "already issued") or makes the
  first slot of the request -- never a second one; the kill points are the scheme's calls into the store and the
  backend, and the checks do not assume an order of the store's writes;
* two processes issuing at once get distinct slots, and one slot with byte-identical keys (h1) or grants (h2) for
  the same request;
* no k, scalar or dedup key reaches the public parameters, the slot list, the setup, a ciphertext, a request, a
  log line or an exception (I14, I16), searched in raw, hex and base64 forms at every alignment;
* missing or unknown markers, truncated fields, a duplicated ``R``, a key of another setup are refused with the
  library's exceptions (02 section 3, rejection rules);
* a component moved between ciphertexts, slots or setups is either refused by a check or decrypts to an
  unrelated value (``decrypt_slot``: "unrelated bits"; I2, I10, I11);
* at the cap a repeated and a new request are indistinguishable and change no file; a refused request reserves
  no slot id (D-j, review V6-F10);
* an old ciphertext and a new h2 key give the explanatory ``FEKeyError``.

Small circuits (``linear_threshold(n=2, bw=3, bx=3)``, ``linear_score(n=2, bw=3, bx=3)`` where an output of
several bits is needed) keep the Python backend fast; the tests use the active backend (``PYFE4AI_BACKEND``).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import random
import shutil
import stat
import sys
import tempfile
import traceback

import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai import circuits
from pyfe4ai.core.exceptions import FEError, FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats import garbled
from pyfe4ai.policy import parse
from pyfe4ai.primitives import bcfe_kem as kem
from pyfe4ai.primitives.garbling import pack_bits, unpack_bits
from pyfe4ai.primitives.groups import ristretto255 as r255
from pyfe4ai.schemes.bcfe import BCFEGCRistretto255KeyGenerator as Generator
from pyfe4ai.schemes.bcfe import gc_ristretto255 as scheme
from pyfe4ai.schemes.common.slots import SlotStore

SCHEME = "bcfe/gc_ristretto255"
SMALL = circuits.linear_threshold(n=2, bw=3, bx=3)          # m = 12 key bits, n = 6 data bits, 1 output bit
SCORE = circuits.linear_score(n=2, bw=3, bx=3, shift=1)     # 7 output bits: a wrong result shows
M = SMALL.circuit.m
W0 = {"weights": [1, -1], "bias": 2}
W1 = {"weights": [-2, 3], "bias": -5}
W2 = {"weights": [3, 0], "bias": 7}
MODES = ["h1", "h1-no-kept-keys", "h2"]

needs_fork = pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
#: fork() of a process with threads: Python warns (3.12+); not the point of these tests
fork_warning = pytest.mark.filterwarnings("ignore::DeprecationWarning")

#: Exit codes of a forked child: killed at its point, finished without reaching it, failed.
KILLED, COMPLETED, FAILED = 0, 5, 3
#: The dedup key of the next-id probe (on a copy of an instance directory).
PROBE = hashlib.sha3_256(b"test_issuance: next slot id probe").digest()


# ---------------------------------------------------------------------------
# helpers


def make(tmp_path, circuit=SMALL, mode="h1", **options):
    """A set-up authority and the crypto class; *mode* is one of :data:`MODES`."""
    options.setdefault("key_mode", "h2" if mode == "h2" else "h1")
    options.setdefault("keep_issued_keys", mode != "h1-no-kept-keys")
    kg, crypto = pyfe4ai.create(SCHEME, circuit=circuit, cache_dir=str(tmp_path), **options)
    kg.setup()
    return kg, crypto


def random_values(layout, rng) -> dict:
    out = {}
    for f in layout.fields:
        vals = [rng.randint(f.min_value, f.max_value) for _ in range(f.count)]
        out[f.name] = vals[0] if f.count == 1 else vals
    return out


def packed_w(w, circuit=SMALL) -> bytes:
    return pack_bits(circuit.key_layout.encode(w), circuit.circuit.m)


def dedup_key(pp: dict, requester: str, w_packed: bytes) -> bytes:
    """The dedup key recomputed from its documented encoding (:mod:`pyfe4ai.core.formats.garbled`), not through
    the library: ``SHAKE-256("PYFE4AI-V06-BCFE-DEDUP" || u32(len(id)) || id || circuit_digest(32) || u32(len(r))
    || r || w)[0:32]``."""
    sid, r = SCHEME.encode("utf-8"), requester.encode("utf-8")
    data = (b"PYFE4AI-V06-BCFE-DEDUP" + len(sid).to_bytes(4, "little") + sid + bytes.fromhex(pp["circuit_digest"])
            + len(r).to_bytes(4, "little") + r + w_packed)
    return hashlib.shake_256(data).digest(32)


def forms(secret: bytes) -> list:
    """*secret* as it could sit in a file or a message: raw, hex (both cases), and standard and URL-safe base64
    at each of the three alignments inside a longer base64 string (the part fixed by *secret* alone)."""
    out = [secret, secret.hex().encode(), secret.hex().upper().encode()]
    for enc in (base64.b64encode, base64.urlsafe_b64encode):
        for shift in range(3):
            text = enc(b"\x00" * shift + secret).rstrip(b"=")
            out.append(text[(4 * shift + 2) // 3 + 1:-2])
    assert all(len(form) >= 8 for form in out)
    return out


def found(secret: bytes, blob: bytes) -> bool:
    return any(form in blob for form in forms(secret))


def snapshot(directory) -> dict:
    """Every file of *directory*: its bytes, mtime, inode and mode; and the directory's mtime (a temporary file
    created and removed changes it)."""
    out = {".": os.stat(directory).st_mtime_ns}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        st = os.stat(path)
        with open(path, "rb") as f:
            out[name] = (f.read(), st.st_mtime_ns, st.st_ino, stat.S_IMODE(st.st_mode))
    return out


def next_slot_id(directory) -> int:
    """The slot id the instance hands out next, read through the store API on a copy of its directory (the
    instance itself is not touched)."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "copy")
        shutil.copytree(directory, target)
        return SlotStore(target).issue(PROBE, 1 << 62, lambda slot_id: ({}, {})).slot_id


def temporary_files(directory) -> list:
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def point(scalar: bytes) -> bytes:
    return r255.encode(r255.mul_base(kem.decode_scalar(scalar)))


def holds_the_key(pks: bytes, scalars: bytes, w_packed: bytes, m: int = M) -> bool:
    """Whether *scalars* are the secret keys of the selected positions ``(j, w_j)`` and ``(m, 0)`` of the table."""
    table = [pks[i:i + 32] for i in range(0, len(pks), 32)]
    bits = unpack_bits(w_packed, m)
    own = [scalars[i:i + 32] for i in range(0, len(scalars), 32)]
    return len(table) == 2 * m + 1 and len(own) == m + 1 and all(
        table[kem.table_index(m, j, bits[j])] == point(own[j]) for j in range(m)) and table[2 * m] == point(own[m])


def slot_of(out: dict) -> int:
    """The slot id of a key (h1) or a grant (h2)."""
    return out["slot"]["slot_id"] if "slot" in out else out["slot_id"]


def check_instance(kg, crypto, ws: dict, keys: dict | None = None) -> dict:
    """The instance's invariants, through the scheme and the store API: the public parameters verify; every
    listed slot has its private record; no temporary file is left; the unselected positions of every slot of
    known ``w`` (*ws*: slot id -> w) recompute from its ``k``; the kept scalars (h1) and the given keys (*keys*:
    slot id -> key) are the secret keys of their slot's table. Returns the public parameters."""
    pp = kg.get_public_parameters()
    parsed = scheme._parse_pp(pp)
    crypto({"keys": {"pp": pp}})                                          # checked as an encryptor checks it
    store = kg._store
    for slot in parsed.slots:
        private = store.private(slot.slot_id)
        assert private is not None, "listed slot {} has no private record".format(slot.slot_id)
        w = ws.get(slot.slot_id)
        if w is None:
            continue
        assert native_backend.ops().bcfe_unselected_ok(parsed.setup.setup_digest, slot.slot_id, packed_w(w),
                                                       private["k"], slot.pks, m=M)
        if "scalars" in private:
            assert private["key_mode"] == "h1" and holds_the_key(slot.pks, private["scalars"], packed_w(w))
        key = (keys or {}).get(slot.slot_id)
        if key is not None:
            assert holds_the_key(slot.pks, garbled.from_b64(key["scalars"], "scalars"), packed_w(w))
    assert temporary_files(kg.directory) == [], "a crashed writer's temporary file"
    return pp


def run_child(body) -> int:
    """Fork; the child runs *body* and exits (:data:`COMPLETED`, or :data:`FAILED` with its traceback on stderr)
    unless *body* ends it first. Returns the pid."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        code = FAILED
        try:
            body()
            code = COMPLETED
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(code)
    return pid


def wait(pid) -> int:
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


# ---------------------------------------------------------------------------
# 1. crash consistency of issuance


#: Where the child dies: before the scheme calls the store, at the backend call that builds the slot, after it
#: returned (the slot's secrets exist in the child's memory only), after the n-th rename into the instance
#: directory during the store's call (whatever the store writes then), after the store returned (the key or
#: grant never reaches the caller).
POINTS = ["before-issue", "in-build", "after-build", "after-write-1", "after-write-2", "after-write-3",
          "after-issue"]


def arm(kg, where: str, report: int) -> None:
    """In the forked child: end the process (``os._exit``) at *where* of the next issuance of *kg*; slot ids
    handed to the backend and the store's result go to the pipe *report*."""
    directory, store = kg.directory, kg._store
    real_issue, real_replace, real_ops = store.issue, os.replace, native_backend.ops
    renames = [0]

    def replace(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if os.path.dirname(os.path.abspath(os.fspath(dst))) == directory:
            renames[0] += 1
            if where == "after-write-{}".format(renames[0]):
                os._exit(KILLED)

    def issue(dedup, max_keys, build):
        if where == "before-issue":
            os._exit(KILLED)
        os.replace = replace
        try:
            out = real_issue(dedup, max_keys, build)
        finally:
            os.replace = real_replace
        os.write(report, b"returned %d\n" % out.slot_id)
        if where == "after-issue":
            os._exit(KILLED)
        return out

    class Ops:
        """The backend, with the slot-building calls of the scheme instrumented."""

        def __getattr__(self, name):
            return getattr(real_ops(), name)

        def _slot(self, name, setup_digest, slot_id, *args, **kwargs):
            os.write(report, b"handed %d\n" % slot_id)
            if where == "in-build":
                os._exit(KILLED)
            out = getattr(real_ops(), name)(setup_digest, slot_id, *args, **kwargs)
            if where == "after-build":
                os._exit(KILLED)
            return out

        def bcfe_new_slot(self, *args, **kwargs):
            return self._slot("bcfe_new_slot", *args, **kwargs)

        def bcfe_place_slot(self, *args, **kwargs):
            return self._slot("bcfe_place_slot", *args, **kwargs)

    store.issue = issue
    instrumented = Ops()
    native_backend.ops = lambda: instrumented  # the scheme calls pyfe4ai.backend.ops()


class Requests:
    """The two requests of a crash test: (alice, W0) issued before the crash, (bob, W1) killed and retried. For
    h2 each request is made once by one holder: a retry sends the same request."""

    REQUESTERS = ("alice", "bob")
    WS = (W0, W1)

    def __init__(self, kg, crypto, mode):
        self.mode = mode
        self.holder = crypto.key_holder(kg.get_public_parameters()) if mode == "h2" else None
        self.requests = [self.holder.request(w) for w in self.WS] if mode == "h2" else None

    def ask(self, kg, i: int) -> dict:
        if self.mode == "h2":
            return kg.authorize(self.requests[i], requester=self.REQUESTERS[i])
        return kg.issue_key(self.WS[i], requester=self.REQUESTERS[i])

    def key(self, out: dict) -> dict:
        return self.holder.finish(out) if self.mode == "h2" else out


@needs_fork
@fork_warning
@pytest.mark.parametrize("where", POINTS)
@pytest.mark.parametrize("mode", MODES)
def test_a_killed_issuance_leaves_a_consistent_instance(tmp_path, mode, where):
    kg, crypto = make(tmp_path, mode=mode)
    requests = Requests(kg, crypto, mode)
    first = requests.ask(kg, 0)
    key0 = requests.key(first)
    before = kg.get_public_parameters()
    read, write = os.pipe()

    def body():
        os.close(read)
        arm(kg, where, write)
        requests.ask(kg, 1)

    pid = run_child(body)
    os.close(write)
    with os.fdopen(read, "rb") as f:
        lines = [line.split() for line in f.read().decode().splitlines()]
    code = wait(pid)
    assert code == KILLED or (code == COMPLETED and where.startswith("after-write")), (
        "the child exited with {} at {}".format(code, where))
    handed = [int(i) for tag, i in lines if tag == "handed"]
    returned = [int(i) for tag, i in lines if tag == "returned"]
    if where == "before-issue":
        assert handed == []
    elif not where.startswith("after-write"):                            # inside the store: 0 or 1, by its order
        assert len(handed) == 1
    assert len(handed) <= 1

    kg = Generator.open(kg.directory)                                    # the authority restarts
    store = kg._store
    pp = check_instance(kg, crypto, {0: W0})
    assert pp["slots"][0] == before["slots"][0], "an earlier slot record changed"
    listed = [slot["slot_id"] for slot in pp["slots"]]
    dedup = dedup_key(pp, "bob", packed_w(W1))
    stored = store.lookup(dedup)
    if where in ("before-issue", "in-build", "after-build"):
        assert stored is None and listed == [0], "a slot of a build that never returned"
    if where == "after-issue":
        assert stored == returned[0]
    if stored is not None:
        assert listed == [0, stored]

    retry = None
    if mode == "h1-no-kept-keys" and stored is not None:                 # issued, its key lost with the child
        with pytest.raises(FEKeyError, match="already issued"):
            requests.ask(kg, 1)
        retry_id = stored
    else:
        retry = requests.ask(kg, 1)
        retry_id = slot_of(retry)
        if stored is not None:                                           # the stored key or grant
            assert retry_id == stored
        else:                                                            # the request's first slot: a new id
            assert retry_id > max(handed + listed) and retry_id not in handed
        if mode == "h1-no-kept-keys":
            with pytest.raises(FEKeyError, match="already issued"):
                requests.ask(kg, 1)
        else:
            assert json.dumps(requests.ask(kg, 1)) == json.dumps(retry), "a second retry gave another key or grant"
    pp = kg.get_public_parameters()
    assert [slot["slot_id"] for slot in pp["slots"]] == [0, retry_id], "a second slot for one request"
    assert store.lookup(dedup) == retry_id and store.lookup(dedup_key(pp, "alice", packed_w(W0))) == 0
    assert next_slot_id(kg.directory) > max(handed + [retry_id]), "a slot id would be handed out again"
    if mode == "h1-no-kept-keys":                                        # the first request: the same answer
        with pytest.raises(FEKeyError, match="already issued"):
            requests.ask(kg, 0)
    else:
        assert json.dumps(requests.ask(kg, 0)) == json.dumps(first)
    key1 = None if retry is None else requests.key(retry)
    check_instance(kg, crypto, {0: W0, retry_id: W1}, {0: key0, retry_id: key1})
    decryptor = crypto({"keys": {"pp": pp}})
    x = random_values(SMALL.data_layout, random.Random(retry_id))
    ct = decryptor.encrypt(x)
    assert decryptor.decrypt(ct, key0) == SMALL.evaluate(W0, x)
    if key1 is not None:
        assert decryptor.decrypt(ct, key1) == SMALL.evaluate(W1, x)


# ---------------------------------------------------------------------------
# 2. concurrency

ROUNDS = 4


def own_w(p: int, i: int) -> dict:
    return {"weights": [p - 2, i - 2], "bias": 10 + i}


def shared_w(i: int) -> dict:
    return {"weights": [3, i - 2], "bias": -10 - i}


@needs_fork
@fork_warning
@pytest.mark.parametrize("key_mode", ["h1", "h2"])
def test_two_processes_issuing_at_once(tmp_path, key_mode):
    """Two processes, each with its own authority object on the instance (``open``), issue at once: each its
    own functions (distinct slots) and the same requests of one requester (one slot each, the same key byte for
    byte under h1 with kept keys, the same grant under h2)."""
    kg, crypto = make(tmp_path, mode=key_mode, max_keys=64)
    pp0 = kg.get_public_parameters()
    holder = crypto.key_holder(pp0) if key_mode == "h2" else None
    shared = [holder.request(shared_w(i)) for i in range(ROUNDS)] if holder else None
    go_read, go_write = os.pipe()
    pids = []
    for p in range(2):
        def body(p=p):
            os.close(go_write)
            gen = Generator.open(kg.directory)
            own_holder = crypto.key_holder(pp0) if key_mode == "h2" else None
            os.read(go_read, 1)                                         # both start together
            results = []
            for i in range(ROUNDS):
                if key_mode == "h1":
                    own = gen.issue_key(own_w(p, i), requester="p{}".format(p))
                    both = gen.issue_key(shared_w(i), requester="shared")
                else:
                    own = own_holder.finish(gen.authorize(own_holder.request(own_w(p, i)), requester="p{}".format(p)))
                    both = gen.authorize(shared[i], requester="shared")
                results.append({"own": own, "shared": both})
            with open(os.path.join(str(tmp_path), "child-{}.json".format(p)), "w") as f:
                json.dump(results, f)

        pids.append(run_child(body))
    os.close(go_read)
    os.write(go_write, b"gg")
    os.close(go_write)
    assert [wait(pid) for pid in pids] == [COMPLETED, COMPLETED]
    got = []
    for p in range(2):
        with open(os.path.join(str(tmp_path), "child-{}.json".format(p))) as f:
            got.append(json.load(f))

    own_ids = [slot_of(r["own"]) for results in got for r in results]
    shared_ids = [slot_of(r["shared"]) for r in got[0]]
    for i in range(ROUNDS):
        assert json.dumps(got[0][i]["shared"]) == json.dumps(got[1][i]["shared"]), "one request, two answers"
    assert len(set(own_ids + shared_ids)) == len(own_ids) + len(shared_ids) == 3 * ROUNDS
    pp = kg.get_public_parameters()
    assert sorted(slot["slot_id"] for slot in pp["slots"]) == list(range(3 * ROUNDS))
    keys = {slot_of(r["own"]): (own_w(p, i), r["own"]) for p, results in enumerate(got)
            for i, r in enumerate(results)}
    for i, r in enumerate(got[0]):
        keys[slot_of(r["shared"])] = (shared_w(i), r["shared"] if key_mode == "h1" else holder.finish(r["shared"]))
    check_instance(kg, crypto, {sid: w for sid, (w, _) in keys.items()}, {sid: k for sid, (_, k) in keys.items()})
    for i in range(ROUNDS):
        assert kg._store.lookup(dedup_key(pp, "shared", packed_w(shared_w(i)))) == shared_ids[i]
    decryptor = crypto({"keys": {"pp": pp}})
    x = {"x": [3, -2]}
    ct = decryptor.encrypt(x)
    for w, key in keys.values():
        assert decryptor.decrypt(ct, key) == SMALL.evaluate(w, x)


def test_lowering_the_cap_during_an_issuance_never_leaves_more_slots_than_max_keys(tmp_path):
    """Another authority object (another process) lowers the cap to the slots listed while an issuance is
    between its read of ``max_keys`` and the store's lock: either the issuance is refused or the lowering is,
    and the public parameters keep verifying. The interleaving is made deterministic by running the other
    process's ``set_max_keys`` at the scheme's call into the store."""
    kg, crypto = make(tmp_path, max_keys=5)
    kg.issue_key(W0, requester="alice")
    kg.issue_key(W1, requester="bob")
    admin = Generator.open(kg.directory)
    real_issue = kg._store.issue

    def issue(dedup, max_keys, build):
        try:
            admin.set_max_keys(2)                                        # 2 slots listed: allowed now
        except FEError:
            pass                                                         # refusing the lowering is correct
        return real_issue(dedup, max_keys, build)

    kg._store.issue = issue
    try:
        kg.issue_key(W2, requester="carol")
    except FEKeyError:
        pass                                                             # refusing the issuance is correct
    pp = admin.get_public_parameters()
    assert pp["slot_list_version"] <= pp["max_keys"]
    crypto({"keys": {"pp": pp}})


# ---------------------------------------------------------------------------
# 3. records hygiene (I14, I16)


@pytest.mark.parametrize("mode", MODES)
def test_no_secret_reaches_a_public_record_file_log_or_message(tmp_path, caplog, mode):
    caplog.set_level(logging.DEBUG, logger="pyfe4ai")
    kg, crypto = make(tmp_path, mode=mode, max_keys=3, issuance_policy=parse("org:bank"))
    holder = crypto.key_holder(kg.get_public_parameters()) if mode == "h2" else None
    asks = [("alice", W0), ("bob", W1), ("carol", W0)]                   # carol: alice's w, a slot of its own
    keys, requests, grants = [], [], []
    for requester, w in asks:
        if holder is None:
            keys.append(kg.issue_key(w, requester=requester, attributes={"org:bank"}))
        else:
            requests.append(holder.request(w))
            grants.append(kg.authorize(requests[-1], requester=requester, attributes={"org:bank"}))
            keys.append(holder.finish(grants[-1]))
    errors = []
    for call in (lambda: kg.issue_key(W2, requester="mallory", attributes={"org:other"}) if holder is None else
                 kg.authorize(holder.request(W2), requester="mallory", attributes={"org:other"}),
                 lambda: kg.issue_key(W0, requester="alice", attributes={"org:bank"}) if holder is None else
                 kg.authorize(requests[0], requester="alice", attributes={"org:bank"})):    # policy; the cap
        with pytest.raises(FEKeyError) as info:
            call()
        errors.append(str(info.value))
    pp = kg.get_public_parameters()
    decryptor = crypto({"keys": {"pp": pp}})
    ct = decryptor.encrypt({"x": [1, 2]})
    for (_, w), key in zip(asks, keys):
        assert decryptor.decrypt(ct, key) == SMALL.evaluate(w, {"x": [1, 2]})

    store = kg._store
    ids = [key["slot_id"] for key in keys]
    ks = [store.private(slot_id)["k"] for slot_id in ids]
    dedups = [dedup_key(pp, requester, packed_w(w)) for requester, w in asks]
    assert [store.lookup(d) for d in dedups] == ids                      # the recomputed keys are the index's
    scalars = [garbled.from_b64(key["scalars"], "scalars") for key in keys]
    each_scalar = [blob[i:i + 32] for blob in scalars for i in range(0, len(blob), 32)]
    files = {}
    for name in os.listdir(kg.directory):
        with open(os.path.join(kg.directory, name), "rb") as f:
            files[name] = f.read()
    public = {
        "pp": json.dumps(pp).encode(), "ciphertext": json.dumps(ct).encode(),
        "slots.json": files["slots.json"], "setup.json": files[scheme.SETUP_FILE],
        "authority.json": files[scheme.AUTHORITY_FILE],                   # private, but holds none of these
        "reprs": (repr(kg) + repr(store) + repr(holder)).encode(), "errors": "\n".join(errors).encode(),
        "logs": "\n".join(r.getMessage() for r in caplog.records).encode(),
    }
    for secret in ks + dedups + each_scalar:
        for name, blob in public.items():
            assert not found(secret, blob), "a secret in {}".format(name)
    private_file = files["slots.private.json"]
    assert all(found(s, private_file) for s in ks + dedups), "the search does not find what is there"
    assert all(found(s, private_file) == (mode == "h1") for s in each_scalar)  # only the kept keys of h1
    if mode != "h1":
        assert not any(found(s, blob) for s in each_scalar for blob in files.values()), (
            "a scalar the authority keeps no copy of (h2: never had) in a file of the instance")
    for i, key in enumerate(keys):                                       # a key holds its own scalars and w only
        blob = json.dumps(key).encode()
        assert not any(found(s, blob) for s in ks + dedups)
        assert not any(found(s, blob) for j, other in enumerate(scalars) if j != i
                       for s in [other[t:t + 32] for t in range(0, len(other), 32)])
    for request, grant, k, own in zip(requests, grants, ks, scalars):    # h2: the request, the grant
        request_blob, grant_blob = json.dumps(request).encode(), json.dumps(grant).encode()
        own_scalars = [own[t:t + 32] for t in range(0, len(own), 32)]
        assert not any(found(s, request_blob) or found(s, grant_blob) for s in own_scalars + dedups)
        assert found(k, grant_blob) and not any(found(other, grant_blob) for other in ks if other != k)
    with open(os.path.join(kg.directory, "slots.json")) as f:
        listed = json.load(f)
    assert set(listed) == {"format", "format_version", "slot_list_version", "slots"}
    assert all(list(slot) == list(garbled.BCFESlot.FIELDS) for slot in listed["slots"])
    assert all(list(slot) == list(garbled.BCFESlot.FIELDS) for slot in pp["slots"])


# ---------------------------------------------------------------------------
# 4. fail-closed parsing


def _two_slots(tmp_path, data_wires="hardwired"):
    kg, crypto = make(tmp_path, data_wires=data_wires)
    key0 = kg.issue_key(W0, requester="alice")
    key1 = kg.issue_key(W1, requester="bob")
    pp = kg.get_public_parameters()
    decryptor = crypto({"keys": {"pp": pp}})
    return kg, crypto, pp, decryptor, decryptor.encrypt({"x": [2, -1]}), key0, key1


#: The bad values of each marker of the public parameters (they list 2 slots).
PP_MARKERS = {
    "bcfe_version": [2, 0, "1", True, 1.0, None],
    "kem": ["ristretto255-hashed-elgamal-v2", "RISTRETTO255-HASHED-ELGAMAL-V1", "", 1, None],
    "gc": ["half-gates-v2", "HALF-GATES-V1", None, 1],
    "gc_hash": ["aes128", "SHAKE128", None, 0],
    "slot_list_version": [1, 3, -1, "2", True, 2.0, None],
}


@pytest.mark.parametrize("marker", sorted(PP_MARKERS))
def test_a_missing_or_unknown_marker_of_the_public_parameters_is_refused(tmp_path, marker):
    _, crypto, pp, _, ct, key0, _ = _two_slots(tmp_path)
    bad_pps = [{k: v for k, v in pp.items() if k != marker}]
    bad_pps += [dict(copy.deepcopy(pp), **{marker: value}) for value in PP_MARKERS[marker]]
    for bad in bad_pps:
        with pytest.raises(FEValidationError):
            crypto({"keys": {"pp": bad}})
        with pytest.raises(FEValidationError):
            crypto.key_holder(bad)
        live = copy.deepcopy(pp)
        decryptor = crypto({"keys": {"pp": live}})
        live.clear()
        live.update(copy.deepcopy(bad))                                  # checked again at every use
        with pytest.raises(FEValidationError):
            decryptor.encrypt({"x": [1, 1]})
        with pytest.raises(FEValidationError):
            decryptor.decrypt(ct, key0)


@pytest.mark.parametrize("record, marker, values, error", [
    ("key", "bcfe_version", [2, 0, "1", True, 1.0, None], FEKeyError),
    ("key", "slot_list_version", [0, -1, "1", True, 1.0, None], FEKeyError),
    ("key", "scheme", ["sife/ddh", None], FEKeyError),
    ("key", "kem", [scheme.KEM], FEKeyError),                             # a field keys do not have
    ("ciphertext", "bcfe_version", [2, "1", True, None], FEValidationError),
    ("ciphertext", "slot_list_version", [-1, "1", True, 1.0, None], FEValidationError),
    ("ciphertext", "scheme", ["sife/ddh"], FEValidationError),
    ("ciphertext", "gc", [scheme.GC], FEValidationError),
])
def test_a_missing_or_unknown_marker_of_a_key_or_ciphertext_is_refused(tmp_path, record, marker, values, error):
    _, _, _, decryptor, ct, key0, _ = _two_slots(tmp_path)
    target = key0 if record == "key" else ct
    bads = [dict(target, **{marker: value}) for value in values]
    if marker in target:
        bads.append({k: v for k, v in target.items() if k != marker})
    for bad in bads:
        with pytest.raises(error):
            decryptor.decrypt(bad, key0) if record == "ciphertext" else decryptor.decrypt(ct, bad)
    assert decryptor.decrypt(ct, key0) == SMALL.evaluate(W0, {"x": [2, -1]})


def test_a_missing_or_unknown_marker_of_a_request_or_grant_is_refused(tmp_path):
    kg, crypto = make(tmp_path, mode="h2")
    holder = crypto.key_holder(kg.get_public_parameters())
    request = holder.request(W0)
    for bad in [dict(request, bcfe_version=v) for v in (2, "1", True, None)] + [
            {k: v for k, v in request.items() if k != "bcfe_version"}, dict(request, scheme="sife/ddh"),
            dict(request, kem=scheme.KEM)]:
        with pytest.raises(FEValidationError):
            kg.authorize(bad, requester="alice")
    assert kg._store.version() == 0 and next_slot_id(kg.directory) == 0
    grant = kg.authorize(request, requester="alice")
    for bad in [dict(grant, bcfe_version=v) for v in (2, "1", True, None)] + [
            dict(grant, slot_list_version=v) for v in (0, -1, "1", True, None)] + [
            {k: v for k, v in grant.items() if k != name} for name in ("bcfe_version", "slot_list_version")]:
        with pytest.raises(FEKeyError):
            holder.finish(bad)
    holder.finish(grant)                                                 # the request stayed open


def test_a_duplicated_R_is_refused_whatever_the_key(tmp_path):
    """02 section 3: two components of one ciphertext with one ``R`` are refused -- in any case of the hex
    digits, and also when neither component is the key's."""
    kg, crypto, _, _, _, key0, _ = _two_slots(tmp_path)
    kg.issue_key(W2, requester="carol")
    decryptor = crypto({"keys": {"pp": kg.get_public_parameters()}})
    ct = decryptor.encrypt({"x": [2, -1]})
    for i, j in ((0, 1), (1, 2), (2, 0)):
        for case in (str.lower, str.upper):
            bad = copy.deepcopy(ct)
            bad["components"][j]["R"] = case(bad["components"][i]["R"])
            with pytest.raises(FEValidationError, match="share an ephemeral key R"):
                decryptor.decrypt(bad, key0)
    assert decryptor.decrypt(ct, key0) == SMALL.evaluate(W0, {"x": [2, -1]})


def test_a_key_of_another_setup_is_refused(tmp_path):
    _, crypto, pp, decryptor, ct, key0, _ = _two_slots(tmp_path / "one")
    other, _ = make(tmp_path / "two")                                    # the same circuit, its own setup
    foreign = other.issue_key(W0, requester="alice")                      # slot 0 as well
    assert foreign["slot_id"] == key0["slot_id"] and foreign["circuit_digest"] == key0["circuit_digest"]
    for bad, match in (
        (foreign, "another setup"),
        (dict(foreign, setup_digest=pp["setup_digest"]), "another slot digest"),
        (dict(foreign, setup_digest=pp["setup_digest"], slot_id=9), "no component for this slot"),
        (dict(key0, setup_digest=pp["setup_digest"].upper()), "setup_digest"),
        (dict(key0, circuit_digest=circuits.linear_threshold(n=2, bw=3, bx=2).circuit.digest().hex()),
         "another circuit"),
    ):
        with pytest.raises(FEKeyError, match=match):
            decryptor.decrypt(ct, bad)


def _cuts(value, hex_field: bool) -> dict:
    out = {"one character": value[:-1], "one group": value[:-2] if hex_field else value[:-4],
           "half": value[:len(value) // 2], "empty": "", "not a str": None, "a number": 7}
    if not hex_field and value.endswith("="):
        out["padding dropped"] = value.rstrip("=")
    return {name: v for name, v in out.items() if v != value}


#: (record, field path, hex or base64, the exception, and whether its message is DECRYPTION_FAILED)
FIELDS = [
    ("pp", ("circuit_encoding",), False, FEValidationError, False),
    ("pp", ("slots", 1, "pks"), False, FEValidationError, False),
    ("pp", ("setup_nonce",), True, FEValidationError, False),
    ("pp", ("slots", 0, "slot_digest"), True, FEValidationError, False),
    ("pp", ("pp_digest",), True, FEValidationError, False),
    ("key", ("scalars",), False, FEKeyError, False),
    ("key", ("w",), False, FEKeyError, False),
    ("key", ("slot_digest",), True, FEKeyError, False),
    ("ciphertext", ("ct_nonce",), True, FEValidationError, False),
    ("ciphertext", ("pp_digest",), True, FEValidationError, False),
    ("ciphertext", ("components", 0, "label_cts"), False, FESchemeError, True),
    ("ciphertext", ("components", 0, "tables"), False, FESchemeError, True),
    ("ciphertext", ("components", 0, "decode"), False, FESchemeError, True),
    ("ciphertext", ("components", 0, "data_active"), False, FESchemeError, True),
    ("ciphertext", ("components", 0, "R"), True, FESchemeError, True),
    ("request", ("pks",), False, FEValidationError, False),
    ("grant", ("slot", "pks"), False, FEKeyError, False),
    ("grant", ("k",), True, FEKeyError, False),
    ("grant", ("pp_digest",), True, FEKeyError, False),
]


def _set(record: dict, path: tuple, value) -> dict:
    out = copy.deepcopy(record)
    target = out
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return out


def _get(record: dict, path: tuple):
    for step in path:
        record = record[step]
    return record


@pytest.mark.parametrize("record, path, hex_field, error, failed", FIELDS,
                         ids=["{}.{}".format(f[0], ".".join(map(str, f[1]))) for f in FIELDS])
def test_a_truncated_field_is_refused_with_the_librarys_exception(tmp_path, record, path, hex_field, error, failed):
    """Every cut of a hex or base64 field gives the documented exception of the library -- never another
    exception (``IndexError``, ``binascii.Error``, a bare ``ValueError``) -- and a component's field the one
    error of a component that does not decrypt (I10). Mode "labels" gives the components data labels."""
    if record in ("request", "grant"):
        kg, crypto = make(tmp_path, mode="h2", data_wires="labels")
        holder = crypto.key_holder(kg.get_public_parameters())
        request = holder.request(W0)
        target = request if record == "request" else kg.authorize(request, requester="alice")
    else:
        kg, crypto, pp, decryptor, ct, key0, _ = _two_slots(tmp_path, data_wires="labels")
        target = {"pp": pp, "key": key0, "ciphertext": ct}[record]
    for name, value in _cuts(_get(target, path), hex_field).items():
        bad = _set(target, path, value)
        expected = error
        if path[-1] == "R" and not isinstance(value, str):
            expected = FEValidationError                                 # the distinct-R check reads it first
        with pytest.raises(FEError) as info:
            if record == "pp":
                crypto({"keys": {"pp": bad}})
            elif record == "key":
                decryptor.decrypt(ct, bad)
            elif record == "ciphertext":
                decryptor.decrypt(bad, key0)
            elif record == "request":
                kg.authorize(bad, requester="bob")
            else:
                holder.finish(bad)
        assert type(info.value) is expected, "{}: {!r}".format(name, info.value)
        if failed and expected is FESchemeError:
            assert str(info.value) == scheme.DECRYPTION_FAILED
    if record == "request":
        assert kg._store.version() == 0
    if record == "grant":
        holder.finish(target)                                            # the request stayed open


@pytest.mark.parametrize("value", [None, [], "pp", 3, b"{}"])
def test_records_that_are_not_dicts_are_refused(tmp_path, value):
    kg, crypto, pp, decryptor, ct, key0, _ = _two_slots(tmp_path)
    with pytest.raises(FEValidationError):
        crypto({"keys": {"pp": value}})
    with pytest.raises(FEValidationError):
        crypto.key_holder(value)
    with pytest.raises(FEKeyError):
        decryptor.decrypt(ct, value)
    with pytest.raises(FEValidationError):
        decryptor.decrypt(value, key0)


@pytest.mark.parametrize("slots", [1.0, object()], ids=["float", "object"])
def test_a_slot_selection_of_another_type_is_refused(tmp_path, slots):
    _, _, _, decryptor, _, _, _ = _two_slots(tmp_path)
    with pytest.raises(FEValidationError):
        decryptor.encrypt({"x": [1, 1]}, slots=slots)


@pytest.mark.parametrize("key_mode", ["h1", "h2"])
def test_a_requester_id_that_is_not_utf8_is_refused(tmp_path, key_mode):
    kg, crypto = make(tmp_path, mode=key_mode)
    with pytest.raises(FEValidationError, match="requester"):
        if key_mode == "h1":
            kg.issue_key(W0, requester="alice\ud800")
        else:
            kg.authorize(crypto.key_holder(kg.get_public_parameters()).request(W0), requester="alice\ud800")


# ---------------------------------------------------------------------------
# 5. mixing components at the scheme level

TRIALS = 6


@pytest.fixture(scope="module")
def mixing(tmp_path_factory):
    """``linear_score`` (7 output bits) with two slots, A (alice) and B (bob); another setup of the same
    circuit with alice's ``w`` in its slot 0; per trial a ciphertext, a donor ciphertext and a ciphertext of the
    other setup, of random data."""
    rng = random.Random(2026)
    kg, crypto = make(tmp_path_factory.mktemp("mix"), SCORE)
    w_a, w_b = random_values(SCORE.key_layout, rng), random_values(SCORE.key_layout, rng)
    key_a, key_b = kg.issue_key(w_a, requester="alice"), kg.issue_key(w_b, requester="bob")
    other, _ = make(tmp_path_factory.mktemp("mix-other"), SCORE)
    foreign_key = other.issue_key(w_a, requester="alice")
    pp = kg.get_public_parameters()
    setup = scheme._parse_pp(pp).setup
    slot_digests = [bytes.fromhex(s["slot_digest"]) for s in pp["slots"]]
    env = {
        "w_a": w_a, "w_b": w_b, "key_a": key_a, "key_b": key_b, "foreign_key": foreign_key, "pp": pp,
        "decryptor": crypto({"keys": {"pp": pp}}),
        "other_pp_digest": garbled.pp_digest(setup.setup_digest, scheme.GC, setup.desc_json, pp["max_keys"] + 1,
                                             slot_digests).hex(),
    }
    encryptor, foreign_encryptor = env["decryptor"], crypto({"keys": {"pp": other.get_public_parameters()}})
    env["trials"] = []
    for _ in range(TRIALS):
        x, x2 = random_values(SCORE.data_layout, rng), random_values(SCORE.data_layout, rng)
        env["trials"].append((x, x2, encryptor.encrypt(x), encryptor.encrypt(x2), foreign_encryptor.encrypt(x)))
    return env


def _component(ct: dict, slot_id: int) -> dict:
    return next(c for c in ct["components"] if c["slot_id"] == slot_id)


def _replace(ct: dict, slot_id: int, component: dict) -> dict:
    out = copy.deepcopy(ct)
    out["components"] = [dict(component, slot_id=slot_id) if c["slot_id"] == slot_id else c
                         for c in out["components"]]
    return out


def _with(component: dict, donor: dict, *fields) -> dict:
    return dict(component, **{f: donor[f] for f in fields})


#: name -> (env, ct, donor, foreign ct) -> (the mixed ciphertext, the key that decrypts it). A is slot 0, B
#: slot 1; the donor is another ciphertext of the same public parameters.
MIXES = {
    "component of another ciphertext": lambda e, ct, d, f: (_replace(ct, 0, _component(d, 0)), e["key_a"]),
    "tables and decoding bits of another ciphertext": lambda e, ct, d, f: (
        _replace(ct, 0, _with(_component(ct, 0), _component(d, 0), "tables", "decode")), e["key_a"]),
    "label ciphertexts and R of another ciphertext": lambda e, ct, d, f: (
        _replace(ct, 0, _with(_component(ct, 0), _component(d, 0), "label_cts", "R")), e["key_a"]),
    "nonce of another ciphertext": lambda e, ct, d, f: (dict(ct, ct_nonce=d["ct_nonce"]), e["key_a"]),
    "pp digest of other public parameters of the setup": lambda e, ct, d, f: (
        dict(ct, pp_digest=e["other_pp_digest"]), e["key_a"]),
    "component of another slot, the original dropped": lambda e, ct, d, f: (
        dict(ct, components=[dict(_component(ct, 1), slot_id=0)]), e["key_a"]),
    "component of another slot of another ciphertext": lambda e, ct, d, f: (
        _replace(ct, 0, _component(d, 1)), e["key_a"]),
    "component of a ciphertext of another setup": lambda e, ct, d, f: (
        _replace(ct, 0, _component(f, 0)), e["key_a"]),
    "key of another setup with this setup's digests": lambda e, ct, d, f: (ct, dict(
        e["foreign_key"], setup_digest=e["key_a"]["setup_digest"], slot_digest=e["key_a"]["slot_digest"],
        pp_digest=e["key_a"]["pp_digest"])),
    "key moved to another slot with that slot's digest": lambda e, ct, d, f: (ct, dict(
        e["key_a"], slot_id=1, slot_digest=e["key_b"]["slot_digest"])),
}


@pytest.mark.parametrize("mix", sorted(MIXES))
def test_a_mixed_component_decrypts_to_an_unrelated_value(mixing, mix):
    """What the scheme does with these mixes: nothing refuses them (no digest covers a component, and a key's
    scalars are bound to nothing the decryptor can check), and the hashes bind every label and table to its
    (pp digest, nonce, slot, position, bit, ``R``) and instance salt (I2, I11): the decryption gives unrelated
    bits -- ``C(w, x)`` (for either ``w`` and either data of the mix) in at most half of the trials, by chance
    (``2^-7`` each), against every trial for the unmixed ciphertext. If a decryption fails it is the one error
    of I10."""
    e = mixing
    hits = 0
    for x, x2, ct, donor, foreign in e["trials"]:
        assert e["decryptor"].decrypt(ct, e["key_a"]) == SCORE.evaluate(e["w_a"], x)    # the control
        assert e["decryptor"].decrypt(ct, e["key_b"]) == SCORE.evaluate(e["w_b"], x)
        mixed, key = MIXES[mix](e, ct, donor, foreign)
        meaningful = [SCORE.evaluate(w, data) for w in (e["w_a"], e["w_b"]) for data in (x, x2)]
        try:
            value = e["decryptor"].decrypt(mixed, key)
        except FESchemeError as exc:
            assert str(exc) == scheme.DECRYPTION_FAILED
            continue
        hits += value in meaningful
    assert hits <= TRIALS // 2, "a mixed component decrypted to C(w, x) in {} of {} trials".format(hits, TRIALS)


def test_mixes_a_check_refuses(mixing):
    """The mixes a check of the scheme refuses: a component of another slot beside the original (two
    components with one ``R``), a component appended to a ciphertext that has its slot (two components of one
    slot), a key relabelled to another slot with its own slot digest (not the public parameters' slot), and a
    ciphertext of another setup (its setup digest)."""
    e = mixing
    _, _, ct, donor, foreign = e["trials"][0]
    decrypt = e["decryptor"].decrypt
    beside = copy.deepcopy(ct)
    beside["components"][0] = dict(_component(ct, 1), slot_id=0)
    with pytest.raises(FEValidationError, match="share an ephemeral key R"):
        decrypt(beside, e["key_a"])
    appended = copy.deepcopy(ct)
    appended["components"].append(_component(donor, 0))
    with pytest.raises(FEValidationError, match="distinct"):
        decrypt(appended, e["key_a"])
    with pytest.raises(FEKeyError, match="another slot digest"):
        decrypt(ct, dict(e["key_a"], slot_id=1))
    with pytest.raises(FEValidationError, match="another setup"):
        decrypt(foreign, e["key_a"])


# ---------------------------------------------------------------------------
# 6. the cap and the dedup (D-j, review V6-F10)


def _ask(kg, holder, w, requester, attributes=(), request=None, label=None):
    if holder is None:
        return kg.issue_key(w, requester=requester, attributes=attributes, label=label)
    return kg.authorize(request if request is not None else holder.request(w), requester=requester,
                        attributes=attributes, label=label)


@pytest.mark.parametrize("mode", MODES)
def test_at_the_cap_a_repeated_and_a_new_request_are_indistinguishable(tmp_path, caplog, mode):
    """At ``max_keys`` a repeated (requester, w) -- under h2 also with other public keys -- and new requests get
    an exception of one type, text, arguments and context, log the same and change no file (bytes, mtime,
    inode, mode, the directory's mtime) and no slot id; under h1 without kept keys the repeated request does
    not get the "already issued" error that would tell it apart."""
    kg, crypto = make(tmp_path, mode=mode, max_keys=2)
    holder = crypto.key_holder(kg.get_public_parameters()) if mode == "h2" else None
    first = holder.request(W0) if holder else None
    _ask(kg, holder, W0, "alice", request=first)
    _ask(kg, holder, W1, "bob")
    attempts = {"repeated": dict(w=W0, requester="alice", request=first),
                "new w": dict(w=W2, requester="alice"),
                "new requester": dict(w=W0, requester="dave")}
    if holder is not None:
        attempts["repeated, other keys"] = dict(w=W0, requester="alice")
        attempts = {name: dict(a, request=a.get("request") or holder.request(a["w"])) for name, a in attempts.items()}
    before, next_id = snapshot(kg.directory), next_slot_id(kg.directory)
    caplog.set_level(logging.DEBUG, logger="pyfe4ai")
    seen = {}
    for name, attempt in attempts.items():
        caplog.clear()
        with pytest.raises(FEError) as info:
            _ask(kg, holder, **attempt)
        exc = info.value
        seen[name] = (type(exc), str(exc), exc.args, repr(exc.__cause__), repr(exc.__context__),
                      exc.__suppress_context__, tuple((r.levelno, r.getMessage()) for r in caplog.records))
        assert snapshot(kg.directory) == before, "{} changed a file".format(name)
    assert len(set(seen.values())) == 1, seen
    assert seen["repeated"][0] is FEKeyError and "max_keys = 2" in seen["repeated"][1]
    assert next_slot_id(kg.directory) == next_id == 2
    assert kg.get_public_parameters()["slot_list_version"] == 2


@pytest.mark.parametrize("key_mode", ["h1", "h2"])
def test_a_refused_request_reserves_no_slot_id(tmp_path, key_mode):
    """The policy (and every other refusal before step 5) comes before any reservation: the files and the next
    slot id (read through the store API) do not change, and the next accepted request gets that id."""
    kg, crypto = make(tmp_path, mode=key_mode, max_keys=3, issuance_policy=parse("org:bank"))
    holder = crypto.key_holder(kg.get_public_parameters()) if key_mode == "h2" else None
    _ask(kg, holder, W0, "alice", attributes={"org:bank"})
    good = {"org:bank"}
    refusals = [
        ("policy", lambda: _ask(kg, holder, W1, "bob", attributes={"org:other"})),
        ("policy, no attributes", lambda: _ask(kg, holder, W1, "bob")),
        ("bad label", lambda: _ask(kg, holder, W1, "bob", attributes=good, label="a\tb")),
        ("bad attributes", lambda: _ask(kg, holder, W1, "bob", attributes="org:bank")),
        ("bad attribute label", lambda: _ask(kg, holder, W1, "bob", attributes={"bank"})),
        ("empty requester", lambda: _ask(kg, holder, W1, "", attributes=good)),
        ("requester not a str", lambda: _ask(kg, holder, W1, None, attributes=good)),
        ("requester not UTF-8", lambda: _ask(kg, holder, W1, "bob\udc00", attributes=good)),
    ]
    if holder is None:
        refusals += [
            ("w out of range", lambda: kg.issue_key({"weights": [1, 9], "bias": 0}, requester="bob", attributes=good)),
            ("w missing a field", lambda: kg.issue_key({"weights": [1, 1]}, requester="bob", attributes=good)),
            ("the other key mode", lambda: kg.authorize({}, requester="bob", attributes=good)),
        ]
    else:
        request = holder.request(W1)
        pks = garbled.from_b64(request["pks"], "pks")
        refusals += [
            ("w out of range", lambda: kg.authorize(dict(request, w={"weights": [1, 9], "bias": 0}),
                                                    requester="bob", attributes=good)),
            ("identity key", lambda: kg.authorize(dict(request, pks=garbled.b64(bytes(32) + pks[32:])),
                                                  requester="bob", attributes=good)),
            ("another setup", lambda: kg.authorize(dict(request, setup_digest="00" * 32), requester="bob",
                                                   attributes=good)),
            ("the other key mode", lambda: kg.issue_key(W1, requester="bob", attributes=good)),
        ]
    before, next_id = snapshot(kg.directory), next_slot_id(kg.directory)
    for name, call in refusals:
        with pytest.raises(FEError):
            call()
        assert snapshot(kg.directory) == before, "{} changed a file".format(name)
    assert next_slot_id(kg.directory) == next_id == 1
    assert slot_of(_ask(kg, holder, W1, "bob", attributes=good)) == next_id


# ---------------------------------------------------------------------------
# 7. an old ciphertext and a new key (h2)


def test_an_old_ciphertext_and_a_new_h2_key(tmp_path):
    """The h2 variant of ``test_scheme``'s test: a ciphertext made before the holder's slot existed has no
    component for it, and the error says why -- with the decryptor's public parameters newer (the slot listed)
    or as old as the ciphertext's (the slot unknown to them)."""
    kg, crypto = make(tmp_path, mode="h2")
    old_pp = kg.get_public_parameters()
    first = crypto.key_holder(old_pp)
    key0 = first.finish(kg.authorize(first.request(W0), requester="alice"))
    old_pp = kg.get_public_parameters()
    old_ct = crypto({"keys": {"pp": old_pp}}).encrypt({"x": [3, 1]})
    second = crypto.key_holder(old_pp)
    key1 = second.finish(kg.authorize(second.request(W1), requester="bob"))
    assert (key1["slot_id"], key1["slot_list_version"]) == (1, 2)
    new_pp = kg.get_public_parameters()
    for pp in (new_pp, old_pp):
        decryptor = crypto({"keys": {"pp": pp}})
        with pytest.raises(FEKeyError, match=r"older than the key \(keys decrypt only ciphertexts made after they "
                                             r"were issued\): it was made with slot list version 1, the key's "
                                             r"slot came with version 2"):
            decryptor.decrypt(old_ct, key1)
        assert decryptor.decrypt(old_ct, key0) == SMALL.evaluate(W0, {"x": [3, 1]})
    decryptor = crypto({"keys": {"pp": new_pp}})
    with pytest.raises(FEKeyError, match="no component for this slot"):
        decryptor.decrypt(decryptor.encrypt({"x": [3, 1]}, slots=[0]), key1)
    assert decryptor.decrypt(decryptor.encrypt({"x": [3, 1]}), key1) == SMALL.evaluate(W1, {"x": [3, 1]})
