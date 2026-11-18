"""The slot store of the bounded-key schemes (:mod:`pyfe4ai.schemes.common.slots`).

The public slot list only grows and counts its slots; a repeated dedup key
returns the stored private record without building; at the cap a hit and a
miss get the same error and nothing is written; a failed build or a crash at
any step burns at most its slot id and never hands an id out twice; a slot
whose private record was written is issued (the next call lists it from its
write-ahead record, a retry is a hit), so one request never gets two slots;
two processes (and several threads and store objects) issuing at once get
distinct ids; the private side is owner-only and never reaches the public
file; damaged files fail closed; a forked child issues even when another
thread of the parent held the store at the fork.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import pickle
import stat
import sys
import threading

import pytest

from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.schemes.common import slots
from pyfe4ai.schemes.common.slots import IssueResult, SlotStore

needs_fork = pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
#: fork() of a process with threads: Python warns (3.12+); the point of these tests
fork_warning = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def dkey(i) -> bytes:
    return hashlib.sha3_256("dedup key {}".format(i).encode()).digest()


def records(slot_id: int, request=None):
    """A public and a private record of slot *slot_id* (fresh private bytes);
    the public record names its *request* if given (for counting slots per
    request)."""
    public = {"label": "slot-{}".format(slot_id), "pks": [b"\x01" * 32, b"\x02" * 32]}
    if request is not None:
        public["request"] = request
    return public, {"k_i": os.urandom(16), "key": {"scalars": [os.urandom(32)], "bits": [0, 1, 1]}}


class Builder:
    """A ``build`` that records every slot id handed to it."""

    def __init__(self, request=None, fail=None):
        self.ids, self.request, self.fail = [], request, fail

    def __call__(self, slot_id):
        self.ids.append(slot_id)
        if self.fail is not None:
            raise self.fail
        return records(slot_id, self.request)


def never(slot_id):  # pragma: no cover - a hit must not build
    raise AssertionError("build() called for a hit")


def files(directory) -> dict:
    """Name -> bytes of every file in *directory*."""
    out = {}
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as f:
            out[name] = f.read()
    return out


def on_disk(directory, name):
    """The JSON of file *name* as it is now (no store call: no recovery)."""
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def listed_on_disk(directory):
    public = on_disk(directory, slots.PUBLIC_FILE)
    return [] if public is None else [slot["slot_id"] for slot in public["slots"]]


def check_consistent(directory, handed=()):
    """As found on disk: the public file lists no slot whose private record
    is absent. After the recovery (the first store call): the files load;
    every listed slot has its private record and one dedup entry, and no
    other slot has one; no request (a record's ``request``) has two slots;
    every slot id handed to a build is handed once and lies below
    next_slot_id; a listed slot was handed out."""
    private = on_disk(directory, slots.PRIVATE_FILE)
    found = listed_on_disk(directory)
    assert not found or set(found) <= {int(i) for i in private["private"]}, "listed without its private record"
    store = SlotStore(directory)
    public = store.public_slots()
    ids = [slot["slot_id"] for slot in public]
    assert ids == sorted(set(ids)) and ids[:len(found)] == found, "the recovery only appends"
    assert store.version() == len(ids) and listed_on_disk(directory) == ids
    assert len(handed) == len(set(handed)), "a slot id was handed out twice: {}".format(handed)
    assert set(ids) <= set(handed) or not handed
    private = on_disk(directory, slots.PRIVATE_FILE)
    assert all(i < private["next_slot_id"] for i in list(handed) + ids)
    assert sorted(int(i) for i in private["private"]) == ids, "a private record of an unlisted slot"
    assert sorted(private["dedup"].values()) == ids, "not one dedup entry per listed slot"
    requests = [slot["request"] for slot in public if "request" in slot]
    assert len(requests) == len(set(requests)), "one request, two slots: {}".format(requests)
    return store


# -- the list, dedup and the cap


def test_a_new_store_is_empty_and_reads_make_no_directory(tmp_path):
    store = SlotStore(tmp_path / "authority" / "bcfe" / "setup")
    assert store.public_slots() == [] and store.version() == 0
    assert store.lookup(dkey(0)) is None and store.private(0) is None
    assert not (tmp_path / "authority").exists()
    assert repr(store) == "SlotStore({!r})".format(str(tmp_path / "authority" / "bcfe" / "setup"))


def test_slots_are_appended_in_order_and_versioned(tmp_path):
    store, build = SlotStore(tmp_path), Builder()
    seen = []
    for i in range(4):
        out = store.issue(dkey(i), 10, build)
        assert out.status == "new" and out.slot_id == i and isinstance(out, IssueResult)
        assert store.version() == i + 1
        listed = store.public_slots()
        assert listed[:len(seen)] == seen, "an earlier slot record changed"
        seen = listed
        assert listed[-1] == {"slot_id": i, "label": "slot-{}".format(i), "pks": [b"\x01" * 32, b"\x02" * 32]}
        assert store.private(i) == out.private and store.lookup(dkey(i)) == i
    assert build.ids == [0, 1, 2, 3]
    listed = store.public_slots()
    listed[0]["label"] = "changed"
    listed.append({"slot_id": 99})
    store.private(0)["k_i"] = b"changed"
    assert store.public_slots() == seen and store.private(0)["k_i"] != b"changed", "reads return copies"
    assert SlotStore(tmp_path).public_slots() == seen, "another store object on the directory"
    with open(tmp_path / slots.PUBLIC_FILE) as f:
        assert json.load(f)["slot_list_version"] == 4


def test_a_dedup_hit_returns_the_stored_record_without_building(tmp_path):
    store = SlotStore(tmp_path)
    first = store.issue(dkey(7), 5, Builder())
    before = files(tmp_path)
    again = store.issue(dkey(7), 5, never)
    assert again == ("hit", first.slot_id, first.private) and again.private is not first.private
    assert SlotStore(tmp_path).issue(bytearray(dkey(7)), 5, never) == ("hit", 0, first.private)
    assert store.version() == 1 and files(tmp_path) == before, "a hit writes nothing"
    assert store.issue(dkey(8), 5, Builder()).slot_id == 1


def test_the_cap_gives_one_error_for_hits_and_misses_and_writes_nothing(tmp_path):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 2, Builder())
    store.issue(dkey(1), 2, Builder())
    before = files(tmp_path)
    with pytest.raises(FEKeyError) as hit:
        store.issue(dkey(0), 2, never)
    with pytest.raises(FEKeyError) as miss:
        store.issue(dkey(2), 2, never)
    with pytest.raises(FEKeyError) as lowered:
        store.issue(dkey(3), 1, never)
    assert type(hit.value) is type(miss.value) is FEKeyError
    assert str(hit.value) == str(miss.value) and "max_keys = 2" in str(hit.value)
    assert "max_keys = 1" in str(lowered.value)
    for exc in (hit.value, miss.value):
        assert dkey(0).hex() not in str(exc) and dkey(2).hex() not in str(exc)
    assert files(tmp_path) == before and store.version() == 2
    assert store.issue(dkey(0), 3, never).status == "hit", "a raised cap"
    assert store.issue(dkey(2), 3, Builder())[:2] == ("new", 2)


@pytest.mark.parametrize("bad", [b"\x00" * 31, b"\x00" * 33, "00" * 32, None, 5, memoryview(b"\x00" * 32)])
def test_a_dedup_key_is_32_bytes(tmp_path, bad):
    store = SlotStore(tmp_path)
    with pytest.raises(FEValidationError, match="32 bytes") as err:
        store.issue(bad, 3, never)
    assert "00" * 16 not in str(err.value)
    with pytest.raises(FEValidationError):
        store.lookup(bad)


@pytest.mark.parametrize("max_keys", [0, -1, True, 2.0, "3", None])
def test_max_keys_is_a_positive_integer(tmp_path, max_keys):
    with pytest.raises(FEValidationError, match="max_keys"):
        SlotStore(tmp_path).issue(dkey(0), max_keys, never)


def test_other_arguments_are_checked(tmp_path):
    with pytest.raises(FEValidationError, match="callable"):
        SlotStore(tmp_path).issue(dkey(0), 3, None)
    for bad in ("", None, 3):
        with pytest.raises(FEValidationError, match="directory"):
            SlotStore(bad)
    for bad in (-1, True, "0", 1.0):
        with pytest.raises(FEValidationError, match="slot id"):
            SlotStore(tmp_path).private(bad)


def test_records_round_trip_and_a_hit_equals_the_new_result(tmp_path):
    store = SlotStore(tmp_path)
    private = {"a": None, "b": True, "c": -3, "d": 2 ** 200, "e": 0.25, "f": "text",
               "g": [b"", b"\xff" * 5, (1, 2)], "h": {"nested": bytearray(b"xy")}}

    def build(slot_id):
        return {"slot_id": slot_id, "pk": b"\x00" * 32, "t": (1, "x")}, private

    new = store.issue(dkey(0), 3, build)
    expected = dict(private, g=[b"", b"\xff" * 5, [1, 2]], h={"nested": b"xy"})
    assert new.private == expected and store.issue(dkey(0), 3, never).private == expected
    assert store.public_slots() == [{"slot_id": 0, "pk": b"\x00" * 32, "t": [1, "x"]}]
    assert store.issue(dkey(1), 3, lambda i: ({}, b"\x07" * 16)).private == b"\x07" * 16
    assert store.private(1) == b"\x07" * 16


# -- a failing build, crashes


@pytest.mark.parametrize("returned", [
    None, ({"a": 1},), ([], {}), ({"a": 1}, 7), ({"a": object()}, {}), ({"a": 1}, {"b": float("nan")}),
    ({1: "x"}, {}), ({"__bytes__": "AA=="}, {}), ({"slot_id": 5}, {}), ({"slot_id": True}, {}),
])
def test_a_build_returning_records_that_cannot_be_stored_burns_its_id(tmp_path, returned):
    store = SlotStore(tmp_path)
    with pytest.raises(FEValidationError):
        store.issue(dkey(0), 3, lambda slot_id: returned)
    assert store.version() == 0 and store.lookup(dkey(0)) is None
    assert store.issue(dkey(0), 3, Builder()).slot_id == 1


def test_a_failing_build_changes_nothing_but_burns_its_id(tmp_path):
    store, build = SlotStore(tmp_path), Builder()
    store.issue(dkey(0), 5, build)
    public = store.public_slots()
    failing = Builder(fail=RuntimeError("build failed"))
    with pytest.raises(RuntimeError, match="build failed"):
        store.issue(dkey(1), 5, failing)
    assert failing.ids == [1]
    assert store.public_slots() == public and store.version() == 1 and store.lookup(dkey(1)) is None
    out = store.issue(dkey(1), 5, build)
    assert out[:2] == ("new", 2), "the id handed to the failed build is never handed out again"
    assert [slot["slot_id"] for slot in store.public_slots()] == [0, 2]
    check_consistent(tmp_path, build.ids + failing.ids)


def test_a_build_using_the_store_is_refused(tmp_path):
    store = SlotStore(tmp_path)
    with pytest.raises(FESchemeError, match="build"):
        store.issue(dkey(0), 3, lambda slot_id: (store.version(), {}))
    assert store.issue(dkey(0), 3, Builder()).slot_id == 1, "the locks were released"


@pytest.mark.parametrize("failing_write", [1, 2, 3])
def test_an_error_at_each_write(tmp_path, monkeypatch, failing_write):
    """The writes of a new slot: 1 the reservation, 2 the private record with
    its dedup entry and the pending copy of its public record (the commit
    point), 3 the public record. An error at write 3, between the two writes,
    leaves an issued slot: the next call lists it and a retry is a hit."""
    store, first, build = SlotStore(tmp_path), Builder("r0"), Builder("r1")
    store.issue(dkey(0), 5, first)
    count, real = [0], slots._atomic_write

    def write(*args):
        count[0] += 1
        if count[0] == failing_write:
            raise OSError("disk full")
        real(*args)

    monkeypatch.setattr(slots, "_atomic_write", write)
    with pytest.raises(OSError, match="disk full"):
        store.issue(dkey(1), 5, build)
    monkeypatch.setattr(slots, "_atomic_write", real)
    assert [n for n in os.listdir(tmp_path) if n.startswith(".slots-")] == []
    assert listed_on_disk(tmp_path) == [0]
    issued = failing_write == 3
    check_consistent(tmp_path, first.ids + build.ids)
    assert store.version() == (2 if issued else 1)
    assert store.lookup(dkey(1)) == (1 if issued else None) and (store.private(1) is not None) == issued
    retry = store.issue(dkey(1), 5, build)
    if issued:
        assert retry.status == "hit" and retry.slot_id == 1 and build.ids == [1]
    else:
        assert retry.status == "new" and retry.slot_id == (1 if failing_write == 1 else 2)
    assert store.issue(dkey(1), 5, never) == ("hit", retry.slot_id, retry.private)
    check_consistent(tmp_path, first.ids + build.ids)


def _interrupted(directory, monkeypatch):
    """A store whose issue of ``dkey(1)`` failed at the public write (after
    its commit point); the private record returned by nothing so far."""
    store = SlotStore(directory)
    store.issue(dkey(0), 5, Builder("r0"))
    real = slots._atomic_write

    def write(directory, name, data, mode):
        if name == slots.PUBLIC_FILE:
            raise OSError("disk full")
        real(directory, name, data, mode)

    monkeypatch.setattr(slots, "_atomic_write", write)
    with pytest.raises(OSError):
        store.issue(dkey(1), 5, Builder("r1"))
    monkeypatch.setattr(slots, "_atomic_write", real)
    assert listed_on_disk(directory) == [0]
    return store


CALLS = {
    "issue another": lambda store: store.issue(dkey(2), 5, Builder("r2")),
    "issue at the cap": lambda store: pytest.raises(FEKeyError, store.issue, dkey(2), 2, never),
    "public_slots": lambda store: store.public_slots(),
    "version": lambda store: store.version(),
    "private": lambda store: store.private(0),
    "lookup": lambda store: store.lookup(dkey(0)),
}


@pytest.mark.parametrize("call", sorted(CALLS))
def test_every_call_first_lists_an_interrupted_slot(tmp_path, monkeypatch, call):
    store = _interrupted(tmp_path, monkeypatch)
    CALLS[call](SlotStore(tmp_path))
    assert listed_on_disk(tmp_path)[:2] == [0, 1], "the interrupted slot was not listed first"
    before = files(tmp_path)
    assert store.public_slots()[1] == dict(records(1, "r1")[0], slot_id=1)
    assert files(tmp_path) == before, "the recovery runs once"
    assert store.issue(dkey(1), 5, never)[:2] == ("hit", 1)
    check_consistent(tmp_path)


def test_the_pending_copy_is_dropped_by_the_next_private_write(tmp_path, monkeypatch):
    store = _interrupted(tmp_path, monkeypatch)
    assert list(on_disk(tmp_path, slots.PRIVATE_FILE)["pending"]) == ["1"]
    store.version()  # lists slot 1; the private file is not rewritten by a read
    assert list(on_disk(tmp_path, slots.PRIVATE_FILE)["pending"]) == ["1"]
    store.issue(dkey(2), 5, Builder("r2"))  # its reservation drops slot 1's copy
    assert list(on_disk(tmp_path, slots.PRIVATE_FILE)["pending"]) == ["2"]
    check_consistent(tmp_path)


#: Where the forked child dies: before / after (the rename of) write n, in
#: build, or in the middle of writing a temporary file.
CRASHES = ["before-1", "after-1", "in-build", "before-2", "after-2", "before-3", "after-3", "torn-temp"]


@needs_fork
@pytest.mark.parametrize("where", CRASHES)
def test_a_crash_at_each_step_never_reuses_a_slot_id(tmp_path, where):
    """Writes as in test_an_error_at_each_write; from the commit point on
    (after-2, before-3, after-3) the slot is issued and a retry is a hit."""
    store, build = SlotStore(tmp_path), Builder("r0")
    store.issue(dkey(0), 5, build)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child, dies at *where*
        try:
            os.close(read)
            count, real = [0], slots._atomic_write

            def dying_write(directory, name, data, mode):
                count[0] += 1
                if where == "before-{}".format(count[0]):
                    os._exit(0)
                if where == "torn-temp" and count[0] == 2:
                    with open(os.path.join(directory, ".slots-torn.tmp"), "wb") as f:
                        f.write(json.dumps(data).encode()[:20])
                    os._exit(0)
                real(directory, name, data, mode)
                if where == "after-{}".format(count[0]):
                    os._exit(0)

            def dying_build(slot_id):
                os.write(write, b"%d\n" % slot_id)
                if where == "in-build":
                    os._exit(0)
                return records(slot_id, "r1")

            slots._atomic_write = dying_write
            store.issue(dkey(1), 5, dying_build)
        finally:
            os._exit(3)  # finished without dying: the test fails
    os.close(write)
    with os.fdopen(read, "rb") as f:
        handed = build.ids + [int(line) for line in f.read().split()]
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    store = check_consistent(tmp_path, handed)
    assert [n for n in os.listdir(tmp_path) if n.startswith(".slots-")] == [], "a crashed writer's temporary file"
    kept = {slot["slot_id"]: store.private(slot["slot_id"]) for slot in store.public_slots()}
    assert all(private is not None for private in kept.values()), "a listed slot without its private record"
    retry = Builder("r1")
    after = SlotStore(tmp_path).issue(dkey(1), 5, retry)
    if where in ("after-2", "before-3", "after-3"):  # issued before the crash
        assert after.status == "hit" and after.slot_id == 1 and after.private == kept[1] and retry.ids == []
    else:
        assert after.status == "new" and after.slot_id not in handed
        assert after.slot_id == (1 if where in ("before-1",) else 2)
    check_consistent(tmp_path, handed + retry.ids)
    for slot_id, private in kept.items():  # a slot's private record never changes
        assert store.private(slot_id) == private


# -- concurrency


def test_threads_and_store_objects_get_distinct_ids(tmp_path):
    stores = [SlotStore(tmp_path), SlotStore(tmp_path)]
    results, errors = [], []

    def run(t):
        try:
            for i in range(6):
                shared = dkey("shared-{}".format(i))
                results.append((shared, stores[t % 2].issue(shared, 100, Builder())))
                own = dkey("thread-{}-{}".format(t, i))
                results.append((own, stores[t % 2].issue(own, 100, Builder())))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors
    new = [out.slot_id for _, out in results if out.status == "new"]
    assert len(new) == len(set(new)) == 6 + 4 * 6 and sorted(new) == list(range(30))
    for key, out in results:
        assert stores[0].lookup(key) == out.slot_id and stores[1].private(out.slot_id) == out.private
    check_consistent(tmp_path, new)


@needs_fork
def test_two_processes_issuing_at_once_get_distinct_ids(tmp_path):
    go_read, go_write = os.pipe()
    children = []
    for p in range(2):
        out_read, out_write = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child
            code = 1
            try:
                os.close(out_read)
                os.close(go_write)
                os.read(go_read, 1)  # both start together
                store, lines = SlotStore(tmp_path), []
                for i in range(12):
                    for key in (dkey("p{}-{}".format(p, i)), dkey("both-{}".format(i))):
                        out = store.issue(key, 100, Builder())
                        lines.append("{} {} {}".format(key.hex(), out.status, out.slot_id))
                os.write(out_write, "\n".join(lines).encode())
                code = 0
            finally:
                os._exit(code)
        os.close(out_write)
        children.append((pid, out_read))
    os.close(go_read)
    os.write(go_write, b"gg")
    os.close(go_write)
    got = []
    for pid, out_read in children:
        with os.fdopen(out_read, "rb") as f:
            got += [line.split() for line in f.read().decode().splitlines()]
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    new = [int(slot_id) for _, status, slot_id in got if status == "new"]
    assert len(new) == len(set(new)) == 36 and sorted(new) == list(range(36))
    by_key = {}
    for key, status, slot_id in got:
        by_key.setdefault(key, set()).add(int(slot_id))
    assert all(len(ids) == 1 for ids in by_key.values()), "one key, two slots"
    shared = [status for key, status, _ in got if key in {dkey("both-{}".format(i)).hex() for i in range(12)}]
    assert shared.count("new") == 12 and shared.count("hit") == 12
    store = check_consistent(tmp_path, new)
    assert store.version() == 36


@needs_fork
@fork_warning
def test_a_forked_child_issues_while_another_thread_of_the_parent_held_the_store(tmp_path):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 10, Builder())
    inside, release = threading.Event(), threading.Event()
    parent_result = []

    def slow_build(slot_id):
        inside.set()
        release.wait(10)
        return records(slot_id)

    thread = threading.Thread(target=lambda: parent_result.append(store.issue(dkey(1), 10, slow_build)))
    thread.start()
    assert inside.wait(10)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child: the parent's thread holds both locks
        code = 1
        try:
            os.close(read)
            assert store._lock_fd is None, "the inherited file-lock descriptor was not dropped"
            assert not store._lock.locked(), "the parent's lock"
            out = store.issue(dkey(2), 10, Builder())  # waits for the parent's file lock
            os.write(write, "{} {}".format(out.status, out.slot_id).encode())
            code = 0
        finally:
            os._exit(code)
    os.close(write)
    release.set()
    thread.join(10)
    with os.fdopen(read, "rb") as f:
        child = f.read().decode()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert parent_result[0].slot_id == 1 and child == "new 2"
    assert store.issue(dkey(3), 10, Builder()).slot_id == 3
    check_consistent(tmp_path, [0, 1, 2, 3])


def test_copies_and_pickles_are_new_store_objects(tmp_path):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 3, Builder())
    for other in (copy.copy(store), copy.deepcopy(store), pickle.loads(pickle.dumps(store))):
        assert other is not store and other._lock is not store._lock
        assert other.directory == store.directory and other.lookup(dkey(0)) == 0


# -- the private side stays private


def test_file_modes(tmp_path):
    store = SlotStore(tmp_path / "s")
    store.issue(dkey(0), 3, Builder())
    mode = lambda name: stat.S_IMODE(os.stat(tmp_path / "s" / name).st_mode)  # noqa: E731
    assert mode(slots.PRIVATE_FILE) == 0o600 and mode(slots.LOCK_FILE) == 0o600
    assert mode(slots.PUBLIC_FILE) == 0o644
    assert stat.S_IMODE(os.stat(tmp_path / "s").st_mode) & 0o077 == 0
    store.issue(dkey(1), 3, Builder())  # rewritten: still owner-only
    assert mode(slots.PRIVATE_FILE) == 0o600


def _encodings(data: bytes):
    yield data
    yield data.hex().encode()
    yield data.hex().upper().encode()
    for enc in (base64.b64encode, base64.urlsafe_b64encode):
        for shift in range(3):  # aligned at any offset inside a longer base64 string
            text = enc(b"\x00" * shift + data).rstrip(b"=")
            yield text[(4 * shift + 2) // 3 + 1:-2]


def test_the_public_file_holds_no_dedup_key_and_no_private_field(tmp_path, caplog):
    store = SlotStore(tmp_path)
    k_i = b"\xa5\x5a" * 8
    key = dkey("secret function")
    caplog.set_level(logging.DEBUG, logger="pyfe4ai")
    out = store.issue(key, 1, lambda slot_id: ({"label": "x", "pk": b"\x01" * 32},
                                               {"k_i": k_i, "stored_key": b"S" * 40}))
    with pytest.raises(FEKeyError) as cap:
        store.issue(key, 1, never)
    with open(tmp_path / slots.PUBLIC_FILE, "rb") as f:
        public = f.read()
    with open(tmp_path / slots.PRIVATE_FILE, "rb") as f:
        private = f.read()
    assert key.hex().encode() in private, "the test's encodings do not find the key"
    exposed = [public, repr(store).encode(), repr(out).encode(), str(cap.value).encode(),
               "\n".join(r.getMessage() for r in caplog.records).encode()]
    for secret in (key, k_i, b"S" * 40):
        for form in _encodings(secret):
            assert len(form) >= 8
            assert not any(form in text for text in exposed), "a private value in a public place"
    for field in (b"k_i", b"stored_key", b"dedup", b"next_slot_id", b"pending"):
        assert field not in public


# -- damaged files fail closed


def _damage(path, how):
    with open(path, "rb") as f:
        data = f.read()
    if how == "truncated":
        data = data[:len(data) // 2]
    elif how == "empty":
        data = b""
    elif how == "garbage":
        data = b"\xff\xfe not json"
    elif how == "array":
        data = b"[]"
    else:
        doc = json.loads(data)
        how(doc)
        data = json.dumps(doc).encode()
    with open(path, "wb") as f:
        f.write(data)


def _set(name, value):
    return lambda doc: doc.__setitem__(name, value)


DAMAGES = {
    "public truncated": (slots.PUBLIC_FILE, "truncated"),
    "public empty": (slots.PUBLIC_FILE, "empty"),
    "public garbage": (slots.PUBLIC_FILE, "garbage"),
    "public not an object": (slots.PUBLIC_FILE, "array"),
    "public other format": (slots.PUBLIC_FILE, _set("format", "other")),
    "public other version": (slots.PUBLIC_FILE, _set("format_version", 2)),
    "public version mismatch": (slots.PUBLIC_FILE, _set("slot_list_version", 1)),
    "public ids not increasing": (slots.PUBLIC_FILE, lambda d: d["slots"].reverse()),
    "public first slot dropped": (slots.PUBLIC_FILE,
                                  lambda d: (d["slots"].pop(0), d.__setitem__("slot_list_version", 1))),
    "public bad bytes": (slots.PUBLIC_FILE, lambda d: d["slots"][0]["pks"][0].__setitem__("__bytes__", "!!")),
    "private truncated": (slots.PRIVATE_FILE, "truncated"),
    "private garbage": (slots.PRIVATE_FILE, "garbage"),
    "private other format": (slots.PRIVATE_FILE, _set("format", slots.PUBLIC_FORMAT)),
    "private next id reused": (slots.PRIVATE_FILE, _set("next_slot_id", 2)),
    "private next id missing": (slots.PRIVATE_FILE, lambda d: d.pop("next_slot_id")),
    "private record of no slot": (slots.PRIVATE_FILE, lambda d: d["private"].__setitem__("7", {})),
    "private record id not canonical": (slots.PRIVATE_FILE, lambda d: d["private"].__setitem__("01", {})),
    "dedup key short": (slots.PRIVATE_FILE, lambda d: d["dedup"].__setitem__("ab", 0)),
    "dedup to no record": (slots.PRIVATE_FILE, lambda d: d["dedup"].__setitem__("cd" * 32, 9)),
    "two dedup keys of one slot": (slots.PRIVATE_FILE, lambda d: d["dedup"].__setitem__("ef" * 32, 0)),
    "listed slot without private record": (slots.PRIVATE_FILE, lambda d: (
        d["private"].pop("0"), d.__setitem__("dedup", {k: v for k, v in d["dedup"].items() if v != 0}))),
    "private record without dedup entry": (slots.PRIVATE_FILE, lambda d: d.__setitem__(
        "dedup", {k: v for k, v in d["dedup"].items() if v != 0})),
    "pending missing": (slots.PRIVATE_FILE, lambda d: d.pop("pending")),
    "pending differs from the listed record": (slots.PRIVATE_FILE,
                                               lambda d: d["pending"]["2"].__setitem__("label", "other")),
    "pending names another slot": (slots.PRIVATE_FILE, lambda d: d["pending"]["2"].__setitem__("slot_id", 0)),
    "pending without private record": (slots.PRIVATE_FILE, lambda d: d["pending"].__setitem__("5", {"slot_id": 5})),
    "pending below a listed id": (slots.PRIVATE_FILE, lambda d: (  # the burned id 1, as if issued
        d["private"].__setitem__("1", {}), d["dedup"].__setitem__("ab" * 32, 1),
        d["pending"].__setitem__("1", {"slot_id": 1}))),
}


@pytest.mark.parametrize("name", sorted(DAMAGES))
def test_damaged_files_fail_closed(tmp_path, name):
    store = SlotStore(tmp_path)  # slots 0 and 2 listed (1 burned), slot 2's pending copy kept
    store.issue(dkey(0), 5, Builder())
    with pytest.raises(RuntimeError):
        store.issue(dkey(1), 5, Builder(fail=RuntimeError("x")))
    store.issue(dkey(1), 5, Builder())
    assert listed_on_disk(tmp_path) == [0, 2] and list(on_disk(tmp_path, slots.PRIVATE_FILE)["pending"]) == ["2"]
    target, how = DAMAGES[name]
    _damage(tmp_path / target, how)
    before = files(tmp_path)
    for call in (lambda: store.issue(dkey(2), 5, never), lambda: store.issue(dkey(0), 5, never),
                 store.public_slots, store.version, lambda: store.private(0), lambda: store.lookup(dkey(0))):
        with pytest.raises(FEConfigError, match="slot store file") as err:
            call()
        assert dkey(0).hex() not in str(err.value)
    assert files(tmp_path) == before, "a damaged store was rewritten"


def test_a_slot_list_without_its_last_slot_is_completed_from_the_pending_copy(tmp_path):
    """The last slot's public record is still pending in the private file: a
    slot list that lost it (the state after a failed public write) gets it
    back; the slots before it have no pending copy and are not."""
    store = SlotStore(tmp_path)
    for i in range(3):
        store.issue(dkey(i), 5, Builder("r{}".format(i)))
    complete = files(tmp_path)
    _damage(tmp_path / slots.PUBLIC_FILE, lambda d: (d["slots"].pop(), d.__setitem__("slot_list_version", 2)))
    assert store.lookup(dkey(2)) == 2 and files(tmp_path) == complete
    check_consistent(tmp_path)


def test_a_missing_private_file_fails_closed_but_a_missing_slot_list_does_not(tmp_path):
    store = SlotStore(tmp_path)
    with pytest.raises(RuntimeError):
        store.issue(dkey(0), 5, Builder(fail=RuntimeError("x")))
    assert not (tmp_path / slots.PUBLIC_FILE).exists(), "only the reservation was written"
    assert store.issue(dkey(0), 5, Builder()).slot_id == 1
    os.unlink(tmp_path / slots.PRIVATE_FILE)
    with pytest.raises(FEConfigError, match="missing while the slot list exists"):
        store.issue(dkey(1), 5, never)


def test_an_unreadable_file_fails_closed(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 5, Builder())
    os.chmod(tmp_path / slots.PRIVATE_FILE, 0)
    try:
        with pytest.raises(FEConfigError, match="unreadable"):
            store.lookup(dkey(0))
    finally:
        os.chmod(tmp_path / slots.PRIVATE_FILE, 0o600)


# -- the store trusts only a directory and files of its own user


def _every_call(store):
    return (lambda: store.issue(dkey(0), 3, never), lambda: store.issue(dkey(9), 3, never), store.public_slots,
            store.version, lambda: store.private(0), lambda: store.lookup(dkey(0)))


def _refused(store, match, path):
    for call in _every_call(store):
        with pytest.raises(FEConfigError, match=match) as err:
            call()
        assert str(path) in str(err.value) and dkey(0).hex() not in str(err.value)


@pytest.mark.parametrize("umask", [0o000, 0o022, 0o077, 0o277])
def test_a_new_directory_is_made_owner_only_whatever_the_umask(tmp_path, umask):
    (tmp_path / "parent").mkdir()
    old = os.umask(umask)
    try:
        store = SlotStore(tmp_path / "parent" / "store")
        store.issue(dkey(0), 3, Builder())
        store.issue(dkey(1), 3, Builder())  # reopens the files made under that umask
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(store.directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(os.path.join(store.directory, slots.LOCK_FILE)).st_mode) == 0o600
    assert SlotStore(store.directory).version() == 2


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o730, 0o703])
def test_a_directory_other_users_can_write_is_refused(tmp_path, mode):
    directory = tmp_path / "store"
    directory.mkdir()
    os.chmod(directory, mode)
    _refused(SlotStore(directory), "mode {:04o} lets other users write".format(mode), directory)
    assert os.listdir(directory) == [] and stat.S_IMODE(os.stat(directory).st_mode) == mode, "changed"


def test_a_store_whose_directory_became_writable_by_others_is_refused(tmp_path):
    store = SlotStore(tmp_path / "s")
    store.issue(dkey(0), 3, Builder())
    before = files(tmp_path / "s")
    os.chmod(tmp_path / "s", 0o777)
    _refused(store, "mode 0777", tmp_path / "s")
    assert files(tmp_path / "s") == before


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o711, 0o701])
def test_a_directory_others_may_only_read_is_used(tmp_path, mode):
    directory = tmp_path / "store"
    directory.mkdir()
    os.chmod(directory, mode)
    store = SlotStore(directory)
    assert store.issue(dkey(0), 3, Builder())[:2] == ("new", 0) and store.lookup(dkey(0)) == 0


def test_a_symlinked_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    SlotStore(real).issue(dkey(0), 3, Builder())
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    _refused(SlotStore(tmp_path / "link"), "symbolic link", tmp_path / "link")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    _refused(SlotStore(tmp_path / "dangling"), "symbolic link", tmp_path / "dangling")
    assert not (tmp_path / "nowhere").exists(), "a dangling link was followed"


def test_a_symbolic_link_above_the_directory_is_followed(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "up").symlink_to(tmp_path / "real", target_is_directory=True)
    SlotStore(tmp_path / "up" / "store").issue(dkey(0), 3, Builder())
    assert SlotStore(tmp_path / "real" / "store").lookup(dkey(0)) == 0


def test_a_file_in_place_of_the_directory_is_refused(tmp_path):
    (tmp_path / "store").write_bytes(b"")
    _refused(SlotStore(tmp_path / "store"), "not a directory", tmp_path / "store")


@pytest.mark.parametrize("name", [slots.PUBLIC_FILE, slots.PRIVATE_FILE, slots.LOCK_FILE])
def test_a_symlinked_store_file_is_refused(tmp_path, name):
    directory = tmp_path / "s"
    SlotStore(directory).issue(dkey(0), 3, Builder())
    os.replace(directory / name, tmp_path / name)  # a valid file, now outside the directory
    os.symlink(tmp_path / name, directory / name)
    target = (tmp_path / name).read_bytes()
    _refused(SlotStore(directory), "symbolic link", directory / name)
    assert os.path.islink(directory / name) and (tmp_path / name).read_bytes() == target, "written through the link"


@pytest.mark.parametrize("kind", ["fifo", "directory"])
@pytest.mark.parametrize("name", [slots.PUBLIC_FILE, slots.PRIVATE_FILE, slots.LOCK_FILE])
def test_a_store_file_that_is_not_a_regular_file_is_refused(tmp_path, name, kind):
    directory = tmp_path / "s"
    SlotStore(directory).issue(dkey(0), 3, Builder())
    os.unlink(directory / name)
    if kind == "fifo":
        os.mkfifo(directory / name, 0o600)  # opening it for reading would block without O_NONBLOCK
    else:
        (directory / name).mkdir(mode=0o700)
    _refused(SlotStore(directory), "not a regular file", directory / name)


@pytest.mark.parametrize("name,mode", [(slots.PRIVATE_FILE, 0o640), (slots.PRIVATE_FILE, 0o604),
                                       (slots.LOCK_FILE, 0o644), (slots.PUBLIC_FILE, 0o664),
                                       (slots.PUBLIC_FILE, 0o646)])
def test_store_files_other_users_can_write_or_read_privately_are_refused(tmp_path, name, mode):
    directory = tmp_path / "s"
    SlotStore(directory).issue(dkey(0), 3, Builder())
    os.chmod(directory / name, mode)
    _refused(SlotStore(directory), "mode {:04o} lets other users".format(mode), directory / name)


def test_a_directory_of_another_user_is_refused(tmp_path, monkeypatch):
    store = SlotStore(tmp_path / "s")
    store.issue(dkey(0), 3, Builder())
    uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: uid + 1)  # as if another user ran the store
    _refused(store, "owned by uid {}, not by this process's uid {}".format(uid, uid + 1), tmp_path / "s")


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="chown to another uid needs root")
@pytest.mark.parametrize("name", [slots.PUBLIC_FILE, slots.PRIVATE_FILE, slots.LOCK_FILE])
def test_a_store_file_of_another_user_is_refused(tmp_path, name):  # pragma: no cover - root only
    directory = tmp_path / "s"
    SlotStore(directory).issue(dkey(0), 3, Builder())
    os.chown(directory / name, 65534, -1)
    _refused(SlotStore(directory), "owned by uid 65534", directory / name)


# -- a cap the caller keeps: a callable cap, run_locked (appended)


def test_a_callable_cap_is_read_once_under_the_lock(tmp_path):
    store = SlotStore(tmp_path)
    reads = []

    def cap():
        reads.append(store._lock_fd is not None)                       # inside the store's file lock
        return 2

    assert store.issue(dkey(0), cap, Builder()).slot_id == 0 and reads == [True]
    assert store.issue(dkey(0), cap, never).status == "hit" and reads == [True, True]
    store.issue(dkey(1), cap, Builder())
    for key in (dkey(0), dkey(2)):                                     # at the cap: hit and miss alike
        with pytest.raises(FEKeyError, match="max_keys = 2"):
            store.issue(key, cap, never)
    assert store.issue(dkey(2), lambda: 3, Builder()).slot_id == 2     # a raised cap


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "3", None])
def test_a_callable_cap_must_return_a_positive_integer(tmp_path, value):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 3, Builder())
    before = files(tmp_path)
    with pytest.raises(FEValidationError, match="max_keys"):
        store.issue(dkey(1), lambda: value, never)
    assert files(tmp_path) == before                                   # nothing written, no id burned
    assert store.issue(dkey(1), 3, Builder()).slot_id == 1


def test_a_callable_cap_that_raises_or_uses_the_store_writes_nothing(tmp_path):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 3, Builder())
    before = files(tmp_path)

    def failing():
        raise RuntimeError("cap unreadable")

    with pytest.raises(RuntimeError, match="cap unreadable"):
        store.issue(dkey(1), failing, never)
    with pytest.raises(FESchemeError, match="must not call it"):
        store.issue(dkey(1), store.version, never)
    assert files(tmp_path) == before and store.issue(dkey(1), 3, Builder()).slot_id == 1


def test_run_locked_runs_under_the_lock_of_issue(tmp_path):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 5, Builder())
    issued, started = [], threading.Event()

    def other_issue():
        started.set()
        issued.append(SlotStore(tmp_path).issue(dkey(1), 5, Builder()).slot_id)

    def fn(listed):
        assert [slot["slot_id"] for slot in listed] == [0]
        thread = threading.Thread(target=other_issue)
        thread.start()
        started.wait(5)
        thread.join(0.3)
        assert thread.is_alive() and issued == []                      # the issue waits for the lock
        return thread, "result"

    thread, result = store.run_locked(fn)
    thread.join(10)
    assert result == "result" and issued == [1] and store.version() == 2


def test_run_locked_gives_the_slot_list_and_checks_its_callable(tmp_path):
    store = SlotStore(tmp_path)
    store.issue(dkey(0), 5, Builder())
    listed = store.run_locked(lambda slots_now: slots_now)
    assert listed == store.public_slots()
    listed[0]["label"] = "changed"                                     # new objects
    assert store.public_slots()[0]["label"] == "slot-0"
    before = files(tmp_path)
    with pytest.raises(FEValidationError, match="callable"):
        store.run_locked(None)
    with pytest.raises(FESchemeError, match="must not call it"):
        store.run_locked(lambda slots_now: store.version())
    with pytest.raises(RuntimeError, match="refused"):
        store.run_locked(lambda slots_now: (_ for _ in ()).throw(RuntimeError("refused")))
    assert files(tmp_path) == before and store.issue(dkey(1), 5, Builder()).slot_id == 1
