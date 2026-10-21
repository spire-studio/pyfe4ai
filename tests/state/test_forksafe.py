"""The library's fork handler (:mod:`pyfe4ai.core.forksafe`).

In a forked child the process-wide locks are new (even when another thread
of the parent held them at the fork), the OS random buffer is empty, the
registered functions have run and the tracked objects are notified; in the
parent nothing changes. A failing handler does not stop the others.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import types
import weakref

import pytest

from pyfe4ai.backend import native_impl
from pyfe4ai.core import forksafe
from pyfe4ai.core.one_time import OneTimePrecomputation
from pyfe4ai.primitives import sampling
from pyfe4ai.schemes.threshold import store

needs_fork = pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
#: fork() of a process with threads: Python warns (3.12+); the point of these tests
fork_warning = pytest.mark.filterwarnings("ignore::DeprecationWarning")

LIBRARY_LOCKS = [(sampling, "_XOF_LOCK"), (native_impl, "_contexts_lock"), (store, "_SHARE_CACHE_LOCK"),
                 (store, "_STORE_LOCK")]


@pytest.fixture
def registry(monkeypatch):
    """The handlers and tracked objects of this test only (the library's stay
    registered and run too; the test's are dropped afterwards)."""
    monkeypatch.setattr(forksafe, "_HANDLERS", list(forksafe._HANDLERS))
    tracked = weakref.WeakSet(forksafe._TRACKED)
    monkeypatch.setattr(forksafe, "_TRACKED", tracked)
    return tracked


def in_child(check):
    """Run ``check()`` in a forked child; ``(exit code, message)``: 0 and its
    return value, or 1 and the exception."""
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        code = 0
        try:
            msg = check() or ""
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            code, msg = 1, "{}: {}".format(type(exc).__name__, exc)
        os.close(read)
        os.write(write, msg.encode()[:60000])
        os._exit(code)
    os.close(write)
    chunks = []
    while True:
        chunk = os.read(read, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), b"".join(chunks).decode()


def held_by_another_thread(owner, name):
    """Context: ``owner.<name>`` held by another thread until the block ends."""
    class Holder:
        def __enter__(self):
            self.lock = getattr(owner, name)
            self.held, self.release = threading.Event(), threading.Event()

            def hold():
                with self.lock:
                    self.held.set()
                    self.release.wait(10)

            self.thread = threading.Thread(target=hold)
            self.thread.start()
            assert self.held.wait(10)
            return self.lock

        def __exit__(self, *exc):
            self.release.set()
            self.thread.join(10)

    return Holder()


@needs_fork
@fork_warning
@pytest.mark.parametrize("owner,name", LIBRARY_LOCKS, ids=["{}.{}".format(o.__name__, n) for o, n in LIBRARY_LOCKS])
def test_a_process_wide_lock_is_new_and_free_in_the_child(owner, name):
    with held_by_another_thread(owner, name) as old:

        def child():
            new = getattr(owner, name)
            assert new is not old, "the parent's lock"
            assert new.acquire(timeout=2), "the new lock is held"
            assert old.locked(), "the parent's lock object changed in the child"

        assert in_child(child) == (0, "")
        assert getattr(owner, name) is old and old.locked(), "the parent's lock changed"


@needs_fork
@fork_warning
def test_renew_lock_replaces_a_registered_lock_in_the_child_only(registry):
    ns = types.ModuleType("probe_namespace")
    ns.LOCK = threading.RLock()
    forksafe.renew_lock(ns, "LOCK", factory=threading.RLock)
    with held_by_another_thread(ns, "LOCK") as old:

        def child():
            assert ns.LOCK is not old and type(ns.LOCK) is type(old)
            assert ns.LOCK.acquire(timeout=2)

        assert in_child(child) == (0, "")
        assert ns.LOCK is old


@needs_fork
def test_the_os_random_buffer_is_empty_in_the_child():
    source = sampling._OS_SOURCE
    source.getrandbits(64)
    buf, pos, lock = source._buf, source._pos, source._lock
    assert buf and pos

    def child():
        assert source._buf == b"" and source._pos == 0, "the parent's random bytes"
        assert source._lock is not lock
        return str(source.getrandbits(64))

    code, drawn = in_child(child)
    assert code == 0, drawn
    assert source._buf is buf and source._pos == pos and source._lock is lock, "the parent's buffer changed"


@needs_fork
def test_registered_functions_and_tracked_objects_run_in_the_child_only(registry):
    cache = {"key": "value"}
    forksafe.call_in_child(cache.clear)

    class Tracked:
        def __init__(self):
            self.state = "parent"
            forksafe.track(self)

        def _after_fork_in_child(self):
            self.state = "child"

    kept, gone = Tracked(), Tracked()
    gone_ref = weakref.ref(gone)
    del gone
    gc.collect()
    assert gone_ref() is None, "tracking kept an object alive"
    assert len(registry) == 1

    def child():
        assert cache == {}, "the cache was not emptied"
        assert kept.state == "child"

    assert in_child(child) == (0, "")
    assert cache == {"key": "value"} and kept.state == "parent"


@needs_fork
def test_a_failing_handler_does_not_stop_the_others(registry, monkeypatch):
    def broken():
        raise RuntimeError("broken handler")

    after = {"ran": False}
    forksafe.call_in_child(broken)
    forksafe.call_in_child(lambda: after.update(ran=True))
    reported = []
    monkeypatch.setattr(forksafe, "_report", lambda handler: reported.append(handler))

    def child():
        assert after["ran"], "a handler after the failing one did not run"
        assert reported == [broken], reported

    assert in_child(child) == (0, "")
    assert not after["ran"] and reported == []


def test_report_prints_like_an_at_fork_hook(capsys):
    def handler():
        raise ValueError("bad state")

    try:
        handler()
    except ValueError:
        forksafe._report(handler)
    err = capsys.readouterr().err
    assert err.startswith("Exception ignored in: <function") and "ValueError: bad state" in err


@needs_fork
def test_one_time_precomputations_are_used_in_the_child():
    class Pre(OneTimePrecomputation):
        _secret = ("parts",)

        def __init__(self):
            super().__init__()
            self.parts = [1, 2, 3]

    pre = Pre()
    assert pre in forksafe._TRACKED

    def child():
        assert pre.used and pre.parts is None

    assert in_child(child) == (0, "")
    assert not pre.used and pre.parts == [1, 2, 3]
    assert pre._claim() == ([1, 2, 3],)


def test_the_library_registers_its_process_wide_state():
    renewed = {(h.owner.__name__, h.name) for h in forksafe._HANDLERS if isinstance(h, forksafe._RenewLock)}
    assert renewed == {(o.__name__, n) for o, n in LIBRARY_LOCKS}
    assert sampling._OS_SOURCE._reset in forksafe._HANDLERS


def test_registration_is_checked(registry):
    ns = types.ModuleType("probe_namespace")
    with pytest.raises(AttributeError, match="no attribute 'LOCK'"):
        forksafe.renew_lock(ns, "LOCK")
    with pytest.raises(TypeError, match="callable"):
        forksafe.call_in_child(42)
    with pytest.raises(TypeError, match="_after_fork_in_child"):
        forksafe.track(object())
