"""What a forked child process renews: the one fork handler of the library.

``os.fork()`` copies the whole memory of the process but only the thread that
called it. A lock that another thread held at the fork stays held for ever
in the child, and per-process random state (buffered OS random bytes, the
secret parts of one-time precomputations) would be used by both processes.
This module registers the library's only ``os.register_at_fork`` handler
(``after_in_child``); in the child it runs, in the order they were
registered:

* :func:`renew_lock` -- replace a process-wide lock (a module attribute) with
  a new one;
* :func:`call_in_child` -- call a function (e.g. drop the buffered OS random
  bytes of :mod:`pyfe4ai.primitives.sampling`);

and then :func:`track` -- call ``_after_fork_in_child()`` of every tracked
object still alive (e.g. mark the one-time precomputations used and drop
their secret parts, :mod:`pyfe4ai.core.one_time`).

Nothing runs in the parent, nor before the fork. A handler that raises is
reported on ``stderr`` as an ``os.register_at_fork`` hook's exception is
(``Exception ignored in: ...`` and the traceback), and the other handlers
still run. The registrations are made at import of the modules that own the
state, so the child renews what the parent had imported. The process-wide
state of the library and what a fork, a copy or a pickle does to it are
listed on the "Threads, processes and copies" page of the documentation.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
import weakref

#: The handlers run in a forked child, in registration order.
_HANDLERS: list = []
#: Objects whose ``_after_fork_in_child()`` a forked child calls (weakly held).
_TRACKED: "weakref.WeakSet" = weakref.WeakSet()


class _RenewLock:
    """Handler of :func:`renew_lock`: ``setattr(owner, name, factory())``."""

    __slots__ = ("owner", "name", "factory")

    def __init__(self, owner, name: str, factory) -> None:
        self.owner, self.name, self.factory = owner, name, factory

    def __call__(self) -> None:
        setattr(self.owner, self.name, self.factory())

    def __repr__(self) -> str:
        return "<renew {}.{}>".format(getattr(self.owner, "__name__", self.owner), self.name)


def renew_lock(owner, name: str, factory=threading.Lock) -> None:
    """In a forked child, replace ``owner.<name>`` (a module's process-wide
    lock) with ``factory()``: the parent's may be held by another thread,
    which the child does not have.

    The code using the lock must read ``owner.<name>`` at every use (``with
    _LOCK:`` in the owning module does), not keep a reference to it.
    """
    if not hasattr(owner, name):
        raise AttributeError("{!r} has no attribute {!r} to renew".format(owner, name))
    _HANDLERS.append(_RenewLock(owner, name, factory))


def call_in_child(func) -> None:
    """Call *func()* in a forked child (held strongly, like ``os.register_at_fork``)."""
    if not callable(func):
        raise TypeError("call_in_child needs a callable, got {!r}".format(func))
    _HANDLERS.append(func)


def track(obj) -> None:
    """Call ``obj._after_fork_in_child()`` in a forked child, if *obj* is still
    alive then (*obj* is held weakly: tracking does not extend its lifetime)."""
    if not callable(getattr(obj, "_after_fork_in_child", None)):
        raise TypeError("{} has no _after_fork_in_child()".format(type(obj).__name__))
    _TRACKED.add(obj)


def _report(handler) -> None:
    """Report an exception of *handler* as CPython reports one of an
    ``os.register_at_fork`` hook (``PyErr_WriteUnraisable``)."""
    try:
        print("Exception ignored in: {!r}".format(handler), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    except Exception:  # pragma: no cover - stderr closed
        pass


def _after_fork_in_child() -> None:
    for handler in list(_HANDLERS):
        try:
            handler()
        except Exception:
            _report(handler)
    for obj in list(_TRACKED):
        hook = obj._after_fork_in_child
        try:
            hook()
        except Exception:
            _report(hook)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
