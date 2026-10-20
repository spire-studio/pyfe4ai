"""One-time use of a precomputation (the offline phase of an encryption).

A precomputation holds the secret randomness or label masks of future
ciphertexts; using it twice encrypts two plaintexts with the same randomness
(or twice under one label), and their difference follows without any key.
:class:`OneTimePrecomputation` makes the use atomic and keeps it in one
process:

* ``_claim`` checks and sets ``used`` under a
  per-object lock and moves the secret parts out of the object, so of two
  threads exactly one gets them;
* a precomputation records the process that made it and refuses to be used
  in another one; after ``os.fork()`` the child's copies are also marked
  used and dropped (``os.register_at_fork``);
* ``copy.copy``, ``copy.deepcopy`` and pickling raise ``TypeError``: a copy
  would be a second, unused handle on the same randomness. Code that clones
  the object deliberately (copying its attributes) is not prevented; the
  object is a handle on a secret, not a capability that cannot be forged.

The schemes validate their inputs before they claim the precomputation, so
refused inputs do not consume it.
"""

from __future__ import annotations

import os
import threading
import weakref

from pyfe4ai.utils.exceptions import FEValidationError

_LIVE: "weakref.WeakSet[OneTimePrecomputation]" = weakref.WeakSet()


def _invalidate_after_fork() -> None:
    for pre in list(_LIVE):
        pre._invalidate()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_invalidate_after_fork)


class OneTimePrecomputation:
    """Base of the offline phases' results: secret, one-time, one process.

    Subclasses list in ``_secret`` the attributes that hold the secret
    parts (dropped by ``_claim`` and in a forked child).
    """

    _secret: tuple = ()

    def __init__(self) -> None:
        self.used = False
        self._lock = threading.Lock()
        self._pid = os.getpid()
        _LIVE.add(self)

    def _claim(self, *names: str) -> tuple:
        """Mark the precomputation used and return the attributes *names*,
        which are set to ``None`` (all of ``_secret`` if none are given).

        Raises:
            FEValidationError: If it was used already or made in another process.
        """
        names = names or self._secret
        if self._pid != os.getpid():  # checked first: a fork may have copied the lock held
            raise FEValidationError("a precomputation cannot be used in another process than the one that made it")
        with self._lock:
            if self.used:
                raise FEValidationError("a precomputation can be used once")
            self.used = True
            parts = tuple(getattr(self, name) for name in names)
            for name in names:
                setattr(self, name, None)
        return parts

    def _invalidate(self) -> None:
        """In a forked child: used, and the secret parts dropped."""
        self.used = True
        for name in self._secret:
            setattr(self, name, None)

    def __copy__(self):
        raise TypeError("a one-time precomputation cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("a one-time precomputation cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("a one-time precomputation cannot be pickled")

    def __reduce__(self):
        raise TypeError("a one-time precomputation cannot be pickled")
