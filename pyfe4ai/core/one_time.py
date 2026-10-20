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

Offline / online encryption of the element-wise schemes with fresh
randomness per ciphertext (:class:`OfflineRandomMixin`): SIFE- and
MIFE-Paillier (ALS16 over DCR) encrypt as ``(g^r, (1 + N)^x pk^r)`` and
MIFE-DDH as ``(g^r, g^{x + u} h^r)`` with a fresh ``r`` per ciphertext:
everything but the plaintext term depends on ``r`` and the key only, and
costs full-size exponentiations. :meth:`OfflineRandomMixin.precompute` draws
``r`` for every element of given layer shapes -- in the order
``encrypt_lst_ndarray`` draws them -- and computes those powers;
``encrypt_lst_ndarray(..., precomputed=...)`` then only multiplies in the
plaintext term. With the same random source the ciphertexts are those of the
plain call. The precomputation holds the secret randomness of future
ciphertexts: keep it like a key. It is one-time
(:class:`OneTimePrecomputation`): the online phase validates its input, then
claims the precomputation atomically; ``copy.copy``, ``copy.deepcopy``,
pickling and use in another process are refused.

A scheme provides ``_random_part()`` (draws ``r``; returns what the
ciphertext needs of it) and ``_encrypt_with(lst_pt, part)`` (the ciphertext
of ``encrypt`` for that part, with its checks), and names in
``_online_bound`` the public parameter that bounds the plaintexts, if
``_encrypt_with`` checks one (checked before the precomputation is used).
"""

from __future__ import annotations

import os
import threading
import weakref

import numpy as np

from pyfe4ai._validation import scale_to_int
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError

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


def check_online_bound(scheme, layers: list) -> None:
    """The plaintext bound ``pp[scheme._online_bound]`` of the online phase,
    checked on all *layers* before the precomputation is claimed."""
    key = getattr(scheme, "_online_bound", None)
    if key is None:
        return
    bound = int(scheme.pp[key])
    if any(a.size and int(np.max(np.abs(a))) > bound for a in layers):
        raise FEValidationError("plaintext exceeds configured {}".format(key))


class RandomPrecomputation(OneTimePrecomputation):
    """The randomness-dependent parts of future ciphertexts. Secret, one-time."""

    _secret = ("parts",)

    def __init__(self, owner, shapes: list, parts: list) -> None:
        super().__init__()
        self.owner = owner
        self.key_id = id(owner.sk)
        self.shapes = shapes
        self.parts = parts

    def __repr__(self) -> str:
        return "RandomPrecomputation(shapes={}, used={})".format(self.shapes, self.used)


class OfflineRandomMixin:
    """precompute / precomputed= for the element-wise ndarray helpers."""

    def precompute(self, shapes) -> RandomPrecomputation:
        """Offline phase of ``encrypt_lst_ndarray`` for layers of *shapes*."""
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        shapes = [tuple(int(d) for d in s) if isinstance(s, (tuple, list)) else (int(s),) for s in shapes]
        parts = []
        for shape in shapes:
            layer = np.empty(shape, dtype=object)
            for i in np.ndindex(*shape):
                layer[i] = self._random_part()
            parts.append(layer)
        return RandomPrecomputation(self, shapes, parts)

    def _encrypt_precomputed(self, lst_ndarray: list, pre) -> list:
        if not isinstance(pre, RandomPrecomputation) or pre.owner is not self or pre.key_id != id(self.sk):
            raise FEValidationError("the precomputation belongs to another encryptor or key")
        if pre.used:
            raise FEValidationError("a precomputation can be used once")
        layers = [scale_to_int(a, self.precision) for a in lst_ndarray]
        if [tuple(np.shape(a)) for a in layers] != pre.shapes:
            raise FEValidationError("layer shapes differ from the precomputed ones {}".format(pre.shapes))
        check_online_bound(self, layers)
        (all_parts,) = pre._claim()  # atomic: of two callers one gets the parts
        out = []
        for ary, parts in zip(layers, all_parts):
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self._encrypt_with([int(w)], parts[i])
                parts[i] = None
            out.append(ary_ct)
        return out
