"""Offline / online encryption of the element-wise schemes with fresh randomness per ciphertext.

SIFE- and MIFE-Paillier (ALS16 over DCR) encrypt as ``(g^r, (1 + N)^x pk^r)``
and MIFE-DDH as ``(g^r, g^{x + u} h^r)`` with a fresh ``r`` per ciphertext:
everything but the plaintext term depends on ``r`` and the key only, and
costs full-size exponentiations. :meth:`OfflineRandomMixin.precompute` draws
``r`` for every element of given layer shapes -- in the order
``encrypt_lst_ndarray`` draws them -- and computes those powers;
``encrypt_lst_ndarray(..., precomputed=...)`` then only multiplies in the
plaintext term. With the same random source the ciphertexts are those of the
plain call. The precomputation holds the secret randomness of future
ciphertexts: keep it like a key, and use it once.

A scheme provides ``_random_part()`` (draws ``r``; returns what the
ciphertext needs of it) and ``_encrypt_with(lst_pt, part)`` (the ciphertext
of :meth:`encrypt` for that part, with its checks).
"""

from __future__ import annotations

import numpy as np

from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError
from pyfe4ai.utils.validation import scale_to_int


class RandomPrecomputation:
    """The randomness-dependent parts of future ciphertexts. Secret, one-time."""

    def __init__(self, owner, shapes: list, parts: list) -> None:
        self.owner = owner
        self.key_id = id(owner.sk)
        self.shapes = shapes
        self.parts = parts
        self.used = False

    def __repr__(self) -> str:
        return "RandomPrecomputation(shapes={}, used={})".format(self.shapes, self.used)


class OfflineRandomMixin:
    """precompute / precomputed= for the element-wise ndarray helpers."""

    def precompute(self, shapes) -> RandomPrecomputation:
        """Offline phase of :meth:`encrypt_lst_ndarray` for layers of *shapes*."""
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
        pre.used = True
        out = []
        for ary, parts in zip(layers, pre.parts):
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self._encrypt_with([int(w)], parts[i])
            out.append(ary_ct)
        pre.parts = None  # one-time
        return out
