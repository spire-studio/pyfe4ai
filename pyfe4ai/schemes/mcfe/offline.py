"""Offline / online phases of the element-wise MCFE schemes over Z_p^* and Z_{N^2}^*.

In MCFE-DDH (CDGPP18), MCFE-Damgard and MCFE-Paillier (ABMPSW20 Fig. 3) a
ciphertext of x under label l is ``X(x) M(l) mod P``: ``X(x)`` is ``g^x`` or
``(1 + N)^x`` and the label mask ``M(l)`` a product of powers of the label's
hashes with the client's secret, which does not depend on x. Decryption
divides ``prod_i c_i^{y_i}`` by a key term of the label and the functional
key. Both label terms are the expensive part (exponents of 256 to 3072 bits)
and are known before the round, so:

* :meth:`OfflineMCFEMixin.precompute` computes a client's masks for every
  element label of given layer shapes; ``encrypt_lst_ndarray(...,
  precomputed=...)`` then only computes ``X(x)`` and one product per
  element. The ciphertexts are those of the plain call (the encryption is
  deterministic). Secret and one-time, as the masks of the Ring-LWE MCFE
  (see :mod:`pyfe4ai.schemes.one_time`: claimed atomically after the input
  is validated; ``copy.copy``, ``copy.deepcopy``, pickling and use in
  another process are refused).
* :meth:`OfflineMCFEMixin.prepare_key` computes the key terms of every
  element label for a functional key; ``decrypt_lst_ndarray_ct`` with that
  key then only multiplies the ciphertexts and solves the discrete log.

A scheme provides ``_label_mask(label)`` (one mask per slot),
``_encrypt_masked(lst_pt, masks)``, ``_key_term(dk, label)`` and
``decrypt(..., term=...)`` with the checks of its plain methods, and names
in ``_online_bound`` the public parameter that bounds the plaintexts, if
``_encrypt_masked`` checks one.
"""

from __future__ import annotations

import numpy as np

from pyfe4ai.schemes.mcfe.labels import element_label
from pyfe4ai.schemes.one_time import OneTimePrecomputation
from pyfe4ai.schemes.random_offline import check_online_bound
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError
from pyfe4ai.utils.validation import scale_to_int


def _shapes(shapes) -> list:
    return [tuple(int(d) for d in s) if isinstance(s, (tuple, list)) else (int(s),) for s in shapes]


class LabelPrecomputation(OneTimePrecomputation):
    """A client's label masks for one label and fixed layer shapes. Secret, one-time."""

    _secret = ("masks",)

    def __init__(self, owner, label: str, shapes: list, masks: list) -> None:
        super().__init__()
        self.owner = owner
        self.key_id = id(owner.sk)
        self.label = label
        self.shapes = shapes
        self.masks = masks  # per layer: an object array of per-slot mask lists

    def __repr__(self) -> str:
        return "LabelPrecomputation(label={!r}, shapes={}, used={})".format(self.label, self.shapes, self.used)


class PreparedKeyTerms:
    """The key terms of every element label of a functional key (not secret beyond the key)."""

    def __init__(self, label: str, shapes: list, terms: list) -> None:
        self.label = label
        self.shapes = shapes
        self.terms = terms

    def __repr__(self) -> str:
        return "PreparedKeyTerms(label={!r}, shapes={})".format(self.label, self.shapes)


class OfflineMCFEMixin:
    """precompute / precomputed= / prepare_key for the element-wise MCFE schemes."""

    def precompute(self, label: str, shapes) -> LabelPrecomputation:
        """Offline phase of ``encrypt_lst_ndarray`` for *label* and layers of *shapes*."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        shapes = _shapes(shapes)
        masks = []
        for l, shape in enumerate(shapes):
            layer = np.empty(shape, dtype=object)
            for i in np.ndindex(*shape):
                layer[i] = self._label_mask(element_label(label, l, i))
            masks.append(layer)
        return LabelPrecomputation(self, label, shapes, masks)

    def _encrypt_precomputed(self, lst_ndarray: list, label: str, pre) -> list:
        """Online phase: ``X(x)`` times the precomputed masks."""
        if not isinstance(pre, LabelPrecomputation) or pre.owner is not self or pre.key_id != id(self.sk):
            raise FEValidationError("the precomputation belongs to another client or key")
        if pre.label != label:
            raise FEValidationError("the precomputation is for another label")
        if pre.used:
            raise FEValidationError("a precomputation can be used once")
        layers = [scale_to_int(a, self.precision) for a in lst_ndarray]
        if [tuple(np.shape(a)) for a in layers] != pre.shapes:
            raise FEValidationError("layer shapes differ from the precomputed ones {}".format(pre.shapes))
        check_online_bound(self, layers)
        (all_masks,) = pre._claim()  # atomic: of two callers one gets the masks
        out = []
        for ary, masks in zip(layers, all_masks):
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self._encrypt_masked([int(w)], masks[i])
                masks[i] = None
            out.append(ary_ct)
        return out

    def prepare_key(self, dk: dict, label: str, shapes) -> dict:
        """*dk* with the key terms of every element label of *label* and *shapes*:
        ``decrypt_lst_ndarray_ct`` then skips the label hashing and exponentiations."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        shapes = _shapes(shapes)
        terms = []
        for l, shape in enumerate(shapes):
            layer = np.empty(shape, dtype=object)
            for i in np.ndindex(*shape):
                layer[i] = self._key_term(dk, element_label(label, l, i))
            terms.append(layer)
        return dict(dk, key_terms=PreparedKeyTerms(label, shapes, terms))

    @staticmethod
    def _prepared_terms(dk: dict, label, sample: list):
        """The prepared terms of *dk* if they are for *label* and these layers."""
        terms = dk.get("key_terms") if isinstance(dk, dict) else None
        if not isinstance(terms, PreparedKeyTerms) or terms.label != label:
            return None
        if [tuple(np.shape(a)) for a in sample] != terms.shapes:
            return None
        return terms.terms
