"""
Ring-LWE identity-based decentralised multi-client inner-product FE.

| Bermudo Mera, Karmakar, Marc, Soleimanian:
| "Efficient Lattice-Based Inner-Product Functional Encryption", PKC 2022
| (ePrint 2021/046) -- Section 6: the identity-based MCFE of
| :mod:`pyfe4ai.schemes.mcfe.ring_lwe`, decentralised with the compiler of
| Abdalla, Benhamouda, Kohlweiss, Waldner (PKC 2019) extended by a
| "generalized distributed sum" (GDSum) of label-dependent functions
| ``H_i(l)`` with ``sum_i H_i(l) = 0``.

* type:     identity-based decentralised multi-client encryption (Ring-LWE, ROM)
* setting:  R_q = Z_q[x] / (x^ring_n + 1)

Encryption and decryption are those of the identity-based MCFE. There is no
authority for functional keys: for weights ``y`` and label ``l`` client ``i``
publishes

* its per-slot IPFE key ``sum_row y_{i,row} s_{i,row}`` (part of the key), and
* ``z_i = sum_row y_{i,row} (u_{i,row} + H(u'_i, l)_row) + M_i(l, y)``,

where the GDSum masks ``M_i(l, y) = sum_{j > i} PRF(K_ij, l, y) - sum_{j < i}
PRF(K_ji, l, y)`` cancel over all clients, so ``sum_i z_i`` is the pad term
``z''`` of the key and each ``z_i`` alone is uniformly masked. Every client
must publish a share; shares for other weights or labels do not combine.

Deviation: the PKC version of BMMS21 states GDSum only through its property;
this module realises it with pairwise PRF masks (SHAKE-256 keyed by
pairwise secrets ``K_ij``, which this prototype samples in-process, as it
does all other keys; a deployment would agree on them by key exchange),
hashed to ``Z_q`` like the label pads (the ``label_hash`` public parameter).
Versions <= 0.2.0 used an ad-hoc label scalar instead.
"""

from __future__ import annotations

import logging
import secrets

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.labels import chunk_label, encode_label
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe import label_pad, label_pad_material
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai import backend
from pyfe4ai.core.formats.ring_lwe import DecentralizedRingLWEEncryptionKey
from pyfe4ai.core.formats.ring_lwe import DecentralizedRingLWEKeyShare
from pyfe4ai.core.formats.ring_lwe import DecentralizedRingLWEParameters
from pyfe4ai.core.formats.ring_lwe import MCFERingLWEFunctionalKey
from pyfe4ai.core.versions import check_key_label_hash
from pyfe4ai.primitives.hashing import hash_to_zq_material, hash_to_zq_vector
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.ring_lwe_utils import check_key_rows, is_values, mat_vec_mul, transpose
from pyfe4ai._validation import int_vector

logger = logging.getLogger(__name__)

_GDSUM_DOMAIN = "pyfe4ai/dmcfe/ring_lwe/GDSum"


def gdsum_material(pair_key: str, context: str, mode: str) -> bytes:
    """The hash input of one PRF term of :func:`gdsum_mask` (*mode*: the
    ``label_hash`` public parameter)."""
    return hash_to_zq_material("{}|{}".format(pair_key, context), domain=_GDSUM_DOMAIN, mode=mode)


def gdsum_mask(
    nid: str, pair_keys: dict, lst_nid: list, context: str, q, ring_n: int, mode: str
) -> list:
    """Return ``M_i(context) = sum_{j>i} PRF(K_ij) - sum_{j<i} PRF(K_ji)`` in Z_q^ring_n
    (the PRF: :func:`~pyfe4ai.primitives.hashing.hash_to_zq_vector` with the
    ``label_hash`` *mode*)."""
    mask = [gp.mpz(0)] * ring_n
    pos = lst_nid.index(nid)
    for other in lst_nid:
        if other == nid:
            continue
        prf = hash_to_zq_vector(
            "{}|{}".format(pair_keys[other], context), q, ring_n, domain=_GDSUM_DOMAIN, mode=mode
        )
        sign = 1 if lst_nid.index(other) > pos else -1
        mask = [(m + sign * v) % q for m, v in zip(mask, prf)]
    return mask


class DecentralizedMCFERingLWEKeyGenerator(MCFERingLWEKeyGenerator):
    """Sets up identity-based MCFE client keys and pairwise GDSum secrets."""

    _scheme_type = CryptoCONST.TYPE_DMCFE_RING_LWE

    def setup(self) -> None:
        """MCFE setup plus a secret ``K_ij`` shared by every pair of clients."""
        super().setup()
        if len(self.lst_nid) < 2:
            raise FEValidationError("decentralised key derivation needs at least two clients")
        pairs = {nid: {} for nid in self.lst_nid}
        for a in range(len(self.lst_nid)):
            for b in range(a + 1, len(self.lst_nid)):
                k = secrets.token_hex(32)
                pairs[self.lst_nid[a]][self.lst_nid[b]] = k
                pairs[self.lst_nid[b]][self.lst_nid[a]] = k
        self.msk["pair_keys"] = pairs

    def get_public_parameters(self) -> dict:
        """MCFE public parameters plus ``lst_nid``."""
        pp = DecentralizedRingLWEParameters(super().get_public_parameters())
        pp.lst_nid = list(self.lst_nid)
        return pp.to_dict()

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Client *nid*'s MCFE keys, its IPFE secret ``s`` and pairwise ``pair_keys``."""
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
        key = DecentralizedRingLWEEncryptionKey(sk)
        key.s = [[gp.digits(v) for v in row] for row in self.msk["sk"][nid]]
        key.pair_keys = dict(self.msk["pair_keys"][nid])
        return key.to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not available: functional keys are derived by the clients."""
        raise NotImplementedError(
            "decentralised MCFE has no central functional key; combine the "
            "clients' derive_function_decryption_key_share() outputs instead"
        )


class DecentralizedMCFERingLWE(MCFERingLWE):
    """Identity-based MCFE encryption / decryption with GDSum key derivation."""

    def derive_function_decryption_key_share(
        self, fusion_weight: dict, label: str, n_chunks: int = 0
    ) -> dict:
        """Publish this client's key share for weights ``y`` and label ``l``.

        Args:
            fusion_weight: Weights of every client (``eta`` integers each).
            label: The label the key will be bound to.
            n_chunks: Number of ndarray chunks the key must cover.

        Returns:
            Dict with ``sk_y`` (this slot's IPFE key), ``z`` and ``z_chunks``
            (masked pad terms), ``label``, ``nid`` and ``weights_id``.

        Raises:
            FEKeyError: Without the client's keys, or if ``u`` or ``s`` has
                fewer than ``eta`` rows of ``ring_n`` values.
        """
        self._check_private_key()
        if not self._has_private_keys() or "pair_keys" not in DecentralizedRingLWEEncryptionKey(self.sk):
            raise FEKeyError("need the client's keys to derive a key share")
        lst_nid = self.pp["lst_nid"]
        if set(fusion_weight) != set(lst_nid):
            raise FEValidationError("fusion weights must cover every client")
        if not isinstance(label, str):
            raise FEValidationError("identity-based MCFE keys need a string label")
        weights = int_vector(
            fusion_weight[self.id], "fusion weight of {}".format(self.id),
            length=self.pp["eta"], bound=int(self.pp["bound_y"]),
        )
        q = gp.mpz(self.pp["q"])
        ring_n = self.pp["ring_n"]
        y = [gp.mpz(v) for v in weights]
        weights_id = fusion_weight_id(fusion_weight)
        sk = DecentralizedRingLWEEncryptionKey(self.sk)
        # eta rows of ring_n values each (fewer: the Python path zipped them away into a wrong key share)
        check_key_rows({"u": sk.get("u"), "s": sk.get("s")}, self.pp["eta"], ring_n, "u or s")
        s = [[gp.mpz(v) for v in row] for row in sk.s]

        labels = [label] + [chunk_label(label, c) for c in range(n_chunks)]
        ring = backend.ring(q, ring_n)
        if ring.native:  # the native step sequence (its checks come in another order)
            terms, sk_y = self._native_key_share(ring, y, s, labels, weights_id)
            return DecentralizedRingLWEKeyShare.new(
                sk_y=sk_y, z=terms[0], z_chunks=terms[1:],
                label=label, nid=self.id, weights_id=weights_id,
                label_hash=self.label_hash,
            ).to_dict()

        def masked_pad_term(lab) -> list:
            z = [gp.mpz(0)] * ring_n
            pad = self._pad(lab)
            for y_row, u_row, h_row in zip(y, sk.u, pad):
                z = [acc + y_row * (gp.mpz(u) + h) for acc, u, h in zip(z, u_row, h_row)]
            context = "{}|{}".format(encode_label(lab), weights_id)
            mask = gdsum_mask(self.id, sk.pair_keys, lst_nid, context, q, ring_n, self.label_hash)
            return [gp.digits((v + m) % q) for v, m in zip(z, mask)]

        share = DecentralizedRingLWEKeyShare.new(
            sk_y=[gp.digits(v) for v in mat_vec_mul(transpose(s), y, q)],
            z=masked_pad_term(label),
            z_chunks=[masked_pad_term(chunk_label(label, c)) for c in range(n_chunks)],
            label=label,
            nid=self.id,
            weights_id=weights_id,
        )
        share.label_hash = self.label_hash
        return share.to_dict()

    def _native_key_share(self, ring, y: list, s: list, labels: list, weights_id: str) -> tuple:
        """The masked pad terms of *labels* and ``sk_y`` of
        :meth:`derive_function_decryption_key_share` on the native *ring* (the
        same values and strings): ``sum_r y_r (u_r + H_r(l)) + M_i(l)`` with the
        label pad and the GDSum PRF terms expanded by its
        :meth:`~pyfe4ai.backend.interface.Ring.label_terms` with the
        ``label_hash`` of the public parameters (which the encryption key must
        have been issued for, as in :meth:`_pad`)."""
        mode = self._encryption_label_hash()
        lst_nid = self.pp["lst_nid"]
        eta, ring_n = self.pp["eta"], self.pp["ring_n"]
        pos = lst_nid.index(self.id)
        others = [o for o in lst_nid if o != self.id]
        signs = [1 if lst_nid.index(o) > pos else -1 for o in others]
        sk = DecentralizedRingLWEEncryptionKey(self.sk)
        base = ring.lincomb(sk.u, y)  # sum_r y_r u_r
        pad_mats = [label_pad_material(sk.u_prime, lab, mode) for lab in labels]
        mask_mats = [
            [gdsum_material(sk.pair_keys[o], "{}|{}".format(encode_label(lab), weights_id), mode)
             for o in others]
            for lab in labels
        ]
        if eta == 1:  # one call: the pad (weight y) and the masks (weights +-1) as slots
            terms = ring.label_terms(base, [[p] + m for p, m in zip(pad_mats, mask_mats)],
                                     [y] + [[sg] for sg in signs], 1, mode)
        else:
            pads = ring.label_terms(base, [[p] for p in pad_mats], [y], eta, mode)
            masks = ring.label_terms([0] * ring_n, mask_mats, [[sg] for sg in signs], 1, mode)
            terms = [ring.lincomb([a, b], [1, 1]) for a, b in zip(pads, masks)]
        sk_y = ring.lincomb(s, y)
        return terms, sk_y

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Sum every client's share into the identity-based MCFE key.

        Raises:
            FESchemeError: If a client's share is missing or the shares were
                derived for different weights, labels or chunk counts.
            FEKeyError: If a share's ``sk_y``, ``z`` or a ``z_chunks`` entry
                is not ``ring_n`` values.
        """
        lst_nid = self.pp["lst_nid"]
        if set(dct_dk_shares) != set(lst_nid):
            raise FESchemeError("key shares of all clients are required")
        shares = [DecentralizedRingLWEKeyShare(sh) for sh in dct_dk_shares.values()]
        first = shares[0]
        if any(
            (sh.weights_id, sh.label, len(sh.z_chunks))
            != (first.weights_id, first.label, len(first.z_chunks))
            for sh in shares
        ):
            raise FESchemeError("key shares were derived for different weights or labels")
        for sh in shares:
            check_key_label_hash(sh.to_dict(), self.label_hash, "key shares were")
        ring_n = self.pp["ring_n"]
        for sh in shares:  # ring_n values each (the Python sum would stop at the shortest)
            if any(not is_values(poly) or len(poly) != ring_n
                   for poly in [sh.get("sk_y"), sh.get("z")] + list(sh.z_chunks)):
                raise FEKeyError("malformed key share: sk_y, z and every z_chunks entry are ring_n values")
        q = gp.mpz(self.pp["q"])
        ring = backend.ring(q, ring_n)

        def total(get) -> list:
            # sum mod q, natively where the extension applies, the same strings (the
            # shares read one after the other, as the values are summed)
            return ring.lincomb((get(sh) for sh in shares), [1] * len(shares))

        key = MCFERingLWEFunctionalKey.new(
            sk_y={nid: DecentralizedRingLWEKeyShare(sh).sk_y for nid, sh in dct_dk_shares.items()},
            z=total(lambda sh: sh.z),
            z_chunks=[total(lambda sh, c=c: sh.z_chunks[c]) for c in range(len(first.z_chunks))],
            label=first.label,
        )
        key.label_hash = self.label_hash
        return key.to_dict()
