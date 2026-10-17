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
from pyfe4ai.schemes.mcfe.ring_lwe import label_pad
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import DEFAULT_LABEL_HASH
from pyfe4ai.utils.crypto_utils import hash_to_zq_vector
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.ring_lwe_utils import mat_vec_mul, transpose
from pyfe4ai.utils.validation import int_vector

logger = logging.getLogger(__name__)

_GDSUM_DOMAIN = "pyfe4ai/dmcfe/ring_lwe/GDSum"


def gdsum_mask(
    nid: str, pair_keys: dict, lst_nid: list, context: str, q, ring_n: int, mode: str = DEFAULT_LABEL_HASH
) -> list:
    """Return ``M_i(context) = sum_{j>i} PRF(K_ij) - sum_{j<i} PRF(K_ji)`` in Z_q^ring_n
    (the PRF: :func:`~pyfe4ai.utils.crypto_utils.hash_to_zq_vector` with the
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
        pp = super().get_public_parameters()
        pp["lst_nid"] = list(self.lst_nid)
        return pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Client *nid*'s MCFE keys, its IPFE secret ``s`` and pairwise ``pair_keys``."""
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
        sk["s"] = [[gp.digits(v) for v in row] for row in self.msk["sk"][nid]]
        sk["pair_keys"] = dict(self.msk["pair_keys"][nid])
        return sk

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
        """
        if not self._has_private_keys() or "pair_keys" not in self.sk:
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
        s = [[gp.mpz(v) for v in row] for row in self.sk["s"]]

        def masked_pad_term(lab) -> list:
            z = [gp.mpz(0)] * ring_n
            pad = self._pad(lab)
            for y_row, u_row, h_row in zip(y, self.sk["u"], pad):
                z = [acc + y_row * (gp.mpz(u) + h) for acc, u, h in zip(z, u_row, h_row)]
            context = "{}|{}".format(encode_label(lab), weights_id)
            mask = gdsum_mask(self.id, self.sk["pair_keys"], lst_nid, context, q, ring_n, self.label_hash)
            return [gp.digits((v + m) % q) for v, m in zip(z, mask)]

        share = {
            "sk_y": [gp.digits(v) for v in mat_vec_mul(transpose(s), y, q)],
            "z": masked_pad_term(label),
            "z_chunks": [masked_pad_term(chunk_label(label, c)) for c in range(n_chunks)],
            "label": label,
            "nid": self.id,
            "weights_id": weights_id,
        }
        if self.label_hash != DEFAULT_LABEL_HASH:
            share["label_hash"] = self.label_hash
        return share

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Sum every client's share into the identity-based MCFE key.

        Raises:
            FESchemeError: If a client's share is missing or the shares were
                derived for different weights, labels or chunk counts.
        """
        lst_nid = self.pp["lst_nid"]
        if set(dct_dk_shares) != set(lst_nid):
            raise FESchemeError("key shares of all clients are required")
        shares = list(dct_dk_shares.values())
        first = shares[0]
        if any(
            (sh["weights_id"], sh["label"], len(sh["z_chunks"]))
            != (first["weights_id"], first["label"], len(first["z_chunks"]))
            for sh in shares
        ):
            raise FESchemeError("key shares were derived for different weights or labels")
        if any(sh.get("label_hash", DEFAULT_LABEL_HASH) != self.label_hash for sh in shares):
            raise FESchemeError("key shares were derived with another label hash than the public parameters")
        q = gp.mpz(self.pp["q"])

        def total(get) -> list:
            acc = [gp.mpz(0)] * self.pp["ring_n"]
            for sh in shares:
                acc = [(a + gp.mpz(v)) % q for a, v in zip(acc, get(sh))]
            return [gp.digits(v) for v in acc]

        key = {
            "sk_y": {nid: sh["sk_y"] for nid, sh in dct_dk_shares.items()},
            "z": total(lambda sh: sh["z"]),
            "z_chunks": [total(lambda sh, c=c: sh["z_chunks"][c]) for c in range(len(first["z_chunks"]))],
            "label": first["label"],
        }
        if self.label_hash != DEFAULT_LABEL_HASH:
            key["label_hash"] = self.label_hash
        return key
