"""
Decentralized function-hiding multi-client inner-product FE over pairings.

| Nguyen, Pointcheval, Schaedlich, "Decentralized Multi-Client Functional
| Encryption with Strong Security", IACR CiC 1(2), 2024 -- Figure 1, the
| selectively function-hiding DMCFE (an extension of the AGT21b FH-MCFE of
| :mod:`pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing`); selective security
| under SXDH in the random-oracle model, static corruptions.

* type:     decentralised multi-client encryption with function-hiding keys
* setting:  asymmetric pairing group (default MNT224), random oracle

Setup samples one FH-IPFE master key ``imsk_i`` per client
(:mod:`pyfe4ai.schemes.mcfe.fh_ipe`) and a sharing ``t~_1 + ... + t~_n = 0``:

* Enc:      ``[omega]_1 = H1(tag)``; ``ct_i = iEnc(imsk_i, [x_hat_i]_1)`` with
  ``x_hat_i = (x_i, omega t~_i, 0, 0^eta, 0, 0)``.
* DKeyGen:  ``[mu]_2 = H2(tag-f)``; ``dk_i = iKeyGen(imsk_i, [y_hat_i]_2)`` with
  ``y_hat_i = (y_i, mu, 0, 0^eta, 0, 0)``.
* Dec:      ``prod_i iDec(dk_i, ct_i) = [<x, y> + omega mu sum t~_i]_T = [<x, y>]_T``.

There is no authority for functional keys: each client derives its own
``dk_i`` from ``tag-f`` and its own weights ``y_i`` only, and keys of all
clients for the same ``tag-f`` are needed; together they decrypt
``sum_i <x_i, y_i>`` for the weights each client chose. The proof allows one
key per client and ``tag-f``: name every new weight vector with a fresh
``tag_f``. A client object refuses a second weight vector under a ``tag_f`` it
already used (other instances cannot know about it), and derives the key
randomness from a client secret, ``tag-f`` and ``y_i``, so asking again
returns the same key. Each client must encrypt at most once per tag. Earlier
versions used ``tag-f = (tag_f, H(y))`` over all clients' weights, so a
corrupted client could test guesses of the other clients' weights; versions
<= 0.2.0 masked inputs with a public label scalar instead.
"""

from __future__ import annotations

import json
import logging
import secrets

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.pairing_backend import ZR
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.validation import int_vector

logger = logging.getLogger(__name__)

_KEY_RNG_DOMAIN = b"pyfe4ai/fh-dmcfe/DKeyGen|"


class DecentralizedMCFEFHMultiIPEKeyGenerator(MCFEFHMultiIPEKeyGenerator):
    """Setup of the NPS24 selectively function-hiding DMCFE."""

    _scheme_type = CryptoCONST.TYPE_DMCFE_FH_MULTI_IPE

    def setup(self) -> None:
        """FH-MCFE setup plus the zero sharing ``t~`` and per-client key seeds."""
        super().setup()
        if len(self.lst_nid) < 2:
            raise FEValidationError("decentralised key derivation needs at least two clients")
        t = [int(self.ctx.group.random(ZR)) for _ in self.lst_nid[:-1]]
        t.append((-sum(t)) % self.ctx.order)
        self.t_tilde = dict(zip(self.lst_nid, t))
        self.key_seeds = {nid: secrets.token_hex(32) for nid in self.lst_nid}

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Client *nid*'s ``imsk``, its share ``t_tilde`` and its key-derivation ``seed``."""
        if nid not in self.msk:
            return None
        sk = {"imsk": self.ctx.serialise_msk(self.msk[nid])}  # DKeyGen needs a_out
        sk["t_tilde"] = str(self.t_tilde[nid])
        sk["seed"] = self.key_seeds[nid]
        return sk

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not available: functional keys are derived by the clients."""
        raise NotImplementedError(
            "decentralised MCFE has no central functional key; combine the "
            "clients' derive_function_decryption_key_share() outputs instead"
        )


class DecentralizedMCFEFHMultiIPE(MCFEFHMultiIPE):
    """Encryption, key derivation and decryption of the NPS24 FH-DMCFE."""

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._tags_used = {}

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """``iEnc(imsk_i, [(x_i, H1(label)^t~_i, 0, ...)]_1)``; once per client and label."""
        x = self._check_plaintext(lst_pt)
        if "t_tilde" not in self.sk:
            raise FEKeyError("need the client's DMCFE keys to encrypt")
        omega_t = self.ctx.h1(label) ** self.ctx.zr(int(self.sk["t_tilde"]))
        x_hat = self.ctx.x_hat(x, omega_t)
        return {"ct": [self.ctx.ser(v) for v in self.ctx.fhipe.encrypt(self.imsk, x_hat)]}

    def derive_function_decryption_key_share(self, fusion_weight, tag_f: str) -> dict:
        """Derive this client's key ``dk_i`` for its weights ``y_i`` (NPS24 DKeyGen).

        Args:
            fusion_weight: This client's weights (``eta`` integers), or a dict
                of weights in which only this client's entry is used.
            tag_f: Non-empty name of the function; all clients use the same
                one for one functional key, and a fresh one for every new
                weight vector (one key per client and ``tag_f``).

        Returns:
            Dict with ``k`` and ``z`` (the FH-IPFE key), ``nid`` and ``tag_f``
            (nothing derived from ``y``: keys are function hiding).

        Raises:
            FESchemeError: If this object already derived a key for other
                weights under *tag_f*.
        """
        if not self._has_private_keys() or "seed" not in self.sk:
            raise FEKeyError("need the client's keys to derive a key share")
        if not isinstance(tag_f, str) or not tag_f:
            raise FEValidationError("tag_f must be a non-empty string")
        weights = fusion_weight.get(self.id) if isinstance(fusion_weight, dict) else fusion_weight
        weights = int_vector(
            weights, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"])
        )
        used = self._tags_used.setdefault(tag_f, weights)
        if used != weights:
            raise FESchemeError(
                "a key for other weights was already derived under this tag_f; use a fresh tag_f"
            )
        # [mu]_2 = H2(tag-f) and a key randomness seeded with (tag-f, y_i):
        # nothing about the other clients' weights enters this client's key
        context = json.dumps([tag_f, weights])
        y_hat = self.ctx.y_hat(weights, self.ctx.h2(tag_f))
        rng = XofRandom(bytes.fromhex(self.sk["seed"]) + _KEY_RNG_DOMAIN + context.encode("utf-8"))
        key, z_sum = self.ctx.fhipe.keygen(self.imsk, y_hat, rng=rng)
        share = self.ctx.key_entry(key, z_sum)
        share.update({"nid": self.id, "tag_f": tag_f})
        return share

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Collect every client's ``dk_i`` into the functional key.

        Raises:
            FESchemeError: If a client's key is missing or the keys were derived
                for different function tags. (Keys for different weights under
                one ``tag_f`` combine into a key that decrypts to garbage.)
        """
        if set(dct_dk_shares) != set(self.pp["lst_nid"]):
            raise FESchemeError("key shares of all clients are required")
        shares = list(dct_dk_shares.values())
        first = shares[0]
        if any(sh["tag_f"] != first["tag_f"] for sh in shares):
            raise FESchemeError("key shares were derived for different function tags")
        if any(sh.get("nid", nid) != nid for nid, sh in dct_dk_shares.items()):
            raise FESchemeError("key share filed under the wrong client id")
        return {
            "dk": {nid: {"k": sh["k"], "z": sh["z"]} for nid, sh in dct_dk_shares.items()},
            "tag_f": first["tag_f"],
        }
