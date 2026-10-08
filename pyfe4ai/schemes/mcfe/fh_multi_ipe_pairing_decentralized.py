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
``dk_i``, and keys of all clients for the same ``tag-f`` are needed. The proof
allows one key per client and ``tag-f``; this module binds ``tag-f`` to the
whole weight vector (``tag-f = (tag_f, H(y))``) and derives the key randomness
from a client secret and ``tag-f``, so asking again returns the same key.
Each client must encrypt at most once per tag. Versions <= 0.2.0 masked
inputs with a public label scalar instead.
"""

from __future__ import annotations

import json
import logging
import secrets

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEKeyGenerator
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.pairing_backend import ZR
from pyfe4ai.utils.sampling_utils import XofRandom

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
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
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

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """``iEnc(imsk_i, [(x_i, H1(label)^t~_i, 0, ...)]_1)``; once per client and label."""
        self._check_plaintext(lst_pt)
        if "t_tilde" not in self.sk:
            raise FEKeyError("need the client's DMCFE keys to encrypt")
        omega_t = self.ctx.h1(label) ** self.ctx.zr(int(self.sk["t_tilde"]))
        x_hat = self.ctx.x_hat([int(v) for v in lst_pt], omega_t)
        return {"ct": [self.ctx.ser(v) for v in self.ctx.fhipe.encrypt(self.imsk, x_hat)]}

    def derive_function_decryption_key_share(self, fusion_weight: dict, tag_f: str = "") -> dict:
        """Derive this client's key ``dk_i`` for the weights ``y`` (NPS24 DKeyGen).

        Args:
            fusion_weight: Weights of every client (``eta`` integers each).
            tag_f: Optional function tag; ``tag-f`` is ``(tag_f, H(y))``.

        Returns:
            Dict with ``k`` and ``z`` (the FH-IPFE key), ``nid``, ``tag_f`` and
            ``weights_id``.
        """
        if not self._has_private_keys() or "seed" not in self.sk:
            raise FEKeyError("need the client's keys to derive a key share")
        lst_nid = self.pp["lst_nid"]
        if not isinstance(fusion_weight, dict) or set(fusion_weight) != set(lst_nid):
            raise FEValidationError("fusion weights must cover every client")
        if not isinstance(tag_f, str):
            raise FEValidationError("tag_f must be a string")
        weights = fusion_weight[self.id]
        if len(weights) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length for client {}".format(self.id))
        if any(abs(int(v)) > int(self.pp["bound_y"]) for v in weights):
            raise FEValidationError("fusion weight exceeds configured bound_y")
        weights_id = fusion_weight_id(fusion_weight)
        context = json.dumps([tag_f, weights_id])
        y_hat = self.ctx.y_hat([int(v) for v in weights], self.ctx.h2(context))
        rng = XofRandom(bytes.fromhex(self.sk["seed"]) + _KEY_RNG_DOMAIN + context.encode("utf-8"))
        key, z_sum = self.ctx.fhipe.keygen(self.imsk, y_hat, rng=rng)
        share = self.ctx.key_entry(key, z_sum)
        share.update({"nid": self.id, "tag_f": tag_f, "weights_id": weights_id})
        return share

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Collect every client's ``dk_i`` into the functional key.

        Raises:
            FESchemeError: If a client's key is missing or the keys were derived
                for different weights or function tags.
        """
        if set(dct_dk_shares) != set(self.pp["lst_nid"]):
            raise FESchemeError("key shares of all clients are required")
        shares = list(dct_dk_shares.values())
        first = shares[0]
        if any((sh["weights_id"], sh["tag_f"]) != (first["weights_id"], first["tag_f"]) for sh in shares):
            raise FESchemeError("key shares were derived for different weights or function tags")
        if any(sh.get("nid", nid) != nid for nid, sh in dct_dk_shares.items()):
            raise FESchemeError("key share filed under the wrong client id")
        return {
            "dk": {nid: {"k": sh["k"], "z": sh["z"]} for nid, sh in dct_dk_shares.items()},
            "weights_id": first["weights_id"],
            "tag_f": first["tag_f"],
        }
