"""
Simple Single Input Functional Encryption over ristretto255
| From "Simple Functional Encryption Schemes for Inner Products"
| Published in: PKC 2015
| By Michel Abdalla, Florian Bourse, Angelo De Caro, and David Pointcheval
| URL: https://eprint.iacr.org/2015/017.pdf

* type:     public-key encryption
* setting:  the prime-order group ristretto255 (RFC 9496), written additively

Construction 1 of ABDP15 (DDH), as :mod:`pyfe4ai.schemes.sife.ddh` but in
ristretto255: ``mpk = (G, H_i = s_i G)``; the key for ``y`` is
``sk_y = <s, y> mod L``; ``ct = (r G, r H_i + x_i G)``; decryption computes
``sum_i y_i ct_i - sk_y ct_0 = <x, y> G`` and a bounded discrete log
(``dlog_bound``, default ``10^(precision + 2)``). See
:mod:`pyfe4ai.schemes.ec_ddh_base`.
"""

from __future__ import annotations

import logging

from pyfe4ai.schemes.ec_ddh_base import ECDDHCryptoBase, ECDDHKeyGeneratorBase
from pyfe4ai.utils import ec_group as eg
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector

logger = logging.getLogger(__name__)


class SIFERistretto255KeyGenerator(ECDDHKeyGeneratorBase):
    """Key generator for DDH-based single-input inner-product FE over ristretto255."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Args:
            config: ``eta`` (plaintext vector length); ``group`` (optional,
                ``"ristretto255"``).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        if isinstance(self.eta, bool) or not isinstance(self.eta, int) or self.eta < 1:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))

    def setup(self) -> None:
        s = [eg.random_scalar() for _ in range(self.eta)]
        self.msk = {"s": s}
        self.mpk = {"h": [eg.mul_base(s_i) for s_i in s]}
        logger.info("SIFE (ristretto255) setup successfully")

    def get_public_parameters(self) -> dict:
        return {"group": eg.GROUP, "eta": self.eta}

    def get_private_keys(self, nid: str = "nid_default") -> dict | None:
        """Encryption key ``{h}``: the elements ``H_i`` as hex encodings."""
        return {"h": [eg.to_hex(h) for h in self.mpk["h"]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """``{z: <s, y> mod L, y}`` for ``credentials["fusion_weight"]``."""
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE decryption key generation")
        y = int_vector(credentials.get("fusion_weight"), "fusion weight", length=self.eta)
        return {"z": eg.scalar_str(sum(s_i * y_i for s_i, y_i in zip(self.msk["s"], y))), "y": y}


class SIFERistretto255(ECDDHCryptoBase):
    """Crypto operations for DDH-based single-input inner-product FE over ristretto255."""

    def encrypt(self, lst_pt: list) -> dict:
        """``{ct0: r G, ct1: [r H_i + x_i G]}`` for ``eta`` integers *lst_pt*."""
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        x = int_vector(lst_pt, "plaintext", length=len(self.sk["h"]))
        h = [eg.from_hex(v, "encryption key") for v in self.sk["h"]]
        r = eg.random_scalar()
        return {
            "ct0": eg.to_hex(eg.mul_base(r)),
            "ct1": [eg.to_hex(eg.add(eg.mul(h_i, r), eg.mul_base(x_i))) for h_i, x_i in zip(h, x)],
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """``<x, y>``, or ``None`` if ``|<x, y>|`` exceeds ``dlog_bound``.

        Raises:
            FEKeyError: If the decryption key is missing.
            FESchemeError: If *fusion_weight* are not the key's weights.
            FEValidationError: If the ciphertext is malformed.
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        y = int_vector(fusion_weight, "fusion weight")
        if y != dk.get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")
        try:
            ct1 = [eg.from_hex(c, "ciphertext") for c in dct_ct["ct1"]]
            ct0 = eg.from_hex(dct_ct["ct0"], "ciphertext")
        except (KeyError, TypeError) as exc:
            raise FEValidationError("malformed ciphertext") from exc
        if len(ct1) < len(y):
            raise FEValidationError("ciphertext is shorter than the decryption key")
        acc = eg.neg(eg.mul(ct0, eg.scalar_from(dk["z"], "decryption key")))
        for c_i, y_i in zip(ct1, y):
            acc = eg.add(acc, eg.mul(c_i, y_i))
        return self._solve_dlog(acc)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError()

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        raise NotImplementedError()

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        raise NotImplementedError()
