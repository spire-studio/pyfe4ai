"""
Simple Single Input Functional Encryption, variable-length vectors
| From "Simple Functional Encryption Schemes for Inner Products"
| Published in: PKC 2015
| By Michel Abdalla, Florian Bourse, Angelo De Caro, and David Pointcheval
| URL: https://eprint.iacr.org/2015/017.pdf

* type:     public-key encryption
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

ABDP15 Construction 1 (:mod:`pyfe4ai.schemes.sife.ddh`) for vectors of any
length up to ``eta``: a key for ``y`` of length ``k <= eta`` is the key of
``(y, 0, ..., 0)``, and a plaintext of length ``m <= eta`` is encrypted as a
prefix of ``(x, 0, ..., 0)`` (the omitted components are a function of a full
ciphertext). The key decrypts any ciphertext with at least ``k`` components.
Security reduces to ABDP15 by this zero padding; there is no separate paper.
"""

from __future__ import annotations

import logging

from pyfe4ai.schemes.sife.ddh import SIFE
from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError
from pyfe4ai._validation import int_vector

logger = logging.getLogger(__name__)


class SIFEDynamicKeyGenerator(SIFEKeyGenerator):
    """ABDP15 key generator issuing keys for weight vectors of length <= ``eta``."""

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Key for a weight prefix ``y`` (``len(y) <= eta``).

        Returns:
            Dict with ``z`` (``<s[:k], y> mod q``), ``y`` and ``size = k``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE decryption key generation")
        y = int_vector(credentials.get("fusion_weight"), "fusion weight")
        if not 0 < len(y) <= self.eta:
            raise FEValidationError("fusion weight must have 1..{} entries".format(self.eta))
        return {"z": str(int(self._key(y))), "y": y, "size": len(y)}


class SIFEDynamic(SIFE):
    """ABDP15 encryption of vectors of length <= ``eta``."""

    def _plaintext(self, lst_pt: list) -> list:
        x = int_vector(lst_pt, "plaintext")
        if not 0 < len(x) <= len(self.sk["gs"]):
            raise FEValidationError("plaintext must have 1..{} entries".format(len(self.sk["gs"])))
        return x
