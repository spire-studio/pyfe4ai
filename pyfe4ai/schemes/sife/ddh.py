"""
Simple Single Input Functional Encryption
| From "Simple Functional Encryption Schemes for Inner Products"
| Published in: PKC 2015
| By Michel Abdalla, Florian Bourse, Angelo De Caro, and David Pointcheval
| URL: https://eprint.iacr.org/2015/017.pdf

* type:     public-key encryption
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

Construction 1 of ABDP15 (DDH): ``mpk = (g, h_i = g^{s_i})``; the key for
``y`` is ``sk_y = <s, y> mod q``; ``ct = (g^r, h_i^r g^{x_i})``; decryption
computes ``prod ct_i^{y_i} / ct_0^{sk_y} = g^{<x, y>}`` and a bounded discrete
log (``dlog_bound``, default ``10^(precision + 2)``). Every ciphertext
component is reduced modulo ``p`` -- versions <= 0.2.0 published the integer
``(h_i^r mod p)(g^{x_i} mod p)``, from which ``x_i`` follows by trial
division -- and keys are reduced modulo ``q``.
"""

from __future__ import annotations

import os
import logging

import gmpy2 as gp

from pyfe4ai.schemes.common.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.groups.integer_groups import _random
from pyfe4ai.primitives.groups.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai._validation import int_vector

logger = logging.getLogger(__name__)


class SIFEKeyGenerator(DDHKeyGeneratorBase):
    """Key generator for DDH-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE
    _verify_keys = ("sec_param", "eta")

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta`` (plaintext vector length).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)

        self._load_parameters()

    def setup(self) -> None:
        """Generate master secret key (``msk``) and master public key (``mpk``)."""
        s = [_random(self.p, self.sec_param) for _ in range(self.eta)]
        g_s = [gp.powmod(self.g, s[i], self.p) for i in range(self.eta)]
        self.msk = {"s": s}
        self.mpk = {"p": self.p, "g": self.g, "gs": g_s}
        logger.info("SIFE setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters (g, p, eta, sec_param).

        Returns:
            Dict of public parameters.
        """
        logger.debug("generating public parameters ...")
        return {
            "g": gp.digits(self.mpk["g"]),
            "p": gp.digits(self.mpk["p"]),
            "eta": self.eta,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default") -> dict | None:
        """Return encryption keys (group element list ``gs``).

        Args:
            nid: Client identifier (unused for single-input schemes).

        Returns:
            Dict with ``gs`` list.
        """
        logger.debug("generating private key(s)...")
        _keys = {
            "gs": [gp.digits(i) for i in self.mpk["gs"]],
        }
        return _keys

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of int).

        Returns:
            Dict with ``z`` (``<s, y> mod q``) and the weights ``y``.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE decryption key generation")
        y = int_vector(credentials.get("fusion_weight"), "fusion weight", length=self.eta)
        return {"z": gp.digits(self._key(y)), "y": y}

    def _key(self, y: list) -> gp.mpz:
        """``<s, y> mod q`` for a weight prefix ``y`` (``len(y) <= eta``)."""
        return sum(gp.mul(s_i, y_i) for s_i, y_i in zip(self.msk["s"], y)) % self.q


class SIFE(IPFEAbsCrypto):
    """Crypto operations for DDH-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and either ``sk`` (encryption keys) or ``dk`` (decryption key);
                ``dlog_bound`` (default ``10^(precision + 2)``) bounds the
                inner products decryption recovers.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.dlog_bound = int(config.get("dlog_bound", pow(10, self.precision + 2)))
        self.dlog_table = None

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_SIFE,
            f"dlog_{self.dlog_bound}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(_dlog_file, self.pp["g"], self.pp["p"], self.dlog_bound)
        )

    def _plaintext(self, lst_pt: list) -> list:
        return int_vector(lst_pt, "plaintext", length=len(self.sk["gs"]))

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector under the DDH-based scheme.

        Args:
            lst_pt: Integer plaintext vector of length ``eta``.

        Returns:
            Dict with ``ct0`` (group element) and ``ct1`` (list of group elements).

        Raises:
            FEKeyError: If public parameters or private keys are missing.
            FEValidationError: If the plaintext is not ``eta`` integers.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        x = self._plaintext(lst_pt)

        sec_param = self.pp["sec_param"]
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        gs = [gp.mpz(i) for i in self.sk["gs"]]

        r = _random(p, sec_param)
        ct0 = gp.digits(gp.powmod(g, r, p))
        ct1 = [
            gp.digits(gp.mul(gp.powmod(gs[i], r, p), gp.powmod(g, x_i, p)) % p)
            for i, x_i in enumerate(x)
        ]
        return {"ct0": ct0, "ct1": ct1}

    def _check_weights(self, dk: dict, fusion_weight: list) -> list:
        y = int_vector(fusion_weight, "fusion weight")
        if y != dk.get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")
        return y

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt a ciphertext to recover the inner product ⟨x, y⟩.

        Args:
            dct_ct: Ciphertext dict from :meth:`encrypt`.
            dk: Decryption key dict from :meth:`SIFEKeyGenerator.get_decryption_keys`.
            fusion_weight: Integer weight vector y (the key's weights).

        Returns:
            The inner product as an integer, or ``None`` if ``|<x, y>|``
            exceeds ``dlog_bound``.

        Raises:
            FEKeyError: If public parameters or decryption key is missing.
            FESchemeError: If *fusion_weight* are not the key's weights.
            FEValidationError: If the ciphertext is malformed.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not dk:
            raise FEKeyError("no decryption key provided.")
        y = self._check_weights(dk, fusion_weight)
        p = gp.mpz(self.pp["p"])
        ct1 = [gp.mpz(c) for c in dct_ct["ct1"]]
        ct0 = gp.mpz(dct_ct["ct0"])
        if len(ct1) < len(y):
            raise FEValidationError("ciphertext is shorter than the decryption key")
        if not all(0 < c < p for c in ct1 + [ct0]):
            raise FEValidationError("ciphertext element out of range")

        _prod = gp.mpz(1)
        for c_i, y_i in zip(ct1, y):
            _prod = gp.mul(_prod, gp.powmod(c_i, y_i, p)) % p
        gf = gp.divm(_prod, gp.powmod(ct0, gp.mpz(dk["z"]), p), p)
        return self._solve_dlog(gf)

    def _solve_dlog(self, g_inner_prod) -> int | None:
        """Solve the discrete log using a cached dlog table."""
        if self.dlog_table is None:
            self._load_dlog_table()
        try:
            return dlog_table_solve(
                g_inner_prod, self.pp["g"], self.pp["p"],
                self.bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system.")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError()

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError()

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError()
