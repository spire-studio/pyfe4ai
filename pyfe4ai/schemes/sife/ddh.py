"""
Simple Single Input Functional Encryption
| From "Simple Functional Encryption Schemes for Inner Products"
| Published in: PKC 2015
| By Michel Abdalla, Florian Bourse, Angelo De Caro, and David Pointcheval
| URL: https://eprint.iacr.org/2015/017.pdf

* type:     public-key encryption
* setting:  Integer based

"""

from __future__ import annotations

import os
import json
import logging

import gmpy2 as gp

from pyfe4ai.schemes.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

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
            Dict with ``z`` (the inner product of secret key and weights).

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE decryption key generation")
        fusion_weights = credentials.get("fusion_weight")
        if not isinstance(fusion_weights, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weights) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")

        z = gp.mpz(0)
        for i in range(self.eta):
            z += gp.mul(self.msk["s"][i], gp.mpz(fusion_weights[i]))

        return {"z": gp.digits(z)}


class SIFE(IPFEAbsCrypto):
    """Crypto operations for DDH-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and either ``sk`` (encryption keys) or ``dk`` (decryption key).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        else:
            self._load_dlog_table()

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_SIFE,
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"],
                pow(10, self.precision + 2),
            )
        )

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector under the DDH-based scheme.

        Args:
            lst_pt: Integer plaintext vector of length ≤ ``eta``.

        Returns:
            Dict with ``ct0`` (group element) and ``ct1`` (list of group elements).

        Raises:
            FEKeyError: If public parameters or private keys are missing.
            FEValidationError: If plaintext length exceeds ``eta``.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) > len(self.sk["gs"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(lst_pt))

        sec_param = self.pp["sec_param"]
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        gs = [gp.mpz(i) for i in self.sk["gs"]]

        r = _random(p, sec_param)
        ct0 = gp.digits(gp.powmod(g, r, p))
        ct1 = [
            gp.digits(
                gp.mul(gp.powmod(gs[i], r, p), gp.powmod(g, gp.mpz(lst_pt[i]), p)),
            )
            for i in range(len(lst_pt))
        ]

        return {"ct0": ct0, "ct1": ct1}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt a ciphertext to recover the inner product ⟨x, y⟩.

        Args:
            dct_ct: Ciphertext dict from :meth:`encrypt`.
            dk: Decryption key dict from :meth:`SIFEKeyGenerator.get_decryption_keys`.
            fusion_weight: Integer weight vector y.

        Returns:
            The inner product as an integer, or ``None`` if out of bound.

        Raises:
            FEKeyError: If public parameters or decryption key is missing.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not dk:
            raise FEKeyError("no decryption key provided.")

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        z = gp.mpz(dk["z"])

        _prod = gp.mpz(1)
        for i in range(len(fusion_weight)):
            _prod = gp.mul(
                _prod, gp.powmod(gp.mpz(dct_ct["ct1"][i]), gp.mpz(fusion_weight[i]), p)
            )
        # _prod = gp.t_mod(_prod, p)
        gf = gp.divm(_prod, gp.powmod(gp.mpz(dct_ct["ct0"]), z, p), p)

        return self._solve_dlog(gf)

    def _solve_dlog(self, g_inner_prod) -> int | None:
        """Solve the discrete log using a cached dlog table."""
        try:
            return dlog_table_solve(
                g_inner_prod, self.pp["g"], self.pp["p"],
                self.bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system.")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
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
