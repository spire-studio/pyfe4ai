"""
LWE-Based Single-Input Inner-Product Functional Encryption
| From "Simple Functional Encryption Schemes for Inner Products"
| By Michel Abdalla, Florian Bourse, Angelo De Caro, David Pointcheval
| Published in: PKC 2015
|
| This module implements the generic simple FE construction instantiated
| under the LWE assumption as a research-oriented Python prototype.

* type:     public-key encryption
* setting:  Integer based
* note:     prototype implementation aligned with the existing crypto-ipfe API

"""

from __future__ import annotations

import json
import logging
import os
import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import center_lwe_vector
from pyfe4ai.utils.lwe_utils import decode_lwe_inner_product
from pyfe4ai.utils.lwe_utils import derive_lwe_parameters
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matrix_to_digits
from pyfe4ai.utils.matrix_utils import matmul_mod
from pyfe4ai.utils.matrix_utils import matvec_mod
from pyfe4ai.utils.matrix_utils import transpose
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.matrix_utils import vector_to_digits
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_bit_vector
from pyfe4ai.utils.sampling_utils import rand_uniform_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)

class SIFELWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for LWE-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta``, ``lwe_n``, ``bound_x``, ``bound_y``.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.lwe_n = config.get("lwe_n", 32)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        """Assign LWE parameters (p, q, m, sigma_q, l_sigma, A) from *param*."""
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma = gp.mpz(param["l_sigma"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration."""
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("lwe_n") == self.lwe_n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
        )

    def _generate_and_save(self, param_file: str) -> None:
        """Derive LWE parameters, sample public matrix **A**, and persist."""
        self.p, self.q, self.m, self.sigma_q, self.l_sigma = derive_lwe_parameters(
            self.bound_x, self.bound_y, self.eta, self.lwe_n
        )

        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "lwe_n": self.lwe_n,
                    "m": self.m,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma_q": self.sigma_q,
                    "l_sigma": gp.digits(self.l_sigma),
                    "A": matrix_to_digits(self.A),
                },
                f,
            )

    def setup(self) -> None:
        """Generate master secret key (``msk``) and master public key (``mpk``)."""
        sk = rand_uniform_matrix(self.lwe_n, self.eta, self.q)
        noise = discrete_gaussian_matrix(self.m, self.eta, self.sigma_q)
        pk = matmul_mod(self.A, sk, self.q)
        for i in range(self.m):
            for j in range(self.eta):
                pk[i][j] = (pk[i][j] + noise[i][j]) % self.q
        self.msk = {"sk": sk}
        self.mpk = {"pk": pk}
        logger.info("SIFE LWE setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters.

        Returns:
            Dict containing ``A``, ``p``, ``q``, ``m``, ``eta``,
            ``lwe_n``, ``bound_x``, ``bound_y``, ``sec_param``.
        """
        return {
            "A": matrix_to_digits(self.A),
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "m": self.m,
            "eta": self.eta,
            "lwe_n": self.lwe_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return encryption keys (public-key matrix ``pk``).

        Args:
            nid: Client identifier (unused for single-input schemes).

        Returns:
            Dict with ``pk`` matrix.
        """
        return {"pk": matrix_to_digits(self.mpk["pk"])}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of int).

        Returns:
            Dict with ``sk_y`` (the projected secret key).

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid or exceed ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE LWE decryption key generation")
        fusion_weights = credentials.get("fusion_weight")
        if not isinstance(fusion_weights, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weights) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")
        if any(abs(int(v)) > self.bound_y for v in fusion_weights):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        y = [gp.mpz(v) for v in fusion_weights]
        sk_y = matvec_mod(self.msk["sk"], y, self.q)
        return {"sk_y": vector_to_digits(sk_y)}


class SIFELWE(IPFEAbsCrypto):
    """Crypto operations for LWE-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE LWE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = digits_to_matrix(self.pp["A"])
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector under the LWE-based scheme.

        Args:
            lst_pt: Integer plaintext vector of length ``eta``.

        Returns:
            Dict with ``ct0`` and ``ct1`` (serialised LWE ciphertext vectors).

        Raises:
            FEKeyError: If public parameters or encryption keys are missing.
            FEValidationError: If plaintext length or values are invalid.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        pk = digits_to_matrix(self.sk["pk"])
        r = rand_bit_vector(self.pp["m"])

        a_trans = transpose(self.A)
        ct0 = matvec_mod(a_trans, r, q)

        pk_trans = transpose(pk)
        ct1 = matvec_mod(pk_trans, r, q)
        centered = center_lwe_vector(lst_pt, gp.mpz(self.pp["p"]), q)
        ct1 = [(v + c) % q for v, c in zip(ct1, centered)]

        return {"ct0": vector_to_digits(ct0), "ct1": vector_to_digits(ct1)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt a ciphertext to recover the inner product ⟨x, y⟩.

        Args:
            dct_ct: Ciphertext dict from :meth:`encrypt`.
            dk: Decryption key dict from :meth:`SIFELWEKeyGenerator.get_decryption_keys`.
            fusion_weight: Integer weight vector y.

        Returns:
            The inner product as an integer.

        Raises:
            FEKeyError: If decryption key is missing.
            FEValidationError: If fusion weight length or values are invalid.
        """
        if not dk:
            raise FEKeyError("no decryption key provided")
        if len(fusion_weight) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        y = [gp.mpz(v) for v in fusion_weight]
        ct0 = digits_to_vector(dct_ct["ct0"])
        ct1 = digits_to_vector(dct_ct["ct1"])
        sk_y = digits_to_vector(dk["sk_y"])

        d = (vecdot_mod(y, ct1, q) - vecdot_mod(ct0, sk_y, q)) % q
        return decode_lwe_inner_product(d, p, q)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarrays element-wise (requires ``eta=1``).

        Args:
            lst_ndarray: List of numpy arrays to encrypt.

        Returns:
            List of object ndarrays containing per-element ciphertexts.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        m = self.pp["m"]
        pk = digits_to_matrix(self.sk["pk"])
        a_trans = transpose(self.A)
        pk_trans = transpose(pk)
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                r = rand_bit_vector(m)
                ct0 = matvec_mod(a_trans, r, q)
                ct1 = matvec_mod(pk_trans, r, q)
                centered = center_lwe_vector([int(w)], p, q)
                ct1 = [(v + c) % q for v, c in zip(ct1, centered)]
                ary_ct[i] = {"ct0": vector_to_digits(ct0), "ct1": vector_to_digits(ct1)}
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Ciphertext arrays to aggregate.
            **kwargs: Must include ``dk`` and ``fusion_weight``.

        Returns:
            Decrypted ndarray list.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise.

        Args:
            dict_ndarray_ct: List of object ndarrays with per-element ciphertexts.
            **kwargs: Must include ``dk`` and ``fusion_weight``.

        Returns:
            List of float ndarrays with decrypted values.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")

        lst_ndarray = []
        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        y = [gp.mpz(v) for v in fusion_weight]
        sk_y = digits_to_vector(dk["sk_y"])
        for ary in dict_ndarray_ct:
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ct = ary[i]
                ct0 = digits_to_vector(ct["ct0"])
                ct1 = digits_to_vector(ct["ct1"])
                d = (vecdot_mod(y, ct1, q) - vecdot_mod(ct0, sk_y, q)) % q
                ary_dec[i] = decode_lwe_inner_product(d, p, q)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
