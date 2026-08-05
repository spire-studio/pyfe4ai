"""
Ring-LWE-Based Single-Input Inner-Product Functional Encryption
| Based on
| "Efficient Lattice-Based Inner-Product Functional Encryption"
| By Bermudo Mera, Karmakar, Marc, Soleimanian
| ePrint: 2021/046
|
| This module follows the ring-LWE inner-product FE line and is additionally
| informed by the GoFE reference implementation in `innerprod/simple/ringlwe.go`.

* type:     public-key encryption
* setting:  Integer based
* note:     research prototype with SIMD-style matrix encryption

"""

from __future__ import annotations

import json
import logging
import math
import os

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.ring_lwe_utils import (
    center_matrix,
    decode_vector,
    mat_vec_mul,
    matrix_check_bound,
    next_ntt_prime,
    poly_add,
    poly_mul,
    poly_mul_negacyclic,
    poly_neg,
    transpose,
)
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)

# Backward-compatible aliases (private names used by internal callers)
_poly_add = poly_add
_poly_neg = poly_neg
_poly_mul_negacyclic = poly_mul_negacyclic
_poly_mul = poly_mul
_matrix_check_bound = matrix_check_bound
_transpose = transpose
_mat_vec_mul = mat_vec_mul
_center_matrix = center_matrix
_decode_vector = decode_vector


class SIFERingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Ring-LWE-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.ring_n = config.get("ring_n", None)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.ring_n = int(param["ring_n"])
        self.sigma1 = float(param["sigma1"])
        self.sigma2 = float(param["sigma2"])
        self.sigma3 = float(param["sigma3"])
        self.A = [gp.mpz(v) for v in param["A"]]

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and (self.ring_n is None or param.get("ring_n") == self.ring_n)
        )

    def _generate_and_save(self, param_file: str) -> None:
        l = self.eta
        kappa = float(self.sec_param)
        sigma = 1.0
        sigma1 = math.sqrt(float(4 * l)) * sigma * float(self.bound_x)
        kappa_sqrt = math.sqrt(kappa)
        p = gp.mpz(self.bound_x * self.bound_y * gp.mpz(2 * l))

        if self.ring_n is None:
            for pow_exp in range(5, 11):
                ring_n = 1 << pow_exp
                sigma2 = math.sqrt(float(2 * (l + 2) * ring_n * ring_n)) * sigma
                sigma2 *= sigma1 * kappa_sqrt
                sigma3 = sigma2 * math.sqrt(2.0)
                q_float = sigma1 * sigma2 * kappa * float(2 * ring_n)
                q_float += kappa_sqrt * sigma3
                q_float *= float(self.bound_y) * float(2 * l)
                q = gp.mpz(int(q_float)) * p
                if q > 0:
                    self.ring_n = ring_n
                    break
        sigma2 = math.sqrt(float(2 * (l + 2) * self.ring_n * self.ring_n)) * sigma
        sigma2 *= sigma1 * kappa_sqrt
        sigma3 = sigma2 * math.sqrt(2.0)
        q_float = sigma1 * sigma2 * kappa * float(2 * self.ring_n)
        q_float += kappa_sqrt * sigma3
        q_float *= float(self.bound_y) * float(2 * l)
        q = gp.mpz(int(q_float)) * p

        self.p = p
        self.q = next_ntt_prime(q + 1, self.ring_n)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.A = rand_uniform_vector(self.ring_n, self.q)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "ring_n": self.ring_n,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma1": self.sigma1,
                    "sigma2": self.sigma2,
                    "sigma3": self.sigma3,
                    "A": [gp.digits(v) for v in self.A],
                },
                f,
            )

    def setup(self) -> None:
        sk = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
        noise = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
        pk = []
        for i in range(self.eta):
            pk_i = _poly_mul(self.A, sk[i], self.q)
            pk.append(_poly_add(pk_i, noise[i], self.q))
        self.msk = {"sk": sk}
        self.mpk = {"pk": pk}
        logger.info("SIFE Ring-LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": [gp.digits(v) for v in self.A],
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "eta": self.eta,
            "ring_n": self.ring_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sigma3": self.sigma3,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        return {"pk": [[gp.digits(v) for v in row] for row in self.mpk["pk"]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE Ring-LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weight) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")
        if any(abs(int(v)) > self.bound_y for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        sk_t = _transpose(self.msk["sk"])
        y_vec = [gp.mpz(v) for v in fusion_weight]
        sk_y = _mat_vec_mul(sk_t, y_vec, self.q)
        return {"sk_y": [gp.digits(v) for v in sk_y]}


class SIFERingLWE(IPFEAbsCrypto):
    """Crypto operations for Ring-LWE-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = [gp.mpz(v) for v in self.pp["A"]]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, matrix_pt: list[list[int]]) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if not isinstance(matrix_pt, list) or not matrix_pt or not isinstance(matrix_pt[0], list):
            raise FEValidationError("plaintext matrix must be a list of rows")
        if len(matrix_pt) != self.pp["eta"]:
            raise FEValidationError("invalid plaintext row count")
        if len(matrix_pt[0]) > self.pp["ring_n"]:
            raise FEValidationError("plaintext column count exceeds ring dimension")
        if not _matrix_check_bound(matrix_pt, gp.mpz(self.pp["bound_x"])):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        pk = [[gp.mpz(v) for v in row] for row in self.sk["pk"]]
        ring_n = self.pp["ring_n"]

        r = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        noise = discrete_gaussian_matrix(self.pp["eta"], ring_n, self.pp["sigma3"])
        ct0 = []
        for i in range(self.pp["eta"]):
            ct0_i = _poly_mul(pk[i], r, q)
            ct0.append(_poly_add(ct0_i, noise[i], q))
        centered = _center_matrix(matrix_pt, gp.mpz(self.pp["p"]), q, ring_n)
        ct0 = [_poly_add(a, b, q) for a, b in zip(ct0, centered)]

        ct1 = _poly_mul(self.A, r, q)
        e = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        ct1 = _poly_add(ct1, e, q)

        return {
            "ct0": [[gp.digits(v) for v in row] for row in ct0],
            "ct1": [gp.digits(v) for v in ct1],
            "k": len(matrix_pt[0]),
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not dk:
            raise FEKeyError("no decryption key provided")
        if len(fusion_weight) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        ct0 = [[gp.mpz(v) for v in row] for row in dct_ct["ct0"]]
        ct1 = [gp.mpz(v) for v in dct_ct["ct1"]]
        sk_y = [gp.mpz(v) for v in dk["sk_y"]]
        y_vec = [gp.mpz(v) for v in fusion_weight]

        ct0_t = _transpose(ct0)
        lhs = _mat_vec_mul(ct0_t, y_vec, q)
        rhs = _poly_mul(ct1, sk_y, q)
        d = _poly_add(lhs, _poly_neg(rhs, q), q)
        return _decode_vector(d, p, q, int(dct_ct["k"]))

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
