"""
Ring-LWE-Based Multi-Client Inner-Product Functional Encryption
| Multi-client label-aware organization inspired by
| "Decentralized multi-client functional encryption for inner product"
| By Jeremy Chotard, Edouard Dufour Sans, Romain Gay, Duong Hieu Phan,
| David Pointcheval
| Published in: ASIACRYPT 2018
|
| This prototype combines the repository's MCFE API with the Ring-LWE
| inner-product FE line from
| "Efficient Lattice-Based Inner-Product Functional Encryption"
| By Bermudo Mera, Karmakar, Marc, Soleimanian
| ePrint: 2021/046

* type:     public-key encryption
* setting:  Integer based
* note:     research prototype with SIMD-style matrix encryption and labels

"""

from __future__ import annotations

import json
import logging
import os
import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.utils.ring_lwe_utils import center_matrix
from pyfe4ai.utils.ring_lwe_utils import decode_vector
from pyfe4ai.utils.ring_lwe_utils import mat_vec_mul
from pyfe4ai.utils.ring_lwe_utils import matrix_check_bound
from pyfe4ai.utils.ring_lwe_utils import poly_add
from pyfe4ai.utils.ring_lwe_utils import poly_mul
from pyfe4ai.utils.ring_lwe_utils import poly_neg
from pyfe4ai.utils.ring_lwe_utils import transpose
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.lwe_utils import label_scalar_from_hash
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)


class MCFERingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Ring-LWE-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MCFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.ring_n = config.get("ring_n", None)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self.bound_u = gp.mpz(config.get("bound_u", 2))
        self.label_modulus = int(config.get("label_modulus", 8))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _effective_bound_x(self) -> gp.mpz:
        return self.bound_x + self.bound_u * gp.mpz(self.label_modulus // 2)

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
            and param.get("n") == self.n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("bound_u") == gp.digits(self.bound_u)
            and param.get("label_modulus") == self.label_modulus
            and (self.ring_n is None or param.get("ring_n") == self.ring_n)
        )

    def _generate_and_save(self, param_file: str) -> None:
        helper = MIFERingLWEKeyGenerator(
            {
                "sec_param": self.sec_param,
                "eta": self.eta,
                "n": self.n,
                "ring_n": self.ring_n,
                "bound_x": int(self._effective_bound_x()),
                "bound_y": int(self.bound_y),
            }
        )
        self.p = helper.p
        self.q = helper.q
        self.ring_n = helper.ring_n
        self.sigma1 = helper.sigma1
        self.sigma2 = helper.sigma2
        self.sigma3 = helper.sigma3
        self.A = helper.A

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "ring_n": self.ring_n,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "bound_u": gp.digits(self.bound_u),
                    "label_modulus": self.label_modulus,
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
        sk = {
            nid: discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
            for nid in self.lst_nid
        }
        u = {
            nid: [
                gp.mpz(_CSPRNG.randint(-int(self.bound_u), int(self.bound_u)))
                for _ in range(self.eta)
            ]
            for nid in self.lst_nid
        }
        pk = {}
        for nid in self.lst_nid:
            noise = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
            pk_nid = []
            for i in range(self.eta):
                pk_i = poly_mul(self.A, sk[nid][i], self.q)
                pk_nid.append(poly_add(pk_i, noise[i], self.q))
            pk[nid] = pk_nid

        self.msk = {"sk": sk, "u": u}
        self.mpk = {"pk": pk}
        logger.info("MCFE Ring-LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": [gp.digits(v) for v in self.A],
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "eta": self.eta,
            "n": self.n,
            "ring_n": self.ring_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "bound_u": gp.digits(self.bound_u),
            "label_modulus": self.label_modulus,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sigma3": self.sigma3,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid not in self.mpk["pk"]:
            return None
        return {
            "pk": [[gp.digits(v) for v in row] for row in self.mpk["pk"][nid]],
            "u": [gp.digits(v) for v in self.msk["u"][nid]],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE Ring-LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        sk_y = {}
        z = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            sk_t = transpose(self.msk["sk"][nid])
            y_vec = [gp.mpz(v) for v in y]
            sk_y[nid] = [gp.digits(v) for v in mat_vec_mul(sk_t, y_vec, self.q)]
            for u_i, y_i in zip(self.msk["u"][nid], y_vec):
                z += u_i * y_i
        return {"sk_y": sk_y, "z": gp.digits(z)}


class MCFERingLWE(IPFEAbsCrypto):
    """Crypto operations for Ring-LWE-based multi-client inner-product FE."""

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

    def encrypt(self, matrix_pt: list[list[int]], label: str) -> dict:
        """Encrypt plaintext and return a ciphertext dict.

            Args:
                matrix_pt: Integer plaintext matrix.
                label: Encryption label for replay protection.
        """
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
        if not matrix_check_bound(matrix_pt, gp.mpz(self.pp["bound_x"])):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        pk = [[gp.mpz(v) for v in row] for row in self.sk["pk"]]
        u = [gp.mpz(v) for v in self.sk["u"]]
        ring_n = self.pp["ring_n"]
        label_scalar = label_scalar_from_hash(
            md5_hash(label, p), self.pp["label_modulus"]
        )

        masked = []
        for row_idx, row in enumerate(matrix_pt):
            masked_row = [gp.mpz(v) + u[row_idx] * label_scalar for v in row]
            masked.append(masked_row)

        r = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        noise = discrete_gaussian_matrix(self.pp["eta"], ring_n, self.pp["sigma3"])
        ct0 = []
        for i in range(self.pp["eta"]):
            ct0_i = poly_mul(pk[i], r, q)
            ct0.append(poly_add(ct0_i, noise[i], q))
        centered = center_matrix(masked, p, q, ring_n)
        ct0 = [poly_add(a, b, q) for a, b in zip(ct0, centered)]

        ct1 = poly_mul(self.A, r, q)
        e = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        ct1 = poly_add(ct1, e, q)

        return {
            "ct0": [[gp.digits(v) for v in row] for row in ct0],
            "ct1": [gp.digits(v) for v in ct1],
            "k": len(matrix_pt[0]),
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
                label: Encryption label for replay protection.
        """
        if not dk:
            raise FEKeyError("no decryption key provided")

        total = None
        for nid, ct in dct_ct.items():
            y = fusion_weight[nid]
            if len(y) != self.pp["eta"]:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            q = gp.mpz(self.pp["q"])
            p = gp.mpz(self.pp["p"])
            ct0 = [[gp.mpz(v) for v in row] for row in ct["ct0"]]
            ct1 = [gp.mpz(v) for v in ct["ct1"]]
            sk_y = [gp.mpz(v) for v in dk["sk_y"][nid]]
            y_vec = [gp.mpz(v) for v in y]

            ct0_t = transpose(ct0)
            lhs = mat_vec_mul(ct0_t, y_vec, q)
            rhs = poly_mul(ct1, sk_y, q)
            d = poly_add(lhs, poly_neg(rhs, q), q)
            dec = decode_vector(d, p, q, int(ct["k"]))
            if total is None:
                total = dec
            else:
                total = [a + b for a, b in zip(total, dec)]

        label_scalar = label_scalar_from_hash(
            md5_hash(label, gp.mpz(self.pp["p"])), self.pp["label_modulus"]
        )
        correction = int(gp.mpz(dk["z"]) * label_scalar)
        return [v - correction for v in total]

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([[int(ary[i])]], label)
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, label)[0]
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
