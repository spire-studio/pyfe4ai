"""
Fully Secure LWE-Based Multi-Input Inner-Product Functional Encryption
| Multi-input organization inspired by
| "Multi-Input Functional Encryption for Inner Products:
|  Function-Hiding Realizations and Constructions without Pairings"
| By Michel Abdalla, Dario Catalano, Dario Fiore, Romain Gay, Bogdan Ursu
| Published in: CRYPTO 2018
|
| Instantiated here with the fully secure LWE single-input line from
| "Fully Secure Functional Encryption for Inner Products,
|  from Standard Assumptions"
| By Shweta Agrawal, Benoit Libert, Damien Stehle
| Published in: CRYPTO 2016

* type:     public-key encryption
* setting:  Integer based
* note:     research-oriented Python prototype aligned with the crypto-ipfe API

"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.lwe_utils import decode_fullysec_lwe_inner_product
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import center_lwe_vector
from pyfe4ai.utils.lwe_utils import derive_fullysec_lwe_parameters
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matrix_to_digits
from pyfe4ai.utils.matrix_utils import matmul_mod
from pyfe4ai.utils.matrix_utils import matvec_mod
from pyfe4ai.utils.matrix_utils import transpose
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.matrix_utils import vector_to_digits
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_uniform_matrix
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)


class MIFEFullySecLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for fully secure LWE-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_FULLYSEC_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.lwe_n = config.get("lwe_n", 64)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.k = gp.mpz(param["k"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma_q = gp.mpz(param["l_sigma_q"])
        self.sigma1 = float(param["sigma1"])
        self.l_sigma1 = gp.mpz(param["l_sigma1"])
        self.sigma2 = float(param["sigma2"])
        self.l_sigma2 = gp.mpz(param["l_sigma2"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("lwe_n") == self.lwe_n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
        )

    def _generate_and_save(self, param_file: str) -> None:
        (
            self.k,
            self.q,
            self.m,
            self.sigma_q,
            self.l_sigma_q,
            self.sigma1,
            self.l_sigma1,
            self.sigma2,
            self.l_sigma2,
        ) = derive_fullysec_lwe_parameters(
            self.bound_x, self.bound_y, self.eta, self.lwe_n
        )
        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=os.path.dirname(param_file), delete=False
        ) as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "lwe_n": self.lwe_n,
                    "m": self.m,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "k": gp.digits(self.k),
                    "q": gp.digits(self.q),
                    "sigma_q": self.sigma_q,
                    "l_sigma_q": gp.digits(self.l_sigma_q),
                    "sigma1": self.sigma1,
                    "l_sigma1": gp.digits(self.l_sigma1),
                    "sigma2": self.sigma2,
                    "l_sigma2": gp.digits(self.l_sigma2),
                    "A": matrix_to_digits(self.A),
                },
                f,
            )
            temp_name = f.name
        os.replace(temp_name, param_file)

    def setup(self) -> None:
        half_cols = self.m // 2
        z = {}
        u = {}
        for nid in self.lst_nid:
            z_nid = [[gp.mpz(0) for _ in range(self.m)] for _ in range(self.eta)]
            for i in range(self.eta):
                for j in range(self.m):
                    if j < half_cols:
                        sampled = discrete_gaussian_matrix(1, 1, self.sigma1)[0][0]
                    else:
                        sampled = discrete_gaussian_matrix(1, 1, self.sigma2)[0][0]
                        if j - half_cols == i:
                            sampled += 1
                    z_nid[i][j] = sampled
            z[nid] = z_nid
            u[nid] = matmul_mod(z_nid, self.A, self.q)

        self.msk = {"z": z}
        self.mpk = {"u": u}
        logger.info("MIFE fully secure LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": matrix_to_digits(self.A),
            "k": gp.digits(self.k),
            "q": gp.digits(self.q),
            "m": self.m,
            "eta": self.eta,
            "n": self.n,
            "lwe_n": self.lwe_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma_q": self.sigma_q,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid not in self.mpk["u"]:
            return None
        return {"pk": matrix_to_digits(self.mpk["u"][nid])}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MIFE fully secure LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        z_y = {}
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            y_vec = [gp.mpz(v) for v in y]
            z_t = transpose(self.msk["z"][nid])
            z_y[nid] = vector_to_digits(matvec_mod(z_t, y_vec, self.q))
        return {"sk_y": z_y}


class MIFEFullySecLWE(IPFEAbsCrypto):
    """Crypto operations for fully secure LWE-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = digits_to_matrix(self.pp["A"])
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        k = gp.mpz(self.pp["k"])
        pk = digits_to_matrix(self.sk["pk"])
        r = rand_uniform_vector(self.pp["lwe_n"], q)
        e0 = discrete_gaussian_matrix(1, self.pp["m"], self.pp["sigma_q"])[0]
        e1 = discrete_gaussian_matrix(1, self.pp["eta"], self.pp["sigma_q"])[0]

        c0 = matvec_mod(self.A, r, q)
        c0 = [(a + b) % q for a, b in zip(c0, e0)]

        c1 = matvec_mod(pk, r, q)
        centered = center_lwe_vector(lst_pt, k, q)
        c1 = [(a + b + c) % q for a, b, c in zip(c1, e1, centered)]
        return {"ct0": vector_to_digits(c0), "ct1": vector_to_digits(c1)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not dk:
            raise FEKeyError("no decryption key provided")

        q = gp.mpz(self.pp["q"])
        k = gp.mpz(self.pp["k"])
        total = 0
        for nid in dct_ct:
            y = fusion_weight[nid]
            if len(y) != self.pp["eta"]:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            y_vec = [gp.mpz(v) for v in y]
            ct0 = digits_to_vector(dct_ct[nid]["ct0"])
            ct1 = digits_to_vector(dct_ct[nid]["ct1"])
            z_y = digits_to_vector(dk["sk_y"][nid])

            mu = (vecdot_mod(y_vec, ct1, q) - vecdot_mod(z_y, ct0, q)) % q
            total += decode_fullysec_lwe_inner_product(mu, k, q)
        return total

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
