"""
Decentralized LWE-Based Multi-Client Inner-Product Functional Encryption
| Decentralized key-share organization inspired by
| "Decentralizing inner-product functional encryption"
| By Michel Abdalla, Fabrice Benhamouda, Markulf Kohlweiss, Hendrik Waldner
| Published in: PKC 2019
|
| This prototype combines decentralized decryption-key-share derivation with
| the LWE-based single-input inner-product FE line from
| "Simple Functional Encryption Schemes for Inner Products"
| By Michel Abdalla, Florian Bourse, Angelo De Caro, David Pointcheval
| Published in: PKC 2015

* type:     public-key encryption
* setting:  Integer based
* note:     research prototype aligned with the existing crypto-ipfe API

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
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.lwe_utils import center_lwe_vector
from pyfe4ai.utils.lwe_utils import decode_lwe_inner_product
from pyfe4ai.utils.lwe_utils import derive_lwe_parameters
from pyfe4ai.utils.lwe_utils import label_scalar_from_hash
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


class DecentralizedMCFELWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for decentralized LWE-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_DMCFE_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.lwe_n = config.get("lwe_n", 32)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        self.bound_u = gp.mpz(config.get("bound_u", 4))
        self.label_modulus = int(config.get("label_modulus", 8))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _effective_bound_x(self) -> gp.mpz:
        return self.bound_x + self.bound_u * gp.mpz(self.label_modulus // 2)

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma = gp.mpz(param["l_sigma"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("lwe_n") == self.lwe_n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("bound_u") == gp.digits(self.bound_u)
            and param.get("label_modulus") == self.label_modulus
            and param.get("lst_nid") == self.lst_nid
        )

    def _generate_and_save(self, param_file: str) -> None:
        eff_bound_x = self._effective_bound_x()
        self.p, self.q, self.m, self.sigma_q, self.l_sigma = derive_lwe_parameters(
            eff_bound_x,
            self.bound_y,
            self.eta,
            self.lwe_n,
            extra_dimension=self.n,
        )
        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "lst_nid": self.lst_nid,
                    "lwe_n": self.lwe_n,
                    "m": self.m,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "bound_u": gp.digits(self.bound_u),
                    "label_modulus": self.label_modulus,
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma_q": self.sigma_q,
                    "l_sigma": gp.digits(self.l_sigma),
                    "A": matrix_to_digits(self.A),
                },
                f,
            )

    def setup(self) -> None:
        sk = {
            nid: rand_uniform_matrix(self.lwe_n, self.eta, self.q)
            for nid in self.lst_nid
        }
        u = {
            nid: [
                gp.mpz(
                    _CSPRNG.randint(-int(self.bound_u), int(self.bound_u))
                )
                for _ in range(self.eta)
            ]
            for nid in self.lst_nid
        }

        v = {}
        nid_selected = _CSPRNG.sample(self.lst_nid, 1)[0]
        for nid in self.lst_nid:
            if nid != nid_selected:
                v[nid] = [
                    gp.mpz(
                        _CSPRNG.randint(
                            -int(self.bound_u), int(self.bound_u)
                        )
                    )
                    for _ in range(self.eta * self.n)
                ]
        lst_v = [vec for vec in v.values()]
        v[nid_selected] = [-(sum(vals)) for vals in zip(*lst_v)] if lst_v else [
            gp.mpz(0) for _ in range(self.eta * self.n)
        ]

        pk = {}
        for nid in self.lst_nid:
            noise = discrete_gaussian_matrix(self.m, self.eta, self.sigma_q)
            pk_nid = matmul_mod(self.A, sk[nid], self.q)
            for i in range(self.m):
                for j in range(self.eta):
                    pk_nid[i][j] = (pk_nid[i][j] + noise[i][j]) % self.q
            pk[nid] = pk_nid

        self.msk = {"sk": sk, "u": u, "v": v}
        self.mpk = {"pk": pk}
        logger.info("Decentralized MCFE LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": matrix_to_digits(self.A),
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "m": self.m,
            "eta": self.eta,
            "n": self.n,
            "lst_nid": self.lst_nid,
            "lwe_n": self.lwe_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "bound_u": gp.digits(self.bound_u),
            "label_modulus": self.label_modulus,
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
            "pk": matrix_to_digits(self.mpk["pk"][nid]),
            "u": vector_to_digits(self.msk["u"][nid]),
            "sk": matrix_to_digits(self.msk["sk"][nid]),
            "v": vector_to_digits(self.msk["v"][nid]),
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        return NotImplementedError


class DecentralizedMCFELWE(IPFEAbsCrypto):
    """Crypto operations for decentralized LWE-based multi-client inner-product FE."""

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

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt plaintext and return a ciphertext dict.

            Args:
                lst_pt: Integer plaintext vector.
                label: Encryption label for replay protection.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        pk = digits_to_matrix(self.sk["pk"])
        u = digits_to_vector(self.sk["u"])
        label_scalar = label_scalar_from_hash(
            md5_hash(label, gp.mpz(self.pp["p"])), self.pp["label_modulus"]
        )
        masked = [gp.mpz(x) + u_i * label_scalar for x, u_i in zip(lst_pt, u)]

        r = rand_bit_vector(self.pp["m"])
        ct0 = matvec_mod(transpose(self.A), r, q)
        ct1 = matvec_mod(transpose(pk), r, q)
        centered = center_lwe_vector(masked, gp.mpz(self.pp["p"]), q)
        ct1 = [(v + c) % q for v, c in zip(ct1, centered)]
        return {"ct0": vector_to_digits(ct0), "ct1": vector_to_digits(ct1)}

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for key-share derivation")
        if self.id not in fusion_weight:
            raise FEValidationError("invalid fusion weight provided")

        q = gp.mpz(self.pp["q"])
        y = fusion_weight[self.id]
        if len(y) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length for client {}".format(self.id))
        if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in y):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        y_vec = [gp.mpz(v) for v in y]
        sk_matrix = digits_to_matrix(self.sk["sk"])
        sk_y = matvec_mod(sk_matrix, y_vec, q)

        u = digits_to_vector(self.sk["u"])
        z_local = gp.mpz(0)
        for u_i, y_i in zip(u, y_vec):
            z_local += u_i * y_i

        ordered_fw = []
        for nid in self.pp["lst_nid"]:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            ordered_fw.extend([gp.mpz(v) for v in fusion_weight[nid]])
        v = digits_to_vector(self.sk["v"])
        if len(v) < len(ordered_fw):
            raise FEValidationError("invalid private key share length")
        z_pad = gp.mpz(0)
        for v_i, w_i in zip(v, ordered_fw):
            z_pad += v_i * w_i

        return {"sk_y": vector_to_digits(sk_y), "z": gp.digits(z_local + z_pad)}

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        return {
            "sk_y": {nid: share["sk_y"] for nid, share in dct_dk_shares.items()},
            "z": gp.digits(sum(gp.mpz(share["z"]) for share in dct_dk_shares.values())),
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

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        total = 0
        for nid, ct in dct_ct.items():
            y = fusion_weight[nid]
            if len(y) != self.pp["eta"]:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            y_vec = [gp.mpz(v) for v in y]
            ct0 = digits_to_vector(ct["ct0"])
            ct1 = digits_to_vector(ct["ct1"])
            sk_y = digits_to_vector(dk["sk_y"][nid])
            d = (vecdot_mod(y_vec, ct1, q) - vecdot_mod(ct0, sk_y, q)) % q
            total += decode_lwe_inner_product(d, p, q)

        label_scalar = label_scalar_from_hash(
            md5_hash(label, p), self.pp["label_modulus"]
        )
        total -= int(gp.mpz(dk["z"]) * label_scalar)
        return total

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
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)], label)
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

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
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, label)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray

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
