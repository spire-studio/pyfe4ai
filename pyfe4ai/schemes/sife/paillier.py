"""
Paillier-Based Single-Input Inner-Product Functional Encryption
| Inspired by the fully secure Paillier instantiation exposed in CiFEr / GoFE
| From "Fully Secure Functional Encryption for Inner Products,
|       from Standard Assumptions"
| By Shweta Agrawal, Benoit Libert, and Damien Stehle
| Published in: CRYPTO 2016

* type:     public-key encryption
* setting:  Integer based
* note:     research-oriented Python prototype aligned with the existing
            crypto-ipfe API style

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
from pyfe4ai.utils.crypto_utils import generate_group_primes
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.sampling_utils import random_below
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)

class SIFEPaillierKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Paillier-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE_PAILLIER
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bit_length = config.get("bit_length", 128)
        self.bound_x = config.get("bound_x", 100)
        self.bound_y = config.get("bound_y", 100)
        self._load_parameters()
        self._validate_bounds()

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.n = gp.mpz(param["n"])
        self.n_square = gp.mpz(param["n_square"])
        self.g = gp.mpz(param["g"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bit_length") == self.bit_length
            and param.get("bound_x") == self.bound_x
            and param.get("bound_y") == self.bound_y
        )

    def _generate_and_save(self, param_file: str) -> None:
        self.p, _ = generate_group_primes(self.bit_length)
        self.q, _ = generate_group_primes(self.bit_length)
        self.n = self.p * self.q
        self.n_square = self.n * self.n
        self._validate_bounds()

        while True:
            g_prime = random_below(self.n_square - 1)
            g = gp.powmod(g_prime, self.n, self.n_square)
            g = gp.powmod(g, 2, self.n_square)
            if gp.gcd(g, self.n_square) == 1:
                self.g = g
                break

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "bit_length": self.bit_length,
                    "bound_x": self.bound_x,
                    "bound_y": self.bound_y,
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "n": gp.digits(self.n),
                    "n_square": gp.digits(self.n_square),
                    "g": gp.digits(self.g),
                },
                f,
            )

    def _validate_bounds(self) -> None:
        x_square_l = gp.mpz(2 * self.eta * (self.bound_x**2))
        y_square_l = gp.mpz(2 * self.eta * (self.bound_y**2))
        if self.n <= x_square_l:
            raise ValueError("bound_x and eta are too large for the chosen bit_length")
        if self.n <= y_square_l:
            raise ValueError("bound_y and eta are too large for the chosen bit_length")

    def setup(self) -> None:
        # Prototype note: we use bounded centered integer sampling for secret
        # keys to keep the Python implementation simple and reproducible.
        sk_bound = max(self.sec_param, self.bound_y * self.eta)
        sec = [
            gp.mpz(_CSPRNG.randint(-sk_bound, sk_bound))
            for _ in range(self.eta)
        ]
        pub = [pow_signed(self.g, s_i, self.n_square) for s_i in sec]
        self.msk = {"s": sec}
        self.mpk = {
            "n": self.n,
            "n_square": self.n_square,
            "g": self.g,
            "pk": pub,
        }
        logger.info("SIFE Paillier setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "n": gp.digits(self.mpk["n"]),
            "n_square": gp.digits(self.mpk["n_square"]),
            "g": gp.digits(self.mpk["g"]),
            "eta": self.eta,
            "sec_param": self.sec_param,
            "bit_length": self.bit_length,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        return {"pk": [gp.digits(v) for v in self.mpk["pk"]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE Paillier decryption key generation")
        fusion_weights = credentials.get("fusion_weight")
        if not isinstance(fusion_weights, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weights) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")
        if any(abs(int(v)) > self.bound_y for v in fusion_weights):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        k = gp.mpz(0)
        for s_i, y_i in zip(self.msk["s"], fusion_weights):
            k += s_i * gp.mpz(y_i)
        return {"k": gp.digits(k)}


class SIFEPaillier(IPFEAbsCrypto):
    """Crypto operations for Paillier-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > self.pp["bound_x"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        g = gp.mpz(self.pp["g"])
        pk = [gp.mpz(v) for v in self.sk["pk"]]

        r = random_below(n // 4)
        c0 = gp.powmod(g, r, n_square)
        c1 = []
        for x_i, pk_i in zip(lst_pt, pk):
            x_term = gp.mpz(1) + gp.mpz(x_i) * n
            x_term %= n_square
            ct_i = (x_term * gp.powmod(pk_i, r, n_square)) % n_square
            c1.append(gp.digits(ct_i))

        return {"ct0": gp.digits(c0), "ct1": c1}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for decryption")
        if not dk:
            raise FEKeyError("no decryption key provided")
        if len(fusion_weight) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length")
        if any(abs(int(v)) > self.pp["bound_y"] for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        c0 = gp.mpz(dct_ct["ct0"])
        k = gp.mpz(dk["k"])

        acc = pow_signed(c0, -k, n_square)
        for ct_i, y_i in zip(dct_ct["ct1"], fusion_weight):
            acc *= pow_signed(gp.mpz(ct_i), gp.mpz(y_i), n_square)
            acc %= n_square

        ret = ((acc - 1) % n_square) // n
        n_half = n // 2
        if ret > n_half:
            ret -= n
        return int(ret)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        g = gp.mpz(self.pp["g"])
        pk = [gp.mpz(v) for v in self.sk["pk"]]
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                x_i = int(w)
                r = random_below(n // 4)
                c0 = gp.powmod(g, r, n_square)
                x_term = (gp.mpz(1) + gp.mpz(x_i) * n) % n_square
                ct_i = (x_term * gp.powmod(pk[0], r, n_square)) % n_square
                ary_ct[i] = {"ct0": gp.digits(c0), "ct1": [gp.digits(ct_i)]}
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

        lst_ndarray = []
        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        k = gp.mpz(dk["k"])
        for ary in dict_ndarray_ct:
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ct = ary[i]
                c0 = gp.mpz(ct["ct0"])
                acc = pow_signed(c0, -k, n_square)
                for ct_j, y_j in zip(ct["ct1"], fusion_weight):
                    acc *= pow_signed(gp.mpz(ct_j), gp.mpz(y_j), n_square)
                    acc %= n_square
                ret = ((acc - 1) % n_square) // n
                n_half = n // 2
                if ret > n_half:
                    ret -= n
                ary_dec[i] = int(ret)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
