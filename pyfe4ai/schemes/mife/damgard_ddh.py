"""
Damgard-Based Multi-Input Inner-Product Functional Encryption
| Multi-input construction inspired by
| "Multi-Input Functional Encryption for Inner Products:
|  Function-Hiding Realizations and Constructions without Pairings"
| By Michel Abdalla, Dario Catalano, Dario Fiore, Romain Gay, Bogdan Ursu
| Published in: CRYPTO 2018
|
| Instantiated here with a Damgard/DDH-based single-input building block inspired by
| Agrawal, Libert, Stehle:
| "Fully Secure Functional Encryption for Inner Products, from Standard
|  Assumptions"
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

from pyfe4ai.schemes.ddh_base import DamgardDDHKeyGeneratorBase
from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)

class MIFEDamgardKeyGenerator(DamgardDDHKeyGeneratorBase):
    """Key generator for Damgard DDH-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_DAMGARD
    _verify_keys = ("sec_param", "eta", "n", "modulus_length", "bound")

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.modulus_length = config.get("modulus_length", self.sec_param)
        self.bound = config.get("bound", 100)
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()
        self._validate_bounds()

    def _extra_param_fields(self) -> dict:
        return {"n": self.n}

    def _validate_bounds(self) -> None:
        prod = gp.mpz(2 * self.eta * (self.bound**2))
        if prod >= self.q:
            raise ValueError(
                "2 * eta * bound^2 should be smaller than the group order"
            )

    def setup(self) -> None:
        rand = _CSPRNG
        s = {
            nid: [gp.mpz(rand.randint(2, int(self.q - 1))) for _ in range(self.eta)]
            for nid in self.lst_nid
        }
        t = {
            nid: [gp.mpz(rand.randint(2, int(self.q - 1))) for _ in range(self.eta)]
            for nid in self.lst_nid
        }
        pk = {
            nid: [
                gp.mod(
                    gp.mul(gp.powmod(self.g, s_i, self.p), gp.powmod(self.h, t_i, self.p)),
                    self.p,
                )
                for s_i, t_i in zip(s[nid], t[nid])
            ]
            for nid in self.lst_nid
        }
        self.msk = {"s": s, "t": t}
        self.mpk = {"p": self.p, "q": self.q, "g": self.g, "h": self.h, "pk": pk}
        logger.info("MIFE Damgard setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "p": gp.digits(self.mpk["p"]),
            "q": gp.digits(self.mpk["q"]),
            "g": gp.digits(self.mpk["g"]),
            "h": gp.digits(self.mpk["h"]),
            "eta": self.eta,
            "n": self.n,
            "bound": self.bound,
            "sec_param": self.sec_param,
            "modulus_length": self.modulus_length,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid not in self.mpk["pk"]:
            return None
        return {"pk": [gp.digits(v) for v in self.mpk["pk"][nid]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MIFE Damgard decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        k1 = {}
        k2 = {}
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            weights = fusion_weight[nid]
            if len(weights) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound for v in weights):
                raise FEValidationError("fusion weight exceeds configured bound")
            acc1 = gp.mpz(0)
            acc2 = gp.mpz(0)
            for s_i, t_i, y_i in zip(self.msk["s"][nid], self.msk["t"][nid], weights):
                y_mpz = gp.mpz(y_i)
                acc1 += s_i * y_mpz
                acc2 += t_i * y_mpz
            k1[nid] = gp.digits(acc1 % self.q)
            k2[nid] = gp.digits(acc2 % self.q)
        return {"k1": k1, "k2": k2}


class MIFEDamgard(IPFEAbsCrypto):
    """Crypto operations for Damgard DDH-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        else:
            self._load_dlog_table()

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_MIFE_DAMGARD,
            f"dlog_{self.pp['bound']}.json",
        )
        bound = self.pp["n"] * self.pp["eta"] * (self.pp["bound"] ** 2) + 1
        self.dlog_table, self.func_bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"], bound,
            )
        )

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > self.pp["bound"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound")

        p = gp.mpz(self.pp["p"])
        q = gp.mpz(self.pp["q"])
        g = gp.mpz(self.pp["g"])
        h = gp.mpz(self.pp["h"])
        pk = [gp.mpz(v) for v in self.sk["pk"]]
        r = _random(q, self.pp["modulus_length"])

        c = gp.powmod(g, r, p)
        dd = gp.powmod(h, r, p)
        e = []
        for x_i, pk_i in zip(lst_pt, pk):
            ct_i = gp.mod(
                gp.mul(gp.powmod(pk_i, r, p), pow_signed(g, gp.mpz(x_i), p)), p
            )
            e.append(gp.digits(ct_i))
        return {"c": gp.digits(c), "d": gp.digits(dd), "e": e}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
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

        p = gp.mpz(self.pp["p"])
        num = gp.mpz(1)
        denom = gp.mpz(1)
        for nid, ct in dct_ct.items():
            for ct_i, y_i in zip(ct["e"], fusion_weight[nid]):
                num = gp.mod(num * pow_signed(gp.mpz(ct_i), gp.mpz(y_i), p), p)
            denom = gp.mod(
                denom
                * gp.powmod(gp.mpz(ct["c"]), gp.mpz(dk["k1"][nid]), p)
                * gp.powmod(gp.mpz(ct["d"]), gp.mpz(dk["k2"][nid]), p),
                p,
            )
        ret = gp.divm(num, denom, p)
        return self._solve_dlog(ret)

    def _solve_dlog(self, encoded_value: gp.mpz) -> int | None:
        try:
            return dlog_table_solve(
                encoded_value, self.pp["g"], self.pp["p"],
                self.func_bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system")
            return None

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
