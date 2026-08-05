"""
Multi Input Functional Encryption
| From "Multi-Input Functional Encryption for Inner Products:
|          Function-Hiding Realizations and Constructions without Pairings"
| Published in: CRYPTO 2018
| By: Michel Abdalla, Dario Catalano, Dario Fiore, Romain Gay, and Bogdan Ursu5
| URL: https://eprint.iacr.org/2017/972.pdf

* type:     public-key encryption
* setting:  Integer based

"""

from __future__ import annotations

import os
import json
import logging

from functools import reduce
import numpy as np
import gmpy2 as gp

from pyfe4ai.schemes.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError


logger = logging.getLogger(__name__)


class MIFEKeyGenerator(DDHKeyGeneratorBase):
    """Key generator for DDH-based multi-input FE (hybrid-alpha variant)."""

    _scheme_type = CryptoCONST.TYPE_MIFE
    _verify_keys = ("sec_param", "eta", "n", "s")

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        if isinstance(self.eta, int):
            self.dict_eta = {nid: self.eta for nid in self.lst_nid}
        elif isinstance(self.eta, dict) and len(self.eta) == self.n:
            self.dict_eta = self.eta
        else:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        self._load_parameters()

    def _extra_param_fields(self) -> dict:
        return {"n": self.n, "s": self.s}

    def setup(self) -> None:
        w, u, g_w = dict(), dict(), dict()

        for nid in self.lst_nid:
            w_nid, u_nid, g_w_nid = [], [], []
            for _ in range(self.dict_eta[nid]):
                _r = _random(self.p, self.sec_param)
                w_nid.append(_r)
                g_w_nid.append(gp.powmod(self.g, _r, self.p))
                u_nid.append(_random(self.p, self.sec_param))
            w[nid] = w_nid
            u[nid] = u_nid
            g_w[nid] = g_w_nid

        self.keys = {
            "g": self.g,
            "p": self.p,
            "g_w": g_w,
            "w": w,
            "u": u,
        }

    def get_public_parameters(self) -> dict:
        logger.debug("generating public parameters ...")
        return {
            "g": gp.digits(self.keys["g"]),
            "p": gp.digits(self.keys["p"]),
            "n": self.n,
            "eta": self.eta,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str) -> dict | None:
        logger.debug("generating private key(s)...")
        # if nid is None or not isinstance(nid, str) or nid not in self.lst_nid:
        #     logger.error("no id or invalid id  provided.")
        #     return None

        _keys = {
            "w": [gp.digits(i) for i in self.keys["w"][nid]],
            "u": [gp.digits(i) for i in self.keys["u"][nid]],
        }
        return _keys

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        _credentials = kwargs.get("credentials", None)
        if not _credentials:
            raise FEKeyError("need to provided credentials for MIFE")
        _fusion_weights = _credentials.get("fusion_weight")
        d = {}
        z = gp.mpz(0)
        for nid, lst_fusion in _fusion_weights.items():
            if nid in self.keys["w"] and nid in self.keys["u"]:
                w_nid = self.keys["w"][nid]
                u_nid = self.keys["u"][nid]
                w_fusion = gp.mpz(0)
                for i in range(len(lst_fusion)):
                    w_fusion += gp.mul(w_nid[i], gp.mpz(lst_fusion[i]))
                d[nid] = gp.digits(w_fusion)
                for i in range(len(lst_fusion)):
                    z += gp.mul(u_nid[i], gp.mpz(lst_fusion[i]))
            else:
                logger.error("invalid identifier in provided fusion weight.")

        return {"d": d, "z": gp.digits(z)}


class MIFE(IPFEAbsCrypto):
    """Crypto operations for DDH-based multi-input FE (hybrid-alpha variant)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        else:
            self._load_dlog_table()

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_MIFE,
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"],
                pow(10, self.precision + 2),
            )
        )

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) > len(self.sk["u"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(lst_pt))
        if not isinstance(lst_pt, list):
            raise FEValidationError("invalid format of input plaintext:{}".format(lst_pt))

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        sec_param = self.pp["sec_param"]
        u = self.sk["u"]
        w = self.sk["w"]

        r = _random(p, sec_param)
        # r = gp.mpz("93023873389307946419466876648394882596")
        t = gp.digits(gp.powmod(g, r, p))

        c = [
            gp.digits(
                gp.powmod(
                    g, gp.mpz(lst_pt[i]) + gp.mpz(u[i]) + gp.mul(gp.mpz(w[i]), r), p
                )
            )
            for i in range(len(lst_pt))
        ]

        return {"t": t, "c": c}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if dct_ct.keys() != dk.keys() and dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct, dk, fusion wight")

        gp_prod = lambda i, j: gp.mul(i, j) % p

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        z = gp.mpz(dk["z"])
        d = dk["d"]
        lst_gf = list()
        for nid in dct_ct.keys():
            f_nid = fusion_weight[nid]
            c_nid = dct_ct[nid]["c"]
            t_nid = gp.mpz(dct_ct[nid]["t"])
            d_nid = gp.mpz(d[nid])
            cf_prod = (
                reduce(
                    gp_prod,
                    [
                        gp.powmod(gp.mpz(c_nid[i]), gp.mpz(f_nid[i]), p)
                        for i in range(len(c_nid))
                    ],
                )
                % p
            )
            lst_gf.append(gp.divm(cf_prod, gp.powmod(t_nid, d_nid, p), p))
        gf = gp.divm(reduce(gp_prod, lst_gf), gp.powmod(g, z, p), p)

        return self._solve_dlog(gf)

    def decrypt_debug(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Decrypt with additional debug output.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if dct_ct.keys() != dk.keys() and dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct, dk, fusion wight")

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        z = gp.mpz(dk["z"])
        d = dk["d"]
        gf = gp.mpz(1)
        for nid in dct_ct.keys():
            f_nid = fusion_weight[nid]
            c_nid = dct_ct[nid]["c"]
            t_nid = gp.mpz(dct_ct[nid]["t"])
            d_nid = gp.mpz(d[nid])
            cf_prod = gp.mpz(1)
            for i in range(len(c_nid)):
                cf_prod = gp.mul(
                    cf_prod, gp.powmod(gp.mpz(c_nid[i]), gp.mpz(f_nid[i]), p)
                )
            logger.debug("DEBUG 1-{}".format(cf_prod))
            gf = gp.mul(gf, gp.divm(cf_prod, gp.powmod(t_nid, d_nid, p), p))
            logger.debug("DEBUG 2-{}".format(gf))
        gf = gp.divm(gf, gp.powmod(g, z, p), p)
        logger.debug("DEBUG 3-{}".format(gf))
        return self._solve_dlog(gf)

    def _solve_dlog(self, g_inner_prod) -> int | None:
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
        lst_ndarray_ct = list()
        for l in range(len(lst_ndarray)):
            ary = (lst_ndarray[l].copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([w])
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def decrypt_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, dk: dict, fusion_weight: dict
    ) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        _sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = list()
        for l in range(len(_sample)):
            ary = _sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {
                    nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct.keys()
                }
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))

        return lst_ndarray

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk, fusion_weight)
