"""
Abdalla, Michel, Fabrice Benhamouda, Markulf Kohlweiss, and Hendrik Waldner. 
"Decentralizing inner-product functional encryption." 
In IACR International Workshop on Public Key Cryptography, pp. 128-157. 
Cham: Springer International Publishing, 2019.

* setting:  Integer based

"""

from __future__ import annotations

import os
import json
import random
import logging

_CSPRNG = random.SystemRandom()

import numpy as np
import gmpy2 as gp


from pyfe4ai.schemes.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError


logger = logging.getLogger(__name__)


class DecentralizedMCFEKeyGenerator(DDHKeyGeneratorBase):
    """Key generator for decentralized DDH-based multi-client inner-product FE."""

    _scheme_type = "dMCFE"
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
        dct_s, dct_u, dct_v = dict(), dict(), dict()

        nid_selected = _CSPRNG.sample(self.lst_nid, 1)[0]
        for nid in self.lst_nid:
            dct_s[nid] = [_random(self.p, self.sec_param) for _ in range(self.eta)]
            dct_u[nid] = [_random(self.p, self.sec_param) for _ in range(self.eta)]
            if nid != nid_selected:
                dct_v[nid] = [
                    _random(self.p, self.sec_param) for _ in range(self.eta * self.n)
                ]
        lst_v = [v for v in dct_v.values()]
        dct_v[nid_selected] = [-sum(_lst_v) for _lst_v in zip(*lst_v)]

        self.mpk = {"g": self.g, "p": self.p}
        self.msk = {"s": dct_s, "u": dct_u, "v": dct_v}
        logger.info("Decentralized MCFE setup - DONE.")

    def get_public_parameters(self) -> dict:
        logger.debug("generating public parameters ...")
        return {
            "g": gp.digits(self.mpk["g"]),
            "p": gp.digits(self.mpk["p"]),
            "n": self.n,
            "eta": self.eta,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str) -> dict | None:
        logger.debug("generating private key(s)...")
        if nid is None or not isinstance(nid, str) or nid not in self.lst_nid:
            logger.error("no id or invalid id  provided.")
            return None

        _keys = {
            "s": [gp.digits(i) for i in self.msk["s"][nid]],
            "u": [gp.digits(i) for i in self.msk["u"][nid]],
            "v": [gp.digits(i) for i in self.msk["v"][nid]],
        }
        return _keys

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not supported: clients derive functional-key shares locally.

            Args:
                sid: Session / decryption-key identifier.

            Raises:
                NotImplementedError: Always.
        """
        raise NotImplementedError(
            "{} has no central decryption-key derivation: in decentralized "
            "MCFE each client derives its key share with "
            "derive_function_decryption_key_share() and the aggregator "
            "combines them with combine_function_decryption_key_share()".format(
                type(self).__name__
            )
        )


class DecentralizedMCFE(IPFEAbsCrypto):
    """Crypto operations for decentralized DDH-based multi-client inner-product FE."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        else:
            self._load_dlog_table()

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, "dMCFE",
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"],
                pow(10, self.precision + 2),
            )
        )

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
        if len(lst_pt) > len(self.sk["u"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(lst_pt))
        if not isinstance(lst_pt, list):
            raise FEValidationError("invalid format of input plaintext:{}".format(lst_pt))

        sec_param = self.pp["sec_param"]
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        s = [gp.mpz(i) for i in self.sk["s"]]
        u = [gp.mpz(i) for i in self.sk["u"]]

        r = _random(p, sec_param)
        ct0 = gp.digits(gp.powmod(g, r, p))
        ct1 = [
            gp.digits(
                gp.mul(
                    gp.powmod(g, gp.mul(r, s[i]), p),
                    gp.powmod(g, lst_pt[i] + gp.mul(u[i], md5_hash(label, p)), p),
                )
            )
            for i in range(len(lst_pt))
        ]

        return {"ct0": ct0, "ct1": ct1}

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        logger.debug("generate functional DK share for - {}".format(fusion_weight))
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if self.id not in fusion_weight:
            raise FEValidationError("invalid fusion weight provided.")
        _fw_nid = fusion_weight[self.id]

        s = [gp.mpz(i) for i in self.sk["s"]]
        u = [gp.mpz(i) for i in self.sk["u"]]
        v = [gp.mpz(i) for i in self.sk["v"]]

        ordered_fw = dict(sorted(fusion_weight.items(), key=lambda x: (x[1], x[0])))
        _lst_fw = []
        for _lst in list(ordered_fw.values()):
            _lst_fw += _lst
        if len(s) != len(u) or len(v) < len(_lst_fw):
            raise FEValidationError("invalid private keys provided.")

        dk0 = sum([gp.mul(s[i], gp.mpz(_fw_nid[i])) for i in range(len(s))])
        dk1 = sum([gp.mul(u[i], gp.mpz(_fw_nid[i])) for i in range(len(u))])
        _v_fw = sum([gp.mul(v[i], gp.mpz(_lst_fw[i])) for i in range(len(_lst_fw))])

        return {"dk0": gp.digits(dk0), "dk1": gp.digits(gp.add(dk1, _v_fw))}

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        logger.debug("combine functional DK shares.")
        dk0 = {nid: dk_share["dk0"] for nid, dk_share in dct_dk_shares.items()}
        dk1 = sum([gp.mpz(dk_share["dk1"]) for _, dk_share in dct_dk_shares.items()])
        return {"dk0": dk0, "dk1": dk1}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
                label: Encryption label for replay protection.
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if dct_ct.keys() != fusion_weight.keys() or len(dct_ct) != len(dk["dk0"]):
            raise FESchemeError("inconsistent input among ct, dk, fusion wight")

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])

        _prod = lambda i, j: gp.mul(i, j)
        _lst_numerators = []
        _lst_denominator = []
        _numerator = gp.mpz(1)
        _denominator = gp.mpz(1)
        for nid, ct in dct_ct.items():
            _lst_fw = fusion_weight[nid]
            _ct1 = ct["ct1"]
            for i in range(len(_ct1)):
                _numerator *= gp.powmod(gp.mpz(_ct1[i]), gp.mpz(_lst_fw[i]), p)
            _denominator *= gp.powmod(gp.mpz(ct["ct0"]), gp.mpz(dk["dk0"][nid]), p)

        gf = gp.divm(
            _numerator,
            gp.mul(
                _denominator,
                gp.powmod(g, gp.mul(gp.mpz(dk["dk1"]), md5_hash(label, p)), p),
            ),
            p,
        )

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
        _label = kwargs.get("label", None)
        lst_ndarray_ct = list()
        for l in range(len(lst_ndarray)):
            ary = (lst_ndarray[l].copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([w], _label)
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def decrypt_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, dk: dict, fusion_weight: dict, label: str
    ) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
                label: Encryption label for replay protection.
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
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, label)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))

        return lst_ndarray

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        _fusion_weight = kwargs.get("fusion_weight", None)
        _label = kwargs.get("label", None)
        if dk is None or _fusion_weight is None or _label is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk, _fusion_weight, _label)
