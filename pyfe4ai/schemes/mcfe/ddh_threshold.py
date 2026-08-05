"""
R. Xu et al., "TAPFed: Threshold Secure Aggregation for Privacy-Preserving Federated Learning," 
in IEEE Transactions on Dependable and Secure Computing, doi: 10.1109/TDSC.2024.3350206.

* type:     public-key encryption
* setting:  Integer based

"""

from __future__ import annotations

import os
import json
import random
import logging

_CSPRNG = random.SystemRandom()

import gmpy2 as gp
import numpy as np


from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import group_generator_threshold_fe
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)


class ThresholdMCFEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for threshold DDH-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_TMCFE
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.eta = config.get("eta", CryptoCONST.tMCFE_ETA)
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        self.lst_sid = config.get(
            "lst_sid", ["sid_{}".format(i) for i in range(self.s)]
        )
        self._load_parameters()
        self.dict_dk = {}

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["group"]["p"])
        self.g = gp.mpz(param["group"]["g"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("s") == self.s
            and param.get("t") == self.t
        )

    def _generate_and_save(self, param_file: str) -> None:
        self.p, self.g = group_generator_threshold_fe(self.sec_param)
        _param = {
            "sec_param": self.sec_param,
            "group": {"p": gp.digits(self.p), "g": gp.digits(self.g)},
            "eta": self.eta,
            "t": self.t,
            "s": self.s,
            "n": self.n,
        }
        with open(param_file, "w") as f:
            json.dump(_param, f)

    def setup(self) -> None:
        alpha = [_random(self.p, self.sec_param) for _ in range(self.eta)]
        g_alpha = [gp.powmod(self.g, alpha[i], self.p) for i in range(self.eta)]

        W = {nid: None for nid in self.lst_nid}
        U = {nid: None for nid in self.lst_nid}
        g_alpha_W = {nid: None for nid in self.lst_nid}
        for nid in self.lst_nid:
            w_id, u_id, g_alpha_w_id = list(), list(), list()
            for i in range(self.eta):
                w_id.append(_random(self.p, self.sec_param))
                u_id.append(_random(self.p, self.sec_param))
                _alpha_i = alpha[i]
                g_alpha_w_id.append(
                    [
                        gp.powmod(self.g, gp.mul(_alpha_i, w_id[j]), self.p)
                        for j in range(self.eta)
                    ]
                )
            W[nid] = w_id
            U[nid] = u_id
            g_alpha_W[nid] = g_alpha_w_id

        self.pp = {
            "p": gp.digits(self.p),
            "g": gp.digits(self.g),
            "eta": self.eta,
            "s": self.s,
            "t": self.t,
            "n": self.n,
            "sec_param": self.sec_param,
        }

        self.msk = {"W": W, "U": U, "g_alpha": g_alpha, "g_alpha_W": g_alpha_W}

        logger.info("Threshold MCFE setup - DONE.")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str) -> dict | None:
        if nid is None:
            logger.error("id of encryption entity is not found.")
            return None
        if nid not in self.lst_nid:
            logger.error("invalid nid format: {}".format(nid))
            return None

        sk = {}
        sk["g_alpha"] = [gp.digits(e) for e in self.msk["g_alpha"]]
        sk["g_alpha_w"] = [
            [gp.digits(e) for e in row_lst] for row_lst in self.msk["g_alpha_W"][nid]
        ]
        sk["u"] = [gp.digits(e) for e in self.msk["U"][nid]]
        return sk

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        _credentials = kwargs.get("credentials", None)
        if not _credentials:
            raise FEKeyError("need to provided credentials for tMCFE")
        _fusion_weights = _credentials.get("fusion_weight")
        _label = _credentials.get("label")

        ordered_fusion_weights = dict(
            sorted(_fusion_weights.items(), key=lambda x: (x[1], x[0]))
        )
        identifier = "{}-{}".format(_label, ordered_fusion_weights)
        if identifier not in self.dict_dk:
            self._generate_decryption_key(identifier, ordered_fusion_weights, _label)
        dk = self.dict_dk[identifier]
        try:
            dk_sid = {"lst_sid": self.lst_sid}
            dk_sid["v0"] = dk["v0"][sid]
            dk_sid["v1"] = {
                nid: dk["v1"][nid][sid] for nid in ordered_fusion_weights.keys()
            }
            return dk_sid
        except Exception as ex:
            logger.error("invalid sid: {} - errors: {}".format(sid, ex))
            return None

    def _generate_decryption_key(
        self, identifier: str, credentials: dict, label: str
    ) -> None:
        # TODO: here, only consider a simple case: each enc entity use
        # max size of allowed input (i.e., `eta`)
        # it is possible to support various size of enc input for different
        # enc entity (less than `eta`)
        """Perform the _generate_decryption_key operation.

            Args:
                identifier: Decryption-key identifier string.
                credentials: Credential dict containing ``fusion_weight`` and optional ``label``.
                label: Encryption label for replay protection.
        """
        logger.debug("generate new dk for: {}".format(identifier))
        if len(credentials) > self.n:
            raise FEValidationError("invalid size of dk generation credentials.")

        f = lambda a, u, m: sum([a[k] * (u**k) for k in range(m)])
        sid_v = {v: k for k, v in enumerate(self.lst_sid)}

        v1 = {}
        _uy = gp.mpz(0)
        for nid, credential in credentials.items():
            if (
                len(credential) != self.eta
                or not isinstance(credential, list)
                or not isinstance(credential[0], int)
            ):
                raise FEValidationError("invalid content of dk generation credentials.")

            u_nid = self.msk["U"][nid]
            w_nid = self.msk["W"][nid]
            _uy_nid = gp.mpz(0)
            _wy_nid = gp.mpz(0)
            for i in range(len(credential)):
                _uy_nid += gp.mul(credential[i], u_nid[i])
                _wy_nid += gp.mul(credential[i], w_nid[i])
            _uy += _uy_nid
            b_nid = [_wy_nid] + [_CSPRNG.randint(1, self.p) for _ in range(1, self.t)]
            v1[nid] = {
                sid: gp.digits(f(b_nid, sid_v[sid], self.t)) for sid in self.lst_sid
            }

        a = [gp.mul(_uy, md5_hash(label, self.p))] + [
            _CSPRNG.randint(1, self.p) for _ in range(1, self.t)
        ]
        v0 = {sid: gp.digits(f(a, sid_v[sid], self.t)) for sid in self.lst_sid}

        self.dict_dk[identifier] = {"v0": v0, "v1": v1}


class ThresholdMCFE(IPFEAbsCrypto):
    """Crypto operations for threshold DDH-based multi-client inner-product FE."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
            self._load_dlog_table()
        logger.info("initialize successfully.")

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_TMCFE,
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"],
                pow(10, self.precision + 2),
            )
        )

    def encrypt(self, pt: list, label: str) -> dict:
        """Encrypt plaintext and return a ciphertext dict.

            Args:
                pt: Integer plaintext vector.
                label: Encryption label for replay protection.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(pt) > len(self.sk["u"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(pt))
        if not isinstance(pt, list):
            raise FEValidationError("invalid format of input plaintext:{}".format(pt))

        lst_pt = [int(round(v * pow(10, self.precision))) for v in pt]

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        sec_param = self.pp["sec_param"]
        u = [gp.mpz(u_i) for u_i in self.sk["u"]]

        r = _random(p, sec_param)

        ct0 = []
        for gaw_lst in self.sk["g_alpha_w"]:
            _ct_lst = []
            for i in range(len(lst_pt)):
                _ct_lst.append(
                    gp.digits(
                        gp.mul(
                            gp.powmod(gp.mpz(gaw_lst[i]), r, p),
                            gp.powmod(
                                g,
                                (gp.mpz(lst_pt[i]) + gp.mul(u[i], md5_hash(label, p))),
                                p,
                            ),
                        )
                        % p
                    )
                )
            ct0.append(_ct_lst)

        ct1 = gp.mpz(1)
        for _ga in self.sk["g_alpha"]:
            ct1 *= gp.powmod(gp.mpz(_ga), r, p)

        return {"ct0": ct0, "ct1": gp.digits(ct1 % p)}

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> dict:
        """Compute a partial decryption share.

            Args:
                dict_ct: Dict of per-client ciphertexts.
                credentials: Credential dict containing ``fusion_weight`` and optional ``label``.
                dk: Functional decryption key.
                lst_sid_enrolled: List of enrolled session / node identifiers.
        """
        if not dk:
            raise FEKeyError("no decryption keys provided.")
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        lst_sid = dk["lst_sid"]
        if self.id not in lst_sid:
            raise FESchemeError("local id:{} is not supported".format(self.id))

        sid_v = {v: k for k, v in enumerate(lst_sid)}

        ct0_prime = gp.mpz(1)
        lst_ct1_prime = list()
        for nid in dict_ct.keys():
            lst_ct0_nid = dict_ct[nid]["ct0"]
            _ct0_nid = gp.mpz(1)
            for r_lst in lst_ct0_nid:
                for k in range(len(r_lst)):
                    _ct0_nid *= gp.powmod(gp.mpz(r_lst[k]), credentials[nid][k], p)
            ct0_prime *= _ct0_nid % p

            ct1_nid = gp.mpz(dict_ct[nid]["ct1"])
            lst_ct1_prime.append(
                gp.digits(
                    gp.powmod(
                        ct1_nid,
                        gp.mul(
                            gp.mpz(dk["v1"][nid]),
                            self.L(self.id, sid_v, lst_sid_enrolled),
                        ),
                        p,
                    )
                )
            )

        ct0_prime = gp.digits(ct0_prime % p)
        ct2_prime = gp.digits(
            gp.powmod(
                g, gp.mul(gp.mpz(dk["v0"]), self.L(self.id, sid_v, lst_sid_enrolled)), p
            )
        )

        return {
            "ct0_prime": ct0_prime,
            "ct1_prime": lst_ct1_prime,
            "ct2_prime": ct2_prime,
        }

    def combine_decrypt(self, dict_ct_prime: dict) -> float:
        p = gp.mpz(self.pp["p"])
        eta = self.pp["eta"]

        lst_ct0_prime = []
        prod_ct1_prime = gp.mpz(1)
        prod_ct2_prime = gp.mpz(1)
        for _, ct_prime_sid in dict_ct_prime.items():
            lst_ct0_prime.append(gp.mpz(ct_prime_sid["ct0_prime"]))
            for ct1_prime in ct_prime_sid["ct1_prime"]:
                prod_ct1_prime *= gp.mpz(ct1_prime)
            prod_ct2_prime *= gp.mpz(ct_prime_sid["ct2_prime"])

        # verification
        ct_const = lst_ct0_prime[0]
        for i in range(1, len(lst_ct0_prime)):
            if ct_const != lst_ct0_prime[i]:
                logger.warning("ct consistency verification failed.")
                return None

        if eta == 1:
            mul_ct_prime = gp.mul(prod_ct1_prime, prod_ct2_prime)
            g_f = gp.divm(ct_const, mul_ct_prime, p)
            # g_f = gp.f_div(ct_const, gp.mul(prod_ct1_prime, prod_ct2_prime)) % p
            f = self._solve_dlog(gp.digits(g_f))
        else:
            g_f = gp.divm(
                ct_const,
                (gp.mul(prod_ct1_prime, gp.powmod(prod_ct2_prime, 2, p)) % p),
                p,
            )
            f = self._solve_dlog(gp.digits(g_f))
            if f is not None:
                f = f / 2
        f = float(f / pow(10, self.precision))
        return f

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        _label = kwargs.get("label", None)
        lst_ndarray_ct = list()
        for l in range(len(lst_ndarray)):
            _ary = lst_ndarray[l]
            _ary_ct = np.empty(_ary.shape, dtype=object)
            for i, w in np.ndenumerate(_ary):
                _ary_ct[i] = self.encrypt([w], _label)
            lst_ndarray_ct.append(_ary_ct)
        return lst_ndarray_ct

    def _share_decrypt_lst_ndarray(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Perform the _share_decrypt_lst_ndarray operation.

            Args:
                dict_ct: Dict of per-client ciphertexts.
                credentials: Credential dict containing ``fusion_weight`` and optional ``label``.
                dk: Functional decryption key.
                lst_sid_enrolled: List of enrolled session / node identifiers.
        """
        sample_lst_ndarray = next(iter(dict_ct.values()))
        lst_ndarray_ct_sd = list()
        for l in range(len(sample_lst_ndarray)):
            _ary = sample_lst_ndarray[l]
            _ary_ct_sd = np.empty(_ary.shape, dtype=object)
            for i, _ in np.ndenumerate(_ary):
                # extract and share decrypt
                w_dict_ct = {nid: dict_ct[nid][l][i] for nid in dict_ct.keys()}
                w_ct_sd = self.share_decrypt(
                    w_dict_ct, credentials, dk, lst_sid_enrolled
                )
                _ary_ct_sd[i] = w_ct_sd
            lst_ndarray_ct_sd.append(_ary_ct_sd)
        return lst_ndarray_ct_sd

    def _combine_decrypt_lst_ndarray(self, dict_ct_prime: dict) -> list:
        sample_lst_ndarray = next(iter(dict_ct_prime.values()))
        lst_ndarray_dec = list()
        for l in range(len(sample_lst_ndarray)):
            _ary = sample_lst_ndarray[l]
            _ary_dec = np.empty(_ary.shape, dtype=object)
            for i, _ in np.ndenumerate(_ary):
                # extract and combine decrypt
                w_dict_ct_prime = {
                    sid: dict_ct_prime[sid][l][i] for sid in dict_ct_prime.keys()
                }
                _ary_dec[i] = self.combine_decrypt(w_dict_ct_prime)
            lst_ndarray_dec.append(_ary_dec)
        return lst_ndarray_dec

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
                credentials: Credential dict containing ``fusion_weight`` and optional ``label``.
                dk: Functional decryption key.
                lst_sid_enrolled: List of enrolled session / node identifiers.
        """
        return self._share_decrypt_lst_ndarray(
            dict_ndarray_ct, credentials, dk, lst_sid_enrolled
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict) -> list:
        return self._combine_decrypt_lst_ndarray(dict_ndarray_ct)

    def _solve_dlog(self, g_inner_prod) -> int | None:
        try:
            return dlog_table_solve(
                g_inner_prod, self.pp["g"], self.pp["p"],
                self.bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system.")
            return None

    @staticmethod
    def L(sid, sid_v: dict, lst_sid_enrolled: list) -> int:
        """Compute the Lagrange interpolation coefficient.

            Args:
                sid: Session / decryption-key identifier.
                sid_v: Set of enrolled node identifiers.
                lst_sid_enrolled: List of enrolled session / node identifiers.
        """
        prod = 1
        for sid_prime in lst_sid_enrolled:
            if sid_v[sid] != sid_v[sid_prime]:
                prod *= sid_v[sid_prime] / (sid_v[sid_prime] - sid_v[sid])
        return int(prod)
