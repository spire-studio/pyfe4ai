"""
R. Xu et al., "TAPFed: Threshold Secure Aggregation for Privacy-Preserving Federated Learning," 
in IEEE Transactions on Dependable and Secure Computing, doi: 10.1109/TDSC.2024.3350206.

* type:     public-key encryption
* setting:  Integer based

"""

from __future__ import annotations

import os
import json
import logging

from functools import reduce
import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import group_generator_threshold_fe
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError


logger = logging.getLogger(__name__)


class ThresholdMIFEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for threshold DDH-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_TMIFE
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.lambd = self.sec_param
        self.eta = config.get("eta", CryptoCONST.tMCFE_ETA)
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        self.lst_sid = config.get(
            "lst_sid", ["sid_{}".format(i) for i in range(self.s)]
        )
        validate_threshold(self.t, self.lst_sid)

        self._load_parameters()

        crypto_configs = os.path.join("config", "authority", "tMIFE")
        if not os.path.exists(crypto_configs):
            os.makedirs(crypto_configs)

        self.dict_dk = {}

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["group"]["p"])
        self.g = gp.mpz(param["group"]["g"])
        # p = 2q + 1 is a safe prime and g generates the order-q subgroup
        self.q = (self.p - 1) // 2

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("lambd") == self.lambd
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("s") == self.s
            and param.get("t") == self.t
        )

    def _generate_and_save(self, param_file: str) -> None:
        self.p, self.g = group_generator_threshold_fe(self.lambd)
        self.q = (self.p - 1) // 2
        param_dict = {
            "lambd": self.lambd,
            "group": {"p": gp.digits(self.p), "g": gp.digits(self.g)},
            "eta": self.eta,
            "t": self.t,
            "s": self.s,
            "n": self.n,
        }
        with open(param_file, "w") as f:
            json.dump(param_dict, f)

    def setup(self) -> None:

        alpha = [_random(self.p, self.lambd) for _ in range(self.eta)]
        g_alpha = [gp.powmod(self.g, alpha[i], self.p) for i in range(self.eta)]

        W = {nid: None for nid in self.lst_nid}
        U = {nid: None for nid in self.lst_nid}
        g_alpha_W = {nid: None for nid in self.lst_nid}
        for nid in self.lst_nid:
            w_id, u_id, g_alpha_w_id = list(), list(), list()
            for _ in range(self.eta):
                w_id.append(_random(self.p, self.lambd))
                u_id.append(_random(self.p, self.lambd))
            W[nid] = w_id
            U[nid] = u_id
            for i in range(self.eta):
                g_alpha_w_id.append(
                    [
                        gp.powmod(self.g, gp.mul(alpha[i], w_id[j]), self.p)
                        for j in range(self.eta)
                    ]
                )
            g_alpha_W[nid] = g_alpha_w_id

        self.pp = {
            "p": gp.digits(self.p),
            "g": gp.digits(self.g),
            "eta": self.eta,
            "s": self.s,
            "t": self.t,
            "n": self.n,
            "lambd": self.lambd,
        }

        self.msk = {"W": W, "U": U, "g_alpha": g_alpha, "g_alpha_W": g_alpha_W}

        logger.info("Threshold MIFE setup - DONE.")

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

    def get_decryption_keys(self, sid: str, **kwargs) -> dict:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        _credentials = kwargs.get("credentials", None)
        if not _credentials:
            raise FEKeyError("need to provided credentials for tMCFE")
        _fusion_weights = _credentials.get("fusion_weight")
        ordered_fw = dict(sorted(_fusion_weights.items(), key=lambda x: (x[1], x[0])))
        identifier = "identifier-{}".format(ordered_fw)
        if identifier not in self.dict_dk:
            self._generate_decryption_key(identifier, ordered_fw)
        dk = self.dict_dk[identifier]
        dk_sid = {"lst_sid": self.lst_sid}
        dk_sid["v0"] = gp.digits(dk["v0"][sid])
        dk_sid["v1"] = {nid: gp.digits(dk["v1"][nid][sid]) for nid in self.lst_nid}
        return dk_sid

    def _generate_decryption_key(self, identifier: str, credentials: dict) -> None:
        """Perform the _generate_decryption_key operation.

            Args:
                identifier: Decryption-key identifier string.
                credentials: Credential dict containing ``fusion_weight`` and optional ``label``.
        """
        if len(credentials) > self.n:
            raise FEValidationError("invalid size of dk generation credentials.")

        # Shamir shares live in Z_q (q = group order) at points 1..s; the
        # secret is the polynomial value at 0
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())

        sum_uy = gp.mpz(0)
        for nid in self.lst_nid:
            cred = credentials[nid]
            u_i = self.msk["U"][nid]
            sum_uy += sum([gp.mul(cred[k], u_i[k]) for k in range(len(cred))])

        a = share_secret(sum_uy, self.t, points, self.q)

        v0 = {sid: a[sid_v[sid]] for sid in self.lst_sid}

        v1 = {}
        for nid in self.lst_nid:
            cred = credentials[nid]
            w_i = self.msk["W"][nid]
            b_i_0 = share_secret(
                sum([gp.mul(cred[k], w_i[k]) for k in range(len(cred))]),
                self.t,
                points,
                self.q,
            )
            v1[nid] = {sid: b_i_0[sid_v[sid]] for sid in self.lst_sid}

        self.dict_dk[identifier] = {"v0": v0, "v1": v1}


class ThresholdMIFE(IPFEAbsCrypto):
    """Crypto operations for threshold DDH-based multi-input inner-product FE."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pp = self.keys["pp"]
        if "sk" in self.keys:
            self.sk = self.keys["sk"]
            self._load_dlog_table()
        logger.info("initialize successfully.")

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_TMIFE,
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"],
                pow(10, self.precision + 2),
            )
        )

    def encrypt(self, pt: list) -> dict:
        if len(pt) > len(self.sk["u"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(pt))
        if not isinstance(pt, list):
            raise FEValidationError("invalid format of input plaintext:{}".format(pt))

        lst_pt = [int(round(v * pow(10, self.precision))) for v in pt]

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        lambd = self.pp["lambd"]
        u = [gp.mpz(u_i) for u_i in self.sk["u"]]

        r = _random(p, lambd)
        g_v_u = [
            gp.powmod(g, (gp.mpz(lst_pt[k]) + u[k]), p) for k in range(len(lst_pt))
        ]
        g_alpha_w_r = [
            [gp.powmod(gp.mpz(gaw_i), r, p) for gaw_i in r_lst]
            for r_lst in self.sk["g_alpha_w"]
        ]
        g_alpha_r = [gp.powmod(gp.mpz(ga_i), r, p) for ga_i in self.sk["g_alpha"]]

        ct0 = [
            [gp.digits(gp.mul(r_lst[k], g_v_u[k]) % p) for k in range(len(r_lst))]
            for r_lst in g_alpha_w_r
        ]
        ct1 = gp.digits(reduce(lambda x, y: gp.mul(x, y) % p, g_alpha_r) % p)

        return {"ct0": ct0, "ct1": ct1}

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
        q = (p - 1) // 2
        lst_sid = dk["lst_sid"]
        if self.id not in lst_sid:
            raise FESchemeError("local id:{} is not supported".format(self.id))
        validate_enrolled(self.id, lst_sid, lst_sid_enrolled, self.pp["t"])

        sid_v = share_points(lst_sid)
        lagrange = self.L(self.id, sid_v, lst_sid_enrolled, q)
        gp_prod = lambda i, j: gp.mul(i, j) % p

        ct0_prime = list()
        ct1_prime = list()
        for nid in dict_ct.keys():
            ct0_nid_lst = dict_ct[nid]["ct0"]
            ct0_nid_y_lst = [
                [
                    gp.powmod(gp.mpz(r_lst[k]), credentials[nid][k], p)
                    for k in range(len(r_lst))
                ]
                for r_lst in ct0_nid_lst
            ]
            ct0_prime.append(
                reduce(gp_prod, [reduce(gp_prod, r_lst) for r_lst in ct0_nid_y_lst])
            )

            ct1_nid = gp.mpz(dict_ct[nid]["ct1"])
            ct1_prime.append(
                gp.digits(
                    gp.powmod(
                        ct1_nid, gp.mul(gp.mpz(dk["v1"][nid]), lagrange) % q, p
                    )
                )
            )

        ct0_prime = gp.digits(reduce(gp_prod, ct0_prime) % p)
        ct2_prime = gp.digits(
            gp.powmod(g, gp.mul(gp.mpz(dk["v0"]), lagrange) % q, p)
        )

        return {
            "ct0_prime": ct0_prime,
            "ct1_prime": ct1_prime,
            "ct2_prime": ct2_prime,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

    @staticmethod
    def L(id: str, dct_id_v: dict, lst_sid_enrolled: list, modulus) -> gp.mpz:
        """Compute the Lagrange coefficient of *id* at 0 modulo the group order.

            Args:
                id: Decryption-server identifier.
                dct_id_v: Mapping of server ids to their (1-based) share points.
                lst_sid_enrolled: List of enrolled server identifiers.
                modulus: Group order ``q``.
        """
        return lagrange_coefficient_at_zero(
            dct_id_v[id], [dct_id_v[s] for s in lst_sid_enrolled], modulus
        )

    def combine_decrypt(self, dict_ct_prime: dict) -> float | None:
        """Combine at least ``t`` partial decryptions into the inner product.

            Args:
                dict_ct_prime: Mapping of server ids to :meth:`share_decrypt` outputs.

            Raises:
                FESchemeError: If fewer than ``t`` shares are given or the shares
                    do not match the enrolled set they were computed for.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        p = gp.mpz(self.pp["p"])
        q = (p - 1) // 2
        eta = self.pp["eta"]

        ct_prime = dict()
        ct_prime["ct0_prime"] = list()
        ct_prime["ct1_prime"] = list()
        ct_prime["ct2_prime"] = list()
        for sid in dict_ct_prime.keys():
            ct_prime_sid = dict_ct_prime[sid]
            ct_prime["ct0_prime"].append(gp.mpz(ct_prime_sid["ct0_prime"]))
            for ct1_prime in ct_prime_sid["ct1_prime"]:
                ct_prime["ct1_prime"].append(gp.mpz(ct1_prime))
            ct_prime["ct2_prime"].append(gp.mpz(ct_prime_sid["ct2_prime"]))

        gp_prod = lambda i, j: gp.mul(i, j) % p

        # verification
        ct_const = ct_prime["ct0_prime"][0]
        for i in range(1, len(ct_prime["ct0_prime"])):
            if ct_const != ct_prime["ct0_prime"][i]:
                logger.warning("the ct consistency verification failed.")
                return None

        prod_ct1_prime = reduce(gp_prod, ct_prime["ct1_prime"])
        prod_ct2_prime = reduce(gp_prod, ct_prime["ct2_prime"])

        # each of the eta rows of ct0 carries one copy of g^{<x,y> + <u,y>},
        # so ct0' / (ct1' * ct2'^eta) = g^{eta * <x,y>}; strip eta in the exponent
        g_eta_f = gp.divm(
            ct_const,
            gp.mul(prod_ct1_prime, gp.powmod(prod_ct2_prime, eta, p)) % p,
            p,
        )
        g_f = gp.powmod(g_eta_f, gp.invert(eta, q), p)
        f = self._solve_dlog(gp.digits(g_f))
        if f is None:
            return None
        return float(f / pow(10, self.precision))

    def encrypt_lst_ndarray(self, lst_ndarray: list) -> list:
        lst_ndarray_ct = list()
        for l in range(len(lst_ndarray)):
            ary = lst_ndarray[l]
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([w])
            lst_ndarray_ct.append(ary_ct)
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
            ary = sample_lst_ndarray[l]
            ary_ct_sd = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                w_dict_ct = {nid: dict_ct[nid][l][i] for nid in dict_ct.keys()}
                w_ct_sd = self.share_decrypt(
                    w_dict_ct, credentials, dk, lst_sid_enrolled
                )
                ary_ct_sd[i] = w_ct_sd
            lst_ndarray_ct_sd.append(ary_ct_sd)
        return lst_ndarray_ct_sd

    def _combine_decrypt_lst_ndarray(self, dict_ct_prime: dict) -> list:
        sample_lst_ndarray = next(iter(dict_ct_prime.values()))
        lst_ndarray_dec = list()
        for l in range(len(sample_lst_ndarray)):
            ary = sample_lst_ndarray[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                w_dict_ct_prime = {
                    sid: dict_ct_prime[sid][l][i] for sid in dict_ct_prime.keys()
                }
                ary_dec[i] = self.combine_decrypt(w_dict_ct_prime)
            lst_ndarray_dec.append(ary_dec)
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
