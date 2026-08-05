"""
Threshold LWE-Based Multi-Input Inner-Product Functional Encryption
| Threshold secure aggregation organization inspired by
| "TAPFed: Threshold Secure Aggregation for Privacy-Preserving Federated Learning"
| Published in: IEEE TDSC
|
| This prototype combines threshold decryption-key sharing with the
| repository's LWE-based MIFE line.

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

from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import decode_lwe_inner_product
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matvec_mod
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)


def _eval_poly(coeffs: list[gp.mpz], x_value: int, modulus: gp.mpz) -> gp.mpz:
    """Perform the _eval_poly operation.

        Args:
            coeffs: Polynomial coefficient list.
            x_value: Evaluation point.
            modulus: Modulus for the arithmetic operation.
    """
    acc = gp.mpz(0)
    x = gp.mpz(x_value)
    power = gp.mpz(1)
    for coeff in coeffs:
        acc = (acc + coeff * power) % modulus
        power = (power * x) % modulus
    return acc


def _share_secret(
    secret: gp.mpz, threshold: int, share_points: list[int], modulus: gp.mpz
) -> dict[int, gp.mpz]:
    """Perform the _share_secret operation.

        Args:
            secret: The secret to share.
            threshold: Minimum number of shares needed for reconstruction.
            share_points: Evaluation points for the shares.
            modulus: Modulus for the arithmetic operation.
    """
    coeffs = [gp.mpz(secret) % modulus] + [
        gp.mpz(_CSPRNG.randrange(0, int(modulus)))
        for _ in range(1, threshold)
    ]
    return {x: _eval_poly(coeffs, x, modulus) for x in share_points}


def _lagrange_at_zero(shares: dict[int, gp.mpz], modulus: gp.mpz) -> gp.mpz:
    """Perform the _lagrange_at_zero operation.

        Args:
            shares: List of ``(x, y)`` share pairs.
            modulus: Modulus for the arithmetic operation.
    """
    result = gp.mpz(0)
    xs = list(shares.keys())
    for x_j in xs:
        num = gp.mpz(1)
        den = gp.mpz(1)
        for x_m in xs:
            if x_m == x_j:
                continue
            num = (num * gp.mpz(x_m)) % modulus
            den = (den * gp.mpz(x_m - x_j)) % modulus
        lagrange = (num * gp.invert(den % modulus, modulus)) % modulus
        result = (result + gp.mpz(shares[x_j]) * lagrange) % modulus
    return result


class ThresholdMIFELWEKeyGenerator(MIFELWEKeyGenerator):
    """Key generator for threshold LWE-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_TMIFE_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        self.t = config.get("t", CryptoCONST.tMIFE_T)
        self.lst_sid = config.get(
            "lst_sid", ["sid_{}".format(i) for i in range(config.get("s", 1))]
        )
        self.dict_dk = {}
        super().__init__(config, **kwargs)

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
            and param.get("s") == self.s
            and param.get("t") == self.t
            and param.get("lst_sid") == self.lst_sid
            and param.get("lwe_n") == self.lwe_n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
        )

    def _generate_and_save(self, param_file: str) -> None:
        super()._generate_and_save(param_file)
        with open(param_file, "r", encoding="utf-8") as f:
            param = json.load(f)
        param["s"] = self.s
        param["t"] = self.t
        param["lst_sid"] = self.lst_sid
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(param, f)

    def get_public_parameters(self) -> dict:
        pp = super().get_public_parameters()
        pp["s"] = self.s
        pp["t"] = self.t
        pp["lst_sid"] = self.lst_sid
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for Threshold MIFE LWE")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        ordered_fusion_weights = dict(sorted(fusion_weight.items(), key=lambda x: x[0]))
        identifier = "{}".format(ordered_fusion_weights)
        if identifier not in self.dict_dk:
            self._generate_threshold_decryption_keys(identifier, ordered_fusion_weights)
        return self.dict_dk[identifier].get(sid)

    def _generate_threshold_decryption_keys(
        self, identifier: str, fusion_weight: dict
    ) -> None:
        """Perform the _generate_threshold_decryption_keys operation.

            Args:
                identifier: Decryption-key identifier string.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        q = gp.mpz(self.q)
        sid_index = {sid: idx + 1 for idx, sid in enumerate(self.lst_sid)}

        full_sk_y = {}
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            y_vec = [gp.mpz(v) for v in y]
            full_sk_y[nid] = matvec_mod(self.msk["sk"][nid], y_vec, q)

        per_sid = {
            sid: {"lst_sid": self.lst_sid, "sid_index": sid_index[sid], "sk_y": {}}
            for sid in self.lst_sid
        }

        for nid in self.lst_nid:
            component_shares = [
                _share_secret(secret, self.t, list(sid_index.values()), q)
                for secret in full_sk_y[nid]
            ]
            for sid, x_value in sid_index.items():
                per_sid[sid]["sk_y"][nid] = [
                    gp.digits(component_shares[i][x_value])
                    for i in range(len(component_shares))
                ]

        self.dict_dk[identifier] = per_sid


class ThresholdMIFELWE(MIFELWE):
    """Crypto operations for threshold LWE-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)

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
            raise FEKeyError("no decryption keys provided")
        if self.id not in dk["lst_sid"]:
            raise FESchemeError("local id:{} is not supported".format(self.id))

        q = gp.mpz(self.pp["q"])
        ct0_prime = {}
        for nid, ct in dict_ct.items():
            y = [gp.mpz(v) for v in credentials[nid]]
            ct0 = digits_to_vector(ct["ct0"])
            ct1 = digits_to_vector(ct["ct1"])
            sk_y_share = [gp.mpz(v) for v in dk["sk_y"][nid]]
            d_share = (vecdot_mod(y, ct1, q) - vecdot_mod(ct0, sk_y_share, q)) % q
            ct0_prime[nid] = gp.digits(d_share)

        return {"ct0_prime": ct0_prime, "sid_index": dk["sid_index"]}

    def combine_decrypt(self, dict_ct_prime: dict) -> float:
        if len(dict_ct_prime) < self.pp["t"]:
            raise FESchemeError("insufficient threshold shares for decryption")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        total = 0
        sample = next(iter(dict_ct_prime.values()))
        for nid in sample["ct0_prime"]:
            shares = {
                int(ct_prime["sid_index"]): gp.mpz(ct_prime["ct0_prime"][nid])
                for ct_prime in dict_ct_prime.values()
            }
            d_value = _lagrange_at_zero(shares, q)
            total += decode_lwe_inner_product(d_value, p, q)
        return float(total)

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
        lst_ndarray_ct_sd = []
        for l in range(len(sample_lst_ndarray)):
            ary = sample_lst_ndarray[l]
            ary_ct_sd = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                w_dict_ct = {nid: dict_ct[nid][l][i] for nid in dict_ct.keys()}
                ary_ct_sd[i] = self.share_decrypt(
                    w_dict_ct, credentials, dk, lst_sid_enrolled
                )
            lst_ndarray_ct_sd.append(ary_ct_sd)
        return lst_ndarray_ct_sd

    def _combine_decrypt_lst_ndarray(self, dict_ct_prime: dict) -> list:
        sample_lst_ndarray = next(iter(dict_ct_prime.values()))
        lst_ndarray_dec = []
        for l in range(len(sample_lst_ndarray)):
            ary = sample_lst_ndarray[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                w_dict_ct_prime = {
                    sid: dict_ct_prime[sid][l][i] for sid in dict_ct_prime.keys()
                }
                ary_dec[i] = self.combine_decrypt(w_dict_ct_prime)
            lst_ndarray_dec.append((ary_dec / pow(10, self.precision)).astype(float))
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
