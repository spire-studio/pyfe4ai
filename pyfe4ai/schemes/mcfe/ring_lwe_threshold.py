"""
Threshold Ring-LWE-Based Multi-Client Inner-Product Functional Encryption
| Threshold secure aggregation organization inspired by
| "TAPFed: Threshold Secure Aggregation for Privacy-Preserving Federated Learning"
| Published in: IEEE TDSC
|
| This prototype combines threshold decryption-key sharing with the Ring-LWE
| multi-client inner-product FE line used in this repository.

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

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.utils.ring_lwe_utils import decode_vector
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.lwe_utils import label_scalar_from_hash
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


def _center_mod(value: gp.mpz, modulus: gp.mpz) -> gp.mpz:
    """Perform the _center_mod operation.

        Args:
            value: Input value.
            modulus: Modulus for the arithmetic operation.
    """
    value %= modulus
    half = modulus // 2
    if value > half:
        value -= modulus
    return value


class ThresholdMCFERingLWEKeyGenerator(MCFERingLWEKeyGenerator):
    """Key generator for threshold Ring-LWE-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_TMCFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        self.lst_sid = config.get(
            "lst_sid", ["sid_{}".format(i) for i in range(config.get("s", 1))]
        )
        self.dict_dk = {}
        super().__init__(config, **kwargs)

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
            super()._param_verification(param)
            and param.get("s") == self.s
            and param.get("t") == self.t
            and param.get("lst_sid") == self.lst_sid
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
            raise FEKeyError("need credentials for Threshold MCFE Ring-LWE")
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

        z = gp.mpz(0)
        full_sk_y = {}
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            y_vec = [gp.mpz(v) for v in fusion_weight[nid]]
            sk_t = np.array(self.msk["sk"][nid], dtype=object).T.tolist()
            full_sk_y[nid] = []
            for row in sk_t:
                acc = gp.mpz(0)
                for a_i, b_i in zip(row, y_vec):
                    acc += a_i * b_i
                full_sk_y[nid].append(acc % q)
            for u_i, y_i in zip(self.msk["u"][nid], y_vec):
                z += u_i * y_i
        z %= q

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

        z_shares = _share_secret(z, self.t, list(sid_index.values()), q)
        for sid, x_value in sid_index.items():
            per_sid[sid]["z"] = gp.digits(z_shares[x_value])

        self.dict_dk[identifier] = per_sid


class ThresholdMCFERingLWE(MCFERingLWE):
    """Crypto operations for threshold Ring-LWE-based multi-client inner-product FE."""

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

        fusion_weight = credentials.get("fusion_weight")
        label = credentials.get("label")
        if fusion_weight is None or label is None:
            raise FEKeyError("need fusion_weight and label for threshold share decryption")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        label_scalar = label_scalar_from_hash(md5_hash(label, p), self.pp["label_modulus"])

        ct0_prime = {}
        for nid, ct in dict_ct.items():
            y = [gp.mpz(v) for v in fusion_weight[nid]]
            ct0 = [[gp.mpz(v) for v in row] for row in ct["ct0"]]
            ct1 = [gp.mpz(v) for v in ct["ct1"]]
            sk_y_share = [gp.mpz(v) for v in dk["sk_y"][nid]]

            ct0_t = np.array(ct0, dtype=object).T.tolist()
            lhs = []
            for row in ct0_t:
                acc = gp.mpz(0)
                for a_i, b_i in zip(row, y):
                    acc += a_i * b_i
                lhs.append(acc % q)
            rhs = []
            for i in range(len(ct1)):
                acc = gp.mpz(0)
                for j in range(len(ct1)):
                    idx = i + j
                    coeff = ct1[i] * sk_y_share[j]
                    if idx >= len(ct1):
                        idx -= len(ct1)
                        coeff = -coeff
                    while len(rhs) <= idx:
                        rhs.append(gp.mpz(0))
                    rhs[idx] = (rhs[idx] + coeff) % q
            if len(rhs) < len(ct1):
                rhs.extend([gp.mpz(0) for _ in range(len(ct1) - len(rhs))])
            d_share = [((lhs[i] - rhs[i]) % q) for i in range(len(ct1))]
            ct0_prime[nid] = [gp.digits(v) for v in d_share]

        z_share = (gp.mpz(dk["z"]) * label_scalar) % q
        return {
            "ct0_prime": ct0_prime,
            "ct1_prime": gp.digits(z_share),
            "sid_index": dk["sid_index"],
            "k": next(iter(dict_ct.values()))["k"],
        }

    def combine_decrypt(self, dict_ct_prime: dict) -> list[int]:
        if len(dict_ct_prime) < self.pp["t"]:
            raise FESchemeError("insufficient threshold shares for decryption")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        total = None
        sample = next(iter(dict_ct_prime.values()))
        for nid in sample["ct0_prime"]:
            d_poly = []
            for idx in range(len(sample["ct0_prime"][nid])):
                shares = {
                    int(ct_prime["sid_index"]): gp.mpz(ct_prime["ct0_prime"][nid][idx])
                    for ct_prime in dict_ct_prime.values()
                }
                d_poly.append(_lagrange_at_zero(shares, q))
            dec = decode_vector(d_poly, p, q, int(sample["k"]))
            if total is None:
                total = dec
            else:
                total = [a + b for a, b in zip(total, dec)]

        z_shares = {
            int(ct_prime["sid_index"]): gp.mpz(ct_prime["ct1_prime"])
            for ct_prime in dict_ct_prime.values()
        }
        z_value = int(_center_mod(_lagrange_at_zero(z_shares, q), q))
        return [v - z_value for v in total]

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
                ary_dec[i] = self.combine_decrypt(w_dict_ct_prime)[0]
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
