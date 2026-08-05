"""
Function-Hiding Inner-Product Encryption over Pairings
| Based on
| "Function-Hiding Inner Product Encryption is Practical"
| By Kim, Lewi, Mandal, Montgomery, Roy, Wu
|
| This prototype follows the GoFE `innerprod/fullysec/fhipe.go` structure and
| adapts it to the repository's single-input FE API.

* type:     public-key encryption
* setting:  Pairing based
* note:     research prototype aligned with the existing crypto-ipfe API

"""

from __future__ import annotations

import json
import logging
import math
import os

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError
from pyfe4ai.utils.pairing_utils import (
    bounded_discrete_log_gt,
    mat_vec_mod,
    matrix_inverse_mod,
    random_nonzero_zr,
    to_zr,
    transpose_mod,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases (private names used by internal callers)
_transpose_mod = transpose_mod
_mat_vec_mod = mat_vec_mod
_random_nonzero_zr = random_nonzero_zr
_to_zr = to_zr
_matrix_inverse_mod = matrix_inverse_mod
_bounded_discrete_log_gt = bounded_discrete_log_gt


class SIFEFHIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for function-hiding inner-product encryption over pairings."""

    _scheme_type = CryptoCONST.TYPE_SIFE_FH_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 16))
        self.bound_y = gp.mpz(config.get("bound_y", 16))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        self.g1 = self.group.random(G1)
        self.g2 = self.group.random(G2)

        while True:
            b = [
                [gp.mpz(int(self.group.random(ZR))) for _ in range(self.eta)]
                for _ in range(self.eta)
            ]
            try:
                b_inv, det = _matrix_inverse_mod(b, self.order)
                break
            except ValueError:
                continue

        b_star = _transpose_mod(b_inv)
        b_star = [
            [(det * entry) % self.order for entry in row]
            for row in b_star
        ]

        self.mpk = {}
        self.msk = {
            "g1": self.group.serialize(self.g1).decode("utf-8"),
            "g2": self.group.serialize(self.g2).decode("utf-8"),
            "b": [[gp.digits(v) for v in row] for row in b],
            "b_star": [[gp.digits(v) for v in row] for row in b_star],
            "det": gp.digits(det),
        }
        logger.info("SIFE FH-IPE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "eta": self.eta,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "pairing_group_param": self.pairing_group_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        return {
            "g2": self.msk["g2"],
            "b_star": self.msk["b_star"],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for FH-IPE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weight) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")
        if any(abs(int(v)) > self.bound_y for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        y = [gp.mpz(v) for v in fusion_weight]
        b = [[gp.mpz(v) for v in row] for row in self.msk["b"]]
        det = gp.mpz(self.msk["det"])
        alpha = _random_nonzero_zr(self.group)
        alpha_int = gp.mpz(int(alpha))

        by = _mat_vec_mod(b, y, self.order)
        by = [(alpha_int * entry) % self.order for entry in by]

        g1 = self.group.deserialize(self.msk["g1"].encode("utf-8"))
        k1 = g1 ** _to_zr(self.group, (alpha_int * det) % self.order)
        k2 = [g1 ** _to_zr(self.group, entry) for entry in by]
        return {
            "k1": self.group.serialize(k1).decode("utf-8"),
            "k2": [self.group.serialize(v).decode("utf-8") for v in k2],
        }


class SIFEFHIPE(IPFEAbsCrypto):
    """Crypto operations for function-hiding inner-product encryption over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.group = PairingGroup(self.pp["pairing_group_param"])
        self.order = gp.mpz(int(self.group.order()))
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        x = [gp.mpz(v) for v in lst_pt]
        b_star = [[gp.mpz(v) for v in row] for row in self.sk["b_star"]]
        g2 = self.group.deserialize(self.sk["g2"].encode("utf-8"))
        beta = _random_nonzero_zr(self.group)
        beta_int = gp.mpz(int(beta))

        bstar_x = _mat_vec_mod(b_star, x, self.order)
        bstar_x = [(beta_int * entry) % self.order for entry in bstar_x]
        c1 = g2 ** _to_zr(self.group, beta_int)
        c2 = [g2 ** _to_zr(self.group, entry) for entry in bstar_x]
        return {
            "c1": self.group.serialize(c1).decode("utf-8"),
            "c2": [self.group.serialize(v).decode("utf-8") for v in c2],
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if len(fusion_weight) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        c1 = self.group.deserialize(dct_ct["c1"].encode("utf-8"))
        c2 = [self.group.deserialize(v.encode("utf-8")) for v in dct_ct["c2"]]
        k1 = self.group.deserialize(dk["k1"].encode("utf-8"))
        k2 = [self.group.deserialize(v.encode("utf-8")) for v in dk["k2"]]

        if len(c2) != self.pp["eta"] or len(k2) != self.pp["eta"]:
            raise FEValidationError("key or ciphertext length mismatch")

        d1 = pair(k1, c1)
        d2 = self.group.init(GT, 1)
        for lhs, rhs in zip(k2, c2):
            d2 *= pair(lhs, rhs)

        bound = int(self.pp["eta"] * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"]))
        return _bounded_discrete_log_gt(self.group, d1, d2, bound)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE")
