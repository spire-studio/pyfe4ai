"""
Multi-Input Secret-Key Quadratic Functional Encryption (MI-SGP)
| Based on the single-input SGP construction from
| "Reading in the Dark: Classifying Encrypted Digits with Functional Encryption"
| By Dufour Sans, Gay, Pointcheval  (ePrint 2018/206)
|
| Extended to multi-input: each client i independently encrypts its own
| pair (x_i, y_i), and decryption recovers the aggregated quadratic form
| sum_i x_i^T F_i y_i  where F_i is the per-client function matrix.
|
| The multi-input composition follows the per-client secret key approach:
| each client holds independent (s_i, t_i) secrets and encrypts using
| independent randomness (gamma_i, W_i).  Decryption combines pairing
| results across all clients.

* type:     secret-key encryption
* setting:  Pairing based
* note:     research prototype aligned with the existing crypto-ipfe API

"""

from __future__ import annotations

import json
import logging
import os

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.quadratic.sgp import (
    _matvec_mod,
    _quadratic_form,
    _rand_invertible_2x2,
    _rand_scalar,
    _serialize_group_vec,
    _deserialize_group_vec,
    _transpose,
)
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.pairing_utils import to_zr
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair
from pyfe4ai.utils.pairing_backend import require_pairing_backend

logger = logging.getLogger(__name__)

TYPE_MULTI_INPUT_QUADRATIC_SGP = "MI_QUADRATIC_SGP"


class MultiInputQuadraticSGPKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for multi-input secret-key quadratic FE (MI-SGP)."""

    _scheme_type = TYPE_MULTI_INPUT_QUADRATIC_SGP
    """Key generator for multi-input secret-key quadratic FE.

    Each client i receives independent secrets (s_i, t_i) and shared
    group generators (g1, g2).  The function key for a set of per-client
    matrices {F_i} embeds sum_i <s_i, F_i t_i> in the exponent.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.d = config.get("d", config.get("n", 2))  # per-client vector dim
        if not self.n:
            raise ValueError("need to provide number of clients")
        self.bound = gp.mpz(config.get("bound", 16))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("d") == self.d
            and param.get("n") == self.n
            and param.get("bound") == gp.digits(self.bound)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "d": self.d,
                    "n": self.n,
                    "bound": gp.digits(self.bound),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        self.g1 = self.group.random(G1)
        self.g2 = self.group.random(G2)

        # Per-client secrets
        dct_s = {
            nid: [_rand_scalar(self.group) % self.order for _ in range(self.d)]
            for nid in self.lst_nid
        }
        dct_t = {
            nid: [_rand_scalar(self.group) % self.order for _ in range(self.d)]
            for nid in self.lst_nid
        }

        self.mpk = {}
        self.msk = {
            "g1": self.group.serialize(self.g1).decode("utf-8"),
            "g2": self.group.serialize(self.g2).decode("utf-8"),
            "s": {nid: [gp.digits(v) for v in vec] for nid, vec in dct_s.items()},
            "t": {nid: [gp.digits(v) for v in vec] for nid, vec in dct_t.items()},
        }
        logger.info("Multi-input Quadratic SGP setup successfully")

    def get_public_parameters(self, **kwargs) -> dict | None:
        return {
            "d": self.d,
            "n": self.n,
            "lst_nid": self.lst_nid,
            "bound": gp.digits(self.bound),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.msk["g1"],
            "g2": self.msk["g2"],
        }

    def get_private_keys(self, nid: str = None, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid is None or nid not in self.msk["s"]:
            return None
        return {
            "g1": self.msk["g1"],
            "g2": self.msk["g2"],
            "s": self.msk["s"][nid],
            "t": self.msk["t"][nid],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Generate functional decryption key for per-client function matrices.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise ValueError("need credentials for MI-Quadratic SGP DK generation")
        function_matrices = credentials.get("function_matrices")
        if not isinstance(function_matrices, dict):
            raise ValueError("function_matrices must be a dict {client_id: matrix}")
        if set(function_matrices.keys()) != set(self.lst_nid):
            raise ValueError("function_matrices must cover all clients")

        for nid, f_matrix in function_matrices.items():
            if (
                not isinstance(f_matrix, list)
                or len(f_matrix) != self.d
                or any(len(row) != self.d for row in f_matrix)
            ):
                raise ValueError(
                    "invalid function_matrix shape for client {}".format(nid)
                )
            if any(
                abs(int(value)) > self.bound
                for row in f_matrix
                for value in row
            ):
                raise ValueError(
                    "function_matrix exceeds bound for client {}".format(nid)
                )

        # Per-client sub-keys: key_i = g2^{<s_i, F_i t_i>}
        g2 = self.group.deserialize(self.msk["g2"].encode("utf-8"))
        serialized_matrices = {}
        per_client_keys = {}
        for nid in self.lst_nid:
            f_matrix = [[gp.mpz(v) for v in row] for row in function_matrices[nid]]
            s_vec = [gp.mpz(v) for v in self.msk["s"][nid]]
            t_vec = [gp.mpz(v) for v in self.msk["t"][nid]]
            exponent_i = _quadratic_form(f_matrix, s_vec, t_vec) % self.order
            key_i = g2 ** to_zr(self.group, exponent_i)
            per_client_keys[nid] = self.group.serialize(key_i).decode("utf-8")
            serialized_matrices[nid] = [
                [gp.digits(v) for v in row] for row in f_matrix
            ]

        return {
            "keys": per_client_keys,
            "f": serialized_matrices,
        }


class MultiInputQuadraticSGP(IPFEAbsCrypto):
    """Multi-input secret-key quadratic FE.

    Each client independently encrypts (x_i, y_i). The decryptor combines
    ciphertexts from all clients and recovers sum_i x_i^T F_i y_i.
    """

    scheme_type = TYPE_MULTI_INPUT_QUADRATIC_SGP

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

    def encrypt(self, payload: dict) -> dict:
        """Encrypt a single client's (x, y) pair.

        ``payload``: dict with keys ``x`` and ``y`` (list[int] of length d).
        """
        if not self._has_private_keys():
            raise ValueError("no encryption key provided")
        if not isinstance(payload, dict):
            raise ValueError("expects a dict payload with x and y")
        x_plain = payload.get("x")
        y_plain = payload.get("y")
        if not isinstance(x_plain, list) or not isinstance(y_plain, list):
            raise ValueError("payload must include list-valued x and y")
        d = self.pp["d"]
        if len(x_plain) != d or len(y_plain) != d:
            raise ValueError("invalid input size")

        bound = gp.mpz(self.pp["bound"])
        x = [gp.mpz(v) for v in x_plain]
        y = [gp.mpz(v) for v in y_plain]
        if any(abs(int(v)) > bound for v in x + y):
            raise ValueError("payload exceeds configured bound")

        g1 = self.group.deserialize(self.sk["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.sk["g2"].encode("utf-8"))
        s_vec = [gp.mpz(v) for v in self.sk["s"]]
        t_vec = [gp.mpz(v) for v in self.sk["t"]]

        gamma = _rand_scalar(self.group) % self.order
        w, w_inv = _rand_invertible_2x2(self.group, self.order)
        w_inv_t = _transpose(w_inv)

        a_rows, b_rows = [], []
        for x_i, y_i, s_i, t_i in zip(x, y, s_vec, t_vec):
            left = [x_i % self.order, (gamma * s_i) % self.order]
            right = [y_i % self.order, (-t_i) % self.order]
            a_i = _matvec_mod(w_inv_t, left, self.order)
            b_i = _matvec_mod(w, right, self.order)
            a_rows.append([g1 ** to_zr(self.group, v) for v in a_i])
            b_rows.append([g2 ** to_zr(self.group, v) for v in b_i])

        return {
            "c0": self.group.serialize(g1 ** to_zr(self.group, gamma)).decode("utf-8"),
            "a": [_serialize_group_vec(self.group, row) for row in a_rows],
            "b": [_serialize_group_vec(self.group, row) for row in b_rows],
        }

    def decrypt(self, dct_ct: dict, dk: dict) -> int:
        """Decrypt aggregated quadratic form from multiple clients.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
        """
        function_matrices = dk["f"]
        per_client_keys = dk["keys"]
        d = self.pp["d"]

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        base = pair(g1, g2)
        dlog_bound = int((gp.mpz(d) ** 2) * (gp.mpz(self.pp["bound"]) ** 3))

        # Per-client decrypt and sum
        total = 0
        for client_id, ct in dct_ct.items():
            if client_id not in function_matrices:
                raise ValueError("missing function matrix for client {}".format(client_id))
            if client_id not in per_client_keys:
                raise ValueError("missing sub-key for client {}".format(client_id))

            f_matrix = [[gp.mpz(v) for v in row] for row in function_matrices[client_id]]
            key_i = self.group.deserialize(per_client_keys[client_id].encode("utf-8"))
            c0 = self.group.deserialize(ct["c0"].encode("utf-8"))
            a_rows = [_deserialize_group_vec(self.group, row) for row in ct["a"]]
            b_rows = [_deserialize_group_vec(self.group, row) for row in ct["b"]]

            # prod_i = e(c0_i, key_i) * prod_{j,k} e(a_{ij}, b_{ik})^{F_i[j][k]}
            #        = e(g1,g2)^{gamma_i * <s_i, F_i t_i>}
            #          * prod_{j,k} e(g1,g2)^{(x_ij*y_ik + gamma_i*s_ij*(-t_ik)) * F_i[j][k]}
            #        = e(g1,g2)^{x_i^T F_i y_i}
            prod = pair(c0, key_i)
            for i in range(d):
                for j in range(d):
                    val = int(f_matrix[i][j]) % int(self.order)
                    if val == 0:
                        continue
                    e = pair(a_rows[i][0], b_rows[j][0]) * pair(a_rows[i][1], b_rows[j][1])
                    prod = prod * (e ** to_zr(self.group, gp.mpz(val)))

            client_result = bounded_discrete_log_gt(self.group, base, prod, dlog_bound)
            total += client_result

        return total

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        raise NotImplementedError("ndarray helpers not yet supported for MI-SGP")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helpers not yet supported for MI-SGP")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helpers not yet supported for MI-SGP")
