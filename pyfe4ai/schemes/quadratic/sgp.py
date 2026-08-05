"""
Secret-Key Quadratic Functional Encryption (SGP)
| Based on
| "Reading in the Dark: Classifying Encrypted Digits with Functional Encryption"
| By Dufour Sans, Gay, Pointcheval
| See ePrint 2018/206
|
| This prototype follows the GoFE `quadratic/sgp.go` structure and adapts it
| to the repository's key-generator / crypto split.

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
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.pairing_utils import to_zr
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)


def _rand_scalar(group: PairingGroup) -> gp.mpz:
    return gp.mpz(int(group.random(ZR)))


def _rand_invertible_2x2(
    group: PairingGroup, modulus: gp.mpz
) -> tuple[list[list[gp.mpz]], list[list[gp.mpz]]]:
    """Perform the _rand_invertible_2x2 operation.

        Args:
            group: Pairing group instance.
            modulus: Modulus for the arithmetic operation.
    """
    while True:
        w = [[_rand_scalar(group) % modulus for _ in range(2)] for _ in range(2)]
        det = (w[0][0] * w[1][1] - w[0][1] * w[1][0]) % modulus
        if det == 0:
            continue
        try:
            det_inv = gp.invert(det, modulus)
        except ZeroDivisionError:
            continue
        inv = [
            [(w[1][1] * det_inv) % modulus, ((-w[0][1]) * det_inv) % modulus],
            [((-w[1][0]) * det_inv) % modulus, (w[0][0] * det_inv) % modulus],
        ]
        return w, inv


def _transpose(matrix: list[list[gp.mpz]]) -> list[list[gp.mpz]]:
    return [list(col) for col in zip(*matrix)]


def _matvec_mod(
    matrix: list[list[gp.mpz]], vector: list[gp.mpz], modulus: gp.mpz
) -> list[gp.mpz]:
    """Perform the _matvec_mod operation.

        Args:
            matrix: Input matrix.
            vector: Input vector.
            modulus: Modulus for the arithmetic operation.
    """
    ret = []
    for row in matrix:
        acc = gp.mpz(0)
        for lhs, rhs in zip(row, vector):
            acc += lhs * rhs
        ret.append(acc % modulus)
    return ret


def _serialize_group_vec(group: PairingGroup, elements: list) -> list[str]:
    """Perform the _serialize_group_vec operation.

        Args:
            group: Pairing group instance.
            elements: Vector of group elements.
    """
    return [group.serialize(elem).decode("utf-8") for elem in elements]


def _deserialize_group_vec(group: PairingGroup, elements: list[str]) -> list:
    """Perform the _deserialize_group_vec operation.

        Args:
            group: Pairing group instance.
            elements: Vector of group elements.
    """
    return [group.deserialize(elem.encode("utf-8")) for elem in elements]


def _quadratic_form(
    matrix: list[list[gp.mpz]], lhs: list[gp.mpz], rhs: list[gp.mpz]
) -> gp.mpz:
    """Perform the _quadratic_form operation.

        Args:
            matrix: Input matrix.
            lhs: Left-hand side vector.
            rhs: Right-hand side vector.
    """
    acc = gp.mpz(0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            acc += value * lhs[i] * rhs[j]
    return acc


class QuadraticSGPKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for secret-key quadratic FE (SGP construction)."""

    _scheme_type = CryptoCONST.TYPE_QUADRATIC_SGP
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.n = config.get("n", config.get("eta", 2))
        self.bound = gp.mpz(config.get("bound", 16))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("n") == self.n
            and param.get("bound") == gp.digits(self.bound)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
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
        self.s = [_rand_scalar(self.group) % self.order for _ in range(self.n)]
        self.t = [_rand_scalar(self.group) % self.order for _ in range(self.n)]
        self.mpk = {}
        self.msk = {
            "g1": self.group.serialize(self.g1).decode("utf-8"),
            "g2": self.group.serialize(self.g2).decode("utf-8"),
            "s": [gp.digits(v) for v in self.s],
            "t": [gp.digits(v) for v in self.t],
        }
        logger.info("Quadratic SGP setup successfully")

    def get_public_parameters(self, **kwargs) -> dict | None:
        return {
            "n": self.n,
            "bound": gp.digits(self.bound),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.msk.get("g1"),
            "g2": self.msk.get("g2"),
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        return dict(self.msk)

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for Quadratic SGP decryption key generation")
        function_matrix = credentials.get("function_matrix")
        if not isinstance(function_matrix, list) or len(function_matrix) != self.n:
            raise FEValidationError("invalid function_matrix")
        if any(not isinstance(row, list) or len(row) != self.n for row in function_matrix):
            raise FEValidationError("invalid function_matrix shape")
        if any(abs(int(value)) > self.bound for row in function_matrix for value in row):
            raise FEValidationError("function_matrix exceeds configured bound")

        f_matrix = [[gp.mpz(value) for value in row] for row in function_matrix]
        s_vec = [gp.mpz(value) for value in self.msk["s"]]
        t_vec = [gp.mpz(value) for value in self.msk["t"]]
        exponent = _quadratic_form(f_matrix, s_vec, t_vec) % self.order
        g2 = self.group.deserialize(self.msk["g2"].encode("utf-8"))
        fe_key = g2 ** to_zr(self.group, exponent)

        return {
            "key": self.group.serialize(fe_key).decode("utf-8"),
            "f": [[gp.digits(value) for value in row] for row in f_matrix],
        }


class QuadraticSGP(IPFEAbsCrypto):
    """Crypto operations for secret-key quadratic FE (SGP construction)."""

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
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if not isinstance(payload, dict):
            raise FEValidationError("Quadratic SGP expects a dict payload with x and y")
        x_plain = payload.get("x")
        y_plain = payload.get("y")
        if not isinstance(x_plain, list) or not isinstance(y_plain, list):
            raise FEValidationError("payload must include list-valued x and y")
        if len(x_plain) != self.pp["n"] or len(y_plain) != self.pp["n"]:
            raise FEValidationError("invalid input size")

        bound = gp.mpz(self.pp["bound"])
        x = [gp.mpz(value) for value in x_plain]
        y = [gp.mpz(value) for value in y_plain]
        if any(abs(int(value)) > bound for value in x + y):
            raise FEValidationError("payload exceeds configured bound")

        g1 = self.group.deserialize(self.sk["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.sk["g2"].encode("utf-8"))
        s_vec = [gp.mpz(value) for value in self.sk["s"]]
        t_vec = [gp.mpz(value) for value in self.sk["t"]]

        gamma = _rand_scalar(self.group) % self.order
        w, w_inv = _rand_invertible_2x2(self.group, self.order)
        w_inv_t = _transpose(w_inv)

        a_rows = []
        b_rows = []
        for x_i, y_i, s_i, t_i in zip(x, y, s_vec, t_vec):
            left = [x_i % self.order, (gamma * s_i) % self.order]
            right = [y_i % self.order, (-t_i) % self.order]

            a_i = _matvec_mod(w_inv_t, left, self.order)
            b_i = _matvec_mod(w, right, self.order)

            a_rows.append([g1 ** to_zr(self.group, value) for value in a_i])
            b_rows.append([g2 ** to_zr(self.group, value) for value in b_i])

        return {
            "c0": self.group.serialize(g1 ** to_zr(self.group, gamma)).decode("utf-8"),
            "a": [_serialize_group_vec(self.group, row) for row in a_rows],
            "b": [_serialize_group_vec(self.group, row) for row in b_rows],
        }

    def decrypt(self, ct: dict, dk: dict) -> int:
        """Decrypt ciphertexts and recover the inner product.

            Args:
                ct: Ciphertext dict.
                dk: Functional decryption key.
        """
        function_matrix = [[gp.mpz(value) for value in row] for row in dk["f"]]
        if len(function_matrix) != self.pp["n"]:
            raise FEValidationError("invalid function matrix length")

        c0 = self.group.deserialize(ct["c0"].encode("utf-8"))
        key = self.group.deserialize(dk["key"].encode("utf-8"))
        a_rows = [_deserialize_group_vec(self.group, row) for row in ct["a"]]
        b_rows = [_deserialize_group_vec(self.group, row) for row in ct["b"]]

        prod = pair(c0, key)
        for i, row in enumerate(function_matrix):
            for j, value in enumerate(row):
                if value == 0:
                    continue
                e = pair(a_rows[i][0], b_rows[j][0]) * pair(a_rows[i][1], b_rows[j][1])
                prod *= e ** to_zr(self.group, value % self.order)

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        base = pair(g1, g2)
        bound = int((gp.mpz(self.pp["n"]) ** 2) * (gp.mpz(self.pp["bound"]) ** 3))
        return bounded_discrete_log_gt(self.group, base, prod, bound)

    def _prepare_quadratic_ndarray_payload(
        self, lhs_lst_ndarray: list, rhs_lst_ndarray: list
    ) -> tuple[list, list]:
        """Perform the _prepare_quadratic_ndarray_payload operation.

            Args:
                lhs_lst_ndarray: See implementation for details.
                rhs_lst_ndarray: See implementation for details.
        """
        if not isinstance(lhs_lst_ndarray, list) or not isinstance(rhs_lst_ndarray, list):
            raise FESchemeError("ndarray helper expects list-valued lhs and rhs arrays")
        if len(lhs_lst_ndarray) != self.pp["n"] or len(rhs_lst_ndarray) != self.pp["n"]:
            raise FEValidationError("invalid ndarray vector length for quadratic payload")
        shape = lhs_lst_ndarray[0].shape
        if any(ary.shape != shape for ary in lhs_lst_ndarray + rhs_lst_ndarray):
            raise FEValidationError("all ndarray inputs must share the same shape")
        lhs_scaled = [
            (ary.copy() * pow(10, self.precision)).astype(int) for ary in lhs_lst_ndarray
        ]
        rhs_scaled = [
            (ary.copy() * pow(10, self.precision)).astype(int) for ary in rhs_lst_ndarray
        ]
        return lhs_scaled, rhs_scaled

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        rhs_lst_ndarray = kwargs.get("rhs_lst_ndarray")
        lhs_scaled, rhs_scaled = self._prepare_quadratic_ndarray_payload(
            lst_ndarray, rhs_lst_ndarray
        )
        ary_ct = np.empty(lhs_scaled[0].shape, dtype=object)
        for i, _ in np.ndenumerate(lhs_scaled[0]):
            x = [int(ary[i]) for ary in lhs_scaled]
            y = [int(ary[i]) for ary in rhs_scaled]
            ary_ct[i] = self.encrypt({"x": x, "y": y})
        return [ary_ct]

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk=dk)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")

        lst_ndarray = []
        for ary in dict_ndarray_ct:
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ary_dec[i] = self.decrypt(ary[i], dk)
            lst_ndarray.append((ary_dec / pow(10, 2 * self.precision)).astype(float))
        return lst_ndarray
