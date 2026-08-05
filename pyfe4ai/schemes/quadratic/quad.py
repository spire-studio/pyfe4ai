"""
Public-Key Quadratic Functional Encryption (Quad)
| Based on
| "A New Paradigm for Public-Key Functional Encryption for Degree-2 Polynomials"
| By Romain Gay
|
| This repository prototype realizes quadratic FE through tensor encoding over
| the repository's partially function-hiding IPE building block. The overall
| organization is aligned with GoFE's `quadratic/quad.go`, while the Python
| implementation favors a simpler composition over a line-by-line port.

* type:     public-key encryption
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
from pyfe4ai.schemes.sife.part_fh_ipe_pairing import SIFEPartFHIPE
from pyfe4ai.schemes.sife.part_fh_ipe_pairing import SIFEPartFHIPEKeyGenerator
from pyfe4ai.utils.pairing_utils import transpose_mod
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)


def _flatten_matrix(matrix: list[list[int | gp.mpz]]) -> list[gp.mpz]:
    return [gp.mpz(value) for row in matrix for value in row]


def _tensor_vector(x: list[gp.mpz], y: list[gp.mpz]) -> list[gp.mpz]:
    """Perform the _tensor_vector operation.

        Args:
            x: Input integer.
            y: See implementation for details.
    """
    return [x_i * y_j for x_i in x for y_j in y]


def _identity_matrix(size: int) -> list[list[gp.mpz]]:
    return [
        [gp.mpz(1 if i == j else 0) for j in range(size)]
        for i in range(size)
    ]


class QuadraticQuadKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for public-key quadratic FE (Quad construction)."""

    _scheme_type = CryptoCONST.TYPE_QUADRATIC_QUAD
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.n = int(config.get("n", 2))
        self.m = int(config.get("m", self.n))
        self.bound = gp.mpz(config.get("bound", 16))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        pass

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("n") == self.n
            and param.get("m") == self.m
            and param.get("bound") == gp.digits(self.bound)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "n": self.n,
                    "m": self.m,
                    "bound": gp.digits(self.bound),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )

    def setup(self) -> None:
        tensor_dim = self.n * self.m
        tensor_bound = self.bound * self.bound
        inner_config = {
            "sec_param": self.sec_param,
            "eta": tensor_dim,
            "bound": gp.digits(tensor_bound),
            "subspace_dim": tensor_dim,
            "coeff_bound": gp.digits(tensor_bound),
            "pairing_group_param": self.pairing_group_param,
        }
        self.inner_kg = SIFEPartFHIPEKeyGenerator(inner_config)
        self.inner_kg.setup()

        inner_pp = self.inner_kg.get_public_parameters()
        v = [[gp.mpz(value) for value in row] for row in self.inner_kg.msk["v"]]
        vt = transpose_mod(v)
        identity = _identity_matrix(tensor_dim)

        inner_pp["bound"] = gp.digits(tensor_bound)
        inner_pp["coeff_bound"] = gp.digits(tensor_bound)
        inner_pp["subspace_dim"] = tensor_dim
        inner_pp["m"] = [[gp.digits(value) for value in row] for row in identity]
        inner_pp["vt_m"] = [[gp.digits(value) for value in row] for row in vt]

        self.pp = {
            "n": self.n,
            "m": self.m,
            "bound": gp.digits(self.bound),
            "pairing_group_param": self.pairing_group_param,
            "inner": inner_pp,
        }
        self.mpk = {}
        self.msk = {}
        logger.info("Quadratic Quad setup successfully")

    def get_public_parameters(self, **kwargs) -> dict | None:
        return self.pp

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        return None

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for Quadratic Quad decryption key generation")
        function_matrix = credentials.get("function_matrix")
        if not isinstance(function_matrix, list) or len(function_matrix) != self.n:
            raise FEValidationError("invalid function_matrix")
        if any(not isinstance(row, list) or len(row) != self.m for row in function_matrix):
            raise FEValidationError("invalid function_matrix shape")
        if any(abs(int(value)) > self.bound for row in function_matrix for value in row):
            raise FEValidationError("function_matrix exceeds configured bound")

        fusion_weight = _flatten_matrix(function_matrix)
        inner_dk = self.inner_kg.get_decryption_keys(
            sid, credentials={"fusion_weight": fusion_weight}
        )
        return {
            "dk": inner_dk,
            "f": [[gp.digits(gp.mpz(value)) for value in row] for row in function_matrix],
        }


class QuadraticQuad(IPFEAbsCrypto):
    """Crypto operations for public-key quadratic FE (Quad construction)."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.inner = SIFEPartFHIPE({"id": self.id, "keys": {"pp": self.pp["inner"]}})

    def encrypt(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise FEValidationError("Quadratic Quad expects a dict payload with x and y")
        x_plain = payload.get("x")
        y_plain = payload.get("y")
        if not isinstance(x_plain, list) or not isinstance(y_plain, list):
            raise FEValidationError("payload must include list-valued x and y")
        if len(x_plain) != self.pp["n"] or len(y_plain) != self.pp["m"]:
            raise FEValidationError("invalid input size")

        bound = gp.mpz(self.pp["bound"])
        x = [gp.mpz(value) for value in x_plain]
        y = [gp.mpz(value) for value in y_plain]
        if any(abs(int(value)) > bound for value in x + y):
            raise FEValidationError("payload exceeds configured bound")

        tensor = _tensor_vector(x, y)
        inner_ct = self.inner.encrypt(tensor)
        return {"ct": inner_ct["ct"]}

    def decrypt(self, ct: dict, dk: dict) -> int:
        """Decrypt ciphertexts and recover the inner product.

            Args:
                ct: Ciphertext dict.
                dk: Functional decryption key.
        """
        return self.inner.decrypt(ct, dk["dk"])

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
        if len(lhs_lst_ndarray) != self.pp["n"] or len(rhs_lst_ndarray) != self.pp["m"]:
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
