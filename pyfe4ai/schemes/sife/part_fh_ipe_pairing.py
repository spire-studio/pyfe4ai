"""
Partially Function-Hiding Inner-Product Encryption over Pairings
| Based on
| "A New Paradigm for Public-Key Functional Encryption for Degree-2 Polynomials"
| By Romain Gay
|
| This prototype follows the GoFE `innerprod/fullysec/part_fh_ipe.go`
| organization and adapts it to the repository's single-input FE API.

* type:     public-key encryption
* setting:  Pairing based
* note:     research prototype aligned with the existing crypto-ipfe API

"""

from __future__ import annotations

import json
import logging
import os
import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.pairing_utils import mat_vec_mod
from pyfe4ai.utils.pairing_utils import random_nonzero_zr
from pyfe4ai.utils.pairing_utils import to_zr
from pyfe4ai.utils.pairing_utils import transpose_mod
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, PairingGroup, pair
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)


def _rand_scalar(order: gp.mpz) -> gp.mpz:
    return gp.mpz(_CSPRNG.randrange(1, int(order)))


def _rand_matrix(rows: int, cols: int, order: gp.mpz) -> list[list[gp.mpz]]:
    """Perform the _rand_matrix operation.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            order: Group order (or exponent order for dequantization).
    """
    return [[_rand_scalar(order) for _ in range(cols)] for _ in range(rows)]


def _rand_bounded_matrix(
    rows: int, cols: int, bound: gp.mpz
) -> list[list[gp.mpz]]:
    """Perform the _rand_bounded_matrix operation.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            bound: Upper bound on absolute values.
    """
    b = int(bound)
    return [[gp.mpz(_CSPRNG.randint(-b, b)) for _ in range(cols)] for _ in range(rows)]


def _matmul_mod(
    left: list[list[gp.mpz]], right: list[list[gp.mpz]], modulus: gp.mpz
) -> list[list[gp.mpz]]:
    """Perform the _matmul_mod operation.

        Args:
            left: Left operand matrix or vector.
            right: Right operand matrix or vector.
            modulus: Modulus for the arithmetic operation.
    """
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    ret = [[gp.mpz(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        row_i = ret[i]
        for k in range(inner):
            left_ik = left[i][k]
            if left_ik == 0:
                continue
            right_k = right[k]
            for j in range(cols):
                row_i[j] += left_ik * right_k[j]
        for j in range(cols):
            row_i[j] = row_i[j] % modulus
    return ret


def _matrix_vec_plain(matrix: list[list[gp.mpz]], vector: list[gp.mpz]) -> list[gp.mpz]:
    """Perform the _matrix_vec_plain operation.

        Args:
            matrix: Input matrix.
            vector: Input vector.
    """
    ret = []
    for row in matrix:
        acc = gp.mpz(0)
        for a_i, b_i in zip(row, vector):
            acc += a_i * b_i
        ret.append(acc)
    return ret


def _g1_pow_vec(group: PairingGroup, g1, scalars: list[gp.mpz]):
    """Perform the _g1_pow_vec operation.

        Args:
            group: Pairing group instance.
            g1: Generator element in G1.
            scalars: Vector of scalar exponents.
    """
    return [g1 ** to_zr(group, s) for s in scalars]


def _g2_pow_vec(group: PairingGroup, g2, scalars: list[gp.mpz]):
    """Perform the _g2_pow_vec operation.

        Args:
            group: Pairing group instance.
            g2: Generator element in G2.
            scalars: Vector of scalar exponents.
    """
    return [g2 ** to_zr(group, s) for s in scalars]


def _serialize_vec(group: PairingGroup, vec) -> list[str]:
    """Perform the _serialize_vec operation.

        Args:
            group: Pairing group instance.
            vec: Input vector.
    """
    return [group.serialize(v).decode("utf-8") for v in vec]


def _deserialize_vec(group: PairingGroup, vec: list[str]):
    """Perform the _deserialize_vec operation.

        Args:
            group: Pairing group instance.
            vec: Input vector.
    """
    return [group.deserialize(v.encode("utf-8")) for v in vec]


class SIFEPartFHIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for partially function-hiding inner-product encryption over pairings."""

    _scheme_type = CryptoCONST.TYPE_SIFE_PART_FH_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound = gp.mpz(config.get("bound", config.get("bound_x", 16)))
        self.subspace_dim = int(config.get("subspace_dim", max(1, min(3, self.eta))))
        self.coeff_bound = gp.mpz(config.get("coeff_bound", 1))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bound") == gp.digits(self.bound)
            and param.get("subspace_dim") == self.subspace_dim
            and param.get("coeff_bound") == gp.digits(self.coeff_bound)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "bound": gp.digits(self.bound),
                    "subspace_dim": self.subspace_dim,
                    "coeff_bound": gp.digits(self.coeff_bound),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        self.g1 = self.group.random(G1)
        self.g2 = self.group.random(G2)
        order = self.order

        a_vec = [gp.mpz(1), _rand_scalar(order)]
        b_vec = [gp.mpz(1), _rand_scalar(order)]
        u = _rand_matrix(self.eta + 2, 2, order)
        v = _rand_matrix(self.eta, 2, order)

        m_entry_bound = max(gp.mpz(1), self.bound // max(1, self.subspace_dim * int(self.coeff_bound)))
        m = _rand_bounded_matrix(self.eta, self.subspace_dim, m_entry_bound)

        ua_scalars = mat_vec_mod(u, a_vec, order)
        vt = transpose_mod(v)
        vt_m = matmul_plain = _matmul_mod(vt, m, order)

        self.pp = {
            "eta": self.eta,
            "bound": gp.digits(self.bound),
            "subspace_dim": self.subspace_dim,
            "coeff_bound": gp.digits(self.coeff_bound),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.group.serialize(self.g1).decode("utf-8"),
            "g2": self.group.serialize(self.g2).decode("utf-8"),
            "a": _serialize_vec(self.group, _g1_pow_vec(self.group, self.g1, a_vec)),
            "ua": _serialize_vec(self.group, _g1_pow_vec(self.group, self.g1, ua_scalars)),
            "vt_m": [[gp.digits(v) for v in row] for row in vt_m],
            "m": [[gp.digits(v) for v in row] for row in m],
        }
        self.msk = {
            "b": [gp.digits(v) for v in b_vec],
            "u": [[gp.digits(v) for v in row] for row in u],
            "v": [[gp.digits(v) for v in row] for row in v],
        }
        logger.info("SIFE partially FH-IPE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        return {
            "b": self.msk["b"],
            "u": self.msk["u"],
            "v": self.msk["v"],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for partially FH-IPE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weight) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")
        if any(abs(int(v)) > self.bound for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound")

        order = self.order
        y = [gp.mpz(v) for v in fusion_weight]
        b = [gp.mpz(v) for v in self.msk["b"]]
        u = [[gp.mpz(v) for v in row] for row in self.msk["u"]]
        v = [[gp.mpz(v) for v in row] for row in self.msk["v"]]

        s = gp.mpz(int(random_nonzero_zr(self.group)))
        bs = [(s * entry) % order for entry in b]
        vbs = mat_vec_mod(v, bs, order)
        y_vbs = [(a + b_i) % order for a, b_i in zip(y, vbs)]
        key2 = bs + y_vbs
        u_t = transpose_mod(u)
        key1 = [(-val) % order for val in mat_vec_mod(u_t, key2, order)]
        key = key1 + key2

        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        return {"k": _serialize_vec(self.group, _g2_pow_vec(self.group, g2, key))}


class SIFEPartFHIPE(IPFEAbsCrypto):
    """Crypto operations for partially function-hiding inner-product encryption over pairings."""

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
        if len(lst_pt) != self.pp["subspace_dim"]:
            raise FESchemeError("public encryption expects subspace coefficients of configured dimension")
        if any(abs(int(v)) > gp.mpz(self.pp["coeff_bound"]) for v in lst_pt):
            raise FEValidationError("subspace coefficients exceed configured coeff_bound")

        t = [gp.mpz(v) for v in lst_pt]
        m = [[gp.mpz(v) for v in row] for row in self.pp["m"]]
        x = _matrix_vec_plain(m, t)
        if any(abs(int(v)) > gp.mpz(self.pp["bound"]) for v in x):
            raise FEValidationError("derived plaintext exceeds configured bound")

        a = _deserialize_vec(self.group, self.pp["a"])
        ua = _deserialize_vec(self.group, self.pp["ua"])
        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        vt_m = [[gp.mpz(v) for v in row] for row in self.pp["vt_m"]]

        r = random_nonzero_zr(self.group)
        r_int = gp.mpz(int(r))
        c = [elem ** r for elem in a]
        uc = [elem ** r for elem in ua]

        vt_mt = mat_vec_mod(vt_m, t, self.order)
        neg_vt_mt = [(-v) % self.order for v in vt_mt]
        mt_g1 = _g1_pow_vec(self.group, g1, x)
        prefix = _g1_pow_vec(self.group, g1, neg_vt_mt)
        cipher2 = prefix + mt_g1
        cipher2 = [lhs * rhs for lhs, rhs in zip(cipher2, uc)]
        cipher = c + cipher2
        return {"ct": _serialize_vec(self.group, cipher)}

    def sec_encrypt(self, lst_pt: list) -> dict:
        if not self._has_private_keys():
            raise FEKeyError("no secret key provided for secret-key encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound")

        x = [gp.mpz(v) for v in lst_pt]
        a = _deserialize_vec(self.group, self.pp["a"])
        ua = _deserialize_vec(self.group, self.pp["ua"])
        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        v = [[gp.mpz(v) for v in row] for row in self.sk["v"]]
        vt = transpose_mod(v)

        r = random_nonzero_zr(self.group)
        c = [elem ** r for elem in a]
        uc = [elem ** r for elem in ua]

        vt_x = mat_vec_mod(vt, x, self.order)
        neg_vt_x = [(-v) % self.order for v in vt_x]
        x_g1 = _g1_pow_vec(self.group, g1, x)
        prefix = _g1_pow_vec(self.group, g1, neg_vt_x)
        cipher2 = prefix + x_g1
        cipher2 = [lhs * rhs for lhs, rhs in zip(cipher2, uc)]
        cipher = c + cipher2
        return {"ct": _serialize_vec(self.group, cipher)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list | None = None):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        cipher = _deserialize_vec(self.group, dct_ct["ct"])
        key = _deserialize_vec(self.group, dk["k"])
        if len(cipher) != self.pp["eta"] + 4 or len(key) != self.pp["eta"] + 4:
            raise FEValidationError("ciphertext or key length mismatch")

        acc = pair(cipher[0], key[0])
        for lhs, rhs in zip(cipher[1:], key[1:]):
            acc *= pair(lhs, rhs)

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        base = pair(g1, g2)
        bound = int(self.pp["eta"] * (gp.mpz(self.pp["bound"]) ** 2))
        return bounded_discrete_log_gt(self.group, base, acc, bound)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        raise NotImplementedError("ndarray helper is not implemented for partially FH-IPE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for partially FH-IPE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for partially FH-IPE")
