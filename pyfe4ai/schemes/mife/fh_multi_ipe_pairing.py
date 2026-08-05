"""
Function-Hiding Multi-Input Inner-Product Encryption over Pairings
| Based on
| "Full-Hiding (Unbounded) Multi-Input Inner Product Functional Encryption
|  from the k-Linear Assumption"
| By Palash Datta, Tatsuaki Okamoto, Jun Tomida
| See ePrint 2018/061
|
| This prototype follows the GoFE `innerprod/fullysec/fh_multi_ipe.go`
| organization and adapts it to the repository's MIFE API.

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
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.pairing_utils import matrix_inverse_mod
from pyfe4ai.utils.pairing_utils import random_nonzero_zr
from pyfe4ai.utils.pairing_utils import to_zr
from pyfe4ai.utils.pairing_utils import transpose_mod
from pyfe4ai.utils.pairing_utils import vector_add_mod
from pyfe4ai.utils.pairing_utils import vector_scalar_mod
from pyfe4ai.utils.pairing_utils import serialize_matrix_g
from pyfe4ai.utils.pairing_utils import deserialize_matrix_g
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)

# Backward-compatible aliases
_vector_add_mod = vector_add_mod
_vector_scalar_mod = vector_scalar_mod
_serialize_matrix_g = serialize_matrix_g
_deserialize_matrix_g = deserialize_matrix_g


def _random_ob_matrix(
    group: PairingGroup, dim: int, mu: gp.mpz, order: gp.mpz
) -> tuple[list[list[gp.mpz]], list[list[gp.mpz]]]:
    """Perform the _random_ob_matrix operation.

        Args:
            group: Pairing group instance.
            dim: Vector dimension.
            mu: Obfuscation parameter.
            order: Group order (or exponent order for dequantization).
    """
    while True:
        b = [
            [gp.mpz(int(group.random(ZR))) for _ in range(dim)]
            for _ in range(dim)
        ]
        try:
            b_inv, _ = matrix_inverse_mod(b, order)
            break
        except ValueError:
            continue
    b_star = transpose_mod(b_inv)
    b_star = [[(mu * entry) % order for entry in row] for row in b_star]
    return b, b_star


class MIFEFHMultiIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for function-hiding multi-slot multi-input IPE over pairings."""

    _scheme_type = CryptoCONST.TYPE_MIFE_FH_MULTI_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.sec_level = int(config.get("sec_level", 2))
        self.eta = int(config.get("eta", CryptoCONST.MIFE_ETA))
        self.bound_x = gp.mpz(config.get("bound_x", 16))
        self.bound_y = gp.mpz(config.get("bound_y", 16))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))
        self.dim = 2 * self.eta + 2 * self.sec_level + 1

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("sec_level") == self.sec_level
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "sec_level": self.sec_level,
                    "eta": self.eta,
                    "n": self.n,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))
        self.dim = 2 * self.eta + 2 * self.sec_level + 1

    def setup(self) -> None:
        mu = gp.mpz(int(random_nonzero_zr(self.group)))
        g1 = self.group.random(G1)
        g2 = self.group.random(G2)
        pub = pair(g1, g2) ** to_zr(self.group, mu)

        bhat = {}
        bstar_hat = {}
        for nid in self.lst_nid:
            b, b_star = _random_ob_matrix(self.group, self.dim, mu, self.order)
            b_hat_rows = []
            b_star_hat_rows = []
            for j in range(self.eta + self.sec_level + 1):
                if j < self.eta:
                    b_hat_rows.append(b[j])
                    b_star_hat_rows.append(b_star[j])
                elif j == self.eta:
                    b_hat_rows.append(b[j + self.eta])
                    b_star_hat_rows.append(b_star[j + self.eta])
                elif j < self.eta + self.sec_level:
                    b_hat_rows.append(b[j - 1 + self.eta + self.sec_level])
                    b_star_hat_rows.append(b_star[j + self.eta])
                else:
                    b_hat_rows.append(b[j - 1 + self.eta + self.sec_level])

            bhat[nid] = [[gp.digits(v) for v in row] for row in b_hat_rows]
            bstar_hat[nid] = [[gp.digits(v) for v in row] for row in b_star_hat_rows]

        self.msk = {
            "bhat": bhat,
            "bstar_hat": bstar_hat,
        }
        self.pp = {
            "sec_level": self.sec_level,
            "eta": self.eta,
            "n": self.n,
            "lst_nid": list(self.lst_nid),
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.group.serialize(g1).decode("utf-8"),
            "g2": self.group.serialize(g2).decode("utf-8"),
            "pub": self.group.serialize(pub).decode("utf-8"),
        }
        logger.info("MIFE FHMultiIPE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid not in self.msk["bhat"]:
            return None
        return {
            "bhat": self.msk["bhat"][nid],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for FHMultiIPE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        gamma = [
            [gp.mpz(int(self.group.random(ZR))) for _ in range(self.n)]
            for _ in range(self.sec_level)
        ]
        if self.n > 1:
            partial_sum = gp.mpz(0)
            for idx in range(self.n - 1):
                partial_sum = (partial_sum + gamma[0][idx]) % self.order
            gamma[0][self.n - 1] = (-partial_sum) % self.order
        else:
            gamma[0][0] = gp.mpz(0)

        dk = {}
        for client_index, nid in enumerate(self.lst_nid):
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            bstar_hat = [
                [gp.mpz(v) for v in row]
                for row in self.msk["bstar_hat"][nid]
            ]
            key_vec = [gp.mpz(0) for _ in range(self.dim)]
            for row_index in range(self.eta + self.sec_level):
                if row_index < self.eta:
                    scalar = gp.mpz(y[row_index])
                else:
                    scalar = gamma[row_index - self.eta][client_index]
                contribution = vector_scalar_mod(
                    bstar_hat[row_index], scalar, self.order
                )
                key_vec = vector_add_mod(key_vec, contribution, self.order)
            g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
            g2_row = [g2 ** to_zr(self.group, value) for value in key_vec]
            dk[nid] = serialize_matrix_g(self.group, [g2_row])[0]

        return {"dk": dk}


class MIFEFHMultiIPE(IPFEAbsCrypto):
    """Crypto operations for function-hiding multi-slot multi-input IPE over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.group = PairingGroup(self.pp["pairing_group_param"])
        self.dim = 2 * self.pp["eta"] + 2 * self.pp["sec_level"] + 1
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
        bhat = [[gp.mpz(v) for v in row] for row in self.sk["bhat"]]
        phi = [gp.mpz(int(self.group.random(ZR))) for _ in range(self.pp["sec_level"])]

        key_vec = [gp.mpz(0) for _ in range(self.dim)]
        for row_index in range(self.pp["eta"] + self.pp["sec_level"] + 1):
            if row_index < self.pp["eta"]:
                scalar = x[row_index]
            elif row_index == self.pp["eta"]:
                scalar = gp.mpz(1)
            else:
                scalar = phi[row_index - self.pp["eta"] - 1]
            contribution = vector_scalar_mod(bhat[row_index], scalar, gp.mpz(int(self.group.order())))
            key_vec = vector_add_mod(key_vec, contribution, gp.mpz(int(self.group.order())))

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        ct = [g1 ** to_zr(self.group, value) for value in key_vec]
        return {"ct": [self.group.serialize(value).decode("utf-8") for value in ct]}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict | None = None):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        pub = self.group.deserialize(self.pp["pub"].encode("utf-8"))
        acc = self.group.init(GT, 1)
        for nid in self.pp.get("lst_nid", list(dct_ct.keys())):
            if nid not in dct_ct or nid not in dk["dk"]:
                continue
            ct = [self.group.deserialize(v.encode("utf-8")) for v in dct_ct[nid]["ct"]]
            key = [self.group.deserialize(v.encode("utf-8")) for v in dk["dk"][nid]]
            for lhs, rhs in zip(ct, key):
                acc *= pair(lhs, rhs)

        bound = int(
            self.pp["n"]
            * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"])
            * gp.mpz(self.pp["bound_y"])
        )
        return bounded_discrete_log_gt(self.group, pub, acc, bound)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
