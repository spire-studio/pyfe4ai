"""
Function-Hiding Multi-Client Inner-Product Encryption over Pairings
| Multi-client label-aware structure inspired by
| "Decentralized multi-client functional encryption for inner product"
| By Jeremy Chotard, Edouard Dufour Sans, Romain Gay, Duong Hieu Phan,
| David Pointcheval
| Published in: ASIACRYPT 2018
|
| Instantiated here with a function-hiding multi-input inner-product FE
| building block based on
| "Full-Hiding (Unbounded) Multi-Input Inner Product Functional Encryption
|  from the k-Linear Assumption"
| By Palash Datta, Tatsuaki Okamoto, Jun Tomida
| See ePrint 2018/061

* type:     public-key encryption
* setting:  Pairing based
* note:     research-oriented Python prototype aligned with the existing
            crypto-ipfe API style

"""

from __future__ import annotations

import json
import logging
import os
import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPE
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.lwe_utils import label_scalar_from_hash
from pyfe4ai.utils.pairing_backend import PairingGroup
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)


class MCFEFHMultiIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for function-hiding multi-client IPE over pairings."""

    _scheme_type = CryptoCONST.TYPE_MCFE_FH_MULTI_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.sec_level = int(config.get("sec_level", 2))
        self.eta = int(config.get("eta", CryptoCONST.MCFE_ETA))
        self.bound_x = gp.mpz(config.get("bound_x", 16))
        self.bound_y = gp.mpz(config.get("bound_y", 16))
        self.u_bound = gp.mpz(config.get("u_bound", max(2, int(self.bound_x))))
        self.label_modulus = int(config.get("label_modulus", 17))
        self.pairing_group_param = config.get("pairing_group_param", "SS512")
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("sec_level") == self.sec_level
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("u_bound") == gp.digits(self.u_bound)
            and param.get("label_modulus") == self.label_modulus
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
                    "u_bound": gp.digits(self.u_bound),
                    "label_modulus": self.label_modulus,
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        half_label = self.label_modulus // 2
        inner_bound_x = self.bound_x + self.u_bound * gp.mpz(half_label)
        inner_config = {
            "sec_param": self.sec_param,
            "sec_level": self.sec_level,
            "eta": self.eta,
            "n": self.n,
            "s": self.s,
            "lst_nid": self.lst_nid,
            "bound_x": gp.digits(inner_bound_x),
            "bound_y": gp.digits(self.bound_y),
            "pairing_group_param": self.pairing_group_param,
        }
        self.inner_kg = MIFEFHMultiIPEKeyGenerator(inner_config)
        self.inner_kg.setup()
        inner_pp = self.inner_kg.get_public_parameters()

        rand = _CSPRNG
        u = {
            nid: [gp.mpz(rand.randint(-int(self.u_bound), int(self.u_bound))) for _ in range(self.eta)]
            for nid in self.lst_nid
        }
        self.pp = {
            "sec_level": self.sec_level,
            "eta": self.eta,
            "n": self.n,
            "lst_nid": list(self.lst_nid),
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "u_bound": gp.digits(self.u_bound),
            "label_modulus": self.label_modulus,
            "pairing_group_param": self.pairing_group_param,
            "inner_bound_x": gp.digits(inner_bound_x),
            "inner": inner_pp,
        }
        self.mpk = {}
        self.msk = {
            "inner": self.inner_kg.msk,
            "u": u,
        }
        logger.info("MCFE FHMultiIPE setup successfully")

    def get_public_parameters(self) -> dict:
        return dict(self.pp)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        inner_sk = self.inner_kg.get_private_keys(nid)
        if inner_sk is None or nid not in self.msk["u"]:
            return None
        return {
            "inner": inner_sk,
            "u": [gp.digits(value) for value in self.msk["u"][nid]],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Perform the get_decryption_keys operation.

            Args:
                sid: Session / decryption-key identifier.
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for MCFE FHMultiIPE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        inner_dk = self.inner_kg.get_decryption_keys(
            sid, credentials={"fusion_weight": fusion_weight}
        )
        z = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            weights = fusion_weight[nid]
            if len(weights) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(value)) > self.bound_y for value in weights):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            for u_i, y_i in zip(self.msk["u"][nid], weights):
                z += u_i * gp.mpz(y_i)

        return {"dk": inner_dk, "z": gp.digits(z)}


class MCFEFHMultiIPE(IPFEAbsCrypto):
    """Crypto operations for function-hiding multi-client IPE over pairings."""

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
            self.inner = MIFEFHMultiIPE(
                {"id": self.id, "keys": {"pp": self.pp["inner"], "sk": self.sk["inner"]}}
            )
        else:
            self.inner = MIFEFHMultiIPE({"id": self.id, "keys": {"pp": self.pp["inner"]}})

    def _label_scalar(self, label: str) -> gp.mpz:
        label_hash = md5_hash(label, self.order)
        return label_scalar_from_hash(label_hash, self.pp["label_modulus"])

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt plaintext and return a ciphertext dict.

            Args:
                lst_pt: Integer plaintext vector.
                label: Encryption label for replay protection.
        """
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        label_scalar = self._label_scalar(label)
        u = [gp.mpz(value) for value in self.sk["u"]]
        masked = [
            gp.mpz(x_i) + label_scalar * u_i
            for x_i, u_i in zip(lst_pt, u)
        ]
        inner_ct = self.inner.encrypt(masked)
        return {"ct": inner_ct["ct"]}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
                label: Encryption label for replay protection.
        """
        total = self.inner.decrypt(dct_ct, dk["dk"])
        label_scalar = self._label_scalar(label)
        correction = gp.mpz(dk["z"]) * label_scalar
        return int(gp.mpz(total) - correction)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)], label)
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, label)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
