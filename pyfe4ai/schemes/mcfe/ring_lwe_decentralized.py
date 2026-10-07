"""
Decentralized Ring-LWE-Based Multi-Client Inner-Product Functional Encryption
| Decentralized key-share organization inspired by
| "Decentralizing inner-product functional encryption"
| By Michel Abdalla, Fabrice Benhamouda, Markulf Kohlweiss, Hendrik Waldner
| Published in: PKC 2019
|
| This prototype combines decentralized decryption-key-share derivation with
| the Ring-LWE multi-client inner-product FE line used in this repository.

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

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.utils.ring_lwe_utils import transpose
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.lwe_utils import label_scalar_from_hash
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix

logger = logging.getLogger(__name__)


class DecentralizedMCFERingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for decentralized Ring-LWE-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_DMCFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.ring_n = config.get("ring_n", None)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self.bound_u = gp.mpz(config.get("bound_u", 2))
        self.label_modulus = int(config.get("label_modulus", 8))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _effective_bound_x(self) -> gp.mpz:
        return self.bound_x + self.bound_u * gp.mpz(self.label_modulus // 2)

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
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("lst_nid") == self.lst_nid
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("bound_u") == gp.digits(self.bound_u)
            and param.get("label_modulus") == self.label_modulus
            and (self.ring_n is None or param.get("ring_n") == self.ring_n)
        )

    def _generate_and_save(self, param_file: str) -> None:
        helper = MIFERingLWEKeyGenerator(
            {
                "sec_param": self.sec_param,
                "eta": self.eta,
                "n": self.n,
                "ring_n": self.ring_n,
                "bound_x": int(self._effective_bound_x()),
                "bound_y": int(self.bound_y),
            }
        )
        self.p = helper.p
        self.q = helper.q
        self.ring_n = helper.ring_n
        self.sigma1 = helper.sigma1
        self.sigma2 = helper.sigma2
        self.sigma3 = helper.sigma3
        self.A = helper.A

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "lst_nid": self.lst_nid,
                    "ring_n": self.ring_n,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "bound_u": gp.digits(self.bound_u),
                    "label_modulus": self.label_modulus,
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma1": self.sigma1,
                    "sigma2": self.sigma2,
                    "sigma3": self.sigma3,
                    "A": [gp.digits(v) for v in self.A],
                },
                f,
            )

    def setup(self) -> None:
        sk = {
            nid: discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
            for nid in self.lst_nid
        }
        u = {
            nid: [
                gp.mpz(_CSPRNG.randint(-int(self.bound_u), int(self.bound_u)))
                for _ in range(self.eta)
            ]
            for nid in self.lst_nid
        }

        v = {}
        nid_selected = _CSPRNG.sample(self.lst_nid, 1)[0]
        for nid in self.lst_nid:
            if nid != nid_selected:
                v[nid] = [
                    gp.mpz(_CSPRNG.randint(-int(self.bound_u), int(self.bound_u)))
                    for _ in range(self.eta * self.n)
                ]
        lst_v = [vec for vec in v.values()]
        v[nid_selected] = [-(sum(vals)) for vals in zip(*lst_v)] if lst_v else [
            gp.mpz(0) for _ in range(self.eta * self.n)
        ]

        pk = {}
        from pyfe4ai.utils.ring_lwe_utils import poly_add, poly_mul

        for nid in self.lst_nid:
            noise = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
            pk_nid = []
            for i in range(self.eta):
                pk_i = poly_mul(self.A, sk[nid][i], self.q)
                pk_nid.append(poly_add(pk_i, noise[i], self.q))
            pk[nid] = pk_nid

        self.msk = {"sk": sk, "u": u, "v": v}
        self.mpk = {"pk": pk}
        logger.info("Decentralized MCFE Ring-LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": [gp.digits(v) for v in self.A],
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "eta": self.eta,
            "n": self.n,
            "lst_nid": self.lst_nid,
            "ring_n": self.ring_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "bound_u": gp.digits(self.bound_u),
            "label_modulus": self.label_modulus,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sigma3": self.sigma3,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid not in self.mpk["pk"]:
            return None
        return {
            "pk": [[gp.digits(v) for v in row] for row in self.mpk["pk"][nid]],
            "u": [gp.digits(v) for v in self.msk["u"][nid]],
            "sk": [[gp.digits(v) for v in row] for row in self.msk["sk"][nid]],
            "v": [gp.digits(v) for v in self.msk["v"][nid]],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not supported: clients derive functional-key shares locally.

            Args:
                sid: Session / decryption-key identifier.

            Raises:
                NotImplementedError: Always.
        """
        raise NotImplementedError(
            "{} has no central decryption-key derivation: in decentralized "
            "MCFE each client derives its key share with "
            "derive_function_decryption_key_share() and the aggregator "
            "combines them with combine_function_decryption_key_share()".format(
                type(self).__name__
            )
        )


class DecentralizedMCFERingLWE(MCFERingLWE):
    """Crypto operations for decentralized Ring-LWE-based multi-client inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for key-share derivation")
        if self.id not in fusion_weight:
            raise FEValidationError("invalid fusion weight provided")

        q = gp.mpz(self.pp["q"])
        y = fusion_weight[self.id]
        if len(y) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length for client {}".format(self.id))
        if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in y):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        y_vec = [gp.mpz(v) for v in y]
        sk_matrix = [[gp.mpz(v) for v in row] for row in self.sk["sk"]]
        sk_t = transpose(sk_matrix)
        sk_y = []
        for row in sk_t:
            acc = gp.mpz(0)
            for a_i, b_i in zip(row, y_vec):
                acc += a_i * b_i
            sk_y.append(acc % q)

        u = [gp.mpz(v) for v in self.sk["u"]]
        z_local = gp.mpz(0)
        for u_i, y_i in zip(u, y_vec):
            z_local += u_i * y_i

        ordered_fw = []
        for nid in self.pp["lst_nid"]:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            ordered_fw.extend([gp.mpz(v) for v in fusion_weight[nid]])
        v = [gp.mpz(val) for val in self.sk["v"]]
        if len(v) < len(ordered_fw):
            raise FEValidationError("invalid private key share length")
        z_pad = gp.mpz(0)
        for v_i, w_i in zip(v, ordered_fw):
            z_pad += v_i * w_i

        return {"sk_y": [gp.digits(v) for v in sk_y], "z": gp.digits(z_local + z_pad)}

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        return {
            "sk_y": {nid: share["sk_y"] for nid, share in dct_dk_shares.items()},
            "z": gp.digits(sum(gp.mpz(share["z"]) for share in dct_dk_shares.values())),
        }
