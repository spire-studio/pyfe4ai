"""
LWE-Based Multi-Client Inner-Product Functional Encryption
| Multi-client label-aware organization inspired by
| "Decentralized multi-client functional encryption for inner product"
| By Jeremy Chotard, Edouard Dufour Sans, Romain Gay, Duong Hieu Phan,
| David Pointcheval
| Published in: ASIACRYPT 2018
|
| This prototype composes per-client simple LWE inner-product FE instances
| using a bounded label embedding, with the underlying LWE line taken from
| "Simple Functional Encryption Schemes for Inner Products"
| By Michel Abdalla, Florian Bourse, Angelo De Caro, David Pointcheval
| Published in: PKC 2015

* type:     public-key encryption
* setting:  Integer based
* note:     prototype implementation aligned with the existing crypto-ipfe API

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
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.lwe_utils import center_lwe_vector
from pyfe4ai.utils.lwe_utils import decode_lwe_inner_product
from pyfe4ai.utils.lwe_utils import derive_lwe_parameters
from pyfe4ai.utils.lwe_utils import label_scalar_from_hash
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matrix_to_digits
from pyfe4ai.utils.matrix_utils import matmul_mod
from pyfe4ai.utils.matrix_utils import matvec_mod
from pyfe4ai.utils.matrix_utils import transpose
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.matrix_utils import vector_to_digits
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_bit_vector
from pyfe4ai.utils.sampling_utils import rand_uniform_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FEValidationError

logger = logging.getLogger(__name__)


class MCFELWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for LWE-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MCFE_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MCFE LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta``, ``n`` (number of clients), ``lwe_n``,
                ``bound_x``, ``bound_y``, ``bound_u``, ``label_modulus``.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.lwe_n = config.get("lwe_n", 32)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        self.bound_u = gp.mpz(config.get("bound_u", 4))
        self.label_modulus = int(config.get("label_modulus", 8))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _effective_bound_x(self) -> gp.mpz:
        """Compute the effective plaintext bound including label masking."""
        return self.bound_x + self.bound_u * gp.mpz(self.label_modulus // 2)

    def _apply_parameters(self, param: dict) -> None:
        """Assign LWE parameters (p, q, m, sigma_q, l_sigma, A) from *param*."""
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma = gp.mpz(param["l_sigma"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration."""
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("lwe_n") == self.lwe_n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("bound_u") == gp.digits(self.bound_u)
            and param.get("label_modulus") == self.label_modulus
        )

    def _generate_and_save(self, param_file: str) -> None:
        """Derive LWE parameters, sample public matrix **A**, and persist."""
        eff_bound_x = self._effective_bound_x()
        self.p, self.q, self.m, self.sigma_q, self.l_sigma = derive_lwe_parameters(
            eff_bound_x,
            self.bound_y,
            self.eta,
            self.lwe_n,
            extra_dimension=self.n,
        )

        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "lwe_n": self.lwe_n,
                    "m": self.m,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "bound_u": gp.digits(self.bound_u),
                    "label_modulus": self.label_modulus,
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma_q": self.sigma_q,
                    "l_sigma": gp.digits(self.l_sigma),
                    "A": matrix_to_digits(self.A),
                },
                f,
            )

    def setup(self) -> None:
        """Generate per-client secret keys, label masks, and public keys."""
        sk = {
            nid: rand_uniform_matrix(self.lwe_n, self.eta, self.q) for nid in self.lst_nid
        }
        u = {
            nid: [
                gp.mpz(_CSPRNG.randint(-int(self.bound_u), int(self.bound_u)))
                for _ in range(self.eta)
            ]
            for nid in self.lst_nid
        }
        pk = {}
        for nid in self.lst_nid:
            noise = discrete_gaussian_matrix(self.m, self.eta, self.sigma_q)
            pk_nid = matmul_mod(self.A, sk[nid], self.q)
            for i in range(self.m):
                for j in range(self.eta):
                    pk_nid[i][j] = (pk_nid[i][j] + noise[i][j]) % self.q
            pk[nid] = pk_nid

        self.msk = {"sk": sk, "u": u}
        self.mpk = {"pk": pk}
        logger.info("MCFE LWE setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters.

        Returns:
            Dict containing ``A``, ``p``, ``q``, ``m``, ``eta``, ``n``,
            ``lwe_n``, ``bound_x``, ``bound_y``, ``bound_u``,
            ``label_modulus``, ``sec_param``.
        """
        return {
            "A": matrix_to_digits(self.A),
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "m": self.m,
            "eta": self.eta,
            "n": self.n,
            "lwe_n": self.lwe_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "bound_u": gp.digits(self.bound_u),
            "label_modulus": self.label_modulus,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return encryption keys for client *nid*.

        Args:
            nid: Client identifier.

        Returns:
            Dict with ``pk`` matrix and ``u`` label mask, or ``None`` if invalid.
        """
        if nid not in self.mpk["pk"]:
            return None
        return {
            "pk": matrix_to_digits(self.mpk["pk"][nid]),
            "u": vector_to_digits(self.msk["u"][nid]),
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to weight lists).

        Returns:
            Dict with ``sk_y`` (per-client projected secret keys) and
            ``z`` (aggregated label masking secret).

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid or exceed ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        sk_y = {}
        z = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            y_vec = [gp.mpz(v) for v in y]
            sk_y[nid] = vector_to_digits(matvec_mod(self.msk["sk"][nid], y_vec, self.q))
            for u_i, y_i in zip(self.msk["u"][nid], y_vec):
                z += u_i * y_i
        return {"sk_y": sk_y, "z": gp.digits(z)}


class MCFELWE(IPFEAbsCrypto):
    """Crypto operations for LWE-based multi-client inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MCFE LWE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = digits_to_matrix(self.pp["A"])
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt a plaintext vector for one client under a label.

        Args:
            lst_pt: Integer plaintext vector of length ``eta``.
            label: Encryption label binding the ciphertext to a session.

        Returns:
            Dict with ``ct0`` and ``ct1`` (serialised LWE ciphertext vectors).

        Raises:
            FEKeyError: If public parameters or encryption keys are missing.
            FEValidationError: If plaintext length or values are invalid.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        pk = digits_to_matrix(self.sk["pk"])
        u = digits_to_vector(self.sk["u"])
        label_scalar = label_scalar_from_hash(
            md5_hash(label, gp.mpz(self.pp["p"])), self.pp["label_modulus"]
        )
        masked = [gp.mpz(x) + u_i * label_scalar for x, u_i in zip(lst_pt, u)]

        r = rand_bit_vector(self.pp["m"])
        a_trans = transpose(self.A)
        ct0 = matvec_mod(a_trans, r, q)

        pk_trans = transpose(pk)
        ct1 = matvec_mod(pk_trans, r, q)
        centered = center_lwe_vector(masked, gp.mpz(self.pp["p"]), q)
        ct1 = [(v + c) % q for v, c in zip(ct1, centered)]

        return {"ct0": vector_to_digits(ct0), "ct1": vector_to_digits(ct1)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt aggregated ciphertexts to recover the multi-client inner product.

        Args:
            dct_ct: Mapping of client IDs to their ciphertext dicts.
            dk: Decryption key dict with ``sk_y`` (per-client) and ``z``.
            fusion_weight: Mapping of client IDs to weight lists.
            label: The encryption label used during :meth:`encrypt`.

        Returns:
            The inner product as an integer.

        Raises:
            FEKeyError: If decryption key is missing.
            FEValidationError: If fusion weight length or values are invalid.
        """
        if not dk:
            raise FEKeyError("no decryption key provided")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        total = 0
        for nid in dct_ct:
            y = fusion_weight[nid]
            if len(y) != self.pp["eta"]:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(int(v)) > gp.mpz(self.pp["bound_y"]) for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            y_vec = [gp.mpz(v) for v in y]
            ct0 = digits_to_vector(dct_ct[nid]["ct0"])
            ct1 = digits_to_vector(dct_ct[nid]["ct1"])
            sk_y = digits_to_vector(dk["sk_y"][nid])

            d = (vecdot_mod(y_vec, ct1, q) - vecdot_mod(ct0, sk_y, q)) % q
            total += decode_lwe_inner_product(d, p, q)

        label_scalar = label_scalar_from_hash(
            md5_hash(label, p), self.pp["label_modulus"]
        )
        total -= int(gp.mpz(dk["z"]) * label_scalar)
        return total

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarrays element-wise with a label (requires ``eta=1``).

        Args:
            lst_ndarray: List of numpy arrays to encrypt.
            **kwargs: Must include ``label``.

        Returns:
            List of object ndarrays containing per-element ciphertexts.
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
        """Aggregate ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight``, and ``label``.

        Returns:
            Decrypted ndarray list.
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
        """Decrypt ndarray ciphertexts from multiple clients.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight``, and ``label``.

        Returns:
            List of float ndarrays with decrypted values.
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
