"""
Chotard, Jérémy, Edouard Dufour Sans, Romain Gay, Duong Hieu Phan, and David Pointcheval.
"Decentralized multi-client functional encryption for inner product."
In Advances in Cryptology ASIACRYPT 2018, Brisbane, QLD, Australia, December 2-6, pp. 703-732.
Springer International Publishing, 2018.
| URL: https://eprint.iacr.org/2017/989

* type:     secret-key multi-client encryption (MCFE, DDH, random oracle)
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

Construction (one coordinate j of client i's vector is one CDGPP18 slot):

* Setup:    ``s_{i,j} <- Z_q^2`` for every client i and coordinate j.
* Encrypt:  ``[u_l] = H(l) in G^2``;  ``c_{i,j} = g^{x_{i,j}} * [u_l]^{s_{i,j}}``.
* KeyGen:   ``d = sum_{i,j} y_{i,j} * s_{i,j} in Z_q^2`` (one aggregated key).
* Decrypt:  ``g^{<x,y>} = prod c_{i,j}^{y_{i,j}} / [u_l]^d``, then a bounded dlog.

Each client must encrypt at most once per label: two ciphertexts of the same
client under the same label reveal the difference of the plaintexts.

"""

from __future__ import annotations

import os
import json
import logging

import numpy as np
import gmpy2 as gp

from pyfe4ai.schemes.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import _random
from pyfe4ai.utils.crypto_utils import hash_to_qr_group
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError


logger = logging.getLogger(__name__)

_LABEL_HASH_DOMAIN = "pyfe4ai/mcfe/ddh/H(label)"


def _element_label(label: str, layer: int, index: tuple) -> str:
    """Unambiguous per-element label for the ndarray helpers.

    CDGPP18 ciphertexts are deterministic per (client slot, label), so every
    array element must be encrypted under its own label; reusing one label
    would let anyone compute ``c_i / c_j = g^{x_i - x_j}`` without a key.
    """
    return json.dumps([label, int(layer), [int(v) for v in index]])


def _hash_label(label: str, p: gp.mpz) -> list:
    """Return ``[u_l] = H(label) in G^2`` (random oracle into the QR subgroup)."""
    if not isinstance(label, str):
        raise FEValidationError("label must be a string, got {!r}".format(label))
    return [hash_to_qr_group(label, p, index=k, domain=_LABEL_HASH_DOMAIN) for k in (0, 1)]


class MCFEKeyGenerator(DDHKeyGeneratorBase):
    """Key generator for DDH-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MCFE
    _verify_keys = ("sec_param", "eta", "n", "s")

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MCFE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta`` (int or per-client dict), ``n`` (number of clients),
                ``s``, ``lst_nid`` (client identifiers).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)

        if isinstance(self.eta, int):
            self.dict_eta = {nid: self.eta for nid in self.lst_nid}
        elif isinstance(self.eta, dict) and len(self.eta) == self.n:
            self.dict_eta = self.eta
        else:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        self._load_parameters()

    def _extra_param_fields(self) -> dict:
        return {"n": self.n, "s": self.s}

    def setup(self) -> None:
        """Generate one independent secret ``s_{i,j} in Z_q^2`` per client slot."""
        self.msk = {
            "s": {
                nid: [
                    [_random(self.q, self.sec_param) for _ in range(2)]
                    for _ in range(self.dict_eta[nid])
                ]
                for nid in self.lst_nid
            }
        }
        self.mpk = {"g": self.g, "p": self.p}
        logger.info("MCFE setup - DONE.")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters (g, p, n, eta, sec_param).

        Returns:
            Dict of public parameters.
        """
        logger.debug("generating public parameters ...")
        return {
            "g": gp.digits(self.mpk["g"]),
            "p": gp.digits(self.mpk["p"]),
            "n": self.n,
            "eta": self.eta,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str) -> dict | None:
        """Return the encryption key of client *nid*.

        Args:
            nid: Client identifier, must be in ``lst_nid``.

        Returns:
            Dict with ``s`` (one ``Z_q^2`` pair per slot of *nid*), or ``None``
            if *nid* is invalid.
        """
        logger.debug("generating private key(s)...")
        if nid is None or not isinstance(nid, str) or nid not in self.lst_nid:
            logger.error("no id or invalid id  provided.")
            return None

        return {
            "s": [[gp.digits(e) for e in s_j] for s_j in self.msk["s"][nid]],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the functional decryption key ``d = sum y_{i,j} s_{i,j}``.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to weight lists).

        Returns:
            Dict with ``d`` (a pair of integers mod q). The key is aggregated
            over all clients, so it gives no per-client decryption capability.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If a client id is unknown or a weight list is
                longer than that client's ``eta``.
        """
        _credentials = kwargs.get("credentials", None)
        if not _credentials:
            raise FEKeyError("need to provided credentials for MCFE")
        _fusion_weights = _credentials.get("fusion_weight")
        if not isinstance(_fusion_weights, dict) or not _fusion_weights:
            raise FEValidationError("invalid fusion weights provided, need a dict")

        d = [gp.mpz(0), gp.mpz(0)]
        for nid, lst_fusion in _fusion_weights.items():
            if nid not in self.msk["s"]:
                raise FEValidationError(
                    "invalid identifier in provided fusion weight: {}".format(nid)
                )
            s_nid = self.msk["s"][nid]
            if len(lst_fusion) > len(s_nid):
                raise FEValidationError(
                    "fusion weight of {} is longer than eta".format(nid)
                )
            for y_j, s_j in zip(lst_fusion, s_nid):
                d[0] += gp.mul(gp.mpz(y_j), s_j[0])
                d[1] += gp.mul(gp.mpz(y_j), s_j[1])

        return {"d": [gp.digits(d_k % self.q) for d_k in d]}


class MCFE(IPFEAbsCrypto):
    """Crypto operations for DDH-based multi-client inner-product FE."""

    def __init__(self, config: dict) -> None:
        """Initialise the MCFE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and either ``sk`` (encryption keys) or ``dk`` (decryption key).
        """
        super().__init__(config)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        else:
            self._load_dlog_table()

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_MCFE,
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"],
                pow(10, self.precision + 2),
            )
        )

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt a plaintext vector for one client under a label.

        Args:
            lst_pt: Integer plaintext vector of length ≤ ``eta``.
            label: Encryption label binding the ciphertext to a session. A
                client must not encrypt twice under the same label.

        Returns:
            Dict with ``c`` (one group element per plaintext slot).

        Raises:
            FEKeyError: If public parameters or private keys are missing.
            FEValidationError: If plaintext length or format is invalid.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if not isinstance(lst_pt, list):
            raise FEValidationError("invalid format of input plaintext:{}".format(lst_pt))
        if len(lst_pt) > len(self.sk["s"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(lst_pt))

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        u_l = _hash_label(label, p)

        c = list()
        for x_j, s_j in zip(lst_pt, self.sk["s"]):
            c_j = gp.powmod(g, gp.mpz(x_j), p)
            for u_k, s_jk in zip(u_l, s_j):
                c_j = gp.mul(c_j, gp.powmod(u_k, gp.mpz(s_jk), p)) % p
            c.append(gp.digits(c_j))

        return {"c": c}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt aggregated ciphertexts to recover the multi-client inner product.

        Args:
            dct_ct: Mapping of client IDs to their ciphertext dicts.
            dk: Decryption key dict with ``d`` (pair of integers mod q).
            fusion_weight: Mapping of client IDs to weight lists; must be the
                weights the key was derived for.
            label: The encryption label used during :meth:`encrypt`.

        Returns:
            The inner product as an integer, or ``None`` if out of bound.

        Raises:
            FEKeyError: If decryption key is missing.
            FESchemeError: If ciphertexts and fusion weights cover different clients.
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if not isinstance(dk.get("d"), (list, tuple)) or len(dk["d"]) != 2:
            raise FEKeyError("malformed MCFE decryption key: expected d = [d0, d1]")
        if dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")

        p = gp.mpz(self.pp["p"])
        u_l = _hash_label(label, p)

        _cf_prod = gp.mpz(1)
        for nid, ct_nid in dct_ct.items():
            f_nid = fusion_weight[nid]
            c_nid = ct_nid["c"]
            if len(c_nid) != len(f_nid):
                raise FESchemeError(
                    "ciphertext and fusion weight of {} differ in length".format(nid)
                )
            for c_j, y_j in zip(c_nid, f_nid):
                c_j = gp.mpz(c_j)
                if not 0 < c_j < p:
                    raise FEValidationError("ciphertext element of {} out of range".format(nid))
                _cf_prod = gp.mul(_cf_prod, gp.powmod(c_j, gp.mpz(y_j), p)) % p

        _ud_prod = gp.mpz(1)
        for u_k, d_k in zip(u_l, dk["d"]):
            _ud_prod = gp.mul(_ud_prod, gp.powmod(u_k, gp.mpz(d_k), p)) % p

        return self._solve_dlog(gp.divm(_cf_prod, _ud_prod, p))

    def _solve_dlog(self, g_inner_prod) -> int | None:
        """Solve the discrete log using a cached dlog table."""
        try:
            return dlog_table_solve(
                g_inner_prod, self.pp["g"], self.pp["p"],
                self.bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system.")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarrays element-wise with a label.

        Each element is encrypted under its own label derived from ``label``,
        the layer and the element index (see :func:`_element_label`).

        Args:
            lst_ndarray: List of numpy arrays to encrypt.
            **kwargs: Must include ``label`` (a string).

        Returns:
            List of object ndarrays containing per-element ciphertexts.
        """
        _label = kwargs.get("label", None)
        lst_ndarray_ct = list()
        for l in range(len(lst_ndarray)):
            ary = (lst_ndarray[l].copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([w], _element_label(_label, l, i))
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def decrypt_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, dk: dict, fusion_weight: dict, label: str
    ) -> list | None:
        """Decrypt ndarray ciphertexts from multiple clients.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            dk: Decryption key dict.
            fusion_weight: Mapping of client IDs to weight lists.
            label: The encryption label.

        Returns:
            List of float ndarrays with decrypted values.
        """
        _sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = list()
        for l in range(len(_sample)):
            ary = _sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {
                    nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct.keys()
                }
                ary_dec[i] = self.decrypt(
                    dct_ct, dk, fusion_weight, _element_label(label, l, i)
                )
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))

        return lst_ndarray

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight``, and ``label``.

        Returns:
            Decrypted ndarray list.
        """
        dk = kwargs.get("dk", None)
        _fusion_weight = kwargs.get("fusion_weight", None)
        _label = kwargs.get("label", None)
        if dk is None or _fusion_weight is None or _label is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk, _fusion_weight, _label)
