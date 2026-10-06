"""
Chotard, Jérémy, Edouard Dufour Sans, Romain Gay, Duong Hieu Phan, and David Pointcheval. 
"Decentralized multi-client functional encryption for inner product." 
In Advances in Cryptology ASIACRYPT 2018, Brisbane, QLD, Australia, December 2-6, pp. 703-732. 
Springer International Publishing, 2018.

* setting:  Integer based

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
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError


logger = logging.getLogger(__name__)


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
        """Generate master secret key (``msk``) and master public key (``mpk``)."""
        vec_a = [1, _random(self.p, self.sec_param)]
        vec_w = [
            [_random(self.p, self.sec_param), _random(self.p, self.sec_param)]
            for _ in range(self.eta)
        ]
        dct_u = {
            nid: [_random(self.p, self.sec_param) for _ in range(self.eta)]
            for nid in self.lst_nid
        }

        g_a = [gp.powmod(self.g, a, self.p) for a in vec_a]
        self.mpk = {"g": self.g, "p": self.p}
        self.msk = {"w": vec_w, "u": dct_u, "g_a": g_a}
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
        """Return encryption keys for client *nid*.

        Args:
            nid: Client identifier, must be in ``lst_nid``.

        Returns:
            Dict with ``g_a``, ``w``, ``u`` keys, or ``None`` if *nid* is invalid.
        """
        logger.debug("generating private key(s)...")
        if nid is None or not isinstance(nid, str) or nid not in self.lst_nid:
            logger.error("no id or invalid id  provided.")
            return None

        _keys = {
            "g_a": [gp.digits(i) for i in self.msk["g_a"]],
            "w": [[gp.digits(i[0]), gp.digits(i[1])] for i in self.msk["w"]],
            "u": [gp.digits(i) for i in self.msk["u"][nid]],
        }
        return _keys

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to weight lists).

        Returns:
            Dict with ``d`` (per-client keys) and ``z`` (aggregated secret).

        Raises:
            FEKeyError: If credentials are missing.
        """
        _credentials = kwargs.get("credentials", None)
        if not _credentials:
            raise FEKeyError("need to provided credentials for MCFE")
        _fusion_weights = _credentials.get("fusion_weight")
        d = {}
        z = gp.mpz(0)
        for nid, lst_fusion in _fusion_weights.items():
            if nid in self.msk["u"]:
                u_nid = self.msk["u"][nid]
                w_fusion = gp.mpz(0)
                for i in range(len(lst_fusion)):
                    wi = self.msk["w"][i]
                    w_fusion += gp.mul(wi[0] + wi[1], gp.mpz(lst_fusion[i]))
                    z += gp.mul(u_nid[i], gp.mpz(lst_fusion[i]))
                d[nid] = gp.digits(w_fusion)
            else:
                logger.error("invalid identifier in provided fusion weight.")

        return {"d": d, "z": gp.digits(z)}


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
            label: Encryption label binding the ciphertext to a session.

        Returns:
            Dict with ``t`` (group element pair) and ``c`` (ciphertext list).

        Raises:
            FEKeyError: If public parameters or private keys are missing.
            FEValidationError: If plaintext length or format is invalid.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) > len(self.sk["u"]):
            raise FEValidationError("invalid size of input plaintext:{}".format(lst_pt))
        if not isinstance(lst_pt, list):
            raise FEValidationError("invalid format of input plaintext:{}".format(lst_pt))

        sec_param = self.pp["sec_param"]
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        lst_ga = [gp.mpz(i) for i in self.sk["g_a"]]
        u = [gp.mpz(i) for i in self.sk["u"]]
        w = [[gp.mpz(lst_wi[0]), gp.mpz(lst_wi[1])] for lst_wi in self.sk["w"]]

        r = _random(p, sec_param)
        t = [gp.digits(gp.powmod(ga, r, p)) for ga in lst_ga]

        _label = md5_hash(label, p)
        c = list()
        for i in range(len(lst_pt)):
            # slot i is masked with its own key component w[i] only, matching
            # the per-slot d = sum_i y_i * (w_i0 + w_i1) used at decryption
            _ga_w = gp.mpz(1)
            for ga in lst_ga:
                _ga_w = gp.mul(_ga_w, gp.powmod(ga, w[i][0] + w[i][1], p)) % p
            _ptu = gp.powmod(g, gp.mpz(lst_pt[i]) + gp.mul(u[i], _label), p)
            c.append(gp.digits(gp.mul(_ptu, gp.powmod(_ga_w, r, p)) % p))

        return {"t": t, "c": c}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt aggregated ciphertexts to recover the multi-client inner product.

        Args:
            dct_ct: Mapping of client IDs to their ciphertext dicts.
            dk: Decryption key dict with ``d`` (per-client) and ``z``.
            fusion_weight: Mapping of client IDs to weight lists.
            label: The encryption label used during :meth:`encrypt`.

        Returns:
            The inner product as an integer, or ``None`` if out of bound.

        Raises:
            FEKeyError: If decryption key is missing.
            FESchemeError: If inputs are inconsistent.
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if dct_ct.keys() != dk.keys() and dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct, dk, fusion wight")

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        z = gp.mpz(dk["z"])
        d = dk["d"]

        _cf_prod = gp.mpz(1)
        _td_prod = gp.mpz(1)
        for nid in dct_ct.keys():
            f_nid = fusion_weight[nid]
            c_nid = dct_ct[nid]["c"]
            t_nid = [gp.mpz(_t) for _t in dct_ct[nid]["t"]]
            d_nid = gp.mpz(d[nid])

            for i in range(len(c_nid)):
                _cf_prod *= gp.powmod(gp.mpz(c_nid[i]), gp.mpz(f_nid[i]), p)
            for _t in t_nid:
                _td_prod *= gp.powmod(_t, d_nid, p)

        gf = gp.divm(
            gp.divm(_cf_prod, _td_prod, p),
            gp.powmod(g, gp.mul(z, md5_hash(label, p)), p),
            p,
        )

        return self._solve_dlog(gf)

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

        Args:
            lst_ndarray: List of numpy arrays to encrypt.
            **kwargs: Must include ``label``.

        Returns:
            List of object ndarrays containing per-element ciphertexts.
        """
        _label = kwargs.get("label", None)
        lst_ndarray_ct = list()
        for l in range(len(lst_ndarray)):
            ary = (lst_ndarray[l].copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([w], _label)
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
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, label)
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
