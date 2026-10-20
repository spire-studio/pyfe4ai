"""
Multi Input Functional Encryption (DDH)
| From "Multi-Input Functional Encryption for Inner Products:
|          Function-Hiding Realizations and Constructions without Pairings"
| Published in: CRYPTO 2018
| By: Michel Abdalla, Dario Catalano, Dario Fiore, Romain Gay, and Bogdan Ursu
| URL: https://eprint.iacr.org/2017/972.pdf -- Fig. 4, instantiated with the
| DDH IPFE of Abdalla, Bourse, De Caro, Pointcheval (PKC 2015, Construction 1)
| in every slot.

* type:     private-key multi-input encryption
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

Slot ``i`` has an independent IPFE key ``s_i in Z_q^eta`` (``h_i = g^{s_i}``)
and a one-time pad ``u_i in Z_q^eta``; client ``i`` holds ``(h_i, u_i)``.

* Enc:      ``ct_i = (g^r, g^{x_i + u_i} h_i^r)`` (every element reduced mod p).
* KeyGen:   ``d_i = <s_i, y_i> mod q`` per slot and ``z = sum_i <u_i, y_i> mod q``.
* Dec:      ``prod_i (prod_j c_{i,j}^{y_{i,j}} / t_i^{d_i}) / g^z = g^{<x, y>}``,
  then a discrete log bounded by ``dlog_bound``.

Each slot's partial result is ``g^{<x_i + u_i, y_i>}``: the pad hides
``<x_i, y_i>``, only the sum over all slots is revealed. As in every MIFE,
there are no labels -- a key holder may combine any ciphertexts of each
slot, so clients must not reuse a key epoch across independent inputs
(use MCFE for repeated aggregation). Deviation from ACFGU18, which uses the
adaptively secure ALS16 IPFE: ABDP15 is selectively secure. Versions <= 0.2.0
did not reduce ciphertexts modulo ``p`` (their factors revealed ``x + u``)
nor keys modulo ``q`` (a key with a large weight revealed the pads).
"""

from __future__ import annotations

import logging
import os
import random

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.common.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.one_time import OfflineRandomMixin
from pyfe4ai.schemes.threshold.binding import check_fusion_weight
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.groups.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.core.formats.dl import MIFEDDHCiphertext
from pyfe4ai.core.formats.dl import MIFEDDHEncryptionKey
from pyfe4ai.core.formats.dl import MIFEDDHFunctionalKey
from pyfe4ai.core.formats.dl import MIFEDDHParameters
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.core.versions import MIFE_DDH_CONSTRUCTION
from pyfe4ai._validation import int_vector
from pyfe4ai._validation import scale_to_int

logger = logging.getLogger(__name__)

_CSPRNG = random.SystemRandom()


class MIFEKeyGenerator(DDHKeyGeneratorBase):
    """Key generator for DDH-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE
    _verify_keys = ("sec_param", "eta", "n", "s")

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta`` (int or per-client dict), ``n`` (number of clients),
                ``s``, ``lst_nid`` (client identifiers).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        if isinstance(self.eta, int):
            self.dict_eta = {nid: self.eta for nid in self.lst_nid}
        elif isinstance(self.eta, dict) and set(self.eta) == set(self.lst_nid):
            self.dict_eta = self.eta
        else:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        self._load_parameters()

    def _extra_param_fields(self) -> dict:
        return {"n": self.n, "s": self.s}

    def _zq(self) -> gp.mpz:
        return gp.mpz(_CSPRNG.randrange(int(self.q)))

    def setup(self) -> None:
        """Sample an IPFE key ``s_i`` and a pad ``u_i`` for every slot."""
        self.msk = {
            "s": {nid: [self._zq() for _ in range(self.dict_eta[nid])] for nid in self.lst_nid},
            "u": {nid: [self._zq() for _ in range(self.dict_eta[nid])] for nid in self.lst_nid},
        }
        self.mpk = {"g": self.g, "p": self.p}
        logger.info("MIFE setup - DONE.")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters (g, p, q, n, eta, sec_param)."""
        return MIFEDDHParameters.new(
            construction=MIFE_DDH_CONSTRUCTION,
            g=gp.digits(self.g),
            p=gp.digits(self.p),
            q=gp.digits(self.q),
            n=self.n,
            eta=self.eta,
            lst_nid=list(self.lst_nid),
            sec_param=self.sec_param,
        ).to_dict()

    def get_private_keys(self, nid: str) -> dict | None:
        """Return client *nid*'s encryption key ``{h, u}`` (``None`` if unknown)."""
        if nid not in self.lst_nid:
            logger.error("no id or invalid id provided.")
            warn_at_caller("no id or invalid id provided.")
            return None
        return MIFEDDHEncryptionKey.new(
            h=[gp.digits(gp.powmod(self.g, s, self.p)) for s in self.msk["s"][nid]],
            u=[gp.digits(u) for u in self.msk["u"][nid]],
        ).to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the functional key for weights ``y`` covering every slot.

        Args:
            sid: Session identifier (unused).
            **kwargs: ``credentials`` with ``fusion_weight`` (dict mapping every
                client ID to ``eta`` integer weights).

        Returns:
            Dict with ``d`` (``<s_i, y_i> mod q`` per slot), ``z`` and
            ``weights_id``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need to provided credentials for MIFE")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or set(fusion_weight) != set(self.lst_nid):
            raise FEValidationError("fusion weights must cover every client")
        d, z = {}, gp.mpz(0)
        for nid in self.lst_nid:
            y = int_vector(fusion_weight[nid], "fusion weight of " + nid, length=self.dict_eta[nid])
            d[nid] = gp.digits(sum(s * y_j for s, y_j in zip(self.msk["s"][nid], y)) % self.q)
            z += sum(u * y_j for u, y_j in zip(self.msk["u"][nid], y))
        return MIFEDDHFunctionalKey.new(
            d=d, z=gp.digits(z % self.q), weights_id=fusion_weight_id(fusion_weight)
        ).to_dict()


class MIFE(OfflineRandomMixin, IPFEAbsCrypto):
    """Crypto operations for DDH-based multi-input inner-product FE."""

    def __init__(self, config: dict) -> None:
        """Initialise the MIFE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` and, for clients, ``sk``;
                ``dlog_bound`` (default ``10^(precision + 2)``) bounds the
                inner products decryption recovers.
        """
        super().__init__(config)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.dlog_bound = int(config.get("dlog_bound", pow(10, self.precision + 2)))
        self.dlog_table = None

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_MIFE, f"dlog_{self.dlog_bound}.json"
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(_dlog_file, self.pp["g"], self.pp["p"], self.dlog_bound)
        )

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt this client's ``eta`` integers: ``(g^r, g^{x_j + u_j} h_j^r)``.

        Raises:
            FEKeyError: If public parameters or private keys are missing.
            FEValidationError: If the plaintext is not ``eta`` integers.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        x = int_vector(lst_pt, "plaintext", length=len(MIFEDDHEncryptionKey(self.sk).u))
        return self._encrypt_with(x, self._random_part())

    def _random_part(self) -> tuple:
        """``(g^r, [h_j^r])`` mod p for a fresh r (see
        :mod:`pyfe4ai.core.one_time`)."""
        p = gp.mpz(self.pp["p"])
        r = gp.mpz(_CSPRNG.randrange(int(gp.mpz(self.pp["q"]))))
        h = MIFEDDHEncryptionKey(self.sk).h
        return gp.powmod(gp.mpz(self.pp["g"]), r, p), [gp.powmod(gp.mpz(h_j), r, p) for h_j in h]

    def _encrypt_with(self, lst_pt: list, part: tuple) -> dict:
        """``(g^r, g^{x_j + u_j} h_j^r)`` for the randomness part *part*;
        ``g^{x_j + u_j} = g^{x_j} g^{u_j}`` with ``g^{u_j}`` cached per key."""
        x = int_vector(lst_pt, "plaintext", length=len(MIFEDDHEncryptionKey(self.sk).u))
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        cached = self.__dict__.get("_g_u")
        if cached is None or cached[0] is not self.sk:
            cached = self._g_u = (self.sk, [gp.powmod(g, gp.mpz(u_j), p) for u_j in MIFEDDHEncryptionKey(self.sk).u])
        t, masks = part
        c = [gp.digits(gp.powmod(g, x_j, p) * g_u % p * m % p) for x_j, g_u, m in zip(x, cached[1], masks)]
        return MIFEDDHCiphertext.new(t=gp.digits(t), c=c).to_dict()

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Recover ``sum_i <x_i, y_i>`` from one ciphertext per slot.

        Returns:
            The inner product, or ``None`` if it is out of the range the
            discrete-log table covers (at least ``|<x, y>| <= dlog_bound``).

        Raises:
            FEKeyError: If the decryption key is missing.
            FESchemeError: If ciphertexts, key and weights do not cover the
                same slots or the weights are not the key's.
            FEValidationError: If a ciphertext is malformed.
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        check_fusion_weight(dk, fusion_weight)
        key = MIFEDDHFunctionalKey.from_dict(dk)
        if not set(dct_ct) == set(key.d) == set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")
        p = gp.mpz(self.pp["p"])
        acc = gp.powmod(gp.mpz(self.pp["g"]), -gp.mpz(key.z), p)
        for nid, ct in dct_ct.items():
            y = int_vector(fusion_weight[nid], "fusion weight of " + nid)
            c = [gp.mpz(v) for v in MIFEDDHCiphertext(ct).c]
            t = gp.mpz(MIFEDDHCiphertext(ct).t)
            if len(c) != len(y) or not all(0 < v < p and gp.legendre(v, p) == 1 for v in c + [t]):
                raise FEValidationError("malformed ciphertext of {}".format(nid))
            for c_j, y_j in zip(c, y):
                acc = acc * gp.powmod(c_j, y_j, p) % p
            acc = acc * gp.powmod(t, -gp.mpz(key.d[nid]), p) % p
        return self._solve_dlog(acc)

    def _solve_dlog(self, g_inner_prod) -> int | None:
        """Solve the discrete log using a cached dlog table."""
        if self.dlog_table is None:
            self._load_dlog_table()
        try:
            return dlog_table_solve(
                g_inner_prod, self.pp["g"], self.pp["p"],
                self.bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system.")
            warn_at_caller("inner-product is out of bound supported by crypto system.")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``).

        Every element is a separate ciphertext under the same slot key, so a
        holder of one functional key can combine elements of different
        positions (MIFE has no labels): use these helpers for one input per
        key epoch, and MCFE for federated aggregation.
        """
        if len(MIFEDDHEncryptionKey(self.sk).u) != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if kwargs.get("precomputed") is not None:
            return self._encrypt_precomputed(lst_ndarray, kwargs["precomputed"])
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def decrypt_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, dk: dict, fusion_weight: dict
    ) -> list | None:
        """Decrypt ndarray ciphertexts from all clients element-wise."""
        _sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(_sample)):
            ary_dec = np.empty(_sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(_sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`)."""
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk, fusion_weight)
