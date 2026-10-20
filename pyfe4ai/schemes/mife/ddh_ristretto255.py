"""
Multi Input Functional Encryption (DDH) over ristretto255
| From "Multi-Input Functional Encryption for Inner Products:
|          Function-Hiding Realizations and Constructions without Pairings"
| Published in: CRYPTO 2018
| By: Michel Abdalla, Dario Catalano, Dario Fiore, Romain Gay, and Bogdan Ursu
| URL: https://eprint.iacr.org/2017/972.pdf -- Fig. 4, instantiated with the
| DDH IPFE of Abdalla, Bourse, De Caro, Pointcheval (PKC 2015, Construction 1)
| in every slot.

* type:     private-key multi-input encryption
* setting:  the prime-order group ristretto255 (RFC 9496), written additively

As :mod:`pyfe4ai.schemes.mife.ddh` in ristretto255. Slot ``i`` has an IPFE
key ``s_i in Z_L^eta`` (``H_i = s_i G``) and a one-time pad ``u_i in Z_L^eta``;
client ``i`` holds ``(H_i, u_i)``.

* Enc:      ``ct_i = (r G, (x_i + u_i) G + r H_i)``.
* KeyGen:   ``d_i = <s_i, y_i> mod L`` per slot and ``z = sum_i <u_i, y_i> mod L``.
* Dec:      ``sum_i (sum_j y_{i,j} c_{i,j} - d_i t_i) - z G = <x, y> G``, then a
  discrete log bounded by ``dlog_bound``.

As in every MIFE there are no labels: clients must not reuse a key epoch
across independent inputs (use MCFE for repeated aggregation).
"""

from __future__ import annotations

import logging

import numpy as np

from pyfe4ai.schemes.common.ec_ddh_base import ECDDHCryptoBase, ECDDHKeyGeneratorBase
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.schemes.threshold.binding import check_fusion_weight, fusion_weight_id
from pyfe4ai.primitives.groups import ec_group as eg
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.formats.dl import MIFEDDHCiphertext
from pyfe4ai.core.formats.dl import MIFEDDHEncryptionKey
from pyfe4ai.core.formats.dl import MIFEDDHFunctionalKey
from pyfe4ai.core.formats.dl import RistrettoParameters
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.core.versions import MIFE_DDH_RISTRETTO255_CONSTRUCTION
from pyfe4ai._validation import int_vector, scale_to_int

logger = logging.getLogger(__name__)


class MIFEDDHRistretto255KeyGenerator(ECDDHKeyGeneratorBase):
    """Key generator for DDH-based multi-input inner-product FE over ristretto255."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Check the configuration.

        Args:
            config: ``n`` (number of clients), ``lst_nid``, ``eta`` (int or
                per-client dict), ``group`` (optional, ``"ristretto255"``).
        """
        super().__init__(config, **kwargs)
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        if isinstance(self.eta, int) and not isinstance(self.eta, bool) and self.eta >= 1:
            self.dict_eta = {nid: self.eta for nid in self.lst_nid}
        elif isinstance(self.eta, dict) and set(self.eta) == set(self.lst_nid):
            self.dict_eta = self.eta
        else:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))

    def setup(self) -> None:
        """Sample an IPFE key ``s_i`` and a pad ``u_i`` for every slot."""
        self.msk = {
            key: {nid: [eg.random_scalar() for _ in range(self.dict_eta[nid])] for nid in self.lst_nid}
            for key in ("s", "u")
        }
        logger.info("MIFE (ristretto255) setup - DONE.")

    def get_public_parameters(self) -> dict:
        return RistrettoParameters.new(
            construction=MIFE_DDH_RISTRETTO255_CONSTRUCTION,
            group=eg.GROUP,
            n=self.n,
            eta=self.eta,
            lst_nid=list(self.lst_nid),
        ).to_dict()

    def get_private_keys(self, nid: str) -> dict | None:
        """Client *nid*'s encryption key ``{h, u}`` (``None`` if unknown)."""
        if nid not in self.lst_nid:
            logger.error("no id or invalid id provided.")
            warn_at_caller("no id or invalid id provided.")
            return None
        return MIFEDDHEncryptionKey.new(
            h=[eg.to_hex(eg.mul_base(s)) for s in self.msk["s"][nid]],
            u=[eg.scalar_str(u) for u in self.msk["u"][nid]],
        ).to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """``{d, z, weights_id}`` for weights covering every slot."""
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need to provided credentials for MIFE")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or set(fusion_weight) != set(self.lst_nid):
            raise FEValidationError("fusion weights must cover every client")
        d, z = {}, 0
        for nid in self.lst_nid:
            y = int_vector(fusion_weight[nid], "fusion weight of " + nid, length=self.dict_eta[nid])
            d[nid] = eg.scalar_str(sum(s * y_j for s, y_j in zip(self.msk["s"][nid], y)))
            z += sum(u * y_j for u, y_j in zip(self.msk["u"][nid], y))
        return MIFEDDHFunctionalKey.new(d=d, z=eg.scalar_str(z), weights_id=fusion_weight_id(fusion_weight)).to_dict()


class MIFEDDHRistretto255(ECDDHCryptoBase):
    """Crypto operations for DDH-based multi-input inner-product FE over ristretto255."""

    def encrypt(self, lst_pt: list) -> dict:
        """``{t: r G, c: [(x_j + u_j) G + r H_j]}`` for this client's ``eta`` integers."""
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        x = int_vector(lst_pt, "plaintext", length=len(MIFEDDHEncryptionKey(self.sk).u))
        r = eg.random_scalar()
        c = []
        sk = MIFEDDHEncryptionKey.from_dict(self.sk)
        for x_j, u_j, h_j in zip(x, sk.u, sk.h):
            point = eg.add(eg.mul_base(x_j + eg.scalar_from(u_j, "encryption key")),
                           eg.mul(eg.from_hex(h_j, "encryption key"), r))
            c.append(eg.to_hex(point))
        return MIFEDDHCiphertext.new(t=eg.to_hex(eg.mul_base(r)), c=c).to_dict()

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """``sum_i <x_i, y_i>``, or ``None`` if beyond ``dlog_bound``.

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
        acc = eg.neg(eg.mul_base(eg.scalar_from(key.z, "decryption key")))
        for nid, ct in dct_ct.items():
            y = int_vector(fusion_weight[nid], "fusion weight of " + nid)
            try:
                c = [eg.from_hex(v) for v in MIFEDDHCiphertext(ct).c]
                t = eg.from_hex(MIFEDDHCiphertext(ct).t)
            except (FEValidationError, KeyError, TypeError) as exc:
                raise FEValidationError("malformed ciphertext of {}".format(nid)) from exc
            if len(c) != len(y):
                raise FEValidationError("malformed ciphertext of {}".format(nid))
            for c_j, y_j in zip(c, y):
                acc = eg.add(acc, eg.mul(c_j, y_j))
            acc = eg.sub(acc, eg.mul(t, eg.scalar_from(key.d[nid], "decryption key")))
        return self._solve_dlog(acc)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``); one input per key epoch."""
        if len(MIFEDDHEncryptionKey(self.sk).u) != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        out = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            out.append(ary_ct)
        return out

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, dk: dict, fusion_weight: dict) -> list | None:
        sample = next(iter(dict_ndarray_ct.values()))
        out = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                ary_dec[i] = self.decrypt({nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct},
                                          dk, fusion_weight)
            out.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return out

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk, fusion_weight)
