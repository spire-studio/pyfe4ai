"""
Multi-client inner-product FE (DDH) over ristretto255
| Chotard, Dufour Sans, Gay, Phan, Pointcheval:
| "Decentralized multi-client functional encryption for inner product",
| ASIACRYPT 2018, https://eprint.iacr.org/2017/989 -- the MCFE of Sec. 4.

* type:     secret-key multi-client encryption (MCFE, DDH, random oracle)
* setting:  the prime-order group ristretto255 (RFC 9496), written additively

As :mod:`pyfe4ai.schemes.mcfe.ddh` in ristretto255 (one coordinate ``j`` of
client ``i`` is one CDGPP18 slot):

* Setup:    ``s_{i,j} <- Z_L^2``.
* Encrypt:  ``[u_l] = H(l) in G^2``; ``c_{i,j} = x_{i,j} G + <s_{i,j}, [u_l]>``.
* KeyGen:   ``d = sum_{i,j} y_{i,j} s_{i,j} in Z_L^2`` (one aggregated key).
* Decrypt:  ``sum y_{i,j} c_{i,j} - <d, [u_l]> = <x, y> G``, then a bounded dlog.

``H(l)_k = hash_to_ristretto255(k || encode_label(l))`` (RFC 9380, DST
``PYFE4AI-MCFE-DDH-H(label)-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_``).
Each client must encrypt at most once per label; the ndarray helpers derive
one label per element.
"""

from __future__ import annotations

import logging

import numpy as np

from pyfe4ai.schemes.common.ec_ddh_base import ECDDHCryptoBase, ECDDHKeyGeneratorBase
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.primitives.groups import ec_group as eg
from pyfe4ai import backend as native_backend
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai._validation import int_vector, scale_to_int

logger = logging.getLogger(__name__)

_LABEL_DST = eg.dst("MCFE-DDH-H(label)")


def _k(k) -> bytes:
    return (int(k) % eg.ORDER).to_bytes(32, "little")


def hash_label(label) -> list:
    """``[u_l] = H(label) in G^2`` (random oracle into ristretto255)."""
    encoded = encode_label(label)
    return [eg.hash_to_group(encoded, _LABEL_DST, index=k) for k in (0, 1)]


class MCFERistretto255KeyGenerator(ECDDHKeyGeneratorBase):
    """Key generator for DDH-based multi-client inner-product FE over ristretto255."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Check the configuration.

        Args:
            config: ``n`` (number of clients), ``lst_nid``, ``eta`` (int or
                per-client dict), ``group`` (optional, ``"ristretto255"``).
        """
        super().__init__(config, **kwargs)
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        if isinstance(self.eta, int) and not isinstance(self.eta, bool) and self.eta >= 1:
            self.dict_eta = {nid: self.eta for nid in self.lst_nid}
        elif isinstance(self.eta, dict) and set(self.eta) == set(self.lst_nid):
            self.dict_eta = self.eta
        else:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))

    def setup(self) -> None:
        """One secret ``s_{i,j} in Z_L^2`` per client slot."""
        self.msk = {
            "s": {
                nid: [[eg.random_scalar(), eg.random_scalar()] for _ in range(self.dict_eta[nid])]
                for nid in self.lst_nid
            }
        }
        logger.info("MCFE (ristretto255) setup - DONE.")

    def get_public_parameters(self) -> dict:
        return {"group": eg.GROUP, "n": self.n, "eta": self.eta}

    def get_private_keys(self, nid: str) -> dict | None:
        """Client *nid*'s key ``{s}`` (one ``Z_L^2`` pair per slot), ``None`` if unknown."""
        if not isinstance(nid, str) or nid not in self.lst_nid:
            logger.error("no id or invalid id  provided.")
            return None
        return {"s": [[eg.scalar_str(e) for e in s_j] for s_j in self.msk["s"][nid]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """``{d: sum y_{i,j} s_{i,j}}`` (a pair of scalars), aggregated over all clients."""
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need to provided credentials for MCFE")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or not fusion_weight:
            raise FEValidationError("invalid fusion weights provided, need a dict")
        d = [0, 0]
        for nid, weights in fusion_weight.items():
            if nid not in self.msk["s"]:
                raise FEValidationError("invalid identifier in provided fusion weight: {}".format(nid))
            s_nid = self.msk["s"][nid]
            if len(weights) > len(s_nid):
                raise FEValidationError("fusion weight of {} is longer than eta".format(nid))
            for y_j, s_j in zip(int_vector(weights, "fusion weight of {}".format(nid)), s_nid):
                d[0] += y_j * s_j[0]
                d[1] += y_j * s_j[1]
        return {"d": [eg.scalar_str(d_k) for d_k in d]}


class MCFERistretto255(ECDDHCryptoBase):
    """Crypto operations for DDH-based multi-client inner-product FE over ristretto255."""

    def _key(self) -> list:
        cache = self.__dict__.get("_s_cache")
        if cache is None or cache[0] is not self.sk["s"]:
            cache = (self.sk["s"], [[eg.scalar_from(v, "encryption key") for v in s_j] for s_j in self.sk["s"]])
            self._s_cache = cache
        return cache[1]

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """``{c: [x_j G + <s_j, H(label)>]}``; a client encrypts at most once per label."""
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if not isinstance(lst_pt, list):
            raise FEValidationError("input plaintext must be a list")
        if len(lst_pt) > len(self.sk["s"]):
            raise FEValidationError("input plaintext has {} values, more than eta".format(len(lst_pt)))
        u = hash_label(label)
        c = []
        for x_j, s_j in zip(int_vector(lst_pt, "plaintext"), self._key()):
            point = eg.add(eg.mul_base(x_j), eg.add(eg.mul(u[0], s_j[0]), eg.mul(u[1], s_j[1])))
            c.append(eg.to_hex(point))
        return {"c": c}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """``sum_i <x_i, y_i>`` under *label*, or ``None`` if beyond ``dlog_bound``.

        Raises:
            FEKeyError: If the decryption key is missing or malformed.
            FESchemeError: If ciphertexts and weights cover different clients.
            FEValidationError: If a ciphertext is malformed.
        """
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if not isinstance(dk.get("d"), (list, tuple)) or len(dk["d"]) != 2:
            raise FEKeyError("malformed MCFE decryption key: expected d = [d0, d1]")
        if dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")
        acc = eg.IDENTITY
        for nid, ct in dct_ct.items():
            weights = fusion_weight[nid]
            c = ct["c"]
            if len(c) != len(weights):
                raise FESchemeError("ciphertext and fusion weight of {} differ in length".format(nid))
            for c_j, y_j in zip(c, int_vector(weights, "fusion weight of {}".format(nid))):
                acc = eg.add(acc, eg.mul(eg.from_hex(c_j, "ciphertext element of {}".format(nid)), y_j))
        u = hash_label(label)
        d = [eg.scalar_from(v, "decryption key") for v in dk["d"]]
        acc = eg.sub(acc, eg.add(eg.mul(u[0], d[0]), eg.mul(u[1], d[1])))
        return self._solve_dlog(acc)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise, every element under its own label."""
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        out = []
        for l, ary_src in enumerate(lst_ndarray):
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            ext = native_backend.ec_ext()
            if (ext is not None and ary.size and self._has_public_parameters() and self._has_private_keys()
                    and len(self.sk["s"]) >= 1):
                # encrypt() of every element in one native call
                s0, s1 = self._key()[0]
                cts = ext.r255_mcfe_encrypt(
                    [encode_label(_element_label(label, l, i)).encode("utf-8") for i in np.ndindex(ary.shape)],
                    _LABEL_DST, [int(w) for w in ary.ravel()], _k(s0), _k(s1), native_backend.get_threads(),
                )
                for i, c in zip(np.ndindex(ary.shape), cts):
                    ary_ct[i] = {"c": [c]}
            else:
                for i, w in np.ndenumerate(ary):
                    ary_ct[i] = self.encrypt([int(w)], _element_label(label, l, i))
            out.append(ary_ct)
        return out

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, dk: dict, fusion_weight: dict, label: str) -> list | None:
        sample = next(iter(dict_ndarray_ct.values()))
        out = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            native = self._decrypt_layer_native(dict_ndarray_ct, dk, fusion_weight, label, l, sample[l])
            if native is not None:
                out.append(native)
                continue
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, _element_label(label, l, i))
            out.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return out

    def _decrypt_layer_native(self, dict_ndarray_ct, dk, fusion_weight, label, l, layer):
        """decrypt() of every element of layer *l* in one native call, with its
        checks; ``None`` when the Python path applies (no Rust group, or
        layouts other than one value per client and element)."""
        ext = native_backend.ec_ext()
        if ext is None or not np.size(layer):
            return None
        if not dk:
            raise FEKeyError("no decryption key provided.")
        if not isinstance(dk.get("d"), (list, tuple)) or len(dk["d"]) != 2:
            raise FEKeyError("malformed MCFE decryption key: expected d = [d0, d1]")
        if dict_ndarray_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")
        nids = list(dict_ndarray_ct)
        ys = []
        for nid in nids:
            weights = fusion_weight[nid]
            if len(weights) != 1:
                return None
            ys.append(int_vector(weights, "fusion weight of {}".format(nid))[0])
        index = list(np.ndindex(np.shape(layer)))
        try:
            cts = [[dict_ndarray_ct[nid][l][i]["c"] for i in index] for nid in nids]
        except (KeyError, TypeError, IndexError):
            return None
        if any(not isinstance(c, list) or len(c) != 1 or not isinstance(c[0], str) for col in cts for c in col):
            return None
        d = [eg.scalar_from(v, "decryption key") for v in dk["d"]]
        try:
            data, mask = ext.r255_mcfe_decrypt(
                [[c[0] for c in col] for col in cts], ys, _k(d[0]), _k(d[1]),
                [encode_label(_element_label(label, l, i)).encode("utf-8") for i in index], _LABEL_DST,
                self.dlog_bound, native_backend.get_threads(),
            )
        except ValueError as exc:
            if exc.args[:1] == ("malformed ciphertext",):
                raise FEValidationError("malformed ciphertext element of {}".format(nids[exc.args[1]])) from exc
            raise
        values = np.frombuffer(data, dtype=np.int64).tolist()
        ary = np.empty(len(values), dtype=object)
        ary[:] = values
        if mask is not None:
            ary[np.frombuffer(mask, dtype=np.uint8).astype(bool)] = None
        return decoded_layer(ary.reshape(np.shape(layer)), l, 10 ** self.precision)

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk, fusion_weight, label)
