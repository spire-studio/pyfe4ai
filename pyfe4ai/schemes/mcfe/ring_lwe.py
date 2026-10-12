"""
Ring-LWE identity-based multi-client inner-product FE.

| Bermudo Mera, Karmakar, Marc, Soleimanian:
| "Efficient Lattice-Based Inner-Product Functional Encryption", PKC 2022
| (ePrint 2021/046) -- Section 6, "MCFE-Compiler" (from multi-input to
| identity-based multi-client IPFE, random oracle; Theorem 5, static
| corruptions) applied to the MIFE of :mod:`pyfe4ai.schemes.mife.ring_lwe`.

* type:     identity-based multi-client encryption (MCFE, Ring-LWE, random oracle)
* setting:  R_q = Z_q[x] / (x^ring_n + 1)

Construction (MIFE notation; ``H(u'_i, l) in R_q^eta`` is a hash keyed by the
client secret ``u'_i``):

* Setup:    MIFE setup plus a secret ``u'_i`` per client.
* Encrypt:  ``ct_{i,l} = MIFE.Enc(msk_i, x_i + H(u'_i, l))``, i.e. the slot
  encrypts ``x_i + u_i + H(u'_i, l)``.
* KeyGen:   for weights ``y`` and label ``l``: the MIFE key
  ``({sk_{i,y}}, z)`` with ``z'' = z + sum_i y_i H(u'_i, l)``.
* Decrypt:  ``MIFE.Dec`` with ``z''``.

Functional keys are bound to a label (identity-based MCFE): a key for
``(y, l)`` only decrypts ciphertexts produced under ``l``. Each client must
encrypt at most once per label. The ndarray helpers pack values into the
``ring_n`` coefficients of one ciphertext per chunk; chunk ``c`` of label
``l`` uses ``H(u'_i, (l, c))``, and a key for ``n_chunks`` chunks carries one
``z''`` per chunk. ``H`` is SHAKE-256 over the client secret and the encoded
label. Versions <= 0.2.0 masked inputs with a small public label scalar.
"""

from __future__ import annotations

import logging
import math
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mcfe.labels import chunk_label, encode_label
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import hash_to_zq_vector
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)

_PAD_DOMAIN = "pyfe4ai/mcfe/ring_lwe/H(u',label)"


def label_pad(u_prime: str, label, q, eta: int, ring_n: int) -> list:
    """Return ``H(u'_i, label)`` as ``eta`` rows of ``ring_n`` coefficients mod q."""
    flat = hash_to_zq_vector(
        "{}|{}".format(u_prime, encode_label(label)), q, eta * ring_n, domain=_PAD_DOMAIN
    )
    return [flat[r * ring_n:(r + 1) * ring_n] for r in range(eta)]


def chunk_key_index(layer: dict, j: int, n_keys: int) -> int:
    """Index of the chunk key for chunk *j* of an ndarray ciphertext layer.

    Raises:
        FESchemeError: Unless the layer's ``first_chunk`` is a non-negative
            integer and all its chunks have a key among the *n_keys*.
    """
    first = layer.get("first_chunk")
    n_chunks = len(layer.get("chunks", []))
    if (
        not isinstance(first, int) or isinstance(first, bool)
        or first < 0 or first + n_chunks > n_keys or not 0 <= j < n_chunks
    ):
        raise FESchemeError(
            "ciphertext chunks starting at {!r} are not covered by the {} chunk keys".format(first, n_keys)
        )
    return first + j

class MCFERingLWEKeyGenerator(MIFERingLWEKeyGenerator):
    """Key generator for BMMS21 identity-based MCFE over Ring-LWE."""

    _scheme_type = CryptoCONST.TYPE_MCFE_RING_LWE

    def setup(self) -> None:
        """MIFE setup plus a secret ``u'_i`` for every client."""
        super().setup()
        self.msk["u_prime"] = {nid: secrets.token_hex(32) for nid in self.lst_nid}

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s keys: MIFE ``pk``, ``u`` and the label-pad secret ``u_prime``."""
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
        sk["u_prime"] = self.msk["u_prime"][nid]
        return sk

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the key for weights ``y`` and label ``l`` (identity-based).

        Args:
            sid: Session identifier (unused).
            **kwargs: ``credentials`` with ``fusion_weight``, ``label`` and
                optionally ``n_chunks`` (number of ndarray chunks to cover).

        Returns:
            Dict with ``sk_y`` (per-slot MIFE keys), ``z`` (pad term of the
            label itself), ``z_chunks`` (pad terms of chunks ``0..n_chunks-1``)
            and ``label``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE Ring-LWE decryption key generation")
        label = credentials.get("label")
        if not isinstance(label, str):
            raise FEValidationError("identity-based MCFE keys need a string label")
        n_chunks = int(credentials.get("n_chunks", 0))
        fusion_weight = credentials.get("fusion_weight")
        key = super().get_decryption_keys(sid, credentials={"fusion_weight": fusion_weight})
        base = [gp.mpz(v) for v in key["z"]]

        def pad_term(lab) -> list:
            z = list(base)
            for nid in self.lst_nid:
                pad = label_pad(self.msk["u_prime"][nid], lab, self.q, self.eta, self.ring_n)
                for y_row, pad_row in zip(fusion_weight[nid], pad):
                    z = [acc + int(y_row) * h for acc, h in zip(z, pad_row)]
            return [gp.digits(v % self.q) for v in z]

        return {
            "sk_y": key["sk_y"],
            "z": pad_term(label),
            "z_chunks": [pad_term(chunk_label(label, c)) for c in range(n_chunks)],
            "label": label,
        }


class MCFERingLWE(MIFERingLWE):
    """Crypto operations for BMMS21 identity-based MCFE over Ring-LWE."""

    def _pad(self, label) -> list:
        return label_pad(
            self.sk["u_prime"], label, gp.mpz(self.pp["q"]), self.pp["eta"], self.pp["ring_n"]
        )

    def encrypt(self, matrix_pt: list[list[int]], label: str) -> dict:
        """Encrypt this client's matrix under *label* (once per client and label)."""
        rows = self._check_plaintext(matrix_pt)
        if "u_prime" not in self.sk:
            raise FEKeyError("encryption key lacks u_prime")
        if not isinstance(label, str):
            raise FEValidationError("label must be a string")
        return self._encrypt_rows(self._padded_rows(rows, extra_pad=self._pad(label)), len(rows[0]))

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Recover the batch of inner products for a key bound to *label*.

        Raises:
            FESchemeError: If the key was derived for another label or the
                ciphertexts do not cover exactly the key's clients.
        """
        if not dk or "z" not in dk:
            raise FEKeyError("no decryption key provided")
        if dk.get("label") != label:
            raise FESchemeError("the key was derived for another label")
        return self._decrypt_with_key_term(dct_ct, dk, dk["z"], fusion_weight)

    def _decrypt_with_key_term(self, dct_ct, dk, z, fusion_weight) -> list:
        if set(dct_ct) != set(dk["sk_y"]) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every client of the key")
        batch = {int(ct["k"]) for ct in dct_ct.values()}
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")
        return self._decrypt_with_pad_term(dct_ct, dk["sk_y"], z, fusion_weight, batch.pop())

    @staticmethod
    def key_chunks(lst_ndarray_ct: list) -> int:
        """Number of chunk keys needed for the output of :meth:`encrypt_lst_ndarray`."""
        return sum(len(layer["chunks"]) for layer in lst_ndarray_ct)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Pack ndarrays into ``ring_n``-coefficient chunks (``eta = 1``).

        Chunk ``c`` (counted across all layers) is encrypted under the chunk
        label ``(label, c)``, so each chunk has its own label pad.

        Returns:
            One dict per layer with ``shape``, ``first_chunk`` and ``chunks``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        ring_n = self.pp["ring_n"]
        out, chunk = [], 0
        for ary_src in lst_ndarray:
            flat = scale_to_int(ary_src, self.precision).ravel().tolist()
            n_layer = max(1, math.ceil(len(flat) / ring_n))
            cts = []
            for j in range(n_layer):
                part = flat[j * ring_n:(j + 1) * ring_n]
                self._check_plaintext([part])
                rows = self._padded_rows([part], extra_pad=self._pad(chunk_label(label, chunk + j)))
                cts.append(self._encrypt_rows(rows, len(part)))
            out.append({"shape": list(np.shape(ary_src)), "first_chunk": chunk, "chunks": cts})
            chunk += n_layer
        return out

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate-decrypt packed ndarray ciphertexts (needs ``dk``, ``fusion_weight``, ``label``)."""
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt packed ndarray ciphertexts with a key covering their chunks."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        if dk.get("label") != label:
            raise FESchemeError("the key was derived for another label")
        sample = next(iter(dict_ndarray_ct.values()))
        if self.key_chunks(sample) > len(dk.get("z_chunks", [])):
            raise FEKeyError("the key covers fewer chunks than the ciphertexts")
        result = []
        for l, layer in enumerate(sample):
            values = []
            for j in range(len(layer["chunks"])):
                dct_ct = {nid: dict_ndarray_ct[nid][l]["chunks"][j] for nid in dict_ndarray_ct}
                z = dk["z_chunks"][chunk_key_index(layer, j, len(dk["z_chunks"]))]
                values.extend(self._decrypt_with_key_term(dct_ct, dk, z, fusion_weight))
            ary = np.empty(len(values), dtype=object)
            ary[:] = values
            result.append(decoded_layer(ary.reshape(layer["shape"]), l, 10 ** self.precision))
        return result
