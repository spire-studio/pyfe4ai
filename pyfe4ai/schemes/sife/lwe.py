"""
LWE-Based Single-Input Inner-Product Functional Encryption
| From "Simple Functional Encryption Schemes for Inner Products"
| By Michel Abdalla, Florian Bourse, Angelo De Caro, David Pointcheval
| Published in: PKC 2015
|
| This module implements the generic simple FE construction instantiated
| under the LWE assumption as a research-oriented Python prototype.

* type:     public-key encryption
* setting:  Integer based
* note:     prototype implementation aligned with the existing crypto-ipfe API

The public matrix ``A`` is published as a seed and expanded with SHAKE-128;
encryption computes ``A^T r`` and ``pk^T r`` exactly with numpy, either over
an ``A`` kept in memory or expanded for every batch (``matrix_mode``). See
:mod:`pyfe4ai.schemes.abdp15_lwe`, which also covers the ``standard``
parameters. Versions <= 0.2.0 published ``A`` entry by entry.
"""

from __future__ import annotations

import logging
import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.abdp15_lwe import ENCRYPT_BATCH
from pyfe4ai.schemes.abdp15_lwe import ABDP15CryptoMixin
from pyfe4ai.schemes.abdp15_lwe import ABDP15KeyGeneratorMixin
from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import center_lwe_vector
from pyfe4ai.utils.lwe_utils import decode_lwe_inner_product
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matrix_to_digits
from pyfe4ai.utils.matrix_utils import matvec_mod
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.matrix_utils import vector_to_digits
from pyfe4ai.utils.sampling_utils import rand_uniform_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)

class SIFELWEKeyGenerator(IPFEAbsKeyGenerator, ABDP15KeyGeneratorMixin):
    """Key generator for LWE-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta``, ``lwe_n`` (default 32; ``"auto"`` for the smallest
                dimension meeting the 128-bit table, the ``standard``
                profile's choice), ``bound_x``, ``bound_y`` and
                ``matrix_mode`` (``"memory"`` or ``"stream"``, see
                :mod:`pyfe4ai.schemes.abdp15_lwe`).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        self._init_abdp15(config)

    def setup(self) -> None:
        """Generate master secret key (``msk``) and master public key (``mpk``)."""
        sk = rand_uniform_matrix(self.lwe_n, self.eta, self.q)
        self.msk = {"sk": sk}
        self.mpk = {"pk": self._public_keys([sk])[0]}
        logger.info("SIFE LWE setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters.

        Returns:
            Dict containing ``A_seed``, ``m``, ``lwe_n``, ``p``, ``q``,
            ``eta``, ``bound_x``, ``bound_y``, ``sec_param``.
        """
        return {
            **self._matrix_parameters(),
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "eta": self.eta,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return encryption keys (public-key matrix ``pk``).

        Args:
            nid: Client identifier (unused for single-input schemes).

        Returns:
            Dict with ``pk`` matrix.
        """
        return {"pk": matrix_to_digits(self.mpk["pk"])}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of int).

        Returns:
            Dict with ``sk_y`` (the projected secret key).

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid or exceed ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE LWE decryption key generation")
        y = int_vector(
            credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=self.bound_y
        )
        sk_y = matvec_mod(self.msk["sk"], [gp.mpz(v) for v in y], self.q)
        return {"sk_y": vector_to_digits(sk_y), "y": y}


class SIFELWE(IPFEAbsCrypto, ABDP15CryptoMixin):
    """Crypto operations for LWE-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE LWE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and optionally ``sk`` (encryption keys); optional
                ``matrix_mode`` (``"memory"`` or ``"stream"``).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self._init_abdp15(config)

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector under the LWE-based scheme.

        Args:
            lst_pt: Integer plaintext vector of length ``eta``.

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
        lst_pt = int_vector(
            lst_pt, "plaintext", length=self.pp["eta"], bound=int(self.pp["bound_x"])
        )
        return self._encrypt_batch([lst_pt])[0]

    def _encrypt_batch(self, plaintexts: list) -> list:
        """Ciphertexts ``(A^T r, pk^T r + floor-encoded x)`` of validated plaintexts."""
        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        ct0s, masks = self._masks(len(plaintexts))
        out = []
        for x, ct0, mask in zip(plaintexts, ct0s, masks):
            centered = center_lwe_vector(x, p, q)
            out.append({
                "ct0": vector_to_digits([gp.mpz(v) for v in ct0]),
                "ct1": vector_to_digits([(gp.mpz(v) + c) % q for v, c in zip(mask, centered)]),
            })
        return out

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt a ciphertext to recover the inner product ⟨x, y⟩.

        Args:
            dct_ct: Ciphertext dict from :meth:`encrypt`.
            dk: Decryption key dict from :meth:`SIFELWEKeyGenerator.get_decryption_keys`.
            fusion_weight: Integer weight vector y.

        Returns:
            The inner product as an integer.

        Raises:
            FEKeyError: If decryption key is missing.
            FEValidationError: If fusion weight length or values are invalid.
        """
        if not dk:
            raise FEKeyError("no decryption key provided")
        y = self._key_weights(dk, fusion_weight)
        return self._decode(dct_ct, y, digits_to_vector(dk["sk_y"]))

    def _key_weights(self, dk: dict, fusion_weight: list) -> list:
        y = int_vector(
            fusion_weight, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"])
        )
        if y != dk.get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")
        return y

    def _decode(self, ct: dict, y: list, sk_y: list) -> int | None:
        """``round(p (<y, ct1> - <ct0, sk_y>) / q)``; ``None`` beyond ``eta X Y``."""
        q = gp.mpz(self.pp["q"])
        ct0 = digits_to_vector(ct["ct0"])
        ct1 = digits_to_vector(ct["ct1"])
        if len(ct0) != len(sk_y) or len(ct1) != len(y):
            raise FEValidationError("ciphertext does not match the key dimensions")
        d = (vecdot_mod([gp.mpz(v) for v in y], ct1, q) - vecdot_mod(ct0, sk_y, q)) % q
        mu = decode_lwe_inner_product(d, gp.mpz(self.pp["p"]), q)
        bound = self.pp["eta"] * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        if abs(int(mu)) > bound:
            logger.error("inner product out of bound (wrong key or ciphertext?)")
            return None
        return int(mu)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarrays element-wise (requires ``eta=1``).

        Args:
            lst_ndarray: List of numpy arrays to encrypt.

        Returns:
            List of object ndarrays containing per-element ciphertexts.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            if np.abs(ary).max(initial=0) > int(self.pp["bound_x"]):
                raise FEValidationError("quantised value exceeds configured bound_x")
            flat = [[int(w)] for w in ary.ravel()]
            cts = []
            for start in range(0, len(flat), ENCRYPT_BATCH):
                cts.extend(self._encrypt_batch(flat[start:start + ENCRYPT_BATCH]))
            ary_ct = np.empty(len(cts), dtype=object)
            ary_ct[:] = cts
            lst_ndarray_ct.append(ary_ct.reshape(ary.shape))
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Ciphertext arrays to aggregate.
            **kwargs: Must include ``dk`` and ``fusion_weight``.

        Returns:
            Decrypted ndarray list.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise.

        Args:
            dict_ndarray_ct: List of object ndarrays with per-element ciphertexts.
            **kwargs: Must include ``dk`` and ``fusion_weight``.

        Returns:
            List of float ndarrays with decrypted values.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")

        y = self._key_weights(dk, fusion_weight)
        sk_y = digits_to_vector(dk["sk_y"])
        lst_ndarray = []
        for ary in dict_ndarray_ct:
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ary_dec[i] = self._decode(ary[i], y, sk_y)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
