"""
Fully Secure LWE-Based Single-Input Inner-Product Functional Encryption
| Based on the LWE variant of
| "Fully Secure Functional Encryption for Inner Products,
|  from Standard Assumptions"
| By Shweta Agrawal, Benoit Libert, Damien Stehle
| Published in: CRYPTO 2016
|
| as recalled in Abdalla, Catalano, Fiore, Gay, Ursu, CRYPTO 2018, Fig. 8.
| Parameters follow GoFE's ``innerprod/fullysec/lwe.go``.

* type:     public-key encryption
* setting:  Integer based

``A <- Z_q^{m x n}``; ``Z`` has ``D_{sigma1}`` entries in its first ``m/2``
columns and ``D_{sigma2}`` entries plus the identity in the others; ``U = Z A``.
The key for ``y`` is ``Z^T y``; ``ct = (A s + e0, U s + e1 + floor(q/K) x)``;
decryption decodes ``<y, ct1> - <Z^T y, ct0>``. Signed messages and weights:
``K = 2 eta X Y + 1`` (versions <= 0.2.0 used ``2 eta X Y``, so ``+-eta X Y``
decoded with a random sign), centred decoding.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.utils.lwe_utils import derive_fullysec_lwe_parameters
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matrix_to_digits
from pyfe4ai.utils.matrix_utils import matmul_mod
from pyfe4ai.utils.matrix_utils import matvec_mod
from pyfe4ai.utils.matrix_utils import transpose
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.matrix_utils import vector_to_digits
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_uniform_matrix
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)

# K = 2 eta X Y + 1, encoding floor(q / K) x, m = 4 n log2 q, B_tau ~ sqrt(m)
_CONSTRUCTION = "als16-lwe/v4"


class SIFEFullySecLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for fully secure LWE-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE_FULLYSEC_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the fully secure SIFE LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta``, ``lwe_n``, ``bound_x``, ``bound_y``.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.lwe_n = config.get("lwe_n", 64)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        """Assign fully-secure LWE parameters from *param*."""
        self.k = gp.mpz(param["k"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma_q = gp.mpz(param["l_sigma_q"])
        self.sigma1 = float(param["sigma1"])
        self.l_sigma1 = gp.mpz(param["l_sigma1"])
        self.sigma2 = float(param["sigma2"])
        self.l_sigma2 = gp.mpz(param["l_sigma2"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration."""
        return (
            param.get("construction") == _CONSTRUCTION
            and param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("lwe_n") == self.lwe_n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
        )

    def _generate_and_save(self, param_file: str) -> None:
        """Derive fully-secure LWE parameters, sample **A**, and persist."""
        (
            self.k,
            self.q,
            self.m,
            self.sigma_q,
            self.l_sigma_q,
            self.sigma1,
            self.l_sigma1,
            self.sigma2,
            self.l_sigma2,
        ) = derive_fullysec_lwe_parameters(
            self.bound_x, self.bound_y, self.eta, self.lwe_n
        )
        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=os.path.dirname(param_file), delete=False
        ) as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "lwe_n": self.lwe_n,
                    "m": self.m,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "k": gp.digits(self.k),
                    "q": gp.digits(self.q),
                    "sigma_q": self.sigma_q,
                    "l_sigma_q": gp.digits(self.l_sigma_q),
                    "sigma1": self.sigma1,
                    "l_sigma1": gp.digits(self.l_sigma1),
                    "sigma2": self.sigma2,
                    "l_sigma2": gp.digits(self.l_sigma2),
                    "A": matrix_to_digits(self.A),
                },
                f,
            )
            temp_name = f.name
        os.replace(temp_name, param_file)

    def setup(self) -> None:
        """Generate master secret key (``msk``) and master public key (``mpk``)."""
        half_cols = self.m // 2
        if self.eta > self.m - half_cols:
            raise FEValidationError("eta must be at most m - m // 2 = {}".format(self.m - half_cols))
        z = [[gp.mpz(0) for _ in range(self.m)] for _ in range(self.eta)]
        for i in range(self.eta):
            for j in range(self.m):
                if j < half_cols:
                    sampled = discrete_gaussian_matrix(1, 1, self.sigma1)[0][0]
                else:
                    sampled = discrete_gaussian_matrix(1, 1, self.sigma2)[0][0]
                    if j - half_cols == i:
                        sampled += 1
                z[i][j] = sampled

        u = matmul_mod(z, self.A, self.q)
        self.msk = {"z": z}
        self.mpk = {"u": u}
        logger.info("SIFE fully secure LWE setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters.

        Returns:
            Dict containing ``A``, ``k``, ``q``, ``m``, ``eta``,
            ``lwe_n``, ``bound_x``, ``bound_y``, and noise parameters.
        """
        return {
            "A": matrix_to_digits(self.A),
            "k": gp.digits(self.k),
            "q": gp.digits(self.q),
            "m": self.m,
            "eta": self.eta,
            "lwe_n": self.lwe_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma_q": self.sigma_q,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return encryption keys (public-key matrix ``pk``).

        Args:
            nid: Client identifier (unused for single-input schemes).

        Returns:
            Dict with ``pk`` matrix.
        """
        return {"pk": matrix_to_digits(self.mpk["u"])}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of int).

        Returns:
            Dict with ``sk_y`` (``Z^T y``) and the weights ``y``.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid or exceed ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for fully secure SIFE LWE decryption key generation")
        y = int_vector(
            credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=self.bound_y
        )
        z_y = matvec_mod(transpose(self.msk["z"]), [gp.mpz(v) for v in y], self.q)
        return {"sk_y": vector_to_digits(z_y), "y": y}


class SIFEFullySecLWE(IPFEAbsCrypto):
    """Crypto operations for fully secure LWE-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the fully secure SIFE LWE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = digits_to_matrix(self.pp["A"])
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector under the fully-secure LWE scheme.

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
        x = int_vector(lst_pt, "plaintext", length=self.pp["eta"], bound=int(self.pp["bound_x"]))

        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["k"])
        pk = digits_to_matrix(self.sk["pk"])
        r = rand_uniform_vector(self.pp["lwe_n"], q)
        e0 = discrete_gaussian_matrix(1, self.pp["m"], self.pp["sigma_q"])[0]
        e1 = discrete_gaussian_matrix(1, self.pp["eta"], self.pp["sigma_q"])[0]

        c0 = matvec_mod(self.A, r, q)
        c0 = [(a + b) % q for a, b in zip(c0, e0)]

        c1 = matvec_mod(pk, r, q)
        c1 = [(a + b + delta * x_i) % q for a, b, x_i in zip(c1, e1, x)]
        return {"ct0": vector_to_digits(c0), "ct1": vector_to_digits(c1)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt a ciphertext to recover the inner product ⟨x, y⟩.

        Args:
            dct_ct: Ciphertext dict from :meth:`encrypt`.
            dk: Decryption key from :meth:`SIFEFullySecLWEKeyGenerator.get_decryption_keys`.
            fusion_weight: Integer weight vector y.

        Returns:
            The inner product, or ``None`` if it decodes beyond ``eta X Y``
            (wrong key or ciphertext).

        Raises:
            FEKeyError: If decryption key is missing.
            FESchemeError: If *fusion_weight* are not the key's weights.
            FEValidationError: If fusion weight or ciphertext are malformed.
        """
        if not dk:
            raise FEKeyError("no decryption key provided")
        y = int_vector(
            fusion_weight, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"])
        )
        if y != dk.get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")

        q = gp.mpz(self.pp["q"])
        ct0 = digits_to_vector(dct_ct["ct0"])
        ct1 = digits_to_vector(dct_ct["ct1"])
        z_y = digits_to_vector(dk["sk_y"])
        if len(ct0) != len(z_y) or len(ct1) != len(y):
            raise FEValidationError("ciphertext does not match the key dimensions")

        mu = (vecdot_mod([gp.mpz(v) for v in y], ct1, q) - vecdot_mod(z_y, ct0, q)) % q
        bound = self.pp["eta"] * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        return decode_scaled(mu, q // gp.mpz(self.pp["k"]), q, bound)

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
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            lst_ndarray_ct.append(ary_ct)
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

        lst_ndarray = []
        for ary in dict_ndarray_ct:
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ary_dec[i] = self.decrypt(ary[i], dk, fusion_weight)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
