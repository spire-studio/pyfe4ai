"""
Ring-LWE-Based Single-Input Inner-Product Functional Encryption
| Based on
| "Efficient Lattice-Based Inner-Product Functional Encryption"
| By Bermudo Mera, Karmakar, Marc, Soleimanian
| ePrint: 2021/046
|
| (PKC 2022), Section 4 (selectively secure Ring-LWE IPFE) with the batching
| of Remark 1; also informed by GoFE's ``innerprod/simple/ringlwe.go``.

* type:     public-key encryption
* setting:  R_q = Z_q[x] / (x^ring_n + 1), ring_n a power of two

Public ``a``; ``s_i, e_i <- D_sigma1``, ``pk_i = a s_i + e_i``. The key for ``y``
is ``sk_y = sum y_i s_i``. A plaintext is an ``eta x k`` matrix (``k <= ring_n``,
column ``j`` in coefficient ``j``): ``ct0 = a r + f0``,
``ct_i = pk_i r + f_i + Delta x_i`` with ``r, f0 <- D_sigma2``,
``f_i <- D_sigma3``; decryption rounds ``sum y_i ct_i - ct0 sk_y``. Parameters
come from :func:`pyfe4ai.utils.ring_lwe_utils.bmms21_parameters`; ``ring_n``
defaults to the toy value ``DEFAULT_RING_N`` (versions <= 0.2.0 claimed to
derive it but always chose 32).
"""

from __future__ import annotations

import json
import secrets
import logging

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.profiles import STANDARD
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.ring_lwe_utils import (
    bmms21_parameters,
    bmms21_selective_parameters,
    center_matrix,
    check_ring_lwe,
    decode_vector,
    mat_vec_mul,
    matrix_check_bound,
    poly_add,
    poly_mul,
    poly_mul_negacyclic,
    poly_neg,
    resolve_ring_n,
    transpose,
)
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

_PARAM_VERSION = "bmms21-sec4.2/v3"  # p = 2 l X Y + 2, Sec. 4.2 widths, seeded a

# Backward-compatible aliases (private names used by internal callers)
_poly_add = poly_add
_poly_neg = poly_neg
_poly_mul_negacyclic = poly_mul_negacyclic
_poly_mul = poly_mul
_matrix_check_bound = matrix_check_bound
_transpose = transpose
_mat_vec_mul = mat_vec_mul
_center_matrix = center_matrix
_decode_vector = decode_vector


class SIFERingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Ring-LWE-based single-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_SIFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE Ring-LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``
                (``kappa`` of the noise bounds), ``eta``, ``bound_x``,
                ``bound_y`` and ``ring_n`` (default
                :data:`~pyfe4ai.utils.ring_lwe_utils.DEFAULT_RING_N`;
                ``"auto"`` for the smallest dimension meeting the 128-bit
                table, the ``standard`` profile's choice).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self.ring_n = resolve_ring_n(config.get("ring_n"), self._params_for)
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_ring_lwe(self.ring_n, self.q, self._params_for(self.ring_n)[1])

    def _params_for(self, ring_n: int) -> tuple:
        """``(q, rlwe_width)`` that :meth:`_generate_and_save` derives for *ring_n*."""
        P = bmms21_selective_parameters(self.sec_param, self.eta, self.bound_x, self.bound_y, ring_n)
        return P["q"], P["rlwe_width"]

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.ring_n = int(param["ring_n"])
        self.sigma1 = float(param["sigma1"])
        self.sigma2 = float(param["sigma2"])
        self.sigma3 = float(param["sigma3"])
        self.a_seed = param["A_seed"]
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)

    def _param_verification(self, param: dict) -> bool:
        """The cached record must match the config and the parameters derived
        from it; only the seed of ``a`` is random."""
        if not (
            param.get("param_version") == _PARAM_VERSION
            and param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("ring_n") == self.ring_n
        ):
            return False
        p, q, sigma1, sigma2, sigma3 = bmms21_parameters(
            self.sec_param, self.eta, self.bound_x, self.bound_y, self.ring_n
        )
        return (param.get("p"), param.get("q")) == (gp.digits(p), gp.digits(q)) and (
            param.get("sigma1"), param.get("sigma2"), param.get("sigma3")
        ) == (sigma1, sigma2, sigma3)

    def _generate_and_save(self, param_file: str) -> None:
        self.p, self.q, self.sigma1, self.sigma2, self.sigma3 = bmms21_parameters(
            self.sec_param, self.eta, self.bound_x, self.bound_y, self.ring_n
        )
        self.a_seed = secrets.token_hex(32)
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "param_version": _PARAM_VERSION,
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "ring_n": self.ring_n,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma1": self.sigma1,
                    "sigma2": self.sigma2,
                    "sigma3": self.sigma3,
                    "A_seed": self.a_seed,
                },
                f,
            )

    def setup(self) -> None:
        sk = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
        noise = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
        pk = []
        for i in range(self.eta):
            pk_i = _poly_mul(self.A, sk[i], self.q)
            pk.append(_poly_add(pk_i, noise[i], self.q))
        self.msk = {"sk": sk}
        self.mpk = {"pk": pk}
        logger.info("SIFE Ring-LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": [gp.digits(v) for v in self.A],
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "eta": self.eta,
            "ring_n": self.ring_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sigma3": self.sigma3,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return the encryption key (public key polynomials ``pk``).

        Args:
            nid: Client identifier (unused for single-input schemes).
        """
        return {"pk": [[gp.digits(v) for v in row] for row in self.mpk["pk"]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of
                int).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE Ring-LWE decryption key generation")
        y = int_vector(
            credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=self.bound_y
        )
        sk_y = _mat_vec_mul(_transpose(self.msk["sk"]), [gp.mpz(v) for v in y], self.q)
        return {"sk_y": [gp.digits(v) for v in sk_y], "y": y}


class SIFERingLWE(IPFEAbsCrypto):
    """Crypto operations for Ring-LWE-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE Ring-LWE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = [gp.mpz(v) for v in self.pp["A"]]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, matrix_pt: list[list[int]]) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if not isinstance(matrix_pt, list) or not matrix_pt or not isinstance(matrix_pt[0], list):
            raise FEValidationError("plaintext matrix must be a list of rows")
        if len(matrix_pt) != self.pp["eta"]:
            raise FEValidationError("invalid plaintext row count")
        k = len(matrix_pt[0])
        if not 0 < k <= self.pp["ring_n"]:
            raise FEValidationError("plaintext column count must be 1..ring_n")
        matrix_pt = [
            int_vector(row, "plaintext row", length=k, bound=int(self.pp["bound_x"]))
            for row in matrix_pt
        ]

        q = gp.mpz(self.pp["q"])
        pk = [[gp.mpz(v) for v in row] for row in self.sk["pk"]]
        ring_n = self.pp["ring_n"]

        r = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        noise = discrete_gaussian_matrix(self.pp["eta"], ring_n, self.pp["sigma3"])
        ct0 = []
        for i in range(self.pp["eta"]):
            ct0_i = _poly_mul(pk[i], r, q)
            ct0.append(_poly_add(ct0_i, noise[i], q))
        centered = _center_matrix(matrix_pt, gp.mpz(self.pp["p"]), q, ring_n)
        ct0 = [_poly_add(a, b, q) for a, b in zip(ct0, centered)]

        ct1 = _poly_mul(self.A, r, q)
        e = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        ct1 = _poly_add(ct1, e, q)

        return {
            "ct0": [[gp.digits(v) for v in row] for row in ct0],
            "ct1": [gp.digits(v) for v in ct1],
            "k": len(matrix_pt[0]),
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not dk:
            raise FEKeyError("no decryption key provided")
        y = int_vector(
            fusion_weight, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"])
        )
        if y != dk.get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        ring_n = self.pp["ring_n"]
        ct0 = [[gp.mpz(v) for v in row] for row in dct_ct["ct0"]]
        ct1 = [gp.mpz(v) for v in dct_ct["ct1"]]
        sk_y = [gp.mpz(v) for v in dk["sk_y"]]
        k = int(dct_ct["k"])
        if (
            len(ct0) != len(y) or any(len(row) != ring_n for row in ct0)
            or len(ct1) != ring_n or len(sk_y) != ring_n or not 0 < k <= ring_n
        ):
            raise FEValidationError("ciphertext does not match the parameters")

        lhs = _mat_vec_mul(_transpose(ct0), [gp.mpz(v) for v in y], q)
        rhs = _poly_mul(ct1, sk_y, q)
        values = _decode_vector(_poly_add(lhs, _poly_neg(rhs, q), q), p, q, k)
        bound = len(y) * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        if any(abs(v) > bound for v in values):
            logger.error("inner product out of bound (wrong key or ciphertext?)")
            return None
        return values

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError()

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError()

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError()
