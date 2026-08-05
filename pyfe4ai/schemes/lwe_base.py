"""Base classes for LWE-family key generators and crypto operations.

Provides shared parameter caching logic (LWE parameter derivation, matrix A
sampling, serialisation) so that concrete SIFE / MIFE / MCFE LWE schemes
only need to declare which configuration fields they use.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto, IPFEAbsKeyGenerator, ParameterCacheMixin
from pyfe4ai.utils.lwe_utils import center_lwe_vector, decode_lwe_inner_product, derive_lwe_parameters
from pyfe4ai.utils.matrix_utils import (
    digits_to_matrix,
    matrix_to_digits,
    matvec_mod,
    transpose,
    vecdot_mod,
)
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix, rand_bit_vector, rand_uniform_matrix

logger = logging.getLogger(__name__)


class LWEKeyGeneratorBase(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Base for LWE key generators sharing (p, q, m, sigma_q, l_sigma, A).

    Subclasses must set ``_scheme_type`` and ``_verify_keys``, and may
    override ``_extra_param_fields`` and ``_extra_verify`` to persist
    and verify additional configuration.
    """

    _verify_keys: tuple[str, ...] = ()

    def _apply_parameters(self, param: dict) -> None:
        """Assign loaded parameters to instance attributes."""
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma = gp.mpz(param["l_sigma"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration keys."""
        return all(param.get(k) == getattr(self, k) for k in self._verify_keys) and self._extra_verify(param)

    def _extra_verify(self, param: dict) -> bool:
        """Subclass hook for additional verification (e.g. bound_x as digits)."""
        return True

    def _derive_lwe_extra_dimension(self) -> int | None:
        """Return ``extra_dimension`` for :func:`derive_lwe_parameters`, or ``None``."""
        return None

    def _effective_bound_x(self) -> gp.mpz:
        """Return the effective bound_x used for parameter derivation.

        MCFE overrides this to account for label masking.
        """
        return self.bound_x

    def _extra_param_fields(self) -> dict:
        """Return extra fields to include in the saved parameter dict."""
        return {}

    def _generate_and_save(self, param_file: str) -> None:
        """Derive LWE parameters, sample public matrix **A**, and persist."""
        self.p, self.q, self.m, self.sigma_q, self.l_sigma = derive_lwe_parameters(
            self._effective_bound_x(),
            self.bound_y,
            self.eta,
            self.lwe_n,
            extra_dimension=self._derive_lwe_extra_dimension(),
        )

        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)

        _param = {
            "sec_param": self.sec_param,
            "eta": self.eta,
            "lwe_n": self.lwe_n,
            "m": self.m,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "sigma_q": self.sigma_q,
            "l_sigma": gp.digits(self.l_sigma),
            "A": matrix_to_digits(self.A),
            **self._extra_param_fields(),
        }
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(_param, f)


class LWECryptoBase(IPFEAbsCrypto):
    """Base for LWE crypto operations sharing encrypt/decrypt core logic.

    Subclasses may override :meth:`_pre_encrypt` to inject label masking.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with scheme configuration.

        Args:
            config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = digits_to_matrix(self.pp["A"])
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def _lwe_encrypt_core(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector using the core LWE routine.

        Args:
            lst_pt: Integer plaintext vector (already masked if applicable).

        Returns:
            Dict with serialised ``ct0`` and ``ct1`` vectors.
        """
        q = gp.mpz(self.pp["q"])
        pk = digits_to_matrix(self.sk["pk"])
        r = rand_bit_vector(self.pp["m"])

        a_trans = transpose(self.A)
        ct0 = matvec_mod(a_trans, r, q)

        pk_trans = transpose(pk)
        ct1 = matvec_mod(pk_trans, r, q)
        centered = center_lwe_vector(lst_pt, gp.mpz(self.pp["p"]), q)
        ct1 = [(v + c) % q for v, c in zip(ct1, centered)]

        from pyfe4ai.utils.matrix_utils import vector_to_digits

        return {"ct0": vector_to_digits(ct0), "ct1": vector_to_digits(ct1)}

    @staticmethod
    def _lwe_decrypt_single(ct0, ct1, sk_y, y_vec, q, p):
        """Decrypt a single client's ciphertext contribution.

        Args:
            ct0: Ciphertext component 0 (vector of mpz).
            ct1: Ciphertext component 1 (vector of mpz).
            sk_y: Projected secret key (vector of mpz).
            y_vec: Weight vector (list of mpz).
            q: LWE modulus.
            p: Plaintext modulus.

        Returns:
            Decoded inner product contribution (int).
        """
        d = (vecdot_mod(y_vec, ct1, q) - vecdot_mod(ct0, sk_y, q)) % q
        return decode_lwe_inner_product(d, p, q)
