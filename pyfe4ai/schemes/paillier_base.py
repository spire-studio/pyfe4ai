"""Base classes for Paillier-family key generators and crypto operations.

Provides shared Paillier group generation (p, q, n_mod, n_square, g),
parameter caching, bound validation, and encrypt/decrypt core logic so
that concrete SIFE / MIFE / MCFE Paillier schemes avoid code duplication.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto, IPFEAbsKeyGenerator, ParameterCacheMixin
from pyfe4ai.utils.crypto_utils import generate_group_primes
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.sampling_utils import random_below

logger = logging.getLogger(__name__)


class PaillierKeyGeneratorBase(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Base for Paillier key generators sharing (p, q, n_mod, n_square, g).

    Subclasses must set ``_scheme_type`` and ``_verify_keys``, and may
    override ``_extra_param_fields``, ``_extra_verify``, and
    ``_validate_bounds`` to persist/verify additional configuration.
    """

    _verify_keys: tuple[str, ...] = ()

    def _apply_parameters(self, param: dict) -> None:
        """Assign loaded parameters to instance attributes."""
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.n_mod = gp.mpz(param["n_mod"])
        self.n_square = gp.mpz(param["n_square"])
        self.g = gp.mpz(param["g"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if cached parameters match the current config."""
        return all(param.get(k) == getattr(self, k) for k in self._verify_keys) and self._extra_verify(param)

    def _extra_verify(self, param: dict) -> bool:
        """Subclass hook for additional verification."""
        return True

    def _extra_param_fields(self) -> dict:
        """Return extra fields to include in the saved parameter dict."""
        return {}

    @abstractmethod
    def _validate_bounds(self) -> None:
        """Validate that the bound constraints hold for the Paillier modulus."""
        raise NotImplementedError

    def _generate_and_save(self, param_file: str) -> None:
        """Generate a fresh Paillier group (p, q, n_mod, n_square, g) and persist."""
        self.p, _ = generate_group_primes(self.bit_length)
        self.q, _ = generate_group_primes(self.bit_length)
        self.n_mod = self.p * self.q
        self.n_square = self.n_mod * self.n_mod
        self._validate_bounds()

        while True:
            g_prime = random_below(self.n_square - 1)
            g = gp.powmod(g_prime, self.n_mod, self.n_square)
            g = gp.powmod(g, 2, self.n_square)
            if gp.gcd(g, self.n_square) == 1:
                self.g = g
                break

        _param = {
            "sec_param": self.sec_param,
            "eta": self.eta,
            "bit_length": self.bit_length,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "n_mod": gp.digits(self.n_mod),
            "n_square": gp.digits(self.n_square),
            "g": gp.digits(self.g),
            **self._extra_param_fields(),
        }
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(_param, f)


class PaillierCryptoBase(IPFEAbsCrypto):
    """Base for Paillier crypto operations sharing encrypt/decrypt core logic."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with scheme configuration.

        Args:
            config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    @staticmethod
    def _paillier_encrypt_element(x_i, pk_i, r, n_mod, n_square):
        """Encrypt a single plaintext element under Paillier.

        Args:
            x_i: Plaintext value (int or mpz).
            pk_i: Public key element (mpz).
            r: Randomness (mpz).
            n_mod: Paillier modulus N (mpz).
            n_square: N^2 (mpz).

        Returns:
            Ciphertext element as mpz.
        """
        x_term = gp.mpz(1) + gp.mpz(x_i) * n_mod
        x_term %= n_square
        return (x_term * gp.powmod(pk_i, r, n_square)) % n_square

    @staticmethod
    def _paillier_decrypt_core(acc, n_mod, n_square):
        """Recover plaintext from accumulated Paillier ciphertext.

        Args:
            acc: Accumulated ciphertext product (mpz).
            n_mod: Paillier modulus N (mpz).
            n_square: N^2 (mpz).

        Returns:
            Decrypted integer.
        """
        ret = ((acc - 1) % n_square) // n_mod
        n_half = n_mod // 2
        if ret > n_half:
            ret -= n_mod
        return int(ret)
