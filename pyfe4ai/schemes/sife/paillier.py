"""
Paillier-Based Single-Input Inner-Product Functional Encryption
| From "Fully Secure Functional Encryption for Inner Products,
|       from Standard Assumptions"
| By Shweta Agrawal, Benoit Libert, and Damien Stehle
| Published in: CRYPTO 2016 (DCR instantiation; recalled as Fig. 9 of
| ABMPSW20, https://eprint.iacr.org/2020/788, Appendix B.1)

* type:     public-key encryption
* setting:  Integer based (Z_{N^2}^*, N = p q with safe primes)

Construction: ``g = g'^{2N} mod N^2``; ``s <- D_{Z^eta, sigma}`` with
``sigma > sqrt(kappa) * N^{5/2}``; ``h_i = g^{s_i}``; ``ct = (g^r,
(1 + N)^{x_i} h_i^r)`` with ``r <- {0..N/4}``; ``sk_y = <s, y>`` over Z;
``<x, y> = (prod ct_i^{y_i} ct_0^{-sk_y} - 1 mod N^2) / N``. The Gaussian
is a rounded continuous Gaussian at ``bits(sigma) + 128`` bits of precision
and ``kappa`` is the config key ``stat_param`` (default 64).

"""

from __future__ import annotations

import json
import secrets
import logging
import os
import gmpy2 as gp
import numpy as np

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.one_time import OfflineRandomMixin
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.groups.integer_groups import generate_group_primes
from pyfe4ai.primitives.groups.modular_utils import check_zn2_elements
from pyfe4ai.primitives.groups.modular_utils import decode_paillier
from pyfe4ai.primitives.groups.modular_utils import pow_signed
from pyfe4ai.primitives.sampling import discrete_gaussian_bigint
from pyfe4ai.primitives.sampling import random_below
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError, warn_at_caller
from pyfe4ai._validation import scale_to_int
from pyfe4ai.primitives.groups.integer_groups import is_dcr_modulus
from pyfe4ai._validation import strict_int
from pyfe4ai.primitives.hashing import paillier_base

logger = logging.getLogger(__name__)

# version <= 0.2.0 caches also stored p and q; they are regenerated
_PARAM_FORMAT = "modulus-only/v1+g-seed"

class SIFEPaillierKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Paillier-based single-input inner-product FE."""

    _holds_dcr_modulus = True  # never cached under standard (see ParameterCacheMixin)

    _scheme_type = CryptoCONST.TYPE_SIFE_PAILLIER
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE Paillier key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``eta``,
                ``bit_length`` (bits of each prime of ``N``, default 128),
                ``bound_x``, ``bound_y`` and ``stat_param`` (default 64).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bit_length = config.get("bit_length", 128)
        self.bound_x = config.get("bound_x", 100)
        self.bound_y = config.get("bound_y", 100)
        self.stat_param = int(config.get("stat_param", 64))
        self._load_parameters()
        self._validate_bounds()

    def _apply_parameters(self, param: dict) -> None:
        self.n = gp.mpz(param["n"])
        self.n_square = gp.mpz(param["n_square"])
        self.g_seed = param["g_seed"]
        self.g = paillier_base(self.g_seed, self.n)

    def _loaded_parameters_valid(self) -> bool:
        n = self.n
        return (
            is_dcr_modulus(n, self.bit_length)
            and self.n_square == n * n
            and 1 < self.g < self.n_square
            and gp.gcd(self.g, n) == 1
            and self.g % n != 1  # g = 1 + N (order N) would reveal s mod N from the public key
        )

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("param_format") == _PARAM_FORMAT
            and param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bit_length") == self.bit_length
            and param.get("bound_x") == self.bound_x
            and param.get("bound_y") == self.bound_y
        )

    def _generate_and_save(self, param_file: str) -> None:
        # the factorization is a decryption trapdoor: keep it local, never persist it
        p, _ = generate_group_primes(self.bit_length)
        while True:
            q, _ = generate_group_primes(self.bit_length)
            if q != p:
                break
        self.n = p * q
        del p, q
        self.n_square = self.n * self.n
        self._validate_bounds()

        self.g_seed = secrets.token_hex(32)
        self.g = paillier_base(self.g_seed, self.n)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "bit_length": self.bit_length,
                    "bound_x": self.bound_x,
                    "bound_y": self.bound_y,
                    "param_format": _PARAM_FORMAT,
                    "n": gp.digits(self.n),
                    "n_square": gp.digits(self.n_square),
                    "g_seed": self.g_seed,
                },
                f,
            )

    def _validate_bounds(self) -> None:
        x_square_l = gp.mpz(2 * self.eta * (self.bound_x**2))
        y_square_l = gp.mpz(2 * self.eta * (self.bound_y**2))
        if self.n <= x_square_l:
            raise ValueError("bound_x and eta are too large for the chosen bit_length")
        if self.n <= y_square_l:
            raise ValueError("bound_y and eta are too large for the chosen bit_length")

    def setup(self) -> None:
        """Sample ``s <- D_{Z^eta, sigma}`` with ``sigma > sqrt(kappa) N^{5/2}`` (ALS16).

        Versions <= 0.2.0 drew ``|s_i| <= max(sec_param, bound_y * eta)``,
        so ``s`` could be recovered from the public key by brute force.
        """
        self.sigma = (
            (gp.isqrt(self.stat_param) + 1) * self.n * self.n * (gp.isqrt(self.n) + 1)
        )
        sec = discrete_gaussian_bigint(self.sigma, self.eta)
        pub = [pow_signed(self.g, s_i, self.n_square) for s_i in sec]
        self.msk = {"s": sec}
        self.mpk = {
            "n": self.n,
            "n_square": self.n_square,
            "g": self.g,
            "pk": pub,
        }
        logger.info("SIFE Paillier setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "n": gp.digits(self.mpk["n"]),
            "n_square": gp.digits(self.mpk["n_square"]),
            "g": gp.digits(self.mpk["g"]),
            "eta": self.eta,
            "sec_param": self.sec_param,
            "bit_length": self.bit_length,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return the encryption key (public key vector ``pk``).

        Args:
            nid: Client identifier (unused for single-input schemes).
        """
        return {"pk": [gp.digits(v) for v in self.mpk["pk"]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of
                int).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE Paillier decryption key generation")
        fusion_weights = credentials.get("fusion_weight")
        if not isinstance(fusion_weights, list):
            raise FEValidationError("invalid fusion weights provided, need a list")
        if len(fusion_weights) != self.eta:
            raise FEValidationError("invalid fusion weights provided, length mismatch")
        if any(abs(strict_int(v, "fusion weight")) > self.bound_y for v in fusion_weights):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        k = gp.mpz(0)
        for s_i, y_i in zip(self.msk["s"], fusion_weights):
            k += s_i * gp.mpz(y_i)
        return {"k": gp.digits(k)}


class SIFEPaillier(OfflineRandomMixin, IPFEAbsCrypto):
    """Crypto operations for Paillier-based single-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE Paillier crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > self.pp["bound_x"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        return self._encrypt_with(lst_pt, self._random_part())

    def _random_part(self) -> tuple:
        """``(g^r, [pk_i^r])`` mod N^2 for a fresh r (see
        :mod:`pyfe4ai.core.one_time`)."""
        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        r = random_below(n // 4)
        return gp.powmod(gp.mpz(self.pp["g"]), r, n_square), [
            gp.powmod(gp.mpz(pk_i), r, n_square) for pk_i in self.sk["pk"]]

    def _encrypt_with(self, lst_pt: list, part: tuple) -> dict:
        """The ciphertext of *lst_pt* for the randomness part *part*."""
        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        c0, masks = part
        c1 = []
        for x_i, m in zip(lst_pt, masks):
            x_term = gp.mpz(1) + gp.mpz(x_i) * n
            x_term %= n_square
            c1.append(gp.digits((x_term * m) % n_square))
        return {"ct0": gp.digits(c0), "ct1": c1}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for decryption")
        if not dk:
            raise FEKeyError("no decryption key provided")
        if len(fusion_weight) != self.pp["eta"]:
            raise FEValidationError("invalid fusion weight length")
        if any(abs(strict_int(v, "fusion weight")) > self.pp["bound_y"] for v in fusion_weight):
            raise FEValidationError("fusion weight exceeds configured bound_y")

        n = gp.mpz(self.pp["n"])
        n_square = gp.mpz(self.pp["n_square"])
        ct1 = dct_ct.get("ct1")
        if not isinstance(ct1, list) or len(ct1) != self.pp["eta"]:
            raise FEValidationError("ciphertext does not match eta")
        c0, *c1 = check_zn2_elements([dct_ct["ct0"], *ct1], n_square)
        k = gp.mpz(dk["k"])

        acc = pow_signed(c0, -k, n_square)
        for ct_i, y_i in zip(c1, fusion_weight):
            acc *= pow_signed(ct_i, gp.mpz(y_i), n_square)
            acc %= n_square

        bound = self.pp["eta"] * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        mu = decode_paillier(acc, n, bound)
        if mu is None:
            logger.error("decryption is not an inner product in range (wrong key or ciphertext?)")
            warn_at_caller(
                "decryption is not an inner product in range (wrong key or ciphertext?)",
            )
        return mu

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarrays element-wise (requires ``eta = 1``).

        Args:
            lst_ndarray: List of numpy arrays, scaled by ``10^precision``.

        Returns:
            One object array of ciphertexts per input array.

        Raises:
            NotImplementedError: If ``eta != 1``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if kwargs.get("precomputed") is not None:
            return self._encrypt_precomputed(lst_ndarray, kwargs["precomputed"])
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self._encrypt_with([int(w)], self._random_part())
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")

        lst_ndarray = []
        for l, ary in enumerate(dict_ndarray_ct):
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                # the checks of decrypt (Z_{N^2}^* members, result in range) apply per element
                ary_dec[i] = self.decrypt(ary[i], dk, fusion_weight)
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
