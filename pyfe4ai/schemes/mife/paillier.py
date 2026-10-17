"""
DCR (Paillier) multi-input inner-product FE.

| Abdalla, Catalano, Fiore, Gay, Ursu:
| "Multi-Input Functional Encryption for Inner Products: Function-Hiding
|  Realizations and Constructions without Pairings", CRYPTO 2018
| (ePrint 2017/972) -- Fig. 4 (transformation for bounded-norm inner products)
| instantiated with the Paillier IPFE of Agrawal, Libert, Stehle (CRYPTO 2016),
| recalled as Fig. 9 with its two-step decryption in Section 4.3.

* type:     private-key multi-input encryption (MIFE, DCR)
* setting:  Integer based (Z_{N^2}^*, N = p q with safe primes)

Construction (slot ``i`` is client ``i``, ``L = N``):

* Setup:    ``g = g'^{2N} mod N^2``; per slot ``s_i <- D_{Z^eta, sigma}`` with
  ``sigma > sqrt(kappa) N^{5/2}``, ``h_{i,j} = g^{s_{i,j}}`` and a
  one-time pad ``u_i <- Z_N^eta``; client ``i`` keeps ``(h_i, u_i)``.
* Encrypt:  ``w = x_i + u_i mod N``; ``C_0 = g^r`` (``r <- {0..N/4}``),
  ``C_j = (1 + w_j N) h_{i,j}^r``.
* KeyGen:   per slot ``d_i = <s_i, y_i>`` over Z, and ``z = sum_i <u_i, y_i> mod N``.
* Decrypt:  Dec1 per slot ``E(<w_i, y_i>) = prod_j C_j^{y_j} C_0^{-d_i}``;
  ``(prod_i E(<w_i, y_i>) * (1 + N)^{-z} - 1 mod N^2) / N``, centred mod N.

The pad hides each slot's partial result; versions <= 0.2.0 had no pad (the
key holder learned every ``<x_i, y_i>``) and drew keys from a range small
enough to brute-force from ``h``. Deviation: all slots share ``N`` and ``g``
instead of running Setup independently per slot (``s_i`` and ``u_i`` are
independent); the factorization of ``N`` is never stored. A decryption needs
ciphertexts of every slot named in the key. Correctness needs
``2 n eta X Y < N``.
"""

from __future__ import annotations

import json
import secrets
import logging
import os
import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.random_offline import OfflineRandomMixin
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import generate_group_primes
from pyfe4ai.utils.modular_utils import check_zn2_elements
from pyfe4ai.utils.modular_utils import decode_paillier
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.sampling_utils import random_below
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import scale_to_int
from pyfe4ai.utils.crypto_utils import is_dcr_modulus
from pyfe4ai.utils.validation import strict_int
from pyfe4ai.utils.crypto_utils import paillier_base

logger = logging.getLogger(__name__)

# version <= 0.2.0 caches also stored p and q; they are regenerated
_PARAM_FORMAT = "modulus-only/v1+g-seed"

class MIFEPaillierKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Paillier-based multi-input inner-product FE."""

    _holds_dcr_modulus = True  # never cached under standard (see ParameterCacheMixin)

    _scheme_type = CryptoCONST.TYPE_MIFE_PAILLIER
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE Paillier key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients, required), ``lst_nid``, ``eta``, ``bit_length`` (bits
                of each prime of ``N``, default 128), ``bound_x``, ``bound_y``
                and ``stat_param`` (default 64).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.bit_length = config.get("bit_length", 128)
        self.bound_x = config.get("bound_x", 100)
        self.bound_y = config.get("bound_y", 100)
        self.stat_param = int(config.get("stat_param", 64))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()
        self._validate_bounds()

    def _apply_parameters(self, param: dict) -> None:
        self.n_mod = gp.mpz(param["n_mod"])
        self.n_square = gp.mpz(param["n_square"])
        self.g_seed = param["g_seed"]
        self.g = paillier_base(self.g_seed, self.n_mod)

    def _loaded_parameters_valid(self) -> bool:
        n = self.n_mod
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
            and param.get("n") == self.n
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
        self.n_mod = p * q
        del p, q
        self.n_square = self.n_mod * self.n_mod
        self._validate_bounds()

        self.g_seed = secrets.token_hex(32)
        self.g = paillier_base(self.g_seed, self.n_mod)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "bit_length": self.bit_length,
                    "bound_x": self.bound_x,
                    "bound_y": self.bound_y,
                    "param_format": _PARAM_FORMAT,
                    "n_mod": gp.digits(self.n_mod),
                    "n_square": gp.digits(self.n_square),
                    "g_seed": self.g_seed,
                },
                f,
            )

    def _validate_bounds(self) -> None:
        # the centred result |<x, y>| <= n eta X Y must stay below N / 2
        if 2 * self.n * self.eta * gp.mpz(self.bound_x) * self.bound_y >= self.n_mod:
            raise FEValidationError(
                "2 * n * eta * bound_x * bound_y must be smaller than N; increase bit_length"
            )

    def setup(self) -> None:
        """Sample per slot ``s_i <- D_{Z^eta, sigma}`` (ALS16) and a pad ``u_i <- Z_N^eta``."""
        self.sigma = (
            (gp.isqrt(self.stat_param) + 1) * self.n_mod * self.n_mod * (gp.isqrt(self.n_mod) + 1)
        )
        s = {nid: discrete_gaussian_bigint(self.sigma, self.eta) for nid in self.lst_nid}
        u = {nid: rand_uniform_vector(self.eta, self.n_mod) for nid in self.lst_nid}
        pk = {
            nid: [pow_signed(self.g, s_i, self.n_square) for s_i in vec]
            for nid, vec in s.items()
        }
        self.msk = {"s": s, "u": u}
        self.mpk = {
            "n_mod": self.n_mod,
            "n_square": self.n_square,
            "g": self.g,
            "pk": pk,
        }
        logger.info("MIFE Paillier setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "n_mod": gp.digits(self.mpk["n_mod"]),
            "n_square": gp.digits(self.mpk["n_square"]),
            "g": gp.digits(self.mpk["g"]),
            "eta": self.eta,
            "n": self.n,
            "sec_param": self.sec_param,
            "bit_length": self.bit_length,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s encryption key (``pk`` and its pad ``u``).

        Args:
            nid: Client identifier.

        Returns:
            The key dict, or ``None`` for an unknown client.
        """
        if nid not in self.mpk["pk"]:
            return None
        return {
            "pk": [gp.digits(v) for v in self.mpk["pk"][nid]],
            "u": [gp.digits(v) for v in self.msk["u"][nid]],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (dict
                ``{nid: list of int}``).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MIFE Paillier decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        dct_k = {}
        z = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            weights = fusion_weight[nid]
            if len(weights) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(strict_int(v, "fusion weight")) > self.bound_y for v in weights):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            k = gp.mpz(0)
            for s_i, u_i, y_i in zip(self.msk["s"][nid], self.msk["u"][nid], weights):
                k += s_i * gp.mpz(int(y_i))
                z += u_i * gp.mpz(int(y_i))
            dct_k[nid] = gp.digits(k)
        return {"k": dct_k, "z": gp.digits(z % self.n_mod)}


class MIFEPaillier(OfflineRandomMixin, IPFEAbsCrypto):
    """Crypto operations for Paillier-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE Paillier crypto object.

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
        if "u" not in self.sk:
            raise FEKeyError("encryption key lacks the one-time pad u")
        return self._encrypt_with(lst_pt, self._random_part())

    def _random_part(self) -> tuple:
        """``(g^r, [pk_i^r])`` mod N^2 for a fresh r (see
        :mod:`pyfe4ai.schemes.random_offline`)."""
        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = gp.mpz(self.pp["n_square"])
        r = random_below(n_mod // 4)
        return gp.powmod(gp.mpz(self.pp["g"]), r, n_square), [
            gp.powmod(gp.mpz(pk_i), r, n_square) for pk_i in self.sk["pk"]]

    def _encrypt_with(self, lst_pt: list, part: tuple) -> dict:
        """The ciphertext of :meth:`encrypt` (with its checks) for the randomness part *part*."""
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > self.pp["bound_x"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")
        if "u" not in self.sk:
            raise FEKeyError("encryption key lacks the one-time pad u")
        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = gp.mpz(self.pp["n_square"])
        c0, masks = part
        c1 = []
        for x_i, u_i, m in zip(lst_pt, self.sk["u"], masks):
            w_i = (gp.mpz(int(x_i)) + gp.mpz(u_i)) % n_mod
            x_term = (1 + w_i * n_mod) % n_square
            c1.append(gp.digits((x_term * m) % n_square))
        return {"ct0": gp.digits(c0), "ct1": c1}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
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

        if "z" not in dk:
            raise FEKeyError("decryption key lacks z")
        if set(dct_ct) != set(dk["k"]) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")
        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = gp.mpz(self.pp["n_square"])
        eta = self.pp["eta"]
        # E(-z) = (1 + N)^{-z} = 1 - z N mod N^2
        acc = (1 - gp.mpz(dk["z"]) * n_mod) % n_square
        for nid, ct in dct_ct.items():
            weights = fusion_weight[nid]
            ct1 = ct.get("ct1")
            if not isinstance(ct1, list) or len(ct1) != eta or len(weights) != eta:
                raise FEValidationError("ciphertext or weights of {} do not match eta".format(nid))
            c0, *c1 = check_zn2_elements([ct["ct0"], *ct1], n_square)
            acc *= pow_signed(c0, -gp.mpz(dk["k"][nid]), n_square)
            acc %= n_square
            for ct_i, y_i in zip(c1, weights):
                acc *= pow_signed(ct_i, gp.mpz(y_i), n_square)
                acc %= n_square

        bound = len(dct_ct) * eta * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        mu = decode_paillier(acc, n_mod, bound)
        if mu is None:
            logger.error("decryption is not an inner product in range (wrong key or ciphertext?)")
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
                ary_ct[i] = self.encrypt([int(w)])
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

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
