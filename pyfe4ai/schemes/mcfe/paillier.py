"""
DCR (Paillier) based multi-client inner-product FE.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Fig. 3 (MCFE based on DCR),
| correctness and security in Appendix B.2 (one-IND secure in the ROM).

* type:     secret-key multi-client encryption (MCFE, DCR, random oracle)
* setting:  Integer based (Z_{N^2}^*, N = p * q with safe primes p, q)

Construction (one coordinate j of client i's vector is one slot; ``n`` below
is the total number of slots):

* Setup:    safe-prime modulus ``N``; ``H: labels -> Z_{N^2}^*`` full-domain hash.
* KeyGen:   ``s_{i,j} <- D_{Z, sigma}`` with
            ``sigma > sqrt(kappa + 2 n log2(2X)) * N^{5/2}`` (adaptive setting).
* Encrypt:  ``c_{i,j} = (1 + N)^{x_{i,j}} * H(l)^{s_{i,j}} mod N^2``.
* KeyDer:   ``k = sum y_{i,j} s_{i,j}`` over the integers (one aggregated key).
* Decrypt:  ``C = prod c_{i,j}^{y_{i,j}} * H(l)^{-k}``;
            ``<x, y> = (C - 1 mod N^2) / N``.

Correctness needs ``|x|, |y| < sqrt(N / 2n)`` (Fig. 3); the result is
returned centred modulo ``N``, so signed inputs are supported. Each client
slot must encrypt at most once per label (one-IND security): two ciphertexts
of one slot under one label reveal ``(1 + N)^{x - x'}``. The ndarray helpers
therefore derive a distinct label per element.

Implementation notes: ``H`` is SHAKE-256 expanded to ``bits(N^2) + 128`` bits
(the paper's implementation uses counter-mode SHA-256); ``D_{Z, sigma}`` is
a rounded continuous Gaussian computed at ``bits(sigma) + 128`` bits of
precision (:func:`pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint`).
``bit_length`` is the size of each prime and is far below any real security
level by default.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import generate_group_primes
from pyfe4ai.utils.crypto_utils import hash_to_zn2_star
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)

_LABEL_HASH_DOMAIN = "pyfe4ai/mcfe/paillier/H(label)"
_CONSTRUCTION = "abmpsw20-dcr/v2"  # v1 caches stored p and q
_DEFAULT_STAT_PARAM = 64


def _hash_label(label, n_mod: gp.mpz) -> gp.mpz:
    return hash_to_zn2_star(encode_label(label), n_mod, domain=_LABEL_HASH_DOMAIN)


def dcr_sigma(n_mod: gp.mpz, n_slots: int, bound_x: int, kappa: int) -> gp.mpz:
    """Return an integer ``sigma > sqrt(kappa + 2 n log2(2X)) * N^{5/2}``.

    This is the adaptive-security condition of ABMPSW20 Fig. 3 / Lemma 27
    (the selective one drops the ``2 n log2(2X)`` term).

        Args:
            n_mod: RSA modulus ``N``.
            n_slots: Total number of slots ``n``.
            bound_x: Plaintext bound ``X``.
            kappa: Statistical security parameter.
    """
    log_term = 2 * n_slots * (int(2 * bound_x).bit_length())  # >= 2n log2(2X)
    factor = gp.isqrt(kappa + log_term) + 1  # > sqrt(kappa + 2n log2(2X))
    n_five_halves = n_mod * n_mod * (gp.isqrt(n_mod) + 1)  # > N^{5/2}
    return factor * n_five_halves


class MCFEPaillierKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for DCR-based multi-client inner-product FE (ABMPSW20 Fig. 3)."""

    _scheme_type = CryptoCONST.TYPE_MCFE_PAILLIER

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients), ``lst_nid``, ``eta`` (slots per client),
                ``bit_length`` (bits of each safe prime), ``bound_x``,
                ``bound_y``, ``stat_param`` (statistical parameter ``kappa``),
                ``sec_param`` (recorded only).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.bit_length = config.get("bit_length", 128)
        self.bound_x = config.get("bound_x", 100)
        self.bound_y = config.get("bound_y", 100)
        self.stat_param = config.get("stat_param", _DEFAULT_STAT_PARAM)
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        if not isinstance(self.eta, int) or self.eta < 1:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        self.n_slots = self.n * self.eta
        self._load_parameters()
        self._validate_bounds()

    def _apply_parameters(self, param: dict) -> None:
        self.n_mod = gp.mpz(param["n_mod"])
        self.n_square = self.n_mod * self.n_mod

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("construction") == _CONSTRUCTION
            and param.get("bit_length") == self.bit_length
        )

    def _generate_and_save(self, param_file: str) -> None:
        # the factorization is a decryption trapdoor (c^lambda reveals x given
        # two ciphertexts of a slot): keep it local, never persist it
        p, _ = generate_group_primes(self.bit_length)
        while True:
            q, _ = generate_group_primes(self.bit_length)
            if q != p:
                break
        self.n_mod = p * q
        del p, q
        self.n_square = self.n_mod * self.n_mod
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "bit_length": self.bit_length,
                    "n_mod": gp.digits(self.n_mod),
                },
                f,
            )

    def _validate_bounds(self) -> None:
        # Fig. 3: X, Y < sqrt(N / 2n), i.e. 2 n X^2 < N
        for name, bound in (("bound_x", self.bound_x), ("bound_y", self.bound_y)):
            if 2 * self.n_slots * gp.mpz(bound) ** 2 >= self.n_mod:
                raise FEValidationError(
                    "{} must satisfy 2 * n * eta * {}^2 < N; increase bit_length".format(
                        name, name
                    )
                )

    def setup(self) -> None:
        """Sample one Gaussian secret ``s_{i,j}`` per client slot."""
        self.sigma = dcr_sigma(self.n_mod, self.n_slots, self.bound_x, self.stat_param)
        self.msk = {
            "s": {
                nid: discrete_gaussian_bigint(self.sigma, self.eta)
                for nid in self.lst_nid
            }
        }
        logger.info("MCFE Paillier (ABMPSW20 DCR) setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters."""
        return {
            "n_mod": gp.digits(self.n_mod),
            "n_square": gp.digits(self.n_square),
            "eta": self.eta,
            "n": self.n,
            "sec_param": self.sec_param,
            "bit_length": self.bit_length,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return the encryption key ``{s}`` of client *nid* (``None`` if unknown)."""
        if nid not in self.msk["s"]:
            return None
        return {"s": [gp.digits(v) for v in self.msk["s"][nid]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the functional key ``k = sum y_{i,j} s_{i,j}`` (over Z).

        Args:
            sid: Session identifier (unused).
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to integer weight lists).

        Returns:
            Dict with ``k``, a single integer aggregated over all clients.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If a client id is unknown or a weight is
                malformed or exceeds ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE Paillier decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or not fusion_weight:
            raise FEValidationError("invalid fusion weights provided, need a dict")

        k = gp.mpz(0)
        for nid, weights in fusion_weight.items():
            if nid not in self.msk["s"]:
                raise FEValidationError("unknown client {} in fusion weight".format(nid))
            if len(weights) > self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            for s_j, y_j in zip(self.msk["s"][nid], weights):
                if abs(int(y_j)) > self.bound_y:
                    raise FEValidationError("fusion weight exceeds configured bound_y")
                k += s_j * gp.mpz(int(y_j))
        return {"k": gp.digits(k)}


class MCFEPaillier(IPFEAbsCrypto):
    """Crypto operations for DCR-based multi-client inner-product FE (ABMPSW20 Fig. 3)."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for clients, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt one client's integer vector under a label.

        Args:
            lst_pt: Integer plaintext vector of length at most ``eta``.
            label: Encryption label; a client must not encrypt twice under
                the same label.

        Returns:
            Dict with ``c`` (one element of ``Z_{N^2}^*`` per slot).

        Raises:
            FEKeyError: If public parameters or encryption keys are missing.
            FEValidationError: If the plaintext is malformed or out of bound.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) > len(self.sk["s"]):
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > self.pp["bound_x"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = n_mod * n_mod
        h_l = _hash_label(label, n_mod)
        c = []
        for x_j, s_j in zip(lst_pt, self.sk["s"]):
            x_term = (1 + gp.mpz(int(x_j)) * n_mod) % n_square  # (1 + N)^x
            c.append(gp.digits(x_term * pow_signed(h_l, gp.mpz(s_j), n_square) % n_square))
        return {"c": c}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Recover ``sum_i <x_i, y_i>`` from all clients' ciphertexts under one label.

        Args:
            dct_ct: Mapping of client IDs to ciphertext dicts.
            dk: Functional key ``{k}`` derived for *fusion_weight*.
            fusion_weight: Mapping of client IDs to weight lists.
            label: The encryption label.

        Returns:
            The inner product (centred modulo ``N``), or ``None`` if the
            result is not a valid decryption (e.g. wrong label or key).

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If ciphertexts and weights cover different clients.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for decryption")
        if not dk or "k" not in dk:
            raise FEKeyError("no decryption key provided")
        if dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")

        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = n_mod * n_mod
        acc = gp.mpz(1)
        for nid, ct in dct_ct.items():
            if len(ct["c"]) != len(fusion_weight[nid]):
                raise FESchemeError(
                    "ciphertext and fusion weight of {} differ in length".format(nid)
                )
            for c_j, y_j in zip(ct["c"], fusion_weight[nid]):
                acc = acc * pow_signed(gp.mpz(c_j), gp.mpz(int(y_j)), n_square) % n_square
        h_l = _hash_label(label, n_mod)
        acc = acc * pow_signed(h_l, -gp.mpz(dk["k"]), n_square) % n_square

        if (acc - 1) % n_mod != 0:
            logger.error("invalid decryption: wrong label, key or ciphertexts")
            return None
        ret = ((acc - 1) % n_square) // n_mod
        if ret > n_mod // 2:
            ret -= n_mod
        return int(ret)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``), one derived label per element.

        Args:
            lst_ndarray: List of numpy arrays.
            **kwargs: Must include ``label``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        lst_ndarray_ct = []
        for l, ary_src in enumerate(lst_ndarray):
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)], _element_label(label, l, i))
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate-decrypt ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight`` and ``label``.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight`` and ``label``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(
                    dct_ct, dk, fusion_weight, _element_label(label, l, i)
                )
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
