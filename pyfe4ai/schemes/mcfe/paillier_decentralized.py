"""
Decentralized Paillier-Based Multi-Client Inner-Product Functional Encryption

This module is a **research-oriented Python prototype** — not a verbatim
transcription of any single paper.  The design draws on two main references:

  * Decentralized DK-share derivation pattern (correlated-noise v-shares
    with sum_i v_i = 0, label-masked encryption, per-client local DK
    computation):
    Inspired by — Abdalla, Benhamouda, Kohlweiss, Waldner,
    "Decentralizing Inner-Product Functional Encryption", PKC 2019.

  * Paillier-based inner-product FE building block (Paillier encryption
    with homomorphic aggregation, L-function extraction):
    Inspired by — Agrawal, Libert, Stehlé,
    "Fully Secure Functional Encryption for Inner Products, from Standard
    Assumptions", CRYPTO 2016.

Key simplifications relative to the papers:

  - Trusted setup (a single party samples all client secrets) rather than
    the interactive DKG protocol described in the PKC 2019 compiler.
  - Label masking uses SHA-256 hashed to a scalar (``md5_hash`` alias),
    not the paper's random-oracle-based PRF family.
  - Paillier generator selection follows a simplified safe-prime pipeline
    rather than the full CRS generation of the CRYPTO 2016 scheme.
  - No formal simulation-based security proof accompanies this code;
    correctness has been verified via unit tests only.

* type:     public-key encryption
* setting:  Integer based

"""

from __future__ import annotations

import json
import logging
import os
import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import generate_group_primes
from pyfe4ai.utils.crypto_utils import md5_hash
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.sampling_utils import random_below

logger = logging.getLogger(__name__)

TYPE_DMCFE_PAILLIER = "dMCFE_PAILLIER"


class DecentralizedMCFEPaillierKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for decentralized Paillier-based multi-client inner-product FE."""

    _scheme_type = TYPE_DMCFE_PAILLIER
    """Trusted setup that distributes per-client secret shares.

    Each client receives its own ``(pk, s, u, v)`` so that functional
    decryption keys can later be derived *locally* by the clients
    without contacting a central authority.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.bit_length = config.get("bit_length", 128)
        self.bound_x = config.get("bound_x", 100)
        self.bound_y = config.get("bound_y", 100)
        if not self.n:
            raise ValueError("need to provide number of clients")
        self._load_parameters()
        self._validate_bounds()

    # ------------------------------------------------------------------
    # Parameter persistence
    # ------------------------------------------------------------------

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.n_mod = gp.mpz(param["n_mod"])
        self.n_square = gp.mpz(param["n_square"])
        self.g = gp.mpz(param["g"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("bit_length") == self.bit_length
            and param.get("bound_x") == self.bound_x
            and param.get("bound_y") == self.bound_y
        )

    def _generate_and_save(self, param_file: str) -> None:
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

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "bit_length": self.bit_length,
                    "bound_x": self.bound_x,
                    "bound_y": self.bound_y,
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "n_mod": gp.digits(self.n_mod),
                    "n_square": gp.digits(self.n_square),
                    "g": gp.digits(self.g),
                },
                f,
            )

    def _validate_bounds(self) -> None:
        x_square_l = gp.mpz(2 * self.n * self.eta * (self.bound_x ** 2))
        y_square_l = gp.mpz(2 * self.n * self.eta * (self.bound_y ** 2))
        if self.n_mod <= x_square_l:
            raise ValueError("bound_x and eta are too large for the chosen bit_length")
        if self.n_mod <= y_square_l:
            raise ValueError("bound_y and eta are too large for the chosen bit_length")

    # ------------------------------------------------------------------
    # Setup — distributes per-client shares
    # ------------------------------------------------------------------

    def setup(self) -> None:
        rand = _CSPRNG
        sk_bound = max(self.sec_param, self.bound_y * self.eta)

        # Per-client secrets: s_i (encryption secret), u_i (label mask)
        s = {
            nid: [gp.mpz(rand.randint(-sk_bound, sk_bound)) for _ in range(self.eta)]
            for nid in self.lst_nid
        }
        u = {
            nid: [gp.mpz(rand.randint(-sk_bound, sk_bound)) for _ in range(self.eta)]
            for nid in self.lst_nid
        }

        # Correlated noise shares v_i for decentralized DK derivation.
        # Constraint: sum_i v_i = 0 (coordinate-wise, over eta*n positions).
        v = {}
        nid_selected = _CSPRNG.sample(self.lst_nid, 1)[0]
        for nid in self.lst_nid:
            if nid != nid_selected:
                v[nid] = [
                    gp.mpz(rand.randint(-sk_bound, sk_bound))
                    for _ in range(self.eta * self.n)
                ]
        lst_v = list(v.values())
        v[nid_selected] = (
            [-sum(vals) for vals in zip(*lst_v)]
            if lst_v
            else [gp.mpz(0) for _ in range(self.eta * self.n)]
        )

        # Public keys for Paillier encryption
        pk = {
            nid: [pow_signed(self.g, s_i, self.n_square) for s_i in vec]
            for nid, vec in s.items()
        }

        self.msk = {"s": s, "u": u, "v": v}
        self.mpk = {
            "n_mod": self.n_mod,
            "n_square": self.n_square,
            "g": self.g,
            "pk": pk,
        }
        logger.info("Decentralized MCFE Paillier setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "n_mod": gp.digits(self.mpk["n_mod"]),
            "n_square": gp.digits(self.mpk["n_square"]),
            "g": gp.digits(self.mpk["g"]),
            "eta": self.eta,
            "n": self.n,
            "lst_nid": self.lst_nid,
            "sec_param": self.sec_param,
            "bit_length": self.bit_length,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Perform the get_private_keys operation.

            Args:
                nid: Node identifier.
        """
        if nid not in self.mpk["pk"]:
            return None
        return {
            "pk": [gp.digits(v) for v in self.mpk["pk"][nid]],
            "s": [gp.digits(v) for v in self.msk["s"][nid]],
            "u": [gp.digits(v) for v in self.msk["u"][nid]],
            "v": [gp.digits(v) for v in self.msk["v"][nid]],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not supported: clients derive functional-key shares locally.

            Args:
                sid: Session / decryption-key identifier.

            Raises:
                NotImplementedError: Always.
        """
        raise NotImplementedError(
            "{} has no central decryption-key derivation: in decentralized "
            "MCFE each client derives its key share with "
            "derive_function_decryption_key_share() and the aggregator "
            "combines them with combine_function_decryption_key_share()".format(
                type(self).__name__
            )
        )


class DecentralizedMCFEPaillier(IPFEAbsCrypto):
    """Decentralized Paillier MCFE: each client encrypts and derives
    functional decryption key shares locally.  A combiner merges shares
    and decrypts the inner product.

    Protocol flow:
        1. Each client i encrypts x_i under label ℓ.
        2. Given a function vector y = (y_1, …, y_n), each client i
           locally computes a DK share from (s_i, u_i, v_i, y).
        3. A combiner collects DK shares, aggregates, then decrypts
           ⟨x, y⟩ from the ciphertexts.
    """

    scheme_type = TYPE_DMCFE_PAILLIER

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    # ------------------------------------------------------------------
    # Encryption (identical to centralized Paillier MCFE)
    # ------------------------------------------------------------------

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt plaintext and return a ciphertext dict.

            Args:
                lst_pt: Integer plaintext vector.
                label: Encryption label for replay protection.
        """
        if not self._has_public_parameters():
            raise ValueError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise ValueError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise ValueError("invalid size of input plaintext")
        if any(abs(int(v)) > self.pp["bound_x"] for v in lst_pt):
            raise ValueError("plaintext exceeds configured bound_x")

        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = gp.mpz(self.pp["n_square"])
        g = gp.mpz(self.pp["g"])
        pk = [gp.mpz(v) for v in self.sk["pk"]]
        u = [gp.mpz(v) for v in self.sk["u"]]
        label_hash = md5_hash(label, n_mod)

        r = random_below(n_mod // 4)
        c0 = gp.powmod(g, r, n_square)
        c1 = []
        for x_i, u_i, pk_i in zip(lst_pt, u, pk):
            masked = gp.mpz(x_i) + u_i * label_hash
            x_term = (gp.mpz(1) + masked * n_mod) % n_square
            ct_i = (x_term * gp.powmod(pk_i, r, n_square)) % n_square
            c1.append(gp.digits(ct_i))
        return {"ct0": gp.digits(c0), "ct1": c1}

    # ------------------------------------------------------------------
    # Decentralized functional decryption key derivation
    # ------------------------------------------------------------------

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        """Derive this client's DK share for the given function vector.

        Each client i computes:
            dk0_i = sum_j  s_i[j] * y_i[j]
            dk1_i = sum_j  u_i[j] * y_i[j]  +  sum_k  v_i[k] * y_flat[k]

        where y_flat is the concatenation of y vectors for all clients
        in a fixed (sorted) order.
        """
        if not self._has_private_keys():
            raise ValueError("no private keys provided for key-share derivation")
        if self.id not in fusion_weight:
            raise ValueError("this client's id is not in fusion_weight")

        y_self = fusion_weight[self.id]
        eta = self.pp["eta"]
        if len(y_self) != eta:
            raise ValueError("fusion weight length mismatch for client {}".format(self.id))
        if any(abs(int(v)) > self.pp["bound_y"] for v in y_self):
            raise ValueError("fusion weight exceeds configured bound_y")

        s = [gp.mpz(v) for v in self.sk["s"]]
        u = [gp.mpz(v) for v in self.sk["u"]]
        v = [gp.mpz(v) for v in self.sk["v"]]

        # dk0: s_i · y_i
        dk0 = sum(s[j] * gp.mpz(y_self[j]) for j in range(eta))

        # dk1: u_i · y_i  +  v_i · y_flat
        dk1_local = sum(u[j] * gp.mpz(y_self[j]) for j in range(eta))

        # Build y_flat in a fixed order (sorted by nid)
        y_flat = []
        for nid in sorted(fusion_weight.keys()):
            if nid not in fusion_weight:
                raise ValueError("fusion weight missing client {}".format(nid))
            y_flat.extend([gp.mpz(v) for v in fusion_weight[nid]])

        if len(v) < len(y_flat):
            raise ValueError("private key v-share length mismatch")
        v_dot = sum(v[k] * y_flat[k] for k in range(len(y_flat)))

        return {"dk0": gp.digits(dk0), "dk1": gp.digits(dk1_local + v_dot)}

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Aggregate per-client DK shares.

        Because sum_i v_i = 0, the correlated-noise terms cancel:
            dk0 = {nid: dk0_nid}
            dk1 = sum_i dk1_i  =  sum_i (u_i · y_i)
        """
        dk0 = {nid: share["dk0"] for nid, share in dct_dk_shares.items()}
        dk1 = sum(gp.mpz(share["dk1"]) for share in dct_dk_shares.values())
        return {"dk0": dk0, "dk1": gp.digits(dk1)}

    # ------------------------------------------------------------------
    # Decryption
    # ------------------------------------------------------------------

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Decrypt the inner product ⟨x, y⟩ from aggregated ciphertexts.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
                label: Encryption label for replay protection.
        """
        if not dk:
            raise ValueError("no decryption key provided")
        if dct_ct.keys() != fusion_weight.keys() or len(dct_ct) != len(dk["dk0"]):
            raise ValueError("inconsistent input among ct, dk, fusion weight")

        n_mod = gp.mpz(self.pp["n_mod"])
        n_square = gp.mpz(self.pp["n_square"])
        label_hash = md5_hash(label, n_mod)

        acc = gp.mpz(1)
        for nid, ct in dct_ct.items():
            # g^{-s_i · y_i · r}
            acc *= pow_signed(gp.mpz(ct["ct0"]), -gp.mpz(dk["dk0"][nid]), n_square)
            acc %= n_square
            # product of ct1[j]^{y_i[j]}
            for ct_j, y_j in zip(ct["ct1"], fusion_weight[nid]):
                acc *= pow_signed(gp.mpz(ct_j), gp.mpz(y_j), n_square)
                acc %= n_square

        # Cancel label masking: g^{sum u_i · y_i · H(label) · n}
        correction = (
            gp.mpz(1) + gp.mpz(dk["dk1"]) * label_hash * n_mod
        ) % n_square
        acc = (acc * gp.invert(correction, n_square)) % n_square

        # L-function: (acc - 1) / n_mod
        ret = ((acc - 1) % n_square) // n_mod
        n_half = n_mod // 2
        if ret > n_half:
            ret -= n_mod
        return int(ret)

    # ------------------------------------------------------------------
    # ndarray helpers
    # ------------------------------------------------------------------

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Perform the encrypt_lst_ndarray operation.

            Args:
                lst_ndarray: List of numpy arrays to encrypt element-wise.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)], label)
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise ValueError("need to provide decryption key, fusion weight and label")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, label)
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise ValueError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )
