"""
LWE multi-input inner-product FE (adaptive security).

| Abdalla, Catalano, Fiore, Gay, Ursu:
| "Multi-Input Functional Encryption for Inner Products: Function-Hiding
|  Realizations and Constructions without Pairings", CRYPTO 2018
| (ePrint 2017/972) -- Fig. 4 (transformation for bounded-norm inner
| products) instantiated with the LWE IPFE of Agrawal, Libert, Stehle
| (CRYPTO 2016, Section 4.1), recalled as Fig. 8 with its two-step decryption
| in Section 4.2.

* type:     private-key multi-input encryption (MIFE, LWE)
* setting:  Integer based (Z_q, Delta = floor(q / K))

Construction (slot ``i`` is client ``i``, ``L = q``):

* Setup*:   parameters of ALS16 derived as for vectors of dimension ``n eta``
  and plaintext bound ``3X`` (Section 4.2, footnote 14 and Fig. 4);
  public ``A``; per slot ``Z_i <- D`` (ALS16 key distribution),
  ``U_i = Z_i A`` and a one-time pad ``u_i <- Z_q^eta``; client
  ``i`` keeps ``(U_i, u_i)``.
* Encrypt:  ``w = x_i + u_i mod q``; ``c0 = A s + e0``,
  ``c1 = U_i s + e1 + Delta * w mod q``.
* KeyGen:   per slot ``Z_i^T y_i mod q``, and ``z = sum_i <u_i, y_i> mod q``.
* Decrypt:  Dec1 per slot ``<y_i, c1> - <Z_i^T y_i, c0> = Delta <w_i, y_i> + noise_i``;
  ``mu = round(centre(sum_i Dec1_i - Delta z) / Delta)`` (Dec2).

The pad hides each slot's partial result; versions <= 0.2.0 had no pad (the
key holder decoded every ``<x_i, y_i>``) and derived single-slot parameters.
Deviation: all slots share ``A`` instead of running Setup independently per
slot (``Z_i`` and ``u_i`` are independent). A decryption needs ciphertexts of
every slot named in the key.
"""

from __future__ import annotations

import json
import secrets
import logging
import os
import tempfile

import gmpy2 as gp
import numpy as np

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.primitives.lattice.lwe_utils import decode_scaled
from pyfe4ai.core.constants import CryptoCONST

from pyfe4ai.primitives.lattice.lwe_utils import derive_fullysec_lwe_parameters
from pyfe4ai.primitives.lattice.matrix_utils import digits_to_matrix
from pyfe4ai.primitives.lattice.matrix_utils import digits_to_vector
from pyfe4ai.primitives.lattice.matrix_utils import matrix_to_digits
from pyfe4ai.primitives.lattice.matrix_utils import matmul_mod
from pyfe4ai.primitives.lattice.matrix_utils import matvec_mod
from pyfe4ai.primitives.lattice.matrix_utils import transpose
from pyfe4ai.primitives.lattice.matrix_utils import vecdot_mod
from pyfe4ai.primitives.lattice.matrix_utils import vector_to_digits
from pyfe4ai.primitives.sampling import discrete_gaussian_matrix
from pyfe4ai.primitives.sampling import rand_uniform_vector
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.versions import record_matches
from pyfe4ai.core.versions import MIFE_FULLYSEC_LWE_RECORD
from pyfe4ai._validation import scale_to_int
from pyfe4ai._validation import strict_int
from pyfe4ai.primitives.lattice.lwe_matrix import SeededMatrix

logger = logging.getLogger(__name__)


def _seeded_matrix(seed: str, rows: int, cols: int, q) -> list:
    """The public matrix ``A``, expanded from *seed* (a cache stores the seed only)."""
    return [[gp.mpz(v) for v in row] for row in SeededMatrix(seed, rows, cols, int(q)).entries()]


class MIFEFullySecLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for fully secure LWE-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_FULLYSEC_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE fully secure LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients, required), ``lst_nid``, ``eta``, ``lwe_n`` (default
                64), ``bound_x`` and ``bound_y``. The ``standard`` profile
                refuses this scheme.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.lwe_n = config.get("lwe_n", 64)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.k = gp.mpz(param["k"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma_q = gp.mpz(param["l_sigma_q"])
        self.sigma1 = float(param["sigma1"])
        self.l_sigma1 = gp.mpz(param["l_sigma1"])
        self.sigma2 = float(param["sigma2"])
        self.l_sigma2 = gp.mpz(param["l_sigma2"])
        self.a_seed = param["A_seed"]
        self.A = _seeded_matrix(self.a_seed, self.m, self.lwe_n, self.q)

    def _param_verification(self, param: dict) -> bool:
        if not record_matches(param, {
            "construction": MIFE_FULLYSEC_LWE_RECORD, "sec_param": self.sec_param, "eta": self.eta, "n": self.n,
            "lwe_n": self.lwe_n, "bound_x": gp.digits(self.bound_x), "bound_y": gp.digits(self.bound_y),
        }):
            return False
        expected = derive_fullysec_lwe_parameters(3 * self.bound_x, self.bound_y, self.n * self.eta, self.lwe_n)
        names = ("k", "q", "m", "sigma_q", "l_sigma_q", "sigma1", "l_sigma1", "sigma2", "l_sigma2")
        if not record_matches(param, {
            name: value if isinstance(value, (int, float)) else gp.digits(value) for name, value in zip(names, expected)
        }):
            return False
        seed = param.get("A_seed")
        return isinstance(seed, str) and len(bytes.fromhex(seed)) == 32

    def _generate_and_save(self, param_file: str) -> None:
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
            3 * self.bound_x, self.bound_y, self.n * self.eta, self.lwe_n
        )
        self.a_seed = secrets.token_hex(32)
        self.A = _seeded_matrix(self.a_seed, self.m, self.lwe_n, self.q)

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=os.path.dirname(param_file), delete=False
        ) as f:
            json.dump(
                {
                    "construction": MIFE_FULLYSEC_LWE_RECORD,
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
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
                    "A_seed": self.a_seed,
                },
                f,
            )
            temp_name = f.name
        os.replace(temp_name, param_file)

    def setup(self) -> None:
        half_cols = self.m // 2
        z = {}
        u = {}
        for nid in self.lst_nid:
            z_nid = [[gp.mpz(0) for _ in range(self.m)] for _ in range(self.eta)]
            for i in range(self.eta):
                for j in range(self.m):
                    if j < half_cols:
                        sampled = discrete_gaussian_matrix(1, 1, self.sigma1)[0][0]
                    else:
                        sampled = discrete_gaussian_matrix(1, 1, self.sigma2)[0][0]
                        if j - half_cols == i:
                            sampled += 1
                    z_nid[i][j] = sampled
            z[nid] = z_nid
            u[nid] = matmul_mod(z_nid, self.A, self.q)

        self.msk = {
            "z": z,
            "pad": {nid: rand_uniform_vector(self.eta, self.q) for nid in self.lst_nid},
        }
        self.mpk = {"u": u}
        logger.info("MIFE fully secure LWE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": matrix_to_digits(self.A),
            "k": gp.digits(self.k),
            "q": gp.digits(self.q),
            "m": self.m,
            "eta": self.eta,
            "n": self.n,
            "lwe_n": self.lwe_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma_q": self.sigma_q,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s encryption key (``pk`` and its pad ``u``).

        Args:
            nid: Client identifier.

        Returns:
            The key dict, or ``None`` for an unknown client.
        """
        if nid not in self.mpk["u"]:
            return None
        return {
            "pk": matrix_to_digits(self.mpk["u"][nid]),
            "u": vector_to_digits(self.msk["pad"][nid]),
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
            raise FEKeyError("need credentials for MIFE fully secure LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        z_y = {}
        pad_term = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(strict_int(v, "fusion weight")) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            y_vec = [gp.mpz(v) for v in y]
            z_t = transpose(self.msk["z"][nid])
            z_y[nid] = vector_to_digits(matvec_mod(z_t, y_vec, self.q))
            pad_term += sum(u * y_j for u, y_j in zip(self.msk["pad"][nid], y_vec))
        return {"sk_y": z_y, "z": gp.digits(pad_term % self.q)}


class MIFEFullySecLWE(IPFEAbsCrypto):
    """Crypto operations for fully secure LWE-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE fully secure LWE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = digits_to_matrix(self.pp["A"])
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        q = gp.mpz(self.pp["q"])
        k = gp.mpz(self.pp["k"])
        pk = digits_to_matrix(self.sk["pk"])
        r = rand_uniform_vector(self.pp["lwe_n"], q)
        e0 = discrete_gaussian_matrix(1, self.pp["m"], self.pp["sigma_q"])[0]
        e1 = discrete_gaussian_matrix(1, self.pp["eta"], self.pp["sigma_q"])[0]

        c0 = matvec_mod(self.A, r, q)
        c0 = [(a + b) % q for a, b in zip(c0, e0)]

        if "u" not in self.sk:
            raise FEKeyError("encryption key lacks the one-time pad u")
        delta = q // k
        c1 = matvec_mod(pk, r, q)
        w = [(gp.mpz(int(x)) + gp.mpz(u)) % q for x, u in zip(lst_pt, self.sk["u"])]
        c1 = [(a + b + delta * w_j) % q for a, b, w_j in zip(c1, e1, w)]
        return {"ct0": vector_to_digits(c0), "ct1": vector_to_digits(c1)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        if not dk or "z" not in dk:
            raise FEKeyError("no decryption key provided")
        if set(dct_ct) != set(dk["sk_y"]) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")

        q = gp.mpz(self.pp["q"])
        k = gp.mpz(self.pp["k"])
        delta = q // k
        total = -delta * gp.mpz(dk["z"])
        for nid in dct_ct:
            y = fusion_weight[nid]
            if len(y) != self.pp["eta"]:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(strict_int(v, "fusion weight")) > gp.mpz(self.pp["bound_y"]) for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            y_vec = [gp.mpz(v) for v in y]
            ct0 = digits_to_vector(dct_ct[nid]["ct0"])
            ct1 = digits_to_vector(dct_ct[nid]["ct1"])
            z_y = digits_to_vector(dk["sk_y"][nid])

            total += vecdot_mod(y_vec, ct1, q) - vecdot_mod(z_y, ct0, q)
        bound = self.pp["n"] * self.pp["eta"] * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        return decode_scaled(total % q, delta, q, bound)

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
