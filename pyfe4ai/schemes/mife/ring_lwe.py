"""
Ring-LWE multi-input inner-product FE.

| Bermudo Mera, Karmakar, Marc, Soleimanian:
| "Efficient Lattice-Based Inner-Product Functional Encryption", PKC 2022
| (ePrint 2021/046) -- Section 6, "MIFE-Compiler" (the transformation of
| Abdalla, Catalano, Fiore, Gay, Ursu, CRYPTO 2018) applied to their
| Ring-LWE IPFE, which has 2-step decryption with a linear encoding
| (Proposition 1).

* type:     private-key multi-input encryption (MIFE, Ring-LWE)
* setting:  R_q = Z_q[x] / (x^ring_n + 1), Delta = floor(q / p)

Construction (slot ``i`` is client ``i``; every row of the plaintext matrix is
one coordinate, its coefficients a batch of independent values):

* Setup:    public ``a``; per slot ``s_{i,row}`` (Gaussian),
  ``pk_{i,row} = a s_{i,row} + e``, and a one-time pad
  ``u_{i,row} <- R_q`` (uniform coefficients); client ``i`` keeps
  ``(pk_i, u_i)``.
* Encrypt:  ``w = x_i + u_i`` (coefficient-wise mod q);
  ``ct0_row = pk_row r + e_row + Delta w_row``, ``ct1 = a r + e``.
* KeyGen:   per slot ``sk_i = sum_row y_{i,row} s_{i,row}`` and
  ``z = sum_i sum_row y_{i,row} u_{i,row}`` (coefficient-wise mod q).
* Decrypt:  Dec1 per slot ``sum_row y_row ct0_row - ct1 sk_i``;
  Dec2 decodes ``sum_i Dec1_i - Delta z`` coefficient-wise.

The pad hides each slot's partial result; versions <= 0.2.0 had no pad and
decoded every slot separately. Parameters are derived as for a single-input
scheme of dimension ``n eta`` (ACFGU18 Setup*), since only the sum over all
slots is decoded, and with the Gaussian widths of BMMS21 Sec. 4.2 for
messages of norm ``3X``: ACFGU18 Fig. 4 needs an IPFE for the class
``F_{m,3X,Y}`` (footnote 12; the hybrids of Lemma 3 encrypt
``x^j - x^1 + x'^1``), while ``p`` stays sized for ``X``. Versions <= 0.3.0
used ``X``, which meets that condition only for nine or more clients.
Deviation: all slots share ``a`` (secrets and pads are independent per
slot). A decryption needs ciphertexts of every slot named in
the key.
"""

from __future__ import annotations

import json
import secrets
import logging

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.schemes.profiles import STANDARD
from pyfe4ai.utils.ring_lwe_utils import bmms21_selective_parameters
from pyfe4ai.utils.ring_lwe_utils import canonical_ring_ciphertext
from pyfe4ai.utils.ring_lwe_utils import check_ring_lwe
from pyfe4ai.utils.ring_lwe_utils import mat_vec_mul
from pyfe4ai.utils.ring_lwe_utils import poly_add
from pyfe4ai.utils.ring_lwe_utils import poly_mul
from pyfe4ai.utils.ring_lwe_utils import poly_neg
from pyfe4ai.utils.ring_lwe_utils import resolve_ring_n
from pyfe4ai.utils.ring_lwe_utils import transpose
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

_CONSTRUCTION = "acfgu18-bmms21/v5"  # sized for n * eta, p = 2T + 2, Sec. 4.2 widths for 3X, seeded a


class MIFERingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Ring-LWE-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE Ring-LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients, required), ``lst_nid``, ``sec_param`` (``kappa`` of
                the noise bounds), ``eta``, ``bound_x``, ``bound_y`` and
                ``ring_n`` (default
                :data:`~pyfe4ai.utils.ring_lwe_utils.DEFAULT_RING_N`;
                ``"auto"`` for the smallest dimension meeting the 128-bit
                table, the ``standard`` profile's choice).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self.ring_n = resolve_ring_n(config.get("ring_n"), self._params_for)
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_ring_lwe(self.ring_n, self.q, self._params_for(self.ring_n)[1])

    def _bmms21(self, ring_n: int) -> dict:
        """BMMS21 Sec. 4.2 parameters for *ring_n*: dimension ``n eta``,
        decoding bound ``X``, widths for messages of norm ``3X`` (ACFGU18
        Fig. 4, see the module docstring)."""
        return bmms21_selective_parameters(
            self.sec_param, self.n * self.eta, self.bound_x, self.bound_y, ring_n,
            security_bound_x=3 * self.bound_x,
        )

    def _bmms21_tuple(self, ring_n: int) -> tuple:
        """``(p, q, sigma1, sigma2, sigma3)`` of :meth:`_bmms21`."""
        P = self._bmms21(ring_n)
        return P["p"], P["q"], P["sigma1"], P["sigma2"], P["sigma3"]

    def _params_for(self, ring_n: int) -> tuple:
        """``(q, rlwe_width)`` that :meth:`_generate_and_save` derives for *ring_n*
        (the width is that of the RLWE problems of the BMMS21 proof)."""
        return self._modulus_for(ring_n), self._bmms21(ring_n)["rlwe_width"]

    def _modulus_for(self, ring_n: int):
        """Modulus ``q`` that :meth:`_generate_and_save` derives for *ring_n*."""
        return self._bmms21_tuple(ring_n)[1]

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
        from it (``q`` as :meth:`_modulus_for` gives it); only the seed of ``a``
        is random."""
        if not (
            param.get("construction") == _CONSTRUCTION
            and param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("ring_n") == self.ring_n
        ):
            return False
        p, _, sigma1, sigma2, sigma3 = self._bmms21_tuple(self.ring_n)
        q = self._modulus_for(self.ring_n)
        return (param.get("p"), param.get("q")) == (gp.digits(p), gp.digits(q)) and (
            param.get("sigma1"), param.get("sigma2"), param.get("sigma3")
        ) == (sigma1, sigma2, sigma3)

    def _generate_and_save(self, param_file: str) -> None:
        # only the sum over all slots is decoded: size p (and the noise) as
        # for a single-input scheme of dimension n * eta (ACFGU18 Setup*),
        # with the widths for 3X (ACFGU18 Fig. 4)
        self.p, self.q, self.sigma1, self.sigma2, self.sigma3 = self._bmms21_tuple(self.ring_n)
        self.a_seed = secrets.token_hex(32)
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)

        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
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
        sk = {
            nid: discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
            for nid in self.lst_nid
        }
        pk = {}
        for nid in self.lst_nid:
            noise = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1)
            pk_nid = []
            for i in range(self.eta):
                pk_i = poly_mul(self.A, sk[nid][i], self.q)
                pk_nid.append(poly_add(pk_i, noise[i], self.q))
            pk[nid] = pk_nid
        pad = {
            nid: [rand_uniform_vector(self.ring_n, self.q) for _ in range(self.eta)]
            for nid in self.lst_nid
        }
        self.msk = {"sk": sk, "u": pad}
        self.mpk = {"pk": pk}
        logger.info("MIFE Ring-LWE (ACFGU18 / BMMS21) setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "A": [gp.digits(v) for v in self.A],
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "eta": self.eta,
            "n": self.n,
            "ring_n": self.ring_n,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "sigma3": self.sigma3,
            "sec_param": self.sec_param,
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
            "pk": [[gp.digits(v) for v in row] for row in self.mpk["pk"][nid]],
            "u": [[gp.digits(v) for v in row] for row in self.msk["u"][nid]],
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
            raise FEKeyError("need credentials for MIFE Ring-LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        sk_y = {}
        z = [gp.mpz(0)] * self.ring_n
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = int_vector(
                fusion_weight[nid], "fusion weight of {}".format(nid),
                length=self.eta, bound=int(self.bound_y),
            )
            sk_t = transpose(self.msk["sk"][nid])
            y_vec = [gp.mpz(v) for v in y]
            sk_y[nid] = [gp.digits(v) for v in mat_vec_mul(sk_t, y_vec, self.q)]
            for y_row, u_row in zip(y_vec, self.msk["u"][nid]):
                z = [acc + y_row * u for acc, u in zip(z, u_row)]
        return {"sk_y": sk_y, "z": [gp.digits(v % self.q) for v in z]}


class MIFERingLWE(IPFEAbsCrypto):
    """Crypto operations for Ring-LWE-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE Ring-LWE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = [gp.mpz(v) for v in self.pp["A"]]
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def _check_plaintext(self, matrix_pt) -> list:
        """Return *matrix_pt* as ``eta`` integer rows of one length ``1..ring_n``."""
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if not isinstance(matrix_pt, list) or not matrix_pt or not isinstance(matrix_pt[0], list):
            raise FEValidationError("plaintext matrix must be a list of rows")
        if len(matrix_pt) != self.pp["eta"]:
            raise FEValidationError("invalid plaintext row count")
        rows = [int_vector(row, "plaintext row", bound=int(self.pp["bound_x"])) for row in matrix_pt]
        k = len(rows[0])
        if not 1 <= k <= self.pp["ring_n"] or any(len(row) != k for row in rows):
            raise FEValidationError(
                "plaintext rows must all have the same length, between 1 and ring_n"
            )
        if "u" not in self.sk:
            raise FEKeyError("encryption key lacks the one-time pad u")
        return rows

    def _padded_rows(self, matrix_pt, extra_pad=None) -> list:
        """Rows ``x + u (+ extra_pad)`` mod q, zero-extended to ``ring_n``."""
        q = gp.mpz(self.pp["q"])
        ring_n = self.pp["ring_n"]
        rows = []
        for i, (row, u_row) in enumerate(zip(matrix_pt, self.sk["u"])):
            padded = [gp.mpz(int(v)) for v in row] + [gp.mpz(0)] * (ring_n - len(row))
            extra = extra_pad[i] if extra_pad is not None else [0] * ring_n
            rows.append([(x + gp.mpz(u) + e) % q for x, u, e in zip(padded, u_row, extra)])
        return rows

    def _encrypt_rows(self, w_rows: list, k: int) -> dict:
        """BMMS21 Ring-LWE encryption of rows already reduced mod q."""
        q = gp.mpz(self.pp["q"])
        ring_n = self.pp["ring_n"]
        delta = q // gp.mpz(self.pp["p"])
        pk = [[gp.mpz(v) for v in row] for row in self.sk["pk"]]
        r = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        noise = discrete_gaussian_matrix(self.pp["eta"], ring_n, self.pp["sigma3"])
        ct0 = []
        for i, w_row in enumerate(w_rows):
            ct0_i = poly_add(poly_mul(pk[i], r, q), noise[i], q)
            ct0.append(poly_add(ct0_i, [delta * w for w in w_row], q))
        e = discrete_gaussian_matrix(1, ring_n, self.pp["sigma2"])[0]
        ct1 = poly_add(poly_mul(self.A, r, q), e, q)
        return {
            "ct0": [[gp.digits(v) for v in row] for row in ct0],
            "ct1": [gp.digits(v) for v in ct1],
            "k": k,
        }

    def encrypt(self, matrix_pt: list[list[int]]) -> dict:
        """Encrypt this slot's matrix: Ring-LWE encryption of ``x + u mod q``.

        Returns:
            Dict with ``ct0`` (one polynomial per row), ``ct1`` and ``k``
            (number of used coefficients).
        """
        rows = self._check_plaintext(matrix_pt)
        return self._encrypt_rows(self._padded_rows(rows), len(rows[0]))

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Recover the batch of inner products ``sum_i <x_i[:, k], y_i>``.

        Returns:
            List with one inner product per batch position (``None`` entries
            if a decoded value exceeds ``n eta X Y``).

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If ciphertexts do not cover exactly the key's slots
                or have different batch lengths.
        """
        if not dk or "z" not in dk:
            raise FEKeyError("no decryption key provided")
        if set(dct_ct) != set(dk["sk_y"]) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")
        batch = {int(ct["k"]) for ct in dct_ct.values()}
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")

        return self._decrypt_with_pad_term(dct_ct, dk["sk_y"], dk["z"], fusion_weight, batch.pop())

    def _decrypt_with_pad_term(self, dct_ct, sk_y_map, z, fusion_weight, k) -> list:
        """Dec2(sum_i Dec1_i - Delta z) for the first *k* coefficients."""
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        total = [(-delta * gp.mpz(v)) % q for v in z]
        for nid, ct in dct_ct.items():
            y = int_vector(
                fusion_weight[nid], "fusion weight of {}".format(nid),
                length=self.pp["eta"], bound=int(self.pp["bound_y"]),
            )
            try:
                _, ct0, ct1 = canonical_ring_ciphertext(ct, int(q), self.pp["eta"], self.pp["ring_n"])
            except FESchemeError as exc:
                raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
            ct0 = [[gp.mpz(v) for v in row] for row in ct0]
            ct1 = [gp.mpz(v) for v in ct1]
            sk_y = [gp.mpz(v) for v in sk_y_map[nid]]
            y_vec = [gp.mpz(int(v)) for v in y]
            # Dec1 of slot nid: Delta <w_i, y_i> + noise (coefficient-wise)
            lhs = mat_vec_mul(transpose(ct0), y_vec, q)
            rhs = poly_mul(ct1, sk_y, q)
            total = poly_add(total, poly_add(lhs, poly_neg(rhs, q), q), q)
        bound = (
            self.pp["n"] * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        )
        return [decode_scaled(v, delta, q, bound) for v in total[:k]]

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
