"""
Adaptively secure Ring-LWE single-input inner-product FE.

| Bermudo Mera, Karmakar, Marc, Soleimanian:
| "Efficient Lattice-Based Inner-Product Functional Encryption", PKC 2022,
| ePrint 2021/046 -- Section 5 (adaptively secure IPFE based on RLWE), with
| Theorem 1 (mhe-RLWE from RLWE), Theorem 2 (Ring-LHL) and the parameter
| constraints of Appendix C.1. It is the ring counterpart of the LWE scheme
| of Agrawal, Libert, Stehle (CRYPTO 2016, :mod:`pyfe4ai.schemes.sife.fullysec_lwe`).

* type:     public-key encryption, adaptive IND security
* setting:  R_q = Z_q[x] / (x^ring_n + 1), q an NTT-friendly prime

Construction (``l = eta`` slots, ``m`` key polynomials, ``Delta = floor(q / p)``):

* Setup:    ``a = (a_1, ..., a_m) <- R_q^m`` uniform with some ``a_j``
  invertible; ``s_i = (s_i1, ..., s_im)``, ``s_ij <- D_sigma1``;
  ``pk_i = <a, s_i>`` (no error term).
* Encrypt:  ``r <- D_sigma2``, ``f0 <- D_sigma2^m``, ``f_i <- D_sigma3``;
  ``ct0 = a r + f0``, ``ct_i = pk_i r + f_i + Delta x_i`` (constant
  coefficient, ``Delta x_i`` linear in ``x_i`` as the proof needs); resampled until some ``ct0_j`` is invertible and
  ``ct0`` is proportional to ``a`` in no CRT slot.
* KeyGen:   ``sk_y = sum_i y_i s_i`` (``m`` polynomials).
* Decrypt:  ``d = sum_i y_i ct_i - <ct0, sk_y>``; round its constant coefficient.

Deviations from the paper:

* ``m >= 5`` key polynomials instead of ``m = 3``. With the constraints of
  Appendix C.1, ``m = 3`` admits no modulus: Theorem 2 (``k = 2``) needs the
  ``s*`` of game G4 -> G5 to have width ``sigma' >= sqrt(n ln(2mn(1 + 1/delta))
  / pi) q^(2/m + eps/2)``, G3 -> G4 needs ``sigma1 >= sqrt(l) 2X sigma'``,
  Theorem 1 needs ``sigma2 > sigma n C sqrt(l + 2)`` with ``C = sqrt(kappa)
  sigma1``, and the decryption noise ``l Y (sqrt(kappa) sigma3 + m n kappa
  sigma1 sigma2)`` then grows like ``q^(4/m)`` while it must stay below
  ``q / 2p``. Theorem 2 holds for every ``m >= 3``, so the proof is
  unchanged. ``m = "auto"`` (default) takes the smallest ``m`` that admits a
  modulus; :func:`pyfe4ai.utils.ring_lwe_utils.bmms21_adaptive_parameters`
  derives the widths and ``q``.
* Signed inputs ``|x| <= X``, ``|y| <= Y``: ``p = 2 l X Y + 2`` with centred
  decoding, ``||x1 - x0||_inf <= 2X``, and complexity leveraging over
  ``(2X + 1)^(2l)`` challenge pairs.
* One vector per ciphertext: batching ``k`` vectors (Remark 1) would make the
  leveraging range over ``(2X + 1)^(2lk)`` pairs.
* Gaussian samples are rounded continuous Gaussians, as elsewhere in the library.

Under the ``standard`` profile ``ring_n = "auto"`` takes the smallest ring
dimension (with the smallest ``m``) whose ``q`` meets
:data:`~pyfe4ai.utils.ring_lwe_utils.RING_LWE_128_LOG_Q` at the width
``sigma`` (about 8) of the RLWE problem of Theorem 1.
"""

from __future__ import annotations

import json
import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.profiles import STANDARD
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.ring_lwe_utils import _get_ntt_root
from pyfe4ai.utils.ring_lwe_utils import bmms21_adaptive_choice
from pyfe4ai.utils.ring_lwe_utils import check_ring_lwe
from pyfe4ai.utils.ring_lwe_utils import decode_vector
from pyfe4ai.utils.ring_lwe_utils import ntt_forward
from pyfe4ai.utils.ring_lwe_utils import ntt_inverse
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int
from pyfe4ai.utils.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

_PARAM_VERSION = "bmms21-sec5/v2"  # a expanded from a stored seed
#: Resampling attempts for ``a`` and for a ciphertext before giving up.
_MAX_TRIES = 64


def encode_message(x: int, q, p) -> gp.mpz:
    """``floor(q / p) x mod q``: linear in ``x``, as the last game of the proof needs."""
    return gp.mpz(q) // gp.mpz(p) * gp.mpz(x) % gp.mpz(q)


def _invertible(poly_ntt: list, q) -> bool:
    """A polynomial is a unit of ``R_q`` iff none of its CRT (NTT) values is 0."""
    return all(v % q for v in poly_ntt)


def independent_at_every_slot(a_ntt: list, c_ntt: list, q) -> bool:
    """``True`` iff ``(a(t), c(t))`` has rank 2 over ``Z_q`` in every CRT slot ``t``.

    *a_ntt* and *c_ntt* hold ``m`` NTT vectors each. Then ``x -> (<a, x>, <c, x>)``
    maps ``R_q^m`` onto ``R_q^2`` (the condition of the Ring-LHL step, which
    the paper states as "``ct0`` is not a multiple of ``a``").
    """
    q = int(q)
    for t in range(len(a_ntt[0])):
        col_a = [int(row[t]) % q for row in a_ntt]
        col_c = [int(row[t]) % q for row in c_ntt]
        j0 = next((j for j, v in enumerate(col_a) if v), None)
        if j0 is None:
            return False
        if all((col_a[j0] * c - col_c[j0] * a) % q == 0 for a, c in zip(col_a, col_c)):
            return False
    return True


def _ntt_all(polys: list, q, root) -> list:
    return [ntt_forward([gp.mpz(v) for v in poly], q, root) for poly in polys]


def _seeded_a(seed: str, q, ring_n: int, m: int) -> list:
    """The ``m`` public polynomials, expanded from *seed* (a cache stores the seed only)."""
    return [seeded_polynomial(seed, q, ring_n, index=j) for j in range(m)]

class SIFEFullySecRingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for the adaptively secure Ring-LWE IPFE of BMMS21 (Sec. 5)."""

    _scheme_type = CryptoCONST.TYPE_SIFE_FULLYSEC_RING_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``
                (``kappa``), ``eta``, ``bound_x``, ``bound_y``, ``ring_n``
                (power of two, ``"auto"``, or ``None`` for the toy default)
                and ``m`` (number of key polynomials, at least 5, or
                ``"auto"``, the default).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self.ring_n_config = config.get("ring_n")
        self.m_config = config.get("m", "auto")
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_ring_lwe(self.ring_n, self.q, self.sigma)

    def _derive(self) -> dict:
        return bmms21_adaptive_choice(
            self.sec_param, self.eta, self.bound_x, self.bound_y, self.ring_n_config,
            self.m_config, standard=self.security_profile == STANDARD,
        )

    def _param_verification(self, param: dict) -> bool:
        """The cached record must match the config and the parameters derived from
        it (only the seed of ``a`` is random; ``a`` must contain a unit)."""
        if not (
            param.get("param_version") == _PARAM_VERSION
            and param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("ring_n_config") == self.ring_n_config
            and param.get("m_config") == self.m_config
            and param.get("profile") == self.security_profile
        ):
            return False
        expected = self._derive()
        if (
            param.get("q") != gp.digits(expected["q"])
            or param.get("p") != gp.digits(expected["p"])
            or (param.get("ring_n"), param.get("m")) != (expected["ring_n"], expected["m"])
            or any(param.get(k) != expected[k] for k in ("sigma", "sigma_lhl", "sigma1", "sigma2", "sigma3"))
        ):
            return False
        q, n = expected["q"], expected["ring_n"]
        a = _seeded_a(param["a_seed"], q, n, expected["m"])
        return any(_invertible(poly, q) for poly in _ntt_all(a, q, _get_ntt_root(q, n)))

    def _apply_parameters(self, param: dict) -> None:
        self.ring_n = int(param["ring_n"])
        self.m = int(param["m"])
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        for key in ("sigma", "sigma_lhl", "sigma1", "sigma2", "sigma3"):
            setattr(self, key, float(param[key]))
        self.a_seed = param["a_seed"]
        self.a = _seeded_a(self.a_seed, self.q, self.ring_n, self.m)

    def _generate_and_save(self, param_file: str) -> None:
        params = self._derive()
        q, n = params["q"], params["ring_n"]
        root = _get_ntt_root(q, n)
        for _ in range(_MAX_TRIES):
            seed = secrets.token_hex(32)
            a = _seeded_a(seed, q, n, params["m"])
            if any(_invertible(poly, q) for poly in _ntt_all(a, q, root)):
                break
        else:
            raise FESchemeError("no invertible element in a; is q an NTT prime?")
        record = {
            "param_version": _PARAM_VERSION,
            "sec_param": self.sec_param,
            "eta": self.eta,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "ring_n_config": self.ring_n_config,
            "m_config": self.m_config,
            "profile": self.security_profile,
            "ring_n": n,
            "m": params["m"],
            "p": gp.digits(params["p"]),
            "q": gp.digits(q),
            "a_seed": seed,
        }
        record.update({k: params[k] for k in ("sigma", "sigma_lhl", "sigma1", "sigma2", "sigma3")})
        self._apply_parameters(record)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(record, f)

    def setup(self) -> None:
        """Sample ``s_i`` (``m`` polynomials per slot) and ``pk_i = <a, s_i>``."""
        q, n = self.q, self.ring_n
        root = _get_ntt_root(q, n)
        a_ntt = _ntt_all(self.a, q, root)
        sk, pk = [], []
        for _ in range(self.eta):
            s_i = discrete_gaussian_matrix(self.m, n, self.sigma1)
            acc = [gp.mpz(0)] * n
            for a_j, s_j in zip(a_ntt, _ntt_all(s_i, q, root)):
                acc = [(u + x * y) % q for u, x, y in zip(acc, a_j, s_j)]
            sk.append(s_i)
            pk.append(ntt_inverse(acc, q, root))
        self.msk = {"sk": sk}
        self.mpk = {"pk": pk}
        logger.info("SIFE FullySec Ring-LWE setup successfully (ring_n=%d, m=%d)", n, self.m)

    def get_public_parameters(self) -> dict:
        """Return ``a``, ``p``, ``q``, ``ring_n``, ``m``, ``eta``, the bounds and the
        encryption widths ``sigma2``, ``sigma3``."""
        return {
            "a": [[gp.digits(v) for v in poly] for poly in self.a],
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "ring_n": self.ring_n,
            "m": self.m,
            "eta": self.eta,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "sigma2": self.sigma2,
            "sigma3": self.sigma3,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return the encryption key: the public keys ``pk_i`` (``eta`` polynomials)."""
        return {"pk": [[gp.digits(v) for v in poly] for poly in self.mpk["pk"]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive ``sk_y = sum_i y_i s_i`` (``m`` polynomials mod ``q``).

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for SIFE FullySec Ring-LWE key generation")
        y = int_vector(
            credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=self.bound_y
        )
        sk_y = []
        for j in range(self.m):
            acc = [gp.mpz(0)] * self.ring_n
            for y_i, s_i in zip(y, self.msk["sk"]):
                acc = [u + y_i * v for u, v in zip(acc, s_i[j])]
            sk_y.append([gp.digits(v % self.q) for v in acc])
        return {"sk_y": sk_y, "y": y}


class SIFEFullySecRingLWE(IPFEAbsCrypto):
    """Crypto operations for the adaptively secure Ring-LWE IPFE of BMMS21 (Sec. 5)."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for encryptors, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self._cache = {}

    def _ring(self) -> tuple:
        q = gp.mpz(self.pp["q"])
        n = int(self.pp["ring_n"])
        return q, n, _get_ntt_root(q, n)

    def _a_ntt(self) -> list:
        if "a" not in self._cache:
            q, _, root = self._ring()
            self._cache["a"] = _ntt_all(self.pp["a"], q, root)
        return self._cache["a"]

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt an integer vector of length ``eta``.

        Returns:
            Dict with ``ct0`` (``m`` polynomials) and ``ct`` (``eta`` polynomials).

        Raises:
            FEKeyError: If public parameters or the encryption key are missing.
            FEValidationError: If the plaintext is malformed or out of bound.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        x = int_vector(lst_pt, "plaintext", length=self.pp["eta"], bound=int(self.pp["bound_x"]))
        q, n, root = self._ring()
        m = int(self.pp["m"])
        a_ntt = self._a_ntt()
        if "pk" not in self._cache:
            self._cache["pk"] = _ntt_all(self.sk["pk"], q, root)
        pk_ntt = self._cache["pk"]
        for _ in range(_MAX_TRIES):
            r_ntt = ntt_forward(discrete_gaussian_matrix(1, n, self.pp["sigma2"])[0], q, root)
            f0 = discrete_gaussian_matrix(m, n, self.pp["sigma2"])
            ct0_ntt = [
                [(u * v + w) % q for u, v, w in zip(a_j, r_ntt, f_j)]
                for a_j, f_j in zip(a_ntt, _ntt_all(f0, q, root))
            ]
            if any(_invertible(c, q) for c in ct0_ntt) and independent_at_every_slot(
                a_ntt, ct0_ntt, q
            ):
                break
        else:
            raise FESchemeError("could not sample a ciphertext with the required ct0")
        noise = discrete_gaussian_matrix(len(x), n, self.pp["sigma3"])
        ct = []
        for x_i, pk_i, f_i in zip(x, pk_ntt, noise):
            poly = ntt_inverse([(u * v) % q for u, v in zip(pk_i, r_ntt)], q, root)
            poly = [(u + v) % q for u, v in zip(poly, f_i)]
            poly[0] = (poly[0] + encode_message(x_i, q, self.pp["p"])) % q
            ct.append(poly)
        return {
            "ct0": [[gp.digits(v) for v in ntt_inverse(c, q, root)] for c in ct0_ntt],
            "ct": [[gp.digits(v) for v in poly] for poly in ct],
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Return ``<x, y>``, or ``None`` if the decoded value exceeds ``eta X Y``.

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If *fusion_weight* differs from the key's weights.
            FEValidationError: If the ciphertext does not match the parameters.
        """
        if not dk or "sk_y" not in dk:
            raise FEKeyError("no decryption key provided")
        y = int_vector(
            fusion_weight, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"])
        )
        if y != dk.get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")
        q, n, root = self._ring()
        m = int(self.pp["m"])
        ct0, ct, sk_y = dct_ct.get("ct0", []), dct_ct.get("ct", []), dk["sk_y"]
        if (
            len(ct0) != m or len(sk_y) != m or len(ct) != len(y)
            or any(len(poly) != n for poly in [*ct0, *ct, *sk_y])
        ):
            raise FEValidationError("ciphertext or key does not match the parameters")
        acc = [gp.mpz(0)] * n
        for c_j, s_j in zip(_ntt_all(ct0, q, root), _ntt_all(sk_y, q, root)):
            acc = [(u + v * w) % q for u, v, w in zip(acc, c_j, s_j)]
        inner = ntt_inverse(acc, q, root)
        d = [gp.mpz(0)] * n
        for y_i, poly in zip(y, ct):
            d = [u + y_i * gp.mpz(v) for u, v in zip(d, poly)]
        d = [(u - v) % q for u, v in zip(d, inner)]
        value = decode_vector(d, gp.mpz(self.pp["p"]), q, 1)[0]
        bound = len(y) * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        if abs(value) > bound:
            logger.error("inner product out of bound (wrong key or ciphertext?)")
            return None
        return value

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``): one ciphertext per element."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        out = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            out.append(ary_ct)
        return out

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`)."""
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, **kwargs)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise; needs ``dk`` and ``fusion_weight``."""
        dk, fusion_weight = kwargs.get("dk"), kwargs.get("fusion_weight")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        out = []
        for l, ary in enumerate(dict_ndarray_ct):
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ary_dec[i] = self.decrypt(ary[i], dk, fusion_weight)
            out.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return out
