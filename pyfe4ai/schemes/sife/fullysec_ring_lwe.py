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
  modulus; :func:`pyfe4ai.primitives.lattice.ring_lwe_utils.bmms21_adaptive_parameters`
  derives the widths and ``q``.
* Signed inputs ``|x| <= X``, ``|y| <= Y``: ``p = 2 l X Y + 2`` with centred
  decoding, ``||x1 - x0||_inf <= 2X``, and complexity leveraging over
  ``(2X + 1)^(2l)`` challenge pairs.
* Gaussian samples (keys and encryption noise) are exact ``D_{Z,sigma}``
  by default (``gaussian_sampler = "facct"``,
  :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct`), whatever the
  library-wide default; ``"rounded"`` (a rounded continuous Gaussian, the
  library default) stays selectable. Rounded keys are ``0.02 / sigma1^2``
  per sample, over ``m n eta`` samples, from ``D_{Z,sigma1}`` in statistical
  distance for an ideal rounded Gaussian -- in the standard configurations
  from about ``2^-47`` (elementwise ``eta = 1``, ``X = 100``) over
  ``2^-44.5`` (packed ``eta = 1``) to ``2^-40.5`` (``eta = 4``, ``X = Y =
  4``), negligible above ``sigma1 = 2^40`` -- and about ``2^-12`` at ``eta =
  1`` with the double-precision path's artifacts (crypto review, October
  2026), terms the proof does not account for. Python FACCT is slower than the rounded
  sampler at the key width ``sigma1`` but faster beyond ``2^40`` (the
  encryption widths), where the rounded sampler needs MPFR Box-Muller.

Ciphertext layouts (``ndarray_layout``):

* ``"elementwise"`` (default): one vector per ciphertext; the ndarray helper
  encrypts every element on its own.
* ``"packed"`` (opt-in, **experimental**, its own parameter record; the
  leveraging argument below is this project's own, checked independently
  but not published or externally reviewed): a ciphertext holds an
  ``eta x k`` matrix, ``k <= ring_n`` (Remark 1: ``Delta x_ij`` is added to
  coefficient ``j`` of ``ct_i``), and decrypts to the ``k`` inner products
  ``<x_j, y>`` of its columns. The ndarray helper packs ``ring_n``
  elements per ciphertext. The statistical games G3* - G5* of the proof
  depend on the challenge only through the span ``V`` of the differences
  ``x1_j - x0_j``, so the leveraging ranges over the possible ``V``,
  ``(D + 1) (4X + 1)^(l D)`` with ``D = min(l, ring_n)``, whatever ``k`` is
  (guessing both challenge matrices would give ``(2X + 1)^(2lk)``); G3 -> G4
  decomposes along a basis ``B`` of ``V``, ``||B|| <= 2X sqrt(l D)``, which
  ``sigma1`` absorbs, and G4 -> G5 applies Theorem 2 once per basis vector
  (:func:`pyfe4ai.primitives.lattice.ring_lwe_utils.bmms21_adaptive_leveraging`). G5*
  writes the differences in that basis mod ``q``, so ``q`` must exceed the
  minors of the basis, ``(2X sqrt(D))^D``; the parameters enforce it (it
  moves e.g. ``eta = 24``, ``X = 4`` to a larger ring). For ``eta = 1``,
  ``V`` is ``{0}`` or ``Q``: two guesses, and the basis ``[1]`` of norm 1
  instead of ``2X`` (``X = 100``, ``Y = 1``: ``m = 12`` and a 107-bit ``q``
  against ``m = 19`` and 111 bits elementwise, whose parameters stay as
  they are). Encryptors and decryptors
  accept ``pp`` flagged ``"packed"`` only if its ``q``, ``p`` and widths are
  the packed derivation.

Under the ``standard`` profile ``ring_n = "auto"`` takes the smallest ring
dimension (with the smallest ``m``) whose ``q`` meets
:data:`~pyfe4ai.primitives.lattice.ring_lwe_utils.RING_LWE_128_LOG_Q` at the width
``sigma`` (about 8) of the RLWE problem of Theorem 1.
"""

from __future__ import annotations

import json
import logging
import math
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.profiles import STANDARD
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.lattice.ring_lwe_utils import _get_ntt_root
from pyfe4ai.primitives.lattice.ring_lwe_utils import bmms21_adaptive_choice
from pyfe4ai.primitives.lattice.ring_lwe_utils import bmms21_adaptive_parameters
from pyfe4ai.primitives.lattice.ring_lwe_utils import check_key_rows
from pyfe4ai.primitives.lattice.ring_lwe_utils import check_ring_lwe
from pyfe4ai.primitives.lattice.ring_lwe_utils import decode_vector
from pyfe4ai.primitives.lattice.ring_lwe_utils import ntt_forward
from pyfe4ai.primitives.lattice.ring_lwe_utils import ntt_inverse
from pyfe4ai.primitives.sampling import scheme_sampler
from pyfe4ai.primitives.sampling import discrete_gaussian_matrix
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.core.formats.ring_lwe import FullySecRingLWECiphertext
from pyfe4ai.core.formats.ring_lwe import FullySecRingLWEEncryptionKey
from pyfe4ai.core.formats.ring_lwe import FullySecRingLWEFunctionalKey
from pyfe4ai.core.formats.ring_lwe import FullySecRingLWELayer
from pyfe4ai.core.formats.ring_lwe import FullySecRingLWEParameters
from pyfe4ai.core.formats.ring_lwe import FullySecRingLWERecord
from pyfe4ai.core.versions import record_matches
from pyfe4ai.core.versions import SIFE_FULLYSEC_RING_LWE_PACKED_RECORD
from pyfe4ai.core.versions import SIFE_FULLYSEC_RING_LWE_RECORD
from pyfe4ai._validation import int_vector
from pyfe4ai._validation import scale_to_int
from pyfe4ai.primitives.lattice.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

#: Values of the ``ndarray_layout`` option; the first is the default.
LAYOUTS = ("elementwise", "packed")
#: Default Gaussian sampler of this scheme (``gaussian_sampler`` option): the
#: exact ``D_{Z,sigma}`` of the proof, whatever the library-wide default is.
DEFAULT_SAMPLER = "facct"


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
                (power of two, ``"auto"``, or ``None`` for the toy default),
                ``m`` (number of key polynomials, at least 5, or
                ``"auto"``, the default), ``ndarray_layout``
                (``"elementwise"``, the default, or ``"packed"``: up to
                ``ring_n`` vectors per ciphertext, with parameters of their own)
                and ``gaussian_sampler`` (``"facct"``, the default, or
                ``"rounded"``) for the keys, also published in ``pp`` as the
                encryptors' default.

        Raises:
            FEValidationError: For an unknown ``ndarray_layout`` or sampler,
                or a ``ring_modulus`` other than ``"prime"`` (the adaptive
                proof's Ring-LHL, Theorem 2, is for a prime ``q``; see
                :doc:`/rns_modulus_review`).
        """
        if config.get("ring_modulus", "prime") != "prime":
            raise FEValidationError(
                "sife/fullysec_ring_lwe supports ring_modulus 'prime' only: its adaptive proof "
                "(Ring-LHL) is for a prime q"
            )
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self.ring_n_config = config.get("ring_n")
        self.m_config = config.get("m", "auto")
        self.ndarray_layout = config.get("ndarray_layout", LAYOUTS[0])
        if self.ndarray_layout not in LAYOUTS:
            raise FEValidationError(
                "ndarray_layout must be one of {}, got {!r}".format(LAYOUTS, self.ndarray_layout)
            )
        self.gaussian_sampler = scheme_sampler(config, default=DEFAULT_SAMPLER, profile=self.security_profile,
                                               use_context=False)
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_ring_lwe(self.ring_n, self.q, self.sigma)

    def _derive(self) -> dict:
        return bmms21_adaptive_choice(
            self.sec_param, self.eta, self.bound_x, self.bound_y, self.ring_n_config,
            self.m_config, standard=self.security_profile == STANDARD, packed=self._packed(),
        )

    def _packed(self) -> bool:
        return self.ndarray_layout == "packed"

    def _param_version(self) -> str:
        return SIFE_FULLYSEC_RING_LWE_PACKED_RECORD if self._packed() else SIFE_FULLYSEC_RING_LWE_RECORD

    def _param_verification(self, param: dict) -> bool:
        """The cached record must match the config and the parameters derived from
        it (only the seed of ``a`` is random; ``a`` must contain a unit)."""
        if not record_matches(param, {
            "param_version": self._param_version(), "ndarray_layout": self.ndarray_layout,
            "sec_param": self.sec_param, "eta": self.eta, "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y), "ring_n_config": self.ring_n_config, "m_config": self.m_config,
            "profile": self.security_profile,
        }):
            return False
        expected = self._derive()
        if not record_matches(param, {
            "q": gp.digits(expected["q"]), "p": gp.digits(expected["p"]), "ring_n": expected["ring_n"],
            "m": expected["m"], **{k: expected[k] for k in ("sigma", "sigma_lhl", "sigma1", "sigma2", "sigma3")},
        }):
            return False
        q, n = expected["q"], expected["ring_n"]
        a = _seeded_a(FullySecRingLWERecord(param).a_seed, q, n, expected["m"])
        return any(_invertible(poly, q) for poly in _ntt_all(a, q, _get_ntt_root(q, n)))

    def _apply_parameters(self, param: dict) -> None:
        record = FullySecRingLWERecord.from_dict(param)
        self.ring_n = int(record.ring_n)
        self.m = int(record.m)
        self.p = gp.mpz(record.p)
        self.q = gp.mpz(record.q)
        for key in ("sigma", "sigma_lhl", "sigma1", "sigma2", "sigma3"):
            setattr(self, key, float(getattr(record, key)))
        self.a_seed = record.a_seed
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
        record = FullySecRingLWERecord.new(
            param_version=self._param_version(),
            sec_param=self.sec_param,
            eta=self.eta,
            bound_x=gp.digits(self.bound_x),
            bound_y=gp.digits(self.bound_y),
            ring_n_config=self.ring_n_config,
            m_config=self.m_config,
            profile=self.security_profile,
            ring_n=n,
            m=params["m"],
            p=gp.digits(params["p"]),
            q=gp.digits(q),
            a_seed=seed,
        )
        record.update(**{k: params[k] for k in ("sigma", "sigma_lhl", "sigma1", "sigma2", "sigma3")})
        if self._packed():  # elementwise records keep their v2 form
            record.ndarray_layout = self.ndarray_layout
        self._apply_parameters(record.to_dict())
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f)

    def setup(self) -> None:
        """Sample ``s_i`` (``m`` polynomials per slot) and ``pk_i = <a, s_i>``."""
        q, n = self.q, self.ring_n
        root = _get_ntt_root(q, n)
        a_ntt = _ntt_all(self.a, q, root)
        sk, pk = [], []
        for _ in range(self.eta):
            s_i = discrete_gaussian_matrix(self.m, n, self.sigma1, sampler=self.gaussian_sampler)
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
        encryption widths ``sigma2``, ``sigma3``, the ``gaussian_sampler`` (and
        ``ndarray_layout`` when packed)."""
        pp = FullySecRingLWEParameters.new(
            a=[[gp.digits(v) for v in poly] for poly in self.a],
            p=gp.digits(self.p),
            q=gp.digits(self.q),
            ring_n=self.ring_n,
            m=self.m,
            eta=self.eta,
            bound_x=gp.digits(self.bound_x),
            bound_y=gp.digits(self.bound_y),
            sigma2=self.sigma2,
            sigma3=self.sigma3,
            sec_param=self.sec_param,
            gaussian_sampler=self.gaussian_sampler,
            security_profile=self.security_profile,
        )
        if self._packed():
            pp.ndarray_layout = self.ndarray_layout
        return pp.to_dict()

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return the encryption key: the public keys ``pk_i`` (``eta`` polynomials)."""
        return FullySecRingLWEEncryptionKey.new(pk=[[gp.digits(v) for v in poly] for poly in self.mpk["pk"]]).to_dict()

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
        return FullySecRingLWEFunctionalKey.new(sk_y=sk_y, y=y).to_dict()


class SIFEFullySecRingLWE(IPFEAbsCrypto):
    """Crypto operations for the adaptively secure Ring-LWE IPFE of BMMS21 (Sec. 5)."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for encryptors, ``keys.sk``; the
        optional ``gaussian_sampler`` (``"facct"`` or ``"rounded"``) of the
        encryption noise defaults to that of ``pp``, else ``"facct"``.

        Raises:
            FEValidationError: For an unknown sampler.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.gaussian_sampler = scheme_sampler(config, self.pp, default=DEFAULT_SAMPLER, use_context=False)
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

    def _packed(self) -> bool:
        """Whether *pp* holds packed parameters. The flag alone does not suffice:
        ``q``, ``p``, ``sigma2`` and ``sigma3`` must be the packed derivation for
        the bounds, ``ring_n`` and ``m`` of *pp*, and ``a`` must have ``m``
        polynomials, since elementwise widths cover one vector per ciphertext
        only. The check is redone whenever one of these values changes.

        Raises:
            FEValidationError: If *pp* is flagged packed but its parameters are not.
        """
        pp = self.pp
        if pp.get("ndarray_layout", LAYOUTS[0]) != "packed":
            return False
        try:
            key = tuple(str(pp[k]) for k in ("sec_param", "eta", "bound_x", "bound_y", "ring_n", "m",
                                              "q", "p", "sigma2", "sigma3")) + (len(pp["a"]),)
        except (KeyError, TypeError) as exc:
            raise FEValidationError("public parameters flagged packed lack {}".format(exc)) from exc
        if self._cache.get("packed_pp") != key:
            try:
                expected = bmms21_adaptive_parameters(
                    int(pp["sec_param"]), int(pp["eta"]), int(pp["bound_x"]), int(pp["bound_y"]),
                    int(pp["ring_n"]), int(pp["m"]), packed=True,
                )
                same = (
                    int(pp["q"]) == int(expected["q"]) and int(pp["p"]) == int(expected["p"])
                    and float(pp["sigma2"]) == expected["sigma2"] and float(pp["sigma3"]) == expected["sigma3"]
                    and len(pp["a"]) == int(pp["m"])
                )
            except (KeyError, TypeError, ValueError, FEValidationError):
                same = False
            if not same:
                raise FEValidationError(
                    "ndarray_layout = 'packed', but q, p, the widths or a of these public "
                    "parameters are not the packed ones for their bounds, ring_n and m"
                )
            self._cache["packed_pp"] = key
        return True

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt an integer vector of length ``eta`` or, with packed parameters,
        an ``eta x k`` matrix (``eta`` rows of ``k <= ring_n`` integers; column ``j``
        goes to coefficient ``j``).

        Returns:
            Dict with ``ct0`` (``m`` polynomials) and ``ct`` (``eta`` polynomials);
            for a matrix also ``k``.

        Raises:
            FEKeyError: If public parameters or the encryption key are missing.
            FEValidationError: If the plaintext is malformed or out of bound, or
                is a matrix without packed parameters (their ``sigma1`` and
                leveraging cover only one vector per ciphertext).
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        eta, bound = self.pp["eta"], int(self.pp["bound_x"])
        if isinstance(lst_pt, list) and lst_pt and isinstance(lst_pt[0], (list, tuple)):
            if not self._packed():
                raise FEValidationError(
                    "a plaintext matrix needs parameters generated with ndarray_layout='packed'"
                )
            if len(lst_pt) != eta:
                raise FEValidationError("plaintext matrix must have eta = {} rows".format(eta))
            k = len(lst_pt[0])
            if not 0 < k <= int(self.pp["ring_n"]):
                raise FEValidationError("plaintext column count must be 1..ring_n")
            ct = self._encrypt_rows([int_vector(row, "plaintext row", length=k, bound=bound) for row in lst_pt])
            FullySecRingLWECiphertext(ct).k = k
            return ct
        x = int_vector(lst_pt, "plaintext", length=eta, bound=bound)
        return self._encrypt_rows([[x_i] for x_i in x])

    def _encrypt_rows(self, rows: list) -> dict:
        """``ct0`` and ``ct`` for the plaintext *rows* (``eta`` lists of ``k`` ints).

        Raises:
            FEKeyError: Unless ``pk`` has ``eta`` rows of ``ring_n`` values (with
                fewer, the ciphertext lacked the missing rows), checked before
                anything is drawn.
        """
        q, n, root = self._ring()
        self._check_private_key()
        check_key_rows({"pk": FullySecRingLWEEncryptionKey(self.sk).get("pk")}, int(self.pp["eta"]), n, "pk")
        sampler = self.gaussian_sampler
        m = int(self.pp["m"])
        a_ntt = self._a_ntt()
        if "pk" not in self._cache:
            self._cache["pk"] = _ntt_all(FullySecRingLWEEncryptionKey(self.sk).pk, q, root)
        pk_ntt = self._cache["pk"]
        for _ in range(_MAX_TRIES):
            r_ntt = ntt_forward(discrete_gaussian_matrix(1, n, self.pp["sigma2"], sampler=sampler)[0], q, root)
            f0 = discrete_gaussian_matrix(m, n, self.pp["sigma2"], sampler=sampler)
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
        noise = discrete_gaussian_matrix(len(rows), n, self.pp["sigma3"], sampler=sampler)
        ct = []
        for row, pk_i, f_i in zip(rows, pk_ntt, noise):
            poly = ntt_inverse([(u * v) % q for u, v in zip(pk_i, r_ntt)], q, root)
            poly = [(u + v) % q for u, v in zip(poly, f_i)]
            for j, x_ij in enumerate(row):
                poly[j] = (poly[j] + encode_message(x_ij, q, self.pp["p"])) % q
            ct.append(poly)
        return FullySecRingLWECiphertext.new(
            ct0=[[gp.digits(v) for v in ntt_inverse(c, q, root)] for c in ct0_ntt],
            ct=[[gp.digits(v) for v in poly] for poly in ct],
        ).to_dict()

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list):
        """Return ``<x, y>``, or ``None`` if the decoded value exceeds ``eta X Y``;
        for a packed ciphertext (one with ``k``) the list of the ``k`` inner
        products, or ``None`` if one of them exceeds the bound.

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If *fusion_weight* differs from the key's weights.
            FEValidationError: If the ciphertext does not match the parameters.
        """
        y = self._checked_key(dk, fusion_weight)
        q, _, root = self._ring()
        values = self._decrypt_values(dct_ct, _ntt_all(FullySecRingLWEFunctionalKey(dk).sk_y, q, root), y)
        if values is None or "k" in FullySecRingLWECiphertext(dct_ct):
            return values
        return values[0]

    def _checked_key(self, dk: dict, fusion_weight: list) -> list:
        """The weights *fusion_weight*, checked against *dk* and the parameters."""
        if not dk or "sk_y" not in FullySecRingLWEFunctionalKey(dk):
            raise FEKeyError("no decryption key provided")
        y = int_vector(
            fusion_weight, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"])
        )
        if y != FullySecRingLWEFunctionalKey(dk).get("y"):
            raise FESchemeError("fusion weights differ from those of the decryption key")
        sk_y = FullySecRingLWEFunctionalKey(dk).sk_y
        if len(sk_y) != int(self.pp["m"]) or any(len(poly) != int(self.pp["ring_n"]) for poly in sk_y):
            raise FEValidationError("ciphertext or key does not match the parameters")
        return y

    def _decrypt_values(self, dct_ct: dict, sk_ntt: list, y: list) -> list | None:
        """The ``k`` decoded coefficients of a ciphertext (``k = 1`` without a batch
        length), or ``None`` if one exceeds ``eta X Y``; *sk_ntt* is ``NTT(sk_y)``."""
        q, n, root = self._ring()
        m = int(self.pp["m"])
        dct_ct = FullySecRingLWECiphertext(dct_ct)
        ct0, ct = dct_ct.get("ct0", []), dct_ct.get("ct", [])
        k = dct_ct.get("k", 1)
        if "k" in dct_ct and (
            not self._packed() or not isinstance(k, int) or isinstance(k, bool) or not 0 < k <= n
        ):
            raise FEValidationError("packed ciphertext does not match the parameters")
        if len(ct0) != m or len(ct) != len(y) or any(len(poly) != n for poly in [*ct0, *ct]):
            raise FEValidationError("ciphertext or key does not match the parameters")
        acc = [gp.mpz(0)] * n
        for c_j, s_j in zip(_ntt_all(ct0, q, root), sk_ntt):
            acc = [(u + v * w) % q for u, v, w in zip(acc, c_j, s_j)]
        inner = ntt_inverse(acc, q, root)
        d = [gp.mpz(0)] * n
        for y_i, poly in zip(y, ct):
            d = [u + y_i * gp.mpz(v) for u, v in zip(d, poly)]
        d = [(u - v) % q for u, v in zip(d, inner)]
        values = decode_vector(d, gp.mpz(self.pp["p"]), q, k)
        bound = len(y) * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        if any(abs(v) > bound for v in values):
            logger.error("inner product out of bound (wrong key or ciphertext?)")
            warn_at_caller("inner product out of bound (wrong key or ciphertext?)")
            return None
        return values

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays (``eta = 1``): one ciphertext per element, or with packed
        parameters ``ring_n`` consecutive elements (row-major) per ciphertext.

        A packed layer is ``{"shape", "first_chunk", "chunks"}``, its chunks
        packed ciphertexts numbered consecutively across the layers (the
        layout of the MCFE Ring-LWE schemes, which
        :class:`~pyfe4ai.ml.EncryptedGradient` serialises).
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if self._packed():
            return self._encrypt_packed_layers(lst_ndarray)
        out = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            out.append(ary_ct)
        return out

    def _encrypt_packed_layers(self, lst_ndarray: list) -> list:
        n = int(self.pp["ring_n"])
        out, first = [], 0
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            flat = [int(v) for v in ary.reshape(-1)]
            if not flat:
                raise FEValidationError("cannot pack an empty array")
            chunks = [self.encrypt([flat[j:j + n]]) for j in range(0, len(flat), n)]
            out.append(FullySecRingLWELayer.new(shape=list(ary.shape), first_chunk=first, chunks=chunks).to_dict())
            first += len(chunks)
        return out

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`)."""
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, **kwargs)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts, element-wise or packed layers (transforming
        ``sk_y`` once); needs ``dk`` and ``fusion_weight``."""
        dk, fusion_weight = kwargs.get("dk"), kwargs.get("fusion_weight")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        out, key, first = [], None, 0
        for l, ary in enumerate(dict_ndarray_ct):
            if isinstance(ary, dict):  # packed layer
                if key is None:
                    y = self._checked_key(dk, fusion_weight)
                    q, _, root = self._ring()
                    key = (y, _ntt_all(FullySecRingLWEFunctionalKey(dk).sk_y, q, root))
                shape, chunks = self._checked_packed_layer(ary, l, first)
                out.append(self._decrypt_packed_layer(shape, chunks, l, *key))
                first += len(chunks)
                continue
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                ary_dec[i] = self.decrypt(ary[i], dk, fusion_weight)
            out.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return out

    def _checked_packed_layer(self, layer, l: int, first: int) -> tuple:
        """``(shape, chunks)`` of packed layer *l*, laid out as
        :meth:`encrypt_lst_ndarray` writes it: a shape of non-negative ints,
        ``first_chunk`` equal to *first* (the chunks of the packed layers
        before it), and chunks of ``ring_n`` elements but the last, covering
        the shape. The order of full chunks is not authenticated (nor is the
        position of an elementwise ciphertext): a ciphertext is not bound to
        its place.

        Raises:
            FEValidationError: For any other layout.
        """
        def whole(v) -> bool:
            return isinstance(v, (int, np.integer)) and not isinstance(v, bool)

        n = int(self.pp["ring_n"])
        if not isinstance(layer, dict) or set(layer) != set(FullySecRingLWELayer.FIELDS):
            raise FEValidationError("packed layer {} must be a dict of shape, first_chunk and chunks".format(l))
        layer = FullySecRingLWELayer(layer)
        shape, chunks = layer.shape, layer.chunks
        if not isinstance(shape, (list, tuple)) or len(shape) > 32 or not all(whole(v) and v >= 0 for v in shape):
            raise FEValidationError("packed layer {} has an invalid shape {!r}".format(l, shape))
        shape = tuple(int(v) for v in shape)
        if not whole(layer.first_chunk) or layer.first_chunk != first:
            raise FEValidationError(
                "packed layer {} starts at chunk {!r}, expected {}".format(l, layer.first_chunk, first)
            )
        if not isinstance(chunks, list) or not chunks or not all(
            isinstance(chunk, dict) and whole(FullySecRingLWECiphertext(chunk).get("k")) for chunk in chunks
        ):
            raise FEValidationError("packed layer {} needs a non-empty list of packed ciphertexts".format(l))
        ks = [int(FullySecRingLWECiphertext(chunk).k) for chunk in chunks]
        if any(k != n for k in ks[:-1]) or not 0 < ks[-1] <= n or sum(ks) != math.prod(shape):
            raise FEValidationError(
                "packed layer {} has chunks of {} elements; its shape {} needs {} in chunks of "
                "ring_n = {} but the last".format(l, ks, shape, math.prod(shape), n)
            )
        return shape, chunks

    def _decrypt_packed_layer(self, shape: tuple, chunks: list, l: int, y: list, sk_ntt: list) -> np.ndarray:
        values = []
        for chunk in chunks:
            vals = self._decrypt_values(chunk, sk_ntt, y)
            values.extend([None] * FullySecRingLWECiphertext(chunk).k if vals is None else vals)
        ary = np.empty(len(values), dtype=object)
        ary[:] = values
        return decoded_layer(ary.reshape(shape), l, 10 ** self.precision)
