"""
Function-Hiding Inner-Product Encryption over Pairings
| "Function-Hiding Inner Product Encryption is Practical"
| By Kim, Lewi, Mandal, Montgomery, Roy, Wu, SCN 2018 (ePrint 2016/440),
| Section 3; structure as in GoFE's ``innerprod/fullysec/fhipe.go``.

* type:     secret-key encryption with function-hiding keys
* setting:  asymmetric pairing group (default BLS12-381); secure in the generic
  bilinear group model

``B <- GL_n(Z_q)`` (rejection sampling), ``B* = det(B) (B^{-1})^T``. The key for
``y`` is ``(g1^{alpha det B}, g1^{alpha B y})``, a ciphertext of ``x`` is
``(g2^beta, g2^{beta B* x})``; decryption pairs them and solves a bounded
discrete log of ``e(g1, g2)^{alpha beta det B <x, y>}`` to base
``e(g1, g2)^{alpha beta det B}``. Keys live in G1 and ciphertexts in G2, as in
the paper (``B`` used transposed, which is identically distributed).

Deviation: every vector is padded as ``x' = (x, 1, 0)``, ``y' = (y, 0, 1)``
(``<x', y'> = <x, y>``) so that ``n = eta + 2 >= 3`` and no vector is zero: with
``n = 1`` (``B* = 1``) ciphertexts and keys revealed ``x`` and ``y`` directly,
and an all-zero vector was recognisable without a key -- the paper requires
non-zero vectors. The encryption key ``(g2, B*)`` determines ``B`` up to a
scalar and is master-secret equivalent.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, ZR, PairingGroup, multi_pair, pair
from pyfe4ai.primitives.pairing.pairing_backend import deserialize_many, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError
from pyfe4ai.core.versions import record_matches
from pyfe4ai._validation import int_vector
from pyfe4ai.primitives.pairing.pairing_utils import (
    bounded_discrete_log_gt,
    mat_vec_mod,
    matrix_inverse_mod,
    random_nonzero_zr,
    to_zr,
    transpose_mod,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases (private names used by internal callers)
_transpose_mod = transpose_mod
_mat_vec_mod = mat_vec_mod
_random_nonzero_zr = random_nonzero_zr
_to_zr = to_zr
_matrix_inverse_mod = matrix_inverse_mod
_bounded_discrete_log_gt = bounded_discrete_log_gt


class SIFEFHIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for function-hiding inner-product encryption over pairings."""

    _scheme_type = CryptoCONST.TYPE_SIFE_FH_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE FH-IPE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``eta``,
                ``bound_x``, ``bound_y`` and ``pairing_group_param``
                (asymmetric group, default ``"BLS12-381"``).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 16))
        self.bound_y = gp.mpz(config.get("bound_y", 16))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return record_matches(param, {
            "sec_param": self.sec_param, "eta": self.eta, "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y), "pairing_group_param": self.pairing_group_param,
        })

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        self.g1 = self.group.random(G1)
        self.g2 = self.group.random(G2)

        dim = self.eta + 2  # padded vectors (x, 1, 0) and (y, 0, 1)
        while True:
            b = [
                [gp.mpz(int(self.group.random(ZR))) for _ in range(dim)]
                for _ in range(dim)
            ]
            try:
                b_inv, det = _matrix_inverse_mod(b, self.order)
                break
            except ValueError:
                continue

        b_star = _transpose_mod(b_inv)
        b_star = [
            [(det * entry) % self.order for entry in row]
            for row in b_star
        ]

        self.mpk = {}
        self.msk = {
            "g1": self.group.serialize(self.g1).decode("utf-8"),
            "g2": self.group.serialize(self.g2).decode("utf-8"),
            "b": [[gp.digits(v) for v in row] for row in b],
            "b_star": [[gp.digits(v) for v in row] for row in b_star],
            "det": gp.digits(det),
        }
        logger.info("SIFE FH-IPE setup successfully")

    def get_public_parameters(self) -> dict:
        return {
            "eta": self.eta,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "pairing_group_param": self.pairing_group_param,
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Return the encryption key (``g2``, ``b_star``); it is equivalent to
        the master secret key.

        Args:
            nid: Client identifier (unused for single-input schemes).
        """
        return {
            "g2": self.msk["g2"],
            "b_star": self.msk["b_star"],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of
                int).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for FH-IPE decryption key generation")
        y = int_vector(
            credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=self.bound_y
        )
        y = [gp.mpz(v) for v in y] + [gp.mpz(0), gp.mpz(1)]
        b = [[gp.mpz(v) for v in row] for row in self.msk["b"]]
        det = gp.mpz(self.msk["det"])
        alpha = _random_nonzero_zr(self.group)
        alpha_int = gp.mpz(int(alpha))

        by = _mat_vec_mod(b, y, self.order)
        by = [(alpha_int * entry) % self.order for entry in by]

        g1 = self.group.deserialize(self.msk["g1"].encode("utf-8"))
        k1, *k2 = pow_many(g1, [(alpha_int * det) % self.order] + by, self.group)
        return {
            "k1": self.group.serialize(k1).decode("utf-8"),
            "k2": [self.group.serialize(v).decode("utf-8") for v in k2],
        }


class SIFEFHIPE(IPFEAbsCrypto):
    """Crypto operations for function-hiding inner-product encryption over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE FH-IPE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.group = PairingGroup(self.pp["pairing_group_param"])
        check_asymmetric(self.group)
        self.order = gp.mpz(int(self.group.order()))
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        x = int_vector(lst_pt, "plaintext", length=self.pp["eta"], bound=int(self.pp["bound_x"]))
        x = [gp.mpz(v) for v in x] + [gp.mpz(1), gp.mpz(0)]
        b_star = [[gp.mpz(v) for v in row] for row in self.sk["b_star"]]
        g2 = self.group.deserialize(self.sk["g2"].encode("utf-8"))
        beta = _random_nonzero_zr(self.group)
        beta_int = gp.mpz(int(beta))

        bstar_x = _mat_vec_mod(b_star, x, self.order)
        bstar_x = [(beta_int * entry) % self.order for entry in bstar_x]
        c1, *c2 = pow_many(g2, [beta_int] + bstar_x, self.group)
        return {
            "c1": self.group.serialize(c1).decode("utf-8"),
            "c2": [self.group.serialize(v).decode("utf-8") for v in c2],
        }

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list | None = None):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict.
                dk: Functional decryption key.
                fusion_weight: Not needed: the key encodes ``y`` and hides it
                    (function hiding). Checked against the bounds if given.
        """
        if fusion_weight is not None:
            int_vector(fusion_weight, "fusion weight", length=self.pp["eta"], bound=int(self.pp["bound_y"]))

        c1, *c2 = deserialize_many(self.group, [dct_ct["c1"]] + list(dct_ct["c2"]))
        k1, *k2 = deserialize_many(self.group, [dk["k1"]] + list(dk["k2"]))

        if len(c2) != self.pp["eta"] + 2 or len(k2) != self.pp["eta"] + 2:
            raise FEValidationError("key or ciphertext length mismatch")

        d1 = pair(k1, c1)
        d2 = multi_pair(k2, c2)

        bound = int(self.pp["eta"] * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"]))
        # d1 = e(k1, c1) is a new base for every decryption: no cached table
        return _bounded_discrete_log_gt(self.group, d1, d2, bound, cache=False)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE")
