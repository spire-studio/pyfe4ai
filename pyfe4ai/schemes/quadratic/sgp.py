"""
Public-key functional encryption for quadratic polynomials (SGP)
| "Reading in the Dark: Classifying Encrypted Digits with Functional Encryption"
| By Dufour-Sans, Gay, Pointcheval (ePrint 2018/206) -- Fig. 2.

* type:     public-key encryption; keys for matrices ``F``, decryption gives
  ``x^T F y``
* setting:  asymmetric pairing group (default BLS12-381); IND-CPA in the generic
  bilinear group model

``s, t <- Z_p^n``; ``pk = (g1^s, g2^t)``, ``msk = (s, t)``.

* Enc(pk, x, y): ``gamma <- Z_p``, ``W <- GL_2``;
  ``a_i = (W^{-1})^T (x_i, gamma s_i)``, ``b_i = W (y_i, -t_i)``;
  ``ct = (g1^gamma, {g1^{a_i}, g2^{b_i}})`` -- computed from ``pk``.
* KeyGen(msk, F): ``dk = (g2^{f(s, t)}, F)`` with ``f(s, t) = sum f_ij s_i t_j``.
* Dec: ``e(g1^gamma, g2^{f(s,t)}) prod e(g1^{a_i}, g2^{b_j})^{f_ij} = [x^T F y]_T``
  and a discrete log in ``[-n^2 B^3, n^2 B^3]``.

Signed inputs with one bound ``B`` for ``x``, ``y`` and ``F``. Versions <= 0.2.0
encrypted with ``(s, t)`` themselves (GoFE's secret-key form) and handed
them to every encryptor, which made each encryptor a key authority.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.pairing_utils import to_zr
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, ZR, PairingGroup, multi_pair, pair
from pyfe4ai.utils.pairing_backend import deserialize_many, multiexp, pow_many
from pyfe4ai.utils.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)


def _rand_scalar(group: PairingGroup) -> gp.mpz:
    return gp.mpz(int(group.random(ZR)))


def _rand_invertible_2x2(
    group: PairingGroup, modulus: gp.mpz
) -> tuple[list[list[gp.mpz]], list[list[gp.mpz]]]:
    """Perform the _rand_invertible_2x2 operation.

        Args:
            group: Pairing group instance.
            modulus: Modulus for the arithmetic operation.
    """
    while True:
        w = [[_rand_scalar(group) % modulus for _ in range(2)] for _ in range(2)]
        det = (w[0][0] * w[1][1] - w[0][1] * w[1][0]) % modulus
        if det == 0:
            continue
        try:
            det_inv = gp.invert(det, modulus)
        except ZeroDivisionError:
            continue
        inv = [
            [(w[1][1] * det_inv) % modulus, ((-w[0][1]) * det_inv) % modulus],
            [((-w[1][0]) * det_inv) % modulus, (w[0][0] * det_inv) % modulus],
        ]
        return w, inv


def _transpose(matrix: list[list[gp.mpz]]) -> list[list[gp.mpz]]:
    return [list(col) for col in zip(*matrix)]


def _matvec_mod(
    matrix: list[list[gp.mpz]], vector: list[gp.mpz], modulus: gp.mpz
) -> list[gp.mpz]:
    """Perform the _matvec_mod operation.

        Args:
            matrix: Input matrix.
            vector: Input vector.
            modulus: Modulus for the arithmetic operation.
    """
    ret = []
    for row in matrix:
        acc = gp.mpz(0)
        for lhs, rhs in zip(row, vector):
            acc += lhs * rhs
        ret.append(acc % modulus)
    return ret


def _serialize_group_vec(group: PairingGroup, elements: list) -> list[str]:
    """Perform the _serialize_group_vec operation.

        Args:
            group: Pairing group instance.
            elements: Vector of group elements.
    """
    return [group.serialize(elem).decode("utf-8") for elem in elements]


def _deserialize_group_vec(group: PairingGroup, elements: list[str]) -> list:
    """Perform the _deserialize_group_vec operation.

        Args:
            group: Pairing group instance.
            elements: Vector of group elements.
    """
    return deserialize_many(group, elements)


def _quadratic_form(
    matrix: list[list[gp.mpz]], lhs: list[gp.mpz], rhs: list[gp.mpz]
) -> gp.mpz:
    """Perform the _quadratic_form operation.

        Args:
            matrix: Input matrix.
            lhs: Left-hand side vector.
            rhs: Right-hand side vector.
    """
    acc = gp.mpz(0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            acc += value * lhs[i] * rhs[j]
    return acc


class QuadraticSGPKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Setup and key generation of DSGPP18's quadratic FE (Fig. 2)."""

    _scheme_type = CryptoCONST.TYPE_QUADRATIC_SGP
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the DSGPP18 quadratic FE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (vector
                length, or ``eta``; default 2), ``bound`` (one bound for ``x``,
                ``y`` and ``F``, default 16) and ``pairing_group_param``
                (asymmetric group, default ``"BLS12-381"``).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.n = config.get("n", config.get("eta", 2))
        self.bound = gp.mpz(config.get("bound", 16))
        if not isinstance(self.n, int) or self.n < 1 or self.bound < 1:
            raise FEValidationError("need integers n >= 1 and bound >= 1")
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("n") == self.n
            and param.get("bound") == gp.digits(self.bound)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "n": self.n,
                    "bound": gp.digits(self.bound),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        self.g1 = self.group.random(G1)
        self.g2 = self.group.random(G2)
        self.s = [_rand_scalar(self.group) % self.order for _ in range(self.n)]
        self.t = [_rand_scalar(self.group) % self.order for _ in range(self.n)]
        ser = lambda e: self.group.serialize(e).decode("utf-8")  # noqa: E731
        self.mpk = {
            "g1_s": [ser(v) for v in pow_many(self.g1, self.s, self.group)],
            "g2_t": [ser(v) for v in pow_many(self.g2, self.t, self.group)],
        }
        self.msk = {
            "g1": ser(self.g1),
            "g2": ser(self.g2),
            "s": [gp.digits(v) for v in self.s],
            "t": [gp.digits(v) for v in self.t],
        }
        logger.info("Quadratic SGP setup successfully")

    def get_public_parameters(self, **kwargs) -> dict | None:
        return {
            "n": self.n,
            "bound": gp.digits(self.bound),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.msk.get("g1"),
            "g2": self.msk.get("g2"),
            "g1_s": self.mpk["g1_s"],
            "g2_t": self.mpk["g2_t"],
        }

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Public-key scheme: encryption needs ``pk`` only (``None``).

        Versions <= 0.2.0 returned the master secret key ``(s, t)``.
        """
        return None

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for a matrix ``F``.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``function_matrix``
                (``n x n`` integer matrix ``F``).
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for Quadratic SGP decryption key generation")
        function_matrix = credentials.get("function_matrix")
        if not isinstance(function_matrix, list) or len(function_matrix) != self.n:
            raise FEValidationError("invalid function_matrix")
        f_matrix = [
            [gp.mpz(v) for v in int_vector(row, "function_matrix row", length=self.n, bound=int(self.bound))]
            for row in function_matrix
        ]
        s_vec = [gp.mpz(value) for value in self.msk["s"]]
        t_vec = [gp.mpz(value) for value in self.msk["t"]]
        exponent = _quadratic_form(f_matrix, s_vec, t_vec) % self.order
        g2 = self.group.deserialize(self.msk["g2"].encode("utf-8"))
        fe_key = g2 ** to_zr(self.group, exponent)

        return {
            "key": self.group.serialize(fe_key).decode("utf-8"),
            "f": [[gp.digits(value) for value in row] for row in f_matrix],
        }


class QuadraticSGP(IPFEAbsCrypto):
    """Encryption (from ``pk``) and decryption of DSGPP18's quadratic FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the DSGPP18 quadratic FE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters,
                which hold the public key).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.group = PairingGroup(self.pp["pairing_group_param"])
        check_asymmetric(self.group)
        self.order = gp.mpz(int(self.group.order()))
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, payload: dict) -> dict:
        """``Enc(pk, (x, y))`` for ``n`` integers each, bounded by ``bound``."""
        if not isinstance(payload, dict):
            raise FEValidationError("Quadratic SGP expects a dict payload with x and y")
        n, bound = self.pp["n"], int(self.pp["bound"])
        x = int_vector(payload.get("x"), "x", length=n, bound=bound)
        y = int_vector(payload.get("y"), "y", length=n, bound=bound)
        group, order = self.group, self.order
        g1, g2, *pk = _deserialize_group_vec(
            group, [self.pp["g1"], self.pp["g2"]] + list(self.pp["g1_s"]) + list(self.pp["g2_t"])
        )
        g1_s, g2_t = pk[:len(self.pp["g1_s"])], pk[len(self.pp["g1_s"]):]

        gamma = _rand_scalar(group) % order
        w, w_inv = _rand_invertible_2x2(group, order)
        w_inv_t = _transpose(w_inv)
        a_rows, b_rows = [], []
        for x_i, y_i, gs_i, gt_i in zip(x, y, g1_s, g2_t):
            # a_i = (W^-1)^T (x_i, gamma s_i), b_i = W (y_i, -t_i), from pk
            a_rows.append([
                multiexp([g1, gs_i], [row[0] * x_i % order, row[1] * gamma % order], group) for row in w_inv_t
            ])
            b_rows.append([multiexp([g2, gt_i], [row[0] * y_i % order, -row[1] % order], group) for row in w])
        return {
            "c0": group.serialize(g1 ** to_zr(group, gamma)).decode("utf-8"),
            "a": [_serialize_group_vec(group, row) for row in a_rows],
            "b": [_serialize_group_vec(group, row) for row in b_rows],
        }

    def decrypt(self, ct: dict, dk: dict) -> int | None:
        """Decrypt ciphertexts and recover the inner product.

            Args:
                ct: Ciphertext dict.
                dk: Functional decryption key.
        """
        n = self.pp["n"]
        function_matrix = [[gp.mpz(value) for value in row] for row in dk["f"]]
        if len(function_matrix) != n or any(len(row) != n for row in function_matrix):
            raise FEValidationError("invalid function matrix shape")
        if len(ct["a"]) != n or len(ct["b"]) != n or any(len(r) != 2 for r in ct["a"] + ct["b"]):
            raise FEValidationError("ciphertext does not match the parameters")

        c0, key, *flat = _deserialize_group_vec(
            self.group, [ct["c0"], dk["key"]] + [v for row in ct["a"] + ct["b"] for v in row]
        )
        a_rows = [flat[2 * i:2 * i + 2] for i in range(n)]
        b_rows = [flat[2 * (n + j):2 * (n + j) + 2] for j in range(n)]

        # prod_ij (e(a_i0, b_j0) e(a_i1, b_j1))^F_ij = prod_i e(a_i0, prod_j b_j0^F_ij) e(a_i1, ...):
        # two G2 multi-scalar multiplications per row and one product of 2 n + 1 pairings
        lhs, rhs = [c0], [key]
        for i, row in enumerate(function_matrix):
            terms = [(b_rows[j], value % self.order) for j, value in enumerate(row) if value]
            if not terms:
                continue
            for k in (0, 1):
                lhs.append(a_rows[i][k])
                rhs.append(multiexp([b[k] for b, _ in terms], [e for _, e in terms], self.group))
        prod = multi_pair(lhs, rhs)

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        base = pair(g1, g2)
        bound = int((gp.mpz(n) ** 2) * (gp.mpz(self.pp["bound"]) ** 3))
        try:
            return bounded_discrete_log_gt(self.group, base, prod, bound)
        except ValueError:
            logger.error("quadratic result out of bound (wrong key or ciphertext?)")
            return None

    def _prepare_quadratic_ndarray_payload(
        self, lhs_lst_ndarray: list, rhs_lst_ndarray: list
    ) -> tuple[list, list]:
        """Perform the _prepare_quadratic_ndarray_payload operation.

            Args:
                lhs_lst_ndarray: See implementation for details.
                rhs_lst_ndarray: See implementation for details.
        """
        if not isinstance(lhs_lst_ndarray, list) or not isinstance(rhs_lst_ndarray, list):
            raise FESchemeError("ndarray helper expects list-valued lhs and rhs arrays")
        if len(lhs_lst_ndarray) != self.pp["n"] or len(rhs_lst_ndarray) != self.pp["n"]:
            raise FEValidationError("invalid ndarray vector length for quadratic payload")
        shape = lhs_lst_ndarray[0].shape
        if any(ary.shape != shape for ary in lhs_lst_ndarray + rhs_lst_ndarray):
            raise FEValidationError("all ndarray inputs must share the same shape")
        lhs_scaled = [
            scale_to_int(ary, self.precision) for ary in lhs_lst_ndarray
        ]
        rhs_scaled = [
            scale_to_int(ary, self.precision) for ary in rhs_lst_ndarray
        ]
        return lhs_scaled, rhs_scaled

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt pairs ``(x, y)`` of ndarrays element-wise.

        Args:
            lst_ndarray: The ``n`` arrays of ``x`` (one per coordinate), all of
                one shape, scaled by ``10^precision``.
            **kwargs: Must include ``rhs_lst_ndarray``, the ``n`` arrays of ``y``.

        Returns:
            A one-element list: the object array of ciphertexts.
        """
        rhs_lst_ndarray = kwargs.get("rhs_lst_ndarray")
        lhs_scaled, rhs_scaled = self._prepare_quadratic_ndarray_payload(
            lst_ndarray, rhs_lst_ndarray
        )
        ary_ct = np.empty(lhs_scaled[0].shape, dtype=object)
        for i, _ in np.ndenumerate(lhs_scaled[0]):
            x = [int(ary[i]) for ary in lhs_scaled]
            y = [int(ary[i]) for ary in rhs_scaled]
            ary_ct[i] = self.encrypt({"x": x, "y": y})
        return [ary_ct]

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk=dk)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")

        lst_ndarray = []
        for l, ary in enumerate(dict_ndarray_ct):
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                value = self.decrypt(ary[i], dk)
                ary_dec[i] = value
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** (2 * self.precision)))
        return lst_ndarray
