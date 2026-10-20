"""
Partially Function-Hiding Inner-Product Encryption over Pairings
| "A New Paradigm for Public-Key Functional Encryption for Degree-2 Polynomials"
| By Romain Gay, PKC 2020 (ePrint 2020/093) -- Fig. 8, CPA variant (no
| QANIZK), with k = 1 (SXDH); structure as in GoFE's
| ``innerprod/fullysec/part_fh_ipe.go``.

* type:     public-key encryption for vectors in the span of a public matrix
  ``M``; secret-key encryption (``Enc'``) of arbitrary vectors
* setting:  asymmetric pairing group (default BLS12-381)

``a, b <- (1, Z_q)``, ``U in Z_q^{(eta+2) x 2}``, ``V in Z_q^{eta x 2}`` and a
random full-rank ``M in Z^{eta x k}`` with ``k = subspace_dim < eta``; the
public key is ``([a]_1, [U a]_1, V^T M, M)`` and ``msk = (b, U, V)``.

* Enc(t):   ``[c = a r ; (-V^T M t ; M t) + U c]_1`` for ``t in Z^k``.
* Enc'(x):  the same with ``(-V^T x ; x)``; needs ``V``, so it is run by the
  key generator (:meth:`SIFEPartFHIPEKeyGenerator.sec_encrypt`).
* KeyGen(y): ``[(-U^T w ; w)]_2`` with ``w = (b s ; y + V b s)``.
* Dec:      ``prod_j e(ct_j, sk_j) = [<x, y>]_T`` and a bounded discrete log.

Every key holder learns ``M^T y`` (encrypt the unit vectors ``t = e_j`` and
decrypt), the leakage the paper allows; for an integer ``y`` with
``|y_i| <= bound`` this often determines ``y`` unless ``eta - k`` is large, so
``k`` defaults to 1 -- the public encryption space and the leakage have the
same dimension ``k``. Public encryption requires ``|t_i| <= coeff_bound`` and
``|M t| <= bound`` so that the final discrete log is feasible. Versions
<= 0.2.0 handed the master secret key to every encryptor and, by default,
used a square ``M`` (``k = eta``), which reveals ``V`` and with it ``y``.
Deviations: ``M`` and ``V^T M`` are published in the clear instead of in G1.
"""

from __future__ import annotations

import json
import logging
import random

import gmpy2 as gp

_CSPRNG = random.SystemRandom()

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.primitives.pairing.pairing_utils import mat_vec_mod
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr
from pyfe4ai.primitives.pairing.pairing_utils import transpose_mod
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup, multi_pair, pair
from pyfe4ai.primitives.pairing.pairing_backend import deserialize_many, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError
from pyfe4ai.core.versions import record_matches
from pyfe4ai.primitives.pairing.pairing_utils import matrix_inverse_mod
from pyfe4ai._validation import int_vector
from pyfe4ai._validation import strict_int

logger = logging.getLogger(__name__)


def _rand_scalar(order: gp.mpz) -> gp.mpz:
    return gp.mpz(_CSPRNG.randrange(1, int(order)))


def _rand_matrix(rows: int, cols: int, order: gp.mpz) -> list[list[gp.mpz]]:
    """Perform the _rand_matrix operation.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            order: Group order (or exponent order for dequantization).
    """
    return [[_rand_scalar(order) for _ in range(cols)] for _ in range(rows)]


def _rand_bounded_matrix(
    rows: int, cols: int, bound: gp.mpz
) -> list[list[gp.mpz]]:
    """Perform the _rand_bounded_matrix operation.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            bound: Upper bound on absolute values.
    """
    b = int(bound)
    return [[gp.mpz(_CSPRNG.randint(-b, b)) for _ in range(cols)] for _ in range(rows)]


def _matmul_mod(
    left: list[list[gp.mpz]], right: list[list[gp.mpz]], modulus: gp.mpz
) -> list[list[gp.mpz]]:
    """Perform the _matmul_mod operation.

        Args:
            left: Left operand matrix or vector.
            right: Right operand matrix or vector.
            modulus: Modulus for the arithmetic operation.
    """
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    ret = [[gp.mpz(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        row_i = ret[i]
        for k in range(inner):
            left_ik = left[i][k]
            if left_ik == 0:
                continue
            right_k = right[k]
            for j in range(cols):
                row_i[j] += left_ik * right_k[j]
        for j in range(cols):
            row_i[j] = row_i[j] % modulus
    return ret


def _matrix_vec_plain(matrix: list[list[gp.mpz]], vector: list[gp.mpz]) -> list[gp.mpz]:
    """Perform the _matrix_vec_plain operation.

        Args:
            matrix: Input matrix.
            vector: Input vector.
    """
    ret = []
    for row in matrix:
        acc = gp.mpz(0)
        for a_i, b_i in zip(row, vector):
            acc += a_i * b_i
        ret.append(acc)
    return ret


def _g1_pow_vec(group: PairingGroup, g1, scalars: list[gp.mpz]):
    """Perform the _g1_pow_vec operation.

        Args:
            group: Pairing group instance.
            g1: Generator element in G1.
            scalars: Vector of scalar exponents.
    """
    return pow_many(g1, scalars, group)


def _g2_pow_vec(group: PairingGroup, g2, scalars: list[gp.mpz]):
    """Perform the _g2_pow_vec operation.

        Args:
            group: Pairing group instance.
            g2: Generator element in G2.
            scalars: Vector of scalar exponents.
    """
    return pow_many(g2, scalars, group)


def _serialize_vec(group: PairingGroup, vec) -> list[str]:
    """Perform the _serialize_vec operation.

        Args:
            group: Pairing group instance.
            vec: Input vector.
    """
    return [group.serialize(v).decode("utf-8") for v in vec]


def _deserialize_vec(group: PairingGroup, vec: list[str]):
    """Perform the _deserialize_vec operation.

        Args:
            group: Pairing group instance.
            vec: Input vector.
    """
    return deserialize_many(group, vec)


class SIFEPartFHIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for partially function-hiding inner-product encryption over pairings."""

    _scheme_type = CryptoCONST.TYPE_SIFE_PART_FH_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE partially FH-IPE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``eta``, ``bound``
                (or ``bound_x``), ``subspace_dim`` (``1 <= subspace_dim <
                eta``, default 1), ``coeff_bound`` (bound on the coefficients
                ``t`` of public encryption, default 1) and
                ``pairing_group_param`` (asymmetric group, default
                ``"BLS12-381"``).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA)
        self.bound = gp.mpz(config.get("bound", config.get("bound_x", 16)))
        # keys leak M^T y (k values): keep k small relative to eta by default
        self.subspace_dim = int(config.get("subspace_dim", 1))
        if not 1 <= self.subspace_dim < self.eta:
            # a square M (k = eta) makes V = (V^T M) M^{-1} public, and keys reveal y
            raise FEValidationError("subspace_dim must satisfy 1 <= subspace_dim < eta")
        self.coeff_bound = gp.mpz(config.get("coeff_bound", 1))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return record_matches(param, {
            "sec_param": self.sec_param, "eta": self.eta, "bound": gp.digits(self.bound),
            "subspace_dim": self.subspace_dim, "coeff_bound": gp.digits(self.coeff_bound),
            "pairing_group_param": self.pairing_group_param,
        })

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "bound": gp.digits(self.bound),
                    "subspace_dim": self.subspace_dim,
                    "coeff_bound": gp.digits(self.coeff_bound),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        self.g1 = self.group.random(G1)
        self.g2 = self.group.random(G2)
        order = self.order

        a_vec = [gp.mpz(1), _rand_scalar(order)]
        b_vec = [gp.mpz(1), _rand_scalar(order)]
        u = _rand_matrix(self.eta + 2, 2, order)
        v = _rand_matrix(self.eta, 2, order)

        m_entry_bound = max(gp.mpz(1), self.bound // max(1, self.subspace_dim * int(self.coeff_bound)))
        while True:  # Gay20 Sec. 3.1: M of full column rank (M^T M invertible)
            m = _rand_bounded_matrix(self.eta, self.subspace_dim, m_entry_bound)
            try:
                matrix_inverse_mod(_matmul_mod(transpose_mod(m), m, order), order)
                break
            except ValueError:
                continue

        ua_scalars = mat_vec_mod(u, a_vec, order)
        vt_m = _matmul_mod(transpose_mod(v), m, order)

        self.pp = {
            "eta": self.eta,
            "bound": gp.digits(self.bound),
            "subspace_dim": self.subspace_dim,
            "coeff_bound": gp.digits(self.coeff_bound),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.group.serialize(self.g1).decode("utf-8"),
            "g2": self.group.serialize(self.g2).decode("utf-8"),
            "a": _serialize_vec(self.group, _g1_pow_vec(self.group, self.g1, a_vec)),
            "ua": _serialize_vec(self.group, _g1_pow_vec(self.group, self.g1, ua_scalars)),
            "vt_m": [[gp.digits(v) for v in row] for row in vt_m],
            "m": [[gp.digits(v) for v in row] for row in m],
        }
        self.msk = {
            "b": [gp.digits(v) for v in b_vec],
            "u": [[gp.digits(v) for v in row] for row in u],
            "v": [[gp.digits(v) for v in row] for row in v],
        }
        logger.info("SIFE partially FH-IPE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Public-key scheme: encryptors need the public parameters only.

        Returns ``None``; versions <= 0.2.0 returned the master secret key.
        """
        return None

    def sec_encrypt(self, lst_pt: list) -> dict:
        """``Enc'(msk, x)`` for any ``x`` with ``|x_i| <= bound`` (needs ``V``)."""
        x = int_vector(lst_pt, "plaintext", length=self.eta, bound=int(self.bound))
        order = self.order
        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        a = _deserialize_vec(self.group, self.pp["a"])
        ua = _deserialize_vec(self.group, self.pp["ua"])
        vt = transpose_mod([[gp.mpz(v) for v in row] for row in self.msk["v"]])
        r = random_nonzero_zr(self.group)
        neg_vt_x = [(-v) % order for v in mat_vec_mod(vt, [gp.mpz(v) for v in x], order)]
        body = _g1_pow_vec(self.group, g1, neg_vt_x + [gp.mpz(v) for v in x])
        cipher = [elem ** r for elem in a] + [lhs * (rhs ** r) for lhs, rhs in zip(body, ua)]
        return {"ct": _serialize_vec(self.group, cipher)}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of
                int).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for partially FH-IPE decryption key generation")
        fusion_weight = int_vector(
            credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=int(self.bound)
        )

        order = self.order
        y = [gp.mpz(v) for v in fusion_weight]
        b = [gp.mpz(v) for v in self.msk["b"]]
        u = [[gp.mpz(v) for v in row] for row in self.msk["u"]]
        v = [[gp.mpz(v) for v in row] for row in self.msk["v"]]

        s = gp.mpz(int(random_nonzero_zr(self.group)))
        bs = [(s * entry) % order for entry in b]
        vbs = mat_vec_mod(v, bs, order)
        y_vbs = [(a + b_i) % order for a, b_i in zip(y, vbs)]
        key2 = bs + y_vbs
        u_t = transpose_mod(u)
        key1 = [(-val) % order for val in mat_vec_mod(u_t, key2, order)]
        key = key1 + key2

        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        return {"k": _serialize_vec(self.group, _g2_pow_vec(self.group, g2, key))}


class SIFEPartFHIPE(IPFEAbsCrypto):
    """Crypto operations for partially function-hiding inner-product encryption over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the SIFE partially FH-IPE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters);
                encryption is public-key.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.group = PairingGroup(self.pp["pairing_group_param"])
        check_asymmetric(self.group)
        self.order = gp.mpz(int(self.group.order()))

    def encrypt(self, lst_pt: list) -> dict:
        """Public ``Enc(t)`` of ``x = M t`` for ``k`` coefficients ``t``."""
        t = [
            gp.mpz(v)
            for v in int_vector(
                lst_pt, "subspace coefficients", length=self.pp["subspace_dim"],
                bound=int(self.pp["coeff_bound"]),
            )
        ]
        m = [[gp.mpz(v) for v in row] for row in self.pp["m"]]
        x = _matrix_vec_plain(m, t)
        if any(abs(strict_int(v, "plaintext")) > gp.mpz(self.pp["bound"]) for v in x):
            raise FEValidationError("derived plaintext exceeds configured bound")

        a = _deserialize_vec(self.group, self.pp["a"])
        ua = _deserialize_vec(self.group, self.pp["ua"])
        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        vt_m = [[gp.mpz(v) for v in row] for row in self.pp["vt_m"]]

        r = random_nonzero_zr(self.group)
        c = [elem ** r for elem in a]
        uc = [elem ** r for elem in ua]

        vt_mt = mat_vec_mod(vt_m, t, self.order)
        neg_vt_mt = [(-v) % self.order for v in vt_mt]
        cipher2 = _g1_pow_vec(self.group, g1, neg_vt_mt + x)
        cipher2 = [lhs * rhs for lhs, rhs in zip(cipher2, uc)]
        cipher = c + cipher2
        return {"ct": _serialize_vec(self.group, cipher)}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list | None = None):
        """Decrypt ciphertexts and recover the inner product.

            Args:
                dct_ct: Ciphertext dict (or dict of per-client ciphertexts).
                dk: Functional decryption key.
                fusion_weight: Fusion weight vector (or dict of per-client weight vectors).
        """
        cipher = _deserialize_vec(self.group, dct_ct["ct"])
        key = _deserialize_vec(self.group, dk["k"])
        if len(cipher) != self.pp["eta"] + 4 or len(key) != self.pp["eta"] + 4:
            raise FEValidationError("ciphertext or key length mismatch")

        acc = multi_pair(cipher, key)

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
        base = pair(g1, g2)
        bound = int(self.pp["eta"] * (gp.mpz(self.pp["bound"]) ** 2))
        return bounded_discrete_log_gt(self.group, base, acc, bound)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for partially FH-IPE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for partially FH-IPE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError("ndarray helper is not implemented for partially FH-IPE")
