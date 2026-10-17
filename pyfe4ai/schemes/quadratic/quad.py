"""
Public-key functional encryption for degree-2 polynomials (Quad)
| "A New Paradigm for Public-Key Functional Encryption for Degree-2 Polynomials"
| By Romain Gay, PKC 2020 (ePrint 2020/093) -- Fig. 4 (Quad), CPA variant,
| over the partially function-hiding IPFE of Fig. 8 with k = 1.

* type:     public-key encryption; keys for matrices ``F``, decryption gives
  ``x^T F y``
* setting:  asymmetric pairing group (default BLS12-381); semi-adaptive
  simulation security from bilateral DLIN, DDH in G1 and SXDH

Setup: ``a <- DDH`` (``(1, a)``), ``B <- DLIN`` (``[[b1, 0], [0, b2], [1, 1]]``),
``U in Z_p^{n x 2}``, ``V in Z_p^{m x 3}`` and
``M = [[a (x) (Id_m | V B), 0], [0, Id_n (x) B]]`` of size
``(2m + 3n) x (m + 2 + 2n)``; the IPFE is set up for ``[M]_1`` and
``pk = ([U a]_1, [V B]_2, pk_IPFE)``.

* Enc(x, y):  ``r <- Z_p``, ``s <- Z_p^2``; ``ct_x = [x + U a r]_1``,
  ``ct_y = [y + V B s]_2``, ``ct_IPFE = Enc_IPFE(r (y ; s) ; x (x) s)``.
* KeyGen(F):  ``sk_F = (F, KeyGen_IPFE(vect(U^T F) ; vect(F V)))``.
* Dec:        ``e(ct_x, F ct_y) / Dec_IPFE = [x^T F y]_T`` and a discrete log
  bounded by ``n m B^3``.

The IPFE (Fig. 8, CPA, no QANIZK): ``a', b' <- (1, Z_p)``,
``U' in Z_p^{(N+2) x 2}``, ``V' in Z_p^{N x 2}``,
``pk_IPFE = ([a']_1, [U' a']_1, [V'^T M]_1, [M]_1)``;
``Enc_IPFE(t) = [c = a' r' ; (-V'^T M t ; M t) + U' c]_1`` computed from
``[M]_1`` (``M`` contains the secrets ``a``, ``B``, ``V B``);
``KeyGen_IPFE(z) = [(-U'^T w ; w)]_2`` with ``w = (b' s' ; z + V' b' s')``;
``Dec_IPFE = prod e(ct_j, sk_j) = [<M t, z>]_T``.

As in the paper, ``M`` is not of full column rank (the ``a (x) V B`` columns
are combinations of the ``a (x) Id_m`` ones); the redundant columns are what
lets the encryptor form ``r (y + V B s)`` from ``[M]_1``.

A ciphertext has ``4n + 2m + 4`` elements of G1 and ``m`` of G2. Signed
inputs (``|x_i|, |y_j|, |F_ij| <= B``): the discrete log is searched in
``[-n m B^3, n m B^3]``. Versions <= 0.2.0 encrypted ``x (x) y`` with an IPFE
instead (ciphertexts of ``n m`` elements, not this construction).
"""

from __future__ import annotations

import logging
import random

import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, ZR, PairingGroup, multi_pair, pair
from pyfe4ai.utils.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)

_CSPRNG = random.SystemRandom()


def _matmul(left: list, right: list, p: int) -> list:
    return [[sum(a * b for a, b in zip(row, col)) % p for col in zip(*right)] for row in left]


def _transpose(matrix: list) -> list:
    return [list(col) for col in zip(*matrix)]


class _Ctx:
    """Group, generators and (de)serialisation for a Quad instance."""

    def __init__(self, group_name: str, g1=None, g2=None) -> None:
        require_pairing_backend()
        self.group = PairingGroup(group_name)
        check_asymmetric(self.group)
        self.p = int(self.group.order())
        self.g1 = g1 if g1 is not None else self.group.random(G1)
        self.g2 = g2 if g2 is not None else self.group.random(G2)
        self.gt = pair(self.g1, self.g2)

    def rand(self) -> int:
        return _CSPRNG.randrange(1, self.p)

    def zr(self, value):
        return self.group.init(ZR, int(value) % self.p)

    def ser(self, element) -> str:
        return self.group.serialize(element).decode("utf-8")

    def de(self, text: str):
        return self.group.deserialize(text.encode("utf-8"))

    def multiexp(self, bases: list, exps: list):
        acc = bases[0] ** self.zr(0)
        for b, e in zip(bases, exps):
            e = int(e) % self.p
            if e:
                acc = acc * (b ** self.zr(e))
        return acc


class QuadraticQuadKeyGenerator(IPFEAbsKeyGenerator):
    """Setup and key generation of Gay20's Quad."""

    _scheme_type = CryptoCONST.TYPE_QUADRATIC_QUAD

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: ``n`` (length of ``x``), ``m`` (length of ``y``), ``bound``
                (``B`` on every entry of ``x``, ``y`` and ``F``) and
                ``pairing_group_param`` (asymmetric, default BLS12-381).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.n = int(config.get("n", 2))
        self.m = int(config.get("m", self.n))
        self.bound = int(config.get("bound", 16))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        if self.n < 1 or self.m < 1:
            raise FEValidationError("n and m must be positive")

    def setup(self) -> None:
        """Sample ``a``, ``B``, ``U``, ``V``, build ``[M]_1`` and set up the IPFE."""
        ctx = _Ctx(self.pairing_group_param)
        p, n, m = ctx.p, self.n, self.m
        rnd = ctx.rand
        a = [1, rnd()]
        b_mat = [[rnd(), 0], [0, rnd()], [1, 1]]
        u = [[rnd(), rnd()] for _ in range(n)]
        v = [[rnd() for _ in range(3)] for _ in range(m)]
        vb = _matmul(v, b_mat, p)
        big_n, big_k = 2 * m + 3 * n, m + 2 + 2 * n
        mat = [[0] * big_k for _ in range(big_n)]
        for i in range(2):  # a (x) (Id_m | V B)
            for row in range(m):
                mat[i * m + row][row] = a[i]
                for c in range(2):
                    mat[i * m + row][m + c] = a[i] * vb[row][c] % p
        for i in range(n):  # Id_n (x) B
            for k in range(3):
                for c in range(2):
                    mat[2 * m + 3 * i + k][m + 2 + 2 * i + c] = b_mat[k][c]
        # partially function-hiding IPFE (Gay20 Fig. 8, k = 1) for [M]_1
        a2 = [1, rnd()]
        b2 = [1, rnd()]
        u2 = [[rnd(), rnd()] for _ in range(big_n + 2)]
        v2 = [[rnd(), rnd()] for _ in range(big_n)]
        v2t_m = _matmul(_transpose(v2), mat, p)
        ua = [(row[0] * a[0] + row[1] * a[1]) % p for row in u]
        u2a2 = [(row[0] * a2[0] + row[1] * a2[1]) % p for row in u2]

        def g1v(values):
            return [ctx.ser(ctx.g1 ** ctx.zr(x)) for x in values]

        self.ctx = ctx
        self.pp = {
            "pairing_group_param": self.pairing_group_param,
            "n": n,
            "m": m,
            "bound": self.bound,
            "sec_param": self.sec_param,
            "g1": ctx.ser(ctx.g1),
            "g2": ctx.ser(ctx.g2),
            "ua": g1v(ua),
            "vb": [[ctx.ser(ctx.g2 ** ctx.zr(x)) for x in row] for row in vb],
            "ipfe": {
                "a": g1v(a2),
                "ua": g1v(u2a2),
                "vt_m": [g1v(row) for row in v2t_m],
                "m": [g1v(row) for row in mat],
            },
        }
        self.msk = {"u": u, "v": v, "ipfe": {"b": b2, "u": u2, "v": v2}}
        logger.info("Quadratic Quad (Gay20 Fig. 4) setup successfully")

    def get_public_parameters(self, **kwargs) -> dict | None:
        """The public key ``([U a]_1, [V B]_2, pk_IPFE)`` and the dimensions."""
        return self.pp

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Public-key scheme: encryption needs the public key only."""
        return None

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """``sk_F = (F, KeyGen_IPFE(vect(U^T F) ; vect(F V)))``.

        Args:
            sid: Session identifier (unused).
            **kwargs: ``credentials`` with ``function_matrix`` (``n x m``
                integers bounded by ``bound``).
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for Quadratic Quad decryption key generation")
        f_rows = credentials.get("function_matrix")
        if not isinstance(f_rows, list) or len(f_rows) != self.n:
            raise FEValidationError("invalid function_matrix")
        f_mat = [int_vector(row, "function_matrix row", length=self.m, bound=self.bound) for row in f_rows]
        ctx, p = self.ctx, self.ctx.p
        utf = _matmul(_transpose(self.msk["u"]), f_mat, p)  # 2 x m
        fv = _matmul(f_mat, self.msk["v"], p)  # n x 3
        z = [x for row in utf for x in row] + [x for row in fv for x in row]
        ipfe = self.msk["ipfe"]
        s2 = ctx.rand()
        bs = [b * s2 % p for b in ipfe["b"]]
        w = bs + [(z_i + row[0] * bs[0] + row[1] * bs[1]) % p for z_i, row in zip(z, ipfe["v"])]
        head = [(-sum(row[c] * w_j for row, w_j in zip(ipfe["u"], w))) % p for c in (0, 1)]
        return {
            "f": [[str(x) for x in row] for row in f_mat],
            "k": [ctx.ser(ctx.g2 ** ctx.zr(x)) for x in head + w],
        }


class QuadraticQuad(IPFEAbsCrypto):
    """Encryption and decryption of Gay20's Quad."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` (the public key)."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        group = PairingGroup(self.pp["pairing_group_param"])
        self.ctx = _Ctx(
            self.pp["pairing_group_param"],
            group.deserialize(self.pp["g1"].encode("utf-8")),
            group.deserialize(self.pp["g2"].encode("utf-8")),
        )

    def encrypt(self, payload: dict) -> dict:
        """Encrypt ``{"x": n integers, "y": m integers}`` (entries bounded by ``bound``)."""
        if not isinstance(payload, dict):
            raise FEValidationError("Quadratic Quad expects a dict payload with x and y")
        pp, ctx = self.pp, self.ctx
        p, bound = ctx.p, int(pp["bound"])
        x = int_vector(payload.get("x"), "x", length=pp["n"], bound=bound)
        y = int_vector(payload.get("y"), "y", length=pp["m"], bound=bound)
        r, s = ctx.rand(), [ctx.rand(), ctx.rand()]
        ct_x = [
            (ctx.g1 ** ctx.zr(x_i)) * (ctx.de(ua_i) ** ctx.zr(r)) for x_i, ua_i in zip(x, pp["ua"])
        ]
        ct_y = [
            (ctx.g2 ** ctx.zr(y_j)) * ctx.multiexp([ctx.de(v) for v in row], s)
            for y_j, row in zip(y, pp["vb"])
        ]
        t = (
            [r * y_j % p for y_j in y]
            + [r * s_c % p for s_c in s]
            + [x_i * s_c % p for x_i in x for s_c in s]
        )
        ipfe = pp["ipfe"]
        r2 = ctx.zr(ctx.rand())
        c = [ctx.de(v) ** r2 for v in ipfe["a"]]
        neg_vtmt = [
            ctx.multiexp([ctx.de(v) for v in row], [(-t_j) % p for t_j in t]) for row in ipfe["vt_m"]
        ]
        mt = [ctx.multiexp([ctx.de(v) for v in row], t) for row in ipfe["m"]]
        body = [lhs * (ctx.de(u) ** r2) for lhs, u in zip(neg_vtmt + mt, ipfe["ua"])]
        return {
            "ct_x": [ctx.ser(v) for v in ct_x],
            "ct_y": [ctx.ser(v) for v in ct_y],
            "ct": [ctx.ser(v) for v in c + body],
        }

    def decrypt(self, ct: dict, dk: dict) -> int | None:
        """``log(e(ct_x, F ct_y) / Dec_IPFE)``; ``None`` beyond ``n m B^3``."""
        if not dk or "k" not in dk:
            raise FEKeyError("no decryption key provided")
        pp, ctx = self.pp, self.ctx
        n, m = pp["n"], pp["m"]
        if len(dk["f"]) != n or any(len(row) != m for row in dk["f"]):
            raise FESchemeError("the key's function matrix does not match the parameters")
        f_mat = [[int(v) for v in row] for row in dk["f"]]
        key = [ctx.de(v) for v in dk["k"]]
        ct_ipfe = [ctx.de(v) for v in ct["ct"]]
        ct_x = [ctx.de(v) for v in ct["ct_x"]]
        ct_y = [ctx.de(v) for v in ct["ct_y"]]
        n_ipfe = 2 * m + 3 * n + 4
        if len(ct_x) != n or len(ct_y) != m or len(ct_ipfe) != n_ipfe or len(key) != n_ipfe:
            raise FESchemeError("ciphertext and key do not match the parameters")
        d = multi_pair(ct_ipfe, key)
        v = multi_pair(ct_x, [ctx.multiexp(ct_y, row) for row in f_mat])
        bound = n * m * int(pp["bound"]) ** 3
        try:
            return bounded_discrete_log_gt(ctx.group, ctx.gt, v / d, bound)
        except ValueError:
            logger.error("quadratic result out of bound (wrong key or ciphertext?)")
            return None

    def _prepare_quadratic_ndarray_payload(self, lhs_lst_ndarray: list, rhs_lst_ndarray: list) -> tuple:
        if not isinstance(lhs_lst_ndarray, list) or not isinstance(rhs_lst_ndarray, list):
            raise FESchemeError("ndarray helper expects list-valued lhs and rhs arrays")
        if len(lhs_lst_ndarray) != self.pp["n"] or len(rhs_lst_ndarray) != self.pp["m"]:
            raise FEValidationError("invalid ndarray vector length for quadratic payload")
        shape = lhs_lst_ndarray[0].shape
        if any(ary.shape != shape for ary in lhs_lst_ndarray + rhs_lst_ndarray):
            raise FEValidationError("all ndarray inputs must share the same shape")
        lhs = [scale_to_int(ary, self.precision) for ary in lhs_lst_ndarray]
        rhs = [scale_to_int(ary, self.precision) for ary in rhs_lst_ndarray]
        return lhs, rhs

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt position-wise pairs ``(x, y)`` taken from ``n`` and ``m`` arrays."""
        lhs, rhs = self._prepare_quadratic_ndarray_payload(lst_ndarray, kwargs.get("rhs_lst_ndarray"))
        ary_ct = np.empty(lhs[0].shape, dtype=object)
        for i, _ in np.ndenumerate(lhs[0]):
            ary_ct[i] = self.encrypt({"x": [int(a[i]) for a in lhs], "y": [int(a[i]) for a in rhs]})
        return [ary_ct]

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt position-wise (delegates to :meth:`decrypt_lst_ndarray_ct`)."""
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk=dk)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise (results scaled by ``10^(2 precision)``)."""
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")
        out = []
        for l, ary in enumerate(dict_ndarray_ct):
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                value = self.decrypt(ary[i], dk)
                ary_dec[i] = value
            out.append(decoded_layer(ary_dec, l, 10 ** (2 * self.precision)))
        return out
