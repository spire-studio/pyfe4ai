"""
Multi-input functional encryption for quadratic functions (cross-user terms)
| Agrawal, Goyal, Tomida, "Multi-Input Quadratic Functional Encryption:
| Stronger Security, Broader Functionality", TCC 2022 (ePrint 2022/1168) --
| Section 5: MIFE for quadratic functions in which every user has its own
| encryption key and any subset of users may be corrupted. Building blocks
| as in the paper: the function-hiding IPFE of [TAO20] recalled in Sec. 5.1,
| the function-hiding SK-MCFE for inner products of Fig. 9 and the SK-MCFE
| for mixed-group inner products of Sec. 3 / Fig. 1.

* type:     multi-input FE with per-user encryption keys
* setting:  asymmetric pairing group (default BLS12-381); selective pos
  message-hiding security (Thm. 5.1) from MDDH_1 (SXDH, for the
  TAO20 IPFE), bilateral MDDH_k with ``k >= 2`` (for the mixed-group
  SK-MCFE) and PRF security

Function class (Def. 2.9): ``n`` users with ``m`` integers each,
``f(x_1, ..., x_n) = sum_{i < k} sum_{j, l} c[(i, j, k, l)] x_i[j] x_k[l]`` --
only terms between *different* users. ``c`` is given as an ``(n m) x (n m)``
matrix ``C[(i - 1) m + j][(k - 1) m + l]`` whose blocks with ``i >= k`` are zero.

Building blocks (all over the same pairing group):

* ``_TAO20``: ``B <- GL_{d+3}(Z_p)``, ``B* = (B^{-1})^T``;
  ``Enc([x]_1; r) = [(x, r, 0, 0) B]_1``, ``KeyGen([y]_2; s) = [(y, 0, s, 0) B*]_2``,
  ``Dec = prod e(ct_j, sk_j) = [<x, y>]_T``; linear in ``(x, r)`` / ``(y, s)``.
* ``_ICFE`` (Fig. 9): one ``_TAO20`` per slot; ``Enc(i, lab, x) = Enc_i((x, 0^L,
  PRF(K, lab), 0))``, keys for ``(y_i, 0^L, r_i, 0)`` with ``sum r_i = 0``.
* ``_MGFE`` (Fig. 1): ``Enc(i, lab, ([x1]_1, [x2]_2))`` with ``z <- Z_p^k``
  gives ``icEnc((x1, 0^m2, z, 0))`` and the IPFE *key* of ``(x2, -z, 0)``;
  keys with ``a <- Z_p^k`` give ``icKeyGen((y1_i, 0^m2, a, 0))_i`` and IPFE
  *ciphertexts* of ``(y2_i, a, 0)``.

The MIFE (Sec. 5.2): the authority samples ``w_(i,j,k,l)``, ``u_ij``,
``u~_ij``, ``v_ij``, ``v~_ij`` and the master keys, and gives user ``i`` the
encryption key ``EK_i`` = master ciphertexts ``MstEnc(i, e_j)`` (``j <= m``) and
``MstEnc(i, 0)`` (``D = 4m + 2k + 17`` of them). A user encrypts by the
homomorphic combination ``sum_j x[j] MCT1_j - (sum_j x[j] - 1) MCT0_1 +
sum_j gamma_j (MCT0_2j - MCT0_2j+1)``. ``MstEnc`` encodes
``b_j = (x[j], 0, s e_(i,j), r u_ij, v_ij, 0^3m)`` as IPFE ciphertexts,
``b~_j = (x[j], 0, s~ w_(*,*,i,j), u~_ij, t v~_ij, 0^3m)`` as IPFE keys,
``(s, 0)`` / ``(s~, 0)`` under a second IPFE and ``f = (r, t, 0^{m^2 n})``,
``h = 0`` under the mixed-group SK-MCFE with the fixed label ``lab0``.
A key for ``c`` is ``(c, mgKeyGen({f~_i, h~_i}), {sigma_ik})``; decryption
returns ``z1 - z2 - z3`` (Sec. 5.2) and a discrete log in
``[-m^2 n^2 C X^2, m^2 n^2 C X^2]``.

Deviations: PRF = HMAC-SHA512 reduced mod p; only the pos variant (every
user must contribute a ciphertext, Remark 2.6; the generic any-transform is
not implemented); ``iSetup(1^lambda; PRF(K, lab0))`` is realised as one IPFE
instance sampled at setup. Encryption keys are large (``(m + D)`` master
ciphertexts per user) and encryption costs a few thousand exponentiations
for small ``n, m``.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import logging
import random
import secrets

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.formats.pairing import MIQFECiphertext
from pyfe4ai.core.formats.pairing import MIQFEEncryptionKey
from pyfe4ai.core.formats.pairing import MIQFEFunctionalKey
from pyfe4ai.core.formats.pairing import MIQFEParameters
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, GT, ZR, PairingGroup, multi_pair, pair
from pyfe4ai.primitives.pairing.pairing_backend import deserialize_many, multiexp, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.primitives.pairing.pairing_utils import matrix_inverse_mod
from pyfe4ai._validation import int_vector

logger = logging.getLogger(__name__)

_CSPRNG = random.SystemRandom()
_LAB0 = b"pyfe4ai/agt22/lab0"


class _Ctx:
    """Pairing group, generators and helpers shared by the building blocks."""

    def __init__(self, group_name: str, g1=None, g2=None) -> None:
        require_pairing_backend()
        self.group = PairingGroup(group_name)
        check_asymmetric(self.group)
        self.p = int(self.group.order())
        self.g1 = g1 if g1 is not None else self.group.random(G1)
        self.g2 = g2 if g2 is not None else self.group.random(G2)
        self._gt = None

    @property
    def gt(self):
        """``e(g1, g2)``, computed on first use (decryption only)."""
        if self._gt is None:
            self._gt = pair(self.g1, self.g2)
        return self._gt

    def rand(self) -> int:
        return _CSPRNG.randrange(self.p)

    def zr(self, value):
        return self.group.init(ZR, int(value) % self.p)

    def ser(self, element) -> str:
        return self.group.serialize(element).decode("utf-8")

    def de(self, text: str):
        return self.group.deserialize(text.encode("utf-8"))

    def de_many(self, texts: list) -> list:
        return deserialize_many(self.group, texts)

    def g1v(self, exps: list) -> list:
        return [self.ser(v) for v in pow_many(self.g1, exps, self.group)]

    def g2v(self, exps: list) -> list:
        return [self.ser(v) for v in pow_many(self.g2, exps, self.group)]

    def multiexp(self, bases: list, exps: list):
        return multiexp(bases, exps, self.group)

    def ip(self, cts: list, keys: list):
        """``prod_j e(ct_j, key_j)`` for serialised G1 / G2 vectors of equal length."""
        if len(cts) != len(keys):
            raise FESchemeError("IPFE ciphertext and key lengths differ")
        return multi_pair(self.de_many(cts), self.de_many(keys))


class _TAO20:
    """Function-hiding IPFE of [TAO20] (AGT22 Sec. 5.1), working on exponents."""

    def __init__(self, ctx: _Ctx, dim: int) -> None:
        self.ctx = ctx
        self.dim = dim
        self.size = dim + 3

    def setup(self) -> dict:
        p = self.ctx.p
        while True:
            b = [[self.ctx.rand() for _ in range(self.size)] for _ in range(self.size)]
            try:
                inv, _ = matrix_inverse_mod(b, p)
                break
            except ValueError:
                continue
        b_star = [[int(inv[r][c]) for r in range(self.size)] for c in range(self.size)]  # (B^-1)^T
        return {"B": b, "Bs": b_star}

    def _times(self, vec: list, mat: list) -> list:
        p = self.ctx.p
        return [sum(v * row[c] for v, row in zip(vec, mat)) % p for c in range(self.size)]

    def enc(self, msk: dict, x: list, r: int) -> list:
        if len(x) != self.dim:
            raise FEValidationError("IPFE input must have {} entries".format(self.dim))
        return self._times(list(x) + [r, 0, 0], msk["B"])

    def key(self, msk: dict, y: list, s: int) -> list:
        if len(y) != self.dim:
            raise FEValidationError("IPFE key vector must have {} entries".format(self.dim))
        return self._times(list(y) + [0, s, 0], msk["Bs"])


class _ICFE:
    """Function-hiding SK-MCFE for inner products (AGT22 Fig. 9), on exponents."""

    def __init__(self, ctx: _Ctx, n: int, length: int) -> None:
        self.ctx = ctx
        self.n = n
        self.length = length
        self.tao = _TAO20(ctx, 2 * length + 2)

    def setup(self) -> dict:
        return {"K": secrets.token_hex(32), "inst": [self.tao.setup() for _ in range(self.n)]}

    def prf(self, msk: dict, lab: bytes) -> int:
        digest = hmac.new(bytes.fromhex(msk["K"]), lab, hashlib.sha512).digest()
        return int.from_bytes(digest, "big") % self.ctx.p

    def enc(self, msk: dict, i: int, lab: bytes, x: list) -> list:
        x_hat = list(x) + [0] * self.length + [self.prf(msk, lab), 0]
        return self.tao.enc(msk["inst"][i], x_hat, self.ctx.rand())

    def keygen(self, msk: dict, ys: list) -> list:
        p = self.ctx.p
        r = [self.ctx.rand() for _ in range(self.n - 1)]
        r.append((-sum(r)) % p)
        return [
            self.tao.key(msk["inst"][i], list(y) + [0] * self.length + [r_i, 0], self.ctx.rand())
            for i, (y, r_i) in enumerate(zip(ys, r))
        ]


class _MGFE:
    """Function-hiding SK-MCFE for mixed-group inner products (AGT22 Fig. 1)."""

    def __init__(self, ctx: _Ctx, n: int, m1: int, m2: int, k: int) -> None:
        self.ctx = ctx
        self.n, self.m1, self.m2, self.k = n, m1, m2, k
        self.ic = _ICFE(ctx, n, m1 + m2 + k + 1)
        self.tao = _TAO20(ctx, m2 + k + 1)

    def setup(self) -> dict:
        return {"ic": self.ic.setup(), "inst": [self.tao.setup() for _ in range(self.n)]}

    def enc(self, msk: dict, i: int, lab: bytes, x1: list, x2: list) -> tuple:
        """Exponents of ``(icCT_i in G1, iSK_i in G2)`` for ``([x1]_1, [x2]_2)``."""
        z = [self.ctx.rand() for _ in range(self.k)]
        x1t = list(x1) + [0] * self.m2 + z + [0]
        x2t = list(x2) + [(-v) % self.ctx.p for v in z] + [0]
        return (
            self.ic.enc(msk["ic"], i, lab, x1t),
            self.tao.key(msk["inst"][i], x2t, self.ctx.rand()),
        )

    def keygen(self, msk: dict, y1s: list, y2s: list) -> tuple:
        """Exponents of ``(icSK_i in G2, iCT_i in G1)`` for ``{[y1_i]_2, [y2_i]_1}``."""
        a = [self.ctx.rand() for _ in range(self.k)]
        ic_sk = self.ic.keygen(msk["ic"], [list(y1) + [0] * self.m2 + a + [0] for y1 in y1s])
        i_ct = [
            self.tao.enc(msk["inst"][i], list(y2) + a + [0], self.ctx.rand())
            for i, y2 in enumerate(y2s)
        ]
        return ic_sk, i_ct


def _dims(n: int, m: int, k: int) -> dict:
    return {"D": 4 * m + 2 * k + 17, "ipfe1": m * n + 3 * m + 4, "m1": m * m * n + 2, "m2": 1}


class MultiInputQuadraticKeyGenerator(IPFEAbsKeyGenerator):
    """Setup and key generation of the AGT22 MIFE for quadratic functions."""

    _scheme_type = CryptoCONST.TYPE_QUADRATIC_MIQFE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: ``n`` (users, >= 2), ``lst_nid``, ``m`` (integers per user),
                ``bound_x`` (``X``), ``bound_c`` (``C``), ``mddh_k`` (``k >= 2``,
                default 2) and ``pairing_group_param`` (asymmetric, default BLS12-381).
        """
        super().__init__(config, **kwargs)
        self.m = int(config.get("m", 2))
        self.bound_x = int(config.get("bound_x", 8))
        self.bound_c = int(config.get("bound_c", 8))
        self.k = int(config.get("mddh_k", 2))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        if not self.n or self.n < 2 or len(self.lst_nid) != self.n:
            raise FEValidationError("need n >= 2 users listed in lst_nid")
        if self.m < 1 or self.k < 2:
            raise FEValidationError("need m >= 1 and mddh_k >= 2 (bilateral MDDH)")
        if self.bound_x < 1 or self.bound_c < 1:
            raise FEValidationError("need bound_x >= 1 and bound_c >= 1")

    def setup(self) -> None:
        """Sample the master key and every user's encryption key (master ciphertexts)."""
        ctx = _Ctx(self.pairing_group_param)
        n, m, k = self.n, self.m, self.k
        dims = _dims(n, m, k)
        rnd = ctx.rand
        self.ctx = ctx
        self.ipfe1 = _TAO20(ctx, dims["ipfe1"])
        self.ipfe2 = _TAO20(ctx, 2)
        self.mg = _MGFE(ctx, n, dims["m1"], dims["m2"], k)
        self.msk = {
            "ipfe1": self.ipfe1.setup(),
            "ipfe2": self.ipfe2.setup(),
            "mg": self.mg.setup(),
            "w": {(i, j, a, b): rnd() for i in range(n) for j in range(m) for a in range(n) for b in range(m)},
            "u": [[rnd() for _ in range(m)] for _ in range(n)],
            "ut": [[rnd() for _ in range(m)] for _ in range(n)],
            "v": [[rnd() for _ in range(m)] for _ in range(n)],
            "vt": [[rnd() for _ in range(m)] for _ in range(n)],
        }
        self.pp = MIQFEParameters.new(
            pairing_group_param=self.pairing_group_param,
            g1=ctx.ser(ctx.g1),
            g2=ctx.ser(ctx.g2),
            n=n,
            m=m,
            mddh_k=k,
            lst_nid=list(self.lst_nid),
            bound_x=self.bound_x,
            bound_c=self.bound_c,
            sec_param=self.sec_param,
        ).to_dict()
        self.ek = {}
        for i, nid in enumerate(self.lst_nid):
            unit = [[1 if a == j else 0 for a in range(m)] for j in range(m)]
            self.ek[nid] = MIQFEEncryptionKey.new(
                mct1=[self._mst_enc(i, e_j) for e_j in unit],
                mct0=[self._mst_enc(i, [0] * m) for _ in range(dims["D"])],
            ).to_dict()
        logger.info("AGT22 multi-input quadratic FE setup successfully")

    def _mst_enc(self, i: int, x: list) -> dict:
        """``MstEnc(MSK, i, x)`` of AGT22 Sec. 5.2."""
        ctx, msk, n, m = self.ctx, self.msk, self.n, self.m
        rnd = ctx.rand
        s, st, r, t = rnd(), rnd(), rnd(), rnd()
        cts, sks = [], []
        for j in range(m):
            e_ij = [0] * (n * m)
            e_ij[i * m + j] = s
            w_col = [st * msk["w"][(a, b, i, j)] % ctx.p for a in range(n) for b in range(m)]
            b_j = [x[j], 0] + e_ij + [r * msk["u"][i][j] % ctx.p, msk["v"][i][j]] + [0] * (3 * m)
            bt_j = [x[j], 0] + w_col + [msk["ut"][i][j], t * msk["vt"][i][j] % ctx.p] + [0] * (3 * m)
            cts.append(ctx.g1v(self.ipfe1.enc(msk["ipfe1"], b_j, rnd())))
            sks.append(ctx.g2v(self.ipfe1.key(msk["ipfe1"], bt_j, rnd())))
        mg_ct, mg_sk = self.mg.enc(msk["mg"], i, _LAB0, [r, t] + [0] * (m * m * n), [0])
        return MIQFECiphertext.new(
            ct=cts,
            sk=sks,
            ct2=ctx.g1v(self.ipfe2.enc(msk["ipfe2"], [s, 0], rnd())),
            sk2=ctx.g2v(self.ipfe2.key(msk["ipfe2"], [st, 0], rnd())),
            mg_ct=ctx.g1v(mg_ct),
            mg_sk=ctx.g2v(mg_sk),
        ).to_dict()

    def get_public_parameters(self) -> dict:
        """Group description, generators and dimensions."""
        return dict(self.pp)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """User *nid*'s encryption key: its master ciphertexts only (no master secret).

        A corrupted user learns its own key; together with a functional key for
        ``c`` it can still evaluate ``c`` on its own chosen inputs, which the
        functionality allows (e.g. ``sum_l c[(i,j,k,l)] x_k[l]``).
        """
        ek = self.ek.get(nid)
        return None if ek is None else copy.deepcopy(ek)

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Key for ``c`` (``credentials["coefficients"]``, an ``nm x nm`` matrix).

        Entries in blocks ``(i, k)`` with ``i >= k`` must be zero (Def. 2.9).
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials with the coefficient matrix")
        n, m, ctx, msk = self.n, self.m, self.ctx, self.msk
        rows = credentials.get("coefficients")
        if not isinstance(rows, list) or len(rows) != n * m:
            raise FEValidationError("coefficients must be an (n m) x (n m) matrix")
        c = [int_vector(row, "coefficient row", length=n * m, bound=self.bound_c) for row in rows]
        for mu in range(n * m):
            for nu in range(n * m):
                if c[mu][nu] and mu // m >= nu // m:
                    raise FEValidationError("only terms x_i[j] x_k[l] with i < k are allowed")
        p = ctx.p

        def coef(i, j, a, b):
            return c[i * m + j][a * m + b]

        f1 = []
        for i in range(n):
            ft0 = sum(
                coef(i, j, a, b) * msk["u"][i][j] * msk["ut"][a][b]
                for j in range(m) for a in range(n) for b in range(m)
            ) % p
            ft1 = sum(
                coef(a, b, i, j) * msk["v"][a][b] * msk["vt"][i][j]
                for j in range(m) for a in range(n) for b in range(m)
            ) % p
            f1.append([ft0, ft1] + [0] * (m * m * n))
        ic_sk, i_ct = self.mg.keygen(msk["mg"], f1, [[0] for _ in range(n)])
        sigma = [
            [sum(coef(i, j, a, b) * msk["w"][(i, j, a, b)] for j in range(m) for b in range(m)) % p
             for a in range(n)]
            for i in range(n)
        ]
        return MIQFEFunctionalKey.new(
            c=[[str(v) for v in row] for row in c],
            mg_sk=[ctx.g2v(v) for v in ic_sk],
            mg_ct=[ctx.g1v(v) for v in i_ct],
            sigma=[[str(v) for v in row] for row in sigma],
        ).to_dict()


class MultiInputQuadratic(IPFEAbsCrypto):
    """Encryption with ``EK_i`` and decryption of the AGT22 MIFE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for users, ``keys.sk`` (their ``EK_i``)."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        probe = PairingGroup(self.pp["pairing_group_param"])
        self.ctx = _Ctx(
            self.pp["pairing_group_param"],
            probe.deserialize(self.pp["g1"].encode("utf-8")),
            probe.deserialize(self.pp["g2"].encode("utf-8")),
        )
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        """Homomorphic combination of this user's master ciphertexts (Sec. 5.2 Enc)."""
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        ctx, m = self.ctx, self.pp["m"]
        x = int_vector(lst_pt, "plaintext", length=m, bound=int(self.pp["bound_x"]))
        sk = MIQFEEncryptionKey.from_dict(self.sk)
        mct1, mct0 = sk.mct1, sk.mct0
        weights = list(zip(mct1, x))
        weights.append((mct0[0], 1 - sum(x)))
        for g in range((len(mct0) - 1) // 2):
            gamma = ctx.rand()
            weights.append((mct0[2 * g + 1], gamma))
            weights.append((mct0[2 * g + 2], -gamma))
        weights = [(mct, a % ctx.p) for mct, a in weights if a % ctx.p]

        def combine(get) -> list:
            # entry j: prod_w get(mct_w)[j] ** a_w, one product of powers per entry
            rows = [get(mct) for mct, _ in weights]
            width = min(len(row) for row in rows)
            elems = iter(ctx.de_many([v for row in rows for v in row[:width]]))
            rows = [[next(elems) for _ in range(width)] for _ in rows]
            exps = [a for _, a in weights]
            return [ctx.ser(ctx.multiexp([row[j] for row in rows], exps)) for j in range(width)]

        return MIQFECiphertext.new(
            ct=[combine(lambda mct, j=j: MIQFECiphertext(mct).ct[j]) for j in range(m)],
            sk=[combine(lambda mct, j=j: MIQFECiphertext(mct).sk[j]) for j in range(m)],
            ct2=combine(lambda mct: MIQFECiphertext(mct).ct2),
            sk2=combine(lambda mct: MIQFECiphertext(mct).sk2),
            mg_ct=combine(lambda mct: MIQFECiphertext(mct).mg_ct),
            mg_sk=combine(lambda mct: MIQFECiphertext(mct).mg_sk),
        ).to_dict()

    def decrypt(self, dct_ct: dict, dk: dict) -> int | None:
        """``z = z1 - z2 - z3`` (Sec. 5.2 Dec) and a bounded discrete log.

        Raises:
            FESchemeError: Unless every user contributes a ciphertext.
        """
        if not dk or "c" not in MIQFEFunctionalKey(dk):
            raise FEKeyError("no decryption key provided")
        lst_nid, n, m = self.pp["lst_nid"], self.pp["n"], self.pp["m"]
        if set(dct_ct) != set(lst_nid):
            raise FESchemeError("need one ciphertext of every user")
        ctx = self.ctx
        cts = [MIQFECiphertext(dct_ct[nid]) for nid in lst_nid]
        key = MIQFEFunctionalKey.from_dict(dk)
        c = [[int(v) for v in row] for row in key.c]
        sigma = [[int(v) for v in row] for row in key.sigma]
        # z1, z2: products of powers in GT, one multi-exponentiation each
        terms = [
            (ctx.ip(cts[mu // m].ct[mu % m], cts[nu // m].sk[nu % m]), c[mu][nu])
            for mu in range(n * m) for nu in range(n * m) if c[mu][nu]
        ]
        z1 = ctx.multiexp(*zip(*terms)) if terms else ctx.group.init(GT, 1)
        terms = [
            (ctx.ip(cts[i].ct2, cts[a].sk2), sigma[i][a]) for i in range(n) for a in range(n) if sigma[i][a]
        ]
        z2 = ctx.multiexp(*zip(*terms)) if terms else ctx.group.init(GT, 1)
        z3 = ctx.group.init(GT, 1)
        for i in range(n):
            z3 *= ctx.ip(cts[i].mg_ct, key.mg_sk[i])
            z3 *= ctx.ip(key.mg_ct[i], cts[i].mg_sk)
        bound = m * m * n * n * int(self.pp["bound_c"]) * int(self.pp["bound_x"]) ** 2
        try:
            return bounded_discrete_log_gt(ctx.group, ctx.gt, z1 / (z2 * z3), bound)
        except ValueError:
            logger.error("result out of bound (wrong key or ciphertexts?)")
            warn_at_caller("result out of bound (wrong key or ciphertexts?)")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not provided: encryption keys and costs are large; use :meth:`encrypt`."""
        raise NotImplementedError("ndarray helpers are not provided for multi-input quadratic FE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not provided (see :meth:`encrypt_lst_ndarray`)."""
        raise NotImplementedError("ndarray helpers are not provided for multi-input quadratic FE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not provided (see :meth:`encrypt_lst_ndarray`)."""
        raise NotImplementedError("ndarray helpers are not provided for multi-input quadratic FE")
