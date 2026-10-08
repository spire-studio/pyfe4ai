"""
Function-hiding single-input inner-product encryption over asymmetric
pairings, with group-encoded inputs and output ``[<x, y>]_T``.

| Abdalla, Catalano, Fiore, Gay, Ursu, "Multi-Input Functional Encryption for
| Inner Products: Function-Hiding Realizations and Constructions without
| Pairings", CRYPTO 2018 (ePrint 2017/972): Fig. 12 for a single slot
| (``n = 1``), built from the AGRW17 scheme of Fig. 24 (inner layer) and the
| ALS16 D_k-MDDH scheme of Fig. 7 over pairing groups (outer layer), with
| ``k = 1`` (SXDH); Appendix B (the LV16 padding ``x || 0``, ``y || 0``)
| turns its weak function hiding into full function hiding (selective).

This is the black-box "FH-IPFE" that the function-hiding MCFE / DMCFE
constructions of :mod:`pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing` use: its
encryption takes ``[x]_1`` and its key generation ``[y]_2`` (vectors of group
elements), and decryption returns ``[<x, y>]_T = e(g1, g2)^{<x, y>}``, so
partial results of several instances can be multiplied.

With ``x' = (x, 0^m)``, ``y' = (y, 0^m)`` (``m2 = 2m``) and ``l = m2 + 3``:

* inner Enc:    ``s <- Z_p``; ``ct_in = [(A s ; (x', z) + U A s)]_1 in G1^l``;
* inner KeyGen: ``r <- Z_p``; ``v = (y', r)``; ``sk_in = [(-U^T v ; v)]_2 in G2^l``,
  ``z_sum = z r``;
* outer KeyGen of ``[ct_in]_1``: ``ct = [(W'^T ct_in ; ct_in)]_1 in G1^{l+2}``;
* outer Enc of ``[sk_in]_2``: ``r' <- Z_p``; ``key = [(-A' r' ; sk_in + W' A' r')]_2``;
* Dec: ``prod_j e(ct_j, key_j) / [z_sum]_T = [<ct_in, sk_in> - z r]_T = [<x, y>]_T``.

``A, A' in Z_p^2``, ``U in Z_p^{(m2+1) x 2}``, ``W' in Z_p^{l x 2}`` and
``z in Z_p`` form the master secret key.
"""

from __future__ import annotations

from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair
from pyfe4ai.utils.exceptions import FEValidationError

SYMMETRIC_GROUPS = ("SS512", "SS1024")


def check_asymmetric(group_name: str) -> None:
    """Reject symmetric pairing groups: the schemes rely on SXDH."""
    if group_name in SYMMETRIC_GROUPS:
        raise FEValidationError(
            "{} is a symmetric pairing group, where SXDH does not hold; use an "
            "asymmetric group such as MNT224 or BN254".format(group_name)
        )


class FHIPE:
    """ACFGU18 Fig. 12 (n = 1, k = 1) with LV16 padding, for vectors of length *m*."""

    def __init__(self, group: PairingGroup, g1, g2, m: int) -> None:
        self.group = group
        self.g1 = g1
        self.g2 = g2
        self.gt = pair(g1, g2)
        self.m = m
        self.m2 = 2 * m
        self.ell = self.m2 + 3
        self.order = int(group.order())

    # -- helpers ---------------------------------------------------------------
    def _zr(self, value: int):
        return self.group.init(ZR, int(value) % self.order)

    def _rand(self, rng=None) -> int:
        if rng is None:
            return int(self.group.random(ZR))
        return rng.getrandbits(self.order.bit_length() + 128) % self.order

    def _multiexp(self, bases: list, exps: list, identity):
        acc = identity
        for b, e in zip(bases, exps):
            e = int(e) % self.order
            if e:
                acc = acc * (b ** self._zr(e))
        return acc

    def identity1(self):
        return self.g1 ** self._zr(0)

    def identity2(self):
        return self.g2 ** self._zr(0)

    # -- algorithms ----------------------------------------------------------
    def setup(self) -> dict:
        """Sample the master secret key (all entries in Z_p)."""
        rand = self._rand
        return {
            "a_in": [rand(), rand()],
            "u": [[rand(), rand()] for _ in range(self.m2 + 1)],
            "z": rand(),
            "a_out": [rand(), rand()],
            "w": [[rand(), rand()] for _ in range(self.ell)],
        }

    def encrypt(self, msk: dict, x_enc: list, rng=None) -> list:
        """Encrypt ``[x]_1`` (``m`` elements of G1); returns ``l + 2`` elements of G1.

        *rng* (an object with ``getrandbits``) makes the randomness deterministic.
        """
        if len(x_enc) != self.m:
            raise FEValidationError("FH-IPE plaintext must have {} entries".format(self.m))
        p = self.order
        x_pad = list(x_enc) + [self.identity1()] * self.m
        s = self._rand(rng)
        as_ = [(a * s) % p for a in msk["a_in"]]
        uas = [(row[0] * as_[0] + row[1] * as_[1]) % p for row in msk["u"]]
        ct_in = [self.g1 ** self._zr(v) for v in as_]
        for j in range(self.m2):
            ct_in.append(x_pad[j] * (self.g1 ** self._zr(uas[j])))
        ct_in.append(self.g1 ** self._zr(msk["z"] + uas[self.m2]))
        # outer KeyGen of [ct_in]_1: (W'^T ct_in ; ct_in)
        head = [
            self._multiexp(ct_in, [row[c] for row in msk["w"]], self.identity1())
            for c in (0, 1)
        ]
        return head + ct_in

    def keygen(self, msk: dict, y_enc: list, rng=None) -> tuple:
        """Key for ``[y]_2`` (``m`` elements of G2); returns (``l + 2`` elements of G2, ``z_sum``).

        *rng* (an object with ``getrandbits``) makes the randomness deterministic.
        """
        if len(y_enc) != self.m:
            raise FEValidationError("FH-IPE key vector must have {} entries".format(self.m))
        p = self.order
        r = self._rand(rng)
        v = list(y_enc) + [self.identity2()] * self.m + [self.g2 ** self._zr(r)]
        sk_in = [
            self._multiexp(v, [(-row[c]) % p for row in msk["u"]], self.identity2())
            for c in (0, 1)
        ] + v
        # outer Enc of [sk_in]_2: (-A' r' ; sk_in + W' A' r')
        r2 = self._rand(rng)
        ar = [(a * r2) % p for a in msk["a_out"]]
        key = [self.g2 ** self._zr(-v_) for v_ in ar]
        for j in range(self.ell):
            row = msk["w"][j]
            key.append(sk_in[j] * (self.g2 ** self._zr(row[0] * ar[0] + row[1] * ar[1])))
        return key, (msk["z"] * r) % p

    def keygen_exponents(self, msk: dict, y: list) -> tuple:
        """Like :meth:`keygen` for a vector known in the clear, returning exponents.

        Returns ``(e, z_sum)`` with the key equal to ``[e]_2``; used to secret-share
        a key (threshold decryption).
        """
        if len(y) != self.m:
            raise FEValidationError("FH-IPE key vector must have {} entries".format(self.m))
        p = self.order
        r = self._rand()
        v = [int(t) % p for t in y] + [0] * self.m + [r]
        sk_in = [sum(-row[c] * v_j for row, v_j in zip(msk["u"], v)) % p for c in (0, 1)] + v
        r2 = self._rand()
        ar = [(a * r2) % p for a in msk["a_out"]]
        e = [(-v_) % p for v_ in ar]
        for j in range(self.ell):
            row = msk["w"][j]
            e.append((sk_in[j] + row[0] * ar[0] + row[1] * ar[1]) % p)
        return e, (msk["z"] * r) % p

    def decrypt(self, ct: list, key: list, z_sum_gt):
        """Return ``[<x, y>]_T`` from a ciphertext, a key and the key's ``[z_sum]_T``."""
        acc = self.group.init(GT, 1)
        for c, k in zip(ct, key):
            acc *= pair(c, k)
        return acc / z_sum_gt
