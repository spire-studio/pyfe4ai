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

from pyfe4ai.utils.pairing_backend import ZR, PairingGroup, multi_pair, multiexp, pair, pow_many
from pyfe4ai.utils.pairing_backend import check_asymmetric  # noqa: F401  (re-export)
from pyfe4ai.utils.exceptions import FEValidationError

class FHIPE:
    """ACFGU18 Fig. 12 (n = 1, k = 1) with LV16 padding, for vectors of length *m*."""

    def __init__(self, group: PairingGroup, g1, g2, m: int) -> None:
        self.group = group
        self.g1 = g1
        self.g2 = g2
        self._gt = None
        self.m = m
        self.m2 = 2 * m
        self.ell = self.m2 + 3
        self.order = int(group.order())

    @property
    def gt(self):
        """``e(g1, g2)``, computed on first use (encryption does not need it)."""
        if self._gt is None:
            self._gt = pair(self.g1, self.g2)
        return self._gt

    # -- helpers ---------------------------------------------------------------
    def _zr(self, value: int):
        return self.group.init(ZR, int(value) % self.order)

    def _rand(self, rng=None) -> int:
        if rng is None:
            return int(self.group.random(ZR))
        return rng.getrandbits(self.order.bit_length() + 128) % self.order

    def _multiexp(self, bases: list, exps: list):
        """prod_j bases[j]^exps[j]: one multi-scalar multiplication on BLS12-381."""
        return multiexp(bases, exps, self.group)

    def _powers(self, base, exps: list) -> list:
        """[base^e for e in exps] with one precomputation for *base* on BLS12-381."""
        return pow_many(base, exps, self.group)

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
        powers = self._powers(self.g1, as_ + uas[: self.m2] + [msk["z"] + uas[self.m2]])
        ct_in = powers[:2]
        for j in range(self.m2):
            ct_in.append(x_pad[j] * powers[2 + j])
        ct_in.append(powers[-1])
        # outer KeyGen of [ct_in]_1: (W'^T ct_in ; ct_in)
        head = [self._multiexp(ct_in, [row[c] for row in msk["w"]]) for c in (0, 1)]
        return head + ct_in

    def keygen(self, msk: dict, y_enc: list, rng=None, r=None) -> tuple:
        """Key for ``[y]_2`` (``m`` elements of G2); returns (``l + 2`` elements of G2, ``z_sum``).

        *rng* (an object with ``getrandbits``) makes the randomness deterministic;
        *r* fixes the inner-layer randomness (shared by all slots of an
        ACFGU18 Fig. 12 key, whose pad term is then ``sum_i z_i r``).
        """
        if len(y_enc) != self.m:
            raise FEValidationError("FH-IPE key vector must have {} entries".format(self.m))
        p = self.order
        r = self._rand(rng) if r is None else int(r) % p
        # outer Enc of [sk_in]_2: (-A' r' ; sk_in + W' A' r')
        r2 = self._rand(rng)
        ar = [(a * r2) % p for a in msk["a_out"]]
        powers = self._powers(
            self.g2, [r] + [-v_ for v_ in ar] + [row[0] * ar[0] + row[1] * ar[1] for row in msk["w"]]
        )
        v = list(y_enc) + [self.identity2()] * self.m + [powers[0]]
        sk_in = [self._multiexp(v, [(-row[c]) % p for row in msk["u"]]) for c in (0, 1)] + v
        key = powers[1:3]
        for j in range(self.ell):
            key.append(sk_in[j] * powers[3 + j])
        return key, (msk["z"] * r) % p

    def keygen_exponents(self, msk: dict, y: list, r=None) -> tuple:
        """Like :meth:`keygen` for a vector known in the clear, returning exponents.

        Returns ``(e, z_sum)`` with the key equal to ``[e]_2``; used to secret-share
        a key (threshold decryption).
        """
        if len(y) != self.m:
            raise FEValidationError("FH-IPE key vector must have {} entries".format(self.m))
        p = self.order
        r = self._rand() if r is None else int(r) % p
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
        if len(ct) != self.ell + 2 or len(key) != self.ell + 2:
            raise FEValidationError("FH-IPE ciphertext and key must have {} elements".format(self.ell + 2))
        return multi_pair(ct, key) / z_sum_gt
