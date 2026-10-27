"""The Python implementation of the backend operations: the reference.

:class:`PythonBackend` and :class:`PythonRing` implement every operation of
:mod:`pyfe4ai.backend.interface` in Python; the native implementation
(:mod:`pyfe4ai.backend.native_impl`) subclasses them and must give the same
outputs. Where the reference algorithm is a primitive of its own -- the NTT
and polynomial product of :mod:`~pyfe4ai.primitives.lattice.ring_lwe_utils`,
the samplers of :mod:`~pyfe4ai.primitives.sampling`, the hashes of
:mod:`~pyfe4ai.primitives.hashing`, the ristretto255 group of
:mod:`~pyfe4ai.primitives.groups.ristretto255` -- the operation calls it;
the kernels of the schemes (encryption and decryption of Ring-LWE tensors,
label terms, the threshold dealing and partial decryptions) are written out
here, with the Python sources of randomness of their callers.
"""

from __future__ import annotations

import math

import gmpy2 as gp

from pyfe4ai.backend.interface import Backend
from pyfe4ai.backend.interface import ReferenceKey
from pyfe4ai.backend.interface import Ring
from pyfe4ai.primitives.sampling import discrete_gaussian_matrix  # the reference's Gaussian rows (tests replace it here)


class PythonRing(Ring):
    """``Z_q[x] / (x^ring_n + 1)`` in Python."""

    native = False

    # -- values mod q

    def hash_to_zq(self, material: bytes, count: int, mode: str = "shake256-wide") -> list:
        from pyfe4ai.primitives.hashing import hash_to_zq_values

        return hash_to_zq_values(material, self.q, count, mode)

    def lincomb(self, rows: list, coeffs: list) -> list:
        acc = [gp.mpz(0)] * self.ring_n
        for row, c in zip(rows, coeffs):
            acc = [a + c * gp.mpz(v) for a, v in zip(acc, row)]
        q = self.q
        return [gp.digits(v % q) for v in acc]

    def reference(self) -> "PythonRing":
        return self

    def label_terms(self, base: list, materials: list, weights: list, eta: int, mode: str,
                    binary: bool = False) -> list:
        from pyfe4ai.core.formats.binary import encode_values

        q, ring_n = self.q, self.ring_n
        base = [gp.mpz(v) for v in base]

        def pad_term(row_materials) -> list:
            z = list(base)
            for material, ys in zip(row_materials, weights):
                flat = self.hash_to_zq(material, eta * ring_n, mode)
                pad = [flat[r * ring_n:(r + 1) * ring_n] for r in range(eta)]
                for y_row, pad_row in zip(ys, pad):
                    z = [acc + int(y_row) * h for acc, h in zip(z, pad_row)]
            if binary:
                return encode_values(z, q)
            return [gp.digits(v % q) for v in z]

        return [pad_term(m) for m in materials]

    # -- Ring-LWE encryption

    def samples_natively(self, *sigmas, sampler: str) -> bool:
        return False

    def encrypt_rows(self, key, reference, w_rows: list, eta: int, sigmas: tuple, sampler: str, binary: bool):
        ct0, ct1 = self._encrypt_reference(reference(), w_rows, eta, sigmas, sampler)
        return self._ciphertext(ct0, ct1, binary)

    def _ciphertext(self, ct0: list, ct1: list, binary: bool):
        """Rows ``ct0`` and ``ct1`` (integers in ``[0, q)``) as their format writes them."""
        if binary:
            from pyfe4ai.core.formats.binary import encode_binary_ciphertext

            return encode_binary_ciphertext(ct0, ct1, self.q, self.ring_n)
        return [[gp.digits(v) for v in row] for row in ct0], [gp.digits(v) for v in ct1]

    def _encrypt_reference(self, key: "ReferenceKey", w_rows: list, eta: int, sigmas: tuple, sampler: str) -> tuple:
        """``(ct0, ct1)`` of :meth:`encrypt_rows` as integers: ``r`` is
        transformed once and multiplied with the cached ``NTT(pk_row)`` and
        ``NTT(a)``, the same ciphertext as ``poly_mul(pk_row, r)`` and
        ``poly_mul(a, r)`` with the same samples."""
        from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_add, poly_mul

        q, ring_n, delta = self.q, self.ring_n, key.delta
        sigma2, sigma3 = sigmas
        ring = key.ntt
        r = discrete_gaussian_matrix(1, ring_n, sigma2, sampler=sampler)[0]
        noise = discrete_gaussian_matrix(eta, ring_n, sigma3, sampler=sampler)
        r_hat = None if ring is None else ring.forward(r)

        def times_r(poly, poly_hat):
            if ring is None:
                return poly_mul(poly, r, q)
            return ring.inverse(ring.mul(poly_hat, r_hat))

        pk_hat = key.pk_hat or [None] * len(key.pk)
        ct0 = []
        for i, w_row in enumerate(w_rows):
            prod = times_r(key.pk[i], pk_hat[i])
            ct0.append([(c + e + delta * w) % q for c, e, w in zip(prod, noise[i], w_row)])
        e = discrete_gaussian_matrix(1, ring_n, sigma2, sampler=sampler)[0]
        return ct0, poly_add(times_r(key.a, key.a_hat), e, q)

    def _padded_row(self, key: "ReferenceKey", part: list, pad: list) -> list:
        """``x + u + H mod q`` of one row (``eta = 1``), *x* zero-extended to ``ring_n``."""
        q = self.q
        padded = [gp.mpz(int(v)) for v in part] + [gp.mpz(0)] * (self.ring_n - len(part))
        return [(x + u + e) % q for x, u, e in zip(padded, key.u[0], pad)]

    def encrypt_tensor(self, key, reference, ints, materials: list, mode: str, sigmas: tuple, sampler: str,
                       binary: bool) -> list:
        ring_n = self.ring_n
        ref = reference()
        flat, out = ints.tolist(), []
        for j, material in enumerate(materials):
            part = flat[j * ring_n:(j + 1) * ring_n]
            row = self._padded_row(ref, part, self.hash_to_zq(material, ring_n, mode))
            ct = self._ciphertext(*self._encrypt_reference(ref, [row], 1, sigmas, sampler), binary)
            out.append((ct, len(part)) if binary else (ct[0], ct[1], len(part)))
        return out

    def precompute_tensor(self, key, reference, materials: list, mode: str, sigmas: tuple, sampler: str) -> tuple:
        ref = reference()
        rows = []
        for material in materials:  # x = 0: the stored ct0 is pk r + e_0 + Delta (u + H)
            row = self._padded_row(ref, [], self.hash_to_zq(material, self.ring_n, mode))
            ct0, ct1 = self._encrypt_reference(ref, [row], 1, sigmas, sampler)
            rows.append((ct0[0], ct1))
        return None, rows

    def encrypt_precomputed(self, native, rows, first: int, ints, delta, binary: bool) -> list:
        if native is not None:  # the handle of a native precomputation: its kernel
            return native.encrypt(first, ints, binary)
        q, ring_n = self.q, self.ring_n
        flat, cts = ints.tolist(), []
        for j in range(math.ceil(len(flat) / ring_n)):
            part = flat[j * ring_n:(j + 1) * ring_n]
            p_row, ct1 = rows[first + j]
            rows[first + j] = None
            ct0 = [(p + delta * x) % q for p, x in zip(p_row, part)] + p_row[len(part):]
            ct = self._ciphertext([ct0], ct1, binary)
            cts.append((ct, len(part)) if binary else (ct[0], ct[1], len(part)))
        return cts


class PythonBackend(Backend):
    """The backend operations in Python (the reference)."""

    name = "python"
    batched_layers = False

    def __init__(self) -> None:
        self._rings: dict = {}

    def ring(self, q, ring_n: int) -> PythonRing:
        key = (int(q), int(ring_n))
        ring = self._rings.get(key)
        if ring is None:
            ring = self._rings.setdefault(key, PythonRing(q, ring_n))
        return ring

    # -- lattices and sampling

    def ntt(self, a: list, q, root, inverse: bool = False) -> list:
        from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu

        if inverse:
            return rlu._ntt_inverse_reference(a, q, root)
        return rlu._ntt_forward_reference(a, q, root)

    def poly_mul(self, a: list, b: list, q) -> list:
        from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu

        return rlu._poly_mul_reference(a, b, q, rlu._get_ntt_root(q, len(a)))

    def gaussian_bigint(self, sigma, count: int, rng, prec: int) -> list:
        from pyfe4ai.primitives import sampling

        return sampling._gaussian_bigint_reference(sigma, count, rng, prec)

    def gaussian_facct(self, sigma, count: int, rng) -> list:
        from pyfe4ai.primitives import sampling

        return sampling._gaussian_facct_reference(sigma, count, rng)

    # -- threshold key sharings over Z_q

    def zq_kernels(self, q) -> bool:
        return False

    def replicated_deal(self, q, values, n_random: int, binary: bool):
        return None  # the caller's reference steps (share_replicated)

    def zq_sum(self, q, count: int, rows, materials: list, subtract: bool = False, binary: bool = False,
               reference: bool = True):
        if not reference:
            return None  # the caller's reference steps
        from pyfe4ai.core.formats.binary import decode_values, encode_values

        if binary:  # rows of words: their sum, as words
            total = [0] * count
            for leaf in rows:
                total = [a + b for a, b in zip(total, decode_values(leaf, q, count))]
            return encode_values(total, q)
        from pyfe4ai.core.exceptions import FESchemeError
        from pyfe4ai.primitives.hashing import hash_to_zq_values

        q = gp.mpz(q)
        acc = [gp.mpz(0)] * count
        for row in rows:
            if len(row) != count:
                raise FESchemeError("key pieces of different layouts")
            acc = [a + gp.mpz(b) for a, b in zip(acc, row)]
        for material in materials:
            values = hash_to_zq_values(material, q, count)
            acc = [a - b for a, b in zip(acc, values)] if subtract else [a + b for a, b in zip(acc, values)]
        return [gp.digits(a % q) for a in acc]

    # -- ristretto255 (elements as points in extended coordinates)

    def r255_add(self, a, b):
        from pyfe4ai.primitives.groups import ec_group as eg

        return eg.r255.add(eg._t(a), eg._t(b))

    def r255_sub(self, a, b):
        from pyfe4ai.primitives.groups import ec_group as eg

        return eg.r255.sub(eg._t(a), eg._t(b))

    def r255_neg(self, a):
        from pyfe4ai.primitives.groups import ec_group as eg

        return eg.r255.neg(eg._t(a))

    def r255_mul(self, point, k: int):
        from pyfe4ai.primitives.groups import ec_group as eg

        return eg.r255.mul(eg._t(point), k)

    def r255_mul_base(self, k: int):
        from pyfe4ai.primitives.groups import ristretto255 as r255

        return r255.mul_base(k)

    def r255_decode(self, data: bytes):
        from pyfe4ai.primitives.groups import ristretto255 as r255

        return r255.decode(data)

    def r255_hash(self, msg: bytes, tag: bytes):
        from pyfe4ai.primitives.groups import ristretto255 as r255

        return r255.hash_to_group(msg, tag)

    def r255_dlog(self, point, bound: int):
        from pyfe4ai.primitives.groups import ec_group as eg

        return eg._dlog_reference(point, bound)

    # -- dense LWE and ristretto255 layers (the batched steps, see Backend.batched_layers)

    def dense_key(self, key_row: list, q):
        return key_row

    def label_dots(self, materials: list, key, q, dim: int, mode: str) -> list:
        import operator

        from pyfe4ai.primitives.hashing import hash_to_zq_values

        return [gp.mpz(sum(map(operator.mul, map(gp.mpz, key), hash_to_zq_values(m, q, dim, mode)))) % q
                for m in materials]

    def r255_mcfe_encrypt(self, msgs: list, dst: bytes, values: list, s0: int, s1: int) -> list:
        from pyfe4ai.primitives.groups import ec_group as eg

        out = []
        for msg, x in zip(msgs, values):
            u = [self.r255_hash(bytes([k]) + msg, dst) for k in (0, 1)]
            out.append(eg.to_hex(self.r255_add(self.r255_mul_base(x),
                                               self.r255_add(self.r255_mul(u[0], s0), self.r255_mul(u[1], s1)))))
        return out

    def r255_mcfe_decrypt(self, cts: list, ys: list, d0: int, d1: int, msgs: list, dst: bytes, bound: int):
        import numpy as np

        values, mask = [], []
        for e, msg in enumerate(msgs):
            acc = None
            for i, (col, y) in enumerate(zip(cts, ys)):
                try:
                    point = self.r255_decode(bytes.fromhex(col[e]))
                except (TypeError, ValueError) as exc:
                    raise ValueError("malformed ciphertext", i) from exc
                term = self.r255_mul(point, y)
                acc = term if acc is None else self.r255_add(acc, term)
            u = [self.r255_hash(bytes([k]) + msg, dst) for k in (0, 1)]
            acc = self.r255_sub(acc, self.r255_add(self.r255_mul(u[0], d0), self.r255_mul(u[1], d1)))
            x = self.r255_dlog(acc, bound)
            values.append(0 if x is None else x)
            mask.append(x is None)
        data = np.array(values, dtype=np.int64).tobytes()
        return data, (np.array(mask, dtype=np.uint8).tobytes() if any(mask) else None)

    # -- BLS12-381

    def bls12_381(self):
        from pyfe4ai.backend import native_impl

        return native_impl.bls12_381_kernels()
