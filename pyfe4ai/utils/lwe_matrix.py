"""Public LWE matrices expanded from a seed, and exact products with them.

The ABDP15 LWE IPFE (``sife/lwe``, ``mife/lwe``, ``mife/lwe_threshold``)
publishes a uniform ``A in Z_q^{m x n}`` with ``m ~ n log2 q`` rows -- about
10^8 entries at 128-bit parameters, too many for a JSON parameter file or a
pure-Python loop. Here ``A`` is published as a 32-byte seed and expanded with
SHAKE-128 (as in FrodoKEM), and the products the scheme needs are computed
exactly with numpy:

* entries are held as ``L = ceil(bits(q) / 31)`` limbs of 31 bits in float64
  arrays; a sum of fewer than ``MAX_ROWS = 2^22`` limbs stays below 2^53 and
  is exact in double precision, so ``r^T A`` for a binary ``r`` (encryption)
  is one BLAS product per limb;
* ``A S mod q`` (key generation) multiplies 16-bit pieces of ``A`` and ``S``,
  whose products summed over fewer than 2^21 columns also stay below 2^53.

``matrix_mode`` trades memory for time:

``"memory"``  ``A`` is expanded once and kept, ``8 L`` bytes per entry (about
              2.4 GB for ``n = 1024``, ``m = 96000`` and a 92-bit ``q``); an
              encryption is one pass over it (about 0.1 s at that size).
``"stream"``  ``A`` is expanded again, chunk by chunk, for every product:
              little memory, but every product pays the expansion (seconds
              at that size); a batch of encryptions shares one expansion.

Expansion format (``EXPANSION_DOMAIN``): rows are taken in units of
``EXPANSION_ROWS``. Unit ``u`` reads ``4 L`` bytes per entry, row-major, from
SHAKE-128 of ``"pyfe4ai/lwe/A/v1|<seed>|<cols>|<q>|<u>"`` as little-endian
32-bit words, masks each to 31 bits (the last to the remaining bits of
``q``) and takes ``sum_j limb_j 2^(31 j)``, uniform below ``2^bits(q)``.
Entries ``>= q`` are replaced, in order, by candidates read the same way from
that string extended with ``"|retry|<c>"``, ``c = 1, 2, ...``.
"""

from __future__ import annotations

import hashlib
import os
import weakref

import numpy as np

from pyfe4ai.utils.exceptions import FEValidationError

LIMB_BITS = 31
EXPANSION_ROWS = 64
EXPANSION_DOMAIN = "pyfe4ai/lwe/A/v1"
#: Accepted values of ``matrix_mode``.
MATRIX_MODES = ("memory", "stream")
#: Sums of 31-bit limbs over fewer rows than this are exact in float64.
MAX_ROWS = 1 << (53 - LIMB_BITS)
#: Rows expanded at a time in ``"stream"`` mode and for ``A S``.
DEFAULT_CHUNK_ROWS = 4096

_LIMB_MASK = (1 << LIMB_BITS) - 1
_PIECE_BITS = 16
_PIECE_MASK = (1 << _PIECE_BITS) - 1
#: Products of 16-bit pieces summed over fewer columns than this are exact.
_MAX_COLS = 1 << (53 - 2 * _PIECE_BITS)


def limb_count(q) -> int:
    """Number of 31-bit limbs of an integer below ``2^bits(q)``."""
    return -(-int(q).bit_length() // LIMB_BITS)


def check_matrix_mode(mode: str) -> str:
    """Return *mode* if it is one of :data:`MATRIX_MODES`."""
    if mode not in MATRIX_MODES:
        raise FEValidationError(
            "matrix_mode must be one of {}, got {!r}".format(MATRIX_MODES, mode)
        )
    return mode


def physical_memory_bytes() -> int | None:
    """Physical memory of this machine, or ``None`` where the OS does not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _check_shape(rows: int, cols: int) -> None:
    if rows < 1 or cols < 1:
        raise FEValidationError("a matrix needs at least one row and one column")
    if rows >= MAX_ROWS:
        raise FEValidationError(
            "{} rows: sums of 31-bit limbs are exact in float64 for fewer than {} rows".format(
                rows, MAX_ROWS
            )
        )
    if cols >= _MAX_COLS:
        raise FEValidationError("at most {} columns".format(_MAX_COLS - 1))


def _limbs_of(values: np.ndarray, n_limbs: int) -> np.ndarray:
    """``(L, ...)`` float64 limbs of an object array of integers in ``[0, 2^(31 L))``."""
    return np.stack(
        [((values >> (LIMB_BITS * j)) & _LIMB_MASK).astype(np.int64) for j in range(n_limbs)]
    ).astype(np.float64)


def _ints_mod(parts: np.ndarray, weight_bits: int, q: int) -> list:
    """``sum_j parts[j] 2^(weight_bits j) mod q`` as nested lists of ints.

    *parts* hold exact integers below 2^63 (int64, or float64 below 2^53).
    """
    acc = parts[0].astype(np.int64).astype(object)
    for j in range(1, len(parts)):
        acc = acc + (parts[j].astype(np.int64).astype(object) << (weight_bits * j))
    return (acc % q).tolist()


def _binary_combinations(blocks, r, q: int, n_limbs: int, cols: int) -> list:
    """``r A mod q`` for the rows of a 0/1 matrix *r*, ``A`` given by row blocks."""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2 or not np.all((r == 0) | (r == 1)):
        raise FEValidationError("expected a 2-D array of 0/1 vectors")
    acc = np.zeros((n_limbs, r.shape[0], cols))
    for start, stop, limbs in blocks:
        r_block = r[:, start:stop]
        for j in range(n_limbs):
            acc[j] += r_block @ limbs[j]
    return _ints_mod(acc, LIMB_BITS, q)


def _pieces(limbs: np.ndarray, n_pieces: int) -> list:
    """16-bit pieces (float64) of integers given as int64 31-bit limbs ``(L, ...)``."""
    out = []
    for w in range(n_pieces):
        bit = _PIECE_BITS * w
        j, shift = divmod(bit, LIMB_BITS)
        piece = limbs[j] >> shift
        if LIMB_BITS - shift < _PIECE_BITS and j + 1 < len(limbs):
            piece = piece | (limbs[j + 1] << (LIMB_BITS - shift))
        out.append((piece & _PIECE_MASK).astype(np.float64))
    return out


def _times(blocks, s, q: int, n_limbs: int, cols: int) -> list:
    """``A S mod q`` for an integer matrix *s* (``cols x k``, entries in ``[0, q)``)."""
    s_obj = np.array(s, dtype=object)
    if s_obj.ndim != 2 or s_obj.shape[0] != cols:
        raise FEValidationError("S must have {} rows".format(cols))
    n_pieces = -(-q.bit_length() // _PIECE_BITS)
    s_pieces = [((s_obj >> (_PIECE_BITS * t)) & _PIECE_MASK).astype(np.float64) for t in range(n_pieces)]
    out = []
    for _, _, limbs in blocks:
        a_pieces = _pieces(limbs.astype(np.int64), n_pieces)
        groups = np.zeros((2 * n_pieces - 1, limbs.shape[1], s_obj.shape[1]), dtype=np.int64)
        for a, a_piece in enumerate(a_pieces):
            for t, s_piece in enumerate(s_pieces):
                groups[a + t] += (a_piece @ s_piece).astype(np.int64)  # < 2^53 each
        out.extend(_ints_mod(groups, _PIECE_BITS, q))
    return out


class LimbMatrix:
    """An explicit matrix over ``Z_q`` (e.g. a public key ``A S + E``) held as limbs."""

    def __init__(self, entries, q) -> None:
        self.q = int(q)
        values = np.array(entries, dtype=object)
        if values.ndim != 2:
            raise FEValidationError("expected a 2-D matrix")
        self.rows, self.cols = values.shape
        _check_shape(self.rows, self.cols)
        self.n_limbs = limb_count(self.q)
        self._limbs = _limbs_of(values, self.n_limbs)

    def binary_combinations(self, r) -> list:
        """``[r_b^T M mod q]_b`` for the rows ``r_b`` of a 0/1 matrix *r*."""
        return _binary_combinations(
            [(0, self.rows, self._limbs)], r, self.q, self.n_limbs, self.cols
        )


class SeededMatrix:
    """Uniform ``A in Z_q^{rows x cols}`` expanded from a hex *seed* (see the module docstring).

    Args:
        seed: Hex string (the public seed).
        rows, cols: Shape of ``A``.
        q: Modulus.
        mode: ``"memory"`` (expand once, keep) or ``"stream"`` (expand per product).
        chunk_rows: Rows per chunk in ``"stream"`` mode and for :meth:`times`.
    """

    _shared = weakref.WeakValueDictionary()

    def __init__(self, seed: str, rows: int, cols: int, q, mode: str = "memory",
                 chunk_rows: int = DEFAULT_CHUNK_ROWS) -> None:
        self.mode = check_matrix_mode(mode)
        _check_shape(rows, cols)
        bytes.fromhex(seed)  # validates the seed
        self.seed, self.rows, self.cols, self.q = seed, int(rows), int(cols), int(q)
        self.n_limbs = limb_count(self.q)
        top_bits = self.q.bit_length() - LIMB_BITS * (self.n_limbs - 1)
        self._top_mask = (1 << top_bits) - 1
        self._q_limbs = [(self.q >> (LIMB_BITS * j)) & _LIMB_MASK for j in range(self.n_limbs)]
        units = -(-max(1, int(chunk_rows)) // EXPANSION_ROWS)
        self.chunk_rows = units * EXPANSION_ROWS
        self._full = None

    @classmethod
    def shared(cls, seed: str, rows: int, cols: int, q, mode: str = "memory") -> "SeededMatrix":
        """One instance per ``(seed, shape, q, mode)`` and process, so that several
        encryptors in one process (e.g. simulated clients) expand ``A`` once."""
        key = (seed, int(rows), int(cols), int(q), mode)
        found = cls._shared.get(key)
        if found is None:
            found = cls(seed, rows, cols, q, mode)
            cls._shared[key] = found
        return found

    # -- expansion -----------------------------------------------------------

    def _candidates(self, material: str, count: int) -> np.ndarray:
        data = hashlib.shake_128(material.encode("utf-8")).digest(4 * self.n_limbs * count)
        words = np.frombuffer(data, dtype="<u4").reshape(count, self.n_limbs).astype(np.int64)
        words &= _LIMB_MASK
        words[:, -1] &= self._top_mask
        return words

    def _at_least_q(self, limbs: np.ndarray) -> np.ndarray:
        ge = np.zeros(len(limbs), dtype=bool)
        eq = np.ones(len(limbs), dtype=bool)
        for j in reversed(range(self.n_limbs)):
            ge |= eq & (limbs[:, j] > self._q_limbs[j])
            eq &= limbs[:, j] == self._q_limbs[j]
        return ge | eq

    def _expand_unit(self, unit: int) -> np.ndarray:
        """int64 limbs ``(L, n_rows, cols)`` of expansion unit *unit*."""
        n_rows = min(EXPANSION_ROWS, self.rows - unit * EXPANSION_ROWS)
        count = n_rows * self.cols
        base = "|".join([EXPANSION_DOMAIN, self.seed, str(self.cols), str(self.q), str(unit)])
        limbs = self._candidates(base, count)
        bad = np.nonzero(self._at_least_q(limbs))[0]
        retry = 0
        while len(bad):
            retry += 1
            fresh = self._candidates(base + "|retry|" + str(retry), len(bad))
            limbs[bad] = fresh
            bad = bad[self._at_least_q(fresh)]
        return limbs.reshape(n_rows, self.cols, self.n_limbs).transpose(2, 0, 1)

    def _expand_rows(self, start: int, stop: int) -> np.ndarray:
        """float64 limbs ``(L, stop - start, cols)``; *start* is a multiple of EXPANSION_ROWS."""
        out = np.empty((self.n_limbs, stop - start, self.cols))
        for row in range(start, stop, EXPANSION_ROWS):
            unit = self._expand_unit(row // EXPANSION_ROWS)
            out[:, row - start:row - start + unit.shape[1]] = unit
        return out

    def _expand_all(self) -> np.ndarray:
        """``A`` as float64 limbs, expanded once (``"memory"`` mode)."""
        if self._full is None:
            needed = 8 * self.n_limbs * self.rows * self.cols
            available = physical_memory_bytes()
            if available is not None and needed > available:
                raise FEValidationError(
                    "matrix_mode='memory' needs {:.1f} GB for the expanded matrix, more than "
                    "the {:.1f} GB of physical memory; use matrix_mode='stream'".format(
                        needed / 2 ** 30, available / 2 ** 30
                    )
                )
            self._full = self._expand_rows(0, self.rows)
        return self._full

    def _blocks(self):
        if self.mode == "memory":
            full = self._expand_all()
            for start in range(0, self.rows, self.chunk_rows):
                stop = min(self.rows, start + self.chunk_rows)
                yield start, stop, full[:, start:stop]
        else:
            for start in range(0, self.rows, self.chunk_rows):
                stop = min(self.rows, start + self.chunk_rows)
                yield start, stop, self._expand_rows(start, stop)

    # -- products ------------------------------------------------------------

    def binary_combinations(self, r) -> list:
        """``[r_b^T A mod q]_b`` for the rows ``r_b`` of a 0/1 matrix *r* (``B x rows``)."""
        if self.mode == "memory":
            blocks = [(0, self.rows, self._expand_all())]
        else:
            blocks = self._blocks()
        return _binary_combinations(blocks, r, self.q, self.n_limbs, self.cols)

    def times(self, s) -> list:
        """``A S mod q`` (``rows x k``) for an integer matrix *s* (``cols x k``)."""
        return _times(self._blocks(), s, self.q, self.n_limbs, self.cols)

    def entries(self) -> list:
        """All entries as Python integers (for tests and small matrices)."""
        out = []
        for _, _, limbs in self._blocks():
            out.extend(_ints_mod(limbs, LIMB_BITS, self.q))
        return out
