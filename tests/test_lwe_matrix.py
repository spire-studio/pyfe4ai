"""Seeded public LWE matrices and their exact products in numpy."""

import hashlib
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.utils import lwe_matrix as lm
from pyfe4ai.utils.exceptions import FEValidationError

SEED = "00" * 31 + "2a"


def reference_entries(seed: str, rows: int, cols: int, q: int) -> list:
    """The expansion format, written out with Python integers."""
    k = q.bit_length()
    n_limbs = -(-k // lm.LIMB_BITS)
    top_bits = k - lm.LIMB_BITS * (n_limbs - 1)

    def candidates(material: str, count: int) -> list:
        data = hashlib.shake_128(material.encode("utf-8")).digest(4 * n_limbs * count)
        out = []
        for i in range(count):
            value = 0
            for j in range(n_limbs):
                limb = int.from_bytes(data[4 * (n_limbs * i + j):4 * (n_limbs * i + j + 1)], "little")
                bits = top_bits if j == n_limbs - 1 else lm.LIMB_BITS
                value |= (limb & ((1 << bits) - 1)) << (lm.LIMB_BITS * j)
            out.append(value)
        return out

    entries = []
    for unit, start in enumerate(range(0, rows, lm.EXPANSION_ROWS)):
        n_rows = min(lm.EXPANSION_ROWS, rows - start)
        base = "|".join([lm.EXPANSION_DOMAIN, seed, str(cols), str(q), str(unit)])
        flat = candidates(base, n_rows * cols)
        retry = 0
        while any(v >= q for v in flat):
            retry += 1
            bad = [i for i, v in enumerate(flat) if v >= q]
            fresh = candidates(base + "|retry|" + str(retry), len(bad))
            for i, v in zip(bad, fresh):
                flat[i] = v
        entries += [flat[r * cols:(r + 1) * cols] for r in range(n_rows)]
    return entries


@pytest.mark.parametrize("q", [
    int(gp.prev_prime(2 ** 62)),                 # two limbs, almost no rejection
    int(gp.next_prime(2 ** 61 + 12345)),         # about half the candidates rejected
    int(gp.next_prime(2 ** 92 + 2 ** 91)),       # three limbs, a quarter rejected
])
def test_expansion_follows_the_format(q):
    matrix = lm.SeededMatrix(SEED, 150, 5, q)
    assert matrix.entries() == reference_entries(SEED, 150, 5, q)


def test_modes_and_block_sizes_give_the_same_matrix():
    q = int(gp.next_prime(2 ** 70 + 3))
    memory = lm.SeededMatrix(SEED, 300, 7, q, mode="memory")
    stream = lm.SeededMatrix(SEED, 300, 7, q, mode="stream", chunk_rows=64 * 3)
    assert memory.entries() == stream.entries()
    assert lm.SeededMatrix("11" * 32, 300, 7, q).entries() != memory.entries()


def _python_binary_combinations(entries, r_rows, q):
    cols = len(entries[0])
    return [
        [sum(e[c] for e, bit in zip(entries, r) if bit) % q for c in range(cols)] for r in r_rows
    ]


@pytest.mark.parametrize("mode", lm.MATRIX_MODES)
def test_binary_combinations_are_exact(mode):
    q = int(gp.next_prime(2 ** 92 - 2 ** 60))
    rng = random.Random(3)
    matrix = lm.SeededMatrix(SEED, 200, 9, q, mode=mode, chunk_rows=128)
    r_rows = [[rng.randrange(2) for _ in range(200)] for _ in range(4)]
    got = matrix.binary_combinations(np.array(r_rows))
    assert got == _python_binary_combinations(matrix.entries(), r_rows, q)


def test_explicit_limb_matrix_combinations_are_exact():
    q = int(gp.next_prime(2 ** 100))
    rng = random.Random(5)
    entries = [[rng.randrange(q) for _ in range(3)] for _ in range(80)]
    r_rows = [[rng.randrange(2) for _ in range(80)] for _ in range(2)]
    got = lm.LimbMatrix(entries, q).binary_combinations(np.array(r_rows))
    assert got == _python_binary_combinations(entries, r_rows, q)


@pytest.mark.parametrize("mode", lm.MATRIX_MODES)
def test_times_is_exact(mode):
    q = int(gp.next_prime(2 ** 92 + 7))
    rng = random.Random(7)
    matrix = lm.SeededMatrix(SEED, 130, 11, q, mode=mode, chunk_rows=64)
    s = [[rng.randrange(q) for _ in range(4)] for _ in range(11)]
    a = matrix.entries()
    expected = [[sum(a[i][k] * s[k][j] for k in range(11)) % q for j in range(4)] for i in range(130)]
    assert matrix.times(s) == expected


def test_too_many_rows_for_exact_sums_are_refused():
    with pytest.raises(FEValidationError, match="rows"):
        lm.SeededMatrix(SEED, 2 ** 22, 4, 101)


def test_unknown_mode_is_refused():
    with pytest.raises(FEValidationError, match="matrix_mode"):
        lm.SeededMatrix(SEED, 10, 4, 101, mode="disk")


def test_shared_instances_expand_once_per_process():
    q = int(gp.next_prime(2 ** 40))
    a = lm.SeededMatrix.shared(SEED, 64, 3, q, "memory")
    assert lm.SeededMatrix.shared(SEED, 64, 3, q, "memory") is a
    assert lm.SeededMatrix.shared(SEED, 64, 3, q, "stream") is not a


def test_memory_mode_refuses_a_matrix_larger_than_physical_memory():
    huge = lm.SeededMatrix(SEED, 4_000_000, 1_000_000, int(gp.next_prime(2 ** 100)), mode="memory")
    with pytest.raises(FEValidationError, match="stream"):
        huge.binary_combinations(np.zeros((1, 4_000_000)))
