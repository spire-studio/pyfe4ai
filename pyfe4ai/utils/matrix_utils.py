"""Integer matrix/vector arithmetic under a modulus."""

from __future__ import annotations

import gmpy2 as gp


def matmul_mod(
    left: list[list[gp.mpz]], right: list[list[gp.mpz]], modulus: gp.mpz
) -> list[list[gp.mpz]]:
    """Multiply two matrices modulo a given modulus.

        Args:
            left: Left operand matrix or vector.
            right: Right operand matrix or vector.
            modulus: Modulus for the arithmetic operation.
    """
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    result = [[gp.mpz(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        row_i = result[i]
        for k in range(inner):
            left_ik = left[i][k]
            if left_ik == 0:
                continue
            right_k = right[k]
            for j in range(cols):
                row_i[j] += left_ik * right_k[j]
        for j in range(cols):
            row_i[j] = row_i[j] % modulus
    return result


def transpose(matrix: list[list[gp.mpz]]) -> list[list[gp.mpz]]:
    return [list(col) for col in zip(*matrix)]


def matvec_mod(
    matrix: list[list[gp.mpz]], vector: list[gp.mpz], modulus: gp.mpz
) -> list[gp.mpz]:
    """Multiply a matrix by a vector modulo a given modulus.

        Args:
            matrix: Input matrix.
            vector: Input vector.
            modulus: Modulus for the arithmetic operation.
    """
    result = []
    for row in matrix:
        acc = gp.mpz(0)
        for a, b in zip(row, vector):
            acc += a * b
        result.append(acc % modulus)
    return result


def vecdot_mod(left: list[gp.mpz], right: list[gp.mpz], modulus: gp.mpz) -> gp.mpz:
    """Compute the dot product of two vectors modulo a given modulus.

        Args:
            left: Left operand matrix or vector.
            right: Right operand matrix or vector.
            modulus: Modulus for the arithmetic operation.
    """
    acc = gp.mpz(0)
    for a, b in zip(left, right):
        acc += a * b
    return acc % modulus


def matrix_to_digits(matrix: list[list[gp.mpz]]) -> list[list[str]]:
    return [[gp.digits(v) for v in row] for row in matrix]


def digits_to_matrix(matrix: list[list[str]]) -> list[list[gp.mpz]]:
    return [[gp.mpz(v) for v in row] for row in matrix]


def vector_to_digits(vector: list[gp.mpz]) -> list[str]:
    return [gp.digits(v) for v in vector]


def digits_to_vector(vector: list[str]) -> list[gp.mpz]:
    return [gp.mpz(v) for v in vector]
