"""Pairing-based helper functions shared across FH-IPE / FH-Multi-IPE / Quadratic schemes.

These were originally private helpers in ``sife/fh_ipe_pairing.py`` and
``mife/fh_multi_ipe_pairing.py``.  They are extracted here so that
cross-package imports reference ``utils`` rather than reaching into
another scheme family's private namespace.
"""

import math

import gmpy2 as gp

from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair


# ---------------------------------------------------------------------------
# Linear-algebra helpers (mod arithmetic on plain mpz matrices/vectors)
# ---------------------------------------------------------------------------

def transpose_mod(matrix: list[list[gp.mpz]]) -> list[list[gp.mpz]]:
    return [list(col) for col in zip(*matrix)]


def mat_vec_mod(
    matrix: list[list[gp.mpz]], vector: list[gp.mpz], modulus: gp.mpz
) -> list[gp.mpz]:
    """Multiply a matrix by a vector modulo a given modulus.

        Args:
            matrix: Input matrix.
            vector: Input vector.
            modulus: Modulus for the arithmetic operation.
    """
    ret = []
    for row in matrix:
        acc = gp.mpz(0)
        for a_i, b_i in zip(row, vector):
            acc = (acc + a_i * b_i) % modulus
        ret.append(acc)
    return ret


def vector_add_mod(lhs: list[gp.mpz], rhs: list[gp.mpz], modulus: gp.mpz) -> list[gp.mpz]:
    """Add two vectors element-wise modulo a given modulus.

        Args:
            lhs: Left-hand side vector.
            rhs: Right-hand side vector.
            modulus: Modulus for the arithmetic operation.
    """
    return [(a + b) % modulus for a, b in zip(lhs, rhs)]


def vector_scalar_mod(vector: list[gp.mpz], scalar: gp.mpz, modulus: gp.mpz) -> list[gp.mpz]:
    """Multiply a vector by a scalar modulo a given modulus.

        Args:
            vector: Input vector.
            scalar: Scalar multiplier.
            modulus: Modulus for the arithmetic operation.
    """
    return [(scalar * value) % modulus for value in vector]


# ---------------------------------------------------------------------------
# Pairing-group helpers
# ---------------------------------------------------------------------------

def random_nonzero_zr(group: PairingGroup):
    while True:
        value = group.random(ZR)
        if int(value) != 0:
            return value


def to_zr(group: PairingGroup, value: gp.mpz | int):
    """Convert an integer to a ZR element in the pairing group.

        Args:
            group: Pairing group instance.
            value: Input value.
    """
    return group.init(ZR, int(gp.mpz(value)))


def matrix_inverse_mod(
    matrix: list[list[gp.mpz]], modulus: gp.mpz
) -> tuple[list[list[gp.mpz]], gp.mpz]:
    """Compute the modular inverse of a matrix via Gaussian elimination.

        Args:
            matrix: Input matrix.
            modulus: Modulus for the arithmetic operation.
    """
    n = len(matrix)
    aug = []
    for i in range(n):
        row = [gp.mpz(v) % modulus for v in matrix[i]]
        row.extend(gp.mpz(1 if i == j else 0) for j in range(n))
        aug.append(row)

    det = gp.mpz(1)
    sign = 1
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] % modulus != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("matrix is not invertible modulo group order")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
            sign *= -1

        pivot_val = aug[col][col] % modulus
        det = (det * pivot_val) % modulus
        inv_pivot = gp.invert(pivot_val, modulus)
        aug[col] = [(v * inv_pivot) % modulus for v in aug[col]]

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] % modulus
            if factor == 0:
                continue
            aug[row] = [
                (aug[row][idx] - factor * aug[col][idx]) % modulus
                for idx in range(2 * n)
            ]

    if sign < 0:
        det = (-det) % modulus
    inverse = [row[n:] for row in aug]
    return inverse, det


def bounded_discrete_log_gt(group: PairingGroup, base, target, bound: int) -> int:
    """Solve a bounded discrete logarithm in GT via baby-step giant-step.

        Args:
            group: Pairing group instance.
            base: Base for modular exponentiation.
            target: Target group element.
            bound: Upper bound on absolute values.
    """
    if bound < 0:
        raise ValueError("bound must be non-negative")
    identity = group.init(GT, 1)
    if target == identity:
        return 0
    m = int(math.isqrt(bound) + 1)

    baby = {}
    cur = identity
    for j in range(m):
        baby[group.serialize(cur)] = j
        cur *= base

    order = gp.mpz(int(group.order()))
    factor = base ** to_zr(group, (order - (m % order)) % order)
    gamma = target
    for i in range(m + 1):
        key = group.serialize(gamma)
        if key in baby:
            cand = i * m + baby[key]
            if cand <= bound:
                return cand
        gamma *= factor

    base_inv = base ** to_zr(group, order - 1)
    baby = {}
    cur = identity
    for j in range(m):
        baby[group.serialize(cur)] = j
        cur *= base_inv

    factor = base_inv ** to_zr(group, (order - (m % order)) % order)
    gamma = target
    for i in range(m + 1):
        key = group.serialize(gamma)
        if key in baby:
            cand = -(i * m + baby[key])
            if -bound <= cand <= bound:
                return cand
        gamma *= factor
    raise ValueError("discrete log not found within configured bound")


# ---------------------------------------------------------------------------
# Group-element serialization helpers
# ---------------------------------------------------------------------------

def serialize_matrix_g(group: PairingGroup, matrix) -> list[list[str]]:
    """Serialize a matrix of group elements to strings.

        Args:
            group: Pairing group instance.
            matrix: Input matrix.
    """
    return [
        [group.serialize(value).decode("utf-8") for value in row]
        for row in matrix
    ]


def deserialize_matrix_g(group: PairingGroup, matrix: list[list[str]]):
    """Deserialize a matrix of strings back to group elements.

        Args:
            group: Pairing group instance.
            matrix: Input matrix.
    """
    return [
        [group.deserialize(value.encode("utf-8")) for value in row]
        for row in matrix
    ]
