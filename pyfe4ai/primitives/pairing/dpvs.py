"""Dual pairing vector spaces (DPVS) over an asymmetric pairing group.

A dual pair of bases is a pair of invertible matrices ``B, B*`` over ``Z_p``
with ``B B*^T = mu I``. Row ``i`` of ``B`` is the exponent vector of the basis
vector ``b_i = g1^(B_i)`` of ``G1^dim`` and row ``j`` of ``B*`` that of
``b*_j = g2^(B*_j)`` of ``G2^dim``, so that the coordinate-wise pairing

    e(b_i, b*_j) = prod_k e(b_ik, b*_jk) = e(g1, g2)^(mu delta_ij).

The module samples the pair, turns rows of a matrix into vectors of group
elements, combines vectors by scalars and pairs two vectors. It is shared by
the DOT18 multi-input scheme (``mu`` random, ``g_T = e(g1, g2)^mu`` public)
and the NPP22 schemes (``mu = 1``). Group elements are handled by
:mod:`pyfe4ai.primitives.pairing.pairing_backend`, so the functions run on
BLS12-381 and on charm groups alike; importing the module needs no backend.
"""

from collections.abc import Sequence

import gmpy2 as gp

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.primitives.pairing.pairing_backend import ZR, PairingGroup, multi_pair, multiexp, pow_many
from pyfe4ai.primitives.pairing.pairing_utils import matrix_inverse_mod, transpose_mod


def random_dual_bases(
    group: PairingGroup, dim: int, mu: gp.mpz, order: gp.mpz
) -> tuple[list[list[gp.mpz]], list[list[gp.mpz]]]:
    """Draw a uniformly random pair of dual orthogonal bases ``(B, B*)`` over ``Z_order``.

    ``B`` is drawn entry by entry from ``group.random(ZR)``, again in full while
    it is singular, and ``B* = mu (B^-1)^T``, so that ``B B*^T = mu I``. The
    DOT18 scheme draws ``mu`` at random (and publishes ``e(g1, g2)^mu``); the
    NPP22 schemes pass ``mu = 1``.

    Args:
        group: Pairing group instance.
        dim: Vector dimension.
        mu: Scalar of the dual bases (``B B*^T = mu I``), nonzero modulo *order*.
        order: Group order, the modulus of the exponents.

    Returns:
        The two ``dim x dim`` matrices of ``mpz``, row by row.

    Raises:
        FEValidationError: For ``mu = 0`` modulo *order*, which makes ``B*`` the zero matrix.
    """
    if mu % order == 0:
        raise FEValidationError("mu must be nonzero modulo the group order: B* would be the zero matrix")
    while True:
        b = [
            [gp.mpz(int(group.random(ZR))) for _ in range(dim)]
            for _ in range(dim)
        ]
        try:
            b_inv, _ = matrix_inverse_mod(b, order)
            break
        except ValueError:
            continue
    b_star = transpose_mod(b_inv)
    b_star = [[(mu * entry) % order for entry in row] for row in b_star]
    return b, b_star


def basis_vectors(
    matrix: Sequence[Sequence[gp.mpz]], rows: Sequence[int], generator, group: PairingGroup
) -> list[list]:
    """The rows ``rows`` of ``matrix`` as vectors of group elements ``generator ** row``.

    Args:
        matrix: Exponent matrix, one row per basis vector (``B`` or ``B*``).
        rows: Indices of the rows to take, in the order of the result.
        generator: Generator of ``G1`` (for ``B``) or ``G2`` (for ``B*``).
        group: Pairing group instance.

    Returns:
        One list of ``len(matrix[i])`` group elements per index ``i`` of ``rows``.
    """
    return [pow_many(generator, matrix[i], group) for i in rows]


def combine(vectors: Sequence[list], scalars: Sequence[int], group: PairingGroup) -> list:
    """The linear combination ``prod_i vectors[i] ** scalars[i]``, coordinate by coordinate.

    Args:
        vectors: Vectors of group elements of one group and one length.
        scalars: One integer (or ``Z_p`` element) per vector, reduced modulo the group order.
        group: Pairing group instance.

    Returns:
        The vector whose coordinate ``j`` is ``prod_i vectors[i][j] ** scalars[i]``.

    Raises:
        FEValidationError: If there is no vector, the vectors differ in length,
            or there is not one scalar per vector.
    """
    if not vectors:
        raise FEValidationError("combine needs at least one vector")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise FEValidationError(
            "combine needs vectors of the same length, got lengths {}".format(sorted({len(v) for v in vectors}))
        )
    if len(scalars) != len(vectors):
        raise FEValidationError(
            "combine needs one scalar per vector, got {} scalars for {} vectors".format(len(scalars), len(vectors))
        )
    return [multiexp([vector[j] for vector in vectors], scalars, group) for j in range(width)]


def vector_pair(xs: list, ys: list):
    """The coordinate-wise pairing ``prod_k e(xs[k], ys[k])`` of a ``G1`` and a ``G2`` vector.

    One final exponentiation on BLS12-381 (see
    :func:`~pyfe4ai.primitives.pairing.pairing_backend.multi_pair`).

    Raises:
        FEValidationError: If the vectors are empty (the product of no
            pairings is not a group element) or differ in length.
    """
    if len(xs) != len(ys):
        raise FEValidationError(
            "vector_pair needs vectors of the same length, got lengths {} and {}".format(len(xs), len(ys))
        )
    if not xs:
        raise FEValidationError("vector_pair needs at least one coordinate")
    return multi_pair(xs, ys)
