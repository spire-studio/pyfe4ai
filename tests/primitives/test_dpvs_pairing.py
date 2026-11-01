"""pyfe4ai.primitives.pairing.dpvs on the test pairing group.

Dual bases in the group (``e(b_i, b*_j) = e(g1, g2)^(mu delta_ij)``) and the
vector operations, checked against the per-element loops they replace. The
tests use ``pow_many``, ``multiexp`` and ``to_zr``, so they run on BLS12-381
and, with ``PYFE4AI_TEST_PAIRING_GROUP=MNT224``, on charm.
"""

import gmpy2 as gp
import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from pyfe4ai.core.exceptions import FEValidationError  # noqa: E402
from pyfe4ai.primitives.pairing import pairing_backend as pb  # noqa: E402
from pyfe4ai.primitives.pairing.dpvs import basis_vectors, combine, encode, random_dual_bases, vector_pair  # noqa: E402
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr, to_zr  # noqa: E402

DIM = 4


@pytest.fixture(scope="module")
def group():
    return pb.PairingGroup(GROUP)


@pytest.fixture(scope="module")
def order(group):
    return gp.mpz(int(group.order()))


@pytest.fixture(scope="module")
def generators(group):
    return group.random(pb.G1), group.random(pb.G2)


def _power(group, element, exponent):
    return element ** to_zr(group, gp.mpz(int(exponent)) % gp.mpz(int(group.order())))


def _bases(group, order, generators, mu):
    """``(B in G1^(DIM x DIM), B* in G2^(DIM x DIM))`` for a fresh pair of dual bases."""
    b, b_star = random_dual_bases(group, DIM, mu, order)
    g1, g2 = generators
    return (
        basis_vectors(b, range(DIM), g1, group),
        basis_vectors(b_star, range(DIM), g2, group),
    )


@pytest.mark.parametrize("scaled", [False, True], ids=["mu=1", "random mu"])
def test_duality_in_the_group(group, order, generators, scaled):
    mu = gp.mpz(int(random_nonzero_zr(group))) if scaled else gp.mpz(1)
    b1, b2 = _bases(group, order, generators, mu)
    g_t = pb.pair(*generators)
    for i in range(DIM):
        for j in range(DIM):
            want = _power(group, g_t, mu) if i == j else group.init(pb.GT, 1)
            assert vector_pair(b1[i], b2[j]) == want


def test_basis_vectors_select_and_order_rows(group, order, generators):
    g2 = generators[1]
    _, b_star = random_dual_bases(group, DIM, gp.mpz(1), order)
    got = basis_vectors(b_star, [2, 0], g2, group)
    assert len(got) == 2
    for vector, row in zip(got, (b_star[2], b_star[0])):
        assert vector == [_power(group, g2, entry) for entry in row]
    assert basis_vectors(b_star, [], g2, group) == []


def test_combine_is_coordinatewise(group, order, generators):
    b1, _ = _bases(group, order, generators, gp.mpz(1))
    u, v = b1[0], b1[1]
    # a negative scalar, one above the order, and a zero are reduced mod the order
    for s, t in ((3, 5), (-2, int(order) + 7), (0, 11), (0, 0)):
        got = combine([u, v], [s, t], group)
        assert len(got) == DIM
        for j in range(DIM):
            assert got[j] == _power(group, u[j], s) * _power(group, v[j], t)


def test_combine_of_three_vectors_and_of_one(group, order, generators):
    b1, _ = _bases(group, order, generators, gp.mpz(1))
    scalars = [4, -9, int(group.random(pb.ZR))]
    got = combine(b1[:3], scalars, group)
    for j in range(DIM):
        want = pb.multiexp([b1[i][j] for i in range(3)], scalars, group)
        assert got[j] == want
    assert combine([b1[3]], [1], group) == b1[3]


def test_combination_of_dual_bases_pairs_to_the_inner_product(group, order, generators):
    # e(sum_i a_i b_i, sum_j c_j b*_j) = e(g1, g2)^(mu <a, c>)
    mu = gp.mpz(int(random_nonzero_zr(group)))
    b1, b2 = _bases(group, order, generators, mu)
    a = [3, -4, 0, int(group.random(pb.ZR))]
    c = [5, 2, 9, -int(group.random(pb.ZR))]
    got = vector_pair(combine(b1, a, group), combine(b2, c, group))
    inner = sum(gp.mpz(x) * gp.mpz(y) for x, y in zip(a, c)) % order
    assert got == _power(group, pb.pair(*generators), mu * inner)


def test_vector_pair_is_the_product_of_pairings(group, generators):
    g1, g2 = generators
    xs = [_power(group, g1, e) for e in (2, 3, 5)]
    ys = [_power(group, g2, e) for e in (7, 11, 13)]
    want = pb.pair(xs[0], ys[0]) * pb.pair(xs[1], ys[1]) * pb.pair(xs[2], ys[2])
    assert vector_pair(xs, ys) == want
    assert want == _power(group, pb.pair(g1, g2), 2 * 7 + 3 * 11 + 5 * 13)


def test_vector_pair_refuses_unequal_lengths(group, generators):
    g1, g2 = generators
    with pytest.raises(ValueError):
        vector_pair([g1, g1], [g2])


def test_combine_refuses_malformed_input_on_the_group(group, generators):
    g1 = generators[0]
    with pytest.raises(FEValidationError):
        combine([], [], group)
    with pytest.raises(FEValidationError):
        combine([[g1, g1], [g1]], [1, 1], group)
    with pytest.raises(FEValidationError):
        combine([[g1], [g1]], [1], group)


def test_encode_is_the_combination_of_the_rows(group, order, generators):
    g1 = generators[0]
    b, _ = random_dual_bases(group, 4, gp.mpz(1), order)
    rows = basis_vectors(b, [0, 1], g1, group)
    assert encode([5, 7], b[:2], g1, order, group) == combine(rows, [5, 7], group)


def test_encode_reduces_every_coefficient_modulo_the_order(group, order, generators):
    g1 = generators[0]
    b, _ = random_dual_bases(group, DIM, gp.mpz(1), order)
    rows = basis_vectors(b, range(DIM), g1, group)
    random_scalar = int(group.random(pb.ZR))
    # negative, zero, above the order, an mpz and a ZR-sized value
    for coefficients in ([3, -4, 0, 9], [0, 0, 0, 0], [-1, int(order) + 7, gp.mpz(2), -int(order)],
                         [random_scalar, 1, random_scalar, -random_scalar]):
        assert encode(coefficients, b, g1, order, group) == combine(rows, coefficients, group)


def test_encode_of_one_row_and_of_a_wide_matrix(group, order, generators):
    g2 = generators[1]
    _, b_star = random_dual_bases(group, DIM, gp.mpz(1), order)
    assert encode([1], b_star[2:3], g2, order, group) == basis_vectors(b_star, [2], g2, group)[0]
    assert encode([6], b_star[2:3], g2, order, group) == [_power(group, g2, 6 * entry) for entry in b_star[2]]
    wide = [[gp.mpz(i * 10 + j + 1) for j in range(9)] for i in range(3)]
    got = encode([2, -1, 5], wide, g2, order, group)
    assert len(got) == 9
    for j in range(9):
        assert got[j] == _power(group, g2, 2 * wide[0][j] - wide[1][j] + 5 * wide[2][j])


def test_encode_pairs_to_the_inner_product(group, order, generators):
    # the NPP22 use: coefficients on the rows of B and of B* of one dual pair
    b, b_star = random_dual_bases(group, DIM, gp.mpz(1), order)
    g1, g2 = generators
    x, y = [3, -4, 0, 8], [5, 2, 9, -1]
    got = vector_pair(encode(x, b, g1, order, group), encode(y, b_star, g2, order, group))
    assert got == _power(group, pb.pair(g1, g2), sum(a * c for a, c in zip(x, y)))


def test_encode_refuses_malformed_input(group, order, generators):
    g1 = generators[0]
    b, _ = random_dual_bases(group, DIM, gp.mpz(1), order)
    with pytest.raises(FEValidationError):
        encode([], [], g1, order, group)
    with pytest.raises(FEValidationError):
        encode([1], b[:2], g1, order, group)
    with pytest.raises(FEValidationError):
        encode([1, 2, 3], b[:2], g1, order, group)
    with pytest.raises(FEValidationError):
        encode([1, 1], [b[0], b[1][:-1]], g1, order, group)
