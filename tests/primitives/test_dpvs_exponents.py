"""The exponent side of pyfe4ai.primitives.pairing.dpvs, on stub groups.

No pairing backend is needed: ``random_dual_bases`` draws exponents from
``group.random(ZR)`` only, and ``combine`` refuses malformed input before it
touches the group. The stub hands out a fixed sequence of draws, so the order
of the draws, the rejection of singular matrices and the scaling by ``mu`` are
pinned. The expected answers of ``test_dual_bases_known_answer`` and
``test_singular_first_draw_and_mu_scaling`` were produced by the original
``_random_ob_matrix`` of the DOT18 module, before it moved here, and are
worked out in the comments; DOT18 keeps its output byte for byte.
"""

import gmpy2 as gp
import pytest

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.primitives.pairing.dpvs import combine, random_dual_bases
from pyfe4ai.primitives.pairing.pairing_backend import ZR

MPZ = type(gp.mpz(0))
ORDER = gp.mpz(7)


class FixedGroup:
    """Stands in for a pairing group: ``random(ZR)`` returns the given draws in turn."""

    def __init__(self, draws):
        self._draws = list(draws)
        self.drawn = 0

    def random(self, kind):
        assert kind == ZR, "the dual bases are drawn from Z_p"
        if self.drawn == len(self._draws):
            raise AssertionError("the sampler drew more than the {} stub values".format(len(self._draws)))
        self.drawn += 1
        return self._draws[self.drawn - 1]


def _times_transpose(b, b_star, order):
    """``B B*^T mod order``."""
    return [[sum(x * y for x, y in zip(row, other)) % order for other in b_star] for row in b]


def test_dual_bases_known_answer():
    group = FixedGroup([1, 2, 3, 4])
    b, b_star = random_dual_bases(group, 2, gp.mpz(1), ORDER)
    # B = [[1, 2], [3, 4]] has det = -2 = 5 and inverse 3 [[4, -2], [-3, 1]] = [[5, 1], [5, 3]] mod 7;
    # B* is the transpose of the inverse
    assert b == [[1, 2], [3, 4]]
    assert b_star == [[5, 5], [1, 3]]
    assert group.drawn == 4


def test_singular_first_draw_and_mu_scaling():
    # the first matrix [[1, 2], [2, 4]] is singular mod 7 and is drawn again in full
    group = FixedGroup([1, 2, 2, 4, 1, 2, 3, 4])
    b, b_star = random_dual_bases(group, 2, gp.mpz(3), ORDER)
    # the same B as above, and 3 [[5, 5], [1, 3]] = [[15, 15], [3, 9]] = [[1, 1], [3, 2]] mod 7
    assert b == [[1, 2], [3, 4]]
    assert b_star == [[1, 1], [3, 2]]
    assert group.drawn == 8


@pytest.mark.parametrize("draws, mu", [([1, 2, 3, 4], 1), ([1, 2, 2, 4, 1, 2, 3, 4], 3)])
def test_dual_bases_are_dual_and_mpz(draws, mu):
    b, b_star = random_dual_bases(FixedGroup(draws), 2, gp.mpz(mu), ORDER)
    assert _times_transpose(b, b_star, ORDER) == [[mu % 7, 0], [0, mu % 7]]
    assert all(isinstance(entry, MPZ) for row in b + b_star for entry in row)


def test_combine_refuses_no_vectors():
    with pytest.raises(FEValidationError, match="at least one vector"):
        combine([], [], None)


def test_combine_refuses_vectors_of_different_lengths():
    with pytest.raises(FEValidationError, match="same length"):
        combine([["a", "b"], ["c"]], [1, 2], None)


@pytest.mark.parametrize("scalars", [[], [1], [1, 2, 3]])
def test_combine_refuses_a_scalar_count_that_is_not_the_vector_count(scalars):
    with pytest.raises(FEValidationError, match="one scalar per vector"):
        combine([["a", "b"], ["c", "d"]], scalars, None)
