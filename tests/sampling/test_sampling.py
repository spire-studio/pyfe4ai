"""Gaussian samplers keep full integer precision at every width."""

import statistics

import pytest

from pyfe4ai.primitives.sampling import discrete_gaussian_bigint
from pyfe4ai.primitives.sampling import discrete_gaussian_matrix


@pytest.mark.parametrize("sigma", [2.0 ** 52, 2.0 ** 71.7])
def test_wide_gaussian_keeps_low_bits(sigma):
    """Rounding a double sample at these widths left every low bit zero."""
    samples = [v for row in discrete_gaussian_matrix(4, 500, sigma) for v in row]
    odd = sum(int(v) & 1 for v in samples) / len(samples)
    assert 0.4 < odd < 0.6
    assert 0.8 * sigma < statistics.pstdev(int(v) for v in samples) < 1.2 * sigma


def test_matrix_shape_for_wide_gaussian():
    m = discrete_gaussian_matrix(3, 2, 2.0 ** 60)
    assert [len(row) for row in m] == [2, 2, 2]


def test_narrow_gaussian_unchanged():
    samples = [v for row in discrete_gaussian_matrix(10, 200, 3.2) for v in row]
    assert 2.5 < statistics.pstdev(int(v) for v in samples) < 4.0


def test_bigint_sampler_width():
    samples = discrete_gaussian_bigint(2 ** 80, 2000)
    assert 0.8 * 2 ** 80 < statistics.pstdev(int(v) for v in samples) < 1.2 * 2 ** 80
