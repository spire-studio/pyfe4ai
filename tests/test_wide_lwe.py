"""Lattice-estimator bounds for the wide-error LWE of ABDP15 (sife/lwe, mife/lwe)."""

import math

import gmpy2 as gp
import pytest

from pyfe4ai.utils.exceptions import FEValidationError
from pyfe4ai.utils.lwe_utils import (
    WIDE_LWE_128_LOG_RATIO,
    abdp15_parameters,
    abdp15_reduction_sigma,
    check_wide_lwe,
    derive_lwe_parameters,
    wide_lwe_n,
)

DIMS = sorted(next(iter(WIDE_LWE_128_LOG_RATIO.values())))


def test_every_width_has_the_same_dimensions():
    assert all(sorted(row) == DIMS for row in WIDE_LWE_128_LOG_RATIO.values())


def test_bounds_grow_with_the_dimension_and_the_width():
    for row in WIDE_LWE_128_LOG_RATIO.values():
        assert all(row[a] <= row[b] for a, b in zip(DIMS, DIMS[1:]))
    widths = sorted(WIDE_LWE_128_LOG_RATIO)
    for a, b in zip(widths, widths[1:]):
        assert all(WIDE_LWE_128_LOG_RATIO[a][n] <= WIDE_LWE_128_LOG_RATIO[b][n] for n in DIMS)


def test_check_uses_the_tabulated_width_and_dimension_below():
    n, log_sigma = DIMS[3], sorted(WIDE_LWE_128_LOG_RATIO)[1]
    bound = WIDE_LWE_128_LOG_RATIO[log_sigma][n]
    sigma = 2.0 ** (log_sigma + 3)  # between two rows: held to the lower one
    check_wide_lwe(n + 10, int(sigma * 2 ** bound), sigma)
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        check_wide_lwe(n + 10, int(sigma * 2 ** (bound + 1)), sigma)


@pytest.mark.parametrize("lwe_n, sigma", [(DIMS[0] - 1, 2.0 ** 30), (DIMS[-1], 2.0 ** 10)])
def test_check_refuses_what_the_table_does_not_cover(lwe_n, sigma):
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        check_wide_lwe(lwe_n, 3, sigma)


def test_auto_picks_the_smallest_dimension_that_passes():
    first_ok = DIMS[5]

    def params_for(n):
        bound = WIDE_LWE_128_LOG_RATIO[32][n]
        ratio = bound if n >= first_ok else bound + 1
        return {"lwe_n": n, "q": 2 ** (32 + ratio), "sigma_q": 2.0 ** 32}

    assert wide_lwe_n(params_for)["lwe_n"] == first_ok
    with pytest.raises(FEValidationError, match="no LWE dimension"):
        wide_lwe_n(lambda n: {"lwe_n": n, "q": 2 ** 400, "sigma_q": 2.0 ** 32})


def test_auto_reads_the_table_at_the_given_width():
    # log2(q / 2^32) is the 2^32-row bound at DIMS[2]: read at sigma_q = 2^40
    # the smallest dimension already passes, read at 2^32 only DIMS[2] does
    q = 2 ** (32 + WIDE_LWE_128_LOG_RATIO[32][DIMS[2]])

    def params_for(n):
        return {"lwe_n": n, "q": q, "sigma_q": 2.0 ** 40}

    assert wide_lwe_n(params_for)["lwe_n"] == DIMS[0]
    assert wide_lwe_n(params_for, lambda p: p["sigma_q"] / 2 ** 8)["lwe_n"] == DIMS[2]


def test_auto_abdp15_parameters_meet_the_table():
    params = abdp15_parameters(gp.mpz(50), gp.mpz(50), 2, "auto")
    assert params["lwe_n"] in DIMS
    check_wide_lwe(params["lwe_n"], params["q"], params["sigma_q"])
    reduced = lambda p: abdp15_reduction_sigma(p["sigma_q"], 100, 2)  # noqa: E731
    params = abdp15_parameters(gp.mpz(50), gp.mpz(50), 2, "auto", lwe_sigma=reduced)
    check_wide_lwe(params["lwe_n"], params["q"], reduced(params))


def test_reduction_width():
    # sigma >= (1 + B sqrt(eta)) sigma' (Bourse 2017, Thm. 8.2.2)
    assert abdp15_reduction_sigma(2.0 ** 30, 100, 4) == 2.0 ** 30 / 201
    assert abdp15_reduction_sigma(7.0, 0, 9) == 7.0


def test_explicit_dimension_is_used_as_given():
    assert abdp15_parameters(gp.mpz(8), gp.mpz(8), 4, 16)["lwe_n"] == 16


def test_modulus_is_the_largest_prime_below_a_power_of_two():
    q = derive_lwe_parameters(gp.mpz(8), gp.mpz(8), 4, 16)[1]
    assert q == gp.prev_prime(gp.mpz(1) << int(q).bit_length())
