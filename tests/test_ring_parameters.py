"""BMMS21 Sec. 4.2 parameters and the Ring-LWE 128-bit table."""

import math

import pytest

from pyfe4ai.utils.exceptions import FEValidationError
from pyfe4ai.utils.ring_lwe_utils import (
    RING_LWE_128_LOG_Q,
    bmms21_selective_parameters,
    check_ring_lwe,
    ring_lwe_bound,
)


def _eta(dim: int, log2_inv_eps: float) -> float:
    """Smoothing parameter of Z^dim for eps = 2^-log2_inv_eps (up to a negligible term)."""
    return math.sqrt((math.log(2 * dim) + log2_inv_eps * math.log(2)) / math.pi)


@pytest.mark.parametrize("kappa, l, X, Y, n", [
    (128, 1, 4, 4, 2048),
    (128, 10, 50, 50, 4096),
    (64, 3, 4, 4, 32),
])
def test_selective_parameters_meet_section_4_2(kappa, l, X, Y, n):
    """Regression: sigma = 1 and sigma1 = 2 sqrt(l) X missed the smoothing
    conditions of the G1 -> G2 and G3 -> G4 steps (the left side of the
    mhe-RLWE condition was 0.71, below eta(Z^(n + n l)) of about 5.6)."""
    P = bmms21_selective_parameters(kappa, l, X, Y, n)
    q, p = int(P["q"]), int(P["p"])
    assert p == 2 * l * X * Y + 2 and q % (2 * n) == 1
    # G1 -> G2: mhe-RLWE from RLWE of width sigma (Theorem 1), C = sqrt(kappa) sigma1
    C = math.sqrt(kappa) * P["sigma1"]
    ratio = P["sigma"] * n * C * math.sqrt(l + 2) / P["sigma2"]
    assert P["sigma"] * math.sqrt(1 - ratio ** 2) > _eta(n + n * l, kappa)
    assert P["sigma3"] == pytest.approx(math.sqrt(2) * P["sigma2"])
    assert P["sigma2"] / math.sqrt(2) >= _eta(n, kappa)
    # G3 -> G4 (Lemma 1, ||alpha|| <= 2 X sqrt(l)): Gamma >= sigma' / sqrt(2) >= eta_eps'(Z^n)
    assert P["sigma1"] >= math.sqrt(2) * 2 * X * math.sqrt(l) * P["sigma_prime"] * (1 - 1e-12)
    assert P["sigma_prime"] / math.sqrt(2) >= _eta(n, kappa + math.log2(n)) * (1 - 1e-12)
    # correctness: l Y (2 n kappa sigma1 sigma2 + sqrt(kappa) sigma3) < floor(q / 2p)
    noise = l * Y * (2 * n * kappa * P["sigma1"] * P["sigma2"] + math.sqrt(kappa) * P["sigma3"])
    assert noise < q // (2 * p)
    # the RLWE problems of the proof: width sigma (G1 -> G2) and sigma' (G5 -> G6)
    assert P["rlwe_width"] == min(P["sigma"], P["sigma_prime"])


def test_ring_table_rows_and_lookup():
    widths = sorted(RING_LWE_128_LOG_Q)
    dims = sorted(RING_LWE_128_LOG_Q[widths[0]])
    for n in dims:
        row = [RING_LWE_128_LOG_Q[s][n] for s in widths]
        assert row == sorted(row)  # wider errors allow a larger q
    for s in widths:
        col = [RING_LWE_128_LOG_Q[s][n] for n in dims]
        assert col == sorted(col)
    # a width between two rows uses the lower row; below the table there is none
    assert ring_lwe_bound(4096, 7.9) == RING_LWE_128_LOG_Q[6.0][4096]
    assert ring_lwe_bound(4096, 100.0) == RING_LWE_128_LOG_Q[max(widths)][4096]
    assert ring_lwe_bound(4096, 1.5) is None
    assert ring_lwe_bound(3000, 8.0) is None


def test_check_ring_lwe():
    bound = RING_LWE_128_LOG_Q[8.0][4096]
    check_ring_lwe(4096, 2 ** bound - 1, 8.0)
    with pytest.raises(FEValidationError, match="below 128-bit"):
        check_ring_lwe(4096, 2 ** bound + 1, 8.0)
    with pytest.raises(FEValidationError, match="below 128-bit"):
        check_ring_lwe(4096, 3, 1.0)
