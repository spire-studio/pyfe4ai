"""Tests for utils.quantization — precision management toolkit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyfe4ai.utils.quantization import (
    QuantizationConfig,
    analyze_precision_loss,
    estimate_bounds,
    validate_bounds,
)


# ── QuantizationConfig ─────────────────────────────────────────────


class TestQuantizationConfig:
    """Basic construction and property tests."""

    def test_decimal_scale_factor(self):
        qc = QuantizationConfig(precision=3, mode="decimal")
        assert qc.scale_factor == 1000

    def test_binary_scale_factor(self):
        qc = QuantizationConfig(precision=8, mode="binary")
        assert qc.scale_factor == 256

    def test_precision_zero(self):
        qc = QuantizationConfig(precision=0)
        assert qc.scale_factor == 1

    def test_negative_precision_raises(self):
        with pytest.raises(ValueError, match="precision must be >= 0"):
            QuantizationConfig(precision=-1)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            QuantizationConfig(precision=3, mode="hex")


class TestQuantizeScalar:
    """Scalar quantize / dequantize round-trip."""

    @pytest.mark.parametrize("mode", ["decimal", "binary"])
    def test_round_trip(self, mode):
        qc = QuantizationConfig(precision=4, mode=mode)
        original = 3.14159
        restored = qc.dequantize(qc.quantize(original))
        assert abs(original - restored) < 1.0 / qc.scale_factor

    def test_negative_value(self):
        qc = QuantizationConfig(precision=3, mode="decimal")
        assert qc.quantize(-2.718) == -2718

    def test_rounding_not_truncation(self):
        qc = QuantizationConfig(precision=1, mode="decimal")
        # 1.55 * 10 = 15.5 → round to 16 (not truncate to 15)
        assert qc.quantize(1.55) == 16

    def test_dequantize_order2(self):
        qc = QuantizationConfig(precision=2, mode="decimal")
        # Quadratic scheme: result scaled by 10^(2*2) = 10000
        assert qc.dequantize(10000, order=2) == 1.0


class TestQuantizeArray:
    """Vectorised array operations."""

    def test_array_round_trip(self):
        qc = QuantizationConfig(precision=3, mode="decimal")
        original = np.array([1.234, -5.678, 0.001, 0.0])
        quantized = qc.quantize_array(original)
        assert quantized.dtype == np.int64
        restored = qc.dequantize_array(quantized)
        np.testing.assert_allclose(original, restored, atol=1e-3)

    def test_2d_shape_preserved(self):
        qc = QuantizationConfig(precision=2, mode="binary")
        arr = np.random.randn(4, 5)
        q = qc.quantize_array(arr)
        assert q.shape == (4, 5)
        assert q.dtype == np.int64

    def test_uses_rounding_not_truncation(self):
        qc = QuantizationConfig(precision=1, mode="decimal")
        arr = np.array([1.55, 2.44999])
        q = qc.quantize_array(arr)
        assert q[0] == 16   # round up
        assert q[1] == 24   # round down

    def test_dequantize_array_order2(self):
        qc = QuantizationConfig(precision=2, mode="decimal")
        arr = np.array([10000, 20000])
        result = qc.dequantize_array(arr, order=2)
        np.testing.assert_allclose(result, [1.0, 2.0])


# ── estimate_bounds ────────────────────────────────────────────────


class TestEstimateBounds:
    """Bound estimation from sample data."""

    def test_basic_bounds(self):
        x = np.array([0.1, -0.2, 0.05])
        y = np.array([0.1, 0.1, 0.1])
        result = estimate_bounds(x, y, precision=2)
        assert result["bound_x"] == 20   # max(|10|, |20|, |5|) = 20
        assert result["bound_y"] == 10
        assert result["eta"] == 3
        # max_ip = 3 * 20 * 10 = 600 < dlog_bound = 10^4 = 10000
        assert result["warnings"] == []

    def test_overflow_warning(self):
        x = np.array([100.0] * 10)
        y = np.array([100.0] * 10)
        result = estimate_bounds(x, y, precision=1)
        # max_ip = 10 * 1000 * 1000 = 10_000_000
        # dlog_bound = 10^3 = 1000 → overflow
        assert len(result["warnings"]) == 1
        assert "exceeds dlog bound" in result["warnings"][0]

    def test_eta_override(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0])
        result = estimate_bounds(x, y, precision=2, eta=1)
        assert result["eta"] == 1

    def test_binary_mode(self):
        x = np.array([0.5])
        y = np.array([1.0])
        result = estimate_bounds(x, y, precision=8, mode="binary")
        assert result["bound_x"] == 128  # 0.5 * 256 = 128
        assert result["bound_y"] == 256


# ── validate_bounds ────────────────────────────────────────────────


class TestValidateBounds:
    """Bound validation checks."""

    def test_safe_bounds(self):
        msgs = validate_bounds(bound_x=10, bound_y=10, eta=3, precision=3)
        assert msgs == []

    def test_overflow_detected(self):
        msgs = validate_bounds(bound_x=1000, bound_y=1000, eta=10, precision=2)
        # 10 * 1000 * 1000 = 10_000_000 > 10^4 = 10_000
        assert any("exceeds dlog table size" in m for m in msgs)

    def test_invalid_bounds(self):
        msgs = validate_bounds(bound_x=0, bound_y=-1, eta=1, precision=1)
        assert len(msgs) >= 2


# ── analyze_precision_loss ─────────────────────────────────────────


class TestAnalyzePrecisionLoss:
    """Precision loss reporting."""

    def test_zero_loss_at_high_precision(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = analyze_precision_loss(arr, precision=10, mode="decimal")
        assert result["max_abs_error"] == 0.0
        assert result["bits_of_precision"] == float("inf")

    def test_positive_loss_at_low_precision(self):
        arr = np.array([1.23456])
        result = analyze_precision_loss(arr, precision=2, mode="decimal")
        assert result["max_abs_error"] > 0
        assert result["max_abs_error"] < 0.01  # within 10^-2

    def test_relative_error_with_zeros(self):
        arr = np.array([0.0, 0.0])
        result = analyze_precision_loss(arr, precision=3)
        assert result["max_rel_error"] == 0.0

    def test_binary_vs_decimal(self):
        arr = np.random.randn(100)
        d = analyze_precision_loss(arr, precision=10, mode="decimal")
        b = analyze_precision_loss(arr, precision=10, mode="binary")
        # binary 2^10=1024 vs decimal 10^10=10B — decimal has much less error
        assert d["max_abs_error"] <= b["max_abs_error"]
