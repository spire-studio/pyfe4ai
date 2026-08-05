"""Precision management toolkit for functional encryption on real-valued data.

FE schemes operate on integers, but AI/ML workloads use floats.  This module
provides three tools to bridge the gap:

* :class:`QuantizationConfig` — float↔int conversion with configurable
  scaling (``decimal`` or ``binary`` mode).
* :func:`estimate_bounds` — auto-compute ``bound_x``, ``bound_y`` and
  discrete-log table requirements from sample data.
* :func:`analyze_precision_loss` — report quantization error metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizationConfig:
    """Immutable quantization strategy for float↔int conversion.

    Parameters
    ----------
    precision : int
        Number of fractional digits (decimal) or bits (binary) to preserve.
    mode : str
        ``"decimal"`` scales by ``10**precision`` (compatible with existing
        PyFE4AI ndarray helpers).  ``"binary"`` scales by ``2**precision``
        (avoids base-10 rounding artefacts, better for ML weights).
    """

    precision: int = 3
    mode: str = "decimal"

    def __post_init__(self) -> None:
        if self.precision < 0:
            raise ValueError("precision must be >= 0")
        if self.mode not in ("decimal", "binary"):
            raise ValueError("mode must be 'decimal' or 'binary'")

    @property
    def scale_factor(self) -> int:
        """Multiplicative factor used to shift floats into integer domain."""
        if self.mode == "decimal":
            return 10 ** self.precision
        return 1 << self.precision  # 2**precision

    # -- scalar operations --------------------------------------------------

    def quantize(self, value: float) -> int:
        """Convert a single float to an integer via rounding."""
        return int(round(value * self.scale_factor))

    def dequantize(self, value: int, order: int = 1) -> float:
        """Convert an integer back to float.

            Args:
                value: Input value.
                order: Group order (or exponent order for dequantization).
        """
        return value / (self.scale_factor ** order)

    # -- array operations (vectorised) --------------------------------------

    def quantize_array(self, arr: np.ndarray) -> np.ndarray:
        """Quantize an entire array in one vectorised operation.

        Uses ``np.round`` (banker's rounding) instead of truncation to
        minimise systematic bias.  Returns int64 array.
        """
        return np.round(arr * self.scale_factor).astype(np.int64)

    def dequantize_array(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        """De-quantize an integer array back to float64.

            Args:
                arr: Input array.
                order: Group order (or exponent order for dequantization).
        """
        return arr.astype(np.float64) / (self.scale_factor ** order)


# ---------------------------------------------------------------------------
# Bounds estimation
# ---------------------------------------------------------------------------


def estimate_bounds(
    x_data: np.ndarray,
    y_weights: np.ndarray,
    precision: int,
    eta: int | None = None,
    mode: str = "decimal",
) -> dict:
    """Auto-compute safe FE bounds from sample data.

        Args:
            x_data: See implementation for details.
            y_weights: See implementation for details.
            precision: Quantization precision.
            eta: Inner-product vector dimension.
            mode: See implementation for details.
    """
    qcfg = QuantizationConfig(precision=precision, mode=mode)

    # Quantize and find max absolute values
    q_x = qcfg.quantize_array(x_data.ravel())
    q_y = qcfg.quantize_array(y_weights.ravel())

    bound_x = int(np.max(np.abs(q_x))) if q_x.size else 0
    bound_y = int(np.max(np.abs(q_y))) if q_y.size else 0

    if eta is None:
        eta = min(q_x.size, q_y.size)

    # Worst-case inner product magnitude: eta * bound_x * bound_y
    max_ip = eta * bound_x * bound_y

    # dlog constraint: result must fit within 10^(precision + 2) for decimal
    # mode.  For binary mode the same constraint applies because PyFE4AI's
    # dlog_solver always uses 10^(precision+2).
    dlog_bound = 10 ** (precision + 2)

    # Minimum precision such that 10^(p+2) >= max_ip
    dlog_precision_min = max(0, math.ceil(math.log10(max(max_ip, 1))) - 2)

    warnings: list[str] = []
    if max_ip > dlog_bound:
        warnings.append(
            f"max inner product ({max_ip}) exceeds dlog bound "
            f"({dlog_bound}) at precision={precision}. "
            f"Increase precision to >= {dlog_precision_min} "
            f"or reduce data / weight magnitudes."
        )

    return {
        "bound_x": bound_x,
        "bound_y": bound_y,
        "eta": eta,
        "max_inner_product": max_ip,
        "dlog_bound": dlog_bound,
        "dlog_precision_min": dlog_precision_min,
        "warnings": warnings,
    }


def validate_bounds(
    bound_x: int,
    bound_y: int,
    eta: int,
    precision: int,
) -> list[str]:
    """Check whether configured bounds are safe for FE operations.

        Args:
            bound_x: Bound on plaintext values.
            bound_y: Bound on weight values.
            eta: Inner-product vector dimension.
            precision: Quantization precision.
    """
    msgs: list[str] = []
    max_ip = eta * bound_x * bound_y
    dlog_bound = 10 ** (precision + 2)

    if max_ip > dlog_bound:
        msgs.append(
            f"eta*bound_x*bound_y = {max_ip} exceeds dlog table size "
            f"{dlog_bound}. Decryption will fail with ValueError."
        )
    if bound_x <= 0:
        msgs.append("bound_x must be > 0")
    if bound_y <= 0:
        msgs.append("bound_y must be > 0")
    return msgs


# ---------------------------------------------------------------------------
# Precision analysis
# ---------------------------------------------------------------------------


def analyze_precision_loss(
    original: np.ndarray,
    precision: int,
    mode: str = "decimal",
) -> dict:
    """Report quantization error metrics for the given data and precision.

        Args:
            original: See implementation for details.
            precision: Quantization precision.
            mode: See implementation for details.
    """
    qcfg = QuantizationConfig(precision=precision, mode=mode)
    quantized = qcfg.quantize_array(original)
    restored = qcfg.dequantize_array(quantized)

    abs_error = np.abs(original - restored)
    max_abs = float(np.max(abs_error)) if abs_error.size else 0.0
    mean_abs = float(np.mean(abs_error)) if abs_error.size else 0.0

    # Relative error (avoid division by zero)
    nonzero_mask = np.abs(original) > 1e-15
    if np.any(nonzero_mask):
        max_rel = float(np.max(abs_error[nonzero_mask] / np.abs(original[nonzero_mask])))
    else:
        max_rel = 0.0

    # Effective bits of precision
    bits = -math.log2(max_abs) if max_abs > 0 else float("inf")

    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
        "bits_of_precision": bits,
    }
