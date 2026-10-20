"""Input validation shared by the schemes."""

from __future__ import annotations

import numbers

import numpy as np

from pyfe4ai.core.exceptions import FEValidationError


def int_vector(values, name: str, length: int | None = None, bound=None) -> list[int]:
    """Return *values* as a list of Python ints, or raise ``FEValidationError``.

    Integer types (``int``, ``numpy.integer``, ``gmpy2.mpz``) and integral
    floats are accepted; booleans and non-integral numbers are rejected
    instead of being truncated silently.

    Args:
        values: Sequence of numbers.
        name: Name used in error messages.
        length: Required length, if any.
        bound: If given, every ``|v|`` must be at most ``bound``.
    """
    if isinstance(values, (str, bytes)):
        raise FEValidationError("{} must be a sequence of integers".format(name))
    try:
        items = list(values)
    except TypeError as exc:
        raise FEValidationError("{} must be a sequence of integers".format(name)) from exc
    out = []
    for v in items:
        if isinstance(v, bool):
            raise FEValidationError("{} must contain integers, got a bool".format(name))
        if isinstance(v, numbers.Integral):
            iv = int(v)
        elif isinstance(v, numbers.Rational) and v.denominator == 1:
            iv = int(v.numerator)  # exact (float(v) would round a large fraction)
        elif (
            isinstance(v, numbers.Real) and not isinstance(v, numbers.Rational)
            and float(v).is_integer() and v == float(v)
        ):
            iv = int(v)
        else:
            # the type only: the value may be a (malformed) secret input
            raise FEValidationError(
                "{} must contain integers, got a non-integral {}".format(name, type(v).__name__)
            )
        if bound is not None and abs(iv) > bound:
            raise FEValidationError("{} exceeds the bound {}".format(name, bound))
        out.append(iv)
    if length is not None and len(out) != length:
        raise FEValidationError(
            "{} must have {} entries, got {}".format(name, length, len(out))
        )
    return out


def strict_int(value, name: str = "input") -> int:
    """*value* as an int; non-integral values raise (see :func:`int_vector`)."""
    return int_vector([value], name)[0]


_MAX_EXACT = 2 ** 53  # floats are exact integers up to here


def scale_to_int(values, precision: int) -> np.ndarray:
    """Return ``round(values * 10^precision)`` as an ``int64`` array.

    The ndarray helpers encode real-valued plaintexts with this: it rounds to
    the nearest integer (ties to even) instead of truncating toward zero,
    multiplies integer arrays exactly, and refuses NaN, infinities and values
    whose scaled magnitude is not exactly representable.

    Raises:
        FEValidationError: For non-numeric, non-finite or too large values.
    """
    if isinstance(precision, bool) or not isinstance(precision, (int, np.integer)) or precision < 0:
        raise FEValidationError("precision must be a non-negative integer, got {!r}".format(precision))
    arr = np.asarray(values)
    scale = 10 ** int(precision)
    if arr.dtype.kind in "iu":
        if arr.size and int(np.max(np.abs(arr.astype(object)))) * scale >= 2 ** 63:
            raise FEValidationError("plaintext too large for precision {}".format(precision))
        return arr.astype(np.int64) * scale
    if arr.dtype.kind not in "fO":
        raise FEValidationError("plaintext must be real-valued, got dtype {}".format(arr.dtype))
    try:
        arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise FEValidationError("plaintext must be real-valued") from exc
    if not np.all(np.isfinite(arr)):
        raise FEValidationError("plaintext contains NaN or infinity")
    scaled = np.rint(arr * float(scale))
    if scaled.size and np.max(np.abs(scaled)) >= _MAX_EXACT:
        raise FEValidationError("plaintext too large for precision {}".format(precision))
    return scaled.astype(np.int64)
