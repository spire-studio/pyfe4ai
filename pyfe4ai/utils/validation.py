"""Input validation shared by the schemes."""

from __future__ import annotations

import numbers

from pyfe4ai.utils.exceptions import FEValidationError


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
        elif isinstance(v, numbers.Real) and float(v).is_integer():
            iv = int(v)
        else:
            raise FEValidationError("{} must contain integers, got {!r}".format(name, v))
        if bound is not None and abs(iv) > bound:
            raise FEValidationError("{} exceeds the bound {}".format(name, bound))
        out.append(iv)
    if length is not None and len(out) != length:
        raise FEValidationError(
            "{} must have {} entries, got {}".format(name, length, len(out))
        )
    return out
