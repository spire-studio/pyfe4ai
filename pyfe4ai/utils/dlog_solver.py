"""Bounded discrete logarithm solvers for integer and pairing groups.

Provides two strategies:

1. **Dlog table** (default): precompute a lookup table of size O(√n) once
   per process (memoised in memory) and reuse it for every solve.  Tables are
   never read from disk: a cache file in the working directory would let
   anyone with write access to it run code (pickle) or silently corrupt
   decryption results, and rebuilding is as cheap as verifying a cached copy.

2. **Baby-step giant-step (BSGS)** (optional): compute baby-step and
   giant-step values on the fly without any disk cache.  O(√n) time and
   space per invocation.  Useful for one-off solves or when disk caching
   is undesirable.
"""

from __future__ import annotations

import functools
import logging
import math

import gmpy2 as gp

from pyfe4ai.utils.pairing_backend import PairingGroup, pair

logger = logging.getLogger(__name__)


def _dlog_table_step_size(bound: int) -> int:
    """Compute table step size m = ceil(sqrt(2·bound + 1))."""
    return math.isqrt(2 * bound + 1) + 1


# ---------------------------------------------------------------------------
# Cached dlog table functions for integer groups (Zp*)  — DEFAULT strategy
# ---------------------------------------------------------------------------


def dlog_build_table(
    g_str: str, p_str: str, bound: int
) -> tuple[dict, int, gp.mpz]:
    """Build a dlog lookup table for discrete log in ``[-bound, bound]``.

        Args:
            g_str: Group generator as a digit string.
            p_str: Prime modulus as a digit string.
            bound: Upper bound on absolute values.
    """
    g = gp.mpz(g_str)
    p = gp.mpz(p_str)
    m = _dlog_table_step_size(bound)
    table: dict[str, int] = {}
    val = gp.mpz(1)
    for j in range(m):
        table[gp.digits(val)] = j
        val = (val * g) % p
    giant = gp.powmod(g, -m, p)
    return table, m, giant


def dlog_table_solve(
    value, g_str: str, p_str: str, bound: int,
    table: dict, m: int, giant: gp.mpz,
) -> int:
    """Solve ``g^x ≡ value (mod p)`` with ``x ∈ [-bound, bound]``.

        Args:
            value: Input value.
            g_str: Group generator as a digit string.
            p_str: Prime modulus as a digit string.
            bound: Upper bound on absolute values.
            table: Baby-step table ``{digits(g^j): j}`` for ``0 <= j < m``.
            m: Baby-step count (table size).
            giant: Giant step ``g^-m mod p``.
    """
    g = gp.mpz(g_str)
    p = gp.mpz(p_str)
    h_shifted = (gp.mpz(value) * gp.powmod(g, bound, p)) % p
    gamma = h_shifted
    n_giants = (2 * bound) // m + 2
    for i in range(n_giants):
        key = gp.digits(gamma)
        if key in table:
            return i * m + table[key] - bound
        gamma = (gamma * giant) % p
    raise ValueError(
        "inner product out of bound supported by crypto system "
        "(bound={})".format(bound)
    )


@functools.lru_cache(maxsize=32)
def _cached_dlog_table(g_str: str, p_str: str, bound: int) -> tuple[dict, int, gp.mpz]:
    logger.debug("building dlog table in memory (bound=%d)", bound)
    return dlog_build_table(g_str, p_str, bound)


def load_or_build_dlog_table(
    filepath: str, g_str: str, p_str: str, bound: int,
) -> tuple[dict, int, int, gp.mpz]:
    """Return a dlog lookup table for ``g`` mod ``p`` covering ``[-bound, bound]``.

    The table is built in memory and memoised per process. *filepath* is kept
    for API compatibility and ignored: earlier versions loaded a pickle (and
    a JSON fallback) from that location without any integrity check, which
    allowed arbitrary code execution / wrong results via a planted file.

    Args:
        filepath: Ignored (legacy cache location).
        g_str: Group generator as a digit string.
        p_str: Prime modulus as a digit string.
        bound: Upper bound on absolute values.
    """
    table, m, giant = _cached_dlog_table(str(g_str), str(p_str), int(bound))
    return table, bound, m, giant


# ---------------------------------------------------------------------------
# On-the-fly BSGS for integer groups (Zp*)  — OPTIONAL backup strategy
# ---------------------------------------------------------------------------


def bsgs_solve_int(
    value, g_str: str, p_str: str, bound: int,
) -> int:
    """Solve ``g^x ≡ value (mod p)`` with ``x ∈ [-bound, bound]`` via BSGS.

    Uses the standard two-pass approach (matching GoFE / CiFEr reference
    implementations): search ``[0, bound]`` with ``g``, then search
    ``[0, bound]`` with ``g⁻¹`` to cover negative solutions.

    Unlike the dlog-table approach, this computes baby-step and giant-step
    values entirely on the fly without reading or writing any disk cache.
    Suitable for one-off discrete-log recovery or when caching is undesirable.

    Complexity: O(√bound) time and space.

    Args:
        value: Target group element ``g^x mod p``.
        g_str: Generator as a :func:`gmpy2.digits` string.
        p_str: Modulus as a :func:`gmpy2.digits` string.
        bound: Search range ``[-bound, bound]``.

    Returns:
        The discrete logarithm *x*.

    Raises:
        ValueError: If no solution is found within the bound.
    """
    h = gp.mpz(value)
    g = gp.mpz(g_str)
    p = gp.mpz(p_str)
    m = math.isqrt(bound) + 1

    # Baby-step: build in-memory table {g^j mod p → j} for j ∈ [0, m)
    baby_table: dict[str, int] = {}
    val = gp.mpz(1)
    for j in range(m):
        baby_table[gp.digits(val)] = j
        val = (val * g) % p

    # Giant-step factor: g^{-m} mod p
    giant = gp.powmod(g, -m, p)

    # Pass 1: search x ∈ [0, bound] — solve g^x ≡ h (mod p)
    gamma = h
    for i in range(m):
        key = gp.digits(gamma)
        if key in baby_table:
            return i * m + baby_table[key]
        gamma = (gamma * giant) % p

    # Pass 2: search x ∈ (0, bound] with g_inv
    # Reuse baby table but with g_inv^{-m} = g^{m} as giant step
    g_inv = gp.invert(g, p)
    giant_neg = gp.powmod(g_inv, -m, p)

    # Build baby table for g_inv
    baby_table_inv: dict[str, int] = {}
    val = gp.mpz(1)
    for j in range(m):
        baby_table_inv[gp.digits(val)] = j
        val = (val * g_inv) % p

    gamma = h
    for i in range(m):
        key = gp.digits(gamma)
        if key in baby_table_inv:
            result = i * m + baby_table_inv[key]
            if result > 0:  # x=0 already found in pass 1
                return -result
        gamma = (gamma * giant_neg) % p

    raise ValueError(
        "discrete log not found in range [-{0}, {0}]".format(bound)
    )


# ---------------------------------------------------------------------------
# On-the-fly BSGS for pairing groups (GT)  — OPTIONAL backup strategy
# ---------------------------------------------------------------------------


def bsgs_solve_pairing(target, g, h, pairing_group_param: str, bound: int) -> int:
    """Solve ``e(g, h)^x = target`` in a pairing group via on-the-fly BSGS.

    Uses the standard approach (matching CiFEr's ``cfe_baby_giant_FP12``):
    build one baby-step table for ``e(g,h)`` and simultaneously check both
    ``target`` and ``target⁻¹`` in each giant step, covering ``[-bound, bound]``
    in a single loop.

    Computes baby-step and giant-step values entirely in memory without
    persisting any lookup table.

    Args:
        target: The target element in GT (output of a pairing).
        g: Generator in G1.
        h: Generator in G2.
        pairing_group_param: PBC pairing parameter name (e.g. ``"MNT224"``).
        bound: Search range ``[-bound, bound]``.

    Returns:
        The discrete logarithm *x*.

    Raises:
        ValueError: If no solution is found within the bound.
    """
    group = PairingGroup(pairing_group_param)
    e_gh = pair(g, h)

    # Check zero case
    identity = e_gh ** 0
    if target == identity:
        return 0

    m = math.isqrt(bound) + 1

    # Baby-step table: {serialize(e_gh^j) → j} for j in [0, m]
    baby_table: dict[str, int] = {}
    val = identity
    for j in range(m + 1):
        baby_table[group.serialize(val).decode("utf-8")] = j
        val = val * e_gh

    # Giant-step factor: e_gh^{-m}
    giant = 1 / (e_gh ** m)

    # Simultaneously search positive and negative
    gamma_pos = target
    gamma_neg = 1 / target
    for i in range(m + 1):
        key_pos = group.serialize(gamma_pos).decode("utf-8")
        if key_pos in baby_table:
            return i * m + baby_table[key_pos]

        key_neg = group.serialize(gamma_neg).decode("utf-8")
        if key_neg in baby_table:
            return -(i * m + baby_table[key_neg])

        gamma_pos = gamma_pos * giant
        gamma_neg = gamma_neg * giant

    raise ValueError(
        "discrete log not found in range [-{0}, {0}]".format(bound)
    )
