"""Ring-LWE helper functions shared across SIFE / MIFE / MCFE Ring-LWE schemes.

These were originally private helpers in ``sife/ring_lwe.py``.  They are
extracted here so that cross-package imports reference ``utils`` rather than
reaching into another scheme family's private namespace.

.. versionchanged:: 0.2
   Added NTT-based polynomial multiplication (O(n log n)) alongside the
   original naive O(n²) implementation.

.. versionchanged:: 0.3
   Added ``next_ntt_prime`` for NTT-friendly parameter generation,
   ``poly_mul`` auto-dispatch with cached root of unity.
"""

import logging

import gmpy2 as gp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NTT (Number Theoretic Transform) for Z_q[X] / (X^n + 1)
# ---------------------------------------------------------------------------

def _find_primitive_root(q: gp.mpz, order: int) -> gp.mpz | None:
    """Find a primitive root of unity of the given *order* modulo *q*.

        Args:
            q: Modulus (or second prime).
            order: Group order (or exponent order for dequantization).
    """
    if (q - 1) % order != 0:
        return None
    # g is a generator candidate; raise it to (q-1)/order
    exponent = (q - 1) // order
    # Try small candidates
    for g_candidate in range(2, min(int(q), 1000)):
        g = gp.mpz(g_candidate)
        root = gp.powmod(g, exponent, q)
        if root == 1:
            continue
        # Verify: root^order == 1 mod q and root^(order/2) != 1 mod q
        if gp.powmod(root, order, q) == 1 and gp.powmod(root, order // 2, q) != 1:
            return root
    return None


def _bit_reverse(x: int, log_n: int) -> int:
    """Bit-reverse an integer *x* within *log_n* bits.

        Args:
            x: Input integer.
            log_n: Log2 of the vector length.
    """
    result = 0
    for _ in range(log_n):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


# ---------------------------------------------------------------------------
# Twiddle factor cache for NTT
# ---------------------------------------------------------------------------

_twiddle_cache: dict[tuple[int, int, int], dict] = {}


def _get_twiddle_factors(q: gp.mpz, n: int, root: gp.mpz) -> dict:
    """Return cached twiddle factors for NTT with given parameters.

        Args:
            q: Modulus (or second prime).
            n: Dimension or number of clients.
            root: Primitive root of unity for NTT.
    """
    key = (int(q), n, int(root))
    if key in _twiddle_cache:
        return _twiddle_cache[key]

    log_n = n.bit_length() - 1
    root_inv = gp.invert(root, q)
    n_inv = gp.invert(gp.mpz(n), q)

    # Pre-twist / un-twist tables
    twist = [gp.powmod(root, i, q) for i in range(n)]
    untwist = [gp.powmod(root_inv, i, q) for i in range(n)]

    # Bit-reversal permutation table
    bit_rev = [_bit_reverse(i, log_n) for i in range(n)]

    # Butterfly twiddle tables per stage (forward)
    w_n_base_fwd = gp.powmod(root, 2, q)
    fwd_twiddles: list[list[gp.mpz]] = []
    length = 2
    while length <= n:
        w_step = gp.powmod(w_n_base_fwd, n // length, q)
        stage = [gp.mpz(1)]
        w = gp.mpz(1)
        for _ in range(1, length // 2):
            w = (w * w_step) % q
            stage.append(w)
        fwd_twiddles.append(stage)
        length <<= 1

    # Butterfly twiddle tables per stage (inverse)
    w_n_base_inv = gp.powmod(root_inv, 2, q)
    inv_twiddles: list[list[gp.mpz]] = []
    length = 2
    while length <= n:
        w_step = gp.powmod(w_n_base_inv, n // length, q)
        stage = [gp.mpz(1)]
        w = gp.mpz(1)
        for _ in range(1, length // 2):
            w = (w * w_step) % q
            stage.append(w)
        inv_twiddles.append(stage)
        length <<= 1

    factors = {
        "twist": twist,
        "untwist": untwist,
        "bit_rev": bit_rev,
        "n_inv": n_inv,
        "fwd_twiddles": fwd_twiddles,
        "inv_twiddles": inv_twiddles,
    }
    _twiddle_cache[key] = factors
    return factors


def ntt_forward(
    a: list[gp.mpz], q: gp.mpz, root: gp.mpz
) -> list[gp.mpz]:
    """In-place iterative Cooley-Tukey NTT (decimation-in-time).

        Args:
            a: First polynomial (coefficient vector).
            q: Modulus (or second prime).
            root: Primitive root of unity for NTT.
    """
    n = len(a)
    tf = _get_twiddle_factors(q, n, root)
    twist = tf["twist"]
    bit_rev = tf["bit_rev"]
    fwd_twiddles = tf["fwd_twiddles"]

    # Pre-twist for negacyclic convolution
    twisted = [(a[i] * twist[i]) % q for i in range(n)]

    # Bit-reversal permutation
    result = [twisted[bit_rev[i]] for i in range(n)]

    # Butterfly stages
    for stage_twiddle in fwd_twiddles:
        length = len(stage_twiddle) * 2
        half = length // 2
        for start in range(0, n, length):
            for j in range(half):
                w = stage_twiddle[j]
                u = result[start + j]
                v = (result[start + j + half] * w) % q
                result[start + j] = (u + v) % q
                result[start + j + half] = (u - v) % q
    return result


def ntt_inverse(
    a: list[gp.mpz], q: gp.mpz, root: gp.mpz
) -> list[gp.mpz]:
    """Inverse NTT: transform back to coefficient domain.

        Args:
            a: First polynomial (coefficient vector).
            q: Modulus (or second prime).
            root: Primitive root of unity for NTT.
    """
    n = len(a)
    tf = _get_twiddle_factors(q, n, root)
    bit_rev = tf["bit_rev"]
    inv_twiddles = tf["inv_twiddles"]
    n_inv = tf["n_inv"]
    untwist = tf["untwist"]

    # Bit-reversal permutation
    result = [a[bit_rev[i]] for i in range(n)]

    # Butterfly stages with inverse twiddles
    for stage_twiddle in inv_twiddles:
        length = len(stage_twiddle) * 2
        half = length // 2
        for start in range(0, n, length):
            for j in range(half):
                w = stage_twiddle[j]
                u = result[start + j]
                v = (result[start + j + half] * w) % q
                result[start + j] = (u + v) % q
                result[start + j + half] = (u - v) % q

    # Scale by n^{-1} and un-twist
    for i in range(n):
        result[i] = (result[i] * n_inv % q * untwist[i]) % q

    return result


def poly_mul_ntt(
    a: list[gp.mpz], b: list[gp.mpz], q: gp.mpz, root: gp.mpz
) -> list[gp.mpz]:
    """Polynomial multiplication in Z_q[X]/(X^n+1) via NTT.

        Args:
            a: First polynomial (coefficient vector).
            b: Second polynomial (coefficient vector).
            q: Modulus (or second prime).
            root: Primitive root of unity for NTT.
    """
    a_ntt = ntt_forward(a, q, root)
    b_ntt = ntt_forward(b, q, root)
    c_ntt = [(x * y) % q for x, y in zip(a_ntt, b_ntt)]
    return ntt_inverse(c_ntt, q, root)


# ---------------------------------------------------------------------------
# NTT-friendly primer generation + auto-dispatch
# ---------------------------------------------------------------------------

# Cache: (q, ring_n) → root.  ``None`` means NTT is not available.
_ntt_root_cache: dict[tuple[int, int], gp.mpz | None] = {}


def _get_ntt_root(q: gp.mpz, ring_n: int) -> gp.mpz | None:
    """Look up (or compute and cache) a primitive 2n-th root of unity mod *q*.

        Args:
            q: Modulus (or second prime).
            ring_n: Polynomial ring dimension.
    """
    key = (int(q), ring_n)
    if key not in _ntt_root_cache:
        _ntt_root_cache[key] = _find_primitive_root(q, 2 * ring_n)
    return _ntt_root_cache[key]


def poly_mul(
    a: list[gp.mpz], b: list[gp.mpz], q: gp.mpz
) -> list[gp.mpz]:
    """Polynomial multiplication with automatic NTT acceleration.

        Args:
            a: First polynomial (coefficient vector).
            b: Second polynomial (coefficient vector).
            q: Modulus (or second prime).
    """
    n = len(a)
    root = _get_ntt_root(q, n)
    if root is not None:
        return poly_mul_ntt(a, b, q, root)
    logger.warning(
        "NTT root not found for q=%s, n=%d; falling back to O(n²) "
        "negacyclic multiplication. Consider using next_ntt_prime() "
        "to choose an NTT-friendly prime.",
        q, n,
    )
    return poly_mul_negacyclic(a, b, q)


def next_ntt_prime(lower_bound: gp.mpz, ring_n: int) -> gp.mpz:
    """Find the smallest prime ``q >= lower_bound`` with ``q ≡ 1 (mod 2·ring_n)``.

        Args:
            lower_bound: Lower bound for the search.
            ring_n: Polynomial ring dimension.
    """
    modulus = 2 * ring_n
    # Start from the first candidate ≡ 1 (mod 2n) that is ≥ lower_bound
    remainder = int(lower_bound) % modulus
    if remainder == 0:
        candidate = int(lower_bound) + 1  # want ≡ 1, not ≡ 0
    elif remainder <= 1:
        candidate = int(lower_bound) + (1 - remainder)
    else:
        candidate = int(lower_bound) + (modulus + 1 - remainder)

    while True:
        q = gp.mpz(candidate)
        if gp.is_prime(q):
            return q
        candidate += modulus


# ---------------------------------------------------------------------------
# Polynomial arithmetic in Z_q[X] / (X^n + 1) — naive O(n²)
# ---------------------------------------------------------------------------

def poly_add(a: list[gp.mpz], b: list[gp.mpz], q: gp.mpz) -> list[gp.mpz]:
    """Add two polynomials element-wise modulo q.

        Args:
            a: First polynomial (coefficient vector).
            b: Second polynomial (coefficient vector).
            q: Modulus (or second prime).
    """
    return [(x + y) % q for x, y in zip(a, b)]


def poly_neg(a: list[gp.mpz], q: gp.mpz) -> list[gp.mpz]:
    """Negate a polynomial modulo q.

        Args:
            a: First polynomial (coefficient vector).
            q: Modulus (or second prime).
    """
    return [(-x) % q for x in a]


def poly_mul_negacyclic(
    a: list[gp.mpz], b: list[gp.mpz], q: gp.mpz
) -> list[gp.mpz]:
    """Multiply two polynomials in the negacyclic ring Z_q[X]/(X^n+1).

        Args:
            a: First polynomial (coefficient vector).
            b: Second polynomial (coefficient vector).
            q: Modulus (or second prime).
    """
    n = len(a)
    ret = [gp.mpz(0) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            idx = i + j
            coeff = a[i] * b[j]
            if idx >= n:
                idx -= n
                coeff = -coeff
            ret[idx] = (ret[idx] + coeff) % q
    return ret


# ---------------------------------------------------------------------------
# Matrix / vector helpers
# ---------------------------------------------------------------------------

def matrix_check_bound(matrix: list[list[int]], bound: gp.mpz) -> bool:
    """Check whether all matrix entries have absolute value within the bound.

        Args:
            matrix: Input matrix.
            bound: Upper bound on absolute values.
    """
    return all(abs(int(v)) <= bound for row in matrix for v in row)


def transpose(matrix: list[list[int | gp.mpz]]) -> list[list[gp.mpz]]:
    if not matrix:
        return []
    return [[gp.mpz(matrix[i][j]) for i in range(len(matrix))] for j in range(len(matrix[0]))]


def mat_vec_mul(matrix: list[list[gp.mpz]], vec: list[gp.mpz], q: gp.mpz) -> list[gp.mpz]:
    """Multiply a matrix by a polynomial vector modulo q.

        Args:
            matrix: Input matrix.
            vec: Input vector.
            q: Modulus (or second prime).
    """
    ret = []
    for row in matrix:
        acc = gp.mpz(0)
        for a_i, b_i in zip(row, vec):
            acc += a_i * b_i
        ret.append(acc % q)
    return ret


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------

def center_matrix(
    matrix: list[list[int]], p: gp.mpz, q: gp.mpz, ring_n: int
) -> list[list[gp.mpz]]:
    """Scale and round matrix entries for LWE decoding.

        Args:
            matrix: Input matrix.
            p: Prime modulus.
            q: Modulus (or second prime).
            ring_n: Polynomial ring dimension.
    """
    centered = [[gp.mpz(0) for _ in range(ring_n)] for _ in range(len(matrix))]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            t = gp.mpz(value) * q
            t = gp.t_div(t, p)
            centered[i][j] = t % q
    return centered


def decode_vector(poly: list[gp.mpz], p: gp.mpz, q: gp.mpz, k: int) -> list[int]:
    """Extract the first k plaintext coefficients by rounding.

        Args:
            poly: Input polynomial coefficient vector.
            p: Prime modulus.
            q: Modulus (or second prime).
            k: Inner-product bound.
    """
    half_q = q // 2
    ret = []
    for value in poly[:k]:
        d = gp.mpz(value) % q
        if d > half_q:
            d -= q
        d = d * p
        d = (d + half_q) // q
        ret.append(int(d))
    return ret
