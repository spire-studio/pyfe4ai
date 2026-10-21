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
import math

import gmpy2 as gp

from pyfe4ai.core.exceptions import warn_at_caller

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
        # root^order == 1 and root^(order/2) == -1 mod q: for a prime q the
        # same roots as root^(order/2) != 1; for a product of primes the
        # condition the negacyclic NTT needs (-1 modulo every prime)
        if gp.powmod(root, order, q) == 1 and gp.powmod(root, order // 2, q) == q - 1:
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

    The backend's :meth:`~pyfe4ai.backend.interface.Backend.ntt`: natively
    where the extension has a context for ``(q, len(a))`` with this root
    (the same values), else ``_ntt_forward_reference``.

        Args:
            a: First polynomial (coefficient vector).
            q: Modulus (or second prime).
            root: Primitive root of unity for NTT.
    """
    from pyfe4ai import backend

    return backend.ops().ntt(a, q, root, False)


def _ntt_forward_reference(a: list[gp.mpz], q: gp.mpz, root: gp.mpz) -> list[gp.mpz]:
    """:func:`ntt_forward` in Python (the reference)."""
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

    The backend's :meth:`~pyfe4ai.backend.interface.Backend.ntt` (natively
    as :func:`ntt_forward`), else ``_ntt_inverse_reference``.

        Args:
            a: First polynomial (coefficient vector).
            q: Modulus (or second prime).
            root: Primitive root of unity for NTT.
    """
    from pyfe4ai import backend

    return backend.ops().ntt(a, q, root, True)


def _ntt_inverse_reference(a: list[gp.mpz], q: gp.mpz, root: gp.mpz) -> list[gp.mpz]:
    """:func:`ntt_inverse` in Python (the reference)."""
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

    For an RNS modulus (:func:`register_rns_modulus`) the root is the CRT of
    the factors' roots (``root^n = -1`` modulo every factor).

        Args:
            q: Modulus (or second prime).
            ring_n: Polynomial ring dimension.
    """
    key = (int(q), ring_n)
    try:
        return _ntt_root_cache[key]
    except KeyError:
        pass
    factors = _rns_factors.get(key[0])
    if factors is None:
        root = _find_primitive_root(q, 2 * ring_n)
    else:
        roots = [_find_primitive_root(gp.mpz(f), 2 * ring_n) for f in factors]
        root = None if None in roots else _crt(roots, factors)
    # the value computed here, not the cache again: register_rns_modulus in
    # another thread may have dropped the entry meanwhile
    _ntt_root_cache[key] = root
    return root


def _crt(residues: list, moduli: tuple) -> gp.mpz:
    """The ``x mod prod(moduli)`` with ``x = residues[i] mod moduli[i]``."""
    x, m = gp.mpz(0), gp.mpz(1)
    for r, f in zip(residues, moduli):
        x += m * ((gp.mpz(r) - x) * gp.invert(m, f) % f)
        m *= f
    return x


# ---------------------------------------------------------------------------
# RNS moduli: q = q1 q2, a product of two NTT-friendly primes
# ---------------------------------------------------------------------------

#: ``ring_modulus`` of the selectively secure BMMS21 schemes (MIFE / MCFE /
#: dMCFE Ring-LWE): ``"prime"`` (default) is an NTT-friendly prime, ``"rns"``
#: a product of two (:func:`rns_ntt_modulus`; see docs/rns_modulus_review.md).
RING_MODULI = ("prime", "rns")
#: Largest bit length of an RNS factor (64-bit words of the native code, whose
#: lazy butterflies need 4 q_i < 2^64).
RNS_MAX_FACTOR_BITS = 62
#: Most factors of an RNS modulus that the native 256-bit ring holds (an
#: NTT-domain value keeps one residue per 64-bit limb); moduli with more run on
#: the Python path.
RNS_NATIVE_FACTORS = 4

#: Known factorisations of RNS moduli, ``q -> (q1, q2)`` (see :func:`register_rns_modulus`).
_rns_factors: dict[int, tuple] = {}


def check_ring_modulus(value) -> str:
    """Return *value* if it is one of :data:`RING_MODULI`."""
    if value not in RING_MODULI:
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError("ring_modulus must be one of {}, got {!r}".format(RING_MODULI, value))
    return value


def register_rns_modulus(q, factors) -> tuple:
    """Record that *q* is the product of the distinct primes *factors*.

    The NTT root (``_get_ntt_root``) and the native context of an RNS
    modulus come from its factors, which the public parameters carry
    (``q_factors``); without them a composite ``q`` has no NTT here.

    Returns:
        The factors as a tuple of ``mpz``.

    Raises:
        FEValidationError: Unless the factors are at least two distinct
            primes whose product is *q*.
    """
    q = int(q)
    try:
        fs = tuple(gp.mpz(int(f)) for f in factors)
    except (TypeError, ValueError) as exc:
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError("q_factors must be integers") from exc
    if len(fs) < 2 or len(set(fs)) != len(fs) or math.prod(int(f) for f in fs) != q or not all(
        f > 2 and gp.is_prime(f) for f in fs
    ):
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError("q_factors must be distinct odd primes whose product is q")
    if _rns_factors.get(q) != fs:
        _rns_factors[q] = fs
        for cache in (_ntt_root_cache, RingNTT._cache):  # computed without the factors
            # a snapshot of the keys (other threads may add entries meanwhile);
            # pop: another thread registering q may have dropped one already
            for key in [k for k in list(cache) if k[0] == q]:
                cache.pop(key, None)
    return fs


def rns_factors(q) -> tuple | None:
    """The registered factors of *q* (:func:`register_rns_modulus`), or ``None``."""
    return _rns_factors.get(int(q))


def prev_ntt_prime(upper_bound, ring_n: int) -> gp.mpz | None:
    """The largest prime ``q <= upper_bound`` with ``q = 1 (mod 2 ring_n)``, or ``None``."""
    modulus = 2 * ring_n
    candidate = int(upper_bound) - (int(upper_bound) - 1) % modulus
    while candidate > modulus:
        if gp.is_prime(candidate):
            return gp.mpz(candidate)
        candidate -= modulus
    return None


def rns_ntt_modulus(lower_bound, ring_n: int, p, factors: int = 2) -> tuple:
    """``(q, (q_1, ..., q_k))``: ``q = q_1 ... q_k >= lower_bound``, a product of
    ``k = factors`` distinct NTT primes, in increasing order.

    ``k - 1`` factors are the largest NTT primes ``<= floor(lower_bound^(1/k))``
    (descending), the last the smallest NTT prime with a product
    ``>= lower_bound`` that is not one of them, advanced while
    ``gcd(floor(q / p), q) != 1`` (``Delta`` is then a unit, as for a prime
    ``q``; see the review). For ``k = 2``: ``q1`` the largest NTT prime
    ``<= sqrt(lower_bound)``, ``q2`` the smallest with ``q1 q2 >=
    lower_bound``, as in earlier versions. Deterministic, so a cached record
    can be checked.

    Raises:
        FEValidationError: If a factor would exceed :data:`RNS_MAX_FACTOR_BITS`
            bits or the bound is too small for *factors* factors.
    """
    from pyfe4ai.core.exceptions import FEValidationError

    lower, k = gp.mpz(lower_bound), int(factors)
    if k < 2:
        raise FEValidationError("an RNS modulus has at least two factors")
    chosen, top = [], gp.iroot(lower, k)[0]
    for _ in range(k - 1):
        f = prev_ntt_prime(top, ring_n)
        if f is None:
            raise FEValidationError("q is too small for an RNS modulus of {} factors at ring_n = {}".format(k, ring_n))
        chosen.append(f)
        top = f - 1
    rest = math.prod(int(f) for f in chosen)
    last = next_ntt_prime(-(-lower // rest), ring_n)
    while last in chosen or gp.gcd((rest * last) // gp.mpz(p), rest * last) != 1:
        last = next_ntt_prime(last + 1, ring_n)
    if int(last).bit_length() > RNS_MAX_FACTOR_BITS:
        raise FEValidationError(
            "an RNS modulus of {} factors below 2^{} is too small here (q needs {} bits)".format(
                k, RNS_MAX_FACTOR_BITS, int(lower).bit_length()
            )
        )
    return gp.mpz(rest) * last, tuple(sorted(chosen + [last]))


def rns_factor_count(lower_bound, min_factors: int = 2) -> int:
    """The number of RNS factors of :func:`rns_ntt_modulus` for a modulus of
    the bit length of *lower_bound*: *min_factors* (two), or more so that
    each is at most 60 bits (four for the 212 to 218 bits of the flooded
    threshold moduli) -- but :data:`RNS_NATIVE_FACTORS` (four, of up to
    :data:`RNS_MAX_FACTOR_BITS` bits) up to 248 bits, the most the native
    ring holds, rather than five."""
    bits = int(gp.mpz(lower_bound)).bit_length()
    k = max(int(min_factors), -(-bits // 60))
    if k > RNS_NATIVE_FACTORS and bits <= RNS_NATIVE_FACTORS * RNS_MAX_FACTOR_BITS:
        k = max(RNS_NATIVE_FACTORS, int(min_factors))
    return k


def rns_ntt_modulus_for(lower_bound, ring_n: int, p, min_factors: int = 2) -> tuple:
    """:func:`rns_ntt_modulus` with :func:`rns_factor_count` factors (at
    least *min_factors*), or one more where the last factor would exceed
    :data:`RNS_MAX_FACTOR_BITS` bits (bounds just below ``2^248`` with four
    factors). Deterministic.

    Raises:
        FEValidationError: As :func:`rns_ntt_modulus` for both counts.
    """
    from pyfe4ai.core.exceptions import FEValidationError

    k = rns_factor_count(lower_bound, min_factors)
    try:
        return rns_ntt_modulus(lower_bound, ring_n, p, k)
    except FEValidationError:
        return rns_ntt_modulus(lower_bound, ring_n, p, k + 1)


def poly_mul(
    a: list[gp.mpz], b: list[gp.mpz], q: gp.mpz
) -> list[gp.mpz]:
    """Polynomial multiplication with automatic NTT acceleration.

    The backend's :meth:`~pyfe4ai.backend.interface.Backend.poly_mul`:
    natively where the extension has a context for ``(q, len(a))`` (the
    same values), else ``_poly_mul_reference``.

        Args:
            a: First polynomial (coefficient vector).
            b: Second polynomial (coefficient vector).
            q: Modulus (or second prime).
    """
    from pyfe4ai import backend

    return backend.ops().poly_mul(a, b, q)


def _poly_mul_reference(a: list[gp.mpz], b: list[gp.mpz], q: gp.mpz, root) -> list[gp.mpz]:
    """:func:`poly_mul` in Python with the root :func:`_get_ntt_root` found
    for ``(q, len(a))`` (the NTT, or the naive product with a warning)."""
    n = len(a)
    if root is not None:
        return poly_mul_ntt(a, b, q, root)
    logger.warning(
        "NTT root not found for q=%s, n=%d; falling back to O(n²) "
        "negacyclic multiplication. Consider using next_ntt_prime() "
        "to choose an NTT-friendly prime.",
        q, n,
    )
    warn_at_caller(
        "NTT root not found for q={}, n={}; falling back to O(n²) negacyclic multiplication. Consider using "
        "next_ntt_prime() to choose an NTT-friendly prime.".format(q, n),
    )
    return poly_mul_negacyclic(a, b, q)


class RingNTT:
    """NTT-domain arithmetic of ``Z_q[x] / (x^n + 1)`` with :func:`poly_mul`'s root.

    ``inverse(mul(forward(a), forward(b)))`` equals ``poly_mul(a, b, q)``, and
    the NTT is linear mod q, so a fixed operand can be transformed once and
    products summed in the NTT domain before a single inverse transform, with
    the same result as summing the :func:`poly_mul` products mod q.
    :meth:`get` returns ``None`` when ``q`` has no primitive ``2n``-th root of
    unity (then :func:`poly_mul` falls back to the naive product).
    """

    _cache: dict = {}

    def __init__(self, q, ring_n: int, root) -> None:
        self.q = gp.mpz(q)
        self.ring_n = ring_n
        self.root = root

    @classmethod
    def get(cls, q, ring_n: int) -> "RingNTT | None":
        key = (int(q), ring_n)
        try:
            return cls._cache[key]
        except KeyError:
            pass
        root = _get_ntt_root(gp.mpz(q), ring_n)
        ntt = None if root is None else cls(q, ring_n, root)
        cls._cache[key] = ntt
        return ntt  # not the cache again (see _get_ntt_root)

    def forward(self, a: list) -> list:
        """NTT of the coefficient vector *a* (integers, any sign)."""
        return ntt_forward(a, self.q, self.root)

    def inverse(self, a_hat: list) -> list:
        """Coefficients, in ``[0, q)``, of the NTT-domain vector *a_hat*."""
        return ntt_inverse(a_hat, self.q, self.root)

    def mul(self, a_hat: list, b_hat: list) -> list:
        """Pointwise product mod q (the NTT of the negacyclic product)."""
        q = self.q
        return [(x * y) % q for x, y in zip(a_hat, b_hat)]

    def mul_add(self, acc: list | None, a_hat: list, b_hat: list) -> list:
        """``acc + a_hat * b_hat`` pointwise mod q (``acc = None``: just the product)."""
        if acc is None:
            return self.mul(a_hat, b_hat)
        q = self.q
        return [(s + x * y) % q for s, x, y in zip(acc, a_hat, b_hat)]


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
    """Encode the rows of a signed matrix as polynomials ``floor(q / p) x mod q``.

    Row ``i`` becomes a polynomial whose first coefficients hold the entries
    of the row. The encoding is linear in ``x``, as the BMMS21 proofs need
    (versions <= 0.2.0 used ``trunc(x q / p)``).

    Args:
        matrix: Input matrix.
        p: Plaintext modulus.
        q: Ciphertext modulus.
        ring_n: Polynomial ring dimension.
    """
    delta = gp.mpz(q) // gp.mpz(p)
    centered = [[gp.mpz(0) for _ in range(ring_n)] for _ in range(len(matrix))]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            centered[i][j] = delta * gp.mpz(value) % q
    return centered


def seeded_polynomial(seed: str, q, ring_n: int, index: int = 0) -> list[gp.mpz]:
    """Uniform element of ``R_q`` expanded from a hex *seed* with SHAKE-256.

    The Ring-LWE schemes publish their public polynomial ``a`` this way: a
    parameter cache stores only the seed, so a planted cache cannot choose
    ``a`` (e.g. ``a = 0``), only another pseudo-random one.
    """
    from pyfe4ai.primitives.hashing import hash_to_zq_vector

    bytes.fromhex(seed)  # a hex string
    return hash_to_zq_vector(seed, q, ring_n, domain="pyfe4ai/ring-lwe/a|{}".format(int(index)))


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


#: Default ring dimension. A toy value chosen for speed; nothing here
#: estimates the hardness of Ring-LWE at the derived (n, q, sigma), and BMMS21
#: (Table 1) uses n >= 2048 for comparable settings.
DEFAULT_RING_N = 32


def check_key_rows(rows: dict, eta: int, ring_n: int, what: str) -> None:
    """Raise unless every entry of *rows* (a key field's name -> its rows, e.g.
    an encryption key's ``pk`` and pad ``u``) is a list of at least *eta* rows
    whose first *eta* are lists of *ring_n* values. With fewer rows or rows of
    another length the encryption and key derivation zipped the key away:
    short ciphertexts, keys that decrypt to wrong values (on the Python path),
    or an ``IndexError`` / ``ValueError`` deep in a backend.

    Args:
        rows: ``{field name: rows}``.
        eta: Rows a key field must have.
        ring_n: Values per row.
        what: The fields as named in the message (``"pk or u"``); constant.

    Raises:
        FEKeyError: With a constant message (no key data).
    """
    from pyfe4ai.core.exceptions import FEKeyError

    if any(not isinstance(value, (list, tuple)) or len(value) < eta for value in rows.values()):
        raise FEKeyError("the encryption key has fewer than eta rows of " + what)
    if any(not isinstance(row, (list, tuple)) or len(row) != ring_n for value in rows.values() for row in value[:eta]):
        raise FEKeyError("the encryption key has a row of " + what + " that is not ring_n values")


def check_ring_n(ring_n) -> int:
    """Return *ring_n* if it is a power of two >= 2 (``x^n + 1`` must be the
    cyclotomic polynomial of BMMS21 and the NTT needs ``2n | q - 1``)."""
    if isinstance(ring_n, bool) or not isinstance(ring_n, int) or ring_n < 2 or ring_n & (ring_n - 1):
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError("ring_n must be a power of two >= 2, got {!r}".format(ring_n))
    return ring_n


#: For each error width ``sigma`` (outer key) and ring dimension ``n`` (inner
#: key): the largest ``k`` such that LWE in dimension ``n`` with ``q = 2^k``,
#: secret and errors ``D_{Z, sigma}`` and unlimited samples costs at least
#: 2^128 operations by the lattice estimator (commit 53da598, MATZOV cost
#: model; primal uSVP, BDD and BDD hybrid, dual and dual hybrid). Ring-LWE is
#: estimated as plain LWE, as in the Homomorphic Encryption Standard, whose
#: table (ternary secret, ``sigma = 3.2``) this estimator puts at about 125
#: bits. A scheme is held to the row of the largest width not above the
#: smallest width among the RLWE problems its proof relies on: wider
#: Gaussians only make the attacks more expensive. Regenerate with
#: ``tools/params/estimate_ring_lwe.py``.
#:
#: :meta hide-value:
RING_LWE_128_LOG_Q = {
    2.0: {1024: 27, 2048: 53, 4096: 107, 8192: 214, 16384: 431, 32768: 868},
    3.0: {1024: 28, 2048: 55, 4096: 108, 8192: 216, 16384: 432, 32768: 870},
    4.0: {1024: 29, 2048: 55, 4096: 109, 8192: 216, 16384: 433, 32768: 870},
    6.0: {1024: 30, 2048: 57, 4096: 110, 8192: 218, 16384: 434, 32768: 872},
    8.0: {1024: 31, 2048: 57, 4096: 111, 8192: 218, 16384: 435, 32768: 872},
    12.0: {1024: 32, 2048: 58, 4096: 112, 8192: 220, 16384: 436, 32768: 874},
    16.0: {1024: 33, 2048: 59, 4096: 113, 8192: 221, 16384: 437, 32768: 874},
}
_RING_DIMS = sorted(RING_LWE_128_LOG_Q[min(RING_LWE_128_LOG_Q)])


def ring_lwe_bound(ring_n: int, sigma) -> int | None:
    """Largest 128-bit ``log2 q`` of :data:`RING_LWE_128_LOG_Q` for *ring_n* and
    the row of the largest tabulated width not above *sigma*, or ``None``."""
    widths = [w for w in RING_LWE_128_LOG_Q if w <= sigma]
    if not widths or ring_n not in RING_LWE_128_LOG_Q[max(widths)]:
        return None
    return RING_LWE_128_LOG_Q[max(widths)][ring_n]


def check_ring_lwe(ring_n: int, q, sigma) -> None:
    """Raise unless Ring-LWE with ``(ring_n, q)`` and width *sigma* meets
    :data:`RING_LWE_128_LOG_Q`."""
    bound = ring_lwe_bound(ring_n, sigma)
    if bound is None or int(q).bit_length() > bound:
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError(
            "ring_n = {} with a {}-bit q and RLWE width {:.2f} is below 128-bit security "
            "(lattice estimator: log2 q <= {} here; tabulated for ring_n in {} and widths "
            ">= {})".format(
                ring_n, int(q).bit_length(), float(sigma), "-" if bound is None else bound,
                _RING_DIMS, min(RING_LWE_128_LOG_Q),
            )
        )


def ring_lwe_auto_n(params_for) -> int:
    """Smallest tabulated ring dimension ``n`` whose ``params_for(n) = (q, sigma)``
    meets :data:`RING_LWE_128_LOG_Q` (the modulus grows with the dimension)."""
    for ring_n in _RING_DIMS:
        q, sigma = params_for(ring_n)
        bound = ring_lwe_bound(ring_n, sigma)
        if bound is not None and int(q).bit_length() <= bound:
            return ring_n
    from pyfe4ai.core.exceptions import FEValidationError

    raise FEValidationError(
        "no ring dimension up to {} reaches 128-bit security for these bounds; "
        "reduce bound_x, bound_y, eta or the number of clients".format(_RING_DIMS[-1])
    )


def resolve_ring_n(ring_n, params_for) -> int:
    """Ring dimension of a config entry: ``"auto"`` selects the smallest one
    meeting :data:`RING_LWE_128_LOG_Q` (see :func:`ring_lwe_auto_n`; *params_for*
    maps a dimension to ``(q, rlwe_width)``), ``None`` gives :data:`DEFAULT_RING_N`."""
    if ring_n == "auto":
        return ring_lwe_auto_n(params_for)
    return check_ring_n(ring_n or DEFAULT_RING_N)


def bmms21_selective_parameters(sec_param: int, eta: int, bound_x, bound_y, ring_n: int, *,
                                security_bound_x=None, ring_modulus: str = "prime") -> dict:
    """Parameters of the selectively secure BMMS21 IPFE (Sec. 4) per Sec. 4.2.

    For ``l = eta`` coordinates bounded by ``X`` (messages) and ``Y``
    (weights), ``kappa = sec_param`` and ``n = ring_n``, with
    ``eta_eps(Z^d) = sqrt(ln(2 d (1 + 1/eps)) / pi)``:

    * G1 -> G2 (mhe-RLWE from RLWE of width ``sigma``, Theorem 1):
      ``sigma = 1.5 eta_eps(Z^(n + n l))`` with ``eps = 2^-kappa`` (at least
      3.2) and ``sigma2 = sqrt(2) sigma n C sqrt(l + 2)``, ``C = sqrt(kappa) sigma1``
      (Lemma 2), so that ``sigma sqrt(1 - (sigma n C sqrt(l + 2) / sigma2)^2)
      > eta``; ``sigma3 = sqrt(2) sigma2``;
    * G3 -> G4 (Lemma 1 with ``||alpha|| <= 2 X sqrt(l)``):
      ``sigma1 = sqrt(2) 2 X sqrt(l) sigma'``, so that ``Gamma >= sigma' / sqrt(2)``,
      and ``sigma' = sqrt(2) eta_eps'(Z^n)`` with ``eps' = 2^-kappa / n``;
      G5 -> G6 then needs RLWE with secrets and errors of width ``sigma'``;
    * correctness: ``l Y (2 n kappa sigma1 sigma2 + sqrt(kappa) sigma3) < q / 2p``
      with ``p = 2 l X Y + 2`` (signed inner products ``|<x, y>| <= l X Y``
      strictly inside ``(-p/2, p/2)``); ``q`` is the first NTT-friendly prime
      above it.

    ``security_bound_x`` (default ``bound_x``) is the message bound ``X`` of
    the G3 -> G4 condition, ``bound_x`` the one of decryption (``p``): the
    MIFE compiler of ACFGU18 (Fig. 4) needs an IPFE that is secure for
    messages of norm below ``3X`` (its Lemma 3 queries the IPFE with
    ``x^j - x^1 + x'^1``), although only ``X``-bounded results are decoded.

    ``ring_modulus="rns"`` replaces the prime by the product of two NTT
    primes above the same bound (:func:`rns_ntt_modulus`).

    Versions <= 0.2.0 used ``sigma = 1`` and ``sigma1 = 2 sqrt(l) X`` (the
    paper's Table 1 has the same gap), which miss both smoothing conditions.

    Returns:
        Dict with ``p``, ``q``, ``sigma``, ``sigma_prime``, ``sigma1``,
        ``sigma2``, ``sigma3`` and ``rlwe_width`` (``min(sigma, sigma')``, the
        width the 128-bit table is read at); with ``"rns"`` also
        ``q_factors``.
    """
    l = int(eta)
    n = int(ring_n)
    kappa = float(sec_param)
    sigma = max(3.2, 1.5 * math.sqrt(_smoothing_log(n + n * l, kappa) / math.pi))
    sigma_prime = math.sqrt(2.0) * math.sqrt(_smoothing_log(n, kappa + math.log2(n)) / math.pi)
    sec_x = bound_x if security_bound_x is None else security_bound_x
    sigma1 = math.sqrt(2.0) * 2.0 * float(sec_x) * math.sqrt(l) * sigma_prime
    sigma2 = math.sqrt(2.0) * sigma * n * math.sqrt(kappa) * sigma1 * math.sqrt(l + 2)
    sigma3 = math.sqrt(2.0) * sigma2
    p = gp.mpz(bound_x) * gp.mpz(bound_y) * gp.mpz(2 * l) + 2
    noise = float(l) * float(bound_y) * (2.0 * n * kappa * sigma1 * sigma2 + math.sqrt(kappa) * sigma3)
    lower = gp.mpz(int(noise) + 1) * 2 * p + 1
    if check_ring_modulus(ring_modulus) == "rns":
        q, factors = rns_ntt_modulus(lower, n, p)
    else:
        q, factors = next_ntt_prime(lower, n), None
    out = {
        "p": p, "q": q, "sigma": sigma, "sigma_prime": sigma_prime,
        "sigma1": sigma1, "sigma2": sigma2, "sigma3": sigma3,
        "rlwe_width": min(sigma, sigma_prime),
    }
    if factors is not None:
        out["q_factors"] = factors
    return out


def bmms21_parameters(sec_param: int, eta: int, bound_x, bound_y, ring_n: int, *,
                      security_bound_x=None, ring_modulus: str = "prime") -> tuple:
    """``(p, q, sigma1, sigma2, sigma3)`` of :func:`bmms21_selective_parameters`."""
    P = bmms21_selective_parameters(sec_param, eta, bound_x, bound_y, ring_n,
                                    security_bound_x=security_bound_x, ring_modulus=ring_modulus)
    return P["p"], P["q"], P["sigma1"], P["sigma2"], P["sigma3"]


# ---------------------------------------------------------------------------
# Adaptively secure IPFE of BMMS21 Sec. 5 (parameters of Appendix C.1)
# ---------------------------------------------------------------------------

#: Smallest number ``m`` of key polynomials for which the constraints of
#: BMMS21 Appendix C.1 can hold (see :func:`bmms21_adaptive_log2_q`).
BMMS21_ADAPTIVE_MIN_M = 5
#: Largest ``m`` that ``m = "auto"`` considers.
BMMS21_ADAPTIVE_MAX_M = 64


def _smoothing_log(dim: int, log2_inv_eps: float) -> float:
    """``ln(2 dim (1 + 1/eps))`` for ``eps = 2^-log2_inv_eps`` (up to a negligible term)."""
    return math.log(2 * dim) + log2_inv_eps * math.log(2)


def bmms21_adaptive_leveraging(eta: int, bound_x, ring_n: int, packed: bool = False) -> tuple:
    """``(log2 L, D)``: the complexity-leveraging factor ``L`` of BMMS21 Sec. 5 and
    the rank ``D`` of the challenge differences that games G3* - G5* decompose.

    One vector per ciphertext (``packed=False``): ``L = (2X + 1)^(2l)``, the
    challenge pairs, and ``D = 1``. Up to ``ring_n`` vectors per ciphertext
    (``packed=True``, Remark 1): the statistical games depend on the
    challenge only through ``V = span(alpha^1, ..., alpha^k)`` of the
    differences ``alpha^j = x1^j - x0^j`` (the keys they answer are those
    with ``y`` orthogonal to ``V``), so guessing ``V`` suffices. ``V`` has a
    dimension ``d <= D = min(l, ring_n)`` and is spanned by ``d`` vectors of
    ``[-2X, 2X]^l``, so ``L = (D + 1) (4X + 1)^(l D)``, independent of the
    number of packed vectors (guessing both challenge matrices would give
    ``(2X + 1)^(2lk)``). For ``l = 1``, ``V`` is ``{0}`` or ``Q``: ``L = 2``.
    """
    l = int(eta)
    if not packed:
        return 2 * l * math.log2(2 * int(bound_x) + 1), 1
    if l == 1:
        return 1.0, 1
    rank = min(l, int(ring_n))
    return l * rank * math.log2(4 * int(bound_x) + 1) + math.log2(rank + 1), rank


def _bmms21_basis_norm_log2(eta: int, bound_x, ring_n: int, packed: bool = False) -> float:
    """log2 of the bound on ``||B||`` of G3 -> G4: the basis ``B`` along which
    Lemma 1 decomposes the key columns. ``||alpha|| <= 2X sqrt(l)`` for one
    difference, ``2X sqrt(l D)`` for ``D`` basis vectors of ``[-2X, 2X]^l``;
    packed with ``l = 1``, ``V = Q`` has the basis ``[1]`` (norm 1)."""
    l = int(eta)
    if packed and l == 1:
        return 0.0
    rank = bmms21_adaptive_leveraging(l, bound_x, ring_n, packed)[1]
    return math.log2(2 * int(bound_x) * math.sqrt(l * rank))


def _bmms21_packed_minor_log2(eta: int, bound_x, ring_n: int) -> float:
    """log2 of the Hadamard bound ``(2X sqrt(D))^D`` on the ``d x d`` minors of a
    basis of the differences (``d <= D = min(l, ring_n)``, entries at most ``2X``).

    Game G5* writes every difference as a rational combination of the basis;
    the denominators divide such a minor, so they are invertible mod the
    prime ``q`` when ``q`` exceeds this bound. ``q`` is public: below it an
    adversary could pick differences whose minor is a multiple of ``q``.
    """
    if int(eta) == 1:
        return 0.0  # the basis [1]: integer coefficients
    rank = min(int(eta), int(ring_n))
    return rank * math.log2(2 * int(bound_x) * math.sqrt(rank))


def _bmms21_adaptive_widths(kappa: int, l: int, bound_x, bound_y, ring_n: int, m: int,
                            log2_q: float, packed: bool = False) -> dict:
    """log2 of the Gaussian widths of BMMS21 Sec. 5 for a modulus of ``log2_q`` bits.

    Signed inputs ``|x| <= X``, ``|y| <= Y``: ``alpha = x1 - x0`` has entries of
    at most ``2X``; complexity leveraging ranges over ``L`` and the differences
    have rank at most ``D`` (:func:`bmms21_adaptive_leveraging`; ``(2X + 1)^(2l)``
    and 1 for one vector per ciphertext).

    * Theorem 1 (mhe-RLWE from RLWE with errors ``D_sigma``):
      ``sigma = 1.5 eta_eps(Z^(n + n l))`` with ``eps = 2^-kappa`` (at least
      3.2) and ``sigma2 = sqrt(2) sigma n C sqrt(l + 2)``, ``C = sqrt(kappa) sigma1``,
      so that ``sigma sqrt(1 - (sigma n C sqrt(l + 2) / sigma2)^2) > eta``;
    * Theorem 2 (Ring-LHL, ``k = 2``):
      ``sigma' = sqrt(n ln(2 m n (1 + 1/delta)) / pi) q^(2/m + eps/2)`` with
      ``(2 D delta + 2^n q^(-eps n) (q^2m / ((q^m - 1)(q^m - q)))^n) L <= 2^-kappa``
      (``D`` applications, one per basis vector of the differences);
    * G3 -> G4 (Lemma 1 for ``sigma1^2 I - sigma'^2 B B^T``, ``B`` a basis of the
      differences): ``sigma1 = sqrt(2) ||B|| sigma'`` with ``||B|| <= 2X sqrt(l D)``
      (1 when packed with ``l = 1``, see ``_bmms21_basis_norm_log2``), so that
      ``Gamma >= sigma' / sqrt(2) > eta_eps'(Z^n)``;
    * ``sigma3 = sqrt(m) sigma2``.
    """
    n = ring_n
    log2_lev, rank = bmms21_adaptive_leveraging(l, bound_x, n, packed)
    # 1.5 eta: Theorem 1 needs sigma sqrt(1 - 1/2) strictly above eta
    sigma = max(3.2, 1.5 * math.sqrt(_smoothing_log(n + n * l, kappa) / math.pi))
    # one bit of slack each for the factor (q^2m / ((q^m - 1)(q^m - q)))^n of the
    # bad-matrix fraction and for ln(1 + 1/delta) ~ ln(1/delta), and log2 D for the
    # D applications of Theorem 2
    log2_inv_delta = kappa + 3 + log2_lev + math.log2(rank)
    log2_q_eps = (n + kappa + 2 + log2_lev) / n  # eps log2 q, from 2^n q^(-eps n)
    log_sp = (
        0.5 * math.log2(n * _smoothing_log(m * n, log2_inv_delta) / math.pi)
        + 2 * log2_q / m + log2_q_eps / 2
    )
    # Lemma 1 also needs sigma' / sqrt(2) above eta_eps'(Z^n), eps' = 2^-kappa / (n m lev)
    eta_g = math.sqrt(_smoothing_log(n, kappa + math.log2(n * m) + log2_lev) / math.pi)
    log_sp = max(log_sp, math.log2(math.sqrt(2) * eta_g) + 1e-9)
    log_s1 = 0.5 + _bmms21_basis_norm_log2(l, bound_x, n, packed) + log_sp
    log_s2 = 0.5 + math.log2(sigma * n * math.sqrt(kappa) * math.sqrt(l + 2)) + log_s1
    return {
        "sigma": math.log2(sigma),
        "sigma_lhl": log_sp,
        "sigma1": log_s1,
        "sigma2": log_s2,
        "sigma3": log_s2 + 0.5 * math.log2(m),
    }


def _bmms21_adaptive_slack(kappa: int, l: int, bound_x, bound_y, ring_n: int, m: int,
                           log2_q: float, packed: bool = False) -> float:
    """``log2(q / 2p) - log2(noise)`` (correctness of Appendix C.1, positive when met)."""
    w = _bmms21_adaptive_widths(kappa, l, bound_x, bound_y, ring_n, m, log2_q, packed)
    p = 2 * l * int(bound_x) * int(bound_y) + 2
    # l Y (sqrt(kappa) sigma3 + m n kappa sigma1 sigma2) = l Y sigma2 (2^u + 2^v), in log2
    u = 0.5 * math.log2(kappa) + w["sigma3"] - w["sigma2"]
    v = math.log2(m * ring_n * kappa) + w["sigma1"]
    log_sum = max(u, v) + math.log2(1 + 2.0 ** -abs(u - v))
    log_noise = math.log2(l * int(bound_y)) + w["sigma2"] + log_sum
    return log2_q - math.log2(2 * p) - log_noise


def bmms21_adaptive_log2_q(sec_param: int, eta: int, bound_x, bound_y, ring_n: int,
                           m: int, packed: bool = False) -> float | None:
    """Smallest ``log2 q`` meeting the constraints of BMMS21 Appendix C.1, or ``None``.

    All widths grow like ``q^(2/m)`` (the Ring-LHL bound on ``sigma'``) and the
    correctness noise like ``q^(4/m)``, so a modulus exists only for ``m > 4``;
    the paper's ``m = 3`` has none. *packed*: up to ``ring_n`` vectors per
    ciphertext (see :func:`bmms21_adaptive_leveraging`); ``q`` must then also
    exceed twice the bound of ``_bmms21_packed_minor_log2`` (a larger ``q`` only
    widens the correctness margin, since the noise grows like ``q^(4/m)``).
    """
    args = (int(sec_param), int(eta), bound_x, bound_y, int(ring_n), int(m))
    if m <= 4 or _bmms21_adaptive_slack(*args, 4000.0, packed) <= 0:
        return None
    lo, hi = 1.0, 4000.0
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        if _bmms21_adaptive_slack(*args, mid, packed) > 0:
            hi = mid
        else:
            lo = mid
    if packed:
        hi = max(hi, _bmms21_packed_minor_log2(eta, bound_x, ring_n) + 1)
    return hi


def bmms21_adaptive_parameters(sec_param: int, eta: int, bound_x, bound_y, ring_n: int,
                               m: int, packed: bool = False) -> dict:
    """Parameters of the adaptively secure BMMS21 IPFE (Sec. 5) for ``m`` key polynomials
    (*packed*: up to ``ring_n`` vectors per ciphertext).

    Returns:
        Dict with ``p`` (``2 l X Y + 2``), ``q`` (NTT-friendly prime), ``m``,
        and the widths ``sigma`` (RLWE), ``sigma_lhl`` (``sigma'``),
        ``sigma1``, ``sigma2``, ``sigma3``; see ``_bmms21_adaptive_widths``.

    Raises:
        FEValidationError: If no modulus meets the constraints (``m <= 4``).
    """
    log2_q = bmms21_adaptive_log2_q(sec_param, eta, bound_x, bound_y, ring_n, m, packed)
    if log2_q is None:
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError(
            "m = {} key polynomials admit no modulus: the Ring-LHL width sigma' grows like "
            "q^(2/m) and the decryption noise like q^(4/m), so m must be at least {}".format(
                m, BMMS21_ADAPTIVE_MIN_M
            )
        )
    q = next_ntt_prime(gp.mpz(2) ** math.ceil(log2_q), ring_n)
    exact = math.log2(int(q))
    if _bmms21_adaptive_slack(int(sec_param), int(eta), bound_x, bound_y, ring_n, m, exact, packed) <= 0:
        raise AssertionError("modulus does not meet the correctness bound")  # q only grew
    if packed and exact <= _bmms21_packed_minor_log2(eta, bound_x, ring_n):
        raise AssertionError("modulus does not exceed the minors of the differences")
    widths = _bmms21_adaptive_widths(int(sec_param), int(eta), bound_x, bound_y, ring_n, m, exact, packed)
    if max(widths.values()) > 1000:
        from pyfe4ai.core.exceptions import FEValidationError

        raise FEValidationError("these bounds need Gaussian widths beyond 2^1000; reduce them")
    out = {key: 2.0 ** value for key, value in widths.items()}
    out.update({"p": gp.mpz(2 * int(eta) * int(bound_x) * int(bound_y) + 2), "q": q, "m": int(m)})
    return out


def bmms21_adaptive_choice(sec_param: int, eta: int, bound_x, bound_y, ring_n, m,
                           standard: bool, packed: bool = False) -> dict:
    """Ring dimension and ``m`` of the adaptive BMMS21 IPFE, then its parameters.

    ``ring_n = "auto"`` tries the dimensions of :data:`RING_LWE_128_LOG_Q` in
    increasing order (``None`` gives :data:`DEFAULT_RING_N`); ``m = "auto"``
    takes the smallest ``m >= 5`` that admits a modulus -- one meeting
    :data:`RING_LWE_128_LOG_Q` at the RLWE width ``sigma`` when *standard* is
    set and ``ring_n`` or ``m`` is ``"auto"``. A small ``m`` keeps the ``l m``
    key polynomials and the ``m``-polynomial functional keys small; a larger
    one lowers ``q`` (``sigma'`` grows like ``q^(2/m)``) and can be passed
    explicitly. *packed*: parameters for up to ``ring_n`` vectors per ciphertext.
    """
    from pyfe4ai.core.exceptions import FEValidationError

    dims = list(_RING_DIMS) if ring_n == "auto" else [check_ring_n(ring_n or DEFAULT_RING_N)]
    if m == "auto":
        ms = range(BMMS21_ADAPTIVE_MIN_M, BMMS21_ADAPTIVE_MAX_M + 1)
    else:
        ms = [int(m)]
    for n in dims:
        for mm in ms:
            log2_q = bmms21_adaptive_log2_q(sec_param, eta, bound_x, bound_y, n, mm, packed)
            if log2_q is None:
                continue
            bits = math.ceil(log2_q) + 1  # bits of the NTT prime just above 2^ceil(log2_q)
            # an explicit (ring_n, m) is left to the caller's check, which explains the refusal
            if standard and (ring_n == "auto" or m == "auto"):
                widths = _bmms21_adaptive_widths(sec_param, eta, bound_x, bound_y, n, mm, log2_q, packed)
                bound = ring_lwe_bound(n, 2.0 ** widths["sigma"])
                if bound is None or bits > bound:
                    continue
            params = bmms21_adaptive_parameters(sec_param, eta, bound_x, bound_y, n, mm, packed)
            params["ring_n"] = n
            return params
    if m != "auto":
        bmms21_adaptive_parameters(sec_param, eta, bound_x, bound_y, dims[0], int(m), packed)  # raises
    where = "ring dimension up to {}".format(dims[-1]) if ring_n == "auto" else "ring_n = {}".format(dims[0])
    raise FEValidationError(
        "no {} and m up to {} reaches 128-bit security for these bounds; reduce bound_x, "
        "bound_y or eta".format(where, BMMS21_ADAPTIVE_MAX_M)
    )
