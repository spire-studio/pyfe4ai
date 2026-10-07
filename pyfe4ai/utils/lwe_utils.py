"""LWE parameter derivation, vector centering, and inner-product decoding."""

from __future__ import annotations

import math
import random

import gmpy2 as gp

from pyfe4ai.utils.exceptions import FEValidationError

_CSPRNG = random.SystemRandom()


SIGMA_CDT = math.sqrt(1.0 / (2.0 * math.log(2.0)))


def derive_lwe_parameters(
    max_plain_bound: gp.mpz,
    bound_y: gp.mpz,
    eta: int,
    lwe_n: int,
    extra_dimension: int | None = None,
) -> tuple[gp.mpz, gp.mpz, int, float, gp.mpz]:
    """Derive LWE scheme parameters from plaintext and weight bounds.

        Args:
            max_plain_bound: Maximum plaintext bound.
            bound_y: Bound on weight values.
            eta: Inner-product vector dimension.
            lwe_n: LWE lattice dimension.
            extra_dimension: Extra dimension flag.
    """
    bound_x_bits = max(1, int(max_plain_bound).bit_length())
    bound_y_bits = max(1, int(bound_y).bit_length())
    p_bits = bound_x_bits + bound_y_bits + max(1, int(eta).bit_length()) + 2
    if extra_dimension is not None:
        p_bits += max(1, int(extra_dimension).bit_length())
    p = gp.next_prime(gp.mpz(_CSPRNG.getrandbits(p_bits)) + (1 << (p_bits - 1)))

    val = float(max_plain_bound) * math.sqrt(eta) + 1.0
    x = val * float(p) * float(bound_y)
    x *= 8.0 * lwe_n * math.sqrt(lwe_n + eta + 1.0)
    x *= math.sqrt(x)
    q_bits = max(p_bits + 2, int(gp.mpz(int(x)).bit_length()) + 1)
    q = gp.next_prime(gp.mpz(_CSPRNG.getrandbits(q_bits)) + (1 << (q_bits - 1)))

    m = (lwe_n + eta + 1) * q_bits + 2 * lwe_n + 1
    sigma = 1.0 / (2.0 * math.sqrt(2.0 * eta * m * lwe_n))
    sigma /= float(p)
    sigma /= float(bound_y)
    sigma_q = sigma * float(q)
    l_sigma = gp.mpz(int(sigma_q / SIGMA_CDT) + 1)
    sigma_q = float(l_sigma) * SIGMA_CDT
    return p, q, m, sigma_q, l_sigma


def derive_fullysec_lwe_parameters(
    bound_x: gp.mpz,
    bound_y: gp.mpz,
    eta: int,
    lwe_n: int,
) -> tuple[gp.mpz, gp.mpz, int, float, gp.mpz, float, gp.mpz, float, gp.mpz]:
    """Derive fully-secure LWE parameters with multiple noise levels.

        Args:
            bound_x: Bound on plaintext values.
            bound_y: Bound on weight values.
            eta: Inner-product vector dimension.
            lwe_n: LWE lattice dimension.
    """
    k = gp.mpz(2) * gp.mpz(eta) * gp.mpz(bound_x) * gp.mpz(bound_y)
    k_float = float(k)
    squared = k_float * k_float
    n_float = float(lwe_n)

    n_bits_q = 1
    sigma = 1.0
    sigma1 = 1.0
    sigma2 = 1.0
    l_sigma1 = gp.mpz(1)
    l_sigma2 = gp.mpz(1)
    i = 1
    while True:
        bound_m_float = max(1.0, n_float * float(i))
        log2_m = math.log2(bound_m_float)
        sqrt_n_log_m = math.sqrt(n_float * log2_m)
        max_float = max(squared, bound_m_float)
        sqrt_max = math.sqrt(max_float)

        sigma1 = sqrt_n_log_m * sqrt_max
        l_sigma1 = gp.mpz(int(sigma1 / SIGMA_CDT) + 1)
        sigma1 = float(l_sigma1) * SIGMA_CDT

        mul_val = math.sqrt(n_float) * math.pow(n_float, 3) * math.pow(math.sqrt(log2_m), 5)
        mul_val *= math.sqrt(bound_m_float)
        sigma2 = mul_val * max_float
        l_sigma2 = gp.mpz(int(sigma2 / SIGMA_CDT) + 1)
        sigma2 = float(l_sigma2) * SIGMA_CDT

        bound2 = math.sqrt(sigma1 * sigma1 + sigma2 * sigma2)
        bound2 *= math.sqrt(n_float)
        sigma = 1.0 / squared
        sigma /= bound2
        sigma /= math.log2(n_float)

        sigma_prime = sigma / k_float
        sigma_prime /= math.pow(n_float, 6) * math.pow(float(n_bits_q), 2)
        sigma_prime /= math.pow(math.sqrt(math.log2(n_float)), 5)

        bound_for_q = math.sqrt(math.log2(n_float)) / sigma_prime
        next_bits_q = max(2, int(math.floor(math.log2(bound_for_q))) + 2)
        if next_bits_q < i:
            break
        n_bits_q = next_bits_q
        i = next_bits_q + 1

    q = gp.next_prime(gp.mpz(_CSPRNG.getrandbits(n_bits_q)) + gp.mpz(1 << (n_bits_q - 1)))
    m = max(eta + 2, int(1.01 * n_float * float(n_bits_q)))
    q_float = float(q)
    sigma_q = sigma * q_float
    l_sigma_q = gp.mpz(int(sigma_q / SIGMA_CDT) + 1)
    sigma_q = float(l_sigma_q) * SIGMA_CDT
    return k, q, m, sigma_q, l_sigma_q, sigma1, l_sigma1, sigma2, l_sigma2


def center_lwe_vector(vector: list[int] | list[gp.mpz], p: gp.mpz, q: gp.mpz) -> list[gp.mpz]:
    """Scale and center a vector for LWE message extraction.

        Args:
            vector: Input vector.
            p: Prime modulus.
            q: Modulus (or second prime).
    """
    ret = []
    for x in vector:
        t = gp.mpz(x) * q
        t = gp.t_div(t, p)
        t %= q
        ret.append(t)
    return ret


def decode_lwe_inner_product(value: gp.mpz, p: gp.mpz, q: gp.mpz) -> int:
    """Extract a plaintext value from an LWE inner-product ciphertext.

        Args:
            value: Input value.
            p: Prime modulus.
            q: Modulus (or second prime).
    """
    d = gp.mpz(value) % q
    half_q = q // 2
    if d > half_q:
        d -= q
    d = d * p
    d = (d + half_q) // q
    return int(d)


def label_scalar_from_hash(hash_value: gp.mpz, label_modulus: int) -> gp.mpz:
    """Convert a hash value to a non-zero centered label scalar.

    The scalar lies in ``[-(M - 1 - M // 2), M // 2]`` excluding 0, for
    ``M = label_modulus``. Zero is excluded because a zero scalar removes the
    per-client label mask ``u * l`` altogether, letting any functional-key
    holder decode each client's plaintext.

        Args:
            hash_value: Hash digest value.
            label_modulus: Modulus for centering the label scalar (>= 2).

        Raises:
            FEValidationError: If ``label_modulus < 2``.
    """
    label_modulus = int(label_modulus)
    if label_modulus < 2:
        raise FEValidationError(
            "label_modulus must be >= 2, got {}".format(label_modulus)
        )
    raw = int(hash_value % gp.mpz(label_modulus - 1)) + 1
    half = label_modulus // 2
    if raw > half:
        raw -= label_modulus
    return gp.mpz(raw)


def decode_fullysec_lwe_inner_product(value: gp.mpz, k: gp.mpz, q: gp.mpz) -> int:
    """Extract a plaintext from a fully-secure LWE ciphertext.

        Args:
            value: Input value.
            k: Inner-product bound.
            q: Modulus (or second prime).
    """
    d = gp.mpz(value) % q
    half_q = q // 2
    if d > half_q:
        d -= q
    q_div_k = q // k
    q_div_2k = q // (k * 2)
    return int((d + q_div_2k) // q_div_k)


#: Gaussian tail cut used in worst-case bounds: a Gaussian sample exceeds
#: 12 standard deviations in absolute value with probability below 2^-100.
GAUSSIAN_TAIL_CUT = 12

_LOG2_4_3 = math.log2(4.0 / 3.0)


def abmpsw20_lwe_parameters(
    n_slots: int,
    bound_x: int,
    bound_y: int,
    lwe_n: int,
    stat_param: int,
    security: str = "selective",
) -> dict:
    """Derive parameters of the ABMPSW20 LWE-based MCFE (SCN 2020, Fig. 4).

    The paper's message and key spaces ``{0..P-1}`` and ``{0..V-1}`` are
    replaced by the signed ranges ``|x| <= X`` and ``|y| <= Y``; the inner
    product then lies in ``[-T, T]`` with ``T = n X Y`` and is encoded with
    ``K = 2T + 1`` and decoded centred. The constraints of Appendix C.2/C.3
    are then instantiated as:

    * correctness, ``q0 > K (nV + 1)``: ``q0`` is the smallest power of two
      ``>= 4 K (n Y + 1)`` (a factor 4 of slack, half of which is reserved
      for the noise of threshold decryption);
    * statistical step (Lemma 34/35), ``sigma >= 10 n P^2`` with
      ``P = 2X + 1`` bounding ``||x^1 - x^0||``: ``sigma_t = 10 n P^2``;
    * LWE noise of the proof, ``alpha' q >= sqrt(n0)``:
      ``sigma_e = ceil(2 sqrt(n0))``;
    * rounding absorbs ``t . e`` (Lemma 28 / game G2),
      ``q >= q0 B n0^{omega(1)}``: ``B = m0 (c sigma_t)(c sigma_e)`` with the
      tail cut ``c`` and ``n0^{omega(1)}`` instantiated as ``2^stat_param``;
      ``q`` is the next prime;
    * leftover hash lemma (Lemma 33), ``m0 log2(4/3) >= log2 q + 2 kappa``
      (selective) or ``>= log2 q + 2 kappa + 4 n log2 P`` (adaptive).

    ``m0`` and ``q`` depend on each other and are iterated to a fixed point.
    In the adaptive setting ``m0`` grows linearly in ``n log2 P`` (about
    4,600 for 100 slots with ``X = Y = 100``, ``kappa = 128``), and every
    label costs ``n0 + m0`` hash outputs and inner-product terms, so large
    ndarray workloads (one label per element) become slow.
    The LWE dimension ``n0`` is taken as given; nothing here estimates the
    concrete hardness of LWE(n0, q, sigma_e).

        Args:
            n_slots: Total number of slots ``n`` (clients x coordinates).
            bound_x: Plaintext bound ``X``.
            bound_y: Weight bound ``Y``.
            lwe_n: LWE secret dimension ``n0``.
            stat_param: Statistical security parameter ``kappa``.
            security: ``"selective"`` or ``"adaptive"`` (choice of ``m0``).

        Returns:
            Dict with integer entries ``q``, ``q0``, ``K``, ``delta``
            (``floor(q / K)``), ``bound_ip`` (``T``), ``n0``, ``m0``,
            ``sigma_t``, ``sigma_e`` and ``B``.
    """
    if security not in ("selective", "adaptive"):
        raise ValueError("security must be 'selective' or 'adaptive'")
    if min(n_slots, bound_x, bound_y, lwe_n, stat_param) < 1:
        raise ValueError("all parameters must be positive")
    bound_ip = n_slots * bound_x * bound_y
    k_mod = 2 * bound_ip + 1
    p_range = 2 * bound_x + 1
    q0 = 1 << (4 * k_mod * (n_slots * bound_y + 1) - 1).bit_length()
    sigma_t = 10 * n_slots * p_range * p_range
    sigma_e = math.ceil(2 * math.sqrt(lwe_n))
    extra = 4 * n_slots * math.log2(p_range) if security == "adaptive" else 0.0

    m0 = 1
    while True:
        b_bound = m0 * (GAUSSIAN_TAIL_CUT * sigma_t) * (GAUSSIAN_TAIL_CUT * sigma_e)
        q = gp.next_prime(gp.mpz(q0) * b_bound << stat_param)
        m0_req = math.ceil((int(q).bit_length() + 2 * stat_param + extra) / _LOG2_4_3)
        if m0_req <= m0:
            break
        m0 = m0_req
    if not q > n_slots * p_range * p_range:  # Lemma 33: n P^2 < q
        raise ValueError("derived q violates n P^2 < q")
    return {
        "q": int(q),
        "q0": q0,
        "K": k_mod,
        "delta": int(q) // k_mod,
        "bound_ip": bound_ip,
        "n0": lwe_n,
        "m0": m0,
        "sigma_t": sigma_t,
        "sigma_e": sigma_e,
        "B": b_bound,
    }


def round_to_q0(value, q, q0) -> gp.mpz:
    """Rounding map ``Z_q -> Z_q0``: ``round((q0 / q) * (value mod q)) mod q0``.

        Args:
            value: Integer (reduced mod ``q`` first).
            q: Large modulus.
            q0: Small modulus.
    """
    q = gp.mpz(q)
    q0 = gp.mpz(q0)
    v = gp.mpz(value) % q
    return ((2 * q0 * v + q) // (2 * q)) % q0


def decode_rounded_inner_product(mu_prime, q, q0, delta, bound_ip) -> int | None:
    """Decode ``mu' ~ (q0 / q) * delta * mu (mod q0)`` to the signed ``mu``.

    Returns the integer ``mu`` closest to ``mu' q / (q0 delta)`` after
    centring ``mu'`` modulo ``q0``, or ``None`` if ``|mu| > bound_ip``.

        Args:
            mu_prime: Value modulo ``q0``.
            q: Large modulus.
            q0: Small modulus.
            delta: Encoding factor ``floor(q / K)``.
            bound_ip: Bound ``T`` on ``|mu|``.
    """
    q0 = gp.mpz(q0)
    mu_c = gp.mpz(mu_prime) % q0
    if mu_c > q0 // 2:
        mu_c -= q0
    num = mu_c * gp.mpz(q)
    den = q0 * gp.mpz(delta)
    mu = (2 * num + den) // (2 * den)
    if abs(mu) > bound_ip:
        return None
    return int(mu)
