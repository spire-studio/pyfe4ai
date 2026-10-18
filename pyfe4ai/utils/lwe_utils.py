"""LWE parameter derivation, vector centering, and inner-product decoding."""

from __future__ import annotations

import math

import gmpy2 as gp




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
    # deterministic, so that a dimension chosen by "auto" is reproducible
    p = gp.next_prime(gp.mpz(1) << (p_bits - 1))

    val = float(max_plain_bound) * math.sqrt(eta) + 1.0
    x = val * float(p) * float(bound_y)
    x *= 8.0 * lwe_n * math.sqrt(lwe_n + eta + 1.0)
    x *= math.sqrt(x)
    q_bits = max(p_bits + 2, int(gp.mpz(int(x)).bit_length()) + 1)
    # the largest q_bits-bit prime: entries of a seeded A (pyfe4ai.utils.lwe_matrix)
    # are then sampled from q_bits-bit candidates with almost no rejection
    q = gp.prev_prime(gp.mpz(1) << q_bits)

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
    # |<x, y>| <= eta X Y takes 2 eta X Y + 1 values: K = 2 eta X Y made +-eta X Y
    # collide (the sign of the noise decided the result)
    k = gp.mpz(2) * gp.mpz(eta) * gp.mpz(bound_x) * gp.mpz(bound_y) + 1
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
        # ALS16 Theorem 2 needs m >= 4 n log2 q (the widths below depend on m)
        bound_m_float = max(1.0, 4.0 * n_float * float(i))
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

        # B_tau: the norm of a row of tau (m entries of width sigma1 or sigma2);
        # decryption correctness needs alpha^-1 >= K^2 B_tau omega(sqrt(log n))
        bound2 = math.sqrt(sigma1 * sigma1 + sigma2 * sigma2)
        bound2 *= math.sqrt(bound_m_float)
        sigma = 1.0 / squared
        sigma /= bound2
        # omega(sqrt(log n)) of the correctness bound: log2 n, but at least the
        # 12-sigma tail cut (log2 n alone left toy dimensions a few sigma of margin)
        sigma /= max(math.log2(n_float), GAUSSIAN_TAIL_CUT)

        sigma_prime = sigma / k_float
        sigma_prime /= math.pow(n_float, 6) * math.pow(float(n_bits_q), 2)
        sigma_prime /= math.pow(math.sqrt(math.log2(n_float)), 5)

        bound_for_q = math.sqrt(math.log2(n_float)) / sigma_prime
        next_bits_q = max(2, int(math.floor(math.log2(bound_for_q))) + 2)
        if next_bits_q < i:
            break
        n_bits_q = next_bits_q
        i = next_bits_q + 1

    # the largest n_bits_q-bit prime (a random draw could have one bit more)
    q = gp.prev_prime(gp.mpz(1) << n_bits_q)
    m = max(eta + 2, 4 * int(lwe_n) * n_bits_q)
    q_float = float(q)
    sigma_q = sigma * q_float
    l_sigma_q = gp.mpz(int(sigma_q / SIGMA_CDT) + 1)
    sigma_q = float(l_sigma_q) * SIGMA_CDT
    return k, q, m, sigma_q, l_sigma_q, sigma1, l_sigma1, sigma2, l_sigma2


def center_lwe_vector(vector: list[int] | list[gp.mpz], p: gp.mpz, q: gp.mpz) -> list[gp.mpz]:
    """Encode a signed vector as ``floor(q / p) x mod q``.

    The encoding is linear in ``x``, as the linear ciphertext homomorphism of
    the ABDP15 PKE requires (versions <= 0.2.0 used ``trunc(x q / p)``, which
    differs from it by a small non-linear term).

    Args:
        vector: Input vector.
        p: Plaintext modulus.
        q: Ciphertext modulus.
    """
    delta = gp.mpz(q) // gp.mpz(p)
    return [delta * gp.mpz(x) % q for x in vector]


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


#: Largest ``q.bit_length()`` for which LWE in dimension ``n`` with a uniform
#: secret modulo ``q``, errors ``D_{Z, ceil(2 sqrt(n))}`` and any number of
#: samples costs at least 2^128 operations, by the lattice estimator
#: (https://github.com/malb/lattice-estimator, commit 53da598, default MATZOV
#: cost model; primal uSVP and BDD, dual, dual hybrid). The error width is the
#: ``sigma_e`` of :func:`abmpsw20_lwe_parameters`, so this bounds the LWE
#: problem of the ABMPSW20 security proof. Regenerate with
#: ``tools/estimate_dense_lwe.py``.
#:
#: :meta hide-value:
DENSE_LWE_128_LOG_Q = {
    4096: 118, 4608: 132, 5120: 146, 5632: 159, 6144: 173,
    6656: 187, 7168: 200, 7680: 214, 8192: 227, 8704: 241,
    9216: 254, 9728: 268, 10240: 282, 10752: 295, 11264: 309,
    11776: 322, 12288: 336, 12800: 350, 13312: 363, 13824: 377,
    14336: 390, 14848: 404, 15360: 418, 15872: 431, 16384: 445,
}


def check_dense_lwe(lwe_n: int, q) -> None:
    """Raise unless LWE in dimension *lwe_n* modulo *q* meets :data:`DENSE_LWE_128_LOG_Q`.

    A dimension between two tabulated ones is held to the bound of the
    smaller one (LWE only gets harder with the dimension and with the error
    width ``2 sqrt(n)``).
    """
    tabulated = [n for n in DENSE_LWE_128_LOG_Q if n <= lwe_n]
    if not tabulated or int(q).bit_length() > DENSE_LWE_128_LOG_Q[max(tabulated)]:
        from pyfe4ai.utils.exceptions import FEValidationError

        raise FEValidationError(
            "lwe_n = {} with a {}-bit q is below 128-bit security (lattice estimator: "
            "log2 q <= {} for lwe_n >= {})".format(
                lwe_n,
                int(q).bit_length(),
                DENSE_LWE_128_LOG_Q[max(tabulated)] if tabulated else "-",
                max(tabulated) if tabulated else min(DENSE_LWE_128_LOG_Q),
            )
        )


def dense_lwe_n(modulus_for) -> int:
    """Smallest tabulated dimension whose modulus ``modulus_for(n)`` meets
    :data:`DENSE_LWE_128_LOG_Q` (the modulus grows with the dimension)."""
    for lwe_n, max_bits in DENSE_LWE_128_LOG_Q.items():
        if int(modulus_for(lwe_n)).bit_length() <= max_bits:
            return lwe_n
    from pyfe4ai.utils.exceptions import FEValidationError

    raise FEValidationError(
        "no LWE dimension up to {} reaches 128-bit security for these bounds; "
        "reduce bound_x, bound_y, eta or the number of clients".format(max(DENSE_LWE_128_LOG_Q))
    )


def resolve_lwe_n(lwe_n, modulus_for, default: int) -> int:
    """LWE dimension of a config entry: ``"auto"`` selects the smallest one
    meeting :data:`DENSE_LWE_128_LOG_Q` (see :func:`dense_lwe_n`), ``None``
    gives *default*."""
    if lwe_n == "auto":
        return dense_lwe_n(modulus_for)
    return int(default if lwe_n is None else lwe_n)


#: For each error width ``2^s`` (outer key) and LWE dimension ``n`` (inner
#: key): the largest ``log2(q / sigma)`` for which LWE with a uniform secret,
#: errors ``D_{Z, 2^s}`` and any number of samples costs at least 2^128
#: operations, by the lattice estimator (commit 53da598, MATZOV cost model;
#: primal uSVP and BDD, dual). The ABDP15 public key ``A S + E`` has very
#: wide errors (``sigma`` of 2^30 to 2^45 for small bounds from the GoFE
#: parameter recipe, more for large ones);
#: its key generators read the table at the narrower width ``sigma'`` of the
#: LWE problem their security proof reduces to
#: (:func:`abdp15_reduction_sigma`). At a fixed ratio the attacks get more
#: expensive as the width grows, so a scheme is held to the row of the largest
#: width not above the one it is checked at. Regenerate with
#: ``tools/estimate_wide_lwe.py``.
#:
#: :meta hide-value:
WIDE_LWE_128_LOG_RATIO = {
    16: {512: 21, 576: 23, 640: 25, 704: 27, 768: 29, 832: 31, 896: 33, 960: 35,
         1024: 36, 1088: 38, 1152: 40, 1216: 42, 1280: 44, 1344: 45, 1408: 47, 1472: 49,
         1536: 51, 1600: 52, 1664: 54, 1728: 56, 1792: 58, 1856: 59, 1920: 61, 1984: 63,
         2048: 65, 2112: 66, 2176: 68, 2240: 70, 2304: 71, 2368: 73, 2432: 75, 2496: 77,
         2560: 78, 2624: 80, 2688: 82, 2752: 83, 2816: 85, 2880: 87, 2944: 89, 3008: 90,
         3072: 92, 3136: 94, 3200: 95, 3264: 97, 3328: 99, 3392: 100, 3456: 102, 3520: 104,
         3584: 106, 3648: 107, 3712: 109, 3776: 111, 3840: 112, 3904: 114, 3968: 116, 4032: 117,
         4096: 119, 4160: 121, 4224: 123, 4288: 124, 4352: 126, 4416: 128, 4480: 129, 4544: 131,
         4608: 133, 4672: 134, 4736: 136, 4800: 138, 4864: 139, 4928: 141, 4992: 143, 5056: 145,
         5120: 146, 5184: 148, 5248: 150, 5312: 151, 5376: 153, 5440: 155, 5504: 156, 5568: 158,
         5632: 160, 5696: 161, 5760: 163, 5824: 165, 5888: 167, 5952: 168, 6016: 170, 6080: 172,
         6144: 173},
    24: {512: 24, 576: 27, 640: 29, 704: 31, 768: 33, 832: 35, 896: 37, 960: 39,
         1024: 40, 1088: 42, 1152: 44, 1216: 46, 1280: 48, 1344: 50, 1408: 52, 1472: 53,
         1536: 55, 1600: 57, 1664: 59, 1728: 61, 1792: 63, 1856: 64, 1920: 66, 1984: 68,
         2048: 70, 2112: 71, 2176: 73, 2240: 75, 2304: 77, 2368: 78, 2432: 80, 2496: 82,
         2560: 84, 2624: 86, 2688: 87, 2752: 89, 2816: 91, 2880: 92, 2944: 94, 3008: 96,
         3072: 98, 3136: 99, 3200: 101, 3264: 103, 3328: 105, 3392: 106, 3456: 108, 3520: 110,
         3584: 111, 3648: 113, 3712: 115, 3776: 117, 3840: 118, 3904: 120, 3968: 122, 4032: 124,
         4096: 125, 4160: 127, 4224: 129, 4288: 130, 4352: 132, 4416: 134, 4480: 136, 4544: 137,
         4608: 139, 4672: 141, 4736: 142, 4800: 144, 4864: 146, 4928: 148, 4992: 149, 5056: 151,
         5120: 153, 5184: 154, 5248: 156, 5312: 158, 5376: 159, 5440: 161, 5504: 163, 5568: 165,
         5632: 166, 5696: 168, 5760: 170, 5824: 171, 5888: 173, 5952: 175, 6016: 177, 6080: 178,
         6144: 180},
    32: {512: 27, 576: 29, 640: 32, 704: 34, 768: 36, 832: 38, 896: 40, 960: 42,
         1024: 44, 1088: 46, 1152: 48, 1216: 50, 1280: 52, 1344: 54, 1408: 56, 1472: 58,
         1536: 59, 1600: 61, 1664: 63, 1728: 65, 1792: 67, 1856: 69, 1920: 71, 1984: 72,
         2048: 74, 2112: 76, 2176: 78, 2240: 80, 2304: 82, 2368: 83, 2432: 85, 2496: 87,
         2560: 89, 2624: 90, 2688: 92, 2752: 94, 2816: 96, 2880: 98, 2944: 99, 3008: 101,
         3072: 103, 3136: 105, 3200: 106, 3264: 108, 3328: 110, 3392: 112, 3456: 113, 3520: 115,
         3584: 117, 3648: 119, 3712: 120, 3776: 122, 3840: 124, 3904: 126, 3968: 127, 4032: 129,
         4096: 131, 4160: 133, 4224: 134, 4288: 136, 4352: 138, 4416: 140, 4480: 141, 4544: 143,
         4608: 145, 4672: 147, 4736: 148, 4800: 150, 4864: 152, 4928: 153, 4992: 155, 5056: 157,
         5120: 159, 5184: 160, 5248: 162, 5312: 164, 5376: 166, 5440: 167, 5504: 169, 5568: 171,
         5632: 172, 5696: 174, 5760: 176, 5824: 178, 5888: 179, 5952: 181, 6016: 183, 6080: 185,
         6144: 186},
    40: {512: 29, 576: 32, 640: 34, 704: 36, 768: 39, 832: 41, 896: 43, 960: 45,
         1024: 47, 1088: 49, 1152: 51, 1216: 53, 1280: 55, 1344: 57, 1408: 59, 1472: 61,
         1536: 63, 1600: 65, 1664: 67, 1728: 69, 1792: 71, 1856: 73, 1920: 75, 1984: 77,
         2048: 78, 2112: 80, 2176: 82, 2240: 84, 2304: 86, 2368: 88, 2432: 90, 2496: 91,
         2560: 93, 2624: 95, 2688: 97, 2752: 99, 2816: 101, 2880: 102, 2944: 104, 3008: 106,
         3072: 108, 3136: 110, 3200: 111, 3264: 113, 3328: 115, 3392: 117, 3456: 118, 3520: 120,
         3584: 122, 3648: 124, 3712: 126, 3776: 127, 3840: 129, 3904: 131, 3968: 133, 4032: 134,
         4096: 136, 4160: 138, 4224: 140, 4288: 141, 4352: 143, 4416: 145, 4480: 147, 4544: 149,
         4608: 150, 4672: 152, 4736: 154, 4800: 156, 4864: 157, 4928: 159, 4992: 161, 5056: 162,
         5120: 164, 5184: 166, 5248: 168, 5312: 169, 5376: 171, 5440: 173, 5504: 175, 5568: 176,
         5632: 178, 5696: 180, 5760: 182, 5824: 183, 5888: 185, 5952: 187, 6016: 189, 6080: 190,
         6144: 192},
}
_WIDE_DIMS = sorted(WIDE_LWE_128_LOG_RATIO[16])


def wide_lwe_bound(lwe_n: int, sigma) -> int | None:
    """Largest 128-bit ``log2(q / sigma)`` of :data:`WIDE_LWE_128_LOG_RATIO` for
    *lwe_n* and *sigma* (rows at or below both), or ``None`` below the table."""
    widths = [s for s in WIDE_LWE_128_LOG_RATIO if 2.0 ** s <= sigma]
    dims = [n for n in _WIDE_DIMS if n <= lwe_n]
    if not widths or not dims:
        return None
    return WIDE_LWE_128_LOG_RATIO[max(widths)][max(dims)]


def abdp15_reduction_sigma(sigma, diff_bound, eta: int) -> float:
    """Error width ``sigma'`` of the LWE problem that the ABDP15 proof reduces to.

    The selective-security proof of the LWE instantiation (ABDP15 full
    version, ePrint 2015/017 as revised on 2015-03-28; written out in
    F. Bourse, PhD thesis, ENS 2017, Thm. 4.3.5 with Thms. 8.2.2, 8.2.3 and
    8.3.2) rewrites the public keys as ``d_i pk + pk'_i`` for the difference
    ``d = x1 - x0`` of the challenge vectors. It needs
    ``sigma >= (1 + B sqrt(eta)) sigma'`` with ``|d_i| <= B``, and the scheme
    is then as hard as LWE with errors of width ``sigma'``, not ``sigma``.
    Theorem 2 of the proceedings version does not cover noisy LWE keys.
    The full-version text has not been checked here: its ePrint record only
    says that the 2015-03-28 revision reworked the LWE instantiation and added
    a second generic theorem. The statement above rests on the thesis.
    """
    return float(sigma) / (1.0 + float(diff_bound) * math.sqrt(eta))


def check_wide_lwe(lwe_n: int, q, sigma, what: str = "sigma") -> None:
    """Raise unless LWE(lwe_n, q, sigma) meets :data:`WIDE_LWE_128_LOG_RATIO`;
    *what* names the width in the message."""
    bound = wide_lwe_bound(lwe_n, sigma)
    ratio = math.log2(int(q)) - math.log2(sigma)
    if bound is None or ratio > bound:
        from pyfe4ai.utils.exceptions import FEValidationError

        raise FEValidationError(
            "lwe_n = {n} with log2(q / {w}) = {r:.1f} and {w} = 2^{s:.1f} is below 128-bit "
            "security (lattice estimator: log2(q / {w}) <= {b} here; tabulated for "
            "lwe_n >= {n0} and {w} >= 2^{s0})".format(
                n=lwe_n, w=what, r=ratio, s=math.log2(sigma), b="-" if bound is None else bound,
                n0=_WIDE_DIMS[0], s0=min(WIDE_LWE_128_LOG_RATIO),
            )
        )


def wide_lwe_n(params_for, lwe_sigma) -> dict:
    """``params_for(n)`` (a dict with ``q`` and ``sigma_q``) for the smallest
    tabulated ``n`` that meets :data:`WIDE_LWE_128_LOG_RATIO` at the width
    ``lwe_sigma(params)``.

    *lwe_sigma* has no default, so that a caller cannot fall back to the
    public key's own width by omission. The ABDP15 key generators pass the
    width of their security reduction (:func:`abdp15_reduction_sigma`).
    """
    for lwe_n in _WIDE_DIMS:
        params = params_for(lwe_n)
        sigma = lwe_sigma(params)
        bound = wide_lwe_bound(lwe_n, sigma)
        if bound is not None and math.log2(int(params["q"])) - math.log2(sigma) <= bound:
            return params
    from pyfe4ai.utils.exceptions import FEValidationError

    raise FEValidationError(
        "no LWE dimension up to {} reaches 128-bit security for these bounds; "
        "reduce bound_x, bound_y, eta or the number of clients".format(_WIDE_DIMS[-1])
    )


def abdp15_parameters(
    bound_x, bound_y, eta: int, lwe_n, extra_dimension: int | None = None, lwe_sigma=None
) -> dict:
    """:func:`derive_lwe_parameters` as a dict, for *lwe_n* or, if it is
    ``"auto"``, for the smallest dimension meeting :data:`WIDE_LWE_128_LOG_RATIO`
    at the width ``lwe_sigma(params)`` (see :func:`wide_lwe_n`; the key
    generators pass the width of their security reduction,
    :func:`abdp15_reduction_sigma`).

    Returns:
        Dict with ``lwe_n``, ``p``, ``q``, ``m``, ``sigma_q`` and ``l_sigma``.

    Raises:
        ValueError: For ``lwe_n = "auto"`` without *lwe_sigma*.
    """
    if lwe_n == "auto" and lwe_sigma is None:
        raise ValueError(
            'lwe_n = "auto" needs lwe_sigma, the error width the 128-bit table is read at '
            "(abdp15_reduction_sigma for the ABDP15 schemes)"
        )

    def params_for(n: int) -> dict:
        p, q, m, sigma_q, l_sigma = derive_lwe_parameters(
            bound_x, bound_y, eta, n, extra_dimension=extra_dimension
        )
        return {"lwe_n": n, "p": p, "q": q, "m": m, "sigma_q": sigma_q, "l_sigma": l_sigma}

    if lwe_n == "auto":
        return wide_lwe_n(params_for, lwe_sigma)
    return params_for(int(lwe_n))


#: Gaussian tail cut used in worst-case bounds. A Gaussian sample exceeds
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
    The LWE dimension ``n0`` is taken as given; :data:`DENSE_LWE_128_LOG_Q`
    bounds ``q`` for 128-bit hardness of LWE(n0, q, sigma_e).

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


def decode_scaled(value, delta, q, bound) -> int | None:
    """Decode ``value ~ delta * mu + noise (mod q)`` to the signed integer ``mu``.

    Used by the two-step decryption of the LWE IPFEs (ACFGU18 Section 4.2,
    encoding ``E(gamma, noise) = gamma * delta + noise mod q``).

    Args:
        value: Value modulo ``q``.
        delta: Scaling factor ``floor(q / p)``.
        q: Modulus.
        bound: Bound on ``|mu|``; larger results return ``None``.
    """
    q = gp.mpz(q)
    delta = gp.mpz(delta)
    v = gp.mpz(value) % q
    if v > q // 2:
        v -= q
    mu = (2 * v + delta) // (2 * delta)
    if abs(mu) > bound:
        return None
    return int(mu)
