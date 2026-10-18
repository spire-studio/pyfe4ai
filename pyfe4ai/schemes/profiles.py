"""Security profiles: parameter presets for the key generators.

The ``security_profile`` entry of a key generator's config selects one:

``"toy"`` (default)
    The library's small defaults -- fast enough for tests and demos, and
    **not secure**. A config without the entry, or with ``"toy"``, is used
    exactly as given.

``"standard"``
    Parameters aiming at 128-bit classical security where this prototype
    can reach it, chosen per scheme family:

    * discrete log (DDH, Damgard / MDDH and their MIFE / MCFE variants): a
      3072-bit safe-prime group, the RFC 3526 MODP group 15 (NIST SP 800-57
      Part 1 Rev. 5, Table 2: L = 3072 for 128 bits). The decentralised
      ``mcfe/ddh_decentralized`` then needs a DCR modulus of about 6410 bits
      for combining key shares, whose safe primes take minutes to generate;
    * DCR (Paillier variants): ``N`` of about 3072 bits (two 1536-bit safe
      primes);
    * DDH over ristretto255 (``*/ddh_ristretto255``): the group is fixed, of
      order about ``2^252`` (Pollard rho about ``2^126``, the 128-bit class of
      X25519 / Ed25519); the profile sets nothing;
    * Ring-LWE: ``kappa = 128`` and the smallest ring dimension whose derived
      modulus ``q`` meets :data:`~pyfe4ai.utils.ring_lwe_utils.RING_LWE_128_LOG_Q`,
      a lattice-estimator table (the method of the LWE tables) read at the
      smallest width among the RLWE problems of the scheme's proof (about 8
      for the BMMS21 parameters of Sec. 4.2 and 5; ``log2 q`` <= 31 / 57 /
      111 / 218 / 435 / 872 for n = 1024 ... 32768 at that width). It replaces the
      Homomorphic Encryption Standard's bound, which this estimator puts at
      about 125 bits; ``sife/fullysec_ring_lwe`` (BMMS21 Sec. 5, adaptive) also
      picks its number ``m`` of key polynomials: the smallest one whose ``q``
      meets that bound;
    * dense LWE, the ABMPSW20 MCFE family (``mcfe/lwe``, ``fullysec_lwe``,
      ``lwe_threshold``, ``lwe_decentralized``): ``lwe_n = "auto"``, the
      smallest dimension ``n0`` whose derived ``q`` meets
      :data:`~pyfe4ai.utils.lwe_utils.DENSE_LWE_128_LOG_Q` -- a lattice
      estimator table for LWE(n0, q, ceil(2 sqrt(n0))) with a uniform secret,
      the problem of the scheme's security proof. These schemes have no public
      matrix: a label is hashed to ``n0 + m0`` values and each ciphertext is
      one inner product, cheaper than a 3072-bit DDH ciphertext. The
      decentralised variant combines key shares under a DCR modulus of at
      least 3072 bits (``STANDARD_DCR_PRIME_BITS`` per prime), so its key
      derivation is not post-quantum;
    * LWE with a seeded public matrix, the ABDP15 IPFE (``sife/lwe``,
      ``mife/lwe``): ``lwe_n = "auto"``, the smallest dimension whose
      ``(q, sigma)`` meets :data:`~pyfe4ai.utils.lwe_utils.WIDE_LWE_128_LOG_RATIO`,
      a lattice estimator table for the LWE problem of the public key
      ``A S + E`` (``lwe_n`` of about 830 to 1100 for small bounds). ``A``
      (``m ~ lwe_n log2 q`` rows, 5 * 10^7 to 10^8 entries) is published as a
      seed; ``matrix_mode`` chooses between keeping it expanded in memory
      (about 0.1 s per encryption, 1.5 to 3 GB) and expanding it for every
      batch (``"stream"``: seconds per batch, about 0.5 GB). See
      :mod:`pyfe4ai.schemes.abdp15_lwe`. ``mife/lwe_threshold`` is selected
      the same way at its flooding-enlarged ``q`` (``lwe_n`` of about 6000
      with ``stat_param = 128``, ``A`` of about 8 * 10^9 entries), and
      ``standard`` fixes its ``matrix_mode`` to ``"stream"``: key generation
      takes over half an hour, an encryption about 10 minutes, or about 11 s
      each in batches of 64;
    * pairing-based schemes: BLS12-381 (the Rust extension) -- about 120 to
      126-bit security after the exTNFS attacks on its 381-bit field
      (Barbulescu and Duquesne, J. Cryptol. 2019), **below** the 128-bit
      target; a :class:`~pyfe4ai.utils.exceptions.FESecurityWarning` says so.

    Refused, because no parameter set is both secure and usable in pure
    Python: ``sife/fullysec_lwe`` and ``mife/fullysec_lwe`` (ALS16: at a
    dimension the lattice estimator puts at 128 bits, n = 3328 to 4352 for
    ``X = Y <= 50`` and ``eta <= 10``, and with ``m = 4 n log2 q`` as its
    Theorem 2 requires, the public matrix has 1 to 2 * 10^10 entries and a
    ciphertext 2.9 to 4.5 * 10^6 coordinates of 219 to 258 bits) and
    ``mcfe/paillier_decentralized`` (its key shares are combined under a DCR
    modulus of more than 15000 bits).

    The statistical parameter ``stat_param`` (rounding and leftover-hash
    slack of ABMPSW20, flooding of the threshold schemes, the masks of
    ``mcfe/ddh_decentralized`` and the Paillier and MCFE2 key distributions)
    is 128, as ``kappa`` of the Ring-LWE schemes: a statistical distance of
    ``2^-128`` per sample, so that ``Q`` samples stay below ``Q 2^-128``.
    That holds for exact Gaussians (the ``"facct"`` sampler, the default of
    the lattice schemes) and for the MPFR path at ``bits(sigma) + 192`` bits;
    the double-precision rounded sampler does not meet it (see the security
    notes in the documentation).

    Values given explicitly in the config must not be weaker than the
    profile's.
"""

from __future__ import annotations

import warnings

from pyfe4ai.schemes.status import status_for_module
from pyfe4ai.utils.exceptions import FESecurityWarning, FEValidationError

TOY = "toy"
STANDARD = "standard"
PROFILES = (TOY, STANDARD)

#: Scheme family of every registered ``(family, variant)``.
#:
#: :meta hide-value:
SCHEME_KIND = {
    **{("sife", v): "dl" for v in ("ddh", "ddh_dynamic", "damgard")},
    **{("mife", v): "dl" for v in ("ddh", "damgard", "ddh_threshold")},
    **{("mcfe", v): "dl" for v in ("ddh", "damgard", "ddh_threshold", "ddh_decentralized")},
    **{(f, "ddh_ristretto255"): "ec" for f in ("sife", "mife", "mcfe")},
    **{(f, "paillier"): "dcr" for f in ("sife", "mife", "mcfe")},
    ("mcfe", "paillier_decentralized"): "dcr",
    **{(f, "lwe"): "lwe_abdp15" for f in ("sife", "mife")},
    **{(f, "fullysec_lwe"): "lwe_matrix" for f in ("sife", "mife")},
    ("mife", "lwe_threshold"): "lwe_abdp15",
    **{("mcfe", v): "lwe" for v in ("lwe", "fullysec_lwe", "lwe_threshold", "lwe_decentralized")},
    **{(f, "ring_lwe"): "ring_lwe" for f in ("sife", "mife", "mcfe")},
    ("sife", "fullysec_ring_lwe"): "ring_lwe",
    **{("mcfe", v): "ring_lwe" for v in ("ring_lwe_threshold", "ring_lwe_decentralized")},
    **{("sife", v): "pairing" for v in ("fh_ipe_pairing", "part_fh_ipe_pairing")},
    **{("mife", v): "pairing" for v in ("fh_ipe_pairing", "fh_multi_ipe_pairing")},
    **{("mcfe", v): "pairing" for v in (
        "fh_multi_ipe_pairing", "fh_multi_ipe_pairing_threshold", "fh_multi_ipe_pairing_decentralized"
    )},
    **{("quadratic", v): "pairing" for v in ("quad", "sgp", "miqfe")},
}

#: Bits of each prime of a DCR modulus under ``standard`` (``N`` of 3072 bits).
STANDARD_DCR_PRIME_BITS = 1536

#: Statistical parameter under ``standard`` (minimum and default).
STANDARD_STAT_PARAM = 128

#: Minimum value of each size parameter under ``standard``; also its default.
#: ``stat_param`` is ignored by the schemes that do not use it.
_STANDARD_MINIMA = {
    "dl": {"sec_param": 3072, "modulus_length": 3072, "stat_param": STANDARD_STAT_PARAM},
    # ristretto255 is fixed (about 126-bit security); nothing to size
    "ec": {},
    "dcr": {"bit_length": STANDARD_DCR_PRIME_BITS, "stat_param": STANDARD_STAT_PARAM},
    "ring_lwe": {"sec_param": 128, "stat_param": STANDARD_STAT_PARAM},
    "lwe": {"stat_param": STANDARD_STAT_PARAM},
    "lwe_abdp15": {"stat_param": STANDARD_STAT_PARAM},
}
STANDARD_PAIRING_GROUP = "BLS12-381"

#: Settings the ``standard`` profile fixes for single schemes, with the reason.
_STANDARD_FIXED = {
    ("mife", "lwe_threshold"): (
        {"matrix_mode": "stream"},
        "its flooding noise needs lwe_n of about 6000 and a public matrix of about 8*10^9 "
        "entries, more than 400 GB when kept in memory",
    ),
}

_REFUSED = {
    "lwe_matrix": "at a 128-bit dimension (lwe_n = 3328 to 4352 for X = Y <= 50, eta <= 10) its "
    "public matrix has 1 to 2*10^10 entries and every ciphertext m = 4 n log2 q = 2.9 to 4.5*10^6 "
    "coordinates of 219 to 258 bits (80 to 145 MB); use sife/fullysec_ring_lwe for adaptive "
    "security, or the Ring-LWE scheme of the same family",
    ("mcfe", "paillier_decentralized"): "its key shares would be combined under a DCR modulus "
    "of more than 15000 bits; use mcfe/ddh_decentralized",
}


def scheme_of(cls: type) -> tuple | None:
    """Return the registered ``(family, variant)`` that *cls* implements or extends."""
    for klass in cls.__mro__:
        found = status_for_module(klass.__module__)
        if found is not None:
            return found[0]
    return None


def resolve(config: dict, scheme: tuple | None) -> dict:
    """Return *config* completed with the presets of its ``security_profile``.

    Args:
        config: A key generator's configuration; returned unchanged (the same
            object) for the ``toy`` profile, otherwise copied.
        scheme: ``(family, variant)`` of the key generator, or ``None`` if it
            is not a registered scheme.

    Raises:
        FEValidationError: For an unknown profile, a scheme the profile does
            not support, or an explicit value weaker than the profile's.
    """
    profile = config.get("security_profile", TOY)
    if profile not in PROFILES:
        raise FEValidationError("unknown security_profile {!r}; use one of {}".format(profile, PROFILES))
    if profile == TOY:
        return config
    kind = SCHEME_KIND.get(scheme)
    if kind is None:
        raise FEValidationError("no '{}' parameters for {}".format(profile, scheme))
    reason = _REFUSED.get(scheme) or _REFUSED.get(kind)
    if reason:
        raise FEValidationError(
            "no '{}' parameter set for {}/{}: {}".format(profile, scheme[0], scheme[1], reason)
        )
    out = dict(config)
    if kind == "pairing":
        group = out.setdefault("pairing_group_param", STANDARD_PAIRING_GROUP)
        if group != STANDARD_PAIRING_GROUP:
            raise FEValidationError(
                "the '{}' profile uses pairing group {}, got {!r}".format(profile, STANDARD_PAIRING_GROUP, group)
            )
        warnings.warn(
            "{}/{}: BLS12-381 gives about 120 to 126-bit security after the exTNFS "
            "attacks, below the 128-bit target of the '{}' profile".format(*scheme, profile),
            FESecurityWarning,
            stacklevel=3,
        )
        return out
    for key, minimum in _STANDARD_MINIMA[kind].items():
        value = out.setdefault(key, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise FEValidationError(
                "the '{}' profile needs {} >= {}, got {!r}".format(profile, key, minimum, value)
            )
    if kind == "ring_lwe":
        out.setdefault("ring_n", "auto")
    if kind in ("lwe", "lwe_abdp15"):
        out.setdefault("lwe_n", "auto")
    fixed, why = _STANDARD_FIXED.get(scheme, ({}, ""))
    for key, value in fixed.items():
        if out.setdefault(key, value) != value:
            raise FEValidationError(
                "the '{}' profile needs {}={!r} for {}/{}: {}".format(
                    profile, key, value, scheme[0], scheme[1], why
                )
            )
    return out
