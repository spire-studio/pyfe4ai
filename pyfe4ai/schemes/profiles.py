"""Security profiles: parameter presets for the key generators.

The ``security_profile`` entry of a key generator's config selects one:

``"toy"`` (default)
    The library's small defaults -- fast enough for tests and demos, and
    **not secure**. A config without the entry, or with ``"toy"``, is used
    exactly as given.

``"standard"``
    Parameters aiming at 128-bit classical security where this pure-Python
    prototype can reach it, chosen per scheme family:

    * discrete log (DDH, Damgard / MDDH and their MIFE / MCFE variants): a
      3072-bit safe-prime group, the RFC 3526 MODP group 15 (NIST SP 800-57
      Part 1 Rev. 5, Table 2: L = 3072 for 128 bits). The decentralised
      ``mcfe/ddh_decentralized`` then needs a DCR modulus of about 6200 bits
      for combining key shares, whose safe primes take minutes to generate;
    * DCR (Paillier variants): ``N`` of about 3072 bits (two 1536-bit safe
      primes);
    * Ring-LWE: ``kappa = 128`` and the smallest ring dimension whose derived
      modulus ``q`` stays within the 128-bit bound of the Homomorphic
      Encryption Standard (Albrecht et al. 2018, Table 1:
      ``log2 q`` <= 27 / 54 / 109 / 218 / 438 / 881 for n = 1024 ... 32768);
      that bound is for ternary secrets and ``sigma = 3.2`` errors, so it is
      conservative for the wider Gaussians used here;
    * pairing-based schemes: BN254, the strongest curve charm-crypto provides
      -- about 100-bit security after the exTNFS attacks (Barbulescu and
      Duquesne, J. Cryptol. 2019), **below** the 128-bit target; a
      :class:`~pyfe4ai.utils.exceptions.FESecurityWarning` says so.

    Refused, because no parameter set is both secure and usable in pure
    Python: the dense-LWE schemes (the public matrix of a secure dimension
    has billions of entries) and ``mcfe/paillier_decentralized`` (its key
    shares are combined under a DCR modulus of more than 15000 bits).

    Values given explicitly in the config must not be weaker than the
    profile's. Statistical parameters (``stat_param``) keep their default,
    ``2^-64`` statistical distance.
"""

from __future__ import annotations

import warnings

from pyfe4ai.schemes.status import status_for_module
from pyfe4ai.utils.exceptions import FESecurityWarning, FEValidationError

TOY = "toy"
STANDARD = "standard"
PROFILES = (TOY, STANDARD)

#: Scheme family of every registered ``(family, variant)``.
SCHEME_KIND = {
    **{("sife", v): "dl" for v in ("ddh", "ddh_dynamic", "damgard")},
    **{("mife", v): "dl" for v in ("ddh", "damgard", "ddh_threshold")},
    **{("mcfe", v): "dl" for v in ("ddh", "damgard", "ddh_threshold", "ddh_decentralized")},
    **{(f, "paillier"): "dcr" for f in ("sife", "mife", "mcfe")},
    ("mcfe", "paillier_decentralized"): "dcr",
    **{("sife", v): "lwe" for v in ("lwe", "fullysec_lwe")},
    **{("mife", v): "lwe" for v in ("lwe", "fullysec_lwe", "lwe_threshold")},
    **{("mcfe", v): "lwe" for v in ("lwe", "fullysec_lwe", "lwe_threshold", "lwe_decentralized")},
    **{(f, "ring_lwe"): "ring_lwe" for f in ("sife", "mife", "mcfe")},
    **{("mcfe", v): "ring_lwe" for v in ("ring_lwe_threshold", "ring_lwe_decentralized")},
    **{("sife", v): "pairing" for v in ("fh_ipe_pairing", "part_fh_ipe_pairing")},
    **{("mife", v): "pairing" for v in ("fh_ipe_pairing", "fh_multi_ipe_pairing")},
    **{("mcfe", v): "pairing" for v in (
        "fh_multi_ipe_pairing", "fh_multi_ipe_pairing_threshold", "fh_multi_ipe_pairing_decentralized"
    )},
    **{("quadratic", v): "pairing" for v in ("quad", "sgp", "miqfe")},
}

#: Minimum value of each size parameter under ``standard``; also its default.
_STANDARD_MINIMA = {
    "dl": {"sec_param": 3072, "modulus_length": 3072},
    "dcr": {"bit_length": 1536},
    "ring_lwe": {"sec_param": 128},
}
STANDARD_PAIRING_GROUP = "BN254"

_REFUSED = {
    "lwe": "secure dense-LWE dimensions are out of reach of this pure-Python prototype; "
    "use a Ring-LWE, DL or DCR scheme",
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
            "{}/{}: BN254, the strongest pairing group of charm-crypto, gives about "
            "100-bit security, below the 128-bit target of the '{}' profile".format(*scheme, profile),
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
    return out
