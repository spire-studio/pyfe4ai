"""The stability tiers of the 1.x compatibility policy, derived from the registry.

Every registered scheme is in one tier, given by its review status
(:class:`~pyfe4ai.core.registry.SchemeStatus`) and nothing else:

====================  ==============================  ==================================================
Tier                  Review status                   Promise within 1.x
====================  ==============================  ==================================================
``stable``            ``reviewed``                    public API, config options and serialized records
                                                      stay backward compatible; every 1.x release reads
                                                      the records 1.0 wrote; changes are deprecated
                                                      (a planned ``FEDeprecationWarning``) before 2.0 
``experimental``      ``experimental``,               none: may change in any release
                      ``unreviewed``
``known-insecure``    ``known-insecure``              none; refused by the ``standard`` profile
====================  ==============================  ==================================================

A stable scheme can still have experimental parts -- option values
(:data:`EXPERIMENTAL_OPTIONS`) or capabilities (:data:`EXPERIMENTAL_CAPABILITIES`)
whose construction or argument is this project's own -- and the records
written with them are experimental too (:func:`config_tier`). Security fixes
are the one exception to the promise: in any release a scheme, an option or
a kind of record found insecure may become known-insecure and be refused by
``standard``.
"""

from __future__ import annotations

from collections.abc import Mapping

from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.registry import EXPERIMENTAL, KNOWN_INSECURE, REVIEWED, UNREVIEWED, get_scheme

STABLE = "stable"
EXPERIMENTAL_TIER = "experimental"
KNOWN_INSECURE_TIER = "known-insecure"
#: The tiers, most stable first.
TIERS = (STABLE, EXPERIMENTAL_TIER, KNOWN_INSECURE_TIER)

#: The tier of each review status.
TIER_OF_STATUS = {
    REVIEWED: STABLE,
    EXPERIMENTAL: EXPERIMENTAL_TIER,
    UNREVIEWED: EXPERIMENTAL_TIER,
    KNOWN_INSECURE: KNOWN_INSECURE_TIER,
}

#: Option values of stable schemes that are experimental: ``{scheme id: {option: values}}`` (key generator
#: options; a value outside the set is in the scheme's tier).
EXPERIMENTAL_OPTIONS: dict[str, dict[str, frozenset]] = {
    # the packed layout rests on this project's own leveraging argument (registry note)
    "sife/fullysec_ring_lwe": {"ndarray_layout": frozenset({"packed"})},
}

#: Capabilities of stable schemes that are experimental: ``{scheme id: capabilities}``.
EXPERIMENTAL_CAPABILITIES: dict[str, frozenset] = {}


def tier(scheme) -> str:
    """The tier of *scheme* (an id ``"family/variant"`` or ``(family, variant)``).

    Raises:
        FEValidationError: For an unknown scheme.
    """
    return TIER_OF_STATUS[get_scheme(scheme).status.status]


def experimental_options(config: Mapping, scheme) -> dict:
    """The options of *config* whose values are experimental for *scheme*: ``{option: value}``."""
    entry = get_scheme(scheme)
    marked = EXPERIMENTAL_OPTIONS.get(entry.id, {})
    return {name: config[name] for name, values in marked.items() if name in config and config[name] in values}


def config_tier(scheme, config: Mapping | None = None) -> str:
    """The tier of the records that *scheme* writes under *config*: the scheme's tier, or ``"experimental"`` when
    a stable scheme is configured with an experimental option value."""
    base = tier(scheme)
    if base == STABLE and config and experimental_options(config, scheme):
        return EXPERIMENTAL_TIER
    return base


def stable_schemes() -> list[str]:
    """The ids of the stable schemes, in registry order."""
    from pyfe4ai.core.registry import SCHEMES

    return [s.id for s in SCHEMES if TIER_OF_STATUS[s.status.status] == STABLE]


def _check_tables() -> None:
    """The experimental tables name stable schemes and options or capabilities they have (checked by the tests)."""
    from pyfe4ai.core.registry import CAPABILITIES

    for scheme, options in EXPERIMENTAL_OPTIONS.items():
        if tier(scheme) != STABLE:
            raise FEValidationError("{}: experimental options are listed for stable schemes only".format(scheme))
        declared = get_scheme(scheme).config.OPTIONS
        unknown = set(options) - set(declared)
        if unknown:
            raise FEValidationError("{}: no options {}".format(scheme, sorted(unknown)))
    for scheme, caps in EXPERIMENTAL_CAPABILITIES.items():
        if tier(scheme) != STABLE or not set(caps) <= set(CAPABILITIES):
            raise FEValidationError("{}: experimental capabilities {} are not capabilities of a stable scheme".format(
                scheme, sorted(caps)))
