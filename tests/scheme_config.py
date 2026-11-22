"""A config written for several schemes, cut to the options of one: the key generators warn of the others, and refuse
them under the ``standard`` profile."""

from pyfe4ai.core import config as configs


def of_scheme(scheme, config: dict) -> dict:
    """The entries of *config* that *scheme* (``(family, variant)`` or ``"family/variant"``) declares."""
    declared = configs.config_class(tuple(scheme.split("/")) if isinstance(scheme, str) else scheme).OPTIONS
    return {key: value for key, value in config.items() if key in declared}
