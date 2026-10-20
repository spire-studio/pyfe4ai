"""Review status of every registered scheme.

Each registered ``(family, variant)`` has a
:class:`~pyfe4ai.core.registry.SchemeStatus`, declared with the scheme in
:mod:`pyfe4ai.core.registry` (which also defines the status and fidelity
values). This module derives the status table from the registry and
provides what uses it: the queries, the
:class:`~pyfe4ai.core.exceptions.FESecurityWarning` emitted when a
known-insecure key generator or crypto object is instantiated, and the
README *Scheme Status* table (:func:`markdown_table`). All schemes
additionally share the caveats of the README (research prototype, toy
default parameters, no side-channel hardening).
"""

from __future__ import annotations

import warnings

from pyfe4ai.core.exceptions import FESecurityWarning
from pyfe4ai.core.registry import (  # the status vocabulary is defined with the registry
    AD_HOC,
    ADAPTED,
    EXPERIMENTAL,
    FAITHFUL,
    KNOWN_INSECURE,
    REVIEWED,
    SCHEMES,
    UNKNOWN,
    UNREVIEWED,
    SchemeStatus,
)

#: Review status of every registered ``(family, variant)``, derived from
#: :data:`pyfe4ai.core.registry.SCHEMES` (in its order): the source of the
#: README *Scheme Status* table and of the status table of the docs.
#:
#: :meta hide-value:
SCHEME_STATUS: dict[tuple[str, str], SchemeStatus] = {scheme.key: scheme.status for scheme in SCHEMES}

_BY_MODULE = {entry.module: key for key, entry in SCHEME_STATUS.items()}

__all__ = [
    "SchemeStatus", "REVIEWED", "EXPERIMENTAL", "UNREVIEWED", "KNOWN_INSECURE", "FAITHFUL", "ADAPTED", "AD_HOC",
    "UNKNOWN", "SCHEME_STATUS", "get_scheme_status", "status_for_module", "warn_if_known_insecure", "markdown_table",
]


def get_scheme_status(family: str, variant: str | None = None) -> SchemeStatus:
    """Return the review status of a registered scheme: ``(family, variant)``,
    or its id ``"family/variant"`` alone.

    Raises:
        KeyError: If the scheme is not registered.
    """
    if variant is None and isinstance(family, str) and "/" in family:
        family, variant = family.split("/", 1)
    return SCHEME_STATUS[(family, variant)]


def status_for_module(module: str) -> tuple[tuple[str, str], SchemeStatus] | None:
    """Return ``((family, variant), status)`` for a scheme module, if registered."""
    key = _BY_MODULE.get(module)
    return None if key is None else (key, SCHEME_STATUS[key])


def warn_if_known_insecure(module: str, stacklevel: int = 3) -> None:
    """Emit :class:`~pyfe4ai.core.exceptions.FESecurityWarning` if *module*
    implements a known-insecure scheme."""
    found = status_for_module(module)
    if found is None:
        return
    (family, variant), entry = found
    if entry.status == KNOWN_INSECURE:
        warnings.warn(
            "{}/{} is known to be insecure: {}. Use it only for functionality "
            "or performance experiments.".format(family, variant, entry.issue),
            FESecurityWarning,
            stacklevel=stacklevel,
        )


def markdown_table() -> str:
    """Render :data:`SCHEME_STATUS` as the README *Scheme Status* table."""
    lines = [
        "| Scheme | Status | Fidelity | Construction | Notes |",
        "|---|---|---|---|---|",
    ]
    for (family, variant), entry in SCHEME_STATUS.items():
        note = (
            "see [Known Issues](#known-issues--security-notice)"
            if entry.status == KNOWN_INSECURE
            else entry.note
        )
        lines.append(
            "| `{}/{}` | {} | {} | {} | {} |".format(
                family, variant, entry.status, entry.fidelity, entry.reference, note
            )
        )
    return "\n".join(lines)
