"""The binding of the keys and ciphertexts of a scheme to its public parameters.

The public parameters of an attribute-based scheme carry ``pp_binding``, a
digest of all their other fields, and every functional key and ciphertext made
under them carries a copy. Objects of two setups are then told apart by
comparing two strings, before any pairing or discrete log. The digest is not a
signature: it catches a mix-up or a corruption, and anyone can compute it for
parameters of their own. Standard library and
:mod:`pyfe4ai.core.exceptions` only.
"""

from __future__ import annotations

import hashlib
import json

from pyfe4ai.core.exceptions import FEValidationError


def pp_digest(pp: dict, domain: bytes) -> str:
    """``SHA-256(domain || text)`` in hexadecimal, *text* the canonical JSON of
    every field of *pp* but ``pp_binding``: compact, keys sorted, non-ASCII
    characters as they are, in UTF-8.

    ``FEValidationError`` if *pp* is not a dict of JSON data.
    """
    if not isinstance(pp, dict):
        raise FEValidationError("public parameters must be a dict, got {}".format(type(pp).__name__))
    body = {name: value for name, value in pp.items() if name != "pp_binding"}
    try:
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:  # also a lone surrogate, which has no UTF-8
        raise FEValidationError("public parameters must be JSON data: {}".format(exc)) from exc
    return hashlib.sha256(domain + text).hexdigest()


def check_binding(pp: dict, record, name: str) -> None:
    """``FEValidationError`` unless *record* carries the ``pp_binding`` of *pp*.

    Args:
        pp: The public parameters.
        record: A key or ciphertext, as a dict or a
            :class:`~pyfe4ai.core.formats.record.Record`.
        name: What *record* is, for the message.
    """
    binding = record.get("pp_binding")
    if binding is None:
        raise FEValidationError("the {} has no pp_binding".format(name))
    if binding != pp["pp_binding"]:
        raise FEValidationError("the {} was made under other public parameters: its pp_binding differs".format(name))
