"""Server and client ids.

A server id (``sid``: an entry of ``lst_sid``, the server of a key share or a
partial decryption) or a client id (``nid``: an entry of ``lst_nid``, the
client of a ciphertext or an encrypted gradient) is a ``str`` or an ``int``
(not a ``bool``): :func:`is_id`. Partial decryptions and the wire records of
the ML adapter carry ids of these types only, and messages that list ids order
them with :func:`id_list`.
"""

from __future__ import annotations

from pyfe4ai.core.exceptions import FEValidationError

#: What a list of ids must be, by option name (the text of the
#: ``FEValidationError`` for a list that is not one).
ID_LISTS = {
    "lst_sid": "lst_sid must be a list of server ids (str or int)",
    "lst_nid": "lst_nid must be a list of client ids (str or int)",
}

#: The ``FEValidationError`` text for a repeated id, by option name.
DUPLICATE_IDS = {
    "lst_sid": "lst_sid contains duplicate server ids",
    "lst_nid": "lst_nid contains duplicate client ids",
}


def is_id(value) -> bool:
    """Whether *value* is a server or client id: a ``str``, or an ``int``
    other than a ``bool``."""
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def id_list(ids) -> list:
    """*ids* sorted for a message: by value, or by ``repr`` where the values
    do not compare (``[1, "a"]``), so that listing ids never raises."""
    ids = list(ids)
    try:
        return sorted(ids)
    except TypeError:
        return sorted(ids, key=repr)


def check_id_list(ids, option: str) -> None:
    """Raise unless *ids* (the value of the option *option*, ``"lst_sid"`` or
    ``"lst_nid"``) lists distinct ids that can key a dict.

    Raises:
        FEValidationError: ``ID_LISTS[option]`` for a value whose entries
            cannot be counted and compared (not a sized collection, or an
            unhashable entry such as a list), ``DUPLICATE_IDS[option]`` for a
            repeated id.
    """
    try:
        distinct = len(set(ids)) == len(ids)
    except TypeError:
        raise FEValidationError(ID_LISTS[option]) from None
    if not distinct:
        raise FEValidationError(DUPLICATE_IDS[option])
