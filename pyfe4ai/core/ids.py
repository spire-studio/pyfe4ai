"""Server and client ids.

A server id (``sid``: an entry of ``lst_sid``, the server of a key share or a
partial decryption) or a client id (``nid``: an entry of ``lst_nid``, the
client of a ciphertext or an encrypted gradient) is a ``str`` or an ``int``
(not a ``bool``): :func:`is_id`. A key generator refuses a ``lst_sid`` or
``lst_nid`` of other values, or one that mixes ``str`` and ``int`` ids
(:func:`check_id_list`); partial decryptions and the wire records of the ML
adapter carry ids of these types only, and messages that list ids order them
with :func:`id_list`.

Where an id enters after setup -- the server or client named when keys are
issued, a crypto object's own ``id``, the enrolled servers of a partial
decryption, the servers of the ML adapter, the ids of partial decryptions and
encrypted gradients read back -- a numpy integer scalar is read as the
``int`` it equals (:func:`as_id`; ``list(rng.choice(lst_sid, t))`` gives
such ids), and then the same rule and the same membership checks apply.
"""

from __future__ import annotations

import operator

import numpy as np

from pyfe4ai.core.exceptions import FEValidationError

#: What a list of ids must be, by option name (the text of the
#: ``FEValidationError`` for a list that is not one).
ID_LISTS = {
    "lst_sid": "lst_sid must be a list of server ids, each a str or an int (convert numpy integers with int(), "
               "other ids with str())",
    "lst_nid": "lst_nid must be a list of client ids, each a str or an int (convert numpy integers with int(), "
               "other ids with str())",
}

#: The ``FEValidationError`` text for a list that mixes ``str`` and ``int``
#: ids, by option name: JSON names ``1`` and ``"1"`` alike, and sorting
#: such ids (canonical JSON digests, messages) fails.
MIXED_IDS = {
    "lst_sid": "lst_sid mixes str and int server ids (convert them all with str(), or all with int())",
    "lst_nid": "lst_nid mixes str and int client ids (convert them all with str(), or all with int())",
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


def as_id(value):
    """*value* read as an id after setup: a numpy integer scalar
    (``numpy.integer``; ``numpy.bool_`` is none) becomes the ``int`` it equals
    (``operator.index``), which it also hashes like; every other value is
    returned unchanged, for :func:`is_id` and the membership checks to judge
    (a float, a ``bool``, ``None`` stay what they are)."""
    return operator.index(value) if isinstance(value, np.integer) else value


def issued_id(value):
    """*value* read as the id a key is issued for: :func:`as_id` of it if that
    is an id, else ``None`` -- which no ``lst_sid`` or ``lst_nid`` holds, so
    that the key generator answers as for an unknown server or client
    (``None``), also for ``True`` or ``1.0``, which equal the id ``1``."""
    value = as_id(value)
    return value if is_id(value) else None


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
    ``"lst_nid"``) is a list (or tuple) of distinct ids, all ``str`` or all
    ``int``.

    The lists themselves are taken as given (numpy integers are refused
    here, not read as ints: the key generator keeps the list, and the public
    parameters and key shares carry it).

    Raises:
        FEValidationError: ``ID_LISTS[option]`` for another value (a string,
            a set, a dict, an iterator) or an entry that is not an id
            (``bytes``, a float, ``None``, a ``bool``, a tuple, a numpy
            integer), ``MIXED_IDS[option]`` for ``str`` and ``int`` ids in one
            list, ``DUPLICATE_IDS[option]`` for a repeated id.
    """
    if not isinstance(ids, (list, tuple)) or not all(is_id(i) for i in ids):
        raise FEValidationError(ID_LISTS[option])
    if len({isinstance(i, str) for i in ids}) > 1:
        raise FEValidationError(MIXED_IDS[option])
    if len(set(ids)) != len(ids):
        raise FEValidationError(DUPLICATE_IDS[option])
