"""The group elements that a pairing crypto object decodes once and keeps.

The key-policy attribute-based schemes read the same group elements at each call: the generators ``g1`` and ``g2`` of
the public parameters, the tables of the public parameters (single-input) or of the client's encryption key
(multi-client) that an encryption combines, and ``e(g1, g2)``, the base of the discrete log. Decoding one costs a
deserialization and the checks of :func:`~pyfe4ai.schemes.common.ac_codec.read_elements`, so a crypto object does it at
the first use of a field and keeps the elements. A field that is never read is never decoded: a malformed element in a
row that no call reads goes unseen, as it does without the cache.
"""

from __future__ import annotations

from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, pair
from pyfe4ai.schemes.common.ac_codec import read_elements


class DecodedElementsMixin:
    """The cached getters of a pairing crypto object.

    The class that mixes this in calls ``_init_decoded`` in its constructor. It has ``group`` and ``pp`` (a dict) for
    the generators and ``_pp_elements``, and ``_client_key`` (a record) for ``_key_elements``. The cache lives
    as long as the object: the public parameters and the key of an object do not change.
    """

    def _init_decoded(self) -> None:
        """Start with nothing decoded."""
        self._decoded = {}
        self._gt = None

    def _pp_elements(self, kind, name: str, index: int | None = None) -> list:
        """The elements of the field *name* of the public parameters (row *index* of a table), of the group *kind*.

        Decoded and checked at their first use, and kept: encryption reads the same ones each time.
        """
        key = (name, index)
        if key not in self._decoded:
            items = self.pp[name] if index is None else self.pp[name][index]
            self._decoded[key] = read_elements(self.group, items, kind, name)
        return self._decoded[key]

    def _key_elements(self, name: str, index: int | None = None) -> list:
        """The elements of the field *name* of the encryption key (row *index* of a table), all of ``G1``, decoded and
        checked at their first use, and kept: encryption reads the same ones each time."""
        key = ("ek", name, index)
        if key not in self._decoded:
            items = getattr(self._client_key, name)
            if index is not None:
                items = items[index]
            self._decoded[key] = read_elements(self.group, items, G1, name)
        return self._decoded[key]

    def _generator(self, name: str, kind):
        """The generator ``g1`` or ``g2`` of the public parameters (*name*; *kind*: ``G1`` or ``G2``), decoded at its
        first use."""
        key = (name, None)
        if key not in self._decoded:
            self._decoded[key] = read_elements(self.group, [self.pp[name]], kind, name)[0]
        return self._decoded[key]

    def _generator_pair(self):
        """``e(g1, g2)``, computed once."""
        if self._gt is None:
            self._gt = pair(self._generator("g1", G1), self._generator("g2", G2))
        return self._gt
