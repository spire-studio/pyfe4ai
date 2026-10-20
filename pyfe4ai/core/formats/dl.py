"""
The serialized objects of the discrete-log and DCR schemes, and the key
shares of the decentralised MCFE key derivation (ABMPSW20 Fig. 14) that the
decentralised DDH, DCR and LWE schemes share.
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record


class MCFE2KeyShare(Record):
    """A client's share of a functional key of the decentralised MCFE
    (:mod:`pyfe4ai.schemes.mcfe.decentralize`): its contribution encrypted under
    the MCFE2 key."""

    c = Field("decimal strings, the MCFE2 ciphertexts of the contribution")
    nid = Field("str, the client")
    weights_id = Field("hex digest of the weights")
