"""
The serialized objects of the threshold key issuance of ``sife/cp_ab_pairing``
(:mod:`pyfe4ai.schemes.sife.cp_ab_pairing_keyissue`): the public record of
the key servers, a server's secret share, a key request and a partial key.

The combined key is the scheme's own
:class:`~pyfe4ai.core.formats.pairing.CPABFunctionalKey`, unchanged. Group
elements are serialised as in :mod:`pyfe4ai.core.formats.pairing`; digests
and keys are lowercase hex.
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record


class CPABKeyServers(Record):
    """The public record of the key servers of one sharing."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.CP_AB_KEYSERVERS_VERSION`)")
    protocol = Field("protocol id (:data:`~pyfe4ai.core.versions.CP_AB_KEYISSUE_PROTOCOL`)")
    pp_binding = Field("the ``pp_binding`` of the public parameters")
    servers = Field("list of the server ids, in the order of their share points")
    points = Field("list of the Shamir share points (ints), one per server")
    threshold = Field("int ``t``: servers per request")
    sharing = Field("``shamir``")
    sharing_id = Field("hex, 16 random bytes drawn at the sharing")


class CPABKeyServerShare(Record):
    """A key server's secret share (as secret as the master key, to its holder)."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.CP_AB_KEYSERVER_SHARE_VERSION`)")
    server = Field("the server id")
    keyservers = Field("hex digest of the :class:`CPABKeyServers` record")
    sharing_id = Field("hex, the ``sharing_id`` of that record")
    u = Field("the share ``U_i``: 2 rows of ``eta`` decimal strings mod ``p``")
    r_key = Field("hex, the 32-byte PRF key of the server's randomness")
    mask_keys = Field("dict from every other server id to the hex 32-byte PRF key the two share")


class CPABKeyRequest(Record):
    """A key request (public: every server of the request reads it)."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.CP_AB_KEYREQUEST_VERSION`)")
    id = Field("hex, 16 random bytes drawn by the requester")
    sid = Field("str, the requester's identity; servers require it to be the authenticated one")
    y = Field("the ``eta`` weights, decimal strings")
    attributes = Field("the labels, sorted, without copies")
    servers = Field("the ``t`` distinct server ids, sorted")
    pp_binding = Field("the ``pp_binding`` of the public parameters")
    keyservers = Field("hex digest of the :class:`CPABKeyServers` record")
    digest = Field("hex, 64 bytes: SHAKE-256 of the canonical encoding of the fields above")


class CPABPartialKey(Record):
    """One server's masked partial key for one request."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.CP_AB_PARTIAL_KEY_VERSION`)")
    server = Field("the server id")
    request = Field("hex, the request's digest")
    k1 = Field("2 elements of G2")
    k0 = Field("2 elements of G2")
    k_attr = Field("dict from ``label#copy`` (every copy of every label of the request) to 2 elements of G2")
