"""
The serialized objects of the identity directory
(:mod:`pyfe4ai.ac.identity_directory`): a directory of per-identity
instances of ``sife/fullysec_ring_lwe``, and the ciphertexts and functional
keys bound to one of its entries.

The inner objects (``pp``, ``pk``, ``ct``, ``dk`` and the ndarray layers)
are those of :mod:`pyfe4ai.core.formats.ring_lwe`, unchanged. Digests are
lowercase hex SHA-256 (:mod:`pyfe4ai.ac.identity_directory` says over what).
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record


class IdentityDirectoryRecord(Record):
    """A published identity directory."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.IDENTITY_DIRECTORY_VERSION`)")
    scheme = Field("the scheme id of every instance (``sife/fullysec_ring_lwe``)")
    pp = Field("the public parameters the instances share")
    pp_digest = Field("hex SHA-256 of ``pp``")
    entries = Field("list of :class:`IdentityEntry`, sorted by identity")
    digest = Field("hex SHA-256 of the version, the scheme, ``pp_digest`` and the entries' identities and digests")


class IdentityEntry(Record):
    """One identity's entry in a directory."""

    identity = Field("str: NFC, 1 to 256 characters, no control or format characters")
    pk = Field("the instance's encryption key (``eta`` polynomials)")
    entry = Field("hex SHA-256 of the version, the scheme, ``pp_digest``, the identity and ``pk``")


class IdentityCiphertext(Record):
    """A vector ciphertext for one identity."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.IDENTITY_DIRECTORY_VERSION`)")
    identity = Field("str, the identity it was encrypted for")
    entry = Field("hex digest of the directory entry it was encrypted under")
    ct = Field("the ``sife/fullysec_ring_lwe`` ciphertext")


class IdentityLayers(Record):
    """Encrypted ndarrays for one identity."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.IDENTITY_DIRECTORY_VERSION`)")
    identity = Field("str, the identity they were encrypted for")
    entry = Field("hex digest of the directory entry they were encrypted under")
    layers = Field("the ``sife/fullysec_ring_lwe`` ndarray layers (element-wise arrays or packed layers)")


class IdentityFunctionalKey(Record):
    """A functional key of one identity's instance."""

    version = Field("format id (:data:`~pyfe4ai.core.versions.IDENTITY_DIRECTORY_VERSION`)")
    identity = Field("str, the identity it was issued to")
    entry = Field("hex digest of the directory entry it was issued under")
    dk = Field("the ``sife/fullysec_ring_lwe`` functional key (``sk_y``, ``y``)")
