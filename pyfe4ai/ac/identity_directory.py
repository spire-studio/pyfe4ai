"""
Identity directory: one ``sife/fullysec_ring_lwe`` instance per identity.

**Experimental** (the directory, its checks and its records; the scheme
underneath, BMMS21 Sec. 5, is reviewed and unchanged). A key centre
(:class:`IdentityKeyCentre`) runs an instance of the adaptively secure
Ring-LWE IPFE for every identity it adds and publishes a directory: the
public parameters the instances share, each identity's encryption key, and
digests that bind them. An encryptor (:class:`IdentityDirectory`) encrypts
for an identity under that identity's key; the identity's holder decrypts
with functional keys of its own instance only.

* **Security.** The instances share the parameter record (``a``, ``q``, the
  widths) and draw their secrets independently. An adversary attacking
  identity B's instance while holding any keys of the other identities is
  simulated from B's instance alone (the reduction draws the other secrets
  itself, ``pk = <a, s>``), so the directory is as secure as one instance;
  challenges for several identities cost a hybrid over the identities.
  Encrypting the same vector for several identities is an "or" of
  identities; there is no "and".
* **Not provided.** The key centre holds every identity's master key (key
  escrow, as in identity-based encryption). Ciphertexts name their identity
  in the clear (no anonymity). The directory's authenticity is the
  application's: the library checks that a directory is consistent and, given
  the digest an encryptor obtained over a trusted channel
  (``expected_digest``), that it is that directory. There is no revocation;
  re-keying an identity is a new key centre.
* **Bindings.** ``pp_digest`` is SHA-256 over the whole ``pp`` (``a``, the
  modulus, the widths, the sampler, the profile, the layout); an entry's
  digest is over the format, the scheme, ``pp_digest``, the identity and its
  ``pk``; the directory's digest is over the format, the scheme, ``pp_digest``
  and the sorted identities with their entry digests. Ciphertexts and keys
  carry the identity and the entry digest, and decryption refuses
  (``FEKeyError``) unless they agree with each other and with the directory.
  An entry's digest does not depend on the other entries: adding identities
  changes the directory's digest only, and earlier ciphertexts and keys stay
  valid.
* **Identities** are case-sensitive ``str`` of 1 to 256 characters in
  Unicode NFC, without control or format characters; nothing is normalised
  silently.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import unicodedata

from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats.identity import IdentityCiphertext
from pyfe4ai.core.formats.identity import IdentityDirectoryRecord
from pyfe4ai.core.formats.identity import IdentityEntry
from pyfe4ai.core.formats.identity import IdentityFunctionalKey
from pyfe4ai.core.formats.identity import IdentityLayers
from pyfe4ai.core.versions import IDENTITY_DIRECTORY_VERSION
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWE
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWEKeyGenerator

#: The scheme every instance of a directory runs.
SCHEME_ID = "sife/fullysec_ring_lwe"
#: The longest identity, in characters.
MAX_IDENTITY_LENGTH = 256

_DOMAIN_PP = b"pyfe4ai/identity-directory/pp/v1|"
_DOMAIN_ENTRY = b"pyfe4ai/identity-directory/entry/v1|"
_DOMAIN_DIRECTORY = b"pyfe4ai/identity-directory/directory/v1|"
# Unicode categories an identity may not contain: control, format, surrogate
_REFUSED_CATEGORIES = frozenset(("Cc", "Cf", "Cs"))


def _canonical(value) -> bytes:
    """Canonical JSON of *value* (sorted keys, no spaces, ASCII; a JSON round
    trip of *value* gives the same bytes)."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise FEValidationError("not a JSON value: {}".format(exc)) from exc


def _sha256(domain: bytes, value) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


def check_identity(identity) -> str:
    """Return *identity* if it is a valid identity, else raise.

    Raises:
        FEValidationError: Unless *identity* is a ``str`` of 1 to
            :data:`MAX_IDENTITY_LENGTH` characters in NFC without control,
            format or surrogate characters.
    """
    if not isinstance(identity, str) or not identity:
        raise FEValidationError("an identity is a non-empty str, got {!r}".format(identity))
    if len(identity) > MAX_IDENTITY_LENGTH:
        raise FEValidationError("an identity has at most {} characters".format(MAX_IDENTITY_LENGTH))
    if any(unicodedata.category(c) in _REFUSED_CATEGORIES for c in identity):
        raise FEValidationError("an identity may not contain control or format characters")
    if unicodedata.normalize("NFC", identity) != identity:
        raise FEValidationError("an identity must be in Unicode NFC (normalise it before use)")
    return identity


def pp_digest(pp: dict) -> str:
    """Hex SHA-256 of the public parameters *pp* (all of them)."""
    return _sha256(_DOMAIN_PP, pp)


def entry_digest(digest_of_pp: str, identity: str, pk: dict) -> str:
    """Hex SHA-256 binding *identity* to its encryption key *pk* under the
    public parameters of digest *digest_of_pp*."""
    return _sha256(_DOMAIN_ENTRY, [IDENTITY_DIRECTORY_VERSION, SCHEME_ID, digest_of_pp, identity, pk])


def directory_digest(digest_of_pp: str, entries: list) -> str:
    """Hex SHA-256 of a directory with the public parameters of digest
    *digest_of_pp* and the *entries* (sorted by identity)."""
    return _sha256(_DOMAIN_DIRECTORY, [IDENTITY_DIRECTORY_VERSION, SCHEME_ID, digest_of_pp,
                                       [[e["identity"], e["entry"]] for e in entries]])


def _same(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and hmac.compare_digest(a.encode(), b.encode())


class IdentityKeyCentre:
    """The key centre: one ``sife/fullysec_ring_lwe`` instance per identity.

    Holds the master key of every identity it adds. Its state is in memory;
    the library defines no format for it.

    Args:
        config: The configuration of ``sife/fullysec_ring_lwe``
            (:class:`~pyfe4ai.schemes.sife.fullysec_ring_lwe.SIFEFullySecRingLWEKeyGenerator`),
            shared by every identity.
    """

    def __init__(self, config: dict | None = None, **kwargs) -> None:
        self._template = SIFEFullySecRingLWEKeyGenerator(dict(config or {}), **kwargs)
        self._pp = self._template.get_public_parameters()
        self._pp_digest = pp_digest(self._pp)
        self._instances = {}
        self._entries = {}

    def add_identity(self, identity: str) -> str:
        """Set up a fresh instance for *identity*; return its entry digest.

        Raises:
            FEValidationError: For an invalid identity.
            FESchemeError: If the identity is already in the directory.
        """
        check_identity(identity)
        if identity in self._instances:
            raise FESchemeError("identity {!r} is already in the directory".format(identity))
        instance = copy.copy(self._template)  # shares the parameter record (a, q, widths)
        instance.setup()  # its own secrets s_i and pk_i
        pk = instance.get_private_keys()
        entry = entry_digest(self._pp_digest, identity, pk)
        self._instances[identity] = instance
        self._entries[identity] = IdentityEntry.new(identity=identity, pk=pk, entry=entry).to_dict()
        return entry

    def identities(self) -> list:
        """The identities, sorted."""
        return sorted(self._instances)

    def directory(self) -> dict:
        """The directory to publish (:class:`~pyfe4ai.core.formats.identity.IdentityDirectoryRecord`)."""
        entries = [copy.deepcopy(self._entries[i]) for i in sorted(self._entries)]
        return IdentityDirectoryRecord.new(
            version=IDENTITY_DIRECTORY_VERSION,
            scheme=SCHEME_ID,
            pp=copy.deepcopy(self._pp),
            pp_digest=self._pp_digest,
            entries=entries,
            digest=directory_digest(self._pp_digest, entries),
        ).to_dict()

    def get_decryption_keys(self, identity: str, fusion_weight: list) -> dict:
        """A functional key for *fusion_weight* of *identity*'s instance.

        Whoever calls this decides that the caller is *identity*; the key
        centre authenticates nobody.

        Raises:
            FEKeyError: For an identity not in the directory.
            FEValidationError: For invalid weights.
        """
        check_identity(identity)
        instance = self._instances.get(identity)
        if instance is None:
            raise FEKeyError("identity {!r} is not in the directory".format(identity))
        dk = instance.get_decryption_keys("identity", credentials={"fusion_weight": fusion_weight})
        return IdentityFunctionalKey.new(
            version=IDENTITY_DIRECTORY_VERSION, identity=identity, entry=self._entries[identity]["entry"], dk=dk,
        ).to_dict()


class IdentityDirectory:
    """A published directory, checked: encrypt for its identities, decrypt
    with keys issued under it.

    Args:
        record: The directory (:meth:`IdentityKeyCentre.directory`).
        expected_digest: The directory digest obtained over a trusted channel;
            without it the directory is only checked for consistency, which
            does not tell it from one a key centre of an attacker published.
        crypto_config: Further options of the
            :class:`~pyfe4ai.schemes.sife.fullysec_ring_lwe.SIFEFullySecRingLWE`
            objects (``precision``, ``gaussian_sampler``).

    Raises:
        FEValidationError: For a malformed or inconsistent directory, or one
            whose digest is not *expected_digest*.
    """

    def __init__(self, record: dict, *, expected_digest: str | None = None, crypto_config: dict | None = None) -> None:
        self._entries, self._pp, self.digest = self._checked(record)
        if expected_digest is not None and not _same(self.digest, expected_digest):
            raise FEValidationError("the directory's digest is not the expected one")
        self._crypto_config = dict(crypto_config or {})
        self._crypto_config.pop("keys", None)
        self._crypto = {}

    @staticmethod
    def _checked(record) -> tuple:
        if not isinstance(record, dict):
            raise FEValidationError("a directory is a dict")
        d = IdentityDirectoryRecord(record)
        try:
            version, scheme, pp, digest_of_pp = d.version, d.scheme, d.pp, d.pp_digest
            entries, digest = d.entries, d.digest
        except KeyError as exc:
            raise FEValidationError("the directory lacks {}".format(exc)) from exc
        if version != IDENTITY_DIRECTORY_VERSION:
            raise FEValidationError("unknown directory format {!r}".format(version))
        if scheme != SCHEME_ID:
            raise FEValidationError("a directory of {!r}, not {}".format(scheme, SCHEME_ID))
        if not isinstance(pp, dict) or not _same(pp_digest(pp), digest_of_pp):
            raise FEValidationError("the directory's public parameters do not match pp_digest")
        if not isinstance(entries, list):
            raise FEValidationError("the directory's entries are a list")
        by_identity = {}
        for e in entries:
            if not isinstance(e, dict):
                raise FEValidationError("a directory entry is a dict")
            try:
                identity, pk, entry = e["identity"], e["pk"], e["entry"]
            except KeyError as exc:
                raise FEValidationError("a directory entry lacks {}".format(exc)) from exc
            check_identity(identity)
            if identity in by_identity:
                raise FEValidationError("identity {!r} appears twice".format(identity))
            if not _same(entry_digest(digest_of_pp, identity, pk), entry):
                raise FEValidationError("the entry of {!r} does not match its digest".format(identity))
            by_identity[identity] = e
        if [e["identity"] for e in entries] != sorted(by_identity):
            raise FEValidationError("the directory's entries are not sorted by identity")
        if not _same(directory_digest(digest_of_pp, entries), digest):
            raise FEValidationError("the directory does not match its digest")
        return by_identity, pp, digest

    def identities(self) -> list:
        """The identities, sorted."""
        return sorted(self._entries)

    @property
    def public_parameters(self) -> dict:
        """The public parameters the instances share (``security_profile``
        among them: check it before relying on the encryption)."""
        return self._pp

    def entry_digest(self, identity: str) -> str:
        """The entry digest of *identity*.

        Raises:
            FEKeyError: For an identity not in the directory.
        """
        return self._entry(identity)["entry"]

    def _entry(self, identity) -> dict:
        check_identity(identity)
        entry = self._entries.get(identity)
        if entry is None:
            raise FEKeyError("identity {!r} is not in the directory".format(identity))
        return entry

    def _scheme(self, identity: str) -> SIFEFullySecRingLWE:
        if identity not in self._crypto:
            config = dict(self._crypto_config, keys={"pp": self._pp, "sk": self._entry(identity)["pk"]})
            self._crypto[identity] = SIFEFullySecRingLWE(config)
        return self._crypto[identity]

    def encrypt(self, identity: str, x: list) -> dict:
        """Encrypt the vector (or, with packed parameters, the matrix) *x* for
        *identity* (:class:`~pyfe4ai.core.formats.identity.IdentityCiphertext`)."""
        entry = self._entry(identity)["entry"]
        ct = self._scheme(identity).encrypt(x)
        return IdentityCiphertext.new(version=IDENTITY_DIRECTORY_VERSION, identity=identity, entry=entry,
                                      ct=ct).to_dict()

    def encrypt_lst_ndarray(self, identity: str, lst_ndarray: list) -> dict:
        """Encrypt ndarrays (``eta = 1``) for *identity*
        (:class:`~pyfe4ai.core.formats.identity.IdentityLayers`)."""
        entry = self._entry(identity)["entry"]
        layers = self._scheme(identity).encrypt_lst_ndarray(lst_ndarray)
        return IdentityLayers.new(version=IDENTITY_DIRECTORY_VERSION, identity=identity, entry=entry,
                                  layers=layers).to_dict()

    def _checked_pair(self, ct: dict, dk: dict, kind) -> str:
        """The identity of *ct* and *dk*, once both are bound to the same entry
        of this directory."""
        for obj, what in ((ct, "ciphertext"), (dk, "key")):
            if not isinstance(obj, dict) or obj.get("version") != IDENTITY_DIRECTORY_VERSION:
                raise FEValidationError("not an identity-directory {} of this format".format(what))
        identity = ct.get("identity")
        if dk.get("identity") != identity:
            raise FEKeyError("the key was issued to another identity than the ciphertext's")
        entry = self._entry(identity)["entry"]
        if not _same(ct.get("entry"), entry):
            raise FEKeyError("the ciphertext was encrypted under another entry of {!r}".format(identity))
        if not _same(dk.get("entry"), entry):
            raise FEKeyError("the key was issued under another entry of {!r}".format(identity))
        if kind not in ct or "dk" not in dk:
            raise FEValidationError("the ciphertext lacks {} or the key lacks dk".format(kind))
        return identity

    def decrypt(self, ct: dict, dk: dict, fusion_weight: list):
        """``<x, y>`` (or the list of a packed ciphertext's inner products), as
        ``sife/fullysec_ring_lwe`` decrypts it.

        Raises:
            FEKeyError: If the ciphertext and the key are not bound to the same
                entry of this directory.
        """
        identity = self._checked_pair(ct, dk, "ct")
        return self._scheme(identity).decrypt(ct["ct"], dk["dk"], fusion_weight)

    def decrypt_lst_ndarray_ct(self, layers: dict, dk: dict, fusion_weight: list) -> list | None:
        """Decrypt encrypted ndarrays (:meth:`encrypt_lst_ndarray`).

        Raises:
            FEKeyError: As :meth:`decrypt`.
        """
        identity = self._checked_pair(layers, dk, "layers")
        return self._scheme(identity).decrypt_lst_ndarray_ct(layers["layers"], dk=dk["dk"],
                                                             fusion_weight=fusion_weight)
