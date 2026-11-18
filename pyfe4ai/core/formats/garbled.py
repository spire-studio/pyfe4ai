"""
The serialized objects of the bounded-key circuit FE ``bcfe/gc_ristretto255``
(:mod:`pyfe4ai.schemes.bcfe.gc_ristretto255`): public parameters with their
slot list, functional keys, ciphertexts, and the two messages of the
holder-generated issuance (h2): the key request and the grant. Also the
three digests the records carry or the authority keeps, with their exact
byte encodings below.

**Encodings in the dicts.** The records are JSON-able dicts. Digests, nonces,
the ephemeral key ``R`` (a ristretto255 point, as the other ristretto255
records of the library write points) and the slot secret ``k`` are lowercase
hex strings of their bytes; the larger binary fields (the circuit encoding,
the public-key table, scalars, packed bits, label ciphertexts, garbled
tables, data labels) are base64 strings (RFC 4648, standard alphabet, with
padding) of their bytes. A reader accepts only these canonical forms
(:func:`from_hex`, :func:`from_b64`). Packed bits are least significant bit
first (bit ``i`` is bit ``i % 8`` of byte ``i // 8``, the unused high bits of
the last byte zero). Lengths are fixed by the circuit: ``m`` key bits,
``n`` data bits, ``n_out`` output bits of ``C``; ``guard(C)`` has ``m + 1``
key wires (:meth:`pyfe4ai.circuits.Circuit.guard`).

**Digests** (``u8``, ``u32``, ``u64``: fixed-length little-endian integers;
``||``: concatenation; every field fixed-length or length-prefixed, so each
encoding is injective -- requirement I15 of the 0.6 security argument):

* slot digest (:func:`slot_digest`)::

    SHAKE-256("PYFE4AI-V06-BCFE-SLOT" || setup_digest(32) || u64(slot_id)
              || opt_text(label) || opt_digest(policy_digest)
              || attribute_set_digest(32) || u32(2m + 1) || pks(32 (2m + 1)))[0:32]

  with ``opt_text(None) = u8(0)``, ``opt_text(s) = u8(1) || u32(len(b)) ||
  b`` (``b`` the UTF-8 bytes of ``s``), ``opt_digest(None) = u8(0)``,
  ``opt_digest(d) = u8(1) || d(32)``. The setup digest
  (:func:`pyfe4ai.primitives.bcfe_kem.setup_digest`) binds the slot to its
  setup (scheme id, ``bcfe_version``, circuit digest, ``data_wires``,
  ``gc_hash``, ``kem``, ``label_bits``, ``setup_nonce``).
* pp digest (:func:`pp_digest`)::

    SHAKE-256("PYFE4AI-V06-BCFE-PP" || setup_digest(32) || u32(len(gc)) || gc
              || u32(len(desc)) || desc || u64(max_keys) || u64(slot_list_version)
              || slot_digest_1(32) || ... || slot_digest_v(32))[0:32]

  with ``gc`` the ASCII bytes of the ``gc`` marker, ``desc`` the UTF-8 bytes
  of the circuit description as canonical JSON
  (:meth:`pyfe4ai.circuits.ParamCircuit.describe_json`: sorted keys, no
  spaces) and ``v = slot_list_version`` slot digests in list order. With the
  setup digest it covers every field of the public parameters: the scheme
  id, the markers, ``label_bits``, ``data_wires``, the circuit (its encoding
  through the circuit digest, its description and layouts directly), the
  setup nonce, ``max_keys``, the version and every slot record.
* dedup key (:func:`dedup_key`; the authority's private index, never
  published, I16)::

    SHAKE-256("PYFE4AI-V06-BCFE-DEDUP" || u32(len(id)) || id || circuit_digest(32)
              || u32(len(r)) || r || w)[0:32]

  with ``id`` the UTF-8 scheme id, ``r`` the UTF-8 bytes of the
  authenticated requester id (a non-empty ``str``) and ``w`` the ``m`` key
  bits packed (``ceil(m / 8)`` bytes, fixed by the circuit digest).

The records' version markers and how a reader refuses a missing or unknown
one are in :mod:`pyfe4ai.core.versions` (:func:`~pyfe4ai.core.versions.check_bcfe_markers`).
"""

from __future__ import annotations

import base64
import binascii
import hashlib

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record

#: Domain-separation strings of the three digests.
SLOT_DST = b"PYFE4AI-V06-BCFE-SLOT"
PP_DST = b"PYFE4AI-V06-BCFE-PP"
DEDUP_DST = b"PYFE4AI-V06-BCFE-DEDUP"
#: Length of every digest (slot, pp, dedup, setup, circuit, policy, attribute set).
DIGEST_BYTES = 32
POINT_BYTES = 32

# ---------------------------------------------------------------------------
# field encodings


def b64(data: bytes) -> str:
    """*data* as base64 (standard alphabet, padded)."""
    return base64.b64encode(bytes(data)).decode("ascii")


def from_b64(value, what: str, length: int | None = None) -> bytes:
    """The bytes of a canonical base64 string (of *length* bytes if given).

    Raises:
        ValueError: For anything else; the message names *what*, never the value.
    """
    if not isinstance(value, str):
        raise ValueError("{} must be a base64 string".format(what))
    try:
        data = base64.b64decode(value.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError):
        raise ValueError("{} is not base64".format(what)) from None
    if b64(data) != value:
        raise ValueError("{} is not canonical base64".format(what))
    if length is not None and len(data) != length:
        raise ValueError("{} must be {} bytes, got {}".format(what, length, len(data)))
    return data


def from_hex(value, what: str, length: int) -> bytes:
    """The *length* bytes of a lowercase hex string.

    Raises:
        ValueError: For anything else; the message names *what*, never the value.
    """
    if not isinstance(value, str) or len(value) != 2 * length:
        raise ValueError("{} must be {} lowercase hex digits".format(what, 2 * length))
    try:
        data = bytes.fromhex(value)
    except ValueError:
        raise ValueError("{} is not hex".format(what)) from None
    if data.hex() != value:
        raise ValueError("{} must be lowercase hex".format(what))
    return data


def _u32(value: int) -> bytes:
    return int(value).to_bytes(4, "little")


def _u64(value: int) -> bytes:
    return int(value).to_bytes(8, "little")


def _fixed(data: bytes, length: int, what: str) -> bytes:
    if not isinstance(data, (bytes, bytearray)) or len(data) != length:
        raise ValueError("{} must be {} bytes".format(what, length))
    return bytes(data)


def _text(value: str, what: str) -> bytes:
    if type(value) is not str:
        raise ValueError("{} must be a str".format(what))
    data = value.encode("utf-8")
    if len(data) >= 1 << 32:
        raise ValueError("{} is too long".format(what))
    return _u32(len(data)) + data


# ---------------------------------------------------------------------------
# digests


def slot_digest(setup_digest: bytes, slot_id: int, label, policy_digest, attribute_set_digest: bytes,
                pks: bytes) -> bytes:
    """The slot digest (see the module docstring): *label* a ``str`` or
    ``None``, *policy_digest* 32 bytes or ``None``, *pks* ``32 (2m + 1)``
    bytes.

    Raises:
        ValueError: For fields of another type or length.
    """
    if isinstance(slot_id, bool) or not isinstance(slot_id, int) or not 0 <= slot_id < 1 << 64:
        raise ValueError("slot_id must be an integer in [0, 2^64)")
    pks = bytes(pks)
    if not pks or len(pks) % POINT_BYTES or (len(pks) // POINT_BYTES) % 2 != 1:
        raise ValueError("pks must be 2m + 1 points of 32 bytes")
    out = SLOT_DST + _fixed(setup_digest, DIGEST_BYTES, "setup_digest") + _u64(slot_id)
    out += b"\x00" if label is None else b"\x01" + _text(label, "label")
    out += b"\x00" if policy_digest is None else b"\x01" + _fixed(policy_digest, DIGEST_BYTES, "policy_digest")
    out += _fixed(attribute_set_digest, DIGEST_BYTES, "attribute_set_digest")
    out += _u32(len(pks) // POINT_BYTES) + pks
    return hashlib.shake_256(out).digest(DIGEST_BYTES)


def pp_digest(setup_digest: bytes, gc: str, description_json: str, max_keys: int, slot_digests) -> bytes:
    """The pp digest (see the module docstring) over *slot_digests* (32
    bytes each, in list order; their number is ``slot_list_version``).

    Raises:
        ValueError: For fields of another type, length or range.
    """
    digests = [_fixed(d, DIGEST_BYTES, "a slot digest") for d in slot_digests]
    if isinstance(max_keys, bool) or not isinstance(max_keys, int) or not 0 <= max_keys < 1 << 64:
        raise ValueError("max_keys must be an integer in [0, 2^64)")
    if type(gc) is not str or not gc.isascii():
        raise ValueError("gc must be an ASCII str")
    out = (PP_DST + _fixed(setup_digest, DIGEST_BYTES, "setup_digest") + _text(gc, "gc")
           + _text(description_json, "the circuit description") + _u64(max_keys) + _u64(len(digests))
           + b"".join(digests))
    return hashlib.shake_256(out).digest(DIGEST_BYTES)


def dedup_key(scheme_id: str, circuit_digest: bytes, requester: str, w_packed: bytes) -> bytes:
    """The authority's dedup key of a request (see the module docstring);
    secret: never published, logged or put in an exception (I16).

    Raises:
        ValueError: For a requester that is not a non-empty ``str``, or fields
            of another type or length.
    """
    if type(requester) is not str or not requester:
        raise ValueError("the requester id must be a non-empty str")
    if not isinstance(w_packed, (bytes, bytearray)):
        raise ValueError("w must be packed bits (bytes)")
    out = (DEDUP_DST + _text(scheme_id, "the scheme id") + _fixed(circuit_digest, DIGEST_BYTES, "circuit_digest")
           + _text(requester, "the requester id") + bytes(w_packed))
    return hashlib.shake_256(out).digest(DIGEST_BYTES)


# ---------------------------------------------------------------------------
# records


class BCFESetup(Record):
    """The setup of a ``bcfe`` instance: the public parameters without the
    slot list (the authority's ``setup.json``; also the first fields of
    :class:`BCFEPublicParameters`)."""

    scheme = Field("the scheme id, ``\"bcfe/gc_ristretto255\"``")
    bcfe_version = Field("int, the format version (:data:`~pyfe4ai.core.versions.BCFE_VERSION`)")
    kem = Field("the label KEM, ``\"ristretto255-hashed-elgamal-v1\"``")
    gc = Field("the garbling scheme, ``\"half-gates-v1\"``")
    gc_hash = Field("the gate hash, ``\"shake128\"``")
    label_bits = Field("int, the label length in bits (128)")
    data_wires = Field("``\"hardwired\"`` (data bits known to the garbler) or ``\"labels\"`` (SS10's input labels)")
    circuit = Field("the circuit description (:meth:`~pyfe4ai.circuits.ParamCircuit.describe`: name, params, "
                    "sizes, digest, the three layouts)")
    circuit_encoding = Field("base64, the ``CIRC v1`` encoding of ``C`` (without the guard bit)")
    circuit_digest = Field("hex, ``SHAKE-256(\"PYFE4AI-V06-BCFE-CIRCUIT\" || encoding)[0:32]``")
    setup_nonce = Field("hex, 32 random bytes drawn by ``setup()``")
    setup_digest = Field("hex, :func:`~pyfe4ai.primitives.bcfe_kem.setup_digest` of the fields above")


class BCFESlot(Record):
    """A slot of the slot list: the public half of one issued key."""

    slot_id = Field("int, never reused (ids increase and may skip a number)")
    label = Field("str or ``None``, the name the authority gave the slot (encryptors select slots by it)")
    policy_digest = Field("hex or ``None``, :func:`~pyfe4ai.policy.policy_digest` of the issuance policy")
    attribute_set_digest = Field("hex, :func:`~pyfe4ai.policy.attribute_set_digest` of the requester's attributes")
    pks = Field("base64, ``32 (2m + 1)`` bytes: ``P[j, 0] || P[j, 1]`` for ``j < m``, then ``P[m, 0]`` (the g bit)")
    slot_digest = Field("hex, :func:`slot_digest`")


class BCFEPublicParameters(BCFESetup):
    """Public parameters of ``bcfe/gc_ristretto255``: the fields of
    :class:`BCFESetup`, then the slot list and the pp digest."""

    max_keys = Field("int, the cap on the number of slots (may be raised, never below the slots listed)")
    slot_list_version = Field("int, the number of slots listed")
    slots = Field("list of :class:`BCFESlot` dicts, in issue order")
    pp_digest = Field("hex, :func:`pp_digest`")


class BCFEKey(Record):
    """A functional key: the secret keys of one slot's selected positions
    and the bits ``w`` that select them (its holder knows ``w``)."""

    scheme = Field("the scheme id")
    bcfe_version = Field("int, the format version")
    setup_digest = Field("hex, the setup the slot belongs to")
    circuit_digest = Field("hex, the digest of ``C``")
    slot_id = Field("int")
    slot_digest = Field("hex, the digest of the slot record")
    pp_digest = Field("hex, the pp digest of the slot list as of this slot's issue")
    slot_list_version = Field("int, the slot list version as of this slot's issue (ciphertexts of an "
                              "earlier version do not cover the slot)")
    scalars = Field("base64, ``32 (m + 1)`` bytes: the secret keys of ``(j, w_j)``, ``j < m``, then of ``(g, 0)``, "
                    "canonical little-endian; secret")
    w = Field("base64, the ``m`` key bits packed; secret")


class BCFEComponent(Record):
    """The part of a ciphertext for one slot."""

    slot_id = Field("int")
    R = Field("hex, the ephemeral key ``r B`` of this (ciphertext, slot)")
    label_cts = Field("base64, ``16 (2m + 1)`` bytes: the key-wire labels, each under its slot public key")
    tables = Field("base64, ``16`` bytes per garbled-table entry")
    decode = Field("base64, the ``n_out`` output decoding bits packed")
    data_active = Field("base64, ``16 n`` bytes of data labels (``data_wires = \"labels\"``), else empty")


class BCFECiphertext(Record):
    """A ciphertext: one component per selected slot of the slot list."""

    scheme = Field("the scheme id")
    bcfe_version = Field("int, the format version")
    setup_digest = Field("hex, the setup of the public parameters it was made with")
    circuit_digest = Field("hex")
    slot_list_version = Field("int, the version of the slot list it was made with")
    pp_digest = Field("hex, the digest of those public parameters (an input of every hash of the ciphertext)")
    ct_nonce = Field("hex, 16 random bytes drawn for this ciphertext")
    components = Field("list of :class:`BCFEComponent` dicts, in slot-list order")


class BCFEKeyRequest(Record):
    """The key holder's request of the holder-generated issuance (h2): the
    function and the holder's public keys for its selected positions."""

    scheme = Field("the scheme id")
    bcfe_version = Field("int, the format version")
    setup_digest = Field("hex, the setup the request is for")
    circuit_digest = Field("hex")
    w = Field("the key values (a dict by field name of the key layout); secret towards everyone but the authority")
    pks = Field("base64, ``32 (m + 1)`` bytes: the holder's keys for ``(j, w_j)``, ``j < m``, then ``(g, 0)``")


class BCFEGrant(Record):
    """The authority's answer to a request (h2): the slot made for it and
    the slot secret ``k`` with which the holder checks the slot (CheckAuth)."""

    scheme = Field("the scheme id")
    bcfe_version = Field("int, the format version")
    setup_digest = Field("hex")
    circuit_digest = Field("hex")
    slot = Field("the :class:`BCFESlot` dict of the slot")
    k = Field("hex, the 16-byte slot secret of the unselected positions; secret (it reveals ``w``)")
    pp_digest = Field("hex, the pp digest of the slot list as of this slot's issue")
    slot_list_version = Field("int, the slot list version as of this slot's issue")
