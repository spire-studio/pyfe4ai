"""Slot public keys and the label KEM of the bounded-key schemes ``bcfe/*``: the reference algorithm.

A key of ``bcfe/*`` is a *slot* of the construction of Sahai and Seyalioglu,
"Worry-Free Encryption: Functional Encryption with Public Keys" (ACM CCS
2010, SS10), Sec. 3.3: one public key per key bit and bit value, ``P[j, b]``
for ``j = 0 .. m - 1`` and ``b = 0, 1``, of which the key holder knows the
secret keys of the *selected* positions ``(j, w_j)`` only, plus one extra key
bit ``g`` (SS10 Sec. 3.2, the guarded circuit ``C'(w || g, x) = C(w, x) AND
NOT g``) with the single position ``(g, 0)``. An encryption garbles
``guard(C)`` for every slot (:mod:`pyfe4ai.primitives.garbling`) and
encrypts each key-wire label ``L_j^b`` under ``P[j, b]``; the holder opens
the labels of ``w || 0`` and evaluates ``C(w, x)``. The byte layouts are
those of the functions below: :func:`kem_mask`, :func:`unselected_pk`,
:func:`setup_digest`, :func:`ephemeral_scalar`, :func:`table_index`.

* **Label KEM** -- hashed ElGamal over ristretto255 with one ephemeral key
  ``R = r B`` per (ciphertext, slot), shared by the ``2m + 1`` labels of the
  slot: ``c[j, b] = L_j^b xor H_K(pp_digest, ct_nonce, slot_id, j, b, R,
  r P[j, b])`` (:func:`kem_mask`). Sharing ``r`` across recipients is the
  multi-recipient setting of Bellare, Boldyreva and Staddon (PKC 2003);
  a direct proof under the strong Diffie-Hellman assumption (Abdalla,
  Bellare and Rogaway, CT-RSA 2001) covers it: every ``H_K`` input carries
  distinct metadata ``(pp_digest, ct_nonce, slot_id, j, b)``, and the
  public keys may be arbitrary group elements.
* **Unselected positions (h')** -- ``P[j, 1 - w_j] = H_P(setup_digest,
  slot_id, j, k)`` (:func:`unselected_pk`, RFC 9380 ``hash_to_ristretto255``
  with a DST of its own), ``k`` a 128-bit secret per slot: nobody knows
  their discrete logarithms, so there is no unselected secret key to erase
  (I4). The center keeps ``k`` private and hands it to the key holder,
  who recomputes the unselected positions (:func:`unselected_ok`, the h'
  part of SS10's CheckAuth, I13); ``k`` is never published (I14).
* **Issuance** -- h1 (:func:`new_slot`): the center draws the ``m + 1``
  selected secret keys; h2 (:func:`place_slot`): the holder hands in its
  ``m + 1`` public keys, which must decode and not be the identity (I8), and
  the center places them at ``(j, w_j)`` and ``(g, 0)``.
* **Randomness** -- explicit: ``rand(n)`` (issuance) and the 32-byte
  ``seed`` of a (ciphertext, slot) (:func:`encrypt_slot`), which yields,
  expanded together with the instance salt, both the garbling
  (:func:`~pyfe4ai.primitives.garbling.expand_seed`) and ``r``
  (:func:`ephemeral_scalar`). The backend operations ``bcfe_*`` draw them
  from the operating system for every call (I1); a fixed seed is for the
  tests only (I16).

Requirements of the security argument implemented here:
I1 (fresh ``Delta``, labels and ``r`` per (ciphertext, slot) from the
caller's seed), I2 and I15 (``H_K``, ``H_P`` and the setup digest are
domain-separated and read fixed-length or length-prefixed fields; ``H_P``
takes the fixed setup digest, not the pp digest), I3 (the g bit has the one
position ``(g, 0)``; ``L_g^1`` is never encrypted or sent, and this module
never forms it -- inside the garbling it is only the zero label of the
output of ``INV(g)``, by the free INV gate), I4 / h' (no unselected secret
key exists), I8 (h2 public keys decode and are not the identity), I13 (the
h' part of CheckAuth, with the length of the table).

Scalars cross this module as 32 little-endian bytes below the group order
``l``; points as their 32-byte canonical encodings (RFC 9496). The selected
secret keys, ``k``, ``w``, the seed, ``r``, ``Delta`` and the labels are
secret. This Python reference makes no constant-time claim (Python integers,
:mod:`~pyfe4ai.primitives.groups.ristretto255` and ``hashlib`` are not
constant time) and cannot wipe its copies; it selects by the bits of ``w``
with masks, not branches, and the native implementation of the backend
operations is the one that wipes.
"""

from __future__ import annotations

import hashlib
import hmac

from pyfe4ai.primitives import garbling
from pyfe4ai.primitives.groups import ristretto255 as r255

#: The order ``l`` of ristretto255: scalars are canonical below it.
ORDER = r255.L

#: u8 codes of the record fields in :func:`setup_digest`.
DATA_WIRES_HARDWIRED = 0
DATA_WIRES_LABELS = 1
GC_HASH_SHAKE128 = 0
KEM_RISTRETTO255_HASHED_ELGAMAL_V1 = 0
#: The record values of those fields and their codes.
DATA_WIRES_CODES = {"hardwired": DATA_WIRES_HARDWIRED, "labels": DATA_WIRES_LABELS}
GC_HASH_CODES = {"shake128": GC_HASH_SHAKE128}
KEM_CODES = {"ristretto255-hashed-elgamal-v1": KEM_RISTRETTO255_HASHED_ELGAMAL_V1}

#: Domain-separation strings.
KEM_DST = b"PYFE4AI-V06-BCFE-KEM"
UNSELECTED_PREFIX = b"PFE6-UNSEL"
UNSELECTED_DST = b"PYFE4AI-V06-BCFE-UNSEL-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_"
SETUP_DST = b"PYFE4AI-V06-BCFE-SETUP"
EPHEMERAL_DST = b"PYFE4AI-V06-BCFE-EPHEM"

#: Lengths in bytes.
DIGEST_BYTES = 32       # pp digest, setup digest, circuit digest, setup nonce
NONCE_BYTES = 16        # ciphertext nonce
SLOT_KEY_BYTES = 16     # k, the secret of the unselected positions (h')
POINT_BYTES = 32
SCALAR_BYTES = 32
SCALAR_DRAW_BYTES = 64  # random bytes per secret scalar (reduced mod l)
LABEL_BYTES = garbling.LABEL_BYTES
SEED_BYTES = garbling.SEED_BYTES


# ------------------------------------------------------------------ checks and encodings

def _as_bytes(data, what: str) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("{} must be bytes, not {}".format(what, type(data).__name__))
    return bytes(data)


def _fixed(data, length: int, what: str) -> bytes:
    data = _as_bytes(data, what)
    if len(data) != length:
        raise ValueError("{} must be {} bytes, got {}".format(what, length, len(data)))
    return data


def _uint(value, bits: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ValueError("{} must be an integer in [0, 2^{})".format(what, bits))
    return value


def _u32(value: int) -> bytes:
    return value.to_bytes(4, "little")


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "little")


def check_key_bits_count(m) -> int:
    """*m*, the number of key bits of ``C`` (``guard(C)`` has ``m + 1`` key
    wires), checked to be an integer in ``[1, 2^26 - 1)`` (``guard(C)`` stays
    below :data:`~pyfe4ai.primitives.garbling.MAX_CIRCUIT_WIRES`).

    Raises:
        ValueError: Otherwise.
    """
    if isinstance(m, bool) or not isinstance(m, int) or not 1 <= m < garbling.MAX_WIRES - 1:
        raise ValueError("m must be an integer in [1, 2^26 - 1), the key bits of the circuit")
    return m


def _key_bits(w_bits, m) -> list:
    """The ``m`` key bits of *w_bits*: a sequence of 0 / 1 (``m`` its length
    unless given) or packed (:func:`~pyfe4ai.primitives.garbling.pack_bits`,
    *m* required)."""
    if m is not None:
        check_key_bits_count(m)
    if isinstance(w_bits, (bytes, bytearray, memoryview)):
        if m is None:
            raise ValueError("packed key bits need m, the number of key bits")
        return garbling.unpack_bits(w_bits, m)
    bits = list(w_bits)
    if m is not None and len(bits) != m:
        raise ValueError("expected {} key bits, got {}".format(m, len(bits)))
    if not bits:
        raise ValueError("w must have at least one key bit")
    return garbling.unpack_bits(garbling.pack_bits(bits, len(bits)), len(bits))


def _cswap(bit: int, a: bytes, b: bytes) -> tuple:
    """``(a, b)`` for ``bit = 0`` and ``(b, a)`` for ``bit = 1``, by a mask
    (equal lengths)."""
    ia, ib = int.from_bytes(a, "little"), int.from_bytes(b, "little")
    t = (ia ^ ib) & -bit
    return (ia ^ t).to_bytes(len(a), "little"), (ib ^ t).to_bytes(len(b), "little")


def _xor16(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(LABEL_BYTES, "little")


def _chunks(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


def encode_scalar(s: int) -> bytes:
    """A scalar in ``[0, l)`` as 32 little-endian bytes.

    Raises:
        ValueError: If *s* is not an integer in ``[0, l)``.
    """
    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < ORDER:
        raise ValueError("a scalar must be an integer in [0, l)")
    return s.to_bytes(SCALAR_BYTES, "little")


def decode_scalar(data) -> int:
    """The scalar of 32 little-endian bytes, which must be canonical (below ``l``).

    Raises:
        TypeError: If *data* is not bytes.
        ValueError: If *data* is not 32 bytes or encodes a value ``>= l``.
    """
    data = _fixed(data, SCALAR_BYTES, "a scalar")
    s = int.from_bytes(data, "little")
    if s >= ORDER:
        raise ValueError("a scalar must be canonical (below the group order)")
    return s


def _draw(rand, nbytes: int) -> bytes:
    data = rand(nbytes)
    if not isinstance(data, (bytes, bytearray)) or len(data) != nbytes:
        raise ValueError("the random source must return {} bytes".format(nbytes))
    return bytes(data)


def _draw_scalar(rand) -> int:
    """A secret scalar: 64 bytes of *rand*, little-endian, mod ``l``; 0 raises."""
    s = int.from_bytes(_draw(rand, SCALAR_DRAW_BYTES), "little") % ORDER
    if s == 0:
        raise ValueError("a secret scalar is 0 (probability 2^-252): issue the slot again")
    return s


def byte_source(data: bytes):
    """``rand(n)`` handing out the bytes of *data* in order (the backend
    operations draw all bytes of a call at once); ``ValueError`` when they
    run out."""
    data = _as_bytes(data, "random bytes")
    pos = 0

    def rand(nbytes: int) -> bytes:
        nonlocal pos
        if pos + nbytes > len(data):
            raise ValueError("the random bytes ran out")
        pos += nbytes
        return data[pos - nbytes:pos]

    return rand


def table_index(m: int, j: int, b: int) -> int:
    """The index of position ``(j, b)`` in the public-key table and in the
    label ciphertexts of a slot with ``m`` key bits: ``2 j + b`` for ``j <
    m``, ``2 m`` for the g bit ``(m, 0)``.

    Raises:
        ValueError: For ``(m, 1)`` (the g bit has no position 1) or a
            position out of range.
    """
    check_key_bits_count(m)
    if (isinstance(b, bool) or not isinstance(b, int) or b not in (0, 1) or isinstance(j, bool)
            or not isinstance(j, int) or not 0 <= j <= m):
        raise ValueError("no position ({}, {}) in a slot of {} key bits".format(j, b, m))
    if j == m and b:
        raise ValueError("the g bit has the one position (m, 0)")
    return 2 * j + b


def _decode_point(data: bytes, what: str):
    try:
        point = r255.decode(data)
    except ValueError:
        raise ValueError("{} is not a canonical ristretto255 encoding".format(what)) from None
    if r255.equal(point, r255.IDENTITY):
        raise ValueError("{} is the identity".format(what))
    return point


def _points(data, count: int, what: str) -> list:
    data = _as_bytes(data, what)
    if len(data) != POINT_BYTES * count:
        raise ValueError("{}: expected {} points of {} bytes ({} bytes), got {} bytes".format(
            what, count, POINT_BYTES, POINT_BYTES * count, len(data)))
    return _chunks(data, POINT_BYTES)


# ------------------------------------------------------------------ the hashes

def _mask_prefix(pp_digest, ct_nonce, slot_id) -> bytes:
    return (KEM_DST + _fixed(pp_digest, DIGEST_BYTES, "pp_digest") + _fixed(ct_nonce, NONCE_BYTES, "ct_nonce")
            + _u64(_uint(slot_id, 64, "slot_id")))


def _mask(prefix: bytes, j: int, b: int, big_r: bytes, z: bytes) -> bytes:
    return hashlib.shake_256(prefix + _u32(j) + bytes([b]) + big_r + z).digest(LABEL_BYTES)


def kem_mask(pp_digest: bytes, ct_nonce: bytes, slot_id: int, j: int, b: int, R: bytes, Z: bytes) -> bytes:
    """``H_K = SHAKE-256("PYFE4AI-V06-BCFE-KEM" || pp_digest(32) || ct_nonce(16)
    || u64(slot_id) || u32(j) || u8(b) || R(32) || Z(32))[0:16]``: the mask
    of the label of key wire *j* of ``guard(C)`` for bit *b* (the g bit is
    ``j = m``, only with ``b = 0``), ``R`` the ephemeral key and ``Z = r
    P[j, b]`` as 32-byte encodings (not decoded here).

    Raises:
        TypeError: For byte fields that are not bytes.
        ValueError: For a field of another length, ``slot_id`` not in ``[0,
            2^64)``, ``j`` not in ``[0, 2^32)`` or ``b`` not 0 or 1.
    """
    prefix = _mask_prefix(pp_digest, ct_nonce, slot_id)
    j = _uint(j, 32, "j")
    if isinstance(b, bool) or not isinstance(b, int) or b not in (0, 1):
        raise ValueError("b must be 0 or 1")
    return _mask(prefix, j, b, _fixed(R, POINT_BYTES, "R"), _fixed(Z, POINT_BYTES, "Z"))


def unselected_pk(setup_digest: bytes, slot_id: int, j: int, k: bytes) -> bytes:
    """``H_P``: the public key of the unselected position of key bit *j*
    (h'), ``hash_to_ristretto255(msg, DST)`` of RFC 9380 (suite
    ``ristretto255_XMD:SHA-512_R255MAP_RO_``) with ``msg = "PFE6-UNSEL" ||
    setup_digest(32) || u64(slot_id) || u32(j) || k(16)`` and the DST
    :data:`UNSELECTED_DST`, as its 32-byte encoding. *k* is secret.

    Raises:
        TypeError: For byte fields that are not bytes.
        ValueError: For a field of another length, ``slot_id`` not in ``[0,
            2^64)`` or ``j`` not in ``[0, 2^32)``.
    """
    msg = (UNSELECTED_PREFIX + _fixed(setup_digest, DIGEST_BYTES, "setup_digest")
           + _u64(_uint(slot_id, 64, "slot_id")) + _u32(_uint(j, 32, "j")) + _fixed(k, SLOT_KEY_BYTES, "k"))
    return r255.encode(r255.hash_to_group(msg, UNSELECTED_DST))


def _code(value, codes: dict, what: str) -> int:
    if isinstance(value, str) and value in codes:
        return codes[value]
    if not isinstance(value, bool) and isinstance(value, int) and value in codes.values():
        return value
    raise ValueError("unknown {} {!r}; choose one of {} (or their codes)".format(what, value, ", ".join(codes)))


def setup_digest(scheme_id, bcfe_version: int, circuit_digest: bytes, data_wires, gc_hash, kem, label_bits: int,
                 setup_nonce: bytes) -> bytes:
    """The setup digest, the fixed input of ``H_P``:
    ``SHAKE-256("PYFE4AI-V06-BCFE-SETUP" || u32(len(id)) || id ||
    u32(bcfe_version) || circuit_digest(32) || u8(data_wires) || u8(gc_hash)
    || u8(kem) || u32(label_bits) || setup_nonce(32))[0:32]``.

    ``id`` is the scheme id (UTF-8 for a string). *data_wires*, *gc_hash*
    and *kem* are the record values (``"hardwired"`` / ``"labels"``,
    ``"shake128"``, ``"ristretto255-hashed-elgamal-v1"``) or their codes
    (:data:`DATA_WIRES_CODES`, :data:`GC_HASH_CODES`, :data:`KEM_CODES`).
    ``max_keys`` and the slot table are not in it (the bound may be raised;
    the input of ``H_P`` must stay fixed for the life of the setup).

    Raises:
        TypeError: For a scheme id that is neither str nor bytes, or byte
            fields that are not bytes.
        ValueError: For a field of another length, an unknown value or
            code, or an integer out of the range of its field.
    """
    if isinstance(scheme_id, str):
        sid = scheme_id.encode("utf-8")
    else:
        sid = _as_bytes(scheme_id, "scheme_id")
    _uint(len(sid), 32, "the length of the scheme id")
    return hashlib.shake_256(
        SETUP_DST + _u32(len(sid)) + sid + _u32(_uint(bcfe_version, 32, "bcfe_version"))
        + _fixed(circuit_digest, DIGEST_BYTES, "circuit_digest")
        + bytes([_code(data_wires, DATA_WIRES_CODES, "data_wires"), _code(gc_hash, GC_HASH_CODES, "gc_hash"),
                 _code(kem, KEM_CODES, "kem")])
        + _u32(_uint(label_bits, 32, "label_bits")) + _fixed(setup_nonce, DIGEST_BYTES, "setup_nonce")
    ).digest(DIGEST_BYTES)


def ephemeral_scalar(salt: bytes, seed: bytes) -> int:
    """The ephemeral key ``r`` of a (ciphertext, slot):
    ``SHAKE-256("PYFE4AI-V06-BCFE-EPHEM" || salt(56) || seed(32))[0:64]``
    little-endian mod ``l``, *salt* the instance salt (:func:`instance_salt`).
    The seed also yields the garbling, through its own domain
    (:func:`~pyfe4ai.primitives.garbling.expand_seed`). Secret.

    The salt is bound in so that one seed used for two (ciphertext, slot)
    instances still gives unrelated ``r``: defence in depth against a
    repeated seed. Design choice; the alternative was the seed alone,
    relying only on a fresh seed per instance from the operating system
    (I1).

    Raises:
        TypeError: If *salt* or *seed* is not bytes.
        ValueError: If *salt* is not 56 bytes, *seed* not 32 bytes, or ``r =
            0`` (probability 2^-252).
    """
    salt = _fixed(salt, garbling.SALT_BYTES, "the instance salt")
    seed = _fixed(seed, SEED_BYTES, "the seed")
    r = int.from_bytes(hashlib.shake_256(EPHEMERAL_DST + salt + seed).digest(SCALAR_DRAW_BYTES), "little") % ORDER
    if r == 0:
        raise ValueError("the ephemeral scalar is 0 (probability 2^-252): draw a new seed")
    return r


def instance_salt(pp_digest: bytes, ct_nonce: bytes, slot_id: int) -> bytes:
    """The 56-byte instance salt of the gate hash, ``pp_digest || ct_nonce ||
    u64(slot_id)``: it binds the gate hash, the seed expansion and ``r``
    to the instance (I11, the gate hash takes the instance salt).

    Raises:
        TypeError, ValueError: For fields of another type or length.
    """
    return (_fixed(pp_digest, DIGEST_BYTES, "pp_digest") + _fixed(ct_nonce, NONCE_BYTES, "ct_nonce")
            + _u64(_uint(slot_id, 64, "slot_id")))


# ------------------------------------------------------------------ issuance

def _table(setup_digest_: bytes, slot_id: int, w: list, k: bytes, selected: list) -> bytes:
    """The ``2m + 1`` public keys: ``selected[j]`` at ``(j, w_j)``, ``H_P`` at
    ``(j, 1 - w_j)``, ``selected[m]`` at ``(g, 0)``."""
    m = len(w)
    out = []
    for j in range(m):
        p0, p1 = _cswap(w[j], selected[j], unselected_pk(setup_digest_, slot_id, j, k))
        out.append(p0)
        out.append(p1)
    out.append(selected[m])
    return b"".join(out)


def new_slot(setup_digest: bytes, slot_id: int, w_bits, rand, m: int | None = None) -> dict:
    """Issue a slot for the key bits *w_bits* (h1, with h'): the center draws
    the secret keys of the selected positions and of the g bit.

    Randomness, read from ``rand(n)`` in this order: 64 bytes per secret
    scalar for ``j = 0 .. m - 1`` (position ``(j, w_j)``) and then the g bit,
    each read little-endian and reduced mod ``l`` (a scalar 0 raises, no
    redraw); then the 16 bytes of ``k``. ``64 (m + 1) + 16`` bytes in all.

    Args:
        setup_digest: The 32-byte setup digest (:func:`setup_digest`).
        slot_id: The slot number, in ``[0, 2^64)``.
        w_bits: The ``m`` key bits: a sequence of 0 / 1, or packed with *m*.
        rand: ``rand(n) -> n bytes``.
        m: The number of key bits (required for packed *w_bits*).

    Returns:
        ``{"scalars": 32 B x (m + 1)`` -- the secret keys of ``(j, w_j)``,
        ``j = 0 .. m - 1``, then of ``(g, 0)``, canonical little-endian;
        secret --, ``"pks": 32 B x (2m + 1)`` -- the table, ``P[j, 0] ||
        P[j, 1]`` for ``j = 0 .. m - 1``, then ``P[m, 0]`` --, ``"k": 16 B``
        -- secret, never published``}``.

    Raises:
        TypeError: For byte fields that are not bytes.
        ValueError: For malformed fields or key bits, random bytes of
            another length, or a secret scalar 0.
    """
    setup_digest = _fixed(setup_digest, DIGEST_BYTES, "setup_digest")
    slot_id = _uint(slot_id, 64, "slot_id")
    w = _key_bits(w_bits, m)
    scalars = [_draw_scalar(rand) for _ in range(len(w) + 1)]
    k = _draw(rand, SLOT_KEY_BYTES)
    selected = [r255.encode(r255.mul_base(s)) for s in scalars]
    return {"scalars": b"".join(encode_scalar(s) for s in scalars),
            "pks": _table(setup_digest, slot_id, w, k, selected), "k": k}


def place_slot(setup_digest: bytes, slot_id: int, w_bits, holder_pks: bytes, rand, m: int | None = None) -> dict:
    """Issue a slot from the key holder's public keys (h2, with h'): the
    holder's ``m + 1`` public keys go to ``(j, w_j)`` and ``(g, 0)``, the
    unselected positions are derived from a fresh ``k``.

    Every holder key must be a canonical encoding of a group element other
    than the identity (I8); no proof of knowledge of its secret key is asked
    (the label KEM is secure for arbitrary public keys). Randomness: the 16
    bytes of ``k`` from ``rand(16)``, after the inputs are checked (the
    backend operation ``bcfe_place_slot`` reads its 16 bytes after checking
    *m* and *w* and before the other inputs; see its contract).

    Args:
        setup_digest, slot_id, w_bits, m: As :func:`new_slot`.
        holder_pks: ``32 B x (m + 1)``, the holder's keys for ``(j, w_j)``,
            ``j = 0 .. m - 1``, then for ``(g, 0)``.
        rand: ``rand(n) -> n bytes``.

    Returns:
        ``{"pks": 32 B x (2m + 1), "k": 16 B}`` as :func:`new_slot`; ``k`` is
        secret (the center hands it to the holder for :func:`unselected_ok`).

    Raises:
        TypeError: For byte fields that are not bytes.
        ValueError: For malformed fields or key bits, another number of
            holder keys than ``m + 1``, a holder key that does not decode or
            is the identity, or random bytes of another length.
    """
    setup_digest = _fixed(setup_digest, DIGEST_BYTES, "setup_digest")
    slot_id = _uint(slot_id, 64, "slot_id")
    w = _key_bits(w_bits, m)
    holder = _points(holder_pks, len(w) + 1, "holder_pks")
    for i, p in enumerate(holder):
        _decode_point(p, "holder public key {}".format(i))
    k = _draw(rand, SLOT_KEY_BYTES)
    return {"pks": _table(setup_digest, slot_id, w, k, holder), "k": k}


def unselected_ok(setup_digest: bytes, slot_id: int, w_bits, k: bytes, pks: bytes, m: int | None = None) -> bool:
    """The h' part of CheckAuth (I13): whether the table *pks* has exactly
    ``2m + 1`` entries and every unselected position ``(j, 1 - w_j)`` holds
    ``H_P(setup_digest, slot_id, j, k)``. The scheme checks the rest: the
    holder's own keys at ``(j, w_j)`` and ``(g, 0)``. *w_bits* and *k* are
    secret.

    Raises:
        TypeError: For byte fields that are not bytes.
        ValueError: For a malformed ``setup_digest``, ``slot_id``, ``k`` or
            key bits (a table of another length is ``False``, not an error).
    """
    setup_digest = _fixed(setup_digest, DIGEST_BYTES, "setup_digest")
    slot_id = _uint(slot_id, 64, "slot_id")
    k = _fixed(k, SLOT_KEY_BYTES, "k")
    w = _key_bits(w_bits, m)
    pks = _as_bytes(pks, "pks")
    if len(pks) != POINT_BYTES * (2 * len(w) + 1):
        return False
    table = _chunks(pks, POINT_BYTES)
    ok = True
    for j, bit in enumerate(w):
        _, unselected = _cswap(bit, table[2 * j], table[2 * j + 1])
        ok &= hmac.compare_digest(unselected, unselected_pk(setup_digest, slot_id, j, k))
    return ok


# ------------------------------------------------------------------ encryption and decryption

def guarded_key_bits(parsed: garbling.ParsedCircuit) -> int:
    """``m`` of a parsed ``guard(C)`` (:meth:`pyfe4ai.circuits.Circuit.guard`):
    its ``m' = m + 1`` key wires minus the g bit.

    Raises:
        ValueError: If the circuit is not ``guard(C)`` of a circuit with at
            least one key bit: ``m' < 2``, or not ending in ``INV(g)``
            followed by ``AND(o_k, INV(g))`` per output in order (the
            outputs), or the g bit read by another gate.
    """
    m_prime, n, gates, n_out = parsed.m, parsed.n, parsed.gates, parsed.n_out
    g = m_prime - 1
    inv = len(gates) - n_out - 1
    if m_prime < 2 or inv < 0 or gates[inv] != (garbling.OP_INV, g, 0):
        raise ValueError("the circuit is not guard(C): no INV(g) before the output gates")
    ng = m_prime + n + inv
    for i in range(n_out):
        op, a, b = gates[inv + 1 + i]
        if op != garbling.OP_AND or b != ng or a >= ng or parsed.outputs[i] != ng + 1 + i:
            raise ValueError("the circuit is not guard(C): output {} is not AND(o, INV(g))".format(i))
    for i, (op, a, b) in enumerate(gates):
        if i != inv and (a == g or (op != garbling.OP_INV and b == g)):
            raise ValueError("the circuit is not guard(C): gate {} reads the g bit".format(i))
    return g


def _check_mode(mode) -> None:
    if mode not in garbling.MODES:
        raise ValueError("unknown garbling mode {!r}; choose one of {}".format(mode, ", ".join(garbling.MODES)))


def encrypt_slot(circuit: bytes, mode: str, pp_digest: bytes, ct_nonce: bytes, slot_id: int, x_bits, pks: bytes,
                 seed: bytes) -> dict:
    """The component of one slot in a ciphertext: garble ``guard(C)`` for
    *x_bits* and encrypt the labels of its key wires under the slot's table.

    From the 32-byte *seed* and the instance salt (:func:`instance_salt`):
    the garbling (``Delta`` and the zero labels,
    :func:`~pyfe4ai.primitives.garbling.garble` with that salt) and ``r``
    (:func:`ephemeral_scalar`), ``R = r B``. The label ciphertexts follow
    the table: ``c[j, b] = L_j^b xor H_K(pp_digest, ct_nonce, slot_id, j,
    b, R, r P[j, b])`` with ``L_j^1 = L_j^0 xor Delta`` for ``j = 0 .. m -
    1``, then ``c[m, 0]`` alone: ``L_g^1`` is never encrypted or sent, nor
    formed here (I3).

    Args:
        circuit: The ``CIRC v1`` encoding of ``guard(C)`` (``m + 1`` key wires).
        mode: ``"hardwired"`` or ``"labels"``.
        pp_digest: 32 bytes; ct_nonce: 16 bytes; slot_id: in ``[0, 2^64)``.
        x_bits: The ``n`` data bits, a list of 0 / 1 or packed (secret).
        pks: The slot's table, ``32 B x (2m + 1)``.
        seed: 32 bytes, fresh from a CSPRNG for every (ciphertext, slot)
            (I1); fixed only in tests (I16). Secret.

    Returns:
        ``{"R": 32 B, "label_cts": 16 B x (2m + 1), "tables": 16 B x table
        entries, "decode": the n_out decoding bits packed, "data_active": 16
        B x n (mode "labels") or b""}``. ``Delta``, the seed, ``r`` and the
        zero labels are not returned.

    Raises:
        TypeError: For inputs that are not bytes (*x_bits* may be a list).
        ValueError: For a malformed circuit or one that is not ``guard(C)``,
            an unknown mode, fields of another length, data bits that are
            not ``n`` bits, a table of another length than ``2m + 1`` points,
            a table entry that does not decode or is the identity, or ``r =
            0``.
    """
    parsed = garbling.parse_circuit(circuit)
    m = guarded_key_bits(parsed)
    _check_mode(mode)
    salt = instance_salt(pp_digest, ct_nonce, slot_id)
    prefix = _mask_prefix(pp_digest, ct_nonce, slot_id)
    table = [_decode_point(p, "public key {} of the slot".format(i))
             for i, p in enumerate(_points(pks, 2 * m + 1, "pks"))]
    r = ephemeral_scalar(salt, seed)
    big_r = r255.encode(r255.mul_base(r))
    garbled = garbling.garble(circuit, mode, salt, x_bits, seed)
    delta = garbled["delta"]
    key0 = _chunks(garbled["key0"], LABEL_BYTES)
    cts = []
    for j in range(m):
        labels = (key0[j], _xor16(key0[j], delta))
        for b in (0, 1):
            z = r255.encode(r255.mul(table[2 * j + b], r))
            cts.append(_xor16(labels[b], _mask(prefix, j, b, big_r, z)))
    z = r255.encode(r255.mul(table[2 * m], r))  # the g bit: (m, 0) only
    cts.append(_xor16(key0[m], _mask(prefix, m, 0, big_r, z)))
    return {"R": big_r, "label_cts": b"".join(cts), "tables": garbled["tables"], "decode": garbled["decode"],
            "data_active": garbled["data_active"]}


def recover_labels(pp_digest: bytes, ct_nonce: bytes, slot_id: int, R: bytes, label_cts: bytes, scalars: bytes,
                   w_bits, m: int | None = None) -> bytes:
    """The ``m + 1`` active key labels a key holder opens: ``c[j, w_j] xor
    H_K(..., j, w_j, R, s_j R)`` for ``j = 0 .. m - 1``, then ``c[m, 0] xor
    H_K(..., m, 0, R, s_g R)`` (the g bit is always 0). Returns them
    concatenated (16 bytes each); they are secret.

    Raises:
        TypeError: For byte fields that are not bytes.
        ValueError: For fields of another length, ``R`` not a canonical
            encoding or the identity, ``label_cts`` of another length than
            ``16 (2m + 1)``, ``scalars`` of another length than ``32 (m +
            1)`` or a scalar not below ``l``, malformed key bits.
    """
    prefix = _mask_prefix(pp_digest, ct_nonce, slot_id)
    w = _key_bits(w_bits, m)
    m = len(w)
    big_r = _fixed(R, POINT_BYTES, "R")
    point_r = _decode_point(big_r, "R")
    cts = _as_bytes(label_cts, "label_cts")
    if len(cts) != LABEL_BYTES * (2 * m + 1):
        raise ValueError("label_cts: expected {} entries of {} bytes ({} bytes), got {} bytes".format(
            2 * m + 1, LABEL_BYTES, LABEL_BYTES * (2 * m + 1), len(cts)))
    cts = _chunks(cts, LABEL_BYTES)
    secret = _as_bytes(scalars, "scalars")
    if len(secret) != SCALAR_BYTES * (m + 1):
        raise ValueError("scalars: expected {} scalars of {} bytes, got {} bytes".format(
            m + 1, SCALAR_BYTES, len(secret)))
    s = [decode_scalar(part) for part in _chunks(secret, SCALAR_BYTES)]
    out = []
    for j, bit in enumerate(w):
        c, _ = _cswap(bit, cts[2 * j], cts[2 * j + 1])
        z = r255.encode(r255.mul(point_r, s[j]))
        out.append(_xor16(c, _mask(prefix, j, bit, big_r, z)))
    z = r255.encode(r255.mul(point_r, s[m]))
    out.append(_xor16(cts[2 * m], _mask(prefix, m, 0, big_r, z)))
    return b"".join(out)


def decrypt_slot(circuit: bytes, mode: str, pp_digest: bytes, ct_nonce: bytes, slot_id: int, R: bytes,
                 label_cts: bytes, tables: bytes, decode: bytes, data_active: bytes, scalars: bytes, w_bits) -> bytes:
    """Decrypt one slot's component of a ciphertext with the slot's key:
    open the active labels of ``w || 0`` (:func:`recover_labels`) and
    evaluate the garbling (:func:`~pyfe4ai.primitives.garbling.evaluate`).
    Returns the ``n_out`` output bits, packed: ``C(w, x)`` for a well-formed
    component, unrelated bits for a component of another slot, ciphertext
    or key.

    Args:
        circuit, mode, pp_digest, ct_nonce, slot_id: As :func:`encrypt_slot`.
        R, label_cts, tables, decode, data_active: The component.
        scalars: ``32 B x (m + 1)``, the slot's secret keys (secret).
        w_bits: The ``m`` key bits (packed, or a list of 0 / 1; secret).

    Raises:
        TypeError: For inputs that are not bytes.
        ValueError: For a malformed circuit or one that is not ``guard(C)``,
            an unknown mode, any field of another length, ``R`` not a
            canonical encoding or the identity, a scalar not below ``l``,
            another number of table entries or data labels, decoding bits
            that are not ``n_out`` packed bits.
    """
    parsed = garbling.parse_circuit(circuit)
    m = guarded_key_bits(parsed)
    _check_mode(mode)
    salt = instance_salt(pp_digest, ct_nonce, slot_id)
    key_active = recover_labels(pp_digest, ct_nonce, slot_id, R, label_cts, scalars, w_bits, m=m)
    return garbling.evaluate(circuit, mode, salt, tables, decode, key_active, data_active)
