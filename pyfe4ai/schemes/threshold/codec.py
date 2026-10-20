"""
Encodings of the threshold artifacts and their strict parsing.

* Key material (nested dicts and lists of polynomials) as value lists --
  ints, ``mpz`` or decimal strings, read by :func:`strict_mpz` -- or binary
  words (:func:`binary_material`: ``ceil(bits(q) / 8)`` bytes per value,
  little-endian, strictly below ``q``), and the kinds of its polynomials
  (:func:`poly_kinds`, :func:`uniform_material`).
* The ciphertexts of the threshold MCFE over Ring-LWE
  (:mod:`pyfe4ai.schemes.mcfe.ring_lwe_threshold`): JSON, tagged over their
  JSON form, or format 2 of
  :func:`~pyfe4ai.core.formats.binary.encode_threshold_ciphertext`
  with what their tags (:func:`binary_tag_message`) and the noise seeds
  (:func:`binary_request_digests`) hash; strict reading of both, and the
  digests of canonical ciphertexts (:func:`ciphertexts_digest`).
* Its partial decryptions: values as decimal strings (JSON) or binary words.

A format-2 encoding is unique (a value ``>= q`` is refused): no client can
re-encode one request to draw a second noise for the same key part.
"""

from __future__ import annotations

import hashlib
import json

import gmpy2 as gp

from pyfe4ai.core.exceptions import FEKeyError, FESchemeError
from pyfe4ai.core.formats.ring_lwe import RingLWECiphertext
from pyfe4ai.core.formats.ring_lwe import RingLWELayer
from pyfe4ai.core.formats.threshold import TMCFERingLWEPartial
from pyfe4ai.core.formats.binary import canonical_ring_ciphertext
from pyfe4ai.core.formats.binary import decode_threshold_ciphertext
from pyfe4ai.core.formats.binary import decode_values
from pyfe4ai.core.formats.binary import encode_values
from pyfe4ai.core.formats.binary import threshold_ciphertext_header

# key material


def _value_bytes(modulus) -> int:
    return (int(modulus).bit_length() + 7) // 8


def _map_polys(material, f):
    """*material* with every list of values (a polynomial) or bytes leaf replaced by ``f(leaf)``."""
    if isinstance(material, dict):
        return {key: _map_polys(value, f) for key, value in material.items()}
    if isinstance(material, (list, tuple)) and (not material or any(isinstance(v, (dict, list, tuple)) for v in material)):
        return [_map_polys(value, f) for value in material]
    return f(material)


def binary_material(material, modulus):
    """*material* (nested dicts and lists of values) with every polynomial as
    ``ceil(bits(q) / 8)``-byte little-endian words (values reduced mod q):
    the binary key shares of the threshold Ring-LWE MCFE."""
    from pyfe4ai.core.formats.binary import encode_values

    q = int(modulus)
    return _map_polys(material, lambda leaf: encode_values(leaf if isinstance(leaf, (list, tuple)) else [leaf], q))


def _bytes_layout(material, width: int):
    """The :func:`~pyfe4ai.schemes.threshold.sharing.piece_layout` of binary key material (a bytes leaf: its
    number of values), or ``None`` for leaves that are not whole words."""
    if isinstance(material, (bytes, bytearray)):
        return len(material) // width if len(material) % width == 0 else None
    if isinstance(material, dict):
        out = {key: _bytes_layout(value, width) for key, value in material.items()}
        return None if any(v is None for v in out.values()) else out
    if isinstance(material, (list, tuple)):
        out = [_bytes_layout(value, width) for value in material]
        return None if any(v is None for v in out) else out
    return None


def _map_bytes(material, f):
    """*material* with every bytes leaf replaced by ``f(leaf)``."""
    if isinstance(material, (bytes, bytearray)):
        return f(material)
    if isinstance(material, dict):
        return {key: _map_bytes(value, f) for key, value in material.items()}
    if isinstance(material, (list, tuple)):
        return [_map_bytes(value, f) for value in material]
    return material


def _bytes_leaves(material):
    """The bytes leaves of *material* in the order of
    :func:`~pyfe4ai.schemes.threshold.sharing._flatten_like`."""
    if isinstance(material, (bytes, bytearray)):
        yield bytes(material)
    elif isinstance(material, dict):
        for value in material.values():
            yield from _bytes_leaves(value)
    elif isinstance(material, (list, tuple)):
        for value in material:
            yield from _bytes_leaves(value)


def strict_mpz(value):
    """A key-share value as ``mpz``: an int, an ``mpz`` or a decimal string
    (floats and other types are refused rather than truncated).

    Raises:
        FEKeyError: For any other value.
    """
    if isinstance(value, float) or not (isinstance(value, (int, str)) or type(value).__name__ == "mpz"):
        raise FEKeyError("malformed key share: values are ints or decimal strings, not {}".format(type(value).__name__))
    try:
        return gp.mpz(value)
    except (ValueError, TypeError) as exc:
        raise FEKeyError("malformed key share value") from exc


def poly_kinds(material) -> set:
    """The kinds of the polynomials of key material: ``"bytes"`` (bytes or
    bytearray), ``"values"`` (a list or tuple of values; only its first
    element is looked at) and ``"other"`` (anything else, e.g. a
    ``memoryview``). The walk is over polynomials, not values."""
    if isinstance(material, (bytes, bytearray)):
        return {"bytes"}
    if isinstance(material, dict):
        return set().union(*(poly_kinds(v) for v in material.values()))
    if isinstance(material, (list, tuple)):
        if not material:
            return set()
        if isinstance(material[0], (dict, list, tuple, bytes, bytearray, memoryview)):
            return set().union(*(poly_kinds(v) for v in material))
        return {"values"}
    if isinstance(material, (int, str)) or type(material).__name__ == "mpz":
        return {"values"}  # a single value
    return {"other"}


def uniform_material(material, modulus):
    """*material* with polynomials of one kind: bytes polynomials decoded to
    values when it mixes them with value lists (what the Python dealing reads
    them as), else unchanged.

    Raises:
        FEKeyError: For polynomials that are neither (``"other"``) or malformed bytes.
    """
    kinds = poly_kinds(material)
    if "other" in kinds:
        raise FEKeyError("malformed key material: polynomials must be value lists or bytes")
    if kinds != {"bytes", "values"}:
        return material
    from pyfe4ai.core.formats.binary import decode_values

    width = _value_bytes(modulus)

    def decode(leaf):
        try:
            return decode_values(leaf, modulus, len(leaf) // width)
        except ValueError as exc:
            raise FEKeyError("malformed binary key material") from exc

    return _map_bytes(material, decode)


def _has_bytes(material) -> bool:
    """Whether *material* holds binary polynomials (its first leaf is bytes;
    key material is of one kind throughout, and only the first path is
    walked, not every value)."""
    while True:
        if isinstance(material, (bytes, bytearray)):
            return True
        if isinstance(material, dict) and material:
            material = next(iter(material.values()))
        elif isinstance(material, (list, tuple)) and material:
            material = material[0]
        else:
            return False


def _unflatten_bytes(layout, blob: bytes, wb: int):
    """*blob* (the values of :func:`~pyfe4ai.schemes.threshold.sharing._flatten_like`
    as ``wb``-byte words) in the
    structure of *layout*, every list of values (or single value) one bytes
    object."""
    pos = 0

    def build(node):
        nonlocal pos
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        if isinstance(node, list):
            return [build(value) for value in node]
        if node == 0:  # an empty list (no chunk keys) stays a list
            return []
        count = 1 if node is None else node
        pos += count * wb
        return blob[pos - count * wb:pos]

    out = build(layout)
    if pos != len(blob):
        raise FESchemeError("key pieces of different layouts")
    return out


def _poly_values(poly, q, ring_n: int) -> list:
    """A key-share polynomial as ``mpz``: a list or tuple of ``ring_n`` ints,
    ``mpz`` or decimal strings (:func:`strict_mpz`),
    or binary (``ceil(bits(q)/8)``-byte words, strictly below q).

    Raises:
        FEKeyError: For malformed binary values.
    """
    if isinstance(poly, (bytes, bytearray)):
        try:
            return [gp.mpz(v) for v in decode_values(poly, q, ring_n)]
        except ValueError as exc:
            raise FEKeyError("malformed binary key share") from exc
    if not isinstance(poly, (list, tuple)) or len(poly) != ring_n:
        raise FEKeyError("malformed key share: a polynomial is ring_n values or ring_n words of bytes")
    return [strict_mpz(v) for v in poly]


def _check_key_polys(polys, q, ring_n: int) -> None:
    """Raise unless every polynomial is ``ring_n`` values (a list or tuple) or
    ``ring_n`` words of ``ceil(bits(q)/8)`` bytes: a shorter one would be
    served as a shorter answer, a longer one read in part.

    Raises:
        FEKeyError: Otherwise.
    """
    words = ring_n * ((int(q).bit_length() + 7) // 8)
    for poly in polys:
        if isinstance(poly, (bytes, bytearray)):
            ok = len(poly) == words
        else:
            ok = isinstance(poly, (list, tuple)) and len(poly) == ring_n
        if not ok:
            raise FEKeyError("malformed key share: a polynomial is ring_n values or ring_n words of bytes")


# ciphertexts


def ciphertexts_digest(canonical: dict) -> str:
    """Return the digest of ciphertexts given in canonical form (``{nid: values}``).

    Partial decryptions seed their flooding noise with it, so it must depend
    only on the values a server computes with, never on how a request encodes
    them: otherwise re-encoding one request draws fresh noise, and averaging
    the partials removes it.
    """
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


_TAG_DOMAIN = "pyfe4ai/tmcfe-ring-lwe/ciphertext-tag/v1"


# binary public parameters (format 2): what the tags and the noise seeds hash
_TAG_DOMAIN_BIN = b"pyfe4ai/tmcfe-ring-lwe/ciphertext-tag/v2-binary|"


_CT0_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/ct0/v2|"


_CT1_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/ct1/v2|"


_KEY_INPUTS_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/key-inputs/v2|"


_CT_DIGEST_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/ct-digest/v2|"


def binary_ciphertext_digests(ct: dict, q, eta: int, ring_n: int) -> tuple:
    """``(k, d0, d1)`` of a format-2 ciphertext dict: its batch length and
    ``d0 = SHA-256(domain || header || ct0 rows)``, ``d1 = SHA-256(domain ||
    ct1)`` (the values are checked when read: :func:`~pyfe4ai.core.formats.binary.decode_threshold_ciphertext`).

    Raises:
        ValueError: Unless ``ct["ct"]`` is a format-2 ciphertext of these
            parameters whose batch length is ``ct["k"]``.
    """
    ct = RingLWECiphertext(ct)
    data = ct.ct
    k = threshold_ciphertext_header(data, q, eta, ring_n)
    if type(ct.get("k")) is not int or ct.k != k:
        raise ValueError("the ciphertext's k is not its header's")
    split = 14 + eta * k * ((int(q).bit_length() + 7) // 8)
    view = memoryview(data)
    d0 = hashlib.sha256(_CT0_DOMAIN)
    d0.update(view[:split])
    d1 = hashlib.sha256(_CT1_DOMAIN)
    d1.update(view[split:])
    return k, d0.digest(), d1.digest()


def binary_tag_message(pp_digest: bytes, encoded_label: str, ct: dict, digests: tuple) -> bytes:
    """What the tag of a format-2 ciphertext covers (HMAC-SHA256 under the
    client-server key): a domain, the digest of the public parameters
    (:func:`~pyfe4ai.schemes.threshold.binding.pp_binding` over
    :func:`~pyfe4ai.schemes.threshold.binding.tmcfe_ring_lwe_binding_fields`), the (chunk) label, the ciphertext's ``label_hash``
    code (for an ndarray chunk its layer's; see ``_chunk_with_code``), the
    batch length and the digests of :func:`binary_ciphertext_digests`.

    Raises:
        ValueError: For a label hash code that is not a byte.
    """
    k, d0, d1 = digests
    code = RingLWECiphertext(ct).get("label_hash", 0)
    if type(code) is not int or not 0 <= code < 256:
        raise ValueError("malformed label hash code")
    label = encoded_label.encode("utf-8")
    return (_TAG_DOMAIN_BIN + pp_digest + len(label).to_bytes(4, "big") + label + bytes([code])
            + k.to_bytes(4, "big") + d0 + d1)


def _client_ids(dict_ct) -> list:
    """The client ids of a request as ``(json.dumps(nid) bytes, nid)``, sorted."""
    return sorted((json.dumps(nid).encode("utf-8"), nid) for nid in dict_ct)


def binary_request_digests(digests: dict) -> tuple:
    """``(ct_digest, key_inputs)`` (hex) of a request of format-2 ciphertexts
    from their :func:`binary_ciphertext_digests` ``{nid: (k, d0, d1)}``:
    SHA-256 over the client ids (``json.dumps``, sorted) with ``(k, d0, d1)``
    each, and over the ids with ``d1`` only -- the values the key part depends
    on, so the flooding noise is seeded with them (a format-2 encoding is
    unique: no re-encoding of one request draws fresh noise)."""
    ct_digest, key_inputs = hashlib.sha256(_CT_DIGEST_DOMAIN), hashlib.sha256(_KEY_INPUTS_DOMAIN)
    for name, nid in _client_ids(digests):
        k, d0, d1 = digests[nid]
        prefix = len(name).to_bytes(4, "big") + name
        ct_digest.update(prefix + k.to_bytes(4, "big") + d0 + d1)
        key_inputs.update(prefix + d1)
    return ct_digest.hexdigest(), key_inputs.hexdigest()


def _tag_payload(encoded_label: str, ct: dict) -> list:
    """What a tMCFE-Ring-LWE ciphertext tag covers: the (chunk) label, the
    ciphertext as serialised and its ``label_hash`` code when it has one --
    for an ndarray chunk its layer's code, see ``_chunk_with_code``
    (ciphertexts of versions <= 0.3.0 have none: their tags stay valid)."""
    ct = RingLWECiphertext(ct)
    payload = [encoded_label, ct.get("k"), ct.get("ct0"), ct.get("ct1")]
    if "label_hash" in ct:
        payload.append(ct.label_hash)
    return payload


def _chunk_with_code(ct, layer) -> dict:
    """An ndarray chunk as tagged and decrypted: with its layer's
    ``label_hash`` code (layers, not chunks, carry it)."""
    if isinstance(ct, dict) and isinstance(layer, dict):
        layer = RingLWELayer(layer)
        if "label_hash" in layer and "label_hash" not in RingLWECiphertext(ct):
            return RingLWECiphertext.extend(ct, label_hash=layer.label_hash).to_dict()
    return ct


def check_ciphertext_format(dict_ct: dict, binary: bool) -> None:
    """Refuse ciphertexts of the other format than the public parameters':
    JSON ones (``ct0``, ``ct1``) under binary public parameters, binary ones
    (``ct``) under JSON ones.

    Raises:
        FESchemeError: For a ciphertext of the other format.
    """
    if binary:
        for nid, ct in dict_ct.items():  # no JSON ciphertexts under binary public parameters
            rec = RingLWECiphertext(ct)
            if isinstance(ct, dict) and ("ct0" in rec or "ct1" in rec):
                raise FESchemeError(
                    "the ciphertext of {} is JSON; the public parameters' format is binary (ct_format, the "
                    "default): encrypt with these public parameters".format(nid))
        return
    for nid, ct in dict_ct.items():  # no binary ciphertexts under JSON public parameters
        if isinstance(ct, dict) and "ct" in RingLWECiphertext(ct):
            raise FESchemeError("the ciphertext of {} is binary; the public parameters' format is JSON".format(nid))


def read_json_ciphertexts(dict_ct: dict, q: int, eta: int, ring_n: int) -> dict:
    """``{nid: (k, ct0, ct1)}`` of JSON ciphertexts, read strictly
    (:func:`~pyfe4ai.core.formats.binary.canonical_ring_ciphertext`).

    Raises:
        FESchemeError: For a malformed ciphertext.
    """
    cts = {}
    for nid, ct in dict_ct.items():
        try:
            cts[nid] = canonical_ring_ciphertext(ct, q, eta, ring_n)
        except FESchemeError as exc:
            raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
    return cts


def read_binary_ciphertexts(dict_ct: dict, k: int, q: int, eta: int, ring_n: int) -> dict:
    """``{nid: (k, ct0, ct1)}`` of format-2 ciphertexts of batch length *k*
    (the Python reading of format 2, strict:
    :func:`~pyfe4ai.core.formats.binary.decode_threshold_ciphertext`).

    Raises:
        FESchemeError: For a malformed ciphertext.
    """
    cts = {}
    for nid, ct in dict_ct.items():
        try:
            _, ct0, ct1 = decode_threshold_ciphertext(RingLWECiphertext(ct).ct, q, eta, ring_n)
        except ValueError as exc:
            raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
        cts[nid] = (k, ct0, ct1)
    return cts


def native_json_inputs(ctx, dict_ct: dict, sk_y: dict, fusion_weight: dict, z_share, eta: int,
                       ring_n: int) -> tuple | None:
    """JSON ciphertexts read natively for a partial decryption: ``(nids, ks,
    packed, ct_digest, key_inputs)``, the client ids sorted, the digests
    those of :func:`ciphertexts_digest`; ``None`` when the Python path must
    read them (values other than exact ints and plain decimal strings,
    other layouts -- also of the key share *sk_y*, *z_share* -- and
    malformed ciphertexts, whose errors it raises). *ctx*: a ring of the
    backend on which the threshold kernels run natively
    (:attr:`~pyfe4ai.backend.interface.Ring.threshold_native`)."""
    if not all(isinstance(nid, str) for nid in dict_ct) or len(z_share) != ring_n:
        return None
    nids = sorted(dict_ct)  # the key order of json.dumps(..., sort_keys=True)
    ks, rows = [], []
    for nid in nids:
        ct = dict_ct[nid]
        if not isinstance(ct, dict) or "ct0" not in RingLWECiphertext(ct) or "ct1" not in RingLWECiphertext(ct):
            return None
        ct = RingLWECiphertext(ct)
        k, ct0 = ct.get("k"), ct.ct0
        if type(k) is not int or not 1 <= k <= ring_n or type(ct0) not in (list, tuple) or len(ct0) != eta:
            return None
        if len(sk_y[nid]) != ring_n or len(fusion_weight[nid]) != eta:
            return None
        ks.append(k)
        rows.append((ct0, ct.ct1))
    out = ctx.threshold_inputs([json.dumps(nid) for nid in nids], ks, rows, eta)
    if out is None:
        return None
    packed, ct_text, key_text = out
    return nids, ks, packed, hashlib.sha256(ct_text).hexdigest(), hashlib.sha256(key_text).hexdigest()


# partial decryptions


def partial_values(values: list, q, binary: bool):
    """Values of a partial decryption as its format writes them: binary words
    (``ceil(bits(q) / 8)`` bytes each) or decimal strings."""
    return encode_values(values, q) if binary else [gp.digits(v) for v in values]


def check_partial_format(partials: list, ct_format: str) -> None:
    """Refuse partial decryptions whose ``w`` and ``c_y`` are not of the
    public parameters' format: bytes (binary) or sequences (JSON).

    Raises:
        FESchemeError: Otherwise.
    """
    values_type = (bytes, bytearray) if ct_format == "binary" else (list, tuple)
    if any(not isinstance(p.get("w"), values_type) or not isinstance(p.get("c_y"), values_type)
           for p in map(TMCFERingLWEPartial, partials)):
        raise FESchemeError("partial decryptions not in the public parameters' format ({})".format(ct_format))


def partial_batch_length(first: dict, partials: list, ring_n: int, binary: bool) -> int:
    """The batch length ``k`` of partial decryptions: an int in ``[1, ring_n]``;
    every JSON partial holds at least ``k`` values (a shorter one would give a
    shorter result).

    Raises:
        FESchemeError: Otherwise.
    """
    k = TMCFERingLWEPartial(first).k
    if type(k) is not int or not 1 <= k <= ring_n or not binary and any(
            len(p.w) < k or len(p.c_y) < k for p in map(TMCFERingLWEPartial, partials)):
        raise FESchemeError("malformed partial decryption")
    return k


def json_partial_difference(first: dict, partials: list, q) -> list:
    """``c_y - sum w`` mod q of JSON partial decryptions, value by value."""
    total = [gp.mpz(v) for v in TMCFERingLWEPartial(first).c_y]
    for part in partials:
        total = [(a - gp.mpz(v)) % q for a, v in zip(total, TMCFERingLWEPartial(part).w)]
    return total


def binary_partial_difference(first: dict, partials: list, q, k: int) -> list:
    """``c_y - sum w`` mod q of binary partial decryptions (``k`` values of
    ``ceil(bits(q) / 8)`` bytes each, strictly below q).

    Raises:
        FESchemeError: For a malformed partial decryption.
    """
    try:
        total = decode_values(TMCFERingLWEPartial(first).c_y, q, k)
        for part in partials:
            total = [(a - v) % q for a, v in zip(total, decode_values(TMCFERingLWEPartial(part).w, q, k))]
    except ValueError as exc:
        raise FESchemeError("malformed partial decryption") from exc
    return total
