"""
Byte codecs of the Ring-LWE ciphertexts and values.

* Format 1 (:func:`encode_binary_ciphertext`, :func:`decode_binary_ciphertext`):
  the ``ct`` of the MIFE / MCFE Ring-LWE crypto objects with
  ``ct_format="binary"`` (:data:`~pyfe4ai.core.versions.CT_FORMATS`);
  decryption reads it and the JSON form (``ct0``, ``ct1``) alike
  (:func:`canonical_ring_ciphertext`).
* Format 2 (:func:`encode_threshold_ciphertext`,
  :func:`decode_threshold_ciphertext`, :func:`threshold_ciphertext_header`,
  :func:`threshold_ciphertext_from_format1`): the ciphertexts of the
  threshold MCFE over Ring-LWE under binary public parameters. Every value
  is strictly below ``q``, so an encoding is unique.
* Values (:func:`encode_values`, :func:`decode_values`): ``ceil(bits(q) / 8)``-byte
  little-endian words -- the polynomials of binary key shares and the
  binary partial decryptions.

Both formats start with ``b"PFRL"`` and a version byte (1 or 2); a reader of
one refuses the other.
"""

from __future__ import annotations

from pyfe4ai.core.exceptions import FESchemeError

#: Magic of the binary formats.
_BIN_MAGIC = b"PFRL"
_BIN_VERSION = 1
_BIN_HEADER = 10


def encode_binary_ciphertext(ct0: list, ct1: list, q, ring_n: int) -> bytes:
    """Binary format 1 of a ciphertext with ``eta`` rows ``ct0`` and ``ct1``.

    ``b"PFRL"``, version 1, ``wb = ceil(bits(q) / 8)``, ``eta`` (2 bytes,
    little-endian), ``log2 ring_n`` and a zero byte; then the coefficients of
    the ``ct0`` rows and of ``ct1``, ``ring_n`` each, as ``wb``-byte
    little-endian integers in ``[0, q)``.
    """
    wb = (int(q).bit_length() + 7) // 8
    eta = len(ct0)
    head = _BIN_MAGIC + bytes([_BIN_VERSION, wb, eta & 0xFF, eta >> 8, ring_n.bit_length() - 1, 0])
    q = int(q)
    return head + b"".join(
        (int(v) % q).to_bytes(wb, "little") for row in list(ct0) + [ct1] for v in row
    )


def decode_binary_ciphertext(data, q, eta: int, ring_n: int) -> tuple:
    """``(ct0, ct1)`` of a format-1 ciphertext, coefficients reduced mod q.

    Raises:
        ValueError: If the header or the length does not match.
    """
    wb = (int(q).bit_length() + 7) // 8
    data = bytes(data)
    head = _BIN_MAGIC + bytes([_BIN_VERSION, wb, eta & 0xFF, eta >> 8, ring_n.bit_length() - 1, 0])
    if len(data) != _BIN_HEADER + (eta + 1) * ring_n * wb or data[:_BIN_HEADER] != head:
        raise ValueError("not a format-1 ciphertext of these parameters")
    q = int(q)
    vals = [
        int.from_bytes(data[i:i + wb], "little") % q
        for i in range(_BIN_HEADER, len(data), wb)
    ]
    rows = [vals[r * ring_n:(r + 1) * ring_n] for r in range(eta + 1)]
    return rows[:eta], rows[eta]


#: Binary format 2, of the threshold Ring-LWE MCFE with ``ct_format="binary"``
#: public parameters (:func:`encode_threshold_ciphertext`).
_BIN2_VERSION = 2
_BIN2_HEADER = 14


def _bin2_head(wb: int, eta: int, ring_n: int, k: int) -> bytes:
    return _BIN_MAGIC + bytes([_BIN2_VERSION, wb, eta & 0xFF, eta >> 8, ring_n.bit_length() - 1, 0]) + int(
        k).to_bytes(4, "little")


def encode_threshold_ciphertext(ct0: list, ct1: list, k: int, q, ring_n: int) -> bytes:
    """Binary format 2 of a ciphertext with batch length *k*.

    ``b"PFRL"``, version 2, ``wb = ceil(bits(q) / 8)``, ``eta`` (2 bytes,
    little-endian), ``log2 ring_n``, a zero byte and ``k`` (4 bytes,
    little-endian); then the first ``k`` coefficients of each ``ct0`` row and
    the ``ring_n`` coefficients of ``ct1``, as ``wb``-byte little-endian
    integers in ``[0, q)``. Decryption reads only the first ``k`` coefficients
    of ``ct0`` (coefficient-wise), while ``ct1`` enters a ring product.
    """
    q = int(q)
    wb = (q.bit_length() + 7) // 8
    rows = [row[:k] for row in ct0] + [ct1]
    return _bin2_head(wb, len(ct0), ring_n, k) + b"".join((int(v) % q).to_bytes(wb, "little") for row in rows for v in row)


def threshold_ciphertext_from_format1(data, k: int, q, eta: int, ring_n: int) -> bytes:
    """Format 2 (:func:`encode_threshold_ciphertext`) of a format-1 ciphertext
    whose values are below q (as this library writes them; a reader of format
    2 checks that again).

    Raises:
        ValueError: If *data* is not a format-1 ciphertext of these parameters.
    """
    wb = (int(q).bit_length() + 7) // 8
    data = bytes(data)
    head = _BIN_MAGIC + bytes([_BIN_VERSION, wb, eta & 0xFF, eta >> 8, ring_n.bit_length() - 1, 0])
    row = ring_n * wb
    if len(data) != _BIN_HEADER + (eta + 1) * row or data[:_BIN_HEADER] != head or not 1 <= k <= ring_n:
        raise ValueError("not a format-1 ciphertext of these parameters")
    view = memoryview(data)
    parts = [_bin2_head(wb, eta, ring_n, k)]
    parts += [view[_BIN_HEADER + r * row:_BIN_HEADER + r * row + k * wb] for r in range(eta)]
    parts.append(view[_BIN_HEADER + eta * row:])
    return b"".join(parts)


def threshold_ciphertext_header(data, q, eta: int, ring_n: int) -> int:
    """The batch length ``k`` of a format-2 ciphertext of these parameters
    (header and length checked; the values are not).

    Raises:
        ValueError: If the header or the length does not match.
    """
    wb = (int(q).bit_length() + 7) // 8
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("a format-2 ciphertext is bytes")
    data = memoryview(data)
    if len(data) < _BIN2_HEADER:
        raise ValueError("not a format-2 ciphertext of these parameters")
    k = int.from_bytes(data[10:_BIN2_HEADER], "little")
    if (
        bytes(data[:10]) != _bin2_head(wb, eta, ring_n, 0)[:10] or not 1 <= k <= ring_n
        or len(data) != _BIN2_HEADER + (eta * k + ring_n) * wb
    ):
        raise ValueError("not a format-2 ciphertext of these parameters")
    return k


def decode_values(data, q, count: int) -> list:
    """*count* ``ceil(bits(q) / 8)``-byte little-endian integers, each strictly
    below *q* (a value ``>= q`` would be a second encoding of one below it).

    Raises:
        ValueError: For another length or a value ``>= q``.
    """
    q = int(q)
    wb = (q.bit_length() + 7) // 8
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != count * wb:
        raise ValueError("expected {} values of {} bytes".format(count, wb))
    data = bytes(data)
    out = [int.from_bytes(data[i:i + wb], "little") for i in range(0, count * wb, wb)]
    if any(v >= q for v in out):
        raise ValueError("a value is not below q")
    return out


def encode_values(values, q) -> bytes:
    """Integers in ``[0, q)`` (reduced mod q) as ``ceil(bits(q) / 8)``-byte little-endian words."""
    q = int(q)
    wb = (q.bit_length() + 7) // 8
    return b"".join((int(v) % q).to_bytes(wb, "little") for v in values)


def decode_threshold_ciphertext(data, q, eta: int, ring_n: int) -> tuple:
    """``(k, ct0, ct1)`` of a format-2 ciphertext: ``eta`` rows of ``k``
    coefficients and ``ring_n`` coefficients, all strictly below q.

    Raises:
        ValueError: If the header, the length or a value does not match.
    """
    k = threshold_ciphertext_header(data, q, eta, ring_n)
    data = bytes(data)
    vals = decode_values(data[_BIN2_HEADER:], q, eta * k + ring_n)
    return k, [vals[r * k:(r + 1) * k] for r in range(eta)], vals[eta * k:]


def canonical_ring_ciphertext(ct, q, eta: int, ring_n: int) -> tuple:
    """Return ``(k, ct0, ct1)`` of a BMMS21 ciphertext with coefficients in ``[0, q)``.

    *ct* holds ``k`` and either ``ct0`` and ``ct1`` (JSON format) or ``ct``
    (binary format, :func:`encode_binary_ciphertext`).

    Raises:
        FESchemeError: Unless *ct* has ``eta`` polynomials ``ct0`` and one
            polynomial ``ct1`` of ``ring_n`` integer coefficients each, and a
            batch length ``1 <= k <= ring_n``.
    """
    try:
        k = ct["k"]
        if "ct" in ct and "ct0" not in ct:
            if not isinstance(ct["ct"], (bytes, bytearray, memoryview)):
                raise TypeError("binary ciphertext must be bytes")
            ct0, ct1 = decode_binary_ciphertext(ct["ct"], q, eta, ring_n)
        else:
            ct0 = [[int(str(v)) % q for v in row] for row in ct["ct0"]]
            ct1 = [int(str(v)) % q for v in ct["ct1"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise FESchemeError("malformed ciphertext") from exc
    if (
        not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= ring_n
        or len(ct0) != eta or any(len(row) != ring_n for row in ct0) or len(ct1) != ring_n
    ):
        raise FESchemeError("malformed ciphertext")
    return k, ct0, ct1
