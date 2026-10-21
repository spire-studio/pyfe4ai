"""Encrypted data containers of the ML adapter and their serialisation:
:class:`EncryptedGradient` and :class:`EncryptedFeatures` (``to_dict`` /
``from_dict``), whose wire forms are the records :class:`GradientWire`
(with :class:`ElementLayerWire` or :class:`PackedLayerWire` layers) and
:class:`FeaturesWire` (:mod:`pyfe4ai.core.formats.record`)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record
from pyfe4ai.core.formats.ring_lwe import RingLWECiphertext
from pyfe4ai.core.formats.ring_lwe import RingLWELayer

_BYTES = (bytes, bytearray, memoryview)


class GradientWire(Record):
    """The wire form of an :class:`EncryptedGradient` (:meth:`EncryptedGradient.to_dict`)."""

    nid = Field("str, the client")
    shape = Field("list of ints, the gradient's shape")
    precision = Field("int, the fixed-point digits")
    mode = Field("the quantization mode")
    ciphertexts = Field("one layer per entry: :class:`ElementLayerWire` or :class:`PackedLayerWrapper`")


class ElementLayerWire(Record):
    """A per-element ciphertext layer on the wire: one scheme ciphertext per index."""

    shape = Field("list of ints")
    data = Field('``{"(i, j, ...)": ciphertext}``')


class PackedLayerWrapper(Record):
    """A packed layer on the wire (``{"packed": layer}``)."""

    packed = Field(":class:`PackedLayerWire`")


class PackedLayerWire(Record):
    """A packed ciphertext layer on the wire: the scheme's layer
    (:class:`~pyfe4ai.core.formats.ring_lwe.RingLWELayer`), the chunk fields
    holding bytes written as base64 strings and named in ``base64``."""

    shape = Field("list of ints")
    first_chunk = Field("int")
    chunks = Field("ciphertexts")
    label_hash = Field("int code (MCFE Ring-LWE; none for ``shake256-wide``)")
    base64 = Field("the chunk fields written as base64 (only if there are some)")


#: The fields a packed layer on the wire may lack.
_PACKED_OPTIONAL = ("label_hash", "base64")


class FeaturesWire(Record):
    """The wire form of an :class:`EncryptedFeatures`."""

    ciphertext = Field("the scheme's ciphertext")
    dim = Field("int")
    precision = Field("int")
    mode = Field("the quantization mode")


def _from_base64(value: Any, what: str) -> bytes:
    """The bytes of a strict base64 string (as ``to_dict`` writes them)."""
    if not isinstance(value, str):
        raise ValueError("{} is not a base64 string".format(what))
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except ValueError as exc:  # binascii.Error, non-ASCII text
        raise ValueError("{} is not base64".format(what)) from exc


@dataclass
class EncryptedGradient:
    """Encrypted gradient array for FL aggregation.

    Produced by :func:`encrypt_gradient`.
    """

    #: One entry per layer, an object array of per-element ciphertexts or,
    #: for the packed Ring-LWE MCFE schemes, a dict
    #: ``{"shape", "first_chunk", "chunks"}`` (see
    #: :meth:`pyfe4ai.schemes.mcfe.ring_lwe.MCFERingLWE.encrypt_lst_ndarray`).
    ciphertexts: list
    shape: tuple
    precision: int
    mode: str
    nid: str

    def to_dict(self) -> dict:
        """Serialise for network transfer (e.g. federated learning).

        A per-element layer is written as ``{"shape", "data"}`` with one
        entry per index, a packed layer as
        ``{"packed": {"shape", "first_chunk", "chunks"}}``. Chunk fields
        holding bytes (binary ciphertexts, e.g. the default format of
        ``mcfe/ring_lwe_threshold``) are written as base64 strings and named
        in the layer's ``"base64"`` list; :meth:`from_dict` decodes them.

        Raises:
            TypeError: If a binary field of one chunk is not bytes in another.
        """
        serialised_cts = []
        for ary in self.ciphertexts:
            if isinstance(ary, dict):  # packed layer
                serialised_cts.append(PackedLayerWrapper.new(packed=self._serialised_packed(ary)).to_dict())
                continue
            flat = {}
            for idx, _ in np.ndenumerate(ary):
                flat[str(idx)] = ary[idx]
            serialised_cts.append(ElementLayerWire.new(shape=list(ary.shape), data=flat).to_dict())
        return GradientWire.new(
            nid=self.nid,
            shape=list(self.shape),
            precision=self.precision,
            mode=self.mode,
            ciphertexts=serialised_cts,
        ).to_dict()

    @staticmethod
    def _serialised_packed(layer: dict) -> dict:
        layer = RingLWELayer(layer)
        binary = sorted({key for chunk in layer.chunks for key, v in chunk.items() if isinstance(v, _BYTES)})
        chunks = list(layer.chunks)
        if binary:
            for j, chunk in enumerate(chunks):
                if any(not isinstance(chunk.get(key), _BYTES) for key in binary):
                    raise TypeError("chunk {} of a packed layer lacks the binary fields {} of the others".format(j, binary))
                chunks[j] = dict(chunk, **{key: base64.b64encode(bytes(chunk[key])).decode("ascii") for key in binary})
        out = PackedLayerWire.new(
            shape=list(layer.shape),
            first_chunk=layer.first_chunk,
            chunks=chunks,
        )
        if "label_hash" in layer:  # the layer's label-hash code (MCFE Ring-LWE)
            out.label_hash = layer.label_hash
        if binary:
            out.base64 = binary
        return out.to_dict()

    #: Largest number of elements :meth:`from_dict` accepts per array.
    MAX_ELEMENTS = 10 ** 7

    #: The quantization modes of a serialised gradient (:class:`~pyfe4ai.ml.quantization.QuantizationConfig`).
    MODES = ("decimal", "binary")

    @classmethod
    def from_dict(cls, d: dict) -> EncryptedGradient:
        """Deserialise from :meth:`to_dict`, checking the field types and the
        layout (the scheme checks the ciphertexts when decrypting).

        Raises:
            ValueError: Unless *d* has exactly the keys ``nid`` (a string),
                ``shape`` (a list of non-negative ints), ``precision`` (a
                non-negative int), ``mode`` (``"decimal"`` or ``"binary"``)
                and ``ciphertexts`` (a non-empty list of layers); if an array's
                shape is invalid or has more than :attr:`MAX_ELEMENTS`
                elements; if a per-element layer is not ``{"shape", "data"}``
                with one ciphertext (a dict) under each index key
                ``to_dict`` writes, ``"(i, j, ...)"``, and nothing else; if a
                packed layer is not a valid ``{"shape", "first_chunk",
                "chunks"}`` layout; or if packed and per-element layers are
                mixed.
        """
        if not isinstance(d, dict) or set(d) != set(GradientWire.FIELDS):
            raise ValueError("invalid encrypted gradient: keys nid, shape, precision, mode and ciphertexts")
        d = GradientWire(d)
        precision = d.precision
        if (not isinstance(d.nid, str) or isinstance(precision, bool) or not isinstance(precision, int)
                or precision < 0 or d.mode not in cls.MODES):
            raise ValueError("invalid encrypted gradient: nid is a string, precision a non-negative int, "
                             "mode 'decimal' or 'binary'")
        shape, _ = cls._checked_shape(d.shape)
        layers = d.ciphertexts
        if not isinstance(layers, list) or not layers:
            raise ValueError("invalid encrypted gradient: ciphertexts must be a non-empty list of layers")
        packed = [isinstance(ct_dict, dict) and "packed" in PackedLayerWrapper(ct_dict) for ct_dict in layers]
        if any(packed) and not all(packed):
            raise ValueError("cannot mix packed and per-element ciphertext layers")
        if any(packed):
            cts = cls._packed_layers(layers)
        else:
            cts = [cls._element_layer(ct_dict) for ct_dict in layers]
        return cls(
            ciphertexts=cts,
            shape=shape,
            precision=precision,
            mode=d.mode,
            nid=d.nid,
        )

    @classmethod
    def _checked_shape(cls, shape) -> tuple[tuple, int]:
        """Return ``(shape, size)`` of a serialised shape, or raise ``ValueError``."""
        if (
            not isinstance(shape, list) or len(shape) > 32
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in shape)
        ):
            raise ValueError("invalid ciphertext shape {!r}".format(shape))
        size = 1
        for v in shape:
            size *= v
        if size > cls.MAX_ELEMENTS:
            raise ValueError("ciphertext array has too many elements ({})".format(size))
        return tuple(shape), size

    @classmethod
    def _element_layer(cls, ct_dict: dict) -> np.ndarray:
        """Rebuild a per-element layer ``{"shape", "data"}`` as an object array:
        ``data`` holds one ciphertext (a dict) under the key ``str(index)`` of
        every index of the shape, as :meth:`to_dict` writes it, and no other
        key (another spelling of an index, e.g. ``"( 0,)"`` or ``"(+0,)"``, is
        refused rather than read as the same position)."""
        if not isinstance(ct_dict, dict) or set(ct_dict) != set(ElementLayerWire.FIELDS):
            raise ValueError("invalid ciphertext layer: keys shape and data")
        ct_dict = ElementLayerWire(ct_dict)
        shape, size = cls._checked_shape(ct_dict.shape)
        data = ct_dict.data
        if not isinstance(data, dict) or len(data) != size:
            raise ValueError("ciphertext array must have an entry for every element")
        ary = np.empty(shape, dtype=object)
        for key, val in data.items():
            idx = cls._index(key)
            if len(idx) != len(shape) or any(not 0 <= i < n for i, n in zip(idx, shape)):
                raise ValueError("ciphertext index {} outside the shape {}".format(key, shape))
            if not isinstance(val, dict):
                raise ValueError("invalid ciphertext layer: every entry is a ciphertext (a dict)")
            ary[idx] = val
        # distinct keys in the one spelling are distinct positions: len(data) == size covers every one
        return ary

    @staticmethod
    def _index(key) -> tuple:
        """The index of a per-element layer's key, written as :meth:`to_dict`
        writes it (``str`` of a tuple of ints: ``"(0,)"``, ``"(1, 2)"``,
        ``"()"``); other spellings are refused."""
        idx = None
        if isinstance(key, str) and key[:1] == "(" and key[-1:] == ")":
            try:
                idx = tuple(int(x) for x in key[1:-1].split(",") if x.strip())
            except ValueError:
                idx = None
        if idx is None or str(idx) != key:
            raise ValueError("invalid ciphertext index key in a per-element layer")
        return idx

    @classmethod
    def _packed_layers(cls, layers: list) -> list:
        """Check serialised packed layers against the layout of ``encrypt_lst_ndarray``.

        Every layer is ``{"packed": {"shape", "first_chunk", "chunks"}}``,
        optionally with the layer's ``label_hash`` code (an int) and
        ``base64``, the chunk fields written as base64 (decoded to bytes;
        every chunk has them as strict base64 strings).
        The chunks of all layers are numbered consecutively from 0, so a
        layer's ``first_chunk`` is the number of chunks before it. Each chunk
        is a dict with a batch length ``k >= 1``; the ``k`` of a layer add up
        to its number of elements, and every chunk but the last of a layer is
        full (has the largest ``k``, i.e. ``ring_n``). The ciphertexts in the
        chunks are checked by the scheme when decrypting.
        """
        out, next_chunk = [], 0
        for wrapped in layers:
            layer = PackedLayerWrapper(wrapped).packed
            if (
                set(wrapped) != set(PackedLayerWrapper.FIELDS) or not isinstance(layer, dict)
                or set(layer) - set(_PACKED_OPTIONAL) != set(PackedLayerWire.FIELDS) - set(_PACKED_OPTIONAL)
                or ("label_hash" in PackedLayerWire(layer) and type(PackedLayerWire(layer).label_hash) is not int)
            ):
                raise ValueError("invalid packed ciphertext layer")
            layer = PackedLayerWire(layer)
            encoded = layer.get("base64", [])
            if (
                not isinstance(encoded, list) or ("base64" in layer and not encoded)
                or any(not isinstance(key, str) or key == "k" for key in encoded) or len(set(encoded)) != len(encoded)
            ):
                raise ValueError("invalid packed ciphertext layer: base64 names the chunk fields written as base64")
            shape, size = cls._checked_shape(layer.shape)
            first, chunks = layer.first_chunk, layer.chunks
            if not isinstance(first, int) or isinstance(first, bool) or first != next_chunk:
                raise ValueError(
                    "packed layer has first chunk {!r}, expected {}".format(first, next_chunk)
                )
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("packed chunks must be a non-empty list")
            chunks = list(chunks)
            for j, chunk in enumerate(chunks):
                k = RingLWECiphertext(chunk).get("k") if isinstance(chunk, dict) else None
                if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                    raise ValueError(
                        "chunk {} of a packed layer is not a ciphertext with a batch "
                        "length k >= 1".format(j)
                    )
                if encoded:
                    chunks[j] = dict(chunk, **{key: _from_base64(chunk.get(key), "chunk {} field {!r}".format(j, key))
                                               for key in encoded})
            covered = sum(RingLWECiphertext(chunk).k for chunk in chunks)
            if covered != size:
                raise ValueError(
                    "packed chunks cover {} elements, the shape {} has {}".format(covered, shape, size)
                )
            out.append(RingLWELayer.new(shape=list(shape), first_chunk=first, chunks=list(chunks)).to_dict())
            if "label_hash" in layer:
                RingLWELayer(out[-1]).label_hash = layer.label_hash
            next_chunk += len(chunks)
        width = max((RingLWECiphertext(chunk).k for layer in out for chunk in RingLWELayer(layer).chunks), default=0)
        if any(RingLWECiphertext(chunk).k != width for layer in out for chunk in RingLWELayer(layer).chunks[:-1]):
            raise ValueError(
                "every chunk but the last of a packed layer must have the full width {}".format(width)
            )
        return out


@dataclass
class EncryptedFeatures:
    """Encrypted feature vector for inference.

    Produced by :func:`encrypt_features`.
    """

    ciphertext: dict
    dim: int
    precision: int
    mode: str

    def to_dict(self) -> dict:
        return FeaturesWire.new(
            ciphertext=self.ciphertext,
            dim=self.dim,
            precision=self.precision,
            mode=self.mode,
        ).to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> EncryptedFeatures:
        d = FeaturesWire(d)
        return cls(
            ciphertext=d.ciphertext,
            dim=d.dim,
            precision=d.precision,
            mode=d.mode,
        )
