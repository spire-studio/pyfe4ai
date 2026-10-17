"""ML framework adapter layer for PyFE4AI.

Provides two high-level APIs that hide FE complexity from ML engineers:

**FL aggregation** (``eta=1``, ndarray helpers):
    Encrypt per-element gradients from *n* clients, aggregate with weights.

    >>> wrapper = FESchemeWrapper("mcfe", "lwe", keygen_cfg, crypto_cfg)
    >>> enc = encrypt_gradient(grad_array, wrapper, label="round-1")
    >>> result = aggregate_gradients(
    ...     {"c0": enc0, "c1": enc1}, wrapper, weights, label="round-1")

**Encrypted inference** (``eta=feature_dim``, vector mode):
    Encrypt a feature vector, compute ``⟨x, w⟩`` without revealing *x*.

    >>> wrapper = FESchemeWrapper("sife", "lwe", keygen_cfg, crypto_cfg)
    >>> enc = encrypt_features(feature_vec, wrapper)
    >>> score = compute_linear(enc, wrapper, model_weights)

PyTorch is optional — functions accept both :class:`numpy.ndarray` and
``torch.Tensor`` inputs.

**Single trust domain.** A :class:`FESchemeWrapper` runs the key generator
itself, so it holds the master secret key and plays every party (clients,
key authority, decryptor). It is meant for experiments and for code that
already trusts one process with everything; a deployment keeps the master
key away from the aggregator. The wrapper applies a default policy that a
key holder would otherwise have to enforce: :func:`aggregate_gradients`
needs a ciphertext and a non-zero weight for every enrolled client (pass
``allow_partial=True`` to relax this), each client encrypts at most once per
label, and each label is aggregated with one weight vector only. These
checks stop the obvious ways of isolating one client, not all of them: a
single aggregate with very unequal weights (e.g. ``(W, 1)`` with ``W`` above
twice the largest gradient entry) separates two clients' gradients, as with
any inner-product FE; choosing the weights is the key holder's
responsibility, and :meth:`FESchemeWrapper.get_decryption_keys` issues keys
for any weights.
"""

from __future__ import annotations

import importlib
import logging
import numbers
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from pyfe4ai.schemes.status import get_scheme_status
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError
from pyfe4ai.utils.quantization import QuantizationConfig
from pyfe4ai.utils.validation import int_vector

logger = logging.getLogger(__name__)

# Optional PyTorch support
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Registry: (scheme_type, variant) → (module_path, keygen_class, crypto_class)
# ---------------------------------------------------------------------------

_SCHEME_REGISTRY: dict[tuple[str, str], tuple[str, str, str]] = {
    # ---- SIFE ----------------------------------------------------------
    ("sife", "ddh"): ("pyfe4ai.schemes.sife.ddh", "SIFEKeyGenerator", "SIFE"),
    ("sife", "ddh_ristretto255"): (
        "pyfe4ai.schemes.sife.ddh_ristretto255",
        "SIFERistretto255KeyGenerator",
        "SIFERistretto255",
    ),
    ("sife", "lwe"): ("pyfe4ai.schemes.sife.lwe", "SIFELWEKeyGenerator", "SIFELWE"),
    ("sife", "paillier"): (
        "pyfe4ai.schemes.sife.paillier",
        "SIFEPaillierKeyGenerator",
        "SIFEPaillier",
    ),
    ("sife", "damgard"): (
        "pyfe4ai.schemes.sife.damgard_ddh",
        "SIFEDamgardKeyGenerator",
        "SIFEDamgard",
    ),
    ("sife", "ring_lwe"): (
        "pyfe4ai.schemes.sife.ring_lwe",
        "SIFERingLWEKeyGenerator",
        "SIFERingLWE",
    ),
    ("sife", "fullysec_ring_lwe"): (
        "pyfe4ai.schemes.sife.fullysec_ring_lwe",
        "SIFEFullySecRingLWEKeyGenerator",
        "SIFEFullySecRingLWE",
    ),
    ("sife", "fullysec_lwe"): (
        "pyfe4ai.schemes.sife.fullysec_lwe",
        "SIFEFullySecLWEKeyGenerator",
        "SIFEFullySecLWE",
    ),
    ("sife", "ddh_dynamic"): (
        "pyfe4ai.schemes.sife.ddh_dynamic",
        "SIFEDynamicKeyGenerator",
        "SIFEDynamic",
    ),
    ("sife", "fh_ipe_pairing"): (
        "pyfe4ai.schemes.sife.fh_ipe_pairing",
        "SIFEFHIPEKeyGenerator",
        "SIFEFHIPE",
    ),
    ("sife", "part_fh_ipe_pairing"): (
        "pyfe4ai.schemes.sife.part_fh_ipe_pairing",
        "SIFEPartFHIPEKeyGenerator",
        "SIFEPartFHIPE",
    ),
    # ---- MIFE ----------------------------------------------------------
    ("mife", "ddh"): ("pyfe4ai.schemes.mife.ddh", "MIFEKeyGenerator", "MIFE"),
    ("mife", "ddh_ristretto255"): (
        "pyfe4ai.schemes.mife.ddh_ristretto255",
        "MIFERistretto255KeyGenerator",
        "MIFERistretto255",
    ),
    ("mife", "lwe"): ("pyfe4ai.schemes.mife.lwe", "MIFELWEKeyGenerator", "MIFELWE"),
    ("mife", "paillier"): (
        "pyfe4ai.schemes.mife.paillier",
        "MIFEPaillierKeyGenerator",
        "MIFEPaillier",
    ),
    ("mife", "damgard"): (
        "pyfe4ai.schemes.mife.damgard_ddh",
        "MIFEDamgardKeyGenerator",
        "MIFEDamgard",
    ),
    ("mife", "ring_lwe"): (
        "pyfe4ai.schemes.mife.ring_lwe",
        "MIFERingLWEKeyGenerator",
        "MIFERingLWE",
    ),
    ("mife", "fullysec_lwe"): (
        "pyfe4ai.schemes.mife.fullysec_lwe",
        "MIFEFullySecLWEKeyGenerator",
        "MIFEFullySecLWE",
    ),
    ("mife", "fh_ipe_pairing"): (
        "pyfe4ai.schemes.mife.fh_ipe_pairing",
        "MIFEFHIPEKeyGenerator",
        "MIFEFHIPE",
    ),
    ("mife", "fh_multi_ipe_pairing"): (
        "pyfe4ai.schemes.mife.fh_multi_ipe_pairing",
        "MIFEFHMultiIPEKeyGenerator",
        "MIFEFHMultiIPE",
    ),
    ("mife", "ddh_threshold"): (
        "pyfe4ai.schemes.mife.ddh_threshold",
        "ThresholdMIFEKeyGenerator",
        "ThresholdMIFE",
    ),
    ("mife", "lwe_threshold"): (
        "pyfe4ai.schemes.mife.lwe_threshold",
        "ThresholdMIFELWEKeyGenerator",
        "ThresholdMIFELWE",
    ),
    # ---- MCFE ----------------------------------------------------------
    ("mcfe", "ddh"): ("pyfe4ai.schemes.mcfe.ddh", "MCFEKeyGenerator", "MCFE"),
    ("mcfe", "ddh_ristretto255"): (
        "pyfe4ai.schemes.mcfe.ddh_ristretto255",
        "MCFERistretto255KeyGenerator",
        "MCFERistretto255",
    ),
    ("mcfe", "lwe"): ("pyfe4ai.schemes.mcfe.lwe", "MCFELWEKeyGenerator", "MCFELWE"),
    ("mcfe", "paillier"): (
        "pyfe4ai.schemes.mcfe.paillier",
        "MCFEPaillierKeyGenerator",
        "MCFEPaillier",
    ),
    ("mcfe", "damgard"): (
        "pyfe4ai.schemes.mcfe.damgard_ddh",
        "MCFEDamgardKeyGenerator",
        "MCFEDamgard",
    ),
    ("mcfe", "ring_lwe"): (
        "pyfe4ai.schemes.mcfe.ring_lwe",
        "MCFERingLWEKeyGenerator",
        "MCFERingLWE",
    ),
    ("mcfe", "fullysec_lwe"): (
        "pyfe4ai.schemes.mcfe.fullysec_lwe",
        "MCFEFullySecLWEKeyGenerator",
        "MCFEFullySecLWE",
    ),
    ("mcfe", "fh_multi_ipe_pairing"): (
        "pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing",
        "MCFEFHMultiIPEKeyGenerator",
        "MCFEFHMultiIPE",
    ),
    ("mcfe", "fh_multi_ipe_pairing_threshold"): (
        "pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold",
        "ThresholdMCFEFHMultiIPEKeyGenerator",
        "ThresholdMCFEFHMultiIPE",
    ),
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): (
        "pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized",
        "DecentralizedMCFEFHMultiIPEKeyGenerator",
        "DecentralizedMCFEFHMultiIPE",
    ),
    ("mcfe", "ddh_threshold"): (
        "pyfe4ai.schemes.mcfe.ddh_threshold",
        "ThresholdMCFEKeyGenerator",
        "ThresholdMCFE",
    ),
    ("mcfe", "ddh_decentralized"): (
        "pyfe4ai.schemes.mcfe.ddh_decentralized",
        "DecentralizedMCFEKeyGenerator",
        "DecentralizedMCFE",
    ),
    ("mcfe", "lwe_threshold"): (
        "pyfe4ai.schemes.mcfe.lwe_threshold",
        "ThresholdMCFELWEKeyGenerator",
        "ThresholdMCFELWE",
    ),
    ("mcfe", "lwe_decentralized"): (
        "pyfe4ai.schemes.mcfe.lwe_decentralized",
        "DecentralizedMCFELWEKeyGenerator",
        "DecentralizedMCFELWE",
    ),
    ("mcfe", "ring_lwe_threshold"): (
        "pyfe4ai.schemes.mcfe.ring_lwe_threshold",
        "ThresholdMCFERingLWEKeyGenerator",
        "ThresholdMCFERingLWE",
    ),
    ("mcfe", "ring_lwe_decentralized"): (
        "pyfe4ai.schemes.mcfe.ring_lwe_decentralized",
        "DecentralizedMCFERingLWEKeyGenerator",
        "DecentralizedMCFERingLWE",
    ),
    ("mcfe", "paillier_decentralized"): (
        "pyfe4ai.schemes.mcfe.paillier_decentralized",
        "DecentralizedMCFEPaillierKeyGenerator",
        "DecentralizedMCFEPaillier",
    ),
    # ---- Quadratic -----------------------------------------------------
    ("quadratic", "quad"): (
        "pyfe4ai.schemes.quadratic.quad",
        "QuadraticQuadKeyGenerator",
        "QuadraticQuad",
    ),
    ("quadratic", "sgp"): (
        "pyfe4ai.schemes.quadratic.sgp",
        "QuadraticSGPKeyGenerator",
        "QuadraticSGP",
    ),
    ("quadratic", "miqfe"): (
        "pyfe4ai.schemes.quadratic.miqfe",
        "MultiInputQuadraticKeyGenerator",
        "MultiInputQuadratic",
    ),
}

# Schemes whose encrypt/decrypt require a label argument.
_LABEL_SCHEMES = frozenset({"mcfe"})

#: Instantiations selected with the ``group`` entry of a key-generator config:
#: ``{group: {(family, variant): variant in that group}}``. ``"modp"`` (or no
#: entry) keeps the 3072-bit ``Z_p^*`` subgroup of the ``ddh`` variants.
_GROUP_VARIANTS = {
    "ristretto255": {(f, "ddh"): "ddh_ristretto255" for f in ("sife", "mife", "mcfe")},
}


def _require_single_decryptor(wrapper: "FESchemeWrapper", api: str) -> None:
    """Reject variants whose decryption is a multi-party protocol.

    The high-level helpers derive one functional key from the key generator
    and decrypt in one shot. Decentralized variants have no central key
    derivation (each client contributes a key share) and threshold variants
    need partial decryptions from at least ``t`` servers, so neither fits.
    """
    if wrapper.variant.endswith("_decentralized"):
        raise NotImplementedError(
            "{}() does not support '{}/{}': decentralized schemes have no "
            "central decryption key. Derive the key with "
            "derive_function_decryption_key_share() on every client and "
            "combine_function_decryption_key_share() on the aggregator, then "
            "call decrypt_lst_ndarray_ct() on the scheme class directly.".format(
                api, wrapper.scheme_type, wrapper.variant
            )
        )
    if wrapper.variant.endswith("_threshold"):
        raise NotImplementedError(
            "{}() does not support '{}/{}': threshold schemes need partial "
            "decryptions from at least t servers (compute_lst_ndarray_ct() "
            "on each server, then decrypt_lst_ndarray_ct() on the combiner). "
            "Use the scheme classes directly.".format(
                api, wrapper.scheme_type, wrapper.variant
            )
        )


def _require_labelled_aggregation(wrapper: "FESchemeWrapper", api: str) -> None:
    """Reject multi-input FE for gradient aggregation.

    MIFE has no labels: the holder of one functional key may combine any
    ciphertext of each slot, so with element-wise (or per-round) encryption
    it learns the differences between a client's values -- the client's
    update up to one offset. Federated aggregation needs MCFE, whose
    ciphertexts are bound to labels.
    """
    if wrapper.scheme_type == "mife":
        raise NotImplementedError(
            "{}() does not support '{}/{}': multi-input FE has no labels, so a "
            "functional-key holder can combine a client's ciphertexts of different "
            "elements or rounds and learn the differences of its update. Use an "
            "MCFE scheme (e.g. 'mcfe/ddh') for gradient aggregation.".format(
                api, wrapper.scheme_type, wrapper.variant
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(data: Any) -> np.ndarray:
    """Convert input to numpy array, handling torch.Tensor transparently."""
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def tensor_to_ndarray(t: Any) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array (detach + cpu + numpy)."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")
    return t.detach().cpu().numpy()


def ndarray_to_tensor(arr: np.ndarray) -> Any:
    """Convert a numpy array to a PyTorch tensor."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# FESchemeWrapper
# ---------------------------------------------------------------------------


class FESchemeWrapper:
    """Unified wrapper around any PyFE4AI scheme.

    Manages the KeyGenerator lifecycle (setup + key distribution) and
    provides a consistent encrypt / decrypt surface regardless of the
    underlying scheme family. The wrapper runs the key generator itself and
    so forms a single trust domain (see the module docstring).

    Every registered ``(scheme_type, variant)`` pair is accepted: the keys of
    :data:`~pyfe4ai.schemes.status.SCHEME_STATUS` (for example
    ``("sife", "ddh")``, ``("mcfe", "lwe_threshold")``), with their review
    status in :attr:`status`. The FL helpers are narrower:
    :func:`encrypt_gradient` and :func:`aggregate_gradients` refuse MIFE (no
    labels, so a key holder could combine a client's ciphertexts across
    elements or rounds), and :func:`aggregate_gradients` and
    :func:`compute_linear` refuse the ``*_decentralized`` and ``*_threshold``
    variants, whose decryption is a multi-party protocol; use the scheme
    classes directly for those.

    Args:
        scheme_type: ``"sife"``, ``"mife"``, ``"mcfe"`` or ``"quadratic"``.
        variant: Variant name within the family, e.g. ``"ddh"``, ``"lwe"``,
            ``"damgard"``, ``"paillier"``, ``"ring_lwe"`` or
            ``"lwe_threshold"`` (see :data:`~pyfe4ai.schemes.status.SCHEME_STATUS`).
        keygen_config: Configuration dict for the KeyGenerator (``sec_param``,
            ``eta``, ``n``, ``bound_x``, ``security_profile``, etc.).
            ``group="ristretto255"`` selects the elliptic-curve instantiation
            of the ``ddh`` variants (``ddh_ristretto255``).
        precision: Quantization precision (0: no quantization).
        qmode: Quantization mode, ``"decimal"`` or ``"binary"``.
        crypto_options: Extra entries of the crypto objects' config, e.g.
            ``{"ct_format": "binary"}`` for the Ring-LWE MIFE / MCFE.

    Raises:
        ValueError: If ``(scheme_type, variant)`` is not registered.
    """

    def __init__(
        self,
        scheme_type: str,
        variant: str,
        keygen_config: dict,
        *,
        precision: int = 0,
        qmode: str = "decimal",
        crypto_options: dict | None = None,
    ) -> None:
        key = (scheme_type.lower(), variant.lower())
        group = keygen_config.get("group")
        if group not in (None, "modp") and key in _SCHEME_REGISTRY:
            by_group = _GROUP_VARIANTS.get(group, {})
            if key in by_group:
                key = (key[0], by_group[key])
            elif key[1] not in by_group.values():
                raise ValueError("{}/{} has no instantiation in group {!r}".format(*key, group))
        if key not in _SCHEME_REGISTRY:
            supported = sorted(f"{s}/{v}" for s, v in _SCHEME_REGISTRY)
            raise ValueError(
                f"Unknown scheme '{scheme_type}/{variant}'. "
                f"Supported: {supported}"
            )

        self.scheme_type, self.variant = key
        self.keygen_config = dict(keygen_config)
        self.crypto_options = dict(crypto_options or {})
        self.qconfig = QuantizationConfig(precision=precision, mode=qmode)
        self.has_label = self.scheme_type in _LABEL_SCHEMES
        #: Review status of the scheme (see :mod:`pyfe4ai.schemes.status`).
        self.status = get_scheme_status(*key)

        mod_path, kg_cls_name, crypto_cls_name = _SCHEME_REGISTRY[key]
        mod = importlib.import_module(mod_path)
        self._kg_cls = getattr(mod, kg_cls_name)
        self._crypto_cls = getattr(mod, crypto_cls_name)

        # KeyGenerator lifecycle
        self.kg = self._kg_cls(self.keygen_config)
        self.kg.setup()
        self.pp = self.kg.get_public_parameters()

        # Cache for crypto instances per nid
        self._encryptors: dict[str, Any] = {}
        # Labels used for encryption, per client, and weights aggregated, per label
        self._encrypted_labels: set[tuple[str, str]] = set()
        self._aggregated: dict[str, dict] = {}

    # -- crypto instance management ----------------------------------------

    def _get_encryptor(self, nid: str) -> Any:
        """Get or create a crypto instance for the given node id."""
        if nid not in self._encryptors:
            sk = self.kg.get_private_keys(nid)
            keys: dict[str, Any] = {"pp": self.pp}
            if sk is not None:
                keys["sk"] = sk
            config = dict(self.crypto_options, id=nid, precision=self.qconfig.precision, keys=keys)
            mode = getattr(self.kg, "matrix_mode", None)  # LWE schemes with a seeded matrix
            if mode is not None:
                config["matrix_mode"] = mode
            self._encryptors[nid] = self._crypto_cls(config)
        return self._encryptors[nid]

    def _get_decryptor(self, sid: str) -> Any:
        """Create a crypto instance for decryption (no sk needed)."""
        return self._crypto_cls(
            dict(self.crypto_options, id=sid, precision=self.qconfig.precision, keys={"pp": self.pp})
        )

    def get_decryption_keys(self, sid: str, **credentials) -> dict:
        """Derive functional decryption keys.

            Args:
                sid: Session / decryption-key identifier.
        """
        return self.kg.get_decryption_keys(sid, credentials=credentials)


# ---------------------------------------------------------------------------
# Encrypted data containers
# ---------------------------------------------------------------------------


@dataclass
class EncryptedGradient:
    """Encrypted gradient array for FL aggregation.

    Produced by :func:`encrypt_gradient`.
    """

    #: One entry per layer: an object array of per-element ciphertexts, or,
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
        ``{"packed": {"shape", "first_chunk", "chunks"}}``.

        Raises:
            TypeError: If a packed chunk holds ``bytes`` (a binary ciphertext
                format). These have no JSON form, and the dict returned here
                is meant for ``json.dumps``.
        """
        serialised_cts = []
        for ary in self.ciphertexts:
            if isinstance(ary, dict):  # packed layer
                serialised_cts.append({"packed": self._serialised_packed(ary)})
                continue
            flat = {}
            for idx, _ in np.ndenumerate(ary):
                flat[str(idx)] = ary[idx]
            serialised_cts.append({"shape": list(ary.shape), "data": flat})
        return {
            "nid": self.nid,
            "shape": list(self.shape),
            "precision": self.precision,
            "mode": self.mode,
            "ciphertexts": serialised_cts,
        }

    @staticmethod
    def _serialised_packed(layer: dict) -> dict:
        for j, chunk in enumerate(layer["chunks"]):
            if any(isinstance(v, (bytes, bytearray, memoryview)) for v in chunk.values()):
                raise TypeError(
                    "chunk {} of a packed layer is a binary ciphertext; to_dict() only "
                    "serialises JSON-compatible chunks".format(j)
                )
        out = {
            "shape": list(layer["shape"]),
            "first_chunk": layer["first_chunk"],
            "chunks": list(layer["chunks"]),
        }
        if "label_hash" in layer:  # the layer's label-hash code (MCFE Ring-LWE)
            out["label_hash"] = layer["label_hash"]
        return out

    #: Largest number of elements :meth:`from_dict` accepts per array.
    MAX_ELEMENTS = 10 ** 7

    @classmethod
    def from_dict(cls, d: dict) -> EncryptedGradient:
        """Deserialise from dict.

        Raises:
            ValueError: If an array's shape is invalid or has more than
                :attr:`MAX_ELEMENTS` elements, or its entries do not cover
                exactly the positions of that shape; if a packed layer does
                not have the layout of :meth:`_packed_layers`; or if packed
                and per-element layers are mixed.
        """
        layers = d["ciphertexts"]
        packed = [isinstance(ct_dict, dict) and "packed" in ct_dict for ct_dict in layers]
        if any(packed) and not all(packed):
            raise ValueError("cannot mix packed and per-element ciphertext layers")
        if any(packed):
            cts = cls._packed_layers(layers)
        else:
            cts = [cls._element_layer(ct_dict) for ct_dict in layers]
        return cls(
            ciphertexts=cts,
            shape=tuple(d["shape"]),
            precision=d["precision"],
            mode=d["mode"],
            nid=d["nid"],
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
        """Rebuild a per-element layer ``{"shape", "data"}`` as an object array."""
        shape, size = cls._checked_shape(ct_dict["shape"])
        ary = np.empty(shape, dtype=object)
        seen = set()
        for key, val in ct_dict["data"].items():
            idx = tuple(
                int(x) for x in key.strip("()").split(",") if x.strip()
            )
            if len(idx) != len(shape) or any(not 0 <= i < n for i, n in zip(idx, shape)):
                raise ValueError("ciphertext index {} outside the shape {}".format(key, shape))
            ary[idx] = val
            seen.add(idx)
        if len(seen) != size:
            raise ValueError("ciphertext array must have an entry for every element")
        return ary

    @classmethod
    def _packed_layers(cls, layers: list) -> list:
        """Check serialised packed layers against the layout of ``encrypt_lst_ndarray``.

        Every layer is ``{"packed": {"shape", "first_chunk", "chunks"}}``,
        optionally with the layer's ``label_hash`` code (an int).
        The chunks of all layers are numbered consecutively from 0, so a
        layer's ``first_chunk`` is the number of chunks before it. Each chunk
        is a dict with a batch length ``k >= 1``; the ``k`` of a layer add up
        to its number of elements, and every chunk but the last of a layer is
        full (has the largest ``k``, i.e. ``ring_n``). The ciphertexts in the
        chunks are checked by the scheme when decrypting.
        """
        out, next_chunk = [], 0
        for wrapped in layers:
            layer = wrapped["packed"]
            if (
                set(wrapped) != {"packed"} or not isinstance(layer, dict)
                or set(layer) - {"label_hash"} != {"shape", "first_chunk", "chunks"}
                or ("label_hash" in layer and type(layer["label_hash"]) is not int)
            ):
                raise ValueError("invalid packed ciphertext layer")
            shape, size = cls._checked_shape(layer["shape"])
            first, chunks = layer["first_chunk"], layer["chunks"]
            if not isinstance(first, int) or isinstance(first, bool) or first != next_chunk:
                raise ValueError(
                    "packed layer has first chunk {!r}, expected {}".format(first, next_chunk)
                )
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("packed chunks must be a non-empty list")
            for j, chunk in enumerate(chunks):
                k = chunk.get("k") if isinstance(chunk, dict) else None
                if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                    raise ValueError(
                        "chunk {} of a packed layer is not a ciphertext with a batch "
                        "length k >= 1".format(j)
                    )
            covered = sum(chunk["k"] for chunk in chunks)
            if covered != size:
                raise ValueError(
                    "packed chunks cover {} elements, the shape {} has {}".format(covered, shape, size)
                )
            out.append({"shape": list(shape), "first_chunk": first, "chunks": list(chunks)})
            if "label_hash" in layer:
                out[-1]["label_hash"] = layer["label_hash"]
            next_chunk += len(chunks)
        width = max((chunk["k"] for layer in out for chunk in layer["chunks"]), default=0)
        if any(chunk["k"] != width for layer in out for chunk in layer["chunks"][:-1]):
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
        return {
            "ciphertext": self.ciphertext,
            "dim": self.dim,
            "precision": self.precision,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EncryptedFeatures:
        return cls(
            ciphertext=d["ciphertext"],
            dim=d["dim"],
            precision=d["precision"],
            mode=d["mode"],
        )


# ---------------------------------------------------------------------------
# FL Aggregation API (eta=1, ndarray helpers)
# ---------------------------------------------------------------------------


def precompute_gradient(
    shape,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
    label: str,
) -> Any:
    """Offline phase of :func:`encrypt_gradient` for a gradient of *shape*.

    For ``mcfe/ring_lwe`` (see
    :meth:`~pyfe4ai.schemes.mcfe.ring_lwe.MCFERingLWE.precompute`): label
    pads, Gaussian samples and NTT products are computed before the gradient
    exists; ``encrypt_gradient(..., precomputed=...)`` then only adds the
    plaintext term. The result is secret and one-time.

    Raises:
        ValueError: If the scheme has no offline phase, or *nid* already
            encrypted under *label*.
    """
    _require_labelled_aggregation(wrapper, "precompute_gradient")
    if not wrapper.has_label or not isinstance(label, str):
        raise ValueError("precomputation needs an MCFE scheme and a string label")
    encryptor = wrapper._get_encryptor(nid)
    precompute = getattr(encryptor, "precompute", None)
    if precompute is None:
        raise ValueError("{}/{} has no offline encryption phase".format(wrapper.scheme_type, wrapper.variant))
    if (nid, label) in wrapper._encrypted_labels:
        raise ValueError("client {} already encrypted under label {!r}".format(nid, label))
    return precompute(label, [tuple(int(d) for d in np.shape(np.empty(shape)))])


def encrypt_gradient(
    gradient: np.ndarray | Any,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
    label: str | None = None,
    precomputed: Any = None,
) -> EncryptedGradient:
    """Encrypt a gradient array for federated aggregation.

        Args:
            gradient: Gradient array (numpy array or ``torch.Tensor``).
            wrapper: FE scheme wrapper instance.
            nid: Node identifier.
            label: Round label; MCFE allows one ciphertext per client and
                label, so a label this wrapper already used for *nid* is
                refused.
            precomputed: The result of :func:`precompute_gradient` for this
                client, label and shape (online phase).
    """
    _require_labelled_aggregation(wrapper, "encrypt_gradient")
    arr = _to_numpy(gradient).astype(np.float64)
    encryptor = wrapper._get_encryptor(nid)

    kwargs: dict[str, Any] = {}
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for encryption")
        if (nid, label) in wrapper._encrypted_labels:
            raise ValueError(
                "client {} already encrypted under label {!r}; use a fresh label "
                "for every round".format(nid, label)
            )
        kwargs["label"] = label
    if precomputed is not None:
        kwargs["precomputed"] = precomputed

    cts = encryptor.encrypt_lst_ndarray([arr], **kwargs)
    if wrapper.has_label:
        wrapper._encrypted_labels.add((nid, label))

    return EncryptedGradient(
        ciphertexts=cts,
        shape=arr.shape,
        precision=wrapper.qconfig.precision,
        mode=wrapper.qconfig.mode,
        nid=nid,
    )


def _integer_weights(weights: dict, qconfig: QuantizationConfig) -> tuple[dict, int]:
    """Return ``({nid: [int]}, scale)``: the weights as integers, multiplied
    by the quantisation scale if any of them is fractional."""
    lists = {
        nid: [w] if np.ndim(w) == 0 else list(np.ravel(w)) for nid, w in weights.items()
    }
    values = [v for lst in lists.values() for v in lst]
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("weights must be finite")
    try:
        return {nid: int_vector(lst, "weight of {}".format(nid)) for nid, lst in lists.items()}, 1
    except FEValidationError:
        pass  # fractional weights: quantise them below
    scale = qconfig.scale_factor
    if scale == 1:
        raise ValueError("fractional weights need a wrapper precision > 0")
    scaled = {
        nid: [Fraction(v) * scale if isinstance(v, numbers.Rational) else float(v) * scale for v in lst]
        for nid, lst in lists.items()
    }
    # exact for rationals; floats (0.1 * 10 = 1.0000000000000002) get an absolute tolerance
    if any(abs(v - round(v)) > 1e-6 for lst in scaled.values() for v in lst):
        raise ValueError(
            "the weights are not multiples of 1/{} at the wrapper's precision; "
            "raise the precision or round the weights".format(scale)
        )
    return {nid: [int(round(v)) for v in lst] for nid, lst in scaled.items()}, scale


@dataclass
class PreparedAggregation:
    """A functional key derived before the round (:func:`prepare_aggregation`)."""

    label: str | None
    sid: str
    fusion_weight: dict
    weight_scale: int
    n_chunks: int | None
    dk: dict
    allow_partial: bool


def _aggregation_key(wrapper, weights, label, sid, nids, allow_partial, n_chunks_fn):
    """Policy checks of :func:`aggregate_gradients` and the functional key:
    ``(fusion_weight, weight_scale, dk)``; *nids* are the clients whose
    ciphertexts will be aggregated, ``n_chunks_fn()`` the number of packed
    chunks a key must cover (``None``: not a packed scheme)."""
    # Normalise weights to {nid: [w]} format for eta=1 schemes
    fusion_weight, weight_scale = _integer_weights(weights, wrapper.qconfig)
    if wrapper.scheme_type == "sife":
        if len(nids) != 1 or set(fusion_weight) != set(nids):
            raise ValueError(
                "single-input FE aggregates one client: pass one ciphertext and its weight"
            )
    elif not allow_partial:
        enrolled = set(getattr(wrapper.kg, "lst_nid", None) or nids)
        if set(nids) != enrolled or set(fusion_weight) != enrolled:
            raise ValueError(
                "aggregation needs a ciphertext and a weight for every enrolled client "
                "(allow_partial=True to override)"
            )
        if any(v == 0 for lst in fusion_weight.values() for v in lst):
            raise ValueError("aggregation needs non-zero weights (allow_partial=True to override)")
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for decryption")
        previous = wrapper._aggregated.setdefault(label, fusion_weight)
        if previous != fusion_weight:
            raise ValueError(
                "label {!r} was already aggregated with other weights".format(label)
            )

    # Build credentials for dk generation
    # SIFE expects fusion_weight as a flat list; MCFE/MIFE expect a dict.
    if wrapper.scheme_type in ("mcfe", "mife"):
        credentials: dict[str, Any] = {"fusion_weight": fusion_weight}
    else:
        # Single-input: take the first (only) client's weight list
        credentials = {"fusion_weight": next(iter(fusion_weight.values()))}
    if wrapper.has_label:
        credentials["label"] = label
    n_chunks = n_chunks_fn()
    if n_chunks is not None:
        # identity-based schemes issue one key term per packed chunk
        credentials["n_chunks"] = n_chunks
    dk = wrapper.kg.get_decryption_keys(sid, credentials=credentials)
    return fusion_weight, weight_scale, dk


def prepare_aggregation(
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list],
    *,
    shape,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
) -> PreparedAggregation:
    """Derive the functional key of an aggregation before the ciphertexts arrive.

    The key depends on the weights, the label and (for the packed Ring-LWE
    MCFE) the number of chunks of a gradient of *shape*, not on the
    ciphertexts; :func:`aggregate_gradients` with ``prepared=`` then only
    decrypts. The same policy checks apply, with the clients of *weights* as
    those whose ciphertexts will be aggregated.
    """
    _require_labelled_aggregation(wrapper, "prepare_aggregation")
    _require_single_decryptor(wrapper, "prepare_aggregation")
    decryptor = wrapper._get_decryptor(sid)
    n_chunks = None
    if getattr(decryptor, "key_chunks", None) is not None:
        ring_n = int(wrapper.pp["ring_n"])
        size = int(np.prod(np.shape(np.empty(shape)), dtype=np.int64))
        n_chunks = max(1, -(-size // ring_n))
    fusion_weight, weight_scale, dk = _aggregation_key(
        wrapper, weights, label, sid, list(weights), allow_partial, lambda: n_chunks
    )
    return PreparedAggregation(label, sid, fusion_weight, weight_scale, n_chunks, dk, allow_partial)


def aggregate_gradients(
    encrypted: dict[str, EncryptedGradient],
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list] | None = None,
    *,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
    prepared: PreparedAggregation | None = None,
) -> np.ndarray:
    """Decrypt and aggregate encrypted gradients from multiple clients.

        Args:
            encrypted: Encrypted data container.
            wrapper: FE scheme wrapper instance.
            weights: Weight per client (a number or a one-element list);
                fractional weights are quantised with the wrapper's
                precision and the result is scaled back.
            sid: Session / decryption-key identifier.
            label: Round label (MCFE); each label is aggregated with one
                weight vector only.
            allow_partial: Allow zero weights and a subset of the enrolled
                clients. Off by default: the wrapper holds the master key,
                and such weights would decrypt a single client's gradient
                (so would very unequal weights; see the module docstring).
            prepared: A key from :func:`prepare_aggregation` (online phase);
                *weights*, if given, and *label* must match it.

        Raises:
            ValueError: If the policy above is violated, or SIFE is given
                more than one client.
    """
    _require_labelled_aggregation(wrapper, "aggregate_gradients")
    _require_single_decryptor(wrapper, "aggregate_gradients")
    decryptor = None

    def n_chunks_fn():
        # Collect ciphertext arrays per nid
        # decrypt_lst_ndarray_ct expects: list of arrays (one per layer)
        # where each array element is a ciphertext dict
        # For SIFE: dict_ndarray_ct is list of arrays
        # For MCFE/MIFE: dict_ndarray_ct is dict[nid -> list of arrays]
        nonlocal decryptor
        sample_enc = next(iter(encrypted.values()))
        decryptor = wrapper._get_decryptor(prepared.sid if prepared is not None else sid)
        key_chunks = getattr(decryptor, "key_chunks", None)
        return key_chunks(sample_enc.ciphertexts) if key_chunks is not None else None

    if prepared is None:
        if weights is None:
            raise ValueError("need the weights (or a prepared aggregation)")
        fusion_weight, weight_scale, dk = _aggregation_key(
            wrapper, weights, label, sid, list(encrypted), allow_partial, n_chunks_fn
        )
    else:
        n_chunks = n_chunks_fn()
        if prepared.label != label:
            raise ValueError("the prepared key is for label {!r}".format(prepared.label))
        if weights is not None and _integer_weights(weights, wrapper.qconfig)[0] != prepared.fusion_weight:
            raise ValueError("the prepared key is for other weights")
        if set(encrypted) != set(prepared.fusion_weight):
            raise ValueError("the prepared key is for the clients {}".format(sorted(prepared.fusion_weight)))
        if n_chunks is not None and (prepared.n_chunks or 0) < n_chunks:
            raise ValueError("the prepared key covers fewer chunks than the ciphertexts")
        fusion_weight, weight_scale, dk = prepared.fusion_weight, prepared.weight_scale, prepared.dk

    if wrapper.scheme_type in ("mcfe", "mife"):
        # Multi-client: {nid: [array_per_layer]}
        dict_ct: dict[str, list] = {}
        for nid, enc in encrypted.items():
            dict_ct[nid] = enc.ciphertexts

        kwargs: dict[str, Any] = {"dk": dk, "fusion_weight": fusion_weight}
        if wrapper.has_label:
            kwargs["label"] = label

        result_arrays = decryptor.decrypt_lst_ndarray_ct(
            dict_ct, **kwargs
        )
    else:
        # Single-input (SIFE): one client, simple list-of-arrays
        enc = next(iter(encrypted.values()))
        fw_list = next(iter(fusion_weight.values()))
        result_arrays = decryptor.decrypt_lst_ndarray_ct(
            enc.ciphertexts, dk=dk, fusion_weight=fw_list
        )

    if weight_scale != 1:
        result_arrays = [np.asarray(ary, dtype=np.float64) / weight_scale for ary in result_arrays]
    if result_arrays and len(result_arrays) == 1:
        return result_arrays[0]
    return np.array(result_arrays)


# ---------------------------------------------------------------------------
# Encrypted Inference API (eta=feature_dim, vector mode)
# ---------------------------------------------------------------------------


def encrypt_features(
    features: np.ndarray | Any,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
) -> EncryptedFeatures:
    """Encrypt a feature vector for privacy-preserving inference.

        Args:
            features: Feature vector (numpy array or ``torch.Tensor``),
                flattened and quantized with the wrapper's precision.
            wrapper: FE scheme wrapper instance.
            nid: Node identifier.
    """
    arr = _to_numpy(features).ravel().astype(np.float64)
    qcfg = wrapper.qconfig
    quantized = qcfg.quantize_array(arr).tolist()

    encryptor = wrapper._get_encryptor(nid)
    ct = encryptor.encrypt(quantized)

    return EncryptedFeatures(
        ciphertext=ct,
        dim=len(quantized),
        precision=qcfg.precision,
        mode=qcfg.mode,
    )


def compute_linear(
    encrypted: EncryptedFeatures,
    wrapper: FESchemeWrapper,
    weights: np.ndarray | Any,
    *,
    sid: str = "sid_infer",
) -> float:
    """Compute ⟨features, weights⟩ on encrypted features.

        Args:
            encrypted: Encrypted data container.
            wrapper: FE scheme wrapper instance.
            weights: Weight vector or dict.
            sid: Session / decryption-key identifier.
    """
    _require_single_decryptor(wrapper, "compute_linear")

    w_arr = _to_numpy(weights).ravel().astype(np.float64)
    qcfg = wrapper.qconfig
    q_weights = qcfg.quantize_array(w_arr).tolist()

    dk = wrapper.get_decryption_keys(sid, fusion_weight=q_weights)

    decryptor = wrapper._get_decryptor(sid)
    raw_result = decryptor.decrypt(encrypted.ciphertext, dk, q_weights)
    if raw_result is None:
        raise FESchemeError(
            "the inner product does not decrypt (result outside the decryption bound, "
            "or key and ciphertext do not match)"
        )

    # Both x and w were quantized → result scaled by scale_factor^2
    return qcfg.dequantize(raw_result, order=2)
