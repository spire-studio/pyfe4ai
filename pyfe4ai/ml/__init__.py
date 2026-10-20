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

**Threshold MCFE** (``mcfe/ring_lwe_threshold``): the wrapper also runs the
``s`` decryption servers. :func:`prepare_aggregation` (or
:func:`aggregate_gradients` without ``prepared``) gives each enrolled server
its label key share of the round and, once per weights, its weight key share
(the scheme's split key shares, its default); :func:`aggregate_gradients`
then has the servers decrypt partially and combines their answers. The two
steps are also separate functions -- :func:`partial_decrypt_gradients` per
server, :func:`combine_partial_decryptions` at the combiner -- whose
:class:`ThresholdPartial` serialises like :class:`EncryptedGradient`
(binary ciphertexts and partial decryptions, the scheme's default format, as
base64).

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

import base64
import importlib
import logging
import numbers
import threading
from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np

from pyfe4ai.core.status import get_scheme_status
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.ml.quantization import QuantizationConfig
from pyfe4ai._validation import int_vector

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

#: Threshold variants whose decryption servers the wrapper runs (FL helpers).
_SERVED_THRESHOLD = frozenset({("mcfe", "ring_lwe_threshold")})

_BYTES = (bytes, bytearray, memoryview)

#: Instantiations selected with the ``group`` entry of a key-generator config:
#: ``{group: {(family, variant): variant in that group}}``. ``"modp"`` (or no
#: entry) keeps the 3072-bit ``Z_p^*`` subgroup of the ``ddh`` variants.
_GROUP_VARIANTS = {
    "ristretto255": {(f, "ddh"): "ddh_ristretto255" for f in ("sife", "mife", "mcfe")},
}


def _require_single_decryptor(wrapper: "FESchemeWrapper", api: str, served_threshold: bool = False) -> None:
    """Reject variants whose decryption is a multi-party protocol.

    The high-level helpers derive one functional key from the key generator
    and decrypt in one shot. Decentralized variants have no central key
    derivation (each client contributes a key share) and threshold variants
    need partial decryptions from at least ``t`` servers, so neither fits --
    but for the helpers that run the servers of :data:`_SERVED_THRESHOLD`
    (*served_threshold*).
    """
    if served_threshold and _serves_threshold(wrapper):
        return
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


def _serves_threshold(wrapper: "FESchemeWrapper") -> bool:
    """Whether the wrapper runs the decryption servers of its scheme."""
    return (wrapper.scheme_type, wrapper.variant) in _SERVED_THRESHOLD


def _require_served_threshold(wrapper: "FESchemeWrapper", api: str) -> None:
    if not _serves_threshold(wrapper):
        raise NotImplementedError(
            "{}() is for the threshold MCFE whose servers the wrapper runs ({}), not '{}/{}'".format(
                api, ", ".join(sorted("/".join(key) for key in _SERVED_THRESHOLD)), wrapper.scheme_type,
                wrapper.variant))


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


def _from_base64(value: Any, what: str) -> bytes:
    """The bytes of a strict base64 string (as ``to_dict`` writes them)."""
    if not isinstance(value, str):
        raise ValueError("{} is not a base64 string".format(what))
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except ValueError as exc:  # binascii.Error, non-ASCII text
        raise ValueError("{} is not base64".format(what)) from exc


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
    :data:`~pyfe4ai.core.status.SCHEME_STATUS` (for example
    ``("sife", "ddh")``, ``("mcfe", "lwe_threshold")``), with their review
    status in :attr:`status`. The FL helpers are narrower:
    :func:`encrypt_gradient` and :func:`aggregate_gradients` refuse MIFE (no
    labels, so a key holder could combine a client's ciphertexts across
    elements or rounds), and :func:`aggregate_gradients` and
    :func:`compute_linear` refuse the ``*_decentralized`` and ``*_threshold``
    variants, whose decryption is a multi-party protocol; use the scheme
    classes directly for those. The exception is ``mcfe/ring_lwe_threshold``,
    whose ``s`` decryption servers the wrapper runs (see the module
    docstring): :func:`aggregate_gradients` and :func:`prepare_aggregation`
    take it, with the enrolled ``servers``.

    A wrapper can be pickled and copied (``multiprocessing``, Ray): the copy
    carries the ``(nid, label)`` pairs used so far and refuses them as the
    original does. Copies do not learn of labels used later by the original
    or by other copies, so encrypt each client's gradient for a label through
    one wrapper. A pickled wrapper carries everything it holds, the master
    secret key and every client's key included: do not ship it to workers
    that should only encrypt. Give such a worker its client's own crypto
    object instead -- the scheme class built from ``wrapper.pp`` and
    ``wrapper.kg.get_private_keys(nid)`` (e.g. ``MCFERingLWE({"id": nid,
    "precision": p, "keys": {"pp": pp, "sk": sk}})`` for ``mcfe/ring_lwe``),
    whose ``encrypt_lst_ndarray`` the worker calls.

    Args:
        scheme_type: ``"sife"``, ``"mife"``, ``"mcfe"`` or ``"quadratic"``.
        variant: Variant name within the family, e.g. ``"ddh"``, ``"lwe"``,
            ``"damgard"``, ``"paillier"``, ``"ring_lwe"`` or
            ``"lwe_threshold"`` (see :data:`~pyfe4ai.core.status.SCHEME_STATUS`).
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
        #: Review status of the scheme (see :mod:`pyfe4ai.core.status`).
        self.status = get_scheme_status(*key)

        mod_path, kg_cls_name, crypto_cls_name = _SCHEME_REGISTRY[key]
        mod = importlib.import_module(mod_path)
        self._kg_cls = getattr(mod, kg_cls_name)
        self._crypto_cls = getattr(mod, crypto_cls_name)

        # KeyGenerator lifecycle
        self.kg = self._kg_cls(self.keygen_config)
        self.kg.setup()
        self.pp = self.kg.get_public_parameters()

        # Cache for crypto instances per nid, and the threshold servers per sid
        self._encryptors: dict[str, Any] = {}
        self._servers: dict[str, Any] = {}
        # Labels used for encryption, per client (reserved before encrypting,
        # under _label_lock), and weights aggregated, per label
        self._encrypted_labels: set[tuple[str, str]] = set()
        self._label_lock = threading.Lock()
        self._aggregated: dict[str, dict] = {}

    def __getstate__(self) -> dict:
        """Pickling and copying: the state without the lock and the cached
        crypto objects (rebuilt from the keys on use; they may hold native
        objects; threshold servers get their weight key shares again), with a
        snapshot of the used ``(nid, label)`` pairs (the copy refuses them
        too)."""
        state = self.__dict__.copy()
        del state["_label_lock"]
        state["_encryptors"] = {}
        state["_servers"] = {}
        with self._label_lock:
            state["_encrypted_labels"] = set(self._encrypted_labels)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_servers", {})
        self._label_lock = threading.Lock()

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

    def _get_server(self, sid: str) -> Any:
        """Decryption server *sid* of the threshold scheme, made on first use
        (it keeps the weight key shares it is given).

        Raises:
            ValueError: If *sid* is not a server of the public parameters.
        """
        if sid not in (self.pp.get("lst_sid") or []):
            raise ValueError("{!r} is not one of the servers {}".format(sid, self.pp.get("lst_sid")))
        with self._label_lock:  # one object per server: the one given its weight key shares
            server = self._servers.get(sid)
            if server is None:
                server = self._servers[sid] = self._crypto_cls(
                    dict(self.crypto_options, id=sid, precision=self.qconfig.precision, keys={"pp": self.pp})
                )
        return server

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
        binary = sorted({key for chunk in layer["chunks"] for key, v in chunk.items() if isinstance(v, _BYTES)})
        chunks = list(layer["chunks"])
        if binary:
            for j, chunk in enumerate(chunks):
                if any(not isinstance(chunk.get(key), _BYTES) for key in binary):
                    raise TypeError("chunk {} of a packed layer lacks the binary fields {} of the others".format(j, binary))
                chunks[j] = dict(chunk, **{key: base64.b64encode(bytes(chunk[key])).decode("ascii") for key in binary})
        out = {
            "shape": list(layer["shape"]),
            "first_chunk": layer["first_chunk"],
            "chunks": chunks,
        }
        if "label_hash" in layer:  # the layer's label-hash code (MCFE Ring-LWE)
            out["label_hash"] = layer["label_hash"]
        if binary:
            out["base64"] = binary
        return out

    #: Largest number of elements :meth:`from_dict` accepts per array.
    MAX_ELEMENTS = 10 ** 7

    @classmethod
    def from_dict(cls, d: dict) -> EncryptedGradient:
        """Deserialise from dict.

        Raises:
            ValueError: If an array's shape is invalid or has more than
                :attr:`MAX_ELEMENTS` elements, or its entries do not cover
                exactly the positions of that shape; if a packed layer is not a
                valid ``{"shape", "first_chunk", "chunks"}`` layout; or if packed
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
            layer = wrapped["packed"]
            if (
                set(wrapped) != {"packed"} or not isinstance(layer, dict)
                or set(layer) - {"label_hash", "base64"} != {"shape", "first_chunk", "chunks"}
                or ("label_hash" in layer and type(layer["label_hash"]) is not int)
            ):
                raise ValueError("invalid packed ciphertext layer")
            encoded = layer.get("base64", [])
            if (
                not isinstance(encoded, list) or ("base64" in layer and not encoded)
                or any(not isinstance(key, str) or key == "k" for key in encoded) or len(set(encoded)) != len(encoded)
            ):
                raise ValueError("invalid packed ciphertext layer: base64 names the chunk fields written as base64")
            shape, size = cls._checked_shape(layer["shape"])
            first, chunks = layer["first_chunk"], layer["chunks"]
            if not isinstance(first, int) or isinstance(first, bool) or first != next_chunk:
                raise ValueError(
                    "packed layer has first chunk {!r}, expected {}".format(first, next_chunk)
                )
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("packed chunks must be a non-empty list")
            chunks = list(chunks)
            for j, chunk in enumerate(chunks):
                k = chunk.get("k") if isinstance(chunk, dict) else None
                if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                    raise ValueError(
                        "chunk {} of a packed layer is not a ciphertext with a batch "
                        "length k >= 1".format(j)
                    )
                if encoded:
                    chunks[j] = dict(chunk, **{key: _from_base64(chunk.get(key), "chunk {} field {!r}".format(j, key))
                                               for key in encoded})
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
    plaintext term. For ``mcfe/ddh``, ``mcfe/damgard`` and ``mcfe/paillier``
    (and their decentralized variants; :mod:`pyfe4ai.schemes.mcfe.offline`)
    the label masks of every element are computed. The result is secret and
    one-time; the ciphertexts are those of the plain call.

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
    with wrapper._label_lock:
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
        # reserve (nid, label) before encrypting: of two concurrent calls one
        # encrypts, the other is refused; a failed encryption releases it
        with wrapper._label_lock:
            if (nid, label) in wrapper._encrypted_labels:
                raise ValueError(
                    "client {} already encrypted under label {!r}; use a fresh label "
                    "for every round".format(nid, label)
                )
            wrapper._encrypted_labels.add((nid, label))
        kwargs["label"] = label
    if precomputed is not None:
        kwargs["precomputed"] = precomputed

    try:
        cts = encryptor.encrypt_lst_ndarray([arr], **kwargs)
    except BaseException:
        if wrapper.has_label:
            with wrapper._label_lock:
                wrapper._encrypted_labels.discard((nid, label))
        raise

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
    """A functional key derived before the round (:func:`prepare_aggregation`);
    for a threshold MCFE the key shares of the enrolled ``servers``
    (``dk = {sid: key share}``) and ``sid`` the combiner.

    ``dk`` is key material (a functional key, or the servers' label key shares
    with their MAC and noise keys): keep it where the wrapper is, and do not
    send, log or store it elsewhere."""

    label: str | None
    sid: str
    fusion_weight: dict
    weight_scale: int
    n_chunks: int | None
    dk: dict
    allow_partial: bool
    servers: list | None = None


def _aggregation_key(wrapper, weights, label, sid, nids, allow_partial, n_chunks_fn, servers=None):
    """Policy checks of :func:`aggregate_gradients` and the functional key:
    ``(fusion_weight, weight_scale, dk)``; *nids* are the clients whose
    ciphertexts will be aggregated, ``n_chunks_fn()`` the number of packed
    chunks a key must cover (``None``: not a packed scheme). For a threshold
    MCFE run by the wrapper ``dk`` holds the key shares of the enrolled
    *servers* (:func:`_server_key_shares`)."""
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
    if servers is not None:
        return fusion_weight, weight_scale, _server_key_shares(wrapper, credentials, servers)
    dk = wrapper.kg.get_decryption_keys(sid, credentials=credentials)
    return fusion_weight, weight_scale, dk


def _enrolled_servers(wrapper, servers) -> list:
    """The servers whose partial decryptions a threshold aggregation combines:
    *servers* (at least ``t`` distinct servers of the public parameters) or
    the first ``t`` of ``lst_sid``.

    Raises:
        ValueError: For other servers.
        NotImplementedError: Under ``flooding_accounting = "kl"``, whose
            servers need a partial-decryption budget of their own.
    """
    if wrapper.pp.get("flooding_accounting", "per-sample") != "per-sample":
        raise NotImplementedError(
            "flooding_accounting={!r}: every server needs a partial_budget of its own; run the servers "
            "with the scheme classes".format(wrapper.pp.get("flooding_accounting"))
        )
    lst_sid, t = list(wrapper.pp["lst_sid"]), int(wrapper.pp["t"])
    if servers is None:
        return lst_sid[:t]
    servers = list(servers)
    if len(servers) < t or len(set(servers)) != len(servers) or not set(servers) <= set(lst_sid):
        raise ValueError("servers must be at least t = {} distinct servers of {}".format(t, lst_sid))
    return servers


def _server_key_shares(wrapper, credentials: dict, servers: list) -> dict:
    """``{sid: key share}`` of the enrolled servers for the round (by default
    label key shares, with the chunk keys only), each server given its weight
    key share first where it lacks it: once per weights and share set."""
    out = {}
    for sid in servers:
        dk = wrapper.kg.get_decryption_keys(sid, credentials=dict(credentials, label_term=False))
        _serving(wrapper, sid, dk, credentials["fusion_weight"])
        out[sid] = dk
    return out


def _serving(wrapper, sid: str, dk: dict, fusion_weight: dict):
    """Server *sid* of the wrapper, given the weight key share that the label
    key share *dk* completes where it lacks it (first use, a copied wrapper's
    new server objects, or a weight part its store dropped)."""
    server = wrapper._get_server(sid)
    if not server.has_weight_key_share(dk):
        server.add_weight_key_share(wrapper.kg.get_weight_key_share(sid, credentials={"fusion_weight": fusion_weight}))
    return server


def prepare_aggregation(
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list],
    *,
    shape,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
    servers: list | None = None,
) -> PreparedAggregation:
    """Derive the functional key of an aggregation before the ciphertexts arrive.

    The key depends on the weights, the label and (for the packed Ring-LWE
    MCFE) the number of chunks of a gradient of *shape*, not on the
    ciphertexts; :func:`aggregate_gradients` with ``prepared=`` then only
    decrypts. The same policy checks apply, with the clients of *weights* as
    those whose ciphertexts will be aggregated.

    For ``mcfe/ring_lwe_threshold`` the key shares of the enrolled *servers*
    (default: the first ``t`` of the public parameters' ``lst_sid``): their
    label key shares of the round, each server given its weight key share
    once per weights (again where a copy of the wrapper or the server's store
    lacks it when the round is decrypted); *sid* is the combiner. The result
    holds key material (see :class:`PreparedAggregation`).
    """
    _require_labelled_aggregation(wrapper, "prepare_aggregation")
    _require_single_decryptor(wrapper, "prepare_aggregation", served_threshold=True)
    if servers is not None and not _serves_threshold(wrapper):
        raise ValueError("servers are those of a threshold MCFE")
    enrolled = _enrolled_servers(wrapper, servers) if _serves_threshold(wrapper) else None
    decryptor = wrapper._get_decryptor(sid)
    n_chunks = None
    if getattr(decryptor, "key_chunks", None) is not None:
        ring_n = int(wrapper.pp["ring_n"])
        size = int(np.prod(np.shape(np.empty(shape)), dtype=np.int64))
        n_chunks = max(1, -(-size // ring_n))
    fusion_weight, weight_scale, dk = _aggregation_key(
        wrapper, weights, label, sid, list(weights), allow_partial, lambda: n_chunks, enrolled
    )
    prepare_key = getattr(decryptor, "prepare_key", None)
    if prepare_key is not None:  # element-wise MCFE: the label terms of the key too
        dk = prepare_key(dk, label, [tuple(int(d) for d in np.shape(np.empty(shape)))])
    return PreparedAggregation(label, sid, fusion_weight, weight_scale, n_chunks, dk, allow_partial, enrolled)


def aggregate_gradients(
    encrypted: dict[str, EncryptedGradient],
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list] | None = None,
    *,
    sid: str = "sid_agg",
    label: str | None = None,
    allow_partial: bool = False,
    prepared: PreparedAggregation | None = None,
    servers: list | None = None,
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
            servers: ``mcfe/ring_lwe_threshold``: the servers whose partial
                decryptions are combined (default: those of *prepared*, or
                the first ``t`` of ``lst_sid``); the wrapper runs them
                (:func:`partial_decrypt_gradients`,
                :func:`combine_partial_decryptions`) and *sid* is the
                combiner. If their label key shares turn out to be of
                different sharings (``MixedSharingsError``), all servers
                fetch them again and decrypt once more.

        Raises:
            ValueError: If the policy above is violated, or SIFE is given
                more than one client.
    """
    _require_labelled_aggregation(wrapper, "aggregate_gradients")
    _require_single_decryptor(wrapper, "aggregate_gradients", served_threshold=True)
    threshold = _serves_threshold(wrapper)
    if servers is not None and not threshold:
        raise ValueError("servers are those of a threshold MCFE")
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
        enrolled = _enrolled_servers(wrapper, servers) if threshold else None
        fusion_weight, weight_scale, dk = _aggregation_key(
            wrapper, weights, label, sid, list(encrypted), allow_partial, n_chunks_fn, enrolled
        )
        if threshold:
            prepared = PreparedAggregation(label, sid, fusion_weight, weight_scale, n_chunks_fn(), dk, allow_partial,
                                           enrolled)
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
        if threshold and servers is not None and list(servers) != prepared.servers:
            raise ValueError("the prepared key shares are for the servers {}".format(prepared.servers))
        fusion_weight, weight_scale, dk = prepared.fusion_weight, prepared.weight_scale, prepared.dk

    if threshold:  # the servers the wrapper runs, then the combiner
        from pyfe4ai.schemes.mcfe.ring_lwe_threshold import MixedSharingsError

        for attempt in (0, 1):
            partials = {
                server: partial_decrypt_gradients(encrypted, wrapper, prepared, server=server)
                for server in prepared.servers
            }
            try:
                return combine_partial_decryptions(partials, wrapper, prepared)
            except MixedSharingsError:
                if attempt:
                    raise
                # the label's parts were shared again between the servers' fetches: all servers fetch them again
                credentials = {"fusion_weight": prepared.fusion_weight, "label": prepared.label,
                               "n_chunks": prepared.n_chunks}
                prepared = replace(prepared, dk=_server_key_shares(wrapper, credentials, prepared.servers))

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

    return _aggregated(result_arrays, weight_scale)


def _aggregated(result_arrays: list, weight_scale: int) -> np.ndarray:
    """The decrypted layers, scaled back from integer weights; one layer as is."""
    if weight_scale != 1:
        result_arrays = [np.asarray(ary, dtype=np.float64) / weight_scale for ary in result_arrays]
    if result_arrays and len(result_arrays) == 1:
        return result_arrays[0]
    return np.array(result_arrays)


# ---------------------------------------------------------------------------
# Threshold MCFE: partial decryptions of the servers the wrapper runs
# ---------------------------------------------------------------------------

#: Fields of one partial decryption (one chunk) of a :class:`ThresholdPartial`.
_PARTIAL_FIELDS = frozenset({"w", "c_y", "k", "ct_digest", "sid", "lst_sid_enrolled", "sharing_id"})


@dataclass
class ThresholdPartial:
    """One server's partial decryptions of a round's gradients (threshold
    MCFE), from :func:`partial_decrypt_gradients` for
    :func:`combine_partial_decryptions`."""

    sid: str
    label: str
    #: :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.compute_lst_ndarray_ct`
    #: of the clients' gradients: per layer ``{"shape", "parts"}``, one partial
    #: decryption per chunk (``w`` and ``c_y`` bytes for binary public
    #: parameters, lists of decimal strings for JSON ones; ``sharing_id``
    #: that of the server's key share, or ``None``). :meth:`to_dict` keeps the
    #: ``sharing_id``; a transport of its own must keep it as well, or the
    #: combiner can no longer tell partials of two sharings apart.
    layers: list

    def to_dict(self) -> dict:
        """Serialise for transfer: ``{"sid", "label", "format", "layers"}``,
        binary ``w`` and ``c_y`` (``format = "binary"``) as base64 strings.

        Raises:
            TypeError: If the partial decryptions mix binary and JSON values.
        """
        parts = [part for layer in self.layers for part in layer["parts"]]
        binary = [isinstance(part["w"], _BYTES) and isinstance(part["c_y"], _BYTES) for part in parts]
        if any(binary) and not all(binary):
            raise TypeError("partial decryptions mixing binary and JSON values")
        fmt = "binary" if binary and all(binary) else "json"

        def written(part):
            if fmt == "json":
                return dict(part, w=list(part["w"]), c_y=list(part["c_y"]), lst_sid_enrolled=list(part["lst_sid_enrolled"]))
            return dict(part, w=base64.b64encode(bytes(part["w"])).decode("ascii"),
                        c_y=base64.b64encode(bytes(part["c_y"])).decode("ascii"),
                        lst_sid_enrolled=list(part["lst_sid_enrolled"]))

        return {
            "sid": self.sid,
            "label": self.label,
            "format": fmt,
            "layers": [{"shape": list(layer["shape"]), "parts": [written(p) for p in layer["parts"]]}
                       for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ThresholdPartial:
        """Deserialise from :meth:`to_dict`, checking the layout (the combiner
        checks the values).

        Raises:
            ValueError: For another layout, a partial decryption of another
                server, or ``w`` / ``c_y`` not of the format (strict base64
                for ``"binary"``, lists of strings for ``"json"``).
        """
        if not isinstance(d, dict) or set(d) != {"sid", "label", "format", "layers"}:
            raise ValueError("invalid partial decryption: keys sid, label, format and layers")
        sid, label, fmt, layers = d["sid"], d["label"], d["format"], d["layers"]
        if not isinstance(sid, str) or not isinstance(label, str) or fmt not in ("binary", "json"):
            raise ValueError("invalid partial decryption: sid and label are strings, format binary or json")
        if not isinstance(layers, list) or not layers:
            raise ValueError("invalid partial decryption: layers must be a non-empty list")
        out = []
        for l, layer in enumerate(layers):
            if not isinstance(layer, dict) or set(layer) != {"shape", "parts"}:
                raise ValueError("invalid partial decryption: layer {} has keys shape and parts".format(l))
            shape, _ = EncryptedGradient._checked_shape(layer["shape"])
            if not isinstance(layer["parts"], list) or not layer["parts"]:
                raise ValueError("invalid partial decryption: layer {} has no parts".format(l))
            parts = []
            for j, part in enumerate(layer["parts"]):
                where = "partial decryption ({}, chunk {})".format(l, j)
                if not isinstance(part, dict) or set(part) != _PARTIAL_FIELDS:
                    raise ValueError("invalid {}: keys {}".format(where, sorted(_PARTIAL_FIELDS)))
                k, enrolled = part["k"], part["lst_sid_enrolled"]
                if type(k) is not int or k < 1 or not isinstance(part["ct_digest"], str) or part["sid"] != sid:
                    raise ValueError("invalid {}: k >= 1, a ct_digest and this server's sid".format(where))
                if not isinstance(enrolled, list) or not all(isinstance(v, str) for v in enrolled):
                    raise ValueError("invalid {}: lst_sid_enrolled is a list of server ids".format(where))
                sharing_id = part["sharing_id"]
                if sharing_id is not None and not (isinstance(sharing_id, str) and 0 < len(sharing_id) <= 128):
                    raise ValueError("invalid {}: sharing_id is a string (or null)".format(where))
                if fmt == "binary":
                    values = {key: _from_base64(part[key], "{} {}".format(where, key)) for key in ("w", "c_y")}
                else:
                    if any(not isinstance(part[key], list) or not all(isinstance(v, str) for v in part[key])
                           for key in ("w", "c_y")):
                        raise ValueError("invalid {}: w and c_y are lists of decimal strings".format(where))
                    values = {"w": list(part["w"]), "c_y": list(part["c_y"])}
                parts.append(dict(part, lst_sid_enrolled=list(enrolled), **values))
            out.append({"shape": list(shape), "parts": parts})
        return cls(sid=sid, label=label, layers=out)


def partial_decrypt_gradients(
    encrypted: dict[str, EncryptedGradient],
    wrapper: FESchemeWrapper,
    prepared: PreparedAggregation,
    *,
    server: str,
) -> ThresholdPartial:
    """Server *server*'s partial decryptions of the clients' gradients for a
    prepared threshold aggregation (``mcfe/ring_lwe_threshold``; the wrapper
    runs the server, with its key share of *prepared*, see
    :func:`prepare_aggregation`).

    Raises:
        ValueError: If *server* is not enrolled in *prepared*, or the clients
            are not those of its weights.
    """
    _require_served_threshold(wrapper, "partial_decrypt_gradients")
    if not prepared.servers or server not in prepared.servers:
        raise ValueError("server {!r} is not enrolled in the prepared aggregation ({})".format(server, prepared.servers))
    if set(encrypted) != set(prepared.fusion_weight):
        raise ValueError("the prepared key is for the clients {}".format(sorted(prepared.fusion_weight)))
    layers = _serving(wrapper, server, prepared.dk[server], prepared.fusion_weight).compute_lst_ndarray_ct(
        {nid: enc.ciphertexts for nid, enc in encrypted.items()},
        {"fusion_weight": prepared.fusion_weight, "label": prepared.label},
        prepared.dk[server],
        prepared.servers,
    )
    return ThresholdPartial(sid=server, label=prepared.label, layers=layers)


def combine_partial_decryptions(
    partials: dict[str, ThresholdPartial],
    wrapper: FESchemeWrapper,
    prepared: PreparedAggregation,
) -> np.ndarray:
    """Combine the enrolled servers' partial decryptions
    (:func:`partial_decrypt_gradients`) into the weighted sum of the
    gradients, as :func:`aggregate_gradients` returns it.

    Raises:
        ValueError: Unless *partials* are those of the prepared servers for
            its label.
        FESchemeError: If they do not combine (see
            :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.decrypt_lst_ndarray_ct`);
            :class:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.MixedSharingsError`
            if their label parts are of different sharings (the key generator
            shared the label again between the servers' fetches): prepare the
            round again and have all servers decrypt with the new key shares
            (:func:`aggregate_gradients` does this once by itself).
    """
    _require_served_threshold(wrapper, "combine_partial_decryptions")
    if set(partials) != set(prepared.servers or []) or any(
            p.sid != sid or p.label != prepared.label for sid, p in partials.items()):
        raise ValueError("need the partial decryptions of the servers {} for label {!r}".format(
            prepared.servers, prepared.label))
    combiner = wrapper._get_decryptor(prepared.sid)
    return _aggregated(combiner.decrypt_lst_ndarray_ct({sid: p.layers for sid, p in partials.items()}),
                       prepared.weight_scale)


# ---------------------------------------------------------------------------
# Encrypted Inference API (eta=feature_dim, vector mode)
# ---------------------------------------------------------------------------


#: SIFE schemes whose ``encrypt`` takes an ``eta x k`` matrix (``k`` vectors
#: in the ring's coefficients): a feature vector is one column of it.
_MATRIX_PLAINTEXT_SIFE = {("sife", "ring_lwe")}


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
    if (wrapper.scheme_type, wrapper.variant) in _MATRIX_PLAINTEXT_SIFE:
        ct = encryptor.encrypt([[v] for v in quantized])  # one column: the feature vector
    else:
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
    if isinstance(raw_result, list):  # a matrix ciphertext of one column: its one inner product
        raw_result = raw_result[0] if len(raw_result) == 1 else None
    if raw_result is None:
        raise FESchemeError(
            "the inner product does not decrypt (result outside the decryption bound, "
            "or key and ciphertext do not match)"
        )

    # Both x and w were quantized → result scaled by scale_factor^2
    return qcfg.dequantize(raw_result, order=2)
