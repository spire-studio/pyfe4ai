"""The scheme wrapper of the ML adapter: :class:`FESchemeWrapper`, its scheme
registry and the policies of the FL helpers, and :class:`PreparedAggregation`.

**Single trust domain.** A :class:`FESchemeWrapper` runs the key generator
itself, so it holds the master secret key and plays every party (clients,
key authority, decryptor); see :mod:`pyfe4ai.ml`.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyfe4ai.core.status import get_scheme_status
from pyfe4ai.ml.quantization import QuantizationConfig

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
    ("sife", "ddh"): ("pyfe4ai.schemes.sife.ddh", "SIFEDDHKeyGenerator", "SIFEDDH"),
    ("sife", "ddh_ristretto255"): (
        "pyfe4ai.schemes.sife.ddh_ristretto255",
        "SIFEDDHRistretto255KeyGenerator",
        "SIFEDDHRistretto255",
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
        "SIFEDDHDynamicKeyGenerator",
        "SIFEDDHDynamic",
    ),
    ("sife", "fh_ipe_pairing"): (
        "pyfe4ai.schemes.sife.fh_ipe_pairing",
        "SIFEFHIPEPairingKeyGenerator",
        "SIFEFHIPEPairing",
    ),
    ("sife", "part_fh_ipe_pairing"): (
        "pyfe4ai.schemes.sife.part_fh_ipe_pairing",
        "SIFEPartFHIPEPairingKeyGenerator",
        "SIFEPartFHIPEPairing",
    ),
    # ---- MIFE ----------------------------------------------------------
    ("mife", "ddh"): ("pyfe4ai.schemes.mife.ddh", "MIFEDDHKeyGenerator", "MIFEDDH"),
    ("mife", "ddh_ristretto255"): (
        "pyfe4ai.schemes.mife.ddh_ristretto255",
        "MIFEDDHRistretto255KeyGenerator",
        "MIFEDDHRistretto255",
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
        "MIFEFHIPEPairingKeyGenerator",
        "MIFEFHIPEPairing",
    ),
    ("mife", "fh_multi_ipe_pairing"): (
        "pyfe4ai.schemes.mife.fh_multi_ipe_pairing",
        "MIFEFHMultiIPEPairingKeyGenerator",
        "MIFEFHMultiIPEPairing",
    ),
    ("mife", "ddh_threshold"): (
        "pyfe4ai.schemes.mife.ddh_threshold",
        "ThresholdMIFEDDHKeyGenerator",
        "ThresholdMIFEDDH",
    ),
    ("mife", "lwe_threshold"): (
        "pyfe4ai.schemes.mife.lwe_threshold",
        "ThresholdMIFELWEKeyGenerator",
        "ThresholdMIFELWE",
    ),
    # ---- MCFE ----------------------------------------------------------
    ("mcfe", "ddh"): ("pyfe4ai.schemes.mcfe.ddh", "MCFEDDHKeyGenerator", "MCFEDDH"),
    ("mcfe", "ddh_ristretto255"): (
        "pyfe4ai.schemes.mcfe.ddh_ristretto255",
        "MCFEDDHRistretto255KeyGenerator",
        "MCFEDDHRistretto255",
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
        "MCFEFHMultiIPEPairingKeyGenerator",
        "MCFEFHMultiIPEPairing",
    ),
    ("mcfe", "fh_multi_ipe_pairing_threshold"): (
        "pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold",
        "ThresholdMCFEFHMultiIPEPairingKeyGenerator",
        "ThresholdMCFEFHMultiIPEPairing",
    ),
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): (
        "pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized",
        "DecentralizedMCFEFHMultiIPEPairingKeyGenerator",
        "DecentralizedMCFEFHMultiIPEPairing",
    ),
    ("mcfe", "ddh_threshold"): (
        "pyfe4ai.schemes.mcfe.ddh_threshold",
        "ThresholdMCFEDDHKeyGenerator",
        "ThresholdMCFEDDH",
    ),
    ("mcfe", "ddh_decentralized"): (
        "pyfe4ai.schemes.mcfe.ddh_decentralized",
        "DecentralizedMCFEDDHKeyGenerator",
        "DecentralizedMCFEDDH",
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
        "QuadraticMIQFEKeyGenerator",
        "QuadraticMIQFE",
    ),
}

# Schemes whose encrypt/decrypt require a label argument.
_LABEL_SCHEMES = frozenset({"mcfe"})

#: Threshold variants whose decryption servers the wrapper runs (FL helpers).
_SERVED_THRESHOLD = frozenset({("mcfe", "ring_lwe_threshold")})

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
    so forms a single trust domain (see :mod:`pyfe4ai.ml`).

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


def _aggregated(result_arrays: list, weight_scale: int) -> np.ndarray:
    """The decrypted layers, scaled back from integer weights; one layer as is."""
    if weight_scale != 1:
        result_arrays = [np.asarray(ary, dtype=np.float64) / weight_scale for ary in result_arrays]
    if result_arrays and len(result_arrays) == 1:
        return result_arrays[0]
    return np.array(result_arrays)
