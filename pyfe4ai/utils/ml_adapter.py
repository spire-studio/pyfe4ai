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
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyfe4ai.utils.quantization import QuantizationConfig

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
    ("mife", "ddh_hybrid_alpha"): (
        "pyfe4ai.schemes.mife.ddh_hybrid_alpha",
        "MIFEKeyGenerator",
        "MIFE",
    ),
    # ---- MCFE ----------------------------------------------------------
    ("mcfe", "ddh"): ("pyfe4ai.schemes.mcfe.ddh", "MCFEKeyGenerator", "MCFE"),
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
    ("quadratic", "multi_input_sgp"): (
        "pyfe4ai.schemes.quadratic.multi_input_sgp",
        "MultiInputQuadraticSGPKeyGenerator",
        "MultiInputQuadraticSGP",
    ),
}

# Schemes whose encrypt/decrypt require a label argument.
_LABEL_SCHEMES = frozenset({"mcfe"})


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
    underlying scheme family.

    Parameters
    ----------
    scheme_type : str
        ``"sife"``, ``"mcfe"``, ``"mife"``, or ``"quadratic"``.
    variant : str
        ``"ddh"``, ``"lwe"``, ``"paillier"``, ``"damgard"``, ``"ring_lwe"``,
        ``"quad"``, or ``"sgp"``.
    keygen_config : dict
        Configuration dict for the KeyGenerator (``sec_param``, ``eta``,
        ``n``, ``bound_x``, etc.).
    precision : int
        Quantization precision (default 0 — no quantization).
    qmode : str
        Quantization mode: ``"decimal"`` or ``"binary"`` (default ``"decimal"``).
    """

    def __init__(
        self,
        scheme_type: str,
        variant: str,
        keygen_config: dict,
        *,
        precision: int = 0,
        qmode: str = "decimal",
    ) -> None:
        """Perform the __init__ operation.

            Args:
                scheme_type: See implementation for details.
                variant: See implementation for details.
                keygen_config: See implementation for details.
                precision: See implementation for details.
                qmode: See implementation for details.
        """
        key = (scheme_type.lower(), variant.lower())
        if key not in _SCHEME_REGISTRY:
            supported = sorted(f"{s}/{v}" for s, v in _SCHEME_REGISTRY)
            raise ValueError(
                f"Unknown scheme '{scheme_type}/{variant}'. "
                f"Supported: {supported}"
            )

        self.scheme_type = scheme_type.lower()
        self.variant = variant.lower()
        self.keygen_config = dict(keygen_config)
        self.qconfig = QuantizationConfig(precision=precision, mode=qmode)
        self.has_label = self.scheme_type in _LABEL_SCHEMES

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

    # -- crypto instance management ----------------------------------------

    def _get_encryptor(self, nid: str) -> Any:
        """Get or create a crypto instance for the given node id."""
        if nid not in self._encryptors:
            sk = self.kg.get_private_keys(nid)
            keys: dict[str, Any] = {"pp": self.pp}
            if sk is not None:
                keys["sk"] = sk
            self._encryptors[nid] = self._crypto_cls(
                {
                    "id": nid,
                    "precision": self.qconfig.precision,
                    "keys": keys,
                }
            )
        return self._encryptors[nid]

    def _get_decryptor(self, sid: str) -> Any:
        """Create a crypto instance for decryption (no sk needed)."""
        return self._crypto_cls(
            {
                "id": sid,
                "precision": self.qconfig.precision,
                "keys": {"pp": self.pp},
            }
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

    ciphertexts: list  # list of np.ndarray(dtype=object)
    shape: tuple
    precision: int
    mode: str
    nid: str

    def to_dict(self) -> dict:
        """Serialise for network transfer (e.g. federated learning)."""
        serialised_cts = []
        for ary in self.ciphertexts:
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

    @classmethod
    def from_dict(cls, d: dict) -> EncryptedGradient:
        """Deserialise from dict."""
        cts = []
        for ct_dict in d["ciphertexts"]:
            shape = tuple(ct_dict["shape"])
            ary = np.empty(shape, dtype=object)
            for key, val in ct_dict["data"].items():
                idx = tuple(
                    int(x) for x in key.strip("()").split(",") if x.strip()
                )
                ary[idx] = val
            cts.append(ary)
        return cls(
            ciphertexts=cts,
            shape=tuple(d["shape"]),
            precision=d["precision"],
            mode=d["mode"],
            nid=d["nid"],
        )


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


def encrypt_gradient(
    gradient: np.ndarray | Any,
    wrapper: FESchemeWrapper,
    *,
    nid: str = "nid_default",
    label: str | None = None,
) -> EncryptedGradient:
    """Encrypt a gradient array for federated aggregation.

        Args:
            gradient: See implementation for details.
            wrapper: FE scheme wrapper instance.
            nid: Node identifier.
            label: Encryption label for replay protection.
    """
    arr = _to_numpy(gradient).astype(np.float64)
    encryptor = wrapper._get_encryptor(nid)

    kwargs: dict[str, Any] = {}
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for encryption")
        kwargs["label"] = label

    cts = encryptor.encrypt_lst_ndarray([arr], **kwargs)

    return EncryptedGradient(
        ciphertexts=cts,
        shape=arr.shape,
        precision=wrapper.qconfig.precision,
        mode=wrapper.qconfig.mode,
        nid=nid,
    )


def aggregate_gradients(
    encrypted: dict[str, EncryptedGradient],
    wrapper: FESchemeWrapper,
    weights: dict[str, float | int | list],
    *,
    sid: str = "sid_agg",
    label: str | None = None,
) -> np.ndarray:
    """Decrypt and aggregate encrypted gradients from multiple clients.

        Args:
            encrypted: Encrypted data container.
            wrapper: FE scheme wrapper instance.
            weights: Weight vector or dict.
            sid: Session / decryption-key identifier.
            label: Encryption label for replay protection.
    """
    # Normalise weights to {nid: [w]} format for eta=1 schemes
    fusion_weight: dict[str, list] = {}
    for nid, w in weights.items():
        if isinstance(w, (int, float)):
            qw = wrapper.qconfig.quantize(w) if isinstance(w, float) else w
            fusion_weight[nid] = [qw]
        else:
            fusion_weight[nid] = list(w)

    # Build credentials for dk generation
    # SIFE expects fusion_weight as a flat list; MCFE/MIFE expect a dict.
    if wrapper.scheme_type in ("mcfe", "mife"):
        credentials: dict[str, Any] = {"fusion_weight": fusion_weight}
    else:
        # Single-input: take the first (only) client's weight list
        credentials = {"fusion_weight": next(iter(fusion_weight.values()))}
    if wrapper.has_label:
        if label is None:
            raise ValueError("MCFE schemes require a label for decryption")
        credentials["label"] = label

    dk = wrapper.kg.get_decryption_keys(sid, credentials=credentials)

    # Collect ciphertext arrays per nid
    # decrypt_lst_ndarray_ct expects: list of arrays (one per layer)
    # where each array element is a ciphertext dict
    sample_enc = next(iter(encrypted.values()))
    n_layers = len(sample_enc.ciphertexts)

    # For SIFE: dict_ndarray_ct is list of arrays
    # For MCFE/MIFE: dict_ndarray_ct is dict[nid -> list of arrays]
    decryptor = wrapper._get_decryptor(sid)

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
            features: See implementation for details.
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
    w_arr = _to_numpy(weights).ravel().astype(np.float64)
    qcfg = wrapper.qconfig
    q_weights = qcfg.quantize_array(w_arr).tolist()

    dk = wrapper.get_decryption_keys(sid, fusion_weight=q_weights)

    decryptor = wrapper._get_decryptor(sid)
    raw_result = decryptor.decrypt(encrypted.ciphertext, dk, q_weights)

    # Both x and w were quantized → result scaled by scale_factor^2
    return qcfg.dequantize(raw_result, order=2)
