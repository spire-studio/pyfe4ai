"""TypedDict schemas for KeyGenerator and Crypto config dicts.

All fields are optional (``total=False``) because every KeyGenerator
provides sensible defaults via ``config.get(key, default)``.  The schemas
exist for documentation and IDE autocompletion when constructing configs::

    from pyfe4ai.schemes.config_types import DDHConfig

    config: DDHConfig = {"sec_param": 128, "eta": 3}
    kg = SIFEKeyGenerator(config)
"""

from __future__ import annotations

from typing import TypedDict


# ── Base ────────────────────────────────────────────────────────────


class FEBaseConfig(TypedDict, total=False):
    """Fields read by :class:`~pyfe4ai.schemes.ipfe.IPFEAbsKeyGenerator`."""

    sec_param: int
    n: int
    s: int
    lst_nid: list[str]


# ── DDH family ──────────────────────────────────────────────────────


class DDHConfig(FEBaseConfig, total=False):
    """Config for standard DDH schemes (SIFE / MIFE / MCFE DDH)."""

    eta: int


class DamgardDDHConfig(FEBaseConfig, total=False):
    """Config for Damgard DDH schemes."""

    eta: int
    modulus_length: int
    bound: int


# ── LWE family ──────────────────────────────────────────────────────


class LWEConfig(FEBaseConfig, total=False):
    """Config for LWE-based schemes (SIFE / MIFE LWE, FullySec LWE)."""

    eta: int
    lwe_n: int
    bound_x: int
    bound_y: int


class MCFELWEConfig(LWEConfig, total=False):
    """Config for MCFE LWE schemes (adds label / noise bounds)."""

    bound_u: int
    label_modulus: int


# ── Ring-LWE family ─────────────────────────────────────────────────


class RingLWEConfig(FEBaseConfig, total=False):
    """Config for Ring-LWE-based schemes."""

    eta: int
    ring_n: int
    bound_x: int
    bound_y: int


class MCFERingLWEConfig(RingLWEConfig, total=False):
    """Config for MCFE Ring-LWE schemes."""

    bound_u: int
    label_modulus: int


# ── Paillier family ─────────────────────────────────────────────────


class PaillierConfig(FEBaseConfig, total=False):
    """Config for Paillier-based schemes."""

    eta: int
    bit_length: int
    bound_x: int
    bound_y: int


# ── Pairing-based family ────────────────────────────────────────────


class PairingIPEConfig(FEBaseConfig, total=False):
    """Config for pairing-based IPE schemes (FH-IPE)."""

    eta: int
    bound_x: int
    bound_y: int
    pairing_group_param: str


class PairingMultiIPEConfig(PairingIPEConfig, total=False):
    """Config for pairing-based multi-input IPE schemes (FH-MULTI-IPE)."""

    sec_level: int
    u_bound: int
    label_modulus: int


class PartFHIPEConfig(FEBaseConfig, total=False):
    """Config for partially function-hiding IPE schemes."""

    eta: int
    bound: int
    bound_x: int
    subspace_dim: int
    coeff_bound: int
    pairing_group_param: str


# ── Threshold extensions ────────────────────────────────────────────


class ThresholdDDHConfig(DDHConfig, total=False):
    """Config for threshold DDH-based schemes."""

    t: int
    lst_sid: list[str]


class ThresholdLWEConfig(LWEConfig, total=False):
    """Config for threshold LWE-based schemes."""

    t: int
    lst_sid: list[str]


class ThresholdMCFELWEConfig(MCFELWEConfig, total=False):
    """Config for threshold MCFE LWE-based schemes."""

    t: int
    lst_sid: list[str]


class ThresholdMCFERingLWEConfig(MCFERingLWEConfig, total=False):
    """Config for threshold MCFE Ring-LWE-based schemes."""

    t: int
    lst_sid: list[str]


class ThresholdPairingMultiIPEConfig(PairingMultiIPEConfig, total=False):
    """Config for threshold pairing-based multi-input IPE schemes."""

    t: int
    lst_sid: list[str]


# ── Quadratic family ────────────────────────────────────────────────


class QuadraticSGPConfig(FEBaseConfig, total=False):
    """Config for SGP quadratic FE schemes."""

    bound: int
    pairing_group_param: str


class QuadraticQuadConfig(FEBaseConfig, total=False):
    """Config for Quad quadratic FE schemes."""

    m: int
    bound: int
    pairing_group_param: str


class MultiInputSGPConfig(FEBaseConfig, total=False):
    """Config for multi-input SGP quadratic FE schemes."""

    d: int
    bound: int
    pairing_group_param: str


# ── Crypto config ───────────────────────────────────────────────────


class CryptoConfig(TypedDict, total=False):
    """Config for :class:`~pyfe4ai.schemes.ipfe.IPFEAbsCrypto` subclasses."""

    keys: dict
    id: str
    type: str
    precision: int
