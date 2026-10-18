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
    """Fields read by :class:`~pyfe4ai.schemes.ipfe.IPFEAbsKeyGenerator`.

    ``sec_param`` is *not* a security level. DDH families use it as the bit
    length of the safe prime ``p`` (and Damgard as the default
    ``modulus_length``); Ring-LWE SIFE uses it as the statistical parameter
    kappa in its noise bounds; LWE families only record it (their hardness is
    set by ``lwe_n``), Paillier families use ``bit_length`` and pairing families
    ``pairing_group_param``.

    ``security_profile`` (``"toy"`` by default, or ``"standard"``) presets
    these size parameters; see :mod:`pyfe4ai.schemes.profiles`.
    ``stat_param`` is the statistical parameter of the schemes that use one
    (ABMPSW20 rounding, flooding, masks, Paillier key widths): 64 by default,
    at least 128 under ``standard``.
    """

    security_profile: str
    sec_param: int
    stat_param: int
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
    """Config for LWE-based schemes (SIFE / MIFE LWE, FullySec LWE).

    ``lwe_n`` is a dimension or ``"auto"``; ``matrix_mode`` (``"memory"`` or
    ``"stream"``) chooses how the ABDP15 schemes hold their seeded matrix.
    """

    eta: int
    lwe_n: int | str
    bound_x: int
    bound_y: int
    matrix_mode: str


class MCFELWEConfig(LWEConfig, total=False):
    """Config for MCFE LWE schemes (ABMPSW20 Fig. 4)."""


# ── Ring-LWE family ─────────────────────────────────────────────────


class RingLWEConfig(FEBaseConfig, total=False):
    """Config for Ring-LWE-based schemes (``ring_n``: a power of two or ``"auto"``;
    ``m``: the number of key polynomials of ``sife/fullysec_ring_lwe``, at
    least 5, or ``"auto"``)."""

    eta: int
    ring_n: int | str
    bound_x: int
    bound_y: int
    m: int | str


class MCFERingLWEConfig(RingLWEConfig, total=False):
    """Config for MCFE Ring-LWE schemes (keys are bound to a label at key generation)."""


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
    """Config for pairing-based multi-input / multi-client IPE schemes (FH-MULTI-IPE)."""


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
    """Config for threshold LWE-based schemes.

    ``flooding_accounting`` (``"per-sample"`` by default, or ``"kl"`` with
    ``max_partial_decryptions`` and ``t = s``) chooses how
    ``mife/lwe_threshold`` sizes its flooding noise; see
    :mod:`pyfe4ai.schemes.flooding_accounting`.
    """

    t: int
    lst_sid: list[str]
    flooding_accounting: str
    max_partial_decryptions: int


class ThresholdMCFELWEConfig(MCFELWEConfig, total=False):
    """Config for threshold MCFE LWE-based schemes."""

    t: int
    lst_sid: list[str]


class ThresholdMCFERingLWEConfig(MCFERingLWEConfig, total=False):
    """Config for threshold MCFE Ring-LWE-based schemes (``flooding_accounting``
    and ``max_partial_decryptions`` as in :class:`ThresholdLWEConfig`)."""

    t: int
    lst_sid: list[str]
    flooding_accounting: str
    max_partial_decryptions: int


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
    """Config for Quad quadratic FE schemes (``x`` of length ``n``, ``y`` of length ``m``)."""

    n: int
    m: int
    bound: int
    pairing_group_param: str


class MultiInputQuadraticConfig(FEBaseConfig, total=False):
    """Config for the AGT22 multi-input quadratic FE (``n`` users, ``m`` integers each)."""

    m: int
    bound_x: int
    bound_c: int
    mddh_k: int
    pairing_group_param: str


# ── Crypto config ───────────────────────────────────────────────────


class CryptoConfig(TypedDict, total=False):
    """Config for :class:`~pyfe4ai.schemes.ipfe.IPFEAbsCrypto` subclasses.

    ``dlog_bound`` (DDH schemes) bounds the discrete log solved at decryption
    (default ``10^(precision + 2)``); ``matrix_mode`` as in :class:`LWEConfig`.
    """

    keys: dict
    id: str
    type: str
    precision: int
    dlog_bound: int
    matrix_mode: str
