"""Review status of every registered scheme.

Each ``(family, variant)`` of the ML-adapter registry has a
:class:`SchemeStatus`:

``status``
    ``"reviewed"``: the implementation was checked against the cited
    construction and no security defect is known. ``"experimental"``:
    implemented and checked here, but the construction or its security
    argument is our own rather than a published one; no defect is known.
    ``"unreviewed"``: not yet
    checked against its paper; no defect is known, but none was looked for
    systematically either. ``"known-insecure"``: a concrete privacy defect is
    known (see ``issue``); the scheme computes correct results but must only
    be used for functionality or performance experiments.

``fidelity``
    ``"faithful"``: the published construction as written.
    ``"adapted"``: the published construction with documented deviations or
    extensions (see ``note``). ``"ad-hoc"``: a composition with no published
    construction behind it. ``"unknown"``: not assessed (unreviewed schemes).

This table is the single source for the README status table and for the
:class:`~pyfe4ai.utils.exceptions.FESecurityWarning` emitted when a
known-insecure key generator is instantiated. All schemes additionally share
the caveats of the README (research prototype, toy default parameters, no
side-channel hardening).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from pyfe4ai.utils.exceptions import FESecurityWarning

REVIEWED = "reviewed"
EXPERIMENTAL = "experimental"
UNREVIEWED = "unreviewed"
KNOWN_INSECURE = "known-insecure"

FAITHFUL = "faithful"
ADAPTED = "adapted"
AD_HOC = "ad-hoc"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class SchemeStatus:
    """Review status of one registered scheme."""

    module: str
    status: str
    fidelity: str
    reference: str
    note: str = ""
    issue: str = ""


_S = "pyfe4ai.schemes."

_FH_IPE_NOTE = (
    "FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric "
    "pairing (default MNT224), symmetric groups rejected"
)
_FIG14_NOTE = (
    "MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) "
    "are sampled in-process"
)

SCHEME_STATUS: dict[tuple[str, str], SchemeStatus] = {
    # --- single-input -------------------------------------------------------
    ("sife", "ddh"): SchemeStatus(_S + "sife.ddh", UNREVIEWED, UNKNOWN, "ABDP15 (DDH)"),
    ("sife", "lwe"): SchemeStatus(_S + "sife.lwe", UNREVIEWED, UNKNOWN, "ABDP15 (LWE)"),
    ("sife", "paillier"): SchemeStatus(
        _S + "sife.paillier", REVIEWED, FAITHFUL, "ALS16 (DCR)",
        note="D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    ("sife", "damgard"): SchemeStatus(_S + "sife.damgard_ddh", UNREVIEWED, UNKNOWN, "ALS16 (DDH)"),
    ("sife", "ring_lwe"): SchemeStatus(_S + "sife.ring_lwe", UNREVIEWED, UNKNOWN, "BMMS21"),
    ("sife", "fullysec_lwe"): SchemeStatus(
        _S + "sife.fullysec_lwe", UNREVIEWED, UNKNOWN, "ALS16 (LWE)"
    ),
    ("sife", "ddh_dynamic"): SchemeStatus(
        _S + "sife.ddh_dynamic", UNREVIEWED, UNKNOWN, "ABDP15 (DDH), dynamic variant"
    ),
    ("sife", "fh_ipe_pairing"): SchemeStatus(
        _S + "sife.fh_ipe_pairing", UNREVIEWED, UNKNOWN, "KLMMRW18"
    ),
    ("sife", "part_fh_ipe_pairing"): SchemeStatus(
        _S + "sife.part_fh_ipe_pairing", UNREVIEWED, UNKNOWN, "Gay20"
    ),
    # --- multi-input --------------------------------------------------------
    ("mife", "ddh"): SchemeStatus(_S + "mife.ddh", UNREVIEWED, UNKNOWN, "ACFGU18 + ABDP15 (DDH)"),
    ("mife", "lwe"): SchemeStatus(
        _S + "mife.lwe", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ABDP15 (LWE)",
        note="ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure "
        "ABDP15 IPFE (same two-step decryption); slots share A",
    ),
    ("mife", "paillier"): SchemeStatus(
        _S + "mife.paillier", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (DCR, Fig. 9)",
        note="slots share N and g; D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    ("mife", "damgard"): SchemeStatus(
        _S + "mife.damgard_ddh", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (DDH, Fig. 7)",
        note="slots share the group and h (the matrix A of Fig. 7)",
    ),
    ("mife", "ring_lwe"): SchemeStatus(
        _S + "mife.ring_lwe", REVIEWED, ADAPTED, "BMMS21 Sec. 6 MIFE-compiler (ACFGU18)",
        note="slots share a; parameters sized for dimension n * eta",
    ),
    ("mife", "fullysec_lwe"): SchemeStatus(
        _S + "mife.fullysec_lwe", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (LWE, Fig. 8)",
        note="slots share A; parameters derived for dimension n * eta and bound 3X",
    ),
    ("mife", "fh_ipe_pairing"): SchemeStatus(
        _S + "mife.fh_ipe_pairing", UNREVIEWED, UNKNOWN, "KLMMRW18, multi-input"
    ),
    ("mife", "fh_multi_ipe_pairing"): SchemeStatus(
        _S + "mife.fh_multi_ipe_pairing", UNREVIEWED, UNKNOWN, "DOT18"
    ),
    ("mife", "ddh_threshold"): SchemeStatus(
        _S + "mife.ddh_threshold", UNREVIEWED, UNKNOWN, "MIFE-DDH + Shamir (TAPFed setting)"
    ),
    ("mife", "lwe_threshold"): SchemeStatus(
        _S + "mife.lwe_threshold", EXPERIMENTAL, AD_HOC, "MIFE-LWE + Shamir + smudging",
        note="MIFE layer as published; the threshold layer (statistical flooding, "
        "derandomised noise, enlarged q) is our own",
    ),
    ("mife", "ddh_hybrid_alpha"): SchemeStatus(
        _S + "mife.ddh_hybrid_alpha", UNREVIEWED, UNKNOWN, "HybridAlpha (Xu+ AISec'19)"
    ),
    # --- multi-client -------------------------------------------------------
    ("mcfe", "ddh"): SchemeStatus(_S + "mcfe.ddh", REVIEWED, FAITHFUL, "CDGPP18"),
    ("mcfe", "lwe"): SchemeStatus(
        _S + "mcfe.lwe", REVIEWED, ADAPTED, "ABMPSW20 Fig. 4, selective parameters",
        note="signed messages and weights (K = 2nXY + 1, centred decoding); "
        "t is a rounded continuous Gaussian",
    ),
    ("mcfe", "paillier"): SchemeStatus(
        _S + "mcfe.paillier", REVIEWED, FAITHFUL, "ABMPSW20 Fig. 3",
        note="D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    ("mcfe", "damgard"): SchemeStatus(
        _S + "mcfe.damgard_ddh", REVIEWED, FAITHFUL, "ABMPSW20 Fig. 2 (D_k-MDDH)",
        note="k = mddh_k (default 1, where it equals CDGPP18)",
    ),
    ("mcfe", "ring_lwe"): SchemeStatus(
        _S + "mcfe.ring_lwe", REVIEWED, ADAPTED, "BMMS21 Sec. 6 MCFE-compiler (identity-based)",
        note="functional keys are bound to a label; ndarray values are packed into "
        "one ciphertext per chunk, each chunk with its own label pad",
    ),
    ("mcfe", "fullysec_lwe"): SchemeStatus(
        _S + "mcfe.fullysec_lwe", REVIEWED, ADAPTED, "ABMPSW20 Fig. 4, adaptive parameters",
        note="signed messages and weights (K = 2nXY + 1, centred decoding); "
        "t is a rounded continuous Gaussian; m0 grows with n log2(2X + 1)",
    ),
    ("mcfe", "fh_multi_ipe_pairing"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing", REVIEWED, ADAPTED,
        "AGT21b Sec. 6.2 (as recapped in NPS24 Sec. 2)", note=_FH_IPE_NOTE,
    ),
    ("mcfe", "fh_multi_ipe_pairing_threshold"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing_threshold", EXPERIMENTAL, AD_HOC,
        "AGT21b FH-MCFE + Shamir",
        note="FH-MCFE layer as published; the Shamir-shared key exponents and "
        "partial decryptions in G_T are our own; " + _FH_IPE_NOTE,
    ),
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing_decentralized", REVIEWED, ADAPTED,
        "NPS24 Fig. 1 (selective FH-DMCFE)",
        note="tag-f is bound to the whole weight vector and keys are derived "
        "deterministically (one key per client and tag-f); " + _FH_IPE_NOTE,
    ),
    ("mcfe", "ddh_threshold"): SchemeStatus(
        _S + "mcfe.ddh_threshold", REVIEWED, ADAPTED, "CDGPP18 + Shamir (TAPFed setting)",
        note="the aggregated key is Shamir-shared; any t servers interpolate in the "
        "exponent (threshold-ElGamal style)",
    ),
    ("mcfe", "ddh_decentralized"): SchemeStatus(
        _S + "mcfe.ddh_decentralized", REVIEWED, ADAPTED,
        "ABMPSW20 Fig. 14 (MCFE1 = CDGPP18)", note=_FIG14_NOTE,
    ),
    ("mcfe", "lwe_threshold"): SchemeStatus(
        _S + "mcfe.lwe_threshold", EXPERIMENTAL, AD_HOC, "ABMPSW20 Fig. 4 + Shamir + flooding",
        note="MCFE layer as published; the threshold layer and its LWE-based "
        "flooding argument are our own (see the module docstring)",
    ),
    ("mcfe", "lwe_decentralized"): SchemeStatus(
        _S + "mcfe.lwe_decentralized", REVIEWED, ADAPTED,
        "ABMPSW20 Fig. 14 (MCFE1 = Fig. 4)", note=_FIG14_NOTE,
    ),
    ("mcfe", "ring_lwe_threshold"): SchemeStatus(
        _S + "mcfe.ring_lwe_threshold", EXPERIMENTAL, AD_HOC,
        "BMMS21 identity-based MCFE + Shamir + smudging",
        note="MCFE layer as published; the threshold layer (statistical flooding, "
        "derandomised noise, enlarged q) is our own",
    ),
    ("mcfe", "ring_lwe_decentralized"): SchemeStatus(
        _S + "mcfe.ring_lwe_decentralized", REVIEWED, ADAPTED,
        "BMMS21 Sec. 6 identity-based DMCFE (ABKW19 + GDSum)",
        note="GDSum realised as pairwise PRF zero-sum masks (its full specification "
        "is only in the ePrint version); pairwise secrets sampled in-process",
    ),
    ("mcfe", "paillier_decentralized"): SchemeStatus(
        _S + "mcfe.paillier_decentralized", REVIEWED, ADAPTED,
        "ABMPSW20 Fig. 14 (MCFE1 = Fig. 3)", note=_FIG14_NOTE,
    ),
    # --- quadratic ----------------------------------------------------------
    ("quadratic", "quad"): SchemeStatus(_S + "quadratic.quad", UNREVIEWED, UNKNOWN, "Gay20"),
    ("quadratic", "sgp"): SchemeStatus(_S + "quadratic.sgp", UNREVIEWED, UNKNOWN, "DSGPP18"),
    ("quadratic", "multi_input_sgp"): SchemeStatus(
        _S + "quadratic.multi_input_sgp", UNREVIEWED, UNKNOWN, "DSGPP18, multi-input"
    ),
}

_BY_MODULE = {entry.module: key for key, entry in SCHEME_STATUS.items()}


def get_scheme_status(family: str, variant: str) -> SchemeStatus:
    """Return the review status of a registered ``(family, variant)``.

    Raises:
        KeyError: If the scheme is not registered.
    """
    return SCHEME_STATUS[(family, variant)]


def status_for_module(module: str) -> tuple[tuple[str, str], SchemeStatus] | None:
    """Return ``((family, variant), status)`` for a scheme module, if registered."""
    key = _BY_MODULE.get(module)
    return None if key is None else (key, SCHEME_STATUS[key])


def warn_if_known_insecure(module: str, stacklevel: int = 3) -> None:
    """Emit :class:`FESecurityWarning` if *module* implements a known-insecure scheme."""
    found = status_for_module(module)
    if found is None:
        return
    (family, variant), entry = found
    if entry.status == KNOWN_INSECURE:
        warnings.warn(
            "{}/{} is known to be insecure: {}. Use it only for functionality "
            "or performance experiments.".format(family, variant, entry.issue),
            FESecurityWarning,
            stacklevel=stacklevel,
        )


def markdown_table() -> str:
    """Render :data:`SCHEME_STATUS` as the README *Scheme Status* table."""
    lines = [
        "| Scheme | Status | Fidelity | Construction | Notes |",
        "|---|---|---|---|---|",
    ]
    for (family, variant), entry in SCHEME_STATUS.items():
        note = (
            "see [Known Issues](#known-issues--security-notice)"
            if entry.status == KNOWN_INSECURE
            else entry.note
        )
        lines.append(
            "| `{}/{}` | {} | {} | {} | {} |".format(
                family, variant, entry.status, entry.fidelity, entry.reference, note
            )
        )
    return "\n".join(lines)
