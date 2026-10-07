"""Review status of every registered scheme.

Each ``(family, variant)`` of the ML-adapter registry has a
:class:`SchemeStatus`:

``status``
    ``"reviewed"``: the implementation was checked against the cited
    construction and no security defect is known. ``"unreviewed"``: not yet
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

_NO_PAD = (
    "no ACFGU18 one-time pad and one key component per client: the "
    "functional-key holder learns every client's <x_i, y_i>"
)
_LABEL_MASK = (
    "the label enters as a public scalar times a fixed per-client secret and "
    "keys / partial decryptions are per client: the functional-key holder "
    "can strip the mask and learn individual inputs"
)
_SMALL_DCR_KEYS = (
    "; secret keys are drawn from |s| <= max(sec_param, bound_y * eta) and "
    "can be brute-forced from public data"
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
        _S + "mife.lwe", KNOWN_INSECURE, AD_HOC, "ACFGU18 + ABDP15 (LWE)", issue=_NO_PAD
    ),
    ("mife", "paillier"): SchemeStatus(
        _S + "mife.paillier", KNOWN_INSECURE, AD_HOC, "ACFGU18 + ALS16 (DCR)",
        issue=_NO_PAD + _SMALL_DCR_KEYS,
    ),
    ("mife", "damgard"): SchemeStatus(
        _S + "mife.damgard_ddh", KNOWN_INSECURE, AD_HOC, "ACFGU18 + ALS16 (DDH)", issue=_NO_PAD
    ),
    ("mife", "ring_lwe"): SchemeStatus(
        _S + "mife.ring_lwe", KNOWN_INSECURE, AD_HOC, "ACFGU18 + BMMS21", issue=_NO_PAD
    ),
    ("mife", "fullysec_lwe"): SchemeStatus(
        _S + "mife.fullysec_lwe", KNOWN_INSECURE, AD_HOC, "ACFGU18 + ALS16 (LWE)",
        issue=_NO_PAD,
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
        _S + "mife.lwe_threshold", KNOWN_INSECURE, AD_HOC, "MIFE-LWE + Shamir",
        issue=_NO_PAD + "; partial decryptions are exact linear functions of the "
        "server's key share (no flooding)",
    ),
    ("mife", "ddh_hybrid_alpha"): SchemeStatus(
        _S + "mife.ddh_hybrid_alpha", UNREVIEWED, UNKNOWN, "HybridAlpha (Xu+ AISec'19)"
    ),
    # --- multi-client -------------------------------------------------------
    ("mcfe", "ddh"): SchemeStatus(_S + "mcfe.ddh", REVIEWED, FAITHFUL, "CDGPP18"),
    ("mcfe", "lwe"): SchemeStatus(
        _S + "mcfe.lwe", REVIEWED, ADAPTED, "ABMPSW20 Fig. 4, selective parameters",
        note="signed messages and weights (K = 2nXY + 1, centred decoding)",
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
        _S + "mcfe.ring_lwe", KNOWN_INSECURE, AD_HOC, "CDGPP18-style + BMMS21",
        issue=_LABEL_MASK,
    ),
    ("mcfe", "fullysec_lwe"): SchemeStatus(
        _S + "mcfe.fullysec_lwe", REVIEWED, ADAPTED, "ABMPSW20 Fig. 4, adaptive parameters",
        note="signed messages and weights (K = 2nXY + 1, centred decoding)",
    ),
    ("mcfe", "fh_multi_ipe_pairing"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing", KNOWN_INSECURE, AD_HOC, "CDGPP18-style + DOT18",
        issue=_LABEL_MASK,
    ),
    ("mcfe", "fh_multi_ipe_pairing_threshold"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing_threshold", KNOWN_INSECURE, AD_HOC,
        "CDGPP18-style + DOT18 + Shamir", issue=_LABEL_MASK,
    ),
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing_decentralized", KNOWN_INSECURE, AD_HOC,
        "CDGPP18-style + DOT18 + ABKW19-style", issue=_LABEL_MASK,
    ),
    ("mcfe", "ddh_threshold"): SchemeStatus(
        _S + "mcfe.ddh_threshold", REVIEWED, ADAPTED, "CDGPP18 + Shamir (TAPFed setting)",
        note="the aggregated key is Shamir-shared; any t servers interpolate in the exponent",
    ),
    ("mcfe", "ddh_decentralized"): SchemeStatus(
        _S + "mcfe.ddh_decentralized", KNOWN_INSECURE, AD_HOC, "ABKW19-style", issue=_LABEL_MASK
    ),
    ("mcfe", "lwe_threshold"): SchemeStatus(
        _S + "mcfe.lwe_threshold", REVIEWED, ADAPTED, "ABMPSW20 Fig. 4 + Shamir + flooding",
        note="the threshold layer and its LWE-based flooding argument are not from a paper",
    ),
    ("mcfe", "lwe_decentralized"): SchemeStatus(
        _S + "mcfe.lwe_decentralized", KNOWN_INSECURE, AD_HOC, "ABKW19-style + LWE",
        issue=_LABEL_MASK,
    ),
    ("mcfe", "ring_lwe_threshold"): SchemeStatus(
        _S + "mcfe.ring_lwe_threshold", KNOWN_INSECURE, AD_HOC, "Ring-LWE MCFE + Shamir",
        issue=_LABEL_MASK + "; partial decryptions are not flooded",
    ),
    ("mcfe", "ring_lwe_decentralized"): SchemeStatus(
        _S + "mcfe.ring_lwe_decentralized", KNOWN_INSECURE, AD_HOC, "ABKW19-style + Ring-LWE",
        issue=_LABEL_MASK,
    ),
    ("mcfe", "paillier_decentralized"): SchemeStatus(
        _S + "mcfe.paillier_decentralized", KNOWN_INSECURE, AD_HOC, "ABKW19-style + ALS16 (DCR)",
        issue=_LABEL_MASK + _SMALL_DCR_KEYS,
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
