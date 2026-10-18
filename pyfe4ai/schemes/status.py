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
    "pairing (default BLS12-381); keys carry a random id, never a digest of y"
)
_FIG14_NOTE = (
    "MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) "
    "are sampled in-process"
)

_R255_NOTE = (
    "ristretto255 (RFC 9496, about 126-bit security: Pollard rho in a group of order 2^252) "
    "instead of a 3072-bit Z_p^* subgroup; {}group arithmetic in Rust (curve25519-dalek) "
    "with the native extension, else in pure Python and not constant-time; new code, "
    "not yet reviewed"
)

#: Review status of every registered ``(family, variant)``: the single source
#: of the README *Scheme Status* table and of the status table of the docs.
#:
#: :meta hide-value:
SCHEME_STATUS: dict[tuple[str, str], SchemeStatus] = {
    # --- single-input -------------------------------------------------------
    ("sife", "ddh"): SchemeStatus(
        _S + "sife.ddh", REVIEWED, FAITHFUL, "ABDP15 Construction 1 (DDH)",
        note="signed x, y; decryption bound dlog_bound (default 10^(precision + 2))",
    ),
    ("sife", "ddh_ristretto255"): SchemeStatus(
        _S + "sife.ddh_ristretto255", UNREVIEWED, UNKNOWN, "ABDP15 Construction 1 (DDH) in ristretto255",
        note=_R255_NOTE.format(""),
    ),
    ("sife", "lwe"): SchemeStatus(
        _S + "sife.lwe", REVIEWED, ADAPTED, "ABDP15 Constructions 2 + 3 (LWE)",
        note="k = 0 (no r-ary decomposition); signed X/Y-bounded inputs with centred "
        "decoding; GoFE parameter recipe; noise is a rounded continuous Gaussian; "
        "A expanded from a public seed (SHAKE-128)",
    ),
    ("sife", "paillier"): SchemeStatus(
        _S + "sife.paillier", REVIEWED, FAITHFUL, "ALS16 (DCR)",
        note="D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    ("sife", "damgard"): SchemeStatus(
        _S + "sife.damgard_ddh", REVIEWED, FAITHFUL, "ALS16 DDH (ACFGU18 Fig. 7, k = 1)",
        note="h = g^a with a discarded",
    ),
    ("sife", "ring_lwe"): SchemeStatus(
        _S + "sife.ring_lwe", REVIEWED, ADAPTED, "BMMS21 Sec. 4 (selective) + Remark 1 batching",
        note="signed messages and weights (p = 2lXY + 2, centred decoding); widths "
        "from the Sec. 4.2 constraints (the paper's Table 1 misses its smoothing "
        "conditions); one ciphertext packs up to ring_n vectors; single NTT prime q; "
        "ring_n defaults to the toy value 32",
    ),
    ("sife", "fullysec_ring_lwe"): SchemeStatus(
        _S + "sife.fullysec_ring_lwe", REVIEWED, ADAPTED, "BMMS21 Sec. 5 (adaptive)",
        note="m >= 5 key polynomials instead of m = 3, which admits no modulus under "
        "the paper's App. C.1 constraints (sigma' >= q^(2/m) from Ring-LHL); signed "
        "inputs (p = 2lXY + 2, leveraging over (2X+1)^(2l)); one vector per ciphertext",
    ),
    ("sife", "fullysec_lwe"): SchemeStatus(
        _S + "sife.fullysec_lwe", REVIEWED, ADAPTED, "ALS16 LWE (ACFGU18 Fig. 8)",
        note="signed inputs (K = 2 eta X Y + 1, centred decoding); GoFE parameter "
        "recipe with m = 4 n log2 q (ALS16 Theorem 2); D_{Z,sigma} realised as a "
        "rounded Gaussian",
    ),
    ("sife", "ddh_dynamic"): SchemeStatus(
        _S + "sife.ddh_dynamic", REVIEWED, ADAPTED, "ABDP15 Construction 1, variable length",
        note="vectors of length <= eta handled as zero padding (prefix keys, truncated "
        "ciphertexts)",
    ),
    ("sife", "fh_ipe_pairing"): SchemeStatus(
        _S + "sife.fh_ipe_pairing", REVIEWED, ADAPTED, "KLMMRW18 Sec. 3",
        note="vectors padded to (x, 1, 0) / (y, 0, 1), so eta = 1 and zero vectors are "
        "safe; keys in G1, ciphertexts in G2; generic bilinear group model; the "
        "encryption key is master-secret equivalent",
    ),
    ("sife", "part_fh_ipe_pairing"): SchemeStatus(
        _S + "sife.part_fh_ipe_pairing", REVIEWED, ADAPTED, "Gay20 Fig. 8 (CPA, k = 1)",
        note="M is a random full-rank public matrix (subspace_dim < eta, default 1), "
        "published with V^T M in the clear; keys leak M^T y, which for small "
        "eta - subspace_dim often determines a bounded y; public Enc needs bounded "
        "t; Enc' is run by the key generator; no QANIZK",
    ),
    # --- multi-input --------------------------------------------------------
    ("mife", "ddh"): SchemeStatus(
        _S + "mife.ddh", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ABDP15 (DDH)",
        note="ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure "
        "ABDP15 IPFE in every slot",
    ),
    ("mife", "ddh_ristretto255"): SchemeStatus(
        _S + "mife.ddh_ristretto255", UNREVIEWED, UNKNOWN, "ACFGU18 Fig. 4 + ABDP15 (DDH) in ristretto255",
        note=_R255_NOTE.format(""),
    ),
    ("mife", "lwe"): SchemeStatus(
        _S + "mife.lwe", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ABDP15 (LWE)",
        note="ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure "
        "ABDP15 IPFE (same two-step decryption), with parameters for dimension n * eta "
        "(footnote 14); slots share A, expanded from a public seed (SHAKE-128)",
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
        note="slots share a; parameters sized for dimension n * eta, with the Sec. 4.2 "
        "widths for bound 3X (ACFGU18 Fig. 4, footnote 12)",
    ),
    ("mife", "fullysec_lwe"): SchemeStatus(
        _S + "mife.fullysec_lwe", REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (LWE, Fig. 8)",
        note="slots share A; parameters derived for dimension n * eta and bound 3X "
        "(ACFGU18 footnote 14), m = 4 n log2 q",
    ),
    ("mife", "fh_ipe_pairing"): SchemeStatus(
        _S + "mife.fh_ipe_pairing", REVIEWED, ADAPTED, "ACFGU18 Fig. 12 (k = 1)",
        note="one r per key shared by all slots; per-slot FH-IPFE as in mcfe/fh_ipe "
        "(LV16 padding of App. B); asymmetric pairing (default BLS12-381); keys carry "
        "a random id, never a digest of y",
    ),
    ("mife", "fh_multi_ipe_pairing"): SchemeStatus(
        _S + "mife.fh_multi_ipe_pairing", REVIEWED, ADAPTED,
        "DOT18 Sec. 3.1 (bounded, k-LIN, k = sec_level)",
        note="n >= 2 enforced (zero sharing r_i); no Remark B.1 SKE layer, so hiding "
        "assumes every slot contributes a ciphertext; slot i's client holds B^_i and "
        "can decode its own slot of a key; keys are group elements only, "
        "decryption does not need y",
    ),
    ("mife", "ddh_threshold"): SchemeStatus(
        _S + "mife.ddh_threshold", REVIEWED, ADAPTED,
        "ACFGU18 Fig. 4 + ABDP15 + Shamir (TAPFed setting)",
        note="the functional key (d_i, z) is Shamir-shared mod q; any t servers "
        "interpolate in the exponent (threshold-ElGamal style), as mcfe/ddh_threshold",
    ),
    ("mife", "lwe_threshold"): SchemeStatus(
        _S + "mife.lwe_threshold", EXPERIMENTAL, AD_HOC, "MIFE-LWE + replicated / Shamir sharing + smudging",
        note="MIFE layer as published; the threshold layer (statistical flooding, "
        "derandomised noise, enlarged q, clients' HMAC tags on ciphertexts so that "
        "servers only decrypt ciphertexts made by the clients, replicated key sharing "
        "for t < s, where Shamir shares let partials reveal the decryption noise "
        "modulo small integers) is our own"
        "; the opt-in sharing='replicated-seeded' expands all key pieces but one from "
        "secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 "
        "keyed by the seeds as a PRF); full replicated sharing stays the default for t < s",
    ),
    # --- multi-client -------------------------------------------------------
    ("mcfe", "ddh"): SchemeStatus(_S + "mcfe.ddh", REVIEWED, FAITHFUL, "CDGPP18"),
    ("mcfe", "ddh_ristretto255"): SchemeStatus(
        _S + "mcfe.ddh_ristretto255", UNREVIEWED, UNKNOWN, "CDGPP18 in ristretto255",
        note=_R255_NOTE.format("labels hashed with RFC 9380 hash_to_ristretto255; "),
    ),
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
        "one ciphertext per chunk, each chunk with its own label pad; parameters as "
        "mife/ring_lwe (widths for bound 3X)",
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
        note="FH-MCFE layer as published; the key exponents Shamir-shared in the "
        "exponent ([e^(k)]_2, [z^(k)]_T), partial decryptions in G_T and labelled, "
        "client-tagged ciphertexts are our own; " + _FH_IPE_NOTE,
    ),
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): SchemeStatus(
        _S + "mcfe.fh_multi_ipe_pairing_decentralized", REVIEWED, ADAPTED,
        "NPS24 Fig. 1 (selective FH-DMCFE)",
        note="explicit tag-f, each client's key depends on (tag-f, y_i) only and is "
        "derived deterministically; one key per client and tag-f (enforced per client "
        "object); " + _FH_IPE_NOTE,
    ),
    ("mcfe", "ddh_threshold"): SchemeStatus(
        _S + "mcfe.ddh_threshold", REVIEWED, ADAPTED, "CDGPP18 + Shamir (TAPFed setting)",
        note="the aggregated key is Shamir-shared; any t servers interpolate in the "
        "exponent (threshold-ElGamal style)",
    ),
    ("mcfe", "ddh_decentralized"): SchemeStatus(
        _S + "mcfe.ddh_decentralized", EXPERIMENTAL, AD_HOC,
        "ABMPSW20 Fig. 14 (MCFE1 = CDGPP18) + carry mask",
        note="Fig. 14 is stated for MCFEs without a discrete log in decryption; "
        "contributions are reduced mod q and masked by q rho_i (rho_i from a per-client "
        "key and H(y)) so that the carry of their sum stays hidden -- our own argument; "
        + _FIG14_NOTE,
    ),
    ("mcfe", "lwe_threshold"): SchemeStatus(
        _S + "mcfe.lwe_threshold", EXPERIMENTAL, AD_HOC, "ABMPSW20 Fig. 4 + replicated / Shamir sharing + flooding",
        note="MCFE layer as published; the threshold layer (replicated key sharing "
        "for t < s, Shamir for t = s) and its LWE-based flooding argument are our "
        "own (see the module docstring)"
        "; the opt-in sharing='replicated-seeded' expands all key pieces but one from "
        "secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 "
        "keyed by the seeds as a PRF); full replicated sharing stays the default for t < s",
    ),
    ("mcfe", "lwe_decentralized"): SchemeStatus(
        _S + "mcfe.lwe_decentralized", REVIEWED, ADAPTED,
        "ABMPSW20 Fig. 14 (MCFE1 = Fig. 4)", note=_FIG14_NOTE,
    ),
    ("mcfe", "ring_lwe_threshold"): SchemeStatus(
        _S + "mcfe.ring_lwe_threshold", EXPERIMENTAL, AD_HOC,
        "BMMS21 identity-based MCFE + replicated / Shamir sharing + smudging",
        note="MCFE layer as published; the threshold layer (statistical flooding, "
        "derandomised noise, enlarged q, clients' HMAC tags on ciphertexts, "
        "replicated key sharing for t < s, where Shamir shares let partials reveal "
        "the decryption noise modulo small integers) is our own"
        "; the opt-in sharing='replicated-seeded' expands all key pieces but one from "
        "secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 "
        "keyed by the seeds as a PRF); full replicated sharing stays the default for t < s",
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
    ("quadratic", "quad"): SchemeStatus(
        _S + "quadratic.quad", REVIEWED, ADAPTED, "Gay20 Fig. 4 over Fig. 8 (CPA)",
        note="signed inputs (discrete log in [-n m B^3, n m B^3]); CPA variant "
        "(no QANIZK); asymmetric pairing (default BLS12-381)",
    ),
    ("quadratic", "sgp"): SchemeStatus(
        _S + "quadratic.sgp", REVIEWED, ADAPTED, "DSGPP18 Fig. 2",
        note="public key pk = (g1^s, g2^t) as in the paper; signed inputs with one "
        "bound B for x, y and F (discrete log in [-n^2 B^3, n^2 B^3]); generic "
        "bilinear group model",
    ),
    ("quadratic", "miqfe"): SchemeStatus(
        _S + "quadratic.miqfe", REVIEWED, ADAPTED, "AGT22 Sec. 5 (MIFE for quadratic functions)",
        note="cross-user terms only (Def. 2.9); per-user encryption keys made of master "
        "ciphertexts; building blocks TAO20 FH-IPFE, Fig. 9 and Fig. 1 as in the paper; "
        "PRF = HMAC-SHA512; pos-security (every user contributes); k = mddh_k >= 2",
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
    """Emit :class:`~pyfe4ai.utils.exceptions.FESecurityWarning` if *module*
    implements a known-insecure scheme."""
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
