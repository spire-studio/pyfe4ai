"""The registry of the schemes: one entry per scheme, the single source of
what the library records about it.

Every registered scheme has an id ``"family/variant"`` -- ``"sife/ddh"``,
``"mcfe/ring_lwe"``, ``"mcfe/ring_lwe_threshold"`` -- and a :class:`Scheme`
entry in :data:`SCHEMES` with

* its module and its classes: the key generator and the crypto class, named
  after the id (:func:`class_name`; imported on first use), and the config
  class of :mod:`pyfe4ai.core.config`;
* its kind (:data:`KINDS`): the family of parameters the security profiles
  size, and the reason the ``standard`` profile refuses it, if it does
  (:func:`pyfe4ai.core.config.resolve` applies both; the minima and presets
  of each kind are in :mod:`pyfe4ai.core.config`);
* its review status (:class:`SchemeStatus`);
* its capabilities (:data:`CAPABILITIES`).

The other tables of the library are derived from it: the status table
(:data:`pyfe4ai.core.status.SCHEME_STATUS`) with the README and
documentation tables rendered from it, the kinds of the security profiles
(:data:`pyfe4ai.core.profiles.SCHEME_KIND`), the schemes of the ML adapter
(:class:`pyfe4ai.ml.FESchemeWrapper`: classes, labels, the servers it runs,
the group variants) and the benchmark scripts. ``tests/api/test_registry.py``
checks the entries against the code.

:func:`create` builds a scheme's key generator by id and returns it with the
scheme's crypto class; :func:`get_scheme` and :func:`list_schemes` query the
registry. All three are exported by :mod:`pyfe4ai`::

    import pyfe4ai

    kg, MCFE = pyfe4ai.create("mcfe/ring_lwe", n=3, bound_x=10, bound_y=10)
    pyfe4ai.list_schemes("mcfe", capabilities="threshold")
    pyfe4ai.get_scheme("mcfe/ring_lwe").capabilities

**Class names.** The classes of a scheme are named after its id, one name
per class: ``Threshold`` or ``Decentralized`` for the variants with that
suffix, the family (``SIFE``, ``MIFE``, ``MCFE``, ``Quadratic``), then the
rest of the variant in CamelCase with its acronyms in capitals; the key
generator adds ``KeyGenerator``, the config class ``Config``:
``sife/ddh`` -- ``SIFEDDH``, ``SIFEDDHKeyGenerator``, ``SIFEDDHConfig``;
``mcfe/ring_lwe_threshold`` -- ``ThresholdMCFERingLWE``;
``mcfe/ddh_decentralized`` -- ``DecentralizedMCFEDDH``;
``mife/fh_multi_ipe_pairing`` -- ``MIFEFHMultiIPEPairing``;
``quadratic/miqfe`` -- ``QuadraticMIQFE``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from pyfe4ai.core.exceptions import FEValidationError

# ---------------------------------------------------------------------------
# review status

REVIEWED = "reviewed"
EXPERIMENTAL = "experimental"
UNREVIEWED = "unreviewed"
KNOWN_INSECURE = "known-insecure"
#: The review status values, best first.
STATUSES = (REVIEWED, EXPERIMENTAL, UNREVIEWED, KNOWN_INSECURE)

FAITHFUL = "faithful"
ADAPTED = "adapted"
AD_HOC = "ad-hoc"
UNKNOWN = "unknown"
#: The fidelity values.
FIDELITIES = (FAITHFUL, ADAPTED, AD_HOC, UNKNOWN)


@dataclass(frozen=True)
class SchemeStatus:
    """Review status of one registered scheme.

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

    ``module`` is the scheme's module, ``reference`` the construction it
    implements. All schemes additionally share the caveats of the README
    (research prototype, toy default parameters, no side-channel hardening).
    """

    module: str
    status: str
    fidelity: str
    reference: str
    note: str = ""
    issue: str = ""


# ---------------------------------------------------------------------------
# kinds, capabilities and refusals

DL = "dl"
EC = "ec"
DCR = "dcr"
LWE = "lwe"
LWE_ABDP15 = "lwe_abdp15"
LWE_MATRIX = "lwe_matrix"
RING_LWE = "ring_lwe"
PAIRING = "pairing"
#: The kinds of scheme the security profiles size parameters for (see
#: :mod:`pyfe4ai.core.profiles`).
KINDS = {
    DL: "discrete log in a safe-prime group (DDH, Damgard / MDDH)",
    EC: "DDH over ristretto255 (a fixed group)",
    DCR: "decisional composite residuosity (Paillier)",
    LWE: "dense LWE without a public matrix (the ABMPSW20 MCFE family)",
    LWE_ABDP15: "LWE with a seeded public matrix (the ABDP15 IPFE)",
    LWE_MATRIX: "LWE with a public matrix, ALS16",
    RING_LWE: "Ring-LWE",
    PAIRING: "asymmetric pairings",
}

LABELS = "labels"
THRESHOLD = "threshold"
DECENTRALIZED = "decentralized"
PAIRING_GROUP = "pairing"
ML_SERVERS = "ml_servers"
BINARY_CT = "binary_ct"
RNS = "rns"
#: The capabilities a scheme may have, with what they mean. ``labels``,
#: ``threshold``, ``decentralized`` and ``pairing`` follow from the id and the
#: kind; the others are declared per scheme.
CAPABILITIES = {
    LABELS: "ciphertexts and functional keys are bound to a label (the MCFE family)",
    THRESHOLD: "functional keys are shared among s servers; t partial decryptions decrypt "
    "(share_decrypt, combine_decrypt)",
    DECENTRALIZED: "the clients derive shares of each functional key, which the decryptor combines "
    "(derive_function_decryption_key_share, combine_function_decryption_key_share)",
    PAIRING_GROUP: "runs over a pairing group (pairing_group_param; BLS12-381 by default)",
    ML_SERVERS: "the ML adapter runs the decryption servers in-process (pyfe4ai.ml: aggregate_gradients, "
    "prepare_aggregation)",
    BINARY_CT: "binary ciphertexts (crypto option ct_format='binary', or the public parameters' format)",
    RNS: "an RNS modulus, a product of NTT primes (key generator option ring_modulus='rns')",
}

_REFUSED_TMIFE_LWE = (
    "read at the width sigma / (1 + 4 X sqrt(eta)) of its security "
    "reduction, the LWE of its flooding-enlarged q (stat_param 128) needs a dimension at the very "
    "end of the 128-bit table (5888 to 6144, a public matrix of 6.5 to 7.8*10^9 entries) for a few "
    "small bounds, e.g. X = Y in {1, 2, 3, 8, 10} at eta = 1, and beyond it for most others "
    "(X = Y = 4 to 6, X = Y >= 12, most eta >= 2; about 6500 for X = Y = 50); use "
    "mife/ddh_threshold, or mcfe/ring_lwe_threshold with labels"
)
_REFUSED_ALS16 = (
    "at a 128-bit dimension (lwe_n = 3328 to 4352 for X = Y <= 50, eta <= 10) its "
    "public matrix has 1 to 2*10^10 entries and every ciphertext m = 4 n log2 q = 2.9 to 4.5*10^6 "
    "coordinates of 219 to 258 bits (80 to 145 MB); use sife/fullysec_ring_lwe for adaptive "
    "security, or the Ring-LWE scheme of the same family"
)
_REFUSED_DMCFE_PAILLIER = (
    "its key shares would be combined under a DCR modulus "
    "of more than 15000 bits; use mcfe/ddh_decentralized"
)

# ---------------------------------------------------------------------------
# class names

#: The family part of the class names.
FAMILIES = {"sife": "SIFE", "mife": "MIFE", "mcfe": "MCFE", "quadratic": "Quadratic"}
#: How each word of a variant is written in a class name.
_WORDS = {
    "ddh": "DDH", "lwe": "LWE", "ring": "Ring", "fullysec": "FullySec", "ristretto255": "Ristretto255",
    "dynamic": "Dynamic", "fh": "FH", "ipe": "IPE", "multi": "Multi", "part": "Part", "pairing": "Pairing",
    "paillier": "Paillier", "damgard": "Damgard", "quad": "Quad", "sgp": "SGP", "miqfe": "MIQFE",
}
_MODIFIERS = ("threshold", "decentralized")


def class_name(scheme: str, role: str = "") -> str:
    """The name of a class of *scheme* (``"family/variant"``): the crypto
    class for ``role=""``, else the name with *role* appended
    (``"KeyGenerator"``, ``"Config"``); see the module docstring.

    Raises:
        KeyError: For a family or a variant word without a spelling.
    """
    family, variant = scheme.split("/")
    words = variant.split("_")
    prefix = words.pop().capitalize() if words[-1] in _MODIFIERS else ""
    return prefix + FAMILIES[family] + "".join(_WORDS[w] for w in words) + role


# ---------------------------------------------------------------------------
# the entries


@dataclass(frozen=True)
class Scheme:
    """A registered scheme (see the module docstring).

    The classes are imported when first read: :attr:`key_generator`,
    :attr:`crypto`, :attr:`config`.
    """

    family: str
    variant: str
    #: The scheme's module.
    module: str
    #: See :data:`KINDS`.
    kind: str
    status: SchemeStatus
    #: See :data:`CAPABILITIES`.
    capabilities: frozenset
    #: Why the ``standard`` profile refuses the scheme; empty if it does not.
    refused: str = ""
    #: The group of a scheme that instantiates another variant of its family
    #: in a group of its own (the ML adapter's ``group`` option), such as
    #: ``ristretto255`` for ``ddh_ristretto255``.
    group: str | None = None
    #: The variant that such a scheme instantiates (``ddh`` for
    #: ``ddh_ristretto255``).
    instance_of: str | None = None

    @property
    def id(self) -> str:
        """``"family/variant"``."""
        return "{}/{}".format(self.family, self.variant)

    @property
    def key(self) -> tuple[str, str]:
        """``(family, variant)``, the key of the derived tables."""
        return self.family, self.variant

    @property
    def crypto_name(self) -> str:
        return class_name(self.id)

    @property
    def key_generator_name(self) -> str:
        return class_name(self.id, "KeyGenerator")

    @property
    def config_name(self) -> str:
        return class_name(self.id, "Config")

    @property
    def key_generator(self) -> type:
        """The key generator class (imports the module)."""
        return getattr(importlib.import_module(self.module), self.key_generator_name)

    @property
    def crypto(self) -> type:
        """The crypto class, whose objects encrypt, decrypt or serve (imports the module)."""
        return getattr(importlib.import_module(self.module), self.crypto_name)

    @property
    def config(self) -> type:
        """The config class (:mod:`pyfe4ai.core.config`)."""
        from pyfe4ai.core import config

        return config.config_class(self.key)

    def has(self, *capabilities: str) -> bool:
        """Whether the scheme has every one of *capabilities*.

        Raises:
            FEValidationError: For a name not in :data:`CAPABILITIES`.
        """
        return set(_capabilities(capabilities)) <= self.capabilities


_S = "pyfe4ai.schemes."


def _scheme(scheme_id: str, kind: str, status: str, fidelity: str, reference: str, *, note: str = "",
            issue: str = "", module: str | None = None, capabilities: Iterable[str] = (), refused: str = "",
            group: str | None = None, instance_of: str | None = None) -> Scheme:
    family, variant = scheme_id.split("/")
    module = _S + (module or "{}.{}".format(family, variant))
    caps = set(capabilities)
    if family == "mcfe":
        caps.add(LABELS)
    if variant.endswith("_threshold"):
        caps.add(THRESHOLD)
    if variant.endswith("_decentralized"):
        caps.add(DECENTRALIZED)
    if kind == PAIRING:
        caps.add(PAIRING_GROUP)
    return Scheme(
        family, variant, module, kind, SchemeStatus(module, status, fidelity, reference, note=note, issue=issue),
        frozenset(caps), refused=refused, group=group, instance_of=instance_of,
    )


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

#: Every registered scheme, in the order of the tables derived from them.
#:
#: :meta hide-value:
SCHEMES: tuple[Scheme, ...] = (
    # --- single-input -------------------------------------------------------
    _scheme(
        "sife/ddh", DL, REVIEWED, FAITHFUL, "ABDP15 Construction 1 (DDH)",
        note="signed x, y; decryption bound dlog_bound (default 10^(precision + 2))",
    ),
    _scheme(
        "sife/ddh_ristretto255", EC, UNREVIEWED, UNKNOWN, "ABDP15 Construction 1 (DDH) in ristretto255",
        note=_R255_NOTE.format(""),
        group="ristretto255", instance_of="ddh",
    ),
    _scheme(
        "sife/lwe", LWE_ABDP15, REVIEWED, ADAPTED, "ABDP15 full version (ePrint 2015/017), LWE",
        note="the LWE instantiation as revised in the full version (no r-ary "
        "decomposition); its selective IND-CPA proof is the full version's Thm. 4.4 "
        "with Thms. 6.6 and 6.7 (Bourse 2017 Thm. 4.3.5 with 8.2.2, 8.2.3, 8.3.2), "
        "not PKC Thm. 2, which does not cover noisy "
        "LWE keys; it reduces to LWE with errors sigma / (1 + 2X sqrt(eta)), the width "
        "standard reads the 128-bit table at; signed X/Y-bounded inputs with centred "
        "decoding; GoFE parameter recipe; pk noise E <- D_{Z,sigma_q} exactly (FACCT) under "
        "standard, the library-wide sampler under toy; A expanded from a public seed (SHAKE-128)",
    ),
    _scheme(
        "sife/paillier", DCR, REVIEWED, FAITHFUL, "ALS16 (DCR)",
        note="D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    _scheme(
        "sife/damgard", DL, REVIEWED, FAITHFUL, "ALS16 DDH (ACFGU18 Fig. 7, k = 1)",
        note="h = g^a with a discarded",
        module="sife.damgard_ddh",
    ),
    _scheme(
        "sife/ring_lwe", RING_LWE, REVIEWED, ADAPTED, "BMMS21 Sec. 4 (selective) + Remark 1 batching",
        note="signed messages and weights (p = 2lXY + 2, centred decoding); widths "
        "from the Sec. 4.2 constraints (the paper's Table 1 misses its smoothing "
        "conditions); one ciphertext packs up to ring_n vectors; single NTT prime q; "
        "ring_n defaults to the toy value 32",
    ),
    _scheme(
        "sife/fullysec_ring_lwe", RING_LWE, REVIEWED, ADAPTED, "BMMS21 Sec. 5 (adaptive)",
        note="m >= 5 key polynomials instead of m = 3, which admits no modulus under "
        "the paper's App. C.1 constraints (sigma' >= q^(2/m) from Ring-LHL); signed "
        "inputs (p = 2lXY + 2, leveraging over (2X+1)^(2l)); one vector per ciphertext. "
        "Experimental: ndarray_layout = \"packed\" (up to ring_n vectors per ciphertext, "
        "leveraging over the span of the differences) rests on this project's own "
        "argument, independently checked, not externally reviewed",
    ),
    _scheme(
        "sife/fullysec_lwe", LWE_MATRIX, REVIEWED, ADAPTED, "ALS16 LWE (ACFGU18 Fig. 8)",
        note="signed inputs (K = 2 eta X Y + 1, centred decoding); GoFE parameter "
        "recipe with m = 4 n log2 q (ALS16 Theorem 2); D_{Z,sigma} realised as a "
        "rounded Gaussian",
        refused=_REFUSED_ALS16,
    ),
    _scheme(
        "sife/ddh_dynamic", DL, REVIEWED, ADAPTED, "ABDP15 Construction 1, variable length",
        note="vectors of length <= eta handled as zero padding (prefix keys, truncated "
        "ciphertexts)",
    ),
    _scheme(
        "sife/fh_ipe_pairing", PAIRING, REVIEWED, ADAPTED, "KLMMRW18 Sec. 3",
        note="vectors padded to (x, 1, 0) / (y, 0, 1), so eta = 1 and zero vectors are "
        "safe; keys in G1, ciphertexts in G2; generic bilinear group model; the "
        "encryption key is master-secret equivalent",
    ),
    _scheme(
        "sife/part_fh_ipe_pairing", PAIRING, REVIEWED, ADAPTED, "Gay20 Fig. 8 (CPA, k = 1)",
        note="M is a random full-rank public matrix (subspace_dim < eta, default 1), "
        "published with V^T M in the clear; keys leak M^T y, which for small "
        "eta - subspace_dim often determines a bounded y; public Enc needs bounded "
        "t; Enc' is run by the key generator; no QANIZK",
    ),
    # --- multi-input --------------------------------------------------------
    _scheme(
        "mife/ddh", DL, REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ABDP15 (DDH)",
        note="ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure "
        "ABDP15 IPFE in every slot",
    ),
    _scheme(
        "mife/ddh_ristretto255", EC, UNREVIEWED, UNKNOWN, "ACFGU18 Fig. 4 + ABDP15 (DDH) in ristretto255",
        note=_R255_NOTE.format(""),
        group="ristretto255", instance_of="ddh",
    ),
    _scheme(
        "mife/lwe", LWE_ABDP15, REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ABDP15 (LWE)",
        note="ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure "
        "ABDP15 IPFE (same two-step decryption; full-version proof, see sife/lwe), with "
        "parameters for dimension n * eta (footnote 14); the slot challenges of the "
        "ACFGU18 reduction differ by up to 4X, so standard reads the 128-bit table at "
        "sigma / (1 + 4X sqrt(eta)); slots share A, expanded from a public seed (SHAKE-128); "
        "pk noise exact (FACCT) under standard",
    ),
    _scheme(
        "mife/paillier", DCR, REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (DCR, Fig. 9)",
        note="slots share N and g; D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    _scheme(
        "mife/damgard", DL, REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (DDH, Fig. 7)",
        note="slots share the group and h (the matrix A of Fig. 7)",
        module="mife.damgard_ddh",
    ),
    _scheme(
        "mife/ring_lwe", RING_LWE, REVIEWED, ADAPTED, "BMMS21 Sec. 6 MIFE-compiler (ACFGU18)",
        note="slots share a; parameters sized for dimension n * eta, with the Sec. 4.2 "
        "widths for bound 3X (ACFGU18 Fig. 4, footnote 12)",
        capabilities=(BINARY_CT, RNS,),
    ),
    _scheme(
        "mife/fullysec_lwe", LWE_MATRIX, REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ALS16 (LWE, Fig. 8)",
        note="slots share A; parameters derived for dimension n * eta and bound 3X "
        "(ACFGU18 footnote 14), m = 4 n log2 q",
        refused=_REFUSED_ALS16,
    ),
    _scheme(
        "mife/fh_ipe_pairing", PAIRING, REVIEWED, ADAPTED, "ACFGU18 Fig. 12 (k = 1)",
        note="one r per key shared by all slots; per-slot FH-IPFE as in mcfe/fh_ipe "
        "(LV16 padding of App. B); asymmetric pairing (default BLS12-381); keys carry "
        "a random id, never a digest of y",
    ),
    _scheme(
        "mife/fh_multi_ipe_pairing", PAIRING, REVIEWED, ADAPTED, "DOT18 Sec. 3.1 (bounded, k-LIN, k = sec_level)",
        note="n >= 2 enforced (zero sharing r_i); no Remark B.1 SKE layer, so hiding "
        "assumes every slot contributes a ciphertext; slot i's client holds B^_i and "
        "can decode its own slot of a key; keys are group elements only, "
        "decryption does not need y",
    ),
    _scheme(
        "mife/ddh_threshold", DL, REVIEWED, ADAPTED, "ACFGU18 Fig. 4 + ABDP15 + Shamir (TAPFed setting)",
        note="the functional key (d_i, z) is Shamir-shared mod q; any t servers "
        "interpolate in the exponent (threshold-ElGamal style), as mcfe/ddh_threshold",
    ),
    _scheme(
        "mife/lwe_threshold", LWE_ABDP15, EXPERIMENTAL, AD_HOC, "MIFE-LWE + replicated / Shamir sharing + smudging",
        note="MIFE layer as published; the threshold layer (statistical flooding, "
        "derandomised noise, enlarged q, clients' HMAC tags on ciphertexts so that "
        "servers only decrypt ciphertexts made by the clients, replicated key sharing "
        "for t < s, where Shamir shares let partials reveal the decryption noise "
        "modulo small integers) is our own"
        "; the opt-in sharing='replicated-seeded' expands all key pieces but one from "
        "secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 "
        "keyed by the seeds as a PRF); full replicated sharing stays the default for t < s"
        "; standard refuses it (at the reduction width of mife/lwe it needs the very end of the 128-bit "
        "table, 5888 to 6144, for a few small bounds and more for most others)",
        refused=_REFUSED_TMIFE_LWE,
    ),
    # --- multi-client -------------------------------------------------------
    _scheme(
        "mcfe/ddh", DL, REVIEWED, FAITHFUL, "CDGPP18",
    ),
    _scheme(
        "mcfe/ddh_ristretto255", EC, UNREVIEWED, UNKNOWN, "CDGPP18 in ristretto255",
        note=_R255_NOTE.format("labels hashed with RFC 9380 hash_to_ristretto255; "),
        group="ristretto255", instance_of="ddh",
    ),
    _scheme(
        "mcfe/lwe", LWE, REVIEWED, ADAPTED, "ABMPSW20 Fig. 4, selective parameters",
        note="signed messages and weights (K = 2nXY + 1, centred decoding); "
        "t <- D_{Z,sigma_t} exactly (FACCT) by default, rounded selectable under toy",
    ),
    _scheme(
        "mcfe/paillier", DCR, REVIEWED, FAITHFUL, "ABMPSW20 Fig. 3",
        note="D_{Z,sigma} realised as a high-precision rounded Gaussian",
    ),
    _scheme(
        "mcfe/damgard", DL, REVIEWED, FAITHFUL, "ABMPSW20 Fig. 2 (D_k-MDDH)",
        note="k = mddh_k (default 1, where it equals CDGPP18)",
        module="mcfe.damgard_ddh",
    ),
    _scheme(
        "mcfe/ring_lwe", RING_LWE, REVIEWED, ADAPTED, "BMMS21 Sec. 6 MCFE-compiler (identity-based)",
        note="functional keys are bound to a label; ndarray values are packed into "
        "one ciphertext per chunk, each chunk with its own label pad; parameters as "
        "mife/ring_lwe (widths for bound 3X)",
        capabilities=(BINARY_CT, RNS,),
    ),
    _scheme(
        "mcfe/fullysec_lwe", LWE, REVIEWED, ADAPTED, "ABMPSW20 Fig. 4, adaptive parameters",
        note="signed messages and weights (K = 2nXY + 1, centred decoding); "
        "t <- D_{Z,sigma_t} exactly (FACCT) by default, rounded selectable under toy; m0 grows with n log2(2X + 1)",
    ),
    _scheme(
        "mcfe/fh_multi_ipe_pairing", PAIRING, REVIEWED, ADAPTED, "AGT21b Sec. 6.2 (as recapped in NPS24 Sec. 2)",
        note=_FH_IPE_NOTE,
    ),
    _scheme(
        "mcfe/fh_multi_ipe_pairing_threshold", PAIRING, EXPERIMENTAL, AD_HOC, "AGT21b FH-MCFE + Shamir",
        note="FH-MCFE layer as published; the key exponents Shamir-shared in the "
        "exponent ([e^(k)]_2, [z^(k)]_T), partial decryptions in G_T and labelled, "
        "client-tagged ciphertexts are our own; " + _FH_IPE_NOTE,
    ),
    _scheme(
        "mcfe/fh_multi_ipe_pairing_decentralized", PAIRING, REVIEWED, ADAPTED, "NPS24 Fig. 1 (selective FH-DMCFE)",
        note="explicit tag-f, each client's key depends on (tag-f, y_i) only and is "
        "derived deterministically; one key per client and tag-f (enforced per client "
        "object); " + _FH_IPE_NOTE,
    ),
    _scheme(
        "mcfe/ddh_threshold", DL, REVIEWED, ADAPTED, "CDGPP18 + Shamir (TAPFed setting)",
        note="the aggregated key is Shamir-shared; any t servers interpolate in the "
        "exponent (threshold-ElGamal style)",
    ),
    _scheme(
        "mcfe/ddh_decentralized", DL, EXPERIMENTAL, AD_HOC, "ABMPSW20 Fig. 14 (MCFE1 = CDGPP18) + carry mask",
        note="Fig. 14 is stated for MCFEs without a discrete log in decryption; "
        "contributions are reduced mod q and masked by q rho_i (rho_i from a per-client "
        "key and H(y)) so that the carry of their sum stays hidden -- our own argument; "
        + _FIG14_NOTE,
    ),
    _scheme(
        "mcfe/lwe_threshold", LWE, EXPERIMENTAL, AD_HOC, "ABMPSW20 Fig. 4 + replicated / Shamir sharing + flooding",
        note="MCFE layer as published; the threshold layer (replicated key sharing "
        "for t < s, Shamir for t = s) and its LWE-based flooding argument are our "
        "own (see the module docstring)"
        "; the opt-in sharing='replicated-seeded' expands all key pieces but one from "
        "secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 "
        "keyed by the seeds as a PRF); full replicated sharing stays the default for t < s",
    ),
    _scheme(
        "mcfe/lwe_decentralized", LWE, REVIEWED, ADAPTED, "ABMPSW20 Fig. 14 (MCFE1 = Fig. 4)",
        note=_FIG14_NOTE,
    ),
    _scheme(
        "mcfe/ring_lwe_threshold", RING_LWE, EXPERIMENTAL, AD_HOC, "BMMS21 identity-based MCFE + replicated / Shamir sharing + smudging",
        note="MCFE layer as published; the threshold layer (statistical flooding, "
        "derandomised noise, enlarged q, clients' HMAC tags on ciphertexts, "
        "replicated key sharing for t < s, where Shamir shares let partials reveal "
        "the decryption noise modulo small integers) is our own"
        "; the opt-in sharing='replicated-seeded' expands all key pieces but one from "
        "secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 "
        "keyed by the seeds as a PRF); full replicated sharing stays the default for t < s"
        "; by default binary ciphertexts (format 2: ct0 cut to the batch length and every value strictly below q, tags over the parameters' digest and the bytes, binary partials and key shares), an RNS modulus (q a product of at least three NTT primes, each above s) and split key shares (sk_y once per weights, the label terms per round); ct_format='json', ring_modulus='prime' and key_shares='full' select the formats of earlier versions",
        capabilities=(ML_SERVERS, BINARY_CT, RNS,),
    ),
    _scheme(
        "mcfe/ring_lwe_decentralized", RING_LWE, REVIEWED, ADAPTED, "BMMS21 Sec. 6 identity-based DMCFE (ABKW19 + GDSum)",
        note="GDSum realised as pairwise PRF zero-sum masks (its full specification "
        "is only in the ePrint version); pairwise secrets sampled in-process",
        capabilities=(BINARY_CT, RNS,),
    ),
    _scheme(
        "mcfe/paillier_decentralized", DCR, REVIEWED, ADAPTED, "ABMPSW20 Fig. 14 (MCFE1 = Fig. 3)",
        note=_FIG14_NOTE,
        refused=_REFUSED_DMCFE_PAILLIER,
    ),
    # --- quadratic ----------------------------------------------------------
    _scheme(
        "quadratic/quad", PAIRING, REVIEWED, ADAPTED, "Gay20 Fig. 4 over Fig. 8 (CPA)",
        note="signed inputs (discrete log in [-n m B^3, n m B^3]); CPA variant "
        "(no QANIZK); asymmetric pairing (default BLS12-381)",
    ),
    _scheme(
        "quadratic/sgp", PAIRING, REVIEWED, ADAPTED, "DSGPP18 Fig. 2",
        note="public key pk = (g1^s, g2^t) as in the paper; signed inputs with one "
        "bound B for x, y and F (discrete log in [-n^2 B^3, n^2 B^3]); generic "
        "bilinear group model",
    ),
    _scheme(
        "quadratic/miqfe", PAIRING, REVIEWED, ADAPTED, "AGT22 Sec. 5 (MIFE for quadratic functions)",
        note="cross-user terms only (Def. 2.9); per-user encryption keys made of master "
        "ciphertexts; building blocks TAO20 FH-IPFE, Fig. 9 and Fig. 1 as in the paper; "
        "PRF = HMAC-SHA512; pos-security (every user contributes); k = mddh_k >= 2",
    ),
)

_BY_KEY = {scheme.key: scheme for scheme in SCHEMES}
_BY_MODULE = {scheme.module: scheme for scheme in SCHEMES}


def _key(scheme) -> tuple:
    if isinstance(scheme, str):
        family, sep, variant = scheme.partition("/")
        return (family, variant) if sep else (scheme,)
    return tuple(scheme) if isinstance(scheme, (tuple, list)) else (scheme,)


def get_scheme(scheme: str | tuple[str, str]) -> Scheme:
    """The registry entry of *scheme*: an id ``"family/variant"`` or a
    ``(family, variant)`` pair.

    Raises:
        FEValidationError: If the scheme is not registered.
    """
    try:
        return _BY_KEY[_key(scheme)]
    except (KeyError, TypeError):
        raise FEValidationError(
            "unknown scheme {!r}; registered: {}".format(scheme, ", ".join(s.id for s in SCHEMES))
        ) from None


def scheme_for_module(module: str) -> Scheme | None:
    """The scheme whose module is *module*, if any."""
    return _BY_MODULE.get(module)


def _capabilities(capabilities) -> tuple:
    names = (capabilities,) if isinstance(capabilities, str) else tuple(capabilities)
    unknown = [c for c in names if c not in CAPABILITIES]
    if unknown:
        raise FEValidationError("unknown capabilities {}; one of {}".format(unknown, ", ".join(CAPABILITIES)))
    return names


def list_schemes(family: str | None = None, *, capabilities: str | Iterable[str] = (), status: str | None = None,
                 kind: str | None = None) -> list[str]:
    """The ids of the registered schemes, in registry order, optionally only
    those of *family*, with all of *capabilities* (:data:`CAPABILITIES`), of
    review *status* (:data:`STATUSES`) or of *kind* (:data:`KINDS`).

    Raises:
        FEValidationError: For an unknown family, capability, status or kind.
    """
    if family is not None and family not in FAMILIES:
        raise FEValidationError("unknown family {!r}; one of {}".format(family, ", ".join(FAMILIES)))
    if status is not None and status not in STATUSES:
        raise FEValidationError("unknown status {!r}; one of {}".format(status, ", ".join(STATUSES)))
    if kind is not None and kind not in KINDS:
        raise FEValidationError("unknown kind {!r}; one of {}".format(kind, ", ".join(KINDS)))
    wanted = set(_capabilities(capabilities))
    return [
        s.id for s in SCHEMES
        if (family is None or s.family == family) and wanted <= s.capabilities
        and (status is None or s.status.status == status) and (kind is None or s.kind == kind)
    ]


class CreatedScheme(NamedTuple):
    """What :func:`create` returns: ``key_generator, crypto = create(...)``."""

    #: The scheme's key generator, built from the config; not set up yet.
    key_generator: Any
    #: The scheme's crypto class, whose objects encrypt (clients, with ``pp``
    #: and ``sk``), decrypt (with ``pp`` and a functional key) or, for the
    #: threshold schemes, decrypt partially and combine.
    crypto: type


def create(scheme: str | tuple[str, str], config: Mapping | None = None, **options) -> CreatedScheme:
    """Build the key generator of *scheme* and return it with the scheme's
    crypto class.

    ``create("mcfe/ring_lwe", config)`` is ``MCFERingLWEKeyGenerator(config)``
    together with ``MCFERingLWE``: the key generator is built exactly as by
    its class (the config is converted to the scheme's config class and
    completed by its security profile, see :mod:`pyfe4ai.core.config`), and
    not set up: call its ``setup()``, then hand out the public parameters
    and keys::

        kg, SIFEDDH = pyfe4ai.create("sife/ddh", eta=3)
        kg.setup()
        pp = kg.get_public_parameters()
        client = SIFEDDH({"keys": {"pp": pp, "sk": kg.get_private_keys()}})

    Args:
        scheme: The scheme's id ``"family/variant"`` (:func:`list_schemes`),
            or ``(family, variant)``.
        config: The key generator's config (a mapping); ``None`` for an
            empty one (the scheme's defaults, ``toy`` profile).
        **options: Config entries, added to *config* (and replacing its
            entries of the same name).

    Returns:
        :class:`CreatedScheme` ``(key_generator, crypto)``.

    Raises:
        FEValidationError: For an unknown scheme, a config that is not a
            mapping, or a config the key generator refuses.
    """
    entry = get_scheme(scheme)
    if options:
        if config is not None and not isinstance(config, Mapping):
            raise FEValidationError(
                "a key generator config must be a mapping, got {}".format(type(config).__name__)
            )
        config = {**(config or {}), **options}
    elif config is None:
        config = {}
    return CreatedScheme(entry.key_generator(config), entry.crypto)
