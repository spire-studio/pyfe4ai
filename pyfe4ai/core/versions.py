"""
Versions and compatibility: what every version marker means, what a missing
marker means, and the checks that accept, refuse or regenerate an artifact.

The serialized objects of a scheme (public parameters, parameter records,
keys, key shares, ciphertexts, partial decryptions; their layouts are in
:mod:`pyfe4ai.core.formats`) carry markers of the derivation that made them.
Versions <= 0.3.0 wrote fewer markers, so for most of them an absent field has
a meaning -- the derivation of those versions:

=============================  ============================  ===============================================
Marker                         Absent means                  Decided by
=============================  ============================  ===============================================
``label_hash`` (pp, keys,      ``"shake256-wide"``           :func:`pp_label_hash`, :func:`key_label_hash`,
key shares)                    (``LEGACY_LABEL_HASH``)       :func:`check_key_label_hash`,
                                                             :func:`check_encryption_key_label_hash`
``label_hash`` code            code 0 (``"shake256-wide"``)  :func:`ciphertext_label_hash`,
(ciphertexts, layers)                                        :func:`check_ciphertext_label_hashes`,
                                                             :func:`check_lwe_ciphertext_label_hash`,
                                                             :func:`check_lwe_layer_label_hash`
``ct_format`` (threshold       ``"json"``                    :func:`pp_ct_format`
pp)
``sharing`` (pp, key           ``"shamir"``                  :func:`check_served_sharing`
shares)                        (:data:`LEGACY_SHARING`)
``threshold_construction``     none (Shamir only)            :func:`check_served_sharing`,
                                                             :func:`check_tmife_lwe_construction`,
                                                             :func:`check_tmcfe_ring_lwe_construction`
``noise_version``              1 (refused)                   :func:`check_flooding_version`
``flooding``                   ``"gaussian"``                :func:`pp_flooding`, :func:`check_key_flooding`
                               (:data:`LEGACY_FLOODING`)
``sharing_id``                 none (combines only with      :func:`key_share_sharing_id`,
                               partials without one)         :func:`check_partials_sharing`
``part`` (tMCFE-Ring-LWE       a whole key share             :func:`key_share_part`
key shares)
``_state_version``             refused                       :func:`check_state`
(pickled state)
record ids (``construction``,  regenerated                   :func:`record_matches`, :func:`caches_record`
``param_version``, ...)
=============================  ============================  ===============================================

Shamir key shares with ``t < s`` (the sharing of versions <= 0.3.0) are
served only under the ``legacy_shamir`` policy of :func:`check_legacy_shamir`.

The identifiers are those of earlier versions; nothing here changes a digest
or a record. The byte formats' own version bytes (format 1 and 2 of the
Ring-LWE ciphertexts) are in :mod:`pyfe4ai.core.formats.binary`.
"""

from __future__ import annotations

from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FESecurityWarning, FEValidationError
from pyfe4ai.core.exceptions import warn_at_caller
from pyfe4ai.primitives.hashing import LABEL_HASH_CODES
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH
from pyfe4ai.primitives.hashing import check_label_hash
from pyfe4ai.primitives.sampling import FLOODING_SAMPLER

# ---------------------------------------------------------------------------
# parameter records: the ids that tell a record's derivation


#: Record id of SIFE LWE (ABDP15 with a seeded ``A``; record field ``construction``).
SIFE_LWE_RECORD = "abdp15-lwe-seeded-a/v1"
#: Record id of MIFE LWE (record field ``construction``).
MIFE_LWE_RECORD = "acfgu18-abdp15-lwe-seeded-a/v2"
#: Record id of SIFE FullySec LWE (``K = 2 eta X Y + 1``, encoding ``floor(q / K) x``,
#: ``m = 4 n log2 q``, ``B_tau ~ sqrt(m)``; record field ``construction``).
SIFE_FULLYSEC_LWE_RECORD = "als16-lwe/v5"
#: Record id of MIFE FullySec LWE (``m = 4 n log2 q``, ``B_tau ~ sqrt(m)``; record field ``construction``).
MIFE_FULLYSEC_LWE_RECORD = "acfgu18-als16-lwe/v4"
#: Record id of SIFE Ring-LWE (``p = 2 l X Y + 2``, Sec. 4.2 widths, seeded ``a``; record field ``param_version``).
SIFE_RING_LWE_RECORD = "bmms21-sec4.2/v3"
#: Record id of SIFE FullySec Ring-LWE, element-wise layout (``a`` expanded from a stored seed; ``param_version``).
SIFE_FULLYSEC_RING_LWE_RECORD = "bmms21-sec5/v2"
#: Record id of SIFE FullySec Ring-LWE, packed layout (leveraging over the span of the
#: differences; v2: ``eta = 1`` with the basis ``[1]``; ``param_version``).
SIFE_FULLYSEC_RING_LWE_PACKED_RECORD = "bmms21-sec5-packed/v2"
#: Record id of the MIFE / MCFE / dMCFE / tMCFE Ring-LWE family (sized for ``n eta``, ``p = 2T + 2``,
#: Sec. 4.2 widths for ``3X``, seeded ``a``; record field ``construction``).
RING_LWE_RECORD = "acfgu18-bmms21/v5"
#: Record format of the SIFE and MIFE Paillier records (versions <= 0.2.0 also stored ``p`` and ``q``;
#: record field ``param_format``).
PAILLIER_RECORD = "modulus-only/v1+g-seed"
#: Record id of MCFE Paillier (v1 records stored ``p`` and ``q``; record field ``construction``).
MCFE_PAILLIER_RECORD = "abmpsw20-dcr/v2"
#: Record id of MCFE Damgard (MDDH; record field ``construction``).
MCFE_DAMGARD_RECORD = "abmpsw20-mddh/v1"
#: Record id of the threshold MCFE over DDH (record field ``construction``).
TMCFE_DDH_RECORD = "cdgpp18-shamir/v1"
#: Record id of the decentralised MCFE over DDH (record field ``construction``).
DMCFE_DDH_RECORD = "abmpsw20-fig14-cdgpp18-dcr-carry-mask/v2"
#: Record id of the decentralised MCFE over LWE (record field ``construction``).
DMCFE_LWE_RECORD = "abmpsw20-fig14-lwe-dcr/v2"
#: Record id of the decentralised MCFE over DCR (record field ``construction``).
DMCFE_PAILLIER_RECORD = "abmpsw20-fig14-dcr-dcr/v1"

#: Public-parameter ``construction`` of MIFE DDH (published, not checked).
MIFE_DDH_CONSTRUCTION = "acfgu18-abdp15/v2"
#: Public-parameter ``construction`` of MIFE DDH over ristretto255 (published, not checked).
MIFE_DDH_RISTRETTO255_CONSTRUCTION = "acfgu18-abdp15-ristretto255/v1"

#: The meaning of an absent field of a parameter record (a record of an
#: earlier version) -- the prime modulus of the Ring-LWE family, the
#: element-wise layout of SIFE FullySec Ring-LWE.
RECORD_ABSENT = {"ring_modulus": "prime", "ndarray_layout": "elementwise"}


def record_matches(param: dict, expected: dict) -> bool:
    """Whether the parameter record *param* has the *expected* value of every
    field (in order, stopping at the first difference; an absent field is
    :data:`RECORD_ABSENT`'s value, else ``None``). A record that does not
    match is regenerated by the parameter cache
    (:class:`~pyfe4ai.core.base.ParameterCacheMixin`)."""
    return all(param.get(field, RECORD_ABSENT.get(field)) == value for field, value in expected.items())


def caches_record(key_generator) -> bool:
    """Whether a key generator keeps its parameter record in the cache:
    not one whose parameters include a DCR modulus under the ``standard``
    profile (no check can tell whether someone knows its factors, so such
    parameters are generated afresh, never loaded or stored)."""
    from pyfe4ai.core.profiles import STANDARD

    return not (key_generator._holds_dcr_modulus and getattr(key_generator, "security_profile", None) == STANDARD)


# ---------------------------------------------------------------------------
# threshold constructions: recorded in public parameters (and parameter
# records) as ``threshold_construction``; ``{construction id: sharing}`` is
# what servers serve (check_served_sharing)

#: The meaning of an absent ``sharing`` (pp and key shares) -- the only sharing of versions <= 0.3.0.
LEGACY_SHARING = "shamir"

#: The construction id of tMIFE-LWE (:mod:`pyfe4ai.schemes.mife.lwe_threshold`) with the Shamir sharing.
TMIFE_LWE_CONSTRUCTION = "acfgu18-abdp15-shamir-smudging-seeded-a/v4"  # v4: flooding at bits + 320, sampler pinned
#: The construction id of tMIFE-LWE with the Shamir sharing and the KL accounting of the flooding.
TMIFE_LWE_CONSTRUCTION_KL = "acfgu18-abdp15-shamir-smudging-kl/v2"
# the parameters are those of the Shamir record; its own record keeps cached
# Shamir records (t < s) of versions <= 0.3.0 from being taken for it
#: The construction id of tMIFE-LWE with the replicated sharing.
TMIFE_LWE_CONSTRUCTION_REPLICATED = "acfgu18-abdp15-replicated-smudging-seeded-a/v1"
#: The construction id of tMIFE-LWE with the seed-expanded replicated sharing.
TMIFE_LWE_CONSTRUCTION_REPLICATED_SEEDED = "acfgu18-abdp15-replicated-seeded-smudging-seeded-a/v1"
#: The sharing of each tMIFE-LWE construction id.
TMIFE_LWE_CONSTRUCTIONS = {
    TMIFE_LWE_CONSTRUCTION: "shamir", TMIFE_LWE_CONSTRUCTION_KL: "shamir",
    TMIFE_LWE_CONSTRUCTION_REPLICATED: "replicated", TMIFE_LWE_CONSTRUCTION_REPLICATED_SEEDED: "replicated-seeded",
}

# recorded in the public parameters (those of versions <= 0.3.0 have none)
#: The construction id of tMCFE-LWE (:mod:`pyfe4ai.schemes.mcfe.lwe_threshold`) with the Shamir sharing.
TMCFE_LWE_CONSTRUCTION = "abmpsw20-shamir-flooding/v1"
#: The construction id of tMCFE-LWE with the replicated sharing.
TMCFE_LWE_CONSTRUCTION_REPLICATED = "abmpsw20-replicated-flooding/v1"
#: The construction id of tMCFE-LWE with the seed-expanded replicated sharing.
TMCFE_LWE_CONSTRUCTION_REPLICATED_SEEDED = "abmpsw20-replicated-seeded-flooding/v1"
#: The sharing of each tMCFE-LWE construction id.
TMCFE_LWE_CONSTRUCTIONS = {TMCFE_LWE_CONSTRUCTION: "shamir", TMCFE_LWE_CONSTRUCTION_REPLICATED: "replicated",
                           TMCFE_LWE_CONSTRUCTION_REPLICATED_SEEDED: "replicated-seeded"}

#: The parameter record (and public ``threshold_construction``) of a sharing
#: and a flooding of tMCFE-Ring-LWE (:mod:`pyfe4ai.schemes.mcfe.ring_lwe_threshold`)
#: under the per-sample accounting (``"kl"``: :data:`TMCFE_RING_LWE_CONSTRUCTION_KL`).
TMCFE_RING_LWE_CONSTRUCTION_IDS = {
    ("shamir", "gaussian"): "bmms21-ibmcfe-shamir-smudging/v2",  # v2: flooding at bits + 320, sampler pinned
    ("shamir", "uniform"): "bmms21-ibmcfe-shamir-smudging-uniform/v1",
    # the parameters are those of the Shamir records; their own records keep cached
    # Shamir records (t < s) of versions <= 0.3.0 from being taken for them
    ("replicated", "gaussian"): "bmms21-ibmcfe-replicated-smudging/v1",
    ("replicated", "uniform"): "bmms21-ibmcfe-replicated-smudging-uniform/v1",
    ("replicated-seeded", "gaussian"): "bmms21-ibmcfe-replicated-seeded-smudging/v1",
    ("replicated-seeded", "uniform"): "bmms21-ibmcfe-replicated-seeded-smudging-uniform/v1",
}
#: The construction id of tMCFE-Ring-LWE with the Shamir sharing, Gaussian flooding and KL accounting.
TMCFE_RING_LWE_CONSTRUCTION_KL = "bmms21-ibmcfe-shamir-smudging-kl/v2"
#: The sharing of each tMCFE-Ring-LWE construction id.
TMCFE_RING_LWE_CONSTRUCTIONS = {
    construction: sharing for (sharing, _), construction in TMCFE_RING_LWE_CONSTRUCTION_IDS.items()
}
TMCFE_RING_LWE_CONSTRUCTIONS[TMCFE_RING_LWE_CONSTRUCTION_KL] = "shamir"


def tmcfe_ring_lwe_construction_id(sharing: str, flooding: str, accounting: str) -> str | None:
    """The tMCFE-Ring-LWE construction id of a sharing, a flooding and a
    flooding accounting (``None`` for a combination the key generator does
    not make)."""
    if accounting == "kl":
        return TMCFE_RING_LWE_CONSTRUCTION_KL if (sharing, flooding) == ("shamir", "gaussian") else None
    return TMCFE_RING_LWE_CONSTRUCTION_IDS.get((sharing, flooding))


def check_served_sharing(pp: dict, dk: dict, constructions: dict) -> None:
    """Raise unless key share *dk* is of the sharing that *pp* describes.

    ``pp["sharing"]`` must match the key share (both default to
    ``"shamir"``, the only sharing of versions <= 0.3.0). Public parameters
    of this version record ``threshold_construction``; it must be one of
    *constructions* (``{construction id: sharing}``) and name that sharing,
    so that artifacts of another construction (a later version, say) are
    refused rather than misread. Only Shamir public parameters may lack it
    (those of versions <= 0.3.0); every version with the replicated sharing
    records it, so replicated public parameters without it are refused.
    """
    sharing = dk.get("sharing", LEGACY_SHARING)
    if pp.get("sharing", LEGACY_SHARING) != sharing:
        raise FESchemeError("the key share is not of the sharing of the public parameters")
    construction = pp.get("threshold_construction")
    if construction is None and sharing != "shamir":
        raise FESchemeError(
            "public parameters of the {} sharing must record threshold_construction".format(sharing)
        )
    if construction is not None and constructions.get(construction) != sharing:
        raise FESchemeError(
            "unknown threshold construction {!r} for a {} key share; known: {}".format(
                construction, sharing, sorted(constructions)
            )
        )


def check_tmife_lwe_construction(pp: dict, kl: bool) -> None:
    """Raise unless tMIFE-LWE public parameters that record a construction id
    record that of their flooding accounting (*kl*: ``"kl"``).

    Raises:
        FESchemeError: For the id of the other accounting.
    """
    construction = pp.get("threshold_construction")
    if construction is not None and (construction == TMIFE_LWE_CONSTRUCTION_KL) != kl:
        raise FESchemeError("the public parameters' threshold_construction is not that of their accounting")


def check_tmcfe_ring_lwe_construction(pp: dict, sharing: str, flooding: str, accounting: str) -> None:
    """Raise unless tMCFE-Ring-LWE public parameters that record a construction
    id record that of the key share's *sharing*, their *flooding* and
    *accounting* (:func:`tmcfe_ring_lwe_construction_id`).

    Raises:
        FESchemeError: Otherwise.
    """
    construction = pp.get("threshold_construction")
    if construction is not None and construction != tmcfe_ring_lwe_construction_id(sharing, flooding, accounting):
        raise FESchemeError("the public parameters' threshold_construction is not that of their flooding")


#: What a server does with a Shamir key share whose ``t < s`` (crypto config
#: entry ``legacy_shamir``): ``"refuse"`` (default) or ``"warn"`` (serve it
#: with an :class:`~pyfe4ai.core.exceptions.FESecurityWarning`). Key shares
#: of versions <= 0.3.0 are refused before this applies (their flooding
#: derivation, see :func:`check_flooding_version`), so ``"warn"`` serves the
#: explicit Shamir ``t < s`` shares of this version under ``toy``. There is
#: deliberately no environment variable: the opt-out is per server object.
LEGACY_SHAMIR_POLICIES = ("refuse", "warn")


def legacy_shamir_policy(config: dict) -> str:
    """The ``legacy_shamir`` entry of a server's crypto *config* (default ``"refuse"``)."""
    policy = config.get("legacy_shamir")
    if policy is None:
        return "refuse"
    if policy not in LEGACY_SHAMIR_POLICIES:
        raise FEValidationError(
            "legacy_shamir must be one of {}, not {!r}".format(LEGACY_SHAMIR_POLICIES, policy)
        )
    return policy


def check_legacy_shamir(sid: str, threshold: int, s: int, legacy_shamir: str, profile) -> None:
    """Serve server *sid*'s Shamir key share with ``t < s`` only under the
    ``legacy_shamir = "warn"`` policy and not under the ``standard`` profile
    of the public parameters (*profile*; ``None`` for those of versions
    <= 0.3.0), with a warning; ``t = s`` is served.

    Raises:
        FESchemeError: For such a key share that is not to be served.

    Warns:
        FESecurityWarning: When it is served; attributed to the application's
            call (the first frame outside pyfe4ai,
            :func:`~pyfe4ai.core.exceptions.warn_at_caller`).
    """
    if threshold >= s:
        return
    from pyfe4ai.core.profiles import STANDARD

    message = (
        "server {}: a Shamir key share with t = {} < s = {} (the sharing of versions "
        "<= 0.3.0) lets partial decryptions reveal the decryption noise modulo small "
        "integers; re-share the threshold keys with this version (sharing="
        "'replicated')".format(sid, threshold, s)
    )
    if legacy_shamir != "warn" or profile == STANDARD:
        raise FESchemeError(
            message + "; during a migration only, a server created with "
            "legacy_shamir='warn' serves it (not under the standard profile)"
        )
    warn_at_caller(message, FESecurityWarning)


# ---------------------------------------------------------------------------
# sharing ids: every sharing a key generator issues has one


#: The message of :class:`MixedSharingsError`.
MIXED_SHARINGS = ("partials come from different sharings of the label's key; fetch the label parts again from "
                  "the key generator for all servers")


class MixedSharingsError(FESchemeError):
    """Partial decryptions of different sharings of one key were combined.

    Every sharing a key generator issues (whole key shares, or the label parts
    of one label) has a random ``sharing_id``, which the servers' partial
    decryptions carry. Shares of two sharings of the same key do not
    reconstruct it: combined, their partials would decode to a wrong value
    inside the decoding bound rather than fail. This happens when the label
    parts of one round are fetched on both sides of their eviction
    (:data:`~pyfe4ai.schemes.threshold.store.LABEL_PARTS_KEPT`); fetch them again for all servers. The check
    needs the ids: keep the ``sharing_id`` with every stored or transmitted
    key share and partial decryption (partials without one combine only with
    each other).
    """


def key_share_sharing_id(dk: dict) -> str | None:
    """The ``sharing_id`` of a key share (``None`` for key shares of earlier
    versions, which have none), which its partial decryptions carry.

    Raises:
        FEKeyError: For a ``sharing_id`` that is not a string.
    """
    sharing_id = dk.get("sharing_id")  # of the shares' sharing (key shares of earlier versions: none)
    if sharing_id is not None and not isinstance(sharing_id, str):
        raise FEKeyError("malformed key share: sharing_id is a string")
    return sharing_id


def check_partials_sharing(partials: list) -> None:
    """Raise unless the partial decryptions are of one sharing: shares of two
    sharings of one key do not reconstruct it (their partials would decode to
    a wrong value). Partials without a ``sharing_id`` combine only with each
    other.

    Raises:
        FESchemeError: For a ``sharing_id`` that is not a string.
        MixedSharingsError: For partials of different sharings.
    """
    sharing_ids = [p.get("sharing_id") for p in partials]
    if any(i is not None and not isinstance(i, str) for i in sharing_ids):
        raise FESchemeError("malformed partial decryption: sharing_id is a string")
    if len(set(sharing_ids)) > 1:
        raise MixedSharingsError(MIXED_SHARINGS)


def key_share_part(dk) -> str | None:
    """The ``part`` of a tMCFE-Ring-LWE key share: ``"weights"``, ``"label"``,
    or ``None`` for a whole key share (versions <= 0.3.0 issued only those;
    ``key_part="full"`` still does)."""
    return dk.get("part")


# ---------------------------------------------------------------------------
# flooding noise: the derivation recorded in the key shares, the distribution
# in the public parameters

#: Version of the flooding-noise derivation in the decryption keys of the
#: threshold schemes (tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE; ``noise_version``):
#: 2 since :data:`~pyfe4ai.primitives.sampling.BIGINT_GUARD_BITS` grew (keys
#: without the field are version 1).
FLOODING_NOISE_VERSION = 2


def check_flooding_version(dk) -> None:
    """Refuse a threshold decryption key of another flooding derivation: it may
    have answered a request with the noise of the earlier derivation, and a
    second noise for the same request would let the answers be averaged.

    Raises:
        FEKeyError: For a key of another version (issue new keys).
    """
    version = dk.get("noise_version", 1) if isinstance(dk, dict) else None
    if version != FLOODING_NOISE_VERSION:
        raise FEKeyError(
            "decryption key of flooding derivation {} (this version derives noise version {}): "
            "issue new decryption keys".format(version, FLOODING_NOISE_VERSION)
        )
    if dk.get("noise_sampler") != FLOODING_SAMPLER:
        raise FEKeyError(
            "decryption key for the flooding sampler {!r} (this version floods with {!r}): "
            "issue new decryption keys".format(dk.get("noise_sampler"), FLOODING_SAMPLER)
        )


#: The flooding of versions <= 0.3.0 and of public parameters (and key shares) without ``flooding``.
LEGACY_FLOODING = "gaussian"


def pp_flooding(pp: dict) -> str:
    """The ``flooding`` of tMCFE-Ring-LWE public parameters (absent: :data:`LEGACY_FLOODING`)."""
    return pp.get("flooding", LEGACY_FLOODING)


def check_key_flooding(pp_value: str, dk: dict, floodings: tuple) -> None:
    """Raise unless the public parameters' flooding *pp_value* is one of
    *floodings* and key share *dk* was issued for it (absent in either:
    :data:`LEGACY_FLOODING`).

    Raises:
        FEValidationError: For an unknown flooding of the public parameters.
        FEKeyError: For a key share of another flooding.
    """
    if pp_value not in floodings:
        raise FEValidationError("unknown flooding {!r}".format(pp_value))
    if dk.get("flooding", LEGACY_FLOODING) != pp_value:
        raise FEKeyError("the key share was issued for another flooding than the public parameters'")


# ---------------------------------------------------------------------------
# label hash: a public parameter of the MCFE schemes, bound to everything that
# depends on it


def label_hash_mode(value) -> str:
    """*value* if it is one of :data:`~pyfe4ai.primitives.hashing.LABEL_HASHES`.

    Raises:
        FEValidationError: Otherwise.
    """
    try:
        return check_label_hash(value)
    except ValueError as exc:
        raise FEValidationError(str(exc)) from exc


def pp_label_hash(pp: dict) -> str:
    """The ``label_hash`` of public parameters (absent: ``"shake256-wide"``,
    the hash of versions <= 0.3.0).

    Raises:
        FEValidationError: If it is not one of
            :data:`~pyfe4ai.primitives.hashing.LABEL_HASHES`.
    """
    return label_hash_mode(pp.get("label_hash", LEGACY_LABEL_HASH))


def key_label_hash(key: dict) -> str:
    """The ``label_hash`` a key, key share or encryption key was issued for
    (absent: ``"shake256-wide"``)."""
    return key.get("label_hash", LEGACY_LABEL_HASH)


def check_key_label_hash(key: dict, mode: str, subject: str = "the key was") -> None:
    """Raise unless *key* (a functional key or key share) was derived for the
    public parameters' label hash *mode* (:func:`key_label_hash`).

    Raises:
        FESchemeError: ``"<subject> derived with another label hash than the
            public parameters"``.
    """
    if key_label_hash(key) != mode:
        raise FESchemeError("{} derived with another label hash than the public parameters".format(subject))


def check_encryption_key_label_hash(sk: dict, mode: str) -> str:
    """*mode* if the encryption key *sk* was issued for that label hash (keys
    without ``label_hash``: ``"shake256-wide"``).

    Raises:
        FEKeyError: If they differ (e.g. public parameters of another version
            than the key): the ciphertexts would decrypt to wrong values.
    """
    key_mode = key_label_hash(sk)
    if key_mode != mode:
        raise FEKeyError(
            "the encryption key was issued for label hash {!r}, the public parameters say {!r}".format(key_mode, mode)
        )
    return mode


def label_hash_code(mode: str) -> int | None:
    """The code a ciphertext of label hash *mode* carries (:data:`~pyfe4ai.primitives.hashing.LABEL_HASH_CODES`),
    ``None`` for ``"shake256-wide"``: its ciphertexts carry none, as in
    versions <= 0.3.0."""
    return None if mode == LEGACY_LABEL_HASH else LABEL_HASH_CODES[mode]


def ciphertext_label_hash(ct) -> str | None:
    """The label hash a ciphertext or ndarray layer dict was produced with (no
    ``label_hash`` field: ``"shake256-wide"``), or ``None`` for an unknown code."""
    code = ct.get("label_hash", 0) if isinstance(ct, dict) else 0
    if type(code) is not int or not 0 <= code < len(LABEL_HASH_CODES):
        return None
    return next(mode for mode, c in LABEL_HASH_CODES.items() if c == code)


def check_ciphertext_label_hashes(cts, mode: str, who: str = "a client") -> None:
    """Raise unless every ciphertext dict of *cts* was produced with *mode*.

    Raises:
        FESchemeError: For a ciphertext of another (or an unknown) label hash.
    """
    for ct in cts:
        if ciphertext_label_hash(ct) != mode:
            raise FESchemeError(
                "a ciphertext of {} was produced with another label hash than the public parameters' {!r}".format(
                    who, mode)
            )


def check_lwe_ciphertext_label_hash(ct: dict, nid, mode: str, absent: int) -> None:
    """Raise unless a dense MCFE-LWE ciphertext carries the code of *mode*
    (*absent*: the code of a ciphertext without one -- 0, the hash of
    versions <= 0.3.0, for a single ciphertext; its layer's for an element of
    an ndarray layer, whose first element carries the code).

    Raises:
        FESchemeError: For a ciphertext of another label hash.
    """
    if ct.get("label_hash", absent) != LABEL_HASH_CODES[mode]:  # the hash this ciphertext was made with
        raise FESchemeError("the ciphertext of {} was produced with another label hash than the public "
                            "parameters' {!r}".format(nid, mode))


def check_lwe_layer_label_hash(first, nid, mode: str) -> None:
    """Raise unless the first element *first* of a dense MCFE-LWE ndarray
    layer carries the code of *mode* (none: ``"shake256-wide"``).

    Raises:
        FESchemeError: For a layer of another (or an unknown) label hash.
    """
    got = first.get("label_hash", 0) if isinstance(first, dict) else 0
    if got != LABEL_HASH_CODES[mode]:
        raise FESchemeError("a ciphertext layer of {} was produced with another label hash than the "
                            "public parameters' {!r}".format(nid, mode))


# ---------------------------------------------------------------------------
# ciphertext formats of the Ring-LWE layers

#: The ciphertext formats of the Ring-LWE schemes, ``"json"`` (decimal
#: strings) and ``"binary"``. For the MIFE / MCFE crypto objects (config
#: ``ct_format``, default ``"json"``) ``"binary"`` is format 1
#: (:func:`~pyfe4ai.core.formats.binary.encode_binary_ciphertext`); for the
#: threshold MCFE over Ring-LWE it is a public parameter (key generator config
#: ``ct_format``, default ``"binary"``; :func:`pp_ct_format`): format 2
#: (:func:`~pyfe4ai.core.formats.binary.encode_threshold_ciphertext`), tags
#: over its digests, binary partial decryptions. ``"json"`` is the only format
#: of versions <= 0.3.0 and that of threshold public parameters without
#: ``ct_format``.
CT_FORMATS = ("json", "binary")


def pp_ct_format(pp: dict) -> str:
    """The ciphertext format of threshold public parameters (absent: ``"json"``).

    Raises:
        FEValidationError: For another value.
    """
    fmt = pp.get("ct_format", "json")
    if fmt not in CT_FORMATS:
        raise FEValidationError("unknown ciphertext format {!r}; one of {}".format(fmt, ", ".join(CT_FORMATS)))
    return fmt


# ---------------------------------------------------------------------------
# pickled state

#: The version of the pickled state of the tMCFE-Ring-LWE objects (a state
#: pickled by another version of pyfe4ai, with other attributes or other
#: derivations, is refused when loaded).
TMCFE_RING_LWE_STATE_VERSION = "pyfe4ai/tmcfe-ring-lwe/state/2026-10-3"


def versioned_state(state: dict, version: str) -> dict:
    """*state* with its ``_state_version``."""
    return dict(state, _state_version=version)


def check_state(obj, state: dict, version: str) -> None:
    """Load *state* into *obj*, refusing one pickled by another version.

    Raises:
        FESchemeError: For a state of another version (or of none).
    """
    if not isinstance(state, dict) or state.get("_state_version") != version:
        raise FESchemeError(
            "this {} was pickled by an incompatible version of pyfe4ai; create the object again".format(
                type(obj).__name__)
        )
    obj.__dict__.update(state)
