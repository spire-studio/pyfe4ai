"""
What the threshold artifacts are bound to, and the checks.

* Key shares to public parameters: :func:`pp_binding` over a scheme's
  binding fields, checked by :func:`check_pp_binding` before every partial
  decryption; the sharing and the threshold construction
  (:func:`check_served_sharing`) against the construction ids each scheme
  records; key shares to the weights they were derived for
  (:func:`fusion_weight_id`).
* Ciphertexts to the servers that decrypt them: HMAC-SHA256 tags under
  client-server keys (:func:`client_server_mac_keys`,
  :func:`tag_ciphertext`, :func:`check_ciphertext_tags`; the tag messages
  are in :mod:`pyfe4ai.schemes.threshold.codec`).
* The threshold MCFE over Ring-LWE: the ciphertext format of its public
  parameters (:func:`pp_ct_format`), the fields its weight and label key
  share parts share (:func:`check_split_parts`), the ``sharing_id`` of every
  sharing it issues (combiners refuse partial decryptions of different
  sharings: :class:`MixedSharingsError`) and the version of the pickled state
  of its objects (:func:`check_state`).

Construction ids, binding fields and state versions are those of earlier
versions; nothing here changes a digest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets

from pyfe4ai._validation import int_vector
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH
from pyfe4ai.schemes.threshold.kl import KL

#: Domain of :func:`pp_binding`.
_PP_BINDING_DOMAIN = b"pyfe4ai/threshold/key-share-pp-binding/v1|"


def pp_binding(pp: dict, fields: tuple, defaults: dict | None = None) -> str:
    """Hex SHA-256 of the public-parameter *fields* a threshold key share is
    bound to (canonical JSON of ``[[field, value], ...]``; a missing field takes
    its value in *defaults* -- what its absence means, e.g. the label hash of
    versions <= 0.3.0 -- else ``null``).

    The key generator writes it into every key share (``pp_binding``) and a
    server compares it with that of its own public parameters
    (:func:`check_pp_binding`) before computing a partial decryption: one key
    share answering a request under public parameters that differ in the
    modulus, the flooding (distribution, width, tail cut, accounting and its
    limit) or the label hash would give a second, differently derived answer
    to the same request, and the difference of the two reveals key material
    (with the label hash: ``lambda <share, H_A(l) - H_B(l)>`` exactly). The
    sharing and the threshold construction are checked on their own
    (:func:`check_served_sharing`).
    """
    defaults = defaults or {}
    payload = json.dumps([[f, pp.get(f, defaults.get(f))] for f in fields], separators=(",", ":"), default=str)
    return hashlib.sha256(_PP_BINDING_DOMAIN + payload.encode("utf-8")).hexdigest()


def check_pp_binding(pp: dict, dk: dict, fields: tuple, defaults: dict | None = None) -> str:
    """Raise unless key share *dk* was issued for public parameters with *pp*'s
    values of *fields* (see :func:`pp_binding`); return *pp*'s digest.

    The servers also seed their flooding noise with the returned digest -- that
    of their own public parameters, equal to the key share's once checked -- so
    that even if this check were bypassed, a request answered under other
    public parameters would get another noise, not an exact difference.

    Raises:
        FEKeyError: For a key share of other public parameters, or one without
            ``pp_binding`` (issue new key shares).
    """
    digest = pp_binding(pp, fields, defaults)
    if not hmac.compare_digest(str(dk.get("pp_binding", "")), digest):
        raise FEKeyError(
            "the key share was issued for other public parameters (modulus, flooding or label hash): "
            "issue new key shares"
        )
    return digest


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
    sharing = dk.get("sharing", "shamir")
    if pp.get("sharing", "shamir") != sharing:
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


# construction ids and binding fields of the threshold LWE schemes: the
# construction ids are recorded in public parameters (and parameter records) as
# ``threshold_construction``, ``{construction id: sharing}`` is what servers
# serve (:func:`check_served_sharing`), and a key share is bound to the
# public-parameter fields (:func:`pp_binding`; a missing field takes its
# default)

#: tMIFE-LWE (:mod:`pyfe4ai.schemes.mife.lwe_threshold`): Shamir sharing.
TMIFE_LWE_CONSTRUCTION = "acfgu18-abdp15-shamir-smudging-seeded-a/v4"  # v4: flooding at bits + 320, sampler pinned
#: tMIFE-LWE: Shamir sharing, KL accounting of the flooding.
TMIFE_LWE_CONSTRUCTION_KL = "acfgu18-abdp15-shamir-smudging-kl/v2"
# the parameters are those of the Shamir record; its own record keeps cached
# Shamir records (t < s) of versions <= 0.3.0 from being taken for it
#: tMIFE-LWE: replicated sharing.
TMIFE_LWE_CONSTRUCTION_REPLICATED = "acfgu18-abdp15-replicated-smudging-seeded-a/v1"
#: tMIFE-LWE: seed-expanded replicated sharing.
TMIFE_LWE_CONSTRUCTION_REPLICATED_SEEDED = "acfgu18-abdp15-replicated-seeded-smudging-seeded-a/v1"
#: tMIFE-LWE: ``{construction id: sharing}``.
TMIFE_LWE_CONSTRUCTIONS = {
    TMIFE_LWE_CONSTRUCTION: "shamir", TMIFE_LWE_CONSTRUCTION_KL: "shamir",
    TMIFE_LWE_CONSTRUCTION_REPLICATED: "replicated", TMIFE_LWE_CONSTRUCTION_REPLICATED_SEEDED: "replicated-seeded",
}
#: tMIFE-LWE: public-parameter fields a key share is bound to.
TMIFE_LWE_BINDING_FIELDS = (
    "q", "p", "lwe_n", "m", "eta", "n", "A_seed", "s", "t", "lst_sid", "sigma_f", "flood_tail_cut",
    "flooding_accounting", "max_partial_decryptions", "stat_param",
)
#: tMIFE-LWE: values of absent binding fields.
TMIFE_LWE_BINDING_DEFAULTS = {"flooding_accounting": "per-sample"}

# recorded in the public parameters (those of versions <= 0.3.0 have none)
#: tMCFE-LWE (:mod:`pyfe4ai.schemes.mcfe.lwe_threshold`): Shamir sharing.
TMCFE_LWE_CONSTRUCTION = "abmpsw20-shamir-flooding/v1"
#: tMCFE-LWE: replicated sharing.
TMCFE_LWE_CONSTRUCTION_REPLICATED = "abmpsw20-replicated-flooding/v1"
#: tMCFE-LWE: seed-expanded replicated sharing.
TMCFE_LWE_CONSTRUCTION_REPLICATED_SEEDED = "abmpsw20-replicated-seeded-flooding/v1"
#: tMCFE-LWE: ``{construction id: sharing}``.
TMCFE_LWE_CONSTRUCTIONS = {TMCFE_LWE_CONSTRUCTION: "shamir", TMCFE_LWE_CONSTRUCTION_REPLICATED: "replicated",
                           TMCFE_LWE_CONSTRUCTION_REPLICATED_SEEDED: "replicated-seeded"}
#: tMCFE-LWE: public-parameter fields a key share is bound to, the label hash
#: among them: one key share answering a label under two label hashes with the
#: same noise would give ``w_A - w_B = lambda <share, H_A(l) - H_B(l)>`` exactly.
TMCFE_LWE_BINDING_FIELDS = ("q", "q0", "K", "delta", "n0", "m0", "s", "t", "lst_sid", "sigma_f", "label_hash")
#: tMCFE-LWE: values of absent binding fields.
TMCFE_LWE_BINDING_DEFAULTS = {"label_hash": LEGACY_LABEL_HASH}

#: tMCFE-Ring-LWE (:mod:`pyfe4ai.schemes.mcfe.ring_lwe_threshold`): the
#: parameter record (and public ``threshold_construction``) of a sharing and a
#: flooding under the per-sample accounting (``"kl"``:
#: :data:`TMCFE_RING_LWE_CONSTRUCTION_KL`).
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
#: tMCFE-Ring-LWE: Shamir sharing, Gaussian flooding, KL accounting.
TMCFE_RING_LWE_CONSTRUCTION_KL = "bmms21-ibmcfe-shamir-smudging-kl/v2"
#: tMCFE-Ring-LWE: ``{construction id: sharing}``.
TMCFE_RING_LWE_CONSTRUCTIONS = {
    construction: sharing for (sharing, _), construction in TMCFE_RING_LWE_CONSTRUCTION_IDS.items()
}
TMCFE_RING_LWE_CONSTRUCTIONS[TMCFE_RING_LWE_CONSTRUCTION_KL] = "shamir"
#: tMCFE-Ring-LWE: public-parameter fields a key share is bound to: what its
#: partial decryptions depend on beyond the sharing and the construction.
TMCFE_RING_LWE_BINDING_FIELDS = (
    "q", "p", "ring_n", "eta", "s", "t", "lst_sid", "flooding", "flood_bits", "sigma_f", "flood_tail_cut",
    "flooding_accounting", "max_partial_decryptions", "stat_param", "label_hash",
)
#: tMCFE-Ring-LWE: values of absent binding fields.
TMCFE_RING_LWE_BINDING_DEFAULTS = {"flooding": "gaussian", "flooding_accounting": "per-sample",
                                   "label_hash": "shake256-wide"}
#: tMCFE-Ring-LWE: fields bound only where the public parameters have them (the
#: digests of public parameters without them, of earlier versions too, stay as
#: they were).
TMCFE_RING_LWE_OPTIONAL_BINDING_FIELDS = ("ct_format", "q_factors")


def tmcfe_ring_lwe_construction_id(sharing: str, flooding: str, accounting: str) -> str | None:
    """The tMCFE-Ring-LWE construction id of a sharing, a flooding and a
    flooding accounting (``None`` for a combination the key generator does
    not make)."""
    if accounting == KL:
        return TMCFE_RING_LWE_CONSTRUCTION_KL if (sharing, flooding) == ("shamir", "gaussian") else None
    return TMCFE_RING_LWE_CONSTRUCTION_IDS.get((sharing, flooding))


def tmcfe_ring_lwe_binding_fields(pp: dict) -> tuple:
    """The public-parameter fields a tMCFE-Ring-LWE key share of *pp* is bound
    to: :data:`TMCFE_RING_LWE_BINDING_FIELDS` and those of
    :data:`TMCFE_RING_LWE_OPTIONAL_BINDING_FIELDS` that *pp* has
    (``ct_format`` of binary public parameters, ``q_factors`` of an RNS
    modulus)."""
    return TMCFE_RING_LWE_BINDING_FIELDS + tuple(f for f in TMCFE_RING_LWE_OPTIONAL_BINDING_FIELDS if f in pp)


#: Ciphertext formats of the threshold MCFE over Ring-LWE (key generator config
#: ``ct_format``, a public parameter): ``"json"`` (decimal strings, tags over
#: the JSON form; the only format of versions <= 0.3.0, and that of public
#: parameters without ``ct_format``) and ``"binary"`` (the default: format 2
#: of :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.encode_threshold_ciphertext`, tags
#: over its digests, binary partial decryptions).
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


def fusion_weight_id(fusion_weight: dict) -> str:
    """Return a canonical digest of a fusion-weight dict.

    Key shares record the digest of the weights they were derived for, so a
    server can refuse to combine ciphertexts under different weights.

    Args:
        fusion_weight: Mapping of client IDs to integer weight lists.
    """
    if not isinstance(fusion_weight, dict) or not fusion_weight:
        raise FEValidationError("invalid fusion weights provided, need a dict")
    canonical = json.dumps(
        {nid: int_vector(lst, "fusion weight of {}".format(nid)) for nid, lst in fusion_weight.items()},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_fusion_weight(dk: dict, fusion_weight: dict) -> None:
    """Raise unless *fusion_weight* are the weights the key share *dk* is for."""
    if dk.get("weights_id") != fusion_weight_id(fusion_weight):
        raise FESchemeError(
            "fusion weights differ from those the key share was derived for"
        )


def client_server_mac_keys(lst_nid: list, lst_sid: list) -> dict:
    """Fresh 32-byte MAC keys (hex), ``{nid: {sid: key}}``, for ciphertext tags.

    Client ``nid`` holds ``{sid: key}`` and tags each ciphertext for every
    server; server ``sid`` holds ``{nid: key}`` and checks the tags meant for
    it (see :func:`check_ciphertext_tags`).
    """
    return {nid: {sid: secrets.token_hex(32) for sid in lst_sid} for nid in lst_nid}


def _tag_message(domain: str, payload) -> bytes:
    return json.dumps([domain, payload], sort_keys=True).encode("utf-8")


def ciphertext_tag(mac_key: str, domain: str, payload) -> str:
    """HMAC-SHA256 (hex) of ``(domain, payload)`` under *mac_key* (hex)."""
    return hmac.new(bytes.fromhex(mac_key), _tag_message(domain, payload), hashlib.sha256).hexdigest()


def tag_ciphertext(ct: dict, mac_keys: dict, domain: str, payload) -> dict:
    """Add ``tags`` (one per server of *mac_keys*) to *ct* and return it."""
    message = _tag_message(domain, payload) if mac_keys else None  # serialised once for all servers
    ct["tags"] = {
        sid: hmac.new(bytes.fromhex(key), message, hashlib.sha256).hexdigest() for sid, key in mac_keys.items()
    }
    return ct


def check_ciphertext_tags(dict_ct: dict, mac_keys: dict, sid: str, domain: str, payload_of) -> None:
    """Raise unless every ciphertext's tag for server *sid* verifies.

    Threshold servers compute partial decryptions that are linear in parts of
    the ciphertexts they are handed; were these chosen freely, the partials
    would reveal the shared functional key. Servers therefore only decrypt
    ciphertexts tagged by their clients.

    Args:
        dict_ct: ``{nid: ciphertext}``.
        mac_keys: The server's ``{nid: key}``.
        sid: The server's id.
        domain: Domain-separation string of the scheme.
        payload_of: ``payload_of(nid, ct)``, the tagged content of a ciphertext.

    Raises:
        FESchemeError: For a missing, malformed or wrong tag.
    """
    for nid, ct in dict_ct.items():
        ok = False
        tags = ct.get("tags") if isinstance(ct, dict) else None
        tag = tags.get(sid) if isinstance(tags, dict) else None
        key = (mac_keys or {}).get(nid)
        if isinstance(tag, str) and isinstance(key, str):
            try:
                expected = ciphertext_tag(key, domain, payload_of(nid, ct))
                ok = hmac.compare_digest(tag.encode("utf-8"), expected.encode("utf-8"))
            except (KeyError, TypeError, ValueError):
                ok = False
        if not ok:
            raise FESchemeError(
                "the ciphertext of {} is not authenticated for server {}".format(nid, sid)
            )


def tag_ciphertext_message(ct: dict, mac_keys: dict, message: bytes) -> dict:
    """Add ``tags`` to *ct*: HMAC-SHA256 (hex) of the raw *message* under each
    server's key of *mac_keys*, and return it (binary ciphertexts, whose tag
    messages are bytes rather than JSON)."""
    ct["tags"] = {sid: hmac.new(bytes.fromhex(key), message, hashlib.sha256).hexdigest() for sid, key in mac_keys.items()}
    return ct


def check_ciphertext_tag_messages(dict_ct: dict, mac_keys: dict, sid: str, message_of) -> None:
    """:func:`check_ciphertext_tags` for tags over raw messages
    (:func:`tag_ciphertext_message`): ``message_of(nid, ct)`` returns the
    bytes a ciphertext's tag covers, or raises ``ValueError`` (``KeyError``,
    ``TypeError``) for a malformed one.

    Raises:
        FESchemeError: For a missing, malformed or wrong tag.
    """
    for nid, ct in dict_ct.items():
        ok = False
        tags = ct.get("tags") if isinstance(ct, dict) else None
        tag = tags.get(sid) if isinstance(tags, dict) else None
        key = (mac_keys or {}).get(nid)
        if isinstance(tag, str) and isinstance(key, str):
            try:
                expected = hmac.new(bytes.fromhex(key), message_of(nid, ct), hashlib.sha256).hexdigest()
                ok = hmac.compare_digest(tag.encode("utf-8"), expected.encode("utf-8"))
            except (KeyError, TypeError, ValueError):
                ok = False
        if not ok:
            raise FESchemeError(
                "the ciphertext of {} is not authenticated for server {}".format(nid, sid)
            )


#: Fields in which the weight and the label part of a tMCFE-Ring-LWE key share
#: must agree (an absent ``sharing`` is ``"shamir"``).
SPLIT_SHARE_FIELDS = ("sid", "lst_sid", "weights_id", "share_set_id", "sharing", "mac_keys", "noise_version",
                      "noise_sampler", "flooding", "label_hash", "pp_binding")


def check_split_parts(weight: dict, dk: dict) -> None:
    """Raise unless the weight part *weight* and the label part *dk* of a key
    share agree in every field of :data:`SPLIT_SHARE_FIELDS`.

    Raises:
        FEKeyError: For parts that do not belong together.
    """
    for field in SPLIT_SHARE_FIELDS:
        if weight.get(field, "shamir" if field == "sharing" else None) != dk.get(
                field, "shamir" if field == "sharing" else None):
            raise FEKeyError("the weight and the label key share do not belong together ({})".format(field))


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


#: The version of the pickled state of the tMCFE-Ring-LWE objects: a state
#: pickled by another version of pyfe4ai (other attributes, other derivations)
#: is refused when loaded.
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
