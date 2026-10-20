"""
What the threshold artifacts are bound to, and the checks.

* Key shares to public parameters: :func:`pp_binding` over a scheme's
  binding fields, checked by :func:`check_pp_binding` before every partial
  decryption; key shares to the weights they were derived for
  (:func:`fusion_weight_id`). The sharing, the threshold construction and the
  other version markers are checked by :mod:`pyfe4ai.core.versions`.
* Ciphertexts to the servers that decrypt them: HMAC-SHA256 tags under
  client-server keys (:func:`client_server_mac_keys`,
  :func:`tag_ciphertext`, :func:`check_ciphertext_tags`; the tag messages
  are in :mod:`pyfe4ai.schemes.threshold.codec`).
* The threshold MCFE over Ring-LWE: the fields its weight and label key
  share parts share (:func:`check_split_parts`).

Binding fields are those of earlier versions; nothing here changes a digest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets

from pyfe4ai._validation import int_vector
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats.threshold import TaggedCiphertext
from pyfe4ai.core.formats.threshold import ThresholdKeyShare
from pyfe4ai.core.versions import LEGACY_FLOODING
from pyfe4ai.core.versions import LEGACY_SHARING
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH

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
    (:func:`~pyfe4ai.core.versions.check_served_sharing`).
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
    if not hmac.compare_digest(str(ThresholdKeyShare(dk).get("pp_binding", "")), digest):
        raise FEKeyError(
            "the key share was issued for other public parameters (modulus, flooding or label hash): "
            "issue new key shares"
        )
    return digest


# the public-parameter fields each threshold LWE scheme binds a key share to
# (:func:`pp_binding`; a missing field takes its default -- what its absence
# means, see :mod:`pyfe4ai.core.versions`)

#: The public-parameter fields a tMIFE-LWE key share is bound to.
TMIFE_LWE_BINDING_FIELDS = (
    "q", "p", "lwe_n", "m", "eta", "n", "A_seed", "s", "t", "lst_sid", "sigma_f", "flood_tail_cut",
    "flooding_accounting", "max_partial_decryptions", "stat_param",
)
#: The values of absent tMIFE-LWE binding fields.
TMIFE_LWE_BINDING_DEFAULTS = {"flooding_accounting": "per-sample"}

#: The public-parameter fields a tMCFE-LWE key share is bound to, the label
#: hash among them (one key share answering a label under two label hashes with
#: the same noise would give ``w_A - w_B = lambda <share, H_A(l) - H_B(l)>`` exactly).
TMCFE_LWE_BINDING_FIELDS = ("q", "q0", "K", "delta", "n0", "m0", "s", "t", "lst_sid", "sigma_f", "label_hash")
#: The values of absent tMCFE-LWE binding fields.
TMCFE_LWE_BINDING_DEFAULTS = {"label_hash": LEGACY_LABEL_HASH}

#: The public-parameter fields a tMCFE-Ring-LWE key share is bound to (what
#: its partial decryptions depend on beyond the sharing and the construction).
TMCFE_RING_LWE_BINDING_FIELDS = (
    "q", "p", "ring_n", "eta", "s", "t", "lst_sid", "flooding", "flood_bits", "sigma_f", "flood_tail_cut",
    "flooding_accounting", "max_partial_decryptions", "stat_param", "label_hash",
)
#: The values of absent tMCFE-Ring-LWE binding fields.
TMCFE_RING_LWE_BINDING_DEFAULTS = {"flooding": LEGACY_FLOODING, "flooding_accounting": "per-sample",
                                   "label_hash": LEGACY_LABEL_HASH}
#: The tMCFE-Ring-LWE fields bound only where the public parameters have them
#: (the digests of public parameters without them, of earlier versions too, stay
#: as they were).
TMCFE_RING_LWE_OPTIONAL_BINDING_FIELDS = ("ct_format", "q_factors")


def tmcfe_ring_lwe_binding_fields(pp: dict) -> tuple:
    """The public-parameter fields a tMCFE-Ring-LWE key share of *pp* is bound
    to: :data:`TMCFE_RING_LWE_BINDING_FIELDS` and those of
    :data:`TMCFE_RING_LWE_OPTIONAL_BINDING_FIELDS` that *pp* has
    (``ct_format`` of binary public parameters, ``q_factors`` of an RNS
    modulus)."""
    return TMCFE_RING_LWE_BINDING_FIELDS + tuple(f for f in TMCFE_RING_LWE_OPTIONAL_BINDING_FIELDS if f in pp)


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
    if ThresholdKeyShare(dk).get("weights_id") != fusion_weight_id(fusion_weight):
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
    TaggedCiphertext(ct).tags = {
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
        tags = TaggedCiphertext(ct).get("tags") if isinstance(ct, dict) else None
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
    TaggedCiphertext(ct).tags = {
        sid: hmac.new(bytes.fromhex(key), message, hashlib.sha256).hexdigest() for sid, key in mac_keys.items()
    }
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
        tags = TaggedCiphertext(ct).get("tags") if isinstance(ct, dict) else None
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
    weight, dk = ThresholdKeyShare(weight), ThresholdKeyShare(dk)
    for field in SPLIT_SHARE_FIELDS:
        if weight.get(field, LEGACY_SHARING if field == "sharing" else None) != dk.get(
                field, LEGACY_SHARING if field == "sharing" else None):
            raise FEKeyError("the weight and the label key share do not belong together ({})".format(field))
