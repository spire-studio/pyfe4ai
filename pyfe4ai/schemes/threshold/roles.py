"""
Server and combiner logic shared by the threshold schemes.

Trust model of every threshold scheme: decryption servers are
honest-but-curious. Fewer than ``t`` of them learn nothing beyond the result,
but nothing proves a partial decryption correct, so a server that deviates
can change the result undetected -- unless the combiner asks for partial
decryptions for several enrolled sets and :func:`combine_consistently`
compares them.

* A server: its policy for Shamir key shares with ``t < s``
  (:func:`~pyfe4ai.core.versions.legacy_shamir_policy`), the checks before a partial decryption
  (:func:`check_key_share_server`, :func:`check_request`), how it sums its
  share for an enrolled set (:func:`enrolled_share_options` for
  :func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled`) and the
  flooding noise of the LWE threshold layers, a function of the server's
  noise key and the request (:func:`noise_seed`; :func:`flooding_noise`,
  :func:`flooding_poly`, :func:`uniform_flooding_poly`).
* A combiner: the checks of a set of partial decryptions
  (:func:`validate_partial_decryptions`) and the combination of one or
  several enrolled sets (:func:`combine_consistently`,
  :func:`combine_elementwise`).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from pyfe4ai.core.exceptions import FESchemeError
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.formats.threshold import PartialDecryption
from pyfe4ai.core.formats.threshold import ThresholdKeyShare
from pyfe4ai.core.versions import check_served_sharing
from pyfe4ai.primitives.lattice.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.primitives.sampling import FLOODING_SAMPLER
from pyfe4ai.primitives.sampling import discrete_gaussian_bigint
from pyfe4ai.schemes.threshold.binding import check_fusion_weight
from pyfe4ai.schemes.threshold.store import share_cache

def check_key_share_server(server_id: str, dk: dict, pp: dict, constructions: dict) -> None:
    """Raise unless key share *dk* is server *server_id*'s and of the sharing
    and a threshold construction of *pp* among *constructions*
    (:func:`~pyfe4ai.core.versions.check_served_sharing`).

    Raises:
        FESchemeError: For another server's key share, another sharing or an
            unknown construction.
    """
    key = ThresholdKeyShare(dk)
    if server_id not in key.lst_sid or key.get("sid") != server_id:
        raise FESchemeError("the key share does not belong to server {}".format(server_id))
    check_served_sharing(pp, dk, constructions)


def check_request(server_id: str, dk: dict, lst_sid_enrolled: list, threshold: int, fusion_weight) -> None:
    """Raise unless server *server_id* may answer for *lst_sid_enrolled*
    (:func:`validate_enrolled`) and *fusion_weight* are the weights its key
    share *dk* was derived for.

    Raises:
        FESchemeError: Otherwise.
    """
    validate_enrolled(server_id, ThresholdKeyShare(dk).lst_sid, lst_sid_enrolled, threshold)
    check_fusion_weight(dk, fusion_weight)


def enrolled_share_options(server) -> dict:
    """The keyword arguments of
    :func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled` for a server
    object: its cache of sums, its ``legacy_shamir`` policy and the
    ``security_profile`` of its public parameters."""
    return {"cache": share_cache(server), "legacy_shamir": server.legacy_shamir,
            "profile": server.pp.get("security_profile")}


def noise_seed(dk: dict, *context) -> bytes:
    """The seed of a partial decryption's flooding noise: key share *dk*'s
    ``noise_key`` (secret, per server) followed by the JSON list of *context*
    -- what the key part depends on (the ciphertext values or the label), the
    tag of the server's share for the enrolled set, and the digest of the
    server's own public parameters (the key share's ``pp_binding``, checked) --
    so that a request for the same key part always draws the same noise, and
    under other public parameters another noise."""
    return bytes.fromhex(ThresholdKeyShare(dk).noise_key) + json.dumps(list(context)).encode("utf-8")


# flooding noise of the partial decryptions


def flooding_noise(sigma: int, rng=None) -> int:
    """Sample ``e <- D_{Z, sigma}`` truncated to ``|e| <= 12 sigma``.

    Uses the arbitrary-precision sampler: ``sigma`` is far wider than a
    double's mantissa, and scaling a 53-bit normal by it would leave the low
    bits of ``e`` deterministic.

    Args:
        sigma: Width.
        rng: Bit source (default: OS CSPRNG). An
            :class:`~pyfe4ai.primitives.sampling.XofRandom` makes the
            sample a function of its seed.
    """
    bound = GAUSSIAN_TAIL_CUT * sigma
    while True:
        e = discrete_gaussian_bigint(sigma, 1, rng=rng, sampler=FLOODING_SAMPLER)[0]
        if abs(e) <= bound:
            return int(e)


def flooding_poly(sigma: int, ring_n: int, rng) -> list:
    """*ring_n* samples of ``D_{Z, sigma}`` cut at ``GAUSSIAN_TAIL_CUT sigma``, drawn in order."""
    bound = GAUSSIAN_TAIL_CUT * sigma
    # one call for ring_n samples draws what ring_n calls for one sample draw
    out = [e for e in discrete_gaussian_bigint(sigma, ring_n, rng=rng, sampler=FLOODING_SAMPLER) if abs(e) <= bound]
    while len(out) < ring_n:  # beyond the tail cut (rare): draw on, one at a time
        e = discrete_gaussian_bigint(sigma, 1, rng=rng, sampler=FLOODING_SAMPLER)[0]
        if abs(e) <= bound:
            out.append(e)
    return out


#: Domain of :func:`uniform_flooding_poly` (the uniform flooding of the threshold MCFE over Ring-LWE).
_UNIFORM_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/uniform-flooding/v1"


def uniform_flooding_poly(bits: int, ring_n: int, seed: bytes) -> list:
    """*ring_n* integers uniform on ``[-2^bits, 2^bits)``, a function of *seed*:
    coefficient ``i`` is block ``i`` of ``ceil((bits + 1) / 8)`` big-endian
    bytes of SHAKE-256 over a domain tag, the seed's length and the seed, its
    low ``bits + 1`` bits, minus ``2^bits``."""
    nb = (bits + 8) // 8
    mask, offset = (1 << (bits + 1)) - 1, 1 << bits
    stream = hashlib.shake_256(_UNIFORM_DOMAIN + len(seed).to_bytes(8, "big") + seed).digest(nb * ring_n)
    from_bytes = int.from_bytes
    return [(from_bytes(stream[i:i + nb], "big") & mask) - offset for i in range(0, nb * ring_n, nb)]


def uniform_flooding_material(seed: bytes) -> bytes:
    """The SHAKE-256 input of :func:`uniform_flooding_poly` for *seed* (the
    native backend draws the noise from it)."""
    return _UNIFORM_DOMAIN + len(seed).to_bytes(8, "big") + seed


# combination


def validate_enrolled(
    sid: str, lst_sid: list[str], lst_sid_enrolled: list[str], threshold: int
) -> None:
    """Check that *sid* may produce a partial decryption for *lst_sid_enrolled*.

    Args:
        sid: Identifier of the server producing the partial decryption.
        lst_sid: Ordered list of all decryption-server identifiers.
        lst_sid_enrolled: Servers whose shares will be combined.
        threshold: Minimum number of shares needed for reconstruction.
    """
    if len(set(lst_sid_enrolled)) != len(lst_sid_enrolled):
        raise FESchemeError(
            "duplicate server ids in enrolled set: {}".format(lst_sid_enrolled)
        )
    unknown = [s for s in lst_sid_enrolled if s not in lst_sid]
    if unknown:
        raise FESchemeError("unknown server ids in enrolled set: {}".format(unknown))
    if sid not in lst_sid_enrolled:
        raise FESchemeError(
            "server {} is not part of the enrolled set {}".format(
                sid, lst_sid_enrolled
            )
        )
    if len(lst_sid_enrolled) < threshold:
        raise FESchemeError(
            "insufficient threshold shares: {} enrolled, need at least t={}".format(
                len(lst_sid_enrolled), threshold
            )
        )


def validate_partial_decryptions(dict_ct_prime: dict, threshold: int) -> None:
    """Check a set of partial decryptions before combining them.

    Each partial decryption carries the server id that produced it and the
    enrolled set its Lagrange coefficient was computed for; the contributors
    must be exactly that enrolled set, each appearing once, and number at
    least *threshold*.

    Args:
        dict_ct_prime: Mapping of server ids to partial decryptions.
        threshold: Minimum number of shares needed for reconstruction.

    Raises:
        FEValidationError: If a partial decryption's ``sid`` is not a string or
            its ``lst_sid_enrolled`` not a list (or tuple) of strings (the
            message holds none of the values).
        FESchemeError: For too few, duplicate or mismatched partial decryptions.
    """
    if len(dict_ct_prime) < threshold:
        raise FESchemeError(
            "insufficient threshold shares for decryption: got {}, need at "
            "least t={}".format(len(dict_ct_prime), threshold)
        )
    contributors = set()
    enrolled = None
    for ct_prime in map(PartialDecryption, dict_ct_prime.values()):
        if "sid" not in ct_prime or "lst_sid_enrolled" not in ct_prime:
            raise FESchemeError(
                "partial decryption is missing its server id / enrolled set"
            )
        enrolled_ids = ct_prime.lst_sid_enrolled
        if not isinstance(ct_prime.sid, str) or not isinstance(enrolled_ids, (list, tuple)) or not all(
                isinstance(sid, str) for sid in enrolled_ids):
            # the messages below sort and print the ids: other types would fail there (TypeError)
            raise FEValidationError(
                "malformed partial decryption: sid and lst_sid_enrolled must be server ids (strings)"
            )
        if ct_prime.sid in contributors:
            raise FESchemeError(
                "duplicate partial decryption from {}".format(ct_prime.sid)
            )
        contributors.add(ct_prime.sid)
        _enrolled = set(ct_prime.lst_sid_enrolled)
        if enrolled is None:
            enrolled = _enrolled
        elif enrolled != _enrolled:
            raise FESchemeError(
                "partial decryptions were computed for different enrolled sets"
            )
    if contributors != enrolled:
        raise FESchemeError(
            "partial decryptions from {} do not match the enrolled set {}".format(
                sorted(contributors), sorted(enrolled)
            )
        )


def partial_groups(partials) -> list:
    """Partial decryptions as a list of groups ``{sid: partial}``, one per enrolled set.

    Accepts one group (a mapping) or several (a list or tuple of mappings).
    """
    if isinstance(partials, dict):
        return [partials]
    if isinstance(partials, (list, tuple)) and partials and all(isinstance(g, dict) for g in partials):
        return list(partials)
    raise FESchemeError("expected a mapping of partial decryptions or a list of such mappings")


def combine_consistently(partials, combine_group):
    """Combine every group of partial decryptions and require a single result.

    Servers are assumed honest-but-curious: nothing proves a partial
    decryption correct, and one server can shift the result of every group it
    belongs to. With more than ``t`` servers, partial decryptions for several
    enrolled sets (a server answers once per set) detect such a server --
    though not which one -- unless it is in every group: to detect any one
    faulty server, leave every server out of some group, e.g. use the
    ``t + 1`` subsets of size ``t`` of ``t + 1`` servers.

    Args:
        partials: ``{sid: partial}`` for one enrolled set, or a list of them.
        combine_group: Function combining one group into a result.

    Raises:
        FESchemeError: If the groups give different results.
    """
    results = [combine_group(group) for group in partial_groups(partials)]
    if any(result != results[0] for result in results[1:]):
        raise FESchemeError(
            "partial decryptions for different enrolled sets disagree: "
            "a server is faulty or malicious"
        )
    return results[0]


def combine_elementwise(dict_ndarray_ct, combine) -> list:
    """Combine element-wise ndarray partial decryptions into object arrays.

    Args:
        dict_ndarray_ct: ``{sid: [array per layer]}`` for one enrolled set, or
            a list of them (see :func:`partial_groups`).
        combine: ``combine(groups, layer, index)``, combining the partials of
            one element (a list of ``{sid: partial}``).

    Raises:
        FESchemeError: If the servers' outputs differ in structure or an
            element does not decrypt (servers disagree, or out of range).
    """
    groups = partial_groups(dict_ndarray_ct)
    if not groups[0]:
        raise FESchemeError("no partial decryptions given")
    sample = next(iter(groups[0].values()))
    outputs = [out for group in groups for out in group.values()]
    if not all(isinstance(out, (list, tuple)) and all(isinstance(a, np.ndarray) for a in out) for out in outputs):
        raise FESchemeError("malformed partial decryptions: expected lists of object arrays")
    shapes = [np.shape(ary) for ary in sample]
    for out in outputs:
        if len(out) != len(shapes) or [np.shape(ary) for ary in out] != shapes:
            raise FESchemeError("partial decryptions differ in shape")
    result = []
    for l, shape in enumerate(shapes):
        ary = np.empty(shape, dtype=object)
        for i in np.ndindex(*shape):
            value = combine([{sid: g[sid][l][i] for sid in g} for g in groups], l, i)
            if value is None or (isinstance(value, float) and value != value):
                raise FESchemeError("element {} of layer {} does not decrypt".format(i, l))
            ary[i] = value
        result.append(ary)
    return result
