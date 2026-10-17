"""
Shared threshold secret-sharing utilities for threshold FE schemes.

Provides polynomial evaluation, Shamir secret sharing, Lagrange
interpolation and modular centering used identically by both the
LWE and Ring-LWE threshold variants, and the replicated sharing
(``sharing = "replicated"``, the default for ``t < s``, see
:func:`resolve_sharing`) of the LWE / Ring-LWE threshold layers.

Why the LWE layers offer a second sharing: they flood partial decryptions
with noise that hides a small unknown (the decryption noise ``nu``) from
whoever learns the result. With ``t - 1`` servers ``C`` corrupted, the
honest partial ``lambda_k^E F(k) + e_k`` of a Shamir sharing equals
something computable from ``C``'s shares and the result, plus ``e_k - M nu``
with ``M = lambda_k^E prod_{j in C} (k - j) / (0 - j)``. That is a rational
number whenever ``t < s``: for ``s = 3, t = 2``, ``C = {2}``,
``E = {1, 3}`` it is ``3 / 4``, and four times the partial minus the
computable part is ``4 e_k - 3 nu``, an integer whose residue mod 4 is
``nu mod 4`` -- whatever the flooding width. A replicated sharing (one
additive piece per ``(t - 1)``-subset ``T`` of the points, held by the
servers outside ``T``) has reconstruction coefficients 0 and 1: ``M`` is 0
or 1, and the flooding argument holds as stated. Its cost is
``C(s - 1, t - 1)`` pieces per server instead of one share (2 for
``s = 3, t = 2``; 6 for ``s = 5, t = 3``). See :func:`share_replicated`.

Trust model of every threshold scheme: decryption servers are
honest-but-curious. Fewer than ``t`` of them learn nothing beyond the result,
but nothing proves a partial decryption correct, so a server that deviates
can change the result undetected -- unless the combiner asks for partial
decryptions for several enrolled sets and :func:`combine_consistently`
compares them.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import random
import secrets
import warnings

import gmpy2 as gp
import numpy as np

from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FESecurityWarning, FEValidationError
from pyfe4ai.utils.validation import int_vector

_CSPRNG = random.SystemRandom()


def eval_poly(coeffs: list[gp.mpz], x_value: int, modulus: gp.mpz) -> gp.mpz:
    """Evaluate a polynomial at *x_value* modulo *modulus*.

    Args:
        coeffs: Polynomial coefficient list.
        x_value: Evaluation point.
        modulus: Modulus for the arithmetic operation.
    """
    acc = gp.mpz(0)
    x = gp.mpz(x_value)
    power = gp.mpz(1)
    for coeff in coeffs:
        acc = (acc + coeff * power) % modulus
        power = (power * x) % modulus
    return acc


def share_secret(
    secret: gp.mpz, threshold: int, share_points: list[int], modulus: gp.mpz
) -> dict[int, gp.mpz]:
    """Create Shamir secret shares of *secret*.

    Args:
        secret: The secret to share.
        threshold: Minimum number of shares needed for reconstruction.
        share_points: Evaluation points for the shares.
        modulus: Modulus for the arithmetic operation.
    """
    coeffs = [gp.mpz(secret) % modulus] + [
        gp.mpz(_CSPRNG.randrange(0, int(modulus)))
        for _ in range(1, threshold)
    ]
    return {x: eval_poly(coeffs, x, modulus) for x in share_points}


def lagrange_coefficient_at_zero(
    x_j: int, xs: list[int], modulus: gp.mpz
) -> gp.mpz:
    """Return the Lagrange basis coefficient of point *x_j* evaluated at 0.

    Computes ``prod_{m != j} x_m / (x_m - x_j)`` modulo *modulus* with a
    modular inverse, so *modulus* must be prime (or at least coprime to every
    difference) and the points in *xs* must be distinct and non-zero.

    Args:
        x_j: Evaluation point whose coefficient is requested.
        xs: All evaluation points taking part in the reconstruction.
        modulus: Modulus for the arithmetic operation.
    """
    num = gp.mpz(1)
    den = gp.mpz(1)
    for x_m in xs:
        if x_m == x_j:
            continue
        num = (num * gp.mpz(x_m)) % modulus
        den = (den * gp.mpz(x_m - x_j)) % modulus
    return (num * gp.invert(den % modulus, modulus)) % modulus


def lagrange_at_zero(shares: dict[int, gp.mpz], modulus: gp.mpz) -> gp.mpz:
    """Reconstruct the secret (polynomial value at 0) from *shares*.

    Args:
        shares: List of ``(x, y)`` share pairs.
        modulus: Modulus for the arithmetic operation.
    """
    result = gp.mpz(0)
    xs = list(shares.keys())
    for x_j in xs:
        lagrange = lagrange_coefficient_at_zero(x_j, xs, modulus)
        result = (result + gp.mpz(shares[x_j]) * lagrange) % modulus
    return result


#: Ways the LWE / Ring-LWE threshold layers share a functional key (config
#: entry ``sharing``): ``"shamir"`` or ``"replicated"`` (see the module
#: docstring and :func:`resolve_sharing`).
SHARING_SCHEMES = ("shamir", "replicated")


def validate_sharing(sharing) -> str:
    """Return *sharing* if it is one of :data:`SHARING_SCHEMES`, else raise."""
    if sharing not in SHARING_SCHEMES:
        raise FEValidationError(
            "sharing must be one of {}, not {!r}".format(SHARING_SCHEMES, sharing)
        )
    return sharing


def resolve_sharing(config: dict, s, t, scheme: str) -> str:
    """The key sharing of a threshold LWE / Ring-LWE layer for *config*.

    ``config["sharing"]`` if given, else ``"replicated"`` for ``t < s`` and
    ``"shamir"`` for ``t = s`` (where the two are equally sound and Shamir
    gives the outputs of versions <= 0.3.0). An explicit ``"shamir"`` with
    ``t < s`` -- the sharing of versions <= 0.3.0, whose partial decryptions
    reveal the decryption noise modulo small integers (module docstring) --
    is refused under the ``standard`` profile and draws a
    :class:`~pyfe4ai.utils.exceptions.FESecurityWarning` otherwise.

    Args:
        config: The key generator's (profile-resolved) config.
        s: Number of servers.
        t: Threshold (checked later by :func:`validate_threshold`).
        scheme: Name of the scheme, for messages.

    Raises:
        FEValidationError: For an unknown sharing, or ``"shamir"`` with
            ``t < s`` under ``standard``.
    """
    below = (
        isinstance(t, int) and not isinstance(t, bool) and isinstance(s, int) and t < s
    )
    sharing = config.get("sharing")
    if sharing is None:
        return "replicated" if below else "shamir"
    validate_sharing(sharing)
    if sharing == "shamir" and below:
        message = (
            "{}: Shamir sharing with t = {} < s = {} lets partial decryptions reveal the "
            "decryption noise modulo small integers (versions <= 0.3.0); use "
            "sharing='replicated' (the default for t < s)".format(scheme, t, s)
        )
        from pyfe4ai.schemes.profiles import STANDARD

        if config.get("security_profile") == STANDARD:
            raise FEValidationError(message)
        warnings.warn(message, FESecurityWarning, stacklevel=4)
    return sharing


def replicated_sets(points: list[int], threshold: int) -> list[tuple[int, ...]]:
    """Index sets of a replicated sharing: the ``(t - 1)``-subsets of *points*, sorted."""
    return list(itertools.combinations(sorted(points), threshold - 1))


def _map_values(material, f):
    if isinstance(material, dict):
        return {key: _map_values(value, f) for key, value in material.items()}
    if isinstance(material, (list, tuple)):
        return [_map_values(value, f) for value in material]
    return f(material)


def _zip_values(a, b, f):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise FESchemeError("key pieces of different layouts")
        return {key: _zip_values(a[key], b[key], f) for key in a}
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise FESchemeError("key pieces of different layouts")
        return [_zip_values(u, v, f) for u, v in zip(a, b)]
    return f(a, b)


def share_replicated(
    material, threshold: int, points: list[int], modulus: gp.mpz
) -> dict[tuple[int, ...], object]:
    """Replicated sharing of every value in *material* (nested dicts and lists).

    Draws one piece ``r_T`` per ``(t - 1)``-subset ``T`` of *points*, of the
    layout of *material* (values as decimal strings), uniform subject to
    ``sum_T r_T = material mod modulus``. The server at point ``x`` receives
    the pieces with ``x`` not in ``T``: any ``t`` servers hold every piece,
    any ``t - 1`` servers ``C`` miss exactly ``r_C``, which is uniform given
    the others. With ``t = s`` this is additive sharing (one piece each).

    Returns:
        ``{T: piece}`` for every index set ``T``.
    """
    sets = replicated_sets(points, threshold)
    modulus = gp.mpz(modulus)

    def draw(_):
        return gp.digits(gp.mpz(_CSPRNG.randrange(0, int(modulus))))

    pieces = {T: _map_values(material, draw) for T in sets[1:]}
    first = _map_values(material, lambda v: gp.mpz(v) % modulus)
    for T in sets[1:]:
        first = _zip_values(first, pieces[T], lambda a, b: (a - gp.mpz(b)) % modulus)
    pieces[sets[0]] = _map_values(first, gp.digits)
    return pieces


def pieces_held(pieces: dict, point: int) -> list[dict]:
    """The pieces of :func:`share_replicated` held by the server at *point*,
    as ``[{"set": [...], "share": piece}]`` (JSON-friendly)."""
    return [{"set": list(T), "share": piece} for T, piece in pieces.items() if point not in T]


def replicated_assignment(
    x_k: int, xs: list[int], points: list[int], threshold: int
) -> list[tuple[int, ...]]:
    """Index sets whose pieces the server at *x_k* adds up for enrolled points *xs*.

    Piece ``r_T`` goes to the smallest enrolled point outside ``T``
    (``|xs| >= t > |T|``, so there is one, and it holds ``r_T``): the
    enrolled servers' sums add up to the secret, each piece counted once.
    """
    return [T for T in replicated_sets(points, threshold) if min(x for x in xs if x not in T) == x_k]


def share_for_enrolled(
    dk: dict, sid: str, lst_sid_enrolled: list[str], threshold: int, modulus: gp.mpz
) -> tuple[dict, gp.mpz, str]:
    """What server *sid* multiplies, by what, for a partial decryption for *lst_sid_enrolled*.

    Returns ``(share, multiplier, tag)``. For a Shamir key share *dk*:
    ``(dk, lambda, digits(lambda))``, ``lambda`` the Lagrange coefficient
    at zero. For a replicated one (``dk["pieces"]``): *dk* with the sum of
    the pieces of :func:`replicated_assignment` in place of the pieces,
    multiplier 1 and a tag naming those pieces. Partial decryptions seed
    their flooding noise with the tag: they depend on the enrolled set only
    through it.

    Raises:
        FEKeyError: If *dk* lacks a piece the server must add.
    """
    sid_v = share_points(dk["lst_sid"])
    xs = [sid_v[s] for s in lst_sid_enrolled]
    if "pieces" not in dk:
        lagrange = lagrange_coefficient_at_zero(sid_v[sid], xs, modulus)
        return dk, lagrange, gp.digits(lagrange)
    held = {tuple(piece["set"]): piece["share"] for piece in dk["pieces"]}
    if not held:
        raise FEKeyError("the key share holds no pieces")
    assigned = replicated_assignment(sid_v[sid], xs, list(sid_v.values()), threshold)
    missing = [T for T in assigned if T not in held]
    if missing:
        raise FEKeyError("the key share lacks the pieces {}".format(missing))
    modulus = gp.mpz(modulus)
    total = _map_values(next(iter(held.values())), lambda _: gp.mpz(0))
    for T in assigned:
        total = _zip_values(total, held[T], lambda a, b: (a + gp.mpz(b)) % modulus)
    share = {key: value for key, value in dk.items() if key != "pieces"}
    share.update(_map_values(total, gp.digits))
    return share, gp.mpz(1), json.dumps(["replicated", [list(T) for T in assigned]])


def share_points(lst_sid: list[str]) -> dict[str, int]:
    """Map each decryption server to its Shamir evaluation point.

    Points start at 1: the polynomial value at 0 is the shared secret itself,
    so no server may ever hold the share at x = 0.

    Args:
        lst_sid: Ordered list of all decryption-server identifiers.
    """
    return {sid: idx + 1 for idx, sid in enumerate(lst_sid)}


def validate_threshold(threshold: int, lst_sid: list[str]) -> None:
    """Reject repeated server ids and thresholds outside ``1 <= t <= s``.

    Args:
        threshold: Minimum number of shares needed for reconstruction.
        lst_sid: Ordered list of all decryption-server identifiers.
    """
    if len(set(lst_sid)) != len(lst_sid):
        raise FEValidationError("lst_sid contains duplicate server ids")
    if not isinstance(threshold, int) or not 1 <= threshold <= len(lst_sid):
        raise FEValidationError(
            "invalid threshold t={} for s={} decryption servers".format(
                threshold, len(lst_sid)
            )
        )


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
    """
    if len(dict_ct_prime) < threshold:
        raise FESchemeError(
            "insufficient threshold shares for decryption: got {}, need at "
            "least t={}".format(len(dict_ct_prime), threshold)
        )
    contributors = set()
    enrolled = None
    for ct_prime in dict_ct_prime.values():
        if "sid" not in ct_prime or "lst_sid_enrolled" not in ct_prime:
            raise FESchemeError(
                "partial decryption is missing its server id / enrolled set"
            )
        if ct_prime["sid"] in contributors:
            raise FESchemeError(
                "duplicate partial decryption from {}".format(ct_prime["sid"])
            )
        contributors.add(ct_prime["sid"])
        _enrolled = set(ct_prime["lst_sid_enrolled"])
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


def center_mod(value: gp.mpz, modulus: gp.mpz) -> gp.mpz:
    """Center *value* into the range ``(-modulus/2, modulus/2]``.

    Args:
        value: Input value.
        modulus: Modulus for the arithmetic operation.
    """
    value %= modulus
    half = modulus // 2
    if value > half:
        value -= modulus
    return value


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


def ciphertexts_digest(canonical: dict) -> str:
    """Return the digest of ciphertexts given in canonical form (``{nid: values}``).

    Partial decryptions seed their flooding noise with it, so it must depend
    only on the values a server computes with, never on how a request encodes
    them: otherwise re-encoding one request draws fresh noise, and averaging
    the partials removes it.
    """
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


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


def check_fusion_weight(dk: dict, fusion_weight: dict) -> None:
    """Raise unless *fusion_weight* are the weights the key share *dk* is for."""
    if dk.get("weights_id") != fusion_weight_id(fusion_weight):
        raise FESchemeError(
            "fusion weights differ from those the key share was derived for"
        )
