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
``nu mod 4`` -- whatever the flooding width. Fewer corrupted servers, or
none, can suffice: from the partials of several enrolled sets the aggregator
alone reads ``nu`` modulo ``(s - 1)!`` for ``t = 1`` (6 for ``s = 5, t = 2``).
A replicated sharing (one
additive piece per ``(t - 1)``-subset ``T`` of the points, held by the
servers outside ``T``) has reconstruction coefficients 0 and 1: ``M`` is 0
or 1, and the flooding argument holds as stated. Its cost is
``C(s - 1, t - 1)`` pieces per server instead of one share (2 for
``s = 3, t = 2``; 6 for ``s = 5, t = 3``). See :func:`share_replicated`.
The opt-in ``sharing = "replicated-seeded"`` (:func:`share_replicated_seeded`)
expands every piece but one from a 32-byte seed, so that a server holds seeds
and at most one full piece; the pieces, and the partial decryptions, are
those of a replicated sharing, but the key shares hide the key only
computationally (SHAKE-256 keyed by the seeds as a pseudo-random function).

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
from collections import OrderedDict

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
#: entry ``sharing``): ``"shamir"``, ``"replicated"`` or the opt-in
#: ``"replicated-seeded"`` (see the module docstring and :func:`resolve_sharing`).
SHARING_SCHEMES = ("shamir", "replicated", "replicated-seeded")


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
        FEValidationError: For a threshold that is not an integer (also a
            ``bool``), an unknown sharing, or ``"shamir"`` with ``t < s``
            under ``standard``.
    """
    if isinstance(t, bool) or not isinstance(t, int):
        raise FEValidationError("invalid threshold t={!r}: need an integer".format(t))
    below = isinstance(s, int) and not isinstance(s, bool) and t < s
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


#: Bytes of a seed of :func:`share_replicated_seeded` (256 bits: many seeds
#: are dealt, and the LWE layers aim at post-quantum security).
SEED_BYTES = 32
_SEED_DOMAIN = "pyfe4ai/threshold/replicated-seeded/piece/v1"


def piece_layout(material):
    """The layout of *material* that :func:`expand_seeded_piece` follows:
    dicts and lists of the same structure, a list of values replaced by its
    length and a single value by ``None`` (JSON-friendly)."""
    if isinstance(material, dict):
        return {key: piece_layout(value) for key, value in material.items()}
    if isinstance(material, (list, tuple)):
        if all(not isinstance(value, (dict, list, tuple)) for value in material):
            return len(material)
        return [piece_layout(value) for value in material]
    return None


def _json_key(key) -> str:
    """*key* as JSON writes a dict key (``0`` becomes ``"0"``): what a key
    share names it after a JSON round trip."""
    try:
        return next(iter(json.loads(json.dumps({key: None}))))
    except (TypeError, ValueError):
        raise FEKeyError("key pieces must name their entries by strings or numbers, not {!r}".format(key)) from None


def _map_layout(layout, leaf, keys=(), path=()):
    """*layout* with every leaf replaced by ``leaf(keys, path, length)``
    (``None`` for a single value): *keys* index the layout as given, *path*
    is their canonical form -- dict keys as JSON writes them, list indices as
    ints -- so that a key share expands to the same pieces after a JSON round
    trip (client ids ``0`` and ``"0"`` name the same entry, and may not both
    occur)."""
    if isinstance(layout, dict):
        names = [_json_key(key) for key in layout]
        if len(set(names)) != len(names):
            raise FEKeyError("key pieces with entries of the same name in JSON: {}".format(sorted(names)))
        return {
            key: _map_layout(value, leaf, keys + (key,), path + (name,))
            for (key, value), name in zip(layout.items(), names)
        }
    if isinstance(layout, list):
        return [_map_layout(value, leaf, keys + (i,), path + (i,)) for i, value in enumerate(layout)]
    if layout is not None and (isinstance(layout, bool) or not isinstance(layout, int) or layout < 0):
        raise FEKeyError("malformed layout of the key pieces")
    return leaf(keys, list(path), layout)


def _select(layout, fields):
    if not isinstance(layout, dict):
        raise FEKeyError("malformed layout of the key pieces")
    if fields is None:
        return layout
    try:
        return {name: layout[name] for name in fields}
    except KeyError:
        raise FEKeyError("malformed layout of the key pieces") from None


def _check_seed(seed, index_set) -> str:
    """*seed* as lower-case hex without spaces (``bytes.fromhex`` also reads
    upper case and spaces, which would otherwise hash differently)."""
    try:
        raw = bytes.fromhex(seed)
        if len(raw) == SEED_BYTES:
            return raw.hex()
    except (TypeError, ValueError):
        pass
    raise FEKeyError("malformed seed of the key piece {}".format(list(index_set)))


def _check_context(ctx, modulus) -> None:
    if ctx is not None and gp.mpz(ctx.q) != modulus:
        raise FESchemeError("the native ring context is for another modulus")


def _leaf_input(seed: str, index_set, path: list, q_text: str) -> str:
    return json.dumps([seed, [int(x) for x in index_set], path, q_text])


def expand_seeded_piece(seed: str, index_set, layout, modulus, fields=None, ctx=None):
    """The piece ``r_T`` of a seeded replicated sharing, of *layout*, as ints mod *modulus*.

    Every list of values (path ``p`` in *layout*, length ``L``) is
    ``hash_to_zq_vector(json([seed, T, p, q]), q, L)`` under the domain
    ``pyfe4ai/threshold/replicated-seeded/piece/v1``: SHAKE-256 blocks of
    ``bits(q) + 128`` bits reduced mod ``q``, each within ``2^-130`` of
    uniform, and pseudo-random as long as the seed is secret (SHAKE-256 as a
    PRF keyed by the seed, which follows a public prefix, on the inputs
    ``(T, p, q)``). The seed is lower-case hex and the dict keys of ``p`` are
    the strings JSON makes of them, so that a key share expands alike before
    and after a JSON round trip.

    Args:
        seed: The piece's seed (hex, :data:`SEED_BYTES` bytes).
        index_set: The set ``T`` of the piece.
        layout: :func:`piece_layout` of the shared material.
        modulus: Modulus ``q``.
        fields: Top-level entries to expand (all if ``None``).
        ctx: A native ring context of *modulus* (any dimension): expand with
            the extension's ``hash_to_zq``, which gives the same values.

    Raises:
        FEKeyError: For a malformed seed or layout.
        FESchemeError: For a context of another modulus.
    """
    from pyfe4ai.utils.crypto_utils import hash_to_zq_material, hash_to_zq_vector

    seed = _check_seed(seed, index_set)
    modulus = gp.mpz(modulus)
    _check_context(ctx, modulus)
    q_text = gp.digits(modulus)

    def leaf(keys, path, length):
        v, count = _leaf_input(seed, index_set, path, q_text), 1 if length is None else length
        if ctx is not None and count:
            from pyfe4ai.utils import native_backend

            values = ctx.unpack(native_backend.ext().hash_to_zq(ctx.cap, hash_to_zq_material(v, _SEED_DOMAIN), count))
        else:
            values = hash_to_zq_vector(v, modulus, count, domain=_SEED_DOMAIN)
        return values[0] if length is None else values

    return _map_layout(_select(layout, fields), leaf)


def _combine_pieces(layout, explicit: list, seeded: list, modulus, ctx=None, subtract=False):
    """``sum(explicit) + sum(seeded)`` (``- sum(seeded)`` if *subtract*) mod
    *modulus*, leaf by leaf, as decimal strings of *layout*.

    *explicit* are pieces of *layout* (ints, mpz or decimal strings),
    *seeded* ``(seed, T)`` pairs of :func:`expand_seeded_piece`. With a native
    ring context *ctx*, leaves of its dimension are expanded and added up by
    the extension (``hash_to_zq`` and ``lincomb``), with the values of the
    Python code; the others in Python.
    """
    from pyfe4ai.utils.crypto_utils import hash_to_zq_material, hash_to_zq_vector

    seeded = [(_check_seed(seed, T), T) for seed, T in seeded]
    modulus = gp.mpz(modulus)
    _check_context(ctx, modulus)
    q_text = gp.digits(modulus)
    ext = None
    if ctx is not None:
        from pyfe4ai.utils import native_backend

        ext = native_backend.ext()

    def leaf(keys, path, length):
        count = 1 if length is None else length
        rows = []
        for piece in explicit:
            value = piece
            for key in keys:
                value = value[key]
            rows.append([value] if length is None else value)
        if ext is not None and count == ctx.ring_n and (rows or seeded):
            packed = [ctx.pack(row) for row in rows] + [
                ext.hash_to_zq(ctx.cap, hash_to_zq_material(_leaf_input(seed, T, path, q_text), _SEED_DOMAIN), count)
                for seed, T in seeded
            ]
            sign = modulus - 1 if subtract else 1
            return ext.lincomb(ctx.cap, packed, [1] * len(rows) + [sign] * len(seeded))
        acc = [gp.mpz(0)] * count
        for row in rows:
            if len(row) != count:
                raise FESchemeError("key pieces of different layouts")
            acc = [a + gp.mpz(b) for a, b in zip(acc, row)]
        for seed, T in seeded:
            values = hash_to_zq_vector(_leaf_input(seed, T, path, q_text), modulus, count, domain=_SEED_DOMAIN)
            acc = [a - b for a, b in zip(acc, values)] if subtract else [a + b for a, b in zip(acc, values)]
        out = [gp.digits(a % modulus) for a in acc]
        return out[0] if length is None else out

    return _map_layout(layout, leaf)


def share_replicated_seeded(material, threshold: int, points: list[int], modulus: gp.mpz, ctx=None) -> dict:
    """Replicated sharing of *material* with pieces expanded from seeds.

    The pieces are those of :func:`share_replicated` -- one per
    ``(t - 1)``-subset ``T`` of *points*, held by the servers outside ``T``,
    adding up to *material* mod *modulus* -- except that each piece but the
    first, ``T0 = {1, ..., t - 1}``, is ``r_T = G(seed_T, T)``
    (:func:`expand_seeded_piece`, a fresh :data:`SEED_BYTES`-byte seed per
    piece) and the first is the correction ``r_T0 = material - sum r_T``.
    A server stores the seeds of its pieces (``C(s - 1, t - 1)`` of them, one
    fewer if it holds ``r_T0``) and ``r_T0`` if it lies outside ``T0``: the
    servers ``1..t-1`` hold no full piece, the others one.

    Hiding is computational: a coalition ``C`` of ``t - 1`` servers other than
    ``T0`` holds ``r_T0`` and misses ``seed_C``, and ``r_T0`` is independent of
    the key only once ``G(seed_C)`` is replaced by a uniform piece. That
    hybrid takes SHAKE-256 to be a PRF keyed by the secret 256-bit seed, which
    follows a public prefix and is expanded over many leaf paths (a standard
    keyed-sponge assumption, of the kind the derandomised flooding noise of
    the threshold layers makes with its key first). ``C = T0`` holds seeds
    only, which are independent of the key, as with :func:`share_replicated`.

    Args:
        ctx: Native ring context of *modulus*: expand and add up its
            dimension's leaves natively (same values).

    Returns:
        ``{"layout", "correction": (T0, r_T0 as decimal strings), "seeds": {T: hex}}``.
    """
    sets = replicated_sets(points, threshold)
    layout = piece_layout(material)
    seeds = {T: secrets.token_hex(SEED_BYTES) for T in sets[1:]}
    correction = _combine_pieces(layout, [material], [(seeds[T], T) for T in sets[1:]], modulus, ctx, subtract=True)
    return {"layout": layout, "correction": (sets[0], correction), "seeds": seeds}


def seeded_pieces_held(sharing: dict, point: int) -> dict:
    """What the server at *point* stores of a :func:`share_replicated_seeded`
    sharing: ``{"pieces": [{"set", "share"}] (r_T0, or none), "seeds":
    [{"set", "seed"}], "layout"}`` (JSON-friendly)."""
    T0, correction = sharing["correction"]
    return {
        "pieces": [] if point in T0 else [{"set": list(T0), "share": correction}],
        "seeds": [{"set": list(T), "seed": seed} for T, seed in sharing["seeds"].items() if point not in T],
        "layout": sharing["layout"],
    }


def replicated_assignment(
    x_k: int, xs: list[int], points: list[int], threshold: int
) -> list[tuple[int, ...]]:
    """Index sets whose pieces the server at *x_k* adds up for enrolled points *xs*.

    Piece ``r_T`` goes to the smallest enrolled point outside ``T``
    (``|xs| >= t > |T|``, so there is one, and it holds ``r_T``): the
    enrolled servers' sums add up to the secret, each piece counted once.
    """
    return [T for T in replicated_sets(points, threshold) if min(x for x in xs if x not in T) == x_k]


#: Entries kept by a cache of :func:`share_for_enrolled` (per scheme object).
SHARE_CACHE_SIZE = 16

#: What a server does with a Shamir key share whose ``t < s`` (crypto config
#: entry ``legacy_shamir``): ``"refuse"`` (default) or ``"warn"`` (serve it
#: with an :class:`~pyfe4ai.utils.exceptions.FESecurityWarning`). Key shares
#: of versions <= 0.3.0 are refused before this applies (their flooding
#: derivation, see :func:`~pyfe4ai.utils.sampling_utils.check_flooding_version`),
#: so ``"warn"`` serves the explicit Shamir ``t < s`` shares of this version
#: under ``toy``. There is deliberately no environment variable: the opt-out
#: is per server object.
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


def check_pp_binding(pp: dict, dk: dict, fields: tuple, defaults: dict | None = None) -> None:
    """Raise unless key share *dk* was issued for public parameters with *pp*'s
    values of *fields* (see :func:`pp_binding`).

    Raises:
        FEKeyError: For a key share of other public parameters, or one without
            ``pp_binding`` (issue new key shares).
    """
    if not hmac.compare_digest(str(dk.get("pp_binding", "")), pp_binding(pp, fields, defaults)):
        raise FEKeyError(
            "the key share was issued for other public parameters (modulus, flooding or label hash): "
            "issue new key shares"
        )


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


def share_for_enrolled(
    dk: dict, sid: str, lst_sid_enrolled: list[str], threshold: int, modulus: gp.mpz,
    fields=None, cache=None, legacy_shamir: str = "refuse", profile=None, ring_n=None,
) -> tuple[dict, gp.mpz, str]:
    """What server *sid* multiplies, by what, for a partial decryption for *lst_sid_enrolled*.

    Returns ``(share, multiplier, tag)``. For a Shamir key share *dk*:
    ``(dk, lambda, digits(lambda))``, ``lambda`` the Lagrange coefficient
    at zero. With ``t < s`` (a key share of versions <= 0.3.0, or an
    explicit ``sharing="shamir"``) such a key share is refused, unless
    *legacy_shamir* is ``"warn"`` and the public parameters do not record
    the ``standard`` *profile*: then it is served with an
    :class:`~pyfe4ai.utils.exceptions.FESecurityWarning`. For a replicated
    one (``dk["pieces"]``): *dk* with the sum of the pieces of
    :func:`replicated_assignment` in place of the pieces (of the entries
    *fields* only, if given), multiplier 1 and a tag naming those pieces.
    Partial decryptions seed their flooding noise with the tag: they depend
    on the enrolled set only through it. A seeded replicated key share
    (``dk["seeds"]``, :func:`share_replicated_seeded`) gives what the
    replicated key share of the expanded pieces gives, tag included.

    Args:
        cache: An :class:`~collections.OrderedDict` that keeps the sums of
            the last :data:`SHARE_CACHE_SIZE` key shares and assignments,
            for key shares that are not modified once handed in (they are
            told apart by identity). The returned share must not be modified.
        ring_n: For seeded pieces: expand them natively when the extension
            has a ring context of ``(modulus, ring_n)``.

        legacy_shamir: ``"refuse"`` or ``"warn"`` (see
            :data:`LEGACY_SHAMIR_POLICIES`).
        profile: ``security_profile`` recorded in the public parameters
            (``None`` for those of versions <= 0.3.0).

    Raises:
        FEKeyError: If *dk* lacks a piece (or its seed) the server must add.
        FESchemeError: For a Shamir key share with ``t < s`` that is not
            to be served.
    """
    sid_v = share_points(dk["lst_sid"])
    xs = [sid_v[s] for s in lst_sid_enrolled]
    if "pieces" not in dk:
        if threshold < len(dk["lst_sid"]):
            from pyfe4ai.schemes.profiles import STANDARD

            message = (
                "server {}: a Shamir key share with t = {} < s = {} (the sharing of versions "
                "<= 0.3.0) lets partial decryptions reveal the decryption noise modulo small "
                "integers; re-share the threshold keys with this version (sharing="
                "'replicated')".format(sid, threshold, len(dk["lst_sid"]))
            )
            if legacy_shamir != "warn" or profile == STANDARD:
                raise FESchemeError(
                    message + "; during a migration only, a server created with "
                    "legacy_shamir='warn' serves it (not under the standard profile)"
                )
            warnings.warn(message, FESecurityWarning, stacklevel=3)
        lagrange = lagrange_coefficient_at_zero(sid_v[sid], xs, modulus)
        return dk, lagrange, gp.digits(lagrange)
    assigned = replicated_assignment(sid_v[sid], xs, list(sid_v.values()), threshold)
    fields = None if fields is None else tuple(fields)
    key = (id(dk), id(dk["pieces"]), tuple(assigned), fields)
    if cache is not None and key in cache and cache[key][0] is dk:
        cache.move_to_end(key)
        return cache[key][1]
    held = {tuple(piece["set"]): piece["share"] for piece in dk["pieces"]}
    seeded = dk.get("sharing") == "replicated-seeded"
    if not seeded and "seeds" in dk:
        raise FEKeyError("seeds in a key share whose sharing is not replicated-seeded")
    seeds = {tuple(entry["set"]): entry["seed"] for entry in dk.get("seeds", [])}
    if not held and not seeds:
        raise FEKeyError("the key share holds no pieces")
    missing = [T for T in assigned if T not in held and T not in seeds]
    if missing:
        raise FEKeyError("the key share lacks the pieces {}".format(missing))
    modulus = gp.mpz(modulus)

    def pick(piece):
        return piece if fields is None else {name: piece[name] for name in fields}

    if seeded:  # expand the assigned seeds and add them up, natively if possible
        ctx = None
        if ring_n is not None:
            from pyfe4ai.utils import native_backend

            ctx = native_backend.ring_context(modulus, ring_n)
        summed = _combine_pieces(
            _select(dk.get("layout"), fields), [pick(held[T]) for T in assigned if T in held],
            [(seeds[T], T) for T in assigned if T not in held], modulus, ctx,
        )
    else:
        total = _map_values(pick(next(iter(held.values()))), lambda _: gp.mpz(0))
        for T in assigned:
            total = _zip_values(total, pick(held[T]), lambda a, b: (a + gp.mpz(b)) % modulus)
        summed = _map_values(total, gp.digits)
    share = {key: value for key, value in dk.items() if key not in ("pieces", "seeds", "layout")}
    share.update(summed)
    out = share, gp.mpz(1), json.dumps(["replicated", [list(T) for T in assigned]])
    if cache is not None:
        cache[key] = (dk, out)  # dk kept alive: its id is not reused while cached
        while len(cache) > SHARE_CACHE_SIZE:
            cache.popitem(last=False)
    return out


def share_cache(obj) -> OrderedDict:
    """The :func:`share_for_enrolled` cache of scheme object *obj* (created on first use)."""
    cache = getattr(obj, "_enrolled_share_cache", None)
    if cache is None:
        cache = obj._enrolled_share_cache = OrderedDict()
    return cache


def share_points(lst_sid: list[str]) -> dict[str, int]:
    """Map each decryption server to its Shamir evaluation point.

    Points start at 1: the polynomial value at 0 is the shared secret itself,
    so no server may ever hold the share at x = 0.

    Args:
        lst_sid: Ordered list of all decryption-server identifiers.
    """
    return {sid: idx + 1 for idx, sid in enumerate(lst_sid)}


def validate_threshold(threshold: int, lst_sid: list[str]) -> None:
    """Reject repeated server ids and thresholds outside ``1 <= t <= s`` (or ``bool``).

    Args:
        threshold: Minimum number of shares needed for reconstruction.
        lst_sid: Ordered list of all decryption-server identifiers.
    """
    if len(set(lst_sid)) != len(lst_sid):
        raise FEValidationError("lst_sid contains duplicate server ids")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 1 <= threshold <= len(lst_sid):
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
