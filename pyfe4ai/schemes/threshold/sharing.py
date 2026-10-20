"""
Key sharing of the threshold schemes: Shamir, replicated and seed-expanded
replicated sharings, their dealing and the sums a server adds up for an
enrolled set.

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

The encodings of the key material (value lists or binary words) are in
:mod:`pyfe4ai.schemes.threshold.codec`, the caches of the sums for an
enrolled set in :mod:`pyfe4ai.schemes.threshold.store`.
"""

from __future__ import annotations

import itertools
import json
import random
import secrets
import warnings

import gmpy2 as gp

from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FESecurityWarning, FEValidationError
from pyfe4ai.core.versions import check_legacy_shamir
from pyfe4ai.schemes.threshold import store
from pyfe4ai.schemes.threshold.codec import _bytes_layout
from pyfe4ai.schemes.threshold.codec import _bytes_leaves
from pyfe4ai.schemes.threshold.codec import _check_key_polys
from pyfe4ai.schemes.threshold.codec import _has_bytes
from pyfe4ai.schemes.threshold.codec import _map_bytes
from pyfe4ai.schemes.threshold.codec import _poly_values
from pyfe4ai.schemes.threshold.codec import _unflatten_bytes
from pyfe4ai.schemes.threshold.codec import _value_bytes
from pyfe4ai.schemes.threshold.codec import binary_material
from pyfe4ai.schemes.threshold.codec import poly_kinds
from pyfe4ai.schemes.threshold.codec import strict_mpz
from pyfe4ai.schemes.threshold.codec import uniform_material

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


def check_rns_servers(s, factors) -> None:
    """Raise unless every factor of an RNS modulus exceeds the number of servers
    *s*: Shamir sharing over ``Z_q = prod F_{q_i}`` (CRT) needs the points
    ``1..s`` and their differences to be units mod every factor.

    Raises:
        FEValidationError: If ``s >= min q_i``.
    """
    if factors and int(s) >= min(int(f) for f in factors):
        raise FEValidationError("an RNS modulus needs every factor above the number of servers s = {}".format(s))


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
    :class:`~pyfe4ai.core.exceptions.FESecurityWarning` otherwise.

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
        from pyfe4ai.core.profiles import STANDARD

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


def _zq_native(modulus):
    """The extension when it does the ``Z_q`` arithmetic of the replicated
    sharings for *modulus* -- built with ``zq_sum``, backend not ``"python"``,
    ``q`` odd in ``[3, 2^253)``, no NTT ring needed --, else ``None``."""
    from pyfe4ai import backend as native_backend

    ext = native_backend.ext()
    if ext is None or native_backend.get_backend() == "python" or not hasattr(ext, "zq_sum"):
        return None
    q = int(modulus)
    return ext if 3 <= q < native_backend.MAX_NATIVE_Q and q % 2 else None


def _flatten_like(layout, material, out: list) -> bool:
    """Append the values of *material* to *out* in the order of *layout* (a
    :func:`piece_layout`); ``False`` if *material* has another layout (the
    Python code then reports it)."""
    if isinstance(layout, dict):
        if not isinstance(material, dict) or set(material) != set(layout):
            return False
        return all(_flatten_like(sub, material[key], out) for key, sub in layout.items())
    if isinstance(layout, list):
        if not isinstance(material, (list, tuple)) or len(material) != len(layout):
            return False
        return all(_flatten_like(sub, value, out) for sub, value in zip(layout, material))
    if layout is None:
        if isinstance(material, (dict, list, tuple)):
            return False
        out.append(material)
        return True
    if not isinstance(material, (list, tuple)) or len(material) != layout:
        return False
    out.extend(material)
    return True


def _unflatten(layout, flat: list):
    """*flat* (values in the order of :func:`_flatten_like`) in the structure of *layout*."""
    pos = 0

    def build(node):
        nonlocal pos
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        if isinstance(node, list):
            return [build(value) for value in node]
        if node is None:
            pos += 1
            return flat[pos - 1]
        pos += node
        return flat[pos - node:pos]

    out = build(layout)
    if pos != len(flat):
        raise FESchemeError("key pieces of different layouts")
    return out


def _share_replicated_native(material, sets: list, modulus: gp.mpz, binary: bool = False):
    """:func:`share_replicated` by the extension -- the same pieces for the
    same random bytes: each random value drawn as
    ``random.SystemRandom.randrange(0, q)`` draws it, from the call's DRBG
    (or the test bytes of :func:`~pyfe4ai.backend.random_input`;
    a test-installed DRBG seed, ``native_backend._seed_override``, makes every
    call read stream 0 of that seed, as for the other tensor calls: tests
    only), in the order of the Python code -- or ``None`` when that code
    applies.

    Raises:
        RuntimeError: If the test bytes run out (they are bounded; the DRBG
            is not).
    """
    ext = _zq_native(modulus)
    if ext is None:
        return None
    from pyfe4ai import backend as native_backend

    width = (int(modulus).bit_length() + 7) // 8
    if _has_bytes(material):  # binary key terms: dealt as bytes, out as bytes
        layout = _bytes_layout(material, width)
        if layout is None:
            return None
        flat, binary = b"".join(_bytes_leaves(material)), True
    else:
        layout = piece_layout(material)
        flat = []
        if not _flatten_like(layout, material, flat):
            return None
    n_random = len(sets) - 1
    # test bytes (None otherwise): a draw is rejected with probability below
    # 1/2, so 4 draws per value and 256 more run out with probability below
    # 2^-180 (Hoeffding), also for q just above a power of two
    draws = (len(flat) // width if isinstance(flat, bytes) else len(flat)) * n_random
    rnd = native_backend.random_input(width * (4 * draws + 256) if draws else 0)
    try:
        out = ext.zq_replicated_deal(
            int(modulus), flat, n_random, rnd, **native_backend.seed_kwargs(), **({"binary": True} if binary else {}))
    except TypeError:  # values other than ints and decimal strings: the Python code reads them
        return None
    except ValueError as exc:
        if binary and "binary values" in str(exc):  # bytes that are not whole words below q
            raise FEKeyError("malformed binary key material") from exc
        if rnd is None:
            raise
        raise RuntimeError(
            "native_backend.random_input: the {} test bytes ran out".format(len(rnd))
        ) from exc
    if binary:
        wb = _value_bytes(modulus)
        pieces = {T: _unflatten_bytes(layout, out[i + 1], wb) for i, T in enumerate(sets[1:])}
        pieces[sets[0]] = _unflatten_bytes(layout, out[0], wb)
        return pieces
    pieces = {T: _unflatten(layout, out[i + 1]) for i, T in enumerate(sets[1:])}
    pieces[sets[0]] = _unflatten(layout, out[0])
    return pieces


def _sum_pieces_native(template, pieces: list, modulus: gp.mpz):
    """The sum of *pieces* mod *modulus* as decimal strings in the layout of
    *template*, by the extension (the values of the Python code), or ``None``
    when that code applies (no pieces, other layouts or values)."""
    ext = _zq_native(modulus) if pieces else None
    if ext is None:
        return None
    layout = piece_layout(template)
    rows = []
    for piece in pieces:
        row = []
        if not _flatten_like(layout, piece, row):
            return None
        rows.append(row)
    try:
        return _unflatten(layout, ext.zq_sum(int(modulus), len(rows[0]), rows, []))
    except TypeError:
        return None


def _sum_binary_pieces(pieces: list, modulus: gp.mpz, skip_first: bool = False):
    """The sum mod q of binary pieces (:func:`~pyfe4ai.schemes.threshold.codec.binary_material`), leaf by leaf,
    as bytes: natively (``zq_sum``) where the extension is built, else in
    Python with the same values. With *skip_first* the first piece only gives
    the layout (the sum of the others; zeros if there are none).

    Raises:
        FESchemeError: For pieces of different layouts, or values that are not
            whole words below q.
    """
    from pyfe4ai.core.formats.binary import decode_values, encode_values

    q, wb = int(modulus), _value_bytes(modulus)
    ext = _zq_native(modulus)

    def add(*leaves):
        if not all(isinstance(leaf, (bytes, bytearray)) for leaf in leaves) or len({len(leaf) for leaf in leaves}) != 1:
            raise FESchemeError("key pieces of different layouts")
        count = len(leaves[0]) // wb
        if skip_first:
            leaves = leaves[1:]
            if not leaves:
                return bytes(count * wb)  # no assigned piece: 0
        try:
            if ext is not None:
                return ext.zq_sum(q, count, [bytes(leaf) for leaf in leaves], [], False, True)
            total = [0] * count
            for leaf in leaves:
                total = [a + b for a, b in zip(total, decode_values(leaf, q, count))]
            return encode_values(total, q)
        except ValueError as exc:
            raise FESchemeError("malformed binary key piece") from exc

    def walk(nodes):
        first = nodes[0]
        if isinstance(first, dict):
            if any(not isinstance(n, dict) or set(n) != set(first) for n in nodes):
                raise FESchemeError("key pieces of different layouts")
            return {key: walk([n[key] for n in nodes]) for key in first}
        if isinstance(first, (list, tuple)):
            if any(not isinstance(n, (list, tuple)) or len(n) != len(first) for n in nodes):
                raise FESchemeError("key pieces of different layouts")
            return [walk([n[i] for n in nodes]) for i in range(len(first))]
        return add(*nodes)

    return walk(pieces)


def share_replicated(
    material, threshold: int, points: list[int], modulus: gp.mpz, binary: bool = False
) -> dict[tuple[int, ...], object]:
    """Replicated sharing of every value in *material* (nested dicts and lists).

    Draws one piece ``r_T`` per ``(t - 1)``-subset ``T`` of *points*, of the
    layout of *material* (values as decimal strings), uniform subject to
    ``sum_T r_T = material mod modulus``. The server at point ``x`` receives
    the pieces with ``x`` not in ``T``: any ``t`` servers hold every piece,
    any ``t - 1`` servers ``C`` miss exactly ``r_C``, which is uniform given
    the others. With ``t = s`` this is additive sharing (one piece each).
    With the native backend the pieces are drawn and the correction computed
    by the extension, with the values of the Python code for the same random
    bytes (``_share_replicated_native``). With *binary* every
    polynomial of a piece is bytes (:func:`~pyfe4ai.schemes.threshold.codec.binary_material`; the same values).

    Returns:
        ``{T: piece}`` for every index set ``T``.
    """
    sets = replicated_sets(points, threshold)
    modulus = gp.mpz(modulus)
    material = uniform_material(material, modulus)  # a mix of bytes and value lists: dealt as values
    if _has_bytes(material):  # binary key terms are dealt as binary pieces
        binary = True
    pieces = _share_replicated_native(material, sets, modulus, binary)
    if pieces is not None:
        return pieces
    if _has_bytes(material):  # the Python code on their values
        from pyfe4ai.core.formats.binary import decode_values

        width = _value_bytes(modulus)

        def decode(leaf):
            try:
                return decode_values(leaf, modulus, len(leaf) // width)
            except ValueError as exc:
                raise FEKeyError("malformed binary key material") from exc

        material = _map_bytes(material, decode)
    if binary:
        return {T: binary_material(piece, modulus) for T, piece in share_replicated(material, threshold, points, modulus).items()}

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
    from pyfe4ai.primitives.hashing import hash_to_zq_material, hash_to_zq_vector

    seed = _check_seed(seed, index_set)
    modulus = gp.mpz(modulus)
    _check_context(ctx, modulus)
    q_text = gp.digits(modulus)

    def leaf(keys, path, length):
        v, count = _leaf_input(seed, index_set, path, q_text), 1 if length is None else length
        if ctx is not None and count:
            from pyfe4ai import backend as native_backend

            values = ctx.unpack(native_backend.ext().hash_to_zq(ctx.cap, hash_to_zq_material(v, _SEED_DOMAIN), count))
        else:
            values = hash_to_zq_vector(v, modulus, count, domain=_SEED_DOMAIN)
        return values[0] if length is None else values

    return _map_layout(_select(layout, fields), leaf)


def _combine_pieces(layout, explicit: list, seeded: list, modulus, ctx=None, subtract=False):
    """``sum(explicit) + sum(seeded)`` (``- sum(seeded)`` if *subtract*) mod
    *modulus*, leaf by leaf, as decimal strings of *layout*.

    *explicit* are pieces of *layout* (ints, mpz or decimal strings),
    *seeded* ``(seed, T)`` pairs of :func:`expand_seeded_piece`. With the
    native backend every leaf is expanded and added up by the extension
    (``zq_sum``: the "shake256-wide" ``hash_to_zq`` and the sums, any odd
    ``q`` below ``2^253``), with the values of the Python code; *ctx*, a
    native ring context of *modulus*, is only checked.
    """
    from pyfe4ai.primitives.hashing import hash_to_zq_material, hash_to_zq_vector

    seeded = [(_check_seed(seed, T), T) for seed, T in seeded]
    modulus = gp.mpz(modulus)
    _check_context(ctx, modulus)
    q_text = gp.digits(modulus)
    ext = _zq_native(modulus)

    def leaf(keys, path, length):
        count = 1 if length is None else length
        rows = []
        for piece in explicit:
            value = piece
            for key in keys:
                value = value[key]
            rows.append([value] if length is None else value)
        if ext is not None and all(len(row) == count for row in rows):
            materials = [hash_to_zq_material(_leaf_input(seed, T, path, q_text), _SEED_DOMAIN) for seed, T in seeded]
            try:
                out = ext.zq_sum(int(modulus), count, rows, materials, subtract)
                return out[0] if length is None else out
            except TypeError:  # values other than ints and decimal strings: read below
                pass
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


def deal_replicated(material, sharing: str, threshold: int, lst_sid: list, modulus, binary: bool = False,
                    ctx=None) -> dict:
    """What each server stores of a replicated sharing of *material* among
    *lst_sid* (points of :func:`share_points`): ``{sid: {"pieces": ...}}``
    (:func:`share_replicated`, :func:`pieces_held`) for ``"replicated"``, or
    ``{sid: {"pieces", "seeds", "layout"}}`` (:func:`share_replicated_seeded`,
    :func:`seeded_pieces_held`) for ``"replicated-seeded"``. *binary* deals
    binary pieces (``"replicated"``), *ctx* expands the seeded pieces natively
    (``"replicated-seeded"``).

    Raises:
        FEValidationError: For another sharing.
    """
    sid_v = share_points(lst_sid)
    points = list(sid_v.values())
    if sharing == "replicated":
        pieces = share_replicated(material, threshold, points, modulus, binary=binary)
        return {sid: {"pieces": pieces_held(pieces, sid_v[sid])} for sid in lst_sid}
    if sharing == "replicated-seeded":
        dealt = share_replicated_seeded(material, threshold, points, modulus, ctx=ctx)
        return {sid: seeded_pieces_held(dealt, sid_v[sid]) for sid in lst_sid}
    raise FEValidationError("not a replicated sharing: {!r}".format(sharing))


def share_polys(polys: list, threshold: int, lst_sid: list, modulus, ring_n: int, binary: bool = False,
                ctx=None) -> list:
    """Shamir shares of every coefficient of *polys* (``ring_n`` values or
    ``ring_n`` binary words each) among *lst_sid* (points of
    :func:`share_points`): ``[{sid: share polynomial}]``, as decimal strings,
    or bytes with *binary*. With *ctx*, a native ring context of *modulus*,
    in one native call (coefficients drawn as :func:`share_secret` draws
    them, in the same order); else :func:`share_secret` per coefficient.
    Every polynomial must be ``ring_n`` values or ``ring_n`` words on its own
    (binary polynomials that are whole words only together would be dealt
    misaligned).

    Raises:
        FEKeyError: For malformed key material.
    """
    from pyfe4ai.core.formats.binary import encode_values

    _check_key_polys(polys, modulus, ring_n)
    sid_v = share_points(lst_sid)
    points = list(sid_v.values())
    shared = None if ctx is None else _share_polys_native(ctx, polys, threshold, points, lst_sid, modulus, binary)
    if shared is not None:
        return shared

    def share_poly(coeffs) -> dict:
        shares = [share_secret(v, threshold, points, modulus) for v in _poly_values(coeffs, modulus, ring_n)]
        if binary:
            return {sid: encode_values([sh[sid_v[sid]] for sh in shares], modulus) for sid in lst_sid}
        return {sid: [gp.digits(sh[sid_v[sid]]) for sh in shares] for sid in lst_sid}

    return [share_poly(poly) for poly in polys]


def _share_polys_native(ctx, polys: list, threshold: int, points: list, lst_sid: list, modulus,
                        binary: bool = False) -> list:
    """:func:`share_polys` in one native call of the extension's ``shamir_share``
    (every polynomial ``ring_n`` values or words of the ring of *ctx*: checked
    here too, for callers other than :func:`share_polys`)."""
    from pyfe4ai import backend as native_backend

    _check_key_polys(polys, modulus, ctx.ring_n)
    width = (int(modulus).bit_length() + 7) // 8
    if all(isinstance(poly, (bytes, bytearray)) for poly in polys):  # binary key terms
        flat = b"".join(polys)
        polys = [range(len(poly) // width) for poly in polys]  # their lengths in values
    elif any(isinstance(poly, (bytes, bytearray)) for poly in polys):
        # bytes iterated as values would be other coefficients: the callers make the material uniform
        raise FEKeyError("key material mixing binary and value polynomials")
    else:
        flat = [v for poly in polys for v in poly]
    rnd = native_backend.random_input(4 * width * len(flat) * (threshold - 1) + 64)
    try:
        out = native_backend.ext().shamir_share(
            ctx.cap, flat, threshold, points, rnd,
            **native_backend.seed_kwargs(), **({"binary": True} if binary else {}),
        )
    except TypeError as exc:  # values that are not ints / decimal strings
        raise FEKeyError("malformed key material") from exc
    except ValueError as exc:
        if rnd is not None and "not enough random bytes" in str(exc):  # a test's random bytes, not the key's
            raise RuntimeError(
                "native_backend.random_input: the {} test bytes ran out".format(len(rnd))) from exc
        raise FEKeyError("malformed key material") from exc  # values not below q, misaligned words
    unit = width if binary else 1
    shared, start = [], 0
    for poly in polys:
        shared.append({sid: out[i][start * unit:(start + len(poly)) * unit] for i, sid in enumerate(lst_sid)})
        start += len(poly)
    return shared


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
    :class:`~pyfe4ai.core.exceptions.FESecurityWarning`. For a replicated
    one (``dk["pieces"]``): *dk* with the sum of the pieces of
    :func:`replicated_assignment` in place of the pieces (of the entries
    *fields* only, if given), multiplier 1 and a tag naming those pieces.
    Partial decryptions seed their flooding noise with the tag: they depend
    on the enrolled set only through it. A seeded replicated key share
    (``dk["seeds"]``, :func:`share_replicated_seeded`) gives what the
    replicated key share of the expanded pieces gives, tag included.

    Args:
        cache: An :class:`~collections.OrderedDict` that keeps the sums of
            the last :data:`~pyfe4ai.schemes.threshold.store.SHARE_CACHE_SIZE` key shares and assignments,
            for key shares that are not modified once handed in (they are
            told apart by identity). The returned share must not be modified.
        ring_n: The ring dimension of a Ring-LWE layer; the native sums
            (``_zq_native``) need no ring, so it is only checked against
            a native ring context of the modulus where there is one.

        legacy_shamir: ``"refuse"`` or ``"warn"`` (see
            :data:`~pyfe4ai.core.versions.LEGACY_SHAMIR_POLICIES`).
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
        check_legacy_shamir(sid, threshold, len(dk["lst_sid"]), legacy_shamir, profile)
        lagrange = lagrange_coefficient_at_zero(sid_v[sid], xs, modulus)
        return dk, lagrange, gp.digits(lagrange)
    assigned = replicated_assignment(sid_v[sid], xs, list(sid_v.values()), threshold)
    fields = None if fields is None else tuple(fields)
    key = (id(dk), id(dk["pieces"]), tuple(assigned), fields)
    if cache is not None:
        hit = store.cached_share(cache, key, dk)
        if hit is not None:
            return hit
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
            from pyfe4ai import backend as native_backend

            ctx = native_backend.ring_context(modulus, ring_n)
        summed = _combine_pieces(
            _select(dk.get("layout"), fields), [pick(held[T]) for T in assigned if T in held],
            [(seeds[T], T) for T in assigned if T not in held], modulus, ctx,
        )
    else:
        try:  # the layout of the server's pieces (it may be assigned none: the sum is then 0)
            template = pick(next(iter(held.values())))
            chosen = [pick(held[T]) for T in assigned]
        except (KeyError, TypeError) as exc:
            raise FEKeyError("malformed key pieces") from exc
        kinds = poly_kinds([template] + chosen)
        if "other" in kinds or len(kinds) > 1:
            raise FEKeyError("malformed key pieces: polynomials of one kind (value lists or bytes) only")
        if kinds == {"bytes"}:  # binary key shares: bytes in, bytes out
            summed = _sum_binary_pieces([template] + chosen, modulus, skip_first=True)
        else:  # natively where possible, with the values of the Python code
            summed = _sum_pieces_native(template, chosen, modulus)
            if summed is None:
                try:
                    total = _map_values(template, lambda _: gp.mpz(0))
                    for piece in chosen:
                        total = _zip_values(total, piece, lambda a, b: (a + strict_mpz(b)) % modulus)
                except (ValueError, TypeError) as exc:
                    raise FEKeyError("malformed key pieces") from exc
                summed = _map_values(total, gp.digits)
    share = {key: value for key, value in dk.items() if key not in ("pieces", "seeds", "layout")}
    share.update(summed)
    out = share, gp.mpz(1), json.dumps(["replicated", [list(T) for T in assigned]])
    if cache is not None:
        store.keep_share(cache, key, dk, out)
    return out
