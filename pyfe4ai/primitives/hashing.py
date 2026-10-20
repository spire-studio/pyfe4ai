"""Hashing to groups and to ``Z_q``: random-oracle hashes into the quadratic residues
of ``Z_p^*`` and into ``Z_{N^2}^*``, the ALS16 Paillier base, and the label hashes
(:data:`LABEL_HASHES`) of the MCFE schemes."""

import hashlib

import gmpy2 as gp


def hash_to_qr_group(v: str, p: gp.mpz, index: int = 0, domain: str = "") -> gp.mpz:
    """Hash a string into the subgroup of quadratic residues of ``Z_p^*``.

    For a safe prime ``p = 2q + 1`` this is the prime-order-``q`` subgroup
    used by the DDH schemes. The digest is expanded with SHAKE-256 to
    ``bits(p) + 128`` bits (so the reduction mod ``p`` is statistically
    close to uniform) and squared; the discrete logarithm of the result
    with respect to any fixed generator is unknown, as required when the
    hash is modelled as a random oracle into the group.

    Args:
        v: Input string.
        p: Safe-prime modulus.
        index: Output index, giving independent outputs for the same *v*.
        domain: Domain-separation tag.
    """
    p = gp.mpz(p)
    n_bytes = (int(p).bit_length() + 128 + 7) // 8
    counter = 0
    while True:
        material = "|".join(
            [domain, str(index), str(counter), str(len(v)), v]
        ).encode("utf-8")
        digest = hashlib.shake_256(material).digest(n_bytes)
        h = gp.powmod(gp.mpz(int.from_bytes(digest, "big")) % p, 2, p)
        if h > 1:
            return h
        counter += 1


def _shake_expand(domain: str, index: int, counter: int, v: str, n_bytes: int) -> int:
    """SHAKE-256 expansion of an unambiguous encoding of ``(domain, index, counter, v)``."""
    material = "|".join([domain, str(index), str(counter), str(len(v)), v]).encode("utf-8")
    return int.from_bytes(hashlib.shake_256(material).digest(n_bytes), "big")


def paillier_base(seed: str, n_mod) -> gp.mpz:
    """The base ``g = H(seed)^(2N) mod N^2`` of the ALS16 DCR schemes.

    A random ``2N``-th residue, as the scheme takes it; a parameter cache
    stores only the seed, so a planted cache cannot choose ``g`` (e.g. of
    order 2, or ``1 + N``), only another pseudo-random one.
    """
    n_mod = gp.mpz(n_mod)
    bytes.fromhex(seed)  # a hex string
    h = hash_to_zn2_star(seed, n_mod, domain="pyfe4ai/als16-dcr/g")
    return gp.powmod(h, 2 * n_mod, n_mod * n_mod)


def hash_to_zn2_star(v: str, n_mod: gp.mpz, domain: str = "") -> gp.mpz:
    """Full-domain hash of a string into ``Z_{N^2}^*`` (random oracle).

    The digest is expanded to ``bits(N^2) + 128`` bits, so the reduction
    mod ``N^2`` is statistically close to uniform; outputs not coprime to
    ``N`` (which would factor ``N``) are re-hashed with a counter.

    Args:
        v: Input string.
        n_mod: RSA modulus ``N``.
        domain: Domain-separation tag.
    """
    n_mod = gp.mpz(n_mod)
    n_square = n_mod * n_mod
    n_bytes = (int(n_square).bit_length() + 128 + 7) // 8
    counter = 0
    while True:
        h = gp.mpz(_shake_expand(domain, 0, counter, v, n_bytes)) % n_square
        if h > 1 and gp.gcd(h, n_mod) == 1:
            return h
        counter += 1


#: How :func:`hash_to_zq_vector` maps a string to ``Z_q`` (the ``label_hash``
#: setting of the MCFE schemes):
#:
#: * ``"shake256-wide"``: one SHAKE-256 stream cut into blocks of
#:   ``bits(q) + 128`` bits, each reduced mod q (statistical distance below
#:   ``2^-128`` per value); the only mode up to v0.3.0
#:   (:data:`LEGACY_LABEL_HASH`);
#: * ``"shake256-rejection"`` / ``"shake128-rejection"``: rejection sampling
#:   mod q (exactly uniform) from a SHAKE-256 / SHAKE128 stream, see
#:   :func:`zq_rejection_parameters`; ``"shake128-rejection"`` is the MCFE
#:   key generators' default (:data:`DEFAULT_LABEL_HASH`).
LABEL_HASHES = ("shake256-wide", "shake256-rejection", "shake128-rejection")

#: The hash of versions <= 0.3.0, of :func:`hash_to_zq_vector` without a mode
#: (e.g. the seeded Ring-LWE ``a``) and of MCFE public parameters and keys
#: without a ``label_hash`` field.
LEGACY_LABEL_HASH = "shake256-wide"

#: The ``label_hash`` the MCFE key generators use unless configured.
DEFAULT_LABEL_HASH = "shake128-rejection"

#: The native backend's code of each label-hash mode.
LABEL_HASH_CODES = {mode: code for code, mode in enumerate(LABEL_HASHES)}


def check_label_hash(mode) -> str:
    """Return *mode* if it is one of :data:`LABEL_HASHES`.

    Raises:
        ValueError: Otherwise.
    """
    if mode not in LABEL_HASHES:
        raise ValueError("unknown label hash {!r}; one of {}".format(mode, ", ".join(LABEL_HASHES)))
    return mode


def hash_to_zq_material(v: str, domain: str = "", mode: str = LEGACY_LABEL_HASH) -> bytes:
    """The XOF input of :func:`hash_to_zq_vector` (also hashed natively,
    see :mod:`pyfe4ai.backend`); the rejection modes use their
    own tag, so no two modes share an input."""
    tag = "zq-vector" if check_label_hash(mode) == "shake256-wide" else "zq-rej/v1"
    return "|".join([domain, tag, str(len(v)), v]).encode("utf-8")


def zq_rejection_parameters(q) -> tuple:
    """``(nb, limit)`` of the rejection modes of :func:`hash_to_zq_vector`:
    candidates are ``nb = ceil(bits(q) / 8) + 1`` big-endian bytes of the
    stream, accepted below ``limit = floor(2^(8 nb) / q) q`` (more than
    ``1 - 2^-8`` of them) and reduced mod q. ``q | limit``, so an accepted
    candidate is uniform mod q."""
    q = int(q)
    nb = (q.bit_length() + 7) // 8 + 1
    return nb, ((1 << (8 * nb)) // q) * q


def _xof(mode: str):
    return hashlib.shake_128 if mode == "shake128-rejection" else hashlib.shake_256


def _rejection_sample(material: bytes, q, length: int, mode: str) -> list:
    """The first *length* accepted candidates of the XOF(material) stream, mod q."""
    if length <= 0:
        return []  # as the shake256-wide mode and the native kernels
    q = gp.mpz(q)
    nb, limit = zq_rejection_parameters(q)
    limit_bytes = limit.to_bytes(nb, "big")  # equal-length big-endian bytes compare as numbers
    from_bytes = _MPZ_FROM_BYTES or (lambda b, order: gp.mpz(int.from_bytes(b, order)))
    xof = _xof(mode)
    n_bytes = nb * (length + length // 64 + 16)
    while True:
        stream = xof(material).digest(n_bytes)  # a longer digest extends a shorter one
        out = []
        for i in range(0, n_bytes - nb + 1, nb):
            block = stream[i:i + nb]
            if block < limit_bytes:
                out.append(from_bytes(block, "big") % q)
                if len(out) == length:
                    return out
        n_bytes *= 2


def hash_to_zq_vector(v: str, q: gp.mpz, length: int, domain: str = "", mode: str = LEGACY_LABEL_HASH) -> list:
    """Hash a string to a vector of *length* elements of ``Z_q`` (random oracle).

    ``mode`` is one of :data:`LABEL_HASHES`. ``"shake256-wide"`` (the
    default here, :data:`LEGACY_LABEL_HASH`) cuts one SHAKE-256 output stream into *length* blocks of ``bits(q) + 128`` bits, each
    reduced mod q, hence statistically close to uniform. Versions <= 0.2.0
    made one SHAKE-256 call per coordinate, so their outputs (and the
    ciphertexts of the schemes that hash labels this way) differ. The
    rejection modes are exactly uniform and read fewer XOF bytes
    (:func:`zq_rejection_parameters`).

    Args:
        v: Input string.
        q: Modulus.
        length: Number of coordinates.
        domain: Domain-separation tag.
        mode: Label-hash mode.
    """
    return _zq_values(hash_to_zq_material(v, domain, mode), q, length, mode)


def hash_to_zq_values(material: bytes, q: gp.mpz, length: int, mode: str = LEGACY_LABEL_HASH) -> list:
    """The *length* values of :func:`hash_to_zq_vector` for its XOF input
    *material* (:func:`hash_to_zq_material` with the same *mode*); the
    reference of the backend's
    :meth:`~pyfe4ai.backend.interface.Ring.hash_to_zq`.

    Raises:
        ValueError: For an unknown *mode*.
    """
    return _zq_values(material, q, length, check_label_hash(mode))


def _zq_values(material: bytes, q: gp.mpz, length: int, mode: str) -> list:
    if mode != "shake256-wide":
        return _rejection_sample(material, q, length, mode)
    q = gp.mpz(q)
    n_bytes = (int(q).bit_length() + 128 + 7) // 8
    stream = hashlib.shake_256(material).digest(length * n_bytes)
    if _MPZ_FROM_BYTES is not None:  # the same values, without a Python int per block
        return [_MPZ_FROM_BYTES(stream[i:i + n_bytes], "big") % q for i in range(0, length * n_bytes, n_bytes)]
    return [
        gp.mpz(int.from_bytes(stream[k * n_bytes:(k + 1) * n_bytes], "big")) % q
        for k in range(length)
    ]


_MPZ_FROM_BYTES = getattr(gp.mpz, "from_bytes", None)  # gmpy2 >= 2.2
