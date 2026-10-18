"""Core cryptographic primitives: safe random generation, group/prime generation, and hashing."""

import functools
import random
import hashlib

import gmpy2 as gp

_CSPRNG = random.SystemRandom()


def _random(maximum: gp.mpz, bits: int = 256):
    """generate random integer with specified bits

    Args:
        maximum (gmpy2.mpz): the maximum threshold of generated integer
        bits (int, optional): the specified bit size of generated integer.
            Defaults to 256.

    Returns:
        gmpy2.mpz: the generated random big integer
    """
    r = gp.mpz(_CSPRNG.getrandbits(bits))
    while r > maximum:
        r = gp.mpz(_CSPRNG.getrandbits(bits))
    return r


def _random_prime(bits):
    r = gp.mpz(_CSPRNG.getrandbits(bits))
    r = gp.bit_set(r, bits - 1)
    return gp.next_prime(r)


def _small_odd_primes(limit: int) -> list:
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(3, limit + 1) if sieve[i]]


#: Odd primes below 2^16 with the inverses of 2 and 4 modulo each.
_SIEVE = [(sp, (sp + 1) // 2, pow(4, -1, sp)) for sp in _small_odd_primes(1 << 16)]
_SIEVE_WINDOW = 1 << 14
#: Below this size the sieve primes could be candidates themselves.
_SIEVE_MIN_BITS = 64


def _safe_prime(bits: int) -> tuple:
    """Return ``(p, q)`` with ``p = 2q + 1`` of exactly *bits* bits, both prime.

    Candidates ``q = q0 + 2k`` (``q0`` random) are sieved so that neither
    ``q`` nor ``2q + 1`` has a prime factor below 2^16, then filtered with a
    base-2 Fermat test; survivors must pass :func:`gmpy2.is_prime` (BPSW plus
    Miller-Rabin rounds) for both ``q`` and ``p``.
    """
    if bits < _SIEVE_MIN_BITS:
        while True:
            q = _random_prime(bits - 1)
            p = q * 2 + 1
            # next_prime can step past 2^(bits - 1), giving a (bits + 1)-bit p
            if p.bit_length() == bits and gp.is_prime(p) and gp.is_prime(q):
                return p, q
    while True:
        q0 = gp.mpz(_CSPRNG.getrandbits(bits - 1)) | (gp.mpz(1) << (bits - 2)) | 1
        alive = bytearray([1]) * _SIEVE_WINDOW
        for sp, inv2, inv4 in _SIEVE:
            r = int(q0 % sp)
            for k in ((-r * inv2) % sp, (-(2 * r + 1) * inv4) % sp):  # sp | q, sp | 2q + 1
                alive[k::sp] = bytes(len(range(k, _SIEVE_WINDOW, sp)))
        for k in range(_SIEVE_WINDOW):
            if not alive[k]:
                continue
            q = q0 + 2 * k
            if q.bit_length() != bits - 1:
                break
            p = 2 * q + 1
            if gp.powmod(2, q - 1, q) != 1 or gp.powmod(2, p - 1, p) != 1:
                continue
            if gp.is_prime(q, 30) and gp.is_prime(p, 30):
                return p, q


def generate_group_primes(bits: int) -> tuple:
    """
    Generates two safe prime numbers with the restriction:p=2q+1.

    Args:
        bits (int): security parameters

    Returns:
        tuple: p, q
    """
    return _safe_prime(bits)


#: Safe primes of the RFC 3526 MODP groups 14 and 15, by bit length:
#: ``p = 2^b - 2^(b-64) - 1 + 2^64 (floor(2^(b-130) pi) + c)``. Used instead
#: of a fresh safe prime when one of these sizes is requested -- generating a
#: 3072-bit safe prime takes minutes, and at these sizes a fixed, published
#: group gives nothing to precomputation.
#:
#: :meta hide-value:
RFC3526_PRIMES = {
    2048: gp.mpz(
        "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
        "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
        "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
        "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
        "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
        "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
        "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
        16,
    ),
    3072: gp.mpz(
        "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
        "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
        "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
        "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
        "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
        "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
        "3995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33"
        "A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7"
        "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864"
        "D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E2"
        "08E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF",
        16,
    ),
}


def _param_generator(bits: int, r: int = 2):
    """Perform the _param_generator operation.

        Args:
            bits: Security parameter (bit length).
            r: Cofactor (default 2).
    """
    if r == 2 and bits in RFC3526_PRIMES:
        p = RFC3526_PRIMES[bits]
        return p, (p - 1) // 2, r
    p, q = _safe_prime(bits)
    return p, q, r


def _g_generator(bits, p, r):
    """Perform the _g_generator operation.

        Args:
            bits: Security parameter (bit length).
            p: Prime modulus.
            r: Cofactor (default 2).
    """
    while True:
        h = _random(p, bits)
        g = gp.powmod(h, r, p)
        if not g == 1:
            break
    return g


def group_generator_threshold_fe(bits: int, r: int = 2):

    """Generate a safe prime and generator for threshold FE.

        Args:
            bits: Security parameter (bit length).
            r: Cofactor (default 2).
    """
    p, _, r = _param_generator(bits, r=r)
    g = _g_generator(bits, p, r)
    return p, g


def is_dl_group(p, q, g, bits: int, cofactor: int = 2) -> bool:
    """Whether ``p = cofactor q + 1`` is a *bits*-bit prime, ``q`` is a prime of
    about ``bits`` bits (at most the cofactor's bits fewer) and ``g`` generates
    the order-``q`` subgroup of ``Z_p^*``; at a size with an RFC 3526 prime,
    ``p`` must be that prime (another safe prime could carry a trapdoor).

    Used to check discrete-log parameters loaded from the parameter cache.
    """
    p, q, g = gp.mpz(p), gp.mpz(q), gp.mpz(g)
    if int(cofactor) == 2 and bits in RFC3526_PRIMES and p != RFC3526_PRIMES[bits]:
        return False
    return (
        p.bit_length() == bits
        and q.bit_length() >= bits - int(cofactor).bit_length()
        and q > 2
        and p == cofactor * q + 1
        and gp.is_prime(q, 40)
        and gp.is_prime(p, 40)
        and 1 < g < p
        and gp.powmod(g, q, p) == 1
    )


@functools.lru_cache(maxsize=1)
def _small_primorial() -> gp.mpz:
    """Product of the primes below 2^16 (trial division of cached moduli)."""
    out, p = gp.mpz(1), gp.mpz(2)
    while p < 1 << 16:
        out *= p
        p = gp.next_prime(p)
    return out


def is_dcr_modulus(n, prime_bits: int) -> bool:
    """Whether *n* can be a product of two distinct *prime_bits*-bit primes:
    odd, composite, not a square, without prime factors below 2^16, of
    ``2 prime_bits - 1`` or ``2 prime_bits`` bits.

    Used to check DCR moduli loaded from the parameter cache (their
    factorisation is never stored, so it cannot be checked further).
    """
    n = gp.mpz(n)
    return (
        n.bit_length() in (2 * prime_bits - 1, 2 * prime_bits)
        and n % 2 == 1
        and gp.gcd(n, _small_primorial()) == 1
        and not gp.is_prime(n, 40)
        and not gp.is_square(n)
    )


def group_generator_fe(bits: int, r: int = 2) -> tuple:
    """Generate a safe-prime group ``(p, q, r, g)`` for the DDH schemes.

    ``p = r q + 1`` is a *bits*-bit prime (for ``r = 2`` the RFC 3526 prime
    at 2048 and 3072 bits) and ``g`` generates the order-``q`` subgroup of
    ``Z_p^*``.

    Args:
        bits: Bit length of ``p`` (``sec_param``; not a security level).
        r: Cofactor (default 2).
    """
    p, q, r = _param_generator(bits, r=r)
    g = _g_generator(bits, p, r)
    return p, q, r, g


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
    see :mod:`pyfe4ai.utils.native_backend`); the rejection modes use their
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
    material = hash_to_zq_material(v, domain, mode)
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
