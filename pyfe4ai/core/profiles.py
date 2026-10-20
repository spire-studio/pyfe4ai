"""Security profiles: parameter presets for the key generators.

The ``security_profile`` entry of a key generator's config selects one:

``"toy"`` (default)
    The library's small defaults -- fast enough for tests and demos, and
    **not secure**. A config without the entry, or with ``"toy"``, is used
    exactly as given.

``"standard"``
    Parameters aiming at 128-bit classical security where this prototype
    can reach it, chosen per scheme family:

    * discrete log (DDH, Damgard / MDDH and their MIFE / MCFE variants): a
      3072-bit safe-prime group, the RFC 3526 MODP group 15 (NIST SP 800-57
      Part 1 Rev. 5, Table 2: L = 3072 for 128 bits). The decentralised
      ``mcfe/ddh_decentralized`` then needs a DCR modulus of about 6410 bits
      for combining key shares, whose safe primes take minutes to generate;
    * DCR (Paillier variants): ``N`` of about 3072 bits (two 1536-bit safe
      primes);
    * DDH over ristretto255 (``*/ddh_ristretto255``): the group is fixed, of
      order about ``2^252`` (Pollard rho about ``2^126``, the 128-bit class of
      X25519 / Ed25519); the profile sets nothing;
    * Ring-LWE: ``kappa = 128`` and the smallest ring dimension whose derived
      modulus ``q`` meets :data:`~pyfe4ai.primitives.lattice.ring_lwe_utils.RING_LWE_128_LOG_Q`,
      a lattice-estimator table (the method of the LWE tables) read at the
      smallest width among the RLWE problems of the scheme's proof (about 8
      for the BMMS21 parameters of Sec. 4.2 and 5; ``log2 q`` <= 31 / 57 /
      111 / 218 / 435 / 872 for n = 1024 ... 32768 at that width). It replaces the
      Homomorphic Encryption Standard's bound, which this estimator puts at
      about 125 bits; ``sife/fullysec_ring_lwe`` (BMMS21 Sec. 5, adaptive) also
      picks its number ``m`` of key polynomials: the smallest one whose ``q``
      meets that bound;
    * dense LWE, the ABMPSW20 MCFE family (``mcfe/lwe``, ``fullysec_lwe``,
      ``lwe_threshold``, ``lwe_decentralized``): ``lwe_n = "auto"``, the
      smallest dimension ``n0`` whose derived ``q`` meets
      :data:`~pyfe4ai.primitives.lattice.lwe_utils.DENSE_LWE_128_LOG_Q` -- a lattice
      estimator table for LWE(n0, q, ceil(2 sqrt(n0))) with a uniform secret,
      the problem of the scheme's security proof. These schemes have no public
      matrix: a label is hashed to ``n0 + m0`` values and each ciphertext is
      one inner product, cheaper than a 3072-bit DDH ciphertext. The
      decentralised variant combines key shares under a DCR modulus of at
      least 3072 bits (``STANDARD_DCR_PRIME_BITS`` per prime), so its key
      derivation is not post-quantum;
    * LWE with a seeded public matrix, the ABDP15 IPFE (``sife/lwe``,
      ``mife/lwe``): ``lwe_n = "auto"``, the smallest dimension whose
      ``(q, sigma')`` meets :data:`~pyfe4ai.primitives.lattice.lwe_utils.WIDE_LWE_128_LOG_RATIO`,
      a lattice estimator table for LWE with a uniform secret, read at the
      error width ``sigma' = sigma / (1 + B sqrt(eta))`` of the LWE problem the
      security proof reduces to (``B = 2 X``, or ``4 X`` in the slots of
      ``mife/lwe``; ABDP15 full version, Thm. 4.4, with the width derived in
      the proof of Thm. 6.6; see :mod:`pyfe4ai.schemes.common.abdp15_lwe`).
      That gives ``lwe_n`` of about 900 to 1700 for small bounds; reading the
      table at the public key's own ``sigma`` would give 14 to 40 percent
      less for ``X = Y >= 8``. ``A``
      (``m ~ lwe_n log2 q`` rows, 10^8 to 4 * 10^8 entries) is published as a
      seed; ``matrix_mode`` chooses between keeping it expanded in memory
      (about 0.1 to 0.3 s per encryption, 2 to 6 GB) and expanding it for
      every batch (``"stream"``: seconds per batch, about 0.5 GB). The noise
      ``E`` of the public keys is sampled exactly (``"facct"``;
      ``"rounded"`` is refused);
    * pairing-based schemes: BLS12-381 (the Rust extension) -- about 120 to
      126-bit security after the exTNFS attacks on its 381-bit field
      (Barbulescu and Duquesne, J. Cryptol. 2019), **below** the 128-bit
      target; a :class:`~pyfe4ai.core.exceptions.FESecurityWarning` says so.

    Refused, because no parameter set is both secure and usable in pure
    Python: ``sife/fullysec_lwe`` and ``mife/fullysec_lwe`` (ALS16: at a
    dimension the lattice estimator puts at 128 bits, n = 3328 to 4352 for
    ``X = Y <= 50`` and ``eta <= 10``, and with ``m = 4 n log2 q`` as its
    Theorem 2 requires, the public matrix has 1 to 2 * 10^10 entries and a
    ciphertext 2.9 to 4.5 * 10^6 coordinates of 219 to 258 bits),
    ``mife/lwe_threshold``. At ``sigma'`` its flooding-enlarged ``q``
    (``stat_param = 128``, two clients) needs an LWE dimension at the very end
    of the table (5888 to 6144, ``A`` of 6.5 to 7.8 * 10^9 entries) for a few
    small bounds, e.g. ``X = Y`` in {1, 2, 3, 8, 10} at ``eta = 1``, and
    beyond it for most others (``X = Y`` = 4 to 6, ``X = Y >= 12``, most
    ``eta >= 2``; about 6500 for ``X = Y = 50``). The opt-in KL accounting
    of the flooding noise (``t = s`` only, with an explicit budget and a
    weaker guarantee) would take about 4864, but the refusal holds with
    either accounting. Also refused:
    ``mcfe/paillier_decentralized`` (its key shares are combined under a DCR
    modulus of more than 15000 bits).

    The statistical parameter ``stat_param`` (rounding and leftover-hash
    slack of ABMPSW20, flooding of the threshold schemes, the masks of
    ``mcfe/ddh_decentralized`` and the Paillier and MCFE2 key distributions)
    is 128, as ``kappa`` of the Ring-LWE schemes: a statistical distance of
    ``2^-128`` per sample, so that ``Q`` samples stay below ``Q 2^-128``.
    That holds for exact Gaussians (the ``"facct"`` sampler, the default of
    the Ring-LWE MIFE / MCFE family, the dense-LWE MCFE keys, SIFE FullySec
    Ring-LWE and, under ``standard``, the ABDP15 public keys) and for the
    MPFR path at ``bits(sigma) + 320`` bits;
    the double-precision rounded sampler does not meet it (see the security
    notes in the documentation).

    ``mcfe/ring_lwe_threshold`` with ``t = s`` (Shamir sharing, Gaussian
    flooding) also accepts, as an explicit opt-in that the profile never
    sets, ``flooding_accounting = "kl"`` with a mandatory
    ``max_partial_decryptions`` (``mife/lwe_threshold`` takes it too, but only
    under ``toy``: this profile refuses that scheme with either accounting). There ``stat_param`` bounds the Hellinger
    distance between the adversary's real and simulated views of a run in
    which every server produces at most that many partial decryptions,
    instead of the per-sample statistical distance above:
    ``stat_param - 6.85`` bits of statistical security in the bit-security
    framework of Micciancio and Walter for decision games, or about
    ``2^-65`` of total statistical distance read conventionally. The servers
    count against the limit in a budget that the deployment gives them and
    must keep across processes and restarts. A different and weaker
    guarantee, with its own parameter records; see
    :mod:`pyfe4ai.schemes.threshold.kl`.

    Values given explicitly in the config must not be weaker than the
    profile's.

The profile's rules -- its minima and presets per kind
(:mod:`pyfe4ai.core.config`) and the refusals (each scheme's ``refused``
in :mod:`pyfe4ai.core.registry`) -- are applied by
:func:`pyfe4ai.core.config.resolve` when a key generator is built; this
module names the profiles and the kind of every scheme.
"""

from __future__ import annotations

from pyfe4ai.core.registry import SCHEMES, scheme_for_module

TOY = "toy"
STANDARD = "standard"
PROFILES = (TOY, STANDARD)

#: Kind of every registered ``(family, variant)`` (the parameters the
#: profiles size: :data:`pyfe4ai.core.registry.KINDS`), derived from the
#: registry.
#:
#: :meta hide-value:
SCHEME_KIND = {scheme.key: scheme.kind for scheme in SCHEMES}

#: Bits of each prime of a DCR modulus under ``standard`` (``N`` of 3072 bits).
STANDARD_DCR_PRIME_BITS = 1536

#: Statistical parameter under ``standard`` (minimum and default).
STANDARD_STAT_PARAM = 128
#: Largest statistical parameter a key generator accepts, in both profiles;
#: the MPFR Gaussians' guard bits (:data:`~pyfe4ai.primitives.sampling.BIGINT_GUARD_BITS`)
#: cover ``stat_param + ceil(log2 N) + 8`` up to this value.
MAX_STAT_PARAM = 256


def scheme_of(cls: type) -> tuple | None:
    """Return the registered ``(family, variant)`` that *cls* implements or extends."""
    for klass in cls.__mro__:
        found = scheme_for_module(klass.__module__)
        if found is not None:
            return found.key
    return None
