"""
Decentralised key derivation for MCFE schemes with a linear functional key.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Appendix D.1, Fig. 14
| ("DMCFE from MCFE"), secure against static corruptions (Theorem 38) if
| MCFE1 is and MCFE2 is sel1-sta-pos+-IND secure.

Fig. 14 turns an MCFE (MCFE1) whose functional key is
``sk_y = sum_i y_i sk_{1,i}`` into a decentralised one:

* KeyGen:        each client has ``sk_{1,i}`` (MCFE1) and an MCFE2 key
                 ``sk_{2,i}`` with ``sum_i sk_{2,i} = 0``.
* KeyDerShare:   client ``i`` publishes
                 ``Enc2(sk_{2,i}, y_i sk_{1,i}, H(y))`` -- an MCFE2 encryption
                 of its key contribution under the label ``H(y)``.
* KeyDerComb:    ``Dec2`` with the all-ones function, whose MCFE2 key is
                 ``sum_i sk_{2,i} = 0``, returns ``sum_i y_i sk_{1,i} = sk_y``.

This module provides MCFE2: the DCR MCFE of ABMPSW20 Fig. 3 (see
:mod:`pyfe4ai.schemes.mcfe.paillier`), one MCFE2 slot per coordinate of a
key contribution, so the all-ones key of each coordinate is the sum of the
clients' MCFE2 keys for that coordinate. Its modulus ``N2`` is sized for the
contributions (Fig. 3 requires ``2 n2 X2^2 < N2`` for ``n2`` slots of bound
``X2``). Implementation choices: ``H(y)`` is a domain-separated hash of the
canonical fusion weights; the zero-sum MCFE2 keys come from pairwise
Gaussian masks ``r_{i,j}`` (client ``i`` adds ``r_{i,j}`` for ``j > i`` and
subtracts it for ``j < i``) -- the "interactively produce" step of Fig. 14
-- which this prototype samples in-process, as it does all other key
material; a deployment would agree on each ``r_{i,j}`` pairwise.
"""

from __future__ import annotations

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.paillier import _hash_label as _dcr_hash_label
from pyfe4ai.schemes.mcfe.paillier import dcr_sigma
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.utils.crypto_utils import generate_group_primes
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError
from pyfe4ai.utils.modular_utils import pow_signed_many
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint

#: Smallest prime size of N2, whatever the contribution bound.
MIN_PRIME_BITS = 128


def key_label(fusion_weight: dict) -> str:
    """Return the MCFE2 label ``H(y)`` binding key shares to the weights ``y``."""
    return "pyfe4ai/dmcfe/H(y)|" + fusion_weight_id(fusion_weight)


def mcfe2_prime_bits(n_slots: int, bound: int, min_bits: int = MIN_PRIME_BITS) -> int:
    """Prime size for ``N2 = p q`` such that ``2 n2 X2^2 < N2`` (ABMPSW20 Fig. 3).

        Args:
            n_slots: Number of MCFE2 slots ``n2`` (clients x contribution size).
            bound: Bound ``X2`` on the absolute value of a contribution entry.
            min_bits: Lower limit on the prime size.
    """
    need = (2 * n_slots * gp.mpz(bound) ** 2).bit_length() + 2
    return max(min_bits, (need + 1) // 2 + 1)


def generate_mcfe2_modulus(prime_bits: int) -> gp.mpz:
    """Return ``N2 = p q`` for fresh safe primes; the primes are discarded."""
    p, _ = generate_group_primes(prime_bits)
    while True:
        q, _ = generate_group_primes(prime_bits)
        if q != p:
            break
    return p * q


def zero_sum_keys(lst_nid: list, dim: int, sigma) -> dict:
    """Return MCFE2 keys ``{nid: [s_{i,0..dim-1}]}`` with ``sum_i s_i = 0``.

    Built from pairwise masks ``r_{i,j} <- D_{Z^dim, sigma}``, so each key is a
    sum of ``n - 1`` Gaussians (width at least ``sigma``) and the keys of all
    clients cancel coordinate-wise.
    """
    if len(lst_nid) < 2:
        raise FEValidationError("decentralised key derivation needs at least two clients")
    keys = {nid: [gp.mpz(0)] * dim for nid in lst_nid}
    for a in range(len(lst_nid)):
        for b in range(a + 1, len(lst_nid)):
            mask = discrete_gaussian_bigint(sigma, dim)
            keys[lst_nid[a]] = [k + r for k, r in zip(keys[lst_nid[a]], mask)]
            keys[lst_nid[b]] = [k - r for k, r in zip(keys[lst_nid[b]], mask)]
    return keys


def mcfe2_key_sigma(n2_mod, n_slots: int, bound: int, stat_param: int):
    """Width of the pairwise masks: the ABMPSW20 Fig. 3 bound for MCFE2."""
    return dcr_sigma(n2_mod, n_slots, bound, stat_param)


def encrypt_contribution(n2_mod, mcfe2_key: list, contribution: list, label: str, bound) -> list:
    """MCFE2 encryption of a key contribution, one Fig. 3 slot per coordinate.

    Returns ``[(1 + N2)^{m_j} H(label)^{s_j} mod N2^2]_j`` as digit strings.
    """
    n2_mod = gp.mpz(n2_mod)
    n2_square = n2_mod * n2_mod
    if len(mcfe2_key) != len(contribution):
        raise FEValidationError("key contribution and MCFE2 key differ in length")
    if any(abs(gp.mpz(m)) > bound for m in contribution):
        raise FEValidationError("key contribution exceeds the MCFE2 bound")
    h = _dcr_hash_label(label, n2_mod)
    masks = pow_signed_many(h, mcfe2_key, n2_square)  # one base: fixed-base method
    return [
        gp.digits((1 + gp.mpz(m) * n2_mod) % n2_square * mask % n2_square)
        for m, mask in zip(contribution, masks)
    ]


def combine_contributions(n2_mod, shares: dict, lst_nid: list) -> list:
    """MCFE2 decryption with the all-ones function (key 0) of every coordinate.

    Args:
        n2_mod: MCFE2 modulus ``N2``.
        shares: Mapping of every client ID to its encrypted contribution.
        lst_nid: All clients; the zero-sum keys only cancel if every client
            contributed.

    Returns:
        ``[sum_i m_{i,j}]_j`` as signed integers (centred mod ``N2``).

    Raises:
        FESchemeError: If a client is missing or a coordinate fails to decode.
    """
    if set(shares) != set(lst_nid):
        raise FESchemeError(
            "key shares of all clients are required, got {}".format(sorted(shares))
        )
    n2_mod = gp.mpz(n2_mod)
    n2_square = n2_mod * n2_mod
    dims = {len(c) for c in shares.values()}
    if len(dims) != 1:
        raise FESchemeError("key shares of different sizes")
    sums = []
    for j in range(dims.pop()):
        acc = gp.mpz(1)
        for nid in lst_nid:
            acc = acc * gp.mpz(shares[nid][j]) % n2_square
        if (acc - 1) % n2_mod != 0:
            raise FESchemeError("key shares were not derived for the same weights")
        value = ((acc - 1) % n2_square) // n2_mod
        if value > n2_mod // 2:
            value -= n2_mod
        sums.append(value)
    return sums
