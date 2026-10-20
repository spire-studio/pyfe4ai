"""
Threshold Ring-LWE identity-based multi-client inner-product FE: the BMMS21
identity-based MCFE with a Shamir-shared functional key and statistically
flooded partial decryptions.

| MCFE layer: :mod:`pyfe4ai.schemes.mcfe.ring_lwe` (BMMS21, PKC 2022, Section 6).
| Threshold layer in the setting of R. Xu et al., "TAPFed: Threshold Secure
| Aggregation for Privacy-Preserving Federated Learning", IEEE TDSC 2024.

* type:     identity-based multi-client encryption, threshold functional decryption
* setting:  R_q = Z_q[x] / (x^ring_n + 1), q a product of NTT-friendly primes
  (default) or an NTT-friendly prime

Defaults (both security profiles, ``standard`` and ``toy``): binary
ciphertexts, key shares and partial decryptions (``ct_format="binary"``), an
RNS modulus (``ring_modulus="rns"``) and key shares split into a weight part
per weights and a label part per label (``key_shares="split"``; whole key
shares for the replicated-seeded sharing). Each is selectable as before --
``ct_format="json"``, ``ring_modulus="prime"``, ``key_shares="full"`` or
``key_part="full"`` per call -- and public parameters, parameter records,
ciphertexts and key shares of the earlier defaults (no ``ct_format``: JSON; no
``q_factors``: a prime) keep working where they did. The defaults change no
construction id; their parameter records are ``param-rns.json`` (a folder's
``param.json`` is the prime's). What does not match fails closed: a server
refuses ciphertexts and key shares of the other format or of other public
parameters (``pp_binding``), and a label part without its weight part
(:meth:`ThresholdMCFERingLWE.add_weight_key_share`); a combiner refuses partial
decryptions of the other format (partial decryptions carry no digest of the
public parameters; servers are honest-but-curious, see below) and partial
decryptions of different sharings of one key (:class:`MixedSharingsError`).

Construction (notation of :mod:`pyfe4ai.schemes.mife.ring_lwe`):

* KeyGen:    the identity-based key ``({sk_i}, z'', z''_chunks)`` for ``(y, l)``;
  every coefficient is Shamir-shared with threshold ``t`` over ``Z_q``.
* ShareDec:  server ``k`` of an enrolled set ``E`` publishes, coefficient-wise,
  ``w_k = lambda_k (sum_i ct1_i sk_i^(k) + Delta z''^(k)) + e_k`` and
  ``c_y = sum_i sum_row y_row ct0_{i,row}``.
* Combine:   ``c_y - sum_{k in E} w_k = Delta <x, y> + noise - sum e_k``; decode.

Flooding: the exact key part differs from what the result and the
ciphertexts determine by the decryption noise
``sum_row y_row (e'_row r + e_row) - e sum_row y_row s_row``, bounded per
coefficient by ``B = n eta Y (2 ring_n 144 sigma1 sigma2 + 12 sigma3)``;
flooding noise ``e_k`` per coefficient smudges it (statistical argument;
uniform by default, Gaussian ``D_{Z, 2^kappa B}`` selectable, see below),
with ``q`` enlarged so that ``s`` flooding terms still decode.
Ciphertexts (and ndarray chunks) carry one HMAC tag per server over the
(chunk) label and the ciphertext (see "Ciphertext format" below); servers
only decrypt ciphertexts tagged for them under the label they serve, since
``w_k`` is linear in ``ct1``.
The noise is seeded with a per-server key (fixed per weights and label),
``lambda_k``, the label or chunk, the ``ct1`` values and the key share's
``pp_binding`` (the public parameters it is bound to), so a request for
the same key part draws the same noise; a corrupted client varying its own
``ct1`` draws fresh noise each time. The flooding argument is statistical
per answer (a statistical distance of about ``2^-stat_param`` per
coefficient, adding up over the answers); it does not bound what many
answers to varied ``ct1`` reveal together: averaging them would take about
``2^(2 stat_param)`` partial decryptions with Gaussian noise (see below
for the uniform default), but other ways of combining these noisy linear
equations are not ruled out. The threshold layer is our own design.

Flooding distribution (key generator config ``flooding``, a public
parameter):

* ``"uniform"`` (default): ``e_k`` uniform on ``[-2^tau, 2^tau)`` with
  ``tau = bits(B) + kappa``, read from the same seeded XOF
  (:func:`uniform_flooding_poly`). By the smudging lemma (Asharov et al.,
  EUROCRYPT 2012, Lemma 2.1) the statistical distance per coefficient is
  ``|E| / 2^(tau + 1) < 2^-(kappa + 1)``, the per-sample bound the Gaussian
  meets as well; ``q`` holds ``s 2^tau`` instead of ``s 12 sigma_f`` (about
  3 bits less), correctness needs no tail cut, and the sampler is exact and
  integer-only (no ``mpfr``). Unlike the Gaussian it has no margin beyond
  that bound: over ``N`` coefficients the distance is ``N 2^-(kappa + 1)``
  (the Gaussian's is ``sqrt(N)`` times its per-sample bound), and telling
  ``E`` from ``E + 1`` by repetition takes about ``2^(tau + 1)`` partial
  decryptions instead of ``2^(2 kappa) B^2``. Parameter records
  ``bmms21-ibmcfe-shamir-smudging-uniform/v1``; public parameters
  ``flooding`` and ``flood_bits``.
* ``"gaussian"``: ``e_k <- D_{Z, 2^kappa B}`` as described above, the only
  flooding of versions <= 0.3.0 (public parameter ``sigma_f``; public
  parameters without ``flooding`` are read as this), drawn by the pinned
  :data:`~pyfe4ai.primitives.sampling.FLOODING_SAMPLER` at ``bits(sigma_f)
  + 320`` bits (records ``bmms21-ibmcfe-shamir-smudging/v2``; the ``v1``
  records of earlier versions are regenerated).

Decryption key shares record the flooding they were issued for (``flooding``,
with ``noise_version`` and ``noise_sampler``); a server refuses a share of
another flooding than its public parameters', which would answer a request
with a second, independent noise.

Sharing (config entry ``sharing``; default ``"replicated"`` for ``t < s`` and
``"shamir"`` for ``t = s``): with ``"shamir"`` and ``t < s`` -- the sharing of
versions <= 0.3.0 -- the smudging argument does not hold: with ``t - 1``
servers ``C`` corrupted an honest partial is computable from ``C``'s shares
and the result up to ``e_k - M nu`` per coefficient,
``M = lambda_k^E prod_{j in C} (k - j) / (0 - j)``, a non-integer rational
``r / d`` for some ``(C, E, k)``, and ``d`` times the difference reveals
``nu mod d`` coefficient by coefficient, whatever ``sigma_f``. Fewer corrupted
servers, or none, can suffice (the aggregator alone reads ``nu`` modulo
``(s - 1)!`` for ``t = 1``), and combining several enrolled sets raises ``d``
(144 for ``s = 5, t = 3``); integer ``M`` reach 4 for ``s = 3, t = 2``, which
multiplies the statistical distance of the flooding by 4. ``"replicated"``
(see :func:`~pyfe4ai.schemes.threshold.threshold_utils.share_replicated`) makes ``M`` 0
or 1, so the argument holds as stated with ``q`` and ``ring_n`` unchanged; a
server holds ``C(s - 1, t - 1)`` key pieces instead of one share (dealt and
added up natively where the extension is built, with the values of the
Python code). With ``t = s`` both are sound. An explicit
``"shamir"`` with ``t < s`` is refused under the ``standard`` profile and
draws an ``FESecurityWarning`` under ``toy``; the replicated sharing has its
own parameter records (``threshold_construction``
``bmms21-ibmcfe-replicated-smudging-uniform/v1``, or
``bmms21-ibmcfe-replicated-smudging/v1`` with the Gaussian flooding), so a
record cached by an earlier version for ``t < s`` is not reused by the
default. Servers refuse Shamir key shares with ``t < s`` unless created with
``legacy_shamir = "warn"`` (never with public parameters that record the
``standard`` profile); key shares of versions <= 0.3.0 are refused in any
case (another flooding derivation, see
:func:`~pyfe4ai.primitives.sampling.check_flooding_version`);
public parameters record ``threshold_construction`` and ``security_profile``,
and servers refuse constructions they do not know. The opt-in
``"replicated-seeded"`` (see
:func:`~pyfe4ai.schemes.threshold.threshold_utils.share_replicated_seeded`) deals the
same kind of pieces, all but one expanded from 32-byte seeds: a server stores
seeds and at most one full piece (the size of a Shamir key share), and its
partial decryptions are those of the replicated key share of the expanded
pieces; the key shares hide the key computationally (SHAKE-256 keyed by the
seeds as a PRF) rather than perfectly. It has its own records
(``bmms21-ibmcfe-replicated-seeded-smudging-uniform/v1``, or
``bmms21-ibmcfe-replicated-seeded-smudging/v1`` with the Gaussian flooding).

``flooding_accounting = "kl"`` with ``max_partial_decryptions = Q`` (opt-in,
``t = s`` with the Shamir sharing and the Gaussian flooding only -- the
flooding defaults to ``"gaussian"`` under ``"kl"`` and ``"uniform"`` is
refused there --, see :mod:`pyfe4ai.schemes.threshold.kl`) sizes the
Gaussian noise for a run in which every server produces at most ``Q`` partial
decryptions (one per ciphertext or chunk): ``KL <= 2^-(stat_param + 1)``
for the ``N = s Q ring_n`` flooded coefficients, i.e.
``sigma_f = B sqrt(N 2^stat_param)``, with a tail cut derived from ``N`` (16
``sigma_f`` at ``N = 2^53``), instead of a statistical distance
``2^-stat_param`` per coefficient. It is a different, weaker guarantee
(``stat_param - 6.85`` bits in the bit-security sense of Micciancio and
Walter, about ``2^-65`` of total statistical distance at
``stat_param = 128``), valid only for the first ``Q`` partial decryptions
of every server, which each server counts in the ``partial_budget`` its
deployment gives it; averaging the noise away would need about
``N 2^stat_param`` requests. Under ``standard`` (the benchmark's three
clients, ``X = 100``, ``Y = 1``, ``s = t = 3``) ``N = 2^53`` gives a ``q`` of
178 bits instead of 215 with the per-sample Gaussian flooding (212 with the
uniform default), at the same ``ring_n`` 8192. These parameters
have their own record (``threshold_construction``
``bmms21-ibmcfe-shamir-smudging-kl/v2``).

Ciphertext format (key generator config ``ct_format``, a public parameter;
:data:`CT_FORMATS`): ``"json"`` (decimal strings, tags over the JSON form,
as in versions <= 0.3.0; public parameters without ``ct_format``) or
``"binary"`` (the default): format 2 of
:func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.encode_threshold_ciphertext` -- the
``ct0`` rows truncated to the batch length ``k`` (decryption reads them
coefficient-wise), ``ct1`` whole, every value a ``ceil(bits(q)/8)``-byte
word strictly below ``q`` -- and binary partial decryptions of the first
``k`` coefficients (about 71 and 54 bytes per element at 215 bits instead
of 225 and 222). Tags (v2) are HMAC-SHA256 under the client-server key of
:func:`binary_tag_message`: the public parameters' digest
(:func:`~pyfe4ai.schemes.threshold.threshold_utils.pp_binding` over
:func:`binding_fields`), the (chunk) label, the ciphertext's label hash
code, ``k`` and SHA-256 digests of the ``ct0`` and ``ct1`` bytes; the noise
is seeded with digests of the ``ct1`` bytes (:func:`binary_request_digests`).
A format-2 encoding is unique -- a value ``>= q`` is refused, also under a
valid tag -- so no client can re-encode one request to draw a second noise
for the same key part; the first ``k`` noise values are those of the full
draw (a prefix of the SHAKE stream for the uniform noise), so answers for
another ``k`` agree on their common coefficients. The format is bound to the
key shares (``ct_format`` enters ``pp_binding`` where the public parameters
have it; JSON digests are unchanged): a key share is served in one format
only. Under binary public parameters JSON ciphertexts, format 1, v1 tags and
JSON partial decryptions are refused, and under JSON ones every binary form.
A client, server or combiner fixes its format when it is made and refuses to
work once the bound fields or the format of its public parameters (the
caller's dict) have changed: one key part, one derivation.

Key shares per round (key generator config ``key_shares``,
:data:`KEY_SHARES`; ``"split"`` by default): a key share for ``(y, l)`` is the
label-independent ``sk_y`` and the label's terms ``z''_l`` (and of its chunks).
:meth:`ThresholdMCFERingLWEKeyGenerator.get_weight_key_share` issues the
shares of ``sk_y`` once per weights (servers keep them:
:meth:`ThresholdMCFERingLWE.add_weight_key_share`), and
``get_decryption_keys`` (``key_part="label"``, the default with split key
shares) the shares of the label terms only (``label_term=False`` leaves out
the label's own ``z``, which only single ciphertexts need): two of six
polynomials per round for two chunks; per round a server needs nothing else
from the key generator. ``key_shares="full"`` (or ``key_part="full"`` per
call) issues whole key shares, as in versions <= 0.3.0; the replicated-seeded
sharing has only these. Both
parts are of one sharing (Shamir points or replicated sets) and carry a
``share_set_id``; a server combines a label part only with the weight part of
the same weights, share set, sharing, binding, flooding, noise derivation and
MAC keys. A linear sharing of ``sk_y`` reused with fresh sharings of the label
terms is a sharing of each round's key, the shares of ``t - 1`` servers stay
independent of the keys, and the noise is seeded as for a full key share;
``NTT(sk_y)`` of a kept weight part is computed once. With binary public
parameters every polynomial of the Shamir and replicated key shares is bytes
(``ceil(bits(q)/8)``-byte words, strictly below ``q``; the seeded sharing keeps
decimal strings). Key material mixing both kinds is dealt as values; key shares
whose polynomials are neither bytes nor lists or tuples (e.g. ``memoryview``),
are of another length, hold values other than ints, ``mpz`` and decimal
strings, or are replicated pieces mixing kinds, are refused (``FEKeyError``).
The key generator keeps the sharings of ``sk_y`` for the last
:data:`WEIGHT_SHARINGS_KEPT` weights: weights issued again after that come
with a new share set -- their weight part and every label part issued from
then on -- which servers holding the old weight part refuse until given the
new one. A label part is of the weights' share set current when it is issued
(one made or cached while another thread evicted that sharing is made again
for the new set) and is kept with that sharing, and dropped with it; per
sharing the label parts of the last :data:`LABEL_PARTS_KEPT` labels are kept
(a label asked for again later gets a new sharing of its terms). Every sharing
the key generator issues -- whole key shares, or the label parts of one label
-- has a random ``sharing_id``, which the servers' partial decryptions carry:
partials of two sharings of one key would decode to a wrong value inside the
decoding bound rather than fail, so combiners refuse a mix
(:class:`MixedSharingsError`; fetch the label parts again for all servers).
The two sharings' answers of one server differ by its share of a sharing of
zero, which ``t - 1`` servers compute from their own parts: a second sharing
reveals nothing. Store and transmit the ``sharing_id`` with every key share
and partial decryption: partials without one (as from key shares of earlier
versions) combine only with each other, so a storage or transport layer that
drops the id on every side would let partials of two sharings combine to a
wrong value unnoticed.

The threshold objects (key generator, clients, servers) are not portable across
versions of pyfe4ai: a state pickled by another version is refused when loaded
(create the object again).

Modulus (key generator config ``ring_modulus``): ``"rns"`` (the default):
``q`` the product of NTT primes below ``2^62``, at least
:data:`MIN_RNS_FACTORS` (three: the native threshold kernels run on the
256-bit ring, also for the smaller ``q`` of the ``toy`` profile) and as many
as make each at most 60 bits (four for the 212 to 218 bits of the
``standard`` profile, about 54 bits each), but four up to 248 bits, the most
the native ring holds (above, five or more on the Python path, with a
warning -- ``"prime"`` stays native below ``2^253``;
:func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.rns_ntt_modulus_for`), of the bit length
of the prime and above the same bound, published as ``q_factors`` (bound to
the key shares); or ``"prime"``, an NTT prime (the modulus of earlier
versions). Every factor must exceed ``s`` (:func:`check_rns_servers`):
Shamir sharing over ``Z_q`` is then one field sharing per factor (CRT), with
units for points and differences; the replicated sharing, the flooding (an
argument over the integers), the label hash (exact rejection sampling on
``[0, q)``) and the decoding bound do not depend on the factorisation (see
docs/rns_modulus_review.md). The parameter record carries ``ring_modulus``
and ``q_factors`` (file ``param-rns.json``, beside the prime's
``param.json``); the construction ids are those of the prime modulus. The
native backend runs the 256-bit kernels with one 64-bit NTT per factor
(NTT-domain values are residue tuples), with the values of the Python code
on the composite ``q``.

With the native backend (:mod:`pyfe4ai.backend`; moduli below
``2^253``, which includes the flooded ``q`` of the standard profile) the MCFE
layer, the Shamir sharing of the key, the parsing, digests and NTT-domain
products of a partial decryption and the combination run natively, with
the values of the Python code; the uniform flooding noise is drawn natively
too (:func:`uniform_flooding_poly`'s values), the Gaussian by
:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint` (natively too,
with the same samples).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from collections import OrderedDict

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.threshold.kl import KL
from pyfe4ai.schemes.threshold.kl import PER_SAMPLE
from pyfe4ai.schemes.threshold.kl import check_kl_threshold
from pyfe4ai.schemes.threshold.kl import flooding_config
from pyfe4ai.schemes.threshold.kl import kl_flooded_samples
from pyfe4ai.schemes.threshold.kl import kl_flooding
from pyfe4ai.schemes.threshold.kl import kl_flooding_width
from pyfe4ai.schemes.threshold.kl import kl_tail_cut
from pyfe4ai.schemes.threshold.kl import server_budget
from pyfe4ai.schemes.mcfe.labels import chunk_label
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import check_ciphertext_label_hashes
from pyfe4ai.schemes.mcfe.ring_lwe import chunk_key_index
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.threshold.threshold_utils import check_ciphertext_tag_messages
from pyfe4ai.schemes.threshold.threshold_utils import check_ciphertext_tags
from pyfe4ai.schemes.threshold.threshold_utils import check_pp_binding
from pyfe4ai.schemes.threshold.threshold_utils import check_served_sharing
from pyfe4ai.schemes.threshold.threshold_utils import pp_binding
from pyfe4ai.schemes.threshold.threshold_utils import legacy_shamir_policy
from pyfe4ai.schemes.threshold.threshold_utils import check_fusion_weight
from pyfe4ai.schemes.threshold.threshold_utils import client_server_mac_keys
from pyfe4ai.schemes.threshold.threshold_utils import tag_ciphertext
from pyfe4ai.schemes.threshold.threshold_utils import tag_ciphertext_message
from pyfe4ai.schemes.threshold.threshold_utils import ciphertexts_digest
from pyfe4ai.schemes.threshold.threshold_utils import fusion_weight_id
from pyfe4ai.schemes.threshold.threshold_utils import pieces_held
from pyfe4ai.schemes.threshold.threshold_utils import poly_kinds
from pyfe4ai.schemes.threshold.threshold_utils import strict_mpz
from pyfe4ai.schemes.threshold.threshold_utils import uniform_material
from pyfe4ai.schemes.threshold.threshold_utils import seeded_pieces_held
from pyfe4ai.schemes.threshold.threshold_utils import share_cache
from pyfe4ai.schemes.threshold.threshold_utils import share_for_enrolled
from pyfe4ai.schemes.threshold.threshold_utils import share_points
from pyfe4ai.schemes.threshold.threshold_utils import share_replicated
from pyfe4ai.schemes.threshold.threshold_utils import share_replicated_seeded
from pyfe4ai.schemes.threshold.threshold_utils import share_secret
from pyfe4ai.schemes.threshold.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold.threshold_utils import combine_consistently
from pyfe4ai.schemes.threshold.threshold_utils import partial_groups
from pyfe4ai.schemes.threshold.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold.threshold_utils import resolve_sharing
from pyfe4ai.schemes.threshold.threshold_utils import validate_threshold
from pyfe4ai import backend as native_backend
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.lattice.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.primitives.lattice.lwe_utils import decode_scaled
from pyfe4ai.primitives.lattice.ring_lwe_utils import canonical_ring_ciphertext
from pyfe4ai.primitives.lattice.ring_lwe_utils import decode_threshold_ciphertext
from pyfe4ai.primitives.lattice.ring_lwe_utils import decode_values
from pyfe4ai.primitives.lattice.ring_lwe_utils import encode_values
from pyfe4ai.primitives.lattice.ring_lwe_utils import threshold_ciphertext_from_format1
from pyfe4ai.primitives.lattice.ring_lwe_utils import threshold_ciphertext_header
from pyfe4ai.primitives.lattice.ring_lwe_utils import mat_vec_mul
from pyfe4ai.primitives.lattice.ring_lwe_utils import bmms21_selective_parameters
from pyfe4ai.primitives.lattice.ring_lwe_utils import next_ntt_prime
from pyfe4ai.primitives.lattice.ring_lwe_utils import register_rns_modulus
from pyfe4ai.primitives.lattice.ring_lwe_utils import RNS_NATIVE_FACTORS
from pyfe4ai.primitives.lattice.ring_lwe_utils import rns_ntt_modulus_for
from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_add
from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_mul
from pyfe4ai.primitives.lattice.ring_lwe_utils import RingNTT
from pyfe4ai.primitives.lattice.ring_lwe_utils import transpose
from pyfe4ai.primitives.sampling import XofRandom
from pyfe4ai.primitives.sampling import (
    FLOODING_NOISE_VERSION,
    FLOODING_SAMPLER,
    check_flooding_version,
    discrete_gaussian_bigint,
)
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

_CONSTRUCTION = "bmms21-ibmcfe-shamir-smudging/v2"  # v2: flooding at bits + 320, sampler pinned
_CONSTRUCTION_UNIFORM = "bmms21-ibmcfe-shamir-smudging-uniform/v1"
_CONSTRUCTION_KL = "bmms21-ibmcfe-shamir-smudging-kl/v2"  # Gaussian flooding, KL accounting
# the parameters are those of the Shamir records; their own records keep cached
# Shamir records (t < s) of versions <= 0.3.0 from being taken for them
_CONSTRUCTION_REPLICATED = "bmms21-ibmcfe-replicated-smudging/v1"
_CONSTRUCTION_REPLICATED_UNIFORM = "bmms21-ibmcfe-replicated-smudging-uniform/v1"
_CONSTRUCTION_REPLICATED_SEEDED = "bmms21-ibmcfe-replicated-seeded-smudging/v1"
_CONSTRUCTION_REPLICATED_SEEDED_UNIFORM = "bmms21-ibmcfe-replicated-seeded-smudging-uniform/v1"
#: the parameter record (and public ``threshold_construction``) of a sharing and a
#: flooding under the per-sample accounting (``"kl"``: :data:`_CONSTRUCTION_KL`)
_CONSTRUCTION_IDS = {
    ("shamir", "gaussian"): _CONSTRUCTION,
    ("shamir", "uniform"): _CONSTRUCTION_UNIFORM,
    ("replicated", "gaussian"): _CONSTRUCTION_REPLICATED,
    ("replicated", "uniform"): _CONSTRUCTION_REPLICATED_UNIFORM,
    ("replicated-seeded", "gaussian"): _CONSTRUCTION_REPLICATED_SEEDED,
    ("replicated-seeded", "uniform"): _CONSTRUCTION_REPLICATED_SEEDED_UNIFORM,
}
_CONSTRUCTIONS = {construction: sharing for (sharing, _), construction in _CONSTRUCTION_IDS.items()}
_CONSTRUCTIONS[_CONSTRUCTION_KL] = "shamir"
#: Public-parameter fields a key share is bound to (see
#: :func:`~pyfe4ai.schemes.threshold.threshold_utils.pp_binding`): what its partial
#: decryptions depend on beyond the sharing and the construction.
_BINDING_FIELDS = (
    "q", "p", "ring_n", "eta", "s", "t", "lst_sid", "flooding", "flood_bits", "sigma_f", "flood_tail_cut",
    "flooding_accounting", "max_partial_decryptions", "stat_param", "label_hash",
)
_BINDING_DEFAULTS = {"flooding": "gaussian", "flooding_accounting": "per-sample", "label_hash": "shake256-wide"}
#: Fields bound only where the public parameters have them (the digests of
#: public parameters without them, of earlier versions too, stay as they were).
_OPTIONAL_BINDING_FIELDS = ("ct_format", "q_factors")


def binding_fields(pp: dict) -> tuple:
    """The public-parameter fields a key share of *pp* is bound to:
    ``_BINDING_FIELDS`` and those of ``_OPTIONAL_BINDING_FIELDS`` that
    *pp* has (``ct_format`` of binary public parameters, ``q_factors`` of an
    RNS modulus)."""
    return _BINDING_FIELDS + tuple(f for f in _OPTIONAL_BINDING_FIELDS if f in pp)


#: Ciphertext formats of the threshold scheme (key generator config
#: ``ct_format``, a public parameter): ``"json"`` (decimal strings, tags over
#: the JSON form; the only format of versions <= 0.3.0, and that of public
#: parameters without ``ct_format``) and ``"binary"`` (the default: format 2
#: of :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.encode_threshold_ciphertext`, tags
#: over its digests, binary partial decryptions).
CT_FORMATS = ("json", "binary")
#: ``ct_format`` of a key generator whose config has none.
DEFAULT_CT_FORMAT = "binary"
#: ``ring_modulus`` of a key generator whose config has none (``"prime"``
#: selectable; see the module docstring).
DEFAULT_RING_MODULUS = "rns"
#: Fewest factors of an RNS modulus of the threshold scheme. Its native
#: kernels run on the 256-bit ring (two factors below ``2^124`` would get the
#: two-prime context of the MIFE / MCFE prototype, which has none of them).
MIN_RNS_FACTORS = 3
#: How a key generator issues key shares (config ``key_shares``): ``"split"``
#: (the default for the Shamir and replicated sharings) -- a weight part per
#: weights (:meth:`ThresholdMCFERingLWEKeyGenerator.get_weight_key_share`) and
#: a label part per label -- or ``"full"`` (whole key shares per label, as in
#: versions <= 0.3.0; the only kind of the replicated-seeded sharing).
KEY_SHARES = ("split", "full")


def pp_ct_format(pp: dict) -> str:
    """The ciphertext format of threshold public parameters (absent: ``"json"``).

    Raises:
        FEValidationError: For another value.
    """
    fmt = pp.get("ct_format", "json")
    if fmt not in CT_FORMATS:
        raise FEValidationError("unknown ciphertext format {!r}; one of {}".format(fmt, ", ".join(CT_FORMATS)))
    return fmt


def _construction_id(sharing: str, flooding: str, accounting: str) -> str | None:
    """The construction id of a sharing, a flooding and a flooding accounting
    (``None`` for a combination the key generator does not make)."""
    if accounting == KL:
        return _CONSTRUCTION_KL if (sharing, flooding) == ("shamir", "gaussian") else None
    return _CONSTRUCTION_IDS.get((sharing, flooding))
_TAG_DOMAIN = "pyfe4ai/tmcfe-ring-lwe/ciphertext-tag/v1"
_UNIFORM_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/uniform-flooding/v1"
# binary public parameters (format 2): what the tags and the noise seeds hash
_TAG_DOMAIN_BIN = b"pyfe4ai/tmcfe-ring-lwe/ciphertext-tag/v2-binary|"
_CT0_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/ct0/v2|"
_CT1_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/ct1/v2|"
_KEY_INPUTS_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/key-inputs/v2|"
_CT_DIGEST_DOMAIN = b"pyfe4ai/tmcfe-ring-lwe/ct-digest/v2|"

#: Flooding distributions of the partial decryptions.
FLOODINGS = ("uniform", "gaussian")

#: The key generator's default flooding.
DEFAULT_FLOODING = "uniform"

#: The flooding of versions <= 0.3.0 and of public parameters without ``flooding``.
LEGACY_FLOODING = "gaussian"


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


def binary_ciphertext_digests(ct: dict, q, eta: int, ring_n: int) -> tuple:
    """``(k, d0, d1)`` of a format-2 ciphertext dict: its batch length and
    ``d0 = SHA-256(domain || header || ct0 rows)``, ``d1 = SHA-256(domain ||
    ct1)`` (the values are checked when read: :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.decode_threshold_ciphertext`).

    Raises:
        ValueError: Unless ``ct["ct"]`` is a format-2 ciphertext of these
            parameters whose batch length is ``ct["k"]``.
    """
    data = ct["ct"]
    k = threshold_ciphertext_header(data, q, eta, ring_n)
    if type(ct.get("k")) is not int or ct["k"] != k:
        raise ValueError("the ciphertext's k is not its header's")
    split = 14 + eta * k * ((int(q).bit_length() + 7) // 8)
    view = memoryview(data)
    d0 = hashlib.sha256(_CT0_DOMAIN)
    d0.update(view[:split])
    d1 = hashlib.sha256(_CT1_DOMAIN)
    d1.update(view[split:])
    return k, d0.digest(), d1.digest()


def binary_tag_message(pp_digest: bytes, encoded_label: str, ct: dict, digests: tuple) -> bytes:
    """What the tag of a format-2 ciphertext covers (HMAC-SHA256 under the
    client-server key): a domain, the digest of the public parameters
    (:func:`~pyfe4ai.schemes.threshold.threshold_utils.pp_binding` over
    :func:`binding_fields`), the (chunk) label, the ciphertext's ``label_hash``
    code (for an ndarray chunk its layer's; see ``_chunk_with_code``), the
    batch length and the digests of :func:`binary_ciphertext_digests`.

    Raises:
        ValueError: For a label hash code that is not a byte.
    """
    k, d0, d1 = digests
    code = ct.get("label_hash", 0)
    if type(code) is not int or not 0 <= code < 256:
        raise ValueError("malformed label hash code")
    label = encoded_label.encode("utf-8")
    return (_TAG_DOMAIN_BIN + pp_digest + len(label).to_bytes(4, "big") + label + bytes([code])
            + k.to_bytes(4, "big") + d0 + d1)


def _client_ids(dict_ct) -> list:
    """The client ids of a request as ``(json.dumps(nid) bytes, nid)``, sorted."""
    return sorted((json.dumps(nid).encode("utf-8"), nid) for nid in dict_ct)


def binary_request_digests(digests: dict) -> tuple:
    """``(ct_digest, key_inputs)`` (hex) of a request of format-2 ciphertexts
    from their :func:`binary_ciphertext_digests` ``{nid: (k, d0, d1)}``:
    SHA-256 over the client ids (``json.dumps``, sorted) with ``(k, d0, d1)``
    each, and over the ids with ``d1`` only -- the values the key part depends
    on, so the flooding noise is seeded with them (a format-2 encoding is
    unique: no re-encoding of one request draws fresh noise)."""
    ct_digest, key_inputs = hashlib.sha256(_CT_DIGEST_DOMAIN), hashlib.sha256(_KEY_INPUTS_DOMAIN)
    for name, nid in _client_ids(digests):
        k, d0, d1 = digests[nid]
        prefix = len(name).to_bytes(4, "big") + name
        ct_digest.update(prefix + k.to_bytes(4, "big") + d0 + d1)
        key_inputs.update(prefix + d1)
    return ct_digest.hexdigest(), key_inputs.hexdigest()


def _poly_values(poly, q, ring_n: int) -> list:
    """A key-share polynomial as ``mpz``: a list or tuple of ``ring_n`` ints,
    ``mpz`` or decimal strings (:func:`~pyfe4ai.schemes.threshold.threshold_utils.strict_mpz`),
    or binary (``ceil(bits(q)/8)``-byte words, strictly below q).

    Raises:
        FEKeyError: For malformed binary values.
    """
    if isinstance(poly, (bytes, bytearray)):
        try:
            return [gp.mpz(v) for v in decode_values(poly, q, ring_n)]
        except ValueError as exc:
            raise FEKeyError("malformed binary key share") from exc
    if not isinstance(poly, (list, tuple)) or len(poly) != ring_n:
        raise FEKeyError("malformed key share: a polynomial is ring_n values or ring_n words of bytes")
    return [strict_mpz(v) for v in poly]


def _check_key_polys(polys, q, ring_n: int) -> None:
    """Raise unless every polynomial is ``ring_n`` values (a list or tuple) or
    ``ring_n`` words of ``ceil(bits(q)/8)`` bytes: a shorter one would be
    served as a shorter answer, a longer one read in part.

    Raises:
        FEKeyError: Otherwise.
    """
    words = ring_n * ((int(q).bit_length() + 7) // 8)
    for poly in polys:
        if isinstance(poly, (bytes, bytearray)):
            ok = len(poly) == words
        else:
            ok = isinstance(poly, (list, tuple)) and len(poly) == ring_n
        if not ok:
            raise FEKeyError("malformed key share: a polynomial is ring_n values or ring_n words of bytes")


#: Guards the weight-share stores and the NTT(sk_y) cache (key generators and
#: servers; several steps of a dict each). The sharing and the NTTs run outside it.
_STORE_LOCK = threading.Lock()


def _reset_store_lock() -> None:
    global _STORE_LOCK
    _STORE_LOCK = threading.Lock()  # a forked child: the parent's may be held by another thread


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_store_lock)

#: The version of the pickled state of the threshold objects: a state pickled
#: by another version of pyfe4ai (other attributes, other derivations) is
#: refused when loaded.
_STATE_VERSION = "pyfe4ai/tmcfe-ring-lwe/state/2026-10-3"


def _versioned_state(state: dict) -> dict:
    return dict(state, _state_version=_STATE_VERSION)


def _check_state(obj, state: dict) -> None:
    """Load *state* into *obj*, refusing one pickled by another version.

    Raises:
        FESchemeError: For a state of another version (or of none).
    """
    if not isinstance(state, dict) or state.get("_state_version") != _STATE_VERSION:
        raise FESchemeError(
            "this {} was pickled by an incompatible version of pyfe4ai; create the object again".format(
                type(obj).__name__)
        )
    obj.__dict__.update(state)


#: Weights whose shares of ``sk_y`` a key generator keeps (least recently used
#: out): a label part for weights no longer kept comes with a new share set,
#: which servers holding the old weight part refuse until given the new one.
WEIGHT_SHARINGS_KEPT = 64

#: Labels whose label parts a key generator keeps per sharing of ``sk_y``
#: (least recently used out; the servers of one label get the same shares while
#: it is kept). A label part asked for again after that is a new sharing of the
#: same terms (same share set and noise key) with another ``sharing_id``:
#: combiners refuse partial decryptions of the two sharings
#: (:class:`MixedSharingsError`), and the servers then fetch the label's parts
#: again, all from the new sharing.
LABEL_PARTS_KEPT = 16


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
    (:data:`LABEL_PARTS_KEPT`); fetch them again for all servers. The check
    needs the ids: keep the ``sharing_id`` with every stored or transmitted
    key share and partial decryption (partials without one combine only with
    each other).
    """


def _native_ring(q, ring_n: int):
    """The native ring context the threshold kernels take for ``(q, ring_n)``
    (prime, or an RNS modulus of :func:`~pyfe4ai.backend.ring_context`
    other than the two-prime context of the MIFE / MCFE prototype, which has
    kernels of its own), or ``None`` for the Python path."""
    ctx = native_backend.ring_context(q, ring_n)
    return None if ctx is None or ctx.legacy_rns else ctx


def check_rns_servers(s, factors) -> None:
    """Raise unless every factor of an RNS modulus exceeds the number of servers
    *s*: Shamir sharing over ``Z_q = prod F_{q_i}`` (CRT) needs the points
    ``1..s`` and their differences to be units mod every factor.

    Raises:
        FEValidationError: If ``s >= min q_i``.
    """
    if factors and int(s) >= min(int(f) for f in factors):
        raise FEValidationError("an RNS modulus needs every factor above the number of servers s = {}".format(s))


def _tag_payload(encoded_label: str, ct: dict) -> list:
    """What a tMCFE-Ring-LWE ciphertext tag covers: the (chunk) label, the
    ciphertext as serialised and its ``label_hash`` code when it has one --
    for an ndarray chunk its layer's code, see ``_chunk_with_code``
    (ciphertexts of versions <= 0.3.0 have none: their tags stay valid)."""
    payload = [encoded_label, ct.get("k"), ct.get("ct0"), ct.get("ct1")]
    if "label_hash" in ct:
        payload.append(ct["label_hash"])
    return payload


def _chunk_with_code(ct, layer) -> dict:
    """An ndarray chunk as tagged and decrypted: with its layer's
    ``label_hash`` code (layers, not chunks, carry it)."""
    if isinstance(ct, dict) and isinstance(layer, dict) and "label_hash" in layer and "label_hash" not in ct:
        return dict(ct, label_hash=layer["label_hash"])
    return ct


def _flooding_poly(sigma: int, ring_n: int, rng) -> list:
    """*ring_n* samples of ``D_{Z, sigma}`` cut at ``GAUSSIAN_TAIL_CUT sigma``, drawn in order."""
    bound = GAUSSIAN_TAIL_CUT * sigma
    # one call for ring_n samples draws what ring_n calls for one sample draw
    out = [e for e in discrete_gaussian_bigint(sigma, ring_n, rng=rng, sampler=FLOODING_SAMPLER) if abs(e) <= bound]
    while len(out) < ring_n:  # beyond the tail cut (rare): draw on, one at a time
        e = discrete_gaussian_bigint(sigma, 1, rng=rng, sampler=FLOODING_SAMPLER)[0]
        if abs(e) <= bound:
            out.append(e)
    return out


class ThresholdMCFERingLWEKeyGenerator(MCFERingLWEKeyGenerator):
    """Key generator for threshold BMMS21 identity-based MCFE over Ring-LWE."""

    _scheme_type = CryptoCONST.TYPE_TMCFE_RING_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for
                :class:`~pyfe4ai.schemes.mcfe.ring_lwe.MCFERingLWEKeyGenerator`,
                plus ``s``, ``t``, ``lst_sid``, ``stat_param`` (``kappa`` of
                the flooding), ``flooding`` (one of :data:`FLOODINGS`,
                default ``"uniform"``, ``"gaussian"`` for the partial
                decryptions of versions <= 0.3.0 and under ``"kl"``; see the
                module docstring), ``sharing`` (``"replicated"``, the default
                for ``t < s``, ``"shamir"``, the default for ``t = s``, or the
                opt-in ``"replicated-seeded"``; see
                :func:`~pyfe4ai.schemes.threshold.threshold_utils.resolve_sharing`)
                and, opt-in, ``flooding_accounting = "kl"`` with
                ``max_partial_decryptions`` (``t = s``, Shamir sharing,
                Gaussian flooding; see the module docstring);
                ``ct_format`` (one of :data:`CT_FORMATS`, default
                ``"binary"``, published in the public parameters, whose
                clients and servers then use format 2; ``"json"`` for the
                format of versions <= 0.3.0);
                ``ring_modulus``: ``"rns"`` (default): the flooded ``q`` is
                a product of at least :data:`MIN_RNS_FACTORS` NTT primes
                below 2^62 (four for the 212 to 218 bits of the
                ``standard`` profile, four up to 248 bits;
                :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.rns_ntt_modulus_for`),
                each above ``s``, published as ``q_factors``, with the
                parameter record ``param-rns.json``; or ``"prime"``, an NTT
                prime (record ``param.json``, as in earlier versions);
                ``key_shares`` (one of :data:`KEY_SHARES`): ``"split"``, the
                default for the Shamir and replicated sharings, or
                ``"full"``, the default (and only choice) for
                ``"replicated-seeded"`` -- what
                :meth:`get_decryption_keys` issues without a ``key_part``.

        Raises:
            FEValidationError: For an unknown ``ct_format``,
                ``ring_modulus`` or ``key_shares``, or ``key_shares="split"``
                with the ``"replicated-seeded"`` sharing.
        """
        self.flooding_accounting, self.max_partial_decryptions = flooding_config(config)
        # the KL bound is for Gaussian flooding noise
        default_flooding = "gaussian" if self.flooding_accounting == KL else DEFAULT_FLOODING
        self.flooding = config.get("flooding", default_flooding)
        if self.flooding not in FLOODINGS:
            raise FEValidationError("unknown flooding {!r}; one of {}".format(self.flooding, ", ".join(FLOODINGS)))
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        s = config.get("s") or len(config.get("lst_sid", [])) or 1
        self.sharing = resolve_sharing(config, s, self.t, "mcfe/ring_lwe_threshold")
        if self.flooding_accounting == KL:
            check_kl_threshold(self.t, s)
            if self.flooding != "gaussian" or self.sharing != "shamir":
                raise FEValidationError(
                    "flooding_accounting='kl' accounts for Gaussian flooding of Shamir key shares "
                    "(t = s): got flooding={!r}, sharing={!r}".format(self.flooding, self.sharing)
                )
        self._threshold_construction = _construction_id(self.sharing, self.flooding, self.flooding_accounting)
        self.lst_sid = config.get("lst_sid", ["sid_{}".format(i) for i in range(s)])
        self.stat_param = int(config.get("stat_param", 64))
        self.ct_format = pp_ct_format({"ct_format": config.get("ct_format", DEFAULT_CT_FORMAT)})
        self.key_shares = config.get("key_shares", "full" if self.sharing == "replicated-seeded" else "split")
        if self.key_shares not in KEY_SHARES:
            raise FEValidationError(
                "key_shares must be one of {}, not {!r}".format(", ".join(KEY_SHARES), self.key_shares))
        if self.key_shares == "split" and self.sharing == "replicated-seeded":
            raise FEValidationError("split key shares need the sharing 'shamir' or 'replicated'; "
                                    "use key_shares='full' with 'replicated-seeded'")
        self.dict_dk = {}
        super().__init__(dict(config, s=s, ring_modulus=config.get("ring_modulus", DEFAULT_RING_MODULUS)), **kwargs)
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)
        check_rns_servers(self.s, self.q_factors)
        if self.q_factors is not None and len(self.q_factors) > RNS_NATIVE_FACTORS:
            logger.warning(
                "ring_modulus='rns': q (%d bits) has %d factors, more than the native ring's %d; "
                "the kernels run on the Python path (the prime modulus is native below 2^253)",
                int(self.q).bit_length(), len(self.q_factors), RNS_NATIVE_FACTORS,
            )

    def _bmms21(self, ring_n: int) -> dict:
        """The MCFE parameters; with a prime ``q`` also for ``ring_modulus="rns"``
        (only a lower bound of the flooded modulus, see :meth:`_flooded_modulus`)."""
        return bmms21_selective_parameters(
            self.sec_param, self.n * self.eta, self.bound_x, self.bound_y, ring_n,
            security_bound_x=3 * self.bound_x,
        )

    def _factors_for(self, ring_n: int) -> list | None:
        factors = self._flooded_modulus(ring_n, *self._bmms21_tuple(ring_n))[2]
        return None if factors is None else [gp.digits(f) for f in factors]

    def _param_verification(self, param: dict) -> bool:
        if not (
            super()._param_verification(param)
            and param.get("threshold_construction") == self._threshold_construction
            and param.get("s") == self.s
            and param.get("stat_param") == self.stat_param
            and all(param.get(key) == value for key, value in self._flooding_fields().items())
        ):
            return False
        flood = self._flooded_modulus(self.ring_n, *self._bmms21_tuple(self.ring_n))[1]
        if self.flooding == "uniform":
            return param.get("flooding") == "uniform" and param.get("flood_bits") == flood
        return param.get("sigma_f") == str(flood)

    def _apply_parameters(self, param: dict) -> None:
        super()._apply_parameters(param)
        if self.flooding == "uniform":
            self.sigma_f, self.flood_bits = None, int(param["flood_bits"])
        else:
            self.sigma_f = int(param["sigma_f"])

    def _flooding_fields(self) -> dict:
        """Record fields of the flooding accounting (none for the default)."""
        if self.flooding_accounting != KL:
            return {}
        return {"flooding_accounting": KL, "max_partial_decryptions": self.max_partial_decryptions}

    def _flooded_modulus(self, ring_n: int, p, q, sigma1, sigma2, sigma3) -> tuple:
        """Return ``(q', flood, factors)``: the BMMS21 modulus *q* enlarged for
        flooding (an NTT prime, or for ``ring_modulus="rns"`` a product of
        NTT primes, ``factors``; else ``None``), and ``sigma_f`` (Gaussian) or
        ``tau`` (uniform on ``[-2^tau, 2^tau)``)."""
        noise_bound = self.n * self.eta * int(self.bound_y) * int(
            2 * ring_n * GAUSSIAN_TAIL_CUT ** 2 * sigma1 * sigma2
            + GAUSSIAN_TAIL_CUT * sigma3 + 1
        )
        if self.flooding == "uniform":
            flood = noise_bound.bit_length() + self.stat_param  # 2^flood > 2^stat_param B
            budget = noise_bound + self.s * (1 << flood)
        elif self.flooding_accounting == KL:
            samples = kl_flooded_samples(self.s, self.max_partial_decryptions, ring_n)
            flood = kl_flooding_width(noise_bound, samples, self.stat_param)
            budget = noise_bound + self.s * kl_tail_cut(samples, self.stat_param, flood) * flood
        else:
            flood = noise_bound << self.stat_param
            budget = noise_bound + self.s * GAUSSIAN_TAIL_CUT * flood
        lower = max(q, p * (2 * budget + 4))
        if self.ring_modulus == "rns":
            q_rns, factors = rns_ntt_modulus_for(lower, ring_n, p, MIN_RNS_FACTORS)
            return q_rns, flood, factors
        return next_ntt_prime(lower, ring_n), flood, None

    def _modulus_for(self, ring_n: int):
        return self._flooded_modulus(ring_n, *self._bmms21_tuple(ring_n))[0]

    def _generate_and_save(self, param_file: str) -> None:
        super()._generate_and_save(param_file)
        self.q, flood, factors = self._flooded_modulus(
            self.ring_n, self.p, self.q, self.sigma1, self.sigma2, self.sigma3
        )
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)
        with open(param_file, "r", encoding="utf-8") as f:
            param = json.load(f)
        if factors is not None:  # the default's records stay as they were
            self.q_factors = register_rns_modulus(self.q, factors)
            param.update({"ring_modulus": "rns", "q_factors": [gp.digits(f) for f in factors]})
        param.update(
            {
                "threshold_construction": self._threshold_construction,
                "s": self.s,
                "stat_param": self.stat_param,
                "q": gp.digits(self.q),
                **self._flooding_fields(),
            }
        )
        if self.flooding == "uniform":
            self.sigma_f, self.flood_bits = None, flood
            param.update({"flooding": "uniform", "flood_bits": flood})
        else:
            self.sigma_f = flood
            param["sigma_f"] = str(flood)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(param, f)

    def __getstate__(self) -> dict:
        return _versioned_state(super().__getstate__())

    def __setstate__(self, state) -> None:
        _check_state(self, state)

    def setup(self) -> None:
        """MCFE setup and the client-server MAC keys; forgets previously shared keys."""
        super().setup()
        self.dict_dk = {}
        self.dict_weight_shares = OrderedDict()
        self.mac_keys = client_server_mac_keys(self.lst_nid, self.lst_sid)
        self._noise_masters = {sid: secrets.token_hex(32) for sid in self.lst_sid}
        pp = self.get_public_parameters()
        self._binding = pp_binding(pp, binding_fields(pp), _BINDING_DEFAULTS)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Client *nid*'s MCFE keys plus ``mac_keys`` (``{sid: key}``) for the
        ciphertext tags."""
        sk = super().get_private_keys(nid, **kwargs)
        if sk is not None:
            sk["mac_keys"] = dict(self.mac_keys[nid])
        return sk

    def get_public_parameters(self) -> dict:
        """MCFE public parameters plus ``s``, ``t``, ``lst_sid``, ``flooding``,
        ``flood_bits`` (uniform) or ``sigma_f`` (Gaussian), ``sharing``,
        ``threshold_construction``, ``security_profile``, ``ct_format`` (only
        ``"binary"``; absent means ``"json"``) and, under ``"kl"``,
        ``flooding_accounting``, ``max_partial_decryptions``, ``stat_param`` and
        ``flood_tail_cut``."""
        pp = super().get_public_parameters()
        pp.update(
            {"s": self.s, "t": self.t, "lst_sid": list(self.lst_sid), "flooding": self.flooding,
             "sharing": self.sharing, "threshold_construction": self._threshold_construction,
             "security_profile": self.security_profile}
        )
        if self.flooding == "uniform":
            pp["flood_bits"] = self.flood_bits
        else:
            pp["sigma_f"] = str(self.sigma_f)
        if self.ct_format != "json":  # the public parameters of JSON stay as they were
            pp["ct_format"] = self.ct_format
        if self.flooding_accounting == KL:
            samples = kl_flooded_samples(self.s, self.max_partial_decryptions, self.ring_n)
            pp.update({
                **self._flooding_fields(),
                "stat_param": self.stat_param,
                "flood_tail_cut": kl_tail_cut(samples, self.stat_param, self.sigma_f),
            })
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of the key for ``(y, label, n_chunks)``.

        Args:
            sid: The server.
            **kwargs: ``credentials`` with ``fusion_weight``, ``label``,
                optionally ``n_chunks`` and ``key_part``: ``"label"`` --
                only the label's terms ``z`` and ``z_chunks``, the per-round
                part; the label-independent ``sk_y`` comes once per weights
                from :meth:`get_weight_key_share` -- or ``"full"``, a whole
                key share (default: ``"label"`` if the config's
                ``key_shares`` is ``"split"``, the default but for the
                replicated-seeded sharing, else ``"full"``) and, with
                ``"label"``, ``label_term`` (default ``True``; ``False``
                leaves out ``z``, which only single ciphertexts of the label
                itself need).

        Returns:
            Dict with shares ``sk_y``, ``z``, ``z_chunks`` and ``label``,
            ``sid``, ``lst_sid``, ``weights_id``, ``noise_key`` and
            ``mac_keys`` (``{nid: key}``), or ``None`` if *sid* is unknown.
            With replicated sharing the shares are replaced by ``sharing``
            and ``pieces`` (``[{"set": T, "share": {"sk_y", "z", "z_chunks"}}]``);
            with ``"replicated-seeded"`` ``pieces`` holds the correction
            piece (or nothing), and ``seeds`` (``[{"set": T, "seed": hex}]``)
            and ``layout`` give the others. A label part has ``part =
            "label"`` and ``share_set_id`` (that of the weight part it
            completes, the weights' current share set when it is issued)
            and no ``sk_y``; label parts are kept with that sharing of
            ``sk_y`` (later calls return the same shares, for the last
            :data:`LABEL_PARTS_KEPT` labels) and dropped with it. Every key
            share part of one sharing has its ``sharing_id`` (the partial
            decryptions carry it; combiners refuse partials of different
            sharings). With binary public parameters
            (``ct_format="binary"``) every polynomial of the Shamir and
            replicated sharings is bytes (``ceil(bits(q)/8)``-byte words, the
            same values).

        Raises:
            FEValidationError: For an unknown ``key_part``, a label part with
                neither ``z`` nor chunk keys, or ``key_part="label"`` with the
                ``"replicated-seeded"`` sharing.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for Threshold MCFE Ring-LWE")
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        part = credentials.get("key_part", "label" if self.key_shares == "split" else "full")
        if part not in ("full", "label"):
            raise FEValidationError("key_part must be 'full' or 'label', not {!r}".format(part))
        label_term = bool(credentials.get("label_term", True))
        weights_id = fusion_weight_id(credentials.get("fusion_weight"))
        n_chunks = int(credentials.get("n_chunks", 0))
        ident = [weights_id, credentials.get("label"), n_chunks]
        if part == "label":
            if not label_term and not n_chunks:
                raise FEValidationError("a label key share needs the label's term or chunk keys")
            ident += [part, label_term]
        ident = json.dumps(ident)
        if part == "label":
            parts = self._label_parts(credentials, weights_id, n_chunks, label_term, ident)
        else:
            if ident not in self.dict_dk:
                key = super().get_decryption_keys(sid, credentials=credentials)
                self.dict_dk[ident] = self._share_key(key, weights_id)
            parts = self.dict_dk[ident]
        dk = dict(parts[sid])
        dk.update(self._share_fields(sid, weights_id))
        return dk

    #: Attempts of :meth:`_label_parts` to issue a label part for a share set
    #: that is not evicted while the label part is made.
    _LABEL_PART_ATTEMPTS = 8

    def _label_parts(self, credentials: dict, weights_id: str, n_chunks: int, label_term: bool, ident: str) -> dict:
        """``{sid: label part}`` for *ident*, of the weights' current share set:
        kept with that sharing of ``sk_y`` (:meth:`_weight_entry`; evicted
        with it, without a scan of the other keys; the last
        :data:`LABEL_PARTS_KEPT` labels), made on first use. A label part,
        cached or made, is handed out only while its sharing is the weights'
        current one (checked under the lock): one whose sharing another thread
        evicted meanwhile is made again for the weights' new share set, so none
        is issued for a sharing the key generator no longer hands out.

        Raises:
            FESchemeError: If the sharing was evicted during every attempt
                (more than :data:`WEIGHT_SHARINGS_KEPT` weights in use at once).
        """
        fusion_weight = credentials.get("fusion_weight")
        for _ in range(self._LABEL_PART_ATTEMPTS):
            entry = self._weight_entry(fusion_weight)
            with _STORE_LOCK:
                if self.dict_weight_shares.get(weights_id) is not entry:
                    continue  # evicted since _weight_entry returned it: the weights' new share set
                cached = entry["label_parts"]
                parts = cached.pop(ident, None)
                if parts is not None:
                    cached[ident] = parts  # most recently used
                    return parts
            made = self._label_part(credentials, weights_id, n_chunks, label_term, entry)
            with _STORE_LOCK:  # still the weights' current share set: keep it (of two threads, one wins)
                if self.dict_weight_shares.get(weights_id) is entry:
                    cached = entry["label_parts"]
                    parts = cached.setdefault(ident, made)
                    while len(cached) > LABEL_PARTS_KEPT:
                        del cached[next(iter(cached))]
                    return parts
        raise FESchemeError(
            "the sharing of these weights was evicted while their label part was made, {} times: more than "
            "WEIGHT_SHARINGS_KEPT = {} weights in use at once".format(self._LABEL_PART_ATTEMPTS, WEIGHT_SHARINGS_KEPT))

    def _label_part(self, credentials: dict, weights_id: str, n_chunks: int, label_term: bool, entry: dict) -> dict:
        """``{sid: label part}``: shares of the terms ``z''`` of the label (if
        *label_term*) and of its chunks, computed for these labels only (binary
        end to end for binary public parameters), of the share set of *entry*
        (:meth:`_weight_entry`)."""
        label = credentials.get("label")
        if not isinstance(label, str):
            raise FEValidationError("identity-based MCFE keys need a string label")
        fusion_weight = credentials.get("fusion_weight")
        labels = ([label] if label_term else []) + [chunk_label(label, c) for c in range(n_chunks)]
        binary = self.ct_format == "binary" and self.sharing != "replicated-seeded"
        # the terms on the MIFE key's z (validated weights; sk_y is the weight part's)
        terms = self._label_terms(self._key_z(fusion_weight), fusion_weight, labels, binary=binary)
        material = {"z": terms[0]} if label_term else {}
        material["z_chunks"] = list(terms[1:] if label_term else terms)
        sharing_id = secrets.token_hex(16)  # this sharing of the terms (another one if they are shared again)
        return {
            sid: dict(shares, label=label, noise_key=self._noise_key(sid, weights_id, label), part="label",
                      share_set_id=entry["share_set_id"], label_hash=self.label_hash, sharing_id=sharing_id)
            for sid, shares in self._share_material(material).items()
        }

    def get_weight_key_share(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of the label-independent part of the key
        for the weights ``y``: the shares of ``sk_y`` (``{nid: sk_{i,y}}``,
        or replicated pieces of it), issued once per weights (later calls
        return the same shares), with ``part = "weights"``, ``share_set_id``
        and the fields of :meth:`get_decryption_keys`. Servers keep it
        (:meth:`ThresholdMCFERingLWE.add_weight_key_share`); per label they
        then need only ``get_decryption_keys(..., key_part="label")``.

        The shares of ``sk_y`` serve every label: a linear sharing reused with
        fresh sharings of the label terms is a sharing of each round's key, and
        ``t - 1`` servers' shares stay independent of the keys (each sharing
        has its own randomness); the flooding noise is seeded as before.

        Args:
            sid: The server.
            **kwargs: ``credentials`` with ``fusion_weight``.

        Raises:
            FEValidationError: With the ``"replicated-seeded"`` sharing.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for Threshold MCFE Ring-LWE")
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        entry = self._weight_entry(credentials.get("fusion_weight"))
        share = dict(entry["shares"][sid], part="weights", share_set_id=entry["share_set_id"])
        share.update(self._share_fields(sid, entry["weights_id"]))
        return share

    def _weight_entry(self, fusion_weight) -> dict:
        """The sharing of ``sk_y`` for the weights, made on first use:
        ``{"weights_id", "share_set_id", "shares", "label_parts"}`` (the label
        parts issued for it, see :meth:`_label_parts`)."""
        if self.sharing == "replicated-seeded":
            raise FEValidationError("split key shares need the sharing 'shamir' or 'replicated'")
        weights_id = fusion_weight_id(fusion_weight)
        with _STORE_LOCK:
            store = self.__dict__.get("dict_weight_shares")
            if not isinstance(store, OrderedDict):
                store = self.dict_weight_shares = OrderedDict(store or {})
            entry = store.get(weights_id)
            if entry is not None:
                store.move_to_end(weights_id)
                return entry
        # the MIFE key of the weights (sk_y; its z is the label terms' base)
        key = MIFERingLWEKeyGenerator.get_decryption_keys(self, None, credentials={"fusion_weight": fusion_weight})
        made = {
            "weights_id": weights_id,
            "share_set_id": secrets.token_hex(16),
            "shares": self._share_material({"sk_y": dict(key["sk_y"])}),
            "label_parts": {},
        }
        with _STORE_LOCK:  # of two threads making it, both return the one kept
            entry = store.setdefault(weights_id, made)
            while len(store) > WEIGHT_SHARINGS_KEPT:  # with their label parts
                store.popitem(last=False)
        return entry

    def _share_fields(self, sid: str, weights_id: str) -> dict:
        """The fields of every key share part of server *sid*."""
        return {
            "sid": sid,
            "lst_sid": list(self.lst_sid),
            "weights_id": weights_id,
            "mac_keys": {nid: self.mac_keys[nid][sid] for nid in self.lst_nid},
            # how the noise is drawn and what the key share is of (both sharings)
            "noise_version": FLOODING_NOISE_VERSION,
            "noise_sampler": FLOODING_SAMPLER,
            "flooding": self.flooding,
            "label_hash": self.label_hash,
            "pp_binding": self._binding,
        }

    def _share_key(self, key: dict, weights_id: str) -> dict:
        material = {"sk_y": dict(key["sk_y"]), "z": key["z"], "z_chunks": list(key["z_chunks"])}
        sharing_id = secrets.token_hex(16)
        return {
            sid: dict(shares, label=key["label"], noise_key=self._noise_key(sid, weights_id, key["label"]),
                      sharing_id=sharing_id)
            for sid, shares in self._share_material(material).items()
        }

    def _share_material(self, material: dict) -> dict:
        """``{sid: shares}`` of *material* (entries of ``sk_y``, ``z``,
        ``z_chunks``) in the sharing: Shamir shares in the structure of
        *material*, or ``sharing`` and the pieces of the replicated (seeded)
        sharing; binary polynomials for binary public parameters (Shamir and
        replicated)."""
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())
        binary = self.ct_format == "binary" and self.sharing != "replicated-seeded"
        try:  # every polynomial ring_n values or ring_n words (a misaligned one is never dealt)
            polys = list(material.get("sk_y", {}).values()) + ([material["z"]] if "z" in material else []) + list(
                material.get("z_chunks", []))
        except (AttributeError, TypeError) as exc:
            raise FEKeyError("malformed key material") from exc
        _check_key_polys(polys, self.q, self.ring_n)
        if self.sharing == "replicated":
            pieces = share_replicated(material, self.t, points, self.q, binary=binary)
            return {sid: {"sharing": "replicated", "pieces": pieces_held(pieces, sid_v[sid])} for sid in self.lst_sid}
        if self.sharing == "replicated-seeded":
            ctx = native_backend.ring_context(self.q, self.ring_n)  # expands the pieces natively
            sharing = share_replicated_seeded(material, self.t, points, self.q, ctx=ctx)
            return {sid: {"sharing": "replicated-seeded", **seeded_pieces_held(sharing, sid_v[sid])}
                    for sid in self.lst_sid}
        # a mix of bytes and value lists is dealt as values (as the Python dealing reads it)
        material = uniform_material(material, self.q)
        sk_y = material.get("sk_y", {})
        polys = list(sk_y.values()) + ([material["z"]] if "z" in material else []) + list(material.get("z_chunks", []))
        shared = self._share_polys_native(polys, points, binary)
        if shared is None:
            def share_poly(coeffs) -> dict:
                shares = [share_secret(v, self.t, points, self.q) for v in _poly_values(coeffs, self.q, self.ring_n)]
                if binary:
                    return {sid: encode_values([sh[sid_v[sid]] for sh in shares], self.q) for sid in self.lst_sid}
                return {sid: [gp.digits(sh[sid_v[sid]]) for sh in shares] for sid in self.lst_sid}

            shared = [share_poly(poly) for poly in polys]
        out = {sid: {} for sid in self.lst_sid}
        pos = 0
        if "sk_y" in material:
            for sid in self.lst_sid:
                out[sid]["sk_y"] = {nid: shared[pos + i][sid] for i, nid in enumerate(sk_y)}
            pos += len(sk_y)
        if "z" in material:
            for sid in self.lst_sid:
                out[sid]["z"] = shared[pos][sid]
            pos += 1
        if "z_chunks" in material:
            for sid in self.lst_sid:
                out[sid]["z_chunks"] = [cs[sid] for cs in shared[pos:]]
        return out

    def _noise_key(self, sid: str, weights_id: str, label) -> str:
        # one per server, weights and label: a re-sharing for another
        # n_chunks (same key part) must not come with fresh noise
        return hmac.new(
            bytes.fromhex(self._noise_masters[sid]),
            json.dumps([weights_id, label]).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _share_polys_native(self, polys: list, points: list, binary: bool = False) -> list | None:
        """:func:`share_secret` of every coefficient of *polys* in one native
        call (coefficients drawn as ``share_secret`` draws them, in the same
        order), as ``[{sid: shares}]`` (with *binary* each share polynomial as
        bytes); ``None`` when the Python path applies."""
        ctx = _native_ring(self.q, self.ring_n)
        if ctx is None:
            return None
        width = (int(self.q).bit_length() + 7) // 8
        if all(isinstance(poly, (bytes, bytearray)) for poly in polys):  # binary key terms
            flat = b"".join(polys)
            polys = [range(len(poly) // width) for poly in polys]  # their lengths in values
        elif any(isinstance(poly, (bytes, bytearray)) for poly in polys):
            # bytes iterated as values would be other coefficients: the callers make the material uniform
            raise FEKeyError("key material mixing binary and value polynomials")
        else:
            flat = [v for poly in polys for v in poly]
        rnd = native_backend.random_input(4 * width * len(flat) * (self.t - 1) + 64)
        try:
            out = native_backend.ext().shamir_share(
                ctx.cap, flat, self.t, points, rnd,
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
            shared.append({sid: out[i][start * unit:(start + len(poly)) * unit] for i, sid in enumerate(self.lst_sid)})
            start += len(poly)
        return shared




class ThresholdMCFERingLWE(MCFERingLWE):
    """Crypto operations for threshold BMMS21 identity-based MCFE over Ring-LWE.

    A server refuses Shamir key shares with ``t < s`` (versions <= 0.3.0)
    unless its config has ``legacy_shamir = "warn"`` (see
    :data:`~pyfe4ai.schemes.threshold.threshold_utils.LEGACY_SHAMIR_POLICIES`).
    """

    _binary_ciphertexts = False  # format 1 only as the step before format 2 (binary public parameters)

    def __init__(self, config: dict, **kwargs) -> None:
        """As :class:`~pyfe4ai.schemes.mcfe.ring_lwe.MCFERingLWE`; under
        ``flooding_accounting = "kl"`` a server needs ``partial_budget``, the
        :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget`
        that its deployment keeps for it; ``legacy_shamir`` as in the class
        docstring. The ciphertext format is that of the public parameters
        (``ct_format``, :func:`pp_ct_format`); a ``ct_format`` in *config*
        must name the same.

        Raises:
            FEValidationError: For a ``ct_format`` other than the public parameters'.
        """
        pp = (config.get("keys") or {}).get("pp") or {}
        self.threshold_format = pp_ct_format(pp)
        if config.get("ct_format", self.threshold_format) != self.threshold_format:
            raise FEValidationError(
                "ct_format {!r} is not the public parameters' {!r}".format(config.get("ct_format"), self.threshold_format)
            )
        binary = self.threshold_format == "binary"
        self._binary_ciphertexts = binary  # the MCFE layer writes format 1, converted to format 2
        super().__init__(dict(config, ct_format="binary" if binary else "json"), **kwargs)
        # the parameters this object was made for: the format above is fixed, and an in-place
        # change of the caller's dict must not let one key part be answered in two derivations
        self._pp_snapshot = self._pp_digest()
        if self.pp.get("q_factors") is not None:
            check_rns_servers(self.pp.get("s", len(self.pp.get("lst_sid", []))), self.pp["q_factors"])
        self.legacy_shamir = legacy_shamir_policy(config)
        self._partial_budget = config.get("partial_budget")
        flood = kl_flooding(self.pp, self.pp["ring_n"]) if getattr(self, "pp", None) else None
        if flood is not None and self._partial_budget is not None:
            server_budget(self.pp, self._partial_budget, flood)

    def _kl_flooding(self):
        """KL flooding parameters and the server's budget, or ``(None, None)``.

        Raises:
            FESchemeError: Under ``"kl"`` without a ``partial_budget``.
        """
        flood = kl_flooding(self.pp, self.pp["ring_n"])
        if flood is None:
            return None, None
        if self.pp.get("flooding", LEGACY_FLOODING) != "gaussian":
            raise FESchemeError("flooding_accounting='kl' public parameters with a flooding other than Gaussian")
        return flood, server_budget(self.pp, self._partial_budget, flood)

    def _mac_keys(self) -> dict:
        mac_keys = (self.sk or {}).get("mac_keys") if self._has_private_keys() else None
        if not mac_keys:
            raise FEKeyError("encryption key lacks the ciphertext MAC keys")
        return mac_keys

    def _pp_digest(self) -> str:
        """:func:`~pyfe4ai.schemes.threshold.threshold_utils.pp_binding` of the public parameters (hex)."""
        return pp_binding(self.pp, binding_fields(self.pp), _BINDING_DEFAULTS)

    def _check_parameters_unchanged(self) -> None:
        """Refuse to work on public parameters changed (in the caller's dict)
        since this object was made: its ciphertext format is fixed then, while
        bindings and noise contexts read the parameters at each call.

        Raises:
            FESchemeError: If their bound fields or format changed.
        """
        if self._pp_digest() != self.__dict__.get("_pp_snapshot") or pp_ct_format(self.pp) != self.threshold_format:
            raise FESchemeError("the public parameters changed after this object was made: make a new one")

    def _tag(self, ct: dict, mac_keys: dict, encoded_label: str, coded: dict, pp_digest=None) -> dict:
        """Tag *ct* for every server: the JSON payload, or for binary public
        parameters its format-2 form (converted from format 1 here) and
        :func:`binary_tag_message`. *coded* is *ct* as tagged (with its layer's
        label hash code)."""
        if self.threshold_format != "binary":
            return tag_ciphertext(ct, mac_keys, _TAG_DOMAIN, _tag_payload(encoded_label, coded))
        q, eta, ring_n = self.pp["q"], self.pp["eta"], self.pp["ring_n"]
        ct["ct"] = threshold_ciphertext_from_format1(ct["ct"], ct["k"], q, eta, ring_n)
        coded = dict(coded, ct=ct["ct"])
        digests = binary_ciphertext_digests(coded, q, eta, ring_n)
        digest = bytes.fromhex(pp_digest or self._pp_digest())
        return tag_ciphertext_message(ct, mac_keys, binary_tag_message(digest, encoded_label, coded, digests))

    def encrypt(self, matrix_pt: list[list[int]], label: str) -> dict:
        """MCFE encryption under *label*, tagged for every server (format 2
        for binary public parameters)."""
        self._check_parameters_unchanged()
        mac_keys = self._mac_keys()
        ct = super().encrypt(matrix_pt, label)
        return self._tag(ct, mac_keys, encode_label(label), ct)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Packed ndarray encryption; every chunk is tagged under its chunk label."""
        self._check_parameters_unchanged()
        mac_keys = self._mac_keys()
        out = super().encrypt_lst_ndarray(lst_ndarray, **kwargs)
        pp_digest = self._pp_digest() if self.threshold_format == "binary" else None
        for layer in out:
            for j, ct in enumerate(layer["chunks"]):
                encoded = encode_label(chunk_label(kwargs["label"], layer["first_chunk"] + j))
                self._tag(ct, mac_keys, encoded, _chunk_with_code(ct, layer), pp_digest)
        return out

    #: Weight key shares a server keeps (:meth:`add_weight_key_share`).
    WEIGHT_SHARES_KEPT = 16

    #: Fields in which the weight and the label part of a key share must agree.
    _SPLIT_FIELDS = ("sid", "lst_sid", "weights_id", "share_set_id", "sharing", "mac_keys", "noise_version",
                     "noise_sampler", "flooding", "label_hash", "pp_binding")

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.pop("_hat_cache", None)  # native contexts
        return _versioned_state(state)

    def __setstate__(self, state) -> None:
        _check_state(self, state)

    def add_weight_key_share(self, share: dict) -> None:
        """Keep this server's weight key share
        (:meth:`ThresholdMCFERingLWEKeyGenerator.get_weight_key_share`) for
        the label key shares (``key_part="label"``) of the same weights and
        share set; the last :data:`WEIGHT_SHARES_KEPT` are kept. ``NTT(sk_y)``
        of the shares is cached across labels.

        Raises:
            FEKeyError: Unless *share* is a weight key share of this server.
        """
        if not isinstance(share, dict) or share.get("part") != "weights" or share.get("sid") != self.id:
            raise FEKeyError("not a weight key share of server {}".format(self.id))
        with _STORE_LOCK:
            store = self.__dict__.setdefault("_weight_shares", OrderedDict())
            store[(share.get("weights_id"), share.get("share_set_id"))] = share
            store.move_to_end((share.get("weights_id"), share.get("share_set_id")))
            while len(store) > self.WEIGHT_SHARES_KEPT:
                store.popitem(last=False)

    def has_weight_key_share(self, dk: dict) -> bool:
        """Whether this server keeps the weight key share that the label key
        share *dk* completes (same weights and share set): ``False`` means
        :meth:`add_weight_key_share` first. A full key share needs none."""
        if not isinstance(dk, dict) or dk.get("part") != "label":
            return True
        with _STORE_LOCK:
            return (dk.get("weights_id"), dk.get("share_set_id")) in self.__dict__.get("_weight_shares", {})

    def _key_parts(self, dk):
        """``(dk, weight)``: a full key share and ``None``, or a label part and
        the weight part kept for it.

        Raises:
            FEKeyError: For a weight part alone, a label part without its
                weight part, or parts that do not belong together.
        """
        if not isinstance(dk, dict) or dk.get("part") is None:
            return dk, None
        if dk.get("part") != "label":
            raise FEKeyError("a weight key share decrypts together with a label key share (key_part='label')")
        with _STORE_LOCK:
            weight = self.__dict__.get("_weight_shares", {}).get((dk.get("weights_id"), dk.get("share_set_id")))
        if weight is None:
            raise FEKeyError(
                "a label key share needs this server's weight key share of the same weights and share set: "
                "add_weight_key_share(key_generator.get_weight_key_share(sid, credentials={'fusion_weight': ...})) "
                "first (key shares are split by default; key_part='full' issues whole ones)")
        for field in self._SPLIT_FIELDS:
            if weight.get(field, "shamir" if field == "sharing" else None) != dk.get(
                    field, "shamir" if field == "sharing" else None):
                raise FEKeyError("the weight and the label key share do not belong together ({})".format(field))
        return dk, weight

    def _enrolled_share(self, dk, weight, lst_sid_enrolled, q, fields):
        """:func:`~pyfe4ai.schemes.threshold.threshold_utils.share_for_enrolled` of a key
        share, or of the label part and its weight part (``sk_y``; summed once
        per weight part and enrolled set)."""
        kwargs = dict(cache=share_cache(self), legacy_shamir=self.legacy_shamir,
                      profile=self.pp.get("security_profile"), ring_n=self.pp["ring_n"])
        if weight is None:
            return share_for_enrolled(dk, self.id, lst_sid_enrolled, self.pp["t"], q, fields=fields, **kwargs)
        held = dk["pieces"][0]["share"] if dk.get("pieces") else dk  # a label part may lack z (label_term=False)
        label_fields = tuple(f for f in fields if f != "sk_y" and f in held)
        share, multiplier, tag = share_for_enrolled(dk, self.id, lst_sid_enrolled, self.pp["t"], q,
                                                    fields=label_fields, **kwargs)
        wshare, wmult, wtag = share_for_enrolled(weight, self.id, lst_sid_enrolled, self.pp["t"], q,
                                                 fields=("sk_y",), **kwargs)
        if (wmult, wtag) != (multiplier, tag):
            raise FEKeyError("the weight and the label key share are of different sharings")
        share = dict(share, sk_y=wshare["sk_y"])
        return share, multiplier, tag

    def _sk_hat(self, ctx, poly, persistent: bool) -> bytes:
        """``NTT(sk share)`` for the native kernels; cached across labels for
        the shares of a kept weight part (*persistent*)."""
        if not persistent:
            return self._ntt_mont(ctx, poly)
        key = (id(poly), id(ctx))
        with _STORE_LOCK:
            cache = self.__dict__.setdefault("_hat_cache", OrderedDict())
            hit = cache.get(key)
            if hit is not None and hit[0] is poly and hit[1] is ctx:
                cache.move_to_end(key)
                return hit[2]
        hat = self._ntt_mont(ctx, poly)
        with _STORE_LOCK:
            cache[key] = (poly, ctx, hat)  # poly and ctx kept alive: their ids are not reused while cached
            while len(cache) > 64:
                cache.popitem(last=False)
        return hat

    @staticmethod
    def _ntt_mont(ctx, poly) -> bytes:
        if isinstance(poly, (bytes, bytearray)):
            try:
                return ctx.ntt_mont(bytes(poly))
            except ValueError as exc:
                raise FEKeyError("malformed binary key share") from exc
        try:
            return ctx.ntt_mont(poly)
        except (TypeError, ValueError) as exc:  # e.g. a value list holding bytes
            raise FEKeyError("malformed key share") from exc

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list,
        _z_share=None, _chunk=None, _sk_hats=None, _share=None,
    ) -> dict:
        """Compute this server's flooded partial decryption.

        Args:
            dict_ct: One ciphertext per client, produced under ``credentials["label"]``.
            credentials: Dict with ``fusion_weight`` and ``label``.
            dk: This server's key share (derived for that label).
            lst_sid_enrolled: The servers whose partials will be combined.

        Returns:
            Dict with ``w``, ``c_y``, ``k``, ``ct_digest``, ``sid``,
            ``lst_sid_enrolled`` and ``sharing_id`` (that of the key share,
            ``None`` for key shares without one).

        Raises:
            FESchemeError: Under ``flooding_accounting = "kl"`` also when the
                server has no ``partial_budget`` or its budget is spent
                (refused requests consume nothing).
        """
        self._check_parameters_unchanged()
        dk, weight = self._key_parts(dk)
        if not dk or not ("sk_y" in dk or "pieces" in dk or weight is not None):
            raise FEKeyError("no decryption keys provided")
        check_flooding_version(dk)
        sharing_id = dk.get("sharing_id")  # of the shares' sharing (key shares of earlier versions: none)
        if sharing_id is not None and not isinstance(sharing_id, str):
            raise FEKeyError("malformed key share: sharing_id is a string")
        flooding = self.pp.get("flooding", LEGACY_FLOODING)
        if flooding not in FLOODINGS:
            raise FEValidationError("unknown flooding {!r}".format(flooding))
        if dk.get("flooding", LEGACY_FLOODING) != flooding:
            raise FEKeyError("the key share was issued for another flooding than the public parameters'")
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        check_served_sharing(self.pp, dk, _CONSTRUCTIONS)
        construction = self.pp.get("threshold_construction")
        accounting = self.pp.get("flooding_accounting", PER_SAMPLE)
        if construction is not None and construction != _construction_id(dk.get("sharing", "shamir"), flooding, accounting):
            raise FESchemeError("the public parameters' threshold_construction is not that of their flooding")
        self._flooding_width()  # malformed flooding parameters: a validation error first
        fusion_weight = credentials.get("fusion_weight")
        if dk.get("label") != credentials.get("label"):
            raise FESchemeError("the key share was derived for another label")
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, fusion_weight)
        q = gp.mpz(self.pp["q"])
        # Shamir: (dk, lambda_k); replicated: the sum of the assigned pieces
        # (sk_y and z; the chunk keys come with _share), 1
        share, multiplier, tag = _share or self._enrolled_share(dk, weight, lst_sid_enrolled, q, ("sk_y", "z"))
        if set(dict_ct) != set(share["sk_y"]):
            raise FESchemeError("need one ciphertext for every client of the key")
        # only ciphertexts the clients tagged for this server under the label
        # (or chunk label) it serves: w is linear in ct1
        tag_label = credentials["label"] if _chunk is None else chunk_label(credentials["label"], _chunk)
        encoded = encode_label(tag_label)
        binary = self.threshold_format == "binary"
        q_int, eta, ring_n = int(q), self.pp["eta"], self.pp["ring_n"]
        digests = {}
        if binary:  # format 2 only: tags over the parameters' digest, label, code, k and the bytes' digests
            pp_digest = bytes.fromhex(self._pp_digest())

            def message_of(nid, ct):
                if not isinstance(ct, dict) or "ct0" in ct or "ct1" in ct:
                    raise ValueError("not a format-2 ciphertext")
                digests[nid] = binary_ciphertext_digests(ct, q_int, eta, ring_n)
                return binary_tag_message(pp_digest, encoded, ct, digests[nid])

            for nid, ct in dict_ct.items():  # no JSON ciphertexts under binary public parameters
                if isinstance(ct, dict) and ("ct0" in ct or "ct1" in ct):
                    raise FESchemeError(
                        "the ciphertext of {} is JSON; the public parameters' format is binary (ct_format, the "
                        "default): encrypt with these public parameters".format(nid))

            check_ciphertext_tag_messages(dict_ct, dk.get("mac_keys"), self.id, message_of)
        else:
            for nid, ct in dict_ct.items():  # no binary ciphertexts under JSON public parameters
                if isinstance(ct, dict) and "ct" in ct:
                    raise FESchemeError("the ciphertext of {} is binary; the public parameters' format is JSON".format(nid))
            check_ciphertext_tags(
                dict_ct, dk.get("mac_keys"), self.id, _TAG_DOMAIN,
                lambda nid, ct: _tag_payload(encoded, ct),
            )
        # the key share of the public parameters' label hash
        self._check_key_label_hash(dk)
        # ... and of their modulus and flooding (width, tail cut, accounting), format and factors
        binding = check_pp_binding(self.pp, dk, binding_fields(self.pp), _BINDING_DEFAULTS)
        delta = q // gp.mpz(self.pp["p"])
        z_share = share.get("z") if _z_share is None else _z_share
        if z_share is None:
            raise FEKeyError("the key share has no term of the label itself (label_term=False): chunks only")
        if not isinstance(share.get("sk_y"), dict) or "other" in poly_kinds([z_share, share["sk_y"]]):
            raise FEKeyError("malformed key share: polynomials must be value lists or bytes")  # e.g. memoryview
        _check_key_polys([z_share, *share["sk_y"].values()], q, ring_n)
        hats = {} if _sk_hats is None else _sk_hats
        ctx = _native_ring(q, ring_n)
        native = cts = None
        if binary:
            batch = {d[0] for d in digests.values()}
            ct_digest, key_inputs = binary_request_digests(digests)
        else:
            native = None if ctx is None else self._native_inputs(ctx, dict_ct, share, fusion_weight, z_share)
            if native is None:
                cts = {}
                for nid, ct in dict_ct.items():
                    try:
                        cts[nid] = canonical_ring_ciphertext(ct, q_int, eta, ring_n)
                    except FESchemeError as exc:
                        raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
                batch = {k for k, _, _ in cts.values()}
            else:
                nids, ks, packed, ct_digest, key_inputs = native
                batch = set(ks)
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")
        k = batch.pop()
        # every (authenticated, well-formed) ciphertext of that label hash too:
        # else the result would be wrong, not refused
        check_ciphertext_label_hashes(dict_ct.values(), self.label_hash)
        if binary and ctx is None:  # the Python reading of format 2 (strict)
            cts = {}
            for nid, ct in dict_ct.items():
                try:
                    _, ct0, ct1 = decode_threshold_ciphertext(ct["ct"], q_int, eta, ring_n)
                except ValueError as exc:
                    raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
                cts[nid] = (k, ct0, ct1)

        if cts is not None:
            key_part = [delta * v % q for v in _poly_values(z_share, q, ring_n)]
            c_y = [gp.mpz(0)] * len(next(iter(cts.values()))[1][0])  # n (JSON) or k (format 2)
            # sum_i ct1_i sk_i in the NTT domain, NTT(sk_i) cached across chunks
            # (_sk_hats); the same sum mod q as adding poly_mul(ct1_i, sk_i)
            ring = RingNTT.get(q, ring_n)
            acc = None
            for nid, (_, ct0, ct1) in cts.items():
                ct0 = [[gp.mpz(v) for v in row] for row in ct0]
                ct1 = [gp.mpz(v) for v in ct1]
                y_vec = [gp.mpz(int(v)) for v in fusion_weight[nid]]
                c_y = poly_add(c_y, mat_vec_mul(transpose(ct0), y_vec, q), q)
                if ring is None:
                    key_part = poly_add(key_part, poly_mul(ct1, _poly_values(share["sk_y"][nid], q, ring_n), q), q)
                    continue
                if nid not in hats:
                    hats[nid] = ring.forward(_poly_values(share["sk_y"][nid], q, ring_n))
                acc = ring.mul_add(acc, ring.forward(ct1), hats[nid])
            if acc is not None:
                key_part = poly_add(key_part, ring.inverse(acc), q)
            if not binary:
                ct_digest = ciphertexts_digest(cts)
                key_inputs = ciphertexts_digest({nid: ct1 for nid, (_, _, ct1) in cts.items()})

        # the noise is seeded with what the key part depends on -- the ct1
        # values and the key term (the label or chunk) -- so a request for the
        # same key part always draws the same noise
        # the partial depends on the enrolled set only through the tag (lambda,
        # or the assigned pieces)
        # and with the digest of this server's public parameters (the key share's
        # pp_binding, checked above): under other parameters a request gets
        # another noise, never the same noise with another key term
        context = json.dumps([key_inputs, encoded, tag, binding]).encode("utf-8")
        seed = bytes.fromhex(dk["noise_key"]) + context
        # format 2 outputs the first k coefficients only, with the first k noise
        # values (for the uniform noise a prefix of the SHAKE stream, for the
        # others those of the full draw): an answer for another k is a prefix
        # or an extension of this one, never a second noise
        count = k if binary else ring_n
        flood, budget = self._kl_flooding()
        if flood is not None:
            flood.check_sampler()
            budget.consume(1)
            noise_args = {"noise": flood.sample(ring_n, XofRandom(seed))[:count]}
        elif self.pp.get("flooding", LEGACY_FLOODING) == "uniform":
            width = self._flooding_width()
            if ctx is not None and width < q.bit_length():  # drawn natively, the same values
                noise_args = {"flood": (width, uniform_flooding_material(seed))}
            else:
                noise_args = {"noise": uniform_flooding_poly(width, count, seed)}
        else:
            noise_args = {"noise": self._flooding_noise(ring_n, seed)[:count]}
        if cts is not None:
            noise = noise_args.get("noise")
            if noise is None:
                noise = uniform_flooding_poly(noise_args["flood"][0], count, seed)
            values = [(multiplier * v + e) % q for v, e in zip(key_part, noise)]
            if binary:
                w, c_y = encode_values(values, q), encode_values(c_y, q)
            else:
                w, c_y = [gp.digits(v) for v in values], [gp.digits(v) for v in c_y]
        else:
            ids = nids if native is not None else [nid for _, nid in _client_ids(dict_ct)]
            for nid in ids:  # NTT(sk_i share), cached across chunks (and labels, for a kept weight part)
                if ("native", nid) not in hats:
                    hats[("native", nid)] = self._sk_hat(ctx, share["sk_y"][nid], weight is not None)
            ys = [[int(v) for v in fusion_weight[nid]] for nid in ids]
            sk_hats = [hats[("native", nid)] for nid in ids]
            if native is not None:
                try:  # the ciphertexts are read already: what the call reads is the key share's
                    w, c_y = native_backend.ext().share_partial(
                        ctx.cap, packed, eta, ys, sk_hats, z_share, delta, multiplier, **noise_args,
                    )
                except (TypeError, ValueError) as exc:
                    raise FEKeyError("malformed key share") from exc
            else:
                try:
                    w, c_y, _ = native_backend.ext().share_partial_bin(
                        ctx.cap, [dict_ct[nid]["ct"] for nid in ids], eta, ys, sk_hats, z_share, delta, multiplier,
                        **noise_args,
                    )
                except (TypeError, ValueError) as exc:
                    if exc.args[:1] == ("malformed ciphertext",):
                        raise FESchemeError("malformed ciphertext of {}".format(ids[exc.args[1]])) from exc
                    if isinstance(z_share, (bytes, bytearray)):
                        raise FEKeyError("malformed binary key share") from exc
                    raise FEKeyError("malformed key share") from exc
        return {
            "w": w,
            "c_y": c_y,
            "k": k,
            "ct_digest": ct_digest,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
            "sharing_id": sharing_id,
        }

    def _flooding_width(self) -> int:
        """``flood_bits`` (uniform) or ``sigma_f`` (Gaussian) of the public parameters.

        Raises:
            FEValidationError: For an unknown flooding or a malformed width.
        """
        flooding = self.pp.get("flooding", LEGACY_FLOODING)
        if flooding == "uniform":
            bits = self.pp.get("flood_bits")
            if type(bits) is not int or bits < 1:
                raise FEValidationError("uniform flooding needs a positive integer flood_bits")
            return bits
        if flooding != "gaussian":
            raise FEValidationError("unknown flooding {!r}".format(flooding))
        try:
            sigma = int(self.pp["sigma_f"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FEValidationError("Gaussian flooding needs an integer sigma_f") from exc
        if sigma < 1:
            raise FEValidationError("Gaussian flooding needs a positive sigma_f")
        return sigma

    def _flooding_noise(self, ring_n: int, seed: bytes) -> list:
        """The flooding polynomial of the public parameters' distribution, a function of *seed*."""
        width = self._flooding_width()
        if self.pp.get("flooding", LEGACY_FLOODING) == "uniform":
            return uniform_flooding_poly(width, ring_n, seed)
        return _flooding_poly(width, ring_n, XofRandom(seed))

    def _native_inputs(self, ctx, dict_ct: dict, dk: dict, fusion_weight: dict, z_share) -> tuple | None:
        """The ciphertexts of :meth:`share_decrypt` read natively: ``(nids, ks,
        packed, ct_digest, key_inputs)``, the client ids sorted, the digests
        those of :func:`ciphertexts_digest`; ``None`` when the Python path must
        read them (values other than exact ints and plain decimal strings,
        other layouts -- also of the key share -- and malformed ciphertexts,
        whose errors it raises)."""
        eta, ring_n = self.pp["eta"], self.pp["ring_n"]
        if not all(isinstance(nid, str) for nid in dict_ct) or len(z_share) != ring_n:
            return None
        nids = sorted(dict_ct)  # the key order of json.dumps(..., sort_keys=True)
        ks, rows = [], []
        for nid in nids:
            ct = dict_ct[nid]
            if not isinstance(ct, dict) or "ct0" not in ct or "ct1" not in ct:
                return None
            k, ct0 = ct.get("k"), ct["ct0"]
            if type(k) is not int or not 1 <= k <= ring_n or type(ct0) not in (list, tuple) or len(ct0) != eta:
                return None
            if len(dk["sk_y"][nid]) != ring_n or len(fusion_weight[nid]) != eta:
                return None
            ks.append(k)
            rows.append((ct0, ct["ct1"]))
        out = native_backend.ext().share_inputs(ctx.cap, [json.dumps(nid) for nid in nids], ks, rows, eta)
        if out is None:
            return None
        packed, ct_text, key_text = out
        return nids, ks, packed, hashlib.sha256(ct_text).hexdigest(), hashlib.sha256(key_text).hexdigest()

    def combine_decrypt(self, partials) -> list | None:
        """Combine partial decryptions from at least ``t`` servers.

        Args:
            partials: ``{sid: share_decrypt(...)}`` for one enrolled set, or a
                list of such mappings for several enrolled sets, whose results
                must then agree (see
                :func:`~pyfe4ai.schemes.threshold.threshold_utils.combine_consistently`:
                servers are assumed honest-but-curious, and only this
                redundancy detects one that returns a wrong partial).

        Raises:
            FESchemeError: If a group is incomplete or invalid, or the groups
                disagree.
        """
        return combine_consistently(partials, self._combine_group)

    def _combine_group(self, dict_ct_prime: dict) -> list | None:
        """Combine at least ``t`` partial decryptions into the batch of inner products.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given or the shares
                do not match the enrolled set they were computed for;
                :class:`MixedSharingsError` if they are of different sharings
                (``sharing_id``).
        """
        self._check_parameters_unchanged()
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = list(dict_ct_prime.values())
        first = partials[0]
        # binary public parameters: bytes only; JSON: sequences only
        binary = self.threshold_format == "binary"
        values_type = (bytes, bytearray) if binary else (list, tuple)
        if any(not isinstance(p.get("w"), values_type) or not isinstance(p.get("c_y"), values_type) for p in partials):
            raise FESchemeError("partial decryptions not in the public parameters' format ({})".format(
                self.threshold_format))
        # shares of two sharings of one key do not reconstruct it (their partials would decode to a wrong value)
        sharing_ids = [p.get("sharing_id") for p in partials]
        if any(i is not None and not isinstance(i, str) for i in sharing_ids):
            raise FESchemeError("malformed partial decryption: sharing_id is a string")
        if len(set(sharing_ids)) > 1:
            raise MixedSharingsError(MIXED_SHARINGS)
        if any(
            (p["c_y"], p["ct_digest"], p["k"]) != (first["c_y"], first["ct_digest"], first["k"])
            for p in partials
        ):
            logger.warning("partial decryptions disagree on the ciphertexts they decrypted")
            return None
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        bound = (
            self.pp["n"] * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        )
        ctx = _native_ring(q, self.pp["ring_n"])
        if binary:
            return self._combine_binary(ctx, first, partials, q, delta, bound)
        k = first["k"]  # JSON: every partial holds at least k values (a shorter one would give a shorter result)
        if type(k) is not int or not 1 <= k <= self.pp["ring_n"] or any(
                len(p["w"]) < k or len(p["c_y"]) < k for p in partials):
            raise FESchemeError("malformed partial decryption")
        if ctx is not None and type(first["k"]) is int:
            out = native_backend.ext().share_combine(
                ctx.cap, first["c_y"], [p["w"] for p in partials], delta, bound, first["k"]
            )
            if out is not None:  # else values or layouts the Python code reads
                return out
        total = [gp.mpz(v) for v in first["c_y"]]
        for part in partials:
            total = [(a - gp.mpz(v)) % q for a, v in zip(total, part["w"])]
        return [decode_scaled(v, delta, q, bound) for v in total[: int(first["k"])]]

    def _combine_binary(self, ctx, first: dict, partials: list, q, delta, bound) -> list | None:
        """:meth:`_combine_group` of binary partial decryptions (``k`` values of
        ``ceil(bits(q) / 8)`` bytes each, strictly below q).

        Raises:
            FESchemeError: For a malformed partial decryption.
        """
        k = first["k"]
        if type(k) is not int or not 1 <= k <= self.pp["ring_n"]:
            raise FESchemeError("malformed partial decryption")
        if ctx is not None:
            try:
                out = native_backend.ext().share_combine_bin(
                    ctx.cap, bytes(first["c_y"]), [bytes(p["w"]) for p in partials], delta, bound, k)
            except ValueError as exc:
                raise FESchemeError("malformed partial decryption") from exc
            if out is not None:  # else a value beyond an int64: decoded below
                return out
        try:
            total = decode_values(first["c_y"], q, k)
            for part in partials:
                total = [(a - v) % q for a, v in zip(total, decode_values(part["w"], q, k))]
        except ValueError as exc:
            raise FESchemeError("malformed partial decryption") from exc
        return [decode_scaled(gp.mpz(v), delta, q, bound) for v in total]

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Partially decrypt every chunk of the clients' packed ndarray ciphertexts
        (under ``"kl"`` refused whole unless the budget covers all of them;
        each partial produced then counts)."""
        sample = next(iter(dict_ndarray_ct.values()))
        packed_share = None
        label_dk, weight = self._key_parts(dk) if dk else (dk, None)
        z_chunks = (label_dk or {}).get("z_chunks", [])
        if label_dk and "pieces" in label_dk and self.id in label_dk.get("lst_sid", []):
            # the assigned pieces (sk_y and the chunk keys), added up once for all chunks
            validate_enrolled(self.id, label_dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
            packed_share = self._enrolled_share(label_dk, weight, lst_sid_enrolled, gp.mpz(self.pp["q"]),
                                                ("sk_y", "z_chunks"))
            z_chunks = packed_share[0]["z_chunks"]
        if self.key_chunks(sample) > len(z_chunks):
            raise FEKeyError("the key share covers fewer chunks than the ciphertexts")
        flood, budget = self._kl_flooding()
        if flood is not None:
            flood.check_sampler()
            budget.check(sum(len(layer["chunks"]) for layer in sample))
        out = []
        sk_hats = {}  # NTT(sk_i), transformed once for all chunks
        for l, layer in enumerate(sample):
            parts = []
            for j in range(len(layer["chunks"])):
                dct_ct = {nid: _chunk_with_code(dict_ndarray_ct[nid][l]["chunks"][j], dict_ndarray_ct[nid][l])
                          for nid in dict_ndarray_ct}
                index = chunk_key_index(layer, j, len(z_chunks))
                parts.append(self.share_decrypt(
                    dct_ct, credentials, dk, lst_sid_enrolled,
                    _z_share=z_chunks[index], _chunk=index, _sk_hats=sk_hats, _share=packed_share,
                ))
            out.append({"shape": layer["shape"], "parts": parts})
        return out

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Combine chunk-wise partial decryptions from at least ``t`` servers.

        Raises:
            FESchemeError: If the servers' outputs differ in structure, or an
                element does not decrypt (servers disagree, or out of range).
        """
        groups = partial_groups(dict_ndarray_ct)
        sample = next(iter(groups[0].values()))
        layout = [(list(layer["shape"]), len(layer["parts"])) for layer in sample]
        for g in groups:
            for out in g.values():
                if [(list(layer["shape"]), len(layer["parts"])) for layer in out] != layout:
                    raise FESchemeError("partial decryptions differ in shape")
        result = []
        for l, layer in enumerate(sample):
            values = []
            for j in range(len(layer["parts"])):
                decoded = self.combine_decrypt(
                    [{sid: g[sid][l]["parts"][j] for sid in g} for g in groups]
                )
                if decoded is None or None in decoded:
                    raise FESchemeError("element ({}, chunk {}) does not decrypt".format(l, j))
                values.extend(decoded)
            if len(values) != int(np.prod(layer["shape"])):
                raise FESchemeError("partial decryptions do not match the shape {}".format(layer["shape"]))
            ary = np.array(values, dtype=float).reshape(layer["shape"])
            result.append(ary / pow(10, self.precision))
        return result
