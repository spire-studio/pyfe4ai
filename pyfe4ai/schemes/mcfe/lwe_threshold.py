"""
Threshold LWE-based multi-client inner-product FE: ABMPSW20 LWE MCFE with a
Shamir-shared functional key and noise-flooded partial decryptions.

| MCFE layer: Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner,
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle
| Model", SCN 2020, https://eprint.iacr.org/2020/788, Fig. 4 (see
| :mod:`pyfe4ai.schemes.mcfe.lwe`).
| Threshold layer in the setting of R. Xu et al., "TAPFed: Threshold Secure
| Aggregation for Privacy-Preserving Federated Learning", IEEE TDSC 2024:
| the aggregated key is Shamir-shared among ``s`` servers, any ``t`` decrypt.

* type:     secret-key multi-client encryption, threshold functional decryption
* setting:  Integer based (Z_q with a rounding map to Z_q0), random oracle

Construction (notation of :mod:`pyfe4ai.schemes.mcfe.lwe`):

* KeyGen:    ``sk_y = sum y_{i,j} Z_{i,j} mod q``; every coordinate is
  Shamir-shared with threshold ``t`` over ``Z_q`` (``q`` prime) at
  points ``1..s``; server ``k`` receives ``sk_y^(k)``.
* ShareDec:  server ``k`` of an enrolled set ``E`` publishes
  ``w_k = lambda_k <sk_y^(k), H(l)> + e_k mod q`` with ``lambda_k``
  its Lagrange coefficient for ``E`` and flooding noise
  ``e_k <- D_{Z, sigma_f}`` (truncated at 12 sigma_f), plus
  ``sum y_{i,j} c_{i,j} mod q0`` for a consistency check. ``e_k`` is
  derived from a per-server secret key, ``l`` and ``lambda_k``, so
  requests for the same label and coefficient get the same answer.
* Combine:   ``v = sum_{k in E} w_k = <sk_y, H(l)> + sum e_k mod q``;
  decode ``mu' = sum y c - round_q0(v)`` as in the MCFE.

Why the noise: without ``e_k`` a partial decryption is an exact linear
function of the server's share, so ``n0 + m0`` partial decryptions of one
server reveal its share and, with ``t`` shares, ``sk_y``; even the combined
values ``<sk_y, H(l)>`` over ``n0 + m0`` labels reveal ``sk_y``. With the noise,
a server's answers are LWE samples ``(H(l), <lambda_k sk_y^(k), H(l)> + e_k)``.
If fewer than ``t - 1`` shares are corrupted the share is uniform; once
``t - 1`` are, an honest answer is equivalent to an LWE sample in ``sk_y``,
whose last ``m0`` coordinates are small (Gaussian of width ``sigma_t``) rather
than uniform. The combined values are LWE samples in ``sk_y`` as well. This is
an LWE-based (computational) argument for dimension ``n0 + m0``, modulus
``q`` and width ``sigma_f``; it is our own and not covered by the cited
papers, which is why the scheme's status is *experimental*.

Noise is derandomised because fresh noise per request would let a combiner
average many answers to the same ``(l, E)``. Answers for different enrolled
sets containing server ``k`` carry independent noise; since the Lagrange
coefficients are small rationals, ``r`` such sets allow averaging that
reduces the noise by at most ``sqrt(r)``.

Sharing (config entry ``sharing``; default ``"replicated"`` for ``t < s`` and
``"shamir"`` for ``t = s``): the argument above takes one answer carrying
``<sk_y, H(l)>``, with multiplier 1, per label and enrolled set. With
``"shamir"`` and ``t < s`` (versions <= 0.3.0) that is not what a coalition
``C`` of ``t - 1`` servers sees: an honest answer is computable from ``C``'s
shares up to ``M_k <sk_y, H(l)> + e_k``,
``M_k = lambda_k^E prod_{j in C} (k - j) / (0 - j)``, and several honest
answers of one enrolled set carry multipliers other than 0 and 1 (rationals
``r_k / d_k``, or integers such as ``-3`` and ``4``). Then
``d_k (w_k - known) - r_k v``, with ``v`` the combined value, is the small
integer ``d_k e_k - r_k sum_E e_j``: the answers fix the noise terms up to
multiples of ``D M`` (``D`` the common denominator; ``sum_E e_j`` modulo
``D``), and ``v`` is an LWE sample whose error is ``D |M|_2`` times narrower
than the ``sigma_f`` of one answer of multiplier 1: ``sqrt(10)`` for
``s = 3, t = 2``, ``C = {2}``, ``E = {1, 3}`` (``M = (3/4, 1/4)``), up to 5
for ``s = 3, t = 2`` and 79 for ``s = 5, t = 3`` per enrolled set, and 9.1 and
869 over all enrolled sets of one label. This is not a break, but the argument
does not cover it. ``"replicated"`` (see
:func:`~pyfe4ai.schemes.threshold_utils.share_replicated`) makes every ``M_k``
0 or 1, with one honest answer of multiplier 1 per enrolled set; the other
answers are computable up to their own noise. ``sigma_f`` and ``q`` are
unchanged. With ``t = s`` both are as described. An explicit ``"shamir"`` with
``t < s`` is refused under the ``standard`` profile and draws an
``FESecurityWarning`` under ``toy`` (this scheme caches no parameters).
Servers refuse Shamir key shares with ``t < s`` (also those of versions <=
0.3.0) unless created with ``legacy_shamir = "warn"``, for a migration only
and never with public parameters that record the ``standard`` profile; public
parameters record ``threshold_construction`` and ``security_profile``, and
servers refuse constructions they do not know.

Statistical smudging in the usual sense (noise hiding a small ciphertext
error) does not apply, because these ciphertexts carry no LWE error, only
rounding. ``sigma_f`` is the largest width that keeps decryption correct:
``(q0 / q) * s * 12 sigma_f <= (n Y + 1) / 2`` uses the half of the error
budget that :func:`pyfe4ai.utils.lwe_utils.abmpsw20_lwe_parameters` reserves
for threshold decryption, which gives a noise-to-modulus ratio of roughly
``1 / (q0 s)``.
"""

from __future__ import annotations

import json
import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.mcfe.lwe import MCFELWE
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe import dot_mod
from pyfe4ai.schemes.threshold_utils import check_fusion_weight
from pyfe4ai.schemes.threshold_utils import check_served_sharing
from pyfe4ai.schemes.threshold_utils import legacy_shamir_policy
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.schemes.threshold_utils import pieces_held
from pyfe4ai.schemes.threshold_utils import share_cache
from pyfe4ai.schemes.threshold_utils import share_for_enrolled
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_replicated
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import combine_consistently
from pyfe4ai.schemes.threshold_utils import combine_elementwise
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import resolve_sharing
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint

logger = logging.getLogger(__name__)

# recorded in the public parameters (those of versions <= 0.3.0 have none)
_CONSTRUCTION = "abmpsw20-shamir-flooding/v1"
_CONSTRUCTION_REPLICATED = "abmpsw20-replicated-flooding/v1"
_CONSTRUCTIONS = {_CONSTRUCTION: "shamir", _CONSTRUCTION_REPLICATED: "replicated"}


def flooding_sigma(params: dict, n_slots: int, bound_y: int, s: int) -> int:
    """Largest ``sigma_f`` with ``(q0 / q) * s * c * sigma_f <= (n Y + 1) / 2``.

        Args:
            params: Output of :func:`~pyfe4ai.utils.lwe_utils.abmpsw20_lwe_parameters`.
            n_slots: Total number of slots ``n``.
            bound_y: Weight bound ``Y``.
            s: Number of decryption servers (upper bound on ``|E|``).
    """
    budget = (n_slots * bound_y + 1) * params["q"] // (2 * params["q0"])
    return int(budget // (s * GAUSSIAN_TAIL_CUT))


def flooding_noise(sigma: int, rng=None) -> int:
    """Sample ``e <- D_{Z, sigma}`` truncated to ``|e| <= 12 sigma``.

    Uses the arbitrary-precision sampler: ``sigma`` is far wider than a
    double's mantissa, and scaling a 53-bit normal by it would leave the low
    bits of ``e`` deterministic.

    Args:
        sigma: Width.
        rng: Bit source (default: OS CSPRNG). An
            :class:`~pyfe4ai.utils.sampling_utils.XofRandom` makes the
            sample a function of its seed.
    """
    bound = GAUSSIAN_TAIL_CUT * sigma
    while True:
        e = discrete_gaussian_bigint(sigma, 1, rng=rng)[0]
        if abs(e) <= bound:
            return int(e)


class ThresholdMCFELWEKeyGenerator(MCFELWEKeyGenerator):
    """Key generator for threshold ABMPSW20 LWE MCFE."""

    _scheme_type = CryptoCONST.TYPE_TMCFE_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`~pyfe4ai.schemes.mcfe.lwe.MCFELWEKeyGenerator`,
                plus ``s`` (number of decryption servers), ``t`` (threshold),
                ``lst_sid`` and ``sharing`` (``"replicated"``, the default
                for ``t < s``, or ``"shamir"``, the default for ``t = s``; see
                :func:`~pyfe4ai.schemes.threshold_utils.resolve_sharing`).
        """
        t = config.get("t", CryptoCONST.tMCFE_T)
        s = config.get("s") or len(config.get("lst_sid", [])) or CryptoCONST.tMCFE_S
        # before the parameters are derived: a refused config costs nothing
        self.sharing = resolve_sharing(config, s, t, "mcfe/lwe_threshold")
        super().__init__(config, **kwargs)
        self.t = t
        if self.s is None:
            self.s = len(config.get("lst_sid", [])) or CryptoCONST.tMCFE_S
        self.lst_sid = config.get("lst_sid", ["sid_{}".format(i) for i in range(self.s)])
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)
        self.sigma_f = flooding_sigma(self.params, self.n_slots, self.bound_y, self.s)
        if self.sigma_f < 1:
            raise FEValidationError("parameters leave no room for flooding noise")
        self.dict_dk = {}

    def setup(self) -> None:
        """Sample the client keys and forget any previously shared keys."""
        super().setup()
        self.dict_dk = {}

    def get_public_parameters(self) -> dict:
        """Return the MCFE public parameters plus ``s``, ``t``, ``lst_sid``,
        ``sigma_f``, ``sharing``, ``threshold_construction`` and ``security_profile``."""
        pp = super().get_public_parameters()
        pp.update(
            {
                "s": self.s,
                "t": self.t,
                "lst_sid": list(self.lst_sid),
                "sigma_f": gp.digits(gp.mpz(self.sigma_f)),
                "sharing": self.sharing,
                "threshold_construction": _CONSTRUCTION if self.sharing == "shamir" else _CONSTRUCTION_REPLICATED,
                "security_profile": self.security_profile,
            }
        )
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of ``sk_y`` for the given weights.

        All servers asking for the same fusion weights receive shares of one
        and the same sharing; the key is label-independent (a ``label`` in
        the credentials is ignored).

        Args:
            sid: Decryption-server identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``.

        Returns:
            Dict with ``sk_y`` (this server's share), ``sid``, ``lst_sid``,
            ``weights_id`` (digest of the weights) and ``noise_key`` (this
            server's secret key for deriving flooding noise), or ``None`` if
            *sid* is unknown. With replicated sharing ``sk_y`` is replaced
            by ``sharing`` and ``pieces`` (``[{"set": T, "share": {"sk_y": ...}}]``).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for Threshold MCFE LWE")
        fusion_weight = credentials.get("fusion_weight")
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        identifier = fusion_weight_id(fusion_weight)
        if identifier not in self.dict_dk:
            self.dict_dk[identifier] = self._share_key(self._aggregate_key(fusion_weight))
        share, noise_key = self.dict_dk[identifier][sid]
        held = {"sk_y": [gp.digits(v) for v in share]} if self.sharing == "shamir" else {
            "sharing": "replicated", "pieces": share,
        }
        return {
            **held,
            "sid": sid,
            "lst_sid": list(self.lst_sid),
            "weights_id": identifier,
            "noise_key": noise_key,
        }

    def _share_key(self, sk_y: list) -> dict:
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())
        if self.sharing == "replicated":
            pieces = share_replicated({"sk_y": list(sk_y)}, self.t, points, self.q)
            return {sid: (pieces_held(pieces, sid_v[sid]), secrets.token_hex(32)) for sid in self.lst_sid}
        shares = [share_secret(v, self.t, points, self.q) for v in sk_y]
        return {
            sid: ([sh[sid_v[sid]] for sh in shares], secrets.token_hex(32))
            for sid in self.lst_sid
        }


class ThresholdMCFELWE(MCFELWE):
    """Crypto operations for threshold ABMPSW20 LWE MCFE.

    A server refuses Shamir key shares with ``t < s`` (versions <= 0.3.0)
    unless its config has ``legacy_shamir = "warn"`` (see
    :data:`~pyfe4ai.schemes.threshold_utils.LEGACY_SHAMIR_POLICIES`).
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.legacy_shamir = legacy_shamir_policy(config)

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list, _share=None
    ) -> dict:
        """Compute this server's noise-flooded partial decryption.

        Args:
            dict_ct: Mapping of client IDs to ciphertexts under one label.
            credentials: Dict with ``fusion_weight`` (the weights the key was
                derived for) and ``label``.
            dk: This server's key share.
            lst_sid_enrolled: The servers whose partial decryptions will be
                combined (at least ``t``, including this one).

        Returns:
            Dict with ``w``, ``c_y``, ``label``, ``sid`` and ``lst_sid_enrolled``.

        Raises:
            FESchemeError: If the enrolled set is invalid or the weights are
                not those the key share was derived for.
        """
        if not dk or not ("sk_y" in dk or "pieces" in dk):
            raise FEKeyError("no decryption keys provided")
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        check_served_sharing(self.pp, dk, _CONSTRUCTIONS)
        fusion_weight = credentials.get("fusion_weight")
        label = credentials.get("label")
        if fusion_weight is None or label is None:
            raise FEKeyError("need fusion_weight and label for threshold share decryption")
        encoded = encode_label(label)
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, fusion_weight)

        # Shamir: (dk, lambda_k); replicated: the sum of the assigned pieces, 1
        share, multiplier, tag = _share or share_for_enrolled(
            dk, self.id, lst_sid_enrolled, self.pp["t"], self.q, cache=share_cache(self),
            legacy_shamir=self.legacy_shamir, profile=self.pp.get("security_profile"),
        )
        share_term = dot_mod([gp.mpz(v) for v in share["sk_y"]], self._hash_label(label), self.q)
        # the partial depends on the enrolled set only through the tag (lambda,
        # or the assigned pieces)
        context = json.dumps([encoded, tag]).encode("utf-8")
        e_k = flooding_noise(
            int(self.pp["sigma_f"]),
            rng=XofRandom(bytes.fromhex(dk["noise_key"]) + context),
        )
        w = (multiplier * share_term + e_k) % self.q
        return {
            "w": gp.digits(w),
            "c_y": int(self._weighted_sum(dict_ct, fusion_weight)),
            "label": encoded,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

    def combine_decrypt(self, partials) -> float | None:
        """Combine partial decryptions from at least ``t`` servers.

        Args:
            partials: ``{sid: share_decrypt(...)}`` for one enrolled set, or a
                list of such mappings for several enrolled sets, whose results
                must then agree (see
                :func:`~pyfe4ai.schemes.threshold_utils.combine_consistently`:
                servers are assumed honest-but-curious, and only this
                redundancy detects one that returns a wrong partial).

        Raises:
            FESchemeError: If a group is incomplete or invalid, or the groups
                disagree.
        """
        return combine_consistently(partials, self._combine_group)

    def _combine_group(self, dict_ct_prime: dict) -> float | None:
        """Combine at least ``t`` partial decryptions into the inner product.

        Returns:
            ``<x, y>`` divided by ``10^precision``, or ``None`` if the servers
            disagree on the ciphertexts or the value is out of range.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given or the shares
                do not match the enrolled set they were computed for.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = list(dict_ct_prime.values())
        c_y, label = partials[0]["c_y"], partials[0]["label"]
        if any(part["c_y"] != c_y or part["label"] != label for part in partials):
            logger.warning("partial decryptions disagree on the ciphertexts they decrypted")
            return None
        v = sum(gp.mpz(part["w"]) for part in partials) % self.q
        mu = self._decode(gp.mpz(c_y), v)
        if mu is None:
            return None
        return float(mu / pow(10, self.precision))

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Partially decrypt every element of clients' ndarray ciphertexts.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists
                produced by
                :meth:`~pyfe4ai.schemes.mcfe.lwe.MCFELWE.encrypt_lst_ndarray`
                under ``credentials["label"]``.
            credentials: Dict with ``fusion_weight`` and ``label``.
            dk: This server's key share.
            lst_sid_enrolled: The servers whose shares will be combined.
        """
        label = credentials.get("label")
        if not isinstance(label, str):
            raise FEValidationError("need a string label")
        sample = next(iter(dict_ndarray_ct.values()))
        share = None
        if dk and "pieces" in dk and self.id in dk.get("lst_sid", []):
            # the assigned pieces, added up once for all elements
            validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
            share = share_for_enrolled(dk, self.id, lst_sid_enrolled, self.pp["t"], self.q, cache=share_cache(self))
        lst_partial = []
        for l in range(len(sample)):
            ary_partial = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                elem_credentials = dict(credentials, label=_element_label(label, l, i))
                ary_partial[i] = self.share_decrypt(
                    dct_ct, elem_credentials, dk, lst_sid_enrolled, _share=share
                )
            lst_partial.append(ary_partial)
        return lst_partial

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Combine element-wise partial decryptions from at least ``t`` servers.

        Args:
            dict_ndarray_ct: Mapping of server ids to the outputs of
                :meth:`compute_lst_ndarray_ct`.

        Raises:
            FESchemeError: If the servers' outputs differ in structure or an
                element does not decrypt (servers disagree, or out of range).
        """
        out = combine_elementwise(dict_ndarray_ct, lambda groups, l, i: self.combine_decrypt(groups))
        return [ary.astype(float) for ary in out]
