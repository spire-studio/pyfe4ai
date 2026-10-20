"""
Threshold LWE multi-input inner-product FE: ACFGU18 MIFE-LWE with a
Shamir-shared functional key and statistically flooded partial decryptions.

| MIFE layer: :mod:`pyfe4ai.schemes.mife.lwe` (ACFGU18 Fig. 4 over the
| ABDP15 LWE IPFE).
| Threshold layer in the setting of R. Xu et al., "TAPFed: Threshold Secure
| Aggregation for Privacy-Preserving Federated Learning", IEEE TDSC 2024:
| the functional key is Shamir-shared among ``s`` servers, any ``t`` decrypt.

* type:     private-key multi-input encryption, threshold functional decryption
* setting:  Integer based (Z_q, q prime, Delta = floor(q / p))

Construction (notation of :mod:`pyfe4ai.schemes.mife.lwe`):

* KeyGen:    the MIFE key ``({sk_i = S_i y_i}, z)``; every coordinate of every
  ``sk_i`` and ``z`` is Shamir-shared with threshold ``t`` over
  ``Z_q`` at points ``1..s``.
* ShareDec:  server ``k`` of an enrolled set ``E`` publishes
  ``w_k = lambda_k (sum_i <ct0_i, sk_i^(k)> + Delta z^(k)) + e_k mod q``
  and ``c_y = sum_i <y_i, ct1_i> mod q`` for a consistency check.
* Combine:   ``c_y - sum_{k in E} w_k = Delta <x, y> + noise - sum e_k``; decode.

Flooding: ``c_y - (sum_i <ct0_i, sk_i> + Delta z) = Delta <x, y> + nu`` where
the ciphertext noise ``nu = sum_i y_i^T E_i^T r_i`` is bounded by
``B = n eta Y m 12 sigma``. From the result and the ciphertexts one can
compute the exact key part only up to ``nu``; the flooding noise
``e_k <- D_{Z, 2^kappa B}`` makes the partial decryptions statistically
indistinguishable from ones simulated from the result (the standard
smudging argument for threshold LWE decryption). ``q`` is enlarged so that
``s`` flooding terms still decode: ``Delta / 2 > B + s * 12 * 2^kappa B``; at
a fixed ``lwe_n`` this lowers the noise-to-modulus ratio of the underlying
LWE instance, and ``m`` grows with ``log2 q`` so that ``A^T r`` stays close to
uniform (versions <= 0.2.0 kept the ``m`` of the smaller base modulus: about
half the length the leftover hash lemma needs). ``e_k`` is derived from a
per-server secret key and the request (see below), so repeating a request
cannot be used to average the noise away. The threshold layer is our own
design.

Ciphertext authentication: ``w_k`` is linear in ``ct0``, so whoever chooses the
ciphertexts a server decrypts could submit ``ct0 = c e_j`` for chosen ``c`` and
read the functional key off the partials coordinate by coordinate (the
flooding hides a fixed noise term, not a chosen multiple of the key). Client
``i`` and server ``k`` therefore share a MAC key; every ciphertext carries one
HMAC-SHA256 tag per server over its serialised ``(ct0, ct1)``, and a server
only decrypts ciphertexts whose tags for it verify. A corrupted client can
still put any ciphertext in its own slot, which reveals that slot's key part
``S_i y_i`` and nothing about the other slots; mixing authentic ciphertexts of
different rounds is the usual multi-input functionality. The noise is seeded
with the server's key, ``lambda_k``, the ``ct0`` values and the key share's
``pp_binding`` (the public parameters it is bound to), so a request for
the same key part draws the same noise; a corrupted client varying its own
``ct0`` changes the key part and so draws fresh noise each time. The flooding
argument is statistical per answer (a statistical distance of about
``2^-stat_param`` per coefficient, adding up over the answers); it does not
bound what many answers to varied ``ct0`` reveal together about the other
slots' part: averaging them would take about ``2^(2 stat_param)`` partial
decryptions, but other ways of combining these noisy linear equations are
not ruled out.

Sharing (config entry ``sharing``; default ``"replicated"`` for ``t < s`` and
``"shamir"`` for ``t = s``): with ``"shamir"`` and ``t < s`` -- the sharing of
versions <= 0.3.0 -- the argument above does not hold. With ``t - 1`` servers
``C`` corrupted, an honest partial is computable from ``C``'s shares and the
result up to ``e_k - M nu``,
``M = lambda_k^E prod_{j in C} (k - j) / (0 - j)``, and ``M`` is a non-integer
rational for some ``(C, E, k)`` (6 of 18 for ``s = 3, t = 2``): for
``M = r / d``, ``d`` times the difference is the small integer
``d e_k - r nu``, which gives ``nu mod d`` whatever ``sigma_f``. Fewer
corrupted servers, or none, can suffice (the aggregator alone reads ``nu``
modulo ``(s - 1)!`` for ``t = 1``), and combining the partials of several
enrolled sets raises ``d`` (144 for ``s = 5, t = 3``); where ``M`` is an
integer it reaches 4 (45 for ``s = 5, t = 3``), and the statistical distance
of the flooding grows ``|M|``-fold. ``"replicated"`` shares the key additively
over the ``(t - 1)``-subsets of the servers (see
:func:`~pyfe4ai.schemes.threshold.sharing.share_replicated`): each server of
``E`` adds up the pieces assigned to it, ``M`` is 0 or 1 and the smudging
argument holds as stated, with ``q`` unchanged; a server holds
``C(s - 1, t - 1)`` pieces instead of one share. With ``t = s`` both are sound
(``M = 1``). An explicit ``"shamir"`` with ``t < s`` draws an
``FESecurityWarning`` under ``toy`` (the ``standard`` profile refuses the
scheme altogether, see below); the two
sharings have their own parameter records (construction
``acfgu18-abdp15-replicated-smudging-seeded-a/v1`` for ``"replicated"``), so a
record cached by an earlier version for ``t < s`` is not reused by the
default. Servers refuse Shamir key shares with ``t < s`` unless created with
``legacy_shamir = "warn"`` (never with public parameters that record the
``standard`` profile); key shares of versions <= 0.3.0 are refused in any
case (another flooding derivation, see
:func:`~pyfe4ai.primitives.sampling.check_flooding_version`);
public parameters record ``threshold_construction`` and
``security_profile``, and servers refuse constructions they do not know.
The opt-in ``"replicated-seeded"`` (see
:func:`~pyfe4ai.schemes.threshold.sharing.share_replicated_seeded`) deals the
same kind of pieces, all but one expanded from 32-byte seeds: a server stores
seeds and at most one full piece, and its partial decryptions are those of
the replicated key share of the expanded pieces; the key shares hide the key
computationally (SHAKE-256 keyed by the seeds as a PRF) rather than
perfectly. Its record is ``acfgu18-abdp15-replicated-seeded-smudging-seeded-a/v1``.

The ``standard`` security profile refuses this scheme. ``lwe_n = "auto"``
reads :data:`pyfe4ai.primitives.lattice.lwe_utils.WIDE_LWE_128_LOG_RATIO` at the width
``sigma / (1 + 4 X sqrt(eta))`` of the MIFE layer's security reduction (see
:mod:`pyfe4ai.schemes.common.abdp15_lwe`). At the flooding-enlarged ``q`` of
``stat_param = 128`` (two clients) that takes the very end of the table
(5888 to 6144, ``A`` of 6.5 to 7.8 * 10^9 entries) for a few small bounds, e.g.
``X = Y`` in {1, 2, 3, 8, 10} at ``eta = 1``, and more than the table's 6144
for most others (``X = Y`` = 4 to 6, ``X = Y >= 12``, most ``eta >= 2``);
``X = Y = 50`` would need about 6500. Read at ``sigma`` itself, the
earlier rule took 5952 for two clients (``q`` of 217 bits, ``m`` of
1.3 * 10^6, ``A`` of 7.8 * 10^9 entries, so ``A`` had to be streamed):
key generation took 38 minutes, one encryption 572 s, a batch of 64
encryptions 679 s and a partial decryption 13 ms.

``flooding_accounting = "kl"`` with ``max_partial_decryptions = Q``
(opt-in, ``t = s`` with the Shamir sharing only, see
:mod:`pyfe4ai.schemes.threshold.kl`) sizes the noise for a run in
which every server produces at most ``Q`` partial decryptions:
``KL <= 2^-(stat_param + 1)`` for the ``N = s Q`` flooded values, i.e.
``sigma_f = B sqrt(N 2^stat_param)``, with a tail cut derived from ``N`` (16
``sigma_f`` at ``N = 2^40``), instead of a statistical distance
``2^-stat_param`` per value. It is a different, weaker guarantee
(``stat_param - 6.85`` bits in the bit-security sense of Micciancio and
Walter, about ``2^-65`` of total statistical distance at ``stat_param =
128``), valid only for the first ``Q`` partial decryptions of every server,
which each server counts in the ``partial_budget`` its deployment gives it.
At ``stat_param = 128`` with ``X = Y = 50``, two clients, ``s = t = 3`` and
``N = 2^40`` it would take ``lwe_n`` about 4864 at ``sigma'`` (``q`` of 173
bits, ``A`` of 4.1 * 10^9 entries, so ``"stream"`` mode), but the
``standard`` profile refuses this scheme with either accounting (a
maintainer decision of October 2026); ``"kl"`` is available under ``toy``.
These parameters have their own record (construction
``acfgu18-abdp15-shamir-smudging-kl/v2``).
"""

from __future__ import annotations

import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.primitives.lattice.lwe_utils import abdp15_parameters
from pyfe4ai.primitives.lattice.lwe_utils import decode_scaled
from pyfe4ai.primitives.lattice.lwe_utils import wide_lwe_n
from pyfe4ai.primitives.lattice.matrix_utils import digits_to_vector
from pyfe4ai.primitives.lattice.matrix_utils import vecdot_mod
from pyfe4ai.primitives.sampling import XofRandom
from pyfe4ai.primitives.sampling import (
    FLOODING_NOISE_VERSION,
    FLOODING_SAMPLER,
    check_flooding_version,
)
from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_BINDING_DEFAULTS
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_BINDING_FIELDS
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_CONSTRUCTION
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_CONSTRUCTION_KL
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_CONSTRUCTION_REPLICATED
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_CONSTRUCTION_REPLICATED_SEEDED
from pyfe4ai.schemes.threshold.binding import TMIFE_LWE_CONSTRUCTIONS
from pyfe4ai.schemes.threshold.binding import check_ciphertext_tags
from pyfe4ai.schemes.threshold.binding import check_pp_binding
from pyfe4ai.schemes.threshold.binding import client_server_mac_keys
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.schemes.threshold.binding import pp_binding
from pyfe4ai.schemes.threshold.binding import tag_ciphertext
from pyfe4ai.schemes.threshold.codec import ciphertexts_digest
from pyfe4ai.schemes.threshold.kl import KL
from pyfe4ai.schemes.threshold.kl import PER_SAMPLE
from pyfe4ai.schemes.threshold.kl import check_kl_threshold
from pyfe4ai.schemes.threshold.kl import flooding_config
from pyfe4ai.schemes.threshold.kl import kl_flooded_samples
from pyfe4ai.schemes.threshold.kl import kl_flooding
from pyfe4ai.schemes.threshold.kl import kl_flooding_width
from pyfe4ai.schemes.threshold.kl import kl_tail_cut
from pyfe4ai.schemes.threshold.kl import server_budget
from pyfe4ai.schemes.threshold.roles import check_key_share_server
from pyfe4ai.schemes.threshold.roles import check_request
from pyfe4ai.schemes.threshold.roles import combine_consistently
from pyfe4ai.schemes.threshold.roles import combine_elementwise
from pyfe4ai.schemes.threshold.roles import enrolled_share_options
from pyfe4ai.schemes.threshold.roles import flooding_noise
from pyfe4ai.schemes.threshold.roles import legacy_shamir_policy
from pyfe4ai.schemes.threshold.roles import noise_seed
from pyfe4ai.schemes.threshold.roles import validate_enrolled
from pyfe4ai.schemes.threshold.roles import validate_partial_decryptions
from pyfe4ai.schemes.threshold.sharing import deal_replicated
from pyfe4ai.schemes.threshold.sharing import resolve_sharing
from pyfe4ai.schemes.threshold.sharing import share_for_enrolled
from pyfe4ai.schemes.threshold.sharing import share_points
from pyfe4ai.schemes.threshold.sharing import share_secret
from pyfe4ai.schemes.threshold.sharing import validate_threshold

logger = logging.getLogger(__name__)

_DEFAULT_STAT_PARAM = 64
_TAG_DOMAIN = "pyfe4ai/tmife-lwe/ciphertext-tag/v1"


def _tag_payload(nid, ct: dict) -> list:
    """What a tMIFE-LWE ciphertext tag covers: ``ct0`` and ``ct1`` as serialised."""
    return [ct.get("ct0"), ct.get("ct1")]


class ThresholdMIFELWEKeyGenerator(MIFELWEKeyGenerator):
    """Key generator for threshold MIFE-LWE (ACFGU18 + Shamir + smudging)."""

    _scheme_type = CryptoCONST.TYPE_TMIFE_LWE
    _construction = TMIFE_LWE_CONSTRUCTION

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`~pyfe4ai.schemes.mife.lwe.MIFELWEKeyGenerator`,
                plus ``s`` (number of servers), ``t`` (threshold), ``lst_sid``,
                ``stat_param`` (statistical parameter ``kappa`` of the
                flooding), ``sharing`` (``"replicated"``, the default for
                ``t < s``, ``"shamir"``, the default for ``t = s``, or the
                opt-in ``"replicated-seeded"``; see
                :func:`~pyfe4ai.schemes.threshold.sharing.resolve_sharing`)
                and, opt-in, ``flooding_accounting = "kl"`` with
                ``max_partial_decryptions`` (``t = s``, Shamir sharing; see
                the module docstring).
        """
        self.t = config.get("t", CryptoCONST.tMIFE_T)
        s = config.get("s") or len(config.get("lst_sid", [])) or 1
        self.sharing = resolve_sharing(config, s, self.t, "mife/lwe_threshold")
        self._construction = {
            "shamir": TMIFE_LWE_CONSTRUCTION, "replicated": TMIFE_LWE_CONSTRUCTION_REPLICATED,
            "replicated-seeded": TMIFE_LWE_CONSTRUCTION_REPLICATED_SEEDED,
        }[self.sharing]
        self.lst_sid = config.get("lst_sid", ["sid_{}".format(i) for i in range(s)])
        self.stat_param = int(config.get("stat_param", _DEFAULT_STAT_PARAM))
        self.flooding_accounting, self.max_partial_decryptions = flooding_config(config)
        if self.flooding_accounting == KL:
            check_kl_threshold(self.t, s)
            if self.sharing != "shamir":
                raise FEValidationError(
                    "flooding_accounting='kl' accounts for the flooding of Shamir key shares (t = s), "
                    "not sharing={!r}".format(self.sharing)
                )
            self._construction = TMIFE_LWE_CONSTRUCTION_KL
        self.dict_dk = {}
        super().__init__(dict(config, s=s), **kwargs)
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)

    def _record_fields(self) -> dict:
        fields = {**super()._record_fields(), "s": self.s, "stat_param": self.stat_param}
        if self.flooding_accounting == KL:
            fields.update(flooding_accounting=KL, max_partial_decryptions=self.max_partial_decryptions)
        return fields

    def _derive(self) -> dict:
        if self.lwe_n_config == "auto":
            return wide_lwe_n(self._threshold_parameters, self._table_sigma)
        return self._threshold_parameters(int(self.lwe_n_config))

    def _threshold_parameters(self, lwe_n: int) -> dict:
        """ABDP15 parameters at *lwe_n* with ``q`` enlarged for ``s`` flooding
        terms and ``m`` for the leftover hash lemma at that ``q``."""
        params = abdp15_parameters(
            self.bound_x, self.bound_y, self.eta, lwe_n, extra_dimension=self.n
        )
        m = params["m"]
        while True:
            # ciphertext noise of a full decryption: |sum_i y_i^T E_i^T r_i|
            noise_bound = (
                self.n * self.eta * int(self.bound_y) * m
                * int(GAUSSIAN_TAIL_CUT * params["sigma_q"] + 1)
            )
            if self.flooding_accounting == KL:
                samples = kl_flooded_samples(self.s, self.max_partial_decryptions, 1)
                sigma_f = kl_flooding_width(noise_bound, samples, self.stat_param)
                budget = noise_bound + self.s * kl_tail_cut(samples, self.stat_param, sigma_f) * sigma_f
            else:
                sigma_f = noise_bound << self.stat_param
                budget = noise_bound + self.s * GAUSSIAN_TAIL_CUT * sigma_f
            # Delta = floor(q / p) must exceed 2 * budget (plus a margin for rounding)
            q = gp.next_prime(max(params["q"], params["p"] * (2 * budget + 4)))
            # A^T r must stay close to uniform for the enlarged q (leftover hash
            # lemma, as in derive_lwe_parameters); a longer r adds noise, so iterate
            m_need = (lwe_n + self.eta + 1) * int(q).bit_length() + 2 * lwe_n + 1
            if m_need <= m:
                break
            m = m_need
        params.update({"q": q, "m": m, "sigma_f": sigma_f})
        return params

    def _apply_parameters(self, param: dict) -> None:
        super()._apply_parameters(param)
        self.sigma_f = int(param["sigma_f"])

    def setup(self) -> None:
        """Sample the MIFE keys and the client-server MAC keys; forget any
        previously shared keys."""
        super().setup()
        self.dict_dk = {}
        self.mac_keys = client_server_mac_keys(self.lst_nid, self.lst_sid)
        self._binding = pp_binding(self.get_public_parameters(), TMIFE_LWE_BINDING_FIELDS, TMIFE_LWE_BINDING_DEFAULTS)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s MIFE encryption key plus ``mac_keys``
        (``{sid: hex}``, one key per server for the ciphertext tags)."""
        sk = super().get_private_keys(nid, **kwargs)
        if sk is not None:
            sk["mac_keys"] = dict(self.mac_keys[nid])
        return sk

    def get_public_parameters(self) -> dict:
        """Return the MIFE public parameters plus ``s``, ``t``, ``lst_sid``,
        ``sigma_f``, ``sharing``, ``threshold_construction``, ``security_profile``
        and, under ``"kl"``, ``flooding_accounting``, ``max_partial_decryptions``,
        ``stat_param`` and ``flood_tail_cut``."""
        pp = super().get_public_parameters()
        pp.update(
            {"s": self.s, "t": self.t, "lst_sid": list(self.lst_sid), "sigma_f": str(self.sigma_f),
             "sharing": self.sharing, "threshold_construction": self._construction,
             "security_profile": self.security_profile}
        )
        if self.flooding_accounting == KL:
            pp.update({
                "flooding_accounting": KL,
                "max_partial_decryptions": self.max_partial_decryptions,
                "stat_param": self.stat_param,
                "flood_tail_cut": kl_tail_cut(
                    kl_flooded_samples(self.s, self.max_partial_decryptions, 1), self.stat_param, self.sigma_f
                ),
            })
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of the MIFE functional key.

        Args:
            sid: Decryption-server identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``.

        Returns:
            Dict with ``sk_y`` (per-slot share vectors), ``z`` (share of the
            pad term), ``sid``, ``lst_sid``, ``weights_id``, ``noise_key`` and
            ``mac_keys`` (``{nid: hex}``, to verify the clients' ciphertext
            tags), or ``None`` if *sid* is unknown. With replicated sharing
            ``sk_y`` and ``z`` are replaced by ``sharing`` and ``pieces``
            (``[{"set": T, "share": {"sk_y": ..., "z": ...}}]``); with
            ``"replicated-seeded"`` ``pieces`` holds the correction piece (or
            nothing), and ``seeds`` and ``layout`` give the others.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for Threshold MIFE LWE")
        fusion_weight = credentials.get("fusion_weight")
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        identifier = fusion_weight_id(fusion_weight)
        if identifier not in self.dict_dk:
            key = super().get_decryption_keys(sid, credentials={"fusion_weight": fusion_weight})
            self.dict_dk[identifier] = self._share_key(key)
        dk = dict(self.dict_dk[identifier][sid])
        dk.update({
            "noise_version": FLOODING_NOISE_VERSION,
            "noise_sampler": FLOODING_SAMPLER,
            "sid": sid,
            "lst_sid": list(self.lst_sid),
            "weights_id": identifier,
            "mac_keys": {nid: self.mac_keys[nid][sid] for nid in self.lst_nid},
            "pp_binding": self._binding,
        })
        return dk

    def _share_key(self, key: dict) -> dict:
        if self.sharing in ("replicated", "replicated-seeded"):
            material = {"sk_y": {nid: list(vec) for nid, vec in key["sk_y"].items()}, "z": key["z"]}
            held = deal_replicated(material, self.sharing, self.t, self.lst_sid, self.q)
            return {
                sid: {"sharing": self.sharing, **held[sid], "noise_key": secrets.token_hex(32)}
                for sid in self.lst_sid
            }
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())
        per_sid = {
            sid: {"sk_y": {}, "noise_key": secrets.token_hex(32)} for sid in self.lst_sid
        }
        for nid, vec in key["sk_y"].items():
            shares = [share_secret(gp.mpz(v), self.t, points, self.q) for v in vec]
            for sid in self.lst_sid:
                per_sid[sid]["sk_y"][nid] = [gp.digits(sh[sid_v[sid]]) for sh in shares]
        z_shares = share_secret(gp.mpz(key["z"]), self.t, points, self.q)
        for sid in self.lst_sid:
            per_sid[sid]["z"] = gp.digits(z_shares[sid_v[sid]])
        return per_sid


class ThresholdMIFELWE(MIFELWE):
    """Crypto operations for threshold MIFE-LWE (ACFGU18 + Shamir + smudging).

    A server refuses Shamir key shares with ``t < s`` (versions <= 0.3.0)
    unless its config has ``legacy_shamir = "warn"`` (see
    :data:`~pyfe4ai.schemes.threshold.roles.LEGACY_SHAMIR_POLICIES`).
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """As :class:`~pyfe4ai.schemes.mife.lwe.MIFELWE`; under
        ``flooding_accounting = "kl"`` a server needs ``partial_budget``, the
        :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget`
        that its deployment keeps for it; ``legacy_shamir`` as in the class
        docstring."""
        super().__init__(config, **kwargs)
        self.legacy_shamir = legacy_shamir_policy(config)
        self._partial_budget = config.get("partial_budget")
        flood = kl_flooding(self.pp, 1) if getattr(self, "pp", None) else None
        if flood is not None and self._partial_budget is not None:
            server_budget(self.pp, self._partial_budget, flood)

    def _kl_flooding(self):
        """KL flooding parameters and the server's budget, or ``(None, None)``.

        Raises:
            FESchemeError: Under ``"kl"`` without a ``partial_budget``.
        """
        flood = kl_flooding(self.pp, 1)
        if flood is None:
            return None, None
        return flood, server_budget(self.pp, self._partial_budget, flood)

    def _encrypt_batch(self, plaintexts: list) -> list:
        """MIFE ciphertexts with ``tags``: one HMAC tag per server."""
        mac_keys = self.sk.get("mac_keys")
        if not mac_keys:
            raise FEKeyError("encryption key lacks the ciphertext MAC keys")
        out = super()._encrypt_batch(plaintexts)
        for ct in out:
            tag_ciphertext(ct, mac_keys, _TAG_DOMAIN, _tag_payload(None, ct))
        return out

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list, _share=None
    ) -> dict:
        """Compute this server's flooded partial decryption.

        Args:
            dict_ct: One ciphertext per slot of the key.
            credentials: Mapping of client IDs to the fusion weights the key
                share was derived for.
            dk: This server's key share.
            lst_sid_enrolled: The servers whose partial decryptions will be
                combined (at least ``t``, including this one).

        Returns:
            Dict with ``w``, ``c_y``, ``ct_digest``, ``sid`` and ``lst_sid_enrolled``.

        Raises:
            FESchemeError: If the enrolled set is invalid, the weights are not
                those of the key share, a ciphertext's tag for this server
                does not verify (it was not produced by that slot's client),
                or, under ``flooding_accounting = "kl"``, the server has no
                ``partial_budget`` or its budget is spent (refused requests
                consume nothing).
        """
        if not dk or not ("pieces" in dk or ("sk_y" in dk and "z" in dk)):
            raise FEKeyError("no decryption keys provided")
        check_flooding_version(dk)
        check_key_share_server(self.id, dk, self.pp, TMIFE_LWE_CONSTRUCTIONS)
        construction = self.pp.get("threshold_construction")
        kl = self.pp.get("flooding_accounting", PER_SAMPLE) == KL
        if construction is not None and (construction == TMIFE_LWE_CONSTRUCTION_KL) != kl:
            raise FESchemeError("the public parameters' threshold_construction is not that of their accounting")
        binding = check_pp_binding(self.pp, dk, TMIFE_LWE_BINDING_FIELDS, TMIFE_LWE_BINDING_DEFAULTS)
        check_request(self.id, dk, lst_sid_enrolled, self.pp["t"], credentials)
        q = gp.mpz(self.pp["q"])
        # Shamir: (dk, lambda_k); replicated: the sum of the assigned pieces, 1
        share, multiplier, tag = _share or share_for_enrolled(
            dk, self.id, lst_sid_enrolled, self.pp["t"], q, **enrolled_share_options(self))
        if set(dict_ct) != set(share["sk_y"]):
            raise FESchemeError("need one ciphertext for every slot of the key")
        check_ciphertext_tags(dict_ct, dk.get("mac_keys"), self.id, _TAG_DOMAIN, _tag_payload)

        delta = q // gp.mpz(self.pp["p"])
        key_part = delta * gp.mpz(share["z"])
        c_y = gp.mpz(0)
        canonical = {}
        for nid, ct in dict_ct.items():
            ct0 = [v % q for v in digits_to_vector(ct["ct0"])]
            ct1 = [v % q for v in digits_to_vector(ct["ct1"])]
            if len(ct0) != len(share["sk_y"][nid]) or len(ct1) != self.pp["eta"]:
                raise FESchemeError("malformed ciphertext of {}".format(nid))
            canonical[nid] = [[int(v) for v in ct0], [int(v) for v in ct1]]
            y_vec = [gp.mpz(int(v)) for v in credentials[nid]]
            key_part += vecdot_mod(ct0, [gp.mpz(v) for v in share["sk_y"][nid]], q)
            c_y += vecdot_mod(y_vec, ct1, q)

        # the noise is seeded with what the key part depends on -- the ct0 values,
        # not their encoding, ct1, extra fields or other servers' tags -- so a
        # request for the same key part always draws the same noise
        ct_digest = ciphertexts_digest(canonical)
        key_inputs = ciphertexts_digest({nid: values[0] for nid, values in canonical.items()})
        # the partial depends on the enrolled set only through the tag (lambda,
        # or the assigned pieces)
        # and with the digest of this server's public parameters (the key share's
        # pp_binding, checked above): under other parameters a request gets
        # another noise, never the same noise with another key term
        rng = XofRandom(noise_seed(dk, key_inputs, tag, binding))
        flood, budget = self._kl_flooding()
        if flood is None:
            e_k = flooding_noise(int(self.pp["sigma_f"]), rng)
        else:
            flood.check_sampler()
            budget.consume(1)
            e_k = int(flood.sample(1, rng)[0])
        return {
            "w": gp.digits((multiplier * key_part + e_k) % q),
            "c_y": gp.digits(c_y % q),
            "ct_digest": ct_digest,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

    def combine_decrypt(self, partials) -> float | None:
        """Combine partial decryptions from at least ``t`` servers.

        Args:
            partials: ``{sid: share_decrypt(...)}`` for one enrolled set, or a
                list of such mappings for several enrolled sets, whose results
                must then agree (see
                :func:`~pyfe4ai.schemes.threshold.roles.combine_consistently`:
                servers are assumed honest-but-curious, and only this
                redundancy detects one that returns a wrong partial).

        Raises:
            FESchemeError: If a group is incomplete or invalid, or the groups
                disagree.
        """
        return combine_consistently(partials, self._combine_group)

    def _combine_group(self, dict_ct_prime: dict) -> float | None:
        """Combine at least ``t`` partial decryptions into ``<x, y>``.

        Returns:
            The inner product, or ``None`` if the servers saw different
            ciphertexts or the value is out of range.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given or the shares
                do not match the enrolled set they were computed for.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = list(dict_ct_prime.values())
        if any(
            part["c_y"] != partials[0]["c_y"] or part["ct_digest"] != partials[0]["ct_digest"]
            for part in partials
        ):
            logger.warning("partial decryptions disagree on the ciphertexts they decrypted")
            return None
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        total = gp.mpz(partials[0]["c_y"]) - sum(gp.mpz(part["w"]) for part in partials)
        bound = (
            self.pp["n"] * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        )
        mu = decode_scaled(total % q, delta, q, bound)
        return None if mu is None else float(mu)

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Partially decrypt every element of the slots' ndarray ciphertexts
        (under ``"kl"`` refused whole unless the budget covers all of them;
        each partial produced then counts)."""
        sample = next(iter(dict_ndarray_ct.values()))
        share = None
        if dk and "pieces" in dk and self.id in dk.get("lst_sid", []):
            # the assigned pieces, added up once for all elements
            validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
            share = share_for_enrolled(
                dk, self.id, lst_sid_enrolled, self.pp["t"], gp.mpz(self.pp["q"]), **enrolled_share_options(self)
            )
        flood, budget = self._kl_flooding()
        if flood is not None:
            flood.check_sampler()
            budget.check(sum(int(np.prod(np.shape(layer))) for layer in sample))
        lst_partial = []
        for l in range(len(sample)):
            ary_partial = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_partial[i] = self.share_decrypt(dct_ct, credentials, dk, lst_sid_enrolled, _share=share)
            lst_partial.append(ary_partial)
        return lst_partial

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Combine element-wise partial decryptions from at least ``t`` servers.

        Raises:
            FESchemeError: If the servers' outputs differ in structure or an
                element does not decrypt (servers disagree, or out of range).
        """
        out = combine_elementwise(dict_ndarray_ct, lambda groups, l, i: self.combine_decrypt(groups))
        return [(ary / pow(10, self.precision)).astype(float) for ary in out]
