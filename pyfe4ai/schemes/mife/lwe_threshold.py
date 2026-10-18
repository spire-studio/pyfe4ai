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
with the server's key, ``lambda_k`` and the ``ct0`` values, so a request for
the same key part draws the same noise; a corrupted client varying its own
``ct0`` changes the key part and so draws fresh noise each time. The flooding
argument is statistical per answer (a statistical distance of about
``2^-stat_param`` per coefficient, adding up over the answers); it does not
bound what many answers to varied ``ct0`` reveal together about the other
slots' part: averaging them would take about ``2^(2 stat_param)`` partial
decryptions, but other ways of combining these noisy linear equations are
not ruled out.

Under the ``standard`` security profile ``lwe_n = "auto"`` picks the
smallest dimension meeting :data:`pyfe4ai.utils.lwe_utils.WIDE_LWE_128_LOG_RATIO`
at the enlarged ``q`` (``stat_param = 128``): about 6000 for ``X = Y = 50``
(5952 for two clients: ``q`` of 217 bits, ``m`` of 1.3 * 10^6, ``A`` of
7.8 * 10^9 entries). ``A`` then needs more than 400 GB in memory, so the
profile fixes ``matrix_mode = "stream"``. Measured on one machine at that
size: key generation 38 minutes (8.5 GB; the machine was busy), one
encryption 572 s (3.9 GB), a batch of 64 encryptions 679 s (5.2 GB), a
partial decryption 13 ms.
"""

from __future__ import annotations

import json
import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.schemes.threshold_utils import check_ciphertext_tags
from pyfe4ai.schemes.threshold_utils import check_fusion_weight
from pyfe4ai.schemes.threshold_utils import client_server_mac_keys
from pyfe4ai.schemes.threshold_utils import tag_ciphertext
from pyfe4ai.schemes.threshold_utils import ciphertexts_digest
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import combine_consistently
from pyfe4ai.schemes.threshold_utils import combine_elementwise
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.lwe_utils import abdp15_parameters
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.utils.lwe_utils import wide_lwe_n
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.sampling_utils import (
    FLOODING_NOISE_VERSION, FLOODING_SAMPLER, check_flooding_version, discrete_gaussian_bigint,
)
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)

_CONSTRUCTION = "acfgu18-abdp15-shamir-smudging-seeded-a/v4"  # v4: flooding at bits + 320, sampler pinned
_DEFAULT_STAT_PARAM = 64
_TAG_DOMAIN = "pyfe4ai/tmife-lwe/ciphertext-tag/v1"


def _tag_payload(nid, ct: dict) -> list:
    """What a tMIFE-LWE ciphertext tag covers: ``ct0`` and ``ct1`` as serialised."""
    return [ct.get("ct0"), ct.get("ct1")]


def _flooding_noise(sigma: int, rng) -> int:
    bound = GAUSSIAN_TAIL_CUT * sigma
    while True:
        e = discrete_gaussian_bigint(sigma, 1, rng=rng, sampler=FLOODING_SAMPLER)[0]
        if abs(e) <= bound:
            return int(e)


class ThresholdMIFELWEKeyGenerator(MIFELWEKeyGenerator):
    """Key generator for threshold MIFE-LWE (ACFGU18 + Shamir + smudging)."""

    _scheme_type = CryptoCONST.TYPE_TMIFE_LWE
    _construction = _CONSTRUCTION

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`~pyfe4ai.schemes.mife.lwe.MIFELWEKeyGenerator`,
                plus ``s`` (number of servers), ``t`` (threshold), ``lst_sid``
                and ``stat_param`` (statistical parameter ``kappa`` of the
                flooding).
        """
        self.t = config.get("t", CryptoCONST.tMIFE_T)
        s = config.get("s") or len(config.get("lst_sid", [])) or 1
        self.lst_sid = config.get("lst_sid", ["sid_{}".format(i) for i in range(s)])
        self.stat_param = int(config.get("stat_param", _DEFAULT_STAT_PARAM))
        self.dict_dk = {}
        super().__init__(dict(config, s=s), **kwargs)
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)

    def _record_fields(self) -> dict:
        return {**super()._record_fields(), "s": self.s, "stat_param": self.stat_param}

    def _derive(self) -> dict:
        if self.lwe_n_config == "auto":
            return wide_lwe_n(self._threshold_parameters)
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

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s MIFE encryption key plus ``mac_keys``
        (``{sid: hex}``, one key per server for the ciphertext tags)."""
        sk = super().get_private_keys(nid, **kwargs)
        if sk is not None:
            sk["mac_keys"] = dict(self.mac_keys[nid])
        return sk

    def get_public_parameters(self) -> dict:
        """Return the MIFE public parameters plus ``s``, ``t``, ``lst_sid``, ``sigma_f``."""
        pp = super().get_public_parameters()
        pp.update(
            {"s": self.s, "t": self.t, "lst_sid": list(self.lst_sid), "sigma_f": str(self.sigma_f)}
        )
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
            tags), or ``None`` if *sid* is unknown.
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
        })
        return dk

    def _share_key(self, key: dict) -> dict:
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
    """Crypto operations for threshold MIFE-LWE (ACFGU18 + Shamir + smudging)."""

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
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
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
                those of the key share, or a ciphertext's tag for this server
                does not verify (it was not produced by that slot's client).
        """
        if not dk or "sk_y" not in dk or "z" not in dk:
            raise FEKeyError("no decryption keys provided")
        check_flooding_version(dk)
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, credentials)
        if set(dict_ct) != set(dk["sk_y"]):
            raise FESchemeError("need one ciphertext for every slot of the key")
        check_ciphertext_tags(dict_ct, dk.get("mac_keys"), self.id, _TAG_DOMAIN, _tag_payload)

        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        key_part = delta * gp.mpz(dk["z"])
        c_y = gp.mpz(0)
        canonical = {}
        for nid, ct in dict_ct.items():
            ct0 = [v % q for v in digits_to_vector(ct["ct0"])]
            ct1 = [v % q for v in digits_to_vector(ct["ct1"])]
            if len(ct0) != len(dk["sk_y"][nid]) or len(ct1) != self.pp["eta"]:
                raise FESchemeError("malformed ciphertext of {}".format(nid))
            canonical[nid] = [[int(v) for v in ct0], [int(v) for v in ct1]]
            y_vec = [gp.mpz(int(v)) for v in credentials[nid]]
            key_part += vecdot_mod(ct0, [gp.mpz(v) for v in dk["sk_y"][nid]], q)
            c_y += vecdot_mod(y_vec, ct1, q)

        sid_v = share_points(dk["lst_sid"])
        lagrange = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], q
        )
        # the noise is seeded with what the key part depends on -- the ct0 values,
        # not their encoding, ct1, extra fields or other servers' tags -- so a
        # request for the same key part always draws the same noise
        ct_digest = ciphertexts_digest(canonical)
        key_inputs = ciphertexts_digest({nid: values[0] for nid, values in canonical.items()})
        # the partial depends on the enrolled set only through lambda
        context = json.dumps([key_inputs, gp.digits(lagrange)]).encode("utf-8")
        e_k = _flooding_noise(
            int(self.pp["sigma_f"]), XofRandom(bytes.fromhex(dk["noise_key"]) + context)
        )
        return {
            "w": gp.digits((lagrange * key_part + e_k) % q),
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
                :func:`~pyfe4ai.schemes.threshold_utils.combine_consistently`:
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
        """Partially decrypt every element of the slots' ndarray ciphertexts."""
        sample = next(iter(dict_ndarray_ct.values()))
        lst_partial = []
        for l in range(len(sample)):
            ary_partial = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_partial[i] = self.share_decrypt(dct_ct, credentials, dk, lst_sid_enrolled)
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
