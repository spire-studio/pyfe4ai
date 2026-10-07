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
             derived from a per-server secret key and ``(l, E)``, so repeated
             requests for the same label and enrolled set get the same answer.
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
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint

logger = logging.getLogger(__name__)


def flooding_sigma(params: dict, n_slots: int, bound_y: int, s: int) -> int:
    """Largest ``sigma_f`` with ``(q0 / q) * s * c * sigma_f <= (n Y + 1) / 2``.

        Args:
            params: Output of :func:`abmpsw20_lwe_parameters`.
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
            rng: Bit source (default: OS CSPRNG); an :class:`XofRandom` makes
                the sample a deterministic function of its seed.
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
            config: As for :class:`MCFELWEKeyGenerator`, plus ``s`` (number of
                decryption servers), ``t`` (threshold) and ``lst_sid``.
        """
        super().__init__(config, **kwargs)
        self.t = config.get("t", CryptoCONST.tMCFE_T)
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
        """Return the MCFE public parameters plus ``s``, ``t``, ``lst_sid``, ``sigma_f``."""
        pp = super().get_public_parameters()
        pp.update(
            {
                "s": self.s,
                "t": self.t,
                "lst_sid": list(self.lst_sid),
                "sigma_f": gp.digits(gp.mpz(self.sigma_f)),
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
            *sid* is unknown.
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
        return {
            "sk_y": [gp.digits(v) for v in share],
            "sid": sid,
            "lst_sid": list(self.lst_sid),
            "weights_id": identifier,
            "noise_key": noise_key,
        }

    def _share_key(self, sk_y: list) -> dict:
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())
        shares = [share_secret(v, self.t, points, self.q) for v in sk_y]
        return {
            sid: ([sh[sid_v[sid]] for sh in shares], secrets.token_hex(32))
            for sid in self.lst_sid
        }


class ThresholdMCFELWE(MCFELWE):
    """Crypto operations for threshold ABMPSW20 LWE MCFE."""

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
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
        if not dk or "sk_y" not in dk:
            raise FEKeyError("no decryption keys provided")
        if self.id not in dk["lst_sid"]:
            raise FESchemeError("local id:{} is not supported".format(self.id))
        fusion_weight = credentials.get("fusion_weight")
        label = credentials.get("label")
        if fusion_weight is None or label is None:
            raise FEKeyError("need fusion_weight and label for threshold share decryption")
        encoded = encode_label(label)
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, fusion_weight)

        sid_v = share_points(dk["lst_sid"])
        lagrange = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], self.q
        )
        share_term = dot_mod([gp.mpz(v) for v in dk["sk_y"]], self._hash_label(label), self.q)
        context = json.dumps([encoded, sorted(lst_sid_enrolled)]).encode("utf-8")
        e_k = flooding_noise(
            int(self.pp["sigma_f"]),
            rng=XofRandom(bytes.fromhex(dk["noise_key"]) + context),
        )
        w = (lagrange * share_term + e_k) % self.q
        return {
            "w": gp.digits(w),
            "c_y": int(self._weighted_sum(dict_ct, fusion_weight)),
            "label": encoded,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

    def combine_decrypt(self, dict_ct_prime: dict) -> float | None:
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
            logger.warning("ct consistency verification failed.")
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
                produced by :meth:`encrypt_lst_ndarray` under ``credentials["label"]``.
            credentials: Dict with ``fusion_weight`` and ``label``.
            dk: This server's key share.
            lst_sid_enrolled: The servers whose shares will be combined.
        """
        label = credentials.get("label")
        if not isinstance(label, str):
            raise FEValidationError("need a string label")
        sample = next(iter(dict_ndarray_ct.values()))
        lst_partial = []
        for l in range(len(sample)):
            ary_partial = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                elem_credentials = dict(credentials, label=_element_label(label, l, i))
                ary_partial[i] = self.share_decrypt(
                    dct_ct, elem_credentials, dk, lst_sid_enrolled
                )
            lst_partial.append(ary_partial)
        return lst_partial

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Combine element-wise partial decryptions from at least ``t`` servers.

        Args:
            dict_ndarray_ct: Mapping of server ids to the outputs of
                :meth:`compute_lst_ndarray_ct`.
        """
        sample = next(iter(dict_ndarray_ct.values()))
        lst_dec = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                ary_dec[i] = self.combine_decrypt(
                    {sid: dict_ndarray_ct[sid][l][i] for sid in dict_ndarray_ct}
                )
            lst_dec.append(ary_dec.astype(float))
        return lst_dec
