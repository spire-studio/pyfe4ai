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
LWE instance. ``e_k`` is derived from a per-server secret key and the
request (enrolled set and ciphertexts), so repeating a request cannot be
used to average the noise away. The threshold layer is our own design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
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
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.utils.lwe_utils import derive_lwe_parameters
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.matrix_utils import matrix_to_digits
from pyfe4ai.utils.matrix_utils import vecdot_mod
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint
from pyfe4ai.utils.sampling_utils import rand_uniform_matrix
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)

_CONSTRUCTION = "acfgu18-abdp15-shamir-smudging/v1"
_DEFAULT_STAT_PARAM = 64


def _flooding_noise(sigma: int, rng) -> int:
    bound = GAUSSIAN_TAIL_CUT * sigma
    while True:
        e = discrete_gaussian_bigint(sigma, 1, rng=rng)[0]
        if abs(e) <= bound:
            return int(e)


class ThresholdMIFELWEKeyGenerator(MIFELWEKeyGenerator):
    """Key generator for threshold MIFE-LWE (ACFGU18 + Shamir + smudging)."""

    _scheme_type = CryptoCONST.TYPE_TMIFE_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`MIFELWEKeyGenerator`, plus ``s`` (number of
                servers), ``t`` (threshold), ``lst_sid`` and ``stat_param``
                (statistical parameter ``kappa`` of the flooding).
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

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma = gp.mpz(param["l_sigma"])
        self.sigma_f = int(param["sigma_f"])
        self.A = digits_to_matrix(param["A"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("construction") == _CONSTRUCTION
            and param.get("sec_param") == self.sec_param
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("s") == self.s
            and param.get("lwe_n") == self.lwe_n
            and param.get("stat_param") == self.stat_param
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
        )

    def _generate_and_save(self, param_file: str) -> None:
        self.p, q_base, self.m, self.sigma_q, self.l_sigma = derive_lwe_parameters(
            self.bound_x, self.bound_y, self.eta, self.lwe_n, extra_dimension=self.n
        )
        # ciphertext noise of a full decryption: |sum_i y_i^T E_i^T r_i|
        noise_bound = (
            self.n * self.eta * int(self.bound_y) * self.m
            * int(GAUSSIAN_TAIL_CUT * self.sigma_q + 1)
        )
        self.sigma_f = noise_bound << self.stat_param
        budget = noise_bound + self.s * GAUSSIAN_TAIL_CUT * self.sigma_f
        # Delta = floor(q / p) must exceed 2 * budget (plus a margin for rounding)
        self.q = gp.next_prime(max(q_base, self.p * (2 * budget + 4)))
        self.A = rand_uniform_matrix(self.m, self.lwe_n, self.q)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "sec_param": self.sec_param,
                    "eta": self.eta,
                    "n": self.n,
                    "s": self.s,
                    "lwe_n": self.lwe_n,
                    "stat_param": self.stat_param,
                    "m": self.m,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "sigma_q": self.sigma_q,
                    "l_sigma": gp.digits(self.l_sigma),
                    "sigma_f": str(self.sigma_f),
                    "A": matrix_to_digits(self.A),
                },
                f,
            )

    def setup(self) -> None:
        """Sample the MIFE keys and forget any previously shared keys."""
        super().setup()
        self.dict_dk = {}

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
            pad term), ``sid``, ``lst_sid``, ``weights_id`` and ``noise_key``,
            or ``None`` if *sid* is unknown.
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
        dk.update({"sid": sid, "lst_sid": list(self.lst_sid), "weights_id": identifier})
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
        """
        if not dk or "sk_y" not in dk or "z" not in dk:
            raise FEKeyError("no decryption keys provided")
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, credentials)
        if set(dict_ct) != set(dk["sk_y"]):
            raise FESchemeError("need one ciphertext for every slot of the key")

        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        key_part = delta * gp.mpz(dk["z"])
        c_y = gp.mpz(0)
        for nid, ct in dict_ct.items():
            ct0 = digits_to_vector(ct["ct0"])
            ct1 = digits_to_vector(ct["ct1"])
            y_vec = [gp.mpz(int(v)) for v in credentials[nid]]
            key_part += vecdot_mod(ct0, [gp.mpz(v) for v in dk["sk_y"][nid]], q)
            c_y += vecdot_mod(y_vec, ct1, q)

        sid_v = share_points(dk["lst_sid"])
        lagrange = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], q
        )
        ct_digest = hashlib.sha256(
            json.dumps(dict_ct, sort_keys=True).encode("utf-8")
        ).hexdigest()
        context = json.dumps([ct_digest, sorted(lst_sid_enrolled)]).encode("utf-8")
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

    def combine_decrypt(self, dict_ct_prime: dict) -> float | None:
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
            logger.warning("ct consistency verification failed.")
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
        """Combine element-wise partial decryptions from at least ``t`` servers."""
        sample = next(iter(dict_ndarray_ct.values()))
        lst_dec = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                ary_dec[i] = self.combine_decrypt(
                    {sid: dict_ndarray_ct[sid][l][i] for sid in dict_ndarray_ct}
                )
            lst_dec.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_dec
