"""
Threshold Ring-LWE identity-based multi-client inner-product FE: the BMMS21
identity-based MCFE with a Shamir-shared functional key and statistically
flooded partial decryptions.

| MCFE layer: :mod:`pyfe4ai.schemes.mcfe.ring_lwe` (BMMS21, PKC 2022, Section 6).
| Threshold layer in the setting of R. Xu et al., "TAPFed: Threshold Secure
| Aggregation for Privacy-Preserving Federated Learning", IEEE TDSC 2024.

* type:     identity-based multi-client encryption, threshold functional decryption
* setting:  R_q = Z_q[x] / (x^ring_n + 1), q an NTT-friendly prime

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
``e_k <- D_{Z, 2^kappa B}`` per coefficient smudges it (statistical
argument), with ``q`` enlarged so that ``s`` flooding terms still decode.
The noise is derived from a per-server secret and the request (enrolled
set and ciphertexts), so repeated requests cannot be averaged. The threshold
layer is our own design.
"""

from __future__ import annotations

import json
import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mcfe.labels import chunk_label
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import chunk_key_index
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.schemes.threshold_utils import check_fusion_weight
from pyfe4ai.schemes.threshold_utils import ciphertexts_digest
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import combine_consistently
from pyfe4ai.schemes.threshold_utils import partial_groups
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.utils.ring_lwe_utils import bmms21_parameters
from pyfe4ai.utils.ring_lwe_utils import canonical_ring_ciphertext
from pyfe4ai.utils.ring_lwe_utils import mat_vec_mul
from pyfe4ai.utils.ring_lwe_utils import next_ntt_prime
from pyfe4ai.utils.ring_lwe_utils import poly_add
from pyfe4ai.utils.ring_lwe_utils import poly_mul
from pyfe4ai.utils.ring_lwe_utils import transpose
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

_CONSTRUCTION = "bmms21-ibmcfe-shamir-smudging/v1"


def _flooding_poly(sigma: int, ring_n: int, rng) -> list:
    bound = GAUSSIAN_TAIL_CUT * sigma
    out = []
    while len(out) < ring_n:
        e = discrete_gaussian_bigint(sigma, 1, rng=rng)[0]
        if abs(e) <= bound:
            out.append(e)
    return out


class ThresholdMCFERingLWEKeyGenerator(MCFERingLWEKeyGenerator):
    """Key generator for threshold BMMS21 identity-based MCFE over Ring-LWE."""

    _scheme_type = CryptoCONST.TYPE_TMCFE_RING_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`MCFERingLWEKeyGenerator`, plus ``s``,
                ``t``, ``lst_sid`` and ``stat_param`` (``kappa`` of the flooding).
        """
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        s = config.get("s") or len(config.get("lst_sid", [])) or 1
        self.lst_sid = config.get("lst_sid", ["sid_{}".format(i) for i in range(s)])
        self.stat_param = int(config.get("stat_param", 64))
        self.dict_dk = {}
        super().__init__(dict(config, s=s), **kwargs)
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)

    def _param_verification(self, param: dict) -> bool:
        if not (
            super()._param_verification(param)
            and param.get("threshold_construction") == _CONSTRUCTION
            and param.get("s") == self.s
            and param.get("stat_param") == self.stat_param
        ):
            return False
        params = bmms21_parameters(
            self.sec_param, self.n * self.eta, self.bound_x, self.bound_y, self.ring_n
        )
        return param.get("sigma_f") == str(self._flooded_modulus(self.ring_n, *params)[1])

    def _apply_parameters(self, param: dict) -> None:
        super()._apply_parameters(param)
        self.sigma_f = int(param["sigma_f"])

    def _flooded_modulus(self, ring_n: int, p, q, sigma1, sigma2, sigma3) -> tuple:
        """Return ``(q', sigma_f)``: the BMMS21 modulus *q* enlarged for flooding."""
        noise_bound = self.n * self.eta * int(self.bound_y) * int(
            2 * ring_n * GAUSSIAN_TAIL_CUT ** 2 * sigma1 * sigma2
            + GAUSSIAN_TAIL_CUT * sigma3 + 1
        )
        sigma_f = noise_bound << self.stat_param
        budget = noise_bound + self.s * GAUSSIAN_TAIL_CUT * sigma_f
        return next_ntt_prime(max(q, p * (2 * budget + 4)), ring_n), sigma_f

    def _modulus_for(self, ring_n: int):
        params = bmms21_parameters(
            self.sec_param, self.n * self.eta, self.bound_x, self.bound_y, ring_n
        )
        return self._flooded_modulus(ring_n, *params)[0]

    def _generate_and_save(self, param_file: str) -> None:
        super()._generate_and_save(param_file)
        self.q, self.sigma_f = self._flooded_modulus(
            self.ring_n, self.p, self.q, self.sigma1, self.sigma2, self.sigma3
        )
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)
        with open(param_file, "r", encoding="utf-8") as f:
            param = json.load(f)
        param.update(
            {
                "threshold_construction": _CONSTRUCTION,
                "s": self.s,
                "stat_param": self.stat_param,
                "sigma_f": str(self.sigma_f),
                "q": gp.digits(self.q),
            }
        )
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(param, f)

    def setup(self) -> None:
        """MCFE setup; forgets previously shared keys."""
        super().setup()
        self.dict_dk = {}

    def get_public_parameters(self) -> dict:
        """MCFE public parameters plus ``s``, ``t``, ``lst_sid`` and ``sigma_f``."""
        pp = super().get_public_parameters()
        pp.update(
            {"s": self.s, "t": self.t, "lst_sid": list(self.lst_sid), "sigma_f": str(self.sigma_f)}
        )
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of the key for ``(y, label, n_chunks)``.

        Returns:
            Dict with shares ``sk_y``, ``z``, ``z_chunks`` and ``label``,
            ``sid``, ``lst_sid``, ``weights_id`` and ``noise_key``, or ``None``
            if *sid* is unknown.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for Threshold MCFE Ring-LWE")
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        weights_id = fusion_weight_id(credentials.get("fusion_weight"))
        ident = json.dumps(
            [weights_id, credentials.get("label"), int(credentials.get("n_chunks", 0))]
        )
        if ident not in self.dict_dk:
            key = super().get_decryption_keys(sid, credentials=credentials)
            self.dict_dk[ident] = self._share_key(key)
        dk = dict(self.dict_dk[ident][sid])
        dk.update({"sid": sid, "lst_sid": list(self.lst_sid), "weights_id": weights_id})
        return dk

    def _share_key(self, key: dict) -> dict:
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())

        def share_poly(coeffs) -> dict:
            shares = [share_secret(gp.mpz(v), self.t, points, self.q) for v in coeffs]
            return {sid: [gp.digits(sh[sid_v[sid]]) for sh in shares] for sid in self.lst_sid}

        sk_shares = {nid: share_poly(v) for nid, v in key["sk_y"].items()}
        z_shares = share_poly(key["z"])
        chunk_shares = [share_poly(zc) for zc in key["z_chunks"]]
        return {
            sid: {
                "sk_y": {nid: sk_shares[nid][sid] for nid in sk_shares},
                "z": z_shares[sid],
                "z_chunks": [cs[sid] for cs in chunk_shares],
                "label": key["label"],
                "noise_key": secrets.token_hex(32),
            }
            for sid in self.lst_sid
        }


class ThresholdMCFERingLWE(MCFERingLWE):
    """Crypto operations for threshold BMMS21 identity-based MCFE over Ring-LWE."""

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list,
        _z_share=None,
    ) -> dict:
        """Compute this server's flooded partial decryption.

        Args:
            dict_ct: One ciphertext per client, produced under ``credentials["label"]``.
            credentials: Dict with ``fusion_weight`` and ``label``.
            dk: This server's key share (derived for that label).
            lst_sid_enrolled: The servers whose partials will be combined.

        Returns:
            Dict with ``w``, ``c_y``, ``k``, ``ct_digest``, ``sid`` and
            ``lst_sid_enrolled``.
        """
        if not dk or "sk_y" not in dk:
            raise FEKeyError("no decryption keys provided")
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        fusion_weight = credentials.get("fusion_weight")
        if dk.get("label") != credentials.get("label"):
            raise FESchemeError("the key share was derived for another label")
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, fusion_weight)
        if set(dict_ct) != set(dk["sk_y"]):
            raise FESchemeError("need one ciphertext for every client of the key")
        q = gp.mpz(self.pp["q"])
        ring_n = self.pp["ring_n"]
        # the noise is seeded with the digest of these canonical values, so a
        # re-encoded request (v + q, extra fields) gets the same noise
        cts = {}
        for nid, ct in dict_ct.items():
            try:
                cts[nid] = canonical_ring_ciphertext(ct, int(q), self.pp["eta"], ring_n)
            except FESchemeError as exc:
                raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
        batch = {k for k, _, _ in cts.values()}
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")

        delta = q // gp.mpz(self.pp["p"])
        z_share = dk["z"] if _z_share is None else _z_share
        key_part = [delta * gp.mpz(v) % q for v in z_share]
        c_y = [gp.mpz(0)] * ring_n
        for nid, (_, ct0, ct1) in cts.items():
            ct0 = [[gp.mpz(v) for v in row] for row in ct0]
            ct1 = [gp.mpz(v) for v in ct1]
            y_vec = [gp.mpz(int(v)) for v in fusion_weight[nid]]
            c_y = poly_add(c_y, mat_vec_mul(transpose(ct0), y_vec, q), q)
            key_part = poly_add(key_part, poly_mul(ct1, [gp.mpz(v) for v in dk["sk_y"][nid]], q), q)

        sid_v = share_points(dk["lst_sid"])
        lagrange = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], q
        )
        ct_digest = ciphertexts_digest(cts)
        context = json.dumps([ct_digest, sorted(lst_sid_enrolled)]).encode("utf-8")
        noise = _flooding_poly(
            int(self.pp["sigma_f"]), ring_n, XofRandom(bytes.fromhex(dk["noise_key"]) + context)
        )
        w = [(lagrange * v + e) % q for v, e in zip(key_part, noise)]
        return {
            "w": [gp.digits(v) for v in w],
            "c_y": [gp.digits(v) for v in c_y],
            "k": batch.pop(),
            "ct_digest": ct_digest,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

    def combine_decrypt(self, partials) -> list | None:
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

    def _combine_group(self, dict_ct_prime: dict) -> list | None:
        """Combine at least ``t`` partial decryptions into the batch of inner products.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given or the shares
                do not match the enrolled set they were computed for.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = list(dict_ct_prime.values())
        first = partials[0]
        if any(
            (p["c_y"], p["ct_digest"], p["k"]) != (first["c_y"], first["ct_digest"], first["k"])
            for p in partials
        ):
            logger.warning("partial decryptions disagree on the ciphertexts they decrypted")
            return None
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        total = [gp.mpz(v) for v in first["c_y"]]
        for part in partials:
            total = [(a - gp.mpz(v)) % q for a, v in zip(total, part["w"])]
        bound = (
            self.pp["n"] * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        )
        return [decode_scaled(v, delta, q, bound) for v in total[: int(first["k"])]]

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Partially decrypt every chunk of the clients' packed ndarray ciphertexts."""
        sample = next(iter(dict_ndarray_ct.values()))
        if self.key_chunks(sample) > len(dk.get("z_chunks", [])):
            raise FEKeyError("the key share covers fewer chunks than the ciphertexts")
        out = []
        for l, layer in enumerate(sample):
            parts = []
            for j in range(len(layer["chunks"])):
                dct_ct = {nid: dict_ndarray_ct[nid][l]["chunks"][j] for nid in dict_ndarray_ct}
                parts.append(self.share_decrypt(
                    dct_ct, credentials, dk, lst_sid_enrolled,
                    _z_share=dk["z_chunks"][chunk_key_index(layer, j, len(dk["z_chunks"]))],
                ))
            out.append({"shape": layer["shape"], "parts": parts})
        return out

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Combine chunk-wise partial decryptions from at least ``t`` servers."""
        groups = partial_groups(dict_ndarray_ct)
        sample = next(iter(groups[0].values()))
        result = []
        for l, layer in enumerate(sample):
            values = []
            for j in range(len(layer["parts"])):
                values.extend(self.combine_decrypt(
                    [{sid: g[sid][l]["parts"][j] for sid in g} for g in groups]
                ))
            ary = np.array(values, dtype=float).reshape(layer["shape"])
            result.append(ary / pow(10, self.precision))
        return result
