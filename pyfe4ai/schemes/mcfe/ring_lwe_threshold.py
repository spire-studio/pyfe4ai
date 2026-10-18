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
Ciphertexts (and ndarray chunks) carry one HMAC tag per server over the
(chunk) label and the ciphertext; servers only decrypt ciphertexts tagged
for them under the label they serve, since ``w_k`` is linear in ``ct1``.
The noise is seeded with a per-server key (fixed per weights and label),
``lambda_k``, the label or chunk and the ``ct1`` values, so a request for
the same key part draws the same noise; a corrupted client varying its own
``ct1`` draws fresh noise each time. The flooding argument is statistical
per answer (a statistical distance of about ``2^-stat_param`` per
coefficient, adding up over the answers); it does not bound what many
answers to varied ``ct1`` reveal together: averaging them would take about
``2^(2 stat_param)`` partial decryptions, but other ways of combining these
noisy linear equations are not ruled out. The threshold layer is our own
design.

With the native backend (:mod:`pyfe4ai.utils.native_backend`; moduli below
``2^253``, which includes the flooded ``q`` of the standard profile) the MCFE
layer, the Shamir sharing of the key, the parsing, digests and NTT-domain
products of a partial decryption and the combination run natively, with
the values of the Python code; the flooding noise is always drawn by
:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint` (natively too,
with the same samples).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mcfe.labels import chunk_label
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import chunk_key_index
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.labels import encode_label
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
from pyfe4ai.schemes.threshold_utils import partial_groups
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils import native_backend
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.utils.ring_lwe_utils import canonical_ring_ciphertext
from pyfe4ai.utils.ring_lwe_utils import mat_vec_mul
from pyfe4ai.utils.ring_lwe_utils import next_ntt_prime
from pyfe4ai.utils.ring_lwe_utils import poly_add
from pyfe4ai.utils.ring_lwe_utils import poly_mul
from pyfe4ai.utils.ring_lwe_utils import RingNTT
from pyfe4ai.utils.ring_lwe_utils import transpose
from pyfe4ai.utils.sampling_utils import XofRandom
from pyfe4ai.utils.sampling_utils import (
    FLOODING_NOISE_VERSION, FLOODING_SAMPLER, check_flooding_version, discrete_gaussian_bigint,
)
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)

_CONSTRUCTION = "bmms21-ibmcfe-shamir-smudging/v2"  # v2: flooding at bits + 320, sampler pinned
_TAG_DOMAIN = "pyfe4ai/tmcfe-ring-lwe/ciphertext-tag/v1"


def _tag_payload(encoded_label: str, ct: dict) -> list:
    """What a tMCFE-Ring-LWE ciphertext tag covers: the (chunk) label and the
    ciphertext as serialised."""
    return [encoded_label, ct.get("k"), ct.get("ct0"), ct.get("ct1")]


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
                plus ``s``, ``t``, ``lst_sid`` and ``stat_param`` (``kappa`` of
                the flooding).
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
        params = self._bmms21_tuple(self.ring_n)
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
        return self._flooded_modulus(ring_n, *self._bmms21_tuple(ring_n))[0]

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
        """MCFE setup and the client-server MAC keys; forgets previously shared keys."""
        super().setup()
        self.dict_dk = {}
        self.mac_keys = client_server_mac_keys(self.lst_nid, self.lst_sid)
        self._noise_masters = {sid: secrets.token_hex(32) for sid in self.lst_sid}

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Client *nid*'s MCFE keys plus ``mac_keys`` (``{sid: key}``) for the
        ciphertext tags."""
        sk = super().get_private_keys(nid, **kwargs)
        if sk is not None:
            sk["mac_keys"] = dict(self.mac_keys[nid])
        return sk

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
            ``sid``, ``lst_sid``, ``weights_id``, ``noise_key`` and
            ``mac_keys`` (``{nid: key}``), or ``None`` if *sid* is unknown.
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
            self.dict_dk[ident] = self._share_key(key, weights_id)
        dk = dict(self.dict_dk[ident][sid])
        dk.update({
            "sid": sid,
            "lst_sid": list(self.lst_sid),
            "weights_id": weights_id,
            "mac_keys": {nid: self.mac_keys[nid][sid] for nid in self.lst_nid},
        })
        return dk

    def _share_key(self, key: dict, weights_id: str) -> dict:
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())
        polys = list(key["sk_y"].values()) + [key["z"]] + list(key["z_chunks"])
        shared = self._share_polys_native(polys, points)
        if shared is None:
            def share_poly(coeffs) -> dict:
                shares = [share_secret(gp.mpz(v), self.t, points, self.q) for v in coeffs]
                return {sid: [gp.digits(sh[sid_v[sid]]) for sh in shares] for sid in self.lst_sid}

            shared = [share_poly(poly) for poly in polys]
        n_sk = len(key["sk_y"])
        sk_shares = dict(zip(key["sk_y"], shared[:n_sk]))
        z_shares = shared[n_sk]
        chunk_shares = shared[n_sk + 1:]
        return {
            sid: {
                "sk_y": {nid: sk_shares[nid][sid] for nid in sk_shares},
                "z": z_shares[sid],
                "z_chunks": [cs[sid] for cs in chunk_shares],
                "label": key["label"],
                "noise_version": FLOODING_NOISE_VERSION,
                "noise_sampler": FLOODING_SAMPLER,
                # one per server, weights and label: a re-sharing for another
                # n_chunks (same key part) must not come with fresh noise
                "noise_key": hmac.new(
                    bytes.fromhex(self._noise_masters[sid]),
                    json.dumps([weights_id, key["label"]]).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            }
            for sid in self.lst_sid
        }

    def _share_polys_native(self, polys: list, points: list) -> list | None:
        """:func:`share_secret` of every coefficient of *polys* in one native
        call (coefficients drawn as ``share_secret`` draws them, in the same
        order), as ``[{sid: shares}]``; ``None`` when the Python path applies."""
        ctx = native_backend.ring_context(self.q, self.ring_n)
        if ctx is None:
            return None
        flat = [v for poly in polys for v in poly]
        width = (int(self.q).bit_length() + 7) // 8
        out = native_backend.ext().shamir_share(
            ctx.cap, flat, self.t, points,
            native_backend.random_input(4 * width * len(flat) * (self.t - 1) + 64),
            **native_backend.seed_kwargs(),
        )
        shared, start = [], 0
        for poly in polys:
            shared.append({sid: out[i][start:start + len(poly)] for i, sid in enumerate(self.lst_sid)})
            start += len(poly)
        return shared


class ThresholdMCFERingLWE(MCFERingLWE):
    """Crypto operations for threshold BMMS21 identity-based MCFE over Ring-LWE."""

    _binary_ciphertexts = False  # tags cover the JSON form of a ciphertext

    def _mac_keys(self) -> dict:
        mac_keys = (self.sk or {}).get("mac_keys") if self._has_private_keys() else None
        if not mac_keys:
            raise FEKeyError("encryption key lacks the ciphertext MAC keys")
        return mac_keys

    def encrypt(self, matrix_pt: list[list[int]], label: str) -> dict:
        """MCFE encryption under *label*, tagged for every server."""
        mac_keys = self._mac_keys()
        ct = super().encrypt(matrix_pt, label)
        return tag_ciphertext(ct, mac_keys, _TAG_DOMAIN, _tag_payload(encode_label(label), ct))

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Packed ndarray encryption; every chunk is tagged under its chunk label."""
        mac_keys = self._mac_keys()
        out = super().encrypt_lst_ndarray(lst_ndarray, **kwargs)
        for layer in out:
            for j, ct in enumerate(layer["chunks"]):
                encoded = encode_label(chunk_label(kwargs["label"], layer["first_chunk"] + j))
                tag_ciphertext(ct, mac_keys, _TAG_DOMAIN, _tag_payload(encoded, ct))
        return out

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list,
        _z_share=None, _chunk=None, _sk_hats=None,
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
        check_flooding_version(dk)
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        fusion_weight = credentials.get("fusion_weight")
        if dk.get("label") != credentials.get("label"):
            raise FESchemeError("the key share was derived for another label")
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, fusion_weight)
        if set(dict_ct) != set(dk["sk_y"]):
            raise FESchemeError("need one ciphertext for every client of the key")
        # only ciphertexts the clients tagged for this server under the label
        # (or chunk label) it serves: w is linear in ct1
        tag_label = credentials["label"] if _chunk is None else chunk_label(credentials["label"], _chunk)
        encoded = encode_label(tag_label)
        check_ciphertext_tags(
            dict_ct, dk.get("mac_keys"), self.id, _TAG_DOMAIN,
            lambda nid, ct: _tag_payload(encoded, ct),
        )
        q = gp.mpz(self.pp["q"])
        ring_n = self.pp["ring_n"]
        delta = q // gp.mpz(self.pp["p"])
        z_share = dk["z"] if _z_share is None else _z_share
        hats = {} if _sk_hats is None else _sk_hats
        ctx = native_backend.ring_context(q, ring_n)
        native = None if ctx is None else self._native_inputs(ctx, dict_ct, dk, fusion_weight, z_share)
        if native is None:
            cts = {}
            for nid, ct in dict_ct.items():
                try:
                    cts[nid] = canonical_ring_ciphertext(ct, int(q), self.pp["eta"], ring_n)
                except FESchemeError as exc:
                    raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
            batch = {k for k, _, _ in cts.values()}
        else:
            nids, ks, packed, ct_digest, key_inputs = native
            batch = set(ks)
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")

        if native is None:
            key_part = [delta * gp.mpz(v) % q for v in z_share]
            c_y = [gp.mpz(0)] * ring_n
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
                    key_part = poly_add(key_part, poly_mul(ct1, [gp.mpz(v) for v in dk["sk_y"][nid]], q), q)
                    continue
                if nid not in hats:
                    hats[nid] = ring.forward([gp.mpz(v) for v in dk["sk_y"][nid]])
                acc = ring.mul_add(acc, ring.forward(ct1), hats[nid])
            if acc is not None:
                key_part = poly_add(key_part, ring.inverse(acc), q)
            ct_digest = ciphertexts_digest(cts)
            key_inputs = ciphertexts_digest({nid: ct1 for nid, (_, _, ct1) in cts.items()})

        sid_v = share_points(dk["lst_sid"])
        lagrange = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], q
        )
        # the noise is seeded with what the key part depends on -- the ct1
        # values and the key term (the label or chunk) -- so a request for the
        # same key part always draws the same noise
        # the partial depends on the enrolled set only through lambda
        context = json.dumps([key_inputs, encoded, gp.digits(lagrange)]).encode("utf-8")
        noise = _flooding_poly(
            int(self.pp["sigma_f"]), ring_n, XofRandom(bytes.fromhex(dk["noise_key"]) + context)
        )
        if native is None:
            w = [gp.digits((lagrange * v + e) % q) for v, e in zip(key_part, noise)]
            c_y = [gp.digits(v) for v in c_y]
        else:
            for nid in nids:  # NTT(sk_i share), cached across chunks
                if ("native", nid) not in hats:
                    hats[("native", nid)] = ctx.ntt_mont(dk["sk_y"][nid])
            w, c_y = native_backend.ext().share_partial(
                ctx.cap, packed, self.pp["eta"], [[int(v) for v in fusion_weight[nid]] for nid in nids],
                [hats[("native", nid)] for nid in nids], z_share, delta, lagrange, noise,
            )
        return {
            "w": w,
            "c_y": c_y,
            "k": batch.pop(),
            "ct_digest": ct_digest,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

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
        bound = (
            self.pp["n"] * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        )
        ctx = native_backend.ring_context(q, self.pp["ring_n"])
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

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Partially decrypt every chunk of the clients' packed ndarray ciphertexts."""
        sample = next(iter(dict_ndarray_ct.values()))
        if self.key_chunks(sample) > len(dk.get("z_chunks", [])):
            raise FEKeyError("the key share covers fewer chunks than the ciphertexts")
        out = []
        sk_hats = {}  # NTT(sk_i), transformed once for all chunks
        for l, layer in enumerate(sample):
            parts = []
            for j in range(len(layer["chunks"])):
                dct_ct = {nid: dict_ndarray_ct[nid][l]["chunks"][j] for nid in dict_ndarray_ct}
                index = chunk_key_index(layer, j, len(dk["z_chunks"]))
                parts.append(self.share_decrypt(
                    dct_ct, credentials, dk, lst_sid_enrolled,
                    _z_share=dk["z_chunks"][index], _chunk=index, _sk_hats=sk_hats,
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
