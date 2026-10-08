"""
Threshold function-hiding multi-client inner-product FE over pairings.

The AGT21b FH-MCFE of :mod:`pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing`
with its functional key Shamir-shared among ``s`` decryption servers, any
``t`` of which decrypt together. Own construction (experimental, no
published security proof):

* the authority computes the FH-MCFE key for ``y`` in the exponent,
  ``dk_i = [e_i]_2`` and ``z = sum_i z_i`` (with ``[z_i]_T`` the per-client
  correction), and Shamir-shares every entry of every ``e_i`` and ``z``
  modulo the group order;
* server ``k`` of an enrolled set ``S`` returns
  ``W_k = prod_{i,j} e(ct_{i,j}, [lambda_k e^(k)_{i,j}]_2) / [lambda_k z^(k)]_T``
  with its Lagrange coefficient ``lambda_k``;
* ``prod_{k in S} W_k = prod_i iDec(dk_i, ct_i) = [<x, y>]_T``, then a
  bounded discrete log.

Fewer than ``t`` servers learn nothing about the key (Shamir). For any
ciphertexts, the partial decryptions of an enrolled set are, in the exponent,
evaluations of a random degree ``t - 1`` polynomial whose constant term is the
FH-MCFE decryption; a combiner can at most compute what the holder of the
FH-MCFE key could, without learning the key itself. Versions <= 0.2.0 shared
a public label-correction term instead.
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEKeyGenerator
from pyfe4ai.schemes.threshold_utils import check_fusion_weight
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.pairing_backend import GT, ZR, pair

logger = logging.getLogger(__name__)


class ThresholdMCFEFHMultiIPEKeyGenerator(MCFEFHMultiIPEKeyGenerator):
    """Key generator: FH-MCFE setup and Shamir-shared functional keys."""

    _scheme_type = CryptoCONST.TYPE_TMCFE_FH_MULTI_IPE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`MCFEFHMultiIPEKeyGenerator`, plus ``s``,
                ``t`` and ``lst_sid``.
        """
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        s = config.get("s") or len(config.get("lst_sid", [])) or 1
        self.lst_sid = config.get("lst_sid", ["sid_{}".format(i) for i in range(s)])
        self.dict_dk = {}
        super().__init__(dict(config, s=s), **kwargs)
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)

    def setup(self) -> None:
        """FH-MCFE setup; forgets previously shared keys."""
        super().setup()
        self.dict_dk = {}

    def get_public_parameters(self) -> dict:
        """FH-MCFE public parameters plus ``s``, ``t`` and ``lst_sid``."""
        pp = super().get_public_parameters()
        pp.update({"s": self.s, "t": self.t, "lst_sid": list(self.lst_sid)})
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of the FH-MCFE key for ``y``.

        All servers asking for the same weights receive shares of one key.

        Args:
            sid: Decryption-server identifier, must be in ``lst_sid``.
            **kwargs: ``credentials`` with ``fusion_weight`` covering every
                client (a ``label`` entry is ignored: keys are label-free).

        Returns:
            Dict with ``k`` (per client the shares of the key exponents),
            ``z``, ``sid``, ``lst_sid`` and ``weights_id``, or ``None`` if
            *sid* is unknown.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for threshold FH-MCFE")
        fusion_weight = credentials.get("fusion_weight")
        self._validate_weights(fusion_weight)
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        weights_id = fusion_weight_id(fusion_weight)
        if weights_id not in self.dict_dk:
            self.dict_dk[weights_id] = self._share_key(fusion_weight)
        dk = dict(self.dict_dk[weights_id][sid])
        dk.update({"sid": sid, "lst_sid": list(self.lst_sid), "weights_id": weights_id})
        return dk

    def _share_key(self, fusion_weight: dict) -> dict:
        ctx = self.ctx
        order = ctx.order
        t = [int(ctx.group.random(ZR)) for _ in self.lst_nid[:-1]]
        t.append((-sum(t)) % order)
        exps, z_total = {}, 0
        for t_i, nid in zip(t, self.lst_nid):
            y_hat = [int(v) for v in fusion_weight[nid]] + [t_i] + [0] * (self.eta + 3)
            e, z_i = ctx.fhipe.keygen_exponents(self.msk[nid], y_hat)
            exps[nid] = e
            z_total = (z_total + z_i) % order
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())

        def share(value) -> dict:
            sh = share_secret(value, self.t, points, order)
            return {sid: str(sh[sid_v[sid]]) for sid in self.lst_sid}

        k_shares = {nid: [share(v) for v in e] for nid, e in exps.items()}
        z_shares = share(z_total)
        return {
            sid: {
                "k": {nid: [sh[sid] for sh in k_shares[nid]] for nid in self.lst_nid},
                "z": z_shares[sid],
            }
            for sid in self.lst_sid
        }


class ThresholdMCFEFHMultiIPE(MCFEFHMultiIPE):
    """Crypto operations for the threshold FH-MCFE."""

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> dict:
        """Compute this server's partial decryption ``W_k`` for an enrolled set.

        Args:
            dict_ct: One ciphertext per client (all under one label).
            credentials: Dict with the ``fusion_weight`` of the key share.
            dk: This server's key share from ``get_decryption_keys``.
            lst_sid_enrolled: The servers whose partials will be combined.

        Returns:
            Dict with ``w``, ``ct_digest``, ``sid`` and ``lst_sid_enrolled``.
        """
        if not dk or "k" not in dk:
            raise FEKeyError("no decryption keys provided")
        if self.id not in dk["lst_sid"]:
            raise FESchemeError("local id:{} is not supported".format(self.id))
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, credentials.get("fusion_weight"))
        if set(dict_ct) != set(dk["k"]):
            raise FESchemeError("need one ciphertext for every client of the key")

        ctx = self.ctx
        sid_v = share_points(dk["lst_sid"])
        lam = int(lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], ctx.order
        ))
        acc = ctx.group.init(GT, 1)
        for nid, ct in dict_ct.items():
            shares = dk["k"][nid]
            if len(ct["ct"]) != len(shares):
                raise FESchemeError("ciphertext of {} has the wrong length".format(nid))
            for c, e in zip(ct["ct"], shares):
                acc *= pair(ctx.de(c), ctx.g2 ** ctx.zr(lam * int(e)))
        acc /= ctx.fhipe.gt ** ctx.zr(lam * int(dk["z"]))
        ct_digest = hashlib.sha256(json.dumps(dict_ct, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "w": ctx.ser(acc),
            "ct_digest": ct_digest,
            "sid": self.id,
            "lst_sid_enrolled": list(lst_sid_enrolled),
        }

    def combine_decrypt(self, dict_ct_prime: dict) -> int | None:
        """Combine at least ``t`` partial decryptions into ``<x, y>``.

        Returns:
            The inner product, or ``None`` if the servers decrypted different
            ciphertexts or the result is out of bound.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given or the shares
                do not match the enrolled set they were computed for.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = list(dict_ct_prime.values())
        if len({p["ct_digest"] for p in partials}) != 1:
            logger.warning("ct consistency verification failed.")
            return None
        acc = self.ctx.group.init(GT, 1)
        for part in partials:
            acc *= self.ctx.de(part["w"])
        return self.ctx.solve(acc)

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> list:
        """Partially decrypt ndarray ciphertexts element-wise."""
        sample = next(iter(dict_ndarray_ct.values()))
        out = []
        for l in range(len(sample)):
            ary = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary[i] = self.share_decrypt(dct_ct, credentials, dk, lst_sid_enrolled)
            out.append(ary)
        return out

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Combine element-wise partial decryptions from at least ``t`` servers."""
        sample = next(iter(dict_ndarray_ct.values()))
        out = []
        for l in range(len(sample)):
            ary = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                ary[i] = self.combine_decrypt({sid: dict_ndarray_ct[sid][l][i] for sid in dict_ndarray_ct})
            out.append((ary / pow(10, self.precision)).astype(float))
        return out
