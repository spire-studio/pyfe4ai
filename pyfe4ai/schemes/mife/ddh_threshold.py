"""
Threshold multi-input inner-product FE (DDH), in the setting of
| R. Xu et al., "TAPFed: Threshold Secure Aggregation for Privacy-Preserving
| Federated Learning", IEEE TDSC, doi: 10.1109/TDSC.2024.3350206.

* type:     private-key multi-input encryption with threshold decryption
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

The MIFE of :mod:`pyfe4ai.schemes.mife.ddh` (ACFGU18 Fig. 4 with the ABDP15
DDH IPFE per slot). Its functional key -- the per-slot ``d_i`` and the pad
term ``z`` -- is Shamir-shared modulo ``q`` among ``s`` decryption servers;
any ``t`` of them decrypt by interpolating in the exponent, as the
threshold layer of :mod:`pyfe4ai.schemes.mcfe.ddh_threshold`:

* server ``k`` with Lagrange coefficient ``lambda_k`` publishes
  ``c_y = prod_{i,j} c_{i,j}^{y_{i,j}}`` (public) and
  ``w_k = prod_i t_i^{lambda_k d_i^(k)} g^{lambda_k z^(k)}``;
* the combiner computes ``c_y / prod_k w_k = g^{<x, y>}`` and a bounded
  discrete log.

Fewer than ``t`` servers learn nothing about the key, and a set of partial
decryptions reveals no more than the MIFE key holder learns (the sum).
Versions <= 0.2.0 used an own IPFE replicated over ``eta`` rows and did not
bind key shares to their weights.
"""

from __future__ import annotations

import hashlib
import json
import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mife.ddh import MIFE, MIFEKeyGenerator
from pyfe4ai.schemes.threshold_utils import check_fusion_weight
from pyfe4ai.schemes.threshold_utils import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold_utils import share_points
from pyfe4ai.schemes.threshold_utils import share_secret
from pyfe4ai.schemes.threshold_utils import validate_enrolled
from pyfe4ai.schemes.threshold_utils import combine_consistently
from pyfe4ai.schemes.threshold_utils import combine_elementwise
from pyfe4ai.schemes.threshold_utils import validate_partial_decryptions
from pyfe4ai.schemes.threshold_utils import validate_threshold
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)


class ThresholdMIFEKeyGenerator(MIFEKeyGenerator):
    """MIFE-DDH setup plus Shamir-shared functional keys."""

    _scheme_type = CryptoCONST.TYPE_TMIFE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`~pyfe4ai.schemes.mife.ddh.MIFEKeyGenerator`,
                plus ``t`` and ``lst_sid`` (``s`` decryption servers).
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

    @property
    def lambd(self) -> int:
        """Alias of ``sec_param`` (name used by versions <= 0.2.0)."""
        return self.sec_param

    def setup(self) -> None:
        """MIFE setup; forgets previously shared keys."""
        super().setup()
        self.dict_dk = {}

    def get_public_parameters(self) -> dict:
        """MIFE public parameters plus ``t`` and ``lst_sid``."""
        pp = super().get_public_parameters()
        pp.update({"t": self.t, "s": self.s, "lst_sid": list(self.lst_sid)})
        return pp

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s Shamir share of the functional key for ``y``.

        All servers asking for the same weights receive shares of one key.

        Returns:
            Dict with ``d`` (per slot), ``z``, ``sid``, ``lst_sid`` and
            ``weights_id``, or ``None`` if *sid* is unknown.
        """
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            return None
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need to provided credentials for tMIFE")
        key = super().get_decryption_keys(sid, credentials=credentials)
        weights_id = key["weights_id"]
        if weights_id not in self.dict_dk:
            self.dict_dk[weights_id] = self._share_key(key)
        dk = dict(self.dict_dk[weights_id][sid])
        dk.update({"sid": sid, "lst_sid": list(self.lst_sid), "weights_id": weights_id})
        return dk

    def _share_key(self, key: dict) -> dict:
        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())

        def share(value) -> dict:
            sh = share_secret(gp.mpz(value), self.t, points, self.q)
            return {sid: gp.digits(sh[sid_v[sid]]) for sid in self.lst_sid}

        d_shares = {nid: share(v) for nid, v in key["d"].items()}
        z_shares = share(key["z"])
        return {
            sid: {"d": {nid: d_shares[nid][sid] for nid in d_shares}, "z": z_shares[sid]}
            for sid in self.lst_sid
        }


class ThresholdMIFE(MIFE):
    """Encryption (with quantisation) and threshold decryption."""

    def _quantise(self, values) -> list:
        return [int(v) for v in scale_to_int(values, self.precision)]

    def encrypt(self, pt: list) -> dict:
        """Encrypt ``eta`` values, quantised with ``10^precision``."""
        return super().encrypt(self._quantise(pt))

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list
    ) -> dict:
        """Compute this server's partial decryption for an enrolled set.

        Args:
            dict_ct: One ciphertext per client.
            credentials: The fusion weights (client ID -> weights) of the key.
            dk: This server's key share.
            lst_sid_enrolled: The servers whose partials will be combined.

        Returns:
            Dict with ``c_y``, ``w``, ``ct_digest``, ``sid`` and
            ``lst_sid_enrolled``.
        """
        if not dk or "d" not in dk:
            raise FEKeyError("no decryption keys provided.")
        if self.id not in dk["lst_sid"] or dk.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        validate_enrolled(self.id, dk["lst_sid"], lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, credentials)
        if not set(dict_ct) == set(credentials) == set(dk["d"]):
            raise FESchemeError("need one ciphertext for every slot of the key")

        p = gp.mpz(self.pp["p"])
        q = gp.mpz(self.pp["q"])
        g = gp.mpz(self.pp["g"])
        sid_v = share_points(dk["lst_sid"])
        lam = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], q
        )
        c_y = gp.mpz(1)
        w = gp.powmod(g, lam * gp.mpz(dk["z"]) % q, p)
        for nid, ct in dict_ct.items():
            y = int_vector(credentials[nid], "fusion weight of " + nid)
            c = [gp.mpz(v) for v in ct["c"]]
            t = gp.mpz(ct["t"])
            if len(c) != len(y) or not all(0 < v < p and gp.legendre(v, p) == 1 for v in c + [t]):
                # outside the order-q subgroup, t^(lambda d) would leak share bits
                raise FEValidationError("malformed ciphertext of {}".format(nid))
            for c_j, y_j in zip(c, y):
                c_y = c_y * gp.powmod(c_j, y_j, p) % p
            w = w * gp.powmod(t, lam * gp.mpz(dk["d"][nid]) % q, p) % p
        ct_digest = hashlib.sha256(json.dumps(dict_ct, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "c_y": gp.digits(c_y),
            "w": gp.digits(w),
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
        """Combine at least ``t`` partial decryptions into ``<x, y> / 10^precision``.

        Returns:
            The result, or ``None`` if the servers decrypted different
            ciphertexts or the result exceeds ``dlog_bound``.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given or the shares
                do not match the enrolled set they were computed for.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = list(dict_ct_prime.values())
        first = partials[0]
        if any((p["c_y"], p["ct_digest"]) != (first["c_y"], first["ct_digest"]) for p in partials):
            logger.warning("partial decryptions disagree on the ciphertexts they decrypted")
            return None
        p = gp.mpz(self.pp["p"])
        w = gp.mpz(1)
        for part in partials:
            w = w * gp.mpz(part["w"]) % p
        f = self._solve_dlog(gp.divm(gp.mpz(first["c_y"]), w, p))
        if f is None:
            return None
        return float(f / pow(10, self.precision))

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list:
        """Encrypt ndarrays element-wise (``eta = 1``); see
        :meth:`MIFE.encrypt_lst_ndarray <pyfe4ai.schemes.mife.ddh.MIFE.encrypt_lst_ndarray>`."""
        if len(self.sk["u"]) != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        out = []
        for ary in lst_ndarray:
            ary_ct = np.empty(np.shape(ary), dtype=object)
            for i, w in np.ndenumerate(np.asarray(ary)):
                ary_ct[i] = self.encrypt([w])
            out.append(ary_ct)
        return out

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
        """Combine element-wise partial decryptions from at least ``t`` servers.

        Raises:
            FESchemeError: If the servers' outputs differ in structure or an
                element does not decrypt (servers disagree, or out of range).
        """
        return combine_elementwise(dict_ndarray_ct, lambda groups, l, i: self.combine_decrypt(groups))
