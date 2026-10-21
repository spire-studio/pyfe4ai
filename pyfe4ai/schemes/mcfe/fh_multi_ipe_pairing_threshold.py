"""
Threshold function-hiding multi-client inner-product FE over pairings.

The AGT21b FH-MCFE of :mod:`pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing`
with its functional key Shamir-shared among ``s`` decryption servers, any
``t`` of which decrypt together. Own construction (experimental, no
published security proof):

* the authority computes the FH-MCFE key for ``y`` in the exponent,
  ``dk_i = [e_i]_2`` and ``z = sum_i z_i`` (with ``[z_i]_T`` the per-client
  correction), Shamir-shares every entry of every ``e_i`` and ``z`` modulo
  the group order and hands server ``k`` its shares in the exponent only:
  ``[e^(k)_{i,j}]_2`` and ``[z^(k)]_T``;
* server ``k`` of an enrolled set ``S`` returns
  ``W_k = (prod_{i,j} e(ct_{i,j}, [e^(k)_{i,j}]_2) / [z^(k)]_T)^lambda_k``
  with its Lagrange coefficient ``lambda_k``;
* ``prod_{k in S} W_k = prod_i iDec(dk_i, ct_i) = [<x, y>]_T``, then a
  bounded discrete log.

Fewer than ``t`` servers learn nothing about the key (Shamir); ``t``
colluding servers interpolate ``[e_i]_2`` and ``[z]_T`` in the exponent, the
FH-MCFE functional key, which hides ``y``. For any ciphertexts, the partial
decryptions of an enrolled set are, in the exponent, evaluations of a random
degree ``t - 1`` polynomial whose constant term is the FH-MCFE decryption; a
combiner can at most compute what the holder of the FH-MCFE key could,
without learning the key itself. Earlier versions handed out the shares of
``e_i`` in ``Z_p``, from which ``t`` colluding servers solved for ``y``;
versions <= 0.2.0 shared a public label-correction term instead.
"""

from __future__ import annotations

import logging
import secrets

import numpy as np

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairing
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairingKeyGenerator
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.threshold.binding import check_ciphertext_tags
from pyfe4ai.schemes.threshold.binding import client_server_mac_keys
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.schemes.threshold.binding import tag_ciphertext
from pyfe4ai.schemes.threshold.codec import ciphertexts_digest
from pyfe4ai.schemes.threshold.roles import combine_elementwise
from pyfe4ai.schemes.threshold.roles import validate_enrolled
from pyfe4ai.schemes.threshold.roles import combine_consistently
from pyfe4ai.schemes.threshold.roles import DISAGREEING_PARTIALS
from pyfe4ai.schemes.threshold.roles import partial_groups
from pyfe4ai.schemes.threshold.roles import validate_partial_decryptions
from pyfe4ai.schemes.threshold.sharing import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold.sharing import share_points
from pyfe4ai.schemes.threshold.sharing import share_secret
from pyfe4ai.schemes.threshold.sharing import validate_threshold
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.formats.pairing import FHCiphertext
from pyfe4ai.core.formats.pairing import FHClientKey
from pyfe4ai.core.formats.pairing import TMCFEFHParameters
from pyfe4ai.core.formats.threshold import TMCFEFHKeyShare
from pyfe4ai.core.formats.threshold import TMCFEFHPartial
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.primitives.pairing.pairing_backend import GT, ZR, multi_pair, pow_many

logger = logging.getLogger(__name__)

_TAG_DOMAIN = "pyfe4ai/tmcfe-fh-multi-ipe/ciphertext-tag/v1"


def _tag_payload(nid, ct: dict) -> list:
    """What a tMCFE-FH ciphertext tag covers: the encoded label and the ciphertext."""
    ct = FHCiphertext(ct)
    return [ct.get("label"), ct.get("ct")]


class ThresholdMCFEFHMultiIPEPairingKeyGenerator(MCFEFHMultiIPEPairingKeyGenerator):
    """Key generator: FH-MCFE setup and Shamir-shared functional keys."""

    _scheme_type = CryptoCONST.TYPE_TMCFE_FH_MULTI_IPE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for
                :class:`~pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing.MCFEFHMultiIPEPairingKeyGenerator`,
                plus ``s``, ``t`` and ``lst_sid``.
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
        """FH-MCFE setup and the client-server MAC keys; forgets previously shared keys."""
        super().setup()
        self.dict_dk = {}
        self.mac_keys = client_server_mac_keys(self.lst_nid, self.lst_sid)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Client *nid*'s ``imsk`` plus ``mac_keys`` (``{sid: key}``) for the
        ciphertext tags."""
        sk = super().get_private_keys(nid, **kwargs)
        if sk is not None:
            FHClientKey(sk).mac_keys = dict(self.mac_keys[nid])
        return sk

    def get_public_parameters(self) -> dict:
        """FH-MCFE public parameters plus ``s``, ``t`` and ``lst_sid``."""
        pp = TMCFEFHParameters(super().get_public_parameters())
        pp.update(s=self.s, t=self.t, lst_sid=list(self.lst_sid))
        return pp.to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s share of the FH-MCFE key for ``y``.

        All servers asking for the same weights receive shares of one key.

        Args:
            sid: Decryption-server identifier, must be in ``lst_sid``.
            **kwargs: ``credentials`` with ``fusion_weight`` covering every
                client (a ``label`` entry is ignored: keys are label-free).

        Returns:
            Dict with ``k`` (per client the shares of the key exponents as
            ``G2`` elements), ``z`` (the share of ``z`` as a ``GT`` element),
            ``sid``, ``lst_sid``, a random ``key_id`` (nothing derived from
            ``y``) and ``mac_keys`` (``{nid: key}``), or ``None`` if *sid* is
            unknown.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for threshold FH-MCFE")
        fusion_weight = credentials.get("fusion_weight")
        self._validate_weights(fusion_weight)
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            warn_at_caller("invalid sid (not one of the key generator's lst_sid)")
            return None
        weights_id = fusion_weight_id(fusion_weight)
        entry = self.dict_dk.get(weights_id)
        if entry is None:
            # one sharing per weights: of two threads dealing at once, both issue the one kept first
            entry = self.dict_dk.setdefault(weights_id, self._share_key(fusion_weight))
        dk = TMCFEFHKeyShare.extend(entry[sid], mac_keys={nid: self.mac_keys[nid][sid] for nid in self.lst_nid})
        return dk.to_dict()

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

        def share(values: list, base) -> list:
            # in the exponent only: shares in Z_p would let t servers solve for y
            shares = [share_secret(value, self.t, points, order) for value in values]
            powers = iter(pow_many(base, [sh[sid_v[sid]] for sh in shares for sid in self.lst_sid], ctx.group))
            return [{sid: ctx.ser(next(powers)) for sid in self.lst_sid} for _ in shares]

        flat = share([v for e in exps.values() for v in e], ctx.g2)
        k_shares, pos = {}, 0
        for nid, e in exps.items():
            k_shares[nid] = flat[pos:pos + len(e)]
            pos += len(e)
        z_shares = share([z_total], ctx.fhipe.gt)[0]
        key_id = secrets.token_hex(16)
        return {
            sid: TMCFEFHKeyShare.new(
                k={nid: [sh[sid] for sh in k_shares[nid]] for nid in self.lst_nid},
                z=z_shares[sid],
                sid=sid,
                lst_sid=list(self.lst_sid),
                key_id=key_id,
            ).to_dict()
            for sid in self.lst_sid
        }


class ThresholdMCFEFHMultiIPEPairing(MCFEFHMultiIPEPairing):
    """Crypto operations for the threshold FH-MCFE."""

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """FH-MCFE encryption under *label*; the ciphertext carries the encoded
        label and a tag for every server over both."""
        mac_keys = FHClientKey(self.sk or {}).get("mac_keys") if self._has_private_keys() else None
        if not mac_keys:
            raise FEKeyError("encryption key lacks the ciphertext MAC keys")
        ct = super().encrypt(lst_pt, label)
        FHCiphertext(ct).label = encode_label(label)
        return tag_ciphertext(ct, mac_keys, _TAG_DOMAIN, _tag_payload(None, ct))

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list, label
    ) -> dict:
        """Compute this server's partial decryption ``W_k`` for an enrolled set.

        Args:
            dict_ct: One ciphertext per client (all under one label).
            credentials: Ignored (the key share encodes ``y``; servers need
                not know the function). Kept for API compatibility.
            dk: This server's key share from ``get_decryption_keys``.
            lst_sid_enrolled: The servers whose partials will be combined.
            label: The label (round) the server agrees to decrypt. The label
                is inside the FH ciphertexts, so the server checks the clients'
                tags over the declared label and the ciphertext.

        Returns:
            Dict with ``w``, ``ct_digest``, ``key_id``, ``label``, ``sid`` and
            ``lst_sid_enrolled``.

        Raises:
            FESchemeError: If a ciphertext is not under *label* or not tagged
                for this server by its client.
        """
        if not dk or "k" not in TMCFEFHKeyShare(dk):
            raise FEKeyError("no decryption keys provided")
        share = TMCFEFHKeyShare.from_dict(dk)
        if self.id not in share.lst_sid or share.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        validate_enrolled(self.id, share.lst_sid, lst_sid_enrolled, self.pp["t"])
        if set(dict_ct) != set(share.k):
            raise FESchemeError("need one ciphertext for every client of the key")
        expected = encode_label(label)
        if any(not isinstance(ct, dict) or FHCiphertext(ct).get("label") != expected for ct in dict_ct.values()):
            raise FESchemeError("the ciphertexts were not encrypted under the requested label")
        check_ciphertext_tags(dict_ct, share.get("mac_keys"), self.id, _TAG_DOMAIN, _tag_payload)

        ctx = self.ctx
        sid_v = share_points(share.lst_sid)
        lam = int(lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], ctx.order
        ))
        lhs, rhs = [], []
        for nid, ct in dict_ct.items():
            shares = share.k[nid]
            if len(FHCiphertext(ct).ct) != len(shares):
                raise FESchemeError("ciphertext of {} has the wrong length".format(nid))
            lhs.extend(ctx.de_many(FHCiphertext(ct).ct))
            rhs.extend(ctx.de_many(shares))
        acc = (multi_pair(lhs, rhs) / ctx.de(share.z)) ** ctx.zr(lam)
        ct_digest = ciphertexts_digest({nid: _tag_payload(nid, ct) for nid, ct in dict_ct.items()})
        return TMCFEFHPartial.new(
            w=ctx.ser(acc),
            ct_digest=ct_digest,
            key_id=share.key_id,
            label=expected,
            sid=self.id,
            lst_sid_enrolled=list(lst_sid_enrolled),
        ).to_dict()

    def combine_decrypt(self, partials, label=None) -> int | None:
        """Combine partial decryptions from at least ``t`` servers.

        Args:
            partials: ``{sid: share_decrypt(...)}`` for one enrolled set, or a
                list of such mappings for several enrolled sets, whose results
                must then agree (see
                :func:`~pyfe4ai.schemes.threshold.roles.combine_consistently`:
                servers are assumed honest-but-curious, and only this
                redundancy detects one that returns a wrong partial).
            label: Optional label the partials must be for.

        Raises:
            FESchemeError: If a group is incomplete or invalid, is for another
                label or its partial decryptions disagree on the ciphertexts
                they decrypted, or the groups disagree.
        """
        if label is not None:
            expected = encode_label(label)
            for group in partial_groups(partials):
                if any(TMCFEFHPartial(part).get("label") != expected for part in group.values()):
                    raise FESchemeError("partial decryptions are for another label than the requested one")
        return combine_consistently(partials, self._combine_group)

    def _combine_group(self, dict_ct_prime: dict) -> int | None:
        """Combine at least ``t`` partial decryptions into ``<x, y>``.

        Returns:
            The inner product, or ``None`` if the result is out of bound.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given, the shares
                do not match the enrolled set they were computed for, or the
                servers decrypted different ciphertexts (or with keys of
                different sharings, ``key_id``).
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        partials = [TMCFEFHPartial(part) for part in dict_ct_prime.values()]
        if len({(p.ct_digest, p.key_id, p.get("label")) for p in partials}) != 1:
            raise FESchemeError(DISAGREEING_PARTIALS)
        acc = self.ctx.group.init(GT, 1)
        for part in partials:
            acc *= self.ctx.de(part.w)
        return self.ctx.solve(acc)

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list,
        label: str,
    ) -> list:
        """Partially decrypt ndarray ciphertexts element-wise; element ``(l, i)``
        must be under its element label derived from *label*."""
        sample = next(iter(dict_ndarray_ct.values()))
        out = []
        for l in range(len(sample)):
            ary = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary[i] = self.share_decrypt(
                    dct_ct, credentials, dk, lst_sid_enrolled, _element_label(label, l, i)
                )
            out.append(ary)
        return out

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct, label: str) -> list:
        """Combine element-wise partial decryptions from at least ``t`` servers.

        Raises:
            FESchemeError: If the servers' outputs differ in structure, are for
                other labels, or an element does not decrypt.
        """
        out = combine_elementwise(
            dict_ndarray_ct,
            lambda groups, l, i: self.combine_decrypt(groups, _element_label(label, l, i)),
        )
        return [(ary / pow(10, self.precision)).astype(float) for ary in out]
