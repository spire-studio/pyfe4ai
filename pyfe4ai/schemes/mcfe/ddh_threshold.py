"""
Threshold multi-client inner-product FE over DDH: CDGPP18 MCFE with a
Shamir-shared functional key.

| MCFE layer: Chotard, Dufour Sans, Gay, Phan, Pointcheval,
| "Decentralized multi-client functional encryption for inner product",
| ASIACRYPT 2018, https://eprint.iacr.org/2017/989 (MCFE from DDH in the ROM;
| same construction as :mod:`pyfe4ai.schemes.mcfe.ddh`).
| Threshold layer in the setting of
| R. Xu et al., "TAPFed: Threshold Secure Aggregation for Privacy-Preserving
| Federated Learning", IEEE TDSC, doi: 10.1109/TDSC.2024.3350206: the
| aggregated functional key is Shamir-shared among ``s`` decryption servers
| and any ``t`` of them decrypt by Lagrange interpolation in the exponent.

* type:     secret-key multi-client encryption, threshold functional decryption
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1), random oracle

Construction (one coordinate j of client i's vector is one CDGPP18 slot):

* Setup:     ``s_{i,j} <- Z_q^2`` for every client slot.
* Encrypt:   ``[u_l] = H(l) in G^2``;  ``c_{i,j} = g^{x_{i,j}} * [u_l]^{s_{i,j}}``.
* KeyGen:    ``d = sum y_{i,j} s_{i,j} in Z_q^2``; each component of ``d`` is
  Shamir-shared with threshold ``t`` over ``Z_q`` at points
  ``1..s``; server ``k`` receives ``d^(k) in Z_q^2``.
* ShareDec:  server ``k`` of an enrolled set ``E`` (``|E| >= t``) publishes
  ``P_k = [u_l]^{lambda_k * d^(k)}`` with ``lambda_k`` its Lagrange
  coefficient at 0 for ``E`` (mod q), and ``C_y = prod c_{i,j}^{y_{i,j}}``.
* Combine:   ``[u_l]^d = prod_{k in E} P_k``;
  ``g^{<x,y>} = C_y / [u_l]^d``, then a bounded dlog.

The functional key is a single aggregated value, so neither a server nor the
combiner can decrypt any individual client; fewer than ``t`` shares of ``d``
are independent of ``d``. Each client must encrypt at most once per label:
two ciphertexts of the same client slot under the same label reveal the
difference of the plaintexts. The ciphertext carries its (public) label so
that the servers can evaluate ``H(l)``.

Deviation from earlier versions (<= 0.2.0): the label used to enter as a
public MD5 scalar and the key carried one share per client, which let the
combiner strip each client's mask; see CHANGELOG 0.3.0.
"""

from __future__ import annotations

import os
import json
import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.common.ddh_base import DDHKeyGeneratorBase
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.threshold.binding import check_fusion_weight
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.schemes.threshold.codec import is_decimal
from pyfe4ai.schemes.threshold.roles import validate_enrolled
from pyfe4ai.schemes.threshold.roles import combine_consistently
from pyfe4ai.schemes.threshold.roles import DISAGREEING_PARTIALS
from pyfe4ai.schemes.threshold.roles import partial_groups
from pyfe4ai.schemes.threshold.roles import combine_elementwise
from pyfe4ai.schemes.threshold.roles import validate_partial_decryptions
from pyfe4ai.schemes.threshold.sharing import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold.sharing import share_points
from pyfe4ai.schemes.threshold.sharing import share_secret
from pyfe4ai.schemes.threshold.sharing import validate_threshold
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.groups.integer_groups import _random
from pyfe4ai.primitives.hashing import hash_to_qr_group
from pyfe4ai.primitives.groups.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.core.formats.dl import MCFEDDHCiphertext
from pyfe4ai.core.formats.dl import MCFEDDHEncryptionKey
from pyfe4ai.core.formats.dl import TMCFEDDHParameters
from pyfe4ai.core.formats.threshold import TMCFEDDHKeyShare
from pyfe4ai.core.formats.threshold import TMCFEDDHPartial
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.core.ids import check_id_list, is_id
from pyfe4ai.core.versions import TMCFE_DDH_RECORD
from pyfe4ai._validation import scale_to_int

logger = logging.getLogger(__name__)

_LABEL_HASH_DOMAIN = "pyfe4ai/mcfe/ddh_threshold/H(label)"


def _hash_encoded(encoded: str, p: gp.mpz) -> list:
    """Return ``[u_l] = H(l) in G^2`` for an encoded label (see :func:`encode_label`)."""
    return [hash_to_qr_group(encoded, p, index=k, domain=_LABEL_HASH_DOMAIN) for k in (0, 1)]


def _fusion_weight_id(fusion_weight: dict) -> str:
    """Canonical identifier of a fusion-weight dict (one sharing per weight set)."""
    return json.dumps(
        {nid: [int(y) for y in lst] for nid, lst in fusion_weight.items()},
        sort_keys=True,
    )


class ThresholdMCFEDDHKeyGenerator(DDHKeyGeneratorBase):
    """Key generator for threshold DDH-based multi-client inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_TMCFE
    _verify_keys = ("sec_param", "eta", "n", "s", "t", "construction")

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the threshold MCFE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta`` (slots per client), ``n`` (number of clients),
                ``s`` (number of decryption servers), ``t`` (threshold),
                ``lst_nid`` and ``lst_sid``.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.tMCFE_ETA)
        self.t = config.get("t", CryptoCONST.tMCFE_T)
        if not isinstance(self.eta, int) or self.eta < 1:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        check_id_list(config.get("lst_sid", []), "lst_sid")  # before len(lst_sid) could raise TypeError
        if self.s is None:
            self.s = len(config.get("lst_sid", [])) or CryptoCONST.tMCFE_S
        self.lst_sid = config.get(
            "lst_sid", ["sid_{}".format(i) for i in range(self.s)]
        )
        if len(self.lst_sid) != self.s:
            raise FEValidationError(
                "lst_sid has {} servers but s={}".format(len(self.lst_sid), self.s)
            )
        validate_threshold(self.t, self.lst_sid)
        self.construction = TMCFE_DDH_RECORD
        self._load_parameters()
        self.dict_dk = {}

    def _extra_param_fields(self) -> dict:
        return {"n": self.n, "s": self.s, "t": self.t, "construction": self.construction}

    def setup(self) -> None:
        """Generate one independent secret ``s_{i,j} in Z_q^2`` per client slot."""
        self.msk = {
            "s": {
                nid: [
                    [_random(self.q, self.sec_param) for _ in range(2)]
                    for _ in range(self.eta)
                ]
                for nid in self.lst_nid
            }
        }
        self.dict_dk = {}
        self.pp = TMCFEDDHParameters.new(
            p=gp.digits(self.p),
            g=gp.digits(self.g),
            eta=self.eta,
            s=self.s,
            t=self.t,
            n=self.n,
            sec_param=self.sec_param,
        ).to_dict()
        logger.info("Threshold MCFE setup - DONE.")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters (p, g, eta, n, s, t, sec_param)."""
        return dict(self.pp)

    def get_private_keys(self, nid: str) -> dict | None:
        """Return the encryption key of client *nid*.

        Args:
            nid: Client identifier, must be in ``lst_nid``.

        Returns:
            Dict with ``s`` (one ``Z_q^2`` pair per slot), or ``None`` if *nid*
            is invalid.
        """
        if nid is None or not is_id(nid) or nid not in self.lst_nid:
            logger.error("no id or invalid id provided.")
            warn_at_caller("no id or invalid id provided.")
            return None
        return MCFEDDHEncryptionKey.new(s=[[gp.digits(e) for e in s_j] for s_j in self.msk["s"][nid]]).to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Return server *sid*'s Shamir share of the functional key for ``y``.

        All servers asking for the same fusion weights receive shares of one
        and the same sharing. The key does not depend on the label; a
        ``label`` entry in the credentials is accepted and ignored.

        Args:
            sid: Decryption-server identifier, must be in ``lst_sid``.
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to integer weight lists).

        Returns:
            Dict with ``d`` (this server's share, a pair mod q), ``sid``,
            ``lst_sid`` and ``weights_id`` (digest of the weights), or
            ``None`` if *sid* is unknown.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If the fusion weights are malformed.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need to provided credentials for tMCFE")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or not fusion_weight:
            raise FEValidationError("invalid fusion weights provided, need a dict")
        if sid not in self.lst_sid:
            logger.error("invalid sid: {}".format(sid))
            warn_at_caller("invalid sid (not one of the key generator's lst_sid)")
            return None

        identifier = _fusion_weight_id(fusion_weight)
        entry = self.dict_dk.get(identifier)
        if entry is None:
            # one sharing per weights: of two threads dealing at once, both issue the one kept first
            entry = self.dict_dk.setdefault(identifier, self._share_functional_key(fusion_weight))
        d_sid = entry[sid]
        return TMCFEDDHKeyShare.new(
            d=[gp.digits(e) for e in d_sid],
            sid=sid,
            lst_sid=list(self.lst_sid),
            weights_id=fusion_weight_id(fusion_weight),
        ).to_dict()

    def _share_functional_key(self, fusion_weight: dict) -> dict:
        """Compute ``d = sum y_{i,j} s_{i,j}`` and Shamir-share it among servers."""
        d = [gp.mpz(0), gp.mpz(0)]
        for nid, lst_y in fusion_weight.items():
            if nid not in self.msk["s"]:
                raise FEValidationError(
                    "invalid identifier in provided fusion weight: {}".format(nid)
                )
            if not isinstance(lst_y, list) or len(lst_y) > self.eta:
                raise FEValidationError(
                    "fusion weight of {} must be a list of at most eta={} "
                    "integers".format(nid, self.eta)
                )
            for y_j, s_j in zip(lst_y, self.msk["s"][nid]):
                if isinstance(y_j, bool) or not isinstance(y_j, (int, np.integer)):
                    raise FEValidationError(
                        "fusion weight of {} must contain integers".format(nid)
                    )
                d[0] += gp.mul(gp.mpz(int(y_j)), s_j[0])
                d[1] += gp.mul(gp.mpz(int(y_j)), s_j[1])

        sid_v = share_points(self.lst_sid)
        points = list(sid_v.values())
        shares = [share_secret(d_k % self.q, self.t, points, self.q) for d_k in d]
        return {sid: [shares[0][sid_v[sid]], shares[1][sid_v[sid]]] for sid in self.lst_sid}


class ThresholdMCFEDDH(IPFEAbsCrypto):
    """Crypto operations for threshold DDH-based multi-client inner-product FE."""

    def __init__(self, config: dict) -> None:
        """Initialise a client (``keys.sk``), server (``keys.dk``) or combiner.

        Args:
            config: Must contain ``keys`` with ``pp`` and optionally ``sk`` or
                ``dk``; ``id`` names the client / server; ``precision`` is the
                number of decimal digits kept when encrypting floats;
                ``dlog_bound`` (default ``10^(precision + 2)``) bounds
                ``|<x, y>|`` (after scaling) for the final discrete log.
        """
        super().__init__(config)
        self.dlog_bound = int(config.get("dlog_bound", pow(10, self.precision + 2)))
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.dlog_table = None

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_TMCFE,
            f"dlog_{self.precision}.json",
        )
        self.dlog_table, self.bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(
                _dlog_file, self.pp["g"], self.pp["p"], self.dlog_bound,
            )
        )

    def encrypt(self, pt: list, label: str) -> dict:
        """Encrypt one client's vector under a label.

        Values are scaled by ``10^precision`` and rounded to integers.

        Args:
            pt: Plaintext vector of length at most ``eta``.
            label: Encryption label; a client must not encrypt twice under
                the same label.

        Returns:
            Dict with ``c`` (one group element per slot) and ``label`` (the
            encoded label, see :func:`pyfe4ai.schemes.mcfe.labels.encode_label`).

        Raises:
            FEKeyError: If public parameters or private keys are missing.
            FEValidationError: If the plaintext or label is malformed.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if not isinstance(pt, list):
            raise FEValidationError("input plaintext must be a list")
        if len(pt) > len(MCFEDDHEncryptionKey(self.sk).s):
            raise FEValidationError("input plaintext has {} values, more than eta".format(len(pt)))

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        encoded = encode_label(label)
        u_l = _hash_encoded(encoded, p)

        c = []
        for v, s_j in zip(scale_to_int(pt, self.precision), MCFEDDHEncryptionKey(self.sk).s):
            x_j = gp.mpz(int(v))
            c_j = gp.powmod(g, x_j, p)
            for u_k, s_jk in zip(u_l, s_j):
                c_j = gp.mul(c_j, gp.powmod(u_k, gp.mpz(s_jk), p)) % p
            c.append(gp.digits(c_j))
        return MCFEDDHCiphertext.new(c=c, label=encoded).to_dict()

    def share_decrypt(
        self, dict_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list, label
    ) -> dict:
        """Compute this server's partial decryption for an enrolled set.

        Args:
            dict_ct: Mapping of client IDs to ciphertexts (all under one label).
            credentials: Mapping of client IDs to the fusion weights the key
                share was derived for.
            dk: This server's key share from ``get_decryption_keys``.
            lst_sid_enrolled: The servers whose partial decryptions will be
                combined (at least ``t``, including this one).
            label: The label the caller wants decrypted; ciphertexts under any
                other label (e.g. those of an earlier round) are refused.

        Returns:
            Dict with ``c_y`` (``prod c^y``), ``u_d`` (``[u_l]^{lambda d^(k)}``),
            ``label``, ``sid`` and ``lst_sid_enrolled``.

        Raises:
            FEKeyError: If the key share is missing or malformed.
            FESchemeError: If the server or enrolled set is invalid, or the
                ciphertexts do not match the fusion weights / a single label.
        """
        if not dk:
            raise FEKeyError("no decryption keys provided.")
        share = TMCFEDDHKeyShare.from_dict(dk)
        if not isinstance(share.get("d"), (list, tuple)) or len(share.d) != 2:
            raise FEKeyError("malformed tMCFE key share: expected d = [d0, d1]")
        if not all(map(is_decimal, share.d)):
            raise FEKeyError("malformed tMCFE key share: d holds decimal strings")
        p = gp.mpz(self.pp["p"])
        q = (p - 1) // 2
        lst_sid = share.lst_sid
        if self.id not in lst_sid or share.get("sid") != self.id:
            raise FESchemeError("the key share does not belong to server {}".format(self.id))
        validate_enrolled(self.id, lst_sid, lst_sid_enrolled, self.pp["t"])
        check_fusion_weight(dk, credentials)

        expected = encode_label(label)
        label = self._common_label(dict_ct)
        if label != expected:
            raise FESchemeError("the ciphertexts were not encrypted under the requested label")
        c_y = self._weighted_product(dict_ct, credentials, p)

        sid_v = share_points(lst_sid)
        lagrange = lagrange_coefficient_at_zero(
            sid_v[self.id], [sid_v[s] for s in lst_sid_enrolled], q
        )
        u_d = gp.mpz(1)
        for u_k, d_k in zip(_hash_encoded(label, p), share.d):
            u_d = gp.mul(u_d, gp.powmod(u_k, gp.mul(gp.mpz(d_k), lagrange) % q, p)) % p

        return TMCFEDDHPartial.new(
            c_y=gp.digits(c_y),
            u_d=gp.digits(u_d),
            label=label,
            sid=self.id,
            lst_sid_enrolled=list(lst_sid_enrolled),
        ).to_dict()

    def combine_decrypt(self, partials, label=None) -> float | None:
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
                if any(TMCFEDDHPartial(part).get("label") != expected for part in group.values()):
                    raise FESchemeError("partial decryptions are for another label than the requested one")
        return combine_consistently(partials, self._combine_group)

    def _combine_group(self, dict_ct_prime: dict) -> float | None:
        """Combine at least ``t`` partial decryptions into the inner product.

        Args:
            dict_ct_prime: Mapping of server ids to :meth:`share_decrypt` outputs.

        Returns:
            ``<x, y>`` divided by ``10^precision``, or ``None`` if
            ``|<x, y>|`` exceeds ``dlog_bound``.

        Raises:
            FESchemeError: If fewer than ``t`` shares are given, the shares
                do not match the enrolled set they were computed for, or the
                servers disagree on the ciphertexts.
        """
        validate_partial_decryptions(dict_ct_prime, self.pp["t"])
        p = gp.mpz(self.pp["p"])

        partials = [TMCFEDDHPartial(part) for part in dict_ct_prime.values()]
        c_y = gp.mpz(partials[0].c_y)
        label = partials[0].get("label")
        if any(
            gp.mpz(part.c_y) != c_y or part.get("label") != label
            for part in partials
        ):
            raise FESchemeError(DISAGREEING_PARTIALS)

        u_d = gp.mpz(1)
        for part in partials:
            u_d = gp.mul(u_d, gp.mpz(part.u_d)) % p

        f = self._solve_dlog(gp.divm(c_y, u_d, p))
        if f is None:
            return None
        return float(f / pow(10, self.precision))

    @staticmethod
    def _common_label(dict_ct: dict) -> str:
        labels = {MCFEDDHCiphertext(ct).get("label") for ct in dict_ct.values()}
        if len(labels) != 1:
            raise FESchemeError(
                "ciphertexts must all carry the same label, got {}".format(labels)
            )
        label = labels.pop()
        if not isinstance(label, str):
            raise FESchemeError("ciphertexts carry no label")
        return label

    @staticmethod
    def _weighted_product(dict_ct: dict, fusion_weight: dict, p: gp.mpz) -> gp.mpz:
        """Return ``prod_{i,j} c_{i,j}^{y_{i,j}} mod p``."""
        if dict_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")
        c_y = gp.mpz(1)
        for nid, ct_nid in dict_ct.items():
            c_nid = MCFEDDHCiphertext(ct_nid).c
            y_nid = fusion_weight[nid]
            if len(c_nid) != len(y_nid):
                raise FESchemeError(
                    "ciphertext and fusion weight of {} differ in length".format(nid)
                )
            for c_j, y_j in zip(c_nid, y_nid):
                c_j = gp.mpz(c_j)
                if not 0 < c_j < p:
                    raise FEValidationError(
                        "ciphertext element of {} out of range".format(nid)
                    )
                c_y = gp.mul(c_y, gp.powmod(c_j, gp.mpz(int(y_j)), p)) % p
        return c_y

    def _solve_dlog(self, g_inner_prod) -> int | None:
        if self.dlog_table is None:
            self._load_dlog_table()
        try:
            return dlog_table_solve(
                g_inner_prod, self.pp["g"], self.pp["p"],
                self.bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system.")
            warn_at_caller("inner-product is out of bound supported by crypto system.")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list:
        """Encrypt a list of ndarrays element-wise (``eta = 1`` slots).

        Every element is encrypted under its own label derived from
        ``label``, the layer and the element index, because a client must
        not encrypt twice under one label.

        Args:
            lst_ndarray: List of numpy arrays.
            **kwargs: Must include ``label``.
        """
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        lst_ndarray_ct = []
        for l in range(len(lst_ndarray)):
            ary = lst_ndarray[l]
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([w], _element_label(label, l, i))
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(
        self, dict_ndarray_ct: dict, credentials: dict, dk: dict, lst_sid_enrolled: list, label: str
    ) -> list:
        """Partially decrypt every element of clients' ndarray ciphertexts.

        Element ``(l, i)`` must be encrypted under its element label derived
        from *label*, so ciphertexts of another round or position are refused.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            credentials: Mapping of client IDs to fusion weights.
            dk: This server's key share.
            lst_sid_enrolled: The servers whose shares will be combined.
            label: The label the ndarrays were encrypted under.
        """
        sample = next(iter(dict_ndarray_ct.values()))
        lst_partial = []
        for l in range(len(sample)):
            ary_partial = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_partial[i] = self.share_decrypt(
                    dct_ct, credentials, dk, lst_sid_enrolled, _element_label(label, l, i)
                )
            lst_partial.append(ary_partial)
        return lst_partial

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct, label: str) -> list:
        """Combine element-wise partial decryptions from at least ``t`` servers.

        Args:
            dict_ndarray_ct: Mapping of server ids to the outputs of
                :meth:`compute_lst_ndarray_ct` (or a list of such mappings for
                several enrolled sets, see :meth:`combine_decrypt`).
            label: The label of the ndarrays; every partial must be for the
                element label of its position.

        Raises:
            FESchemeError: If the servers' outputs differ in structure, an
                element does not decrypt (out of range), or the partial
                decryptions of an element disagree on the ciphertexts (the
                error and message of ``combine_decrypt``).
        """
        return combine_elementwise(
            dict_ndarray_ct,
            lambda groups, l, i: self.combine_decrypt(groups, _element_label(label, l, i)),
        )

    @staticmethod
    def L(sid, sid_v: dict, lst_sid_enrolled: list, modulus) -> gp.mpz:
        """Compute the Lagrange coefficient of *sid* at 0 modulo the group order.

        Args:
            sid: Decryption-server identifier.
            sid_v: Mapping of server ids to their (1-based) share points.
            lst_sid_enrolled: List of enrolled server identifiers.
            modulus: Group order ``q``.
        """
        return lagrange_coefficient_at_zero(
            sid_v[sid], [sid_v[s] for s in lst_sid_enrolled], modulus
        )
