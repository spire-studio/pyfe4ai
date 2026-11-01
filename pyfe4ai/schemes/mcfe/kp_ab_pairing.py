"""
Multi-Client Key-Policy Attribute-Based Inner-Product Functional Encryption over Pairings
| "Multi-Client Functional Encryption with Fine-Grained Access Control"
| By Ky Nguyen, Duong Hieu Phan and David Pointcheval, ASIACRYPT 2022
| (ePrint 2022/215) -- Sec. 5.2 Fig. 3, with the monotone span programs of :mod:`pyfe4ai.policy`.

* type:     multi-client encryption of one scalar per client under a tag (a label) and a set of attribute labels
  (:mod:`pyfe4ai.policy`); a functional key for a weight vector ``y`` and a policy opens ``<x, y>`` when the labels
  satisfy the policy
* setting:  asymmetric pairing group (default BLS12-381), random oracle (the hash of the tag to ``G1``)

``(F, F*)`` are dual bases of ``G1^8`` and ``G2^8``, shared by all clients, and each client ``i`` has its own dual
bases ``(H_i, H_i*)`` of ``G1^4`` and ``G2^4`` (:mod:`pyfe4ai.primitives.pairing.dpvs`); all have ``mu = 1``, and the
``mu`` that the text of Fig. 3 samples is not used. ``S, U`` are in ``(Z_q^*)^n`` and ``p_1 + ... + p_n = 1``. A label
hashes to the ``j`` in ``Z_q`` that stands for it (:func:`~pyfe4ai.policy.attributes.attribute_to_zq`). The master key
is ``msk = (S, U, f1*, f2*, f3*, (h*_i,1, h*_i,2, h*_i,3)_i)``, the last as the elements of ``G2^8`` and ``G2^4`` that
the paper has, and the exponents of the bases are dropped at the end of setup. The encryption key of client ``i`` is
``ek_i = (s_i, u_i, p_i H_i^(1), p_i H_i^(2), h_i,3, f1, f2, f3)``, with the two first rows of ``H_i`` as vectors of
``Z_q`` and the others as elements of ``G1``. The public parameters hold the generators and the configuration only.

* KeyGen(y, P):  for the span program ``M`` of ``P`` (rows ``j`` with the labels of the leaves), a random ``a0``,
  the shares ``a_j = (M (a0, r))_j``, a random ``z`` and, for each client ``i`` and each row, a random ``pi_ij``:
  ``k*_ij = (pi_ij (j, 1), a_j z, 0, ..., 0)_F*`` and ``k*_i,ipfe = (<S, y>, <U, y>, a0 z, 0)_H*_i``. The key is
  ``P``, ``y``, the ``k*_ij`` and ``k*_i,ipfe``.
* Enc(ek_i, x_i, tag, S):  for a set of labels ``S``, the points ``[omega]_1, [omega']_1 = H(tag, S)``, a random
  ``psi_i`` and a random ``sigma_ij`` for each label ``j`` of ``S``: ``c_ij = (sigma_ij (1, -j), psi_i, 0, ..., 0)_F``,
  ``t_i = [omega s_i + omega' u_i + x_i]_1`` and
  ``c_i,ipfe = p_i (omega h_i,1 + omega' h_i,2) + psi_i h_i,3 = (omega p_i, omega' p_i, psi_i, 0)_H_i``.
* Dec:  for ``S`` that satisfies ``P`` and the reconstruction coefficients ``w_j`` of ``M`` for the rows of ``S``:
  ``sum_i (sum_j w_j e(c_ij, k*_ij) - e(c_i,ipfe, k*_i,ipfe) + e(t_i, [y_i]_2)) = [<x, y>]_T``, since
  ``(1, -j) . (j, 1) = 0``, ``sum_j w_j a_j = a0`` and ``sum_i p_i = 1``, and a bounded discrete log gives the
  result. It takes one ``multi_pair`` of ``n (8 |A| + 4) + 1`` pairs for the ``|A|`` rows that the coefficients use.

NPP22 Thm. 14 and Cor. 15 prove this construction adaptively IND-secure from SXDH in the random-oracle model, with
the notion of Def. 12: no client encrypts twice under one tag, and the challenge ciphertexts are those of all honest
clients. The hash of this module is RFC 9380 ``hash_to_curve`` under a domain separation tag, taken as the random
oracle. This implementation is unreviewed: it has had no cryptographic review yet.

A client encrypts once under a tag. Two ciphertexts of one client under one tag have the same ``omega, omega'`` (the
same ``H(tag, S)``) and their ``t`` values differ by the difference of the plaintexts, in the exponent. The scheme
layer does not check this, as with the other labelled multi-client schemes (:mod:`pyfe4ai.schemes.mcfe.labels`).
All clients must use the same set of attributes under a tag, since it is hashed with the tag (NPP22 Remark 11), and a
decryption needs the ciphertexts of all ``n`` clients, also those whose weight is 0, since ``p_1 + ... + p_n = 1``
(NPP22 Remark 10). Ciphertexts are malleable and not authenticated: the security notion of the paper (Def. 9) lets the
adversary corrupt clients and ask for functional keys, and says nothing about a ciphertext that was modified or forged.

The ``label`` and the ``attributes_digest`` of a ciphertext are claims of its client. The points ``H(tag, S)`` are
inside ``t`` and ``c_ipfe``, and checking them against the two fields would need the master key, so a decryption does
not hash again. It checks that the ciphertexts of all clients agree on both fields, that the digest is the one of the
listed attributes and, if the caller gives a ``label``, that this is the one the ciphertexts claim. One client that
hashed other points than it says gives a ``ValueError`` from the discrete log (no value, and not a wrong one); all
clients relabelled alike decrypt to the correct value, since the fields are not what a ciphertext is made of. A caller
who needs to know that ciphertexts belong to a round authenticates them.

The policy of a key (``dk["policy"]``) and the attributes of a ciphertext (``ct["attributes"]``, with the tag in
``ct["label"]``) are public: they travel in the clear and hide nothing. Keys are trusted input to the decryption: the
``pp_binding`` of a key tells the objects of two setups apart and is not a signature. A client key ``ek_i`` is secret.

A key has one ``a0``, one set of shares ``a_j`` and one ``z`` for all clients, and only the ``pi_ij`` are fresh for each
client and row. This is the reading of Fig. 3, which writes ``a_j`` and ``z`` without an index ``i`` in ``k*_ij``, and
of Sec. 3.3 ("all clients i using the same a0"). The proof text of App. B.4 says "for each i" where it samples ``a0``,
while the forms of its keys have no index; this module follows Fig. 3 and Sec. 3.3.

Deviations:

* ``g1`` and ``g2`` are random generators drawn at setup, not those of the group description.
* ``y_i`` and ``x_i`` may be negative within their bounds, not only in ``[0, B]``; the discrete log covers
  ``|<x, y>| <= n bound_x bound_y``. Each client encrypts one scalar (``eta = 1``), as Remark 10 has it; a vector is
  sent as several scalars, each under a tag of its own.
* ``[y_i]_2`` (``m_i`` of Fig. 3) is not part of the key: the decryption computes it from ``y``.
* ``H(tag, S)`` hashes the encoded label, the attribute-set digest, the identifier of the scheme and the
  ``pp_binding`` (:func:`~pyfe4ai.schemes.common.kp_ab.tag_points`), so that two setups and two schemes never share
  points. The attribute set is the sorted list of the distinct labels, however a client wrote it down.
* A label is an attribute by its hash to ``Z_q``, where the paper takes the attributes in ``Z_q``. A label on several
  rows of a policy takes a fresh ``pi`` for each, as ``List-Att(A)`` of the paper, which allows repetitions.
* The policy is compiled to a span program by :func:`~pyfe4ai.policy.msp.compile_formula` (nested Shamir sharing),
  with at most ``max_rows`` rows, and a ciphertext carries at most ``max_attributes`` labels of at most
  :data:`~pyfe4ai.schemes.common.kp_ab.MAX_LABEL_CHARS` characters each; the key generator refuses a policy with a
  longer label, which no ciphertext could satisfy. A decryption reads a policy text of at most
  ``max_rows * (MAX_LABEL_CHARS + 400)`` characters, which the key generator also keeps to. These are limits of this
  implementation, not of the paper: a ciphertext and a key come from outside, and no universe of labels bounds them.
* The dicts of a key and of the ciphertexts are keyed by the client ids as they are, like those of the function-hiding
  multi-client scheme. JSON turns an integer key into a string, so an application that moves them through JSON uses
  string ids.
"""

from __future__ import annotations

import copy
import json
import logging

import gmpy2 as gp

from pyfe4ai._validation import int_vector
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FEValidationError
from pyfe4ai.core.formats.pairing import MCKPABCiphertext
from pyfe4ai.core.formats.pairing import MCKPABClientKey
from pyfe4ai.core.formats.pairing import MCKPABFunctionalKey
from pyfe4ai.core.formats.pairing import MCKPABParameters
from pyfe4ai.core.formats.pairing import MCKPABRecord
from pyfe4ai.core.ids import as_id, check_id_list, id_list, is_id, issued_id
from pyfe4ai.core.versions import record_matches
from pyfe4ai.policy import (
    ATTR_HASH,
    DEFAULT_MAX_ROWS,
    POLICY_COMPILER,
    attribute_set_digest,
    canonical_encoding,
    from_canonical,
    policy_digest,
)
from pyfe4ai.policy.attributes import clip
from pyfe4ai.primitives.pairing.dpvs import basis_vectors, combine, random_dual_bases
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, ZR, multi_pair, multiexp, multiexp_many, pair
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, PairingGroup, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr
from pyfe4ai.primitives.pairing.pairing_utils import serialize_matrix_g
from pyfe4ai.schemes.common.ac_binding import check_binding, pp_digest
from pyfe4ai.schemes.common.ac_codec import check_dlog_limit as _check_dlog_limit
from pyfe4ai.schemes.common.ac_codec import check_length as _check_length
from pyfe4ai.schemes.common.ac_codec import check_marker as _check_marker
from pyfe4ai.schemes.common.ac_codec import copy_parameters as _copy_parameters
from pyfe4ai.schemes.common.ac_codec import decimal as _decimal
from pyfe4ai.schemes.common.ac_codec import load_group as _load_group
from pyfe4ai.schemes.common.ac_codec import positive_int as _positive_int
from pyfe4ai.schemes.common.ac_codec import read_elements as _elements
from pyfe4ai.schemes.common.ac_codec import read_record as _record
from pyfe4ai.schemes.common.ac_codec import require_fields as _require_fields
from pyfe4ai.schemes.common.ac_codec import scalar as _scalar
from pyfe4ai.schemes.common.ac_codec import serialize_vec as _serialize_vec
from pyfe4ai.schemes.common.kp_ab import DEFAULT_MAX_ATTRIBUTES, F_DIM, H_DIM, TAG_HASH
from pyfe4ai.schemes.common.kp_ab import attribute_pairs, attribute_vector, check_policy_labels, check_policy_length
from pyfe4ai.schemes.common.kp_ab import compile_policy, held_attributes, key_row, policy_length_limit
from pyfe4ai.schemes.common.kp_ab import read_attributes, read_policy, tag_points
from pyfe4ai.schemes.mcfe.labels import encode_label

logger = logging.getLogger(__name__)

_PP_DOMAIN = b"pyfe4ai/mcfe/kp-ab/pp-binding/v1|"
_SCHEME_ID = "mcfe/kp_ab_pairing"


def _by_client(entries, what: str) -> dict:
    """The entries of the dict *entries*, keyed by the client ids they name (a numpy integer is the ``int`` it equals).

    Raises:
        FEValidationError: For something that is not a dict, and for a key that is no client id.
    """
    if not isinstance(entries, dict):
        raise FEValidationError("{} must be a dict keyed by client, got {}".format(what, type(entries).__name__))
    out = {}
    for key, value in entries.items():
        nid = as_id(key)
        if not is_id(nid):
            raise FEValidationError("{} has the key {}, which is no client id".format(what, clip(repr(key))))
        out[nid] = value
    return out


def _check_clients(entries: dict, lst_nid: list, what: str) -> None:
    """Raise unless the keys of *entries* are the clients of *lst_nid*, naming those that are missing or unknown."""
    known = set(lst_nid)
    missing = [nid for nid in lst_nid if nid not in entries]
    unknown = id_list(nid for nid in entries if nid not in known)
    if missing or unknown:
        raise FEValidationError("{} must cover every client and no other: missing {}, not a client {}".format(
            what, clip(repr(missing), 120), clip(repr(unknown), 120)))


class MCFEKPABPairingKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for multi-client key-policy attribute-based IPFE over pairings."""

    _scheme_type = CryptoCONST.TYPE_MCFE_KP_AB
    _config_only_record = True  # the record holds the configuration only (see ParameterCacheMixin)

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the multi-client key-policy attribute-based IPFE key generator.

        Args:
            config: Scheme configuration. Needed: ``n``, the number of clients. Recognised keys: ``lst_nid`` (the
                client ids, default ``nid_0`` to ``nid_{n-1}``; their order fixes that of the shares ``p_i``),
                ``eta`` (only 1: each client encrypts one scalar under a tag), ``bound_x``, ``bound_y``,
                ``max_rows`` (the rows a policy's span program may have, default 64), ``max_attributes`` (the labels
                a ciphertext may carry, default 64) and ``pairing_group_param`` (asymmetric group, default
                ``"BLS12-381"``).

        Raises:
            FEConfigError: Without ``n``.
            FEValidationError: For a limit that is not a positive int, ``eta`` other than 1, ``n * bound_x * bound_y``
                above the discrete-log limit, a symmetric pairing group, or client ids that are not a list of
                ``n`` distinct ids.
        """
        n = config.get("n", None)
        if n is None:
            raise FEConfigError("need n, the number of clients")
        _positive_int(n, "n")
        super().__init__(config, **kwargs)
        require_pairing_backend()
        eta = config.get("eta", 1)
        if type(eta) is not int or eta != 1:
            raise FEValidationError(
                "eta must be 1: each client encrypts one scalar under a tag (NPP22 Remark 10), got {}".format(
                    clip(repr(eta))))
        self.eta = 1
        self.lst_nid = list(self.lst_nid)
        self.bound_x = _positive_int(config.get("bound_x", 16), "bound_x")
        self.bound_y = _positive_int(config.get("bound_y", 16), "bound_y")
        self.max_rows = _positive_int(config.get("max_rows", DEFAULT_MAX_ROWS), "max_rows")
        self.max_attributes = _positive_int(config.get("max_attributes", DEFAULT_MAX_ATTRIBUTES), "max_attributes")
        _check_dlog_limit(self.n, self.bound_x, self.bound_y, "n")
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        self.max_policy_length = policy_length_limit(self.max_rows)
        self.pp = None
        self.msk = None
        self._client_keys = None
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return record_matches(param, {
            "sec_param": self.sec_param, "n": self.n, "lst_nid": self.lst_nid, "bound_x": str(self.bound_x),
            "bound_y": str(self.bound_y), "max_rows": self.max_rows, "max_attributes": self.max_attributes,
            "pairing_group_param": self.pairing_group_param, "policy_compiler": POLICY_COMPILER,
            "attr_hash": ATTR_HASH, "tag_hash": TAG_HASH,
        })

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                MCKPABRecord.new(
                    sec_param=self.sec_param,
                    n=self.n,
                    lst_nid=list(self.lst_nid),
                    bound_x=str(self.bound_x),
                    bound_y=str(self.bound_y),
                    max_rows=self.max_rows,
                    max_attributes=self.max_attributes,
                    pairing_group_param=self.pairing_group_param,
                    policy_compiler=POLICY_COMPILER,
                    attr_hash=ATTR_HASH,
                    tag_hash=TAG_HASH,
                ).to_dict(),
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        group, order = self.group, self.order
        g1 = group.random(G1)
        g2 = group.random(G2)

        f, f_star = random_dual_bases(group, F_DIM, gp.mpz(1), order)
        s = {nid: gp.mpz(int(random_nonzero_zr(group))) for nid in self.lst_nid}
        u = {nid: gp.mpz(int(random_nonzero_zr(group))) for nid in self.lst_nid}
        # an n-out-of-n sharing of 1
        p = [gp.mpz(int(group.random(ZR))) for _ in range(self.n - 1)]
        p.append((1 - sum(p)) % order)

        body = MCKPABParameters.new(
            n=self.n,
            lst_nid=list(self.lst_nid),
            eta=1,
            bound_x=str(self.bound_x),
            bound_y=str(self.bound_y),
            max_rows=self.max_rows,
            max_attributes=self.max_attributes,
            pairing_group_param=self.pairing_group_param,
            policy_compiler=POLICY_COMPILER,
            attr_hash=ATTR_HASH,
            tag_hash=TAG_HASH,
            g1=group.serialize(g1).decode("utf-8"),
            g2=group.serialize(g2).decode("utf-8"),
        ).to_dict()
        self.pp = MCKPABParameters.extend(body, pp_binding=pp_digest(body, _PP_DOMAIN)).to_dict()

        f_g1 = serialize_matrix_g(group, basis_vectors(f, [0, 1, 2], g1, group))
        client_keys, h_star = {}, {}
        for nid, p_i in zip(self.lst_nid, p):
            h, h_dual = random_dual_bases(group, H_DIM, gp.mpz(1), order)
            h_star[nid] = basis_vectors(h_dual, [0, 1, 2], g2, group)
            client_keys[nid] = MCKPABClientKey.new(
                nid=nid,
                s=gp.digits(s[nid]),
                u=gp.digits(u[nid]),
                p_h1=[gp.digits(p_i * entry % order) for entry in h[0]],
                p_h2=[gp.digits(p_i * entry % order) for entry in h[1]],
                h3=_serialize_vec(group, basis_vectors(h, [2], g1, group)[0]),
                f=[list(row) for row in f_g1],
                pp_binding=self.pp["pp_binding"],
            ).to_dict()
        # what the paper keeps: S, U and f*_1..3, (h*_i,1..3)_i as elements of G2, and the encryption keys; the
        # bases F, H_i, F*, H*_i stay behind
        self.msk = {
            "s": {nid: gp.digits(value) for nid, value in s.items()},
            "u": {nid: gp.digits(value) for nid, value in u.items()},
            "f_star": basis_vectors(f_star, [0, 1, 2], g2, group),
            "h_star": h_star,
        }
        self._client_keys = client_keys
        logger.info("MCFE key-policy attribute-based IPFE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """The encryption key ``ek_i`` of client *nid*, or ``None`` for an id that is no client.

        The key is secret, and holds nothing of another client.

        Raises:
            FEKeyError: Before :meth:`setup`.
        """
        if self._client_keys is None:
            raise FEKeyError("no encryption keys: run setup() before asking for the key of a client")
        nid = issued_id(nid)    # numpy integers as ints; a value that is no id is an unknown client
        if nid is None or nid not in self._client_keys:
            return None
        return copy.deepcopy(self._client_keys[nid])

    def _read_weights(self, fusion_weight) -> dict:
        """The weight of each client as an int: *fusion_weight* must give one list of one weight within ``bound_y``
        for every client."""
        weights = _by_client(fusion_weight, "fusion weights")
        _check_clients(weights, self.lst_nid, "fusion weights")
        return {nid: int_vector(weights[nid], "fusion weight of {}".format(nid), length=1, bound=self.bound_y)[0]
                for nid in self.lst_nid}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given weights and policy.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (a dict from every client to a list of one
                int) and ``policy`` (a :class:`~pyfe4ai.policy.formula.Formula` or its text).

        Raises:
            FEKeyError: Before :meth:`setup`, or without credentials.
            FEValidationError: For credentials that are not a dict with both entries, weights that do not cover
                every client and no other or are out of bounds, a policy that does not parse, has more than
                ``max_rows`` rows, has a label of more than
                :data:`~pyfe4ai.schemes.common.kp_ab.MAX_LABEL_CHARS` characters (no ciphertext can hold it) or is
                longer than a decryption reads (``max_rows * (MAX_LABEL_CHARS + 400)`` characters).
        """
        if self.msk is None:
            raise FEKeyError("no master key: run setup() before deriving a functional key")
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for key-policy attribute-based IPFE decryption key generation")
        if not isinstance(credentials, dict):
            raise FEValidationError("credentials must be a dict with the entries fusion_weight and policy, got {}"
                                    .format(type(credentials).__name__))
        for entry in ("fusion_weight", "policy"):
            if entry not in credentials:
                raise FEValidationError("credentials need the entry {}".format(entry))
        y = self._read_weights(credentials["fusion_weight"])
        formula = read_policy(credentials["policy"])
        # compile first: it bounds the leaves of a formula built from shared sub-formulas
        msp = compile_policy(formula, self.order, self.max_rows)
        check_policy_labels(msp.row_labels)                         # no ciphertext could satisfy such a row
        encoding = canonical_encoding(formula)
        check_policy_length(encoding, self.max_policy_length)       # no decryption would read it

        group, order, msk = self.group, int(self.order), self.msk
        # one a0, one set of shares and one z for the key, whatever the client; only the pi are fresh for each
        a0, z = int(group.random(ZR)), int(group.random(ZR))
        shares = msp.shares(a0, [int(group.random(ZR)) for _ in range(msp.num_cols - 1)])
        s_y = sum(int(msk["s"][nid]) * y[nid] for nid in self.lst_nid) % order
        u_y = sum(int(msk["u"][nid]) * y[nid] for nid in self.lst_nid) % order
        k_rows, k_ipfe = {}, {}
        for nid in self.lst_nid:
            k_rows[nid] = serialize_matrix_g(group, [
                key_row(msk["f_star"], label, int(group.random(ZR)), share * z, order, group)
                for label, share in zip(msp.row_labels, shares)
            ])
            k_ipfe[nid] = _serialize_vec(group, combine(msk["h_star"][nid], [s_y, u_y, a0 * z % order], group))

        return MCKPABFunctionalKey.new(
            policy=encoding,
            policy_digest=policy_digest(formula),
            policy_compiler=POLICY_COMPILER,
            y={nid: str(y[nid]) for nid in self.lst_nid},
            k_rows=k_rows,
            k_ipfe=k_ipfe,
            pp_binding=self.pp["pp_binding"],
        ).to_dict()


class MCFEKPABPairing(IPFEAbsCrypto):
    """Crypto operations for multi-client key-policy attribute-based IPFE over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the multi-client key-policy attribute-based IPFE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters), and for a client ``sk`` (its encryption
                key) and optionally ``id`` (the client, which must be the one of ``sk``). An aggregator needs
                ``pp`` only. The object keeps a copy of both, so later changes to those dicts do not reach it.

        Raises:
            FEKeyError: Without public parameters.
            FEValidationError: For public parameters that do not agree with their ``pp_binding``, or whose fields
                do not agree with each other, and for an encryption key that is not one of these parameters or of
                the client ``id``.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        pp = self.keys.get("pp")
        if pp is None:
            raise FEKeyError("no public parameters: keys needs the entry pp")
        pp = _copy_parameters(pp)       # what is checked below stays what is used, whatever the caller's dict does
        self._read_parameters(pp)
        self.pp = pp
        self._decoded = {}
        self._gt = None
        self.group = _load_group(pp["pairing_group_param"])
        self.order = gp.mpz(int(self.group.order()))
        self._secret = None
        self._client_key = None
        self._nid = None
        if self._has_private_keys():
            self._check_private_key()
            self._read_client_key(_copy_parameters(self.keys["sk"], "the client key"))

    def _read_parameters(self, pp: dict) -> None:
        """Check *pp*, which comes from outside, and keep what its fields say.

        The digest binds the fields to each other and nothing to a key authority, so the lengths are checked too:
        they bound what later calls read.
        """
        parameters = _record(MCKPABParameters, pp, "public parameters")
        _require_fields(parameters, "public parameters")
        if pp_digest(pp, _PP_DOMAIN) != parameters.pp_binding:
            raise FEValidationError("the public parameters do not agree with their pp_binding")
        _check_marker(parameters.policy_compiler, POLICY_COMPILER, "the policy_compiler of the public parameters")
        _check_marker(parameters.attr_hash, ATTR_HASH, "the attr_hash of the public parameters")
        _check_marker(parameters.tag_hash, TAG_HASH, "the tag_hash of the public parameters")
        if type(parameters.pairing_group_param) is not str:
            raise FEValidationError("pairing_group_param must be a str, got {}".format(
                type(parameters.pairing_group_param).__name__))

        self.n = _positive_int(parameters.n, "n")
        check_id_list(parameters.lst_nid, "lst_nid")
        if len(parameters.lst_nid) != self.n:
            raise FEValidationError("lst_nid has {} clients but n = {}".format(len(parameters.lst_nid), self.n))
        self.lst_nid = list(parameters.lst_nid)
        if _positive_int(parameters.eta, "eta") != 1:
            raise FEValidationError("eta must be 1, got {}".format(parameters.eta))
        self.bound_x = _positive_int(_decimal(parameters.bound_x, "bound_x"), "bound_x")
        self.bound_y = _positive_int(_decimal(parameters.bound_y, "bound_y"), "bound_y")
        self.max_rows = _positive_int(parameters.max_rows, "max_rows")
        self.max_attributes = _positive_int(parameters.max_attributes, "max_attributes")
        _check_dlog_limit(self.n, self.bound_x, self.bound_y, "n")
        self.max_policy_length = policy_length_limit(self.max_rows)

    def _read_client_key(self, sk: dict) -> None:
        """Check the encryption key *sk*, which comes from outside, and keep its exponents."""
        ek = _record(MCKPABClientKey, sk, "client key")
        _require_fields(ek, "client key")
        check_binding(self.pp, ek, "client key")
        nid = as_id(ek.nid)
        if not is_id(nid) or nid not in self.lst_nid:
            raise FEValidationError("the client key is for {}, which is no client of the public parameters".format(
                clip(repr(ek.nid))))
        if self.id is not None and self.id != nid:
            raise FEValidationError("the client key is for {} but the id of the crypto object is {}".format(
                clip(repr(nid)), clip(repr(self.id))))
        _check_length(ek.p_h1, H_DIM, "p_h1")
        _check_length(ek.p_h2, H_DIM, "p_h2")
        _check_length(ek.h3, H_DIM, "h3")
        _check_length(ek.f, 3, "f")
        for row in ek.f:
            _check_length(row, F_DIM, "f")
        order = int(self.order)
        self._nid = nid
        self._secret = {
            "s": _scalar(ek.s, "s", order),
            "u": _scalar(ek.u, "u", order),
            "p_h1": [_scalar(value, "p_h1", order) for value in ek.p_h1],
            "p_h2": [_scalar(value, "p_h2", order) for value in ek.p_h2],
        }
        self._client_key = ek

    def _key_elements(self, name: str, index: int | None = None) -> list:
        """The elements of the field *name* of the encryption key (row *index* of a table), decoded and checked at
        their first use, and kept: encryption reads the same ones each time."""
        key = ("ek", name, index)
        if key not in self._decoded:
            items = getattr(self._client_key, name)
            if index is not None:
                items = items[index]
            self._decoded[key] = _elements(self.group, items, G1, name)
        return self._decoded[key]

    def _generator(self, name: str, kind):
        """The generator ``g1`` or ``g2`` of the public parameters, decoded at its first use."""
        key = (name, None)
        if key not in self._decoded:
            self._decoded[key] = _elements(self.group, [self.pp[name]], kind, name)[0]
        return self._decoded[key]

    def _generator_pair(self):
        """``e(g1, g2)``, computed once."""
        if self._gt is None:
            self._gt = pair(self._generator("g1", G1), self._generator("g2", G2))
        return self._gt

    def encrypt(self, lst_pt: list, label, attributes=None) -> dict:
        """Encrypt one scalar for a tag and a set of attributes.

        A client must encrypt once under a tag; the scheme layer does not check it.

        Args:
            lst_pt: The one integer ``x_i`` with ``|x_i| <= bound_x``, in a list.
            label: The tag: a ``str``, or an element label of :mod:`pyfe4ai.schemes.mcfe.labels`. All clients of a
                decryption use the same one.
            attributes: The labels the ciphertext is for: any collection of valid labels (a ``set``, or a list
                with repeats), at most ``max_attributes`` distinct ones of at most ``MAX_LABEL_CHARS`` characters
                each. The ciphertext lists them sorted and each once. All clients of a decryption use the same
                set. The empty collection is allowed: no policy is satisfied by it.

        Returns:
            The ciphertext, a dict.

        Raises:
            FEKeyError: Without the encryption key ``sk``.
            FEValidationError: For a plaintext out of bounds, a label that is not valid, no attributes, or
                attributes that are not a collection of valid labels within the limits.
        """
        if self._secret is None:
            raise FEKeyError("no encryption key: keys needs the entry sk to encrypt")
        x = int_vector(lst_pt, "plaintext", length=1, bound=self.bound_x)[0]
        encoded_label = encode_label(label)
        labels = held_attributes(attributes, self.max_attributes)
        digest = attribute_set_digest(labels)
        group, order, secret = self.group, int(self.order), self._secret
        g1 = self._generator("g1", G1)
        f = [self._key_elements("f", i) for i in range(3)]
        h3 = self._key_elements("h3")

        omega, omega_prime = tag_points(group, _SCHEME_ID, self.pp["pp_binding"], encoded_label, digest)
        psi = int(group.random(ZR))
        c_attr = [attribute_vector(f, name, int(group.random(ZR)), psi, order, group) for name in labels]
        # t = omega^s omega'^u g1^x and c_ipfe[k] = omega^(p h1[k]) omega'^(p h2[k]) h3[k]^psi, in one call
        rows = [[omega, omega_prime, g1]] + [[omega, omega_prime, h3[k]] for k in range(H_DIM)]
        exponents = [[secret["s"], secret["u"], x % order]]
        exponents += [[secret["p_h1"][k], secret["p_h2"][k], psi] for k in range(H_DIM)]
        t, *c_ipfe = multiexp_many(rows, exponents, group)
        return MCKPABCiphertext.new(
            nid=self._nid,
            label=encoded_label,
            attributes=labels,
            attributes_digest=digest,
            policy_compiler=POLICY_COMPILER,
            attr_hash=ATTR_HASH,
            tag_hash=TAG_HASH,
            c_attr=serialize_matrix_g(group, c_attr),
            t=group.serialize(t).decode("utf-8"),
            c_ipfe=_serialize_vec(group, c_ipfe),
            pp_binding=self.pp["pp_binding"],
        ).to_dict()

    def _check_key(self, key: MCKPABFunctionalKey, rows: int) -> dict:
        """Check the clients and the shape of the fields of *key*, and return the weights as ints by client."""
        y = _by_client(key.y, "y of the functional key")
        k_rows = _by_client(key.k_rows, "k_rows of the functional key")
        k_ipfe = _by_client(key.k_ipfe, "k_ipfe of the functional key")
        for entries, what in ((y, "y"), (k_rows, "k_rows"), (k_ipfe, "k_ipfe")):
            _check_clients(entries, self.lst_nid, "{} of the functional key".format(what))
        weights = {}
        for nid in self.lst_nid:
            _check_length(k_rows[nid], rows, "k_rows of {}".format(nid))
            for row in k_rows[nid]:
                _check_length(row, F_DIM, "k_rows of {}".format(nid))
            _check_length(k_ipfe[nid], H_DIM, "k_ipfe of {}".format(nid))
            weights[nid] = _decimal(y[nid], "y of {}".format(nid))
            if abs(weights[nid]) > self.bound_y:
                raise FEValidationError("y of {} is {}, beyond the limit bound_y = {}".format(
                    nid, weights[nid], self.bound_y))
        return weights

    def _check_tags(self, cts: dict, label) -> list:
        """Check that the ciphertexts agree on the tag and on the set of attributes, and return the attributes.

        The label and the digest are claims of the clients: a decryption does not hash them again (the points are
        inside ``t`` and ``c_ipfe``), but a client that used other ones gives a value that does not decrypt, and this
        names it before any pairing.
        """
        first_nid = self.lst_nid[0]
        first = cts[first_nid]
        for nid in self.lst_nid:
            for name in ("label", "attributes_digest"):
                value = getattr(cts[nid], name)
                if type(value) is not str:
                    raise FEValidationError("the {} of the ciphertext of {} must be a str".format(name, nid))
                if value != getattr(first, name):
                    raise FEValidationError("the ciphertexts of {} and {} differ in their {}: {} and {}".format(
                        first_nid, nid, name, clip(repr(getattr(first, name))), clip(repr(value))))
        if label is not None:
            claimed = encode_label(label)       # a FEValidationError if it cannot be encoded, never a ValueError
            if claimed != first.label:
                raise FEValidationError("the ciphertexts claim the label {}, not {}".format(
                    clip(repr(first.label)), clip(repr(claimed))))
        sets = {}
        for nid in self.lst_nid:
            sets[nid] = read_attributes(cts[nid].attributes, self.max_attributes)
            if attribute_set_digest(sets[nid]) != cts[nid].attributes_digest:
                raise FEValidationError(
                    "the attributes_digest of the ciphertext of {} is not the digest of its attributes".format(nid))
        return sets[first_nid]

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict | None = None, label=None):
        """Decrypt the ciphertexts of all clients and recover the inner product.

        The bindings are checked before anything else, then the fields and the version markers, then that the
        ciphertexts are those of all clients under one tag and one set of attributes, then the policy of the key and
        the shape of the records, before any group element is read. A ciphertext is decoded whole: every vector of it,
        also those the key does not use, so that a malformed ciphertext is refused whichever key opens it. A key is
        read where the policy uses it.

        Args:
            dct_ct: The ciphertexts, one for every client, by client id.
            dk: Functional decryption key.
            fusion_weight: Optional; if given it must be the ``y`` of the key, as a dict from every client to a list
                of one int.
            label: Optional; if given it must be the label that the ciphertexts claim (the ``label`` field of each,
                which all agree on): this checks the claim and not how the ciphertexts were made. A label that cannot
                be encoded (:func:`~pyfe4ai.schemes.mcfe.labels.encode_label`) is a ``FEValidationError``.

        Returns:
            ``<x, y>``.

        Raises:
            FEKeyError: If the attributes of the ciphertexts do not satisfy the policy of the functional key.
            FEValidationError: For an object made under other public parameters, ciphertexts that are not those of
                every client and no other, or that claim different tags or sets of attributes, or a label or weights
                that are not those of the key and ciphertexts, or an object that is malformed.
            ValueError: If the discrete log is not found: the key and the ciphertexts do not fit, whatever their
                fields say.
        """
        key = _record(MCKPABFunctionalKey, dk, "functional key")
        cts = {nid: _record(MCKPABCiphertext, ct, "ciphertext of {}".format(nid))
               for nid, ct in _by_client(dct_ct, "the ciphertexts").items()}

        check_binding(self.pp, key, "functional key")
        for nid, ct in cts.items():
            check_binding(self.pp, ct, "ciphertext of {}".format(nid))
        _require_fields(key, "functional key")
        for nid, ct in cts.items():
            _require_fields(ct, "ciphertext of {}".format(nid))
        _check_marker(key.policy_compiler, POLICY_COMPILER, "the policy_compiler of the functional key")
        for nid, ct in cts.items():
            owner = "the ciphertext of {}".format(nid)
            _check_marker(ct.policy_compiler, POLICY_COMPILER, "the policy_compiler of " + owner)
            _check_marker(ct.attr_hash, ATTR_HASH, "the attr_hash of " + owner)
            _check_marker(ct.tag_hash, TAG_HASH, "the tag_hash of " + owner)

        _check_clients(cts, self.lst_nid, "the ciphertexts")
        for nid, ct in cts.items():
            named = as_id(ct.nid)       # a numpy integer is the int it equals; an array is not compared with a str
            if not is_id(named):
                raise FEValidationError("the ciphertext given for {} names the client {}, which is no client id".format(
                    clip(repr(nid)), clip(repr(ct.nid))))
            if named != nid:
                raise FEValidationError("the ciphertext given for {} is the one of {}".format(
                    clip(repr(nid)), clip(repr(named))))
        attributes = self._check_tags(cts, label)

        if type(key.policy) is not str:
            raise FEValidationError("the policy of a functional key must be a str")
        check_policy_length(key.policy, self.max_policy_length)
        formula = from_canonical(key.policy)
        if type(key.policy_digest) is not str or policy_digest(formula) != key.policy_digest:
            raise FEValidationError("the policy_digest of the functional key is not the digest of its policy")
        msp = compile_policy(formula, self.order, self.max_rows)

        y = self._check_key(key, msp.num_rows)
        for nid in self.lst_nid:
            ct = cts[nid]
            _check_length(ct.c_attr, len(attributes), "c_attr of {}".format(nid))
            for row in ct.c_attr:
                _check_length(row, F_DIM, "c_attr of {}".format(nid))
            if type(ct.t) is not str:
                raise FEValidationError("t of {} must be a serialised element of G1".format(nid))
            _check_length(ct.c_ipfe, H_DIM, "c_ipfe of {}".format(nid))
        if fusion_weight is not None:
            given = _by_client(fusion_weight, "the fusion weight")
            _check_clients(given, self.lst_nid, "the fusion weight")
            for nid in self.lst_nid:
                if int_vector(given[nid], "fusion weight of {}".format(nid), length=1) != [y[nid]]:
                    raise FEValidationError("the fusion weight of {} is not the y of the functional key".format(nid))

        weights = msp.reconstruction(attributes)
        if weights is None:
            raise FEKeyError("the attributes {} of the ciphertexts do not satisfy the policy of the functional key"
                             .format(clip(repr(attributes), 120)))

        group, order = self.group, int(self.order)
        lhs, rhs, t = [], [], []
        for nid in self.lst_nid:
            ct = cts[nid]
            flat = _elements(group, [item for row in ct.c_attr for item in row], G1, "c_attr of {}".format(nid))
            c_attr = [flat[i:i + F_DIM] for i in range(0, len(flat), F_DIM)]
            t.append(_elements(group, [ct.t], G1, "t of {}".format(nid))[0])
            c_ipfe = _elements(group, ct.c_ipfe, G1, "c_ipfe of {}".format(nid))
            k_ipfe = _elements(group, key.k_ipfe[nid], G2, "k_ipfe of {}".format(nid))
            k_rows = {row: _elements(group, key.k_rows[nid][row], G2, "k_rows of {}".format(nid)) for row in weights}
            pairs = attribute_pairs(weights, msp.row_labels, attributes, c_attr, k_rows, group, order)
            lhs.extend(pairs[0])
            rhs.extend(pairs[1])
            lhs.extend(element ** (order - 1) for element in c_ipfe)
            rhs.extend(k_ipfe)
        lhs.append(multiexp(t, [y[nid] % order for nid in self.lst_nid], group))
        rhs.append(self._generator("g2", G2))
        target = multi_pair(lhs, rhs)
        return bounded_discrete_log_gt(group, self._generator_pair(), target, self.n * self.bound_x * self.bound_y)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for the attribute-based IPFE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for the attribute-based IPFE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for the attribute-based IPFE")
