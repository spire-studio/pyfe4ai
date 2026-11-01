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
(NPP22 Remark 10). Ciphertexts are malleable and not authenticated: the security notion of the paper (Def. 12, Fig. 2)
lets the adversary corrupt clients and ask for functional keys, and says nothing about a ciphertext that was modified
or forged.

The ``label`` and the ``attributes_digest`` of a ciphertext are claims of its client. The points ``H(tag, S)`` are
inside ``t`` and ``c_ipfe``, and checking them against the two fields would need the master key, so a decryption does
not hash again. It checks that the ciphertexts of all clients agree on both fields, that the digest is the one of the
listed attributes and, if the caller gives a ``label``, that this is the one the ciphertexts claim. One client that
hashed other points than it says gives a ``ValueError`` from the discrete log (a ``FESchemeError`` from the array
helpers; no value, and not a wrong one). All clients that change the ``label`` alike still decrypt to the correct
value, since the label is not what a ciphertext is made of. The list of attributes is not a claim of that kind: it
says which vector of ``c_attr`` pairs with which row of the key. All clients that change it alike get a ``FEKeyError``
(the policy is not satisfied) or no value (the ``ValueError`` of the discrete log), never a wrong one; a label that no
row of the policy reads can be renamed without effect. A caller who needs to know that ciphertexts belong to a round
authenticates them.

The ndarray helpers encrypt each element under ``element_label(label, layer, index)``, a tag of its own, and decrypt
all elements of all layers in one batch. The points of the key that every element is paired with (the rows the policy
uses, with their reconstruction coefficients folded in, and ``-k*_i,ipfe``) are prepared once per call
(:func:`~pyfe4ai.primitives.pairing.pairing_backend.prepare_g2`; nothing is kept between calls), and each element is
then one product of pairings over them. The elements of an array are for one set of attributes. All elements are
checked first; they are then decoded, paired and solved 256 at a time, so that the memory of the decoded group
elements does not grow with the size of the arrays (see :meth:`MCFEKPABPairing.decrypt_lst_ndarray_ct`).
:meth:`MCFEKPABPairing.check_ndarray_ct` makes the checks that need no key and no decoded point, for a caller that
issues the functional key before it decrypts.

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
from contextlib import contextmanager

import gmpy2 as gp
import numpy as np

from pyfe4ai._validation import int_vector, scale_to_int
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.base import decoded_layer
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
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, ZR, multi_pair, multi_pair_prepared, multiexp
from pyfe4ai.primitives.pairing.pairing_backend import multiexp_many, prepare_g2
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, PairingGroup, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr
from pyfe4ai.primitives.pairing.pairing_utils import serialize_matrix_g
from pyfe4ai.schemes.common.ac_binding import check_binding, pp_digest
from pyfe4ai.schemes.common.ac_codec import check_dlog_limit as _check_dlog_limit
from pyfe4ai.schemes.common.ac_codec import check_length as _check_length
from pyfe4ai.schemes.common.ac_codec import check_marker as _check_marker
from pyfe4ai.schemes.common.ac_codec import check_tags as _check_tags_of
from pyfe4ai.schemes.common.ac_codec import copy_parameters as _copy_parameters
from pyfe4ai.schemes.common.ac_codec import decimal as _decimal
from pyfe4ai.schemes.common.ac_codec import has_tags as _has_tags
from pyfe4ai.schemes.common.ac_codec import load_group as _load_group
from pyfe4ai.schemes.common.ac_codec import positive_int as _positive_int
from pyfe4ai.schemes.common.ac_codec import read_elements as _elements
from pyfe4ai.schemes.common.ac_codec import read_record as _record
from pyfe4ai.schemes.common.ac_codec import require_fields as _require_fields
from pyfe4ai.schemes.common.ac_codec import scalar as _scalar
from pyfe4ai.schemes.common.ac_codec import serialize_vec as _serialize_vec
from pyfe4ai.schemes.common.ac_decoded import DecodedElementsMixin
from pyfe4ai.schemes.common.kp_ab import DEFAULT_MAX_ATTRIBUTES, F_DIM, H_DIM, TAG_HASH
from pyfe4ai.schemes.common.kp_ab import attribute_pairs, attribute_vector, check_policy_labels, check_policy_length
from pyfe4ai.schemes.common.kp_ab import compile_policy, held_attributes, key_row, policy_length_limit
from pyfe4ai.schemes.common.kp_ab import read_attributes, read_policy, tag_points, vector_positions
from pyfe4ai.schemes.mcfe.labels import element_label, encode_label

logger = logging.getLogger(__name__)

_PP_DOMAIN = b"pyfe4ai/mcfe/kp-ab/pp-binding/v1|"
_SCHEME_ID = "mcfe/kp_ab_pairing"

#: The elements of an array that :meth:`MCFEKPABPairing.decrypt_lst_ndarray_ct` decodes, pairs and solves at a time.
#: An element holds ``n (8 |S| + 5)`` decoded group elements (``n`` clients, ``|S|`` attributes), some hundred bytes
#: each: this bounds the memory of the decoded elements, whatever the size of the arrays.
_BATCH_ELEMENTS = 256


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


@contextmanager
def _in_element(layer: int, index: tuple):
    """Say, in a ``FEValidationError`` raised inside the block, which element of which layer it is about."""
    try:
        yield
    except FEValidationError as exc:
        raise FEValidationError("layer {}, element {}: {}".format(layer, index, exc)) from exc


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


class MCFEKPABPairing(DecodedElementsMixin, IPFEAbsCrypto):
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
        self._init_decoded()
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

    def _check_records(self, key, cts: dict) -> None:
        """Check the bindings, then the fields, then the version markers of *key* and of the ciphertexts *cts*, a dict
        from where each comes from (its client, or its client and element in an array) to its record. *key* is
        ``None`` where only the ciphertexts are checked."""
        if key is not None:
            check_binding(self.pp, key, "functional key")
        for whose, ct in cts.items():
            check_binding(self.pp, ct, "ciphertext of {}".format(whose))
        if key is not None:
            _require_fields(key, "functional key")
        for whose, ct in cts.items():
            _require_fields(ct, "ciphertext of {}".format(whose))
        if key is not None:
            _check_marker(key.policy_compiler, POLICY_COMPILER, "the policy_compiler of the functional key")
        for whose, ct in cts.items():
            owner = "the ciphertext of {}".format(whose)
            _check_marker(ct.policy_compiler, POLICY_COMPILER, "the policy_compiler of " + owner)
            _check_marker(ct.attr_hash, ATTR_HASH, "the attr_hash of " + owner)
            _check_marker(ct.tag_hash, TAG_HASH, "the tag_hash of " + owner)

    def _check_ciphertexts(self, cts: dict, label) -> list:
        """Check that *cts* are the ciphertexts of all clients and no other, each in the slot of its client, under one
        tag and one set of attributes (and under *label*, if it is not ``None``), and return the attributes."""
        _check_clients(cts, self.lst_nid, "the ciphertexts")
        for nid, ct in cts.items():
            named = as_id(ct.nid)       # a numpy integer is the int it equals; an array is not compared with a str
            if not is_id(named):
                raise FEValidationError("the ciphertext given for {} names the client {}, which is no client id".format(
                    clip(repr(nid)), clip(repr(ct.nid))))
            if named != nid:
                raise FEValidationError("the ciphertext given for {} is the one of {}".format(
                    clip(repr(nid)), clip(repr(named))))
        return self._check_tags(cts, label)

    def _read_key(self, key) -> tuple:
        """The span program of the policy of *key* and the weights ``y`` of its clients as ints, after the checks of
        the policy, its digest and the shape of the key."""
        if type(key.policy) is not str:
            raise FEValidationError("the policy of a functional key must be a str")
        check_policy_length(key.policy, self.max_policy_length)
        formula = from_canonical(key.policy)
        if type(key.policy_digest) is not str or policy_digest(formula) != key.policy_digest:
            raise FEValidationError("the policy_digest of the functional key is not the digest of its policy")
        msp = compile_policy(formula, self.order, self.max_rows)
        return msp, self._check_key(key, msp.num_rows)

    def _check_shapes(self, cts: dict, attributes: list) -> None:
        """Check the length of every field of the ciphertexts *cts* that is read as a vector."""
        for nid in self.lst_nid:
            ct = cts[nid]
            _check_length(ct.c_attr, len(attributes), "c_attr of {}".format(nid))
            for row in ct.c_attr:
                _check_length(row, F_DIM, "c_attr of {}".format(nid))
            if type(ct.t) is not str:
                raise FEValidationError("t of {} must be a serialised element of G1".format(nid))
            _check_length(ct.c_ipfe, H_DIM, "c_ipfe of {}".format(nid))

    def _check_fusion_weight(self, fusion_weight, y: dict) -> None:
        """Check that *fusion_weight*, if given, is the weights *y* of the key."""
        if fusion_weight is not None:
            given = _by_client(fusion_weight, "the fusion weight")
            _check_clients(given, self.lst_nid, "the fusion weight")
            for nid in self.lst_nid:
                if int_vector(given[nid], "fusion weight of {}".format(nid), length=1) != [y[nid]]:
                    raise FEValidationError("the fusion weight of {} is not the y of the functional key".format(nid))

    @staticmethod
    def _satisfying_weights(msp, attributes: list) -> dict:
        """The reconstruction coefficients of the rows of *msp* that *attributes* satisfy.

        Raises:
            FEKeyError: If *attributes* do not satisfy the policy.
        """
        weights = msp.reconstruction(attributes)
        if weights is None:
            raise FEKeyError("the attributes {} of the ciphertexts do not satisfy the policy of the functional key"
                             .format(clip(repr(attributes), 120)))
        return weights

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

        self._check_records(key, cts)
        attributes = self._check_ciphertexts(cts, label)
        msp, y = self._read_key(key)
        self._check_shapes(cts, attributes)
        self._check_fusion_weight(fusion_weight, y)
        weights = self._satisfying_weights(msp, attributes)

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

    def encrypt_lst_ndarray(self, lst_ndarray: list, label: str | None = None, attributes=None) -> list:
        """Encrypt arrays element by element, each element under a tag of its own.

        The tag of the element at ``index`` of layer ``layer`` is ``element_label(label, layer, index)``
        (:mod:`pyfe4ai.schemes.mcfe.labels`), so that no two elements of a client share one. A client calls this once
        under a label; the scheme layer does not check it.

        Args:
            lst_ndarray: The arrays to encrypt, real or integer valued, scaled by ``10 ** precision`` and rounded.
                Each has at least one dimension.
            label: The label of the round, a ``str``; all clients of a decryption use the same one.
            attributes: The labels the ciphertexts are for, as for :meth:`encrypt`: the same set for every element.
                An iterator is read once.

        Returns:
            One object array of ciphertexts for every array, of its shape.

        Raises:
            FEKeyError: Without the encryption key ``sk``.
            FEValidationError: For a label that is not a ``str``, attributes that are not a collection of valid labels
                within the limits, an array that is not made of numbers or is a scalar, and an element beyond
                ``bound_x``.
        """
        if self._secret is None:
            raise FEKeyError("no encryption key: keys needs the entry sk to encrypt")
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray, got {}".format(clip(repr(label))))
        labels = held_attributes(attributes, self.max_attributes)       # once: an iterator has no second reading
        out = []
        for layer, source in enumerate(lst_ndarray):
            scaled = scale_to_int(source, self.precision)
            if scaled.ndim == 0:
                raise FEValidationError("layer {} of the arrays to encrypt is a scalar: it needs at least one "
                                        "dimension".format(layer))
            cts = np.empty(scaled.shape, dtype=object)
            for index, value in np.ndenumerate(scaled):
                cts[index] = self.encrypt([int(value)], element_label(label, layer, index), attributes=labels)
            out.append(cts)
        return out

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list:
        """Aggregate the ciphertext arrays of all clients: :meth:`decrypt_lst_ndarray_ct`."""
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, **kwargs)

    def _read_arrays(self, dict_ndarray_ct) -> dict:
        """The ciphertext arrays by client: a list of arrays for every client, and for all of them the layers and
        shapes of the first client."""
        arrays = _by_client(dict_ndarray_ct, "the ciphertext arrays")
        _check_clients(arrays, self.lst_nid, "the ciphertext arrays")
        for nid in self.lst_nid:
            if type(arrays[nid]) not in (list, tuple):
                raise FEValidationError("the ciphertext arrays of {} must be a list of arrays, got {}".format(
                    nid, type(arrays[nid]).__name__))
        first = self.lst_nid[0]
        for nid in self.lst_nid:        # the first client is checked first, so that it is a list of arrays below
            if len(arrays[nid]) != len(arrays[first]):
                raise FEValidationError("the ciphertext arrays of {} hold {} layers, those of {} hold {}".format(
                    nid, len(arrays[nid]), first, len(arrays[first])))
            for layer, array in enumerate(arrays[nid]):
                if not isinstance(array, np.ndarray):
                    raise FEValidationError("the ciphertext arrays of {}: layer {} must be an ndarray, got {}".format(
                        nid, layer, type(array).__name__))
                if array.ndim == 0:
                    raise FEValidationError("the ciphertext arrays of {}: layer {} is a scalar: it needs at least one "
                                            "dimension".format(nid, layer))
                if array.shape != arrays[first][layer].shape:
                    raise FEValidationError("the ciphertext arrays of {}: layer {} has the shape {}, that of {} has {}"
                                            .format(nid, layer, array.shape, first, arrays[first][layer].shape))
        return arrays

    @staticmethod
    def _span_name(field: str, who) -> str:
        """The name of a vector of the key (*who*: its client) or of a ciphertext (*who*: ``(client, layer, index)``),
        for a message."""
        if isinstance(who, tuple):
            return "{} of {}, layer {}, element {}".format(field, *who)
        return "{} of {}".format(field, who)

    def _decode(self, spans: list, kind) -> list:
        """The elements of *kind* in all of *spans*, a list of ``(field, who, items)``, decoded in one call.

        Every span is checked as :func:`~pyfe4ai.schemes.common.ac_codec.read_elements` checks a vector. When one is
        not, the spans are read one at a time to find the first and name it.

        Raises:
            FEValidationError: Naming the first span with an item that is no element of *kind*.
        """
        items = [item for _, _, part in spans for item in part]
        try:
            return _elements(self.group, items, kind, "the elements")
        except FEValidationError:
            for field, who, part in spans:
                _elements(self.group, part, kind, self._span_name(field, who))
            raise

    def _ciphertext_spans(self, elements: list, cts: list):
        """The vectors of ``G1`` of the ciphertexts of *elements*, as ``(field, who, items)``, per element and
        client: the rows of ``c_attr`` flattened, then ``[t]``, then ``c_ipfe``."""
        for (layer, index), by_client in zip(elements, cts):
            for nid in self.lst_nid:
                ct, who = by_client[nid], (nid, layer, index)
                yield "c_attr", who, [item for row in ct.c_attr for item in row]
                yield "t", who, [ct.t]
                yield "c_ipfe", who, ct.c_ipfe

    def _check_field_tags(self, elements: list, cts: list) -> None:
        """Check that every vector of ``G1`` of the ciphertexts is a list of strings with the type tag of ``G1``.

        This is what :func:`~pyfe4ai.schemes.common.ac_codec.read_elements` checks first, and it costs no decoding:
        a decryption makes it for all elements before it prepares anything. What it cannot tell is an item with the
        right tag that is no element of the group; that is found when the slice of the item is decoded.

        Raises:
            FEValidationError: Naming the first vector that is not.
        """
        for field, who, part in self._ciphertext_spans(elements, cts):
            if not _has_tags(part, G1):
                _check_tags_of(part, G1, self._span_name(field, who))

    def _key_points(self, key, weights: dict) -> list:
        """The points of ``G2`` that every element is paired with, for :func:`prepare_g2`: for each client, in order,
        the eight points of each row the policy uses (in the order of *weights*) and the four of ``k*_i,ipfe``; the
        last is ``g2``.

        The reconstruction coefficient ``w`` of a row and the sign of ``k*_i,ipfe`` sit in the points
        (``e(c, w k) = e(c, k)^w``, ``e(c, -k) = e(c, k)^-1``). They are the same for every element, so they cost one
        exponentiation of each point of the key, in one call, rather than one of each ``G1`` element of each element;
        a coefficient of 1 costs none.
        """
        group, order = self.group, int(self.order)
        spans = []
        for nid in self.lst_nid:
            spans.extend(("k_rows", nid, key.k_rows[nid][row]) for row in weights)
            spans.append(("k_ipfe", nid, key.k_ipfe[nid]))
        points = self._decode(spans, G2)
        folded, exponents, at = [], [], 0
        for _ in self.lst_nid:
            for weight in weights.values():
                if weight % order != 1:
                    folded.extend(range(at, at + F_DIM))
                    exponents.extend([weight % order] * F_DIM)
                at += F_DIM
            folded.extend(range(at, at + H_DIM))
            exponents.extend([order - 1] * H_DIM)
            at += H_DIM
        powers = multiexp_many([[points[slot]] for slot in folded], [[exponent] for exponent in exponents], group)
        for slot, power in zip(folded, powers):
            points[slot] = power
        points.append(self._generator("g2", G2))
        return points

    def _element_rows(self, elements: list, cts: list, used: list, attributes: list, y: dict) -> list:
        """For each of *elements*, the row of ``G1`` elements that is paired with :meth:`_key_points`: for each
        client the vector of ``c_attr`` of each used row (*used*: their positions among the *attributes*) and
        ``c_ipfe``, and last ``prod_i t_i^y_i``, which goes with ``g2``. All the ``G1`` fields of these elements are
        decoded in one call."""
        order, n = int(self.order), len(self.lst_nid)
        decoded = self._decode(list(self._ciphertext_spans(elements, cts)), G1)
        t_at = F_DIM * len(attributes)              # in the elements of a ciphertext: c_attr, then t, then c_ipfe
        rows, masks = [], []
        for element in range(len(cts)):
            row, t = [], []
            for client in range(n):
                start = (element * n + client) * (t_at + 1 + H_DIM)
                for vector in used:
                    row.extend(decoded[start + F_DIM * vector:start + F_DIM * (vector + 1)])
                row.extend(decoded[start + t_at + 1:start + t_at + 1 + H_DIM])
                t.append(decoded[start + t_at])
            rows.append(row)
            masks.append(t)
        exponent_row = [y[nid] % order for nid in self.lst_nid]
        for row, mask in zip(rows, multiexp_many(masks, [exponent_row] * len(masks), self.group)):
            row.append(mask)
        return rows

    def _check_arrays(self, dict_ndarray_ct, label: str, key=None) -> tuple:
        """The checks of a decryption of arrays that need no decoded point, in the order of :meth:`decrypt`: the
        structure of the arrays, the ciphertexts as records, their bindings, fields and version markers (with those of
        *key*, a functional key record, unless it is ``None``), the clients, tags and digests of each element under
        its label, one set of attributes for all elements, (the policy and shape of *key*,) the length of every field
        and the type tag of every vector of ``G1``.

        Returns:
            ``(shapes, elements, cts, attributes, msp, y)``: the shape of each layer, the ``(layer, index)`` of each
            element, its records by client, the common set of attributes (``None`` without elements), and, for a *key*,
            its span program and weights (``None`` otherwise).
        """
        arrays = self._read_arrays(dict_ndarray_ct)
        shapes = [array.shape for array in arrays[self.lst_nid[0]]]
        elements = [(layer, index) for layer, shape in enumerate(shapes) for index in np.ndindex(shape)]

        flat, cts = {}, []
        for layer, index in elements:
            by_client = {}
            for nid in self.lst_nid:
                where = "{}, layer {}, element {}".format(nid, layer, index)
                by_client[nid] = flat[where] = _record(MCKPABCiphertext, arrays[nid][layer][index],
                                                       "ciphertext of " + where)
            cts.append(by_client)
        self._check_records(key, flat)
        attributes, first = None, None
        for (layer, index), by_client in zip(elements, cts):
            with _in_element(layer, index):
                held = self._check_ciphertexts(by_client, element_label(label, layer, index))
            if attributes is None:
                attributes, first = held, (layer, index)
            elif held != attributes:
                raise FEValidationError(
                    "layer {}, element {}: the attributes {} are not those of layer {}, element {}, {}: the elements "
                    "of an array are for one set of attributes".format(
                        layer, index, clip(repr(held), 120), first[0], first[1], clip(repr(attributes), 120)))
        msp, y = (None, None) if key is None else self._read_key(key)
        for (layer, index), by_client in zip(elements, cts):
            with _in_element(layer, index):
                self._check_shapes(by_client, attributes)
        self._check_field_tags(elements, cts)
        return shapes, elements, cts, attributes, msp, y

    def check_ndarray_ct(self, dict_ndarray_ct: dict, label: str | None = None) -> None:
        """Check the ciphertext arrays of all clients as far as :meth:`decrypt_lst_ndarray_ct` can before it has a key:
        every check that needs neither a functional key nor a decoded group element.

        These are the structure of the arrays (a list of arrays for every client, the same layers and shapes), the
        ciphertexts as records, their bindings, fields and version markers, for each element the clients (all of them
        and no other, each in its own slot), the tag (each element under the label that :meth:`encrypt_lst_ndarray`
        gave it, for *label*), the attributes and their digest, one set of attributes for all elements, the length of
        every field, and that every vector of ``G1`` is a list of strings with the type tag of ``G1``. The messages
        are those of :meth:`decrypt_lst_ndarray_ct`, which makes the same checks (with those of its key) before it
        decodes anything.

        What this does not see. An item with the right type tag that is no element of the group is found when it is
        decoded. The ``label`` and the attributes of a ciphertext are claims of its client: the points ``H(tag, S)`` are
        inside ``t`` and ``c_ipfe``, and only the master key could check them. Clients that disagree on the label or
        the digest are refused, and one client whose points differ from its claim gives no value at the discrete log
        (a ``FESchemeError`` from the array helpers, a ``ValueError`` from :meth:`decrypt`). But if all clients copy
        the label of a round onto ciphertexts of an earlier round, that relabelled replay passes this check and
        decrypts, silently, to the sum of the round the ciphertexts were made in: the key covers every tag. Attributes
        that all clients change alike are different: they give a ``FEKeyError`` or no value, never a wrong one (a
        label that no row of the policy reads can be renamed without effect, and the value is the right one). A
        caller who must know that ciphertexts belong to a round has to authenticate them. A caller that hands the arrays
        to a service that issues keys calls this first, so that a key is not issued for arrays that a decryption would
        refuse.

        Args:
            dict_ndarray_ct: The ciphertext arrays by client id, as for :meth:`decrypt_lst_ndarray_ct`.
            label: The label the arrays were encrypted under, a ``str``.

        Raises:
            FEValidationError: For a label that is not a ``str``, and for the arrays that :meth:`decrypt_lst_ndarray_ct`
                refuses before it decodes a point, as it does; the message names the layer and the element, the client
                and the layer where the shape of an array is wrong, or the clients and the counts where they send
                other numbers of layers.
        """
        if not isinstance(label, str):
            raise FEValidationError("need a string label for check_ndarray_ct, got {}".format(clip(repr(label))))
        self._check_arrays(dict_ndarray_ct, label)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, dk: dict | None = None,
                               fusion_weight: dict | None = None, label: str | None = None) -> list:
        """Decrypt the ciphertext arrays of all clients, all elements with one preparation of the key.

        The elements go through the checks of :meth:`decrypt` (the label of an element must be the one that
        :meth:`encrypt_lst_ndarray` gave it), and they must all be for one set of attributes. These checks, those of
        :meth:`check_ndarray_ct` with the key's, and that every vector of ``G1`` is a list of strings with the type tag
        of ``G1``, are made for all elements before anything is decoded, prepared or paired. The key is read once. Its
        points are prepared once, and the elements are then decoded, paired (one product of pairings each, over the
        prepared points) and solved ``_BATCH_ELEMENTS`` (256) at a time. A string with the right tag that is no element
        of the group is found when the slice that holds it is decoded, after the slices before it were paired; the call
        then raises and returns nothing, as for any other malformed input. The type tags are checked with the lengths,
        before the fusion weight and the satisfaction of the policy: an input both unsatisfied and wrongly tagged gets
        a ``FEValidationError`` here and a ``FEKeyError`` from :meth:`decrypt`.

        Memory: the ciphertexts are the caller's. The call adds about 4 KB for each of them (a record view, a name and
        the lists of the checks, measured at ``n = 10``, ``|S| = 3``) and the prepared points of the key, up to about
        20 KB each, ``n (8 rows + 4) + 1`` of them. The decoded group elements, ``n (8 |S| + 5)`` for each element of
        a slice, and the rows built from them are held for one slice at a time: about 0.5 to 0.8 MB per element of the
        slice at ``n = 10``, ``|S| = 3`` (measured), some 250 MB for a slice of 256, whatever the size of the arrays.

        Args:
            dict_ndarray_ct: The ciphertext arrays by client id: for each client a list of arrays (the layers, each of
                at least one dimension) of ciphertexts, with the same shapes for all clients.
            dk: Functional decryption key.
            fusion_weight: Optional; if given it must be the ``y`` of the key, as for :meth:`decrypt`.
            label: The label the arrays were encrypted under, a ``str``.

        Returns:
            One float array for every layer, the inner products divided by ``10 ** precision``.

        Raises:
            FEKeyError: Without a key, or if the attributes of the ciphertexts do not satisfy the policy of the key.
            FEValidationError: As :meth:`decrypt`, and for a label that is not a ``str``, arrays that do not have the
                same layers and shapes for all clients, and elements that are for different sets of attributes. The
                message names the layer and the element, the client and the layer for a wrong shape, or the clients
                and the counts for other numbers of layers.
            FESchemeError: For the first element whose discrete log was not found (the key and the ciphertexts do not
                fit, whatever their fields say).
        """
        if dk is None:
            raise FEKeyError("need to provide the decryption key dk")
        if not isinstance(label, str):
            raise FEValidationError("need a string label for decrypt_lst_ndarray_ct, got {}".format(clip(repr(label))))
        key = _record(MCKPABFunctionalKey, dk, "functional key")
        shapes, elements, cts, attributes, msp, y = self._check_arrays(dict_ndarray_ct, label, key)
        self._check_fusion_weight(fusion_weight, y)

        values = [np.empty(shape, dtype=object) for shape in shapes]
        if elements:
            weights = self._satisfying_weights(msp, attributes)
            used = vector_positions(weights, msp.row_labels, attributes)
            prepared = prepare_g2(self._key_points(key, weights))               # the lines are not kept after the call
            base, bound = self._generator_pair(), self.n * self.bound_x * self.bound_y
            for start in range(0, len(elements), _BATCH_ELEMENTS):
                part, part_cts = elements[start:start + _BATCH_ELEMENTS], cts[start:start + _BATCH_ELEMENTS]
                products = multi_pair_prepared(self._element_rows(part, part_cts, used, attributes, y), prepared)
                for (layer, index), product in zip(part, products):
                    try:
                        values[layer][index] = bounded_discrete_log_gt(self.group, base, product, bound)
                    except ValueError:
                        values[layer][index] = None     # decoded_layer names the element
        return [decoded_layer(array, layer, 10 ** self.precision) for layer, array in enumerate(values)]
