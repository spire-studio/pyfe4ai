"""
Multi-Client Key-Policy Attribute-Based Inner-Product Functional Encryption over Pairings, with an attribute set for
each client
| "Multi-Client Functional Encryption with Fine-Grained Access Control"
| By Ky Nguyen, Duong Hieu Phan and David Pointcheval, ASIACRYPT 2022
| (ePrint 2022/215) -- Sec. 5.4, on the scheme of Sec. 5.2 Fig. 3 (:mod:`pyfe4ai.schemes.mcfe.kp_ab_pairing`).

* type:     multi-client encryption of one scalar per client under a tag (a label) and *the client's own* set of
  attribute labels (:mod:`pyfe4ai.policy`); a functional key for a weight vector ``y`` and a policy opens ``<x, y>``
  when the set of every client satisfies the policy
* setting:  asymmetric pairing group (default BLS12-381), random oracle (the hash of the tag to ``G1``)
* status:   **known-insecure**. App. B.5 of the paper states Theorem 25 for this construction and sketches its proof as
  changes to that of Thm. 14. Thm. 25 as stated fails: two keys of one ``y`` that open neither of two rounds (two tags),
  where the set of each client satisfies one of the keys in both rounds, give the difference of the two rounds' inner
  products (see "The attack that the fifth coordinate does not stop" below). The scheme is kept for functionality and
  performance experiments and as the record of that attack. Its key generator and crypto objects emit a
  ``FESecurityWarning`` that names the issue, and the ``standard`` security profile refuses it (``FEValidationError``,
  as for any refused scheme). A crypto review of the 0.5 code confirmed the attack, on the algebra and on this
  implementation, and that the warning and the refusal take effect at every entry point; the implementation was not
  reviewed beyond that.

Sec. 5.2 hashes ``(tag, S)``, so all clients must agree on the set of attributes at the time of encryption. Sec. 5.4
hashes the tag only (``[omega]_1, [omega']_1 = H(tag)``) and lets each client ``i`` encrypt for a set ``S_i`` of its
own. That opens a mix-and-match attack on the keys: for two clients with ``S_1`` and ``S_2``, two keys of the policies
``A`` and ``A'`` with ``A(S_1) = A'(S_2) = 1`` and ``A'(S_1) = A(S_2) = 0``, for the same ``y``, let one take the
parts of client 1 from the first and those of client 2 from the second, and add up to ``<x, y>``, although neither key
opens the ciphertexts. The scheme closes it with a fifth dimension of the bases ``H_i`` and ``H_i*`` (dimension five):

* ``ek_i = (s_i, u_i, p_i H_i^(1), p_i H_i^(2), h_i,3, theta_i h_i,5, f1, f2, f3)`` with ``theta_i`` in ``Z_q^*``;
* ``c_i,ipfe = (omega p_i, omega' p_i, psi_i, 0, theta_i)_H_i``;
* ``k*_i,ipfe = (<S, y>, <U, y>, a_i,0 z, 0, d_i)_H*_i``, with the ``d_i`` of a key drawn afresh so that
  ``sum_i theta_i d_i = 0`` (the first ``n - 1`` at random, the last solved: ``_zero_sum_coordinates``).

The terms ``theta_i d_i`` add up to zero in a decryption with one key, so it is as in Fig. 3. With the parts of two
keys under one tag they add up to ``theta_1 d_A,1 + theta_2 d_B,2``, which is not zero except with negligible
probability. This holds for any two keys, also of one policy and one ``y``: under one tag the keys of one function
cannot be spliced either.

The attack that the fifth coordinate does not stop. ``theta_i d_{K,i}`` depends on the key ``K`` and the client, not on
the tag. Take two keys ``A`` and ``B`` of one ``y``, and two tags, such that the set of each client satisfies one of
the keys in both rounds (each key is given to the clients whose set it satisfies). The part of a client, ``V_i(tag)``
(its attribute rows, minus ``c_i,ipfe`` against ``k*_i,ipfe``, plus ``e(t_i, [y_i]_2)``), carries the same obstacle
``theta_i d_{K,i}`` under both tags, so ``sum_i (V_i(tag) - V_i(tag'))`` is ``[<x_tag - x_tag', y>]_T``. Neither key
opens either round (each refuses both with ``FEKeyError``), yet together they give the difference of the two rounds'
inner products. The two tags can be two labels or two elements of one array,
which the helpers encrypt under a tag each. Remark 10 and the game of Fig. 2 let the adversary encrypt under a tag other
than the challenge tag, and Def. 9 calls it admissible (``Rel = 0`` for both keys and both rounds, so its condition 3
does not hold), so Thm. 25 as stated fails.
``test_two_tags_and_two_keys_read_the_difference_of_two_rounds`` reproduces it from the fields of the records.
``mcfe/kp_ab_pairing`` is not affected: one attribute set per tag, so a key that gives a part opens the tag.

Everything else is as in :mod:`pyfe4ai.schemes.mcfe.kp_ab_pairing`, which this module extends (its documentation
applies): ``(F, F*)`` are dual bases of ``G1^8`` and ``G2^8`` shared by all clients, the rows of a policy sit in the
key as ``k*_ij = (pi_ij (j, 1), a_ij z, 0, ..., 0)_F*``, a ciphertext lists one vector of ``F`` for each attribute of
its client, ``t_i = [omega s_i + omega' u_i + x_i]_1`` with ``S, U`` in ``(Z_q^*)^n`` and ``p_1 + ... + p_n = 1``.

* The master key is ``msk = (S, U, (theta_i)_i, f1*, f2*, f3*, (h*_i,1, h*_i,2, h*_i,3)_i)``, plus ``h*_i,5``, the
  fifth vector of each ``H_i*``: the paper's list of ``msk`` leaves it out, and ``k*_i,ipfe`` needs it. The master key
  holds the four vectors ``h*_i,1, h*_i,2, h*_i,3, h*_i,5`` of each client as elements of ``G2^5`` (the exponents of
  the bases are dropped at the end of setup). The paper's "``eki`` contains furthermore ``theta_i h_i,5``" is the field
  ``theta_h5`` of the client key.
* KeyGen(y, P): a random ``z`` for the key, and for each client ``i`` its own root ``a_i,0`` and the shares
  ``a_ij = (M (a_i,0, r_i))_j`` of the span program ``M`` of ``P`` with fresh ``r_i``. The client's ``a_i,0`` is in
  ``k*_i,ipfe`` as Sec. 5.4 writes it (``a_{i,0} z``), and ``z`` has no index, as in Fig. 3 (the "per client" root is
  the paper's notation; ``mcfe/kp_ab_pairing`` has one root for all). The term ``psi_i a_i,0 z`` of a client cancels
  against its own ``c_i,ipfe`` only. The vectors ``k*_ij`` are made for all rows of the policy, for every client.
* Dec: for each client, the coefficients ``w_ij`` that reconstruct ``M`` from *its own* attributes, and the sum
  ``sum_i (sum_j w_ij e(c_ij, k*_ij) - e(c_i,ipfe, k*_i,ipfe) + e(t_i, [y_i]_2)) = [<x, y> - sum_i theta_i d_i]_T``.
  A client whose attributes do not satisfy the policy gives a ``FEKeyError`` that names it (the first in the order of
  the clients), before any pairing, also when its weight is 0: the shares ``p_i`` sum to 1 only over all clients
  (NPP22 Remark 10), so no decryption leaves one out. The pairing is one ``multi_pair`` of
  ``sum_i (8 |A_i| + 5) + 1`` pairs, ``|A_i|`` the rows that client ``i`` uses.

A client encrypts once under a tag; that is necessary here and, as the attack above shows, not enough. Two ciphertexts
of one client under one tag have the same ``omega, omega'`` and ``t`` values that differ by the difference of the
plaintexts in the exponent (``t_1 / t_2 = g1^(x_1 - x_2)``). In ``mcfe/kp_ab_pairing`` the points also depend on the set
of attributes, so a client that encrypts for two sets does not give that away; here the tag does not bind the set. The
scheme layer does not check it, as with the other labelled multi-client schemes (:mod:`pyfe4ai.schemes.mcfe.labels`).

Of a ciphertext only the ``label`` is a claim that the clients must agree on; the set of attributes of a client is its
own (the ciphertexts carry no ``attributes_digest``), so a decryption cannot check one client's set against another's.
The list of attributes says which vector of ``c_attr`` pairs with which row of the key: a client that changes it gets
a ``FEKeyError`` or no value (the ``ValueError`` of the discrete log), never a wrong one. The rest of the discussion of
the claims in :mod:`pyfe4ai.schemes.mcfe.kp_ab_pairing` holds. Ciphertexts are malleable and not authenticated; a
caller who needs to know that ciphertexts belong to a round authenticates them.

The ndarray helpers encrypt each element under ``element_label(label, layer, index)`` and decrypt all elements of all
layers in one batch, as in the scheme of Sec. 5.2, with one difference: a client has one set of attributes for the whole
call, which may differ from that of another client, so the rows of the key that a decryption uses differ between
clients (and so do the sizes of the ciphertexts). The elements of one client are for one set of attributes. The ML
adapter does not take this scheme (``_require_labelled_aggregation``): the clients of one label may hold different
sets, which the adapter does not aggregate. :meth:`MCFEKPABPairingPerClient.check_ndarray_ct` is as in the scheme of
Sec. 5.2.

Deviations, those of :mod:`pyfe4ai.schemes.mcfe.kp_ab_pairing` (random generators ``g1`` and ``g2``, signed ``x_i`` and
``y_i``, ``[y_i]_2`` computed by the decryption, a label hashed to ``Z_q`` as the attribute, the span program of
:func:`~pyfe4ai.policy.msp.compile_formula` with at most ``max_rows`` rows, at most ``max_attributes`` labels in a
ciphertext, dicts keyed by the client ids as they are, ``eta = 1``) and these:

* ``H(tag)`` hashes the encoded label, the identifier of the scheme and the ``pp_binding`` (the digest is ``None``,
  :func:`~pyfe4ai.schemes.common.kp_ab.tag_points`), so that two setups and two schemes never share points.
* The public parameters are those of the scheme of Sec. 5.2, with a ``pp_binding`` under another domain
  (``b"pyfe4ai/mcfe/kp-ab-per-client/pp-binding/v1|"``): objects of the two schemes never open each other.
* ``theta_i`` is drawn in ``Z_q^*`` (nonzero), so that the last ``d_i`` can be solved for. With one client ``d_1`` is 0:
  there is nothing to splice.
"""

from __future__ import annotations

import gmpy2 as gp

from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError
from pyfe4ai.core.formats.pairing import PCKPABCiphertext, PCKPABClientKey, PCKPABFunctionalKey
from pyfe4ai.policy.attributes import clip
from pyfe4ai.primitives.pairing.dpvs import encode
from pyfe4ai.primitives.pairing.pairing_backend import ZR
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr
from pyfe4ai.schemes.common.ac_codec import check_length as _check_length
from pyfe4ai.schemes.common.ac_codec import serialize_vec as _serialize_vec
from pyfe4ai.schemes.mcfe.kp_ab_pairing import MCFEKPABPairing, MCFEKPABPairingKeyGenerator

#: The dimension of ``H_i`` and ``H_i*``: the four of the scheme of Sec. 5.2 and the fifth coordinate.
H_DIM = 5
_PP_DOMAIN = b"pyfe4ai/mcfe/kp-ab-per-client/pp-binding/v1|"
_SCHEME_ID = "mcfe/kp_ab_pairing_per_client"


def _zero_sum_coordinates(thetas, order, group) -> list[int]:
    """The fifth coordinates ``d_1, ..., d_n`` of the vectors ``k*_i,ipfe`` of one functional key: integers in
    ``[0, order)`` with ``sum_i theta_i d_i = 0`` modulo *order*.

    The first ``n - 1`` are drawn at random (``group.random(ZR)``) and the last is solved for, which is why every
    ``theta`` must be nonzero. A key draws its own, independently of every other key; the ``d_i`` of a key are not
    stored anywhere after the key is made.

    Args:
        thetas: The ``theta_i`` of the clients, in the order of the clients, each nonzero modulo *order*.
        order: The group order.
        group: Pairing group instance.

    Raises:
        FEValidationError: Without a ``theta``, or for one that is 0 modulo *order*.
    """
    order = int(order)
    thetas = [int(theta) for theta in thetas]
    if not thetas:
        raise FEValidationError("need at least one theta")
    if any(theta % order == 0 for theta in thetas):
        raise FEValidationError(
            "every theta must be nonzero modulo the group order: the last d could not be solved for")
    d = [int(group.random(ZR)) % order for _ in thetas[:-1]]
    rest = sum(theta * value for theta, value in zip(thetas, d)) % order
    d.append(-rest * pow(thetas[-1], -1, order) % order)
    return d


class MCFEKPABPairingPerClientKeyGenerator(MCFEKPABPairingKeyGenerator):
    """Key generator for multi-client key-policy attribute-based IPFE over pairings with an attribute set for each
    client (NPP22 Sec. 5.4; **known-insecure**, see the module documentation: constructing it emits a
    ``FESecurityWarning`` and the ``standard`` profile refuses it). The options are those of
    :class:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairingKeyGenerator`.

    Setup draws a secret ``theta_i`` for each client, whose ``theta_i h_i,5`` goes into the encryption key and which
    the master key keeps with ``h*_i,5``; a functional key gives each client its own root and shares and the fifth
    coordinates of ``_zero_sum_coordinates``.
    """

    _scheme_type = CryptoCONST.TYPE_MCFE_KP_AB_PC
    H_DIM = H_DIM
    _pp_domain = _PP_DOMAIN
    #: ``h*_1, h*_2, h*_3`` and ``h*_5``: the paper's list of the master key leaves out the fifth.
    _H_STAR_ROWS = (0, 1, 2, 4)
    _ClientKey = PCKPABClientKey
    _FunctionalKey = PCKPABFunctionalKey

    def _client_extras(self, nid, h, g1) -> tuple[dict, dict]:
        """``theta_i`` in ``Z_q^*``, ``theta_h5 = theta_i h_i,5`` for the client key and ``theta_i`` for the master
        key."""
        group = self.group
        theta = gp.mpz(int(random_nonzero_zr(group)))
        theta_h5 = _serialize_vec(group, encode([theta], [h[4]], g1, self.order, group))
        return {"theta_h5": theta_h5}, {"theta": gp.digits(theta)}

    def _roots(self, msp) -> dict:
        """A root ``a_i,0`` and shares ``a_ij = (M (a_i,0, r_i))_j`` of its own, with fresh ``r_i``, for each client."""
        group = self.group
        roots = {}
        for nid in self.lst_nid:
            a0 = int(group.random(ZR))
            roots[nid] = (a0, msp.shares(a0, [int(group.random(ZR)) for _ in range(msp.num_cols - 1)]))
        return roots

    def _extra_coordinates(self) -> dict:
        """The fifth coordinate ``d_i`` of each client's ``k*_i,ipfe``, with ``sum_i theta_i d_i = 0``."""
        thetas = [int(self.msk["theta"][nid]) for nid in self.lst_nid]
        d = _zero_sum_coordinates(thetas, int(self.order), self.group)
        return {nid: (value,) for nid, value in zip(self.lst_nid, d)}


class MCFEKPABPairingPerClient(MCFEKPABPairing):
    """Crypto operations for multi-client key-policy attribute-based IPFE over pairings with an attribute set for each
    client (NPP22 Sec. 5.4; **known-insecure**, see the module documentation).

    Each client encrypts for a set of attributes of its own; a decryption needs the ciphertexts of all clients under one
    tag, and the set of every client must satisfy the policy of the key. The public parameters and the configuration
    are those of :class:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing`.
    """

    H_DIM = H_DIM
    _pp_domain = _PP_DOMAIN
    _scheme_id = _SCHEME_ID
    _ClientKey = PCKPABClientKey
    _FunctionalKey = PCKPABFunctionalKey
    _Ciphertext = PCKPABCiphertext
    #: The sets of attributes of the clients are their own: only the tag is common.
    _CLAIMS = ("label",)

    def _check_extra_key_fields(self, ek) -> None:
        _check_length(ek.theta_h5, self.H_DIM, "theta_h5")

    def _attribute_claim(self, labels: list):
        """``None``: the tag does not depend on the set of attributes, and a ciphertext carries no digest of it."""
        return None

    def _complete_ipfe(self, c_ipfe: list) -> list:
        """``c_i,ipfe = omega p_i H_i^(1) + omega' p_i H_i^(2) + psi_i h_i,3 + theta_i h_i,5``: the fifth coordinate
        ``theta_i`` is the key's ``theta_h5``, multiplied in."""
        return [element * fifth for element, fifth in zip(c_ipfe, self._key_elements("theta_h5"))]

    def _unsatisfied(self, nid, attributes: list) -> FEKeyError:
        return FEKeyError("the attributes {} of the ciphertext of {} do not satisfy the policy of the functional key"
                          .format(clip(repr(attributes), 120), nid))

    def _other_attributes(self, held: dict, attributes: dict, where: tuple, first: tuple) -> FEValidationError:
        nid = next(nid for nid in self.lst_nid if held[nid] != attributes[nid])
        return FEValidationError(
            "layer {}, element {}: the attributes {} of {} are not those of layer {}, element {}, {}: the elements "
            "of an array are for one set of attributes for each client".format(
                where[0], where[1], clip(repr(held[nid]), 120), nid, first[0], first[1],
                clip(repr(attributes[nid]), 120)))

    def encrypt(self, lst_pt: list, label, attributes=None) -> dict:
        """Encrypt one scalar for a tag and the client's own set of attributes.

        A client must encrypt once under a tag. That is necessary here, and not enough (see the module documentation,
        the attack of two tags and two keys): the points of the tag do not depend on the set of attributes, so two
        ciphertexts of one client under one tag, whatever their sets, give the difference of the plaintexts away. The
        scheme layer does not check it.

        Args:
            lst_pt: The one integer ``x_i`` with ``|x_i| <= bound_x``, in a list.
            label: The tag: a ``str``, or an element label of :mod:`pyfe4ai.schemes.mcfe.labels`. All clients of a
                decryption use the same one.
            attributes: The labels the ciphertext is for: any collection of valid labels (a ``set``, or a list with
                repeats), at most ``max_attributes`` distinct ones of at most ``MAX_LABEL_CHARS`` characters each. The
                ciphertext lists them sorted and each once. Each client uses a set of its own; the set of every client
                of a decryption must satisfy the policy. The empty collection is allowed: no policy is satisfied by it.

        Returns:
            The ciphertext, a dict.

        Raises:
            FEKeyError: Without the encryption key ``sk``.
            FEValidationError: For a plaintext out of bounds, a label that is not valid, no attributes, or
                attributes that are not a collection of valid labels within the limits.
        """
        return super().encrypt(lst_pt, label, attributes=attributes)

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict | None = None, label=None):
        """Decrypt the ciphertexts of all clients and recover the inner product.

        The order of the checks is that of
        :meth:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing.decrypt`: the bindings, the fields and the version
        markers, then that the ciphertexts are those of all clients under one tag, then the policy of the key and the
        shape of the records, then that the attributes of every client satisfy the policy, before any group element is
        read. The set of attributes of each client is its own, and each client reconstructs with its own set.

        Args:
            dct_ct: The ciphertexts, one for every client, by client id.
            dk: Functional decryption key.
            fusion_weight: Optional; if given it must be the ``y`` of the key, as a dict from every client to a list
                of one int.
            label: Optional; if given it must be the label that the ciphertexts claim (the ``label`` field of each,
                which all agree on): this checks the claim and not how the ciphertexts were made.

        Returns:
            ``<x, y>``.

        Raises:
            FEKeyError: If the attributes of a client do not satisfy the policy of the functional key; the message
                names the first such client, also one whose weight is 0.
            FEValidationError: For an object made under other public parameters, ciphertexts that are not those of
                every client and no other, or that claim different tags, or a label or weights that are not those of
                the key and ciphertexts, or an object that is malformed.
            ValueError: If the discrete log is not found: the key and the ciphertexts do not fit, whatever their
                fields say (two keys that were spliced, for one).
        """
        return super().decrypt(dct_ct, dk, fusion_weight=fusion_weight, label=label)

    def encrypt_lst_ndarray(self, lst_ndarray: list, label: str | None = None, attributes=None) -> list:
        """Encrypt arrays element by element, each element under a tag of its own, all for the client's own set.

        As :meth:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing.encrypt_lst_ndarray`. A client calls this
        once under a label; the scheme layer does not check it. *attributes* is the set of this client for every
        element, and may differ from that of other clients.
        """
        return super().encrypt_lst_ndarray(lst_ndarray, label=label, attributes=attributes)

    def check_ndarray_ct(self, dict_ndarray_ct: dict, label: str | None = None) -> None:
        """Check the ciphertext arrays of all clients as far as :meth:`decrypt_lst_ndarray_ct` can before it has a key.

        As :meth:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing.check_ndarray_ct`, except that the elements
        of a client are for one set of attributes and another client may have another: there is no digest to compare,
        and the error for a client whose elements differ names the client.
        """
        super().check_ndarray_ct(dict_ndarray_ct, label=label)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, dk: dict | None = None,
                               fusion_weight: dict | None = None, label: str | None = None) -> list:
        """Decrypt the ciphertext arrays of all clients, all elements with one preparation of the key.

        As :meth:`~pyfe4ai.schemes.mcfe.kp_ab_pairing.MCFEKPABPairing.decrypt_lst_ndarray_ct`, with a set of
        attributes for each client: the rows of the key that the decryption uses, and so the points that it prepares
        and the elements that it decodes, differ between clients. A client whose attributes do not satisfy the policy
        gives a ``FEKeyError`` that names it, before any point is decoded.
        """
        return super().decrypt_lst_ndarray_ct(dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label)
