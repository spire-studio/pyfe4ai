"""
Key-Policy Attribute-Based Inner-Product Functional Encryption over Pairings
| "Multi-Client Functional Encryption with Fine-Grained Access Control"
| By Ky Nguyen, Duong Hieu Phan and David Pointcheval, ASIACRYPT 2022
| (ePrint 2022/215) -- Sec. 4 Fig. 1 without the boxed components (the selective version),
| with the monotone span programs of :mod:`pyfe4ai.policy`.

* type:     public-key encryption of a vector under a set of attribute labels
  (:mod:`pyfe4ai.policy`); a functional key for a vector ``y`` and a policy opens ``<x, y>`` when the labels
  satisfy the policy
* setting:  asymmetric pairing group (default BLS12-381)

``(F, F*)`` are dual bases of ``G1^8`` and ``G2^8``, ``(H, H*)`` of ``G1^4`` and ``G2^4``
(:mod:`pyfe4ai.primitives.pairing.dpvs`), ``mu, z`` are in ``Z_q^*`` and ``S, U`` in ``(Z_q^*)^eta``. A label hashes
to the ``j`` in ``Z_q`` that stands for it (:func:`~pyfe4ai.policy.attributes.attribute_to_zq`). The public
parameters hold ``h1 + mu h2``, ``h3``, ``f1, f2, f3`` and ``[s_i + mu u_i]_1``; the master key is
``msk = (z, S, U, f1*, f2*, f3*, h1*, h2*, h3*)``, the last six as the elements of ``G2^8`` and ``G2^4`` that
the paper has, and the exponents of the bases and ``mu`` are dropped at the end of setup.

* KeyGen(y, P):  for the span program ``M`` of ``P`` (rows ``j`` with the labels of the leaves), a random ``a0`` and
  the shares ``a_j = (M (a0, r))_j``, and a random ``pi_j`` for each row: ``k*_j = (pi_j (j, 1), a_j z, 0, ..., 0)_F*``
  and ``k*_ipfe = (<S, y>, <U, y>, a0 z, 0)_H*``. The key is ``P``, ``y``, the ``k*_j`` and ``k*_ipfe``.
* Enc(x, S):  for a set of labels ``S``, random ``omega, psi`` and ``sigma_j`` for each label ``j`` of ``S``:
  ``c_j = (sigma_j (1, -j), psi, 0, ..., 0)_F``, ``t_i = [omega (s_i + mu u_i) + x_i]_1`` and
  ``c_ipfe = (omega, mu omega, psi, 0)_H = omega (h1 + mu h2) + psi h3``.
* Dec:  for ``S`` that satisfies ``P`` and the reconstruction coefficients ``w_j`` of ``M`` for the rows of ``S``:
  ``sum_j w_j e(c_j, k*_j) + sum_i e(t_i, [y_i]_2) - e(c_ipfe, k*_ipfe) = [<x, y>]_T``, since
  ``(1, -j) . (j, 1) = 0`` and ``sum_j w_j a_j = a0``, and a bounded discrete log gives the result.

NPP22 Thm. 7 proves the version with the boxed components (dual bases of dimensions that grow with ``eta``)
adaptively IND-CPA secure from SXDH, for one challenge ciphertext and any number of keys. The paper says that the
selective version, which is the scheme of this module, can be proved with similar techniques, and gives no proof of
it. Several challenge ciphertexts follow by a hybrid argument, as the scheme is a public-key scheme.

Ciphertexts are malleable and not authenticated, and the proof is against eavesdroppers only: anyone can multiply
``t_i`` by ``g1``, which adds ``y_i`` to what a key decrypts, or multiply two ciphertexts of one set of attributes,
which adds their inner products. Authenticate ciphertexts where tampering matters.

The policy of a key (``dk["policy"]``) and the attributes of a ciphertext (``ct["attributes"]``) are public: they
travel in the clear and hide nothing. Keys are trusted input to the decryption: the ``pp_binding`` of a key tells
the objects of two setups apart and is not a signature. This implementation is unreviewed: it has had no
cryptographic review yet.

Deviations:

* ``g1`` and ``g2`` are random generators drawn at setup, not those of the group description.
* ``y_i`` and ``x_i`` may be negative within their bounds, not only in ``[0, B]``; the discrete log covers
  ``|<x, y>| <= eta bound_x bound_y``.
* ``[y_i]_2`` (``m*_i`` of Fig. 1) is not part of the key: the decryption computes it from ``y``.
* A label is an attribute by its hash to ``Z_q``, where the paper takes the attributes in ``Z_q``. A label on several
  rows of a policy takes a fresh ``pi`` for each, as ``List-Att(A)`` of the paper, which allows repetitions.
* The policy is compiled to a span program by :func:`~pyfe4ai.policy.msp.compile_formula` (nested Shamir sharing),
  with at most ``max_rows`` rows, and a ciphertext carries at most ``max_attributes`` labels: limits of this
  implementation, not of the paper.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp

from pyfe4ai._validation import int_vector
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError
from pyfe4ai.core.formats.pairing import KPABFunctionalKey
from pyfe4ai.core.formats.pairing import KPABParameters
from pyfe4ai.core.formats.pairing import KPABRecord
from pyfe4ai.core.versions import record_matches
from pyfe4ai.policy import ATTR_HASH, DEFAULT_MAX_ROWS, POLICY_COMPILER, canonical_encoding, policy_digest
from pyfe4ai.primitives.pairing.dpvs import basis_vectors, combine, encode, random_dual_bases
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, ZR, PairingGroup, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr
from pyfe4ai.primitives.pairing.pairing_utils import serialize_matrix_g
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.ac_codec import check_dlog_limit as _check_dlog_limit
from pyfe4ai.schemes.common.ac_codec import positive_int as _positive_int
from pyfe4ai.schemes.common.ac_codec import serialize_vec as _serialize_vec
from pyfe4ai.schemes.common.kp_ab import DEFAULT_MAX_ATTRIBUTES, F_DIM, H_DIM
from pyfe4ai.schemes.common.kp_ab import compile_policy, key_row, read_policy

logger = logging.getLogger(__name__)

_PP_DOMAIN = b"pyfe4ai/sife/kp-ab/pp-binding/v1|"


class SIFEKPABPairingKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for key-policy attribute-based IPFE over pairings."""

    _scheme_type = CryptoCONST.TYPE_SIFE_KP_AB
    _config_only_record = True  # the record holds the configuration only (see ParameterCacheMixin)

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key-policy attribute-based IPFE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``eta``, ``bound_x``, ``bound_y``, ``max_rows`` (the
                rows a policy's span program may have, default 64), ``max_attributes`` (the labels a ciphertext
                may carry, default 64) and ``pairing_group_param`` (asymmetric group, default ``"BLS12-381"``).

        Raises:
            FEValidationError: For a limit that is not a positive int, or ``eta * bound_x * bound_y`` above the
                discrete-log limit, or a symmetric pairing group.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = _positive_int(config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA), "eta")
        self.bound_x = _positive_int(config.get("bound_x", 16), "bound_x")
        self.bound_y = _positive_int(config.get("bound_y", 16), "bound_y")
        self.max_rows = _positive_int(config.get("max_rows", DEFAULT_MAX_ROWS), "max_rows")
        self.max_attributes = _positive_int(config.get("max_attributes", DEFAULT_MAX_ATTRIBUTES), "max_attributes")
        _check_dlog_limit(self.eta, self.bound_x, self.bound_y)
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        self.pp = None
        self.msk = None
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def _param_verification(self, param: dict) -> bool:
        return record_matches(param, {
            "sec_param": self.sec_param, "eta": self.eta, "bound_x": str(self.bound_x),
            "bound_y": str(self.bound_y), "max_rows": self.max_rows, "max_attributes": self.max_attributes,
            "pairing_group_param": self.pairing_group_param, "policy_compiler": POLICY_COMPILER,
            "attr_hash": ATTR_HASH,
        })

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                KPABRecord.new(
                    sec_param=self.sec_param,
                    eta=self.eta,
                    bound_x=str(self.bound_x),
                    bound_y=str(self.bound_y),
                    max_rows=self.max_rows,
                    max_attributes=self.max_attributes,
                    pairing_group_param=self.pairing_group_param,
                    policy_compiler=POLICY_COMPILER,
                    attr_hash=ATTR_HASH,
                ).to_dict(),
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        group, order = self.group, self.order
        g1 = group.random(G1)
        g2 = group.random(G2)

        h, h_star = random_dual_bases(group, H_DIM, gp.mpz(1), order)
        f, f_star = random_dual_bases(group, F_DIM, gp.mpz(1), order)
        mu = gp.mpz(int(random_nonzero_zr(group)))
        z = gp.mpz(int(random_nonzero_zr(group)))
        s = [gp.mpz(int(random_nonzero_zr(group))) for _ in range(self.eta)]
        u = [gp.mpz(int(random_nonzero_zr(group))) for _ in range(self.eta)]

        body = KPABParameters.new(
            eta=self.eta,
            bound_x=str(self.bound_x),
            bound_y=str(self.bound_y),
            max_rows=self.max_rows,
            max_attributes=self.max_attributes,
            pairing_group_param=self.pairing_group_param,
            policy_compiler=POLICY_COMPILER,
            attr_hash=ATTR_HASH,
            g1=group.serialize(g1).decode("utf-8"),
            g2=group.serialize(g2).decode("utf-8"),
            h_mask=_serialize_vec(group, encode([1, mu], h[:2], g1, order, group)),
            h3=_serialize_vec(group, basis_vectors(h, [2], g1, group)[0]),
            f=serialize_matrix_g(group, basis_vectors(f, [0, 1, 2], g1, group)),
            t=_serialize_vec(group, pow_many(g1, [(s_i + mu * u_i) % order for s_i, u_i in zip(s, u)], group)),
        ).to_dict()
        self.pp = KPABParameters.extend(body, pp_binding=pp_digest(body, _PP_DOMAIN)).to_dict()
        # what the paper keeps: z, S, U and f*_1..3, h*_1..3 as elements of G2; F, H, F*, H* and mu stay behind
        self.msk = {
            "z": gp.digits(z),
            "s": [gp.digits(value) for value in s],
            "u": [gp.digits(value) for value in u],
            "f_star": basis_vectors(f_star, [0, 1, 2], g2, group),
            "h_star": basis_vectors(h_star, [0, 1, 2], g2, group),
        }
        logger.info("SIFE key-policy attribute-based IPFE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Public-key scheme: encryptors need the public parameters only."""
        return None

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given weights and policy.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list of int) and ``policy`` (a
                :class:`~pyfe4ai.policy.formula.Formula` or its text).

        Raises:
            FEKeyError: Before :meth:`setup`, or without credentials.
            FEValidationError: For credentials that are not a dict with both entries, weights out of bounds, a
                policy that does not parse or has more than ``max_rows`` rows.
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
        y = int_vector(credentials["fusion_weight"], "fusion weight", length=self.eta, bound=self.bound_y)
        formula = read_policy(credentials["policy"])
        # compile first: it bounds the leaves of a formula built from shared sub-formulas
        msp = compile_policy(formula, self.order, self.max_rows)

        group, order, msk = self.group, int(self.order), self.msk
        z = int(msk["z"])
        a0 = int(group.random(ZR))
        shares = msp.shares(a0, [int(group.random(ZR)) for _ in range(msp.num_cols - 1)])
        k_rows = [
            key_row(msk["f_star"], label, int(group.random(ZR)), share * z, order, group)
            for label, share in zip(msp.row_labels, shares)
        ]
        s_y = sum(int(s_i) * y_i for s_i, y_i in zip(msk["s"], y)) % order
        u_y = sum(int(u_i) * y_i for u_i, y_i in zip(msk["u"], y)) % order
        k_ipfe = combine(msk["h_star"], [s_y, u_y, a0 * z % order], group)

        return KPABFunctionalKey.new(
            policy=canonical_encoding(formula),
            policy_digest=policy_digest(formula),
            policy_compiler=POLICY_COMPILER,
            y=[str(value) for value in y],
            k_rows=serialize_matrix_g(group, k_rows),
            k_ipfe=_serialize_vec(group, k_ipfe),
            pp_binding=self.pp["pp_binding"],
        ).to_dict()
