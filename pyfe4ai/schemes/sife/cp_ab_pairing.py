"""
Ciphertext-Policy Attribute-Based Inner-Product Functional Encryption over Pairings
| "Inner-Product Functional Encryption with Fine-Grained Access Control"
| By Michel Abdalla, Dario Catalano, Romain Gay and Bogdan Ursu, ASIACRYPT 2020
| (ePrint 2020/577) -- Sec. 3.1 Fig. 1 with the predicate encoding of App. B.2
| (read-once monotone span programs).

* type:     public-key encryption of a vector under a monotone policy over
  attribute labels (:mod:`pyfe4ai.policy`); a functional key for a vector ``y``
  and a set of labels opens ``<x, y>`` when the labels satisfy the policy
* setting:  asymmetric pairing group (default BLS12-381)

``a = (1, a2)``, ``b = (1, b2)``, ``U in Z_p^{2 x eta}`` and one
``W_i in Z_p^{2 x 2}`` for each index ``i``: ``i < size`` is a slot of the
universe (a label and a copy, :meth:`~pyfe4ai.policy.attributes.Universe.index`)
and ``i = size + k`` is column ``k`` of a span program. The public parameters
hold ``[a]_1, [b]_2, [U^T a]_1, [W_i^T a]_1, [W_i b]_2``; ``msk = U``, and the
``W_i`` are dropped at the end of setup.

* Enc(P, x):  for the span program ``M`` of ``P``, with row ``j`` on slot
  ``idx_j``, and a random ``s``: ``c1 = [a s]_1``,
  ``c2[j] = [W_idx_j^T a s + sum_k M_jk W_{size+k}^T a s]_1`` and
  ``c3 = [x + U^T a s]_1``.
* KeyGen(S, y):  for a random ``r``: ``k1 = [b r]_2``,
  ``k0 = [U y + W_size b r]_2``, and ``k_attr[l#c] = [W_idx(l,c) b r]_2`` for
  every copy ``c`` of every label ``l`` in ``S``.
* Dec:  for ``sum_j w_j M_j = e_1`` over the rows of ``S``, put
  ``Y = prod_i c3[i]^y[i]``, ``F = prod_j c2[j]^w_j`` and
  ``K = k0 prod_j k_attr[row j]^w_j``. Then
  ``e(Y, g2) e(F, k1) / e(c1, K) = [<x, y>]_T``, since
  ``a^T W b = b^T W^T a``, and a bounded discrete log gives the result.

ACGU20 Thm. 3.2 proves the scheme of Fig. 1 selectively simulation secure
(SEL-SIM) from DDH in G1 and G2, for a secure predicate encoding. The proof is
selective: it covers a challenge (the policy and the plaintext) chosen before
the public parameters are seen. The policy of a ciphertext (``ct["policy"]``) and
the attribute list of a key (``dk["attributes"]``) are public: they travel in
the clear and hide nothing. Keys are trusted input to ``decrypt``: the
``pp_binding`` of a key tells the objects of two setups apart and is not a
signature. This implementation is unreviewed: it has had no cryptographic
review yet.

Deviations:

* Rows of attributes outside the policy are omitted from the ciphertext. Such
  a row has ``M_j = 0`` and no decryption reads it.
* A label on several rows of a policy takes one copy per row, up to
  ``max_copies`` (:func:`~pyfe4ai.policy.read_once.read_once_rows`): the paper's
  span programs are read-once, and here each slot is one of its attributes. A
  key holds every copy of each of its labels.
* The Decode of App. B.2 as printed lacks the minus sign on its first ``n``
  entries, and is used with it corrected: ``(-x_1 w_1, ..., -x_n w_n, 1, w_1,
  ..., w_n)``, which gives ``F`` its plus sign above; as printed,
  ``(0 ; C | K) d`` is not ``(1, 0, ..., 0)``.
* ``x_i`` and ``y_i`` may be negative within their bounds, not only in
  ``[0, B]``; the discrete log covers ``|<x, y>| <= eta bound_x bound_y``.
* ``k = 1`` (SXDH) only.
"""

from __future__ import annotations

import json
import logging
import re

import gmpy2 as gp

from pyfe4ai._validation import int_vector
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FEValidationError
from pyfe4ai.core.formats.pairing import CPABCiphertext
from pyfe4ai.core.formats.pairing import CPABFunctionalKey
from pyfe4ai.core.formats.pairing import CPABParameters
from pyfe4ai.core.formats.pairing import CPABRecord
from pyfe4ai.core.versions import record_matches
from pyfe4ai.policy import (
    DEFAULT_MAX_ROWS,
    POLICY_COMPILER,
    Formula,
    Universe,
    canonical_encoding,
    compile_formula,
    from_canonical,
    parse,
    policy_digest,
    read_once_rows,
)
from pyfe4ai.policy.attributes import clip
from pyfe4ai.primitives.pairing.bls12_381 import MAX_DLOG_BOUND
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, ZR, PairingGroup, multi_pair, pair
from pyfe4ai.primitives.pairing.pairing_backend import deserialize_many, multiexp, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.primitives.pairing.pairing_utils import mat_vec_mod
from pyfe4ai.primitives.pairing.pairing_utils import random_nonzero_zr
from pyfe4ai.primitives.pairing.pairing_utils import serialize_matrix_g
from pyfe4ai.primitives.pairing.pairing_utils import transpose_mod
from pyfe4ai.schemes.common.ac_binding import check_binding, pp_digest

logger = logging.getLogger(__name__)

_PP_DOMAIN = b"pyfe4ai/sife/cp-ab/pp-binding/v1|"
_DEFAULT_MAX_COPIES = 4
_DEFAULT_MAX_COLUMNS = 16
_GROUP_NAMES = {G1: "G1", G2: "G2"}

# a decimal string of these objects (bound_x, bound_y, the entries of y) has at most 16 digits, so it stands for
# an integer below 10**16, a little above 2**48; the pattern refuses a longer string before int() reads it
_DECIMAL = re.compile(r"0|-?[1-9][0-9]{0,15}")


def _positive_int(value, name: str) -> int:
    # type(value) is int: a bool or a float would be a second spelling of the same limit
    if type(value) is not int or value < 1:
        raise FEValidationError("{} must be a positive int".format(name))
    return value


def _check_dlog_limit(eta: int, bound_x: int, bound_y: int) -> None:
    if eta * bound_x * bound_y > MAX_DLOG_BOUND:
        raise FEValidationError(
            "eta * bound_x * bound_y must be at most 2**48, the limit of the discrete log"
        )


def _decimal(value, name: str) -> int:
    """The int that *value*, a decimal string in its one spelling, stands for."""
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise FEValidationError("{} must hold decimal strings of integers of at most 16 digits".format(name))
    return int(value)


def _check_length(items, length: int, name: str) -> None:
    if type(items) is not list or len(items) != length:
        raise FEValidationError("{} must be a list of {} entries".format(name, length))


def _rand_matrix(group: PairingGroup, rows: int, cols: int) -> list[list[gp.mpz]]:
    return [[gp.mpz(int(group.random(ZR))) for _ in range(cols)] for _ in range(rows)]


def _serialize_vec(group: PairingGroup, vec) -> list[str]:
    return [group.serialize(v).decode("utf-8") for v in vec]


def _pairs(items: list) -> list[list]:
    return [items[i:i + 2] for i in range(0, len(items), 2)]


def _elements(group: PairingGroup, items, kind, name: str) -> list:
    """*items* deserialized, all of the group *kind*, or ``FEValidationError`` naming the field *name*."""
    # the type tag before the element: charm reads almost any text as an element of some group, and
    # never returns from some strings tagged G2, which must not reach it from a field of G1 elements
    tag = "{}:".format(kind)
    if type(items) is not list or not all(type(item) is str and item.startswith(tag) for item in items):
        raise FEValidationError("{} must be a list of serialised elements of {}".format(name, _GROUP_NAMES[kind]))
    try:
        return deserialize_many(group, items)
    except ValueError as exc:
        raise FEValidationError("{} holds an element that does not deserialize".format(name)) from exc


def _record(cls, data, name: str):
    """The record view of the dict *data*."""
    if not isinstance(data, dict):
        raise FEValidationError("the {} must be a dict, got {}".format(name, type(data).__name__))
    return cls(data)


def _require_fields(record, name: str) -> None:
    for field in record.FIELDS:
        if field not in record:
            raise FEValidationError("the {} has no field {}".format(name, field))


class SIFECPABPairingKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for ciphertext-policy attribute-based IPFE over pairings."""

    _scheme_type = CryptoCONST.TYPE_SIFE_CP_AB

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the attribute-based IPFE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``universe`` (the
                attribute labels, required), ``eta``, ``bound_x``, ``bound_y``,
                ``max_copies`` (copies of each label, default 4), ``max_columns``
                and ``max_rows`` (limits of a policy's span program, default 16
                and 64) and ``pairing_group_param`` (asymmetric group, default
                ``"BLS12-381"``).

        Raises:
            FEConfigError: Without a ``universe``.
            FEValidationError: For a limit that is not a positive int, or
                ``eta * bound_x * bound_y`` above the discrete-log limit.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        if config.get("universe") is None:
            raise FEConfigError("the attribute-based IPFE needs the config option universe")
        self.eta = _positive_int(config.get("eta", CryptoCONST.SIFE_DEFAULT_ETA), "eta")
        self.bound_x = _positive_int(config.get("bound_x", 16), "bound_x")
        self.bound_y = _positive_int(config.get("bound_y", 16), "bound_y")
        self.max_copies = _positive_int(config.get("max_copies", _DEFAULT_MAX_COPIES), "max_copies")
        self.max_columns = _positive_int(config.get("max_columns", _DEFAULT_MAX_COLUMNS), "max_columns")
        self.max_rows = _positive_int(config.get("max_rows", DEFAULT_MAX_ROWS), "max_rows")
        _check_dlog_limit(self.eta, self.bound_x, self.bound_y)
        self.universe = Universe(config["universe"], self.max_copies)
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
            "bound_y": str(self.bound_y), "universe": self.universe.encoding(),
            "max_columns": self.max_columns, "max_rows": self.max_rows,
            "pairing_group_param": self.pairing_group_param, "policy_compiler": POLICY_COMPILER,
        })

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                CPABRecord.new(
                    sec_param=self.sec_param,
                    eta=self.eta,
                    bound_x=str(self.bound_x),
                    bound_y=str(self.bound_y),
                    universe=self.universe.encoding(),
                    max_columns=self.max_columns,
                    max_rows=self.max_rows,
                    pairing_group_param=self.pairing_group_param,
                    policy_compiler=POLICY_COMPILER,
                ).to_dict(),
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))

    def setup(self) -> None:
        group, order = self.group, self.order
        g1 = group.random(G1)
        g2 = group.random(G2)

        a = [gp.mpz(1), gp.mpz(int(random_nonzero_zr(group)))]
        b = [gp.mpz(1), gp.mpz(int(random_nonzero_zr(group)))]
        u = _rand_matrix(group, 2, self.eta)
        w = [_rand_matrix(group, 2, 2) for _ in range(self.universe.size + self.max_columns)]

        ua = mat_vec_mod(transpose_mod(u), a, order)
        w_a = [value for w_i in w for value in mat_vec_mod(transpose_mod(w_i), a, order)]
        w_b = [value for w_i in w for value in mat_vec_mod(w_i, b, order)]

        body = CPABParameters.new(
            eta=self.eta,
            bound_x=str(self.bound_x),
            bound_y=str(self.bound_y),
            universe=self.universe.encoding(),
            max_columns=self.max_columns,
            max_rows=self.max_rows,
            pairing_group_param=self.pairing_group_param,
            policy_compiler=POLICY_COMPILER,
            universe_digest=self.universe.digest(),
            g1=group.serialize(g1).decode("utf-8"),
            g2=group.serialize(g2).decode("utf-8"),
            a=_serialize_vec(group, pow_many(g1, a, group)),
            b=_serialize_vec(group, pow_many(g2, b, group)),
            ua=_serialize_vec(group, pow_many(g1, ua, group)),
            w_a=serialize_matrix_g(group, _pairs(pow_many(g1, w_a, group))),
            w_b=serialize_matrix_g(group, _pairs(pow_many(g2, w_b, group))),
        ).to_dict()
        self.pp = CPABParameters.extend(body, pp_binding=pp_digest(body, _PP_DOMAIN)).to_dict()
        self.msk = {"u": [[gp.digits(value) for value in row] for row in u]}
        logger.info("SIFE attribute-based IPFE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str = "nid_default", **kwargs) -> dict | None:
        """Public-key scheme: encryptors need the public parameters only."""
        return None

    def _held_labels(self, attributes) -> list[str]:
        """The sorted distinct labels of *attributes*, all of the universe."""
        if isinstance(attributes, (str, bytes, bytearray)):
            raise FEValidationError("attributes must be a collection of labels, not a bare {}".format(
                type(attributes).__name__))
        try:
            items = list(attributes)
        except TypeError as exc:
            raise FEValidationError("attributes must be a collection of labels") from exc
        for label in items:
            if label not in self.universe:
                raise FEValidationError("attribute label {} is not in the universe".format(clip(repr(label))))
        return sorted(set(items))

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given weights and labels.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (list
                of int) and ``attributes`` (the labels the key holds).

        Raises:
            FEKeyError: Before :meth:`setup`, or without credentials.
            FEValidationError: For weights out of bounds or labels outside the
                universe.
        """
        if self.msk is None:
            raise FEKeyError("no master key: run setup() before deriving a functional key")
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for attribute-based IPFE decryption key generation")
        y = int_vector(credentials.get("fusion_weight"), "fusion weight", length=self.eta, bound=self.bound_y)
        attributes = self._held_labels(credentials.get("attributes"))

        group, pp, size = self.group, self.pp, self.universe.size
        g2 = group.deserialize(pp["g2"].encode("utf-8"))
        b = deserialize_many(group, pp["b"])
        w_b_size = deserialize_many(group, pp["w_b"][size])
        u = [[gp.mpz(value) for value in row] for row in self.msk["u"]]
        u_y = mat_vec_mod(u, [gp.mpz(value) for value in y], self.order)

        r = random_nonzero_zr(group)
        k1 = [elem ** r for elem in b]
        k0 = [multiexp([g2, elem], [u_y_t, r], group) for u_y_t, elem in zip(u_y, w_b_size)]
        k_attr = {}
        for label in attributes:
            for copy in range(1, self.universe.max_copies + 1):
                entry = deserialize_many(group, pp["w_b"][self.universe.index(label, copy)])
                k_attr["{}#{}".format(label, copy)] = _serialize_vec(group, [elem ** r for elem in entry])

        return CPABFunctionalKey.new(
            attributes=attributes,
            y=[str(value) for value in y],
            k1=_serialize_vec(group, k1),
            k0=_serialize_vec(group, k0),
            k_attr=k_attr,
            pp_binding=pp["pp_binding"],
        ).to_dict()


class SIFECPABPairing(IPFEAbsCrypto):
    """Crypto operations for ciphertext-policy attribute-based IPFE over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the attribute-based IPFE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters);
                encryption is public-key.

        Raises:
            FEKeyError: Without public parameters.
            FEValidationError: For public parameters that do not agree with
                their ``pp_binding``, or whose fields do not agree with each
                other.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        pp = self.keys.get("pp")
        if pp is None:
            raise FEKeyError("no public parameters: keys needs the entry pp")
        self._read_parameters(pp)
        self.pp = pp
        try:
            self.group = PairingGroup(pp["pairing_group_param"])
        except ImportError:     # no backend for a group that may exist: the installation, not the parameters
            raise
        except Exception as exc:    # charm raises AssertionError or its own Error for a name it does not know
            raise FEValidationError("pairing_group_param {} is not a pairing group".format(
                clip(repr(pp["pairing_group_param"])))) from exc
        check_asymmetric(self.group)
        self.order = gp.mpz(int(self.group.order()))

    def _read_parameters(self, pp: dict) -> None:
        """Check *pp*, which comes from outside, and keep what its fields say.

        The digest binds the fields to each other and nothing to a key
        authority, so the lengths are checked too: they bound what later
        calls read, ``max_copies`` of a hostile ``universe`` among it.
        """
        parameters = _record(CPABParameters, pp, "public parameters")
        _require_fields(parameters, "public parameters")
        if pp_digest(pp, _PP_DOMAIN) != parameters.pp_binding:
            raise FEValidationError("the public parameters do not agree with their pp_binding")
        if parameters.policy_compiler != POLICY_COMPILER:
            raise FEValidationError("policy compiler {} is not this version's {!r}".format(
                clip(repr(parameters.policy_compiler)), POLICY_COMPILER))
        if type(parameters.pairing_group_param) is not str:
            raise FEValidationError("pairing_group_param must be a str, got {}".format(
                type(parameters.pairing_group_param).__name__))
        universe = Universe.from_encoding(parameters.universe)
        if parameters.universe_digest != universe.digest():
            raise FEValidationError("the universe_digest of the public parameters is not the digest of their universe")

        self.eta = _positive_int(parameters.eta, "eta")
        self.bound_x = _positive_int(_decimal(parameters.bound_x, "bound_x"), "bound_x")
        self.bound_y = _positive_int(_decimal(parameters.bound_y, "bound_y"), "bound_y")
        self.max_columns = _positive_int(parameters.max_columns, "max_columns")
        self.max_rows = _positive_int(parameters.max_rows, "max_rows")
        _check_dlog_limit(self.eta, self.bound_x, self.bound_y)
        _check_length(parameters.a, 2, "a")
        _check_length(parameters.b, 2, "b")
        _check_length(parameters.ua, self.eta, "ua")
        for name in ("w_a", "w_b"):
            table = getattr(parameters, name)
            _check_length(table, universe.size + self.max_columns, name)
            for entry in table:
                _check_length(entry, 2, name)
        self.universe = universe
        self.max_policy_length = self.max_rows * (max(len(label) for label in universe.labels) + 400)

    def _check_policy_length(self, policy: str) -> None:
        """Refuse a policy text of more than ``max_policy_length`` characters.

        The limit is ``max_rows`` times the longest label of the universe plus a
        margin per row, and it counts the characters of the canonical encoding.
        That encoding is JSON, which writes ``"`` and ``\\`` of a label as two
        characters each, so a policy over labels made mostly of those can be
        refused at encryption although its labels fit: on purpose, because the
        limit bounds the text that a ciphertext carries and a decryption reads.
        """
        if len(policy) > self.max_policy_length:
            raise FEValidationError(
                "the policy has {} characters, more than the limit of {} "
                "(max_rows * (longest label of the universe + 400))".format(len(policy), self.max_policy_length)
            )

    def _compile(self, formula: Formula):
        """The span program of *formula* and the (label, copy) of its rows, under the limits of the parameters."""
        msp = compile_formula(formula, int(self.order), max_rows=self.max_rows)
        if msp.num_cols > self.max_columns:
            raise FEValidationError("the policy needs {} columns, more than the limit max_columns = {}".format(
                msp.num_cols, self.max_columns))
        for label in msp.row_labels:
            if label not in self.universe:
                raise FEValidationError("attribute label {} of the policy is not in the universe".format(
                    clip(repr(label))))
        return msp, read_once_rows(msp, self.universe.max_copies)

    def encrypt(self, lst_pt: list, policy=None) -> dict:
        """Encrypt a vector under a policy.

        Args:
            lst_pt: The ``eta`` integers ``x`` with ``|x_i| <= bound_x``.
            policy: A :class:`~pyfe4ai.policy.formula.Formula` or its text
                (:func:`~pyfe4ai.policy.encoding.parse`).

        Raises:
            FEValidationError: For a plaintext out of bounds, no policy, or a
                policy outside the limits of the public parameters.
        """
        x = int_vector(lst_pt, "plaintext", length=self.eta, bound=self.bound_x)
        if isinstance(policy, str):
            policy = parse(policy)
        if not isinstance(policy, Formula):
            raise FEValidationError("the policy must be a Formula or its text, got {}".format(type(policy).__name__))
        # compile first: it bounds the leaves of a formula built from shared sub-formulas
        msp, rows = self._compile(policy)
        encoding = canonical_encoding(policy)
        self._check_policy_length(encoding)     # no decryption would read it

        group, order, pp = self.group, int(self.order), self.pp
        size = self.universe.size
        indices = [self.universe.index(label, copy) for label, copy in rows]
        columns = [[(k, entry) for k, entry in enumerate(row) if entry != 0] for row in msp.matrix]
        used = set(indices) | {size + k for row in columns for k, _ in row}
        w_a = {index: _elements(group, pp["w_a"][index], G1, "w_a") for index in used}

        s = int(random_nonzero_zr(group))
        c1 = [multiexp([elem], [s], group) for elem in _elements(group, pp["a"], G1, "a")]
        c2 = []
        for index, row in zip(indices, columns):
            exps = [s] + [(s * entry) % order for _, entry in row]
            bases = [w_a[index]] + [w_a[size + k] for k, _ in row]
            c2.append([multiexp([base[t] for base in bases], exps, group) for t in (0, 1)])
        g1 = _elements(group, [pp["g1"]], G1, "g1")[0]
        ua = _elements(group, pp["ua"], G1, "ua")
        c3 = [multiexp([g1, ua_i], [x_i % order, s], group) for x_i, ua_i in zip(x, ua)]

        return CPABCiphertext.new(
            policy=encoding,
            policy_digest=policy_digest(policy),
            policy_compiler=POLICY_COMPILER,
            c1=_serialize_vec(group, c1),
            c2=serialize_matrix_g(group, c2),
            c3=_serialize_vec(group, c3),
            pp_binding=pp["pp_binding"],
        ).to_dict()

    def _check_key(self, key: CPABFunctionalKey) -> list[int]:
        """Check the shape of *key*, and return its weights ``y`` as ints."""
        _check_length(key.k1, 2, "k1")
        _check_length(key.k0, 2, "k0")
        if type(key.k_attr) is not dict:
            raise FEValidationError("k_attr must be a dict from label#copy to two group elements")
        for entry in key.k_attr.values():
            _check_length(entry, 2, "k_attr")
        if type(key.attributes) is not list or not all(type(label) is str for label in key.attributes):
            raise FEValidationError("attributes must be a list of labels")
        _check_length(key.y, self.eta, "y")
        y = [_decimal(value, "y") for value in key.y]
        if any(abs(value) > self.bound_y for value in y):
            raise FEValidationError("y has an entry above the limit bound_y = {}".format(self.bound_y))
        return y

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: list | None = None):
        """Decrypt a ciphertext and recover the inner product.

        The bindings are checked before the policy is read, and the policy
        before any group element is.

        Args:
            dct_ct: Ciphertext.
            dk: Functional decryption key.
            fusion_weight: Optional; if given it must be the ``y`` of the key.

        Raises:
            FEKeyError: If the attributes of the key do not satisfy the policy.
            FEValidationError: For an object made under other public
                parameters, or one that is malformed.
            ValueError: If the discrete log is not found: the key and the
                ciphertext do not fit, whatever their fields say.
        """
        ct = _record(CPABCiphertext, dct_ct, "ciphertext")
        key = _record(CPABFunctionalKey, dk, "functional key")

        check_binding(self.pp, ct, "ciphertext")
        check_binding(self.pp, key, "functional key")
        _require_fields(ct, "ciphertext")
        _require_fields(key, "functional key")
        if ct.policy_compiler != POLICY_COMPILER:
            raise FEValidationError("policy compiler {} is not this version's {!r}".format(
                clip(repr(ct.policy_compiler)), POLICY_COMPILER))

        if type(ct.policy) is not str:
            raise FEValidationError("the policy of a ciphertext must be a str")
        self._check_policy_length(ct.policy)
        formula = from_canonical(ct.policy)
        if policy_digest(formula) != ct.policy_digest:
            raise FEValidationError("the policy_digest of the ciphertext is not the digest of its policy")
        msp, rows = self._compile(formula)
        _check_length(ct.c1, 2, "c1")
        _check_length(ct.c2, msp.num_rows, "c2")
        for row in ct.c2:
            _check_length(row, 2, "c2")
        _check_length(ct.c3, self.eta, "c3")
        y = self._check_key(key)
        if fusion_weight is not None:
            if int_vector(fusion_weight, "fusion weight", length=self.eta) != y:
                raise FEValidationError("the fusion weight is not the y of the functional key")

        weights = msp.reconstruction(key.attributes)
        if weights is None:
            raise FEKeyError("the attributes {} of the functional key do not satisfy the policy".format(
                clip(repr(key.attributes), 120)))
        names = ["{}#{}".format(*rows[j]) for j in weights]
        for name in names:
            if name not in key.k_attr:
                raise FEValidationError(
                    "k_attr has no entry {}: the attributes and the k_attr of the key disagree".format(
                        clip(repr(name)))
                )

        group, order, pp = self.group, int(self.order), self.pp
        c1 = _elements(group, ct.c1, G1, "c1")
        c2 = [_elements(group, ct.c2[j], G1, "c2") for j in weights]
        c3 = _elements(group, ct.c3, G1, "c3")
        k1 = _elements(group, key.k1, G2, "k1")
        k0 = _elements(group, key.k0, G2, "k0")
        k_attr = [_elements(group, key.k_attr[name], G2, "k_attr") for name in names]
        g1 = _elements(group, [pp["g1"]], G1, "g1")[0]
        g2 = _elements(group, [pp["g2"]], G2, "g2")[0]

        exps = list(weights.values())
        folded_y = multiexp(c3, [value % order for value in y], group)
        folded_c = [multiexp([row[t] for row in c2], exps, group) for t in (0, 1)]
        folded_k = [multiexp([k0[t]] + [entry[t] for entry in k_attr], [1] + exps, group) for t in (0, 1)]
        c1_inv = [multiexp([elem], [order - 1], group) for elem in c1]
        target = multi_pair([folded_y, folded_c[0], folded_c[1], c1_inv[0], c1_inv[1]],
                            [g2, k1[0], k1[1], folded_k[0], folded_k[1]])
        return bounded_discrete_log_gt(group, pair(g1, g2), target, self.eta * self.bound_x * self.bound_y)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for the attribute-based IPFE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for the attribute-based IPFE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError("ndarray helper is not implemented for the attribute-based IPFE")
