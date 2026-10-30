"""Monotone access policies: formulas over attribute labels, and their span programs.

A policy is a formula built from :class:`~pyfe4ai.policy.formula.Leaf`,
:class:`~pyfe4ai.policy.formula.And`, :class:`~pyfe4ai.policy.formula.Or` and
:class:`~pyfe4ai.policy.formula.Threshold` nodes; a leaf is an attribute label
(:mod:`pyfe4ai.policy.attributes`). The submodules are

* :mod:`~pyfe4ai.policy.formula` -- the node types, :func:`~pyfe4ai.policy.formula.evaluate`
  and :func:`~pyfe4ai.policy.formula.validate_formula`;
* :mod:`~pyfe4ai.policy.attributes` -- attribute labels, the
  :class:`~pyfe4ai.policy.attributes.Universe` of labels a setup covers and
  the hash of a label to ``Z_q``;
* :mod:`~pyfe4ai.policy.encoding` -- the text syntax (:func:`~pyfe4ai.policy.encoding.parse`),
  the canonical encoding and the policy digest;
* :mod:`~pyfe4ai.policy.msp` -- compiling a formula to a monotone span program
  (nested Shamir sharing with the Lewko-Waters rules), its shares and its
  reconstruction coefficients;
* :mod:`~pyfe4ai.policy.zero_one` -- the same for formulas rewritten so that
  every reconstruction coefficient is 0 or 1;
* :mod:`~pyfe4ai.policy.read_once` -- one copy of a label for each row that
  carries it;
* :mod:`~pyfe4ai.policy.structure` -- the minimal qualified and the maximal
  unqualified sets of a short list of holders, and the Q2 check.

The names in ``__all__`` are imported here; the submodules define them. Pure
Python: the standard library and :mod:`pyfe4ai.core.exceptions` only.
"""

from pyfe4ai.policy.attributes import Universe, attribute_to_zq, validate_label
from pyfe4ai.policy.encoding import POLICY_COMPILER, canonical_encoding, from_canonical, parse, policy_digest
from pyfe4ai.policy.formula import (
    DEFAULT_MAX_ROWS,
    MAX_DEPTH,
    And,
    Formula,
    Leaf,
    Or,
    Threshold,
    evaluate,
    validate_formula,
)
from pyfe4ai.policy.msp import MSP, compile_formula
from pyfe4ai.policy.read_once import read_once_rows
from pyfe4ai.policy.structure import MAX_HOLDERS, q2_violation, qualified_families
from pyfe4ai.policy.zero_one import compile_zero_one, threshold_to_cnf

__all__ = [
    "And",
    "DEFAULT_MAX_ROWS",
    "Formula",
    "Leaf",
    "MAX_DEPTH",
    "MAX_HOLDERS",
    "MSP",
    "Or",
    "POLICY_COMPILER",
    "Threshold",
    "Universe",
    "attribute_to_zq",
    "canonical_encoding",
    "compile_formula",
    "compile_zero_one",
    "evaluate",
    "from_canonical",
    "parse",
    "policy_digest",
    "q2_violation",
    "qualified_families",
    "read_once_rows",
    "threshold_to_cnf",
    "validate_formula",
    "validate_label",
]
