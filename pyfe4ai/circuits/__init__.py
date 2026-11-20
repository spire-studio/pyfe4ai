"""Boolean circuits ``C(w, x)`` for the bounded-key schemes (``bcfe``).

The key holder's secret parameter ``w`` and the encryptor's data ``x`` are
bit strings; a public circuit computes ``C(w, x)``, and a key for ``w`` lets
its holder learn ``C(w, x)`` from a ciphertext of ``x``. The submodules are

* :mod:`~pyfe4ai.circuits.circuit` -- :class:`~pyfe4ai.circuits.circuit.Circuit`: the gate list (XOR,
  AND, INV), the canonical encoding ``CIRC v1`` and its digest, the guard bit
  of SS10 (Sahai and Seyalioglu, CCS 2010) as :meth:`~pyfe4ai.circuits.circuit.Circuit.guard`, the wire
  kinds and garbling costs (:meth:`~pyfe4ai.circuits.circuit.Circuit.kinds`, :meth:`~pyfe4ai.circuits.circuit.Circuit.stats`)
  and the plain evaluation;
* :mod:`~pyfe4ai.circuits.builder` -- :class:`~pyfe4ai.circuits.builder.Builder`: gates with constant
  folding and structural hashing, word arithmetic (adders, shift-and-add and
  radix-4 Booth multiplication, comparisons, ReLU, argmax);
* :mod:`~pyfe4ai.circuits.layout` -- :class:`~pyfe4ai.circuits.layout.Field` and the
  :class:`~pyfe4ai.circuits.layout.KeyLayout`, :class:`~pyfe4ai.circuits.layout.DataLayout` and :class:`~pyfe4ai.circuits.layout.OutputLayout` that map
  values to bits and back;
* :mod:`~pyfe4ai.circuits.library` -- ready circuits
  (:func:`~pyfe4ai.circuits.library.linear_threshold`, :func:`~pyfe4ai.circuits.library.linear_score`, :func:`~pyfe4ai.circuits.library.mlp`,
  :func:`~pyfe4ai.circuits.library.argmax_scores`) as :class:`~pyfe4ai.circuits.library.ParamCircuit` objects bundling the
  circuit and its layouts.

Garbling costs follow the half-gates scheme of Zahur, Rosulek and Evans
(EUROCRYPT 2015, ZRE15): XOR and INV free, two table entries for an AND of
two labelled wires, one for an AND with a garbler-known input. The kind of
every wire -- labelled or garbler-known -- depends on the circuit and the
mode of the data wires only, never on the values of ``w`` or ``x``, so
the layout and the number of a ciphertext's gate tables do not depend on
``w`` or ``x`` (their contents do; the garbling hides them, as the scheme's
security claim states).

Pure Python: the standard library, NumPy (layouts accept NumPy integer
arrays) and :mod:`pyfe4ai.core.exceptions`.
"""

from pyfe4ai.circuits.builder import MUL_METHODS, Builder
from pyfe4ai.circuits.circuit import (
    AND,
    CIRCUIT_DIGEST_DOMAIN,
    CIRCUIT_MAGIC,
    CIRCUIT_VERSION,
    INV,
    MODES,
    WIRE_K,
    WIRE_L,
    XOR,
    Circuit,
)
from pyfe4ai.circuits.layout import DataLayout, Field, KeyLayout, Layout, OutputLayout
from pyfe4ai.circuits.library import (
    ParamCircuit,
    argmax_scores,
    linear_acc_bits,
    linear_score,
    linear_threshold,
    mlp,
)

__all__ = [
    "AND",
    "Builder",
    "CIRCUIT_DIGEST_DOMAIN",
    "CIRCUIT_MAGIC",
    "CIRCUIT_VERSION",
    "Circuit",
    "DataLayout",
    "Field",
    "INV",
    "KeyLayout",
    "Layout",
    "MODES",
    "MUL_METHODS",
    "OutputLayout",
    "ParamCircuit",
    "WIRE_K",
    "WIRE_L",
    "XOR",
    "argmax_scores",
    "linear_acc_bits",
    "linear_score",
    "linear_threshold",
    "mlp",
]
