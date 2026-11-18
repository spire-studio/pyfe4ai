"""Allocations of the garbling and slot operations are bounded by the bytes
the caller supplies.

* ``CIRC v1`` allows fewer than 2^26 wires (``m + n + n_gates``,
  :data:`pyfe4ai.primitives.garbling.MAX_CIRCUIT_WIRES`): a header of a few
  dozen bytes that claims more is refused at once by the three decoders
  (:func:`~pyfe4ai.primitives.garbling.parse_circuit`,
  :meth:`pyfe4ai.circuits.Circuit.decode`, the native one behind
  ``gc_garble``), by the constructor, ``guard()`` and the builder, and by the
  backend operations that size per-wire state from the counts
  (``gc_garble``, ``gc_evaluate``, ``bcfe_encrypt_slot``,
  ``bcfe_decrypt_slot``), in both implementations, before anything is
  sized by the counts.
* ``bcfe_new_slot`` / ``bcfe_place_slot`` check ``m`` and the packed ``w``
  (``ceil(m / 8)`` bytes, zero padding) before they draw a random byte, so
  the ``64 (m + 1) + 16`` bytes of a new slot are backed by input bytes.

"Refused at once" is checked as: ``ValueError`` within a second (a sized
allocation of 2^30 wires would take far longer and gigabytes).
"""

from __future__ import annotations

import struct
import time

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.circuits import Builder, Circuit
from pyfe4ai.circuits import builder as builder_module
from pyfe4ai.circuits import circuit as circuit_module
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.primitives import garbling as gc

BACKENDS = ["python", pytest.param("native", marks=pytest.mark.skipif(
    not native_backend.available(), reason="native extension not built"))]
XOR, AND, INV = 1, 2, 3
SALT, PP, NONCE, SETUP = bytes(56), bytes(32), bytes(16), bytes(range(32))
#: Counts at and far above the bound (below 2^32 - 8, so that guard()'s wire numbers still fit a u32 field).
HUGE = (1 << 26, (1 << 26) + 5, 1 << 30, (1 << 32) - 8)


def raw(m: int, n: int, gates: list, outputs: list) -> bytes:
    """A ``CIRC v1`` encoding with any counts in its header."""
    enc = b"PFE6CIRC" + bytes([1]) + struct.pack("<4I", m, n, len(gates), len(outputs))
    enc += b"".join(struct.pack("<BII", op, a, b) for op, a, b in gates)
    return enc + struct.pack("<%dI" % len(outputs), *outputs)


def guarded_header(m: int, n: int) -> bytes:
    """``guard(C)`` of a gate-free ``C`` (output: key wire 0) with *m* key and *n* data bits: 47 bytes for m = 1."""
    m_prime, ng = m + 1, m + 1 + n
    return raw(m_prime, n, [(INV, m, 0), (AND, 0, ng)], [ng + 1])


def quickly_refused(call, error=ValueError) -> None:
    start = time.perf_counter()
    with pytest.raises(error):
        call()
    assert time.perf_counter() - start < 1.0


def test_the_bound_is_2_to_the_26_everywhere():
    assert gc.MAX_CIRCUIT_WIRES == gc.MAX_WIRES == 1 << 26
    assert circuit_module.MAX_CIRCUIT_WIRES == circuit_module.MAX_WIRES == 1 << 26
    if native_backend.available():  # the native decoder: 2^26 wires do not decode
        ext = native_backend.ext()
        with pytest.raises(ValueError, match="2\\^26"):
            ext.gc_garble(raw((1 << 26) - 1, 0, [(INV, 0, 0)], [0]), 0, SALT, b"")


@pytest.mark.parametrize("m,n", [(m, 1) for m in HUGE] + [(1, n) for n in HUGE])
def test_the_decoders_refuse_a_short_header_with_huge_counts(m, n):
    data = raw(m, n, [], [0])
    assert len(data) == 29
    quickly_refused(lambda: gc.parse_circuit(data))
    quickly_refused(lambda: Circuit.decode(data))
    if native_backend.available():
        ext = native_backend.ext()
        quickly_refused(lambda: ext.gc_garble(data, 0, SALT, b""))


@pytest.mark.parametrize("impl", BACKENDS)
@pytest.mark.parametrize("huge", HUGE)
def test_the_operations_refuse_a_short_header_with_huge_counts(impl, huge, monkeypatch):
    monkeypatch.setattr(native_backend, "_random_override", lambda k: bytes(k))
    with native_backend.backend(impl):
        ops = native_backend.ops()
        big_m = raw(huge, 0, [], [0])  # 29 bytes
        quickly_refused(lambda: ops.gc_garble(big_m, "hardwired", SALT, b""))
        quickly_refused(lambda: ops.gc_evaluate(big_m, "hardwired", SALT, b"", b"\x00", bytes(16), b""))
        big_n = guarded_header(1, huge)  # 47 bytes; hardwired data wires carry no labels at all
        assert len(big_n) == 47
        quickly_refused(lambda: ops.gc_evaluate(big_n, "hardwired", SALT, b"", b"\x00", bytes(32), b""))
        quickly_refused(lambda: ops.bcfe_encrypt_slot(big_n, "hardwired", PP, NONCE, 0, b"", bytes(96)))
        quickly_refused(lambda: ops.bcfe_decrypt_slot(big_n, "hardwired", PP, NONCE, 0, bytes(32), bytes(48), b"",
                                                      b"\x00", b"", bytes(64), b"\x00"))
        quickly_refused(lambda: ops.bcfe_decrypt_slot(guarded_header(huge, 0), "hardwired", PP, NONCE, 0,
                                                      bytes(32), bytes(48), b"", b"\x00", b"", bytes(64), b"\x00"))


def test_the_constructor_guard_and_builder_refuse_too(monkeypatch):
    with pytest.raises(ValueError, match="2\\^26"):
        Circuit(1 << 25, 1 << 25, [], [0])
    with pytest.raises(FEValidationError, match="2\\^26"):
        Builder(1 << 25, 1 << 25)
    # build() checks the circuit it makes (with a small bound standing in for 2^26)
    monkeypatch.setattr(builder_module, "MAX_WIRES", 7)
    b = Builder(2, 2)
    b.output(b.xor(b.and_(b.key(0), b.data(0)), b.and_(b.key(1), b.data(1))))
    with pytest.raises(FEValidationError, match="wires"):
        b.build()
    monkeypatch.setattr(builder_module, "MAX_WIRES", 16)
    assert b.build().n_wires == 7
    # guard() of a circuit at the bound
    monkeypatch.setattr(circuit_module, "MAX_WIRES", 8)
    c = Circuit(2, 2, [(XOR, 0, 2), (AND, 1, 3)], [4, 5])
    with pytest.raises(ValueError, match="2\\^26"):
        c.guard()


@pytest.mark.parametrize("impl", BACKENDS)
def test_slots_check_m_and_w_before_drawing(impl, monkeypatch):
    drawn = []

    def hook(k: int) -> bytes:
        drawn.append(k)
        return bytes(range(256)) * (k // 256) + bytes(range(k % 256))

    monkeypatch.setattr(native_backend, "_random_override", hook)
    with native_backend.backend(impl):
        ops = native_backend.ops()
        cases = [
            (1 << 20, b"\x01"),          # a huge m that one byte of w does not back
            (1 << 30, b"\x01"),          # m out of range
            (0, b""),
            (3, b"\x08"),                # a padding bit set
            (3, b"\x01\x00"),            # one byte too many
            (9, b"\x01"),                # one byte too few
        ]
        for m, w in cases:
            quickly_refused(lambda: ops.bcfe_new_slot(SETUP, 1, w, m=m))
            quickly_refused(lambda: ops.bcfe_place_slot(SETUP, 1, w, bytes(32 * 4), m=m))
        quickly_refused(lambda: ops.bcfe_new_slot(SETUP, 1, [1, 0, 1], m=3), TypeError)
        assert drawn == []
        # well-formed m and w: the draw happens, then the other inputs are checked
        quickly_refused(lambda: ops.bcfe_new_slot(SETUP[:31], 1, b"\x05", m=3))
        assert drawn == [64 * 4 + 16]
        quickly_refused(lambda: ops.bcfe_place_slot(SETUP, 1, b"\x05", bytes(32 * 3), m=3))
        assert drawn == [64 * 4 + 16, 16]
