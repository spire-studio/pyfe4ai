"""Known answers of the half-gates garbling and the native garbling against the reference.

* The KAT ``tests/bcfe/kat/gc_v1.json`` (written by ``tests/bcfe/gen_kat.py``
  from the Python reference; specification: 0.6 P1 plan, section 3): its
  answers against :func:`pyfe4ai.primitives.garbling.garble` /
  :func:`~pyfe4ai.primitives.garbling.evaluate` with the seed passed, and
  against the backend operations ``gc_garble`` / ``gc_evaluate`` of both
  implementations with the seed fed through the test hook
  ``pyfe4ai.backend._random_override``; the digests of its library circuits
  against what :mod:`pyfe4ai.circuits` builds today (a silent change of the
  builder fails here); its seeds, salts and inputs against their derivation
  from the case names; the file against what the generator writes.
* Native against reference on random circuits (both modes, random sizes,
  ``m``, ``n``, seeds and salts): a quick run of 50 circuits always, a large
  one (5,000 circuits up to 50,000 gates and the library circuits at
  realistic sizes) only with ``PYFE4AI_GC_CROSSCHECK=large``::

      PYFE4AI_GC_CROSSCHECK=large python -m pytest tests/bcfe/test_gc_kat.py -k large -q

  The large run is not marked ``extended``: that mark means "needs a pairing
  backend" (tests/conftest.py marks every other test ``core``, and the full
  lanes run both), so an environment variable selects it instead.
* The two decoders of ``CIRC v1`` (:func:`pyfe4ai.primitives.garbling.parse_circuit`
  and :meth:`pyfe4ai.circuits.Circuit.decode`) agree on the library circuits.
* ``gc_garble`` (it returns ``Delta``) and ``gc_evaluate`` are for tests and
  known answers only: no source of the schemes or the ML adapter names them.

The native tests are skipped without the extension.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import hashlib
import json
import math
import os
import random
from pathlib import Path

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai import circuits
from pyfe4ai.circuits import Circuit
from pyfe4ai.primitives import garbling as gc
from tests.bcfe import gen_kat

KAT_PATH = Path(__file__).resolve().parent / "kat" / "gc_v1.json"
KAT = json.loads(KAT_PATH.read_text(encoding="utf-8"))
CASES = KAT["cases"]
IDS = [case["name"] for case in CASES]
LIBRARY_CASES = [case for case in CASES if case["circuit"]["source"] == "library"]
MODES = ("hardwired", "labels")
FIELDS = ("delta", "key0", "data_active", "tables", "decode")
#: ``PYFE4AI_GC_CROSSCHECK=large`` runs the large cross-check.
LARGE = os.environ.get("PYFE4AI_GC_CROSSCHECK", "").strip().lower() == "large"

native = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
large = pytest.mark.skipif(not LARGE, reason="the large cross-check runs with PYFE4AI_GC_CROSSCHECK=large")


# ------------------------------------------------------------------ helpers

def bits(text: str) -> list:
    return [int(ch) for ch in text]


def bit_string(values) -> str:
    return "".join(str(v) for v in values)


def active_keys(garbling: dict, w) -> bytes:
    """``key0[j] xor w_j Delta`` for every key wire."""
    delta = int.from_bytes(garbling["delta"], "little")
    key0 = garbling["key0"]
    return b"".join((int.from_bytes(key0[16 * j:16 * j + 16], "little") ^ (delta if bit else 0)).to_bytes(16, "little")
                    for j, bit in enumerate(w))


def w_of(v: int, m: int) -> list:
    """The key bits of the integer *v*, wire 0 first."""
    return [(v >> j) & 1 for j in range(m)]


@functools.lru_cache(maxsize=None)
def _library(function: str, params_json: str) -> Circuit:
    return getattr(circuits, function)(**json.loads(params_json)).circuit


def library_circuit(function: str, params: dict) -> Circuit:
    """``pyfe4ai.circuits.<function>(**params).circuit`` (built once)."""
    return _library(function, json.dumps(params, sort_keys=True))


def garbled_encoding(case: dict) -> bytes:
    """The encoding of ``guard(C)`` the case garbles: stored, or built by the library
    (whose digests must be those of the KAT)."""
    info = case["circuit"]
    if info["source"] != "library":
        return bytes.fromhex(info["encoding"])
    c = library_circuit(info["function"], info["params"])
    assert c.digest().hex() == info["unguarded_digest"], (
        "pyfe4ai.circuits.{}({}) builds another circuit than the KAT's: a change of the builder; regenerate "
        "the KAT (tests/bcfe/gen_kat.py) only if the change is intended".format(info["function"], info["params"]))
    guarded = c.guard()
    assert guarded.digest().hex() == info["digest"]
    return guarded.encode()


def check_garbling(case: dict, res: dict) -> None:
    """*res* (``garble`` / ``gc_garble``) against the case's expected answers."""
    exp, info = case["expected"], case["circuit"]
    assert set(res) == set(FIELDS)
    assert all(type(res[f]) is bytes for f in FIELDS)
    assert res["delta"].hex() == exp["delta"] and res["delta"][0] & 1
    for field in ("key0", "data_active", "tables"):
        if field in exp:
            assert res[field].hex() == exp[field], field
        assert hashlib.sha256(res[field]).hexdigest() == exp[field + "_sha256"], field
    assert res["decode"].hex() == exp["decode"]
    assert len(res["tables"]) == 16 * exp["table_entries"]
    assert len(res["key0"]) == 16 * info["m"]
    assert len(res["data_active"]) == (16 * info["n"] if case["mode"] == "labels" else 0)


class CountingExtension:
    """The extension, counting the calls of its ``gc_*`` functions."""

    def __init__(self, ext) -> None:
        self._ext, self.calls = ext, collections.Counter()

    def __getattr__(self, name):
        attr = getattr(self._ext, name)
        if not name.startswith("gc_"):
            return attr

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return attr(*args, **kwargs)

        return counted


@contextlib.contextmanager
def seeded(seed: bytes, drawn: list = None):
    """The backend operations draw *seed* (``_random_override``; the request is recorded in *drawn*)."""

    def take(nbytes: int) -> bytes:
        if drawn is not None:
            drawn.append(nbytes)
        assert nbytes == len(seed)
        return seed

    before = native_backend._random_override
    native_backend._random_override = take
    try:
        yield
    finally:
        native_backend._random_override = before


# ------------------------------------------------------------------ the KAT file

def test_header():
    assert KAT["version"] == 1
    assert KAT["spec"] == "04_p1_plan section 3, 2026-11-19 rev. 2 (salted seed expansion)"
    dst = KAT["dst"]
    # the specification's strings, not only the code's constants
    assert dst["gate_hash"] == "PFE6GC02" and gc.GATE_HASH_DST == b"PFE6GC02"
    assert dst["garbling_prg"] == "PYFE4AI-V06-BCFE-GBPRG" and gc.PRG_DST == b"PYFE4AI-V06-BCFE-GBPRG"
    assert dst["circuit_digest"] == "PYFE4AI-V06-BCFE-CIRCUIT"
    assert circuits.CIRCUIT_DIGEST_DOMAIN == b"PYFE4AI-V06-BCFE-CIRCUIT"
    assert len(set(IDS)) == len(IDS)
    sources = collections.Counter((case["circuit"]["source"], case["mode"]) for case in CASES)
    assert sources[("hand", "hardwired")] == sources[("hand", "labels")] >= 12
    assert sources[("library", "hardwired")] == 3 and sources[("library", "labels")] == 2
    assert sources[("random", "hardwired")] == sources[("random", "labels")] == 1


def test_the_file_is_what_the_generator_writes():
    assert gen_kat.dumps(gen_kat.build_kat()) == KAT_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_seed_salt_and_inputs_are_derived_from_the_name(case):
    name = case["name"].encode()
    dst = {k: v.encode() for k, v in KAT["dst"].items()}
    info = case["circuit"]
    assert case["seed"] == hashlib.shake_256(dst["kat_seed"] + name).hexdigest(32)
    assert case["salt"] == hashlib.shake_256(dst["kat_salt"] + name).hexdigest(56)
    n, m = info["n"], info["m"] - 1  # the key bits of C; w ends with g = 0
    xs = hashlib.shake_256(dst["kat_x"] + name).digest((n + 7) // 8)
    ws = hashlib.shake_256(dst["kat_w"] + name).digest((m + 7) // 8)
    assert case["x"] == bit_string((xs[i // 8] >> (i % 8)) & 1 for i in range(n))
    assert case["w"] == bit_string((ws[i // 8] >> (i % 8)) & 1 for i in range(m)) + "0"
    assert case["x_packed"] == gc.pack_bits(bits(case["x"]), n).hex()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_delta_and_labels_are_the_seed_expansion(case):
    """The seed expansion with hashlib alone: ``SHAKE-256("PYFE4AI-V06-BCFE-GBPRG"
    || salt || seed)`` read as ``Delta`` (then ``Delta[0] |= 1``), the zero
    labels of the key wires, then (mode ``"labels"``) of the data wires,
    whose active labels are ``W0 xor x_i Delta``."""
    info, exp = case["circuit"], case["expected"]
    m, n = info["m"], info["n"] if case["mode"] == "labels" else 0
    salt, seed = bytes.fromhex(case["salt"]), bytes.fromhex(case["seed"])
    stream = hashlib.shake_256(b"PYFE4AI-V06-BCFE-GBPRG" + salt + seed).digest(16 * (1 + m + n))
    delta = bytes([stream[0] | 1]) + stream[1:16]
    assert exp["delta"] == delta.hex()
    assert exp["key0_sha256"] == hashlib.sha256(stream[16:16 + 16 * m]).hexdigest()
    d = int.from_bytes(delta, "little")
    data_active = b"".join(
        (int.from_bytes(stream[16 * (1 + m + i):16 * (2 + m + i)], "little") ^ (d if bit else 0)).to_bytes(16, "little")
        for i, bit in enumerate(bits(case["x"])[:n]))
    assert exp["data_active_sha256"] == hashlib.sha256(data_active).hexdigest()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_circuits_of_the_cases(case):
    """Each case's ``guard(C)``: digest and sizes; the two decoders agree on it;
    for the hand-written circuits, ``guard`` of the stored ``C`` gives it."""
    info = case["circuit"]
    enc = garbled_encoding(case)
    decoded = Circuit.decode(enc)
    parsed = gc.parse_circuit(enc)
    assert hashlib.shake_256(b"PYFE4AI-V06-BCFE-CIRCUIT" + enc).hexdigest(32) == info["digest"]
    assert (decoded.m, decoded.n, decoded.n_gates, decoded.n_out) == (info["m"], info["n"], info["n_gates"],
                                                                      info["n_out"])
    assert (parsed.m, parsed.n, parsed.gates, parsed.outputs) == (decoded.m, decoded.n, decoded.gates,
                                                                  decoded.outputs)
    if info["source"] == "hand":
        c = Circuit.decode(bytes.fromhex(info["unguarded_encoding"]))
        assert c.digest().hex() == info["unguarded_digest"]
        assert c.guard().encode() == enc
    if info["source"] == "random":
        assert info["n_gates"] == 2000


def _gate_classes(enc: bytes, mode: str) -> list:
    """``"AND-LK"``, ``"INV-K"``, ... per gate of *enc* in *mode*."""
    parsed = gc.parse_circuit(enc)
    kinds = gc.wire_kinds(parsed, mode)
    name = {gc.KIND_L: "L", gc.KIND_K: "K"}
    op_name = {gc.OP_XOR: "XOR", gc.OP_AND: "AND", gc.OP_INV: "INV"}
    return [op_name[op] + "-" + name[kinds[a]] + ("" if op == gc.OP_INV else name[kinds[b]])
            for op, a, b in parsed.gates]


ALL_CLASSES = {"AND-LL", "AND-LK", "AND-KL", "AND-KK", "XOR-LL", "XOR-LK", "XOR-KL", "XOR-KK", "INV-L", "INV-K"}


def test_the_cases_cover_every_gate_kind_pair():
    for source in ("hand", "random"):
        for mode in MODES:
            seen = set()
            for case in CASES:
                if case["circuit"]["source"] == source and case["mode"] == mode:
                    seen.update(_gate_classes(garbled_encoding(case), mode))
            assert seen == (ALL_CLASSES if mode == "hardwired" else {"AND-LL", "XOR-LL", "INV-L"}), (source, mode)
    # the gate of each one-gate circuit (the first gate of guard(C)) is the one its name says, in mode "hardwired"
    one_gate = [case for case in CASES if case["circuit"]["source"] == "hand" and case["mode"] == "hardwired"
                and case["circuit"]["n_gates"] == 3]
    assert len(one_gate) == 10
    for case in one_gate:
        op, kinds = case["name"].split("-")[1:3]
        assert _gate_classes(garbled_encoding(case), "hardwired")[0] == "{}-{}".format(op, kinds).upper()


@pytest.mark.parametrize("case", LIBRARY_CASES, ids=[case["name"] for case in LIBRARY_CASES])
def test_library_circuits_are_what_the_builder_builds_today(case):
    info = case["circuit"]
    c = library_circuit(info["function"], info["params"])
    assert c.digest().hex() == info["unguarded_digest"], (
        "pyfe4ai.circuits.{}({}) changed: digest {} instead of the KAT's {}".format(
            info["function"], info["params"], c.digest().hex(), info["unguarded_digest"]))
    g = c.guard()
    assert g.digest().hex() == info["digest"]
    assert (g.m, g.n, g.n_gates, g.n_out) == (info["m"], info["n"], info["n_gates"], info["n_out"])


# ------------------------------------------------------------------ the answers

def _check_outputs(case: dict, evaluate) -> None:
    """``evaluate(w) -> packed output`` against the case's outputs (and, for the
    tiny circuits, against the output of every key)."""
    info, exp = case["circuit"], case["expected"]
    out = evaluate(bits(case["w"]))
    assert gc.unpack_bits(out, info["n_out"]) == bits(exp["output"])
    assert exp["output"] == exp["plain_output"]
    for v, expected in enumerate(exp.get("output_by_w", ())):
        assert bit_string(gc.unpack_bits(evaluate(w_of(v, info["m"])), info["n_out"])) == expected, v


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_reference_gives_the_known_answers(case):
    enc = garbled_encoding(case)
    mode, salt, seed = case["mode"], bytes.fromhex(case["salt"]), bytes.fromhex(case["seed"])
    res = gc.garble(enc, mode, salt, bits(case["x"]), seed)  # the bits as a list
    check_garbling(case, res)
    assert gc.garble(enc, mode, salt, bytes.fromhex(case["x_packed"]), seed) == res  # and packed
    _check_outputs(case, lambda w: gc.evaluate(enc, mode, salt, res["tables"], res["decode"], active_keys(res, w),
                                               res["data_active"]))
    # the plain circuit agrees
    decoded = Circuit.decode(enc)
    assert bit_string(decoded.evaluate_plain(bits(case["w"]), bits(case["x"]))) == case["expected"]["plain_output"]
    for v, expected in enumerate(case["expected"].get("output_by_w", ())):
        assert bit_string(decoded.evaluate_plain(w_of(v, decoded.m), bits(case["x"]))) == expected


@pytest.mark.parametrize("impl", ["python", pytest.param("native", marks=native)])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_backends_give_the_known_answers(case, impl, monkeypatch):
    enc = garbled_encoding(case)
    mode, salt, seed = case["mode"], bytes.fromhex(case["salt"]), bytes.fromhex(case["seed"])
    spy = None
    if native_backend.available():
        spy = CountingExtension(native_backend._ext)
        monkeypatch.setattr(native_backend, "_ext", spy)
    drawn = []
    with native_backend.backend(impl):
        ops = native_backend.ops()
        with seeded(seed, drawn):
            res = ops.gc_garble(enc, mode, salt, bytes.fromhex(case["x_packed"]))
        assert drawn == [32]  # one seed per garbling, from the hook
        check_garbling(case, res)
        _check_outputs(case, lambda w: ops.gc_evaluate(enc, mode, salt, res["tables"], res["decode"],
                                                       active_keys(res, w), res["data_active"]))
    if spy is not None:
        evaluations = 1 + len(case["expected"].get("output_by_w", ()))
        assert spy.calls == ({"gc_garble": 1, "gc_evaluate": evaluations} if impl == "native" else {})


# ------------------------------------------------------------------ native against the reference

def _log_uniform(rng: random.Random, low: int, high: int) -> int:
    return min(high, max(low, int(math.exp(rng.uniform(math.log(low), math.log(high + 1))))))


def random_instance(rng: random.Random, max_gates: int, max_m: int, max_n: int) -> tuple:
    """``(garbled circuit, w, x)``: a random circuit of 1 .. *max_gates* gates
    (uniform or log-uniform, even odds), random ``m`` and ``n``; half of them ``guard(C)``, the
    others with outputs drawn among the wires labelled in mode
    ``"hardwired"`` (so labelled in both modes). Inputs are drawn from the
    data wires known to the garbler, from the last 16 wires or uniformly;
    the gate mix varies per circuit."""
    m, n = rng.randint(1, max_m), rng.randint(0, max_n)
    size = rng.randint(1, max_gates) if rng.random() < 0.5 else _log_uniform(rng, 1, max_gates)
    n_out = rng.randint(1, 16)
    guard = rng.random() < 0.5
    count = max(0, size - 1 - n_out) if guard else size
    p_xor = rng.uniform(0.1, 0.6)
    p_and = rng.uniform(0.2, 0.9 - p_xor)
    p_known = rng.choice((0.0, 0.2, 0.4))
    known = list(range(m, m + n))
    labelled = [True] * m + [False] * n  # in mode "hardwired"

    def pick(o: int) -> int:
        r = rng.random()
        if r < p_known and known:
            return rng.choice(known)
        if r < p_known + 0.35:
            return o - 1 - rng.randrange(min(o, 16))
        return rng.randrange(o)

    gates = []
    for i in range(count):
        o = m + n + i
        r = rng.random()
        op = gc.OP_XOR if r < p_xor else gc.OP_AND if r < p_xor + p_and else gc.OP_INV
        if o == 1:
            op = gc.OP_INV
        a, b = pick(o), 0
        if op != gc.OP_INV:
            b = pick(o)
            while b == a:
                b = pick(o)
        gates.append((op, a, b))
        is_l = labelled[a] if op == gc.OP_INV else labelled[a] or labelled[b]
        labelled.append(is_l)
        if not is_l:
            known.append(o)
    if guard:
        c = Circuit(m, n, gates, [rng.randrange(m + n + count) for _ in range(n_out)]).guard()
        w = [rng.randrange(2) for _ in range(m)] + [int(rng.random() < 0.1)]  # g = 1 now and then
    else:
        pool = [v for v, is_l in enumerate(labelled) if is_l]
        c = Circuit(m, n, gates, [rng.choice(pool) for _ in range(n_out)])
        w = [rng.randrange(2) for _ in range(m)]
    return c, w, [rng.randrange(2) for _ in range(n)]


def _first_difference(a: bytes, b: bytes) -> str:
    if len(a) != len(b):
        return "lengths {} and {}".format(len(a), len(b))
    return "first at byte {}".format(next(i for i in range(len(a)) if a[i] != b[i]))


def crosscheck(c: Circuit, mode: str, w, x, seed: bytes, salt: bytes, tag: str) -> None:
    """Native ``gc_garble`` / ``gc_evaluate`` against the reference, and the
    evaluation against the plain circuit."""
    enc = c.encode()
    xp = gc.pack_bits(x, c.n)
    ref = gc.garble(enc, mode, salt, xp, seed)
    with native_backend.backend("native"):
        ops = native_backend.ops()
        with seeded(seed):
            nat = ops.gc_garble(enc, mode, salt, xp)
        for field in FIELDS:
            if nat[field] != ref[field]:
                pytest.fail("{} ({}): native {} differs from the reference, {}".format(
                    tag, mode, field, _first_difference(nat[field], ref[field])))
        key_active = active_keys(ref, w)
        y_nat = ops.gc_evaluate(enc, mode, salt, ref["tables"], ref["decode"], key_active, ref["data_active"])
    y_ref = gc.evaluate(enc, mode, salt, ref["tables"], ref["decode"], key_active, ref["data_active"])
    plain = gc.pack_bits(c.evaluate_plain(w, x), c.n_out)
    assert y_nat == y_ref == plain, "{} ({}): native {}, reference {}, plain {}".format(
        tag, mode, y_nat.hex(), y_ref.hex(), plain.hex())


def run_random_crosscheck(seed: int, count: int, max_gates: int, max_m: int, max_n: int) -> collections.Counter:
    """*count* random circuits (:func:`random_instance`), each in both modes with a
    fresh random seed and salt. Returns gate and garbling counts."""
    rng = random.Random(seed)
    stats = collections.Counter()
    for i in range(count):
        c, w, x = random_instance(rng, max_gates, max_m, max_n)
        stats["circuits"] += 1
        stats["gates"] += c.n_gates
        for mode in MODES:
            crosscheck(c, mode, w, x, rng.randbytes(32), rng.randbytes(56),
                       "circuit {} of random.Random({}) (m={}, n={}, gates={})".format(i, seed, c.m, c.n, c.n_gates))
            stats["garblings"] += 1
    return stats


@native
def test_native_is_the_reference_on_random_circuits():
    stats = run_random_crosscheck(20261119, 50, 3000, 64, 64)
    assert stats["circuits"] == 50 and stats["garblings"] == 100


#: The library circuits at realistic sizes of the large cross-check.
LARGE_LIBRARY = (
    ("linear_threshold", {"n": 100, "bw": 16, "bx": 16}),
    ("linear_threshold", {"n": 100, "bw": 16, "bx": 16, "mul": "shift_add"}),
    ("linear_score", {"n": 100, "bw": 16, "bx": 16, "shift": 8}),
    ("mlp", {"n": 16, "hidden": 8, "bw": 8, "bx": 8, "bh": 8, "shift": 6}),
    ("argmax_scores", {"k": 10, "n": 32, "bw": 8, "bx": 8}),
)


def run_library_crosscheck(entries=LARGE_LIBRARY, seed: int = 7) -> collections.Counter:
    """Every library circuit of *entries*, guarded: the two decoders agree, and
    native against reference in both modes on random inputs."""
    rng = random.Random(seed)
    stats = collections.Counter()
    for function, params in entries:
        c = getattr(circuits, function)(**params).circuit.guard()
        _check_decoders(c)
        w = [rng.randrange(2) for _ in range(c.m - 1)] + [0]
        x = [rng.randrange(2) for _ in range(c.n)]
        for mode in MODES:
            crosscheck(c, mode, w, x, rng.randbytes(32), rng.randbytes(56), "{}({})".format(function, params))
            stats["garblings"] += 1
        stats["gates"] += c.n_gates
    return stats


@native
@large
def test_large_native_is_the_reference_on_random_circuits():
    stats = run_random_crosscheck(1119, 5000, 50000, 300, 300)
    assert stats["circuits"] == 5000


@native
@large
def test_large_native_is_the_reference_on_library_circuits():
    stats = run_library_crosscheck()
    assert stats["garblings"] == 2 * len(LARGE_LIBRARY)


# ------------------------------------------------------------------ the two decoders

def _check_decoders(c: Circuit) -> None:
    enc = c.encode()
    parsed = gc.parse_circuit(enc)
    decoded = Circuit.decode(enc)
    assert (parsed.m, parsed.n, parsed.n_wires) == (decoded.m, decoded.n, decoded.n_wires) == (c.m, c.n, c.n_wires)
    assert parsed.gates == decoded.gates == c.gates
    assert parsed.outputs == decoded.outputs == c.outputs
    assert decoded.encode() == enc and decoded == c


DECODE = (
    ("linear_threshold", {"n": 4, "bw": 4, "bx": 4}),
    ("linear_threshold", {"n": 3, "bw": 6, "bx": 5, "mul": "shift_add"}),
    ("linear_score", {"n": 3, "bw": 5, "bx": 3, "shift": 2}),
    ("linear_score", {"n": 2, "bw": 7, "bx": 4, "shift": 0}),
    ("mlp", {"n": 2, "hidden": 2, "bw": 3, "bx": 3, "bh": 3, "shift": 1}),
    ("argmax_scores", {"k": 3, "n": 2, "bw": 4, "bx": 4}),
)


@pytest.mark.parametrize("function,params", DECODE, ids=["{}-{}".format(f, i) for i, (f, _) in enumerate(DECODE)])
def test_the_two_decoders_agree_on_library_circuits(function, params):
    c = library_circuit(function, params)
    for circuit in (c, c.guard()):
        _check_decoders(circuit)


# ------------------------------------------------------------------ gc_garble is for tests only

#: Where the library's schemes live: none of their sources may name the operations that hand out Delta.
SCHEME_SOURCES = ("pyfe4ai/schemes", "pyfe4ai/ml")


def test_the_schemes_never_call_gc_garble_or_gc_evaluate():
    """``gc_garble`` returns ``Delta`` and the zero labels: tests and
    known-answer vectors only. The schemes encrypt with
    ``bcfe_encrypt_slot`` (``Delta`` never leaves the implementation) and
    decrypt with ``bcfe_decrypt_slot``; no source under
    :data:`SCHEME_SOURCES` names ``gc_garble`` or ``gc_evaluate``, not even in a
    string or comment."""
    root = Path(__file__).resolve().parents[2]
    scanned, found = 0, []
    for top in SCHEME_SOURCES:
        for path in sorted((root / top).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            scanned += 1
            for name in ("gc_garble", "gc_evaluate"):
                if name in text:
                    found.append("{}: {}".format(path.relative_to(root), name))
    assert scanned > 10 and not found, found
    # the scan would see a use (the bcfe scheme does name the fused operations)
    scheme = (root / "pyfe4ai/schemes/bcfe/gc_ristretto255.py").read_text(encoding="utf-8")
    assert "bcfe_encrypt_slot" in scheme and "bcfe_decrypt_slot" in scheme
