"""Native dealing and piece sums of the replicated sharings (plain and seeded)
of the three LWE threshold layers: the values, key shares and partial
decryptions of the Python code, bit for bit."""

import json
from itertools import combinations

import gmpy2 as gp
import pytest

from pyfe4ai.schemes import threshold_utils as tu
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator
from pyfe4ai.utils import native_backend
from tests.test_native_backend import ByteStream, _BytesSystemRandom, native_calls  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")

# one modulus per native word size: u64, u128, 256-bit words (none NTT-friendly)
MODULI = [gp.next_prime(gp.mpz(2) ** 61), gp.next_prime(gp.mpz(2) ** 100), gp.next_prime(gp.mpz(2) ** 216)]
ST = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (5, 3)]


def _material(q):
    """Nested dicts, lists and tuples; ints (also negative and beyond q), mpz and decimal strings."""
    return {
        "sk_y": {"c0": [5, -3, int(q) + 7, gp.mpz(11)], 1: ["0", "007", str(int(q) - 1), 2 ** 300]},
        "z": -1,
        "z_chunks": [[1, 2], (3, 4), []],
        "mixed": [9, [10, 11]],
    }


def _items(pieces):
    """The pieces as JSON, in their order (byte identity includes the order)."""
    return json.dumps([[list(T), piece] for T, piece in pieces.items()])


def _deal(backend, material, t, points, q, seed=6):
    """share_replicated on *backend*, its random values drawn from ByteStream(seed)
    (by the extension, or by the Python code where the native backend leaves
    the values to it)."""
    with native_backend.backend(backend), pytest.MonkeyPatch.context() as mp:
        mp.setattr(tu, "_CSPRNG", _BytesSystemRandom(ByteStream(seed)))
        if backend == "native":
            mp.setattr(native_backend, "_random_override", ByteStream(seed).take)
        return tu.share_replicated(material, t, points, q)


@pytest.mark.parametrize("q", MODULI, ids=lambda q: "%d-bit" % int(q).bit_length())
@pytest.mark.parametrize("s,t", ST)
def test_dealing_draws_the_python_pieces_from_the_same_bytes(native_calls, q, s, t):
    points = list(range(1, s + 1))
    material = _material(q)
    py = _deal("python", material, t, points, q)
    assert native_calls["zq_replicated_deal"] == 0
    nat = _deal("native", material, t, points, q)
    assert native_calls["zq_replicated_deal"] == 1
    assert _items(nat) == _items(py)
    want = gp.mpz(-1) % q
    assert sum(gp.mpz(p["z"]) for p in nat.values()) % q == want


def test_dealing_with_the_drbg_reconstructs_and_differs():
    q = MODULI[2]
    material = _material(q)
    with native_backend.backend("native"):
        a = tu.share_replicated(material, 2, [1, 2, 3], q)
        b = tu.share_replicated(material, 2, [1, 2, 3], q)
    assert a[(1,)]["sk_y"]["c0"] != b[(1,)]["sk_y"]["c0"]
    for pieces in (a, b):
        got = [sum(gp.mpz(p["sk_y"]["c0"][i]) for p in pieces.values()) % q for i in range(4)]
        assert got == [gp.mpz(v) % q for v in material["sk_y"]["c0"]]


def test_seeded_drbg_dealing_reads_stream_zero_in_order(monkeypatch):
    """With the DRBG seeded (tests), the draws are those of SystemRandom.randrange
    on ChaCha20 stream 0 of the call, read sequentially."""
    from tests.test_native_backend import FixedBytes

    q = MODULI[2]
    material = _material(q)
    seed = bytes(range(32))
    stream = native_backend.ext().drbg_bytes(seed, 0, 1 << 16)
    with native_backend.backend("python"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(tu, "_CSPRNG", _BytesSystemRandom(FixedBytes(stream)))
        py = tu.share_replicated(material, 3, [1, 2, 3, 4], q)
    monkeypatch.setattr(native_backend, "_seed_override", seed)
    with native_backend.backend("native"):
        assert _items(tu.share_replicated(material, 3, [1, 2, 3, 4], q)) == _items(py)


def test_dealing_refuses_short_test_bytes(monkeypatch):
    q = MODULI[0]
    with native_backend.backend("native"):
        monkeypatch.setattr(native_backend, "_random_override", lambda n: b"\x00" * 7)
        with pytest.raises(RuntimeError, match="the 7 test bytes ran out"):
            tu.share_replicated({"z": [1, 2, 3]}, 2, [1, 2, 3], q)


@pytest.mark.parametrize("bits", [61, 100, 252])
def test_moduli_just_above_a_power_of_two_deal_from_the_same_bytes(bits):
    """Review I1: about half of the draws are rejected there; the test bytes
    suffice for one value at (s, t) = (2, 2) whatever the seed."""
    import random

    from tests.test_native_backend import FixedBytes

    q = gp.next_prime(gp.mpz(2) ** bits)
    for seed in range(400):
        data = random.Random(seed).randbytes(1 << 14)
        with native_backend.backend("python"), pytest.MonkeyPatch.context() as mp:
            mp.setattr(tu, "_CSPRNG", _BytesSystemRandom(FixedBytes(data)))
            py = tu.share_replicated({"z": 5}, 2, [1, 2], q)
        with native_backend.backend("native"), pytest.MonkeyPatch.context() as mp:
            mp.setattr(native_backend, "_random_override", FixedBytes(data).take)
            assert _items(tu.share_replicated({"z": 5}, 2, [1, 2], q)) == _items(py)


def test_the_test_hooks_are_unset_in_use():
    """Review I2: a DRBG seed or random bytes injected by a test make every
    call draw the same randomness; they are unset unless a test sets them."""
    assert native_backend._seed_override is None and native_backend._random_override is None
    assert native_backend.seed_kwargs() == {} and native_backend.random_input(32) is None


@pytest.mark.parametrize("value", [1.5, " 12 ", "+5", "1_0", "-4", "", None])
def test_other_values_are_left_to_the_python_code(value):
    """Values other than ints and decimal strings (also signed ones, which
    the extension reads as gmpy2 does): the extension declines them, and the
    native backend gives what (or raises as) the Python code does."""
    q = MODULI[1]
    material = {"z": [1, value]}
    with native_backend.backend("native"):
        declined = tu._share_replicated_native(material, tu.replicated_sets([1, 2, 3], 2), q) is None
    assert declined == (value != "-4")
    try:
        py = _deal("python", material, 2, [1, 2, 3], q)
    except Exception as exc:  # noqa: BLE001 - whatever the Python code raises
        with pytest.raises(type(exc)):
            _deal("native", material, 2, [1, 2, 3], q)
        return
    assert _items(_deal("native", material, 2, [1, 2, 3], q)) == _items(py)


@pytest.mark.parametrize("q", MODULI, ids=lambda q: "%d-bit" % int(q).bit_length())
@pytest.mark.parametrize("s,t", [(3, 2), (4, 2), (5, 3), (4, 4)])
@pytest.mark.parametrize("seeded", [False, True], ids=["replicated", "replicated-seeded"])
def test_share_for_enrolled_sums_as_the_python_code(native_calls, q, s, t, seeded):
    sids = ["sid_%d" % i for i in range(s)]
    points = list(range(1, s + 1))
    material = {"sk_y": {"a": [1, 2, 3], 7: ["4", 5, 6]}, "z": 8, "z_chunks": [[9, 10], [11, 12]]}
    with native_backend.backend("python"):
        if seeded:
            sharing = tu.share_replicated_seeded(material, t, points, q)
        else:
            pieces = tu.share_replicated(material, t, points, q)
    dks = {}
    for k, sid in enumerate(sids, 1):
        held = tu.seeded_pieces_held(sharing, k) if seeded else {"pieces": tu.pieces_held(pieces, k)}
        dks[sid] = {"sharing": "replicated-seeded" if seeded else "replicated", "lst_sid": sids, **held}
    for transport in (lambda o: o, lambda o: json.loads(json.dumps(o))):
        for size in range(t, s + 1):
            for enrolled in combinations(sids, size):
                total = gp.mpz(0)
                for sid in enrolled:
                    dk = transport(dks[sid])
                    for fields in (None, ("sk_y", "z")):
                        with native_backend.backend("python"):
                            py = tu.share_for_enrolled(dk, sid, list(enrolled), t, q, fields=fields)
                        with native_backend.backend("native"):
                            nat = tu.share_for_enrolled(dk, sid, list(enrolled), t, q, fields=fields)
                        assert json.dumps(nat, default=str) == json.dumps(py, default=str)
                    total += gp.mpz(nat[0]["z"])
                assert total % q == 8
    assert native_calls["zq_sum"] > 0


def test_sums_of_other_layouts_or_values_are_the_python_codes():
    q = MODULI[0]
    template = {"z": [1, 2], "y": 3}
    assert tu._sum_pieces_native(template, [{"z": [1, 2], "y": 3}, {"z": [1], "y": 3}], q) is None
    assert tu._sum_pieces_native(template, [{"z": [1, 2], "x": 3}], q) is None
    assert tu._sum_pieces_native(template, [{"z": [1, 2.5], "y": 3}], q) is None
    with native_backend.backend("native"):
        assert tu._sum_pieces_native(template, [{"y": 4, "z": ["1", 2]}, template], q) == {"z": ["2", "4"], "y": "7"}
        assert tu._sum_pieces_native(template, [], q) is None


# ------------------------------------------------------------------ layers

def _layer(kind, s, t, sharing):
    sids = ["sid_%d" % i for i in range(s)]
    base = {"sec_param": 64, "eta": 1, "n": 2, "s": s, "t": t, "lst_nid": ["c0", "c1"], "lst_sid": sids,
            "bound_x": 6, "bound_y": 6, "sharing": sharing}
    if kind == "mife":
        kg = ThresholdMIFELWEKeyGenerator(dict(base, lwe_n=16))
        cls = ThresholdMIFELWE
    elif kind == "ring":
        # JSON values and whole key shares, as the other schemes here (sent as JSON below)
        kg = ThresholdMCFERingLWEKeyGenerator(dict(base, ring_n=16, ct_format="json", key_shares="full"))
        cls = ThresholdMCFERingLWE
    else:
        kg = ThresholdMCFELWEKeyGenerator(dict(base, lwe_n=16, bound_u=2, label_modulus=8))
        cls = ThresholdMCFELWE
    kg.setup()
    return kg, cls, sids


def _request(kind, kg, cls, pp):
    y = {"c0": [2], "c1": [-3]}
    if kind == "mife":
        cts = {n: cls({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([5]) for n in y}
        return cts, {"fusion_weight": y}, y, -5
    x = [[5, -1]] if kind == "ring" else [5]
    cts = {n: cls({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(x, "L") for n in y}
    cred = {"fusion_weight": y, "label": "L"}
    return cts, cred, cred, [-5, 1] if kind == "ring" else -5.0


@pytest.mark.parametrize("kind", ["mife", "ring", "lwe"])
@pytest.mark.parametrize("sharing", ["replicated", "replicated-seeded"])
@pytest.mark.parametrize("s,t", [(3, 2), (4, 3)])
def test_layer_partials_are_the_same_on_both_backends(native_calls, kind, sharing, s, t):
    """Key shares dealt natively decrypt, and every partial decryption is the
    Python one (cold caches: a new server object per call)."""
    kg, cls, sids = _layer(kind, s, t, sharing)
    pp = kg.get_public_parameters()
    cts, cred, arg, want = _request(kind, kg, cls, pp)
    with native_backend.backend("native"):
        dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
    assert native_calls["zq_replicated_deal" if sharing == "replicated" else "zq_sum"] > 0
    dks = json.loads(json.dumps(dks))  # as the servers receive them
    for size in range(t, s + 1):
        for enrolled in combinations(sids, size):
            group = {}
            for sid in enrolled:
                parts = []
                for backend in ("python", "native"):
                    with native_backend.backend(backend):
                        server = cls({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}})
                        parts.append(server.share_decrypt(cts, arg, dks[sid], list(enrolled)))
                assert parts[0] == parts[1]
                group[sid] = parts[1]
            assert cls({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == want


@pytest.mark.parametrize("kind", ["mife", "ring", "lwe"])
def test_layer_key_shares_dealt_from_the_same_bytes_are_the_python_ones(kind):
    kg, cls, sids = _layer(kind, 3, 2, "replicated")
    pp = kg.get_public_parameters()
    _, cred, _, _ = _request(kind, kg, cls, pp)
    out = {}
    for backend in ("python", "native"):
        kg.dict_dk = {}
        with native_backend.backend(backend), pytest.MonkeyPatch.context() as mp:
            if backend == "python":
                mp.setattr(tu, "_CSPRNG", _BytesSystemRandom(ByteStream(9)))
            else:
                mp.setattr(native_backend, "_random_override", ByteStream(9).take)
            out[backend] = {sid: kg.get_decryption_keys(sid, credentials=cred)["pieces"] for sid in sids}
    assert json.dumps(out["native"]) == json.dumps(out["python"])
