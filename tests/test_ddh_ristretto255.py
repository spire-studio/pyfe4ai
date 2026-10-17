"""DDH schemes over ristretto255: the group, the label hash and the schemes.

The pure-Python ristretto255 is checked against libsodium (through ctypes,
when the library is installed) and RFC 9380 test vectors; the SIFE / MIFE /
MCFE schemes against their plaintext results.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ddh_ristretto255 as mcfe_ec
from pyfe4ai.schemes.mife import ddh_ristretto255 as mife_ec
from pyfe4ai.schemes.sife import ddh_ristretto255 as sife_ec
from pyfe4ai.utils import ec_group
from pyfe4ai.utils import ristretto255 as R
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _sodium():
    path = ctypes.util.find_library("sodium")
    if not path:
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.sodium_init()
        lib.crypto_core_ristretto255_from_hash  # libsodium >= 1.0.18
    except (OSError, AttributeError):
        return None
    return lib


SODIUM = _sodium()


# ---------------------------------------------------------------- the group

def test_generator_and_identity():
    assert R.encode(R.BASE).hex() == "e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76"
    assert R.encode(R.IDENTITY) == bytes(32)
    assert R.encode(R.mul(R.BASE, R.L)) == bytes(32)
    assert R.equal(R.mul(R.BASE, R.L + 5), R.mul_base(5))
    assert R.SQRT_AD_MINUS_ONE == 25063068953384623474111414158702152701244531502492656460079210482610430750235
    assert R.INVSQRT_A_MINUS_D == 54469307008909316920995813868745141605393597292927456921205312896311721017578


def test_group_laws():
    rng = random.Random(3)
    a, b = rng.randrange(R.L), rng.randrange(R.L)
    pa, pb = R.mul_base(a), R.mul_base(b)
    assert R.encode(R.add(pa, pb)) == R.encode(R.mul_base(a + b))
    assert R.encode(R.mul(pa, b)) == R.encode(R.mul(pb, a))
    assert R.encode(R.sub(pa, pa)) == bytes(32)
    assert R.encode(R.decode(R.encode(pa))) == R.encode(pa)


@pytest.mark.parametrize("msg,expected", [
    (b"", "6b9a7312411d92f921c6f68ca0b6380730a1a4d982c507211a90964c394179ba"),
    (b"abc", "0da749f12fbe5483eb066a5f595055679b976e93abe9be6f0f6318bce7aca8dc"),
    (b"abcdef0123456789", "087e45a86e2939ee8b91100af1583c4938e0f5fc6c9db4b107b83346bc967f58"),
])
def test_expand_message_xmd_sha512_rfc9380_vectors(msg, expected):
    assert R.expand_message_xmd(msg, b"QUUX-V01-CS02-with-expander-SHA512-256", 32).hex() == expected


def test_decode_rejects_non_canonical_encodings():
    for bad in (
        (R.P).to_bytes(32, "little"),          # s = p
        (R.P + 2).to_bytes(32, "little")[:32],  # >= p
        (1).to_bytes(32, "little"),             # negative (odd) s
        bytes(31),
    ):
        with pytest.raises(ValueError):
            R.decode(bad)
    with pytest.raises(FEValidationError):
        ec_group.from_hex("zz" * 32)


@pytest.mark.skipif(SODIUM is None, reason="libsodium with ristretto255 not found")
def test_matches_libsodium():
    def out():
        return ctypes.create_string_buffer(32)

    rng = random.Random(9)
    for _ in range(30):
        k = rng.randrange(1, R.L)
        buf = out()
        assert SODIUM.crypto_scalarmult_ristretto255_base(buf, k.to_bytes(32, "little")) == 0
        assert R.encode(R.mul_base(k)) == buf.raw
        uniform = rng.randbytes(64)
        buf = out()
        SODIUM.crypto_core_ristretto255_from_hash(buf, uniform)
        point = R.from_uniform_bytes(uniform)
        assert R.encode(point) == buf.raw
        k2 = rng.randrange(1, R.L)
        buf2 = out()
        SODIUM.crypto_scalarmult_ristretto255(buf2, k2.to_bytes(32, "little"), buf.raw)
        assert R.encode(R.mul(point, k2)) == buf2.raw
    agree = 0
    for _ in range(500):
        data = rng.randbytes(32)
        try:
            R.decode(data)
            mine = True
        except ValueError:
            mine = False
        assert mine == (SODIUM.crypto_core_ristretto255_is_valid_point(data) == 1)
        agree += mine
    assert agree  # some random strings are valid encodings


def test_label_hash_is_domain_separated():
    a = mcfe_ec.hash_label("round-1")
    b = mcfe_ec.hash_label("round-2")
    assert R.encode(a[0]) != R.encode(a[1]) != R.encode(b[0])
    assert R.encode(ec_group.hash_to_group("x", ec_group.dst("A"))) != R.encode(
        ec_group.hash_to_group("x", ec_group.dst("B")))


def test_bounded_dlog():
    for x in (-1000, -1, 0, 1, 77, 999, 1000):
        assert ec_group.dlog(R.mul_base(x), 1000) == x
    assert ec_group.dlog(R.mul_base(1001), 1000) is None
    assert ec_group.dlog(R.mul_base(123456), 1000) is None


# --------------------------------------------------------------- the schemes

def test_sife_round_trip_and_checks():
    kg = sife_ec.SIFERistretto255KeyGenerator({"eta": 4})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = sife_ec.SIFERistretto255({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dec = sife_ec.SIFERistretto255({"keys": {"pp": pp}})
    x, y = [3, -4, 0, 7], [2, 5, -1, 1]
    ct = enc.encrypt(x)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    assert dec.decrypt(ct, dk, y) == sum(a * b for a, b in zip(x, y))
    with pytest.raises(FESchemeError):
        dec.decrypt(ct, dk, [1, 1, 1, 1])
    with pytest.raises(FEValidationError):
        dec.decrypt(dict(ct, ct0="00" * 31 + "01"), dk, y)  # odd s: not an encoding
    with pytest.raises(FEValidationError):
        enc.encrypt([1, 2])
    with pytest.raises(FEKeyError):
        sife_ec.SIFERistretto255({"keys": {"pp": dict(pp, group="modp")}})
    with pytest.raises(FEValidationError):
        sife_ec.SIFERistretto255KeyGenerator({"group": "modp"})


def test_mife_round_trip_and_pads():
    nids = ["a", "b"]
    kg = mife_ec.MIFERistretto255KeyGenerator({"n": 2, "lst_nid": nids, "eta": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: mife_ec.MIFERistretto255({"keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in nids}
    agg = mife_ec.MIFERistretto255({"keys": {"pp": pp}})
    x = {"a": [5, -2], "b": [1, 9]}
    y = {"a": [1, 2], "b": [-3, 1]}
    cts = {n: clients[n].encrypt(x[n]) for n in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    assert agg.decrypt(cts, dk, y) == sum(a * b for n in nids for a, b in zip(x[n], y[n]))
    with pytest.raises(FESchemeError):
        agg.decrypt({"a": cts["a"]}, dk, y)
    bad = dict(cts, b=dict(cts["b"], c=["ff" * 32, cts["b"]["c"][1]]))
    with pytest.raises(FEValidationError, match="malformed ciphertext of b"):
        agg.decrypt(bad, dk, y)


def test_mcfe_round_trip_labels_and_errors():
    nids = ["a", "b", "c"]
    kg = mcfe_ec.MCFERistretto255KeyGenerator({"n": 3, "lst_nid": nids, "eta": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: mcfe_ec.MCFERistretto255({"keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in nids}
    agg = mcfe_ec.MCFERistretto255({"keys": {"pp": pp}})
    x = {"a": [5, -2], "b": [1, 9], "c": [0, -7]}
    y = {"a": [1, 2], "b": [-3, 1], "c": [2, 2]}
    cts = {n: clients[n].encrypt(x[n], "L1") for n in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    expected = sum(a * b for n in nids for a, b in zip(x[n], y[n]))
    assert agg.decrypt(cts, dk, y, "L1") == expected
    assert agg.decrypt(cts, dk, y, "L2") != expected  # another label: garbage (or None)
    with pytest.raises(FESchemeError):
        agg.decrypt({"a": cts["a"]}, dk, y, "L1")
    with pytest.raises(FEKeyError):
        agg.decrypt(cts, {"d": ["1"]}, y, "L1")
    with pytest.raises(FEValidationError):
        clients["a"].encrypt([1, 2, 3], "L1")
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("s", credentials={"fusion_weight": {"zz": [1]}})


def test_wrapper_selects_the_group_and_aggregates():
    from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = ["a", "b", "c"]
    w = FESchemeWrapper("mcfe", "ddh", {"group": "ristretto255", "n": 3, "lst_nid": nids,
                                        "security_profile": "standard"}, precision=2)
    assert (w.scheme_type, w.variant) == ("mcfe", "ddh_ristretto255")
    assert w.status.status == "unreviewed"
    rng = np.random.default_rng(2)
    g = {n: rng.uniform(-1, 1, (2, 3)).round(2) for n in nids}
    enc = {n: encrypt_gradient(g[n], w, nid=n, label="r") for n in nids}
    out = aggregate_gradients(enc, w, {n: 0.5 for n in nids}, label="r")
    assert np.allclose(out, 0.5 * sum(g.values()))
    with pytest.raises(ValueError, match="no instantiation in group"):
        FESchemeWrapper("mcfe", "lwe", {"group": "ristretto255", "n": 3})
    with pytest.raises(ValueError, match="no instantiation in group"):
        FESchemeWrapper("mcfe", "ddh", {"group": "p256", "n": 3})


def test_modp_key_generators_refuse_another_group():
    from pyfe4ai.schemes.mcfe.ddh import MCFEKeyGenerator

    with pytest.raises(FEValidationError, match="ddh_ristretto255"):
        MCFEKeyGenerator({"n": 2, "sec_param": 256, "group": "ristretto255"})
    MCFEKeyGenerator({"n": 2, "sec_param": 256, "group": "modp"})


def test_ndarray_failure_is_reported():
    nids = ["a", "b"]
    kg = mcfe_ec.MCFERistretto255KeyGenerator({"n": 2, "lst_nid": nids})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: mcfe_ec.MCFERistretto255({"precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
               for n in nids}
    agg = mcfe_ec.MCFERistretto255({"precision": 0, "dlog_bound": 50, "keys": {"pp": pp}})
    enc = {n: clients[n].encrypt_lst_ndarray([np.array([3, 40])], label="r") for n in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": {n: [1] for n in nids}})
    with pytest.raises(FESchemeError, match=r"element \(1,\) of layer 0"):  # 80 > dlog_bound
        agg.decrypt_lst_ndarray_ct(enc, dk, {n: [1] for n in nids}, "r")
