"""BLS12-381 from the Rust extension (pyfe4ai.utils.bls12_381) behind charm's interface."""

import base64

import pytest

from pyfe4ai.utils import bls12_381 as B

pytestmark = pytest.mark.skipif(not B.available(), reason="Rust extension not built")

P = 0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB
R = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001

# RFC 9380 J.9.1 and J.10.1: (msg, x, y), coordinates of Fp2 as (c0, c1)
G1_DST = b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_"
G1_VECTORS = [
    (b"",
     0x052926ADD2207B76CA4FA57A8734416C8DC95E24501772C814278700EED6D1E4E8CF62D9C09DB0FAC349612B759E79A1,
     0x08BA738453BFED09CB546DBB0783DBB3A5F1F566ED67BB6BE0E8C67E2E81A4CC68EE29813BB7994998F3EAE0C9C6A265),
    (b"abc",
     0x03567BC5EF9C690C2AB2ECDF6A96EF1C139CC0B2F284DCA0A9A7943388A49A3AEE664BA5379A7655D3C68900BE2F6903,
     0x0B9C15F3FE6E5CF4211F346271D7B01C8F3B28BE689C8429C85B67AF215533311F0B8DFAAA154FA6B88176C229F2885D),
]
G2_DST = b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_RO_"
G2_VECTORS = [
    (b"",
     (0x0141EBFBDCA40EB85B87142E130AB689C673CF60F1A3E98D69335266F30D9B8D4AC44C1038E9DCDD5393FAF5C41FB78A,
      0x05CB8437535E20ECFFAEF7752BADDF98034139C38452458BAEEFAB379BA13DFF5BF5DD71B72418717047F5B0F37DA03D),
     (0x0503921D7F6A12805E72940B963C0CF3471C7B2A524950CA195D11062EE75EC076DAF2D4BC358C4B190C0C98064FDD92,
      0x12424AC32561493F3FE3C260708A12B7C620E7BE00099A974E259DDC7D1F6395C3C811CDD19F1E8DBF3E9ECFDCBAB8D6)),
    (b"abc",
     (0x02C2D18E033B960562AAE3CAB37A27CE00D80CCD5BA4B7FE0E7A210245129DBEC7780CCC7954725F4168AFF2787776E6,
      0x139CDDBCCDC5E91B9623EFD38C49F81A6F83F175E80B06FC374DE9EB4B41DFE4CA3A230ED250FBE3A2ACF73A41177FD8),
     (0x1787327B68159716A37440985269CF584BCB1E621D3A7202BE6EA05C4CFE244AEB197642555A0645FB87BF7466B2BA48,
      0x00AA65DAE3C8D732D10ECD2C50F8A1BAF3001578F71C694E03866E9F3D49AC1E1CE70DD94A733534F106D4CEC0EDDD16)),
]


def _compressed(x_words, y_largest):
    """The ZCash compressed encoding: x big-endian (c1 first in Fp2), flags in the top bits."""
    raw = b"".join(w.to_bytes(48, "big") for w in x_words)
    return bytes([raw[0] | 0x80 | (0x20 if y_largest else 0)]) + raw[1:]


@pytest.fixture(scope="module")
def group():
    return B.PairingGroup("BLS12-381")


@pytest.mark.parametrize("msg,x,y", G1_VECTORS)
def test_hash_to_g1_rfc9380(msg, x, y):
    assert B._ext.bls_hash(B.G1, msg, G1_DST).to_bytes() == _compressed([x], y > (P - 1) // 2)


@pytest.mark.parametrize("msg,x,y", G2_VECTORS)
def test_hash_to_g2_rfc9380(msg, x, y):
    largest = y[1] > (P - 1) // 2 or (y[1] == 0 and y[0] > (P - 1) // 2)
    assert B._ext.bls_hash(B.G2, msg, G2_DST).to_bytes() == _compressed([x[1], x[0]], largest)


def test_group_hash_is_deterministic_and_domain_separated(group):
    h = group.hash("pyfe4ai/fh-mcfe/H1(tag)|label", B.G1)
    assert h == B.PairingGroup().hash(b"pyfe4ai/fh-mcfe/H1(tag)|label", B.G1)
    assert h != group.hash("pyfe4ai/fh-mcfe/H1(tag)|label2", B.G1)
    assert h.to_bytes() != B._ext.bls_hash(B.G1, b"pyfe4ai/fh-mcfe/H1(tag)|label", G1_DST).to_bytes()
    assert group.hash("x", B.G2).type == B.G2
    with pytest.raises(ValueError):
        group.hash("x", B.ZR)


def test_order_and_generators(group):
    assert group.order() == R
    assert group.generator(B.G1).to_bytes().hex().startswith("97f1d3a73197d794")
    g1, g2 = group.generator(B.G1), group.generator(B.G2)
    assert B.pair(g1, g2) == group.generator(B.GT)


def test_bilinearity_and_multi_pair(group):
    g1, g2 = group.random(B.G1), group.random(B.G2)
    a, b = group.random(B.ZR), group.random(B.ZR)
    e = B.pair(g1, g2)
    assert B.pair(g1 ** a, g2 ** b) == e ** (a * b) == e ** (int(a) * int(b))
    assert B.pair(g1 ** -1, g2) == 1 / e == e ** (R - 1)
    xs = [group.random(B.G1) for _ in range(4)]
    ys = [group.random(B.G2) for _ in range(4)]
    acc = group.init(B.GT, 1)
    for x, y in zip(xs, ys):
        acc *= B.pair(x, y)
    assert B.multi_pair(xs, ys) == acc
    with pytest.raises(ValueError):
        B.pair(g1, g1)  # asymmetric: check_asymmetric relies on this


def test_group_operations(group):
    g = group.random(B.G1)
    assert g * g == g ** 2 and (g * g) / g == g
    assert g ** group.init(B.ZR, 0) == group.init(B.G1)
    assert g ** group.init(B.ZR, -3) == 1 / (g ** 3)
    t = group.random(B.GT)
    assert (t / t) == group.init(B.GT, 1) == t ** 0


def test_zr_arithmetic(group):
    a, b = group.random(B.ZR), group.random(B.ZR)
    assert int(a + b) == (int(a) + int(b)) % R
    assert int(a - b) == (int(a) - int(b)) % R
    assert int(a * b) == int(a) * int(b) % R
    assert int(a / b) == int(a) * pow(int(b), -1, R) % R
    assert int(group.init(B.ZR, -5)) == R - 5
    assert group.init(B.ZR, R + 7) == 7


def test_serialisation_round_trip_and_format(group):
    for kind in (B.G1, B.G2, B.GT):
        for x in (group.random(kind), group.random(kind) ** group.init(B.ZR, 0)):
            s = group.serialize(x)
            assert isinstance(s, bytes) and s.startswith(b"%d:" % kind)
            assert group.deserialize(s) == x
            assert group.deserialize(s.decode("utf-8")) == x
            assert x != s.decode("utf-8")  # a string never equals an element
            with pytest.raises(ValueError):
                int(s.decode("utf-8"))
            assert group.ismember(group.deserialize(s))
    assert [len(group.random(k).to_bytes()) for k in (B.G1, B.G2, B.GT)] == [48, 96, 576]


def test_deserialisation_rejects_bad_encodings(group):
    g = group.serialize(group.random(B.G1))
    raw = bytearray(base64.b64decode(g.split(b":")[1]))
    raw[-1] ^= 1
    bad = b"1:" + base64.b64encode(bytes(raw))
    for data in (bad, b"1:" + g.split(b":")[1][:-4], b"2:" + g.split(b":")[1], b"9:AA==", b"garbage"):
        try:
            x = group.deserialize(data)
        except ValueError:
            continue
        # flipping the last bit may land on another x of a subgroup point
        assert data == bad and group.ismember(x)


def test_gt_dlog_matches_python_solver(group):
    from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt

    base = B.pair(group.random(B.G1), group.random(B.G2))
    for x in (0, 1, -1, 999, -1000, 1000):
        target = base ** group.init(B.ZR, x)
        assert B.gt_dlog(base, target, 1000) == x
        assert bounded_discrete_log_gt(group, base, target, 1000) == x
    assert B.gt_dlog(base, base ** 1001, 1000) is None
    with pytest.raises(ValueError):
        bounded_discrete_log_gt(group, base, base ** 1001, 1000)


def test_backend_dispatch():
    from pyfe4ai.utils import pairing_backend as pb

    g = pb.PairingGroup()
    assert isinstance(g, pb.PairingGroup) and g.groupType() == "BLS12-381"
    assert pb.DEFAULT_PAIRING_GROUP == "BLS12-381"
    pb.check_asymmetric("BLS12-381")
    pb.check_asymmetric(g)
    e = pb.pair(g.random(pb.G1), g.random(pb.G2))
    assert e.type == pb.GT
