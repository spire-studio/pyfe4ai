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


# -- exponentiation, products of powers, batch deserialisation ---------------

def _ref_pow(group, g, k):
    """g^k by square-and-multiply with the group operation only (independent of ** and msm)."""
    k %= R
    acc, base = group.init(g.type), g
    while k:
        if k & 1:
            acc = acc * base
        base = base * base
        k >>= 1
    return acc


EXPONENTS = [0, 1, 2, 3, -1, -7, 255, 2**64 - 1, 2**64, 2**128 + 1, R - 1, R, R + 5, 3 * R - 2,
             0x5A3C_96E1_0F2D_7B48_C3A5_1E9F_6D02_B7C4_8A1F_3E6D_9B05_C27A_E4F1_6083_D5B9_2C71]


@pytest.mark.parametrize("kind", [B.G1, B.G2, B.GT])
def test_pow_matches_square_and_multiply(group, kind):
    g = group.random(kind)
    for k in EXPONENTS:
        assert g ** k == _ref_pow(group, g, k)
        assert g ** group.init(B.ZR, k) == _ref_pow(group, g, k)
    assert group.init(kind) ** 12345 == group.init(kind)


@pytest.mark.parametrize("kind", [B.G1, B.G2, B.GT])
def test_msm_and_pow_many(group, kind):
    gs = [group.random(kind) for _ in range(5)] + [group.init(kind)]
    for n in (1, 2, 3, 6, 17):
        bases = [gs[(3 * i) % len(gs)] for i in range(n)]
        ks = [EXPONENTS[(5 * i + n) % len(EXPONENTS)] for i in range(n)]
        want = group.init(kind)
        for b, k in zip(bases, ks):
            want = want * _ref_pow(group, b, k)
        assert B.msm(bases, ks) == want
        assert B.msm(bases, [group.init(B.ZR, k) for k in ks]) == want
    # one base: NAF table below 12 powers, windows of 4, 5, 6 bits from 12, 32, 96 (G1, G2)
    for n in (2, 5, 12, 40, 100):
        ks = [EXPONENTS[(7 * i + n) % len(EXPONENTS)] + i for i in range(n)]
        assert B.pow_many(gs[0], ks) == [_ref_pow(group, gs[0], k) for k in ks]
    assert B.msm(gs[:2], [0, R]) == group.init(kind)
    assert B.pow_many(group.init(kind), [1, 5]) == [group.init(kind)] * 2
    assert B.pow_many(gs[0], []) == []


@pytest.mark.parametrize("kind", [B.G1, B.G2])
def test_msm_pippenger_range(group, kind):
    # from 512 terms on, arkworks' bucket method; below, the GLV/Straus method
    gs = [group.random(kind) for _ in range(520)]
    ks = [int(group.random(B.ZR)) for _ in range(520)]
    assert B.msm(gs, ks) == B.msm(gs[:260], ks[:260]) * B.msm(gs[260:], ks[260:])


def test_msm_rejects_bad_input(group):
    g1, g2 = group.random(B.G1), group.random(B.G2)
    for bases, ks in (([], []), ([g1], [1, 2]), ([g1, g2], [1, 2]), ([group.init(B.ZR, 1)], [1])):
        with pytest.raises((ValueError, TypeError)):
            B.msm(bases, ks)
    with pytest.raises(TypeError):
        B.msm([g1], ["1"])
    with pytest.raises(TypeError):
        B.pow_many(group.init(B.ZR, 2), [1])


def test_deserialize_many(group):
    xs = [group.random(k) for k in (B.G1, B.G2, B.GT)] + [group.init(B.ZR, 9), group.init(B.G1)]
    texts = [group.serialize(x) for x in xs] * 2
    assert group.deserialize_many(texts) == xs * 2
    assert group.deserialize_many([t.decode("utf-8") for t in texts]) == xs * 2
    assert group.deserialize_many([]) == []
    raw = bytearray(xs[2].to_bytes())
    raw[7] ^= 4
    for bad in (b"3:" + base64.b64encode(bytes(raw)), b"9:AA==", b"garbage", b"1:" + texts[0][2:-4]):
        with pytest.raises(ValueError):
            group.deserialize_many(texts[:3] + [bad])


# -- membership in GT --------------------------------------------------------
# Fq12 = Fq6[w]/(w^2 - v), Fq6 = Fq2[v]/(v^3 - (1 + u)), Fq2 = Fq[u]/(u^2 + 1), as in
# arkworks, serialised as 12 coefficients of 48 bytes (little-endian), c0 before c1.

def _f2_mul(a, b):
    return ((a[0] * b[0] - a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P)


def _f2_add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def _f2_xi(a):  # a (1 + u)
    return ((a[0] - a[1]) % P, (a[0] + a[1]) % P)


def _f6_mul(a, b):
    c = [(0, 0)] * 5
    for i in range(3):
        for j in range(3):
            c[i + j] = _f2_add(c[i + j], _f2_mul(a[i], b[j]))
    return (_f2_add(c[0], _f2_xi(c[3])), _f2_add(c[1], _f2_xi(c[4])), c[2])


def _f6_add(a, b):
    return tuple(_f2_add(x, y) for x, y in zip(a, b))


def _f12_mul(a, b):
    c1c1 = _f6_mul(a[1], b[1])
    return (_f6_add(_f6_mul(a[0], b[0]), (_f2_xi(c1c1[2]), c1c1[0], c1c1[1])),
            _f6_add(_f6_mul(a[0], b[1]), _f6_mul(a[1], b[0])))


def _f12_pow(a, e):
    acc = (((1, 0), (0, 0), (0, 0)), ((0, 0),) * 3)
    while e:
        if e & 1:
            acc = _f12_mul(acc, a)
        a = _f12_mul(a, a)
        e >>= 1
    return acc


def _f12_from_bytes(data):
    c = [int.from_bytes(data[48 * i:48 * i + 48], "little") for i in range(12)]
    f6 = lambda o: tuple((c[o + 2 * i], c[o + 2 * i + 1]) for i in range(3))  # noqa: E731
    return (f6(0), f6(6))


def _f12_to_bytes(a):
    return b"".join(x.to_bytes(48, "little") for half in a for f2 in half for x in f2)


def test_gt_membership_check():
    import random

    group = B.PairingGroup()
    s, t = group.random(B.GT), group.random(B.GT)
    # the Python tower agrees with the extension
    assert _f12_to_bytes(_f12_mul(_f12_from_bytes(s.to_bytes()), _f12_from_bytes(t.to_bytes()))) == (s * t).to_bytes()
    one = _f12_pow(_f12_from_bytes(s.to_bytes()), 0)
    assert group.deserialize(group.serialize(s)) == s
    assert B._ext.bls_deserialize(B.GT, _f12_to_bytes(one)) == group.init(B.GT, 1)
    rnd = random.Random(5)
    f = (tuple((rnd.randrange(P), rnd.randrange(P)) for _ in range(3)),
         tuple((rnd.randrange(P), rnd.randrange(P)) for _ in range(3)))
    # h is in the cyclotomic subgroup (order p^4 - p^2 + 1) but not in GT: only f^p == f^u rejects it
    h = _f12_pow(f, (P ** 6 - 1) * (P ** 2 + 1))
    assert _f12_pow(h, R) != one
    for bad in (f, h, _f12_mul(h, _f12_from_bytes(s.to_bytes())), ((((0, 0),) * 3), (((0, 0),) * 3))):
        with pytest.raises(ValueError):
            B._ext.bls_deserialize(B.GT, _f12_to_bytes(bad))
    with pytest.raises(ValueError):
        B._ext.bls_deserialize(B.GT, P.to_bytes(48, "little") + s.to_bytes()[48:])  # non-canonical
