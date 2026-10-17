"""pairing_backend.multiexp / pow_many / deserialize_many on the test pairing group.

On BLS12-381 they are one call into the Rust extension (multi-scalar
multiplication, a shared table for one base, one batch of decodings); on
charm groups (``PYFE4AI_TEST_PAIRING_GROUP=MNT224``) the loops they replace.
Either way the results are those of the per-term loop.
"""

import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from pyfe4ai.utils import pairing_backend as pb  # noqa: E402


@pytest.fixture(scope="module")
def group():
    return pb.PairingGroup(GROUP)


def _loop(group, bases, exps):
    order = int(group.order())
    acc = bases[0] ** group.init(pb.ZR, 0)
    for b, e in zip(bases, exps):
        acc = acc * (b ** group.init(pb.ZR, int(e) % order))
    return acc


@pytest.mark.parametrize("kind", [pb.G1, pb.G2, pb.GT])
def test_multiexp_and_pow_many_match_the_loop(group, kind):
    order = int(group.order())
    gs = [group.random(kind) for _ in range(4)]
    exps = [0, 1, -3, 12345, order - 1, order + 2, int(group.random(pb.ZR)), -int(group.random(pb.ZR))]
    for n in (1, 3, 8):
        bases = [gs[i % 4] for i in range(n)]
        assert pb.multiexp(bases, exps[:n], group) == _loop(group, bases, exps[:n])
        assert pb.multiexp(bases, [group.init(pb.ZR, e % order) for e in exps[:n]], group) == _loop(
            group, bases, exps[:n])
    want = [gs[0] ** group.init(pb.ZR, e % order) for e in exps]
    assert pb.pow_many(gs[0], exps, group) == want
    assert pb.multiexp(gs[:2], [0, 0], group) == gs[0] ** group.init(pb.ZR, 0)


def test_multiexp_rejects_empty_or_unequal(group):
    g = group.random(pb.G1)
    for bases, exps in (([], []), ([g], [1, 2])):
        with pytest.raises(ValueError):
            pb.multiexp(bases, exps, group)


def test_deserialize_many_round_trip(group):
    xs = [group.random(pb.G1), group.random(pb.G2), group.random(pb.GT), group.random(pb.G1)]
    texts = [group.serialize(x).decode("utf-8") for x in xs]
    assert pb.deserialize_many(group, texts + texts[:1]) == xs + xs[:1]
    assert pb.deserialize_many(group, [t.encode("utf-8") for t in texts]) == xs
