"""pyfe4ai.primitives.pairing.pairing_backend.hash_to_g1 on the test pairing group.

A message is hashed to G1 under a domain separation tag. BLS12-381 follows RFC 9380 with the tag as its DST (the
known answers are in test_bls12_381.py); a charm group hashes the tag, a separator and the message. The tests below
hold on either, so they run on BLS12-381 and, with ``PYFE4AI_TEST_PAIRING_GROUP=MNT224``, on charm.
"""

import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from pyfe4ai.core.exceptions import FEValidationError  # noqa: E402
from pyfe4ai.primitives.pairing import pairing_backend as pb  # noqa: E402
from pyfe4ai.primitives.pairing.pairing_backend import hash_to_g1  # noqa: E402


@pytest.fixture(scope="module")
def group():
    return pb.PairingGroup(GROUP)


def test_hash_to_g1_depends_on_the_dst(group):
    a, b = hash_to_g1(group, b"m", b"D1"), hash_to_g1(group, b"m", b"D2")
    assert a != b and a == hash_to_g1(group, b"m", b"D1")


def test_hash_to_g1_depends_on_the_message(group):
    assert hash_to_g1(group, b"m1", b"D") != hash_to_g1(group, b"m2", b"D")
    assert hash_to_g1(group, b"", b"D") != hash_to_g1(group, b"\x01", b"D")
    assert hash_to_g1(group, b"m", b"D") != hash_to_g1(group, b"m ", b"D")


@pytest.mark.skipif(not pb.is_native_group(GROUP), reason="the DST of RFC 9380 is for BLS12-381")
def test_hash_to_g1_takes_binary_messages_and_tags_on_bls12_381(group):
    assert hash_to_g1(group, b"", b"D") != hash_to_g1(group, b"\x00", b"D")
    assert hash_to_g1(group, b"a\x00b", b"D") != hash_to_g1(group, b"a\x00c", b"D")
    assert hash_to_g1(group, b"m", b"A|B") != hash_to_g1(group, b"B|m", b"A")
    assert hash_to_g1(group, b"m", b"D\x00") != hash_to_g1(group, b"m", b"D")


@pytest.mark.skipif(pb.is_native_group(GROUP), reason="a charm group only")
@pytest.mark.parametrize("msg,dst", [(b"a\x00b", b"D"), (b"\x00", b"D"), (b"m", b"D\x00"), (b"m", b"A|B")])
def test_hash_to_g1_refuses_what_a_charm_group_would_collide(group, msg, dst):
    # charm hashes a C string joined as dst|msg: it ends at a NUL, and a | in the tag moves into the message
    with pytest.raises(FEValidationError, match="charm"):
        hash_to_g1(group, msg, dst)


def test_hash_to_g1_gives_an_element_of_g1(group):
    h = hash_to_g1(group, b"m", b"D")
    assert h.type == pb.G1
    assert h != group.init(pb.G1)
    # it can be used as a base of the group
    assert pb.pair(h, group.random(pb.G2)) != group.init(pb.GT, 1)


def test_hash_to_g1_takes_the_longest_dst_and_a_long_message(group):
    assert hash_to_g1(group, b"m" * 10_000, b"D" * 255).type == pb.G1


@pytest.mark.parametrize("dst", [b"", b"D" * 256, "D", None, 7, bytearray(b"D")])
def test_hash_to_g1_refuses_a_bad_dst(group, dst):
    with pytest.raises(FEValidationError, match="dst"):
        hash_to_g1(group, b"m", dst)


@pytest.mark.parametrize("msg", ["m", None, 7, bytearray(b"m"), [b"m"]])
def test_hash_to_g1_refuses_a_message_that_is_not_bytes(group, msg):
    with pytest.raises(FEValidationError, match="msg"):
        hash_to_g1(group, msg, b"D")
