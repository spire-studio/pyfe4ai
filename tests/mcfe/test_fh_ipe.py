"""ACFGU18 Fig. 12 (n = 1) function-hiding IPE building block."""

import random

import pytest

pytest.importorskip("charm")

from pyfe4ai.schemes.mcfe.fh_ipe import FHIPE, check_asymmetric  # noqa: E402
from pyfe4ai.utils.exceptions import FEValidationError  # noqa: E402
from pyfe4ai.utils.pairing_backend import G1, G2, ZR, PairingGroup  # noqa: E402


@pytest.fixture(scope="module")
def fhipe():
    group = PairingGroup("MNT224")
    return FHIPE(group, group.random(G1), group.random(G2), m=3)


def _enc(fh, vec, g):
    return [g ** fh.group.init(ZR, v % fh.order) for v in vec]


def test_decrypts_inner_product_in_gt(fhipe):
    msk = fhipe.setup()
    for _ in range(3):
        x = [random.randint(-5, 5) for _ in range(3)]
        y = [random.randint(-5, 5) for _ in range(3)]
        ct = fhipe.encrypt(msk, _enc(fhipe, x, fhipe.g1))
        key, z_sum = fhipe.keygen(msk, _enc(fhipe, y, fhipe.g2))
        out = fhipe.decrypt(ct, key, fhipe.gt ** fhipe.group.init(ZR, z_sum))
        assert out == fhipe.gt ** fhipe.group.init(ZR, sum(a * b for a, b in zip(x, y)) % fhipe.order)


def test_encryption_and_keys_are_randomised(fhipe):
    msk = fhipe.setup()
    x = _enc(fhipe, [1, 2, 3], fhipe.g1)
    y = _enc(fhipe, [1, 0, 1], fhipe.g2)
    assert fhipe.encrypt(msk, x) != fhipe.encrypt(msk, x)
    assert fhipe.keygen(msk, y)[0] != fhipe.keygen(msk, y)[0]


def test_symmetric_groups_rejected():
    with pytest.raises(FEValidationError):
        check_asymmetric("SS512")
    check_asymmetric("MNT224")


def test_keygen_exponents_match_group_keygen(fhipe):
    msk = fhipe.setup()
    x = [3, -1, 2]
    y = [1, 4, -2]
    e, z_sum = fhipe.keygen_exponents(msk, y)
    key = [fhipe.g2 ** fhipe.group.init(ZR, v) for v in e]
    ct = fhipe.encrypt(msk, _enc(fhipe, x, fhipe.g1))
    out = fhipe.decrypt(ct, key, fhipe.gt ** fhipe.group.init(ZR, z_sum))
    assert out == fhipe.gt ** fhipe.group.init(ZR, (3 - 4 - 4) % fhipe.order)


def test_deterministic_rng(fhipe):
    from pyfe4ai.utils.sampling_utils import XofRandom

    msk = fhipe.setup()
    y = _enc(fhipe, [1, 2, 3], fhipe.g2)
    k1 = fhipe.keygen(msk, y, rng=XofRandom(b"k" * 32 + b"tag"))
    k2 = fhipe.keygen(msk, y, rng=XofRandom(b"k" * 32 + b"tag"))
    assert k1 == k2
