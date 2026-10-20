"""BLS12-381 backend (pyfe4ai.primitives.pairing.bls12_381, rust/native/src/pairing.rs):
regression tests of the October 2026 code review.

Before the fixes:

* every SIFE FH-IPE decryption built a GT baby-step table for its own base
  ``e(k1, c1)`` and the process-wide table cache never evicted it (memory
  grew by megabytes per decryption);
* encodings were not canonical: bytes after the compressed element and type
  tags such as ``"01"`` or ``"+1"`` were accepted (same element, other string);
* a type tag outside 0..255 raised ``OverflowError`` instead of ``ValueError``;
* a bound of ``2^63`` or more wrapped ``2 bound + 1`` and missed in-range logs.
"""

import base64
import os
import subprocess
import sys
import textwrap

import pytest

from pyfe4ai.primitives.pairing import bls12_381 as bls

pytestmark = pytest.mark.skipif(not bls.available(), reason="BLS12-381 backend not built") \
    if hasattr(bls, "available") else []

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _group():
    from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup

    return PairingGroup("BLS12-381")


_LEAK = textwrap.dedent(
    """
    import os, random, resource, sys
    os.chdir(sys.argv[1])
    os.environ["PYFE4AI_CACHE_DIR"] = os.path.join(sys.argv[1], "config")
    from pyfe4ai.schemes.sife.fh_ipe_pairing import SIFEFHIPE, SIFEFHIPEKeyGenerator
    kg = SIFEFHIPEKeyGenerator({"sec_param": 64, "eta": 10, "bound_x": 100, "bound_y": 100,
                                "pairing_group_param": "BLS12-381"})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFEFHIPE({"id": "nid_default", "keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dec = SIFEFHIPE({"id": "sid_0", "keys": {"pp": pp}})
    y = [random.randint(-100, 100) for _ in range(10)]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    def rss():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10
    x = [1] * 10
    dec.decrypt(enc.encrypt(x), dk, y)
    start = rss()
    for _ in range(12):
        assert dec.decrypt(enc.encrypt(x), dk, y) == sum(y)
    print(rss() - start)
    """
)


def test_fh_ipe_decryptions_do_not_grow_memory(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    res = subprocess.run([sys.executable, "-c", _LEAK, str(tmp_path)], capture_output=True, text=True,
                         timeout=600, env=env)
    assert res.returncode == 0, res.stderr[-2000:]
    growth = float(res.stdout.strip().splitlines()[-1])
    assert growth < 30, "12 SIFE FH-IPE decryptions (bound 10^5) grew the RSS by {:.0f} MB".format(growth)


@pytest.mark.parametrize("kind", ["ZR", "G1", "G2", "GT"])
def test_trailing_bytes_are_rejected(kind):
    from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, GT, ZR, pair

    g = _group()
    elem = {"ZR": lambda: g.random(ZR), "G1": lambda: g.random(G1), "G2": lambda: g.random(G2),
            "GT": lambda: pair(g.random(G1), g.random(G2))}[kind]()
    tag, body = g.serialize(elem).split(b":", 1)
    padded = tag + b":" + base64.b64encode(base64.b64decode(body) + b"\x00junk")
    with pytest.raises(ValueError):
        g.deserialize(padded)


@pytest.mark.parametrize("tag", [b"01", b" 1", b"+1"])
def test_type_tag_is_canonical(tag):
    from pyfe4ai.primitives.pairing.pairing_backend import G1

    g = _group()
    _, body = g.serialize(g.random(G1)).split(b":", 1)
    with pytest.raises(ValueError):
        g.deserialize(tag + b":" + body)


@pytest.mark.parametrize("tag", [b"256", b"-1"])
def test_out_of_range_tag_raises_value_error(tag):
    from pyfe4ai.primitives.pairing.pairing_backend import G1

    g = _group()
    _, body = g.serialize(g.random(G1)).split(b":", 1)
    with pytest.raises(ValueError):
        g.deserialize(tag + b":" + body)


def test_gt_dlog_with_a_bound_of_2_63():
    from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, pair

    g = _group()
    gt = pair(g.random(G1), g.random(G2))
    try:
        found = bls.gt_dlog(gt, gt ** 5, 2 ** 63)
    except (ValueError, OverflowError):
        return  # refusing the bound is fine
    assert found == 5
