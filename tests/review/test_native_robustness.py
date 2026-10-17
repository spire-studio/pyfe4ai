"""Review 2026-10 (main 810a4a5): robustness of the native extension's inputs.

Fail on the reviewed commit:

* size arithmetic wraps and a Rust panic surfaces as ``PanicException``,
  a ``BaseException`` that the library's ``except Exception`` fallbacks do not
  catch (``dense_dots`` with a huge count; ``decrypt_tensor`` with
  ``Delta = 0``); a panic while the precomputation's mutex is held poisons it;
* the ristretto255 MCFE decryption parses a ciphertext differently natively
  (``"+f"`` accepted as ``"0f"``) and in Python (spaces accepted);
* ``PYFE4AI_THREADS`` accepts any huge integer, after which every native call
  raises ``OverflowError`` instead of falling back.
"""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from pyfe4ai.utils import native_backend

ext = native_backend.ext()
pytestmark = pytest.mark.skipif(ext is None, reason="native extension not built")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _is_plain_error(exc: BaseException) -> bool:
    return isinstance(exc, Exception)


def test_dense_dots_size_overflow_is_a_value_error():
    try:
        ext.dense_dots([b"m", b"n"], 8, 2 ** 61 + 1, bytes(8), 1, 2)
    except BaseException as exc:  # noqa: BLE001
        assert _is_plain_error(exc), "{}: {}".format(type(exc).__name__, exc)
        return


def test_share_combine_with_delta_zero_is_a_value_error():
    from pyfe4ai.utils.ring_lwe_utils import next_ntt_prime
    import gmpy2 as gp

    q = next_ntt_prime(gp.mpz(2) ** 40, 16)
    ctx = native_backend.ring_context(q, 16)
    try:
        ext.share_combine(ctx.cap, ["1"] * 16, [["0"] * 16], 0, 10, 16)
    except BaseException as exc:  # noqa: BLE001
        assert _is_plain_error(exc), "{}: {}".format(type(exc).__name__, exc)


def test_bad_first_chunk_does_not_poison_the_precomputation():
    from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator

    nids = ["a", "b", "c"]
    kg = MCFERingLWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "lst_nid": nids, "ring_n": 16,
                                  "bound_x": 100, "bound_y": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = MCFERingLWE({"id": "a", "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys("a")}})
    with native_backend.backend("native"):  # also under PYFE4AI_BACKEND=python
        pre = enc.precompute("L", [(16,)])
    nkey, cap = pre.native
    x = np.ones(16, dtype=np.int64)
    with pytest.raises(BaseException) as info:  # noqa: PT011
        ext.encrypt_precomputed(nkey, cap, 2 ** 64 - 1, x, 1, False)
    assert _is_plain_error(info.value), "{}: {}".format(type(info.value).__name__, info.value)
    ext.encrypt_precomputed(nkey, cap, 0, x, 1, False)  # still usable once


def test_ristretto255_hex_parsing_is_the_same_on_both_backends():
    from pyfe4ai.schemes.mcfe import ddh_ristretto255 as mcfe_ec

    nids = ["a", "b"]
    kg = mcfe_ec.MCFERistretto255KeyGenerator({"n": 2, "lst_nid": nids})
    kg.setup()
    pp = kg.get_public_parameters()
    cl = {n: mcfe_ec.MCFERistretto255({"precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
          for n in nids}
    agg = mcfe_ec.MCFERistretto255({"precision": 0, "dlog_bound": 1000, "keys": {"pp": pp}})
    enc = {n: cl[n].encrypt_lst_ndarray([np.array([3, 40])], label="r") for n in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": {n: [1] for n in nids}})
    w = {n: [1] for n in nids}
    h = enc["a"][0][(0,)]["c"][0]
    variants = [" ".join(h[i:i + 2] for i in range(0, 64, 2))]
    plus = [h[:i] + "+" + h[i + 1:] for i in range(0, 64, 2) if h[i] == "0"]
    variants += plus[:1]
    for v in variants:
        e = copy.deepcopy(enc)
        e["a"][0][(0,)]["c"][0] = v
        outcome = {}
        for be in ("native", "python"):
            with native_backend.backend(be):
                try:
                    outcome[be] = np.ravel(agg.decrypt_lst_ndarray_ct(e, dk, w, "r")[0]).tolist()
                except Exception as exc:  # noqa: BLE001
                    outcome[be] = type(exc).__name__
        assert outcome["native"] == outcome["python"], (v[:12], outcome)


def test_huge_thread_count_is_refused():
    env = dict(os.environ, PYFE4AI_THREADS=str(10 ** 22),
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    code = "from pyfe4ai.utils import native_backend as nb; print(nb.get_threads())"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    if res.returncode != 0:
        assert "ValueError" in res.stderr
        return
    assert int(res.stdout.strip()) <= (os.cpu_count() or 1) * 64, "PYFE4AI_THREADS accepted " + res.stdout.strip()
