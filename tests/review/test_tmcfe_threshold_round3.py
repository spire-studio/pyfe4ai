"""Remaining items of the threshold MCFE review (October 2026 code review).

Before the fixes:

* one parameter record per scheme type: a prime and an RNS configuration in
  one config folder overwrote each other's record, so alternating them
  regenerated the parameters (and ``a``) every time;
* an RNS modulus of 241 to 248 bits got five factors of at most 60 bits, one
  more than the native ring holds, so its kernels ran on the Python path,
  although four factors below 2^62 fit;
* ``_STORE_LOCK`` and ``_SHARE_CACHE_LOCK`` -- like the older
  ``native_backend._contexts_lock`` -- were not replaced in a forked child: a
  fork while another thread held one left the child's copy locked for ever,
  and its next split-share or replicated partial decryption hung
  (``sampling._XOF_LOCK`` is reset after a fork for this reason);
* a polynomial given as a list of values holding a bytes element passed
  ``poly_kinds`` (it looks at the first element); the native NTT and partial
  readings raised ``TypeError`` instead of ``FEKeyError``.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.schemes.threshold import store as threshold_store
from pyfe4ai import backend as native_backend

NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _kg(**extra):
    # the defaults when these tests were written: JSON, a prime q, whole key shares
    cfg = dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=SIDS, eta=1, ring_n=32, sec_param=64, bound_x=100, bound_y=3,
               stat_param=64, ct_format="json", ring_modulus="prime", key_shares="full")
    cfg.update(extra)
    return tm.ThresholdMCFERingLWEKeyGenerator(cfg)


def test_i7_prime_and_rns_records_do_not_overwrite_each_other():
    prime, rns = _kg(), _kg(ring_modulus="rns")
    assert _kg().a_seed == prime.a_seed, "the RNS configuration overwrote the prime record"
    assert _kg(ring_modulus="rns").a_seed == rns.a_seed, "the prime configuration overwrote the RNS record"


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
def test_i8_rns_moduli_up_to_248_bits_run_natively():
    kg = _kg(ring_modulus="rns", stat_param=172)  # q of about 241 to 248 bits
    bits = int(kg.q).bit_length()
    assert 240 < bits <= 248, bits
    assert len(kg.q_factors) == 4, "{} factors for a {}-bit q".format(len(kg.q_factors), bits)
    with native_backend.backend("native"):
        ctx = native_backend.ring_context(kg.q, kg.ring_n)
    assert ctx is not None and not ctx.legacy_rns


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() of a process with threads: the point here
@pytest.mark.parametrize("where,name", [(threshold_store, "_STORE_LOCK"), (threshold_store, "_SHARE_CACHE_LOCK"),
                                        (native_backend, "_contexts_lock")])
def test_n1_module_locks_are_new_in_a_forked_child(where, name):
    held, release = threading.Event(), threading.Event()

    def holder():
        with getattr(where, name):
            held.set()
            release.wait(10)

    th = threading.Thread(target=holder)
    th.start()
    held.wait(10)
    pid = os.fork()
    if pid == 0:  # child: the lock as the module now has it
        ok = getattr(where, name).acquire(timeout=2)
        os._exit(0 if ok else 1)
    release.set()
    th.join()
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, \
        "{}.{} is still held in the forked child".format(where.__name__, name)


@pytest.mark.parametrize("backend", ["python"] + (["native"] if native_backend.available() else []))
@pytest.mark.parametrize("ct_format", ["json", "binary"])
@pytest.mark.parametrize("field", ["sk_y", "z"])
def test_i2b_native_a_value_list_holding_bytes_is_a_key_error(backend, ct_format, field):
    """Native readings: a polynomial that is a list of values with a
    bytes element passes poly_kinds (it looks at the first element) and the
    native NTT / partial reading raises TypeError instead of FEKeyError."""
    from pyfe4ai.core.formats import binary as fmt_binary
    from pyfe4ai.core.exceptions import FEKeyError, FESchemeError

    kg = _kg(t=3, ct_format=ct_format)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    cts = {nid: cl.encrypt([[1, 2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)

    def as_list(poly):
        vals = fmt_binary.decode_values(poly, pp["q"], pp["ring_n"]) if isinstance(poly, bytes) else poly
        return [str(int(v)) for v in vals][:-1] + [b"\x01\x02"]

    bad = dict(dk, z=as_list(dk["z"])) if field == "z" else dict(dk, sk_y=dict(dk["sk_y"], c1=as_list(dk["sk_y"]["c1"])))
    with native_backend.backend(backend):
        srv = tm.ThresholdMCFERingLWE({"id": "s0", "precision": 0, "keys": {"pp": pp}})
        try:
            srv.share_decrypt(cts, creds, bad, SIDS)
        except (FEKeyError, FESchemeError):
            return
        except Exception as exc:  # noqa: BLE001
            pytest.fail("{}: {}".format(type(exc).__name__, exc))
        pytest.fail("served")
