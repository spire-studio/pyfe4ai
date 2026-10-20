"""One-time use of the offline phases (pyfe4ai.core.one_time).

Every precomputation -- the randomness of SIFE/MIFE-Paillier and MIFE-DDH,
the label masks of MCFE-DDH/Damgard/Paillier, the Ring-LWE MCFE chunks on
both backends -- is claimed atomically, refuses copies and pickles, and is
not usable in another process. Refused inputs do not consume it.
"""

from __future__ import annotations

import copy
import os
import pickle
import threading

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgard, MCFEDamgardKeyGenerator
from pyfe4ai.schemes.mcfe.ddh import MCFE, MCFEKeyGenerator
from pyfe4ai.schemes.mife import paillier as mife_paillier
from pyfe4ai.schemes.sife import paillier as sife_paillier
from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEValidationError

NIDS = ["c0", "c1", "c2"]
BACKENDS = ["python"] + (["native"] if native_backend.available() else [])


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _sife():
    kg = sife_paillier.SIFEPaillierKeyGenerator({"eta": 1, "bit_length": 256, "bound_x": 20, "bound_y": 3})
    kg.setup()
    enc = sife_paillier.SIFEPaillier({"keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys()},
                                      "precision": 0})
    return enc, lambda shape: enc.precompute([shape]), lambda x, pre: enc.encrypt_lst_ndarray([x], precomputed=pre)


def _mife_paillier():
    kg = mife_paillier.MIFEPaillierKeyGenerator({"bit_length": 256, "bound_x": 20, "bound_y": 3, "n": 2, "eta": 1,
                                                 "lst_nid": ["a", "b"]})
    kg.setup()
    enc = mife_paillier.MIFEPaillier({"id": "a", "precision": 0,
                                      "keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys("a")}})
    return enc, lambda shape: enc.precompute([shape]), lambda x, pre: enc.encrypt_lst_ndarray([x], precomputed=pre)


def _mcfe(kg_cls, cls, cfg):
    def make():
        kg = kg_cls(dict(cfg, n=len(NIDS), eta=1, lst_nid=NIDS))
        kg.setup()
        enc = cls({"id": "c0", "precision": 0,
                   "keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys("c0")}})
        return (enc, lambda shape: enc.precompute("L", [shape]),
                lambda x, pre: enc.encrypt_lst_ndarray([x], label="L", precomputed=pre))
    return make


def _ring_lwe():
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": NIDS, "eta": 1, "ring_n": 16, "sec_param": 64,
                                            "bound_x": 20, "bound_y": 1})
    kg.setup()
    enc = mcfe_rlwe.MCFERingLWE({"id": "c0", "precision": 0,
                                 "keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys("c0")}})
    return (enc, lambda shape: enc.precompute("L", [shape]),
            lambda x, pre: enc.encrypt_lst_ndarray([x], label="L", precomputed=pre))


CASES = {
    "sife/paillier": _sife,
    "mife/paillier": _mife_paillier,
    "mcfe/ddh": _mcfe(MCFEKeyGenerator, MCFE, {"sec_param": 256, "bound_x": 20, "bound_y": 3}),
    "mcfe/damgard": _mcfe(MCFEDamgardKeyGenerator, MCFEDamgard, {"sec_param": 256, "bound": 20}),
    "mcfe/ring_lwe": _ring_lwe,
}
# the schemes whose online phase checks a plaintext bound (as their plain call)
BOUNDED = {"mife/paillier", "mcfe/damgard", "mcfe/ring_lwe"}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_copies_and_pickles_are_refused(case, backend):
    with native_backend.backend(backend):
        _, precompute, _ = CASES[case]()
        pre = precompute((3,))
        for clone in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError, match="one-time precomputation"):
                clone(pre)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_refused_input_does_not_consume_it(case, backend):
    with native_backend.backend(backend):
        _, precompute, encrypt = CASES[case]()
        pre = precompute((3,))
        with pytest.raises(FEValidationError, match="shapes"):
            encrypt(np.arange(4), pre)
        if case in BOUNDED:
            with pytest.raises(FEValidationError, match="bound"):
                encrypt(np.array([1, 2, 10 ** 6]), pre)
        assert not pre.used
        encrypt(np.arange(3), pre)
        assert pre.used
        with pytest.raises(FEValidationError, match="once"):
            encrypt(np.arange(3), pre)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_one_of_many_threads_gets_it(case, backend):
    with native_backend.backend(backend):
        _, precompute, encrypt = CASES[case]()
        pre = precompute((64,))
        results, barrier = [], threading.Barrier(4)

        def run():
            with native_backend.backend(backend):
                barrier.wait()
                try:
                    encrypt(np.arange(64) % 20, pre)
                    results.append("ok")
                except FEValidationError:
                    results.append("refused")

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert sorted(results) == ["ok", "refused", "refused", "refused"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_a_forked_child_cannot_use_it(case, backend):
    with native_backend.backend(backend):
        _, precompute, encrypt = CASES[case]()
        pre = precompute((3,))
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            try:
                with native_backend.backend(backend):
                    encrypt(np.arange(3), pre)
                os.write(write, b"used")
            except FEValidationError:
                os.write(write, b"refused" if pre.used else b"refused, not invalidated")
            except BaseException as exc:  # noqa: BLE001
                os.write(write, type(exc).__name__.encode())
            os._exit(0)
        os.close(write)
        outcome = os.read(read, 100).decode()
        os.close(read)
        os.waitpid(pid, 0)
        assert outcome == "refused"
        encrypt(np.arange(3), pre)  # still the parent's, once


def test_native_precomputation_refuses_another_process():
    if not native_backend.available() or not hasattr(os, "fork"):
        pytest.skip("needs the native extension and fork")
    with native_backend.backend("native"):
        _, precompute, _ = _ring_lwe()
        pre = precompute((16,))
    nkey, cap = pre.native
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the capsule itself (as a direct caller of the extension would)
        try:
            native_backend.ext().encrypt_precomputed(nkey, cap, 0, np.ones(16, dtype=np.int64), 1, False)
            os.write(write, b"used")
        except ValueError as exc:
            os.write(write, str(exc).encode())
        os._exit(0)
    os.close(write)
    outcome = os.read(read, 200).decode()
    os.close(read)
    os.waitpid(pid, 0)
    assert outcome == "precomputation made in another process"
