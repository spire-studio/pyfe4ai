"""Opt-in KL accounting of the flooding noise (October 2026 code review).

Before the fixes:

* KL mode was accepted with t < s, where an honest partial decryption is the
  simulation plus e - M nu with a rational Lagrange-derived M (the
  Lagrange-multiplier leak, tests/test_threshold_lagrange_leak.py): with
  s = 3, t = 2, server 2 corrupted and the enrolled set {1, 3} (M = 3/4),
  4 w - 3 (c_y - Delta mu) - 3 F(2) = 4 e - 3 nu gives nu mod 4 from every KL
  partial, whatever sigma_f; the KL bound assumes M = 1;
* the partial-decryption budget Q failed open: it lived in each server object
  (a fresh object per request had a fresh budget), the public keyword
  ``_budgeted=True`` skipped it, and a shared budget with a larger limit than
  the public parameters' Q was accepted;
* refused (invalid) requests consumed budget.
"""

import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.threshold import flooding_accounting as fa
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.matrix_utils import digits_to_vector

NIDS = ["nid_0", "nid_1", "nid_2"]
SIDS = ["sid_0", "sid_1", "sid_2"]
TMIFE = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS,
         "lwe_n": 16, "bound_x": 6, "bound_y": 6}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _tmife(q_max, **extra):
    kg = ThresholdMIFELWEKeyGenerator({**TMIFE, **extra, "flooding_accounting": "kl", "max_partial_decryptions": q_max})
    kg.setup()
    pp = kg.get_public_parameters()
    encs = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    return kg, pp, encs


def test_kl_partials_do_not_reveal_nu_mod_d_with_t_below_s():
    try:
        kg, pp, encs = _tmife(1000)
    except (FEValidationError, ValueError):
        return  # KL refused for t < s: the fix
    y = {nid: [random.randint(-6, 6)] for nid in NIDS}
    cred = {"fusion_weight": y}
    dk0 = kg.get_decryption_keys("sid_0", credentials=cred)
    dk1 = kg.get_decryption_keys("sid_1", credentials=cred)  # the corrupted server
    full = MIFELWEKeyGenerator.get_decryption_keys(kg, "sid_0", credentials=cred)  # ground truth only
    srv0 = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk0}})
    q = gp.mpz(pp["q"])
    delta = q // gp.mpz(pp["p"])

    def keypart(sk, z, cts):
        acc = delta * gp.mpz(z)
        for nid, ct in cts.items():
            ct0 = [gp.mpz(v) % q for v in digits_to_vector(ct["ct0"])]
            acc += sum(a * gp.mpz(b) for a, b in zip(ct0, sk[nid]))
        return acc % q

    def centred(v):
        v %= q
        return int(v - q) if v > q // 2 else int(v)

    hits = 0
    for _ in range(10):
        x = {nid: [random.randint(-6, 6)] for nid in NIDS}
        cts = {nid: encs[nid].encrypt(x[nid]) for nid in NIDS}
        w0 = gp.mpz(srv0.share_decrypt(cts, y, dk0, ["sid_0", "sid_2"])["w"])
        mu = sum(x[n][0] * y[n][0] for n in NIDS)
        c_y = sum(gp.mpz(y[n][0]) * gp.mpz(digits_to_vector(cts[n]["ct1"])[0]) for n in NIDS) % q
        big_x = centred(4 * w0 - 3 * (c_y - delta * mu) - 3 * keypart(dk1["sk_y"], dk1["z"], cts))
        nu = centred(c_y - delta * mu - keypart(full["sk_y"], full["z"], cts))
        hits += (3 * -big_x) % 4 == nu % 4
    assert hits < 10, "nu mod 4 read off {} of 10 KL partials (s = 3, t = 2)".format(hits)


def _request(encs, i):
    return {nid: encs[nid].encrypt([i % 5]) for nid in NIDS}


def test_budget_holds_across_server_objects():
    kg, pp, encs = _tmife(2, t=3)  # t = s: the leak above does not apply
    y = {nid: [1] for nid in NIDS}
    dk0 = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    served = 0
    for i in range(4):
        try:
            ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk0}}).share_decrypt(
                _request(encs, i), y, dk0, SIDS)
            served += 1
        except FESchemeError:
            pass
    assert served <= pp["max_partial_decryptions"], "served {} partials, Q = 2".format(served)


def test_budget_cannot_be_skipped_or_enlarged():
    kg, pp, encs = _tmife(2, t=3)
    y = {nid: [1] for nid in NIDS}
    dk0 = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    srv = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk0}})
    with pytest.raises((TypeError, FESchemeError)):
        for i in range(3):
            srv.share_decrypt(_request(encs, i), y, dk0, SIDS, _budgeted=True)
    with pytest.raises((FEValidationError, FESchemeError, ValueError)):
        big = fa.PartialDecryptionBudget(10 ** 9)
        srv = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk0}, "partial_budget": big})
        for i in range(3):
            srv.share_decrypt(_request(encs, i), y, dk0, SIDS)
    with pytest.raises((FEValidationError, ValueError)):
        fa.PartialDecryptionBudget(2, used=-10 ** 6)


def test_refused_requests_do_not_consume_budget():
    kg, pp, encs = _tmife(100, t=3)
    y = {nid: [1] for nid in NIDS}
    dk0 = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    budget = fa.PartialDecryptionBudget(100)  # explicit (the fixed code requires it; the first version accepted it)
    srv = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk0}, "partial_budget": budget})
    grads = {nid: [np.array([1, 2, 3])] for nid in NIDS}
    enc = {nid: encs[nid].encrypt_lst_ndarray(grads[nid]) for nid in NIDS}
    for bad in (["sid_0"], ["sid_0", "sid_0", "sid_1"], ["sid_1", "sid_2"]):
        with pytest.raises(FESchemeError):
            srv.compute_lst_ndarray_ct(enc, y, dk0, bad)
    assert budget.used == 0, "refused requests consumed {} units".format(budget.used)
    srv.compute_lst_ndarray_ct(enc, y, dk0, SIDS)
    assert budget.used == 3
