"""Partial decryptions for several enrolled sets must agree."""

import pytest

from pyfe4ai.schemes.threshold_utils import combine_consistently, partial_groups
from pyfe4ai.utils.exceptions import FESchemeError


def test_groups_are_normalised():
    one = {"sid_0": 1, "sid_1": 2}
    assert partial_groups(one) == [one]
    assert partial_groups([one, one]) == [one, one]
    with pytest.raises(FESchemeError):
        partial_groups([])


def test_disagreeing_groups_are_rejected():
    combine = lambda group: sum(group.values())
    assert combine_consistently([{"a": 1, "b": 2}, {"a": 1, "c": 2}], combine) == 3
    with pytest.raises(FESchemeError, match="disagree"):
        combine_consistently([{"a": 1, "b": 2}, {"a": 1, "c": 52}], combine)


# -- scheme level: s = 3, t = 2, partial decryptions for all three pairs -------

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFE, ThresholdMCFEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFE, ThresholdMIFEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

NIDS = ["nid_0", "nid_1"]
SIDS = ["sid_0", "sid_1", "sid_2"]
PAIRS = [["sid_0", "sid_1"], ["sid_0", "sid_2"], ["sid_1", "sid_2"]]
X = {"nid_0": [3], "nid_1": [-2]}
Y = {"nid_0": [2], "nid_1": [5]}  # <x, y> = -4


def _mife_ddh():
    kg = ThresholdMIFEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2, "eta": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMIFE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n]) for n in NIDS}
    dks = {s: kg.get_decryption_keys(s, credentials={"fusion_weight": Y}) for s in SIDS}
    share = lambda s, e: ThresholdMIFE({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(cts, Y, dks[s], e)
    tamper = lambda part: dict(part, w=gp.digits(gp.mpz(part["w"]) * gp.mpz(pp["g"]) % gp.mpz(pp["p"])))
    return ThresholdMIFE({"id": "c", "keys": {"pp": pp}}), share, tamper


def _mife_lwe():
    kg = ThresholdMIFELWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2,
                                       "eta": 1, "lwe_n": 16, "bound_x": 4, "bound_y": 5})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n]) for n in NIDS}
    dks = {s: kg.get_decryption_keys(s, credentials={"fusion_weight": Y}) for s in SIDS}
    share = lambda s, e: ThresholdMIFELWE({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(cts, Y, dks[s], e)
    q, delta = gp.mpz(pp["q"]), gp.mpz(pp["q"]) // gp.mpz(pp["p"])
    tamper = lambda part: dict(part, w=gp.digits((gp.mpz(part["w"]) + 3 * delta) % q))
    return ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}), share, tamper


def _mcfe_ddh():
    kg = ThresholdMCFEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2, "eta": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMCFE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n], "r1") for n in NIDS}
    dks = {s: kg.get_decryption_keys(s, credentials={"fusion_weight": Y}) for s in SIDS}
    share = lambda s, e: ThresholdMCFE({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(cts, Y, dks[s], e, label="r1")
    tamper = lambda part: dict(part, u_d=gp.digits(gp.mpz(part["u_d"]) * gp.mpz(pp["g"]) % gp.mpz(pp["p"])))
    return ThresholdMCFE({"id": "c", "keys": {"pp": pp}}), share, tamper


def _mcfe_lwe():
    kg = ThresholdMCFELWEKeyGenerator({"n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2, "eta": 1, "lwe_n": 16})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMCFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n], "r1") for n in NIDS}
    cred = {"fusion_weight": Y, "label": "r1"}
    dks = {s: kg.get_decryption_keys(s, credentials=cred) for s in SIDS}
    share = lambda s, e: ThresholdMCFELWE({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(cts, cred, dks[s], e)
    tamper = lambda part: dict(part, w=gp.digits((gp.mpz(part["w"]) + 3 * gp.mpz(pp["delta"])) % gp.mpz(pp["q"])))
    return ThresholdMCFELWE({"id": "c", "keys": {"pp": pp}}), share, tamper


def _mcfe_ring():
    kg = ThresholdMCFERingLWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2,
                                           "eta": 1, "ring_n": 16, "bound_x": 4, "bound_y": 5})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMCFERingLWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([X[n]], "r1") for n in NIDS}
    cred = {"fusion_weight": Y, "label": "r1"}
    dks = {s: kg.get_decryption_keys(s, credentials=cred) for s in SIDS}
    share = lambda s, e: ThresholdMCFERingLWE({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).share_decrypt(cts, cred, dks[s], e)
    q, delta = gp.mpz(pp["q"]), gp.mpz(pp["q"]) // gp.mpz(pp["p"])
    tamper = lambda part: dict(part, w=[gp.digits((gp.mpz(part["w"][0]) + 3 * delta) % q)] + part["w"][1:])
    return ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}), share, tamper


@pytest.mark.parametrize("build", [_mife_ddh, _mife_lwe, _mcfe_ddh, _mcfe_lwe, _mcfe_ring])
def test_partial_decryptions_for_several_enrolled_sets_must_agree(build, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    combiner, share, tamper = build()
    groups = [{s: share(s, pair) for s in pair} for pair in PAIRS]
    honest = combiner.combine_decrypt(groups)
    assert honest == combiner.combine_decrypt(groups[0])
    assert honest in (-4, -4.0, [-4])
    forged = [{s: (tamper(p) if s == "sid_1" else p) for s, p in g.items()} for g in groups]
    with pytest.raises(FESchemeError, match="disagree"):
        combiner.combine_decrypt(forged)
