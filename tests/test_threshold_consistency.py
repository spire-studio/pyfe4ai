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


def test_ndarray_combiners_refuse_bad_partials(tmp_path, monkeypatch):
    """A disagreeing partial crashed (TypeError) or gave NaN; a reshaped
    output was taken from the first server."""
    import numpy as np

    from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

    monkeypatch.chdir(tmp_path)
    kg = ThresholdMIFELWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2,
                                       "eta": 1, "lwe_n": 16, "bound_x": 50, "bound_y": 5})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMIFELWE({"id": n, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
           .encrypt_lst_ndarray([np.ones((2, 2))]) for n in NIDS}
    dks = {s: kg.get_decryption_keys(s, credentials={"fusion_weight": Y}) for s in SIDS}
    enrolled = ["sid_0", "sid_1"]
    parts = {s: ThresholdMIFELWE({"id": s, "keys": {"pp": pp, "dk": dks[s]}}).compute_lst_ndarray_ct(cts, Y, dks[s], enrolled)
             for s in enrolled}
    combiner = ThresholdMIFELWE({"id": "c", "precision": 1, "keys": {"pp": pp}})
    assert np.allclose(combiner.decrypt_lst_ndarray_ct(parts)[0], 7)
    bad = parts["sid_1"][0].copy()
    bad[0, 0] = dict(bad[0, 0], c_y="1")
    with pytest.raises(FESchemeError, match="does not decrypt"):
        combiner.decrypt_lst_ndarray_ct(dict(parts, sid_1=[bad]))
    with pytest.raises(FESchemeError, match="shape"):
        combiner.decrypt_lst_ndarray_ct(dict(parts, sid_1=[parts["sid_1"][0][:1]]))


def test_noise_depends_on_the_lagrange_coefficient_not_the_enrolled_set(tmp_path, monkeypatch):
    """sid_0 has lambda = 2 in both {sid_0, sid_1} and {sid_0, sid_2, sid_3};
    seeding the noise with the set drew fresh noise for the same key part."""
    from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

    monkeypatch.chdir(tmp_path)
    sids = ["sid_0", "sid_1", "sid_2", "sid_3"]
    kg = ThresholdMIFELWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 4, "lst_sid": sids, "t": 2,
                                       "eta": 1, "lwe_n": 16, "bound_x": 4, "bound_y": 5})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n]) for n in NIDS}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": Y})
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    a = server.share_decrypt(cts, Y, dk, ["sid_0", "sid_1"])
    b = server.share_decrypt(cts, Y, dk, ["sid_0", "sid_2", "sid_3"])
    assert a["w"] == b["w"]


def test_ring_noise_keys_do_not_change_with_a_resharing(tmp_path, monkeypatch):
    """A key requested with another n_chunks was re-shared under a fresh noise
    key, so the same key part came with fresh noise."""
    monkeypatch.chdir(tmp_path)
    kg = ThresholdMCFERingLWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2,
                                           "eta": 1, "ring_n": 16, "bound_x": 4, "bound_y": 5})
    kg.setup()
    one = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": Y, "label": "r", "n_chunks": 1})
    two = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": Y, "label": "r", "n_chunks": 2})
    assert one["noise_key"] == two["noise_key"]


@pytest.mark.parametrize("partials", [
    {"sid_0": [[[1]]], "sid_1": [[[1]]]},                 # nested lists, not arrays
    [],                                                   # no group at all
    {},                                                   # an empty group
])
def test_elementwise_combination_raises_scheme_errors_on_malformed_output(partials):
    from pyfe4ai.schemes.threshold_utils import combine_elementwise

    with pytest.raises(FESchemeError):
        combine_elementwise(partials, lambda groups, l, i: 1)



def test_keys_of_the_earlier_flooding_derivation_are_refused():
    """A key share issued before the flooding precision changed (no noise_version)
    may have answered requests with the earlier noise: every threshold LWE server
    refuses it, and new key shares carry the current version."""
    from pyfe4ai.utils.exceptions import FEKeyError
    from pyfe4ai.utils.sampling_utils import FLOODING_NOISE_VERSION, check_flooding_version

    base = {"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2, "eta": 1}
    cases = [
        (ThresholdMIFELWEKeyGenerator({**base, "lwe_n": 16, "bound_x": 4, "bound_y": 5}), ThresholdMIFELWE,
         {"fusion_weight": Y}, lambda c, cts, dk: c.share_decrypt(cts, Y, dk, SIDS[:2]),
         lambda kg, pp, n: ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n])),
        (ThresholdMCFELWEKeyGenerator({**base, "lwe_n": 16}), ThresholdMCFELWE,
         {"fusion_weight": Y, "label": "r1"},
         lambda c, cts, dk: c.share_decrypt(cts, {"fusion_weight": Y, "label": "r1"}, dk, SIDS[:2]),
         lambda kg, pp, n: ThresholdMCFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt(X[n], "r1")),
        (ThresholdMCFERingLWEKeyGenerator({**base, "ring_n": 16, "bound_x": 4, "bound_y": 5}), ThresholdMCFERingLWE,
         {"fusion_weight": Y, "label": "r1"},
         lambda c, cts, dk: c.share_decrypt(cts, {"fusion_weight": Y, "label": "r1"}, dk, SIDS[:2]),
         lambda kg, pp, n: ThresholdMCFERingLWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([X[n]], "r1")),
    ]
    for kg, cls, cred, share, enc in cases:
        kg.setup()
        pp = kg.get_public_parameters()
        cts = {n: enc(kg, pp, n) for n in NIDS}
        dk = kg.get_decryption_keys(SIDS[0], credentials=cred)
        assert dk["noise_version"] == FLOODING_NOISE_VERSION
        server = cls({"id": SIDS[0], "keys": {"pp": pp, "dk": dk}})
        share(server, cts, dk)  # accepted
        old = {k: v for k, v in dk.items() if k != "noise_version"}
        with pytest.raises(FEKeyError, match="flooding derivation 1"):
            share(server, cts, old)
    with pytest.raises(FEKeyError):
        check_flooding_version({"noise_version": 3})
    with pytest.raises(FEKeyError, match="flooding sampler"):
        check_flooding_version({"noise_version": FLOODING_NOISE_VERSION, "noise_sampler": "facct"})


def test_flooding_noise_ignores_the_selected_sampler():
    """The flooding sampler is pinned: one request gets the same noise whatever
    sampler a gaussian_sampler() block or the process selects."""
    from pyfe4ai.utils import sampling_utils as su

    kg = ThresholdMCFERingLWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2,
                                           "eta": 1, "ring_n": 16, "bound_x": 4, "bound_y": 5})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMCFERingLWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([X[n]], "r1")
           for n in NIDS}
    cred = {"fusion_weight": Y, "label": "r1"}
    dk = kg.get_decryption_keys(SIDS[0], credentials=cred)
    assert dk["noise_sampler"] == su.FLOODING_SAMPLER
    server = ThresholdMCFERingLWE({"id": SIDS[0], "keys": {"pp": pp, "dk": dk}})
    first = server.share_decrypt(cts, cred, dk, SIDS[:2])["w"]
    with su.gaussian_sampler("facct"):
        assert server.share_decrypt(cts, cred, dk, SIDS[:2])["w"] == first
    default = su.get_gaussian_sampler()
    try:
        su.set_gaussian_sampler("facct")
        assert server.share_decrypt(cts, cred, dk, SIDS[:2])["w"] == first
    finally:
        su.set_gaussian_sampler(default)


def test_stat_param_is_capped_by_the_mpfr_guard():
    from pyfe4ai.schemes import profiles
    from pyfe4ai.utils import sampling_utils as su
    from pyfe4ai.utils.exceptions import FEValidationError

    base = {"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "t": 2, "eta": 1, "ring_n": 16,
            "bound_x": 4, "bound_y": 5}
    kg = ThresholdMCFERingLWEKeyGenerator({**base, "stat_param": profiles.MAX_STAT_PARAM})
    sigma = int(kg.sigma_f)
    assert su.bigint_precision(sigma) - sigma.bit_length() >= profiles.MAX_STAT_PARAM + 56 + 8
    for bad in (profiles.MAX_STAT_PARAM + 1, True, 64.0):
        with pytest.raises(FEValidationError, match="stat_param"):
            ThresholdMCFERingLWEKeyGenerator({**base, "stat_param": bad})
