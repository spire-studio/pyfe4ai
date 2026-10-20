"""Exact sampler by default (October 2026 code review).

No default draw from the double-precision rounded path was found in the nine
schemes in scope (every key and noise site, backends, threads, profiles, ML
adapter). Before the fixes:

* the threshold flooding noise followed the process-wide sampler: one server
  answered the same request with a different noise after
  ``set_gaussian_sampler`` / ``gaussian_sampler()`` changed -- the "second
  noise for the same request" the noise versioning is meant to prevent;
* ``security_profile="standard"`` accepted ``gaussian_sampler="rounded"``
  without a refusal or a warning, although the profile documents that the
  double path does not meet its 2^-128 budget;
* the MPFR guard was a fixed 192 bits (covering stat_param <= 128), while the
  standard profile accepted a larger stat_param for the flooding.
"""

import warnings

import pytest

from pyfe4ai.utils import sampling_utils as su
from pyfe4ai.utils.exceptions import FESecurityWarning, FEValidationError

N, S = ["a", "b"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    su.set_gaussian_sampler("rounded")


# the default tMCFE flooding is "uniform", which draws no Gaussian, so the default
# config alone does not exercise the sampler pin
@pytest.mark.parametrize("flooding", ["gaussian", "uniform"])
def test_flooding_noise_does_not_follow_the_process_sampler(flooding):
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE as T
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator as TK

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kg = TK({"sec_param": 64, "eta": 1, "ring_n": 32, "bound_x": 4, "bound_y": 4, "n": 2, "lst_nid": N,
                 "s": 3, "t": 2, "lst_sid": S, "flooding": flooding, "key_shares": "full"})
        kg.setup()
    pp = kg.get_public_parameters()
    assert pp["flooding"] == flooding
    cts = {n: T({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([[1, 2]], "L") for n in N}
    cred = {"fusion_weight": {n: [1] for n in N}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=cred)
    srv = T({"id": "s0", "keys": {"pp": pp, "dk": dk}})
    first = srv.share_decrypt(cts, cred, dk, ["s0", "s1"])["w"]
    with su.gaussian_sampler("facct"):
        second = srv.share_decrypt(cts, cred, dk, ["s0", "s1"])["w"]
    assert first == second, "one request answered with two flooding noises"


def test_standard_profile_refuses_or_warns_about_rounded():
    from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            MCFERingLWEKeyGenerator({"n": 2, "lst_nid": N[:2], "eta": 1, "bound_x": 100, "bound_y": 1,
                                     "security_profile": "standard", "gaussian_sampler": "rounded"})
        except FEValidationError:
            return
    assert any(issubclass(w.category, FESecurityWarning) for w in caught), "rounded accepted silently under standard"


def test_mpfr_guard_covers_the_accepted_stat_param():
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator as TK

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # the MPFR guard concerns the Gaussian flooding (the default flooding is
            # uniform)
            kg = TK({"n": 2, "lst_nid": N, "s": 2, "lst_sid": S[:2], "t": 2, "eta": 1, "bound_x": 4, "bound_y": 4,
                     "stat_param": 256, "security_profile": "standard", "flooding": "gaussian"})
    except FEValidationError:
        return  # stat_param above the guard refused
    sigma = int(kg.sigma_f)
    guard = su.bigint_precision(sigma) - int(sigma).bit_length()
    assert guard >= 256 + 8, "stat_param 256 accepted with {} MPFR guard bits".format(guard)
