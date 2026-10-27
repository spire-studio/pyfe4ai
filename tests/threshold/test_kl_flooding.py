"""KL accounting of the threshold flooding (``flooding_accounting = "kl"``):
parameters, records, the t = s condition, the run-time limit, the sampler,
and that the default is untouched."""

import json
import math
import os
import random
import threading
import time

import numpy as np
import pytest

from pyfe4ai.core import config as configs
from pyfe4ai.schemes.threshold import kl as fa
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives import sampling as sampling_utils
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.primitives.lattice.lwe_utils import check_wide_lwe
from pyfe4ai.primitives.lattice.ring_lwe_utils import ring_lwe_bound
from pyfe4ai.primitives.sampling import XofRandom

NIDS = ["nid_0", "nid_1", "nid_2"]
SIDS = ["sid_0", "sid_1", "sid_2"]
# the default accounting with t < s, as the existing threshold tests use it
# whole key shares: the servers below are given one key share (split ones: test_threshold_split_shares.py)
TMCFE = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS,
         "ring_n": 16, "bound_x": 4, "bound_y": 4, "key_shares": "full"}
TMIFE = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS,
         "lwe_n": 16, "bound_x": 6, "bound_y": 6}
# "kl" needs t = s
KL = {"t": 3, "flooding_accounting": "kl", "max_partial_decryptions": 1000}
KL_FIELDS = {"flooding_accounting", "max_partial_decryptions", "stat_param", "flood_tail_cut"}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _tmcfe_noise_bound(kg) -> int:
    C = GAUSSIAN_TAIL_CUT
    return kg.n * kg.eta * int(kg.bound_y) * int(2 * kg.ring_n * C ** 2 * kg.sigma1 * kg.sigma2 + C * kg.sigma3 + 1)


def _tmife_noise_bound(kg) -> int:
    return kg.n * kg.eta * int(kg.bound_y) * kg.m * int(GAUSSIAN_TAIL_CUT * kg.sigma_q + 1)


def _budgets(pp, limit=None):
    return {sid: fa.PartialDecryptionBudget(limit or pp["max_partial_decryptions"]) for sid in SIDS}


# -- configuration -------------------------------------------------------------


def test_default_accounting_is_per_sample():
    assert configs.flooding_config({}) == ("per-sample", None)
    assert configs.flooding_config({"flooding_accounting": "per-sample"}) == ("per-sample", None)
    assert configs.flooding_config(KL) == ("kl", 1000)


@pytest.mark.parametrize("config, match", [
    ({"flooding_accounting": "renyi"}, "unknown flooding_accounting"),
    ({"flooding_accounting": "kl"}, "needs max_partial_decryptions"),
    ({"flooding_accounting": "kl", "max_partial_decryptions": 0}, "needs max_partial_decryptions"),
    ({"flooding_accounting": "kl", "max_partial_decryptions": -5}, "needs max_partial_decryptions"),
    ({"flooding_accounting": "kl", "max_partial_decryptions": 2.0 ** 30}, "needs max_partial_decryptions"),
    ({"flooding_accounting": "kl", "max_partial_decryptions": True}, "needs max_partial_decryptions"),
    ({"max_partial_decryptions": 100}, "applies to flooding_accounting='kl' only"),
])
def test_invalid_configs_are_refused(config, match):
    with pytest.raises(FEValidationError, match=match):
        configs.flooding_config(config)
    for kg_cls, base in ((ThresholdMCFERingLWEKeyGenerator, TMCFE), (ThresholdMIFELWEKeyGenerator, TMIFE)):
        with pytest.raises(FEValidationError, match=match):
            kg_cls({**base, "t": 3, **config})


@pytest.mark.parametrize("kg_cls, base", [
    (ThresholdMCFERingLWEKeyGenerator, TMCFE), (ThresholdMIFELWEKeyGenerator, TMIFE),
])
@pytest.mark.parametrize("t", [1, 2])
def test_kl_needs_t_equal_to_s(kg_cls, base, t):
    with pytest.raises(FEValidationError, match="needs t = s"):
        kg_cls({**base, **KL, "t": t})
    kg_cls({**base, **KL})  # t = s = 3


def test_kl_public_parameters_with_t_below_s_are_refused():
    kg = ThresholdMIFELWEKeyGenerator({**TMIFE, **KL})
    kg.setup()
    pp = dict(kg.get_public_parameters(), t=2)
    with pytest.raises(FESchemeError, match="t != s"):
        fa.kl_flooding(pp, 1)


def test_dense_lwe_threshold_refuses_the_option():
    cfg = {"eta": 1, "n": 2, "s": 3, "t": 3, "lst_nid": NIDS[:2], "lst_sid": SIDS, "bound_x": 4, "bound_y": 4}
    for extra in (KL, {"max_partial_decryptions": 10}):
        with pytest.raises(FEValidationError, match="no statistical flooding"):
            ThresholdMCFELWEKeyGenerator({**cfg, **extra})


# -- width, tail cut, sampler -----------------------------------------------------


@pytest.mark.parametrize("bound, samples, stat_param", [
    (1, 1, 1), (3, 7, 64), (2 ** 71 + 12345, 2 ** 53, 128), (10 ** 20, 3 * 8192 * 1000, 128), (5, 2 ** 40, 80),
])
def test_kl_width_is_the_smallest_meeting_the_bound(bound, samples, stat_param):
    sigma = fa.kl_flooding_width(bound, samples, stat_param)
    # samples * bound^2 / (2 sigma^2) <= 2^-(stat_param + 1), exactly in integers
    assert (samples * bound * bound) << (stat_param + 1) <= 2 * sigma * sigma
    assert (samples * bound * bound) << (stat_param + 1) > 2 * (sigma - 1) * (sigma - 1)
    assert fa.kl_divergence_log2(bound, samples, sigma) <= -(stat_param + 1) + 1e-9


def test_kl_width_grows_with_the_square_root_of_the_run():
    bound = 2 ** 70
    one = fa.kl_flooding_width(bound, 2 ** 30, 128)
    assert math.log2(fa.kl_flooding_width(bound, 2 ** 50, 128) / one) == pytest.approx(10, abs=1e-6)
    assert math.log2(fa.kl_flooding_width(bound, 2 ** 30, 130) / one) == pytest.approx(1, abs=1e-6)


def test_gaussian_tail_bound():
    mpmath = pytest.importorskip("mpmath")  # not a dependency of the library

    for cut in (3, 12, 16, 20, 25):
        exact = float(mpmath.log(mpmath.erfc(cut / mpmath.sqrt(2)), 2))
        # Mills' ratio overshoots by a factor below 1 + 1 / cut^2
        assert exact <= fa.gaussian_tail_log2(cut) <= exact + math.log2(1 + 1 / cut ** 2)


@pytest.mark.parametrize("log_samples, stat_param, cut", [
    (16, 64, 12), (53, 128, 16), (40, 128, 16), (30, 128, 15), (54, 256, 21), (155, 128, 20), (200, 256, 26),
])
def test_tail_cut_follows_the_run(log_samples, stat_param, cut):
    samples = 2 ** log_samples
    assert fa.kl_tail_cut(samples, stat_param) == cut
    # N Pr[|D| > T sigma] (doubled) <= 2^-(stat_param + 6), and T - 1 would not do (above 12)
    assert log_samples + 1 + fa.gaussian_tail_log2(cut) <= -(stat_param + 6)
    if cut > GAUSSIAN_TAIL_CUT:
        assert log_samples + 1 + fa.gaussian_tail_log2(cut - 1) > -(stat_param + 6)


def test_sampler_precision():
    sigma = 2 ** 161 + 5
    assert fa.kl_sampler_precision(sigma, 2 ** 64, 256) == 162 + 256 + 64 + 11
    assert fa.kl_sampler_precision(sigma, 2 ** 64 + 1, 256) == 162 + 256 + 65 + 11
    # never below the sampler's own default bits(sigma) + BIGINT_GUARD_BITS
    guard = sampling_utils.BIGINT_GUARD_BITS
    assert fa.kl_sampler_precision(sigma, 2 ** 53, 128) == 162 + guard
    assert fa.kl_sampler_precision(sigma, 10, 64) == 162 + guard
    assert fa.facct_max_log_samples(128) == 51


class _Recording(XofRandom):
    def __init__(self, seed: bytes) -> None:
        super().__init__(seed)
        self.widths = []

    def getrandbits(self, k: int) -> int:
        self.widths.append(k)
        return super().getrandbits(k)


def test_rounded_sampler_takes_a_higher_precision_only():
    rng = _Recording(b"seed")
    default = sampling_utils.discrete_gaussian_bigint(2 ** 100, 8, rng=rng, sampler="rounded")
    assert set(rng.widths) == {101 + sampling_utils.BIGINT_GUARD_BITS}  # bigint_precision
    rng = _Recording(b"seed")
    higher = sampling_utils.discrete_gaussian_bigint(2 ** 100, 8, rng=rng, sampler="rounded", precision=500)
    assert set(rng.widths) == {500} and len(rng.widths) == 16
    # XofRandom draws each width from its own block: the extra bits only refine u1, u2
    assert higher == default
    with pytest.raises(ValueError, match="at least"):
        sampling_utils.discrete_gaussian_bigint(2 ** 100, 8, rng=rng, sampler="rounded",
                                                precision=100 + sampling_utils.BIGINT_GUARD_BITS)


def test_flooding_samples_respect_the_cut():
    values = fa.flooding_samples(1000, 4000, XofRandom(b"cut"), tail_cut=1)
    assert len(values) == 4000 and max(abs(v) for v in values) <= 1000
    assert any(abs(v) > 1000 for v in sampling_utils.discrete_gaussian_bigint(1000, 4000, rng=XofRandom(b"cut")))


# -- the budget -------------------------------------------------------------------


def test_budget():
    budget = fa.PartialDecryptionBudget(5)
    budget.check(5)
    assert budget.used == 0
    budget.consume()
    budget.consume(4)
    assert budget.used == 5 and budget.remaining == 0
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        budget.consume()
    assert budget.used == 5
    budget = fa.PartialDecryptionBudget(5, used=2)
    with pytest.raises(FESchemeError):
        budget.consume(4)
    assert budget.used == 2  # a refused request consumes nothing
    with pytest.raises(FEValidationError):
        budget.consume(-1)
    with pytest.raises(AttributeError):
        budget.used = 0  # the count only grows
    with pytest.raises(AttributeError):
        budget.limit = 10 ** 9


class _SlowCheck(fa.PartialDecryptionBudget):
    """Widens the window between the check and the increment of consume()."""

    def check(self, count: int = 1) -> None:
        super().check(count)
        time.sleep(0.002)


def _race(consume, threads: int) -> tuple:
    barrier = threading.Barrier(threads)
    served, refused = [], []

    def worker():
        barrier.wait()
        try:
            consume()
            served.append(1)
        except FESchemeError:
            refused.append(1)

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for th in pool:
        th.start()
    for th in pool:
        th.join()
    return len(served), len(refused)


def test_budget_is_atomic_across_threads():
    budget = _SlowCheck(10)
    assert _race(budget.consume, 32) == (10, 22)
    assert budget.used == 10 and budget.remaining == 0
    budget = _SlowCheck(10, used=7)
    assert _race(lambda: budget.consume(2), 16) == (1, 15)
    assert budget.used == 9


def test_servers_sharing_a_budget_across_threads_stay_within_it():
    kg, pp = _tmcfe_setup({**TMCFE, **KL, "max_partial_decryptions": 3})
    cts, cred, _ = _tmcfe_request(kg, pp)
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    budget = _SlowCheck(3)
    servers = [ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}, "partial_budget": budget})
               for _ in range(8)]
    calls = iter(servers)
    lock = threading.Lock()

    def one():
        with lock:
            srv = next(calls)
        srv.share_decrypt(cts, cred, dk, SIDS)

    assert _race(one, 8) == (3, 5)
    assert budget.used == 3


@pytest.mark.parametrize("limit, used", [(0, 0), (-1, 0), (5, -1), (2, -10 ** 6), (5, 6), (True, 0), (5, 1.0), (5.0, 0)])
def test_budget_values_are_validated(limit, used):
    with pytest.raises(FEValidationError):
        fa.PartialDecryptionBudget(limit, used=used)


# -- the default is untouched ---------------------------------------------------


def test_default_tmcfe_parameters_and_pp_are_unchanged():
    # the per-sample Gaussian flooding (the default flooding is uniform, see
    # tests/threshold/test_uniform_flooding.py); t < s: the replicated record
    kg = ThresholdMCFERingLWEKeyGenerator(dict(TMCFE, flooding="gaussian"))
    assert kg.sigma_f == _tmcfe_noise_bound(kg) << kg.stat_param
    budget = _tmcfe_noise_bound(kg) + kg.s * GAUSSIAN_TAIL_CUT * kg.sigma_f
    assert kg.q // kg.p >= 2 * budget + 4 - 1
    kg.setup()
    pp = kg.get_public_parameters()
    assert not KL_FIELDS & set(pp)
    with open("config/authority/{}/{}".format(kg._scheme_type, kg._param_file_name())) as f:
        record = json.load(f)
    assert record["threshold_construction"] == "bmms21-ibmcfe-replicated-smudging/v1"
    assert "flooding_accounting" not in record and "max_partial_decryptions" not in record
    uniform = ThresholdMCFERingLWEKeyGenerator(dict(TMCFE))
    uniform.setup()
    assert not KL_FIELDS & set(uniform.get_public_parameters())


def test_default_tmife_parameters_and_pp_are_unchanged():
    kg = ThresholdMIFELWEKeyGenerator(dict(TMIFE))
    assert kg._construction == "acfgu18-abdp15-replicated-smudging-seeded-a/v1"  # t < s: replicated
    assert kg.sigma_f == _tmife_noise_bound(kg) << kg.stat_param
    assert "flooding_accounting" not in kg._record_fields()
    kg.setup()
    assert not KL_FIELDS & set(kg.get_public_parameters())


# -- KL parameters and records ------------------------------------------------


def test_tmcfe_kl_parameters():
    kg = ThresholdMCFERingLWEKeyGenerator({**TMCFE, **KL})
    bound = _tmcfe_noise_bound(kg)
    samples = kg.s * KL["max_partial_decryptions"] * kg.ring_n
    assert kg.sigma_f == fa.kl_flooding_width(bound, samples, kg.stat_param)
    cut = fa.kl_tail_cut(samples, kg.stat_param, kg.sigma_f)
    assert kg.q // kg.p >= 2 * (bound + kg.s * cut * kg.sigma_f) + 3
    kg.setup()
    pp = kg.get_public_parameters()
    assert {k: pp[k] for k in KL_FIELDS} == {
        "flooding_accounting": "kl", "max_partial_decryptions": 1000, "stat_param": 64, "flood_tail_cut": cut,
    }
    default = ThresholdMCFERingLWEKeyGenerator(dict(TMCFE, flooding="gaussian"))  # per-sample Gaussian
    assert kg.flooding == "gaussian" and kg.sigma_f < default.sigma_f and kg.q < default.q
    with pytest.raises(FEValidationError, match="Gaussian flooding"):
        ThresholdMCFERingLWEKeyGenerator({**TMCFE, **KL, "flooding": "uniform"})
    with pytest.raises(FEValidationError, match="Gaussian flooding"):
        ThresholdMCFERingLWEKeyGenerator({**TMCFE, **KL, "sharing": "replicated"})
    with pytest.raises(FESchemeError, match="flood_tail_cut"):
        fa.kl_flooding(dict(pp, flood_tail_cut=cut + 1), pp["ring_n"])


def test_tmife_kl_parameters():
    kg = ThresholdMIFELWEKeyGenerator({**TMIFE, **KL})
    bound = _tmife_noise_bound(kg)
    samples = kg.s * KL["max_partial_decryptions"]
    assert kg.sigma_f == fa.kl_flooding_width(bound, samples, kg.stat_param)
    cut = fa.kl_tail_cut(samples, kg.stat_param, kg.sigma_f)
    assert kg.q // kg.p >= 2 * (bound + kg.s * cut * kg.sigma_f) + 3
    assert kg._construction == "acfgu18-abdp15-shamir-smudging-kl/v2"
    assert kg._record_fields()["max_partial_decryptions"] == 1000
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["flooding_accounting"] == "kl" and pp["flood_tail_cut"] == cut


@pytest.mark.parametrize("kg_cls, base", [
    (ThresholdMCFERingLWEKeyGenerator, {**TMCFE, "t": 3}), (ThresholdMIFELWEKeyGenerator, {**TMIFE, "t": 3}),
])
def test_records_of_the_accountings_never_verify_for_each_other(kg_cls, base):
    seq = [base, {**base, **KL}, {**base, **KL, "max_partial_decryptions": 2000}, base, {**base, **KL}]
    fresh = {}
    for cfg in seq:
        kg = kg_cls(dict(cfg))
        key = (cfg.get("flooding_accounting"), cfg.get("max_partial_decryptions"))
        fresh.setdefault(key, (kg.q, kg.sigma_f))
        # whatever the cache held before, the parameters are those of this config
        assert (kg.q, kg.sigma_f) == fresh[key]
    assert len({v for v in fresh.values()}) == 3


# -- round trips and the run-time limit -------------------------------------------


def _tmcfe_setup(cfg):
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    return kg, kg.get_public_parameters()


def _tmcfe_request(kg, pp, label="lbl"):
    x = {nid: [[random.randint(-4, 4) for _ in range(5)]] for nid in NIDS}
    y = {nid: [random.randint(-4, 4)] for nid in NIDS}
    cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
           for nid in NIDS}
    expected = [sum(x[n][0][j] * y[n][0] for n in NIDS) for j in range(5)]
    return cts, {"fusion_weight": y, "label": label}, expected


def _tmcfe_server(kg, pp, sid, cred, budgets):
    dk = kg.get_decryption_keys(sid, credentials=cred)
    extra = {"partial_budget": budgets[sid]} if budgets else {}
    return ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}, **extra}), dk


def test_tmcfe_kl_round_trip():
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    budgets = _budgets(pp)
    cts, cred, expected = _tmcfe_request(kg, pp)
    parts = {}
    for sid in SIDS:
        srv, dk = _tmcfe_server(kg, pp, sid, cred, budgets)
        parts[sid] = srv.share_decrypt(cts, cred, dk, SIDS)
    assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == expected
    assert all(budgets[sid].used == 1 for sid in SIDS)


def test_tmcfe_kl_servers_need_a_budget():
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    cts, cred, _ = _tmcfe_request(kg, pp)
    srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, None)
    with pytest.raises(FESchemeError, match="needs a partial_budget"):
        srv.share_decrypt(cts, cred, dk, SIDS)
    with pytest.raises(FEValidationError, match="more than max_partial_decryptions"):
        ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk},
                              "partial_budget": fa.PartialDecryptionBudget(1001)})
    with pytest.raises(FEValidationError, match="PartialDecryptionBudget"):
        ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}, "partial_budget": object()})


def test_tmcfe_budget_has_no_bypass():
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    cts, cred, _ = _tmcfe_request(kg, pp)
    srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, _budgets(pp))
    with pytest.raises(TypeError):
        srv.share_decrypt(cts, cred, dk, SIDS, _budgeted=True)


def test_tmcfe_kl_repeats_count_and_the_limit_holds_across_objects():
    kg, pp = _tmcfe_setup({**TMCFE, **KL, "max_partial_decryptions": 3})
    budgets = _budgets(pp)
    cts, cred, _ = _tmcfe_request(kg, pp)
    srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, budgets)
    first = srv.share_decrypt(cts, cred, dk, SIDS)
    assert srv.share_decrypt(cts, cred, dk, SIDS)["w"] == first["w"]  # derandomised, but counted
    # another object answering for the same server shares its budget
    other, _ = _tmcfe_server(kg, pp, "sid_0", cred, budgets)
    other.share_decrypt(cts, cred, dk, SIDS)
    assert budgets["sid_0"].used == 3
    for server in (srv, other):
        with pytest.raises(FESchemeError, match="partial decryption limit reached"):
            server.share_decrypt(cts, cred, dk, SIDS)


def test_tmcfe_refused_requests_consume_nothing():
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    budgets = _budgets(pp)
    cts, cred, _ = _tmcfe_request(kg, pp)
    srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, budgets)
    for bad in (["sid_0", "sid_1"], ["sid_0", "sid_0", "sid_1"], ["sid_1", "sid_2"]):
        with pytest.raises(FESchemeError):
            srv.share_decrypt(cts, cred, dk, bad)
    tampered = {nid: dict(ct) for nid, ct in cts.items()}
    tampered["nid_0"]["tags"] = {sid: "00" * 32 for sid in SIDS}
    with pytest.raises(FESchemeError):
        srv.share_decrypt(tampered, cred, dk, SIDS)
    assert budgets["sid_0"].used == 0


def _ndarray_request(kg, pp):
    rng = np.random.default_rng(3)
    grads = {nid: [rng.uniform(-0.4, 0.4, (3, 7)).round(1), rng.uniform(-0.4, 0.4, (40,)).round(1)] for nid in NIDS}
    enc = {nid: ThresholdMCFERingLWE({"id": nid, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           .encrypt_lst_ndarray(grads[nid], label="round") for nid in NIDS}
    sample = ThresholdMCFERingLWE({"id": "c", "precision": 1, "keys": {"pp": pp}})
    cred = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "round", "n_chunks": sample.key_chunks(enc[NIDS[0]])}
    return grads, enc, cred


def test_tmcfe_ndarray_counts_every_chunk():
    kg, pp = _tmcfe_setup({**TMCFE, **KL, "bound_x": 10, "max_partial_decryptions": 100})
    grads, enc, cred = _ndarray_request(kg, pp)
    chunks = sum(len(layer["chunks"]) for layer in enc[NIDS[0]])
    assert chunks > 2
    budgets = _budgets(pp)
    parts = {}
    for sid in SIDS:
        srv = ThresholdMCFERingLWE({"id": sid, "precision": 1, "keys": {"pp": pp}, "partial_budget": budgets[sid]})
        parts[sid] = srv.compute_lst_ndarray_ct(enc, cred, kg.get_decryption_keys(sid, credentials=cred), SIDS)
        assert budgets[sid].used == chunks
    out = ThresholdMCFERingLWE({"id": "c", "precision": 1, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(parts)
    for got, *layers in zip(out, *(grads[n] for n in NIDS)):
        assert np.allclose(got, sum(layers), atol=0.05)
    # refused whole when the budget does not cover every chunk
    short = fa.PartialDecryptionBudget(chunks - 1)
    srv = ThresholdMCFERingLWE({"id": "sid_0", "precision": 1, "keys": {"pp": pp}, "partial_budget": short})
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        srv.compute_lst_ndarray_ct(enc, cred, dk, SIDS)
    # and invalid requests consume nothing
    with pytest.raises(FESchemeError):
        srv.compute_lst_ndarray_ct(enc, cred, dk, ["sid_1", "sid_2"])
    assert short.used == 0


def test_tmcfe_default_servers_need_no_budget():
    kg, pp = _tmcfe_setup(dict(TMCFE))
    cts, cred, expected = _tmcfe_request(kg, pp)
    parts = {}
    for sid in SIDS[:2]:
        srv, dk = _tmcfe_server(kg, pp, sid, cred, None)
        assert srv._kl_flooding() == (None, None)
        parts[sid] = srv.share_decrypt(cts, cred, dk, SIDS[:2])
    assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == expected


def test_tmcfe_kl_servers_sample_at_the_kl_precision(monkeypatch):
    seen = []
    real = sampling_utils.discrete_gaussian_bigint

    def spy(sigma, count, rng=None, sampler=None, precision=None):
        seen.append((int(sigma), precision))
        return real(sigma, count, rng=rng, sampler=sampler, precision=precision)

    monkeypatch.setattr(fa, "discrete_gaussian_bigint", spy)
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    cts, cred, _ = _tmcfe_request(kg, pp)
    srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, _budgets(pp))
    srv.share_decrypt(cts, cred, dk, SIDS)
    sigma = int(pp["sigma_f"])
    assert seen and set(seen) == {(sigma, fa.kl_sampler_precision(sigma, 3 * 1000 * 16, 64))}


def test_tmcfe_kl_flooding_sampler_is_pinned():
    """The KL flooding is drawn by FLOODING_SAMPLER (the MPFR rounded
    Gaussian), whatever a gaussian_sampler() block selects: the same partial
    with and without a "facct" block, and no FACCT limit on N."""
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    cts, cred, expected = _tmcfe_request(kg, pp)
    srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, _budgets(pp))
    plain = srv.share_decrypt(cts, cred, dk, SIDS)
    with sampling_utils.gaussian_sampler("facct"):
        srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, _budgets(pp))
        assert srv.share_decrypt(cts, cred, dk, SIDS)["w"] == plain["w"]
        # N = 3 * 2^120 * 16 flooded samples: beyond FACCT's 2^-185, but the
        # rounded flooding sampler at kl_sampler_precision covers it
        kg, pp = _tmcfe_setup({**TMCFE, **KL, "max_partial_decryptions": 2 ** 120})
        budgets = _budgets(pp)
        cts, cred, expected = _tmcfe_request(kg, pp)
        parts = {}
        for sid in SIDS:
            srv, dk = _tmcfe_server(kg, pp, sid, cred, budgets)
            parts[sid] = srv.share_decrypt(cts, cred, dk, SIDS)
        assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == expected
        assert budgets["sid_0"].used == 1


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
def test_tmcfe_kl_partials_identical_on_both_backends():
    with native_backend.backend("python"):
        kg, pp = _tmcfe_setup({**TMCFE, **KL, "ring_n": 64})
        cts, cred, _ = _tmcfe_request(kg, pp)
        srv, dk = _tmcfe_server(kg, pp, "sid_0", cred, _budgets(pp))
        python = srv.share_decrypt(cts, cred, dk, SIDS)
    with native_backend.backend("native"):
        assert native_impl.ring_context(pp["q"], pp["ring_n"]) is not None
        srv, _ = _tmcfe_server(kg, pp, "sid_0", cred, _budgets(pp))
        native = srv.share_decrypt(cts, cred, dk, SIDS)
    assert native["w"] == python["w"] and native["c_y"] == python["c_y"]


def _tmife_setup(cfg):
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    encs = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    return kg, pp, encs


def test_tmife_kl_round_trip():
    kg, pp, encs = _tmife_setup({**TMIFE, **KL})
    budgets = _budgets(pp)
    x = {nid: [random.randint(-6, 6)] for nid in NIDS}
    y = {nid: [random.randint(-6, 6)] for nid in NIDS}
    cts = {nid: encs[nid].encrypt(x[nid]) for nid in NIDS}
    parts = {}
    for sid in SIDS:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        srv = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}, "partial_budget": budgets[sid]})
        parts[sid] = srv.share_decrypt(cts, y, dk, SIDS)
    assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == sum(x[n][0] * y[n][0] for n in NIDS)


def test_tmife_servers_refuse_beyond_the_limit_and_without_a_budget():
    kg, pp, encs = _tmife_setup({**TMIFE, **KL, "max_partial_decryptions": 2})
    y = {nid: [1] for nid in NIDS}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    with pytest.raises(FESchemeError, match="needs a partial_budget"):
        ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}}).share_decrypt(
            {nid: encs[nid].encrypt([1]) for nid in NIDS}, y, dk, SIDS)
    budget = fa.PartialDecryptionBudget(2)
    for i in range(2):
        cts = {nid: encs[nid].encrypt([i]) for nid in NIDS}
        ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}, "partial_budget": budget}).share_decrypt(
            cts, y, dk, SIDS)
    srv = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}, "partial_budget": budget})
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        srv.share_decrypt(cts, y, dk, SIDS)
    with pytest.raises(TypeError):
        srv.share_decrypt(cts, y, dk, SIDS, _budgeted=True)


def test_tmife_ndarray_counts_every_element():
    kg, pp, _ = _tmife_setup({**TMIFE, **KL, "bound_x": 50, "max_partial_decryptions": 11})
    x = {nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2))] for nid in NIDS}
    y = {nid: [1] for nid in NIDS}
    enc = {nid: ThresholdMIFELWE({"id": nid, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           .encrypt_lst_ndarray(x[nid]) for nid in NIDS}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    budget = fa.PartialDecryptionBudget(11)
    srv = ThresholdMIFELWE({"id": "sid_0", "precision": 1, "keys": {"pp": pp, "dk": dk}, "partial_budget": budget})
    for bad in (["sid_0"], ["sid_0", "sid_0", "sid_1"], ["sid_1", "sid_2"]):
        with pytest.raises(FESchemeError):
            srv.compute_lst_ndarray_ct(enc, y, dk, bad)
    assert budget.used == 0  # refused requests consume nothing
    srv.compute_lst_ndarray_ct(enc, y, dk, SIDS)
    assert budget.used == 10  # one partial per element
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        srv.compute_lst_ndarray_ct(enc, y, dk, SIDS)  # 10 more do not fit
    assert budget.used == 10


# -- the standard profile ---------------------------------------------------------

STANDARD_TMCFE = {"n": 3, "lst_nid": NIDS, "eta": 1, "bound_x": 100, "bound_y": 1, "s": 3, "t": 3,
                  "lst_sid": SIDS, "security_profile": "standard"}


def test_standard_tmcfe_kl_keeps_the_ring_and_shrinks_q():
    uniform = ThresholdMCFERingLWEKeyGenerator(dict(STANDARD_TMCFE))  # the default flooding
    assert int(uniform.q).bit_length() == 212
    default = ThresholdMCFERingLWEKeyGenerator(dict(STANDARD_TMCFE, flooding="gaussian"))
    q_per_server = 2 ** 53 // (3 * 8192)
    kg = ThresholdMCFERingLWEKeyGenerator(
        {**STANDARD_TMCFE, "flooding_accounting": "kl", "max_partial_decryptions": q_per_server}
    )
    assert kg.stat_param == 128 and kg.ring_n == default.ring_n == 8192
    q, width = kg._params_for(kg.ring_n)
    assert kg.q == q and int(q).bit_length() <= ring_lwe_bound(kg.ring_n, width)
    assert int(default.q).bit_length() == 215 and int(kg.q).bit_length() == 178
    samples = 3 * q_per_server * 8192
    assert fa.kl_divergence_log2(_tmcfe_noise_bound(kg), samples, kg.sigma_f) <= -129
    assert fa.kl_tail_cut(samples, 128, kg.sigma_f) == 16


def test_standard_kl_keeps_the_statistical_minimum():
    with pytest.raises(FEValidationError, match="stat_param >= 128"):
        ThresholdMCFERingLWEKeyGenerator(
            {**STANDARD_TMCFE, "stat_param": 100, "flooding_accounting": "kl", "max_partial_decryptions": 10}
        )


def test_standard_refuses_tmife_also_with_kl():
    """A maintainer decision (October 2026): tMIFE-LWE stays refused under
    standard with either accounting."""
    cfg = {"n": 2, "lst_nid": NIDS[:2], "eta": 1, "bound_x": 50, "bound_y": 50, "s": 3, "t": 3,
           "lst_sid": SIDS, "security_profile": "standard"}
    for extra in ({}, {"flooding_accounting": "kl", "max_partial_decryptions": 2 ** 40 // 3}):
        with pytest.raises(FEValidationError, match="no 'standard' parameter set for mife/lwe_threshold"):
            ThresholdMIFELWEKeyGenerator({**cfg, **extra})


def test_tmife_kl_dimension_at_the_reduction_width():
    """What "kl" takes at the standard sizes (stat_param 128, X = Y = 50, two
    clients, s = t = 3, N = 2^40), derived under toy: lwe_n 4864 at sigma'
    (the table's width), q of 173 bits; the per-sample accounting needs more
    than the table."""
    cfg = {"n": 2, "lst_nid": NIDS[:2], "eta": 1, "bound_x": 50, "bound_y": 50, "s": 3, "t": 3,
           "lst_sid": SIDS, "lwe_n": "auto", "stat_param": 128}
    kg = ThresholdMIFELWEKeyGenerator({**cfg, "flooding_accounting": "kl", "max_partial_decryptions": 2 ** 40 // 3})
    check_wide_lwe(kg.lwe_n, kg.q, kg._table_sigma({"sigma_q": kg.sigma_q}))
    assert (kg.lwe_n, int(kg.q).bit_length()) == (4864, 173)
    with pytest.raises(FEValidationError):
        ThresholdMIFELWEKeyGenerator(dict(cfg))


def test_malformed_kl_public_parameters_raise_library_errors():
    kg, pp = _tmcfe_setup({**TMCFE, **KL})
    flood = fa.kl_flooding(pp, pp["ring_n"])
    assert flood.max_partial_decryptions == KL["max_partial_decryptions"]
    for bad in ({k: v for k, v in pp.items() if k != "stat_param"}, dict(pp, sigma_f="x"),
                dict(pp, max_partial_decryptions=None), dict(pp, flood_tail_cut=0)):
        with pytest.raises((FEValidationError, FESchemeError)):
            fa.kl_flooding(bad, pp["ring_n"])


def test_a_budget_cannot_be_copied_or_pickled():
    import copy
    import pickle

    budget = fa.PartialDecryptionBudget(4)
    for duplicate in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            duplicate(budget)

