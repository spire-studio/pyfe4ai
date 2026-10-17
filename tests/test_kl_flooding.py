"""KL accounting of the threshold flooding (``flooding_accounting = "kl"``):
parameters, records, run-time limit, sampler, and that the default is untouched."""

import math
import random

import numpy as np
import pytest

from pyfe4ai.schemes import flooding_accounting as fa
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import sampling_utils
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.lwe_utils import check_wide_lwe
from pyfe4ai.utils.ring_lwe_utils import ring_lwe_bound
from pyfe4ai.utils.sampling_utils import XofRandom

NIDS = ["nid_0", "nid_1", "nid_2"]
SIDS = ["sid_0", "sid_1", "sid_2"]
TMCFE = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS,
         "ring_n": 16, "bound_x": 4, "bound_y": 4}
TMIFE = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS,
         "lwe_n": 16, "bound_x": 6, "bound_y": 6}
KL = {"flooding_accounting": "kl", "max_partial_decryptions": 1000}
KL_FIELDS = {"flooding_accounting", "max_partial_decryptions", "stat_param", "flood_tail_cut"}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _tmcfe_noise_bound(kg) -> int:
    C = GAUSSIAN_TAIL_CUT
    return kg.n * kg.eta * int(kg.bound_y) * int(2 * kg.ring_n * C ** 2 * kg.sigma1 * kg.sigma2 + C * kg.sigma3 + 1)


def _tmife_noise_bound(kg) -> int:
    return kg.n * kg.eta * int(kg.bound_y) * kg.m * int(GAUSSIAN_TAIL_CUT * kg.sigma_q + 1)


# -- configuration -------------------------------------------------------------


def test_default_accounting_is_per_sample():
    assert fa.flooding_config({}) == ("per-sample", None)
    assert fa.flooding_config({"flooding_accounting": "per-sample"}) == ("per-sample", None)
    assert fa.flooding_config(KL) == ("kl", 1000)


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
        fa.flooding_config(config)
    for kg_cls, base in ((ThresholdMCFERingLWEKeyGenerator, TMCFE), (ThresholdMIFELWEKeyGenerator, TMIFE)):
        with pytest.raises(FEValidationError, match=match):
            kg_cls({**base, **config})


def test_dense_lwe_threshold_refuses_the_option():
    cfg = {"eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": NIDS[:2], "lst_sid": SIDS, "bound_x": 4, "bound_y": 4}
    for extra in (KL, {"max_partial_decryptions": 10}):
        with pytest.raises(FEValidationError, match="no statistical flooding"):
            ThresholdMCFELWEKeyGenerator({**cfg, **extra})


# -- the width ------------------------------------------------------------------


@pytest.mark.parametrize("bound, samples, stat_param", [
    (1, 1, 1), (3, 7, 64), (2 ** 71 + 12345, 2 ** 53, 128), (10 ** 20, 3 * 8192 * 1000, 128), (5, 2 ** 40, 80),
])
def test_kl_width_is_the_smallest_meeting_the_bound(bound, samples, stat_param):
    sigma = fa.kl_flooding_width(bound, samples, stat_param)
    # samples * bound^2 / (2 sigma^2) <= 2^-stat_param, exactly in integers
    assert (samples * bound * bound) << stat_param <= 2 * sigma * sigma
    assert (samples * bound * bound) << stat_param > 2 * (sigma - 1) * (sigma - 1)
    assert fa.kl_divergence_log2(bound, samples, sigma) <= -stat_param + 1e-9


def test_kl_width_grows_with_the_square_root_of_the_run():
    bound = 2 ** 70
    one = fa.kl_flooding_width(bound, 2 ** 30, 128)
    assert math.log2(fa.kl_flooding_width(bound, 2 ** 50, 128) / one) == pytest.approx(10, abs=1e-6)
    assert math.log2(fa.kl_flooding_width(bound, 2 ** 30, 130) / one) == pytest.approx(1, abs=1e-6)


def test_sampler_precision():
    sigma = 2 ** 161 + 5
    assert fa.kl_sampler_precision(sigma, 2 ** 53, 128) == 162 + 128 + 53 + 8
    assert fa.kl_sampler_precision(sigma, 2 ** 53 + 1, 128) == 162 + 128 + 54 + 8
    # never below the sampler's own default bits(sigma) + 128
    assert fa.kl_sampler_precision(sigma, 10, 64) == 162 + 128


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
    assert set(rng.widths) == {229}  # bits(sigma) + 128
    rng = _Recording(b"seed")
    higher = sampling_utils.discrete_gaussian_bigint(2 ** 100, 8, rng=rng, sampler="rounded", precision=300)
    assert set(rng.widths) == {300} and len(rng.widths) == 16
    # XofRandom draws each width from its own block: the extra bits only refine u1, u2
    assert higher == default
    with pytest.raises(ValueError, match="at least"):
        sampling_utils.discrete_gaussian_bigint(2 ** 100, 8, rng=rng, sampler="rounded", precision=228)


def test_kl_servers_sample_at_the_kl_precision(monkeypatch):
    seen = []
    real = sampling_utils.discrete_gaussian_bigint

    def spy(sigma, count, rng=None, sampler=None, precision=None):
        seen.append((int(sigma), precision))
        return real(sigma, count, rng=rng, sampler=sampler, precision=precision)

    monkeypatch.setattr(fa, "discrete_gaussian_bigint", spy)
    _, pp, _, _, parts, expected, _ = _tmcfe_round({**TMCFE, **KL}, ["sid_0", "sid_1"])
    sigma = int(pp["sigma_f"])
    want = fa.kl_sampler_precision(sigma, 3 * 1000 * 16, 64)
    assert seen and set(seen) == {(sigma, want)}


def test_flooding_samples_respect_the_cut():
    values = fa.flooding_samples(1000, 4000, XofRandom(b"cut"), tail_cut=1)
    assert len(values) == 4000 and max(abs(v) for v in values) <= 1000
    assert any(abs(v) > 1000 for v in sampling_utils.discrete_gaussian_bigint(1000, 4000, rng=XofRandom(b"cut")))


def test_budget():
    budget = fa.PartialDecryptionBudget(5)
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
    with pytest.raises(ValueError):
        budget.consume(-1)


# -- the default is untouched ---------------------------------------------------


def test_default_tmcfe_parameters_and_pp_are_unchanged():
    kg = ThresholdMCFERingLWEKeyGenerator(dict(TMCFE))
    assert kg.sigma_f == _tmcfe_noise_bound(kg) << kg.stat_param
    budget = _tmcfe_noise_bound(kg) + kg.s * GAUSSIAN_TAIL_CUT * kg.sigma_f
    assert kg.q // kg.p >= 2 * budget + 4 - 1
    kg.setup()
    pp = kg.get_public_parameters()
    assert not KL_FIELDS & set(pp)
    import json
    with open("config/authority/{}/param.json".format(kg._scheme_type)) as f:
        record = json.load(f)
    assert record["threshold_construction"] == "bmms21-ibmcfe-shamir-smudging/v1"
    assert "flooding_accounting" not in record and "max_partial_decryptions" not in record


def test_default_tmife_parameters_and_pp_are_unchanged():
    kg = ThresholdMIFELWEKeyGenerator(dict(TMIFE))
    assert kg._construction == "acfgu18-abdp15-shamir-smudging-seeded-a/v3"
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
    assert kg.q // kg.p >= 2 * (bound + kg.s * fa.KL_TAIL_CUT * kg.sigma_f) + 3
    kg.setup()
    pp = kg.get_public_parameters()
    assert {k: pp[k] for k in KL_FIELDS} == {
        "flooding_accounting": "kl", "max_partial_decryptions": 1000, "stat_param": 64, "flood_tail_cut": 20,
    }
    default = ThresholdMCFERingLWEKeyGenerator(dict(TMCFE))
    assert kg.sigma_f < default.sigma_f and kg.q < default.q


def test_tmife_kl_parameters():
    kg = ThresholdMIFELWEKeyGenerator({**TMIFE, **KL})
    bound = _tmife_noise_bound(kg)
    assert kg.sigma_f == fa.kl_flooding_width(bound, kg.s * KL["max_partial_decryptions"], kg.stat_param)
    assert kg.q // kg.p >= 2 * (bound + kg.s * fa.KL_TAIL_CUT * kg.sigma_f) + 3
    assert kg._construction == "acfgu18-abdp15-shamir-smudging-kl/v1"
    assert kg._record_fields()["max_partial_decryptions"] == 1000
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["flooding_accounting"] == "kl" and pp["flood_tail_cut"] == 20


@pytest.mark.parametrize("kg_cls, base", [
    (ThresholdMCFERingLWEKeyGenerator, TMCFE), (ThresholdMIFELWEKeyGenerator, TMIFE),
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


def _tmcfe_round(cfg, enrolled, label="lbl", servers=None, budget=None):
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    x = {nid: [[random.randint(-4, 4) for _ in range(5)]] for nid in NIDS}
    y = {nid: [random.randint(-4, 4)] for nid in NIDS}
    cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
           for nid in NIDS}
    cred = {"fusion_weight": y, "label": label}
    servers = servers or {}
    parts = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=cred)
        srv = servers.setdefault(sid, ThresholdMCFERingLWE(
            {"id": sid, "keys": {"pp": pp, "dk": dk}, **({"partial_budget": budget} if budget else {})}
        ))
        parts[sid] = srv.share_decrypt(cts, cred, dk, enrolled)
    expected = [sum(x[n][0][j] * y[n][0] for n in NIDS) for j in range(5)]
    return kg, pp, cts, cred, parts, expected, servers


@pytest.mark.parametrize("enrolled", [["sid_0", "sid_1"], ["sid_1", "sid_2"], ["sid_0", "sid_1", "sid_2"]])
def test_tmcfe_kl_round_trip(enrolled):
    _, pp, _, _, parts, expected, _ = _tmcfe_round({**TMCFE, **KL}, enrolled)
    assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == expected


def test_tmcfe_kl_noise_is_derandomised_and_repeats_count():
    kg, pp, cts, cred, parts, _, servers = _tmcfe_round({**TMCFE, **KL}, ["sid_0", "sid_1"])
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    again = servers["sid_0"].share_decrypt(cts, cred, dk, ["sid_0", "sid_1"])
    assert again["w"] == parts["sid_0"]["w"]
    assert servers["sid_0"]._partial_budget.used == 2  # repeated requests count too


def test_tmcfe_servers_refuse_beyond_the_limit():
    cfg = {**TMCFE, "flooding_accounting": "kl", "max_partial_decryptions": 3}
    kg, pp, cts, cred, _, _, servers = _tmcfe_round(cfg, ["sid_0", "sid_1"])
    srv = servers["sid_0"]
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    srv.share_decrypt(cts, cred, dk, ["sid_0", "sid_2"])
    srv.share_decrypt(cts, cred, dk, ["sid_0", "sid_1", "sid_2"])
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        srv.share_decrypt(cts, cred, dk, ["sid_0", "sid_1"])
    # a fresh object of the same server must be given the shared budget
    shared = fa.PartialDecryptionBudget(3, used=srv._partial_budget.used)
    other = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}, "partial_budget": shared})
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        other.share_decrypt(cts, cred, dk, ["sid_0", "sid_1"])


def test_tmcfe_invalid_requests_consume_nothing():
    cfg = {**TMCFE, "flooding_accounting": "kl", "max_partial_decryptions": 3}
    kg, pp, cts, cred, _, _, servers = _tmcfe_round(cfg, ["sid_0", "sid_1"])
    srv = servers["sid_0"]
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    with pytest.raises(FESchemeError):
        srv.share_decrypt(cts, cred, dk, ["sid_0"])  # below the threshold
    assert srv._partial_budget.used == 1


def _ndarray_round(cfg, budget_limit=None):
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    rng = np.random.default_rng(3)
    grads = {nid: [rng.uniform(-0.4, 0.4, (3, 7)).round(1), rng.uniform(-0.4, 0.4, (40,)).round(1)] for nid in NIDS}
    enc = {nid: ThresholdMCFERingLWE({"id": nid, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           .encrypt_lst_ndarray(grads[nid], label="round") for nid in NIDS}
    sample = ThresholdMCFERingLWE({"id": "c", "precision": 1, "keys": {"pp": pp}})
    cred = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "round", "n_chunks": sample.key_chunks(enc[NIDS[0]])}
    return kg, pp, grads, enc, cred


def test_tmcfe_ndarray_counts_every_chunk_up_front():
    cfg = {**TMCFE, "bound_x": 10, "flooding_accounting": "kl", "max_partial_decryptions": 100}
    kg, pp, grads, enc, cred = _ndarray_round(cfg)
    chunks = sum(len(layer["chunks"]) for layer in enc[NIDS[0]])
    assert chunks > 2
    enrolled = ["sid_0", "sid_2"]
    parts = {}
    for sid in enrolled:
        srv = ThresholdMCFERingLWE({"id": sid, "precision": 1, "keys": {"pp": pp}})
        parts[sid] = srv.compute_lst_ndarray_ct(enc, cred, kg.get_decryption_keys(sid, credentials=cred), enrolled)
        assert srv._partial_budget.used == chunks
    out = ThresholdMCFERingLWE({"id": "c", "precision": 1, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(parts)
    for got, *layers in zip(out, *(grads[n] for n in NIDS)):
        assert np.allclose(got, sum(layers), atol=0.05)
    short = fa.PartialDecryptionBudget(chunks - 1)
    srv = ThresholdMCFERingLWE({"id": "sid_0", "precision": 1, "keys": {"pp": pp}, "partial_budget": short})
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        srv.compute_lst_ndarray_ct(enc, cred, kg.get_decryption_keys("sid_0", credentials=cred), enrolled)
    assert short.used == 0


def test_tmcfe_default_servers_have_no_limit():
    _, pp, cts, cred, parts, expected, servers = _tmcfe_round(dict(TMCFE), ["sid_0", "sid_1"])
    assert servers["sid_0"]._kl_flooding() == (None, None)


def test_tmcfe_kl_with_facct_sampler():
    with sampling_utils.gaussian_sampler("facct"):
        _, pp, _, _, parts, expected, _ = _tmcfe_round({**TMCFE, **KL}, ["sid_0", "sid_1"])
        assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == expected
        # N = 3 * 2^120 * 16 flooded samples exceed what FACCT's 2^-185 covers at stat_param 64
        cfg = {**TMCFE, "flooding_accounting": "kl", "max_partial_decryptions": 2 ** 120}
        with pytest.raises(FEValidationError, match="FACCT"):
            _tmcfe_round(cfg, ["sid_0", "sid_1"])


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
def test_tmcfe_kl_partials_identical_on_both_backends():
    with native_backend.backend("python"):
        kg, pp, cts, cred, parts, expected, _ = _tmcfe_round({**TMCFE, "ring_n": 64, **KL}, ["sid_0", "sid_2"])
    dk = kg.get_decryption_keys("sid_0", credentials=cred)
    with native_backend.backend("native"):
        assert native_backend.ring_context(pp["q"], pp["ring_n"]) is not None
        native = ThresholdMCFERingLWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}}).share_decrypt(
            cts, cred, dk, ["sid_0", "sid_2"])
    assert native["w"] == parts["sid_0"]["w"] and native["c_y"] == parts["sid_0"]["c_y"]


def _tmife_round(cfg, enrolled):
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    x = {nid: [random.randint(-6, 6)] for nid in NIDS}
    y = {nid: [random.randint(-6, 6)] for nid in NIDS}
    cts = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
           for nid in NIDS}
    servers, parts = {}, {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        servers[sid] = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        parts[sid] = servers[sid].share_decrypt(cts, y, dk, enrolled)
    return kg, pp, cts, y, parts, sum(x[n][0] * y[n][0] for n in NIDS), servers


@pytest.mark.parametrize("enrolled", [["sid_0", "sid_1"], ["sid_0", "sid_2"], ["sid_0", "sid_1", "sid_2"]])
def test_tmife_kl_round_trip(enrolled):
    _, pp, _, _, parts, expected, _ = _tmife_round({**TMIFE, **KL}, enrolled)
    assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts) == expected


def test_tmife_servers_refuse_beyond_the_limit():
    cfg = {**TMIFE, "flooding_accounting": "kl", "max_partial_decryptions": 2}
    kg, pp, cts, y, parts, _, servers = _tmife_round(cfg, ["sid_0", "sid_1"])
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    again = servers["sid_0"].share_decrypt(cts, y, dk, ["sid_0", "sid_1"])
    assert again["w"] == parts["sid_0"]["w"]
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        servers["sid_0"].share_decrypt(cts, y, dk, ["sid_0", "sid_1"])


def test_tmife_ndarray_counts_every_element():
    cfg = {**TMIFE, "bound_x": 50, "flooding_accounting": "kl", "max_partial_decryptions": 11}
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    x = {nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2))] for nid in NIDS}
    y = {nid: [1] for nid in NIDS}
    enc = {nid: ThresholdMIFELWE({"id": nid, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           .encrypt_lst_ndarray(x[nid]) for nid in NIDS}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    srv = ThresholdMIFELWE({"id": "sid_0", "precision": 1, "keys": {"pp": pp, "dk": dk}})
    srv.compute_lst_ndarray_ct(enc, y, dk, ["sid_0", "sid_1"])
    assert srv._partial_budget.used == 10  # one partial per element
    with pytest.raises(FESchemeError, match="partial decryption limit reached"):
        srv.compute_lst_ndarray_ct(enc, y, dk, ["sid_0", "sid_1"])  # 10 more do not fit
    assert srv._partial_budget.used == 10


# -- the standard profile ---------------------------------------------------------

STANDARD_TMCFE = {"n": 3, "lst_nid": NIDS, "eta": 1, "bound_x": 100, "bound_y": 1, "s": 3, "t": 2,
                  "lst_sid": SIDS, "security_profile": "standard"}


def test_standard_tmcfe_kl_keeps_the_ring_and_shrinks_q():
    default = ThresholdMCFERingLWEKeyGenerator(dict(STANDARD_TMCFE))
    q_per_server = 2 ** 53 // (3 * 8192)
    kg = ThresholdMCFERingLWEKeyGenerator(
        {**STANDARD_TMCFE, "flooding_accounting": "kl", "max_partial_decryptions": q_per_server}
    )
    assert kg.stat_param == 128 and kg.ring_n == default.ring_n == 8192
    q, width = kg._params_for(kg.ring_n)
    assert kg.q == q and int(q).bit_length() <= ring_lwe_bound(kg.ring_n, width)
    assert int(default.q).bit_length() == 215 and int(kg.q).bit_length() == 178
    samples = 3 * q_per_server * 8192
    assert fa.kl_divergence_log2(_tmcfe_noise_bound(kg), samples, kg.sigma_f) <= -128


def test_standard_kl_keeps_the_statistical_minimum():
    with pytest.raises(FEValidationError, match="stat_param >= 128"):
        ThresholdMCFERingLWEKeyGenerator(
            {**STANDARD_TMCFE, "stat_param": 100, "flooding_accounting": "kl", "max_partial_decryptions": 10}
        )


def test_standard_tmife_kl_dimension():
    cfg = {"n": 2, "lst_nid": NIDS[:2], "eta": 1, "bound_x": 50, "bound_y": 50, "s": 3, "t": 2,
           "lst_sid": SIDS, "security_profile": "standard"}
    default = ThresholdMIFELWEKeyGenerator(dict(cfg))
    kg = ThresholdMIFELWEKeyGenerator({**cfg, "flooding_accounting": "kl", "max_partial_decryptions": 2 ** 40 // 3})
    check_wide_lwe(kg.lwe_n, kg.q, kg.sigma_q)
    assert kg.matrix_mode == "stream"
    assert (default.lwe_n, int(default.q).bit_length()) == (5952, 217)
    assert (kg.lwe_n, int(kg.q).bit_length()) == (4352, 172)
    assert kg.m * kg.lwe_n < 0.45 * default.m * default.lwe_n
