"""Security profiles: ``toy`` leaves configs alone, ``standard`` raises them."""

import warnings

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai import FESecurityWarning
from pyfe4ai.schemes import profiles
from pyfe4ai.schemes import status as st
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.mife.lwe import MIFELWE, MIFELWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator
from pyfe4ai.schemes.sife.lwe import SIFELWE, SIFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.sife.ddh import SIFE, SIFEKeyGenerator
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE, SIFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_utils import RFC3526_PRIMES, generate_group_primes
from pyfe4ai.utils.exceptions import FEValidationError
from pyfe4ai.utils.lwe_utils import (
    DENSE_LWE_128_LOG_Q,
    WIDE_LWE_128_LOG_RATIO,
    abmpsw20_lwe_parameters,
    check_dense_lwe,
    check_wide_lwe,
)
from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient
from pyfe4ai.utils.ring_lwe_utils import (
    RING_LWE_128_LOG_Q,
    bmms21_selective_parameters,
    check_ring_lwe,
    ring_lwe_bound,
    resolve_ring_n,
)

STANDARD = {"security_profile": "standard"}


@pytest.fixture(autouse=True)
def _param_cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# -- resolution --------------------------------------------------------------


def test_every_registered_scheme_has_a_kind():
    assert set(profiles.SCHEME_KIND) == set(st.SCHEME_STATUS)


@pytest.mark.parametrize("config", [{"eta": 3}, {"eta": 3, "security_profile": "toy"}])
def test_toy_returns_the_config_itself(config):
    assert profiles.resolve(config, ("sife", "ddh")) is config
    assert profiles.resolve(config, None) is config


def test_default_profile_keeps_toy_parameters():
    kg = SIFEKeyGenerator({"eta": 3})
    assert kg.security_profile == "toy"
    assert kg.sec_param == 128 and int(kg.p).bit_length() == 128


def test_unknown_profile_is_rejected():
    with pytest.raises(FEValidationError, match="unknown security_profile"):
        SIFEKeyGenerator({"eta": 3, "security_profile": "high"})


def test_standard_does_not_modify_the_callers_config():
    config = {"eta": 3, **STANDARD}
    profiles.resolve(config, ("sife", "ddh"))
    assert config == {"eta": 3, **STANDARD}


@pytest.mark.parametrize(
    "scheme, expected",
    [
        (("mcfe", "ddh"), {"sec_param": 3072, "modulus_length": 3072}),
        (("sife", "paillier"), {"bit_length": 1536}),
        (("mife", "ring_lwe"), {"sec_param": 128, "ring_n": "auto"}),
    ],
)
def test_standard_presets(scheme, expected):
    out = profiles.resolve(dict(STANDARD), scheme)
    assert {k: out[k] for k in expected} == expected


@pytest.mark.parametrize(
    "scheme, weak",
    [
        (("sife", "ddh"), {"sec_param": 2048}),
        (("mife", "damgard"), {"modulus_length": 1024}),
        (("mcfe", "paillier"), {"bit_length": 1024}),
        (("sife", "ring_lwe"), {"sec_param": 64}),
        (("mcfe", "ddh"), {"sec_param": "3072"}),
    ],
)
def test_standard_rejects_weaker_explicit_values(scheme, weak):
    with pytest.raises(FEValidationError, match="profile needs"):
        profiles.resolve({**STANDARD, **weak}, scheme)


@pytest.mark.parametrize("scheme", [
    ("mcfe", "lwe"), ("mcfe", "lwe_threshold"), ("mcfe", "ddh_decentralized"), ("sife", "paillier"),
    ("mcfe", "paillier"), ("mife", "lwe_threshold"), ("mcfe", "ring_lwe_threshold"),
])
def test_standard_statistical_parameter_is_128(scheme):
    """stat_param kept its default 64 (and accepted any explicit value) under
    standard: a 2^-64 distance per sample, against a 128-bit claim."""
    assert profiles.resolve(dict(STANDARD), scheme)["stat_param"] == 128
    assert profiles.resolve({**STANDARD, "stat_param": 160}, scheme)["stat_param"] == 160
    for weak in (64, 1):
        with pytest.raises(FEValidationError, match="stat_param"):
            profiles.resolve({**STANDARD, "stat_param": weak}, scheme)


def test_standard_accepts_stronger_explicit_values():
    out = profiles.resolve({**STANDARD, "bit_length": 2048}, ("mife", "paillier"))
    assert out["bit_length"] == 2048


@pytest.mark.parametrize(
    "scheme",
    [
        ("sife", "fullysec_lwe"),
        ("mife", "fullysec_lwe"),
        ("mcfe", "paillier_decentralized"),
    ],
)
def test_standard_refuses_schemes_without_a_feasible_parameter_set(scheme):
    with pytest.raises(FEValidationError, match="no 'standard' parameter set"):
        profiles.resolve(dict(STANDARD), scheme)


def test_standard_pairing_uses_bn254_and_warns():
    with pytest.warns(FESecurityWarning, match="100-bit"):
        out = profiles.resolve(dict(STANDARD), ("quadratic", "miqfe"))
    assert out["pairing_group_param"] == "BN254"
    with pytest.raises(FEValidationError, match="BN254"):
        profiles.resolve({**STANDARD, "pairing_group_param": "MNT224"}, ("sife", "fh_ipe_pairing"))


def test_unregistered_key_generator_has_no_standard_parameters():
    class Custom(IPFEAbsKeyGenerator):
        def setup(self):
            pass

        def get_public_parameters(self, **kwargs):
            return {}

        def get_private_keys(self, nid, **kwargs):
            return None

        def get_decryption_keys(self, sid, **kwargs):
            return {}

    assert Custom({}).security_profile == "toy"
    with pytest.raises(FEValidationError, match="no 'standard' parameters"):
        Custom(dict(STANDARD))


def test_subclass_inherits_the_scheme_of_its_base():
    class MySIFE(SIFEKeyGenerator):
        pass

    assert profiles.scheme_of(MySIFE) == ("sife", "ddh")
    kg = MySIFE(config={"eta": 2, **STANDARD})
    assert kg.p == RFC3526_PRIMES[3072]


# -- discrete log ------------------------------------------------------------


@pytest.mark.parametrize("bits, c", [(2048, 124476), (3072, 1690314)])
def test_rfc3526_primes(bits, c):
    gp.get_context().precision = bits + 64
    pi_part = int(gp.floor(gp.mpfr(2) ** (bits - 130) * gp.const_pi()))
    expected = 2 ** bits - 2 ** (bits - 64) - 1 + 2 ** 64 * (pi_part + c)
    p = RFC3526_PRIMES[bits]
    assert p == expected and gp.is_prime(p) and gp.is_prime((p - 1) // 2)


def test_standard_sife_ddh_round_trip():
    kg = SIFEKeyGenerator({"eta": 3, **STANDARD})
    kg.setup()
    assert kg.sec_param == 3072 and kg.p == RFC3526_PRIMES[3072]
    pp = kg.get_public_parameters()
    enc = SIFE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": [3, 2, -1]})
    assert SIFE({"keys": {"pp": pp}}).decrypt(enc.encrypt([1, 2, 3]), dk, [3, 2, -1]) == 4


def test_standard_sife_ring_lwe_round_trip():
    kg = SIFERingLWEKeyGenerator({"eta": 2, "bound_x": 50, "bound_y": 50, **STANDARD})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFERingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": [2, -3]})
    ct = enc.encrypt([[1, 2, 3], [4, 5, -6]])
    assert SIFERingLWE({"keys": {"pp": pp}}).decrypt(ct, dk, [2, -3]) == [-10, -11, 24]


@pytest.mark.parametrize("variant", ["ddh", "damgard", "ring_lwe", "lwe", "fullysec_lwe"])
def test_standard_mcfe_aggregation(variant):
    config = {
        "eta": 1, "n": 2, "lst_nid": ["nid_0", "nid_1"], "bound": 50, "bound_x": 50, "bound_y": 50,
        **STANDARD,
    }
    wrapper = FESchemeWrapper("mcfe", variant, config, precision=1)
    enc = {
        nid: encrypt_gradient(np.array(g), wrapper, nid=nid, label="round-1")
        for nid, g in (("nid_0", [2.0, 4.0]), ("nid_1", [1.0, 3.0]))
    }
    result = aggregate_gradients(enc, wrapper, {"nid_0": [1], "nid_1": [1]}, label="round-1")
    np.testing.assert_allclose(result, [3.0, 7.0], atol=0.2)


# -- dense LWE (ABMPSW20 MCFE) ----------------------------------------------

MCFE_LWE = [("mcfe", v) for v in ("lwe", "fullysec_lwe", "lwe_threshold", "lwe_decentralized")]
MCFE_LWE_CONFIG = {"n": 2, "eta": 1, "bound_x": 50, "bound_y": 50}


@pytest.mark.parametrize("scheme", MCFE_LWE)
def test_standard_selects_the_lwe_dimension_of_mcfe_lwe(scheme):
    assert profiles.resolve(dict(STANDARD), scheme)["lwe_n"] == "auto"


def test_dense_lwe_table_grows_with_the_dimension():
    dims = sorted(DENSE_LWE_128_LOG_Q)
    assert all(DENSE_LWE_128_LOG_Q[a] < DENSE_LWE_128_LOG_Q[b] for a, b in zip(dims, dims[1:]))


def test_check_dense_lwe_uses_the_largest_tabulated_dimension_below():
    n0 = sorted(DENSE_LWE_128_LOG_Q)[1]
    bound = DENSE_LWE_128_LOG_Q[n0]
    check_dense_lwe(n0 + 1, 2 ** bound - 1)
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        check_dense_lwe(n0 + 1, 2 ** bound)
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        check_dense_lwe(min(DENSE_LWE_128_LOG_Q) - 1, 3)


def test_standard_mcfe_lwe_picks_the_smallest_estimated_dimension():
    kg = MCFELWEKeyGenerator({**MCFE_LWE_CONFIG, **STANDARD})
    assert kg.lwe_n in DENSE_LWE_128_LOG_Q
    assert int(kg.q).bit_length() <= DENSE_LWE_128_LOG_Q[kg.lwe_n]
    for n0 in (d for d in DENSE_LWE_128_LOG_Q if d < kg.lwe_n):
        q = abmpsw20_lwe_parameters(2, 50, 50, n0, kg.stat_param)["q"]
        assert q.bit_length() > DENSE_LWE_128_LOG_Q[n0]


def test_toy_mcfe_lwe_accepts_an_automatic_dimension():
    kg = MCFELWEKeyGenerator({**MCFE_LWE_CONFIG, "lwe_n": "auto"})
    assert kg.lwe_n in DENSE_LWE_128_LOG_Q


@pytest.mark.parametrize("lwe_n", [32, 2048])
def test_standard_rejects_an_insecure_explicit_lwe_dimension(lwe_n):
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        MCFELWEKeyGenerator({**MCFE_LWE_CONFIG, "lwe_n": lwe_n, **STANDARD})


def test_standard_accepts_a_larger_explicit_lwe_dimension():
    lwe_n = max(DENSE_LWE_128_LOG_Q) + 100
    assert MCFELWEKeyGenerator({**MCFE_LWE_CONFIG, "lwe_n": lwe_n, **STANDARD}).lwe_n == lwe_n


def test_standard_threshold_mcfe_lwe_uses_an_estimated_dimension():
    kg = ThresholdMCFELWEKeyGenerator({**MCFE_LWE_CONFIG, "s": 3, "t": 2, **STANDARD})
    assert int(kg.q).bit_length() <= DENSE_LWE_128_LOG_Q[kg.lwe_n]


def test_standard_dmcfe_lwe_combines_key_shares_under_a_3072_bit_modulus():
    kg = DecentralizedMCFELWEKeyGenerator({**MCFE_LWE_CONFIG, **STANDARD})
    assert kg.n2_prime_bits == profiles.STANDARD_DCR_PRIME_BITS
    assert int(kg.n2_mod).bit_length() >= 2 * profiles.STANDARD_DCR_PRIME_BITS - 1


def test_toy_dmcfe_lwe_sizes_its_modulus_by_the_key_shares():
    kg = DecentralizedMCFELWEKeyGenerator({**MCFE_LWE_CONFIG, "lwe_n": 16})
    assert kg.n2_prime_bits < profiles.STANDARD_DCR_PRIME_BITS


# -- LWE with a seeded public matrix (ABDP15) -------------------------------


@pytest.mark.parametrize("scheme", [("sife", "lwe"), ("mife", "lwe")])
def test_standard_selects_the_lwe_dimension_of_abdp15(scheme):
    assert profiles.resolve(dict(STANDARD), scheme)["lwe_n"] == "auto"


@pytest.mark.parametrize("kg_cls, config", [
    (SIFELWEKeyGenerator, {"eta": 2, "bound_x": 50, "bound_y": 50}),
    (MIFELWEKeyGenerator, {"n": 2, "eta": 1, "bound_x": 50, "bound_y": 50}),
])
def test_standard_abdp15_meets_the_wide_lwe_table(kg_cls, config):
    kg = kg_cls({**config, **STANDARD})
    assert kg.lwe_n in WIDE_LWE_128_LOG_RATIO[16]
    check_wide_lwe(kg.lwe_n, kg.q, kg.sigma_q)


def test_standard_rejects_an_insecure_explicit_abdp15_dimension():
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        SIFELWEKeyGenerator({"eta": 2, "bound_x": 50, "bound_y": 50, "lwe_n": 256, **STANDARD})


def test_standard_sife_lwe_round_trip_streaming_the_matrix():
    config = {"eta": 2, "bound_x": 50, "bound_y": 50, "matrix_mode": "stream", **STANDARD}
    kg = SIFELWEKeyGenerator(config)
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFELWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}, "matrix_mode": "stream"})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": [3, -7]})
    ct = enc.encrypt([50, -50])
    assert SIFELWE({"keys": {"pp": pp}}).decrypt(ct, dk, [3, -7]) == 500


def test_wrapper_passes_the_matrix_mode_to_its_encryptors():
    wrapper = FESchemeWrapper("sife", "lwe", {"eta": 2, "lwe_n": 16, "matrix_mode": "stream"})
    assert wrapper._get_encryptor("nid_0").matrix_mode == "stream"


def test_wrapper_uses_the_matrix_mode_of_its_key_generator():
    # the mode a profile fixed for the key generator, not only an explicit config entry
    wrapper = FESchemeWrapper("sife", "lwe", {"eta": 2, "lwe_n": 16})
    wrapper.kg.matrix_mode = "stream"
    assert wrapper._get_encryptor("nid_0").matrix_mode == "stream"


TMIFE_CONFIG = {"n": 2, "eta": 1, "s": 3, "t": 2, "bound_x": 50, "bound_y": 50}


def test_standard_threshold_mife_lwe_streams_its_matrix():
    out = profiles.resolve(dict(STANDARD), ("mife", "lwe_threshold"))
    assert out["lwe_n"] == "auto" and out["matrix_mode"] == "stream"
    with pytest.raises(FEValidationError, match="stream"):
        profiles.resolve({**STANDARD, "matrix_mode": "memory"}, ("mife", "lwe_threshold"))


def test_standard_threshold_mife_lwe_meets_the_table_at_its_enlarged_q():
    kg = ThresholdMIFELWEKeyGenerator({**TMIFE_CONFIG, **STANDARD})
    check_wide_lwe(kg.lwe_n, kg.q, kg.sigma_q)
    assert kg.matrix_mode == "stream"
    assert kg.m >= (kg.lwe_n + kg.eta + 1) * int(kg.q).bit_length() + 2 * kg.lwe_n + 1


# -- Ring-LWE ----------------------------------------------------------------


def test_standard_ring_lwe_picks_the_smallest_estimated_dimension():
    kg = SIFERingLWEKeyGenerator({"eta": 4, "bound_x": 10, "bound_y": 10, **STANDARD})
    assert kg.ring_n == 4096
    width = bmms21_selective_parameters(128, 4, 10, 10, 4096)["rlwe_width"]
    assert int(kg.q).bit_length() <= ring_lwe_bound(4096, width)
    # the dimension below would need a larger q than the bound allows
    P = bmms21_selective_parameters(128, 4, 10, 10, 2048)
    assert int(P["q"]).bit_length() > ring_lwe_bound(2048, P["rlwe_width"])


@pytest.mark.parametrize("ring_n", [32, 2048])
def test_standard_rejects_an_insecure_explicit_ring_dimension(ring_n):
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        SIFERingLWEKeyGenerator({"eta": 4, "bound_x": 10, "bound_y": 10, "ring_n": ring_n, **STANDARD})


def test_threshold_ring_lwe_dimension_accounts_for_flooding():
    kg = ThresholdMCFERingLWEKeyGenerator(
        {"n": 3, "eta": 2, "bound_x": 10, "bound_y": 10, "s": 3, "t": 2, **STANDARD}
    )
    q, width = kg._params_for(kg.ring_n)
    assert kg.q == q and int(q).bit_length() <= ring_lwe_bound(kg.ring_n, width)


def test_ring_n_helpers():
    assert resolve_ring_n(None, None) == 32
    assert resolve_ring_n(64, None) == 64
    assert resolve_ring_n("auto", lambda n: (2 ** 100, 8.0)) == 4096
    with pytest.raises(FEValidationError, match="no ring dimension"):
        resolve_ring_n("auto", lambda n: (2 ** 1000, 8.0))
    bound = RING_LWE_128_LOG_Q[8.0][1024]
    check_ring_lwe(1024, 2 ** bound - 1, 8.0)
    with pytest.raises(FEValidationError):
        check_ring_lwe(1024, 2 ** bound, 8.0)
    with pytest.raises(FEValidationError):
        check_ring_lwe(512, 3, 8.0)


def test_toy_ring_lwe_keeps_the_default_dimension():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kg = SIFERingLWEKeyGenerator({"eta": 4})
    assert kg.ring_n == 32


@pytest.mark.parametrize("bits, runs", [(12, 2000), (16, 2000), (63, 20), (64, 20), (65, 20), (256, 5)])
def test_safe_primes_have_the_requested_size(bits, runs):
    # next_prime may step past 2^bits for small sizes (about 1% of 16-bit draws)
    for _ in range(runs):
        p, q = generate_group_primes(bits)
        assert p == 2 * q + 1 and int(p).bit_length() == bits
        assert gp.is_prime(p, 40) and gp.is_prime(q, 40)


def test_a_read_only_mapping_config_gets_the_profile_too():
    """The metaclass only resolved dict configs: a Mapping such as
    MappingProxyType kept a 128-bit group while recording "standard"."""
    from types import MappingProxyType

    kg = SIFEKeyGenerator(MappingProxyType({"eta": 2, **STANDARD}))
    assert kg.sec_param == 3072 and int(kg.p).bit_length() == 3072


def test_a_config_that_is_not_a_mapping_is_refused():
    """An object with .get but no Mapping interface skipped the profile while
    the key generator recorded "standard"."""
    class Config:
        def __init__(self, d):
            self.d = d

        def get(self, key, default=None):
            return self.d.get(key, default)

    with pytest.raises(FEValidationError, match="mapping"):
        SIFEKeyGenerator(Config({"eta": 2, **STANDARD}))
