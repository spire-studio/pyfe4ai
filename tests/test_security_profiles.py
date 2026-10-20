"""Security profiles: ``toy`` leaves configs alone, ``standard`` raises them."""

import json
import os
import warnings

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai import FESecurityWarning
from pyfe4ai.core import config as configs
from pyfe4ai.core import profiles
from pyfe4ai.core import status as st
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.schemes.mife.lwe import MIFELWE, MIFELWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator
from pyfe4ai.schemes.sife.lwe import SIFELWE, SIFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.sife.ddh import SIFEDDH, SIFEDDHKeyGenerator
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE, SIFERingLWEKeyGenerator
from pyfe4ai.primitives.groups.integer_groups import RFC3526_PRIMES, generate_group_primes
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.primitives.lattice.lwe_utils import (
    DENSE_LWE_128_LOG_Q,
    WIDE_LWE_128_LOG_RATIO,
    abdp15_parameters,
    abmpsw20_lwe_parameters,
    check_dense_lwe,
    check_wide_lwe,
    wide_lwe_n,
)
from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient
from pyfe4ai.primitives.lattice.ring_lwe_utils import (
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
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


# -- resolution --------------------------------------------------------------


def test_every_registered_scheme_has_a_kind():
    assert set(profiles.SCHEME_KIND) == set(st.SCHEME_STATUS)


@pytest.mark.parametrize("config", [{"eta": 3}, {"eta": 3, "security_profile": "toy"}])
def test_toy_returns_the_config_itself(config):
    assert configs.resolve(config, ("sife", "ddh")) is config
    assert configs.resolve(config, None) is config


def test_default_profile_keeps_toy_parameters():
    kg = SIFEDDHKeyGenerator({"eta": 3})
    assert kg.security_profile == "toy"
    assert kg.sec_param == 128 and int(kg.p).bit_length() == 128


def test_unknown_profile_is_rejected():
    with pytest.raises(FEValidationError, match="unknown security_profile"):
        SIFEDDHKeyGenerator({"eta": 3, "security_profile": "high"})


def test_standard_does_not_modify_the_callers_config():
    config = {"eta": 3, **STANDARD}
    configs.resolve(config, ("sife", "ddh"))
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
    out = configs.resolve(dict(STANDARD), scheme)
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
        configs.resolve({**STANDARD, **weak}, scheme)


@pytest.mark.parametrize("scheme", [
    ("mcfe", "lwe"), ("mcfe", "lwe_threshold"), ("mcfe", "ddh_decentralized"), ("sife", "paillier"),
    ("mcfe", "paillier"), ("mcfe", "ring_lwe_threshold"),
])
def test_standard_statistical_parameter_is_128(scheme):
    """stat_param kept its default 64 (and accepted any explicit value) under
    standard: a 2^-64 distance per sample, against a 128-bit claim."""
    assert configs.resolve(dict(STANDARD), scheme)["stat_param"] == 128
    assert configs.resolve({**STANDARD, "stat_param": 160}, scheme)["stat_param"] == 160
    for weak in (64, 1):
        with pytest.raises(FEValidationError, match="stat_param"):
            configs.resolve({**STANDARD, "stat_param": weak}, scheme)


def test_standard_accepts_stronger_explicit_values():
    out = configs.resolve({**STANDARD, "bit_length": 2048}, ("mife", "paillier"))
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
        configs.resolve(dict(STANDARD), scheme)


def test_standard_pairing_uses_bls12_381_and_warns():
    with pytest.warns(FESecurityWarning, match="120 to 126-bit"):
        out = configs.resolve(dict(STANDARD), ("quadratic", "miqfe"))
    assert out["pairing_group_param"] == "BLS12-381"
    for other in ("MNT224", "BN254"):
        with pytest.raises(FEValidationError, match="BLS12-381"):
            configs.resolve({**STANDARD, "pairing_group_param": other}, ("sife", "fh_ipe_pairing"))


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
    class MySIFE(SIFEDDHKeyGenerator):
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
    kg = SIFEDDHKeyGenerator({"eta": 3, **STANDARD})
    kg.setup()
    assert kg.sec_param == 3072 and kg.p == RFC3526_PRIMES[3072]
    pp = kg.get_public_parameters()
    enc = SIFEDDH({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": [3, 2, -1]})
    assert SIFEDDH({"keys": {"pp": pp}}).decrypt(enc.encrypt([1, 2, 3]), dk, [3, 2, -1]) == 4


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
    assert configs.resolve(dict(STANDARD), scheme)["lwe_n"] == "auto"


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
    assert configs.resolve(dict(STANDARD), scheme)["lwe_n"] == "auto"


def _PK_SIGMA(params):
    """The old rule: the table read at the public key's own width."""
    return params["sigma_q"]


@pytest.mark.parametrize("kg_cls, config, spread", [
    (SIFELWEKeyGenerator, {"eta": 2, "bound_x": 50, "bound_y": 50}, 2),
    (MIFELWEKeyGenerator, {"n": 2, "eta": 1, "bound_x": 50, "bound_y": 50}, 4),
])
def test_standard_abdp15_meets_the_wide_lwe_table_at_the_reduction_width(kg_cls, config, spread):
    kg = kg_cls({**config, **STANDARD})
    assert kg.lwe_n in WIDE_LWE_128_LOG_RATIO[16]
    sigma_prime = kg.sigma_q / (1 + spread * config["bound_x"] * config["eta"] ** 0.5)
    check_wide_lwe(kg.lwe_n, kg.q, sigma_prime)
    # the table is read at sigma', not at the width sigma of the public key
    smaller = abdp15_parameters(
        config["bound_x"], config["bound_y"], config.get("n", 1) * config["eta"], "auto",
        lwe_sigma=_PK_SIGMA,
    )
    assert smaller["lwe_n"] < kg.lwe_n


# lwe_n of `standard` (B = 2X for SIFE, 4X per MIFE slot); the old rule, which
# read the table at the public key's sigma, gave 704, 896, 960, 1344, 896,
# 1088, 960 and 1088
@pytest.mark.parametrize("kg_cls, config, lwe_n", [
    (SIFELWEKeyGenerator, {"eta": 1, "bound_x": 20, "bound_y": 20}, 1088),
    (SIFELWEKeyGenerator, {"eta": 10, "bound_x": 20, "bound_y": 20}, 1152),
    (SIFELWEKeyGenerator, {"eta": 100, "bound_x": 20, "bound_y": 20}, 1344),
    (SIFELWEKeyGenerator, {"eta": 100, "bound_x": 1000, "bound_y": 1000}, 1984),
    (SIFELWEKeyGenerator, {"eta": 10, "bound_x": 50, "bound_y": 50}, 1280),
    (SIFELWEKeyGenerator, {"eta": 10, "bound_x": 256, "bound_y": 256}, 1536),
    (MIFELWEKeyGenerator, {"n": 10, "eta": 10, "bound_x": 20, "bound_y": 20}, 1344),
    (MIFELWEKeyGenerator, {"n": 10, "eta": 100, "bound_x": 20, "bound_y": 20}, 1600),
])
def test_standard_abdp15_dimensions(kg_cls, config, lwe_n):
    if "n" in config:
        config = {**config, "lst_nid": ["nid_{}".format(i) for i in range(config["n"])]}
    assert kg_cls({**config, **STANDARD}).lwe_n == lwe_n


def test_standard_rejects_an_insecure_explicit_abdp15_dimension():
    with pytest.raises(FEValidationError, match="below 128-bit security"):
        SIFELWEKeyGenerator({"eta": 2, "bound_x": 50, "bound_y": 50, "lwe_n": 256, **STANDARD})
    # enough at the public key's sigma (the old rule's choice), not at sigma'
    with pytest.raises(FEValidationError, match="sigma'"):
        SIFELWEKeyGenerator({"eta": 10, "bound_x": 50, "bound_y": 50, "lwe_n": 896, **STANDARD})


def test_standard_rederives_a_record_cached_under_the_old_rule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    config = {"eta": 2, "bound_x": 50, "bound_y": 50, **STANDARD}
    kg = SIFELWEKeyGenerator(config)
    path = tmp_path / "config" / "authority" / kg._scheme_type / "param.json"
    record = json.loads(path.read_text())
    old = kg._serialised(abdp15_parameters(gp.mpz(50), gp.mpz(50), 2, "auto", lwe_sigma=_PK_SIGMA))
    assert old["lwe_n"] < kg.lwe_n
    path.write_text(json.dumps({**record, **old}))
    again = SIFELWEKeyGenerator(config)
    assert again.lwe_n == kg.lwe_n
    assert json.loads(path.read_text())["A_seed"] != record["A_seed"]


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


def test_standard_refuses_threshold_mife_lwe():
    with pytest.raises(FEValidationError, match="very end of the 128-bit table"):
        configs.resolve(dict(STANDARD), ("mife", "lwe_threshold"))
    with pytest.raises(FEValidationError, match="no 'standard' parameter set"):
        ThresholdMIFELWEKeyGenerator({**TMIFE_CONFIG, **STANDARD})


def test_threshold_mife_lwe_has_no_128_bit_dimension_at_its_reduction_width():
    # why standard refuses it: at stat_param 128 the old rule (sigma) took 5952,
    # at sigma' no tabulated dimension is enough
    kg = ThresholdMIFELWEKeyGenerator({**TMIFE_CONFIG, "lwe_n": 16, "stat_param": 128})
    assert wide_lwe_n(kg._threshold_parameters, _PK_SIGMA)["lwe_n"] == 5952
    with pytest.raises(FEValidationError, match="no LWE dimension"):
        wide_lwe_n(kg._threshold_parameters, kg._table_sigma)
    with pytest.raises(FEValidationError, match="no LWE dimension"):
        ThresholdMIFELWEKeyGenerator({**TMIFE_CONFIG, "lwe_n": "auto", "stat_param": 128})


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

    kg = SIFEDDHKeyGenerator(MappingProxyType({"eta": 2, **STANDARD}))
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
        SIFEDDHKeyGenerator(Config({"eta": 2, **STANDARD}))
