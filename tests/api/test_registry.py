"""The scheme registry (:mod:`pyfe4ai.core.registry`) against the code: the
classes and their names, the config classes, the tables derived from it
(status, kinds, refusals, the ML adapter, the benchmark scripts) and the
capabilities, checked on the schemes themselves where that is cheap."""

import importlib
import importlib.util
import inspect
import os
import pkgutil
import types
from pathlib import Path

import pytest

import pyfe4ai.schemes
from pyfe4ai.core import config as configs
from pyfe4ai.core import profiles, registry
from pyfe4ai.core import status as st
from pyfe4ai.core.base import IPFEAbsCrypto, IPFEAbsKeyGenerator
from pyfe4ai.core.exceptions import FESecurityWarning, FEValidationError
from pyfe4ai.ml import wrapper as ml_wrapper

ROOT = Path(__file__).resolve().parents[2]
SCHEMES = registry.SCHEMES
IDS = [s.id for s in SCHEMES]

#: Base classes of several schemes, defined outside the registered modules.
SHARED_BASES = {"DDHKeyGeneratorBase", "DamgardDDHKeyGeneratorBase", "ECDDHKeyGeneratorBase", "ECDDHCryptoBase"}


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(str(tmp_path), "config"))


def test_entries_are_unique_and_complete():
    assert len(SCHEMES) == len(set(IDS)) == 42
    assert {s.family for s in SCHEMES} == set(registry.FAMILIES)
    for s in SCHEMES:
        assert s.kind in registry.KINDS
        assert s.capabilities <= set(registry.CAPABILITIES)
        assert s.status.module == s.module
        assert s.status.status in registry.STATUSES and s.status.fidelity in registry.FIDELITIES
        assert registry.get_scheme(s.id) is s and registry.get_scheme(s.key) is s


@pytest.mark.parametrize("scheme", SCHEMES, ids=IDS)
def test_classes_are_named_after_the_id(scheme):
    module = importlib.import_module(scheme.module)
    kg, crypto = scheme.key_generator, scheme.crypto
    assert (kg.__name__, crypto.__name__) == (registry.class_name(scheme.id, "KeyGenerator"),
                                              registry.class_name(scheme.id))
    assert kg.__module__ == crypto.__module__ == module.__name__
    assert issubclass(kg, IPFEAbsKeyGenerator) and issubclass(crypto, IPFEAbsCrypto)
    assert scheme.config.__name__ == registry.class_name(scheme.id, "Config") and scheme.config.SCHEME == scheme.key


def test_naming_rule():
    assert registry.class_name("mcfe/ring_lwe") == "MCFERingLWE"
    assert registry.class_name("mcfe/ring_lwe_threshold", "KeyGenerator") == "ThresholdMCFERingLWEKeyGenerator"
    assert registry.class_name("mcfe/ddh_decentralized") == "DecentralizedMCFEDDH"
    assert registry.class_name("mife/fh_multi_ipe_pairing", "Config") == "MIFEFHMultiIPEPairingConfig"
    assert registry.class_name("quadratic/miqfe") == "QuadraticMIQFE"


def test_every_scheme_class_is_registered():
    """A key generator or crypto class in pyfe4ai.schemes is a registered one
    (or a shared base class): a new scheme cannot be left out of the registry."""
    registered = {s.key_generator for s in SCHEMES} | {s.crypto for s in SCHEMES}
    found = set()
    for info in pkgutil.walk_packages(pyfe4ai.schemes.__path__, "pyfe4ai.schemes."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and obj.__module__ == module.__name__ and not name.startswith("_")
                    and issubclass(obj, (IPFEAbsKeyGenerator, IPFEAbsCrypto)) and name not in SHARED_BASES):
                found.add(obj)
    assert found == registered


@pytest.mark.parametrize("family", list(registry.FAMILIES))
def test_family_packages_export_their_schemes_without_aliases(family):
    package = importlib.import_module("pyfe4ai.schemes." + family)
    expected = []
    for s in SCHEMES:
        if s.family == family:
            expected += [s.crypto_name, s.key_generator_name]
    assert package.__all__ == expected
    classes = {name: obj for name, obj in vars(package).items() if inspect.isclass(obj)}
    assert sorted(classes) == sorted(expected)
    assert all(classes[name].__name__ == name for name in expected)


def test_config_classes_are_the_registrys():
    assert list(configs.config_classes()) == [s.key for s in SCHEMES]
    for s in SCHEMES:
        assert configs.config_class(s.key) is s.config
        assert getattr(configs, s.config_name) is s.config


def test_status_and_kinds_are_derived():
    assert list(st.SCHEME_STATUS) == [s.key for s in SCHEMES]
    assert all(st.SCHEME_STATUS[s.key] is s.status for s in SCHEMES)
    assert profiles.SCHEME_KIND == {s.key: s.kind for s in SCHEMES}
    for s in SCHEMES:
        assert st.get_scheme_status(*s.key) is st.get_scheme_status(s.id) is s.status
        assert st.status_for_module(s.module) == (s.key, s.status)
        assert profiles.scheme_of(s.key_generator) == profiles.scheme_of(s.crypto) == s.key
    assert st.SchemeStatus is registry.SchemeStatus


@pytest.mark.parametrize("scheme", SCHEMES, ids=IDS)
def test_refusals_are_the_profiles(scheme):
    config = scheme.config({"security_profile": "standard"})
    if scheme.refused:
        with pytest.raises(FEValidationError, match="no 'standard' parameter set for {}: ".format(scheme.id)) as info:
            configs.resolve(config, scheme.key)
        assert str(info.value).endswith(scheme.refused)
    elif scheme.kind == registry.PAIRING:
        with pytest.warns(FESecurityWarning):
            configs.resolve(config, scheme.key)
    else:
        configs.resolve(config, scheme.key)


@pytest.mark.parametrize("scheme", SCHEMES, ids=IDS)
def test_structural_capabilities(scheme):
    crypto, caps = scheme.crypto, scheme.capabilities
    assert (registry.LABELS in caps) == (scheme.family == "mcfe")
    threshold = hasattr(crypto, "share_decrypt") and hasattr(crypto, "combine_decrypt")
    assert (registry.THRESHOLD in caps) == threshold == issubclass(scheme.config, configs.ThresholdConfig)
    decentralized = hasattr(crypto, "derive_function_decryption_key_share") and hasattr(
        crypto, "combine_function_decryption_key_share")
    assert (registry.DECENTRALIZED in caps) == decentralized
    assert (registry.PAIRING_GROUP in caps) == ("pairing_group_param" in scheme.config.OPTIONS)
    assert (registry.ML_SERVERS in caps) <= (registry.THRESHOLD in caps)


#: Ring-LWE configs small enough to set up quickly (every scheme of the kind).
_RING = {
    "sife/ring_lwe": {"ring_n": 16, "eta": 2},
    "sife/fullysec_ring_lwe": {"ring_n": 16, "eta": 2},
    "mife/ring_lwe": {"n": 2, "ring_n": 16},
    "mcfe/ring_lwe": {"n": 2, "ring_n": 16},
    "mcfe/ring_lwe_decentralized": {"n": 2, "ring_n": 16},
    "mcfe/ring_lwe_threshold": {"n": 2, "ring_n": 16, "t": 2, "lst_sid": ["s0", "s1"]},
}


def test_the_ring_lwe_configs_cover_the_kind():
    assert set(_RING) == set(registry.list_schemes(kind=registry.RING_LWE))


def _has_bytes(obj) -> bool:
    if isinstance(obj, (bytes, bytearray)):
        return True
    if isinstance(obj, dict):
        return any(_has_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_bytes(v) for v in obj)
    return False


@pytest.mark.parametrize("scheme_id", sorted(_RING))
def test_binary_ciphertexts_and_rns(scheme_id):
    """binary_ct: the crypto objects write binary ciphertexts when asked
    (``ct_format="binary"``; the threshold scheme by default); rns: the key
    generator takes ``ring_modulus="rns"``."""
    scheme = registry.get_scheme(scheme_id)
    kg, crypto = registry.create(scheme_id, _RING[scheme_id])
    kg.setup()
    pp = kg.get_public_parameters()
    nid = kg.lst_nid[0] if getattr(kg, "lst_nid", None) else None
    sk = kg.get_private_keys(nid) if nid is not None else kg.get_private_keys()
    client = crypto({"id": nid, "ct_format": "binary", "keys": {"pp": pp, "sk": sk}})
    if scheme.has(registry.LABELS):
        ct = client.encrypt([[1]], "L")
    elif scheme_id == "sife/fullysec_ring_lwe":
        ct = client.encrypt([1, 0])
    else:
        ct = client.encrypt([[1] for _ in range(kg.eta)])  # eta rows, one vector
    assert _has_bytes(ct) == scheme.has(registry.BINARY_CT)
    if scheme.has(registry.RNS):
        kg = scheme.key_generator(dict(_RING[scheme_id], ring_modulus="rns"))
        kg.setup()
        assert kg.get_public_parameters().get("q_factors")
    else:
        with pytest.raises(FEValidationError):
            kg = scheme.key_generator(dict(_RING[scheme_id], ring_modulus="rns"))
            kg.setup()


def test_the_ml_adapter_serves_the_registered_schemes():
    def stub(scheme):
        return types.SimpleNamespace(scheme_type=scheme.family, variant=scheme.variant)

    for s in SCHEMES:
        assert ml_wrapper._serves_threshold(stub(s)) == s.has(registry.ML_SERVERS)
    with pytest.raises(ValueError, match=r"Unknown scheme 'mcfe/nope'. Supported: \[") as info:
        ml_wrapper.FESchemeWrapper("mcfe", "nope", {})
    assert str(sorted(IDS)) in str(info.value)
    for s in SCHEMES:
        if s.group is not None:
            assert ml_wrapper._group_variant((s.family, s.instance_of), s.group) == s.key
            assert ml_wrapper._group_variant(s.key, s.group) == s.key
    with pytest.raises(ValueError, match="has no instantiation in group 'ristretto255'"):
        ml_wrapper._group_variant(("mcfe", "ddh_threshold"), "ristretto255")


def test_the_ml_adapter_labels_and_classes():
    wrapper = ml_wrapper.FESchemeWrapper("mcfe", "ddh", {"n": 2})
    scheme = registry.get_scheme("mcfe/ddh")
    assert (wrapper._kg_cls, wrapper._crypto_cls, wrapper.has_label) == (scheme.key_generator, scheme.crypto, True)
    assert wrapper.status is scheme.status
    assert ml_wrapper.FESchemeWrapper("sife", "ddh", {"eta": 2}).has_label is False


def _load(path):
    spec = importlib.util.spec_from_file_location("bench_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_benchmark_scripts_cover_the_registry():
    coverage = _load(ROOT / "benchmarks" / "commands" / "fl_coverage.py")
    assert set(coverage.CONFIGS) == {s.key for s in SCHEMES}
    compare = _load(ROOT / "benchmarks" / "commands" / "fl_compare.py")
    for key, _, _ in compare.FAMILIES.values():
        registry.get_scheme(key)
    assert set(compare.OFFLINE_FAMILIES) <= set(compare.FAMILIES)


def test_every_benchmark_command_has_a_module_and_help(capsys, monkeypatch):
    """benchmarks/run.py: one module per command (and per file), whose main() parses its own options."""
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    run = _load(ROOT / "benchmarks" / "run.py")
    modules = {c.module for c in run.COMMANDS.values()}
    files = {p.stem for p in (ROOT / "benchmarks" / "commands").glob("*.py")} - {"__init__"}
    assert modules == files and len(run.COMMANDS) == 17
    for name in run.COMMANDS:
        with pytest.raises(SystemExit) as stop:
            run.main([name, "--help"])
        assert stop.value.code == 0, name
        assert "usage: run.py " + name in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        run.main(["nope"])
    assert stop.value.code == 2
