"""The top-level API: pyfe4ai.create, the registry queries and the exported
exceptions (and their listing on the API page of the documentation)."""

import os
import re
import types
from pathlib import Path

import pytest

import pyfe4ai
from pyfe4ai.core import config as configs
from pyfe4ai.core import exceptions, registry, versions

DOCS = Path(__file__).resolve().parents[2] / "docs" / "api" / "index.rst"


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(str(tmp_path), "config"))


def test_create_returns_the_key_generator_and_the_crypto_class():
    kg, crypto = pyfe4ai.create("sife/ddh", {"eta": 3})
    scheme = pyfe4ai.get_scheme("sife/ddh")
    assert type(kg) is scheme.key_generator and crypto is scheme.crypto
    assert isinstance(pyfe4ai.create("sife/ddh", eta=3), pyfe4ai.CreatedScheme)
    assert not hasattr(kg, "msk")  # built, not set up
    kg.setup()
    pp = kg.get_public_parameters()
    ct = crypto({"precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys()}}).encrypt([1, 2, 3])
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1, 1, 1]})
    assert crypto({"precision": 0, "keys": {"pp": pp}}).decrypt(ct, dk, [1, 1, 1]) == 6


def test_create_is_the_classs_constructor():
    """The same config conversion and profile as building the class."""
    kg = pyfe4ai.create(("mcfe", "ring_lwe"), {"n": 2}, ring_n=16).key_generator
    assert isinstance(kg, registry.get_scheme("mcfe/ring_lwe").key_generator)
    assert kg.n == 2 and kg.ring_n == 16
    given = {"n": 2, "ring_n": 32}
    assert pyfe4ai.create("mcfe/ring_lwe", given, ring_n=16).key_generator.ring_n == 16
    assert given == {"n": 2, "ring_n": 32}  # the options are added to a copy
    with pytest.raises(exceptions.FEValidationError, match="no 'standard' parameter set for mcfe/paillier_decentralized"):
        pyfe4ai.create("mcfe/paillier_decentralized", security_profile="standard")
    with pytest.raises(exceptions.FEValidationError, match="must be a mapping"):
        pyfe4ai.create("sife/ddh", [("eta", 3)])
    with pytest.raises(exceptions.FEValidationError, match="must be a mapping"):
        pyfe4ai.create("sife/ddh", [("eta", 3)], sec_param=128)


@pytest.mark.parametrize("bad", ["sife/nope", "sife", "SIFE/DDH", ("sife",), None, 3])
def test_unknown_schemes(bad):
    with pytest.raises(exceptions.FEValidationError, match="unknown scheme"):
        pyfe4ai.create(bad)
    with pytest.raises(exceptions.FEValidationError, match="unknown scheme"):
        pyfe4ai.get_scheme(bad)


def test_list_schemes():
    ids = pyfe4ai.list_schemes()
    assert ids == [s.id for s in registry.SCHEMES] and len(ids) == 42
    assert pyfe4ai.list_schemes("quadratic") == ["quadratic/quad", "quadratic/sgp", "quadratic/miqfe"]
    assert pyfe4ai.list_schemes("mcfe", capabilities="threshold") == [
        "mcfe/fh_multi_ipe_pairing_threshold", "mcfe/ddh_threshold", "mcfe/lwe_threshold", "mcfe/ring_lwe_threshold"]
    assert pyfe4ai.list_schemes(capabilities=["threshold", "ml_servers"]) == ["mcfe/ring_lwe_threshold"]
    assert pyfe4ai.list_schemes(status="unreviewed") == [
        "sife/ddh_ristretto255", "mife/ddh_ristretto255", "mcfe/ddh_ristretto255"]
    assert pyfe4ai.list_schemes(kind="lwe_matrix") == ["sife/fullysec_lwe", "mife/fullysec_lwe"]
    for kwargs in ({"family": "tmcfe"}, {"capabilities": "thresold"}, {"status": "ok"}, {"kind": "lattice"}):
        with pytest.raises(exceptions.FEValidationError, match="unknown"):
            pyfe4ai.list_schemes(**kwargs)
    assert set(pyfe4ai.CAPABILITIES) == {"labels", "threshold", "decentralized", "pairing", "ml_servers",
                                          "binary_ct", "rns"}


def test_get_scheme():
    scheme = pyfe4ai.get_scheme("mcfe/ring_lwe_threshold")
    assert scheme is pyfe4ai.get_scheme(("mcfe", "ring_lwe_threshold"))
    assert (scheme.id, scheme.key, scheme.kind) == ("mcfe/ring_lwe_threshold", ("mcfe", "ring_lwe_threshold"), "ring_lwe")
    assert scheme.crypto.__name__ == "ThresholdMCFERingLWE"
    assert scheme.key_generator.__name__ == "ThresholdMCFERingLWEKeyGenerator"
    assert scheme.config is configs.ThresholdMCFERingLWEConfig
    assert scheme.has("threshold", "labels") and not scheme.has("decentralized")
    assert scheme.status is pyfe4ai.get_scheme_status("mcfe", "ring_lwe_threshold")
    assert pyfe4ai.get_scheme_status("mcfe/ring_lwe_threshold") is scheme.status
    assert isinstance(scheme.status, pyfe4ai.SchemeStatus) and isinstance(scheme, pyfe4ai.Scheme)
    with pytest.raises(exceptions.FEValidationError, match="unknown capabilities"):
        scheme.has("binary")
    with pytest.raises(KeyError):
        pyfe4ai.get_scheme_status("mcfe/nope")


def test_exceptions_are_exported():
    for name in ("FEError", "FEConfigError", "FEKeyError", "FEValidationError", "FESchemeError",
                 "FESecurityWarning", "FEWarning"):
        assert getattr(pyfe4ai, name) is getattr(exceptions, name)
    assert pyfe4ai.MixedSharingsError is versions.MixedSharingsError
    assert issubclass(pyfe4ai.MixedSharingsError, pyfe4ai.FESchemeError)
    public = [name for name, obj in vars(exceptions).items()
              if isinstance(obj, type) and issubclass(obj, BaseException) and obj.__module__ == exceptions.__name__]
    assert set(public) <= set(pyfe4ai.__all__)


def test_the_exports_are_listed_on_the_api_page():
    text = DOCS.read_text(encoding="utf-8")
    section = text.split("Public and internal modules", 1)[0]
    for name in pyfe4ai.__all__:
        assert re.search(r"[.~`]{}\b".format(re.escape(name)), section), name
    exported = {name for name, obj in vars(pyfe4ai).items()
                if not name.startswith("_") and not isinstance(obj, types.ModuleType)}
    assert exported == set(pyfe4ai.__all__) - {"__version__"}
