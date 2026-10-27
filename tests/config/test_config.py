"""The typed configurations of :mod:`pyfe4ai.core.config`: the conversion at
entry, the declarations against what the key generators read and use, and
the documentation page rendered from them."""

import ast
import importlib
import os
import re
from pathlib import Path

import gmpy2 as gp
import pytest

from pyfe4ai.core import config as configs
from pyfe4ai.core import profiles
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.status import SCHEME_STATUS
from pyfe4ai.primitives.pairing.pairing_backend import pairing_available
from tests.pairing_group import GROUP, GROUP_MARKS

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "configuration.rst")

_SIDS2, _SIDS3 = ["sid_0", "sid_1"], ["sid_0", "sid_1", "sid_2"]

#: A small config every key generator accepts (toy profile).
MINIMAL = {
    ("sife", "ddh"): {}, ("sife", "ddh_ristretto255"): {}, ("sife", "lwe"): {}, ("sife", "paillier"): {},
    ("sife", "damgard"): {}, ("sife", "ring_lwe"): {}, ("sife", "fullysec_ring_lwe"): {},
    ("sife", "fullysec_lwe"): {}, ("sife", "ddh_dynamic"): {}, ("sife", "fh_ipe_pairing"): {},
    ("sife", "part_fh_ipe_pairing"): {},
    ("mife", "ddh"): {"n": 2}, ("mife", "ddh_ristretto255"): {"n": 2}, ("mife", "lwe"): {"n": 2},
    ("mife", "paillier"): {"n": 2}, ("mife", "damgard"): {"n": 2}, ("mife", "ring_lwe"): {"n": 2},
    ("mife", "fullysec_lwe"): {"n": 2}, ("mife", "fh_ipe_pairing"): {"n": 2},
    ("mife", "fh_multi_ipe_pairing"): {"n": 2}, ("mife", "ddh_threshold"): {"n": 2, "lst_sid": _SIDS2},
    ("mife", "lwe_threshold"): {"n": 2, "lst_sid": _SIDS2},
    ("mcfe", "ddh"): {"n": 2}, ("mcfe", "ddh_ristretto255"): {"n": 2}, ("mcfe", "lwe"): {"n": 2},
    ("mcfe", "paillier"): {"n": 2}, ("mcfe", "damgard"): {"n": 2}, ("mcfe", "ring_lwe"): {"n": 2},
    ("mcfe", "fullysec_lwe"): {"n": 2}, ("mcfe", "fh_multi_ipe_pairing"): {"n": 2},
    ("mcfe", "fh_multi_ipe_pairing_threshold"): {"n": 2, "lst_sid": _SIDS2},
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): {"n": 2}, ("mcfe", "ddh_threshold"): {"n": 2},
    ("mcfe", "ddh_decentralized"): {"n": 2}, ("mcfe", "lwe_threshold"): {"n": 2},
    ("mcfe", "lwe_decentralized"): {"n": 2}, ("mcfe", "ring_lwe_threshold"): {"n": 2, "lst_sid": _SIDS3},
    ("mcfe", "ring_lwe_decentralized"): {"n": 2}, ("mcfe", "paillier_decentralized"): {"n": 2},
    ("quadratic", "quad"): {}, ("quadratic", "sgp"): {}, ("quadratic", "miqfe"): {"n": 2},
}


def generator(scheme):
    """The key generator class of *scheme*, named in its config class's docstring."""
    path = re.search(r":class:`~([\w.]+)`", configs.config_class(scheme).__doc__).group(1)
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


#: The schemes as parameters, the pairing ones marked (native with BLS12-381: see tests/pairing_group.py).
SCHEMES = [pytest.param(scheme, marks=GROUP_MARKS if profiles.SCHEME_KIND[scheme] == "pairing" else ())
           for scheme in sorted(MINIMAL)]


def _skip_without_pairing(scheme):
    if profiles.SCHEME_KIND[scheme] == "pairing" and not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP))


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(str(tmp_path), "config"))


def test_every_scheme_has_a_config_class():
    assert set(configs.config_classes()) == set(SCHEME_STATUS) == set(MINIMAL)
    for scheme, cls in configs.config_classes().items():
        assert cls.SCHEME == scheme and issubclass(cls, configs.SchemeConfig)
        assert generator(scheme).__module__ == SCHEME_STATUS[scheme].module


def test_the_documentation_page_is_current():
    with open(DOCS, encoding="utf-8") as f:
        assert f.read() == configs.options_rst(), "regenerate docs/configuration.rst from options_rst()"


def test_configs_are_dicts_with_declared_options():
    cfg = configs.ThresholdMCFERingLWEConfig(n=3, t=3, lst_sid=_SIDS3)
    assert isinstance(cfg, dict) and cfg == {"n": 3, "t": 3, "lst_sid": _SIDS3}
    assert cfg.n == 3 and cfg.t == 3 and cfg.eta == 1 and cfg.ring_modulus == "rns" and cfg.ct_format == "binary"
    assert cfg.sharing is None and cfg.key_shares is None  # derived by the key generator
    assert configs.MCFERingLWEConfig().ring_modulus == "prime"
    assert repr(configs.SIFEDDHConfig(eta=3)) == "SIFEDDHConfig({'eta': 3})"
    assert configs.SIFEDDHConfig.eta.default == 5 and configs.MIFEDDHConfig.eta.default == 1
    with pytest.raises(TypeError):
        configs.ETA(defualt=3)


def test_key_generators_get_their_config_class():
    from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator

    seen = []
    original = MCFERingLWEKeyGenerator.__init__

    def init(self, config, **kwargs):
        seen.append(config)
        original(self, config, **kwargs)

    given = {"n": 2, "ring_n": 16}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MCFERingLWEKeyGenerator, "__init__", init)
        MCFERingLWEKeyGenerator(given)
        MCFERingLWEKeyGenerator(config=configs.MCFERingLWEConfig(given))
    assert [type(c) for c in seen] == [configs.MCFERingLWEConfig] * 2
    assert seen[0] == given and seen[0] is not given
    with pytest.raises(FEValidationError, match="must be a mapping"):
        MCFERingLWEKeyGenerator([("n", 2)])


def test_resolve_keeps_the_class_and_copies_under_standard():
    toy = configs.MCFEDDHConfig(n=2)
    assert configs.resolve(toy, ("mcfe", "ddh")) is toy
    std = configs.MCFERingLWEConfig(n=2, security_profile="standard")
    out = configs.resolve(std, ("mcfe", "ring_lwe"))
    assert type(out) is configs.MCFERingLWEConfig and out is not std
    assert out["ring_n"] == "auto" and out["sec_param"] == out["stat_param"] == 128 and "ring_n" not in std
    assert type(configs.resolve({"security_profile": "standard"}, ("mcfe", "ring_lwe"))) is dict


def test_threshold_rules():
    assert configs.resolve_sharing({}, 3, 2, "x") == "replicated"
    assert configs.resolve_sharing({}, 3, 3, "x") == "shamir"
    with pytest.warns(Warning, match="Shamir sharing with t = 2 < s = 3"):
        assert configs.resolve_sharing({"sharing": "shamir"}, 3, 2, "x") == "shamir"
    with pytest.raises(FEValidationError, match="Shamir sharing"):
        configs.resolve_sharing({"sharing": "shamir", "security_profile": "standard"}, 3, 2, "x")
    with pytest.raises(FEValidationError, match="sharing must be one of"):
        configs.resolve_sharing({"sharing": "additive"}, 3, 2, "x")
    assert configs.threshold_key_shares({}, "replicated-seeded") == "full"
    assert configs.threshold_key_shares({}, "shamir") == "split"
    with pytest.raises(FEValidationError, match="split key shares"):
        configs.threshold_key_shares({"key_shares": "split"}, "replicated-seeded")
    with pytest.raises(FEValidationError, match="accounts for Gaussian flooding"):
        configs.check_kl_flooding("uniform", "shamir")
    with pytest.raises(FEValidationError, match="not sharing='replicated'"):
        configs.check_kl_sharing("replicated")
    with pytest.raises(FEValidationError, match="no statistical flooding"):
        configs.check_no_flooding_options({"max_partial_decryptions": None})


def _recording(cls, seen):
    def record(name):
        seen.add(name)

    class Recording(cls):
        def get(self, key, default=None):
            record(key)
            return super().get(key, default)

        def __getitem__(self, key):
            record(key)
            return super().__getitem__(key)

        def __contains__(self, key):
            record(key)
            return super().__contains__(key)

        def setdefault(self, key, default=None):
            record(key)
            return super().setdefault(key, default)

    return Recording


@pytest.mark.parametrize("scheme", SCHEMES, ids="/".join)
def test_key_generators_read_declared_options_only(scheme, monkeypatch):
    """The options a key generator reads from its converted config (built from
    a minimal config, then set up) are declared. Reads of a copy (a base
    class given ``dict(config, ...)``) and of branches the minimal config
    does not take are not seen; the inventory of every option's reads is the
    config-matrix probe of the batch-3 records."""
    _skip_without_pairing(scheme)
    seen = set()
    original, key_generator = configs.config_class, generator(scheme)
    monkeypatch.setattr(configs, "config_class", lambda s: _recording(original(s), seen))
    kg = key_generator(dict(MINIMAL[scheme]))
    kg.setup()
    declared = original(scheme).OPTIONS
    assert seen and seen <= set(declared), sorted(seen - set(declared))


_ATTRIBUTE_DEFAULTS = {"eta", "bound_x", "bound_y", "bound", "bit_length", "ring_modulus", "label_hash", "t",
                       "flooding_accounting", "ct_format", "matrix_mode", "mddh_k", "m", "sec_level",
                       "subspace_dim", "coeff_bound", "bound_c", "sec_param", "group", "stat_param"}


@pytest.mark.parametrize("scheme", SCHEMES, ids="/".join)
def test_declared_defaults_are_the_key_generators(scheme):
    _skip_without_pairing(scheme)
    cls = configs.config_class(scheme)
    kg = generator(scheme)(dict(MINIMAL[scheme]))
    checked = 0
    for name, option in cls.OPTIONS.items():
        if (name not in _ATTRIBUTE_DEFAULTS or name in MINIMAL[scheme] or option.default == "auto"
                or isinstance(option.default, configs.Derived)):
            continue
        if not hasattr(kg, name):
            continue
        value = getattr(kg, name)
        if isinstance(value, type(gp.mpz(0))):
            value = int(value)
        assert value == option.default, (name, value, option.default)
        checked += 1
    assert checked


def test_accepted_values_are_the_codes():
    """The accepted values the classes declare (and the documentation page
    shows) are the code's own constants, in their order."""
    from pyfe4ai.primitives.hashing import LABEL_HASHES
    from pyfe4ai.primitives.lattice.lwe_matrix import MATRIX_MODES
    from pyfe4ai.primitives.lattice.ring_lwe_utils import RING_MODULI
    from pyfe4ai.primitives.sampling import GAUSSIAN_SAMPLERS
    from pyfe4ai.schemes.sife.fullysec_ring_lwe import LAYOUTS

    assert configs.LABEL_HASH.allowed == LABEL_HASHES
    assert configs.MATRIX_MODE.allowed == MATRIX_MODES
    assert configs.GAUSSIAN_SAMPLER.allowed == GAUSSIAN_SAMPLERS
    assert configs.RING_MODULUS.allowed == RING_MODULI
    assert configs.SIFEFullySecRingLWEConfig.ndarray_layout.allowed == LAYOUTS
    assert configs.SchemeConfig.security_profile.allowed == profiles.PROFILES
    for cls in configs.config_classes().values():
        for name, option in list(cls.OPTIONS.items()) + list(cls.CRYPTO.items()):
            if name in ("gaussian_sampler", "label_hash", "matrix_mode") and isinstance(option.allowed, tuple):
                assert set(option.allowed) <= set({"gaussian_sampler": GAUSSIAN_SAMPLERS, "label_hash": LABEL_HASHES,
                                                   "matrix_mode": MATRIX_MODES}[name]), (cls.__name__, name)


# ---------------------------------------------------------------------------
# options are read with config.get (review item I3-d of batch 3)

_CONFIG_NAME = re.compile(r"(^|_)(config|cfg|conf|options)$", re.IGNORECASE)


def _option_names() -> set:
    names = set(configs.SchemeConfig.OPTIONS) | set(configs.CryptoConfig.OPTIONS)
    for cls in configs.config_classes().values():
        names |= set(cls.OPTIONS) | set(cls.CRYPTO)
    return names


def _attribute_reads(source: str, names: set) -> list:
    """``(line, receiver, option)`` of every attribute of a config-like name
    (``config``, ``self.keygen_config``, ``cfg``, ...) read as an option."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in names:
            receiver = node.value
            name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", None)
            if name and _CONFIG_NAME.search(name):
                out.append((node.lineno, name, node.attr))
    return out


def test_library_code_reads_no_option_as_an_attribute():
    """The library reads options with ``config.get(name, default)``, the
    default written where it is read; the defaults declared with
    :class:`~pyfe4ai.core.config.Option` only document them and must not
    become behaviour through an attribute read (``config.ring_n``). No
    module of pyfe4ai reads an option as an attribute of a config."""
    names = _option_names()
    assert {"ring_n", "eta", "sharing", "ct_format", "cache_dir"} <= names
    assert _attribute_reads("def f(config, self):\n    return config.ring_n, self.keygen_config.eta, cfg.x\n",
                            names) == [(2, "config", "ring_n"), (2, "keygen_config", "eta")]
    package = Path(configs.__file__).resolve().parents[1]
    hits = []
    for path in sorted(package.rglob("*.py")):
        for line, receiver, option in _attribute_reads(path.read_text(encoding="utf-8"), names):
            hits.append("{}:{}: {}.{}".format(path.relative_to(package.parent), line, receiver, option))
    assert not hits, hits


@pytest.mark.parametrize("scheme", SCHEMES, ids="/".join)
def test_key_generators_read_no_option_as_an_attribute(scheme, monkeypatch):
    """At run time: building and setting up every key generator reads no
    option of its config object as an attribute."""
    _skip_without_pairing(scheme)
    reads = []
    original = configs.Option.__get__

    def get(self, config, owner=None):
        if config is not None:
            reads.append(self.name)
        return original(self, config, owner)

    monkeypatch.setattr(configs.Option, "__get__", get)
    assert configs.MCFERingLWEConfig(n=2).ring_modulus == "prime" and reads == ["ring_modulus"]
    reads.clear()
    kg = generator(scheme)(dict(MINIMAL[scheme]))
    kg.setup()
    assert not reads, reads
