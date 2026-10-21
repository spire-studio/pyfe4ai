"""Where the pairing schemes' FESecurityWarning of the ``standard`` profile points.

The ``standard`` profile runs the pairing schemes on BLS12-381 (about 120
to 126-bit security) with a
:class:`~pyfe4ai.core.exceptions.FESecurityWarning`. It was attributed to a
line inside the library (``core/base.py``, where key generators resolve
their config), so Python's default filter showed it once per process for
every scheme and call site together, and filters on the application's
module did not match it. It is now attributed to the application's call,
the first frame outside pyfe4ai (as :func:`~pyfe4ai.core.exceptions.warn_at_caller`
does for :class:`~pyfe4ai.core.exceptions.FEWarning`): its category is
unchanged, and the default filter shows it once per call site of the
application.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai.core import config as configs
from pyfe4ai.core.exceptions import FESecurityWarning

STANDARD = {"security_profile": "standard"}
needs_bls = pytest.mark.skipif(not native_backend.available(), reason="BLS12-381 needs the native extension")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _security_warnings(caught):
    return [w for w in caught if issubclass(w.category, FESecurityWarning)]


def test_the_pairing_warning_points_at_the_application_call():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        configs.resolve(dict(STANDARD), ("quadratic", "miqfe"))
    (w,) = _security_warnings(caught)
    assert w.category is FESecurityWarning and "120 to 126-bit" in str(w.message)
    assert w.filename == __file__, w.filename


def test_the_default_filter_shows_it_once_per_call_site():
    shown = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            configs.resolve(dict(STANDARD), ("quadratic", "miqfe"))  # one call site: shown once
        configs.resolve(dict(STANDARD), ("quadratic", "miqfe"))  # another: shown again
        shown = _security_warnings(caught)
    assert len(shown) == 2 and all(w.filename == __file__ for w in shown)
    assert shown[0].lineno != shown[1].lineno


@needs_bls
@pytest.mark.parametrize("scheme,options", [("mcfe/fh_multi_ipe_pairing", {"n": 2, "lst_nid": ["c0", "c1"]}),
                                            ("sife/fh_ipe_pairing", {})], ids=["mcfe", "sife"])
def test_building_a_pairing_key_generator_warns_at_the_application_call(scheme, options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyfe4ai.create(scheme, dict(STANDARD, **options))
    (w,) = _security_warnings(caught)
    assert w.filename == __file__ and "BLS12-381" in str(w.message)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FESecurityWarning, module=__name__)
        with pytest.raises(FESecurityWarning):  # a filter on the application's module matches it
            pyfe4ai.create(scheme, dict(STANDARD, **options))


# -- the other FESecurityWarnings: their fixed stack levels landed inside pyfe4ai on these paths

NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
RING = dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=SIDS, eta=1, ring_n=16, sec_param=64, bound_x=10, bound_y=3)


def _only_here(fn, text):
    """The FESecurityWarnings of *fn*: at least one, every one with *text* and attributed to this file."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    found = _security_warnings(caught)
    assert found and all(text in str(w.message) for w in found), [str(w.message) for w in found]
    assert [w.filename for w in found] == [__file__] * len(found), [w.filename for w in found]


def test_explicit_shamir_below_s_points_at_the_application_call():
    """It pointed at registry.py (pyfe4ai.create) or ml/wrapper.py (the ML adapter)."""
    from pyfe4ai.ml import FESchemeWrapper

    _only_here(lambda: pyfe4ai.create("mcfe/ring_lwe_threshold", dict(RING, sharing="shamir")), "Shamir sharing")
    _only_here(lambda: FESchemeWrapper("mcfe", "ring_lwe_threshold", dict(RING, sharing="shamir")), "Shamir sharing")


def test_a_legacy_shamir_key_share_served_points_at_the_application_call():
    """Through compute_lst_ndarray_ct it pointed at mife/lwe_threshold.py."""
    from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

    kg = ThresholdMIFELWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS,
                                       "lst_sid": SIDS, "lwe_n": 16, "bound_x": 6, "bound_y": 6, "sharing": "shamir"})
    kg.setup()
    pp = kg.get_public_parameters()
    weights = {n: [1] for n in NIDS}
    cts = {n: ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt_lst_ndarray(
        [np.ones(2)]) for n in NIDS}
    dk = kg.get_decryption_keys("s0", credentials={"fusion_weight": weights})
    server = ThresholdMIFELWE({"id": "s0", "legacy_shamir": "warn", "keys": {"pp": pp, "dk": dk}})
    _only_here(lambda: server.compute_lst_ndarray_ct(cts, weights, dk, ["s0", "s1"]), "Shamir key share")


def test_a_sampler_override_points_at_the_application_call():
    """It pointed at ring_lwe_threshold.py (a tMCFE-Ring-LWE client) or ml/wrapper.py."""
    from pyfe4ai.ml import FESchemeWrapper, encrypt_gradient
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(RING))
    kg.setup()
    pp = kg.get_public_parameters()
    _only_here(lambda: tm.ThresholdMCFERingLWE({"id": "c0", "gaussian_sampler": "rounded",
                                                "keys": {"pp": pp, "sk": kg.get_private_keys("c0")}}), "overrides")
    wrapper = FESchemeWrapper("mcfe", "ring_lwe", dict(n=1, lst_nid=["c0"], eta=1, ring_n=16, sec_param=64,
                                                       bound_x=10, bound_y=3), crypto_options={"gaussian_sampler": "rounded"})
    _only_here(lambda: encrypt_gradient(np.ones(2), wrapper, nid="c0", label="L"), "overrides")


def test_a_known_insecure_scheme_points_at_the_application_call(monkeypatch):
    """It pointed at the scheme's own module (the constructor chain above IPFEAbsKeyGenerator / IPFEAbsCrypto)."""
    import dataclasses

    from pyfe4ai.core import status as st
    from pyfe4ai.schemes.mife.lwe import MIFELWE, MIFELWEKeyGenerator

    key = ("mife", "lwe")
    monkeypatch.setitem(st.SCHEME_STATUS, key, dataclasses.replace(st.SCHEME_STATUS[key], status=st.KNOWN_INSECURE,
                                                                     issue="test issue"))
    config = {"n": 2, "lwe_n": 8, "bound_x": 2, "bound_y": 2}
    _only_here(lambda: MIFELWEKeyGenerator(dict(config)), "known to be insecure")
    _only_here(lambda: pyfe4ai.create("mife/lwe", dict(config)), "known to be insecure")
    kg = MIFELWEKeyGenerator(dict(config))
    kg.setup()
    _only_here(lambda: MIFELWE({"id": "c0", "keys": {"pp": kg.get_public_parameters()}}), "known to be insecure")
