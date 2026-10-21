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
