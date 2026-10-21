"""Configuration of the test suite: the parameter cache of a run, and the markers.

The tests are grouped by topic, one package per group (``tests/<group>/``);
every test is marked with its group (``pytest -m threshold`` runs
``tests/threshold``; not ``tests/native``, whose name is the ``native``
mark below) and with:

* ``core`` or ``extended``: the staged core-only release lane runs ``-m core``;
  ``extended`` is what needs charm-crypto (the pairing schemes, the
  pairing-group checks, quadratic FE), by the path of the test module below
  ``tests/``;
* ``native``: the test exercises the native code -- the extension
  ``pyfe4ai._rust_native`` or the C ABI of ``rust/core`` -- and is skipped
  (or not generated) without it: a ``skipif`` whose reason names the
  extension or ``rust/core``, a parameter whose value is ``"native"`` (the
  backend), a pairing test module on a group only the extension provides
  (BLS12-381, the default; see tests/pairing_group.py), an explicit mark
  (``@pytest.mark.native``, or ``tests.pairing_group.GROUP_MARKS``) on a
  test that skips itself in its body, and the review tests of
  :data:`NATIVE_IN_REVIEW`. ``-m "not native"`` then runs, with the
  extension, the tests that also run without it;
* ``slow``: the tests of :data:`SLOW`.

The markers are registered in ``pyproject.toml`` (``--strict-markers``).
See CONTRIBUTING.md, "Tests".
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

#: PYFE4AI_CACHE_DIR before the run, and the run's parameter cache.
_CACHE = {}

TESTS = Path(__file__).resolve().parent

#: Tests that take 2 s or more on an Apple M2 Max in one of the five
#: configurations of the suite (native, python, facct, rounded, threads=4):
#: about two thirds of the suite's time. Listed here, by node id below
#: ``tests/``, rather than decorated, so that tests/review stays as the
#: review wrote it and single parameter sets can be named.
SLOW = frozenset({
    "review/test_native_facct_precision.py::test_crafted_bits_give_the_reference_sample",
    "review/test_adapter_label_race.py::test_one_ciphertext_per_client_and_label_under_threads",
    "review/test_offline_one_time.py::test_two_threads_cannot_use_one_precomputation",
    "review/test_rns_precompute_deadlock.py::test_concurrent_use_of_an_rns_precomputation_does_not_deadlock",
    "review/test_precompute_one_time.py::test_concurrent_use_of_a_native_precomputation_does_not_deadlock",
    "review/test_native_facct_wide_review.py::test_signal_handler_drawing_from_an_xof_random_does_not_hang",
    "sife/test_fullysec_ring_lwe_packed.py::test_packed_decryption_equals_per_element_decryption[10-50-50-standard-2]",
    "sife/test_fullysec_ring_lwe_packed.py::test_packed_decryption_equals_per_element_decryption[4-4-4-standard-2]",
    "sife/test_fullysec_ring_lwe_packed.py::test_packed_decryption_equals_per_element_decryption[1-100-1-standard-2]",
    "sife/test_fullysec_ring_lwe.py::test_standard_round_trip",
    "sife/test_fullysec_lwe_hardening.py::test_extreme_inner_products[1]",
    "sife/test_fullysec_lwe_hardening.py::test_extreme_inner_products[-1]",
    "mife/test_fullysec_lwe.py::test_entire_process",
    "mife/test_fullysec_lwe.py::test_entire_process_lst_ndarray",
    "quadratic/test_quad.py::test_entire_process_lst_ndarray",
    "params/test_security_profiles.py::test_standard_sife_lwe_round_trip_streaming_the_matrix",
    "params/test_security_profiles.py::test_standard_dmcfe_lwe_combines_key_shares_under_a_3072_bit_modulus",
    "config/test_config.py::test_key_generators_read_no_option_as_an_attribute[sife/fullysec_lwe]",
    "config/test_config.py::test_key_generators_read_no_option_as_an_attribute[mife/fullysec_lwe]",
    "config/test_config.py::test_key_generators_read_declared_options_only[sife/fullysec_lwe]",
    "config/test_config.py::test_key_generators_read_declared_options_only[mife/fullysec_lwe]",
})

#: The tests of tests/review that skip themselves (in their body) without the
#: extension -- ``"extension"`` -- or without the pairing group of the tests
#: -- ``"group"``, native when only the extension implements it (see
#: tests/pairing_group.py): marked here, as tests/review stays as written.
NATIVE_IN_REVIEW = {
    "review/test_precompute_one_time.py::test_concurrent_use_of_a_native_precomputation_does_not_deadlock":
        "extension",
    "review/test_refactor_batch2_review.py::test_unknown_sids_do_not_grow_the_warning_registry[tMCFE-FH-pairing]":
        "group",
    **{"review/test_tmcfe_threshold_recheck.py::test_i2b_malformed_value_list_key_shares_are_refused[{}]".format(case):
       "extension" for case in ("binary shamir nested sk_y (TypeError)", "binary shamir z value list n+1 (ValueError)",
                                "shamir None value (TypeError)", "shamir float sk_y (TypeError)",
                                "shamir non-decimal string (ValueError)")},
}

#: A ``skipif`` whose reason says this marks a test ``native``.
_NATIVE_REASON = re.compile(r"native|rust|BLS12-381 backend", re.IGNORECASE)
#: Path parts of the ``extended`` tests (they need charm-crypto): the pairing
#: schemes (also the FH-IPFE building block mcfe/test_fh_ipe.py), the
#: pairing-group checks and quadratic FE.
_EXTENDED = ("fh_ipe", "fh_multi_ipe", "pairing", "quadratic")


def pytest_configure(config):
    # the parameter caches of the run go to a temporary directory, neither the
    # user's cache nor the working directory (set before the test modules are
    # imported); tests that need a cache of their own set PYFE4AI_CACHE_DIR
    _CACHE["before"] = os.environ.get("PYFE4AI_CACHE_DIR")
    _CACHE["dir"] = tempfile.mkdtemp(prefix="pyfe4ai-param-cache-")
    os.environ["PYFE4AI_CACHE_DIR"] = _CACHE["dir"]


def pytest_unconfigure(config):
    if "dir" not in _CACHE:
        return
    shutil.rmtree(_CACHE.pop("dir"), ignore_errors=True)
    before = _CACHE.pop("before")
    if before is None:
        os.environ.pop("PYFE4AI_CACHE_DIR", None)
    else:
        os.environ["PYFE4AI_CACHE_DIR"] = before


def pytest_collection_modifyitems(config, items):
    for item in items:
        rel = _relative(item)
        group = rel.split("/", 1)[0] if "/" in rel else None
        if group is not None and group != "native":  # "native" is the mark of what needs the native code
            item.add_marker(group)
        item.add_marker("extended" if any(token in rel for token in _EXTENDED) else "core")
        name = "{}::{}".format(rel, item.nodeid.partition("::")[2])
        if _needs_native(item) or _native_in_review(name):
            item.add_marker("native")
        if name in SLOW:
            item.add_marker("slow")
    _consume_the_missing_extension_notice()


def _relative(item) -> str:
    """The path of the item's module below ``tests/`` (posix)."""
    path = Path(item.path).resolve()
    try:
        return path.relative_to(TESTS).as_posix()
    except ValueError:  # pragma: no cover - a test outside tests/
        return path.name


def _needs_native(item) -> bool:
    if item.get_closest_marker("native") is not None:
        return True
    for mark in item.iter_markers("skipif"):
        if _NATIVE_REASON.search(str(mark.kwargs.get("reason", ""))):
            return True
    callspec = getattr(item, "callspec", None)
    if callspec is not None and any(isinstance(v, str) and v == "native" for v in callspec.params.values()):
        return True
    module = getattr(item, "module", None)
    if module is not None and getattr(module, "skip_unless_available", None) is not None:
        from tests import pairing_group

        return module.skip_unless_available is pairing_group.skip_unless_available and pairing_group.needs_extension()
    return False


def _native_in_review(name: str) -> bool:
    need = NATIVE_IN_REVIEW.get(name)
    if need == "group":
        from tests import pairing_group

        return pairing_group.needs_extension()
    return need == "extension"


def _consume_the_missing_extension_notice() -> None:
    """Without the extension the library gives one FEWarning at its first
    fallback to Python (:data:`pyfe4ai.backend.MISSING_EXTENSION_NOTICE`):
    given now, silently, so that it does not land in whichever test runs first
    (a test that records warnings, or a run with ``-W error``).
    tests/backend/test_backend_info.py checks the notice in fresh interpreters."""
    from pyfe4ai import backend

    backend._NATIVE_NOTICE.clear()
