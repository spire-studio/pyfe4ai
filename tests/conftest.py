from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

#: PYFE4AI_CACHE_DIR before the run, and the run's parameter cache.
_CACHE = {}


def pytest_configure(config):
    # the parameter caches of the run go to a temporary directory, neither the
    # user's cache nor the working directory (set before the test modules are
    # imported); tests that need a cache of their own set PYFE4AI_CACHE_DIR
    _CACHE["before"] = os.environ.get("PYFE4AI_CACHE_DIR")
    _CACHE["dir"] = tempfile.mkdtemp(prefix="pyfe4ai-param-cache-")
    os.environ["PYFE4AI_CACHE_DIR"] = _CACHE["dir"]
    config.addinivalue_line(
        "markers",
        "extended: non-core pairing-heavy or quadratic FE coverage kept outside the core-only release lane.",
    )
    config.addinivalue_line(
        "markers",
        "core: core FE/IPFE tests intended for the staged core-only release lane.",
    )


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
        path = Path(str(item.fspath))
        path_str = str(path)
        name = path.name
        if _is_extended_test(path_str, name):
            item.add_marker("extended")
        else:
            item.add_marker("core")


def _is_extended_test(path_str: str, name: str) -> bool:
    # everything that needs charm-crypto: the pairing schemes (also the FH-IPFE
    # building block mcfe/test_fh_ipe.py), the pairing-group checks and quadratic FE
    extended_tokens = (
        "fh_ipe",
        "fh_multi_ipe",
        "pairing",
        "quadratic",
    )
    return any(token in path_str or token in name for token in extended_tokens)
