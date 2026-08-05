from __future__ import annotations

from pathlib import Path


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "extended: non-core pairing-heavy or quadratic FE coverage kept outside the core-only release lane.",
    )
    config.addinivalue_line(
        "markers",
        "core: core FE/IPFE tests intended for the staged core-only release lane.",
    )


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
    extended_tokens = (
        "fh_ipe_pairing",
        "fh_multi_ipe_pairing",
        "part_fh_ipe_pairing",
        "quadratic",
    )
    return any(token in path_str or token in name for token in extended_tokens)
