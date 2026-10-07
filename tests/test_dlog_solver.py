"""Tests for the cached dlog table in utils.dlog_solver."""

from __future__ import annotations

import json
import pathlib
import pickle

import gmpy2 as gp

from pyfe4ai.utils.crypto_utils import group_generator_fe
from pyfe4ai.utils.dlog_solver import dlog_table_solve, load_or_build_dlog_table


class _Payload:
    def __init__(self, marker: pathlib.Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (pathlib.Path.touch, (self.marker,))


def _group():
    p, _, _, g = group_generator_fe(64)
    return gp.digits(g), gp.digits(p)


def test_planted_pickle_is_never_loaded(tmp_path):
    """Regression (H3): a pickle placed next to the cache path was unpickled,
    i.e. anyone able to write to ./config could execute code."""
    g, p = _group()
    cache = tmp_path / "dlog_3.json"
    marker = tmp_path / "payload-ran"
    with open(tmp_path / "dlog_3.pkl", "wb") as f:
        pickle.dump(_Payload(marker), f)

    table, bound, m, giant = load_or_build_dlog_table(str(cache), g, p, 1000)
    assert not marker.exists()
    assert dlog_table_solve(gp.powmod(gp.mpz(g), -42, gp.mpz(p)), g, p, bound, table, m, giant) == -42


def test_no_cache_file_is_written(tmp_path):
    g, p = _group()
    cache = tmp_path / "sub" / "dlog_3.json"
    table, bound, m, giant = load_or_build_dlog_table(str(cache), g, p, 1000)
    assert not (tmp_path / "sub").exists()
    assert dlog_table_solve(gp.powmod(gp.mpz(g), 999, gp.mpz(p)), g, p, bound, table, m, giant) == 999


def test_planted_json_cache_is_ignored(tmp_path):
    """A JSON cache with a wrong table used to be trusted as long as its
    g / p / bound fields matched, silently corrupting decryption results."""
    g, p = _group()
    m = 46
    gg, pp = gp.mpz(g), gp.mpz(p)
    bogus = {gp.digits(gp.powmod(gg, j, pp)): (j + 1) % m for j in range(m)}
    cache = tmp_path / "dlog_3.json"
    cache.write_text(json.dumps({
        "g": g, "p": p, "func_bound": 1000, "strategy": "dlog_table",
        "step_size": m, "giant_step": gp.digits(gp.powmod(gg, -m, pp)),
        "dlog_table": bogus,
    }))

    table, bound, m, giant = load_or_build_dlog_table(str(cache), g, p, 1000)
    for x in range(-50, 51):
        assert dlog_table_solve(gp.powmod(gg, x, pp), g, p, bound, table, m, giant) == x
