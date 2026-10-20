"""Precision of the native FACCT tables (October 2026 code review).

``native_backend._facct_table`` computed ``(-params.neg_c).as_mantissa_exp()``
in gmpy2's default 53-bit context, so ``c = 1 / (2 sigma^2)`` reached the Rust
acceptance test with 53 significant bits. The fixed-point threshold was then
off by up to about 2^-55, far outside the 2^-100 band that is handed to the
exact Python formula: the native samples were not the reference's for a given
bit source (documented: "for a given bit source the samples are the
reference's", "an error below 2^-110"), and the native output was about 2^-57
from D_{Z,sigma} per sample instead of about 2^-185. The mantissa is now
negated exactly, and tables are checked against the exact formula when built.
"""

import os
import subprocess
import sys

import gmpy2 as gp
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.primitives import sampling as su

pytestmark = pytest.mark.skipif(native_backend.ext() is None or not hasattr(native_backend.ext(), "FacctTable"),
                                reason="native FACCT not built")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


@pytest.mark.parametrize("sigma", [3.2, 3.19, 6.4, 17.3, 437881.7381962938])
def test_acceptance_constant_has_128_correct_bits(monkeypatch, sigma):
    captured = {}

    def table(k, y_bits, cdt, c_mant, c_exp, exact):
        captured.update(c_mant=c_mant, c_exp=c_exp)
        return object()

    monkeypatch.setattr(native_backend._ext, "FacctTable", table, raising=False)
    native_backend._facct_table.cache_clear()
    try:
        native_backend._facct_table(su._exact(sigma))
    finally:
        native_backend._facct_table.cache_clear()
    params = su._facct_params(su._exact(sigma))
    with gp.context(precision=1000):
        exact = -params.neg_c
        passed = gp.mpfr(captured["c_mant"]) * gp.mpfr(2) ** captured["c_exp"]
        rel = abs(passed - exact) / exact
        bits = -gp.log2(rel) if rel else 1000
    assert bits >= 120, "c = 1/(2 sigma^2) passed with only {:.1f} correct bits".format(float(bits))


def test_crafted_bits_give_the_reference_sample(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    res = subprocess.run([sys.executable, os.path.join(HERE, "poc", "poc_facct_divergence.py")], cwd=tmp_path,
                         capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "crafted first trial:     python == native: True" in res.stdout, res.stdout
