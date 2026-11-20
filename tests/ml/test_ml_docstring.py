"""The examples of the :mod:`pyfe4ai.ml` module docstring pass as doctests.

Each section of the docstring (federated aggregation, encrypted inference, policy-gated aggregation) brings its imports
and data, so a reader can also copy one alone. The doctests run in a fresh interpreter with a parameter cache of their
own. The policy-gated section runs on the default pairing group, BLS12-381, which the Rust extension provides.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pairing_group import skip_unless_available

skip_unless_available()

from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, pairing_available  # noqa: E402

if not pairing_available(DEFAULT_PAIRING_GROUP):       # the tests' group may be another one (PYFE4AI_TEST_PAIRING_GROUP)
    pytest.skip("the policy-gated example runs on {}, which no backend implements".format(DEFAULT_PAIRING_GROUP),
                allow_module_level=True)

pytestmark = pytest.mark.native                      # the default group is BLS12-381, from the extension

ROOT = Path(__file__).resolve().parents[2]

RUN = """
import doctest, sys
import pyfe4ai.ml
result = doctest.testmod(pyfe4ai.ml)
print(result.attempted, result.failed)
sys.exit(1 if result.failed else 0)
"""


def test_the_examples_of_the_module_docstring_pass_as_doctests(tmp_path):
    import pyfe4ai.ml

    examples = pyfe4ai.ml.__doc__.count(">>> ")
    assert examples >= 25                               # three sections with their imports, data and a checked result
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"))
    res = subprocess.run([sys.executable, "-c", RUN], capture_output=True, text=True, env=env, cwd=str(tmp_path),
                         timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.split() == [str(examples), "0"], res.stdout
