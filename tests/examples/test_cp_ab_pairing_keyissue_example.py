"""The example of the threshold key issuance of sife/cp_ab_pairing (0.7, experimental): it runs as a script in a
fresh interpreter, with a parameter cache of its own, exits with 0, writes nothing to stderr, and prints the lines
that show what it is for -- a key issued by two of three servers that decrypts, a request the approval rule refuses
and an altered partial key that the requester's check refuses.

The file name holds "pairing", so the test is ``extended``; it runs on BLS12-381, which the Rust extension provides.
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
    pytest.skip("the example runs on {}, which no backend implements".format(DEFAULT_PAIRING_GROUP),
                allow_module_level=True)

pytestmark = pytest.mark.native                      # the default group is BLS12-381, from the extension

ROOT = Path(__file__).resolve().parents[2]
NAME = "sife_cp_ab_pairing_keyissue_demo.py"
EXPECTED = [
    "key issued by ks1 and ks3: <x, y> = 9",
    "request for y = [1, 0, 0]: refused by the approval rule",
    "partial keys of ks2 (altered) and ks3: refused by the check against pp",
]


def test_the_example_runs_and_shows_what_it_is_for(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"))
    res = subprocess.run([sys.executable, str(ROOT / "examples" / NAME)], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path), timeout=600)
    assert res.returncode == 0, res.stderr
    assert res.stderr == ""
    lines = res.stdout.splitlines()
    for line in EXPECTED:
        assert line in lines


def test_the_example_is_listed_in_the_readme():
    readme = (ROOT / "examples" / "README.md").read_text("utf-8")
    assert "`{}`".format(NAME) in readme.split("## Pairing-Based Examples", 1)[1]
