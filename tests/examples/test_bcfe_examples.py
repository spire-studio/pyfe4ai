"""The examples of the bounded-key scheme ``bcfe/gc_ristretto255`` (0.6): each runs as a script in a fresh
interpreter, with a parameter cache of its own, exits with 0 without a warning and prints exactly the lines that show
what it is for -- the credit example the decisions only, the holder-generated example the refusal of a forged grant
by CheckAuth and the decrypted score.

They need no pairing backend and run on the backend of the suite's lane (``PYFE4AI_BACKEND`` is inherited): the
native one when the extension is built, else the Python reference, which takes about a second for these circuits.
The authority of each example lives in a temporary directory of its own, not in the parameter cache. Where the
extension is not built (``pip install`` without cargo), the example gives the library's one missing-extension FEWarning,
as any first use of the library does; that notice is the only stderr allowed then.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyfe4ai import backend
from pyfe4ai.backend import MISSING_EXTENSION_NOTICE

ROOT = Path(__file__).resolve().parents[2]

#: The example and its whole output.
EXAMPLES = {
    "bcfe_credit_scoring.py": [
        "applicant A: approve",
        "applicant B: decline",
    ],
    "bcfe_holder_generated_keys.py": [
        "CheckAuth: refused a grant whose other positions are not derived from its k",
        "risk score: 16",
    ],
}


def _without_the_missing_extension_notice(stderr: str) -> str:
    """*stderr* without the missing-extension notice (its warning line and the source line after it), which only a
    process without the extension gives; with the extension built nothing is removed."""
    if backend.available():
        return stderr
    out, skip = [], False
    for line in stderr.splitlines(keepends=True):
        if MISSING_EXTENSION_NOTICE in line:
            skip = True
            continue
        if skip and line.startswith(" "):
            skip = False
            continue
        skip = False
        out.append(line)
    return "".join(out)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_example_runs_and_shows_what_it_is_for(name, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"))
    res = subprocess.run([sys.executable, str(ROOT / "examples" / name)], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path), timeout=600)
    assert res.returncode == 0, res.stderr
    # no other warning: an example is what a new user runs first
    assert _without_the_missing_extension_notice(res.stderr) == ""
    assert res.stdout.splitlines() == EXAMPLES[name]
    # the example's authority lives in a temporary directory, not in the user's parameter cache
    assert not (tmp_path / "cache" / "authority").exists()


def test_the_examples_are_listed_in_the_readme():
    readme = (ROOT / "examples" / "README.md").read_text("utf-8")
    section = readme.split("## Bounded-Key FE", 1)[1].split("\n## ", 1)[0]
    for name in EXAMPLES:
        assert "`{}`".format(name) in section
