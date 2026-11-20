"""The example of the identity directory pyfe4ai.ac (0.7, experimental): it runs as a script in a fresh interpreter,
with a parameter cache of its own, exits with 0, writes nothing to stderr, and prints the lines that show what it is
for -- the key of the ciphertext's identity decrypts, another identity's key is refused, and adding an identity
changes the directory's digest but not earlier ciphertexts and keys.

It runs on lattices only (sife/fullysec_ring_lwe), on the native or the Python backend.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "ac_identity_directory_minimal.py"
EXPECTED = [
    "identities: ['alice', 'bob']",
    "key of alice: <x, y> = 9",
    "key of bob: refused (issued to another identity)",
    "directory with carol, checked against the old digest: refused",
    "identities: ['alice', 'bob', 'carol']",
    "earlier ciphertext, key of alice: <x, y> = 9",
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
    assert "`{}`".format(NAME) in readme.split("## LWE Examples", 1)[1].split("## ", 1)[0]
