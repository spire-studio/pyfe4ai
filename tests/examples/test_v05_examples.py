"""The examples of the 0.5 schemes (attribute-based IPFE and MCFE, and the policy-gated ML adapter): each runs as a
script in a fresh interpreter, with a parameter cache of its own, exits with 0 and prints the lines that show what it
is for -- the result of a key whose attributes satisfy the policy, and the refusal of one whose attributes do not.

The examples run on the default pairing group, BLS12-381, which the Rust extension provides.
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
    pytest.skip("the examples run on {}, which no backend implements".format(DEFAULT_PAIRING_GROUP),
                allow_module_level=True)

pytestmark = pytest.mark.native                      # the default group is BLS12-381, from the extension

ROOT = Path(__file__).resolve().parents[2]
REFUSED = "refused (the attributes do not satisfy the policy)"

#: The example and the lines of its output that show it works.
EXAMPLES = {
    "sife_cp_ab_pairing_minimal.py": [
        "key for ['org:A', 'role:regulator']: <x, y> = 9",
        "key for ['org:A']: " + REFUSED,
        "key for ['id:alice']: <x, y> = 9",
        "key for ['id:bob']: " + REFUSED,
    ],
    "sife_kp_ab_pairing_minimal.py": [
        "key for and(purpose:research, region:EU): <x, y> = 9",
        "key for and(purpose:research, region:US): " + REFUSED,
    ],
    "mcfe_kp_ab_pairing_minimal.py": [
        "key for and(purpose:research, region:EU): sum_i x_i y_i = -21",
        "key for and(purpose:research, region:US): " + REFUSED,
    ],
    "ml_fl_aggregation_policy.py": [
        "policy and(purpose:research, region:EU): FE result [22. 28.]",
        "policy and(purpose:research, region:US): " + REFUSED,
    ],
}


#: The examples of a known-insecure scheme: they print its ``FESecurityWarning`` (2026-11, see the scheme's issue).
KNOWN_INSECURE = {"mcfe_kp_ab_pairing_minimal.py", "ml_fl_aggregation_policy.py"}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_example_runs_and_shows_what_it_is_for(name, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"))
    res = subprocess.run([sys.executable, str(ROOT / "examples" / name)], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path), timeout=600)
    assert res.returncode == 0, res.stderr
    # no warning, an example is what a new user runs first; except the one that a known-insecure scheme must give
    expected = [line for line in res.stderr.splitlines() if "FESecurityWarning" in line]
    if name in KNOWN_INSECURE:
        assert expected and all("mcfe/kp_ab_pairing is known to be insecure" in line for line in expected)
        assert "FESecurityWarning" in res.stderr and all(
            line.startswith((" ", str(ROOT))) or "FESecurityWarning" in line for line in res.stderr.splitlines())
    else:
        assert res.stderr == ""
    lines = res.stdout.splitlines()
    for line in EXAMPLES[name]:
        assert line in lines


def ml_page_block(section: str, page: str | None = None) -> str:
    """The first code block of *section* (a heading) of ``docs/api/ml.rst``, or of *page*, dedented."""
    if page is None:
        page = (ROOT / "docs" / "api" / "ml.rst").read_text("utf-8")
    lines = page.split("{}\n{}\n".format(section, "-" * len(section)), 1)[1].splitlines()
    start = lines.index(".. code-block:: python") + 2
    block = []
    for line in lines[start:]:
        if line and not line.startswith("   "):
            break
        block.append(line[3:])
    return "\n".join(block).strip() + "\n"


def run_block(block: str, check: str, tmp_path) -> subprocess.CompletedProcess:
    """*block* and then *check* as one script in a fresh interpreter, with a parameter cache of its own."""
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"))
    return subprocess.run([sys.executable, "-c", block + check], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path), timeout=600)


# a reader copies a block alone, so it brings its imports and data; it runs and gives the weighted sum


def test_the_policy_gated_block_of_the_ml_page_runs_as_written(tmp_path):
    block = ml_page_block("Policy-gated aggregation")
    assert '"kp_ab_pairing"' in block and '{"c0": 1, "c1": 2, "c2": 3}' in block
    check = "\nimport numpy\nassert numpy.allclose(total, grad['c0'] + 2 * grad['c1'] + 3 * grad['c2']), total\n"
    res = run_block(block, check, tmp_path)
    assert res.returncode == 0, res.stderr


def test_the_threshold_block_of_the_ml_page_runs_as_written(tmp_path):
    block = ml_page_block("Threshold MCFE")
    assert '"ring_lwe_threshold"' in block and '{"nid_0": 1, "nid_1": 2, "nid_2": 1}' in block
    check = ("\nimport numpy\n"
             "assert numpy.allclose(total, grad['nid_0'] + 2 * grad['nid_1'] + grad['nid_2']), total\n")
    res = run_block(block, check, tmp_path)
    assert res.returncode == 0, res.stderr


def test_the_examples_are_listed_in_the_readme():
    readme = (ROOT / "examples" / "README.md").read_text("utf-8")
    pairing = readme.split("## Pairing-Based Examples", 1)[1]
    for name in EXAMPLES:
        assert "`{}`".format(name) in pairing
