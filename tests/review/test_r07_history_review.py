"""R-0.7 round 3 (3120d70 + 99b6c59): H-1, the suite in its default order.

test_an_exception_of_the_rule_refuses_and_leaves_no_record (tests/sife/test_cp_ab_pairing_keyissue.py) is
parametrized with exception *instances* (ValueError("no"), KeyError("y")) that live as long as the module. Raised by
the approval rule, each keeps its __traceback__, so the frames of issue() and of the test, and with them the five
key servers of that case, stay alive. Since 99b6c59 every key server is tracked by pyfe4ai.core.forksafe, and
tests/state/test_forksafe.py::test_registered_functions_and_tracked_objects_run_in_the_child_only asserts that its
own object is the only one tracked: 11 == 1 when tests/sife runs first, as in the full suite."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
def test_h1_the_forksafe_test_passes_after_the_key_issuance_tests(tmp_path):
    env = dict(os.environ, PYFE4AI_CACHE_DIR=str(tmp_path))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/sife/test_cp_ab_pairing_keyissue.py::test_an_exception_of_the_rule_refuses_and_leaves_no_record",
         "tests/state/test_forksafe.py::test_registered_functions_and_tracked_objects_run_in_the_child_only"],
        capture_output=True, text=True, env=env, timeout=300)
    assert run.returncode == 0, run.stdout[-1500:]
