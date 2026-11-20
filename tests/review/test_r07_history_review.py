"""R-0.7 round 3 (3120d70 + 99b6c59): H-1, key servers kept alive by the exception test.

test_an_exception_of_the_rule_refuses_and_leaves_no_record (tests/sife/test_cp_ab_pairing_keyissue.py) was
parametrized with exception *instances* (ValueError("no"), KeyError("y")) that live as long as the module. Raised by
the approval rule, each kept its __traceback__, so the frames of issue() and of the test, and with them the five key
servers of that case, stayed alive. Since 99b6c59 every key server is tracked by pyfe4ai.core.forksafe, and the
then absolute count of tests/state/test_forksafe.py failed (11 == 1). b0f1458 raises fresh instances.

Since d84c597 that forksafe test counts only its own objects, so it no longer catches a regression (the re-check's
RC-1). This test checks the cause directly: in a fresh interpreter, after the exception test has run and a garbage
collection, no key server is left in forksafe's registry. With b0f1458 reverted it finds the key servers of the
cases that raise a kept instance."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyfe4ai.primitives.pairing.pairing_backend import pairing_available
from tests.pairing_group import GROUP, needs_group

ROOT = Path(__file__).resolve().parents[2]
NODE = "tests/sife/test_cp_ab_pairing_keyissue.py::test_an_exception_of_the_rule_refuses_and_leaves_no_record"

#: Run the exception test with pytest in this interpreter, then count the key servers still tracked for fork.
PROBE = """
import gc, sys
import pytest
code = pytest.main(["-q", "-p", "no:cacheprovider", sys.argv[1]])
from pyfe4ai.core import forksafe
from pyfe4ai.schemes.sife.cp_ab_pairing_keyissue import SIFECPABPairingKeyServer
gc.collect()
left = sum(isinstance(obj, SIFECPABPairingKeyServer) for obj in list(forksafe._TRACKED))
print("exit", int(code), "key servers left", left)
"""


# The key-issuance module skips itself without a backend for GROUP (e.g. the core lane, PYFE4AI_NO_NATIVE=1): the
# probe would then run no test, so this test needs the group too.
@needs_group
@pytest.mark.skipif(not pairing_available(GROUP), reason="no backend for pairing group {}".format(GROUP))
@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
def test_h1_no_key_server_stays_tracked_after_the_exception_test(tmp_path):
    env = dict(os.environ, PYFE4AI_CACHE_DIR=str(tmp_path))
    run = subprocess.run([sys.executable, "-c", PROBE, NODE], capture_output=True, text=True, env=env, cwd=str(ROOT),
                         timeout=300)
    assert run.returncode == 0, run.stderr[-1500:]
    assert run.stdout.splitlines()[-1] == "exit 0 key servers left 0", run.stdout[-1500:]
