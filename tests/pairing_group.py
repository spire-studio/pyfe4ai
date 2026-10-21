"""The pairing group the pairing-scheme tests run on.

The default group, BLS12-381 from the Rust extension, unless the environment
variable ``PYFE4AI_TEST_PAIRING_GROUP`` names another one (e.g. ``MNT224``
to run the same tests on charm-crypto). Test modules skip when no backend
implements the group.
"""

import os

import pytest

from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, is_native_group, pairing_available

GROUP = os.environ.get("PYFE4AI_TEST_PAIRING_GROUP", DEFAULT_PAIRING_GROUP)


def skip_unless_available() -> None:
    """Skip the calling test module if no backend implements :data:`GROUP`."""
    if not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP), allow_module_level=True)


def needs_extension() -> bool:
    """Whether only the Rust extension implements :data:`GROUP` (BLS12-381): the
    pairing test modules are then ``native`` (see tests/conftest.py)."""
    return is_native_group(GROUP)


#: The marks of a test (or parameter set) that needs :data:`GROUP` and skips
#: itself without it: ``native`` when only the extension implements the group.
GROUP_MARKS = [pytest.mark.native] if needs_extension() else []


def needs_group(test):
    """Decorator: mark *test* with :data:`GROUP_MARKS`."""
    for mark in GROUP_MARKS:
        test = mark(test)
    return test
