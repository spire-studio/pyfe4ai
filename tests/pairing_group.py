"""The pairing group the pairing-scheme tests run on.

The default group, BLS12-381 from the Rust extension, unless the environment
variable ``PYFE4AI_TEST_PAIRING_GROUP`` names another one (e.g. ``MNT224``
to run the same tests on charm-crypto). Test modules skip when no backend
implements the group.
"""

import os

import pytest

from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, pairing_available

GROUP = os.environ.get("PYFE4AI_TEST_PAIRING_GROUP", DEFAULT_PAIRING_GROUP)


def skip_unless_available() -> None:
    """Skip the calling test module if no backend implements :data:`GROUP`."""
    if not pairing_available(GROUP):
        pytest.skip("no backend for pairing group {}".format(GROUP), allow_module_level=True)
