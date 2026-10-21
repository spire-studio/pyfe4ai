"""Custom exception hierarchy for PyFE4AI.

All scheme-level exceptions inherit from :class:`FEError`, allowing consumers
to catch broad (``except FEError``) or narrow (``except FEValidationError``).
"""

import os
import sys
import warnings


class FEError(Exception):
    """Base exception for all PyFE4AI errors."""


class FEConfigError(FEError):
    """Configuration or parameter file error (missing, invalid, corrupt)."""


class FEKeyError(FEError):
    """Missing or invalid cryptographic keys / credentials."""


class FEValidationError(FEError, ValueError):
    """Input validation failure (bound exceeded, shape mismatch, etc.)."""


class FESchemeError(FEError, RuntimeError):
    """Internal scheme logic error (unexpected state, unsupported operation)."""


class FESecurityWarning(UserWarning):
    """A scheme with a known security defect is being used.

    See :mod:`pyfe4ai.core.status` and the README *Scheme Status* table.
    """


class FEWarning(UserWarning):
    """A condition an application should see that does not stop the call.

    For example a cached parameter record that no longer matches its
    configuration and is regenerated (new parameters and seeds), a request
    answered with ``None`` (an unknown id, a decryption out of range) or a
    slower fallback. The library also logs
    them (``logging.getLogger("pyfe4ai")`` has only a ``NullHandler``, so
    without a logging configuration this warning is what an application
    sees). Not a subclass of :class:`FESecurityWarning`: filtering one leaves
    the other.

    It is attributed to the application's call (the first caller outside
    pyfe4ai, see :func:`warn_at_caller`), so Python's default filter shows it
    once per call site of the application, not once per library line. Its text
    holds no caller-supplied values (an unknown id is in the log record only).
    """


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_package(filename: str) -> bool:
    return os.path.abspath(filename).startswith(_PACKAGE_DIR + os.sep)


def warn_at_caller(message: str, category: type = FEWarning) -> None:
    """Issue *message* as a *category* warning attributed to the first frame
    outside the pyfe4ai package -- the application's call, however deep the
    library's own calls are -- or, if every frame is inside it, to the library
    function that issues it. Used by the library for :class:`FEWarning`."""
    frame, level = sys._getframe(1), 2
    while frame is not None and _in_package(frame.f_code.co_filename):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level if frame is not None else 2)
