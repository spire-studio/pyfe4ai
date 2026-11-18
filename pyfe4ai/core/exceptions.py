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

    See :mod:`pyfe4ai.core.status` and the README *Scheme Status* table. Like
    :class:`FEWarning`, it is attributed to the application's call (the first
    caller outside pyfe4ai, :func:`warn_at_caller`).
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
    holds no caller-supplied values but the name of a config option (an
    unknown id is in the log record only).
    """


class FEDeprecationWarning(FEWarning):
    """A name, an option or an option value of the stable tier is deprecated.

    It keeps working for the rest of 1.x and goes in 2.0 at the earliest (the
    compatibility policy). A subclass of :class:`FEWarning`, hence of
    ``UserWarning``: unlike ``DeprecationWarning``, Python shows it by default,
    once per call site of the application (:func:`warn_at_caller`); filter it
    by this class to silence deprecations only. Issued by
    :func:`warn_deprecated`.
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


def warn_deprecated(what: str, since: str, use: str | None = None) -> None:
    """Issue the :class:`FEDeprecationWarning` of *what* (a name, an option,
    an option value), deprecated since version *since*, with its replacement
    *use* if there is one: ``"<what> is deprecated since pyfe4ai <since> and
    will be removed in 2.0; use <use>."``"""
    message = "{} is deprecated since pyfe4ai {} and will be removed in 2.0".format(what, since)
    warn_at_caller(message + ("; use {}.".format(use) if use else "."), FEDeprecationWarning)
