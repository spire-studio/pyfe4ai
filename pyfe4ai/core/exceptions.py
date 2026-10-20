"""Custom exception hierarchy for PyFE4AI.

All scheme-level exceptions inherit from :class:`FEError`, allowing consumers
to catch broad (``except FEError``) or narrow (``except FEValidationError``).
"""


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
    answered with ``None`` (an unknown id, a decryption out of range, partial
    decryptions that disagree) or a slower fallback. The library also logs
    them (``logging.getLogger("pyfe4ai")`` has only a ``NullHandler``, so
    without a logging configuration this warning is what an application
    sees). Not a subclass of :class:`FESecurityWarning`: filtering one leaves
    the other.
    """
