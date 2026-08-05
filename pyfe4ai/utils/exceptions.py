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
