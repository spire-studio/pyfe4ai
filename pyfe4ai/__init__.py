"""pyfe4ai: Python Functional Encryption for AI Security and Privacy."""

import logging

__version__ = "0.3.0"

# Convenience re-exports so users can write: from pyfe4ai import SIFE
from pyfe4ai.schemes.sife import SIFE, SIFEKeyGenerator  # noqa: F401
from pyfe4ai.schemes.mife import MIFE, MIFEKeyGenerator  # noqa: F401
from pyfe4ai.schemes.mcfe import MCFE, MCFEKeyGenerator  # noqa: F401
from pyfe4ai.core.status import SchemeStatus, get_scheme_status  # noqa: F401
from pyfe4ai.core.exceptions import FESecurityWarning  # noqa: F401

__all__ = [
    "__version__",
    "SIFE",
    "SIFEKeyGenerator",
    "MIFE",
    "MIFEKeyGenerator",
    "MCFE",
    "MCFEKeyGenerator",
    "SchemeStatus",
    "get_scheme_status",
    "FESecurityWarning",
]

# The modules log to logging.getLogger(__name__); the library configures no
# handlers or log files of its own (that is the application's choice).
logging.getLogger(__name__).addHandler(logging.NullHandler())
