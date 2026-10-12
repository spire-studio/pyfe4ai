"""pyfe4ai: Python Functional Encryption for AI Security and Privacy."""

__version__ = "0.3.0.dev0"

# Convenience re-exports so users can write: from pyfe4ai import SIFE
from pyfe4ai.schemes.sife import SIFE, SIFEKeyGenerator  # noqa: F401
from pyfe4ai.schemes.mife import MIFE, MIFEKeyGenerator  # noqa: F401
from pyfe4ai.schemes.mcfe import MCFE, MCFEKeyGenerator  # noqa: F401
from pyfe4ai.schemes.status import SchemeStatus, get_scheme_status  # noqa: F401
from pyfe4ai.utils.exceptions import FESecurityWarning  # noqa: F401

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
