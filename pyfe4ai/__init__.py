"""pyfe4ai: Python Functional Encryption for AI Security and Privacy.

The package exports its entry points:

* :func:`create` builds the key generator of a scheme given by its id
  ``"family/variant"`` and returns it with the scheme's crypto class;
* :func:`list_schemes`, :func:`get_scheme` (a :class:`Scheme`: classes,
  kind, capabilities, review status), :data:`CAPABILITIES` and
  :func:`get_scheme_status` query the scheme registry
  (:mod:`pyfe4ai.core.registry`);
* the exceptions and warnings of the library.

The scheme classes themselves are exported by the family packages
:mod:`pyfe4ai.schemes.sife`, :mod:`~pyfe4ai.schemes.mife`,
:mod:`~pyfe4ai.schemes.mcfe` and :mod:`~pyfe4ai.schemes.quadratic`, the ML
adapter by :mod:`pyfe4ai.ml`.
"""

import logging

__version__ = "0.3.0"

from pyfe4ai.core.exceptions import FEConfigError  # noqa: E402
from pyfe4ai.core.exceptions import FEError  # noqa: E402
from pyfe4ai.core.exceptions import FEKeyError  # noqa: E402
from pyfe4ai.core.exceptions import FESchemeError  # noqa: E402
from pyfe4ai.core.exceptions import FESecurityWarning  # noqa: E402
from pyfe4ai.core.exceptions import FEValidationError  # noqa: E402
from pyfe4ai.core.exceptions import FEWarning  # noqa: E402
from pyfe4ai.core.registry import CAPABILITIES  # noqa: E402
from pyfe4ai.core.registry import CreatedScheme  # noqa: E402
from pyfe4ai.core.registry import Scheme  # noqa: E402
from pyfe4ai.core.registry import SchemeStatus  # noqa: E402
from pyfe4ai.core.registry import create  # noqa: E402
from pyfe4ai.core.registry import get_scheme  # noqa: E402
from pyfe4ai.core.registry import list_schemes  # noqa: E402
from pyfe4ai.core.status import get_scheme_status  # noqa: E402
from pyfe4ai.core.versions import MixedSharingsError  # noqa: E402

__all__ = [
    "__version__",
    # schemes
    "create",
    "CreatedScheme",
    "list_schemes",
    "get_scheme",
    "Scheme",
    "CAPABILITIES",
    "get_scheme_status",
    "SchemeStatus",
    # exceptions and warnings
    "FEError",
    "FEConfigError",
    "FEKeyError",
    "FEValidationError",
    "FESchemeError",
    "MixedSharingsError",
    "FESecurityWarning",
    "FEWarning",
]

# The modules log to logging.getLogger(__name__); the library configures no
# handlers or log files of its own (that is the application's choice).
logging.getLogger(__name__).addHandler(logging.NullHandler())
