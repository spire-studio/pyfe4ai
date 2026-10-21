"""pyfe4ai: Python Functional Encryption for AI Security and Privacy.

The package exports its entry points:

* :func:`~pyfe4ai.core.registry.create` builds the key generator of a
  scheme given by its id ``"family/variant"`` and returns it with the
  scheme's crypto class;
* :func:`~pyfe4ai.core.registry.list_schemes`,
  :func:`~pyfe4ai.core.registry.get_scheme` (a
  :class:`~pyfe4ai.core.registry.Scheme`: classes, kind, capabilities,
  review status), :data:`~pyfe4ai.core.registry.CAPABILITIES` and
  :func:`~pyfe4ai.core.status.get_scheme_status` query the scheme registry
  (:mod:`pyfe4ai.core.registry`);
* ``backend_info()`` (:func:`pyfe4ai.backend.backend_info`) reports the
  backend in use (native extension or Python), the extension's build, the
  threads, the Gaussian sampler and the parameter cache directory;
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
    # backend
    "backend_info",
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


def backend_info() -> dict:
    """The backend the calling thread uses and how it was chosen: whether the
    native extension is available (and its version and build), the threads
    per native call, the Gaussian sampler and the parameter cache directory.

    Returns :func:`pyfe4ai.backend.backend_info` (which documents the keys).
    Importing :mod:`pyfe4ai` does not load the backend or the extension; this
    call does.
    """
    from pyfe4ai import backend

    return backend.backend_info()


# The modules log to logging.getLogger(__name__); the library configures no
# handlers or log files of its own (that is the application's choice).
logging.getLogger(__name__).addHandler(logging.NullHandler())
