Installation
============

PyFE4AI is installed from a clone of its repository. It needs Python 3.10
or later, ``gmpy2`` and ``numpy``; an optional Rust extension makes the
lattice schemes much faster and provides the BLS12-381 pairing group.

From source
-----------

.. code-block:: bash

   git clone https://github.com/spire-studio/pyfe4ai.git
   cd pyfe4ai
   conda env create -f environment.yml    # Python and gmpy2 from conda-forge
   conda activate pyfe4ai
   pip install -e ".[test]"               # the package, with pytest

The dependencies are declared in ``pyproject.toml``; ``environment.yml``
only adds what conda provides better than pip (Python itself and ``gmpy2``
built against conda-forge's GMP and MPFR). Without conda, a virtual
environment with ``pip install -e ".[test]"`` works as well where ``gmpy2``
wheels exist for the platform.

The Rust extension
------------------

With a Rust toolchain (``cargo``) on the ``PATH``, ``pip install`` also
builds the native extension ``pyfe4ai._rust_native`` (sources in ``rust/``).
It runs the Ring-LWE, dense-LWE and threshold hot paths natively, provides
ristretto255 and the BLS12-381 pairing group that the pairing-based schemes
use by default, and computes the same values as the Python code, which
remains the reference and the fallback.

* The first build downloads the Rust crates of ``rust/native/Cargo.toml``
  from crates.io; later builds use cargo's cache.
* In a source checkout, ``python tools/build_rust.py`` builds (or rebuilds)
  the extension in place, ``--offline`` with cached crates only.
* ``PYFE4AI_NO_NATIVE=1 pip install -e .`` skips it.
* A failed build does not fail the installation: ``pip`` prints a warning
  and the package runs its Python implementation.

Check what you got:

.. code-block:: python

   import pyfe4ai

   info = pyfe4ai.backend_info()
   print(info["native_available"], info["native_version"], info["backend"])

``backend_info()`` reports the library version, whether the extension is
available (or why it could not be imported), its version and build, the
backend the calling thread uses, the threads per native call, the Gaussian
sampler and the parameter cache directory. Without the extension, the first
operation that falls back to Python issues one
:class:`~pyfe4ai.core.exceptions.FEWarning` per process saying so; the
Python implementation is about 5 to 35 times slower on the lattice schemes.
:doc:`environment` lists the environment variables that select the backend
and the threads.

Other pairing curves
--------------------

The pairing-based schemes (function-hiding IPE, the FH-MCFE family and
quadratic FE) run on BLS12-381 from the Rust extension. Other curves
(``pairing_group_param = "MNT224"``, ``"BN254"``, ...) need
`charm-crypto-framework <https://github.com/JHUISI/charm>`_, which is built
from source and needs GMP, OpenSSL and PBC 1.0.0 first:

.. code-block:: bash

   pip install -e ".[test,pairing]"

The repository's ``docker/Dockerfile.pairing`` builds an image with all of
them (``docker build -f docker/Dockerfile.pairing -t pyfe4ai-pairing .``).
Without the extension and without charm-crypto, the pairing-based schemes
are not available; every other scheme is.

Running the tests
-----------------

.. code-block:: bash

   python -m pytest -q             # all tests
   python -m pytest -m core -q     # everything that needs no pairing backend
   python -m pytest -m "not slow" -q

With the extension built, run the suite on both backends
(``PYFE4AI_BACKEND=native`` and ``PYFE4AI_BACKEND=python``): the native
code must match the Python reference bit for bit. ``CONTRIBUTING.md`` in the
repository describes the test groups and marks.

Building this documentation
---------------------------

.. code-block:: bash

   pip install -e ".[doc]"
   cd docs && make html            # _build/html/index.html
