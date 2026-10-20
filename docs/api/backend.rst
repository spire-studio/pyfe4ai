Native Backend
==============

The optional Rust extension ``pyfe4ai._rust_native`` (sources in ``rust/``)
runs the Ring-LWE, dense-LWE and threshold hot paths, ristretto255 and the
BLS12-381 pairing group. ``pip install`` builds it when ``cargo`` is on the
``PATH`` (in a source checkout also ``python tools/build_rust.py``;
``PYFE4AI_NO_NATIVE=1`` skips it). The Python code stays the reference and
the fallback: the native code computes the same values, and the tests check
it bit for bit. The backend is chosen per process, per block or with the
``PYFE4AI_BACKEND`` environment variable (``"auto"``, ``"python"`` or
``"native"``), and the threads of one native call with ``PYFE4AI_THREADS``:

.. code-block:: python

   from pyfe4ai import backend

   backend.describe()             # build and selection, for benchmark records
   backend.set_backend("python")  # the whole process: "auto" (default), "python", "native"
   with backend.backend("native"):
       ...                        # this thread, inside the block (ImportError if not built)
   backend.set_threads(4)         # threads per native call (default 1)

.. automodule:: pyfe4ai.backend
   :members:
