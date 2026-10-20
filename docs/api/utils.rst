Utilities
=========

Native Backend
--------------

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

   from pyfe4ai.utils import native_backend

   native_backend.describe()             # build and selection, for benchmark records
   native_backend.set_backend("python")  # the whole process: "auto" (default), "python", "native"
   with native_backend.backend("native"):
       ...                               # this thread, inside the block (ImportError if not built)
   native_backend.set_threads(4)         # threads per native call (default 1)

.. automodule:: pyfe4ai.utils.native_backend
   :members:

Exceptions
----------

.. automodule:: pyfe4ai.utils.exceptions
   :members:
   :undoc-members:

Crypto Utilities
----------------

.. automodule:: pyfe4ai.utils.crypto_utils
   :members:
   :undoc-members:

Crypto Constants
----------------

.. automodule:: pyfe4ai.utils.crypto_constants
   :members:
   :undoc-members:

Sampling Utilities
------------------

.. automodule:: pyfe4ai.utils.sampling_utils
   :members:
   :undoc-members:

Matrix Utilities
----------------

.. automodule:: pyfe4ai.utils.matrix_utils
   :members:
   :undoc-members:

Modular Arithmetic
------------------

.. automodule:: pyfe4ai.utils.modular_utils
   :members:
   :undoc-members:

Input Validation
----------------

.. automodule:: pyfe4ai.utils.validation
   :members:
   :undoc-members:

LWE Utilities
-------------

.. automodule:: pyfe4ai.utils.lwe_utils
   :members:
   :undoc-members:

Seeded LWE Matrices
-------------------

.. automodule:: pyfe4ai.utils.lwe_matrix
   :members:
   :undoc-members:

Ring-LWE Utilities
------------------

.. automodule:: pyfe4ai.utils.ring_lwe_utils
   :members:
   :undoc-members:

Discrete-Log Solver
-------------------

.. automodule:: pyfe4ai.utils.dlog_solver
   :members:
   :undoc-members:

ML Adapter
----------

For ``mcfe/ring_lwe_threshold`` the wrapper also runs the decryption servers
(one trust domain, for experiments). :func:`~pyfe4ai.utils.ml_adapter.prepare_aggregation`
and :func:`~pyfe4ai.utils.ml_adapter.aggregate_gradients` take ``servers=``
(default: the first ``t`` of the public parameters' ``lst_sid``) and give each
enrolled server its label key share for the round and, once per weights, its
weight key share. The messages of a deployment are those of the separate
steps: :func:`~pyfe4ai.utils.ml_adapter.partial_decrypt_gradients` at each
server and :func:`~pyfe4ai.utils.ml_adapter.combine_partial_decryptions` at
the combiner, exchanging :class:`~pyfe4ai.utils.ml_adapter.ThresholdPartial`
objects:

.. code-block:: python

   prepared = prepare_aggregation(wrapper, weights, shape=(3, 7), label="round-a",
                                  servers=["sid_1", "sid_2"])
   received = {nid: EncryptedGradient.from_dict(d) for nid, d in client_messages.items()}
   wire = {sid: partial_decrypt_gradients(received, wrapper, prepared, server=sid).to_dict()
           for sid in prepared.servers}
   total = combine_partial_decryptions(
       {sid: ThresholdPartial.from_dict(d) for sid, d in wire.items()}, wrapper, prepared)

Every partial decryption carries the ``sharing_id`` of the server's key
share. Keep it when partials (or key shares) are stored or transmitted by
other means: the combiner refuses partials of two sharings of one key
(:class:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.MixedSharingsError`; prepare
the round again), but partials without an id combine only with each other,
so dropping it on every side would let two sharings be mixed unnoticed.
``examples/mcfe_ring_lwe_threshold_demo.py`` runs both the scheme classes and
the adapter.

.. automodule:: pyfe4ai.utils.ml_adapter
   :members:
   :undoc-members:

Quantization
------------

.. automodule:: pyfe4ai.utils.quantization
   :members:
   :undoc-members:

Pairing Backend
---------------

.. automodule:: pyfe4ai.utils.pairing_backend
   :members:
   :undoc-members:

Pairing Utilities
-----------------

.. automodule:: pyfe4ai.utils.pairing_utils
   :members:
   :undoc-members:

BLS12-381
---------

.. automodule:: pyfe4ai.utils.bls12_381
   :members:

Elliptic-Curve Group (ristretto255)
-----------------------------------

.. automodule:: pyfe4ai.utils.ec_group
   :members:

.. automodule:: pyfe4ai.utils.ristretto255
   :members:
