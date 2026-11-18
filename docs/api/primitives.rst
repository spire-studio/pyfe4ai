Primitives
==========

.. automodule:: pyfe4ai.primitives
   :no-members:

Lattices
--------

.. automodule:: pyfe4ai.primitives.lattice
   :no-members:

LWE Parameters
^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.lattice.lwe_utils
   :members:
   :undoc-members:

Seeded LWE Matrices
^^^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.lattice.lwe_matrix
   :members:
   :undoc-members:

Ring-LWE
^^^^^^^^

.. automodule:: pyfe4ai.primitives.lattice.ring_lwe_utils
   :members:
   :undoc-members:

Matrices and Digits mod q
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.lattice.matrix_utils
   :members:
   :undoc-members:

Gaussian Sampling
-----------------

.. automodule:: pyfe4ai.primitives.sampling
   :members:
   :undoc-members:

Hashing
-------

.. automodule:: pyfe4ai.primitives.hashing
   :members:
   :undoc-members:

Groups
------

.. automodule:: pyfe4ai.primitives.groups
   :no-members:

Integer Groups
^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.groups.integer_groups
   :members:
   :undoc-members:

Modular Arithmetic
^^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.groups.modular_utils
   :members:
   :undoc-members:

Discrete-Log Solver
^^^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.groups.dlog_solver
   :members:
   :undoc-members:

Elliptic-Curve Group (ristretto255)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.groups.ec_group
   :members:

.. automodule:: pyfe4ai.primitives.groups.ristretto255
   :members:

Pairings
--------

.. automodule:: pyfe4ai.primitives.pairing
   :no-members:

Pairing Backend
^^^^^^^^^^^^^^^

The attribute-based schemes of NPP22 use two pieces of the backend beyond the
group operations. ``prepare_g2`` and ``multi_pair_prepared`` take one product
of pairings per row of G1 elements over the same G2 points, which the points
of a functional key are for every element of an array: on BLS12-381 the lines
of the Miller loops are computed once, scaled to a first coefficient of 1
where no line has 0 there, and each row is one Miller loop and one final
exponentiation, on ``get_threads()`` threads (see :doc:`/threading`).
``hash_to_g1`` hashes a message to G1 under a domain separation tag: RFC 9380
on BLS12-381, so that the known answers of the RFC reproduce, and on a charm
group, which has no such tag, the tag, a bar and the message as one string.
:func:`~pyfe4ai.primitives.pairing.dpvs.encode` in the dual pairing vector
spaces below builds a basis combination from the exponents with one batch of
powers.

.. automodule:: pyfe4ai.primitives.pairing.pairing_backend
   :members:
   :undoc-members:

Pairing Utilities
^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.pairing.pairing_utils
   :members:
   :undoc-members:

Dual Pairing Vector Spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.primitives.pairing.dpvs
   :members:
   :undoc-members:

BLS12-381
^^^^^^^^^

.. automodule:: pyfe4ai.primitives.pairing.bls12_381
   :members:

Bounded-Key FE
--------------

The building blocks of the bounded-key schemes (:doc:`bcfe`), as reference
algorithms. The backend operations (:mod:`pyfe4ai.backend.interface`) give
the same bytes in both implementations; the schemes call the slot operations
of the second module, which garble and encrypt in one call.

Garbled Circuits
^^^^^^^^^^^^^^^^

The half-gates garbling and its evaluation, on the canonical circuit
encoding of :mod:`pyfe4ai.circuits` (backend operations ``gc_garble`` and
``gc_evaluate``).

.. automodule:: pyfe4ai.primitives.garbling
   :members:

Slot Keys and the Label KEM
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The public keys of a slot, their hash-derived unselected positions and the
hashed ElGamal encryption of the wire labels (backend operations
``bcfe_new_slot``, ``bcfe_place_slot``, ``bcfe_unselected_ok``,
``bcfe_encrypt_slot`` and ``bcfe_decrypt_slot``). The operations that need
randomness take it from the operating system on every call, and the native
implementation wipes ``Delta``, the seeds, ``r`` and the labels; this Python
reference makes no constant-time claim and cannot wipe its copies.

.. automodule:: pyfe4ai.primitives.bcfe_kem
   :members:
