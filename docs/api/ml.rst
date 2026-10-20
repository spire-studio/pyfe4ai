ML Adapter
==========

:mod:`pyfe4ai.ml` runs SIFE and MCFE schemes for encrypted linear inference
and federated gradient aggregation behind one wrapper:

.. code-block:: python

   from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

The functions and classes below are importable from :mod:`pyfe4ai.ml`; they
are defined in its submodules ``wrapper`` (the wrapper; its schemes are
those of :mod:`pyfe4ai.core.registry`), ``serialization`` (the encrypted containers), ``gradients``
(federated aggregation), ``threshold`` (the decryption servers of the
threshold MCFE) and ``inference`` (encrypted linear inference).

Threshold MCFE
--------------

For ``mcfe/ring_lwe_threshold`` the wrapper also runs the decryption servers
(one trust domain, for experiments). :func:`~pyfe4ai.ml.prepare_aggregation`
and :func:`~pyfe4ai.ml.aggregate_gradients` take ``servers=``
(default: the first ``t`` of the public parameters' ``lst_sid``) and give each
enrolled server its label key share for the round and, once per weights, its
weight key share. The messages of a deployment are those of the separate
steps: :func:`~pyfe4ai.ml.partial_decrypt_gradients` at each
server and :func:`~pyfe4ai.ml.combine_partial_decryptions` at
the combiner, exchanging :class:`~pyfe4ai.ml.ThresholdPartial`
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
(:class:`~pyfe4ai.core.versions.MixedSharingsError`; prepare
the round again), but partials without an id combine only with each other,
so dropping it on every side would let two sharings be mixed unnoticed.
``examples/mcfe_ring_lwe_threshold_demo.py`` runs both the scheme classes and
the adapter.

API
---

.. automodule:: pyfe4ai.ml
   :members:
   :undoc-members:

Quantization
------------

.. automodule:: pyfe4ai.ml.quantization
   :members:
   :undoc-members:
