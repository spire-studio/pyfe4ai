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

Policy-gated aggregation
------------------------

``mcfe/kp_ab_pairing`` releases an aggregate only for a policy that the
clients' attributes satisfy. Each client encrypts its gradient for a set of
attributes (``encrypt_gradient(..., attributes=...)``, the same set for all
clients of a round, written as a set or a list in any order), and the key
that :func:`~pyfe4ai.ml.aggregate_gradients` (or
:func:`~pyfe4ai.ml.prepare_aggregation`) issues is for a policy, a
:class:`~pyfe4ai.policy.formula.Formula` or its text:

.. code-block:: python

   nids = ["c0", "c1", "c2"]
   wrapper = FESchemeWrapper("mcfe", "kp_ab_pairing",
                             {"n": 3, "lst_nid": nids, "bound_x": 1000, "bound_y": 10}, precision=1)
   attributes = {"purpose:research", "region:EU"}
   encrypted = {nid: encrypt_gradient(grad[nid], wrapper, nid=nid, label="round-1", attributes=attributes)
                for nid in nids}
   total = aggregate_gradients(encrypted, wrapper, {"c0": 1, "c1": 2, "c2": 3}, label="round-1",
                               policy="and(purpose:research, region:EU)")

The aggregation is refused before a key is issued and before any pairing is
computed when the scheme's own checks of the ciphertexts refuse them
(``ValueError``, naming the layer and the element, the client and the layer for
the shape of an array, or the clients and the counts for other numbers of
layers: records that are malformed or of other parameters,
ciphertexts that claim another label, arrays of other shapes or with other
numbers of layers, clients that encrypted for different sets of attributes),
when the set of attributes does not satisfy the policy
(:class:`~pyfe4ai.core.exceptions.FEKeyError`), when a client is missing, and
for ``allow_partial=True`` (``ValueError``: the shares of the clients' keys add
up to 1, so a result opens only with the ciphertexts of every enrolled client,
also those of weight 0).

``mcfe/kp_ab_pairing_per_client`` (NPP22 Sec. 5.4, experimental), where each
client encrypts for a set of attributes of its own, has the capability too, and
the adapter refuses it (``NotImplementedError``, at the first line of each
entry point): the clients of one label may hold different sets, which the
adapter does not aggregate. Use the scheme classes
(:mod:`pyfe4ai.schemes.mcfe.kp_ab_pairing_per_client`) directly.

What the checks cannot see. The ``label`` and the attributes of a ciphertext are
claims of its client: the points they were hashed to are inside ``t`` and
``c_ipfe``, and only the master key could check them. A client whose points
differ from its claim gives no value (a
:class:`~pyfe4ai.core.exceptions.FESchemeError` at the discrete logarithm,
after the key is issued), and clients that disagree on the label are refused; but if all
clients copy the label of a round onto ciphertexts of an earlier round, that
relabelled replay is aggregated, silently, to the sum of the round the
ciphertexts were made in. Attributes that all clients change alike give
``FEKeyError`` or no value, never a wrong one (a label that no row of the
policy reads can be renamed without effect, and the value is the right one). A caller who must know that
ciphertexts belong to a round has to authenticate them. An item that has the
type tag of a group element and is none is found when it is decoded. These
are found, if at all, after the key is issued; the label is then tied to the
weights of that call, and a second try under it needs the same weights.

A :class:`~pyfe4ai.ml.PreparedAggregation` keeps the canonical encoding of its
policy in ``policy``; the attributes are judged against it when the round is
aggregated, and a policy that differs from it is refused. Each client
encrypts once per label, as with the other MCFE schemes. The per-client
variant (the clients of one label may hold different sets of attributes) is
not supported by the adapter.

**What it is for.** Small vectors and statistics, up to about 10^3 elements
per aggregate, released under an access policy. It is not a way to aggregate
the gradients of a large model: every element is a ciphertext of its own, of
a few kilobytes, and the aggregator pairs several group elements per client
and element. With 10 clients, 10^3 elements and ciphertexts for three
attributes the aggregation takes about 52 s on one thread and about 9 s on eight, and a
client uploads about 2.4 KB per element as JSON. For comparison, the
Ring-LWE MCFE of the ``fl_aggregation_10k`` benchmark (10^4 elements,
``standard`` profile) takes the client 22.2 ms and the aggregator 10.4 ms, at
25.9 B per element in the compact encoding (:doc:`/choosing`). The adapter
warns (:class:`~pyfe4ai.core.exceptions.FEWarning`) of an aggregation of more
than 1000 elements, with the time it estimates, and of a discrete-logarithm
bound ``n * bound_x * bound_y`` above ``2**26``: ``bound_x`` bounds the
quantised entries, the gradient values times ``10 ** precision``.

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
