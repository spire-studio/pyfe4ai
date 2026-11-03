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
weight key share. A label is tied to the weights of one call; the section
"Policy-gated aggregation" below describes the tie, and what happens to the
key shares of the call that loses a race for a label. The messages of a
deployment are those of the separate
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

``mcfe/kp_ab_pairing_per_client`` (NPP22 Sec. 5.4, known-insecure), where each
client encrypts for a set of attributes of its own, has the capability too, and
the adapter refuses it (``NotImplementedError``, at the first line of the
aggregation entry points ``encrypt_gradient``, ``precompute_gradient``,
``prepare_aggregation`` and ``aggregate_gradients``): the clients of one label
may hold different sets, which the adapter does not aggregate. The scheme
warns when it is built (``FESecurityWarning``; see :doc:`/security`). Use the scheme classes
(:mod:`pyfe4ai.schemes.mcfe.kp_ab_pairing_per_client`) directly.

What the checks cannot see. The ``label`` and the attributes of a ciphertext are
claims of its client: the points they were hashed to are inside ``t`` and
``c_ipfe``, and a decryption does not hash them again. The encryption key of a
client can check the claims of its own ciphertext (``t`` divided by the points
of the claimed tag and set, taken to the powers ``s_i`` and ``u_i`` of the key,
is a small power of ``g1`` only if the claims are right; the check also
recovers the plaintext), and the master key can check all of them. An
aggregator that holds neither cannot. A client whose points differ from its
claim gives no value (a
:class:`~pyfe4ai.core.exceptions.FESchemeError` at the discrete logarithm,
after the key is issued), and ciphertexts that disagree on the label are
refused; but whoever holds the ciphertexts of a whole round (an aggregator, a
store, a relay) can write another label into all of them, with no key and no
client taking part, and they are aggregated, silently, under that label: what
decrypts is the value of the round in which the ciphertexts were made. This is
a matter of integrity, not of confidentiality. Attributes that all clients
change alike give ``FEKeyError`` or no value, never a wrong one (an attribute
that no row of the policy reads can be renamed without effect, and the value is
the right one, if it keeps its place in the sorted list; a list out of order is
refused). A deployment that must know that ciphertexts belong to a round has
each client sign or MAC the label together with the record, and the decryptor
verifies it before it decrypts. A MAC needs a key that each client shares with
the decryptor and does not protect against the decryptor itself; a signature
does. An
item that has the
type tag of a group element and is none is found when it is decoded. These
are found, if at all, after the key is issued; the label is then tied to the
weights of that call, and a second try under it needs the same weights. A
request that the key generator refuses (weights beyond ``bound_y``, a policy
it does not take) issues no key and leaves the label free. Of two calls that
pass the check at once with other weights for one label, the first to tie it
keeps the label and the other is refused (``ValueError``) only after it has
obtained its key; in the threshold MCFE that the wrapper runs, the losing call
has already obtained key shares from the servers (its weight key share is added
to the wrapper's server objects where they lack it), and these shares never
reach the caller.

A functional key of ``mcfe/kp_ab_pairing`` is for a weight vector and a policy and
takes no tag, so a key issued or prepared for any round opens every other round
(as for every MCFE scheme whose functional keys are not bound to a label). The
rule that a label is aggregated with one weight vector only therefore
constrains the adapter's own calls, not the keys already issued. Take round 1
aggregated with the weights ``(1, 1, 1)`` and
:func:`~pyfe4ai.ml.prepare_aggregation` for round 2 with ``(1, 2, 1)``: the
adapter refuses to aggregate round 1 with ``(1, 2, 1)``, but the key of the
:class:`~pyfe4ai.ml.PreparedAggregation` (its ``dk``), handed to the decryptor,
opens round 1 too, and the difference of the two values is the gradient of the
second client in round 1. Two rounds and two weight vectors that differ in one
client's weight can thus isolate that client; the rule is no defence against
a holder of the keys, and which keys exist is the key holder's decision.

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
attributes the aggregation takes about 50 s on one thread and about 8 s on eight, and a
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
