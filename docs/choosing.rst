Choosing a scheme
=================

A scheme is named ``"family/variant"``: the family says who encrypts and
who decrypts, the variant which hardness assumption and construction it
uses. This page explains the families and variants, what the registry
records about each scheme, and which scheme fits which task. The registry
answers the same questions at runtime:

.. code-block:: python

   import pyfe4ai

   pyfe4ai.list_schemes("mcfe", capabilities="labels", status="reviewed")
   pyfe4ai.list_schemes(kind="ring_lwe")
   scheme = pyfe4ai.get_scheme("mcfe/ring_lwe_threshold")
   print(scheme.status.status, sorted(scheme.capabilities), scheme.refused or "accepted under standard")

Families
--------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Family
     - Who encrypts, who decrypts
     - Use it for
   * - ``sife`` -- single-input
     - one encryptor; a key holder learns ``<x, y>`` for the weights ``y``
       of its functional key
     - one party's vector scored by another party's linear model
   * - ``mife`` -- multi-input
     - ``n`` clients each encrypt a slot; the key holder learns
       ``sum_i <x_i, y_i>``. No labels: any ciphertext of a slot combines
       with any of the others under one key
     - one-shot joint inputs; **not** for repeated rounds such as gradients
       (see :doc:`security`)
   * - ``mcfe`` -- multi-client
     - ``n`` clients encrypt under a shared label; ciphertexts of one label
       combine, those of different labels do not
     - federated aggregation, one label per round
   * - ``mcfe/*_decentralized``
     - as MCFE, but functional keys need no authority: the clients derive
       shares of each functional key, and the decryptor combines them. The
       setup is still trusted: in this library one key generator samples
       every client's keys, including the pairwise zero-sum secrets, in
       process; a deployment without a trusted party needs a distributed
       setup (e.g. pairwise key exchange), which the library does not
       provide
     - aggregation without a key authority for functional keys (trusted
       setup)
   * - ``mife/*_threshold``, ``mcfe/*_threshold``
     - the functional key is shared among ``s`` decryption servers; any
       ``t`` of them decrypt together, fewer learn nothing beyond the result
       (honest-but-curious servers)
     - no single party able to decrypt; see :doc:`threshold_ml`
   * - ``*_fh_*`` -- function-hiding
     - functional keys also hide the weights ``y``
     - private models
   * - ``quadratic``
     - the key holder learns ``x^T F y`` (``sgp``, ``quad``), or the
       cross-user quadratic terms of several clients (``miqfe``)
     - quadratic scores, e.g. a bilinear model
   * - ``bcfe`` -- bounded keys (*experimental*)
     - one encryptor; a key holder learns ``C(w, x)`` for a public boolean
       circuit ``C`` (:mod:`pyfe4ai.circuits`) and the ``w`` of its key. At
       most ``max_keys`` keys in all, and a key opens only ciphertexts made
       after it was issued; a ciphertext grows with the keys it is
       encrypted to (:doc:`api/bcfe`)
     - a non-linear function of one party's data -- a linear decision or
       score, a small network -- for small models and few keys (a few to a
       few dozen), issued before the data is encrypted. The key holder
       learns its own ``w``, and the encryptor is trusted (:doc:`security`)

Assumptions
-----------

The variant names the hardness assumption; the registry's *kind* groups
the schemes by how the security profiles size them
(:data:`~pyfe4ai.core.registry.KINDS`).

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Kind
     - Variants
     - Properties
   * - ``dl``
     - ``ddh``, ``ddh_dynamic``, ``damgard`` and the threshold and
       decentralized DDH variants
     - 3072-bit groups under ``standard``; decryption solves a discrete log,
       so results must be small (``dlog_bound``)
   * - ``ec``
     - ``ddh_ristretto255``
     - DDH in the ristretto255 group (about 126-bit, fixed size); small
       ciphertexts and fast with the Rust extension; *unreviewed*
   * - ``dcr``
     - ``paillier``, ``paillier_decentralized``
     - Paillier moduli of 3072 bits under ``standard``, generated for every
       key generator; no discrete log at decryption
   * - ``lwe``
     - the MCFE ``lwe``, ``fullysec_lwe``, ``lwe_threshold``,
       ``lwe_decentralized``
     - dense LWE without a public matrix (ABMPSW20); a label is hashed per
       element, so it suits short vectors
   * - ``lwe_abdp15``
     - ``sife/lwe``, ``mife/lwe``, ``mife/lwe_threshold``
     - LWE with a seeded public matrix; large public keys and slow key
       generation under ``standard``; ``mife/lwe_threshold`` is refused
       there
   * - ``lwe_matrix``
     - ``sife/fullysec_lwe``, ``mife/fullysec_lwe`` (ALS16)
     - adaptive security, but refused under ``standard`` (no usable
       parameters; use ``sife/fullysec_ring_lwe``)
   * - ``ring_lwe``
     - ``ring_lwe`` in every family, its threshold and decentralized MCFE,
       ``sife/fullysec_ring_lwe``
     - one ciphertext carries many values (packed tensors), and the Rust
       extension makes it the fastest choice for large vectors and
       gradients; post-quantum assumption
   * - ``pairing``
     - the function-hiding and quadratic schemes and the attribute-based
       ones (``sife/cp_ab_pairing``, ``sife/kp_ab_pairing``,
       ``mcfe/kp_ab_pairing``, ``mcfe/kp_ab_pairing_per_client``)
     - BLS12-381, about 120 to 126-bit security (below the 128-bit target);
       needs the Rust extension (or charm-crypto for other curves)

Capabilities
------------

Each scheme lists its capabilities
(:data:`~pyfe4ai.core.registry.CAPABILITIES`):

* ``labels`` -- ciphertexts are bound to a label (the MCFE family); the
  scheme states whether its functional keys are too;
* ``threshold`` -- the functional key is shared among ``s`` servers, and
  ``t`` partial decryptions decrypt (``share_decrypt``,
  ``combine_decrypt``);
* ``decentralized`` -- the clients derive shares of each functional key
  (``derive_function_decryption_key_share``,
  ``combine_function_decryption_key_share``);
* ``pairing`` -- runs over a pairing group (``pairing_group_param``);
* ``ml_servers`` -- the ML adapter runs the decryption servers in-process
  (``mcfe/ring_lwe_threshold``);
* ``binary_ct`` -- binary ciphertexts (``ct_format="binary"``, about a third
  of the JSON size);
* ``rns`` -- an RNS modulus, a product of NTT primes
  (``ring_modulus="rns"``);
* ``access_control`` -- the ciphertext or the key carries an access policy
  or an attribute set (:mod:`pyfe4ai.policy`), and decryption succeeds only
  if the attributes satisfy the policy, within the security proof of the
  scheme. In ``sife/cp_ab_pairing`` the ciphertext carries the policy
  (``encrypt(..., policy=...)``) and the key the attributes; in the
  key-policy schemes (``sife/kp_ab_pairing``, ``mcfe/kp_ab_pairing``,
  ``mcfe/kp_ab_pairing_per_client``) the key carries the policy
  (``credentials={"fusion_weight": y, "policy": ...}``) and the ciphertext the
  attributes (``encrypt(..., attributes=...)``).
* ``bounded_keys`` -- the key generator is stateful: every distinct function
  it issues gets a slot of its own, up to ``max_keys`` slots in all, and a
  ciphertext is encrypted to the slots that exist when it is made, so a key
  decrypts only ciphertexts made after its slot (``bcfe/gc_ristretto255``).
* ``distributed_keygen`` -- functional keys are issued jointly by ``t`` of
  ``s`` key servers, none of which holds the master key after setup
  (``sife/cp_ab_pairing``, :mod:`pyfe4ai.schemes.sife.cp_ab_pairing_keyissue`;
  experimental, and secure only with authenticated, confidential channels
  between the requester and the servers, see :doc:`security`).

Review status
-------------

Every scheme carries a review status (the full table is on
:doc:`security`): *reviewed* -- checked against its paper, *faithful* or
*adapted* with the deviations listed; *experimental* -- the construction or
its security argument is the project's own (the threshold layers of
tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE, and the carry
mask of dMCFE-DDH); *unreviewed* -- not yet checked (the three
``ddh_ristretto255`` schemes); *known-insecure* -- a
concrete privacy defect is known. Two schemes are: ``mcfe/kp_ab_pairing``
(NPP22 Sec. 5.2), where one corrupted client opens every functional key with
the rows of ``Z_q`` of its encryption key, so two keys whose policy the
attributes do not satisfy give an inner product, an honest client's input
among them; and ``mcfe/kp_ab_pairing_per_client``, the version in which each
client has an attribute set of its own (Sec. 5.4), which has that defect and
another: its fifth coordinate does not depend on the tag, so two keys of one
``y`` that open neither of two rounds, where the set of each client satisfies
one of the keys in both rounds, give the difference of the two rounds' inner
products. Both warn when built and the ``standard`` profile refuses them; the
ML adapter takes ``mcfe/kp_ab_pairing`` (for experiments) and not the
per-client variant. They are for functionality and performance experiments
and as the record of the attacks.

Which scheme for which task
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Task
     - Schemes
   * - Aggregate the gradients of many clients at one aggregator
     - ``mcfe/ring_lwe`` (fast with the Rust extension, packed tensors);
       ``mcfe/ddh`` or ``mcfe/ddh_ristretto255`` for short vectors;
       through :mod:`pyfe4ai.ml` (:doc:`threshold_ml`)
   * - The same without a key authority for functional keys (the setup is
       still trusted)
     - ``mcfe/ring_lwe_decentralized`` or ``mcfe/lwe_decentralized``
       (reviewed); ``mcfe/ddh_decentralized`` (experimental)
   * - The same without a single decryptor
     - ``mcfe/ddh_threshold`` (reviewed); ``mcfe/ring_lwe_threshold``
       (experimental; the ML adapter runs its servers); see
       :doc:`threshold_ml`
   * - Score one party's features with another party's linear model
     - ``sife/ddh`` (short vectors), ``sife/ring_lwe`` (many vectors per
       ciphertext), ``sife/paillier`` (large values); through
       :func:`~pyfe4ai.ml.encrypt_features` and
       :func:`~pyfe4ai.ml.compute_linear`
   * - Adaptive (not only selective) security
     - ``sife/fullysec_ring_lwe``; ``sife/damgard`` and ``sife/paillier``
       (ALS16)
   * - Hide the model weights too
     - ``sife/fh_ipe_pairing``, ``mife/fh_ipe_pairing``,
       ``mcfe/fh_multi_ipe_pairing`` and its threshold and decentralized
       variants
   * - Quadratic functions ``x^T F y``
     - ``quadratic/sgp`` or ``quadratic/quad``; across several users
       ``quadratic/miqfe``
   * - Post-quantum assumptions
     - the ``lwe`` and ``ring_lwe`` variants (the decentralized
       ``lwe_decentralized`` derives its keys under a DCR modulus, which is
       not post-quantum)
   * - Decrypt only for keys whose attributes satisfy an access policy
       chosen at encryption
     - ``sife/cp_ab_pairing`` (selective security for one
       challenge ciphertext, and ciphertexts are malleable, so authenticate
       them where tampering matters; policies of ``and``, ``or`` and
       threshold gates over attribute labels, see :doc:`api/policy`)
   * - Decrypt only for one named identity (or one of several), with
       post-quantum assumptions
     - the identity directory :mod:`pyfe4ai.ac.identity_directory`
       (experimental): one ``sife/fullysec_ring_lwe`` instance per identity,
       the key centre holds every identity's master key, ciphertexts name
       their identity; no "and" of identities, no attribute policies on
       lattices (:doc:`security`)
   * - Give a key a policy that decides which ciphertexts it opens, for one
       encryptor and without a secret encryption key
     - ``sife/kp_ab_pairing`` (selective version; any label,
       no universe)
   * - Release the sum of small vectors or statistics only for a key whose
       policy the clients' attributes satisfy
     - no secure choice: ``mcfe/kp_ab_pairing`` (one set of attributes for all
       clients of a label; through :mod:`pyfe4ai.ml` with ``attributes=`` and
       ``policy=``, see :doc:`api/ml`) does this, but it is *known-insecure*:
       a decryptor that colludes with one client reads what the policy
       withholds (:doc:`security`). For experiments only.

``mcfe/kp_ab_pairing`` is known-insecure (above), and is for aggregates of up
to about 10^3 elements, not for the gradients of a large model. Every element is a ciphertext of its
own, and the aggregator pairs several group elements per client and element.
With 10 clients, 10^3 elements and ciphertexts for three attributes the
aggregation takes about 50 s on one thread and about 8 s on eight, and a
client uploads about 2.4 KB per element as JSON. The Ring-LWE MCFE
(``mcfe/ring_lwe``, 10^4 elements, three clients, ``standard`` profile, native
backend, one thread, binary ciphertexts) takes the client 17.95 ms and the
aggregator 6.24 ms, at 24.6 B per element (``benchmarks/PERFORMANCE.md``):
use ``mcfe/ring_lwe`` when there is no policy to enforce.

Attribute-based schemes
-----------------------

Four schemes tie a decryption to an access policy: ``sife/cp_ab_pairing``
(Abdalla, Catalano, Gay and Ursu, ACGU20) and the three of Nguyen, Phan and
Pointcheval (NPP22). They differ in three things, and in how far their papers'
proofs go (:doc:`security`).

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * -
     - ``sife/cp_ab_pairing``
     - ``sife/kp_ab_pairing``
     - ``mcfe/kp_ab_pairing``
     - ``mcfe/kp_ab_pairing_per_client``
   * - Where the policy sits
     - the ciphertext (CP); the key holds the attributes
     - the key (KP); the ciphertext holds the attributes
     - the key (KP)
     - the key (KP)
   * - Attribute labels
     - a *small universe*: the labels, and the copies of each, are fixed at
       setup (``universe``, ``max_copies``) and the public parameters grow
       with them; a label outside it is refused
     - a *large universe*: any label of up to 1024 characters, hashed to
       ``Z_q``; no universe, only ``max_attributes`` per ciphertext and
       ``max_rows`` per policy
     - as ``sife/kp_ab_pairing``
     - as ``sife/kp_ab_pairing``
   * - Who encrypts, with what key
     - one party, *public-key*: the public parameters are enough
     - one party, *public-key*: the public parameters are enough
     - ``n`` clients, *private-key*: each holds a secret encryption key and
       encrypts one scalar per tag, and a decryption needs all ``n``
     - as ``mcfe/kp_ab_pairing``, each client for a set of attributes of its own
   * - Proof in the paper
     - selective (ACGU20 Thm. 3.2)
     - adaptive for the version with the boxed components (NPP22 Thm. 7);
       this is the selective one, without a proof of its own
     - adaptive, random-oracle model (NPP22 Thm. 14, Cor. 15); fails with one
       corrupted client, which the proof misses
     - Thm. 25 fails as stated, and the attack of Sec. 5.2 applies
   * - Status
     - reviewed
     - reviewed
     - known-insecure
     - known-insecure

*Ciphertext-policy or key-policy.* Take ``sife/cp_ab_pairing`` when the party
that encrypts decides who may read, and each reader's key states what it holds
(a role, an organisation). Take a key-policy scheme when the party that issues
keys decides what each key may open, and the data is described by the
attributes of its ciphertext (a purpose, a region): a key for
``and(purpose:research, region:EU)`` opens only ciphertexts made for both
labels.

*Identity equality.* A policy of one leaf, such as ``id:alice``, gives the
functionality of identity-based IPFE: ``<x, y>`` when the identities match and
nothing otherwise. With ``sife/cp_ab_pairing`` the ciphertext is under the
policy ``id:alice`` and the key for the attributes ``["id:alice"]`` opens it;
with ``sife/kp_ab_pairing`` the ciphertext is for the attributes
``["id:alice"]`` and the key for the policy ``id:alice`` opens it. A key for
another identity (in the ciphertext-policy scheme also one for no attribute)
is refused with :class:`~pyfe4ai.core.exceptions.FEKeyError`. This is the
general scheme with a policy of one leaf, and the identity is a label like any
other (one of the ``universe`` in ``sife/cp_ab_pairing``). It is *not* the
encoding that ACGU20 gives in its App. C.1 (Identity-Based Inner-Product
Functional Encryption), which the library does not implement: that is a
function encoding of its own (``EncCt(x, id)``, ``EncKey(y, id')``), not a
monotone span program.

*Small or large universe.* ``sife/cp_ab_pairing`` suits a vocabulary of labels
that is fixed and small: every label and copy adds a pair of G1 and a pair of
G2 elements to the public parameters, and a pair of G2 elements to each key
that holds the label. The key-policy schemes take any label, with no
registration: a ciphertext holds eight G1 elements per attribute and a key
eight G2 elements per row of its policy (in the MCFE schemes a key holds them
for every client, and a ciphertext is that of one client).

*Public or private key.* With ``sife/cp_ab_pairing`` and ``sife/kp_ab_pairing``
anyone who holds the public parameters encrypts, so a data owner needs no
secret. ``mcfe/kp_ab_pairing`` is the multi-client scheme: every client holds a
secret key, the sum over the clients is released by a key whose policy the
clients' attributes satisfy, and one client missing, also one of weight 0,
means no result. It is *known-insecure*: one corrupted client opens every key,
so the policy does not hold against a decryptor that colludes with a client.
The ML adapter aggregates through it with a policy (:doc:`api/ml`), for
experiments. ``mcfe/kp_ab_pairing_per_client`` lets the clients hold
different sets, and is *known-insecure* too. Neither is a choice for
protecting data.

The two single-input schemes are *reviewed* (checked against their papers) and
the two multi-client ones are *known-insecure*. Ciphertexts are malleable and carry their policy or
attributes in the clear, and none is constant-time: :doc:`security` lists what
each paper proves and what it does not.

The registered schemes
----------------------

Classes, kind, capabilities and whether the ``standard`` profile accepts
each scheme, from the registry (:mod:`pyfe4ai.core.registry`):

.. scheme-registry-table::

The ML adapter takes the family and the variant as two arguments
(``FESchemeWrapper("mcfe", "ring_lwe", config)``), not the id. Its
federated helpers accept the labelled schemes (MCFE; SIFE for one client),
not MIFE, and of the threshold and decentralized variants only
``mcfe/ring_lwe_threshold``, whose servers it runs; the others are used
through their scheme classes. ``python benchmarks/run.py fl-coverage`` in a
source checkout lists, scheme by scheme, whether the federated path works.
