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
     - the function-hiding and quadratic schemes
     - BLS12-381, about 120 to 126-bit security (below the 128-bit target);
       needs the Rust extension (or charm-crypto for other curves)

Capabilities
------------

Each scheme lists its capabilities
(:data:`~pyfe4ai.core.registry.CAPABILITIES`):

* ``labels`` -- ciphertexts and functional keys are bound to a label (the
  MCFE family);
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
  (``ring_modulus="rns"``).

Review status
-------------

Every scheme carries a review status (the full table is on
:doc:`security`): *reviewed* -- checked against its paper, *faithful* or
*adapted* with the deviations listed; *experimental* -- the construction or
its security argument is the project's own (the threshold layers of
tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE, and the carry
mask of dMCFE-DDH); *unreviewed* -- not yet checked (the three
``ddh_ristretto255`` schemes). No scheme is currently *known-insecure*.

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
