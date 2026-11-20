Threat Model
============

Who is trusted, what each family reveals by design, and what is out of
scope, for every scheme of the library. This page makes no claim of its own:
it gathers those of :doc:`security`, :doc:`choosing`, the README's `Known
Issues / Security Notice
<https://github.com/spire-studio/pyfe4ai#known-issues--security-notice>`__ and
the registry notes (:func:`~pyfe4ai.core.status.get_scheme_status`), which give the
details. Where this page and :doc:`security` differ, :doc:`security` is
right and this page is wrong.

PyFE4AI is a research library and has not been externally audited.

In brief
--------

* **The key authority is trusted in every family**, the decentralized ones
  included: one key generator runs the setup in one process and issues every
  client's keys, the pairwise zero-sum secrets included.
* **Functional encryption reveals function values by design.** The holder of
  a key for ``y`` learns ``<x, y>`` (or ``x^T F y``, or ``C(w, x)``); keys
  for several ``y`` and one ``x`` give the projection of ``x`` on the span of
  those ``y``. Which keys to issue is the deployment's decision.
* **Records are not authenticated.** Ciphertexts are malleable, and
  ``pp_binding`` and ``pp_digest`` are digests, not signatures. Integrity,
  authenticity and confidential channels are the application's.
* **Servers are honest-but-curious, or corrupted below their threshold.**
  Fewer than ``t`` decryption servers learn nothing beyond the result, but
  one deviating server can change the result unnoticed.
* **Callers keep the one-time rules.** A client encrypts one input per
  label, a precomputation is used once. The scheme classes do not check the
  first; the ML adapter encrypts once per client and label.
* **Parameters.** The default ``toy`` profile is not secure; ``standard``
  aims at 128 bits of classical security (the pairing schemes on BLS12-381
  at about 120 to 126).
* **Side channels.** Only the native paths listed in :ref:`side-channels`
  were checked; the Python layer and the BLS12-381 scalar multiplication are
  not constant-time.
* **The ML adapter plays every role in one trust domain**, for experiments
  only.

Roles
-----

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Role
     - In the library
     - Secrets held
   * - Key authority
     - a key generator object (``*KeyGenerator``), its ``setup()``
     - the master secret key; in MIFE and MCFE every client's encryption
       key; in the threshold schemes the whole functional key before it is
       shared
   * - Encryptor (client)
     - a crypto object with ``sk``; in the public-key schemes anyone with the
       public parameters
     - its plaintext, its encryption key (MIFE, MCFE), the randomness of the
       encryption
   * - Key holder (decryptor)
     - a crypto object with a functional key
     - the functional key
   * - Decryption server (threshold)
     - the caller of ``share_decrypt``
     - one share of a functional key
   * - Combiner (threshold), key combiner (decentralized)
     - the callers of ``combine_decrypt`` and
       ``combine_function_decryption_key_share``
     - none (a combiner of key shares holds the combined key and becomes a
       key holder)
   * - Parameter cache
     - the files under ``cache_dir``
     - no secret (the DCR moduli of ``standard`` are not cached), but its
       integrity matters
   * - Storage and network
     - the application
     - none; it sees every record

Assumptions
-----------

The guarantees below hold only while all of these hold.

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - No.
     - Assumption
   * - A-1
     - The operating system's random source is available and unpredictable
       (``os.urandom``, ``getentropy``; the native ChaCha20 generator draws
       a seed from it for every call).
   * - A-2
     - Only trusted parties can write to the parameter cache directory.
   * - A-3
     - The key authority runs the setup honestly and keeps the master secret
       key.
   * - A-4
     - Callers keep the one-time rules: one ciphertext per client and label,
       one use per precomputation.
   * - A-5
     - The application keeps records intact in storage and in transit (the
       library authenticates neither ciphertexts nor keys).
   * - A-6
     - ``security_profile="standard"`` is used; ``toy`` is for functional and
       performance experiments.
   * - A-7
     - In a threshold deployment the servers run in separate trust domains,
       and fewer than ``t`` of them are corrupted.
   * - A-8
     - A stateful authority (the slot list of ``bcfe/gc_ristretto255``, the
       key servers of the threshold key issuance) runs from one copy of its
       state at a time: never two copies issuing at once, and after a backup
       is restored only that copy goes on.

Out of scope for the whole library
----------------------------------

* **Integrity under active attack:** chosen-ciphertext attacks, tampered
  ciphertexts, forged records.
* **Malicious key authorities and malicious threshold servers** (a partial
  decryption carries no proof of correctness).
* **Side channels** beyond the paths of :ref:`side-channels`: the Python
  layer (CPython integers, gmpy2), the BLS12-381 scalar multiplication, the
  length of the decimal encodings of records; power, electromagnetic and
  fault attacks.
* **Denial of service:** input validation reduces crashes, but no resource
  bound is promised.
* **Pickles from untrusted sources:** a pickle can run any code, and pickles
  are not compatible across versions (:doc:`compatibility`).
* **Erasing keys from memory:** Python objects cannot be wiped reliably; the
  native core overwrites some copies of secrets, which is not a promise.
* **Quantum adversaries**, except for the ``lwe`` and ``ring_lwe`` variants
  (and ``mcfe/lwe_decentralized`` derives its keys under a DCR modulus, which
  is not post-quantum).
* **Dependencies and the supply chain:** defects of gmpy2, numpy, arkworks,
  curve25519-dalek or PyO3.
* **What function values let one infer**, for instance from keys with
  different weights on the same data.

By family
---------

*Leaks by design* is what a scheme's function itself hands out; *out of
scope* lists what is particular to the family (the list above holds for all).
Each scheme is *stable* unless marked otherwise (:doc:`compatibility`).

Single-input inner-product FE (public key)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schemes: ``sife/ddh``, ``sife/ddh_dynamic``, ``sife/damgard``,
``sife/paillier``, ``sife/lwe``, ``sife/ring_lwe``,
``sife/fullysec_ring_lwe``, ``sife/fullysec_lwe``;
``sife/ddh_ristretto255`` (*unreviewed*).

* **Trusted:** the key authority. Encryptors use the public parameters only.
* **Leaks by design:** a key for ``y`` gives ``<x, y>``; ``y`` is public
  (the keys do not hide the function). ``sife/ddh_dynamic`` treats a short
  vector as zero-padded and truncates its ciphertext, so the ciphertext
  shows the vector's length.
* **Security notion:** selective for ``sife/ddh``, ``sife/lwe`` and
  ``sife/ring_lwe``; adaptive for ``sife/fullysec_ring_lwe``,
  ``sife/damgard`` and ``sife/paillier``; ``sife/fullysec_lwe`` is adaptive,
  but ``standard`` refuses it.
* **Out of scope:** colluding keys give the projection of ``x`` on the span
  of their ``y``. The packed layout of ``sife/fullysec_ring_lwe``
  (``ndarray_layout="packed"``) is experimental and rests on the project's
  own argument.

Function-hiding inner-product FE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schemes: ``sife/fh_ipe_pairing``, ``sife/part_fh_ipe_pairing``,
``mife/fh_ipe_pairing``, ``mife/fh_multi_ipe_pairing``,
``mcfe/fh_multi_ipe_pairing``; the threshold and decentralized variants are
below.

* **Trusted:** the key authority, and **the encryptor**: in
  ``sife/fh_ipe_pairing`` the encryption key is equivalent to the master
  secret key.
* **Leaks by design:** ``<x, y>``; ``y`` is hidden, except in
  ``sife/part_fh_ipe_pairing``, which hides it only in part: the public
  matrix ``M`` and ``V^T M`` are published, keys reveal ``M^T y``, and for a
  small ``eta - subspace_dim`` that often determines a bounded ``y``.
* **Security notion:** ``sife/fh_ipe_pairing`` in the generic bilinear group
  model (KLMMRW18); the others as in their references.
* **Out of scope:** the pairing groups give about 120 to 126 bits under
  ``standard``; the BLS12-381 scalar multiplication is not constant-time.

Multi-input inner-product FE (no labels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schemes: ``mife/ddh``, ``mife/damgard``, ``mife/paillier``, ``mife/lwe``,
``mife/ring_lwe``, ``mife/fullysec_lwe``, and the two function-hiding
schemes above; ``mife/ddh_ristretto255`` (*unreviewed*); the quadratic
``quadratic/miqfe`` alike (below).

* **Trusted:** the key authority, which also makes every slot's encryption
  key.
* **Leaks by design:** ``sum_i <x_i, y_i>``. **There are no labels:** within
  one key epoch any ciphertext of a slot combines with any ciphertexts of
  the other slots, so a client that encrypts several inputs leaks their
  differences.
* **Security notion:** ``mife/ddh`` and ``mife/lwe`` put the selectively
  secure ABDP15 scheme in every slot, so what holds is the selective
  analogue of ACFGU18 Thm. 3.
* **Out of scope:** repeated rounds (gradient aggregation, for instance):
  use MCFE.

Multi-client FE (labels)
~~~~~~~~~~~~~~~~~~~~~~~~

Schemes: ``mcfe/ddh``, ``mcfe/damgard``, ``mcfe/paillier``, ``mcfe/lwe``,
``mcfe/fullysec_lwe``, ``mcfe/ring_lwe``, ``mcfe/fh_multi_ipe_pairing``;
``mcfe/ddh_ristretto255`` (*unreviewed*).

* **Trusted:** the key authority, which makes every client's encryption key
  and issues the functional keys.
* **Leaks by design:** ``sum_i <x_i, y_i>`` per label; ciphertexts of
  different labels do not combine.
* **Caller's duty:** **one ciphertext per client and label.** Two
  ciphertexts under one label leak the difference of their plaintexts; the
  scheme classes do not check this, the ML adapter encrypts once per client
  and label. A decryption needs the ciphertexts of all ``n`` clients under
  the label.
* **Out of scope:** who chooses the labels and whether one repeats is the
  deployment's. ``mcfe/ring_lwe`` is identity-based (one functional key per
  label).

Threshold decryption
~~~~~~~~~~~~~~~~~~~~

A functional key is shared among ``s`` servers, and ``t`` of them decrypt
together. Schemes: ``mife/ddh_threshold``, ``mcfe/ddh_threshold``;
``mife/lwe_threshold`` (*experimental*, refused by ``standard``),
``mcfe/lwe_threshold`` (*experimental*), ``mcfe/ring_lwe_threshold``
(*experimental*), ``mcfe/fh_multi_ipe_pairing_threshold`` (*experimental*).
The threshold layers of the experimental schemes are the project's own,
without a published proof.

* **Trusted:** the key authority, which as the dealer sees the whole
  functional key; the servers are honest-but-curious, fewer than ``t`` of
  them corrupted at once (A-7).
* **Leaks by design:** the result. The flooding noise in partial
  decryptions protects the shares, accounted one sample per partial
  decryption; ``flooding_accounting="kl"`` needs
  ``max_partial_decryptions``, a bound on their number.
* **Integrity:** a partial decryption carries no proof of correctness, so
  one deviating server changes the result unnoticed. ``combine_decrypt``
  raises when the results of several server sets disagree, and when the
  partial decryptions of one set were made for different ciphertexts.
* **Caller's duty:** **store and transmit the** ``sharing_id`` with every key
  share and partial decryption; without it, partial decryptions of two
  sharings can combine to a wrong value without an error. The servers of
  tMIFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE decrypt only ciphertexts
  that carry their clients' MAC tags.
* **Out of scope:** partial decryptions carry no digest of the public
  parameters, so a combiner given those of another deployment is not warned
  (it holds no secret). For ``t < s`` the LWE threshold layers use
  replicated sharing by default, since Shamir shares there let partial
  decryptions reveal the decryption noise modulo small integers; an explicit
  ``"shamir"`` with ``t < s`` draws a warning under ``toy``, and servers
  refuse such key shares unless created with ``legacy_shamir = "warn"``.

Decentralized multi-client FE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clients derive shares of each functional key, and the decryptor combines
them. Schemes: ``mcfe/lwe_decentralized``, ``mcfe/ring_lwe_decentralized``,
``mcfe/paillier_decentralized`` (refused by ``standard``),
``mcfe/fh_multi_ipe_pairing_decentralized``; ``mcfe/ddh_decentralized``
(*experimental*: its carry mask is the project's own).

* **Trusted:** **the setup is still trusted**: one key generator makes every
  client's keys in one process, the pairwise zero-sum secrets included. A
  deployment without a trusted party needs a distributed setup (pairwise key
  exchange, for instance), which the library does not provide. After the
  setup, functional keys need no authority.
* **Leaks by design:** as multi-client FE; a combined functional key is an
  ordinary functional key.
* **Out of scope:** a distributed setup; clients that deviate in their key
  shares. ``mcfe/paillier_decentralized`` is refused by ``standard`` (its key
  shares are combined under a DCR modulus of more than 15000 bits).

Attribute-based inner-product FE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schemes: ``sife/cp_ab_pairing``, ``sife/kp_ab_pairing``;
``mcfe/kp_ab_pairing`` (*known-insecure*) and
``mcfe/kp_ab_pairing_per_client`` (*known-insecure*), both refused by
``standard``.

* **Trusted:** the key authority, which issues keys with attributes or with
  policies.
* **Leaks by design:** a key whose policy fits gives ``<x, y>``. **Policies
  and attribute sets are public:** a key carries its policy (key-policy) or
  its attributes (ciphertext-policy) in the clear, and a ciphertext its
  attributes or its policy.
* **Security notion:** ``sife/cp_ab_pairing`` selective simulation security
  for one challenge ciphertext (ACGU20 Thm. 3.2); ``sife/kp_ab_pairing`` is
  the selective version of NPP22 Fig. 1, whose paper proves only the adaptive
  version, so **the version implemented has no proof of its own**.
* **Out of scope:** ciphertexts are malleable and not authenticated. The two
  multi-client schemes are known to be insecure: one corrupted client opens
  every functional key (:doc:`security`).

Quadratic FE
~~~~~~~~~~~~

Schemes: ``quadratic/sgp``, ``quadratic/quad``, ``quadratic/miqfe``.

* **Trusted:** the key authority.
* **Leaks by design:** ``x^T F y`` with a public ``F``. ``quadratic/miqfe``
  is multi-input without labels, with the same combining issue as above.
* **Security notion:** ``quadratic/quad`` CPA (Gay20);
  ``quadratic/sgp`` in the generic bilinear group model (DSGPP18);
  ``quadratic/miqfe`` as in AGT22 Sec. 5.

ristretto255 variants
~~~~~~~~~~~~~~~~~~~~~

Schemes: ``sife/ddh_ristretto255`` (*unreviewed*), ``mife/ddh_ristretto255``
(*unreviewed*), ``mcfe/ddh_ristretto255`` (*unreviewed*). Trust and leaks are
those of the DDH scheme of the family; the group gives about 126 bits. With
the native extension the group arithmetic is curve25519-dalek's (secret
scalars take its constant-time multiplication), without it pure Python,
which is not constant-time. They are in the experimental tier of the
compatibility promise (:doc:`compatibility`).

ML adapter (``pyfe4ai.ml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pyfe4ai.ml.FESchemeWrapper` runs the key generator itself, so it
holds the master secret key and plays the clients, the key authority and the
decryptor in one trust domain; for ``mcfe/ring_lwe_threshold`` it also runs
the decryption servers, where the threshold then adds no protection. **Use it
for experiments, not as a deployment.** A deployment uses the separate steps
(:func:`~pyfe4ai.ml.partial_decrypt_gradients`,
:func:`~pyfe4ai.ml.combine_partial_decryptions`), each party in its own trust
domain.

Bounded-key circuit FE
~~~~~~~~~~~~~~~~~~~~~~

Scheme: ``bcfe/gc_ristretto255`` (*experimental*; its security argument is
the project's own, reviewed internally, not externally).

* **Trusted:** the authority. **The encryptor is assumed honest.** The slot
  list (the public parameters) reaches encryptors over an authenticated
  channel (``pp_digest`` is not a signature). The authority's directory
  holds the slot secrets (and, by default, the keys it issued): it is kept
  like a master key, and one copy of it is in use (A-8). The library cannot
  authenticate requester ids: the caller does.
* **Leaks by design:** a key holder learns ``C(w, x)`` and its own ``w``
  (no function privacy). A key decrypts only ciphertexts made after its slot
  existed. The slot list shows the number of slots, their public keys and its
  version; a ciphertext grows with the slots it covers. Every decryption
  failure is reported as one error.
* **Security notion:** IND security in the random-oracle model under strong
  Diffie-Hellman in ristretto255, not simulation security. The bound is on
  the keys issued in all (``max_keys``), not the collusion bound of GVW12 or
  AV19: any number of the issued keys may collude.
* **Out of scope:** malicious encryptors (another circuit, selective
  failures, a different ``x`` per slot); forged slot lists; function privacy;
  quantum adversaries (the labels are encrypted with hashed ElGamal);
  revocation and key rotation.

Threshold key issuance and identity directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both are *experimental*; the scheme underneath each is unchanged.

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - Threshold key issuance of ``sife/cp_ab_pairing``
       (``distributed_keygen``)
     - Identity directory (``pyfe4ai.ac``)
   * - Trusted
     - the dealer, which runs once and discards every secret after
       delivering the shares; fewer than ``t`` key servers corrupted, chosen
       before setup (static corruption); **mutually authenticated,
       confidential channels** (mTLS or an equivalent), with the requester's
       identity taken from the transport; an approval rule that decides on
       the weights too
     - the key centre, which holds every identity's master key (key escrow)
       and gives an identity's keys to that identity only; encryptors obtain
       the directory's digest (``expected_digest``) over a trusted channel;
       mapping people to identities is the application's
   * - Leaks by design
     - every server sees the whole request; a repeated request gets the same
       partial key; a failed combination does not tell which server
       cheated; the combined key leaks what any key of the scheme leaks
     - ciphertexts name their identity (no anonymity); the directory lists
       every identity and its keys; the identity fields of ciphertexts and
       keys are claims, not authenticated
   * - Security notion
     - threshold selective simulation security under static corruption, by
       the project's own argument (reduced to ACGU20 Thm. 3.2); the scheme
       itself stays SEL-SIM
     - instances share the parameter record and draw their secrets
       independently, so an attack on one identity reduces to an attack on
       one instance of ``sife/fullysec_ring_lwe``
   * - Out of scope
     - adaptive corruption, share refresh, a dealerless key generation,
       malicious dealers, identifying a cheating server, request privacy,
       channels that are not authenticated or not confidential, denial of
       service
     - revocation and key rotation, anonymity, chosen-ciphertext attacks,
       signed directories, authentication of requesters by the key centre

The library provides no multi-authority attribute-based inner-product FE and
no lattice-based attribute-policy inner-product FE (:doc:`security`, section
"Attribute-based IPFE not provided").

Security notions checked against the papers
-------------------------------------------

Before 1.0 the security notions that the documentation names were compared,
scheme by scheme, with the theorems of the cited papers. Every notion named
agreed with its theorem:

.. list-table::
   :header-rows: 1
   :widths: 36 40 24

   * - Scheme
     - Notion named in the documentation
     - Theorem
   * - ``sife/cp_ab_pairing``
     - SEL-SIM, one challenge ciphertext; SEL-IND for several by a hybrid
     - ACGU20 Thm. 3.2
   * - ``sife/kp_ab_pairing``
     - the selective version implemented has no proof of its own; the
       theorem is for the adaptive version
     - NPP22 Thm. 7
   * - ``mcfe/kp_ab_pairing``, ``mcfe/kp_ab_pairing_per_client``
     - the claims do not hold (known-insecure)
     - NPP22 Thm. 14, Cor. 15; Thm. 25
   * - ``sife/lwe``, ``mife/lwe``
     - selective IND-CPA, by the proof of the full version (``mife/lwe``:
       its selective analogue of ACFGU18 Thm. 3)
     - ABDP15 (ePrint 2015/017) Thm. 4.4, 6.6, 6.7
   * - ``sife/damgard``, ``sife/paillier``, ``sife/fullysec_lwe``
     - adaptive
     - ALS16 Thm. 1, 5, 2 and 3
   * - ``sife/ring_lwe``
     - selective
     - BMMS21 Thm. 3
   * - ``sife/fullysec_ring_lwe``
     - adaptive (with ``m >= 5``, a deviation stated in its note)
     - BMMS21 Thm. 4
   * - ``sife/fh_ipe_pairing``
     - generic bilinear group model
     - KLMMRW18 Thm. 3.1
   * - ``sife/part_fh_ipe_pairing``, ``quadratic/quad``
     - CPA
     - Gay20 Thm. 2, Thm. 1
   * - ``mife/fh_multi_ipe_pairing``
     - every slot needs a ciphertext (no secret-key layer of Remark B.1)
     - DOT18 Thm. 3.1
   * - ``mcfe/lwe``, ``mcfe/fullysec_lwe``
     - selective and adaptive parameters
     - ABMPSW20 Thm. 3.8
   * - ``mcfe/fh_multi_ipe_pairing_decentralized``
     - selective FH-DMCFE
     - NPS24 Fig. 1
   * - ``quadratic/sgp``
     - generic bilinear group model
     - DSGPP18 Thm. 2
   * - ``quadratic/miqfe``
     - pos-security
     - AGT22 Thm. 5.1
   * - ``bcfe/gc_ristretto255``
     - IND in the random-oracle model, strong DH; not SIM
     - the project's own argument
   * - threshold key issuance; identity directory
     - threshold SEL-SIM under static corruption; one instance of BMMS21
       Sec. 5 per identity
     - the project's own argument; BMMS21 Thm. 4

Not compared this way: ``sife/ddh_dynamic``; ``mife/paillier``,
``mife/damgard``, ``mife/ring_lwe``, ``mife/fullysec_lwe``,
``mife/fh_ipe_pairing``; ``mcfe/damgard``, ``mcfe/paillier``;
``mcfe/lwe_decentralized``, ``mcfe/paillier_decentralized``;
``mife/ddh_threshold``, ``mcfe/ddh_threshold``; the three ristretto255
schemes; the project's own threshold layers and carry mask, which have no
published proof. The notions of ``sife/ddh``, ``mcfe/ddh``,
``mcfe/fh_multi_ipe_pairing``, ``mcfe/ring_lwe`` and
``mcfe/ring_lwe_decentralized`` were read in their papers but are not named
in the documentation.
