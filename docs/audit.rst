Audit Package
=============

PyFE4AI has **not been externally audited**. This page is the entry point
for an external audit: what to audit first, where the material is, and the
questions we would like answered. It is not an audit and claims nothing that
:doc:`security` and :doc:`threat_model` do not.

* The package is published with the release, like the rest of the
  documentation, so that anyone can check the library with it.
* The results of an external audit will be published, here and in
  ``SECURITY.md``, with the fixes they lead to.
* Of the project's internal reviews, the conclusions and the fixes are
  published (`Internal reviews`_); the working material is not.
* A finding that should not be public yet goes through the reporting channel
  of ``SECURITY.md``, not through this page.

Scope
-----

The audit covers the *stable* tier of the compatibility promise
(:doc:`compatibility`) first, by priority. Every registered scheme is in one
of the groups below (the test ``tests/api/test_audit_package.py`` keeps them
in step with the registry).

* **A1** -- ``sife/ring_lwe``, ``sife/fullysec_ring_lwe``, ``mife/ring_lwe``,
  ``mcfe/ring_lwe``, ``mcfe/ring_lwe_decentralized``, ``sife/cp_ab_pairing``,
  ``sife/kp_ab_pairing``, ``mife/ddh_threshold``, ``mcfe/ddh_threshold``.
  The Ring-LWE schemes carry the ML adapter (gradient aggregation) and the
  largest part of the native core (NTT, RNS moduli, packing); their widths are
  derived by the project from the constraints of BMMS21 Sec. 4.2, and
  ``sife/fullysec_ring_lwe`` uses ``m >= 5`` key polynomials instead of the
  paper's 3. The attribute-based schemes are new in 0.5 (the version of
  ``sife/kp_ab_pairing`` implemented has no proof of its own), with a new
  policy compiler. The two DDH threshold schemes are the only threshold
  schemes of the stable tier.
* **A2** -- ``sife/lwe``, ``mife/lwe``, ``mcfe/lwe``, ``mcfe/fullysec_lwe``,
  ``mcfe/lwe_decentralized``, ``sife/paillier``, ``mife/paillier``,
  ``mcfe/paillier``. Dense-LWE parameters and the exact Gaussian sampler
  (FACCT); the DCR key derivation of ``mcfe/lwe_decentralized``; the DCR
  moduli, which ``standard`` does not cache, and the rounded Gaussian of the
  DCR schemes.
* **A3** -- ``sife/ddh``, ``sife/ddh_dynamic``, ``sife/damgard``,
  ``mife/ddh``, ``mife/damgard``, ``mcfe/ddh``, ``mcfe/damgard``,
  ``sife/fh_ipe_pairing``, ``sife/part_fh_ipe_pairing``,
  ``mife/fh_ipe_pairing``, ``mife/fh_multi_ipe_pairing``,
  ``mcfe/fh_multi_ipe_pairing``, ``mcfe/fh_multi_ipe_pairing_decentralized``,
  ``quadratic/quad``, ``quadratic/sgp``, ``quadratic/miqfe``. Safe-prime
  groups and discrete logarithms (the oldest and most reviewed code), and
  standard constructions over pairings (BLS12-381 arithmetic from arkworks).
* **A, formats and correctness only** -- ``sife/fullysec_lwe``,
  ``mife/fullysec_lwe``, ``mcfe/paillier_decentralized``. Stable, but
  ``standard`` refuses them (their sizes are not practical), so only their
  formats and correctness under ``toy`` matter.
* **B (optional, later)** -- ``mife/lwe_threshold``, ``mcfe/lwe_threshold``,
  ``mcfe/ring_lwe_threshold``, ``mcfe/fh_multi_ipe_pairing_threshold``,
  ``mcfe/ddh_decentralized``, ``bcfe/gc_ristretto255``,
  ``sife/ddh_ristretto255``, ``mife/ddh_ristretto255``,
  ``mcfe/ddh_ristretto255``; with them the threshold key issuance of
  ``sife/cp_ab_pairing`` and the identity directory. The experimental tier:
  constructions and arguments of the project's own and unreviewed variants,
  to be audited once their protocols have been stable for a release.
* **Not in scope** -- ``mcfe/kp_ab_pairing``, ``mcfe/kp_ab_pairing_per_client``:
  known to be insecure; the attacks are stated in :doc:`security` and
  reproduced by tests.

Besides the schemes:

* **The native core** (``rust/core``, ``rust/native``): bit-for-bit agreement
  with the Python reference (the existing agreement tests); memory safety at
  the PyO3 boundary, the ``unsafe`` blocks, the overwriting of copies of
  secrets; constant time on the paths of :ref:`side-channels`, and the paths
  stated not to be constant-time.
* **Parameters**: the ``standard`` floors and presets
  (``pyfe4ai/core/profiles.py``), the lattice-estimator scripts of
  ``tools/params/`` and their output, the DCR and DL sizes.
* **Sampling**: the exact discrete Gaussian (FACCT) and the rounded one, the
  rules choosing between them, the flooding noise.
* **Hashing**: the label hashes, the domain separation of ``hash_to_curve``,
  the SHAKE expansions.
* **Parsing and checks**: the strict parsing of records, ``pp_binding``, the
  MAC tags.
* **The parameter cache**: its trust model and the regeneration of records
  that do not match.
* **The compatibility rules** (:doc:`compatibility`): whether any path lets a
  record known to be insecure be accepted again.

Not in scope: ``benchmarks/``, ``examples/`` and the site; the ML adapter as a
deployment (it plays every role in one trust domain); side channels of the
Python layer.

Material
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - What
     - Where
   * - A fixed commit and tag, reproducible build
     - the release; ``docker/`` for the images of the test matrix
   * - Threat model
     - :doc:`threat_model`
   * - Every scheme: reference, fidelity, deviations, status
     - the registry (:func:`~pyfe4ai.core.status.get_scheme_status`), the
       README status table, :doc:`security`, the module docstrings
   * - What the papers prove, and the notions checked against them
     - :doc:`security`; :doc:`threat_model` (section "Security notions
       checked against the papers")
   * - Arguments of the project's own
     - :doc:`security` (threshold layers, bounded-key FE, threshold key
       issuance, identity directory, the ABDP15 LWE proof used);
       :doc:`rns_modulus_review`; the module docstrings
   * - Compatibility rules and deprecations
     - :doc:`compatibility`, :doc:`deprecations`,
       :mod:`pyfe4ai.core.stability`
   * - Known and fixed issues
     - the README's Known Issues, ``SECURITY.md``, ``CHANGELOG.md``
   * - Parameter derivation
     - ``tools/params/`` (lattice-estimator), :doc:`configuration`
   * - Regression tests of reviews and attacks
     - ``tests/review/``, ``tests/hardening/`` (strict parsing, format
       mutations), ``tests/compat/`` (record fixtures), the agreement tests
       of the native backend
   * - Side channels
     - :ref:`side-channels`; the harness ``rust/core/examples/ct_dudect/``
       (``main.rs`` and the generated ``data.rs``)
   * - Dependencies
     - ``pyproject.toml``, ``rust/*/Cargo.toml`` and ``Cargo.lock``: gmpy2,
       numpy, arkworks, curve25519-dalek, PyO3

Questions for the auditors
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 6 74 20

   * - No.
     - Question
     - Scope
   * - Q1
     - The Ring-LWE widths and moduli, derived from BMMS21 Sec. 4.2 with the
       smoothing conditions its Table 1 leaves out: do they meet every
       condition of the paper's proofs?
     - A1
   * - Q2
     - ``sife/fullysec_ring_lwe`` uses ``m >= 5`` key polynomials, since
       ``m = 3`` admits no modulus under the constraints of App. C.1: is the
       reasoning right, and is the scheme secure with it?
     - A1
   * - Q3
     - ``sife/kp_ab_pairing`` implements the selective version of NPP22 Fig. 1,
       which the paper does not prove on its own: is there an acceptable
       argument for it?
     - A1
   * - Q4
     - Do the changes of ``sife/cp_ab_pairing`` to ACGU20 (``a = (1, a2)``,
       the rows of attributes outside the policy omitted, the decoding of
       App. B.2 with its sign corrected) keep Thm. 3.2?
     - A1
   * - Q5
     - Threshold DDH: do the Shamir shares and partial decryptions reveal no
       more than the result to honest-but-curious servers, and are the
       consistency checks of the combination sufficient?
     - A1
   * - Q6
     - The native FACCT sampler agrees with the Python reference bit for bit
       and defers to the exact threshold in a band of 2^-100: does that add
       a measurable bias or timing difference? Is the constant-time
       acceptance test and exponential right?
     - A2
   * - Q7
     - The constant-time results: the paths measured, the method, the
       thresholds and the list of exceptions. Please look first at the
       garbling with the seed secret: without the processor's
       data-independent-timing mode (DIT) the shape of its time distribution
       depends on the seed (fresh seeds in both classes were clean in three
       runs, two fixed seeds against each other flagged in eight of nine runs
       (three pairs, three runs each)), and with DIT set, as the garbling does
       since 1.0, the same pairs were clean in nine runs (three each), while
       every mean stayed within the control; the mechanism was not found, and
       whether this could be exploited was not evaluated. Is setting DIT
       around the garbling alone (not the label encryption) the right
       remedy?
     - native core
   * - Q8
     - The compatibility rules and their security exception: can a record
       known to be insecure be accepted again?
     - all
   * - Q9
     - The trust model of the parameter cache: is local write access the
       only premise needed?
     - all
   * - Q10
     - The label hashes and the domain separation of ``hash_to_curve``.
     - all
   * - Q11
     - The flooding noise of the threshold layers, accounted one sample per
       partial decryption; replicated sharing and its seeded variant.
     - B
   * - Q12
     - The stateful issuance of ``bcfe/gc_ristretto255`` and the threshold
       key issuance of ``sife/cp_ab_pairing``.
     - B
   * - Q13
     - The timing method: both classes run the same code on one buffer drawn
       from a seed, and a result is judged by its effect size against a
       SHAKE-256 control. Is it adequate, can it stand in for a formal
       check, and how should cropped tests that flag with the means within
       the control be read? For the garbling with the seed secret (Q7), an
       A/A test with fresh seeds and two fixed seeds against each other
       separated the method from the secret.
     - native core
   * - Q14
     - FACCT is a rejection sampler, whose time follows the number of
       candidates and the order of its decisions, neither of which depends on
       the values returned. Is replaying one sample's decisions with other
       values (``facct_values``, ``facct_values_cond``) the right test that
       the time does not depend on the values, and is preparing the same cache
       state for both classes enough (a first version, which did not, showed
       differences between identical values)?
     - A2
   * - Q15
     - The garbling of ``bcfe/gc_ristretto255`` selects by the bits of ``x``
       with masks, and its label KEM by the bits of ``w``. The garbling was
       timed by the core's harness, the KEM only from Python (a resolution of
       25 to 110 ns per call): is that enough, and what would a finer
       measurement of the KEM need?
     - B
   * - Q16
     - The threshold key issuance leaves mutually authenticated, confidential
       channels to the application and does not encrypt partial keys
       itself: is that division acceptable?
     - B

Internal reviews
----------------

The project reviews its own changes (a separate review of each change that
touches security, with tests that reproduce every finding, then a re-check
of the fixes). These are not external audits. Their conclusions and fixes:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Area
     - Conclusion and fix
   * - Threshold LWE layers (v0.3.0)
     - Shamir sharing with ``t < s`` let partial decryptions reveal the
       decryption noise modulo small integers. Fixed in v0.4.0 by replicated
       sharing, the default for ``t < s`` (``SECURITY.md``, "Fixed Issues").
   * - Parameters and samplers (v0.3.0)
     - Samplers short of their budgets, Ring-LWE widths for norm ``X``
       instead of ``3X``, ABDP15 dimensions read at the wrong width, key
       shares not bound to their parameters, a race in the ML adapter. Fixed
       in v0.4.0 (``SECURITY.md``, ``CHANGELOG.md``).
   * - ABDP15 over LWE
     - Thm. 2 of the PKC 2015 paper does not hold for LWE; the library
       relies on the proof of the full version (ePrint 2015/017) and sizes
       the parameters by it (:doc:`security`).
   * - NPP22 multi-client schemes (0.5)
     - ``mcfe/kp_ab_pairing``: one corrupted client opens every functional
       key; ``mcfe/kp_ab_pairing_per_client``: two tags and two keys give
       the difference of two rounds. Both registered as known-insecure and
       refused by ``standard``; the attacks are reproduced by tests
       (:doc:`security`).
   * - Native kernels after ``fork``
     - Forked children could crash on macOS after multi-threaded native
       calls. Fixed: native calls in a forked child run single-threaded
       (``CHANGELOG.md``).
   * - Bounded-key FE (0.6)
     - The security argument was reviewed twice; it holds as stated in
       :doc:`security` after fixes, with the honest encryptor and the other
       conditions listed there.
   * - Threshold key issuance and identity directory (0.7)
     - The issuance argument was reviewed, revised and re-checked; the
       conditions stated in :doc:`security` came out of it (only ``True``
       approves a request, the requester comes from the transport, partial
       keys are secret). The directory's identity rule was made independent
       of the reader's Unicode version.
   * - Public claims and compatibility (1.0)
     - The security claims of the documentation were checked against the
       registry and the papers (:doc:`threat_model`), and the compatibility
       promise against the record formats; the wording was corrected where
       it said too much or too little.
   * - Native FACCT timing (1.0)
     - The acceptance test and the exponential depended on the values drawn:
       replaying one sample's decisions with other values moved the time by
       about 4 %. Fixed in 1.0: both are branch-free, with the same samples
       bit for bit, and the same replay (corrected for a cache artefact of its
       first version) shows no difference (:ref:`side-channels`).
       Tests of whole seeds with the same counts could not separate the
       values from the order of the decisions.
