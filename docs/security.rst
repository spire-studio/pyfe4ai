Security & Status
=================

PyFE4AI is a research prototype: no scheme is hardened against side
channels, and the default parameters are toy values. This page lists the
review status of every scheme and the parameter presets; the README sections
`Scheme Status <https://github.com/spire-studio/pyfe4ai#scheme-status>`__,
`Security Profiles <https://github.com/spire-studio/pyfe4ai#security-profiles>`__
and `Known Issues / Security Notice
<https://github.com/spire-studio/pyfe4ai#known-issues--security-notice>`__
add measured costs and the remaining limitations.

Scheme status
-------------

|n_schemes| registered schemes: |n_reviewed| reviewed, |n_experimental|
experimental, |n_unreviewed| unreviewed and |n_known_insecure|
known-insecure. The experimental ones are the threshold layers of tMIFE-LWE,
tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE and the carry mask of
dMCFE-DDH: our own constructions on top of published schemes, with no
published security proof. The known-insecure one is
``mcfe/kp_ab_pairing_per_client`` (NPP22 Sec. 5.4, an attribute set for each
client): the fifth coordinate of the paper does not depend on the tag, so two
keys of one ``y`` that open neither of two rounds, where the set of each client
satisfies one of the keys in both rounds, give the difference of the two
rounds' inner products, and Theorem 25 of the paper fails as stated.
It is kept for functionality and performance experiments and as the record of
the attack. Query a scheme at runtime with
:func:`pyfe4ai.get_scheme_status(family, variant)
<pyfe4ai.core.status.get_scheme_status>` (or ``"family/variant"``) or
:attr:`FESchemeWrapper(...).status <pyfe4ai.ml.FESchemeWrapper.status>`;
the key generator and crypto objects of a known-insecure scheme emit
:class:`~pyfe4ai.core.exceptions.FESecurityWarning`, and the ``standard``
security profile refuses it (``FEValidationError``, as it refuses any scheme
with a ``refused`` reason). The table is rendered
from :data:`~pyfe4ai.core.status.SCHEME_STATUS`, which is derived from the
scheme registry (:mod:`pyfe4ai.core.registry`); the status and fidelity
values are defined with :class:`~pyfe4ai.core.registry.SchemeStatus`.

.. scheme-status-table::

Security profiles
-----------------

The ``security_profile`` entry of a key generator's config selects its
parameters: ``"toy"`` (the default) is fast and **not secure**;
``"standard"`` aims at 128-bit classical security and refuses explicit
values below it. Its exceptions: the pairing-based schemes run on
BLS12-381, about 120-126-bit after the exTNFS attacks (with a
:class:`~pyfe4ai.core.exceptions.FESecurityWarning`),
and ``sife/fullysec_lwe``, ``mife/fullysec_lwe`` (ALS16),
``mife/lwe_threshold`` and ``mcfe/paillier_decentralized`` are refused, as is
every known-insecure scheme (``mcfe/kp_ab_pairing_per_client``).
:mod:`pyfe4ai.core.profiles` below gives the parameters per family,
including ``matrix_mode`` for the ABDP15 LWE schemes; :doc:`configuration`
lists every option with its default under each profile, and the option
combinations that are refused.

The pairing backends differ in how they treat a serialised group element. The
default BLS12-381 backend checks the encoding and the subgroup of every group
element it deserializes and is not affected by what follows. For the groups
that charm provides (MNT224, BN254, ...) two things have been observed: its
deserializer may hang on a malformed G2 string, and it accepts junk under a
valid type tag. Keys and public parameters for the charm groups must
therefore come from a trusted source. Ciphertexts come from outside: the
attribute-based schemes (``sife/cp_ab_pairing``, ``sife/kp_ab_pairing``,
``mcfe/kp_ab_pairing`` and ``mcfe/kp_ab_pairing_per_client``) hold only G1
elements in them, refuse any other type tag before charm reads a string, and
refuse a string that is not the one its element serialises to. charm ignores
bytes after an element, and strings cut short have decoded to the honest
element, which suggests that it reads past the end of its buffer; that check
runs after the read, and the other pairing schemes do not make it. Prefer
BLS12-381 for ciphertexts from parties you do not trust. A charm group has no
domain separation tag either: there the key-policy MCFE schemes hash a tag to
G1 by hashing the domain separation tag, a bar and the message as one C string
(see :func:`~pyfe4ai.primitives.pairing.pairing_backend.hash_to_g1`), and only
on BLS12-381 is that hash the RFC 9380 one that the schemes' random-oracle
argument takes.

For the ABDP15 LWE schemes (``sife/lwe``, ``mife/lwe``) the 128 bits are
those of the LWE problem that their security proof reduces to. The
lattice-estimator table is read at the error width
``sigma' = sigma / (1 + B sqrt(eta))`` of that problem (``B = 2 X``, or
``4 X`` per slot of ``mife/lwe``), not at the wider ``sigma`` of the public
key. That makes the dimension 17 to 67 percent larger for
``8 <= X = Y <= 1000``, and up to about twice as large at ``X = Y = 1``.

The proof is the ABDP15 full version (ePrint 2015/017, Oct. 2015), Thm. 4.4
with Thms. 6.6 and 6.7 (Construction 6.5, Sec. 6.2), written out in
F. Bourse's PhD thesis (ENS 2017), Thm. 4.3.5 with Thms. 8.2.2, 8.2.3,
8.3.2. It is not Theorem 2 of the PKC 2015 paper, whose simulation does not
cover noisy LWE keys. The width ``sigma'`` is the condition derived in the
proof of Thm. 6.6 (= Bourse, Thm. 8.2.2); the full version's Sec. 6.2
parameter list states the stronger, sufficient ``(l B + 1) sigma'``.

Hypotheses of the papers that the library meets differently:

- The literal ``p > l B^2`` of Thm. 4.4 / Sec. 6.2 is not needed for the
  Regev instance. The change of basis acts on secret keys mod the LWE prime
  ``q``. The library decodes the integer inner product (``p > 2 eta X Y +
  1``), so admissible keys are orthogonal to ``x1 - x0`` over ``Z``.
- The ``4 X`` per slot of ``mife/lwe`` is the library's ACFGU18 composition,
  covered by neither text.
- The thesis's LWE hardness bound (an absolute width above ``2 sqrt(n)``;
  not stated in the full version) is met under ``standard``:
  ``sigma' >= 2^16``, the table's smallest width.

``mife/lwe_threshold`` is refused, like the ALS16 schemes. Once its modulus
is enlarged for flooding (``stat_param = 128``), it needs a dimension at the
very end of the table at that width (5888 to 6144, with a public matrix of
6.5 to 7.8 * 10^9 entries) for a few small bounds, e.g. ``X = Y`` in
{1, 2, 3, 8, 10} at ``eta = 1``, and beyond it for most others (``X = Y`` =
4 to 6, ``X = Y >= 12``, most ``eta >= 2``).

The opt-in KL-divergence accounting of the flooding noise
(:mod:`pyfe4ai.schemes.threshold.kl`) would bring it to about 4864 for
two clients, ``s = t = 3`` and ``2^40`` flooded values. It applies only with
``flooding_accounting = "kl"``, ``t = s`` and an explicit
``max_partial_decryptions`` budget; the public matrix would still have
4.1 * 10^9 entries (``"stream"`` mode), and the guarantee is weaker and
holds only for the first ``max_partial_decryptions`` partial decryptions of
every server: about ``stat_param - 6.85`` bits in the bit-security sense of
Micciancio and Walter, a total statistical distance of about ``2^-65`` at
``stat_param = 128``. The refusal under ``standard`` holds with either
accounting; ``"kl"`` applies to ``mife/lwe_threshold`` under ``toy``.

The arguments of the security reviews that are part of the documentation:

.. toctree::
   :maxdepth: 1

   rns_modulus_review

Threshold trust model
---------------------

The decryption servers of the threshold schemes (tMIFE, tMCFE) are
honest-but-curious: fewer than ``t`` of them learn nothing beyond the result,
but a partial decryption carries no proof of correctness, so one deviating
server changes the result unnoticed. ``combine_decrypt`` accepts partial
decryptions for several enrolled sets and raises if they disagree (see
:func:`~pyfe4ai.schemes.threshold.roles.combine_consistently`); it also
raises (:class:`~pyfe4ai.core.exceptions.FESchemeError`, for one value and
for an ndarray element alike) when the partial decryptions of one set were
computed for different ciphertexts. In
tMIFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE, servers only decrypt
ciphertexts carrying their clients' MAC tags. With the binary ciphertexts of
tMCFE-Ring-LWE (``ct_format="binary"``, its default; ``"json"`` selects the
format of earlier versions) the tags also cover the digest of the public
parameters, and the encoding is unique (values strictly below ``q``), so that
the flooding noise, seeded with the ciphertexts' bytes, cannot be redrawn by
re-encoding a ciphertext. Its key shares are split by default: a server keeps
its share of the label-independent part of the key once per weights and
receives the shares of the round's label terms per round; a linear sharing
reused with fresh sharings of the label terms is a sharing of each round's
key, the shares of ``t - 1`` servers stay independent of the keys, and the
noise is seeded as for a whole key share (``key_shares="full"`` issues whole
key shares). A server refuses a label part without the weight part of the
same weights and share set, and key shares and ciphertexts of another format
or other public parameters; a combiner refuses partial decryptions of another
format, and partial decryptions of different sharings of one key: every
sharing the key generator issues has a random ``sharing_id`` that the
partials carry, since partials of two sharings (a label shared again after it
left the key generator's cache, ``LABEL_PARTS_KEPT``) would decode to a wrong
value inside the decoding bound rather than fail. A second sharing reveals
nothing: one server's answers under the two sharings differ by its share of
a sharing of zero, which ``t - 1`` servers compute from their own parts.
**Store and transmit the** ``sharing_id`` **with every key share and partial
decryption**: partials without one (as from key shares of earlier versions)
combine only with each other, so a storage or transport layer that drops the
id on every side would let partials of two sharings combine to a wrong value
unnoticed.
Partial decryptions carry no digest of the public parameters, so a
combiner given those of another deployment of the same format is not warned
(it holds no secret; under the honest-but-curious model above the servers
send the partials of its deployment).

Attribute-based schemes
-----------------------

Four schemes release a result only to a key whose policy and attributes fit.
In ``sife/cp_ab_pairing`` the policy is in the ciphertext and the attributes
are in the key (ciphertext-policy); in ``sife/kp_ab_pairing``,
``mcfe/kp_ab_pairing`` and ``mcfe/kp_ab_pairing_per_client`` the policy is in
the key and the attributes are in the ciphertext (key-policy). All run over
pairings. The first three were checked against their papers (status *reviewed*:
checked against the cited construction, no security defect known; it is not a
security audit) and the last one is known-insecure (below). The constructions
are those of Abdalla, Catalano, Gay and Ursu (ACGU20, ASIACRYPT 2020, ePrint
2020/577) and of Nguyen, Phan and Pointcheval (NPP22, ASIACRYPT 2022, ePrint
2022/215, revision of 2022-09-18); the module documentation of each scheme
lists how the library differs from its paper, and :doc:`choosing` compares
them.

What the papers prove, scheme by scheme:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Scheme
     - Proof in the paper, and what it does not cover
   * - ``sife/cp_ab_pairing``
     - ACGU20 Sec. 3.1 Fig. 1, Thm. 3.2: selective simulation security
       (SEL-SIM) from DDH in G1 and G2 (here: SXDH) for a secure predicate
       encoding. Selective means that the policy and the plaintext of the
       challenge are chosen before the public parameters are seen, and the
       theorem is for one challenge ciphertext; several ciphertexts give
       SEL-IND by a hybrid argument. No adaptive security.
   * - ``sife/kp_ab_pairing``
     - NPP22 Sec. 4 Fig. 1. The library implements the selective version,
       without the boxed components of the figure. Thm. 7 proves the version
       *with* the boxed components adaptively IND-CPA secure from SXDH, for one
       challenge ciphertext and any number of keys; the paper says that the
       selective version can be proved with similar techniques and does not
       prove it. The scheme as implemented has no proof of its own.
   * - ``mcfe/kp_ab_pairing``
     - NPP22 Sec. 5.2 Fig. 3, Thm. 14 and Cor. 15: adaptive IND security
       from SXDH in the random-oracle model, in the game of Def. 12 and
       Fig. 2. The adversary may corrupt clients (the oracle ``Corrupt`` gives
       it their encryption keys), ask for functional keys and for ciphertexts
       under tags other than the challenge tag. The random oracle is the hash
       of the tag and the set of attributes to G1; the library takes RFC 9380
       ``hash_to_curve`` under its own domain separation tag for it.
       Admissibility (Def. 9) and Remark 10 restrict the adversary: no client
       encrypts twice under one tag, and every honest client has a challenge
       ciphertext. The notion does not cover a partial set of challenge
       ciphertexts (an all-or-nothing layer would, the paper says; the library
       has none), so a decryption needs the ciphertexts of all ``n`` clients,
       also of one whose weight is 0.
       Remark 11: all clients of a tag use one set of attributes, which the
       tag hash enforces.
   * - ``mcfe/kp_ab_pairing_per_client``
     - NPP22 Sec. 5.4, Thm. 25 (App. B.5): one-time IND security from SXDH,
       with a proof that the paper only sketches, as changes to the proof of
       Thm. 14. **Thm. 25 as stated fails**: see the next part.

What none of the proofs covers, and what the schemes share:

* **Ciphertexts are malleable and not authenticated.** The games let the
  adversary look, not change a ciphertext and ask for a decryption. Anyone
  can multiply one ``t_i`` (or ``c3[i]``) of a ciphertext by ``g1``, which adds
  ``y_i`` to what every key for ``y`` opens, or multiply two ciphertexts of one
  policy or attribute set, which adds their inner products. Authenticate
  ciphertexts where tampering matters.
* **The policy and the attribute set are public.** A functional key carries
  its policy (key-policy) or its list of attributes (ciphertext-policy) in the
  clear, and a ciphertext carries its list of attributes or its policy. None
  of these schemes hides them. The key-policy schemes hash a label to ``Z_q``
  with a public hash.
* **In the two MCFE schemes a client encrypts once under a tag.** Two
  ciphertexts of one client under one tag (and one set of attributes in
  ``mcfe/kp_ab_pairing``) have the same hash points, and give away the
  difference of the two plaintexts: in ``mcfe/kp_ab_pairing`` the points hash
  the tag and the set, in ``mcfe/kp_ab_pairing_per_client`` the tag alone, so
  there any two of them do, whatever their sets. The scheme classes do not
  check it, as with the other labelled schemes; the ML adapter reserves each
  ``(client, label)`` pair.
* **Keys are trusted input.** The ``pp_binding`` of a key, a ciphertext or a
  record is a digest of the public parameters. It tells the objects of two
  setups apart and is not a signature.
* **The claims of a ciphertext.** The ``label`` and the ``attributes_digest``
  of a ciphertext of ``mcfe/kp_ab_pairing`` are claims of its client. The
  encryption key of a client can check the claims of its own ciphertext (the
  check also recovers its plaintext) and the master key those of all of them;
  an aggregator that holds neither cannot. A decryption does not check them
  against the points, and a key opens every tag, so whoever holds the ciphertexts of a whole
  round, an aggregator with no key among them, can write another label into
  all of them, with no client taking part, and they decrypt, silently, to the
  value of the round in which they were made. This is a matter of integrity,
  not of confidentiality. A deployment that needs the claims checked has each
  client sign or MAC the label together with the record, and the decryptor
  verifies it before it decrypts. A MAC needs a key that each client shares
  with the decryptor and does not protect against the decryptor itself; a
  signature does, and anyone can verify it. What a decryption
  does and does not catch is on :doc:`api/ml` and in the module documentation.
* **Reviewed against the papers, not audited, not constant-time.** The first
  three were checked against their papers (status *reviewed*, with the limits
  above and the fidelity of the status table); that is not a security audit.
  The per-client variant is known-insecure (below). None of the four is
  constant-time (see above, and the two sections on the timing of ``encrypt``
  below).

``mcfe/kp_ab_pairing_per_client``: an attack on NPP22 Sec. 5.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attack is stated here because the scheme is registered and the paper's
theorem is cited for it. The test that reproduces it is
``test_two_tags_and_two_keys_read_the_difference_of_two_rounds`` in
``tests/mcfe/test_kp_ab_pairing_per_client.py``.

*The construction.* Sec. 5.2 hashes the tag and the set of attributes, so all
clients of a round encrypt for one set. Sec. 5.4 hashes the tag alone and lets
client ``i`` encrypt for a set ``S_i`` of its own. The paper notes (Sec. 5.4,
Remark 11) that this opens a mix-and-match attack on the keys: for two clients
and two keys of one ``y`` whose policies ``A`` and ``A'`` satisfy
``A(S_1) = A'(S_2) = 1`` and ``A'(S_1) = A(S_2) = 0``, the part of client 1
taken from the first key and that of client 2 taken from the second add up to
the inner product, which neither key opens. The paper closes it with a fifth
dimension of the bases ``H_i``, ``H_i*``. It adds two vectors:

* the ciphertext of client ``i`` has ``c_{i,ipfe} = (omega p_i, omega' p_i,
  psi_i, 0, theta_i)`` in the basis ``H_i``, where ``theta_i`` is drawn at
  setup and sits in the client's encryption key;
* the functional key has ``k_{i,ipfe} = (<S, y>, <U, y>, a_{i,0} z, 0,
  d_{A,i})`` in the basis ``H_i*``, where ``(d_{A,i})_i`` is drawn afresh for
  each key with ``sum_i theta_i d_{A,i} = 0``.

A decryption with one key is unchanged, since the terms ``theta_i d_{A,i}`` add
up to zero. The parts of two keys under one tag add up to ``theta_1 d_{A,1} +
theta_2 d_{A',2}``, which is not zero except with negligible probability, so
under one tag the splice no longer works.

*Why the fifth coordinate does not help across tags.* The part of client ``i``
in a decryption is its attribute vectors paired with the rows of the key, minus
``c_{i,ipfe}`` paired with ``k_{i,ipfe}``, plus ``e(t_i, [y_i]_2)``. The
pairing of the two fifth coordinates puts ``-theta_i d_{K,i}`` into the
exponent of that part when it is made from key ``K``. ``theta_i`` is the same
in every ciphertext of client ``i``, whatever the tag, and ``d_{K,i}`` is fixed
when ``K`` is made: the term depends on the key and the client, not on the tag.
The rest of the part is as in Sec. 5.2, and two keys of one ``y`` hold the same
``<S, y>`` and ``<U, y>``. So if client ``i`` takes its part from the same key
``K_i`` in two rounds (two tags), the product of the parts of one round is
``[<x, y> - sum_i theta_i d_{K_i,i}]_T``, and the quotient of the two rounds is
``[<x - x', y>]_T``: the term is the same in both and cancels.

*The adversary.* In the game of Fig. 2 it

1. asks for two functional keys ``A`` and ``B`` of one ``y``, such that the set
   ``S_i`` of every client satisfies one of the policies and, for each of the
   two, some client's set does not satisfy it;
2. queries ``LoR`` for every honest client under the challenge tag, with
   ``(x_i^0, x_i^1)`` and ``S_i``;
3. queries ``Enc`` for every client under a second tag, with plaintexts
   ``x'_i`` of its choice and the same sets. Remark 10 and Fig. 2 forbid ``Enc``
   under the challenge tag and a repeated ``(i, tag, ac-ct_i)``; the adversary
   makes neither query;
4. takes the part of client ``i`` from the key that ``S_i`` satisfies, in both
   rounds, and divides the product of the parts of the challenge round by
   that of the second round. It decodes ``<x^b - x', y>`` with a bounded
   discrete logarithm, adds ``<x', y>``, and compares the result with
   ``<x^0, y>`` and ``<x^1, y>``.

It is admissible under Def. 9: no client is queried twice under the challenge
tag (condition 1), every honest client has a challenge ciphertext (condition 2),
and neither key opens either round, since ``Rel(ac-k, (ac-ct_i)_i) = 1`` needs
the set of *every* client to satisfy the policy and for each key one set does
not, and condition 3 applies only to a queried key with ``Rel = 1``. It wins
whenever ``<x^0, y>`` differs from ``<x^1, y>``. Thm. 25 states that the
scheme is one-time IND secure from SXDH, against the adversaries that are
admissible, so it fails as stated.

*What it reads.* The difference of the inner products of two rounds under one
``y``, ``<x - x', y>``, from one pair of keys of that ``y``; it reads ``<x, y>``
when the other round is known, as it is in the game. This is not a bound on the
leak: with pairs of keys for several ``y``, the unit vectors say, it is
``x - x'`` coordinate by coordinate, and the ``x_i`` themselves when one round
is known. Both rounds may be two labels,
or two elements of one array, which the array helpers encrypt under a tag each.
Each key refuses both rounds on its own: its decryption raises
:class:`~pyfe4ai.core.exceptions.FEKeyError`. The test builds the parts of the
clients from the fields of the records, for two clients with the sets
``org:p`` and ``org:q``, keys for the policies ``org:p`` and ``org:q`` with
``y = (2, 3)``, and the plaintexts ``(5, -3)`` and ``(1, 7)``: the splice of one
round alone gives no discrete logarithm, and the quotient of the two rounds
decodes to ``-22``, which is ``<x - x', y>``.

*Sec. 5.2 is not affected.* In ``mcfe/kp_ab_pairing`` the tag hash includes the
set of attributes and all clients of a tag encrypt for one set. A key whose
policy that set satisfies gives the part of every client, so it opens the round
by itself; a key that does not gives no client's part. The attack needs a key
that gives the parts of some clients and not of others, which cannot happen
there, and there is no fifth coordinate. This says why the attack does not
carry over, not that the scheme is secure: Thm. 14 is the paper's, and checking
the implementation against the paper is not a proof.

*In the library.* The scheme is kept for functionality and performance
experiments and as the record of the attack. Its status is *known-insecure*;
its key generator and crypto objects emit a
:class:`~pyfe4ai.core.exceptions.FESecurityWarning` that names the issue, the
``standard`` profile refuses it (``FEValidationError``), and the ML adapter does
not take it. No repair is implemented.

The fixed-tag MIFE variant of NPP22 (Remark 16) is not provided. It has one
ciphertext per slot, so this attack does not reach it, but its proof rests on
the same Thm. 25.

*Which version of the paper.* Thm. 25 and App. B.5 are in the full version
(ePrint 2022/215, revision of 2022-09-18), which the statements above cite. The
ASIACRYPT 2022 proceedings version prints the same construction in Sec. 5.4,
with Remark 16, and no theorem: it says that the proof follows from that of
Thm. 14 with few changes and leaves the sketch to the full version. The attack
applies to the construction as printed in both.

Offline / online encryption
---------------------------

``precompute`` of ``sife/cp_ab_pairing`` computes a ciphertext under a policy
but for its plaintext term, and ``encrypt(x, precomputed=...)`` multiplies that
term in. The precomputation opens the ciphertext without a key, so it is kept
like a key and used once: a second use of its randomness gives ``x - x'`` to
anyone.

Like every exponentiation of this library neither path is constant-time. Both
take ``|x_i|`` as the exponent, not ``x_i`` modulo the group order: the online
phase raises ``g1`` to it and divides for a negative ``x_i``, and ``encrypt``
takes the inverse of ``g1`` as the base of a negative ``x_i``. So the time of
both hardly shows the signs: at ``eta = 1000`` the share of negative coordinates
moved either by less than the noise of one measurement. The plaintext is all the
online phase works on: one measurement of it still tells roughly how large the
``x_i`` are and how many of them are zero, and its table reads depend on ``x``.
``encrypt`` depends on ``x`` as well, but the term of the random ``s`` takes
about 95 percent of its time and adds noise of its own. So one timed call shows
the sizes and the zeros better on the online phase, and the signs hardly on
either.

Timing of ``encrypt`` in the key-policy schemes
-----------------------------------------------

The key-policy schemes have no offline phase. The ``encrypt`` of
``sife/kp_ab_pairing`` and of ``mcfe/kp_ab_pairing`` (which
``mcfe/kp_ab_pairing_per_client`` takes over) does the same as that of
``sife/cp_ab_pairing`` above: it takes ``|x_i|`` as the exponent, with the
inverse of ``g1`` as the base of a negative ``x_i``. In ``sife/kp_ab_pairing``
at ``eta = 1000`` all-negative plaintexts took 4.2 sigma of one call longer than
all-positive ones with ``x_i`` modulo the group order, and 0.1 sigma with
``|x_i|``. Neither is constant-time: the sizes of the ``x_i`` and the zeros
still show in their time, and in the multi-client scheme every element of a
vector is a call of its own.

ML adapter
----------

:class:`~pyfe4ai.ml.FESchemeWrapper` runs the key generator
itself, so it holds the master secret key and plays clients, key authority
and decryptor in one trust domain: use it for experiments, not as a
deployment (see :mod:`pyfe4ai.ml`). For
``mcfe/ring_lwe_threshold`` it also runs the decryption servers, so the
threshold adds no protection there; the separate steps
(:func:`~pyfe4ai.ml.partial_decrypt_gradients`,
:func:`~pyfe4ai.ml.combine_partial_decryptions`) show the
messages of a deployment, whose servers run in their own trust domains.

Gaussian sampling
-----------------

The lattice and DCR proofs sample secrets and noise from the discrete
Gaussian ``D_{Z,sigma}``. There are two samplers.

``"facct"`` (:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct`)
samples ``D_{Z,sigma}`` exactly, to about 2^-185 per sample, for every width
(natively below ``2^448``). It is the default of these schemes, which record
their sampler as ``gaussian_sampler`` in their public parameters: the
Ring-LWE MIFE / MCFE family (``sife``, ``mife``, ``mcfe`` ``ring_lwe``, the
decentralized and threshold MCFE), the dense-LWE MCFE keys ``t``
(``mcfe/lwe``, ``fullysec_lwe``, ``lwe_threshold``, ``lwe_decentralized``),
``sife/fullysec_ring_lwe`` and, under ``standard``, the noise ``E`` of the
ABDP15 public keys (``sife/lwe`` and ``mife/lwe``; ``mife/lwe_threshold``,
whose MIFE layer would do the same, is refused under ``standard``). For
them, in this order: the
``gaussian_sampler`` of the key generator's or crypto object's config; else
the one the public parameters record; else, under the ``toy`` profile only
and not for ``sife/fullysec_ring_lwe``, the one a ``gaussian_sampler()``
block selects; else ``"facct"``. The process-wide choice
(``set_gaussian_sampler``, ``PYFE4AI_GAUSSIAN_SAMPLER``) does not apply to
them. The ``standard`` profile refuses ``"rounded"`` for them, and a crypto
object whose config overrides a recorded ``"facct"`` with ``"rounded"``
warns (``FESecurityWarning``; refused under ``standard``).

ABDP15 under ``toy`` takes the ``gaussian_sampler`` of its key generator's
config, else the library-wide choice below, and records it as well.

``"rounded"`` rounds a continuous Gaussian. It is the library-wide default,
and so the sampler of the other lattice and DCR schemes unless chosen
otherwise: ALS16 (``sife`` / ``mife`` ``fullysec_lwe``), the DCR schemes and,
under ``toy``, ABDP15 SIFE / MIFE LWE (and the keys of tMIFE-LWE); the
schemes above can select it under ``toy`` (``gaussian_sampler =
"rounded"``). The *ideal* rounded Gaussian (exact uniforms and arithmetic) is
a mixture of centred discrete Gaussians, about ``0.02 / sigma^2`` from
``D_{Z,sigma}`` per sample: 2^-9 at ``sigma = 3.2``, about 2^-33 per sample at
the Ring-LWE ``sigma1`` -- about 2^-18 for one MIFE key of ``eta ring_n``
samples. Its implementations add their own terms (crypto review, October
2026):

* the double-precision path
  (:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_matrix` and the
  native kernels, widths up to ``2^40``): the two outputs of a Box-Muller pair
  share one radius (rings that show in the tail), exact half-integer ties
  round to even (about ``2^-53.8 sigma`` per sample), and no value exceeds
  about ``8.57 sigma`` (the largest radius 53-bit uniforms give). One
  ciphertext's noise is about 2^-12..2^-4 and one key about 2^-25..2^-15 from
  i.i.d. ``D_{Z,sigma}``, against budgets of 2^-128: use it for toy parameters
  and benchmarks, not where a proof needs ``D_{Z,sigma}``;
* the MPFR path (:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint`,
  wider widths: the DCR keys and masks, all threshold flooding) differs from
  the ideal rounded Gaussian only by its arithmetic, at ``bits(sigma) + 320``
  bits (``BIGINT_GUARD_BITS``: ``bits(sigma) + stat_param + ceil(log2 N) +
  8`` for every accepted ``stat_param``, at most 256, and up to 2^56 samples
  per key). At the earlier ``bits(sigma) + 128`` it was 2 to 19 bits short
  of the budgets. The threshold schemes' Gaussian flooding always uses it
  (``FLOODING_SAMPLER``, recorded in their decryption keys), whatever the
  sampler selected elsewhere; the default uniform flooding of
  ``mcfe/ring_lwe_threshold`` uses no Gaussian sampler.

The library-wide choice -- which applies to the other schemes and to ABDP15
under ``toy`` only -- is
made per call (``sampler=``), per context or per process:

.. code-block:: python

   from pyfe4ai.primitives.sampling import gaussian_sampler, set_gaussian_sampler

   with gaussian_sampler("facct"):     # this context
       ...
   set_gaussian_sampler("facct")       # the whole process

or ``PYFE4AI_GAUSSIAN_SAMPLER=facct`` in the environment. ``"facct"``
costs 0.5 to 3 microseconds per sample natively (several in CPython). Its
algorithm needs no secret-dependent branches, but CPython and big-integer
arithmetic are not constant-time, so neither sampler resists timing side
channels. For a given bit source its output is the same on every platform
and on both backends, so it comes with known-answer tests. Parties that
derive noise from a shared seed (the Gaussian flooding noise of the
threshold LWE schemes, and of ``mcfe/ring_lwe_threshold`` with
``flooding="gaussian"``) must use the same sampler; the default
``flooding="uniform"`` of ``mcfe/ring_lwe_threshold`` draws its noise
without a Gaussian sampler (uniform on ``[-2^tau, 2^tau)`` from the seeded
SHAKE-256 stream, the same statistical distance per coefficient, see the
module).

The key generator of a scheme that records its sampler publishes it in
``pp`` as the encryptors' default; public parameters without it (earlier
versions) mean ``"facct"``, and decryption does not depend on it. With the rounded
sampler, ``sife/fullysec_ring_lwe``'s keys are ``0.02 / sigma1^2`` per
sample, over ``m n eta`` key samples, from ``D_{Z,sigma1}`` for the ideal
rounded Gaussian -- in the standard configurations about 2^-47 (elementwise
``eta = 1``, ``X = 100``), 2^-44.5 (packed ``eta = 1``) and 2^-40.5 (``eta =
4``, ``X = Y = 4``) -- and about 2^-12 at ``eta = 1`` with the
double-precision path's artifacts, terms its adaptive proof does not budget.

Status API
----------

:class:`~pyfe4ai.core.registry.SchemeStatus`, which this module re-exports,
is documented with the scheme registry (:mod:`pyfe4ai.core.registry`).

.. automodule:: pyfe4ai.core.status
   :members:
   :undoc-members:
   :exclude-members: SchemeStatus

Profiles API
------------

.. automodule:: pyfe4ai.core.profiles
   :members:
   :undoc-members:
