Security & Status
=================

PyFE4AI is a research library, not externally audited: nothing is
guaranteed to be constant-time (`Side channels`_ lists what was checked), and
the default parameters are toy values. This page lists the
review status of every scheme and the parameter presets; the README sections
`Scheme Status <https://github.com/spire-studio/pyfe4ai#scheme-status>`__,
`Security Profiles <https://github.com/spire-studio/pyfe4ai#security-profiles>`__
and `Known Issues / Security Notice
<https://github.com/spire-studio/pyfe4ai#known-issues--security-notice>`__
add measured costs and the remaining limitations.

Trust assumptions in brief
--------------------------

What every scheme of the library assumes, whatever its family; the sections
below add what each family assumes on top.

* **The key authority is trusted.** The key generator holds the master secret
  key and, in the multi-input and multi-client schemes, every client's
  encryption key. This holds for the decentralized schemes too, whose setup
  one key generator runs in one process, for the dealer of the threshold key
  issuance (once, before it discards everything), and for the authority of
  ``bcfe/gc_ristretto255``.
* **Functional keys reveal function values by design.** A key for ``y`` gives
  ``<x, y>`` (or ``x^T F y``, or ``C(w, x)``); several keys for one ``x`` give
  everything those values determine. Which keys to issue is the deployment's
  decision.
* **Records are not authenticated.** Ciphertexts are malleable, and
  ``pp_binding`` and ``pp_digest`` are digests, not signatures: the integrity
  and the authenticity of keys, ciphertexts and public parameters are the
  application's to ensure, as are authenticated, confidential channels between
  parties.
* **Servers are honest-but-curious, or corrupted below their threshold.**
  Fewer than ``t`` decryption servers, or (statically) fewer than ``t`` key
  servers, are corrupted, and a partial decryption carries no proof of
  correctness (`Threshold trust model`_).
* **Callers keep the one-time rules.** A client encrypts one input per label;
  a precomputation is used once; ``bcfe`` assumes an honest encryptor.
* **One copy of a stateful authority.** An authority that keeps state (the
  slot list of ``bcfe``, the records of the key servers) is run from one copy
  of it at a time.
* **Parameters.** ``toy`` is not secure; ``standard`` aims at 128 bits (the
  pairing schemes on BLS12-381 at about 120 to 126).
* **Not covered:** chosen-ciphertext attacks and tampering, malicious
  authorities, side channels beyond `Side channels`_, pickles from untrusted
  sources, and quantum adversaries for every scheme that is not built on LWE
  or Ring-LWE.

Scheme status
-------------

|n_schemes| registered schemes: |n_reviewed| reviewed, |n_experimental|
experimental, |n_unreviewed| unreviewed and |n_known_insecure|
known-insecure. The experimental ones are the threshold layers of tMIFE-LWE,
tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE, the carry mask of
dMCFE-DDH and the bounded-key scheme ``bcfe/gc_ristretto255`` (see
:ref:`bcfe-security`): our own constructions on top of published schemes, with no
published security proof. The known-insecure ones are
``mcfe/kp_ab_pairing`` (NPP22 Sec. 5.2: one corrupted client opens every
functional key with the rows of ``Z_q`` of its encryption key, so two keys
whose policy the attributes do not satisfy give an inner product, an honest
client's input among them; the proof of Theorem 14 misses this) and
``mcfe/kp_ab_pairing_per_client`` (Sec. 5.4, an attribute set for each
client), which has that defect too and another: the fifth coordinate of the
paper does not depend on the tag, so two keys of one ``y`` that open neither
of two rounds, where the set of each client satisfies one of the keys in both
rounds, give the difference of the two rounds' inner products, and Theorem 25
of the paper fails as stated. They are kept for functionality and
performance experiments and as the record of the attacks. Query a scheme at
runtime with
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

The *Tier* column is each scheme's tier in the 1.x compatibility policy
(:doc:`compatibility`, :mod:`pyfe4ai.core.stability`): it follows from the
status, and a stable scheme's experimental options or capabilities are named
next to it.

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
every known-insecure scheme (``mcfe/kp_ab_pairing``,
``mcfe/kp_ab_pairing_per_client``).
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
pairings. ``sife/cp_ab_pairing`` and ``sife/kp_ab_pairing`` were checked
against their papers (status *reviewed*: checked against the cited
construction, no security defect known, not a security audit);
``mcfe/kp_ab_pairing`` and ``mcfe/kp_ab_pairing_per_client`` are
known-insecure (below). The constructions
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
       tag hash enforces. **The notion fails with one corrupted client**, and
       the proof misses it: see `mcfe/kp_ab_pairing: a corrupted client opens
       every functional key`_. The scheme is known-insecure.
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
* **Reviewed against the papers, not audited, not constant-time.**
  ``sife/cp_ab_pairing`` and ``sife/kp_ab_pairing`` were checked against
  their papers (status *reviewed*, with the limits above and the fidelity of
  the status table); that is not a security audit. ``mcfe/kp_ab_pairing`` and
  ``mcfe/kp_ab_pairing_per_client`` are known-insecure (below). None of the
  four is constant-time (see above, and the two sections on the timing of
  ``encrypt`` below).

``mcfe/kp_ab_pairing``: a corrupted client opens every functional key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attack is stated here because the scheme is registered and the paper's
theorem is cited for it. The test that reproduces it is
``test_a_corrupted_client_opens_every_key_and_reads_an_honest_input`` in
``tests/mcfe/test_kp_ab_pairing.py``; ``test_a_corrupted_client_opens_every_key_here_too``
in ``tests/mcfe/test_kp_ab_pairing_per_client.py`` runs it on the per-client
variation, which has the same rows.

*The rows of the client key.* In NPP22 Sec. 5.2 Fig. 3 the encryption key of
client ``i`` holds ``p_i H_i^(1)`` and ``p_i H_i^(2)``, the first two vectors
of its basis ``H_i`` times its share ``p_i``, as vectors of ``Z_q``: the client
needs them to raise the hashed points ``[omega]_1, [omega']_1`` to
``c_i,ipfe = (omega p_i, omega' p_i, psi_i, 0)_H_i``. A functional key holds
``k*_i,ipfe = (<S, y>, <U, y>, a0 z, 0)_H*_i``, four points of ``G2``, and
``H_i`` and ``H_i*`` are dual. So the client's rows applied to the coordinates
of ``k*_i,ipfe`` -- a multi-exponentiation in ``G2``, with no pairing and no
use of the policy -- give ``[p_i <S, y>]_2`` and ``[p_i <U, y>]_2``, for every
key.

*Reading an inner product.* With a key of weight on client ``i`` alone
(``y = e_i``, any policy) the client gets ``[p_i s_i]_2``, and with its own
``s_i``, ``[p_i]_2``. For any other key ``y`` and the ciphertexts of a round,
``e(prod_j t_j^(y_j), [p_i]_2) / (e([omega]_1, [p_i <S, y>]_2)
e([omega']_1, [p_i <U, y>]_2)) = [p_i <x, y>]_T``, and a bounded discrete log
in base ``e(g1, [p_i]_2)`` gives ``<x, y>``. For ``y = e_h`` that is the input
of an honest client ``h``. Neither key opens the round through the scheme
(``decrypt`` refuses both with ``FEKeyError`` when the round's attributes do not
satisfy their policies).

*Why Thm. 14 does not hold.* Def. 9 admits both keys: their policies are not
satisfied by the challenge attributes, so its condition on the inner products
does not apply, and corrupting one client is allowed. The adversary reads an
honest client's challenge input, so the notion of Def. 12 fails. The proof
(App. B.4) changes the rows given to a corrupted client in the step from G4 to
G5 and checks that the decryptions agree in the two games, not these
openings, which differ between them.

*What it means here.* The policy of a key does not hold against a decryptor
that colludes with one client: with two keys of any policies it reads what the
policies were to withhold. Without a corrupted client the rows stay secret and
the attack does not apply, but the scheme's claim is security under
corruption. ``sife/kp_ab_pairing`` (Sec. 4) is not affected: its public key
holds the group elements ``h1 + mu h2`` and ``h3``, not rows of ``Z_q``. A
key-side mask of the first two coordinates (fresh ``d_i``, ``e_i`` per key with
``sum_i p_i d_i = sum_i p_i e_i = 0``) stops this computation, but no proof of
the masked construction is known, and the library does not implement it. The
scheme is kept for functionality and performance experiments, and as the
record of the attack; the ML adapter's policy-gated aggregation runs on it and
is for experiments only.

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

*Sec. 5.2 is not affected by this attack* (it has the one above). In
``mcfe/kp_ab_pairing`` the tag hash includes the
set of attributes and all clients of a tag encrypt for one set. A key whose
policy that set satisfies gives the part of every client, so it opens the round
by itself; a key that does not gives no client's part. The attack needs a key
that gives the parts of some clients and not of others, which cannot happen
there, and there is no fifth coordinate. This says why the attack does not
carry over, not that the scheme is secure: it is not (see `mcfe/kp_ab_pairing:
a corrupted client opens every functional key`_).

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

Threshold key issuance (experimental)
-------------------------------------

:mod:`pyfe4ai.schemes.sife.cp_ab_pairing_keyissue` issues the functional
keys of ``sife/cp_ab_pairing`` jointly: a trusted dealer runs the setup,
Shamir-shares the master key ``U`` among ``s`` key servers and hands each
its share once; any ``t`` of them answer a request with masked partial keys,
whose product the requester checks against the public parameters and keeps
as an ordinary key of the scheme. The public parameters, ciphertexts and
decryption do not change, nor does the scheme's own security (SEL-SIM, one
challenge ciphertext). The protocol and its security argument are this
project's own; the argument (static corruption of fewer than ``t`` servers,
any number of corrupted users, honest users; a reduction to ACGU20 Thm. 3.2
that uses the key oracle once per request every honest server of the
request accepted) was reviewed, revised and re-checked within the project;
it is not published and not externally reviewed. With ``t = 1`` there is no
threshold: every server's share is the master key itself.

It holds only under conditions the library does not provide:

* **Mutually authenticated, confidential channels** (mTLS or an
  equivalent) between the requester and each server. A partial key is
  determined by the request, so anyone who can replay a user's request or read
  the ``t`` partial keys of one gets that user's key.
  ``issue(request, authenticated_requester)`` takes the peer identity the
  transport authenticated, never a value read from the request, and refuses
  a request for another ``sid``.
* **Partial keys are secret**: any ``t`` of one request are the requester's
  key. Send them to the authenticated requester only; do not log or store
  them (the library logs none).
* **The approval rule decides on the weights too.** ``approve(request,
  requester, history)`` sees the whole request and the requests this server
  accepted before, as read-only views; only a return value of ``True``
  approves, and a rule that raises refuses. The refusal chains the rule's
  exception, whose traceback reaches the server and its secret share: do not
  keep these exceptions or pass them to error reporters that capture local
  variables. A rule that checks the
  attributes only lets an approved user take ``eta`` keys of unit vectors and
  recover every plaintext its attributes open. With ``t - 1`` corrupted
  servers, keys are as safe as the most permissive honest server's rule.
* **The dealer is trusted** and deletes ``U``, the shares and the PRF keys
  after delivering them (the dealer object drops each share it delivers;
  Python cannot wipe memory, so run it in a process that ends).
* Fewer than ``t`` servers are corrupted, chosen before setup; there is no
  share refresh (a new sharing draws new PRF keys).
* **One object per server and per dealer.** A key server keeps its records
  (request digests and contents, request ids, the approval history) in memory
  only, and the library defines no format to persist them: a restarted server
  starts with none, approves earlier requests afresh, and gives a stateful
  rule an empty history (keep such state, e.g. counters, in the rule). A
  server issues in the process that made it only and cannot be copied or
  pickled. A copy of a dealer holds the same undelivered shares, so keep one
  dealer object.

A corrupted server can make an issuance fail, not hand out a wrong key: the
combiner checks every component with pairings, and any key that passes
decrypts correctly. It cannot tell which server cheated; retry with other
servers. Servers refuse a request digest used before for other content, and a
request id that the same ``sid`` used before for other content; a repeat of an
accepted request gets the same partial key without a new approval. Every server sees the requester, the attributes and the weights.

Attribute-based IPFE not provided
---------------------------------

Further attribute-based and identity-based IPFE constructions were read for
this library and are not implemented, for these reasons:

* **DP23** (Datta and Pal, PKC 2023; ePrint 2023/565, its only version as of
  2026-11, of 2023-04-20): two decentralized multi-authority schemes (Secs. 5
  and 6). Both are broken in their own static-corruption model by the attack
  below, which has not been reviewed independently. Two users with
  different GIDs and one corrupted authority whose attribute appears in the
  policy decrypt together, although none of them satisfies the policy, also
  with keys for ``u`` such that ``<v0 - v1, u> != 0``. The corrupted
  authority divides its own key out of its row of the column-wise ciphertext
  part ``C4`` and obtains ``[f_j]_1``, the random value of column ``j`` that
  all rows share; with it each user removes the GID-dependent factor from its
  row values, and the rows of different users then combine. The smallest
  example is the policy ``and(a, b, c)`` with the Lewko-Waters matrix,
  authority ``b`` corrupted, one user holding ``a`` and another holding ``c``.
  With honestly compiled matrices the attack needs the corrupted authority's
  rows to reach the columns of the users' rows (it succeeded on 14 of 99
  colluding configurations of random AND/OR policies, a lower bound); for
  every policy, the static model lets the adversary choose the matrix (a
  column change whose first row is ``e1``), and then it succeeded on all 99.
  The proofs of Thms. 5.1 and 6.1 replace the challenge matrix by the one of
  the zero-out lemma (Lemma 3.1, from RW15), which keeps the distribution of
  the row shares only, while these ciphertexts also expose the columns. The
  attack was checked on both schemes in an exponent model (it succeeded in
  all 30 trials of each, and in none without a corrupted authority). The size
  comparison of Tables 1.1 and 1.2
  counts every group element as 128 bits; a BLS12-381 ``GT`` element is 576
  bytes.
* **AGT21b** Sec. 3 (Agrawal, Goyal and Tomida, TCC 2021): multi-authority
  ABE composed with IPFE over composite-order groups (``N = p1 p2 p3``), for
  which the library has no backend. The paper mentions a prime-order version
  in the generic group model without giving it, and its proof is in the full
  version (ePrint 2020/1266), which was not checked. Its Lewko-Waters
  row-wise zero shares do not expose the columns as DP23 does (an argument,
  not checked).
* **ACGU20** Secs. 4 and 5 (identity-based IPFE from LWE): at 128 bits the
  parameters their proofs need give, by this project's estimate, a public
  trapdoor block of 4 to 15 GB that cannot be expanded from a seed,
  ciphertexts of 12 to 28 MB and keys of 3 to 28 MB depending on their
  encoding (Sec. 4; Sec. 5, estimated at one point, ``n = 4096``, gives
  ciphertexts of about 180 MB). App. A of the paper first requires ``m`` of about ``4.8 n log q`` and
  then takes ``m = 2 n log q``. Identity-based access control on lattices is
  offered as a directory of per-identity instances instead (below).
* **CT26** (Cini and Tairi, ePrint 2026/062, preprint of 2026-01-16) and
  **PD21** (Pal and Dutta, ePrint 2021/178, preprint): IPFE with attribute
  policies from LWE. The proof of CT26 floods noise twice with
  superpolynomial ratios (``q`` about ``V^2 P lambda1^11``), about 1.2 GB per
  coordinate at ``lambda1 = 2^40``. Its attribute version, like PD21, admits a
  single key that decrypts the challenge ciphertext; the lattice
  attribute-based IPFE constructions read for this library are all bounded in
  this way.

Identity directory (experimental)
---------------------------------

:mod:`pyfe4ai.ac.identity_directory` provides identity-based access control
on lattices without a new scheme. A key centre
(:class:`~pyfe4ai.ac.identity_directory.IdentityKeyCentre`) runs one instance
of ``sife/fullysec_ring_lwe`` per identity (reviewed in its default
element-wise layout; with ``ndarray_layout="packed"`` the directory runs the
packed layout, which is experimental) and publishes a directory of the
identities' encryption keys; an encryptor
(:class:`~pyfe4ai.ac.identity_directory.IdentityDirectory`) encrypts for an
identity, and only keys of that identity's instance decrypt it.

* The instances share the parameter record (``a``, ``q``, the widths) and
  draw their secrets independently, so an attack on one identity's instance,
  with any keys of the other identities, reduces to an attack on that
  instance alone; several challenge ciphertexts cost a hybrid over the
  ciphertexts. Encrypting one vector for several identities gives an "or" of
  identities; there is no "and".
* The key centre holds every identity's master key (key escrow, as in
  identity-based encryption), and it authenticates nobody: whoever calls
  ``get_decryption_keys`` decides who the identity is.
* Ciphertexts name their identity in the clear (no anonymity).
* The library checks that a directory is consistent and, given the digest
  the encryptor obtained over a trusted channel (``expected_digest``), that
  it is that directory. Without ``expected_digest`` a directory published by
  an attacker passes.
* Ciphertexts and keys carry the identity and the digest of its entry;
  decryption refuses (``FEKeyError``) unless they agree with each other and
  with the directory. These fields are claims, not authenticated: they catch
  a wrong key or a wrong directory, not tampering. A ciphertext whose fields
  are rewritten to another identity's is decrypted by that identity's key to
  a wrong value without an error: like the scheme underneath, the directory
  has no ciphertext integrity and no CCA security. Sign or MAC ciphertexts
  where integrity matters. Adding identities keeps earlier ciphertexts and
  keys valid. There is no revocation.
* The key centre applies the identity rule when it adds an identity, under
  its own Python's Unicode database: NFC, no control, format, surrogate or
  unassigned code points, at most 256 characters, case-sensitive, nothing
  normalised silently. Readers only check that an identity is a string of 1 to
  256 characters that encodes to UTF-8, so a directory loads the same under
  every Python. The rule
  does not catch confusable identities (a Cyrillic ``а`` for a Latin ``a``) or
  spaces: mapping people to identities is the application's.
* Keep one key centre object: a copy (``copy``, pickle, a forked child)
  adds identities of its own, so two copies can publish two directories with
  different instances under one identity.
* The directory, its checks and its records are experimental; the scheme
  underneath is unchanged.

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

Side channels
-------------

Nothing in PyFE4AI is guaranteed to be constant-time. The Python layer is
not: CPython and gmpy2 integers, the modular exponentiations of the DDH,
Damgård and Paillier schemes, the decimal encoding of keys in their records,
and the pure-Python ristretto255 fallback. Nor is the BLS12-381 group of the
native extension (arkworks), whose scalar multiplications are variable-time,
also for secret scalars.

For the lattice kernels of the native core (``rust/core``) and the garbling
of ``bcfe/gc_ristretto255``, the paths below were checked before 1.0 in two
ways:

* by reading the release build's code for branches on secret values,
  secret-indexed memory accesses and instructions whose latency depends on
  their operands (for the signed reduction also its inlined assembly after
  link-time optimisation);
* by statistical timing tests in the style of dudect (Reparaz, Balasch and
  Verbauwhede, DATE 2017; the targets of ``rust/core/examples/ct_dudect/main.rs``):
  fixed against random secret inputs, Welch's t-test, judged by the effect
  size against a constant-time control (SHAKE-256, whose two classes differ
  by -0.0012 % [-0.0037, +0.0014] at 8·10^7 batches), on an Apple M4 Max
  with nothing else timed. In the *seeded* design both classes run the same
  code, which draws the secret from a seed into the same buffer, the fixed
  class reusing one seed; in the *scalar* design the secret is one value.
  An earlier design, which copied the inputs of the two classes from two
  buffers, gave a false signal for the signed reduction; no row below rests
  on it.

Except for the exact Gaussian sampler, which is not constant-time (below),
no timing difference beyond the control's was found within the measured
sample sizes. That is evidence for
these paths on this machine and compiler, not a guarantee.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Path
     - Design, batches
     - Result
   * - reduction of signed values (noise, randomness, plaintext) mod ``q``
     - seeded, 8·10^7 batches of 4096 reductions
     - within the control; no branch on the sign in the assembly; a division
       only for values of at least ``q``, which noise never reaches
   * - Montgomery products (64-bit words), by a secret factor
     - scalar, 4·10^7 batches of 4096
     - no difference
   * - forward NTT (``n = 1024``, 60-bit ``q``)
     - seeded, 3·10^7
     - within the control
   * - the MCFE Ring-LWE encryption of 4096 elements (four chunks), the
       plaintext secret
     - seeded, 8·10^5
     - within the control
   * - Shamir sharing of 256 secrets, ``t = 3``
     - seeded, 5·10^7
     - within the control
   * - the half-gates garbling of ``bcfe/gc_ristretto255`` (data bits
       hardwired), the garbler's data bits secret; the same with the garbling
       seed (offset and labels) secret
     - seeded, 8·10^6 each
     - within the control
   * - the exact Gaussian sampler (FACCT, ``sigma = 1000``), 1024 values, the
       random bits secret; the default sampler of the noise of the lattice
       MIFE and MCFE schemes
     - seeded, 2·10^5; seeds of one shape (as many candidates, draws of
       ``y``, signs and outputs of ``y = 0``), 4·10^5; one shape with and
       without outputs of ``y = 0``, 6·10^5
     - **not constant-time.** The time follows the number of candidates: a
       fixed seed differs from random ones by -2.65 % [-2.67, -2.64], and a
       regression over 300 seeds on a second machine gives about 345 ns per
       call of the bit source and 117 ns per candidate, with residuals at the
       measurement noise. In a rejection sampler the number of candidates
       does not depend on the values returned. With the whole shape held
       equal, seven seeds of one shape still differ by +0.031 % [+0.021,
       +0.041] (about 480 ns, t = 6.1): the values of the candidates
       (returned or rejected) move the time by a few hundred nanoseconds per
       1024 samples. Two paths
       depend on the values: an output with ``y = 0`` (a multiple of ``k``,
       about one in 1024) skips the acceptance test (a seed with seven such
       outputs ran 395 ns faster than one with none, t = 5.4: the direction
       of the skip, but no more than seeds of one shape differ by), and the
       fixed-point exponential of the test starts from a floating-point
       estimate with correction loops and early returns, a few nanoseconds
       per candidate by the code (not measured on its own). (An earlier test held only the number of random-bit
       calls equal, which lets candidates trade against redraws of ``y``;
       its -0.067 % is about 9 to 14 candidates.)

Not checked yet, in the native core: the RNS products and the CRT
(``rns.rs``, used by default by the lattice MIFE, MCFE, dMCFE and tMCFE
schemes), the dealing of replicated key shares (``replicated.rs``), the
dense-LWE encryption and key terms, the exact Gaussian sampler of wide
widths (``facct_wide.rs``), the arithmetic of 256-bit words (moduli of 125
bits or more), decryption, and the threshold partial decryptions.
Outside it, nothing was measured: the label selection by ``w`` and the label
encryption of ``bcfe/gc_ristretto255`` were read but not timed, and in the
native ristretto255 code the scalar multiplications are curve25519-dalek's,
which is designed to be constant-time (not checked here), the encoding of a
plaintext branches on its sign, and decryption's bounded discrete logarithm
(baby-step giant-step) takes a time that depends on the result.

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

.. _bcfe-security:

Bounded-key FE
--------------

``bcfe/gc_ristretto255`` (:doc:`api/bcfe`) is **experimental** and
**adapted**. It follows the single-key scheme of Sahai and Seyalioglu (SS10,
ACM CCS 2010, Sec. 3.2 and 3.3) with the half-gates garbling of Zahur,
Rosulek and Evans (ZRE15, EUROCRYPT 2015, Fig. 2), with changes of its own:
one instance (*slot*) per issued function, the data bits folded into the
garbling by default (``data_wires = "hardwired"``), the public keys of the
unselected positions hashed to the group, and an instance salt in the gate
hash. Its security argument is the project's own: the 0.6 security argument,
reviewed internally twice; no external review.

*What is claimed.* IND security in the random-oracle model -- for the gate
hash, the hash that masks the labels and the hash to the group -- under the
strong Diffie-Hellman assumption in ristretto255 (Abdalla, Bellare and
Rogaway, CT-RSA 2001). The adversary chooses the functions of the keys
adaptively, corrupts key holders, registers public keys of its own as a key
holder of ``key_mode = "h2"`` and sees any number of challenge ciphertexts;
it cannot tell which of two inputs ``x0``, ``x1`` was encrypted as long as
every key it holds gives the same result on both, ``C(w, x0) = C(w, x1)``,
for every challenge encrypted to that key's slot. A slot made after a
ciphertext has no component in it, so its key puts no condition on that
ciphertext. Both data-wire modes are covered: ``"labels"`` is SS10's form,
to which the half-gates argument of ZRE15 applies directly; the default
``"hardwired"`` rests on a lemma of the argument of our own, for gates with
one input the garbler knows.

*The concrete bound.* The advantage is at most ``(1 + max_keys)`` times that
of strong DH, plus about ``q / 2^127`` for an adversary of ``q`` hash
queries (the chance that one of them hits a label offset of the garbling),
plus negligible terms (an ephemeral key guessed in advance, two ciphertexts
with one nonce). The main term grows with neither the circuit nor the
number of ciphertexts. Under the usual estimates
-- strong DH as hard as in the generic group of order about ``2^252``, and
running time equal to the number of hash queries -- a 16-bit linear model
with ``2^30`` ciphertexts and 64 slots comes to a bound of about ``2^-61``
for an adversary of ``2^64`` operations and about ``2^-45`` for one of
``2^80``. The limit is the 128-bit label itself (127 random bits of the
offset), whatever the circuit and the ciphertexts. These are estimates in
the models above, not a theorem, and the scheme is not called "128-bit
secure". Longer labels would tighten the bound, at the cost of larger
tables; ``label_bits`` takes 128 only.

*Read before use:*

* **Total issuance bounded by** ``max_keys``. Every distinct (requester,
  function) takes a slot, up to ``max_keys`` slots, and a ciphertext holds
  one component per slot it is encrypted to. This is a bound on the keys
  issued in all, *not* the collusion-bounded notion of Gorbunov,
  Vaikuntanathan and Wee (GVW12, CRYPTO 2012) or of Ananth and
  Vaikuntanathan (AV19, TCC 2019): any number of the issued keys may collude.
* **Keys decrypt only ciphertexts made after their slot existed.** A key
  issued later has no component in an earlier ciphertext
  (:class:`~pyfe4ai.core.exceptions.FEKeyError`, saying so); the data must be
  encrypted again under the new public parameters.
* **The key holder learns its own** ``w``: no function privacy. The key
  carries ``w``, and the authority sees it at issuance. The scheme hides
  ``x`` from the key holders (they learn ``C(w, x)``) and ``w`` from the
  encryptors (the slot list does not show it).
* **Classical security only: harvest now, decrypt later.** The labels are
  encrypted with hashed ElGamal over ristretto255: a ciphertext recorded
  today gives ``x`` up to a future quantum computer.
* **Honest encryptor assumed.** A malicious encryptor can (1) garble another
  circuit, so that a key holder's result is any function of ``w`` it likes,
  which leaks ``w`` once the result is published or returned to it; (2)
  cause a selective failure, a bad label at one position, and learn a bit of
  ``w`` from whether the decryption works; (3) use a different ``x`` for each
  slot, so that the results of the key holders disagree. The library
  reports every failure inside a component as one error
  (:class:`~pyfe4ai.core.exceptions.FESchemeError` with
  :data:`~pyfe4ai.schemes.bcfe.gc_ristretto255.DECRYPTION_FAILED`, whatever
  the cause). Do not send results, or whether a decryption worked, back to
  an encryptor, and do not publish raw results; take ciphertexts from
  encryptors you trust. Nothing detects (3).
* **The library cannot authenticate requester ids.** The deduplication is
  per ``requester``: the same requester asking again for the same ``w`` gets
  the same key (h1, with ``keep_issued_keys``) or the same grant (h2),
  another requester a slot of its own, and once ``max_keys`` slots exist
  every request gets the same error, so that the deduplication tells nothing
  about the functions of others. This holds only if callers pass an id that
  they have authenticated: with an id anyone can claim, a requester takes
  another's key (h1, with ``keep_issued_keys``) or tests whether another
  requester holds a key for a given ``w``.
* **The issuance policy is the authority's decision, not cryptographic
  access control.** ``issuance_policy`` is evaluated by the authority on the
  attributes the caller passes, before any slot is taken. The public slot
  record carries the digest of the policy; the digest of the attribute set
  stays in the authority's private record for its audit (unsalted, a public
  copy would let anyone test guesses of a requester's attributes). It
  decides whom the authority builds a slot for; it is as good as the
  authority and the attributes it is given. The cryptography enforces only
  that a key computes ``C(w, .)``; no ciphertext carries a policy.
* **The public parameters are trusted as distributed.** An encryptor
  encrypts to every slot of the slot list it holds (or to those it selects,
  ``encrypt(x, slots=...)``), so a forged slot list, with a slot whose
  secret keys the forger knows, gives ``x`` to the forger. The
  ``pp_digest`` covers the slot list but is not a signature, and the slot
  list is not signed. Get the public parameters over an authenticated
  channel, pin the ``pp_digest`` you trust, and check
  ``slot_list_version`` and the slots you encrypt to.
* **The authority.** With h1 it keeps every key it issued by default
  (``keep_issued_keys``) and can decrypt those slots; its directory holds
  them and the slot secrets: keep it as a master key. With h2 it never sees
  the holder's secret keys and cannot decrypt the holder's slot on its own.
  No secret key of an unselected position exists in either mode, so the
  authority and a key holder together cannot open a second set of labels;
  the authority can still make slots for any ``w`` it likes, up to
  ``max_keys`` and to the slots the encryptors select.
* **One copy of the authority's directory in use.** The bound on the keys
  issued holds per directory: two copies of it (a second host, a backup
  restored while the original goes on) each issue up to ``max_keys`` slots
  and hand out the same slot ids again. Each slot list still holds at most
  ``max_keys`` slots and each slot one key, so what a ciphertext reveals is
  bounded as before, but the count of keys issued in all is not. Issue from
  one copy only, and keep the directory free of macOS ACLs (the library reads
  the mode bits only; see :doc:`api/bcfe`, "Issuance").
* **Not constant-time.** The Python backend makes no constant-time claim and
  cannot wipe its copies of secrets. The native implementation selects by
  the bits of ``w`` and ``x`` with masks, draws its randomness from the
  operating system and wipes ``Delta``, the seeds, ``r`` and the labels, but
  its timing has not been tested.

Not covered by the argument: simulation security, the secrecy of ``w`` from
its key holder, a malicious encryptor, a forged slot list, side channels and
quantum adversaries.

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
