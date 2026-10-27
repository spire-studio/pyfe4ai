Threshold decryption and the ML adapter
=======================================

A threshold scheme shares every functional key among ``s`` decryption
servers: any ``t`` of them decrypt together, and fewer learn nothing beyond
the result. This page walks through the threshold flow, the threshold MCFE
over Ring-LWE with its split key shares, the accounting of the flooding
noise, and federated learning with :mod:`pyfe4ai.ml`, and lists what fails
and why.

The servers are **honest-but-curious**: a partial decryption carries no
proof of correctness, so one server that deviates changes the result
unnoticed (see `Several enrolled sets`_). The threshold layers of
tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE are the
project's own constructions (status *experimental*; :doc:`security`).

Roles and flow
--------------

* The **key generator** (the key authority) sets up the scheme, hands every
  client its encryption key and every server its key share of a functional
  key: ``get_decryption_keys(sid, credentials={"fusion_weight": y, ...})``.
* The **clients** encrypt, as in the scheme without a threshold.
* An **enrolled set** of at least ``t`` servers is chosen; each enrolled
  server computes a partial decryption with ``share_decrypt``, for that
  set.
* A **combiner** (any party; it holds no secret) adds up the partial
  decryptions with ``combine_decrypt`` and decodes the result.

With the threshold MCFE over DDH (``mcfe/ddh_threshold``, reviewed):

.. code-block:: python

   import pyfe4ai

   x = {"c0": [1, 2], "c1": [3, -1], "c2": [0, 4]}
   y = {"c0": [2, 1], "c1": [1, 1], "c2": [-1, 3]}
   servers = ["s0", "s1", "s2"]

   kg, TMCFE = pyfe4ai.create("mcfe/ddh_threshold", {
       "eta": 2, "n": 3, "lst_nid": list(x), "s": 3, "t": 2, "lst_sid": servers})
   kg.setup()
   pp = kg.get_public_parameters()

   label = "round-1"
   cts = {nid: TMCFE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
          for nid in x}

   def partials(enrolled):
       out = {}
       for sid in enrolled:
           dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y, "label": label})
           server = TMCFE({"id": sid, "keys": {"pp": pp, "dk": dk}})
           out[sid] = server.share_decrypt(cts, y, dk, enrolled, label)
       return out

   combiner = TMCFE({"id": "combiner", "keys": {"pp": pp}})
   print(combiner.combine_decrypt(partials(["s0", "s2"])))  # 18.0

The arguments of ``share_decrypt`` follow the scheme: the clients'
ciphertexts, then the weights of the key (``mife/ddh_threshold``,
``mife/lwe_threshold``, ``mcfe/ddh_threshold``) or the credentials
``{"fusion_weight": y, "label": label}`` (``mcfe/lwe_threshold``,
``mcfe/ring_lwe_threshold``), the server's key share, the enrolled set, and
the label for ``mcfe/ddh_threshold`` and
``mcfe/fh_multi_ipe_pairing_threshold``. The examples
``mcfe_threshold_demo.py``, ``mcfe_lwe_threshold_demo.py``,
``mife_lwe_threshold_demo.py``, ``mcfe_fh_multi_ipe_threshold_demo.py`` and
``mcfe_ring_lwe_threshold_demo.py`` in the repository's ``examples/`` run
each of them. Server and client ids are ``str`` or ``int``, one type per
list (:doc:`configuration`).

Several enrolled sets
~~~~~~~~~~~~~~~~~~~~~

``combine_decrypt`` also takes a list of ``{sid: partial}`` mappings, one
per enrolled set, and raises
:class:`~pyfe4ai.core.exceptions.FESchemeError` if their results disagree.
With the ``t + 1`` subsets of size ``t`` of ``t + 1`` servers, every server
is left out of some set, so one deviating server is detected (not
identified):

.. code-block:: python

   print(combiner.combine_decrypt([partials(["s0", "s1"]), partials(["s1", "s2"]), partials(["s0", "s2"])]))

Sharings
~~~~~~~~

The DDH and pairing threshold layers share the key with Shamir's sharing
and interpolate in the exponent. The LWE layers (``mife/lwe_threshold``,
``mcfe/lwe_threshold``, ``mcfe/ring_lwe_threshold``) flood every partial
decryption with noise and share the key

* with Shamir's sharing for ``t = s``;
* with a replicated sharing for ``t < s`` (``sharing="replicated"``, the
  default): Shamir shares with ``t < s`` let partial decryptions reveal the
  decryption noise modulo small integers (the leak of v0.3.0; see ``SECURITY.md`` and
  :doc:`upgrading`). A server then holds ``C(s - 1, t - 1)`` key
  pieces; the opt-in ``sharing="replicated-seeded"`` expands all of them but
  one from secret seeds (smaller key shares, computational instead of
  perfect hiding).

An explicit ``sharing="shamir"`` with ``t < s`` is refused under the
``standard`` profile and warns under ``toy``; servers refuse such key shares
unless their config has ``legacy_shamir="warn"``.

Threshold MCFE over Ring-LWE
----------------------------

``mcfe/ring_lwe_threshold`` is the threshold scheme for large vectors and
gradients: one ciphertext carries a whole tensor in chunks, and the Rust
extension runs it natively. Its defaults (both profiles) are binary
ciphertexts (``ct_format="binary"``), an RNS modulus
(``ring_modulus="rns"``) and **split key shares** (``key_shares="split"``):

* the **weight part** -- the label-independent part of the key, for one
  weight vector -- is issued once:
  ``kg.get_weight_key_share(sid, credentials={"fusion_weight": y})``, and
  each server keeps it with ``add_weight_key_share``;
* per round, ``get_decryption_keys(sid, credentials={"fusion_weight": y,
  "label": label})`` issues only the **label part**, the shares of the
  label's terms (``key_part="full"`` asks for a whole key share instead).

.. code-block:: python

   import numpy as np
   import pyfe4ai

   clients, servers = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
   kg, TMCFE = pyfe4ai.create("mcfe/ring_lwe_threshold", {
       "eta": 1, "n": 3, "lst_nid": clients, "s": 3, "t": 2, "lst_sid": servers,
       "ring_n": 16, "bound_x": 50, "bound_y": 2})                 # toy sizes
   kg.setup()
   pp = kg.get_public_parameters()
   weights = {"c0": [1], "c1": [2], "c2": [1]}
   enrolled = ["s0", "s2"]

   # once per weights: each server keeps its weight key share
   srv = {sid: TMCFE({"id": sid, "keys": {"pp": pp}}) for sid in enrolled}
   for sid, server in srv.items():
       server.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": weights}))

   client = {nid: TMCFE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in clients}
   combiner = TMCFE({"id": "combiner", "keys": {"pp": pp}})

   # a round of single vectors: eta rows per client, here a batch of two values
   label = "round-1"
   x = {"c0": [[3, -1]], "c1": [[2, 5]], "c2": [[-4, 0]]}
   cts = {nid: client[nid].encrypt(x[nid], label) for nid in clients}
   creds = {"fusion_weight": weights, "label": label}
   partials = {sid: server.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), enrolled)
               for sid, server in srv.items()}
   print(combiner.combine_decrypt(partials))  # [3, 9]

   # a round of gradients: ndarrays, packed into chunks
   label = "round-2"
   grads = {nid: np.random.default_rng(i).integers(-5, 6, (3, 7)) for i, nid in enumerate(clients)}
   cts = {nid: client[nid].encrypt_lst_ndarray([grads[nid]], label=label) for nid in clients}
   creds = {"fusion_weight": weights, "label": label,
            "n_chunks": TMCFE.key_chunks(cts["c0"]), "label_term": False}
   parts = {sid: server.compute_lst_ndarray_ct(cts, creds, kg.get_decryption_keys(sid, credentials=creds), enrolled)
            for sid, server in srv.items()}
   total = combiner.decrypt_lst_ndarray_ct(parts)[0]
   print(np.array_equal(total, sum(weights[nid][0] * grads[nid] for nid in clients)))  # True

For ndarrays the label part covers the chunks (``n_chunks``, from
``key_chunks`` of a client's ciphertexts); ``label_term=False`` leaves out
the term that only single ciphertexts of the label need.

What the key generator keeps:

* the sharings of the last 64 weight vectors
  (:data:`~pyfe4ai.schemes.threshold.store.WEIGHT_SHARINGS_KEPT`); weights
  issued again after that get a new sharing, and their servers need the new
  weight part (``has_weight_key_share(dk)`` tells a server whether it keeps
  the weight part a label part completes);
* the label parts of the last 16 labels per weight vector
  (:data:`~pyfe4ai.schemes.threshold.store.LABEL_PARTS_KEPT`); a label part
  asked for again later is a **new sharing** of the same key.

Sharing ids
~~~~~~~~~~~

Every sharing the key generator issues -- whole key shares, or the label
parts of one label -- has a random ``sharing_id``, and every partial
decryption carries the id of its key share. Partials of two sharings of one
key would decode to a wrong value inside the decoding bound instead of
failing, so the combiner refuses them with
:class:`~pyfe4ai.core.versions.MixedSharingsError` (an ``FESchemeError``):
fetch the label parts again for all enrolled servers and decrypt again.
**Store and transmit the** ``sharing_id`` **with every key share and partial
decryption**: partials without one (as from key shares of earlier versions)
combine only with each other, so a storage or transport layer that drops it
on every side would let two sharings be mixed unnoticed.

Accounting of the flooding noise
--------------------------------

By default (``flooding_accounting="per-sample"``) the flooding noise of a
partial decryption is within statistical distance ``2^-stat_param`` of its
simulation per flooded value, with no limit on the number of partial
decryptions; over a run the distances add up (the number of flooded values
times ``2^-stat_param``). The opt-in ``flooding_accounting="kl"``
(``mife/lwe_threshold`` and ``mcfe/ring_lwe_threshold``, ``t = s`` with
Shamir's sharing only) bounds a whole run of at most
``max_partial_decryptions`` partial decryptions per server instead, for a
smaller modulus and a weaker guarantee: a Hellinger distance below
``2^(-stat_param/2)`` for the first ``max_partial_decryptions`` partial
decryptions of every server, about ``stat_param - 6.85`` bits in the
bit-security sense of Micciancio and Walter for decision games -- a total
statistical distance of about ``2^-65`` at ``stat_param = 128``, not
``2^-121`` (:mod:`pyfe4ai.schemes.threshold.kl`). Each server then counts
its partial decryptions in a
:class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget` that its
deployment creates and passes in the server's config:

.. code-block:: python

   import pyfe4ai
   from pyfe4ai.schemes.threshold.kl import PartialDecryptionBudget

   clients, servers = ["c0", "c1"], ["s0", "s1", "s2"]
   kg, TMCFE = pyfe4ai.create("mcfe/ring_lwe_threshold", {
       "eta": 1, "n": 2, "lst_nid": clients, "s": 3, "t": 3, "lst_sid": servers,
       "ring_n": 16, "bound_x": 50, "bound_y": 2,
       "flooding_accounting": "kl", "max_partial_decryptions": 1000})
   kg.setup()
   pp = kg.get_public_parameters()

   budgets = {sid: PartialDecryptionBudget(1000) for sid in servers}   # one per server
   srv = {sid: TMCFE({"id": sid, "keys": {"pp": pp}, "partial_budget": budgets[sid]}) for sid in servers}
   weights = {"c0": [1], "c1": [1]}
   for sid, server in srv.items():
       server.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": weights}))

   label = "round-1"
   cts = {nid: TMCFE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([[7, 1]], label)
          for nid in clients}
   creds = {"fusion_weight": weights, "label": label}
   partials = {sid: s.share_decrypt(cts, creds, kg.get_decryption_keys(sid, credentials=creds), servers)
               for sid, s in srv.items()}
   print(TMCFE({"id": "combiner", "keys": {"pp": pp}}).combine_decrypt(partials), budgets["s0"].remaining)  # [14, 2] 999

The guarantee holds only if a server's count covers all its partial
decryptions under the setup: a server that runs as several processes or
restarts must share or persist its count (for instance a subclass with an
atomic, durable ``consume``). A budget cannot be copied or pickled, and it
refuses to count in another process than the one that made it (after
``os.fork()`` parent and child would each count from the copied value).
``standard`` refuses ``mife/lwe_threshold`` with either accounting.

Federated learning with the ML adapter
--------------------------------------

:class:`~pyfe4ai.ml.FESchemeWrapper` runs a scheme's key generator and
gives the helpers of :mod:`pyfe4ai.ml` what they need:
:func:`~pyfe4ai.ml.encrypt_gradient` (a client) and
:func:`~pyfe4ai.ml.aggregate_gradients` (the aggregator), with
:func:`~pyfe4ai.ml.precompute_gradient` and
:func:`~pyfe4ai.ml.prepare_aggregation` for an offline phase before the
round. It holds the master key and plays every party, a single trust
domain meant for experiments. Its default policy:

* a client encrypts at most once per label (``ValueError`` otherwise);
* :func:`~pyfe4ai.ml.aggregate_gradients` needs a ciphertext and a non-zero
  weight for every enrolled client (``allow_partial=True`` relaxes this)
  and aggregates a label with one weight vector only;
* a gradient filed under another client than the one it names is refused.

These checks are per wrapper object, and they do not stop very unequal
weights, which separate two clients' gradients in one aggregate: the
weights are the key holder's choice. A pickled or copied wrapper carries the
master key and every client's key; ship a worker its client's own crypto
object instead (the scheme class built from ``wrapper.pp`` and
``wrapper.kg.get_private_keys(nid)``).

The threshold MCFE in the adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``mcfe/ring_lwe_threshold`` the wrapper also runs the ``s`` decryption
servers. :func:`~pyfe4ai.ml.prepare_aggregation` and
:func:`~pyfe4ai.ml.aggregate_gradients` take ``servers=`` (default: the
first ``t`` of ``lst_sid``) and give each enrolled server its label key
share of the round and, once per weights, its weight key share. The
messages of a deployment are those of the separate steps:
:func:`~pyfe4ai.ml.partial_decrypt_gradients` at each server and
:func:`~pyfe4ai.ml.combine_partial_decryptions` at the combiner, with
:class:`~pyfe4ai.ml.EncryptedGradient` and
:class:`~pyfe4ai.ml.ThresholdPartial` serialised as JSON:

.. code-block:: python

   import json
   import numpy as np
   from pyfe4ai.ml import (EncryptedGradient, FESchemeWrapper, ThresholdPartial, aggregate_gradients,
                           combine_partial_decryptions, encrypt_gradient, partial_decrypt_gradients,
                           prepare_aggregation)

   clients, servers = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
   wrapper = FESchemeWrapper("mcfe", "ring_lwe_threshold", {
       "eta": 1, "n": 3, "lst_nid": clients, "s": 3, "t": 2, "lst_sid": servers,
       "ring_n": 16, "bound_x": 50, "bound_y": 2}, precision=1)
   weights = {"c0": 1, "c1": 2, "c2": 1}
   grads = {nid: np.round(np.random.default_rng(i).uniform(-2, 2, (3, 7)), 1) for i, nid in enumerate(clients)}

   # in-process: the wrapper runs the enrolled servers
   enc = {nid: encrypt_gradient(grads[nid], wrapper, nid=nid, label="round-a") for nid in clients}
   total = aggregate_gradients(enc, wrapper, weights, label="round-a", servers=["s0", "s1"])

   # the messages of a deployment
   prepared = prepare_aggregation(wrapper, weights, shape=(3, 7), label="round-b", servers=["s1", "s2"])
   sent = {nid: json.dumps(encrypt_gradient(grads[nid], wrapper, nid=nid, label="round-b").to_dict())
           for nid in clients}
   received = {nid: EncryptedGradient.from_dict(json.loads(m)) for nid, m in sent.items()}
   wire = {sid: json.dumps(partial_decrypt_gradients(received, wrapper, prepared, server=sid).to_dict())
           for sid in prepared.servers}
   total_b = combine_partial_decryptions(
       {sid: ThresholdPartial.from_dict(json.loads(m)) for sid, m in wire.items()}, wrapper, prepared)
   print(np.allclose(total, total_b), np.allclose(total, sum(weights[n] * grads[n] for n in clients)))  # True True

If the label key shares of the enrolled servers turn out to be of
different sharings, :func:`~pyfe4ai.ml.aggregate_gradients` fetches them
again for all servers and decrypts once more; the separate steps raise
:class:`~pyfe4ai.core.versions.MixedSharingsError` (prepare the round
again). A :class:`~pyfe4ai.ml.PreparedAggregation` holds key material (the
servers' label key shares). The adapter refuses ``flooding_accounting="kl"``
and the other threshold and decentralized variants, which are used through
their scheme classes as above.

What fails, and why
-------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Error
     - Cause and remedy
   * - ``FESchemeError``: partial decryptions disagree on the ciphertexts
       they decrypted
     - the servers of one enrolled set decrypted different ciphertexts (a
       server answered another round, or deviated); in earlier versions
       ``combine_decrypt`` returned ``None`` here
   * - ``FESchemeError``: insufficient threshold shares
     - fewer than ``t`` partial decryptions, or a partial computed for
       another enrolled set
   * - ``FESchemeError``: the results of several enrolled sets disagree
     - one server deviates (`Several enrolled sets`_)
   * - :class:`~pyfe4ai.core.versions.MixedSharingsError`
     - tMCFE-Ring-LWE partials of two sharings of one key (a label part
       issued again); fetch the label parts again for all servers
   * - ``FEKeyError``: a label key share needs this server's weight key share
     - call ``add_weight_key_share`` with the server's weight part of the
       same weights first
   * - ``FESchemeError``: the key share was derived for another label
     - each round needs its own label part
   * - ``FESchemeError``: need one ciphertext for every client of the key
     - the request lacks a client of the weights
   * - ``FESchemeError``: the ciphertext of a client is not authenticated for
       this server
     - tMIFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE servers decrypt only
       ciphertexts their clients tagged for them, under the label they serve
   * - ``FEKeyError`` for a key share of v0.3.0, of another flooding or of
       other public parameters
     - key shares are bound to their public parameters (``pp_binding``) and
       to the flooding derivation; re-share the keys with this version
       (:doc:`upgrading`)
   * - ``FESchemeError`` for a Shamir key share with ``t < s``
     - re-share with the default replicated sharing, or create the server
       with ``legacy_shamir="warn"`` (``toy`` only)
   * - ``FESchemeError``: partial decryption limit reached
     - the server's ``PartialDecryptionBudget`` is spent (``"kl"``); a server
       without a budget, or with a budget made in another process, is
       refused the same way
   * - ``FEValidationError`` for a server or client id
     - ids are ``str`` or ``int``, one type per list; numpy integers are read
       as ints after setup (:doc:`configuration`)
   * - ``ValueError`` from the ML adapter
     - its policy: a label used twice by one client, a missing client or a
       zero weight without ``allow_partial=True``, a gradient filed under
       another client, or servers or weights keyed by values that are no ids
   * - ``NotImplementedError`` from the ML adapter
     - MIFE (no labels) for gradients, or a threshold or decentralized
       variant other than ``mcfe/ring_lwe_threshold``
   * - ``None``
     - the result is outside the decoding range (bounds too small for the
       data; with an :class:`~pyfe4ai.core.exceptions.FEWarning`), or a key
       was requested for an unknown server (with an ``FEWarning``) or an
       unknown client (with an ``FEWarning`` in the MIFE and MCFE DDH
       schemes and their ristretto255, threshold and decentralized variants;
       the other schemes return ``None`` without a warning)
