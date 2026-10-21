Threads, Processes and Copies
=============================

What the library keeps between calls, which objects several threads may
use at once, and what ``copy.copy``, ``copy.deepcopy``, ``pickle`` and
``os.fork()`` do to it.

State lives in the objects that own it: a key generator keeps its master
key, the sharings it issued and its caches; a crypto object its keys and
key caches; a threshold server its weight key shares and their caches; a
precomputation its secret randomness. Process-wide state is limited to the
selection of the backend and the sampler, tables derived from public
parameters, the locks of the threshold stores, the OS random buffer and the
registries filled at import (`Process-wide state`_).
``tests/test_global_state.py`` keeps that list closed: a new module-level
cache, lock or registry fails it unless it is added there and here.

Selection: backend, threads, sampler
------------------------------------

* :func:`pyfe4ai.backend.set_backend` and ``PYFE4AI_BACKEND`` select the
  backend for the process; ``with pyfe4ai.backend.backend(name):`` selects
  it for the current thread inside the block (a ``threading.local``):
  threads started inside the block use the process-wide selection.
* :func:`pyfe4ai.backend.set_threads` and ``PYFE4AI_THREADS`` set how many
  threads *one* native call may use over the chunks of a tensor (Rust
  threads, with the GIL released), for the whole process. It does not limit
  how many Python threads call the library: ``P`` Python threads running
  native calls at once use up to ``P`` times that many threads.
* :func:`pyfe4ai.primitives.sampling.set_gaussian_sampler` and
  ``PYFE4AI_GAUSSIAN_SAMPLER`` select the Gaussian sampler for the process;
  ``with gaussian_sampler(name):`` for the current context (a
  ``contextvars.ContextVar``): threads started inside the block use the
  process-wide one. The schemes that record their sampler in their public
  parameters ignore the process-wide choice
  (:func:`~pyfe4ai.primitives.sampling.scheme_sampler`).

Set the process-wide values once, before threads start: every call reads
the current value, and changing it while other threads run changes their
next calls.

Objects several threads may share
---------------------------------

These are made for concurrent use, and the review tests run them from
threads:

* **Threshold MCFE over Ring-LWE** (``mcfe/ring_lwe_threshold``): a server
  object answering :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.share_decrypt`
  and :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.add_weight_key_share`
  from several threads (its weight key shares, the cache of their NTTs and
  the cache of the sums of replicated pieces are guarded by the store
  locks); the key generator's
  :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWEKeyGenerator.get_weight_key_share`
  and the label parts of
  :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWEKeyGenerator.get_decryption_keys`
  (a label part is issued only while its sharing of ``sk_y`` is the
  weights' current one; with more weights in use at once than the
  generator keeps, ``WEIGHT_SHARINGS_KEPT`` = 64, a request whose sharing
  is evicted during each of its attempts fails with
  :class:`~pyfe4ai.core.exceptions.FESchemeError`). Whole key shares
  (``key_part="full"``) of one label are not synchronised: two threads
  asking for the first time may get shares of two sharings, whose partial
  decryptions combiners refuse
  (:class:`~pyfe4ai.core.versions.MixedSharingsError`); issue them from one
  thread.
* The threshold LWE layers (``mife/lwe_threshold``,
  ``mcfe/lwe_threshold``): a server's partial decryption reads the key
  share, and the cache of the sums of replicated pieces is guarded the same
  way. The key generators issue one sharing per weights: of several threads
  asking for the key shares of the same weights for the first time, each may
  deal, but all of them issue the sharing stored first. (These key shares
  carry no sharing id: shares of two sharings of one key would combine to a
  wrong value without an error, so the key generator never hands out two.)
* :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget`: checking
  and counting a partial decryption is one step under its lock.
* One-time precomputations (the offline phases): of several threads using
  one, exactly one succeeds.
* :class:`~pyfe4ai.ml.FESchemeWrapper`: a client's label is reserved under
  the wrapper's lock before it encrypts, and its in-process threshold
  servers are made once.
* :class:`~pyfe4ai.primitives.sampling.XofRandom`: every draw claims its own
  block (which thread gets which block depends on the scheduling).

Everything else is not synchronised. Crypto objects only read their keys
when they encrypt or decrypt, and their caches (the key in the form
encryption uses, the native key) are built on first use: several threads
may share one, and their first calls may each build the cache (the same
values; one of them is kept). A key generator's ``setup()`` replaces its
keys: do not call it while other threads use the generator. When in doubt,
use one object per thread, or a lock of your own.

The native calls release the GIL; the Python implementation holds it, so
threads speed up the native path only.

Copies and pickles
------------------

``pickle``, ``copy.copy`` and ``copy.deepcopy`` use the same state
(``__getstate__``):

* **Caches that may hold native objects are left out** and rebuilt on first
  use: the prepared master key of a MIFE / MCFE Ring-LWE key generator
  (``_native_msk``), the encryption keys of the Ring-LWE crypto objects
  (``_enc_key``, ``_nkey``) and a threshold server's NTTs of its key shares
  (``_hat_cache``); a :class:`~pyfe4ai.ml.FESchemeWrapper` leaves out its
  lock, its crypto objects and its servers (servers get their weight key
  shares again).
* **Everything else is kept**, secrets included: keys, the sharings a
  threshold key generator issued (``dict_weight_shares``, ``dict_dk``), a
  server's weight key shares, and the caches the objects keep in plain
  Python values -- a threshold server's sums of replicated pieces with the
  key shares they were summed from (``_enrolled_share_cache``), the key row
  of a dense-LWE MCFE client (``_dense_key_<backend>``), the scalars of a
  ristretto255 MCFE key (``_s_cache``) and ``g^u`` of a MIFE-DDH key
  (``_g_u``). A pickle of a key generator, client or server is as secret as
  its keys.
* A **shallow copy** (``copy.copy``) shares those stores and caches with the
  original (the same dictionaries): the process-wide store locks keep the
  two consistent, but they are one store. A deep copy or a pickle has its
  own.
* The threshold MCFE over Ring-LWE objects carry the version of their
  state; a state of another version is refused when loaded.
* **Never copied or pickled** (``TypeError``): one-time precomputations (a
  copy would be a second use of the same randomness), a
  :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget` (a copy
  would count separately) and the BLS12-381 pairing group (it holds the
  extension module). Locks are never part of a pickle.

Processes and ``fork``
----------------------

The library registers one ``os.register_at_fork`` handler
(:mod:`pyfe4ai.core.forksafe`). In a forked child it

* replaces the process-wide locks with new ones (``_XOF_LOCK``,
  ``_contexts_lock``, ``_SHARE_CACHE_LOCK``, ``_STORE_LOCK``): a lock that
  another thread of the parent held would otherwise stay held in the child;
* drops the buffered OS random bytes of the samplers (``_OS_RANDOM``) and
  gives the buffer a new lock, so that the two processes never use the same
  bytes;
* marks every one-time precomputation used and drops its secret parts (the
  native ones also refuse to run in another process than the one that made
  them).

Nothing else changes in the child: it has a copy of every object of the
parent, keys and caches included, and the public tables. The native code
keeps no random generator state between calls (one DRBG per call), so a
fork duplicates none. The locks of individual objects
(:class:`~pyfe4ai.ml.FESchemeWrapper`,
:class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget`) are not
renewed: fork while no other thread is inside them, or make the objects in
the child. A :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget`
copied into a child counts separately from the parent's: after the fork the
parent and the child each count from the copied ``used``, so together they
can produce more partial decryptions than ``max_partial_decryptions`` and
the KL flooding guarantee no longer holds. Keep a server's budget in the one
process that answers for it (or share and persist its count). Python warns about ``fork()`` in a process with threads; the
``spawn`` and ``forkserver`` start methods of ``multiprocessing`` avoid
all of the above, and their workers rebuild every cache.

Process-wide state
------------------

Every module- or class-level value that changes after import, or that the
library fills on first use (``tests/test_global_state.py`` checks this
list). "Fork": what a forked child does with it. None of it holds secrets
except the OS random buffer (random bytes not yet used).

.. list-table::
   :header-rows: 1
   :widths: 30 46 24

   * - Name (module)
     - What, bound
     - Fork
   * - ``_state``, ``_default_backend``, ``_threads`` (:mod:`pyfe4ai.backend`)
     - the backend per thread and per process, the threads per native call
     - kept
   * - ``_random_override``, ``_seed_override``, ``_bigint_widen``, ``_ext`` (:mod:`pyfe4ai.backend`)
     - test hooks: the native calls' random bytes and DRBG seed, the native
       sampler's margin, the extension module; ``None`` / 0 / the
       extension in use
     - kept
   * - ``_PYTHON``, ``_NATIVE`` (:mod:`pyfe4ai.backend`)
     - the two implementations; each keeps its Python rings per ``(q,
       ring_n)`` (unbounded: one per parameter set)
     - kept
   * - ``_contexts``, ``_contexts_lock`` (:mod:`pyfe4ai.backend.native_impl`)
     - native ring contexts per ``(q, ring_n, factors)`` (NTT tables;
       unbounded) and their lock
     - lock renewed
   * - ``_facct_table``, ``_facct_wide_table``, ``_bigint_width``
       (:mod:`pyfe4ai.backend.native_impl`); ``_facct_params``
       (:mod:`pyfe4ai.primitives.sampling`)
     - sampler tables per Gaussian width (LRU, 64 / 64 / 256 / 64)
     - kept
   * - ``_ec_group``, ``_bls_kernels`` (:mod:`pyfe4ai.backend.native_impl`)
     - a module and the extension's BLS12-381 functions, set on first use
     - kept
   * - ``_default_sampler``, ``_sampler_override``
       (:mod:`pyfe4ai.primitives.sampling`)
     - the Gaussian sampler per process and per context
     - kept
   * - ``_XOF_LOCK`` (:mod:`pyfe4ai.primitives.sampling`)
     - guards the block claims of every ``XofRandom`` (never held while
       drawing)
     - renewed
   * - ``_OS_RANDOM`` (:mod:`pyfe4ai.primitives.sampling`)
     - OS random bytes read 64 KiB at a time, and its lock
     - bytes dropped, lock renewed
   * - ``_CSPRNG`` (sampling, integer groups, sharing, several schemes)
     - ``random.SystemRandom``: reads the OS, keeps no state
     - kept
   * - ``_twiddle_cache``, ``_ntt_root_cache``, ``RingNTT._cache``,
       ``_rns_factors`` (:mod:`pyfe4ai.primitives.lattice.ring_lwe_utils`)
     - NTT tables, roots and rings per ``(q, ring_n)``, and the factors of
       the registered RNS moduli (unbounded: one per parameter set)
     - kept
   * - ``SeededMatrix._shared`` (:mod:`pyfe4ai.primitives.lattice.lwe_matrix`)
     - the public matrices of the seeded dense-LWE schemes, per seed, held
       weakly (dropped with the last object using them)
     - kept
   * - ``_BASE_COMB`` (:mod:`pyfe4ai.primitives.groups.ristretto255`),
       ``_dlog_table`` (:mod:`pyfe4ai.primitives.groups.ec_group`),
       ``_cached_dlog_table`` (:mod:`pyfe4ai.primitives.groups.dlog_solver`),
       ``_small_primorial`` (:mod:`pyfe4ai.primitives.groups.integer_groups`)
     - the base-point table, baby-step tables (LRU, 16 / 32) and a constant
     - kept
   * - ``PAIRING_IMPORT_ERROR``, ``_CharmPairingGroup``, ``_charm_pair``,
       ``_CharmGroup`` (:mod:`pyfe4ai.primitives.pairing.pairing_backend`)
     - set at import, by whether charm is installed
     - kept
   * - ``_SHARE_CACHE_LOCK``, ``_STORE_LOCK``
       (:mod:`pyfe4ai.schemes.threshold.store`)
     - guard the threshold stores, which are attributes of the key
       generators and servers; process-wide because a shallow copy shares
       its stores with the original
     - renewed
   * - ``_CONFIGS``, ``STANDARD_FIXED``, ``SchemeConfig.OPTIONS``,
       ``SchemeConfig.CRYPTO`` (:mod:`pyfe4ai.core.config`),
       ``Record._fields`` (:mod:`pyfe4ai.core.formats.record`)
     - the config class of every scheme (filled when the classes are
       defined), constants and class defaults
     - kept
   * - ``_HANDLERS``, ``_TRACKED`` (:mod:`pyfe4ai.core.forksafe`)
     - what a forked child runs, and the objects it notifies (held weakly)
     - run

The parameter cache on disk (``PYFE4AI_CACHE_DIR``, see
:func:`~pyfe4ai.core.base.cache_dir`) is shared by every process that uses
the same directory.
