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
``tests/state/test_global_state.py`` keeps that list closed: a new module-level
cache, lock or registry fails it unless it is added there and here.

Selection: backend, threads, sampler
------------------------------------

* :func:`pyfe4ai.backend.set_backend` and ``PYFE4AI_BACKEND`` select the
  backend for the process; ``with pyfe4ai.backend.backend(name):`` selects
  it for the current thread inside the block (a ``threading.local``):
  threads started inside the block use the process-wide selection.
* :func:`pyfe4ai.backend.set_threads` and ``PYFE4AI_THREADS`` set how many
  threads *one* native call may use over the chunks of a tensor, the
  elements of a BLS12-381 batch, the pairs of a product of pairings or the rows
  of a batch of products over prepared G2 points (Rust threads, with the GIL
  released), for the whole process. It does not limit
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
* The key generators of the threshold DDH and FH-MCFE layers
  (``mife/ddh_threshold``, ``mcfe/ddh_threshold``,
  ``mcfe/fh_multi_ipe_pairing_threshold``) issue one sharing per weights the
  same way (shares of two sharings would not combine).
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
encryption uses, the native key, the decoded public parameters) are built on
first use: several threads
may share one, and their first calls may each build the cache (the same
values; one of them is kept). A key generator's ``setup()`` replaces its
keys: do not call it while other threads use the generator. When in doubt,
use one object per thread, or a lock of your own.

The native calls release the GIL; the Python implementation holds it, so
threads speed up the native path only.

The batched pairing kernel
--------------------------

``decrypt_lst_ndarray_ct`` of ``mcfe/kp_ab_pairing`` and
``mcfe/kp_ab_pairing_per_client`` decrypts all elements of an array with one
functional key. Every element is paired with the same G2 points of the key, so
a call prepares them once and pairs one row of G1 elements per element over
them:

* :func:`~pyfe4ai.primitives.pairing.pairing_backend.prepare_g2` stores the
  lines of the Miller loops of the points (on BLS12-381, in the native
  extension; on a charm group it returns the points themselves), and
  :func:`~pyfe4ai.primitives.pairing.pairing_backend.multi_pair_prepared` takes
  one product of pairings per row over those lines, one final exponentiation
  per row, with the same results as ``multi_pair`` for each row.
* **Threads.** One call takes up to ``get_threads()`` threads over the rows,
  however few they are (a row costs at least a final exponentiation, far more
  than starting a thread); ``prepare_g2`` takes the points on that many threads
  from 16 points on. The results do not depend on the number of threads. The
  elements are decoded, paired and solved in slices of at most ``2**16``
  decoded group elements (``n (8 |S| + 5)`` per element, and at least one
  element per slice), so that the memory of the decoded elements does not grow
  with the size of the arrays, nor with the clients and their attributes as
  long as one element fits the budget. A slice is never smaller than one
  element: once ``n (8 |S| + 5)`` exceeds ``2**16`` (``n`` of 127 or more at 64
  attributes) it is one element, and the memory grows with ``n |S|`` again. A
  slice of few elements gives the threads few rows: 12 at ``n = 10`` and
  ``|S| = 64``. Eight threads still pay: on an ARM64 Mac the whole aggregation
  of 40 elements with ``|S| = 64`` took 2.4, 13.4 and 20.1 s on eight threads
  against 14.5, 72.7 and 92.0 s on one at ``n = 10``, 50 and 64, where a slice
  holds 12, 2 and 1 elements.
  A charm group pairs the rows one after the other in the calling thread.
* **State.** Nothing is kept between calls: a prepared set lives for one
  decryption and no object holds one, so none is part of the state of a crypto
  object or of its pickle. It has no text form, and is as sensitive as the key
  elements it comes from. It is read-only, so threads may pair rows over one
  set at once, and it is never copied or pickled (``TypeError``).

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
* **A threshold server's cache of the key shares callers passed in is left
  out** (``mife/lwe_threshold``, ``mcfe/lwe_threshold``,
  ``mcfe/ring_lwe_threshold``): the sums of replicated pieces for an
  enrolled set, with the key shares they were summed from
  (``_enrolled_share_cache``), are summed again on use. A pickle of an
  earlier version that holds the cache loads without it.
* **Everything else is kept**, secrets included: keys, the sharings a
  threshold key generator issued (``dict_weight_shares``, ``dict_dk``), a
  server's weight key shares, and the caches the objects keep in plain
  Python values -- the key row of a dense-LWE MCFE client
  (``_dense_key_<backend>``), the scalars of a ristretto255 MCFE key
  (``_s_cache``) and ``g^u`` of a MIFE-DDH key (``_g_u``). A pickle of a key
  generator, client or server is as secret as its keys.
* A **shallow copy** (``copy.copy``) shares those stores and caches with the
  original (the same dictionaries): the process-wide store locks keep the
  two consistent, but they are one store. A deep copy or a pickle has its
  own.
* The threshold MCFE over Ring-LWE objects carry the version of their
  state; a state of another version is refused when loaded.
* **Never copied or pickled** (``TypeError``): one-time precomputations (a
  copy would be a second use of the same randomness), a
  :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget` (a copy
  would count separately), the BLS12-381 pairing group (it holds the
  extension module) and a prepared set of G2 points
  (:func:`~pyfe4ai.primitives.pairing.pairing_backend.prepare_g2`, see `The
  batched pairing kernel`_). Locks are never part of a pickle.

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
  them);
* gives the copies of the objects that have a lock of their own a new one
  (:class:`~pyfe4ai.ml.FESchemeWrapper`,
  :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget`): another
  thread of the parent may have held the old one.

A :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget` counts in
the process that made it only: in a forked child its
:meth:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget.check` and
:meth:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget.consume` raise
:class:`~pyfe4ai.core.exceptions.FESchemeError`, and so does a ``"kl"``
server that counts against the copy. Otherwise the parent and the child
would each count from the copied ``used``, together produce more partial
decryptions than ``max_partial_decryptions``, and the KL flooding guarantee
would no longer hold. Make a server's budget in the one process that
answers for it (a subclass may share and persist the count, see the class).

**Native threads in a forked child.** In a process forked from one that has
run a native kernel with more than one thread, every native call runs with
one thread, whatever :func:`~pyfe4ai.backend.set_threads` or
``PYFE4AI_THREADS`` says, in the child and in its own children.

* *Why:* on macOS the Rust standard library waits for threads on a
  libdispatch semaphore, and libdispatch refuses to work in the child of a
  fork from a multi-threaded process. A native call that waited for its
  worker threads in such a child killed it (SIGTRAP, "BUG IN CLIENT OF
  LIBDISPATCH: Use-after-free of dispatch_semaphore_t"). This was the case in
  0.4.0 for any kernel run with several threads before the fork.
* *How:* the native core records the process that first runs several
  threads and compares the process id at each multi-threaded call, so the
  rule holds for every native call (through the backend or
  ``pyfe4ai._rust_native`` directly) and for every fork (``os.fork``,
  ``multiprocessing``, a fork made by a C extension). Calling ``set_threads``
  in the child does not lift it.
* *Who is not affected:* a child of a process that never ran several
  threads keeps them (and, once it runs them, its own children are
  limited); a process started with ``spawn``, ``forkserver`` or ``exec``
  starts afresh.
* *Cost:* native calls in a forked child are single-threaded. Pre-fork
  servers and fork-based worker pools that want several native threads per
  worker should start their workers with ``spawn`` or ``forkserver``, or
  fork before the parent runs any multi-threaded native call.
  ``pyfe4ai._rust_native.fork_thread_limit(n)`` returns the thread count a
  call asking for ``n`` gets in the current process.
* Linux waits on a futex and was not seen to crash; the rule applies there
  too, so the behaviour is the same on every platform.

Nothing else changes in the child: it has a copy of every object of the
parent, keys and caches included, and the public tables. The native code
keeps no random generator state between calls (one DRBG per call), and its
worker threads end before a call returns, so a fork duplicates none. Python
warns about ``fork()`` in a process with threads; the ``spawn`` and
``forkserver`` start methods of ``multiprocessing`` avoid all of the above,
and their workers rebuild every cache.

Process-wide state
------------------

Every module- or class-level value that changes after import, or that the
library fills on first use (``tests/state/test_global_state.py`` checks this
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
   * - ``_NATIVE_NOTICE`` (:mod:`pyfe4ai.backend`)
     - whether the missing-extension notice is still due (one entry, popped
       by the first fallback to Python without the extension)
     - kept (a child whose parent gave the notice gives none)
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
       the registered RNS moduli (unbounded: one per parameter set);
       registering an RNS modulus drops its roots and rings computed without
       the factors, also while other threads use the caches
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
