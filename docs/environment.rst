Environment variables and the parameter cache
=============================================

Environment variables
---------------------

The library reads these when it is imported (set them before
``import pyfe4ai`` or in the shell); the functions in the last column change
the same settings at runtime.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Variable
     - Effect
     - At runtime
   * - ``PYFE4AI_BACKEND``
     - ``auto`` (default): the Rust extension where it is built and takes
       the parameters, Python otherwise; ``python``: always the Python
       implementation (no missing-extension notice); ``native``: as
       ``auto``, and without the extension the Python implementation runs
       with the notice. Another value raises ``ValueError`` at import.
     - :func:`pyfe4ai.backend.set_backend`, ``with
       pyfe4ai.backend.backend(name):`` (these two raise ``ImportError`` for
       ``"native"`` without the extension)
   * - ``PYFE4AI_THREADS``
     - threads one native call may use over the chunks of a tensor (default
       1, clamped to :data:`~pyfe4ai.backend.MAX_THREADS`); not an integer:
       ``ValueError`` at import
     - :func:`pyfe4ai.backend.set_threads`
   * - ``PYFE4AI_GAUSSIAN_SAMPLER``
     - the library-wide Gaussian sampler, ``rounded`` (default) or
       ``facct``; the schemes that record their sampler in their public
       parameters (the Ring-LWE family, the dense-LWE MCFE keys, SIFE
       FullySec Ring-LWE, ABDP15 under ``standard``) ignore it
       (:doc:`security`)
     - :func:`~pyfe4ai.primitives.sampling.set_gaussian_sampler`, ``with
       gaussian_sampler(name):``
   * - ``PYFE4AI_CACHE_DIR``
     - the parameter cache directory (below); a ``cache_dir`` entry of a
       key generator's config takes precedence
     - ``cache_dir`` in the config
   * - ``PYFE4AI_NO_NATIVE``
     - set at **installation**: ``pip install`` does not build the Rust
       extension
     - --

:func:`pyfe4ai.backend_info() <pyfe4ai.backend.backend_info>` reports the values in effect. Set the
process-wide values once, before threads start (:doc:`threading`).

These only concern a source checkout:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``PYFE4AI_TEST_PAIRING_GROUP``
     - the pairing group of the pairing-scheme tests (default BLS12-381),
       e.g. ``MNT224`` with charm-crypto
   * - ``PYFE4AI_RUST_CORE_LIB``
     - the path of the ``rust/core`` C library that
       ``tests/native/test_rust_core.py`` loads (default:
       ``rust/core/target/release``, or ``$CARGO_TARGET_DIR/release``)
   * - ``PYFE4AI_TEST_LOG_DIR``
     - where the tests that configure logging write their log files
       (default: a ``pyfe4ai-tests/logs`` folder in the temporary directory)
   * - ``CARGO_TARGET_DIR``
     - cargo's build directory, also for ``tools/build_rust.py``

The parameter cache
-------------------

Key generators keep the parameters they generate -- discrete-log groups,
lattice dimensions and moduli, the seeds of public matrices -- in a
parameter cache and load them again for the same configuration, which saves
the expensive steps of ``setup()``.

* **Where.** ``<cache>/authority/<scheme type>/param.json`` (RNS moduli in
  ``param-rns.json``), where ``<cache>`` is the ``cache_dir`` entry of the
  key generator's config, else ``PYFE4AI_CACHE_DIR``, else the per-user
  cache directory (:func:`~pyfe4ai.core.base.default_cache_dir`):
  ``~/Library/Caches/pyfe4ai`` on macOS, ``$XDG_CACHE_HOME/pyfe4ai`` or
  ``~/.cache/pyfe4ai`` on Linux, ``%LOCALAPPDATA%\pyfe4ai\Cache`` on Windows.
  The DDH crypto objects keep their discrete-log tables under
  ``<cache>/crypto/``. Version 0.3.0 used ``./config/`` of the working
  directory: move ``./config/authority`` to the new place, or set
  ``PYFE4AI_CACHE_DIR=./config``, to keep its records.
* **One record per scheme type.** A key generator whose configuration does
  not match the cached record (other dimensions, bounds or profile, or a
  record of an earlier version) generates new parameters and seeds,
  replaces the record and warns:

  .. code-block:: text

     FEWarning: the parameter record .../authority/SIFE/param.json does not match this
     configuration (for example a record of an earlier version): regenerated with new
     parameters and seeds; keys, public parameters and ciphertexts made with the old
     record no longer match

  Expect it when several configurations of one scheme, or several examples,
  run one after another. Keys and ciphertexts already made keep working with
  the public parameters they were made with; setting up the earlier
  configuration again draws new parameters. Give each configuration its own
  ``cache_dir`` to keep both. An unreadable record is regenerated the same
  way; records are written atomically.
* **Checks.** A cached record is used only if it matches the configuration:
  discrete-log groups are checked for primality, size, safe-prime structure
  and generator order, DCR moduli for size and small factors (their factors
  are never stored), and the lattice parameters are re-derived or expanded
  from stored seeds; under ``standard`` the lattice moduli are checked
  against the 128-bit tables again.
* **Trust.** No check can tell whether someone knows the factors of a DCR
  modulus. Under ``standard`` the Paillier moduli are generated for every
  key generator and never stored; under ``toy`` a planted modulus would be
  used. Use a cache directory that nobody else can write to. Processes that
  share a directory share its records.
