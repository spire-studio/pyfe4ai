API Reference
=============

Top-level package
-----------------

.. automodule:: pyfe4ai
   :no-members:

``pyfe4ai`` exports the entry points of the library:

* :func:`~pyfe4ai.core.registry.create` -- the key generator of a scheme
  given by its id ``"family/variant"``, built from a config and not yet set
  up, with the scheme's crypto class
  (:class:`~pyfe4ai.core.registry.CreatedScheme`);
* the queries of the scheme registry (:mod:`pyfe4ai.core.registry`):
  :func:`~pyfe4ai.core.registry.list_schemes`,
  :func:`~pyfe4ai.core.registry.get_scheme` (a
  :class:`~pyfe4ai.core.registry.Scheme`: classes, kind, capabilities,
  review status), :data:`~pyfe4ai.core.registry.CAPABILITIES`,
  :func:`~pyfe4ai.core.status.get_scheme_status` and
  :class:`~pyfe4ai.core.registry.SchemeStatus`;
* ``backend_info()`` (:func:`~pyfe4ai.backend.backend_info`) -- the
  backend the calling thread uses (the native extension or the Python
  implementation) and how it was chosen, the extension's version and build,
  the threads per native call, the Gaussian sampler and the parameter cache
  directory; importing :mod:`pyfe4ai` does not load the extension, calling
  it does. Without the extension the library also says so once, with an
  :class:`~pyfe4ai.core.exceptions.FEWarning` at the first fallback to
  Python (:doc:`backend`);
* the exceptions: :class:`~pyfe4ai.core.exceptions.FEError` and its
  subclasses :class:`~pyfe4ai.core.exceptions.FEConfigError`,
  :class:`~pyfe4ai.core.exceptions.FEKeyError`,
  :class:`~pyfe4ai.core.exceptions.FEValidationError` (also a
  ``ValueError``), :class:`~pyfe4ai.core.exceptions.FESchemeError` (also a
  ``RuntimeError``) and :class:`~pyfe4ai.core.versions.MixedSharingsError`
  (an ``FESchemeError``);
* the warnings :class:`~pyfe4ai.core.exceptions.FESecurityWarning` and
  :class:`~pyfe4ai.core.exceptions.FEWarning`;
* ``__version__``.

The scheme classes are exported by the family packages
(:mod:`pyfe4ai.schemes.sife`, :mod:`~pyfe4ai.schemes.mife`,
:mod:`~pyfe4ai.schemes.mcfe`, :mod:`~pyfe4ai.schemes.quadratic`), named
after their scheme's id (see :mod:`pyfe4ai.core.registry`). The status and
security-profile modules are documented on :doc:`../security`.

Public and internal modules
---------------------------

**Public** -- what applications use; changes are listed in the
`changelog <https://github.com/spire-studio/pyfe4ai/blob/main/CHANGELOG.md>`__
(from version 0.3.0: :doc:`../upgrading`):

* :mod:`pyfe4ai` (above);
* the family packages :mod:`pyfe4ai.schemes.sife`,
  :mod:`~pyfe4ai.schemes.mife`, :mod:`~pyfe4ai.schemes.mcfe` and
  :mod:`~pyfe4ai.schemes.quadratic`: the key generators and crypto classes
  and their documented methods (the scheme modules, e.g.
  :mod:`pyfe4ai.schemes.mcfe.ring_lwe`, document them);
* :mod:`pyfe4ai.core.registry`, :mod:`pyfe4ai.core.config` (the config
  classes and their options), :mod:`pyfe4ai.core.status`,
  :mod:`pyfe4ai.core.profiles`, :mod:`pyfe4ai.core.exceptions` and
  :mod:`pyfe4ai.core.ids` (the rules for server and client ids, with
  ``is_id`` and ``as_id``);
* :mod:`pyfe4ai.ml` and :mod:`pyfe4ai.ml.quantization`;
* from :mod:`pyfe4ai.backend`: ``available``, ``get_backend``,
  ``set_backend``, ``backend``, ``active_backend``, ``get_threads``,
  ``set_threads``, ``implementation``, ``describe``, ``backend_info`` and
  ``MISSING_EXTENSION_NOTICE``;
* the few entry points in otherwise internal modules that the
  documentation names: ``get_gaussian_sampler``, ``set_gaussian_sampler``
  and ``gaussian_sampler`` (:mod:`pyfe4ai.primitives.sampling`),
  ``pairing_available`` and ``DEFAULT_PAIRING_GROUP``
  (:mod:`pyfe4ai.primitives.pairing.pairing_backend`),
  ``PartialDecryptionBudget`` (:mod:`pyfe4ai.schemes.threshold.kl`) and
  ``default_cache_dir`` (:mod:`pyfe4ai.core.base`).

**Internal** -- documented here for contributors and reviewers; they may
change in any version without notice:

* :mod:`pyfe4ai.core.base` (the base classes and the parameter cache),
  :mod:`pyfe4ai.core.formats` and :mod:`pyfe4ai.core.versions` (the
  classes that describe the serialized objects and their version markers;
  the serialized formats themselves are documented and versioned there),
  :mod:`pyfe4ai.core.one_time`, :mod:`pyfe4ai.core.forksafe` (the fork
  handler; what threads, copies and ``fork`` do is on :doc:`../threading`),
  :mod:`pyfe4ai.core.constants`;
* :mod:`pyfe4ai.schemes.common`, :mod:`pyfe4ai.schemes.threshold` and
  the helpers of the scheme modules other than their classes;
* :mod:`pyfe4ai.primitives` and the rest of :mod:`pyfe4ai.backend`: the
  interface the scheme and primitive layers call (:func:`pyfe4ai.backend.ops`,
  :func:`pyfe4ai.backend.ring`, :mod:`pyfe4ai.backend.interface`) and its
  two implementations (:mod:`pyfe4ai.backend.python_impl`,
  :mod:`pyfe4ai.backend.native_impl`);
* ``pyfe4ai._validation`` and the native extension ``pyfe4ai._rust_native``.

The package is organised in layers:

* :mod:`pyfe4ai.core` -- base classes and the parameter cache, the
  configurations (:mod:`pyfe4ai.core.config`), the serialized objects
  (:mod:`pyfe4ai.core.formats`) and their versions (:mod:`pyfe4ai.core.versions`),
  security profiles, review status, one-time precomputations, the fork
  handler, exceptions;
* ``pyfe4ai.schemes`` -- the schemes by family (``sife``, ``mife``,
  ``mcfe``, ``quadratic``), with the components several families share in
  :mod:`pyfe4ai.schemes.common` and :mod:`pyfe4ai.schemes.threshold`;
* :mod:`pyfe4ai.primitives` -- lattices, Gaussian sampling, hashing, groups
  and pairings;
* :mod:`pyfe4ai.backend` -- the operations the schemes and primitives take
  from a backend (:mod:`pyfe4ai.backend.interface`), their native (Rust)
  and Python implementations, the selection between them, and threads;
* :mod:`pyfe4ai.ml` -- the ML adapter (encrypted inference, federated
  aggregation) and quantization.

.. toctree::
   :maxdepth: 2

   core
   sife
   mife
   mcfe
   quadratic
   common
   config
   primitives
   backend
   ml
