API Reference
=============

Top-level package
-----------------

.. automodule:: pyfe4ai
   :no-members:

``pyfe4ai`` re-exports the DDH schemes of the three main families and the
status API:

* :class:`~pyfe4ai.schemes.sife.ddh.SIFE`,
  :class:`~pyfe4ai.schemes.sife.ddh.SIFEKeyGenerator`,
  :class:`~pyfe4ai.schemes.mife.ddh.MIFE`,
  :class:`~pyfe4ai.schemes.mife.ddh.MIFEKeyGenerator`,
  :class:`~pyfe4ai.schemes.mcfe.ddh.MCFE` and
  :class:`~pyfe4ai.schemes.mcfe.ddh.MCFEKeyGenerator`;
* :class:`~pyfe4ai.core.status.SchemeStatus` and
  :func:`~pyfe4ai.core.status.get_scheme_status`;
* :class:`~pyfe4ai.core.exceptions.FESecurityWarning`;
* ``__version__``.

The status and security-profile modules are documented on
:doc:`../security`.

The package is organised in layers:

* :mod:`pyfe4ai.core` -- base classes and the parameter cache, security
  profiles, review status, one-time precomputations, exceptions;
* :mod:`pyfe4ai.schemes` -- the schemes by family (``sife``, ``mife``,
  ``mcfe``, ``quadratic``), with the components several families share in
  :mod:`pyfe4ai.schemes.common` and :mod:`pyfe4ai.schemes.threshold`;
* :mod:`pyfe4ai.primitives` -- lattices, Gaussian sampling, hashing, groups
  and pairings;
* :mod:`pyfe4ai.backend` -- selection of the native (Rust) or Python
  implementation, and threads;
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
   config_types
   primitives
   backend
   ml
