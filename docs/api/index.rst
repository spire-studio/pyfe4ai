API Reference
=============

Top-level package
-----------------

.. automodule:: pyfe4ai
   :no-members:

``pyfe4ai`` re-exports the DDH schemes of the three main families and the
status API:

* :class:`~pyfe4ai.schemes.sife.ddh.SIFEDDH`,
  :class:`~pyfe4ai.schemes.sife.ddh.SIFEDDHKeyGenerator`,
  :class:`~pyfe4ai.schemes.mife.ddh.MIFEDDH`,
  :class:`~pyfe4ai.schemes.mife.ddh.MIFEDDHKeyGenerator`,
  :class:`~pyfe4ai.schemes.mcfe.ddh.MCFEDDH` and
  :class:`~pyfe4ai.schemes.mcfe.ddh.MCFEDDHKeyGenerator`;
* :class:`~pyfe4ai.core.status.SchemeStatus` and
  :func:`~pyfe4ai.core.status.get_scheme_status`;
* :class:`~pyfe4ai.core.exceptions.FESecurityWarning`;
* ``__version__``.

The status and security-profile modules are documented on
:doc:`../security`.

The package is organised in layers:

* :mod:`pyfe4ai.core` -- base classes and the parameter cache, the
  configurations (:mod:`pyfe4ai.core.config`), the serialized objects
  (:mod:`pyfe4ai.core.formats`) and their versions (:mod:`pyfe4ai.core.versions`),
  security profiles, review status, one-time precomputations, exceptions;
* ``pyfe4ai.schemes`` -- the schemes by family (``sife``, ``mife``,
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
   config
   primitives
   backend
   ml
