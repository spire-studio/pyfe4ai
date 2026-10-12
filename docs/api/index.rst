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
* :class:`~pyfe4ai.schemes.status.SchemeStatus` and
  :func:`~pyfe4ai.schemes.status.get_scheme_status`;
* :class:`~pyfe4ai.utils.exceptions.FESecurityWarning`;
* ``__version__``.

The status and security-profile modules are documented on
:doc:`../security`.

.. toctree::
   :maxdepth: 2

   base
   sife
   mife
   mcfe
   quadratic
   utils
   config_types
