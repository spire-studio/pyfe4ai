Deprecations
============

A name, an option or an option value of the stable tier that is deprecated
keeps working for the rest of 1.x and issues
:class:`~pyfe4ai.core.exceptions.FEDeprecationWarning` (a subclass of
:class:`~pyfe4ai.core.exceptions.FEWarning`, shown by default) at the
application's call. It is removed in 2.0 at the earliest, and only after it
has warned in at least one 1.x minor release (:doc:`compatibility`). To see
every deprecation an application hits, turn the warnings into errors in its
tests:

.. code-block:: python

   import warnings
   from pyfe4ai import FEDeprecationWarning

   warnings.simplefilter("error", FEDeprecationWarning)

.. list-table::
   :header-rows: 1
   :widths: 40 15 25 20

   * - Deprecated
     - Since
     - Use instead
     - Removed in
   * - ``ThresholdMIFEDDHKeyGenerator.lambd`` (``mife/ddh_threshold``), the
       name of versions <= 0.2.0
     - 1.0.0
     - ``sec_param``
     - 2.0

Changed in 1.0 without a deprecation
------------------------------------

Versions up to 0.5 promised no compatibility, so 1.0 changes these directly:

* ``ThresholdMCFEDDH.L`` (``mcfe/ddh_threshold``), a static method computing a
  Lagrange coefficient that no caller used, is private.

The option names of versions before 0.2.0 (``bound_u``, ``label_modulus``,
``u_bound``; :data:`~pyfe4ai.core.config.LEGACY_OPTIONS`) are not
deprecated: they have done nothing since 0.2.0, and a key generator that
gets one ignores it with an :class:`~pyfe4ai.core.exceptions.FEWarning`,
not an :class:`~pyfe4ai.core.exceptions.FEDeprecationWarning`. A filter that
turns deprecations into errors does not catch them.
