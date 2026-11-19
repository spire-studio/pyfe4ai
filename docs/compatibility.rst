Compatibility policy
====================

PyFE4AI follows `semantic versioning <https://semver.org>`__ from 1.0 on, for
the part of the library that this page calls **stable**. Everything else can
change in any release.

Tiers
-----

Every registered scheme is in one tier, derived from its review status
(:func:`~pyfe4ai.core.status.get_scheme_status`) and nothing else
(:mod:`pyfe4ai.core.stability`). Of the |n_schemes| registered schemes,
|n_reviewed| are *reviewed*, hence stable:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Tier
     - Review status
     - What 1.x promises
   * - stable
     - *reviewed* (|n_reviewed|)
     - backward compatibility of the public API, the configuration options
       and the serialized records (below)
   * - experimental
     - *experimental* (|n_experimental|), *unreviewed* (|n_unreviewed|)
     - nothing: may change in any release, without a deprecation period
   * - known-insecure
     - *known-insecure* (|n_known_insecure|)
     - nothing; refused by ``security_profile="standard"``

A stable scheme can have experimental parts: option values and capabilities
whose construction or argument is this project's own (the packed layout of
``sife/fullysec_ring_lwe``; the threshold key issuance ``distributed_keygen``
of ``sife/cp_ab_pairing``)
(:data:`~pyfe4ai.core.stability.EXPERIMENTAL_OPTIONS`,
:data:`~pyfe4ai.core.stability.EXPERIMENTAL_CAPABILITIES`). The records written
with them are experimental too: those of an experimental option value
(:func:`~pyfe4ai.core.stability.config_tier` tells them from a scheme's
configuration), and those of an experimental capability, whatever the
configuration (for ``distributed_keygen``: the key servers' public record, a
server's share, a key request and a partial key, the records of
:mod:`pyfe4ai.core.formats.keyissue`; the combined key is the scheme's own
functional key, which is stable). Records written under the
``toy`` and the ``standard`` profiles have the same formats, so both are
covered.

A scheme moves from experimental to stable in a minor release, after an
independent review. A stable scheme never becomes experimental; if it turns
out to be insecure it becomes known-insecure (see `Security fixes`_).

What stays compatible
---------------------

Python API
~~~~~~~~~~

* The names in ``__all__`` of :mod:`pyfe4ai`, of the family packages
  :mod:`pyfe4ai.schemes.sife`, :mod:`~pyfe4ai.schemes.mife`,
  :mod:`~pyfe4ai.schemes.mcfe` and :mod:`~pyfe4ai.schemes.quadratic` (the
  classes of stable schemes), of :mod:`pyfe4ai.ml`, :mod:`pyfe4ai.policy`
  and :mod:`pyfe4ai.core.status`.
* :func:`pyfe4ai.create <pyfe4ai.core.registry.create>` and the registry queries, with the fields of
  :class:`~pyfe4ai.core.registry.Scheme` and :class:`~pyfe4ai.core.registry.SchemeStatus` (fields may be
  added).
* The public methods and properties of the classes of stable schemes. Pass
  the parameters without a default **by position**: their names are not part
  of the promise. The names of parameters with a default are. Parameters
  whose name starts with an underscore are not. New parameters with a
  default may be added at the end. A method that raises
  :class:`NotImplementedError` in 1.0 may be implemented in a minor release.
* The exception classes and their hierarchy, and which class a documented
  error raises. Messages are not part of the promise.
* The configuration options of stable schemes (key generator and crypto
  objects): their names, their meaning, the values they accept (values may be
  added) and their defaults, except the sizes the ``toy`` profile picks and
  the minima of ``standard``, which may only grow. The default profile stays
  ``toy`` throughout 1.x.

Serialized records
~~~~~~~~~~~~~~~~~~

A record is a dict that a public method returns and another accepts: public
parameters, parameter records, encryption keys, functional keys, key shares,
ciphertexts, partial decryptions, and the wire forms of :mod:`pyfe4ai.ml`
(``EncryptedGradient.to_dict()`` and the like). For the records of stable
schemes:

* **Every 1.x release reads the records 1.0 wrote** and computes the same
  results from them. The promise runs one way: 1.0 need not read what a later
  1.x writes.
* A later release only adds optional fields, whose absence means what 1.0
  did. No field is removed, renamed or re-encoded.
* A record with no version marker has the 1.0 layout.
* The values of the version markers (:mod:`pyfe4ai.core.versions`) never
  change and are never reused: a new derivation gets a new identifier, and
  the old one stays accepted.
* A 1.x key generator with the same configuration reuses a parameter record
  that 1.0 cached, without regenerating it (except the DCR moduli that
  ``standard`` never caches).
* Records of different derivations may still be refused when mixed, with an
  explicit error (for example :class:`~pyfe4ai.core.versions.MixedSharingsError`).

How a record is stored or transmitted -- JSON, base64 for bytes, or anything
else -- is the application's choice.

The test suite keeps a set of records for every stable scheme
(``tests/compat/fixtures/``, the set of 1.0.0 written by the release) and
every later release must read them and compute the same results; each set
also freezes the list of the schemes that were stable when it was written.

Deprecations
------------

A stable name, option or value is deprecated before it goes: using it issues
:class:`~pyfe4ai.core.exceptions.FEDeprecationWarning`, a subclass of
:class:`~pyfe4ai.core.exceptions.FEWarning` that Python shows by default
(issued by :func:`~pyfe4ai.core.exceptions.warn_deprecated`). It keeps working for
the rest of 1.x and is removed in 2.0 at the earliest, after at least one
minor release with the warning. :doc:`deprecations` lists them.

Version numbers
---------------

``MAJOR.MINOR.PATCH``:

* **Patch** releases fix bugs and security issues, and improve performance
  with the same outputs; they add no public names, options or record fields
  (except where a security fix needs a new marker).
* **Minor** releases add: schemes, options and option values, record fields
  that may be absent, identifiers; they promote experimental schemes after a
  review, deprecate, raise the minima of ``standard``, and may drop a Python
  version announced one minor release before.
* **Major** releases remove what was deprecated, or change incompatibly.

Security fixes
--------------

If a stable scheme, option or kind of record turns out to be insecure, any
release -- a patch release too -- may make it known-insecure (it then issues
:class:`~pyfe4ai.core.exceptions.FESecurityWarning` and ``standard`` refuses it), refuse the
affected records under ``standard`` with an explicit error, or raise the
minima of ``standard``. Such a change is listed under *Security* in the
changelog, announced in a security advisory, and :doc:`upgrading` says how to
move on. Reading records under ``toy`` stays possible where it can, so that
old data can still be decrypted before re-keying.

A security fix only refuses. It never renames a name or a field, and never
changes the meaning or the encoding of anything that is still accepted.

Not covered
-----------

* Pickles: objects pickled by one version do not load in another.
* The experimental and known-insecure tiers, experimental option values, and
  experimental capabilities: ``distributed_keygen`` of ``sife/cp_ab_pairing``,
  its classes and its four kinds of record.
* Bounded-key FE: ``pyfe4ai.schemes.bcfe`` (``bcfe/gc_ristretto255`` is
  experimental), the circuits of ``pyfe4ai.circuits`` (which ``import
  pyfe4ai`` loads, though none of its names is public) and the records of
  ``pyfe4ai.core.formats.garbled``.
* The identity directory: ``pyfe4ai.ac``, its classes, its checks and the
  records of ``pyfe4ai.core.formats.identity``, including what they add around
  the keys and ciphertexts of ``sife/fullysec_ring_lwe`` (the records of the
  scheme itself stay covered).
* The parameter sizes of the ``toy`` profile.
* Internal modules: :mod:`pyfe4ai.primitives`, :mod:`pyfe4ai.backend` (except
  the keys of :func:`~pyfe4ai.backend.backend_info`), ``pyfe4ai.schemes.common``,
  ``pyfe4ai.schemes.threshold``, the :class:`~pyfe4ai.core.formats.record.Record`
  classes of :mod:`pyfe4ai.core.formats` (the fields of the records are
  covered, the classes are not), the native extension, every name that
  starts with an underscore, and the names of parameters without a default
  (pass them by position).
* Performance, the bytes of randomised outputs, and the text of errors,
  warnings and log messages.
* The layout of the parameter cache directory (the records in it are
  covered), and sharing it between versions: later versions read the
  records of earlier ones, but an earlier version regenerates a record it
  does not know and overwrites it, so give each version its own cache
  directory.
