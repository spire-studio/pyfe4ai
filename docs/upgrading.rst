Upgrading from 0.3.0
====================

Version 0.4.0 reorganises the package: modules moved, scheme classes are
renamed after their scheme ids, the top-level package exports different
names, and schemes are built with :func:`pyfe4ai.create
<pyfe4ai.core.registry.create>`. Some defaults and serialized formats
changed too, and the threshold LWE layers need new keys. This page lists
what code written for 0.3.0 has to change; the
`changelog <https://github.com/spire-studio/pyfe4ai/blob/main/CHANGELOG.md>`__
has every change.

Before you upgrade
------------------

* **Threshold LWE layers** (``mife/lwe_threshold``,
  ``mcfe/ring_lwe_threshold``, ``mcfe/lwe_threshold``): 0.4.0 refuses every
  key share issued by 0.3.0. Finish the rounds encrypted under the old keys
  with 0.3.0 first; then re-key with 0.4.0 (`Re-keying`_). With ``t < s``
  their 0.3.0 key shares leak part of the decryption noise (see the README's
  Known Issues); the fix changes how the key is shared.
* **Pickles** do not carry across versions: objects pickled by 0.3.0 that
  refer to a moved module or a renamed class do not load. Keep keys, public
  parameters and ciphertexts in their JSON form, and build the objects
  again.
* **The parameter cache** moved from ``./config/`` to the per-user cache
  directory (:doc:`environment`); move ``./config/authority`` there or set
  ``PYFE4AI_CACHE_DIR=./config`` to keep the 0.3.0 records (some are
  regenerated anyway, see `Formats and defaults`_).

Imports and names
-----------------

The top-level package no longer exports scheme classes. Build a scheme from
its id, or import its classes, under their new names, from the family
package:

.. code-block:: python

   # 0.3.0
   from pyfe4ai import SIFE, SIFEKeyGenerator
   kg = SIFEKeyGenerator({"sec_param": 128, "eta": 3})

.. code-block:: python

   # 0.4.0
   import pyfe4ai
   kg, SIFEDDH = pyfe4ai.create("sife/ddh", {"sec_param": 128, "eta": 3})
   # or: from pyfe4ai.schemes.sife import SIFEDDH, SIFEDDHKeyGenerator

``pyfe4ai`` exports ``create`` and ``CreatedScheme``, the registry queries
``list_schemes``, ``get_scheme``, ``CAPABILITIES``, ``get_scheme_status``
and ``SchemeStatus``, every exception and warning (``FEError``,
``FEConfigError``, ``FEKeyError``, ``FEValidationError``, ``FESchemeError``,
``MixedSharingsError``, ``FESecurityWarning``, ``FEWarning``) and
``backend_info``. ``import pyfe4ai`` no longer imports the scheme modules.

Renamed classes
~~~~~~~~~~~~~~~

Each class is named after its scheme id: ``Threshold`` or ``Decentralized``,
then the family, then the variant in CamelCase; the key generator adds
``KeyGenerator`` (``SIFEDDH``, ``SIFEDDHKeyGenerator``). No alias remains at
an old name. Renamed since 0.3.0 (each with its ``...KeyGenerator``):

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Scheme
     - 0.3.0
     - 0.4.0
   * - ``sife/ddh``
     - ``SIFE``
     - ``SIFEDDH``
   * - ``sife/ddh_dynamic``
     - ``SIFEDynamic`` (alias ``DynamicSIFE``)
     - ``SIFEDDHDynamic``
   * - ``sife/damgard``
     - ``SIFEDamgard`` (alias ``SIFEDamgardDDH``)
     - ``SIFEDamgard``
   * - ``sife/fh_ipe_pairing``
     - ``SIFEFHIPE``
     - ``SIFEFHIPEPairing``
   * - ``sife/part_fh_ipe_pairing``
     - ``SIFEPartFHIPE``
     - ``SIFEPartFHIPEPairing``
   * - ``mife/ddh``
     - ``MIFE``
     - ``MIFEDDH``
   * - ``mife/damgard``
     - ``MIFEDamgard`` (alias ``MIFEDamgardDDH``)
     - ``MIFEDamgard``
   * - ``mife/fh_ipe_pairing``
     - ``MIFEFHIPE``
     - ``MIFEFHIPEPairing``
   * - ``mife/fh_multi_ipe_pairing``
     - ``MIFEFHMultiIPE``
     - ``MIFEFHMultiIPEPairing``
   * - ``mife/ddh_threshold``
     - ``ThresholdMIFE``
     - ``ThresholdMIFEDDH``
   * - ``mcfe/ddh``
     - ``MCFE``
     - ``MCFEDDH``
   * - ``mcfe/damgard``
     - ``MCFEDamgard`` (alias ``MCFEDamgardDDH``)
     - ``MCFEDamgard``
   * - ``mcfe/fh_multi_ipe_pairing``
     - ``MCFEFHMultiIPE``
     - ``MCFEFHMultiIPEPairing``
   * - ``mcfe/fh_multi_ipe_pairing_threshold``
     - ``ThresholdMCFEFHMultiIPE``
     - ``ThresholdMCFEFHMultiIPEPairing``
   * - ``mcfe/fh_multi_ipe_pairing_decentralized``
     - ``DecentralizedMCFEFHMultiIPE``
     - ``DecentralizedMCFEFHMultiIPEPairing``
   * - ``mcfe/ddh_threshold``
     - ``ThresholdMCFE``
     - ``ThresholdMCFEDDH``
   * - ``mcfe/ddh_decentralized``
     - ``DecentralizedMCFE`` (alias ``DMCFE``)
     - ``DecentralizedMCFEDDH``
   * - ``mcfe/lwe_decentralized``
     - ``DecentralizedMCFELWE`` (alias ``DMCFELWE``)
     - ``DecentralizedMCFELWE``
   * - ``mcfe/ring_lwe_decentralized``
     - ``DecentralizedMCFERingLWE`` (alias ``DMCFERingLWE``)
     - ``DecentralizedMCFERingLWE``
   * - ``quadratic/miqfe``
     - ``MultiInputQuadratic``
     - ``QuadraticMIQFE``

The other classes keep their names (``SIFELWE``, ``MCFERingLWE``,
``ThresholdMCFERingLWE``, ``DecentralizedMCFEPaillier``, ``QuadraticSGP``,
...). :func:`pyfe4ai.list_schemes <pyfe4ai.core.registry.list_schemes>`
and the table on :doc:`choosing` give every scheme's classes.

Moved modules
~~~~~~~~~~~~~

No module remains at its 0.3.0 path:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.3.0
     - 0.4.0
   * - ``pyfe4ai.utils.ml_adapter``
     - :mod:`pyfe4ai.ml`
   * - ``pyfe4ai.utils.quantization``
     - :mod:`pyfe4ai.ml.quantization`
   * - ``pyfe4ai.utils.exceptions`` (and the exceptions of
       ``pyfe4ai.utils``)
     - :mod:`pyfe4ai.core.exceptions` (also exported by ``pyfe4ai``)
   * - ``pyfe4ai.utils.crypto_constants``
     - :mod:`pyfe4ai.core.constants`
   * - ``pyfe4ai.utils.crypto_utils``
     - :mod:`pyfe4ai.primitives.hashing` (hashes) and
       :mod:`pyfe4ai.primitives.groups.integer_groups` (groups)
   * - ``pyfe4ai.utils.sampling_utils``
     - :mod:`pyfe4ai.primitives.sampling`
   * - ``pyfe4ai.utils.ring_lwe_utils``, ``lwe_utils``, ``lwe_matrix``,
       ``matrix_utils``
     - :mod:`pyfe4ai.primitives.lattice` (same module names)
   * - ``pyfe4ai.utils.dlog_solver``, ``modular_utils``
     - :mod:`pyfe4ai.primitives.groups` (same module names)
   * - ``pyfe4ai.utils.pairing_backend``, ``pairing_utils``
     - :mod:`pyfe4ai.primitives.pairing` (same module names)
   * - ``pyfe4ai.utils.validation``
     - ``pyfe4ai._validation`` (internal)
   * - ``pyfe4ai.schemes.ipfe``
     - :mod:`pyfe4ai.core.base`
   * - ``pyfe4ai.schemes.status``
     - :mod:`pyfe4ai.core.status` (``SchemeStatus`` and the status values:
       :mod:`pyfe4ai.core.registry`)
   * - ``pyfe4ai.schemes.profiles``
     - :mod:`pyfe4ai.core.profiles` (``resolve`` and the profile tables:
       :mod:`pyfe4ai.core.config`)
   * - ``pyfe4ai.schemes.config_types`` (``TypedDict`` classes, also
       re-exported by ``pyfe4ai.schemes``)
     - :mod:`pyfe4ai.core.config` (config classes, below)
   * - ``pyfe4ai.schemes.abdp15_lwe``, ``ddh_base``
     - :mod:`pyfe4ai.schemes.common` (same module names)
   * - ``pyfe4ai.schemes.threshold_utils``
     - :mod:`pyfe4ai.schemes.threshold.sharing` (Shamir sharing,
       Lagrange coefficients), :mod:`~pyfe4ai.schemes.threshold.binding`
       (weights and ciphertext tags), :mod:`~pyfe4ai.schemes.threshold.codec`
       (``ciphertexts_digest``) and :mod:`~pyfe4ai.schemes.threshold.roles`
       (enrolled sets, combination)
   * - ``pyfe4ai.logger`` (``init_logging_config``), ``pyfe4ai/config.yaml``
     - removed: configure logging in the application (see *Warnings and
       logging* below)

Configuration types
~~~~~~~~~~~~~~~~~~~

The ``TypedDict`` classes of ``pyfe4ai.schemes.config_types`` are replaced
by the config classes of :mod:`pyfe4ai.core.config`, one per scheme. They
are ``dict`` subclasses, so dict configs keep working, and the names that
remain build configs as before (``DDHConfig(sec_param=128, eta=3)``).
Renamed: ``MultiInputQuadraticConfig`` → ``QuadraticMIQFEConfig``,
``PairingMultiIPEConfig`` → ``MIFEFHMultiIPEPairingConfig``,
``PartFHIPEConfig`` → ``SIFEPartFHIPEPairingConfig``,
``ThresholdDDHConfig`` → ``ThresholdConfig`` (and
``ThresholdMIFEDDHConfig``, ``ThresholdMCFEDDHConfig``),
``ThresholdLWEConfig`` → ``ThresholdSharingConfig`` (and
``ThresholdMIFELWEConfig``), ``ThresholdPairingMultiIPEConfig`` →
``ThresholdMCFEFHMultiIPEPairingConfig``; ``FEBaseConfig`` →
``SchemeConfig``. :doc:`configuration` lists every scheme's options.

Behaviour changes
-----------------

Partial decryptions that disagree raise
   ``combine_decrypt`` of every threshold scheme raises
   :class:`~pyfe4ai.core.exceptions.FESchemeError` ("partial decryptions
   disagree on the ciphertexts they decrypted") when the partial
   decryptions of one enrolled set were computed for different ciphertexts;
   0.3.0 returned ``None``. Code that tested for ``None`` to detect a faulty
   server catches ``FESchemeError``. ``None`` remains the answer to a value
   out of range.

Server and client ids
   ``lst_sid`` and ``lst_nid`` hold ``str`` or ``int`` ids, one type per
   list; key generators refuse other values (``bytes``, floats, ``None``,
   ``bool``, tuples, numpy integers) and mixed lists with
   ``FEValidationError``. Convert numpy integers with ``int()`` and other ids
   with ``str()`` before building the key generator. After setup a numpy
   integer is read as the ``int`` it equals wherever an id is given, and
   values such as ``1.0`` or ``True`` are no ids (0.3.0 issued client 1's
   key for them in most MIFE and MCFE schemes). Integer client ids now work
   in the DDH and pairing schemes, and integer server ids in tagged
   threshold ciphertexts after a JSON round trip. See
   :doc:`configuration`.

Stricter inputs
   Ring-LWE key rows of shape ``(ring_n, 1)`` and three-dimensional key
   containers are refused (``FEKeyError``; ``numpy.ravel`` each row).
   Threshold DDH servers accept key-share values only as the decimal
   strings the key generators write. ``EncryptedGradient.from_dict`` and
   ``EncryptedFeatures.from_dict`` check the layout and the field types of
   what ``to_dict`` writes (``ValueError`` otherwise); packed Ring-LWE
   gradients now serialise (0.3.0 raised ``KeyError``). The ML adapter
   refuses a gradient filed under another client than the one it names.

Warnings and logging
   The library no longer configures logging or writes log files (``./logs``);
   the ``pyfe4ai`` logger has a ``NullHandler``. What an application should
   see is also issued as :class:`~pyfe4ai.core.exceptions.FEWarning`: a
   regenerated parameter record, a request answered with ``None``, a
   slower fallback, and, once per process, a missing Rust extension. Every
   ``FEWarning`` and ``FESecurityWarning`` is attributed to the
   application's call (the first frame outside pyfe4ai), so Python's
   default filter shows it once per call site, and a filter on a module
   matches the application's module, not ``pyfe4ai.*``; filters by category
   (``warnings.filterwarnings("ignore", category=FEWarning)``) work as
   before.

Pairing groups
   The pairing-based schemes default to BLS12-381 from the Rust extension,
   under both profiles (0.3.0: MNT224 under ``toy``, BN254 under
   ``standard``). Keys, public parameters and ciphertexts of 0.3.0 cannot be
   read on it: pass ``pairing_group_param="MNT224"``, with charm-crypto
   installed, to keep using them.

The ``standard`` profile refuses more
   ``mife/lwe_threshold``, an explicit ``sharing="shamir"`` with ``t < s``,
   ``gaussian_sampler="rounded"`` for the schemes that sample exactly by
   default, and ABDP15 dimensions below the new rule. Both profiles refuse
   ``stat_param`` above 256 and thresholds that are not integers. The
   ``ddh`` key generators refuse a ``group`` other than ``"modp"`` (the
   elliptic-curve DDH is the ``ddh_ristretto255`` variant).

Formats and defaults
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schemes
     - What changed
   * - Threshold LWE layers
     - new derivation of the flooding noise (``noise_version`` 2), key
       shares bound to their public parameters (``pp_binding``), replicated
       sharing for ``t < s`` with its own parameter records; 0.3.0 key
       shares are refused (``FEKeyError``)
   * - ``mcfe/ring_lwe_threshold``
     - defaults: binary ciphertexts, key shares and partial decryptions
       (``ct_format="binary"``), an RNS modulus (``ring_modulus="rns"``),
       split key shares (``key_shares="split"``, :doc:`threshold_ml`) and
       uniform flooding (``flooding="uniform"``); ``ct_format="json"``,
       ``ring_modulus="prime"``, ``key_shares="full"`` and
       ``flooding="gaussian"`` select the 0.3.0 formats. Key shares and
       partial decryptions carry a ``sharing_id``: store and transmit it
       with them
   * - MCFE Ring-LWE and dense-LWE families
     - labels are hashed with ``label_hash="shake128-rejection"``, so new
       ciphertexts and functional-key terms differ from 0.3.0; data without
       the field (0.3.0) is read as ``"shake256-wide"`` and keeps working
       together
   * - ``sife/lwe``, ``mife/lwe``
     - ``lwe_n = "auto"`` picks 17 to 67 % larger dimensions (sized at the
       error width of the security reduction): records cached for ``"auto"``
       are regenerated with a new public matrix, so generate keys again
   * - Ring-LWE MIFE / MCFE
     - parameters sized for messages of norm ``3X`` (``q`` grows by 3 to 4
       bits); cached records are re-derived
   * - Ring-LWE family, dense-LWE MCFE keys, ``sife/fullysec_ring_lwe``,
       ABDP15 under ``standard``
     - exact Gaussian sampling (FACCT) by default, recorded in the public
       parameters; ``PYFE4AI_GAUSSIAN_SAMPLER`` and
       ``set_gaussian_sampler`` no longer apply to them

Re-keying
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Deployment
     - What to do
   * - ``mife/lwe_threshold``, ``mcfe/lwe_threshold``,
       ``mcfe/ring_lwe_threshold``
     - finish the 0.3.0 rounds with 0.3.0; run 0.4.0's key generator
       (``setup()``, ``get_private_keys``, ``get_decryption_keys``) and
       hand out new client keys and key shares; discard the old key shares
   * - ``sife/lwe``, ``mife/lwe`` with ``lwe_n = "auto"``
     - generate keys again (the cached record is regenerated)
   * - pairing-based schemes with 0.3.0 keys
     - generate keys on BLS12-381, or keep ``pairing_group_param="MNT224"``
       with charm-crypto
   * - other schemes
     - keys and ciphertexts of 0.3.0 keep working; regenerating them is
       still the safe choice, since serialized formats can change between
       versions of this research prototype

Tooling
-------

* ``requirements.txt`` is removed: ``pyproject.toml`` lists the
  dependencies (``pip install -e ".[test]"``); PyYAML is no longer needed.
* ``Dockerfile.pairing`` moved to ``docker/Dockerfile.pairing``; the
  parameter estimators to ``tools/params/``.
* The benchmark scripts are commands of ``benchmarks/run.py``
  (``python benchmarks/run.py --help``), with the same options.
* The tests moved into one directory per topic, with pytest marks for the
  groups (``CONTRIBUTING.md``).
