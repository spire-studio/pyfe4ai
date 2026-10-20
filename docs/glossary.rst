Glossary
========

Parameter names and conventions used across the PyFE4AI schemes. Key
generators take a configuration dict (``SIFEKeyGenerator(config)``); the
accepted keys of every scheme, their defaults and the refused combinations
are listed on :doc:`configuration` (from the config classes of
:mod:`pyfe4ai.core.config`).

Scheme taxonomy
---------------

.. list-table::
   :header-rows: 1
   :widths: 16 22 28 34

   * - Prefix
     - Full name
     - Parties
     - Description
   * - SIFE
     - Single-input FE
     - 1 encryptor, 1 decryptor
     - Inner-product FE on one vector
   * - MIFE
     - Multi-input FE
     - *n* encryptors (slots), 1 decryptor
     - Each client encrypts its slot independently; no labels
   * - MCFE
     - Multi-client FE
     - *n* encryptors, 1 decryptor
     - Ciphertexts are bound to a common :term:`label`
   * - dMCFE
     - Decentralized MCFE
     - *n* encryptors, no central key authority
     - Each client derives a share of the functional key; the shares are
       combined
   * - tMIFE / tMCFE
     - Threshold MIFE / MCFE
     - *n* encryptors, *t*-of-*s* decryption servers
     - The functional key is Shamir-shared among *s* servers; any *t* of
       them decrypt together
   * - FH-
     - Function-hiding
     - as the family
     - Functional keys also hide the weights *y*
   * - Quadratic
     - Quadratic FE
     - 1 or *n* encryptors, 1 decryptor
     - Computes ``x^T F y`` instead of an inner product

Configuration keys
------------------

.. glossary::

   ``security_profile``
      ``"toy"`` (the default) keeps the small default parameters: fast, and
      **not secure**. ``"standard"`` fills in parameters aiming at 128-bit
      classical security and refuses explicit values below them. Its
      exceptions: pairing-based schemes get BLS12-381 (about 120-126-bit, with a
      :class:`~pyfe4ai.core.exceptions.FESecurityWarning`), and
      ``sife/fullysec_lwe``, ``mife/fullysec_lwe`` and
      ``mcfe/paillier_decentralized`` are refused. See
      :mod:`pyfe4ai.core.profiles` and :doc:`security`.

   ``sec_param``
      **Not a security level.** The bit length of the safe prime ``p`` in
      the discrete-log (DDH) families and the default ``modulus_length`` of
      the Damgård schemes; ``kappa`` in the Ring-LWE noise bounds; only
      recorded by the LWE, Paillier and pairing families, whose sizes come
      from ``lwe_n``, ``bit_length`` and ``pairing_group_param``. The
      default 128 keeps tests fast and is insecure. For real parameters set
      ``security_profile="standard"``: 3072-bit RFC 3526 groups for the
      discrete-log families, 3072-bit DCR moduli for Paillier,
      lattice-estimator tables for LWE and Ring-LWE, and ``stat_param =
      128``.

   ``stat_param``
      Statistical security parameter: rounding and leftover-hash slack of
      the ABMPSW20 schemes, noise flooding of the threshold schemes, the
      masks of ``mcfe/ddh_decentralized`` and the Paillier and MCFE2 key
      distributions. Default 64; 128 (and at least 128) under ``standard``.
      With ``flooding_accounting = "kl"`` (``mife/lwe_threshold``,
      ``mcfe/ring_lwe_threshold``) it bounds the Hellinger distance of a
      run of at most ``max_partial_decryptions`` partial decryptions per
      server instead of the statistical distance per sample.

   ``flooding_accounting``
      How the threshold schemes ``mife/lwe_threshold`` and
      ``mcfe/ring_lwe_threshold`` size their flooding noise:
      ``"per-sample"`` (default; statistical distance ``2^-stat_param`` per
      flooded value, no limit) or ``"kl"`` (opt-in, ``t = s`` only; a
      bound for a whole run of at most ``max_partial_decryptions`` partial
      decryptions per server, about ``stat_param - 7`` bits in the
      bit-security sense of Micciancio and Walter; each server counts in
      the ``partial_budget`` its deployment provides; much smaller ``q``
      and, for ``mife/lwe_threshold``, ``lwe_n``). See
      :mod:`pyfe4ai.schemes.threshold.kl`.

   ``eta``
      Length of each client's input vector (an ``int``; MIFE-DDH also
      accepts a ``dict[str, int]`` per client).

   ``n``
      Number of clients (input slots); in quadratic ``sgp`` and ``quad``,
      the length of ``x``.

   ``lst_nid``
      Client ids, length ``n`` (default ``nid_0`` ... ``nid_{n-1}``).

   ``s``
      Number of decryption servers of the threshold schemes (tMIFE, tMCFE),
      with ids ``lst_sid``. Other key generators accept it but do not use
      it (MIFE- and MCFE-DDH record it with their cached parameters).
      ``s`` is also the name of secret-key vectors, e.g. ``msk["s"]``.

   ``t``
      Threshold: the number of servers whose partial decryptions are
      combined (``t <= s``, default 2). Fewer than ``t`` learn nothing
      beyond the result.

   ``lst_sid``
      Decryption-server ids of the threshold schemes, length ``s``.

   ``bound_x`` / ``bound_y``
      Bounds on the inputs and on the weights, ``|x_i| <= bound_x`` and
      ``|y_i| <= bound_y``. Used by most families; quadratic ``miqfe`` takes
      ``bound_x`` and ``bound_c`` (coefficients of the quadratic function).

   ``bound``
      One bound for inputs and weights in ``sife/damgard``,
      ``mife/damgard``, ``mcfe/damgard``, ``sife/part_fh_ipe_pairing`` and
      the quadratic ``quad`` and ``sgp`` (``x``, ``y`` and ``F``).

   ``dlog_bound``
      Crypto-object key of the DDH schemes: the range of the discrete log
      solved at decryption (default ``10^(precision + 2)``).

   ``modulus_length``
      Bit length of the group modulus of the Damgård schemes (default
      ``sec_param``).

   ``bit_length``
      Bits of each prime of the Paillier modulus ``N`` (default 128; 1536
      under ``standard``, an ``N`` of 3072 bits).

   ``lwe_n``
      LWE secret dimension (default 32 or 64, depending on the scheme). The
      ABDP15 (``sife/lwe``, ``mife/lwe``, ``mife/lwe_threshold``) and
      ABMPSW20 (``mcfe/lwe`` family) schemes also take ``"auto"``, the
      ``standard`` choice: the smallest dimension whose derived modulus
      meets the 128-bit lattice-estimator table (for ABDP15 read at the
      error width of its security reduction, see
      :func:`~pyfe4ai.primitives.lattice.lwe_utils.abdp15_reduction_sigma`).

   ``ring_n``
      Ring-LWE ring dimension, a power of two; default
      :data:`~pyfe4ai.primitives.lattice.ring_lwe_utils.DEFAULT_RING_N` (32, a toy
      value). ``"auto"``, the ``standard`` choice, selects the smallest
      dimension (1024 to 32768) whose derived modulus meets
      :data:`~pyfe4ai.primitives.lattice.ring_lwe_utils.RING_LWE_128_LOG_Q` (see
      :func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.resolve_ring_n`).

   ``matrix_mode``
      How the ABDP15 LWE schemes (``sife/lwe``, ``mife/lwe``,
      ``mife/lwe_threshold``) hold their seeded public matrix ``A``:
      ``"memory"`` (default) keeps it expanded, ``"stream"`` expands it again
      for every batch (little memory, slower). Accepted by key generator and
      crypto configs. See :mod:`pyfe4ai.schemes.common.abdp15_lwe`.

   ``pairing_group_param``
      Pairing group of the pairing-based schemes; default ``"BLS12-381"``
      (the Rust extension; other names are charm-crypto groups). Only
      asymmetric (type-3) groups are accepted: symmetric ones such as
      ``"SS512"`` are rejected. ``standard`` requires ``"BLS12-381"``, about
      120-126-bit, with a :class:`~pyfe4ai.core.exceptions.FESecurityWarning`.
      The ``group`` attribute of a scheme object is the ``PairingGroup``
      built from it (:mod:`pyfe4ai.primitives.pairing.pairing_backend`).

   ``sec_level``
      ``k`` of k-LIN in ``mife/fh_multi_ipe_pairing`` (default 2).

   ``mddh_k``
      ``k`` of the D_k-MDDH assumption in ``mcfe/damgard`` (default 1) and
      quadratic ``miqfe`` (at least 2, default 2).

   ``precision``
      Crypto-object key: the ndarray helpers scale real values by
      ``10^precision`` and round them to integers (default 0).

Labels, tags and keys
---------------------

.. glossary::

   ``label``
      Encryption label of the multi-client schemes (MCFE, dMCFE, tMCFE),
      e.g. a training round. Ciphertexts are bound to it and decrypt together
      only under the same label; a client must encrypt at most once per
      label. In the identity-based Ring-LWE MCFE variants, functional keys
      are bound to a label too.

   ``tag_f``
      Name of the function in the decentralized FH-MCFE
      (``mcfe/fh_multi_ipe_pairing_decentralized``): each client derives its
      key share for ``(tag_f, y_i)``, at most one per client and ``tag_f``.

   ``mac_keys`` / ``tags``
      Client-to-server MAC keys of tMIFE-LWE, tMCFE-Ring-LWE and
      tMCFE-FH-Multi-IPE. A client's ``sk`` carries ``mac_keys``
      (``{sid: key}``) and adds ``tags`` (``{sid: HMAC-SHA256}``) to each
      ciphertext; a server's key share carries ``{nid: key}``, and the
      server only decrypts ciphertexts whose tag for it verifies.

.. list-table:: Key artifacts
   :header-rows: 1
   :widths: 22 30 48

   * - Name
     - Role
     - Notes
   * - ``mpk`` / ``pp``
     - Master public key / public parameters
     - ``get_public_parameters()``; group elements, moduli, seeds
   * - ``msk``
     - Master secret key
     - Kept by the key generator
   * - ``sk``
     - Per-client encryption key
     - ``get_private_keys(nid)``; ``None`` where encryption needs only
       ``pp`` (e.g. quadratic ``sgp``)
   * - ``dk``
     - Functional decryption key
     - ``get_decryption_keys(sid, credentials=...)``; a per-server key share
       in the threshold schemes
   * - ``fusion_weight``
     - Weights *y* of the inner product
     - A list for SIFE, ``{nid: list}`` for multi-input and multi-client
       schemes

Review status and trust
-----------------------

.. glossary::

   reviewed
      Checked against the cited construction; no security defect known.
      Fidelity *faithful* (as published) or *adapted* (documented
      deviations).

   experimental
      Implemented and checked here, but the construction or its security
      argument is our own (fidelity *ad-hoc*); no defect known.

   unreviewed
      Not yet checked against its paper.

   known-insecure
      A concrete privacy defect is known; the key generator and crypto
      objects emit :class:`~pyfe4ai.core.exceptions.FESecurityWarning`.

   honest-but-curious
      The trust model of the threshold decryption servers: fewer than ``t``
      learn nothing beyond the result, but partial decryptions carry no proof
      of correctness, so a deviating server can change the result
      unnoticed.

See :doc:`security` for the status of every scheme.

Cryptographic variables
-----------------------

.. list-table:: Discrete log (DDH): ``p, q, r, g``
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - Description
   * - ``p``
     - Safe-prime modulus, ``p = r q + 1``
   * - ``q``
     - Prime order of the subgroup
   * - ``r``
     - Cofactor (2)
   * - ``g``
     - Generator of the order-``q`` subgroup of ``Z_p^*``

.. list-table:: Damgård DDH (SIFE, MIFE): ``p, q, g, h``
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - Description
   * - ``p``, ``q``, ``g``
     - As for DDH
   * - ``h``
     - Second generator, hashed into the group from a seed (``h_seed``), so
       nobody knows its discrete log to the base ``g``

.. list-table:: Paillier (DCR)
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - Description
   * - ``n``
     - Modulus ``N = p q`` of two safe primes; the factors are never stored
   * - ``n_square``
     - ``N^2``
   * - ``g``
     - Public base in ``Z_{N^2}^*``

.. list-table:: LWE and Ring-LWE
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - Description
   * - ``q``
     - Ciphertext modulus
   * - ``p``
     - Plaintext modulus
   * - ``m``
     - Number of LWE samples (rows of ``A``); in ``sife/fullysec_ring_lwe``
       the number of key polynomials
   * - ``sigma_q``, ``sigma1``, ``sigma2``, ...
     - Gaussian widths of the noise and keys
   * - ``lwe_n`` / ``ring_n``
     - LWE dimension / ring dimension (see above)

.. list-table:: Pairing-based
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - Description
   * - ``pairing_group_param``
     - Name of the pairing group (see above)
   * - ``order``
     - Group order

API method naming
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Method
     - Class
     - Purpose
   * - ``setup()``
     - KeyGenerator
     - Generate ``mpk`` / ``msk``
   * - ``get_public_parameters()``
     - KeyGenerator
     - Serialise ``pp`` for distribution
   * - ``get_private_keys(nid)``
     - KeyGenerator
     - Serialise a client's encryption key
   * - ``get_decryption_keys(sid, credentials=...)``
     - KeyGenerator
     - Derive a functional decryption key (a server's key share in the
       threshold schemes; refused by the decentralized ones)
   * - ``encrypt(...)``
     - Crypto
     - Encrypt a plaintext vector (under a ``label`` in MCFE)
   * - ``decrypt(ct, dk, ...)``
     - Crypto
     - Decrypt with a functional key
   * - ``share_decrypt(...)`` / ``combine_decrypt(partials)``
     - Crypto
     - Threshold schemes: a server's partial decryption / combining ``t``
       of them
   * - ``derive_function_decryption_key_share(...)`` /
       ``combine_function_decryption_key_share(shares)``
     - Crypto
     - Decentralized schemes: a client's key share / combining the shares
   * - ``encrypt_lst_ndarray(lst_ndarray, ...)``
     - Crypto
     - Encrypt numpy arrays element-wise
   * - ``compute_lst_ndarray_ct(...)``
     - Crypto
     - Compute on ndarray ciphertexts (in the threshold schemes, a server's
       partial decryptions)
   * - ``decrypt_lst_ndarray_ct(...)``
     - Crypto
     - Decrypt ndarray ciphertexts to arrays of results; an element that
       does not decrypt raises ``FESchemeError``
