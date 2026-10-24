pyfe4ai — Python Functional Encryption for AI
==============================================

**PyFE4AI** is a research-oriented Python library of functional encryption
(FE) schemes for AI security and privacy. It covers single-input (SIFE),
multi-input (MIFE), multi-client (MCFE), decentralized, threshold,
function-hiding, and quadratic constructions under DDH (3072-bit groups or
ristretto255), Damgård-DDH, Paillier, LWE, Ring-LWE and pairing-based
assumptions. Every scheme is built from its id with
:func:`pyfe4ai.create("family/variant", config) <pyfe4ai.core.registry.create>`,
which returns a key generator and a crypto class; the arguments of
encryption and decryption are family-specific, and
:class:`~pyfe4ai.ml.FESchemeWrapper` offers a uniform layer for ML
workloads. An optional Rust extension runs the hot paths natively. Check
each scheme's review status (``pyfe4ai.get_scheme_status``) before relying
on its privacy.

`Project homepage <https://spire-studio.github.io/pyfe4ai/>`_ ·
`GitHub <https://github.com/spire-studio/pyfe4ai>`_ ·
`Issues <https://github.com/spire-studio/pyfe4ai/issues>`_ ·
`Changelog <https://github.com/spire-studio/pyfe4ai/blob/main/CHANGELOG.md>`_

Where to start
--------------

* New to the library: :doc:`install`, then :doc:`quickstart`.
* Picking a scheme for a task: :doc:`choosing`.
* Threshold decryption and federated learning: :doc:`threshold_ml`.
* Coming from version 0.3.0: :doc:`upgrading`.

Security & status
-----------------

PyFE4AI is a research prototype, not production-audited cryptography.

* **Review status**: |n_schemes| registered schemes, |n_reviewed| reviewed,
  |n_experimental| experimental, |n_unreviewed| unreviewed and
  |n_known_insecure| known-insecure. The experimental ones are the threshold
  layers of tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE and
  the carry mask of dMCFE-DDH (our own constructions, no published proof).
  ``pyfe4ai.get_scheme_status("family/variant")`` returns a scheme's entry.
* **Security profiles**: ``security_profile="toy"``, the default, is **not
  secure**; ``"standard"`` aims at 128-bit security (3072-bit DL groups
  and DCR moduli, lattice-estimator bounds for LWE and Ring-LWE,
  ``stat_param`` 128), except that pairing-based schemes get BLS12-381
  (about 120-126-bit, with a ``FESecurityWarning``) and ``sife/fullysec_lwe``,
  ``mife/fullysec_lwe``, ``mife/lwe_threshold`` and
  ``mcfe/paillier_decentralized`` are refused. The ABDP15 LWE schemes
  take ``matrix_mode`` (``"memory"`` or ``"stream"``) to trade memory for
  time.
* **Threshold schemes**: decryption servers are honest-but-curious; a
  deviating server can change the result unnoticed.
* **ML adapter**: ``FESchemeWrapper`` runs the key generator and holds the
  master key, a single trust domain meant for experiments.

Details: :doc:`security`, and the README sections
`Scheme Status <https://github.com/spire-studio/pyfe4ai#scheme-status>`__,
`Security Profiles <https://github.com/spire-studio/pyfe4ai#security-profiles>`__
and `Known Issues / Security Notice
<https://github.com/spire-studio/pyfe4ai#known-issues--security-notice>`__.

Contents
--------

.. toctree::
   :maxdepth: 2

   install
   quickstart
   choosing
   security
   configuration
   environment
   threshold_ml
   threading
   glossary
   upgrading
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
