pyfe4ai — Python Functional Encryption for AI
==============================================

**PyFE4AI** is a research-oriented Python library of functional encryption
(FE) schemes for AI security and privacy. It covers single-input (SIFE),
multi-input (MIFE), multi-client (MCFE), decentralized, threshold,
function-hiding, and quadratic constructions across DDH, Damgård-DDH,
Paillier, LWE, Ring-LWE, and pairing backends. Every scheme follows the same
two-object model (a key generator and a crypto object), but operation names
and arguments are family-specific; ``FESchemeWrapper`` offers a uniform layer
for ML workloads. Check each scheme's review status
(``pyfe4ai.get_scheme_status``) before relying on its privacy.

`Project homepage <../>`_ ·
`GitHub <https://github.com/spire-studio/pyfe4ai>`_ ·
`Issues <https://github.com/spire-studio/pyfe4ai/issues>`_

Installation
------------

.. code-block:: bash

   git clone https://github.com/spire-studio/pyfe4ai.git
   cd pyfe4ai
   conda env create -f environment.yml
   conda activate pyfe4ai
   pip install -e ".[test]"

With a Rust toolchain (``cargo``) on the ``PATH``, ``pip install`` also
builds the optional native extension (:mod:`pyfe4ai.backend`):
the Ring-LWE, dense-LWE and threshold hot paths, ristretto255, and the
BLS12-381 pairing group that the pairing-based FE families use by default.
Other pairing curves (MNT224, BN254, ...) need charm-crypto,
``pip install -e ".[test,pairing]"`` (requires the PBC library; see the
repository README for platform notes).

Minimal example
---------------

.. code-block:: python

   from pyfe4ai import SIFE, SIFEKeyGenerator

   x, y = [2, 1, 3], [4, 5, 6]

   # sec_param = bit length of the DDH modulus (NOT a security level);
   # 128 keeps the demo fast but is insecure -- for real use replace it with
   # "security_profile": "standard" (a 3072-bit group)
   kg = SIFEKeyGenerator({"sec_param": 128, "eta": len(x)})
   kg.setup()
   pp, sk = kg.get_public_parameters(), kg.get_private_keys()
   dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

   ct = SIFE({"precision": 3, "keys": {"pp": pp, "sk": sk}}).encrypt(x)
   result = SIFE({"precision": 3, "keys": {"pp": pp}}).decrypt(ct, dk, y)
   print(result)  # <x, y> = 31

Security & status
-----------------

* **Review status**: |n_schemes| registered schemes, |n_reviewed| reviewed,
  |n_experimental| experimental, |n_unreviewed| unreviewed and
  |n_known_insecure| known-insecure. The experimental ones are the threshold
  layers of tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE and
  the carry mask of dMCFE-DDH (our own constructions, no published proof).
  ``pyfe4ai.get_scheme_status(family, variant)`` returns a scheme's entry.
* **Security profiles**: ``security_profile="toy"``, the default, is **not
  secure**; ``"standard"`` aims at 128-bit security (3072-bit DL groups
  and DCR moduli, lattice-estimator bounds for LWE and Ring-LWE,
  ``stat_param`` 128), except that pairing-based schemes get BLS12-381
  (about 120-126-bit, with a ``FESecurityWarning``) and ``sife/fullysec_lwe``,
  ``mife/fullysec_lwe`` and ``mcfe/paillier_decentralized`` are refused. The ABDP15 LWE schemes
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

   security
   glossary
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
