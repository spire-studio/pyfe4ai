pyfe4ai — Python Functional Encryption for AI
==============================================

**PyFE4AI** is a research-oriented Python library of functional encryption
(FE) schemes for AI security and privacy. It covers single-input (SIFE),
multi-input (MIFE), multi-client (MCFE), decentralized, threshold,
function-hiding, and quadratic constructions across DDH, Damgård-DDH,
Paillier, LWE, Ring-LWE, and pairing backends — all under one consistent
``setup / keygen / encrypt / decrypt`` API.

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

For pairing-based FE families, additionally install
``pip install -e ".[test,pairing]"`` (requires the PBC library; see the
repository README for platform notes).

Minimal example
---------------

.. code-block:: python

   from pyfe4ai import SIFE, SIFEKeyGenerator

   x, y = [2, 1, 3], [4, 5, 6]

   kg = SIFEKeyGenerator({"sec_param": 128, "eta": len(x)})
   kg.setup()
   pp, sk = kg.get_public_parameters(), kg.get_private_keys()
   dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

   ct = SIFE({"precision": 3, "keys": {"pp": pp, "sk": sk}}).encrypt(x)
   result = SIFE({"precision": 3, "keys": {"pp": pp}}).decrypt(ct, dk, y)
   print(result)  # <x, y> = 31

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
