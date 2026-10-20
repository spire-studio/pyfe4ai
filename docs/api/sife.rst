SIFE — Single-Input Functional Encryption
==========================================

.. automodule:: pyfe4ai.schemes.sife
   :no-members:

The key generator and crypto class of every scheme below except the
adaptively secure Ring-LWE one are importable from
:mod:`pyfe4ai.schemes.sife`, where ``SIFEDamgardDDH`` and ``DynamicSIFE``
(and their ``...KeyGenerator``) are aliases of
:class:`~pyfe4ai.schemes.sife.damgard_ddh.SIFEDamgard` and
:class:`~pyfe4ai.schemes.sife.ddh_dynamic.SIFEDynamic`.

DDH
---

.. automodule:: pyfe4ai.schemes.sife.ddh
   :members:
   :undoc-members:

DDH over ristretto255
---------------------

.. automodule:: pyfe4ai.schemes.sife.ddh_ristretto255
   :members:
   :undoc-members:

DDH Dynamic
-----------

.. automodule:: pyfe4ai.schemes.sife.ddh_dynamic
   :members:
   :undoc-members:

Damgård DDH
-----------

.. automodule:: pyfe4ai.schemes.sife.damgard_ddh
   :members:
   :undoc-members:

LWE
---

.. automodule:: pyfe4ai.schemes.sife.lwe
   :members:
   :undoc-members:

Fully-Secure LWE
-----------------

.. automodule:: pyfe4ai.schemes.sife.fullysec_lwe
   :members:
   :undoc-members:

Ring-LWE
--------

.. automodule:: pyfe4ai.schemes.sife.ring_lwe
   :members:
   :undoc-members:

Adaptively Secure Ring-LWE
--------------------------

.. automodule:: pyfe4ai.schemes.sife.fullysec_ring_lwe
   :members:
   :undoc-members:

Paillier
--------

.. automodule:: pyfe4ai.schemes.sife.paillier
   :members:
   :undoc-members:

FH-IPE (Pairing)
-----------------

.. automodule:: pyfe4ai.schemes.sife.fh_ipe_pairing
   :members:
   :undoc-members:

Partial FH-IPE (Pairing)
-------------------------

.. automodule:: pyfe4ai.schemes.sife.part_fh_ipe_pairing
   :members:
   :undoc-members:
