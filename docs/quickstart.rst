Quick start
===========

Functional encryption lets a key holder learn a chosen function of
encrypted data -- here an inner product -- and nothing else. Every scheme
in PyFE4AI has two objects: a **key generator**, which sets up the public
parameters and issues keys, and a **crypto class**, whose objects encrypt
(with the public parameters and a client's key) or decrypt (with a
functional key).

A first example
---------------

.. code-block:: python

   import pyfe4ai

   x, y = [2, 1, 3], [4, 5, 6]

   # sec_param = bit length of the DDH modulus (NOT a security level);
   # 128 keeps the demo fast but is insecure -- for real use replace it with
   # "security_profile": "standard" (a 3072-bit group)
   kg, SIFEDDH = pyfe4ai.create("sife/ddh", {"sec_param": 128, "eta": len(x)})
   kg.setup()
   pp, sk = kg.get_public_parameters(), kg.get_private_keys()
   dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

   ct = SIFEDDH({"precision": 3, "keys": {"pp": pp, "sk": sk}}).encrypt(x)
   result = SIFEDDH({"precision": 3, "keys": {"pp": pp}}).decrypt(ct, dk, y)
   print(result)  # <x, y> = 31

:func:`pyfe4ai.create <pyfe4ai.core.registry.create>` takes a scheme id
``"family/variant"`` and a config, and returns the scheme's key generator --
built, not yet set up -- with its crypto class. Options can also be passed
as keywords: ``pyfe4ai.create("sife/ddh", eta=3)``. The classes are
importable by name too (``from pyfe4ai.schemes.sife import SIFEDDH,
SIFEDDHKeyGenerator``), named after the id.

The key generator's methods are the same in every family:

* ``setup()`` draws the parameters and the master key;
* ``get_public_parameters()`` returns ``pp``, for everyone;
* ``get_private_keys(nid)`` returns a client's encryption key (``sk``);
* ``get_decryption_keys(sid, credentials={"fusion_weight": y, ...})``
  returns a functional key for the weights ``y`` (in the threshold schemes,
  one server's key share).

The crypto objects take ``{"keys": {"pp": pp, "sk": sk}}`` to encrypt and
``{"keys": {"pp": pp}}`` to decrypt; their ``encrypt`` and ``decrypt``
arguments depend on the family (a vector, a vector and a label, ...). The
module of each scheme documents them (:doc:`api/index`), and the
:doc:`glossary` lists the configuration keys.

Several clients
---------------

In multi-client FE (MCFE) every client encrypts its own vector under a
shared label, such as a training round; the functional key reveals only the
weighted sum over the clients, and only for ciphertexts of one label:

.. code-block:: python

   import pyfe4ai

   x = {"alice": [1, 2], "bob": [3, -1], "carol": [0, 4]}
   y = {"alice": [2, 1], "bob": [1, 1], "carol": [-1, 3]}

   kg, MCFEDDH = pyfe4ai.create("mcfe/ddh", {"eta": 2, "n": 3, "lst_nid": list(x)})
   kg.setup()
   pp = kg.get_public_parameters()

   label = "round-1"
   cts = {nid: MCFEDDH({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
          for nid in x}

   dk = kg.get_decryption_keys("aggregator", credentials={"fusion_weight": y})
   print(MCFEDDH({"id": "aggregator", "keys": {"pp": pp}}).decrypt(cts, dk, y, label))  # 18

A client must encrypt at most one vector per label; the MCFE schemes assume
it, and the ML adapter below enforces it.

Federated learning
------------------

:mod:`pyfe4ai.ml` wraps a scheme for gradient aggregation:

.. code-block:: python

   import numpy as np
   from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

   clients = ["c0", "c1", "c2"]
   wrapper = FESchemeWrapper("mcfe", "ring_lwe",
                             {"eta": 1, "n": 3, "lst_nid": clients, "bound_x": 10**4, "bound_y": 10},
                             precision=3)
   grads = {nid: np.random.default_rng(i).uniform(-1, 1, 1000) for i, nid in enumerate(clients)}
   enc = {nid: encrypt_gradient(g, wrapper, nid=nid, label="round-1") for nid, g in grads.items()}
   total = aggregate_gradients(enc, wrapper, {nid: 1 for nid in clients}, label="round-1")
   print(np.allclose(total, sum(grads.values()), atol=1e-2))  # True

The wrapper runs the key generator, so it holds the master key and plays
every party: it is meant for experiments. :doc:`threshold_ml` covers the
adapter's policy checks, the threshold MCFE whose decryption servers it
runs, and the separate messages of a deployment.

Real parameters
---------------

The default ``security_profile="toy"`` keeps parameters small: fast, and
**not secure**. ``"standard"`` aims at 128-bit security and refuses explicit
values below it:

.. code-block:: python

   import pyfe4ai

   kg, SIFEDDH = pyfe4ai.create("sife/ddh", {"eta": 3, "security_profile": "standard"})  # 3072-bit group

The pairing-based schemes stay at about 120 to 126 bits (BLS12-381, with a
:class:`~pyfe4ai.core.exceptions.FESecurityWarning`), and four schemes have
no parameter set that is both secure and usable, so ``standard`` refuses
them: see :doc:`security` and :doc:`configuration`.

Finding a scheme
----------------

.. code-block:: python

   import pyfe4ai

   pyfe4ai.list_schemes()                                  # all 42 ids
   pyfe4ai.list_schemes("mcfe", capabilities="threshold")  # the threshold MCFEs
   scheme = pyfe4ai.get_scheme("mcfe/ring_lwe")
   print(scheme.kind, sorted(scheme.capabilities), scheme.status.status)

:doc:`choosing` explains the families, their capabilities and review
status, and which scheme fits which task.
