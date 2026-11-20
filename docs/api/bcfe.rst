BCFE — Bounded-Key Functional Encryption for Circuits
=====================================================

.. automodule:: pyfe4ai.schemes.bcfe
   :no-members:

The family has one scheme, ``bcfe/gc_ristretto255`` (status *experimental*,
fidelity *adapted*). Its key generator and crypto class are importable from
:mod:`pyfe4ai.schemes.bcfe` under their own names; its circuits come from
:mod:`pyfe4ai.circuits`, which ``import pyfe4ai`` imports. Read
:ref:`bcfe-security` before use.

What the scheme is
------------------

A public boolean circuit ``C(w, x)`` is fixed at setup: a library circuit of
:mod:`pyfe4ai.circuits` such as the decision of a linear model,
:func:`~pyfe4ai.circuits.library.linear_threshold`. A functional key fixes the
secret parameter ``w`` (the weights and bias of the model). Its holder learns
``C(w, x)`` from a ciphertext of ``x``, and nothing else about ``x``. Anyone
with the public parameters encrypts.

Every key is a *slot*: one instance of the single-key scheme of Sahai and
Seyalioglu ("Worry-Free Encryption", ACM CCS 2010, SS10, Sec. 3.3), with the
guard bit of their Sec. 3.2, issued for one function. A ciphertext garbles
``C`` once per slot with the half-gates scheme of Zahur, Rosulek and Evans
(EUROCRYPT 2015, ZRE15, Fig. 2), with fresh randomness for every slot, and
encrypts the labels of the key wires to the slot's public keys (hashed
ElGamal over ristretto255). Three properties follow, and they decide where
the scheme fits:

* **At most** ``max_keys`` **keys are issued in all.** Every distinct
  (requester, function) takes a slot until ``max_keys`` slots exist
  (:meth:`~pyfe4ai.schemes.bcfe.gc_ristretto255.BCFEGCRistretto255KeyGenerator.set_max_keys`
  raises the cap). This bounds the keys issued, not the keys that collude:
  any number of the issued keys may collude. It is *not* the bounded-collusion
  notion of Gorbunov, Vaikuntanathan and Wee (CRYPTO 2012) or Ananth and
  Vaikuntanathan (TCC 2019), whose setups for ``Q`` keys let ``Q`` keys be
  issued at any time, also after the ciphertexts.
* **A key decrypts only ciphertexts made after its slot existed.** A
  ciphertext has one component for each slot of the public parameters it was
  made with (or for each slot the encryptor selects,
  ``encrypt(x, slots=...)``). A key issued later has no component in it, and
  ``decrypt`` says so (:class:`~pyfe4ai.core.exceptions.FEKeyError`). Issue
  the keys first, then hand the public parameters to the encryptors.
* **A ciphertext grows with the slots it is encrypted to**, and so does the
  time to make it (`Sizes`_).

So the scheme is for small models (a linear score or decision, a small
network), few keys (a few to a few dozen) and keys issued before the data is
encrypted.

What the scheme is not
----------------------

* **Not function-hiding.** The key holder learns ``w``: its key carries it.
  The authority learns it too (both issuance modes send it ``w``). The
  encryptors do not: the slot list does not show it. What the scheme
  protects is ``x``, from the key holders.
* **Not simulation-secure, and not post-quantum.** The claim is IND
  security against classical adversaries; a ciphertext recorded today gives
  ``x`` up to a future quantum computer.
* **Not secure against a malicious encryptor.** An encryptor that garbles
  wrongly can make a key holder's result a function of ``w`` of its choice.
* **Not an access control.** The issuance policy is the authority's
  decision on whom it builds a slot for; the cryptography enforces only that
  a key computes ``C(w, .)``.
* **Not for large models.** One ciphertext component of a 100-feature linear
  model with 16-bit weights and features is about 1.1 MB (`Sizes`_).

The security claim, its assumptions and everything it does not cover are on
:ref:`bcfe-security`.

Usage
-----

The authority issues keys (``key_mode = "h1"``, the default), a data owner
encrypts, a key holder decrypts:

.. code-block:: python

   import pyfe4ai
   from pyfe4ai import circuits
   from pyfe4ai.policy import parse

   model = circuits.linear_threshold(n=4, bw=8, bx=8)     # [sum_i w_i x_i + bias >= 0]
   kg, BCFE = pyfe4ai.create("bcfe/gc_ristretto255", circuit=model, max_keys=4,
                             issuance_policy=parse("and(org:bank, role:scoring)"))
   kg.setup()                         # its state: <cache_dir>/authority/bcfe/<setup digest>

   key = kg.issue_key({"weights": [3, -2, 1, -4], "bias": 10},
                      requester=authenticated_id,          # the id your application authenticated
                      attributes={"org:bank", "role:scoring"}, label="credit-v3")
   pp = kg.get_public_parameters()    # the slot list, with the new slot: hand it to the data owners

   ct = BCFE({"keys": {"pp": pp}}).encrypt({"x": [5, 1, 2, 0]})   # the data owner
   BCFE({"keys": {"pp": pp}}).decrypt(ct, key)                    # the key holder: {"decision": 1}

With ``key_mode = "h2"`` the key holder makes the key pairs of its slot
itself (SS10's Pre, Auth and CheckAuth), and the authority alone cannot
decrypt that slot:

.. code-block:: python

   kg, BCFE = pyfe4ai.create("bcfe/gc_ristretto255", circuit=model, max_keys=4, key_mode="h2",
                             issuance_policy=parse("and(org:bank, role:scoring)"))
   kg.setup()

   holder = BCFE.key_holder(kg.get_public_parameters())    # the key holder
   # the request holds w and m + 1 public keys: it goes to the authority only
   request = holder.request({"weights": [3, -2, 1, -4], "bias": 10})
   grant = kg.authorize(request, requester=authenticated_id,
                        attributes={"org:bank", "role:scoring"}, label="credit-v3")
   key = holder.finish(grant)                   # CheckAuth, then the key
   pp = kg.get_public_parameters()              # the slot list, with the new slot

The examples ``examples/bcfe_credit_scoring.py`` and
``examples/bcfe_holder_generated_keys.py`` run both flows.

Issuance
--------

:meth:`~pyfe4ai.schemes.bcfe.gc_ristretto255.BCFEGCRistretto255KeyGenerator.issue_key`
(h1) and
:meth:`~pyfe4ai.schemes.bcfe.gc_ristretto255.BCFEGCRistretto255KeyGenerator.authorize`
(h2) take the same steps in the same order, and a step that fails changes
nothing:

1. ``w`` through the circuit's key layout (out of range:
   :class:`~pyfe4ai.core.exceptions.FEValidationError`);
2. the issuance policy on the requester's attributes
   (:func:`~pyfe4ai.policy.formula.evaluate`), before anything is reserved
   (:class:`~pyfe4ai.core.exceptions.FEKeyError`);
3. deduplication per requester: the same ``requester`` asking again for the
   same ``w`` gets the same key, byte for byte (h1, with
   ``keep_issued_keys``, the default), or, for the same request, the same
   grant (h2; a new request with other public keys is refused, since the
   authority does not replace a slot's keys); another requester with the
   same ``w`` gets a slot of its own. The library cannot
   authenticate ``requester``: pass the id that your application has
   authenticated;
4. the cap: once ``max_keys`` slots exist, every request gets the same
   :class:`~pyfe4ai.core.exceptions.FEKeyError`, whether step 3 found a slot
   or not;
5. the slot: its public record is appended to the slot list, the
   authority's private record kept beside it.

In both modes the public keys of the positions a key does not select are
hashed to the group from a fresh secret ``k`` of the slot, so no secret key
of them exists. With h1 the authority makes the selected key pairs and, by
default, keeps every key it issued (it can then decrypt every slot it
kept). With h2 the holder makes them and sends only the public keys; the
authority keeps no key of the slot, so a lost key is not issued again (ask
for a new slot, with another function or requester id), and the grant
carries ``k``, which reveals ``w``: it goes to the key holder only.

The authority's state is a directory, ``<cache_dir>/authority/bcfe/<setup
digest>``: ``setup.json`` (public), ``authority.json`` and the slot store's
files (:class:`~pyfe4ai.schemes.common.slots.SlotStore`: the public slot
list, and a private file of mode ``0o600`` with the slot secrets ``k``, the
kept keys and the deduplication index). Keep it as you would a master key;
:meth:`~pyfe4ai.schemes.bcfe.gc_ristretto255.BCFEGCRistretto255KeyGenerator.open`
resumes an instance from it. The store is append-only and crash-consistent:
a crash at any step never gives a slot two keys, and at worst uses up a
slot id or a slot. Threads, other authority objects and other processes may
issue on one directory at once (a lock per store object and a file lock,
``fcntl``, so POSIX only); a forked child renews the store's lock
(:doc:`../threading`).

Two rules for the directory, which the library cannot check:

* **One copy in use.** The locks order the calls on one directory, not on
  its copies. Two copies (a second host, or a backup restored while the
  original goes on issuing) each hand out the same slot ids again, to
  different slots, and each counts its own ``max_keys``. Every slot still
  has one key, and the slot list of one copy still holds at most its
  ``max_keys`` slots, but a key of one copy fails the key check against a
  slot of the same id in the other. Issue from one copy only. After
  restoring a backup, issue from the restored directory alone, publish its
  public parameters again, and treat the slots issued after the backup as
  lost: their holders decrypt the ciphertexts made before the restore, not
  those made under the restored slot list.
* **No ACL.** The store trusts a directory and files of its own user that
  no other user can write, checked through one descriptor of the directory
  per call (a directory renamed or swapped during a call is not used). It
  reads the mode bits only: a Linux ACL entry that lets another user write
  shows in the group bits and is refused, a macOS ACL (``chmod +a``, shown by
  ``ls -led``) does not, so do not put one on the directory.

The key holder of h2
(:class:`~pyfe4ai.schemes.bcfe.gc_ristretto255.BCFEGCRistretto255KeyHolder`)
keeps the secret keys of its open requests and is never copied or pickled
(``TypeError``).

Sizes
-----

One slot of each library circuit below, in raw bytes; the JSON records carry
them in base64, a third more. The linear circuits have 100 features:
:func:`~pyfe4ai.circuits.library.linear_threshold` (the decision
``[sum_i w_i x_i + bias >= 0]``) and
:func:`~pyfe4ai.circuits.library.linear_score` (the score itself); the MLP
is :func:`~pyfe4ai.circuits.library.mlp` with 64 features, 16 hidden ReLU
units and one output decision, all values 8-bit (``shift = 8``).

.. list-table::
   :header-rows: 1
   :widths: 34 12 10 12 14 18

   * - Circuit
     - ``data_wires``
     - Key bits ``m``
     - Table entries
     - Slot public keys (in ``pp``)
     - Ciphertext component
   * - Linear decision, 16-bit weights and features
     - ``"hardwired"``
     - 1632
     - 64,459
     - 104,480
     - 1,083,617
   * - the same
     - ``"labels"``
     - 1632
     - 96,248
     - 104,480
     - 1,617,841
   * - Linear decision, 8-bit weights and features
     - ``"hardwired"``
     - 816
     - 16,427
     - 52,256
     - 288,993
   * - the same
     - ``"labels"``
     - 816
     - 25,816
     - 52,256
     - 452,017
   * - Linear decision, 16-bit weights, 8-bit features
     - ``"hardwired"``
     - 1624
     - 29,243
     - 103,968
     - 519,905
   * - the same
     - ``"labels"``
     - 1624
     - 45,032
     - 103,968
     - 785,329
   * - Linear score, 16-bit weights and features
     - ``"hardwired"``
     - 1632
     - 64,535
     - 104,480
     - 1,084,837
   * - the same
     - ``"labels"``
     - 1632
     - 96,324
     - 104,480
     - 1,619,061
   * - MLP 64-16-1, 8-bit
     - ``"hardwired"``
     - 8593
     - 174,054
     - 549,984
     - 3,059,889
   * - the same
     - ``"labels"``
     - 8593
     - 241,510
     - 549,984
     - 4,147,377

.. list-table:: Median times of one slot, 1 / 4 threads (milliseconds; Apple M4 Max, native backend, ``benchmarks/run.py bcfe --runs 7``)
   :header-rows: 1
   :widths: 34 14 16 18 18

   * - Circuit
     - ``data_wires``
     - Issue (h1)
     - Encrypt
     - Decrypt
   * - Linear decision, 16-bit
     - ``"hardwired"``
     - 23.9 / 7.3
     - 83.2 / 35.8
     - 21.1 / 12.6
   * - Linear decision, 16-bit
     - ``"labels"``
     - 23.9 / 7.3
     - 90.9 / 43.6
     - 25.0 / 16.5
   * - Linear decision, 8-bit
     - ``"hardwired"``
     - 12.0 / 3.7
     - 37.4 / 13.7
     - 8.7 / 4.5
   * - Linear decision, 8-bit
     - ``"labels"``
     - 12.0 / 3.7
     - 39.7 / 16.0
     - 9.8 / 5.5
   * - Linear decision, 16-bit weights, 8-bit features
     - ``"hardwired"``
     - 23.8 / 7.3
     - 73.6 / 26.4
     - 16.3 / 7.8
   * - Linear score, 16-bit
     - ``"hardwired"``
     - 24.0 / 7.3
     - 83.1 / 35.7
     - 21.1 / 12.7
   * - MLP 64-16-1, 8-bit
     - ``"hardwired"``
     - 126.2 / 38.2
     - 395.0 / 146.2
     - 87.4 / 42.7
   * - MLP 64-16-1, 8-bit
     - ``"labels"``
     - 126.2 / 38.3
     - 411.4 / 163.0
     - 95.6 / 50.7

Encryption is dominated by the label encryption (one variable-base
multiplication per label), not by the garbling; it scales with the number of
slots a ciphertext is encrypted to.

A ciphertext to ``k`` slots is ``k`` components and a 16-byte nonce, and a
key ``32 (m + 1)`` bytes of scalars. ``data_wires = "hardwired"`` (the
default) folds the data bits into the garbled tables; ``"labels"`` is SS10's
form, with the data bits as input labels in the ciphertext.
For another circuit, :meth:`~pyfe4ai.circuits.circuit.Circuit.stats` of
``circuit.guard()`` gives the table entries of a slot (the scheme garbles
``guard(C)``), and ``python benchmarks/run.py bcfe`` in a
source checkout measures the sizes and the times of issuing, encrypting and
decrypting one slot on the machine at hand. The native extension runs the
garbling and the slot operations; the Python implementation gives the same
bytes, far more slowly (its ristretto255 arithmetic is pure Python), and is
meant for small circuits and tests.

Public and internal modules
---------------------------

:mod:`pyfe4ai.schemes.bcfe` (the key generator, the crypto class and the key
holder it returns) and :mod:`pyfe4ai.circuits` are public. Like the rest of
an experimental scheme, their classes, options and records may still change.
:mod:`pyfe4ai.core.circuit_base`, :mod:`pyfe4ai.core.formats.garbled` (the
records, documented and versioned there),
:mod:`pyfe4ai.schemes.common.slots` and the primitives
(:mod:`pyfe4ai.primitives.garbling`, :mod:`pyfe4ai.primitives.bcfe_kem`, on
:doc:`primitives`) are internal.

Garbled Circuits over ristretto255
----------------------------------

.. automodule:: pyfe4ai.schemes.bcfe.gc_ristretto255
   :members:
   :undoc-members:

Circuits
--------

.. automodule:: pyfe4ai.circuits
   :no-members:

Circuit
^^^^^^^

.. automodule:: pyfe4ai.circuits.circuit
   :members:
   :undoc-members:

Builder
^^^^^^^

.. automodule:: pyfe4ai.circuits.builder
   :members:
   :undoc-members:

Layouts
^^^^^^^

.. automodule:: pyfe4ai.circuits.layout
   :members:
   :undoc-members:

Library Circuits
^^^^^^^^^^^^^^^^

.. automodule:: pyfe4ai.circuits.library
   :members:
   :undoc-members:

Base Classes
------------

.. automodule:: pyfe4ai.core.circuit_base
   :members:
   :undoc-members:

Serialized Objects
------------------

.. automodule:: pyfe4ai.core.formats.garbled
   :members:

Slot Store
----------

.. automodule:: pyfe4ai.schemes.common.slots
   :members:
   :undoc-members:
