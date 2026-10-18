Security & Status
=================

PyFE4AI is a research prototype: no scheme is hardened against side
channels, and the default parameters are toy values. This page lists the
review status of every scheme and the parameter presets; the README sections
`Scheme Status <https://github.com/spire-studio/pyfe4ai#scheme-status>`__,
`Security Profiles <https://github.com/spire-studio/pyfe4ai#security-profiles>`__
and `Known Issues / Security Notice
<https://github.com/spire-studio/pyfe4ai#known-issues--security-notice>`__
add measured costs and the remaining limitations.

Scheme status
-------------

|n_schemes| registered schemes: |n_reviewed| reviewed, |n_experimental|
experimental, |n_unreviewed| unreviewed and |n_known_insecure|
known-insecure. The experimental ones are the threshold layers of tMIFE-LWE,
tMCFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE and the carry mask of
dMCFE-DDH: our own constructions on top of published schemes, with no
published security proof. Query a scheme at runtime with
:func:`pyfe4ai.get_scheme_status(family, variant)
<pyfe4ai.schemes.status.get_scheme_status>` or
:attr:`FESchemeWrapper(...).status <pyfe4ai.utils.ml_adapter.FESchemeWrapper.status>`;
the key generator and crypto objects of a known-insecure scheme emit
:class:`~pyfe4ai.utils.exceptions.FESecurityWarning`. The table is rendered
from :data:`~pyfe4ai.schemes.status.SCHEME_STATUS`; the status and fidelity
values are defined in :mod:`pyfe4ai.schemes.status`.

.. scheme-status-table::

Security profiles
-----------------

The ``security_profile`` entry of a key generator's config selects its
parameters: ``"toy"`` (the default) is fast and **not secure**;
``"standard"`` aims at 128-bit classical security and refuses explicit
values below it. Its exceptions: the pairing-based schemes run on
BLS12-381, about 120-126-bit after the exTNFS attacks (with a
:class:`~pyfe4ai.utils.exceptions.FESecurityWarning`),
and ``sife/fullysec_lwe``, ``mife/fullysec_lwe`` (ALS16),
``mife/lwe_threshold`` and ``mcfe/paillier_decentralized`` are refused.
:mod:`pyfe4ai.schemes.profiles` below gives the parameters per family,
including ``matrix_mode`` for the ABDP15 LWE schemes.

For the ABDP15 LWE schemes (``sife/lwe``, ``mife/lwe``) the 128 bits are
those of the LWE problem that their security proof reduces to. The
lattice-estimator table is read at the error width
``sigma' = sigma / (1 + B sqrt(eta))`` of that problem (``B = 2 X``, or
``4 X`` per slot of ``mife/lwe``), not at the wider ``sigma`` of the public
key. That makes the dimension 17 to 67 percent larger for
``8 <= X = Y <= 1000``, and up to about twice as large at ``X = Y = 1``. The
proof is the second
generic theorem of the ABDP15 full version (ePrint 2015/017, written out in
F. Bourse's PhD thesis, ENS 2017, Thm. 4.3.5), not Theorem 2 of the PKC 2015
paper, whose simulation does not cover noisy LWE keys; this rests on the
thesis until the full-version text has been checked.

``mife/lwe_threshold`` is refused, like the ALS16 schemes. Once its modulus
is enlarged for flooding (``stat_param = 128``), it needs a dimension at the
very end of the table at that width (5888 to 6144, with a public matrix of
6.5 to 7.8 * 10^9 entries) for a few small bounds, e.g. ``X = Y`` in
{1, 2, 3, 8, 10} at ``eta = 1``, and beyond it for most others (``X = Y`` =
4 to 6, ``X = Y >= 12``, most ``eta >= 2``).

An opt-in KL-divergence accounting of the flooding noise (in development)
would bring it to about 4864 for two clients, ``s = t = 3`` and ``2^40``
flooded values. It applies only with ``flooding_accounting = "kl"``,
``t = s`` and an explicit ``max_partial_decryptions`` budget; the public
matrix would still have 4.1 * 10^9 entries (``"stream"`` mode), and the
guarantee is weaker, about ``stat_param - 6.85`` bits. The refusal is to be
re-evaluated when that accounting is merged.

Threshold trust model
---------------------

The decryption servers of the threshold schemes (tMIFE, tMCFE) are
honest-but-curious: fewer than ``t`` of them learn nothing beyond the result,
but a partial decryption carries no proof of correctness, so one deviating
server changes the result unnoticed. ``combine_decrypt`` accepts partial
decryptions for several enrolled sets and raises if they disagree (see
:func:`~pyfe4ai.schemes.threshold_utils.combine_consistently`). In
tMIFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE, servers only decrypt
ciphertexts carrying their clients' MAC tags.

ML adapter
----------

:class:`~pyfe4ai.utils.ml_adapter.FESchemeWrapper` runs the key generator
itself, so it holds the master secret key and plays clients, key authority
and decryptor in one trust domain: use it for experiments, not as a
deployment (see :mod:`pyfe4ai.utils.ml_adapter`).

Gaussian sampling
-----------------

The lattice and DCR proofs sample secrets and noise from the discrete
Gaussian ``D_{Z,sigma}``. The default sampler, ``"rounded"``, rounds a
continuous Gaussian: per sample it is about ``0.02 / sigma^2`` away in
statistical distance (2^-9 at ``sigma = 3.2``, about 2^-33 at the Ring-LWE
``sigma1``), and the double-precision path never returns values beyond
about ``8.6 sigma``. The opt-in sampler ``"facct"``
(:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_facct`) samples
``D_{Z,sigma}`` exactly, to about 2^-185 per sample, for every width:

.. code-block:: python

   from pyfe4ai.utils.sampling_utils import gaussian_sampler, set_gaussian_sampler

   with gaussian_sampler("facct"):     # this context
       ...
   set_gaussian_sampler("facct")       # the whole process

or ``PYFE4AI_GAUSSIAN_SAMPLER=facct`` in the environment. It costs a few
microseconds per sample in CPython, several times the rounded sampler. Its
algorithm needs no secret-dependent branches, but CPython and big-integer
arithmetic are not constant-time, so neither sampler resists timing side
channels; ``"facct"`` is the reference for a native implementation that
can. For a given bit source its output is the same on every platform, so it
comes with known-answer tests. Parties that derive noise from a shared seed
(the threshold schemes' flooding noise) must use the same sampler.

Status API
----------

.. automodule:: pyfe4ai.schemes.status
   :members:
   :undoc-members:

Profiles API
------------

.. automodule:: pyfe4ai.schemes.profiles
   :members:
   :undoc-members:
