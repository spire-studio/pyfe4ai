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
and ``sife/fullysec_lwe``, ``mife/fullysec_lwe`` (ALS16) and
``mcfe/paillier_decentralized`` are refused. :mod:`pyfe4ai.schemes.profiles`
below gives the parameters per family, including ``matrix_mode`` for the
ABDP15 LWE schemes.

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
Gaussian ``D_{Z,sigma}``. There are two samplers.

``"facct"`` (:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_facct`)
samples ``D_{Z,sigma}`` exactly, to about 2^-185 per sample, for every width
(natively below ``2^448``). It is the default -- recorded as
``gaussian_sampler`` in the public parameters, whatever the library-wide
choice -- of the schemes whose proofs budget ``D_{Z,sigma}``: the Ring-LWE
MIFE / MCFE family (``sife``, ``mife``, ``mcfe`` ``ring_lwe``, the
decentralized and threshold MCFE), the dense-LWE MCFE keys ``t``
(``mcfe/lwe``, ``fullysec_lwe``, ``lwe_threshold``, ``lwe_decentralized``)
and ``sife/fullysec_ring_lwe``.

``"rounded"`` rounds a continuous Gaussian; it is the library-wide default
elsewhere and selectable in those schemes (``gaussian_sampler =
"rounded"``). The *ideal* rounded Gaussian (exact uniforms and arithmetic) is
a mixture of centred discrete Gaussians, about ``0.02 / sigma^2`` from
``D_{Z,sigma}`` per sample: 2^-9 at ``sigma = 3.2``, about 2^-33 per sample at
the Ring-LWE ``sigma1`` -- about 2^-18 for one MIFE key of ``eta ring_n``
samples. Its implementations add their own terms (crypto review, October
2026):

* the double-precision path
  (:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_matrix` and the
  native kernels, widths up to ``2^40``): the two outputs of a Box-Muller pair
  share one radius (rings that show in the tail), exact half-integer ties
  round to even (about ``2^-53.8 sigma`` per sample), and no value exceeds
  about ``8.57 sigma`` (the largest radius 53-bit uniforms give). One
  ciphertext's noise is about 2^-12..2^-4 and one key about 2^-25..2^-15 from
  i.i.d. ``D_{Z,sigma}``, against budgets of 2^-128: use it for toy parameters
  and benchmarks, not where a proof needs ``D_{Z,sigma}``;
* the MPFR path (:func:`~pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint`,
  wider widths: the DCR keys and masks, all threshold flooding) differs from
  the ideal rounded Gaussian only by its arithmetic, at ``bits(sigma) + 320``
  bits (``BIGINT_GUARD_BITS``: ``bits(sigma) + stat_param + ceil(log2 N) +
  8`` for every accepted ``stat_param``, at most 256, and up to 2^56 samples
  per key). At the earlier ``bits(sigma) + 128`` it was 2 to 19 bits short
  of the budgets. The threshold schemes' flooding always uses it
  (``FLOODING_SAMPLER``, recorded in their decryption keys), whatever the
  sampler selected elsewhere.

The library-wide choice applies to the other schemes:

.. code-block:: python

   from pyfe4ai.utils.sampling_utils import gaussian_sampler, set_gaussian_sampler

   with gaussian_sampler("facct"):     # this context
       ...
   set_gaussian_sampler("facct")       # the whole process

or ``PYFE4AI_GAUSSIAN_SAMPLER=facct`` in the environment. ``"facct"``
costs 0.5 to 3 microseconds per sample natively (several in CPython). Its
algorithm needs no secret-dependent branches, but CPython and big-integer
arithmetic are not constant-time, so neither sampler resists timing side
channels. For a given bit source its output is the same on every platform
and on both backends, so it comes with known-answer tests. Parties that
derive noise from a shared seed (the threshold schemes' flooding noise)
must use the same sampler.

The schemes above take ``gaussian_sampler = "rounded"`` in their key
generator or crypto config to select the rounded sampler; the key generator
publishes its choice in ``pp`` as the encryptors' default, and public
parameters without it (earlier versions) mean ``"facct"``. With the rounded
sampler, ``sife/fullysec_ring_lwe``'s keys at ``eta = 1`` are about 2^-47
from ``D_{Z,sigma1}`` for the ideal rounded Gaussian (``0.02 / sigma1^2``
over its ``m n eta`` key samples) and about 2^-12 with the double-precision
path's artifacts, terms its adaptive proof does not budget.

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
