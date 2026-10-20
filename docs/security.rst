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
<pyfe4ai.core.status.get_scheme_status>` or
:attr:`FESchemeWrapper(...).status <pyfe4ai.ml.FESchemeWrapper.status>`;
the key generator and crypto objects of a known-insecure scheme emit
:class:`~pyfe4ai.core.exceptions.FESecurityWarning`. The table is rendered
from :data:`~pyfe4ai.core.status.SCHEME_STATUS`; the status and fidelity
values are defined in :mod:`pyfe4ai.core.status`.

.. scheme-status-table::

Security profiles
-----------------

The ``security_profile`` entry of a key generator's config selects its
parameters: ``"toy"`` (the default) is fast and **not secure**;
``"standard"`` aims at 128-bit classical security and refuses explicit
values below it. Its exceptions: the pairing-based schemes run on
BLS12-381, about 120-126-bit after the exTNFS attacks (with a
:class:`~pyfe4ai.core.exceptions.FESecurityWarning`),
and ``sife/fullysec_lwe``, ``mife/fullysec_lwe`` (ALS16),
``mife/lwe_threshold`` and ``mcfe/paillier_decentralized`` are refused.
:mod:`pyfe4ai.core.profiles` below gives the parameters per family,
including ``matrix_mode`` for the ABDP15 LWE schemes.

For the ABDP15 LWE schemes (``sife/lwe``, ``mife/lwe``) the 128 bits are
those of the LWE problem that their security proof reduces to. The
lattice-estimator table is read at the error width
``sigma' = sigma / (1 + B sqrt(eta))`` of that problem (``B = 2 X``, or
``4 X`` per slot of ``mife/lwe``), not at the wider ``sigma`` of the public
key. That makes the dimension 17 to 67 percent larger for
``8 <= X = Y <= 1000``, and up to about twice as large at ``X = Y = 1``.

The proof is the ABDP15 full version (ePrint 2015/017, Oct. 2015), Thm. 4.4
with Thms. 6.6 and 6.7 (Construction 6.5, Sec. 6.2), written out in
F. Bourse's PhD thesis (ENS 2017), Thm. 4.3.5 with Thms. 8.2.2, 8.2.3,
8.3.2. It is not Theorem 2 of the PKC 2015 paper, whose simulation does not
cover noisy LWE keys. The width ``sigma'`` is the condition derived in the
proof of Thm. 6.6 (= Bourse, Thm. 8.2.2); the full version's Sec. 6.2
parameter list states the stronger, sufficient ``(l B + 1) sigma'``.

Hypotheses of the papers that the library meets differently:

- The literal ``p > l B^2`` of Thm. 4.4 / Sec. 6.2 is not needed for the
  Regev instance. The change of basis acts on secret keys mod the LWE prime
  ``q``. The library decodes the integer inner product (``p > 2 eta X Y +
  1``), so admissible keys are orthogonal to ``x1 - x0`` over ``Z``.
- The ``4 X`` per slot of ``mife/lwe`` is the library's ACFGU18 composition,
  covered by neither text.
- The thesis's LWE hardness bound (an absolute width above ``2 sqrt(n)``;
  not stated in the full version) is met under ``standard``:
  ``sigma' >= 2^16``, the table's smallest width.

``mife/lwe_threshold`` is refused, like the ALS16 schemes. Once its modulus
is enlarged for flooding (``stat_param = 128``), it needs a dimension at the
very end of the table at that width (5888 to 6144, with a public matrix of
6.5 to 7.8 * 10^9 entries) for a few small bounds, e.g. ``X = Y`` in
{1, 2, 3, 8, 10} at ``eta = 1``, and beyond it for most others (``X = Y`` =
4 to 6, ``X = Y >= 12``, most ``eta >= 2``).

The opt-in KL-divergence accounting of the flooding noise
(:mod:`pyfe4ai.schemes.threshold.kl`) would bring it to about 4864 for
two clients, ``s = t = 3`` and ``2^40`` flooded values. It applies only with
``flooding_accounting = "kl"``, ``t = s`` and an explicit
``max_partial_decryptions`` budget; the public matrix would still have
4.1 * 10^9 entries (``"stream"`` mode), and the guarantee is weaker, about
``stat_param - 6.85`` bits. The refusal under ``standard`` holds with either
accounting; ``"kl"`` applies to ``mife/lwe_threshold`` under ``toy``.

Threshold trust model
---------------------

The decryption servers of the threshold schemes (tMIFE, tMCFE) are
honest-but-curious: fewer than ``t`` of them learn nothing beyond the result,
but a partial decryption carries no proof of correctness, so one deviating
server changes the result unnoticed. ``combine_decrypt`` accepts partial
decryptions for several enrolled sets and raises if they disagree (see
:func:`~pyfe4ai.schemes.threshold.roles.combine_consistently`). In
tMIFE-LWE, tMCFE-Ring-LWE and tMCFE-FH-Multi-IPE, servers only decrypt
ciphertexts carrying their clients' MAC tags. With the binary ciphertexts of
tMCFE-Ring-LWE (``ct_format="binary"``, its default; ``"json"`` selects the
format of earlier versions) the tags also cover the digest of the public
parameters, and the encoding is unique (values strictly below ``q``), so that
the flooding noise, seeded with the ciphertexts' bytes, cannot be redrawn by
re-encoding a ciphertext. Its key shares are split by default: a server keeps
its share of the label-independent part of the key once per weights and
receives the shares of the round's label terms per round; a linear sharing
reused with fresh sharings of the label terms is a sharing of each round's
key, the shares of ``t - 1`` servers stay independent of the keys, and the
noise is seeded as for a whole key share (``key_shares="full"`` issues whole
key shares). A server refuses a label part without the weight part of the
same weights and share set, and key shares and ciphertexts of another format
or other public parameters; a combiner refuses partial decryptions of another
format, and partial decryptions of different sharings of one key: every
sharing the key generator issues has a random ``sharing_id`` that the
partials carry, since partials of two sharings (a label shared again after it
left the key generator's cache, ``LABEL_PARTS_KEPT``) would decode to a wrong
value inside the decoding bound rather than fail. A second sharing reveals
nothing: one server's answers under the two sharings differ by its share of
a sharing of zero, which ``t - 1`` servers compute from their own parts.
**Store and transmit the** ``sharing_id`` **with every key share and partial
decryption**: partials without one (as from key shares of earlier versions)
combine only with each other, so a storage or transport layer that drops the
id on every side would let partials of two sharings combine to a wrong value
unnoticed.
Partial decryptions carry no digest of the public parameters, so a
combiner given those of another deployment of the same format is not warned
(it holds no secret; under the honest-but-curious model above the servers
send the partials of its deployment).

ML adapter
----------

:class:`~pyfe4ai.ml.FESchemeWrapper` runs the key generator
itself, so it holds the master secret key and plays clients, key authority
and decryptor in one trust domain: use it for experiments, not as a
deployment (see :mod:`pyfe4ai.ml`). For
``mcfe/ring_lwe_threshold`` it also runs the decryption servers, so the
threshold adds no protection there; the separate steps
(:func:`~pyfe4ai.ml.partial_decrypt_gradients`,
:func:`~pyfe4ai.ml.combine_partial_decryptions`) show the
messages of a deployment, whose servers run in their own trust domains.

Gaussian sampling
-----------------

The lattice and DCR proofs sample secrets and noise from the discrete
Gaussian ``D_{Z,sigma}``. There are two samplers.

``"facct"`` (:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct`)
samples ``D_{Z,sigma}`` exactly, to about 2^-185 per sample, for every width
(natively below ``2^448``). It is the default of these schemes, which record
their sampler as ``gaussian_sampler`` in their public parameters: the
Ring-LWE MIFE / MCFE family (``sife``, ``mife``, ``mcfe`` ``ring_lwe``, the
decentralized and threshold MCFE), the dense-LWE MCFE keys ``t``
(``mcfe/lwe``, ``fullysec_lwe``, ``lwe_threshold``, ``lwe_decentralized``),
``sife/fullysec_ring_lwe`` and, under ``standard``, the noise ``E`` of the
ABDP15 public keys (``sife/lwe`` and ``mife/lwe``; ``mife/lwe_threshold``,
whose MIFE layer would do the same, is refused under ``standard``). For
them, in this order: the
``gaussian_sampler`` of the key generator's or crypto object's config; else
the one the public parameters record; else, under the ``toy`` profile only
and not for ``sife/fullysec_ring_lwe``, the one a ``gaussian_sampler()``
block selects; else ``"facct"``. The process-wide choice
(``set_gaussian_sampler``, ``PYFE4AI_GAUSSIAN_SAMPLER``) does not apply to
them. The ``standard`` profile refuses ``"rounded"`` for them, and a crypto
object whose config overrides a recorded ``"facct"`` with ``"rounded"``
warns (``FESecurityWarning``; refused under ``standard``).

ABDP15 under ``toy`` takes the ``gaussian_sampler`` of its key generator's
config, else the library-wide choice below, and records it as well.

``"rounded"`` rounds a continuous Gaussian. It is the library-wide default,
and so the sampler of the other lattice and DCR schemes unless chosen
otherwise: ALS16 (``sife`` / ``mife`` ``fullysec_lwe``), the DCR schemes and,
under ``toy``, ABDP15 SIFE / MIFE LWE (and the keys of tMIFE-LWE); the
schemes above can select it under ``toy`` (``gaussian_sampler =
"rounded"``). The *ideal* rounded Gaussian (exact uniforms and arithmetic) is
a mixture of centred discrete Gaussians, about ``0.02 / sigma^2`` from
``D_{Z,sigma}`` per sample: 2^-9 at ``sigma = 3.2``, about 2^-33 per sample at
the Ring-LWE ``sigma1`` -- about 2^-18 for one MIFE key of ``eta ring_n``
samples. Its implementations add their own terms (crypto review, October
2026):

* the double-precision path
  (:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_matrix` and the
  native kernels, widths up to ``2^40``): the two outputs of a Box-Muller pair
  share one radius (rings that show in the tail), exact half-integer ties
  round to even (about ``2^-53.8 sigma`` per sample), and no value exceeds
  about ``8.57 sigma`` (the largest radius 53-bit uniforms give). One
  ciphertext's noise is about 2^-12..2^-4 and one key about 2^-25..2^-15 from
  i.i.d. ``D_{Z,sigma}``, against budgets of 2^-128: use it for toy parameters
  and benchmarks, not where a proof needs ``D_{Z,sigma}``;
* the MPFR path (:func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint`,
  wider widths: the DCR keys and masks, all threshold flooding) differs from
  the ideal rounded Gaussian only by its arithmetic, at ``bits(sigma) + 320``
  bits (``BIGINT_GUARD_BITS``: ``bits(sigma) + stat_param + ceil(log2 N) +
  8`` for every accepted ``stat_param``, at most 256, and up to 2^56 samples
  per key). At the earlier ``bits(sigma) + 128`` it was 2 to 19 bits short
  of the budgets. The threshold schemes' Gaussian flooding always uses it
  (``FLOODING_SAMPLER``, recorded in their decryption keys), whatever the
  sampler selected elsewhere; the default uniform flooding of
  ``mcfe/ring_lwe_threshold`` uses no Gaussian sampler.

The library-wide choice -- which applies to the other schemes and to ABDP15
under ``toy`` only -- is
made per call (``sampler=``), per context or per process:

.. code-block:: python

   from pyfe4ai.primitives.sampling import gaussian_sampler, set_gaussian_sampler

   with gaussian_sampler("facct"):     # this context
       ...
   set_gaussian_sampler("facct")       # the whole process

or ``PYFE4AI_GAUSSIAN_SAMPLER=facct`` in the environment. ``"facct"``
costs 0.5 to 3 microseconds per sample natively (several in CPython). Its
algorithm needs no secret-dependent branches, but CPython and big-integer
arithmetic are not constant-time, so neither sampler resists timing side
channels. For a given bit source its output is the same on every platform
and on both backends, so it comes with known-answer tests. Parties that
derive noise from a shared seed (the Gaussian flooding noise of the
threshold LWE schemes, and of ``mcfe/ring_lwe_threshold`` with
``flooding="gaussian"``) must use the same sampler; the default
``flooding="uniform"`` of ``mcfe/ring_lwe_threshold`` draws its noise
without a Gaussian sampler (uniform on ``[-2^tau, 2^tau)`` from the seeded
SHAKE-256 stream, the same statistical distance per coefficient, see the
module).

The key generator of a scheme that records its sampler publishes it in
``pp`` as the encryptors' default; public parameters without it (earlier
versions) mean ``"facct"``, and decryption does not depend on it. With the rounded
sampler, ``sife/fullysec_ring_lwe``'s keys are ``0.02 / sigma1^2`` per
sample, over ``m n eta`` key samples, from ``D_{Z,sigma1}`` for the ideal
rounded Gaussian -- in the standard configurations about 2^-47 (elementwise
``eta = 1``, ``X = 100``), 2^-44.5 (packed ``eta = 1``) and 2^-40.5 (``eta =
4``, ``X = Y = 4``) -- and about 2^-12 at ``eta = 1`` with the
double-precision path's artifacts, terms its adaptive proof does not budget.

Status API
----------

.. automodule:: pyfe4ai.core.status
   :members:
   :undoc-members:

Profiles API
------------

.. automodule:: pyfe4ai.core.profiles
   :members:
   :undoc-members:
