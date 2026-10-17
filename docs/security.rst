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
