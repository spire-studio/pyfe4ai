"""Shared machinery of the threshold schemes.

* :mod:`~pyfe4ai.schemes.threshold.sharing`: Shamir, replicated and
  seed-expanded replicated sharing, dealt and added up for an enrolled set.
* :mod:`~pyfe4ai.schemes.threshold.binding`: what key shares, ciphertexts and
  partial decryptions are bound to (``pp_binding``, construction ids, weights,
  client tags, sharing ids, formats, state versions) and the checks.
* :mod:`~pyfe4ai.schemes.threshold.codec`: encodings of the key material, the
  threshold ciphertexts and partial decryptions (JSON and binary), strict parsing.
* :mod:`~pyfe4ai.schemes.threshold.store`: the caches and stores of key share
  parts and sums, their locks.
* :mod:`~pyfe4ai.schemes.threshold.roles`: server and combiner logic (checks,
  flooding noise, combination).
* :mod:`~pyfe4ai.schemes.threshold.kl`: the KL accounting of the flooding.
"""
