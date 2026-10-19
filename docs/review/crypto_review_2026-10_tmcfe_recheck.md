# Re-check of the threshold MCFE review fixes (October 2026)

Review only: `perf/tmcfe-threshold` (02ae981) and `main` (b51989f) are
unchanged. This branch (`review/crypto-2026-10-tmcfe-recheck`, from 02ae981)
adds `tests/review/test_tmcfe_threshold_recheck.py` and this note. The review
being re-checked is `docs/review/crypto_review_2026-10_tmcfe.md` (4de73ce).

## Verdict on the fixes (4de73ce..02ae981)

| Id | Verdict | Evidence |
| --- | --- | --- |
| L1 | fixed in the bindings: `share_partial` (lib.rs) and `share_partial_bin` (threshold.rs) hold the noise key's material (`secret_flood`), the copied NTT(key share) bytes (`secret_rows`) and words (`secret_hats`), `z` and the noise in `SecretBytes` / `SecretWords`, wiped on drop, so every `?` and early return wipes them (errors inside `read_poly` / `read_values` wipe what they read); `ntt_mont` wipes `x` and `y` (both contexts), `pack` its words; `uniform_flooding` its stream | reading |
| L2 | fixed: mixed material is dealt as values on both backends, for Shamir, replicated, prime / RNS, JSON / binary; `_share_polys_native` refuses a mix | probe r2, archived p6, review test |
| I2 | fixed as reported: mixed replicated pieces and `memoryview` polynomials raise `FEKeyError`; on the default path several malformed shares that b51989f served with another partial (bytes or memoryview `z`, nested lists) are now refused | probe r3 (and against b51989f), p10, review tests |
| I3 | fixed: clients (`encrypt`, `encrypt_lst_ndarray`), servers (`share_decrypt`, hence `compute_lst_ndarray_ct`, full and split shares) and combiners (`combine_decrypt`, `decrypt_lst_ndarray_ct`) refuse every in-place change, addition or removal of a bound field and of the format; restoring the dict restores service; JSON round trips and deep copies of pp, pickle / copy of the objects and one dict shared by every role work; the noise derivation reads bound fields only | probe r4 |
| Info | locks: done, no nested acquisition; threads (p8, r5) without errors or wrong values; the generator's weight store is bounded and a mismatch of parts is always refused; the RNS NTT test reference runs on the Python backend. Not done and not security issues: one parameter record per scheme type (the record check recomputes `q`, `ring_modulus` and `q_factors`, so a prime record is never used for RNS or the reverse: regeneration only) and five-factor RNS moduli on the Python path (same values, p5b) | reading, probes |

Defaults: dealing (replicated, Shamir, Gaussian, seeded), partials (single and
ndarray), combination and native encryption are byte-identical to b51989f on
both backends (fresh inputs, and the archived inputs of the review: the same
SHA-256 as its b51989f outputs). Suites on 02ae981: native, Python backend,
FACCT, rounded and 4 threads each 1970 passed / 19 skipped / 7 xfailed;
`cargo test` (rust/core) 29 passed.

## New findings (this branch's tests fail on 02ae981)

| Id | Severity | Where | Test |
| --- | --- | --- | --- |
| L2b | low (latent) | `ring_lwe_threshold.py:979-981` `_share_polys_native`: all-bytes material is joined and cut by `len(poly) // width` without checking each polynomial; polynomials that are not whole words but add up to whole words are dealt shifted by a byte (shares of other values, no error); a value `>= q` or a total that is not whole words raises `ValueError`. Python refuses all (`FEKeyError`). No caller passes such material | `test_l2b_shamir_dealing_checks_each_binary_polynomial` |
| I2b | info (as on b51989f for JSON / prime) | `share_decrypt` Python path, `_poly_values`, Python sums, native `pack` / `read_poly` errors: a Shamir `z` shorter than `ring_n` gives a shorter partial (an empty `z` an empty result), floats are truncated by `gmpy2.mpz` (Python reads and replicated sums: another result), wrong lengths and bad values raise `IndexError`, `ZeroDivisionError`, `TypeError`, `ValueError`; under binary parameters value-list polynomials fail natively with `ValueError` / `TypeError` | `test_i2b_malformed_value_list_key_shares_are_refused` |
| I10 | info | key generator: `dict_dk` keeps label parts with the share set of their first issuance; once `WEIGHT_SHARINGS_KEPT` evicts the weights' sharing, weight parts come with a new share set only, so the cached label part can never be served by a server holding the current weight part (fail closed: availability). Label parts made while their sharing is evicted before any weight part was fetched name a set that is never handed out | `test_i10_a_cached_label_part_is_served_after_its_sharing_was_evicted` |
| I11 | info (compatibility) | objects pickled by b51989f: generators raise `AttributeError` (`ct_format`, since 5af955b), clients and servers `FESchemeError("the public parameters changed ...")` since 02ae981 (no `_pp_snapshot`), although nothing changed | `test_i11_objects_pickled_by_b51989f_keep_working` |

Residual hygiene (reading, no test): the core kernel `rlwe::share_partial` of
the JSON native path (unchanged since b51989f, also used with RNS) leaves
`acc` and `kp` (sum of ct1 times the key share) unwiped, which
`threshold::share_partial_k` wipes; the legacy two-prime `RnsCtx::ntt_mont_std`
leaves its residues `r` (MIFE / MCFE prototype only); Python copies (packed
bytes, cached hats) cannot be wiped.
