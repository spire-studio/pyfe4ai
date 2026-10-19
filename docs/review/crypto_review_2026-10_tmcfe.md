# Cryptographic code review: threshold MCFE speed-ups (October 2026)

Review only: `perf/tmcfe-threshold` and `main` are unchanged. This branch
(`review/crypto-2026-10-tmcfe`, from 0716163) adds the tests below.

## Reviewed commits

| Commit | Content |
| --- | --- |
| 5af955b | A: `ct_format="binary"` (format 2, tags v2, binary partials, native uniform flooding) |
| a0a0f75 | B: `ring_modulus="rns"` for the threshold MCFE (RNS backend of `Ring<U256>`) |
| 2cd104e | C1 + C2: split key shares (`get_weight_key_share`, `key_part="label"`), binary key shares |
| 3d06df6, 89d0be3, 0716163 | docs; `_has_bytes` on the first leaf; benchmark options and figures |

Range `b51989f..0716163`. No critical, high or medium finding. The design
claims hold: format 2 is a unique encoding and is refused behind a valid tag
for every non-canonical form (both backends, prime and RNS); one key part
draws one noise (shorter `k` gives prefixes, `ct0` does not enter the noise);
tags v2 bind the parameters' digest, the (chunk) label, the label-hash code,
`k` and both byte digests, and v1 / JSON / other-server tags are refused;
`pp_binding` stays injective with the optional fields; the RNS ring equals the
Python code on the composite `q` on edge values; RNS is sound for the
replicated, seeded, Shamir (`s < min q_i`, checked), KL and Gaussian
constructions; split shares refuse every mismatch of parts; JSON / prime /
full-share defaults are byte-identical to b51989f (dealing with a fixed random
stream, partials, combination, native encryption with a fixed DRBG seed).

## Findings and their tests

| Id | Severity | Where | Test |
| --- | --- | --- | --- |
| L1 | low | `rust/native/src/threshold.rs` `share_partial_bin`, `rust/native/src/lib.rs` `share_partial` (1954-1956), `ntt_mont`: the `z` share read by `read_poly` is not wiped when `noise_values`, `delta_mod` or `index_mod` fail after it (early `?`); the NTT(sk share) words (`hats`) and the `sk_hats` copies are never wiped (pattern of `share_partial` at b51989f, copied into the new function); `ntt_mont` leaves `x` / `y` | reading |
| L2 | low (latent) | `pyfe4ai/schemes/mcfe/ring_lwe_threshold.py:942-946` `_share_polys_native`: material mixing bytes polynomials and value lists is flattened by iterating the bytes (each byte a coefficient): wrong shares without an error; the Python dealing is right. No caller passes a mix today | `test_tmcfe_threshold_review.py::test_l2_native_shamir_dealing_of_mixed_material` |
| I2 | info | `threshold_utils.share_for_enrolled` / `_sum_pieces_native` fallback, `_poly_values`, native `pack`: mixed-type replicated pieces and `memoryview` polynomials are refused, but with `ValueError` / `TypeError` / `IndexError` instead of `FEKeyError` | `test_tmcfe_threshold_review.py::test_i2_malformed_key_material_is_a_key_error` |
| I3 | info | `ring_lwe_threshold.py:984-992` vs `_pp_digest` / `check_pp_binding`: the format is fixed at construction, the binding read from `self.pp` (the caller's dict) at each call; after an in-place `pp["ct_format"] = "binary"` a server serves JSON ciphertexts with a binary key share and the JSON noise inputs (two noises for one key part across server objects) | `test_tmcfe_threshold_review.py::test_i3_server_format_follows_the_parameters_it_binds` |

Further informational items (no test) are in the review report: the
reference of `tests/test_threshold_rns.py::test_ntt_and_products_identical` is
computed natively under the default backend; the weight-share store, the
NTT(sk_y) cache and the share-sum cache are unlocked `OrderedDict`s (no
failure in 640 threaded calls; identity checks rule out stale values) and the
key generator's `_weight_entry` is check-then-set; generator-side stores grow
per weights and per round; one parameter record per scheme type (prime and RNS
configurations in one folder regenerate each other's record); RNS moduli above
240 bits have five factors and run on the Python path.

`tests/review/test_tmcfe_threshold_invariants.py` (passing on 0716163) keeps
the checked invariants: format-2 refusals behind valid tags, one key part one
noise, tag refusals, the RNS ring on edge values against the Python backend,
RNS with the seeded, KL, Gaussian replicated and split constructions.

## Round 3: the remaining items (`fix/tmcfe-threshold-review`)

The fixes of 02ae981 were re-checked twice; the verdicts and the re-check's
own findings (L2b, I2b, I10, I11) are in
`docs/review/crypto_review_2026-10_tmcfe_recheck.md` (5f502ab). This review's
re-check of 02ae981 agrees (every test above passes on both backends, the
probes pass, the five suite configurations pass) and adds that 02ae981 also
fixed a defect the review had missed: binary replicated key shares of a server
that an enrolled set larger than `t` assigns no piece raised `IndexError`
(test
`test_threshold_split_shares.py::test_binary_replicated_shares_with_a_server_assigned_no_piece`).

On `fix/tmcfe-threshold-review` (from 5f502ab), with
`tests/review/test_tmcfe_threshold_round3.py` (fails on 02ae981 and 5f502ab,
passes here):

| Id | Severity | Where (02ae981) | Fix | Test |
| --- | --- | --- | --- | --- |
| I7 | info | `schemes/ipfe.py` `_load_parameters`: one `param.json` per scheme type | `ParameterCacheMixin._param_file_name`; the MIFE / MCFE Ring-LWE family (threshold included) keeps RNS records in `param-rns.json` | `test_i7_prime_and_rns_records_do_not_overwrite_each_other` |
| I8 | info | `utils/ring_lwe_utils.py` `rns_factor_count`: five 60-bit factors from 241 bits, one more than the native ring holds | four factors of up to 62 bits up to 248 bits (`RNS_NATIVE_FACTORS`), a fifth only where the last would exceed 62 bits (`rns_ntt_modulus_for`); above, a warning (Python path) | `test_i8_rns_moduli_up_to_248_bits_run_natively` |
| N1 | low | `ring_lwe_threshold._STORE_LOCK`, `threshold_utils._SHARE_CACHE_LOCK` (new in 02ae981), `native_backend._contexts_lock` (earlier): not replaced in a forked child; a fork while another thread holds one deadlocks the child's next use | `os.register_at_fork(after_in_child=...)` makes new locks, as for `sampling_utils._XOF_LOCK` | `test_n1_module_locks_are_new_in_a_forked_child` |
| I2b (native part) | info | `ring_lwe_threshold._ntt_mont` and the native `share_partial` / `share_partial_bin` calls: a value list holding a bytes element passes `poly_kinds` (first element only) and the native readings raise `TypeError` | those errors raise `FEKeyError` ("malformed key share"); the ciphertexts are read before, so their errors stay `FESchemeError` | `test_i2b_native_a_value_list_holding_bytes_is_a_key_error` |
