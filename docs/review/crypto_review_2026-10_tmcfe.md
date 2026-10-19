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
