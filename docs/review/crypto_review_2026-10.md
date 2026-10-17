# Cryptographic code review, October 2026 (performance work)

Review only: the reviewed branches are unchanged. Each finding below has a
failing test on a review branch cut from the reviewed commit (`tests/review/`).
Later increments are reviewed as diffs against these commits.

## Reviewed commits

| Branch | Commit | Reviewed range | Review branch |
| --- | --- | --- | --- |
| local `main` | 810a4a5 | `426aea8..810a4a5` (v0.3.0 to main, first-parent merges) | `review/crypto-2026-10` |
| `perf/phase2` | 34d2063 | `810a4a5..34d2063` | `review/crypto-2026-10-phase2` |
| `exp/hash-and-flooding` | 3a84434 | `810a4a5..3a84434` | `review/crypto-2026-10-hash` |
| `exp/rns-modulus` | 256b6a9 | `810a4a5..256b6a9` | `review/crypto-2026-10-rns` |

Next increments: the Rust `discrete_gaussian_bigint` (from 34d2063), the KL
flooding prototype (`exp/kl-flooding`, from 810a4a5), a packed layout for
`sife/fullysec_ring_lwe`.

## Findings and their tests

| Id | Severity | Commit | Where | Test |
| --- | --- | --- | --- | --- |
| H1 | high | 34d2063 | `schemes/random_offline.py:62-74`, `schemes/mcfe/offline.py:91-103`: one-time check not atomic; `copy.copy` / fork keep `used = False` -> same `r` twice | phase2 `test_offline_one_time.py` |
| M1 | medium | 810a4a5 | `schemes/mcfe/ring_lwe.py` precompute + native `PrecomputedObj`: reused after `fork()` | `test_precompute_one_time.py::test_precomputation_is_not_reused_after_fork` |
| M2 | medium | all four | `rust/native/src/lib.rs` `encrypt_precomputed` / `precomputed_info` (main 948/970; phase2 1089/1111; hash 964/986; rns 1021/1046, `rns.rs` 276-288): mutex held across `py.detach` -> deadlock | `test_precompute_one_time.py::test_concurrent_use...`, rns `test_rns_precompute_deadlock.py` |
| M3 | medium | 810a4a5 | `rust/native/src/pairing.rs:518-537` GT table cache never evicted; `sife/fh_ipe_pairing.py:250-254` new base per decryption | `test_bls12_381_review.py::test_fh_ipe_decryptions_do_not_grow_memory` |
| M4 | medium | 34d2063 | `utils/native_backend.py:212`: `-neg_c` in the 53-bit default context -> native FACCT acceptance off by 2^-55 | phase2 `test_native_facct_precision.py` |
| M5 | medium | 3a84434 | `schemes/mcfe/ring_lwe.py:83-92`, `mcfe/lwe.py:264`: `label_hash` defaulted, not bound to ciphertexts; tMCFE shares drop it | hash `test_label_hash_binding.py` |
| M6 | medium (pre-existing) | 810a4a5 | `utils/ml_adapter.py:744-755`: (nid, label) checked before, recorded after encryption | `test_adapter_label_race.py` |
| L1 | low | 810a4a5 | `pairing.rs:129,136-139`, `utils/bls12_381.py:96-125`: trailing bytes, lax tags, `OverflowError` | `test_bls12_381_review.py` |
| L2 | low | 810a4a5 | `pairing.rs:526-551`, `ec.rs:109-126`: `2 bound + 1` wraps, no cap | `test_bls12_381_review.py::test_gt_dlog_with_a_bound_of_2_63` |
| L3 | low | 810a4a5 | `dense.rs:44-49`, `rlwe.rs:212-214`, `rlwe.rs:499`: unchecked sizes, divide by zero -> `PanicException` / poisoned mutex | `test_native_robustness.py` |
| L4 | low | 810a4a5 | `ec.rs:183` vs `utils/ec_group.py`: ristretto hex parsed differently by the backends | `test_native_robustness.py` |
| L5 | low | 810a4a5 | `utils/native_backend.py:99`: `PYFE4AI_THREADS` not validated or capped | `test_native_robustness.py` |
| L6 | low (pre-existing) | 810a4a5 | `utils/sampling_utils.py:149`: `gauss_next` cached globally, duplicated by fork | `test_gauss_next_fork.py` |
| L7 | low | 810a4a5, 34d2063 | `rust/core/src/rlwe.rs` (`encrypt_chunk`, `precompute`), `CallDrbg::rng`: r, NTT(r), t, per-chunk buffers, seed copies not wiped | reading |
| L8 | low (latent) | 810a4a5 | `pairing.rs:238-251`: `ZR ** k` reduces k mod r, not r - 1 | reading / agent PoC |
| T1 | test bug (pre-existing) | 4081616.. | `tests/mcfe/test_ddh_threshold.py:245`: tolerance below the quantisation error (about 0.35 % of runs) | `test_flaky_ddh_threshold.py` |

Informational items (no test) are listed in the review report.

## Fixes

On `fix/crypto-review-2026-10` (from `perf/phase2` 34d2063), with the tests of
`review/crypto-2026-10` and `review/crypto-2026-10-phase2` in `tests/review/`
(all passing) and further tests in `tests/test_one_time.py` and
`tests/test_review_fixes.py`. M5 is fixed on `exp/hash-and-flooding`, the RNS
copy of M2 (`rns.rs`) on `exp/rns-modulus`, with the same locking pattern.

| Id | Fix |
| --- | --- |
| H1, M1 | `schemes/one_time.py`: the online phase validates its input, then claims the precomputation under a per-object lock (check and set `used`, move the secret parts out); `copy`, `deepcopy` and pickling raise `TypeError`; use in another process than the creator's is refused, and a forked child's copies are invalidated (`os.register_at_fork`). Applies to `RandomPrecomputation`, `LabelPrecomputation` and `RingLWEPrecomputation` (both backends); the native `PrecomputedObj` records its pid too. `test_offline_one_time.py::test_a_copied_precomputation_cannot_encrypt_twice` now also accepts the refusal of the copy itself. |
| M2 | `lib.rs`: `lock_pre` waits with `MutexExt::lock_py_attached` (detached while waiting); a poisoned lock is a `ValueError`. |
| M3 | `dlog.rs`: LRU table cache (16 tables / 128 MiB per cache, real hash-table sizes) for GT and ristretto255, GT tables keyed by a 128-bit keyed SipHash of the element; `gt_dlog(..., cache=False)` (SIFE FH-IPE) builds an uncached table of about `sqrt(2 bound + 1)` steps. |
| M4 | `native_backend._facct_table` negates the mantissa of `neg_c` exactly; `FacctTable` checks five thresholds against `exact()` when built. |
| M6 | `ml_adapter`: `(nid, label)` reserved under `_label_lock` before encrypting, released on failure. |
| L1 | Strict BLS12-381 strings: tag `^[0-3]$`, canonical base64, exact encoding length (Rust and Python); `OverflowError` caught. |
| L2 | Bounds above `2^48` raise `ValueError` (Rust and Python); at most 1,835,008 baby steps (96 MiB) per native table, `2^16` per pure-Python table. |
| L3 | Checked `count * kl * 8`, `nb * count` (`dense_dots`) and `first + chunks`; `Delta = 0` refused at the bindings and in `decode_scaled`. |
| L4 | Exactly 64 hex digits (either case) on both backends. |
| L5 | `PYFE4AI_THREADS` validated and clamped to `MAX_THREADS`; `set_threads` refuses more. |
| L6 | `_rounded_gaussians`: the Box-Muller of `random.gauss` without the cached second value. |
| L7 | `rlwe.rs`: `wipe_words` for `r`, `NTT(r)`, `t`, the pads and the per-chunk precompute buffers (the stack copy of the DRBG seed that `from_seed` takes by value is not addressed). |
| L8 | `ZR ** k` reduces an integer `k` mod `r - 1`; a ZR exponent keeps its representative. |
| T1 | `test_ddh_threshold.py` compares with the product of the quantised inputs (tolerance `1e-6`). |
| docs | FACCT `2^-110` (conditional on the 128-bit constant, now checked), "wiped when used" (native only), tMCFE / tMIFE flooding remark (per-answer statistical argument only). |

### Re-check of 0ca10e8

`review/crypto-2026-10-fix` (67549be) re-checked the fixes and found five low
issues, fixed on the same branch (`tests/review/test_fix_recheck.py`):

| Id | Fix |
| --- | --- |
| N1 | `FESchemeWrapper.__getstate__` / `__setstate__`: pickling and copying work again; the copy carries a snapshot of the used `(nid, label)` pairs (refused by the copy), not the lock or the cached crypto objects. Copies do not coordinate afterwards. The Ring-LWE key generator and MIFE / SIFE Ring-LWE objects drop their native key caches when pickled. |
| N2 | The per-table cap is a byte budget (`MAX_TABLE_BYTES` = 96 MiB, 1,835,008 steps); the time / memory trade-off above bounds of about `2^33` is documented with measurements in `dlog.rs` and the CHANGELOG. |
| N3 | Sizes follow hashbrown's allocation (`table_bytes`, checked against `HashMap::capacity` by a unit test); the figures are per cache. |
| N4 | GT keys are 128 bits of keyed SipHash-1-3 over the element instead of SHA-256 of its encoding: faster than both the SHA-256 keys and the 576-byte keys of 34d2063. |
| N5 | The pure-Python ristretto255 table is capped at `2^16` baby steps; the CHANGELOG distinguishes the native and Python caps. |
| info | "copy.copy, copy.deepcopy and pickling are refused" (deliberate cloning is not prevented); `check_table` documented as detecting errors above about `2^-102`. |

The deadlock test of `test_precompute_one_time.py` now runs its subprocess with
`PYFE4AI_BACKEND=native`: under `PYFE4AI_BACKEND=python` with the FACCT sampler
its 3M-element Python precomputation exceeded the 120 s timeout.
