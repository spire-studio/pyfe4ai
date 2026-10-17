# Performance of the FL aggregation path, per scheme family

Measurements of branch `perf/native-core` (2026-10-17) against v0.3.0 and
against homomorphic encryption (TenSEAL 0.3.18, i.e. Microsoft SEAL) on the
same workload. Raw numbers: `results/perf_native_core_2026-10-17.json`
(first session: a C++ prototype of the native backend, host and docker) and
`results/perf_rust_linux_2026-10-17.json` (the Rust backend in docker, DRBG
and BLS12-381 measurements), `results/perf_tmcfe_wide_q_2026-10-17.json`
(the threshold Ring-LWE MCFE in 256-bit words) and
`results/perf_native_bls_msm_2026-10-17.json` (multi-scalar multiplication on
BLS12-381); the last two were developed on side branches, now merged.
Branch `perf/phase2` (from local `main` 810a4a5) adds the items of *Phase 2*
below; its numbers are in that section.
The C++ prototype was removed once the Rust backend matched it (same kernels,
same outputs, the same tests); its numbers stay here as measured.

## Native coverage (status)

| family | native (Rust) | still Python |
| --- | --- | --- |
| MCFE Ring-LWE (`ring_lwe`, `ring_lwe_decentralized`) | ndarray encryption (rounded or FACCT noise), offline / online phases, key derivation, aggregation, single-vector label pads; dMCFE key shares and their combination | setup |
| MCFE Ring-LWE threshold | any `q` below 2^253 (the flooded `q` of `standard` has 212 bits: 256-bit words): ndarray encryption, key derivation, Shamir shares of the key, partial decryptions (parsing, digests, NTT-domain products), combination | ciphertext tags (JSON + HMAC), setup; the flooding noise is native since the wide-Gaussian sampler below |
| MIFE Ring-LWE | encryption, key derivation, decryption (no ndarray helpers) | setup |
| SIFE Ring-LWE | encryption, `<ct0, y>`, NTTs and products (also of setup) | decoding, setup sampling |
| SIFE FullySec Ring-LWE | NTTs and products (`ring_lwe_utils` dispatch), the wide Gaussian noise (`discrete_gaussian_bigint`) | slot checks, string conversions |
| MCFE dense LWE (`lwe`, `fullysec_lwe`, `lwe_decentralized`) | label hashing and dot products of ndarray encryption / decryption | rounding, decoding, key derivation, single vectors; dMCFE DCR key shares |
| MCFE LWE threshold | ndarray encryption (as `lwe`) | partial decryptions (label hashed again in Python per element), flooding, combination, Shamir shares |
| ABDP15 SIFE / MIFE LWE, MIFE LWE threshold, ALS16 | the wide Gaussians of ALS16 and the threshold flooding (`discrete_gaussian_bigint`); ABDP15 encryption: numpy matrix products | decryption, keys, big-integer matrix products of ALS16, Shamir shares |
| DDH over ristretto255 | MCFE ndarray encryption and decryption with dlog in one call each; group operations | MIFE / SIFE call per group operation |
| DDH over Z_p^*, Damgård, Paillier | -- (gmpy2) | `powmod`; but the label masks / key terms (MCFE) and the randomness powers (SIFE / MIFE Paillier, MIFE DDH) can be computed offline (*Phase 2*), and the dlog tables are keyed by value |
| Pairing schemes, quadratic FE | BLS12-381 group operations, `multi_pair`, products of powers and powers of one base (`multiexp`, `pow_many`), batch deserialisation, G_T dlog, hash to G1 / G2 | matrix algebra mod p, Shamir shares of tMCFE keys; the G_T dlog table is built once per base and bound |

## Phase 2 (branch `perf/phase2`)

Host (M2 Max, Python 3.13), interleaved with local `main` 810a4a5 where
both apply, medians of 3 runs unless stated.

| item | before | after |
| --- | ---: | ---: |
| 1. u64 words for 62-63-bit `q` (`bound_x = 2`, precision 0: 63 bits since the 3X fix), 10^4, binary: client / aggregator / online aggregator | 2.77 / 7.25 / 2.97 ms | 2.04 / 5.14 / 1.59 ms |
| 2. `sampler="facct"`, standard, 10^4, binary, client (1 / 4 threads) | 355 ms (Python) | 18.3 / 6.4 ms (rounded: 2.7 / 1.0 ms) |
| 2. FACCT sampler alone, per sample (sigma 3.2 .. 2^40) | 5.6-7.3 us (Python) | 0.39-0.52 us |
| 3. SIFE Ring-LWE (eta 10, ring_n 4096): setup / encrypt / decrypt | 374 / 200 / 40 ms | 151 / 16 / 15.5 ms |
| 3. SIFE FullySec Ring-LWE (eta 4): setup / encrypt / decrypt | 1257 / 1078 / 290 ms | 510 / 807 / 67 ms |
| 3. dMCFE Ring-LWE, 10^4: key share per client / combination | 25.3 / 12.6 ms | 4.7 / 2.0 ms |
| 5. MCFE DDH / Damgård / Paillier, 20 elements, online client | 212 / 212 / 717 ms | 0.67 / 0.67 / 0.39 ms |
| 5. ... online aggregator | 220 / 225 / 717 ms | 4.6 / 6.9 / 0.9 ms |
| 5. SIFE / MIFE Paillier, MIFE DDH, online encryption per element | 28.7 / 30.5 / 11.4 ms | 0.03-0.04 ms |
| 6. MCFE Ring-LWE single-vector encryption (eta 4) | 141 ms (Python) | 15 ms |

1. **u64 words up to 2^63.** Montgomery and Shoup products and sums need
   only 2q < 2^64; at 62-63 bits the NTT keeps its values in [0, q)
   instead of Harvey's lazy [0, 4q). The 3X fix (q +3 bits) had moved this
   configuration from 60 to 63 bits and into u128 words (+45% / +51%);
   this recovers it.
2. **Native FACCT** (`rust/core/src/facct.rs`): the getrandbits calls and
   decisions of `discrete_gaussian_facct`; the tables come from Python, the
   acceptance test is decided in fixed point (error below 2^-110) and the
   tests it cannot decide (about 2^-99 of them) call the Python formula, so
   the samples for a given bit source are the reference's (its KATs hold
   natively). 30% of the time is ChaCha20, 50 bytes per trial. The default
   stays the rounded sampler.
3. **SIFE Ring-LWE and dMCFE natively; NTT dispatch.**
   `ring_lwe_utils.ntt_forward` / `ntt_inverse` / `poly_mul` (so `RingNTT`
   and every Python caller) use the extension when it has a context for
   (q, n) with the same root. SIFE Ring-LWE encrypts through the native
   `encrypt`; dMCFE key shares are `pad_terms` and `lincomb` calls. SIFE
   FullySec is bound by `discrete_gaussian_bigint` (MPFR Box-Muller, one
   `os.urandom` per `getrandbits`); so are tMCFE's flooding noise and ALS16.
   dMCFE / tMCFE separately, 10^4 elements: dMCFE client 2.8 ms, key share
   4.7 ms per client, combination 2.0 ms, decryption 3.0 ms; tMCFE (q of 212
   bits) client 20.2 ms, key shares 41.3 ms, partial decryption 216 ms per
   server, combination 4.4 ms (unchanged by this branch).
4. **ABDP15 / ALS16 (profiled, nothing ported).** ABDP15 `memory`: 11.1 ms
   per element to encrypt at standard (lwe_n 768), 95% in the numpy / BLAS
   product; `stream`: 60% SHAKE-128 (OpenSSL) -- a Rust port would not beat
   either; decryption 0.25 ms per element. ALS16 (toy only: `standard`
   refuses it): 436-500 ms per element, 70% `discrete_gaussian_bigint`,
   27% big-integer matrix-vector products.
5. **Offline phases for Z_p^* and DCR** (`pyfe4ai/schemes/mcfe/offline.py`,
   `pyfe4ai/schemes/random_offline.py`): the label masks prod H(l)^s of
   MCFE-DDH / Damgård / Paillier and the label terms of a functional key, and
   the randomness powers g^r, pk^r of SIFE- / MIFE-Paillier and MIFE-DDH, are
   computed before the plaintext (the ML adapter's `precompute_gradient` /
   `prepare_aggregation` use the MCFE ones); the same ciphertexts. The
   offline work equals the former total. dlog tables keyed by value instead
   of decimal strings (half the online aggregation). Multi-exponentiation:
   not used -- the weights of the FL path are small and the large exponents
   are moved offline.
6. **Single-vector label pads natively** (identical; 3.8 -> 2.9 ms per pad).

Decided since: FACCT stays opt-in (rounded default; 2.7 vs 18.3 ms per
client above); the native `discrete_gaussian_bigint` that reproduces the MPFR
reference and the packed SIFE FullySec ciphertexts (two versions, opt-in) are
the next sections.

## Native wide Gaussian (branch `perf/native-gaussian-bigint`, from `perf/phase2`)

`sampling_utils.discrete_gaussian_bigint`, the rounded Gaussian of widths
above 2^40 (the noise of SIFE FullySec Ring-LWE and ALS16, the flooding noise
of tMIFE-LWE, tMCFE-LWE and tMCFE-Ring-LWE), runs in Rust
(`rust/core/src/bigauss.rs`, `native_backend.gaussian_bigint`) for widths
below 2^448 when `rng` is an `XofRandom` or `None`, and returns the
reference's integers for the same bits:

* **Same bits.** `b1, b2 = getrandbits(p)` per sample, `p = bits(ceil(sigma))
  + 128`, the same calls and bytes as the reference; an `XofRandom`'s counter
  advances as in Python. With `rng=None` the bytes come from the per-call
  DRBG (stream `c` for samples `256 c .. 256 c + 255`) instead of one
  `os.urandom` per `getrandbits`: the same distribution, as for the other
  native samplers. Any other `rng` (or a replaced `_CSPRNG`), another MPFR
  rounding mode or trap, or a wider sigma keeps the Python code.
* **Same integers.** `X = sig sqrt(-2 ln u1) cos(2 pi u2)` is evaluated in
  fixed point with `n = ceil((s + 64) / 64)` fraction words for `sig < 2^s`
  (2 words for FullySec and ALS16, 5 for the flooding widths): `-ln u1` by two
  table reductions and a short series (for `u1` near 1 with relative
  precision), the square root by Newton with an exact residual check, the
  cosine by quadrant and octant symmetry, a 257-entry table and Taylor
  series; the tables (ln 2, pi, logarithms, cos / sin) are computed in Rust at
  704 bits on first use (about 2 ms). Its error is below `2^(s + 10 - 64 n)`
  (largest measured: `92 * 2^(s - 64 n)` over 4000 random and edge inputs per
  width); the MPFR value, a dozen roundings at `p` bits, is within `2^-118`
  of `X`. A sample is returned when `X` is farther than
  `2^(max(s + 16 - 64 n, -100) + 1)` from every half-integer, where both
  round alike; the others (at most about `2^-45` of them, when `s` is just
  below a multiple of 64; about `2^-62` at FullySec's `s = 47`) are computed in Python with
  the reference formula from the same `b1, b2` (`_box_muller_rint`, which the
  reference loop now calls as well).
* **Tests** (`tests/test_native_gaussian_bigint.py`): known answers taken
  from the reference before the port, on both backends; agreement on
  `XofRandom` streams (1200 samples, then single draws, then the stream's next
  bits) and on the same OS-style bytes, at the widths of FullySec Ring-LWE
  (2^47, 2^49), ALS16 (2^54, 2^64), tMIFE-LWE (2^99, 2^195), tMCFE-Ring-LWE
  (2^199, 2^217), 3.2 and 2^40 + 1; `discrete_gaussian_matrix` and the
  tMIFE / tMCFE flooding helpers; the reference formula for undecided
  samples (margin widened until half or all are undecided); the error bound
  against MPFR at `2p + 256` bits; thread-count independence; the cases that
  keep the Python code.

Host (M2 Max, Python 3.13), medians of 3:

| | Python | native | native, 4 threads |
| --- | ---: | ---: | ---: |
| per sample, sigma 2^49 (FullySec), `rng=None` / `XofRandom` | 11.9 / 7.8 us | 0.56 / 0.98 us | 0.24 us |
| per sample, sigma 2^54 (ALS16) | 8.6 / 7.8 us | 0.55 / 0.98 us | 0.24 us |
| per sample, sigma 2^99 (tMIFE toy) | 9.5 / 8.7 us | 0.89 / 1.27 us | 0.33 us |
| per sample, sigma 2^195 .. 2^217 (tMIFE / tMCFE standard) | 10.2-10.5 / 9.7-14.8 us | 2.0 / 2.3-2.4 us | 0.63-0.66 us |
| one sample per call (the tMIFE flooding loop), 2^195 | 13.8 us | 5.1 us | |
| tMCFE flooding polynomial, 8192 coefficients | 81 ms | 20.8 ms | 7.6 ms |
| SIFE FullySec Ring-LWE (eta 4, standard) encryption, `perf/phase2` vs this branch, interleaved | 964 ms | 120 ms | 90 ms |
| ALS16 SIFE FullySec LWE (toy, eta 4) encryption | 483 ms | 189 ms | |
| tMCFE-Ring-LWE (q 215 bits, 10^4 elements, 3 clients) partial decryption per server, `perf/phase2` vs this branch, interleaved | 225 ms | 107 ms | |
| ... aggregator (key shares, two partial decryptions, combination) | 501 ms | 262 ms | |

The `XofRandom` column costs two SHAKE-256 calls per sample (about 0.2 us
each). In a FullySec encryption the samples now take 42 ms and their
conversion to `mpz` 8 ms; the rest is the scheme's Python code (strings, slot
checks). FullySec setup and decryption do not draw wide Gaussians and are
unchanged. Scheme-level timings with 4 threads were noisy (the sampling part
scales as in the table).

## SIFE FullySec Ring-LWE: packed ciphertexts, two versions (2026-10-18)

Two opt-in ways to put up to `ring_n` vectors into one BMMS21 Sec. 5
ciphertext (Remark 1), both on top of the native `discrete_gaussian_bigint`
(91d0369); both keep the per-element layout as the default, unchanged, and
read it:

* **span** -- `perf/fullysec-packing-native`: `exp/fullysec-packing` (671d3ab,
  `ndarray_layout="packed"`, records `bmms21-sec5-packed/v1`) merged with
  91d0369. The statistical games G3*-G5* use the challenge only through the
  span V of the differences `x1^j - x0^j`, so the complexity leveraging
  ranges over the possible V, `(D + 1)(4X + 1)^(lD)` with `D = min(eta,
  ring_n)`; G3 -> G4 decomposes along a basis of V (`sigma1` gains
  `sqrt(D)`), and the Ring-LHL is applied once per basis vector.
* **all pairs** -- `perf/fullysec-packed` (5222384, `batch = k | "auto"`,
  records `bmms21-sec5-batch/v1`): the leveraging over all
  `(2X + 1)^(2lk)` challenge pairs, as Remark 1 states it; it enters the
  statistical steps only (Ring-LHL `delta` and `eps`, Lemma 1).

**Decryption.** On both branches, column `j` of a packed ciphertext decrypts
to what a one-vector ciphertext of column `j` decrypts to, under packed and
under per-element keys, and that is `<x_j, y>`. The tests cover four toy
configurations plus the standard ones: all three on the span branch, eta 1
and eta 4 on the all-pairs branch. Its eta 10 case is checked by the
assertions of the benchmark below.

Standard profile, host (M2 Max, Python 3.13, Rust extension), medians of
2 runs per variant. Other processes held the load average at 7-10, so the
runs alternate between the variants. "Per element" means per vector of
`eta` values. "Leveraging" is log2 of the number of guesses each argument
pays for; it is `(2X + 1)^(2l)` for one vector per ciphertext.

| configuration | layout | ring_n | m | log2 q | leveraging (bits) | compact B / el | JSON B / el | enc ms / el | dec ms / el | setup s (+ params) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eta 1, X 100, Y 1 (10^4-element gradient, precision 2) | per element | 4096 | 19 | 111 | 15.3 | 1,146,880 | 3,042,891 | 144 | 92 | 0.17 (+0.11) |
| | span | 4096 | 19 | 111 | 9.6 | 344 (280 per full ct) | 913 | 0.039 | 0.016 | 0.17 (+0.11) |
| | all pairs | 8192 | 9 | 205 | 125,355 | 426 (260 per full ct) | 1,074 | 0.048 | 0.025 | 0.15 (+0.14) |
| eta 4, X = Y = 4 (one full ciphertext) | per element | 4096 | 17 | 110 | 25.4 | 1,204,224 | 3,167,892 | 148 | 67 | 0.47 (+0.07) |
| | span | 4096 | 18 | 111 | 67.7 | 308 | 817 | 0.041 | 0.018 | 0.58 (+0.09) |
| | all pairs | 8192 | 9 | 218 | 207,744 | 364 | 903 | 0.042 | 0.016 | 0.54 (+0.13) |
| eta 10, X = Y = 50 (one full ciphertext) | per element | 8192 | 8 | 218 | 133.2 | 4,128,768 | 10,244,005 | 422 | 135 | 1.14 (+0.10) |
| | span | 8192 | 9 | 205 | 768.6 | 494 | 1,246 | 0.061 | 0.018 | 1.17 (+0.13) |
| | all pairs | 16384 | 10 | 431 | 2,181,763 | 1,080 | 2,672 | 0.232 | 0.078 | 8.4 (+0.73) |

How to read the table:
* "Compact" is `(m + eta) ring_n ceil(log2 q / 8)` bytes per ciphertext.
  The gradient's 10^4 elements fill three ciphertexts of 4096 or two of
  8192, so the narrower chunks pad less.
* Per ciphertext, enc / dec ms: eta 4, span 167 / 73, all pairs 347 / 135;
  eta 10, span 498 / 147, all pairs 3803 / 1276. The all-pairs eta 10
  modulus (431 bits) is beyond the native 2^253 words, so its NTTs run in
  Python.
* At eta 1 the two versions are close: span pads less over 10^4 elements;
  all pairs is 7% smaller per full ciphertext, since its larger q comes
  with fewer key polynomials. At eta 10 all pairs needs twice the ring,
  2.2x the bytes and about 4x the time.
* The span version relies on a new argument, under independent check and in
  the code review below. The all-pairs version relies on the leveraging as
  the paper states it, and on the reading that the leveraging enters only
  the statistical steps.

**Cryptographic code review scope.** Both versions go into the planned
cryptographic code review:
* span (671d3ab):
  - the complexity leveraging over the span V of the differences, `(D + 1)(4X + 1)^(lD)`;
  - G3 -> G4 along a basis of V (Lemma 1 for `sigma1^2 I - sigma'^2 B B^T`);
  - the Ring-LHL once per basis vector (`delta` with `log2 D`);
  - the changed m and q: eta 4, m 17 -> 18 and q 110 -> 111 bits; eta 10, m 8 -> 9 and q 218 -> 205 bits.
* all pairs (5222384):
  - the leveraging over `(2X + 1)^(2lk)`, and that it enters only Theorem 2's `delta` and `eps` and Lemma 1's smoothing parameter, not Theorem 1;
  - the resulting m, q and ring_n.

## Workload and method

`bench_fl_compare.py`: every one of 3 clients encrypts a gradient of
10^4 values in `[-1, 1]` with two decimal digits (`encrypt_gradient`); the
aggregator derives the functional key for the round label and decrypts the
sum with weight 1 (`aggregate_gradients`). `security_profile="standard"`,
`bound_x = 100`, `bound_y = 1`, as in the report's aggregation example. For
MCFE over Ring-LWE this gives `ring_n = 4096` and a 77-bit `q` (u128 words
in the native code); `bound_x = 2`, precision 0 gives a 60-bit `q` (u64
words). Client time is one client's encryption, aggregator time includes the
key derivation (`dkgen`). Medians of 9 rounds after one warm-up round; one
thread unless stated. `--offline` precomputes the client masks and the
aggregator's key before each round and reports the online phase.

Machines: *host* = Apple M2 Max, macOS, Python 3.13, clang; *docker* = the
Linux aarch64 VM of Docker Desktop on the same machine (12 CPUs), PyFE4AI in
the `pyfe4ai-pairing` image (Python 3.10, GCC 13), TenSEAL in `pyfe4ai-he`
(Python 3.11, `whitepaper/scripts/he_baseline.py`, means of 9 rounds). The
FE / HE comparison uses the docker numbers only. The Rust backend was built
in a throwaway container of the same, unchanged `pyfe4ai-pairing` image with
a rustup minimal toolchain (rustc 1.99.1, aarch64-unknown-linux-gnu) and the
crates of the host's cache, offline. Docker timings drift by about 10% over
a day; numbers compared with each other come from one session.

## MCFE over Ring-LWE (BMMS21), 10^4 elements, 3 clients

| docker, 77-bit q | client ms | aggregator ms | of which dkgen | ct B/element |
| --- | ---: | ---: | ---: | ---: |
| v0.3.0 (Python) | 229.7 | 367.5 | 41.8 | 66.0 |
| this branch, Python backend | 135.9 | 211.1 | 35.8 | 66.0 |
| native, JSON ciphertexts | 4.58 | 10.76 | 5.18 | 66.0 |
| native, binary ciphertexts | 3.41 | 9.21 | 5.33 | 24.6 |
| native, binary, 4 threads | 1.38 | 4.46 | 2.30 | 24.6 |
| native, binary, online after `--offline` | 0.23 | 3.60 | - | 24.6 |
| native, binary, online, 4 threads | 0.21 | 2.02 | - | 24.6 |
| TenSEAL CKKS, N = 8192 | 7.54 | 2.59 | - | 99.4 |
| TenSEAL BFV, N = 4096 | 2.91 | 1.22 | - | 26.6 |

*native* in these tables is the C++ prototype (first session); the Rust
backend in docker, with the prototype and TenSEAL in one later session, is
under *Rust backend* below (Rust binary: client 3.51 ms, aggregator 9.49 ms).

Offline work per round (docker, one thread): 3.3 ms per client, 5.2 ms for
the aggregator (key terms of 4 labels for 3 clients).

| host, 77-bit q | client ms | aggregator ms |
| --- | ---: | ---: |
| v0.3.0 | 197.5 | 280.0 |
| Python backend | 120.7 | 159.0 |
| native JSON / binary | 4.01 / 3.24 | 8.83 / 7.15 |
| native binary, 4 threads | 1.29 | 3.50 |
| native binary, online / offline | 0.21 / 3.13 | 2.79 / 4.25 |

60-bit q (docker): v0.3.0 217.1 / 346.0 ms, native binary 2.01 / 5.06 ms
(4 threads 0.86 / 2.74 ms).

10^5 elements (docker, 25 chunks): native binary client 28.4 ms (1 thread),
8.7 ms (4), 5.4 ms (8); aggregator 56.7 ms (30.3 ms dkgen), 21.3 ms, 15.2 ms,
online after `--offline` with 8 threads 0.76 / 6.1 ms. TenSEAL CKKS 64.4 /
22.7 ms, BFV 25.3 / 10.9 ms.

Speedup over v0.3.0 (docker, one thread): client 50x (JSON) to 67x
(binary), aggregator 34x to 40x; with the offline phase the online latency
is 1000x (client) and 100x (aggregator) lower.

### Where the time goes (native, 77-bit q, per 4096-coefficient chunk)

| kernel | host ms | docker ms |
| --- | ---: | ---: |
| forward / inverse NTT (u128 words, Shoup + Harvey) | 0.12 / 0.11 | 0.12 / 0.11 |
| forward NTT, u64 words (60-bit q) | 0.027 | 0.028 |
| label pad: SHAKE-256 of 106 KB + reduction | 0.25 | 0.27 |
| OS random bytes for the samples (86 KB) | 0.32 | 0.14 |
| 4096 decimal strings out (JSON format) | 0.16 | 0.18 |

Client: 3 NTTs, one label pad, 3 x 4096 Gaussians and their random bytes per
chunk. Aggregator: per chunk one forward NTT per client and one inverse;
per round one label pad per client for the label and every chunk.

### What remains slower than HE, and why

* Key derivation (aggregator, 4.2 to 5.3 ms of 7 to 9 ms): `(chunks + 1) x
  clients` label pads `H(u'_i, l)` per round, each a SHAKE-256 expansion of
  `ring_n (bits(q) + 128)` bits. Scheme-inherent (identity-based MCFE binds
  keys to labels); labels are known in advance, so `prepare_aggregation`
  moves it before the round.
* Decryption: one forward NTT per client and chunk (`sum_i ct1_i sk_i` with
  a different key per client) against one decryption of the summed HE
  ciphertext. Scheme-inherent; it grows with the number of clients.
* u128 arithmetic for `q > 2^62`: an NTT costs 4x the u64 one. See *RNS*.
* JSON ciphertexts: 0.36 ms of formatting per chunk on the client and the
  parsing on the aggregator; `ct_format="binary"` removes both and is 2.7x
  smaller (24.6 B per element, BFV 26.6, CKKS 99.4).
* Trust model differs: HE needs a decryptor holding the secret key that does
  not collude with the aggregator; the FE aggregator holds a key for the
  weighted sum only, and the key generator is a trusted party.

### Rust backend

`rust/core` (no dependencies) ports these kernels; its C ABI lets
`tests/test_rust_core.py` check them against the Python code bit for bit.
`pyfe4ai._rust_native` (PyO3, `rust/native`) binds it; it replaced the C++
prototype after passing the same bit-identity tests and the comparison
below (`pip install` builds it with cargo when a Rust toolchain is present).

Docker, 10^4 elements, commit `eb9cc29`, one session (medians of 3 runs of
9 rounds; TenSEAL measured right after):

| docker, 77-bit q, same session | client ms | aggregator ms | of which dkgen |
| --- | ---: | ---: | ---: |
| Rust, JSON ciphertexts | 4.37 | 12.07 | 5.19 |
| Rust, binary | 3.51 | 9.49 | 4.89 |
| Rust, binary, 4 threads | 1.54 | 4.83 | 2.14 |
| Rust, binary, online after `--offline` | 0.25 | 4.51 | - |
| Rust, binary, online, 4 threads | 0.26 | 2.34 | - |
| C++ prototype, JSON / binary | 5.35 / 3.95 | 12.65 / 10.30 | 5.98 / 5.87 |
| C++ prototype, binary, online | 0.28 | 4.32 | - |
| TenSEAL CKKS, N = 8192 | 8.01 | 3.01 | - |
| TenSEAL BFV, N = 4096 | 3.30 | 1.31 | - |

Offline phase: Rust 3.50 ms per client and 5.27 ms for the aggregator
(4 threads: 1.31 / 2.19 ms), C++ 3.86 / 6.07 ms. 10^5 elements, Rust,
binary: client 30.4 / 8.5 / 5.2 ms with 1 / 4 / 8 threads, aggregator 60.7
(dkgen 31.4) / 23.5 / 17.3 ms, online after `--offline` with 8 threads 0.91 /
7.6 ms; TenSEAL CKKS 66.8 / 23.2 ms, BFV 26.2 / 11.4 ms.

Rust is faster than C++ everywhere except the online decryption of the
aggregator (+4% in docker, +9% on the host: 3.46 vs 3.18 ms), which spends
its time in moving data between Python and the kernels (the kernels
themselves are on par: forward NTT 0.095 ms, one decryption chunk of 3
clients 0.58 ms). `eb9cc29` removed most of that gap (it was +20%): integers
are read without Python-level `operator.index` / `%` and without the
`OverflowError` that every negative key coefficient used to raise and drop,
and binary ciphertexts are decoded per chunk without the GIL instead of
being copied.

### Sampling randomness (DRBG)

The Rust backend expands one 32-byte OS seed per tensor call with ChaCha20
(`rand_chacha`), chunk `c` reading stream `c`; seed, generator states and
sample bytes are overwritten when the call ends and nothing persists across
calls (a fork or VM snapshot can only duplicate the call in progress).
Before, every sample byte came from the OS (`getentropy` on macOS: 256 bytes
per call, 0.32 ms per 4096-coefficient chunk; `getrandom` on Linux:
0.14 ms). ChaCha20 is a standard stream cipher, not an SP 800-90A DRBG,
which matters only for FIPS validation; the distribution is unchanged (the
same Box-Muller consumption, checked by tests with an injected seed).

Client, 10^4 elements, Rust, one thread; `8f09fc9` (OS bytes), `2b47eea`
(the DRBG alone) and `a722be3` (the DRBG and faster string output),
interleaved, medians of 3 runs of 9 rounds:

| | macOS 8f09fc9 | macOS 2b47eea | macOS a722be3 | docker 8f09fc9 | docker 2b47eea | docker a722be3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JSON | 4.94 ms | 4.44 ms | 4.19 ms | 4.22 ms | 4.26 ms | 3.92 ms |
| binary | 3.76 ms | 3.34 ms | 3.30 ms | 3.28 ms | 3.37 ms | 3.34 ms |
| offline phase (precompute) | 3.78 ms | 3.16 ms | 3.15 ms | 3.22 ms | 3.20 ms | 3.20 ms |
| online phase, binary | 0.28 ms | 0.30 ms | 0.28 ms | 0.21 ms | 0.21 ms | 0.21 ms |

The DRBG saves 0.4 to 0.6 ms per 10^4 elements on macOS and nothing on
Linux, where `getrandom` is already cheap (10^5 elements, docker: binary
27.4 vs 28.1 ms, offline phase 26.4 vs 26.2 ms, within noise). The JSON
difference between `2b47eea` and `a722be3` is the string output, not the
DRBG.

### RNS (not adopted; needs a decision)

With `q` a product of two NTT-friendly primes of about 39 bits, every NTT
becomes two u64 NTTs (2 x 0.027 ms instead of 0.12 ms). Estimated from the
kernel timings: client -17% (binary 3.4 -> about 2.9 ms), aggregator online
-25% (3.6 -> about 2.8 ms). It changes the parameters (composite `q`;
BMMS21's analysis and the parameter derivation assume a prime) and the
binary format, so it needs a security review first; the 60-bit-q rows show
what u64 arithmetic gives.

## Threshold MCFE over Ring-LWE (`mcfe/ring_lwe_threshold`), 10^4 elements

The flooded modulus (statistical parameter 64 on top of BMMS21's `q`) has
212 bits at `ring_n` 8192 under `standard`, beyond the 125-bit u128 words, so
the scheme ran entirely in Python. `rust/core` now has a 256-bit word (four
u64 limbs): Montgomery products with R = 2^256, Shoup products for the
twiddles (`q < 2^253` keeps 4q below 2^256 for the lazy butterflies);
`AnyRing` picks u64 / u128 / 256-bit words by the size of `q`. With it the
MCFE layer (encryption, key terms) runs natively, and three kernels take the
threshold layer: `shamir_share` (the key's coefficients, drawn as
`SystemRandom.randrange(0, q)` draws them, from the per-call DRBG),
`share_inputs` / `share_partial` (a partial decryption's parsing, the JSON
text that `ciphertexts_digest` hashes, and `w = lambda (Delta z + sum_i ct1_i
sk_i) + e`, `c_y` in the NTT domain) and `share_combine`. The flooding noise
is still drawn by `discrete_gaussian_bigint` (mpfr at `bits(sigma_f) + 128`
= 324 bits), now with one call per polynomial instead of one per coefficient
(the same draws, 24% faster); tags are serialised once for all servers.
Since the native wide-Gaussian sampler (section above) the flooding noise is
computed in Rust with the same samples: partial decryption 225 -> 107 ms per
server.

Host (Apple M2 Max, Python 3.13), 3 clients, `s = 3`, `t = 2`, weight 1,
`bound_x = 100`, precision 2; `bench_fl_compare.py --families ring_lwe
ring_lwe_threshold`, before (`50e3188`) and after interleaved, medians of 2
runs (3 / 5 rounds each); "aggregator" is key shares + both partial
decryptions + combination on one machine:

| `ring_lwe_threshold`, 212-bit q, ring_n 8192 | client ms | key shares (2 servers) | partial, per server | combine | aggregator ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| before (`q` beyond the native words: Python) | 213.5 | 275.9 | 528.0 | 18.0 | 1347.9 |
| after, Python backend | 204.6 | 276.0 | 478.7 | 17.9 | 1251.8 |
| after, native | **19.6** | **41.3** | **211.2** | **4.3** | **468.3** |
| after, native, 4 threads | 13.7 | 32.3 | 211.8 | 4.3 | 461.2 |
| `ring_lwe` (77-bit q), before / after | 3.66 / 3.46 | -- | -- | -- | 9.19 / 8.90 |

One 8192-coefficient chunk of a partial decryption takes 103 ms: the
flooding noise 76 ms (Python), the tag checks 7.5 ms (Python),
`share_partial` 8.1 ms (three forward NTTs, one inverse, reading the noise)
and `share_inputs` 6.8 ms. Kernels at 212 bits (`cargo run --release
--example kernels`): NTT 0.97 / 0.83 ms forward / inverse (77-bit q, 4096:
0.11 / 0.09 ms), encryption of 10^4 elements 10.9 ms (two chunks), one
decryption chunk of 3 clients 5.3 ms. The client's other 9 ms are the
ciphertexts' Python strings and the tags (one JSON serialisation per chunk,
one HMAC per server and chunk). The flooding noise of a chunk now takes 21 ms
natively instead of 76 ms (same samples); beyond that, a faster partial
decryption needs a different flooding distribution or a smaller `sigma_f`,
which change the scheme and need a review.

## Dense LWE: MCFE of ABMPSW20 (`mcfe/lwe`), 200 elements

`n0 + m0 = 7724`, 187-bit `q`. One LWE sample per element: the label of every
element is hashed to `Z_q^7724` (309 KB of SHAKE-256) and multiplied with the
client's key. `lwe_matrix.py` (float64 limbs + BLAS) serves the seeded-matrix
ABDP15 schemes, not this one.

| docker | client ms | aggregator ms |
| --- | ---: | ---: |
| v0.3.0 | 1240 | 1249 |
| Python backend (cached keys, `mpz.from_bytes`) | 604 | 609 |
| native, 1 thread | 148 | 167 |
| native, 4 threads | 40 | 49 |

Host: 1021 / 1030 -> 550 / 557 (Python) -> 140 / 147 (native) -> 38 / 45 ms
(4 threads). Per element 0.7 ms per client natively, of which 0.44 ms is the
SHAKE-256 expansion; 10^4 elements take about 7 s per client and thread.

## DDH over ristretto255 (`*/ddh_ristretto255`, new, unreviewed)

The DDH constructions in ristretto255 (RFC 9496) instead of a 3072-bit
`Z_p^*` subgroup, selected with `group="ristretto255"`. MCFE, 3 clients,
host, per element:

| | client | aggregator | ciphertext |
| --- | ---: | ---: | ---: |
| 3072-bit group (Python on gmpy2) | 10.75 ms | 11.10 ms | 936 B |
| ristretto255, pure Python | 3.47 ms | 4.56 ms | 75 B |
| ristretto255, Rust, 1 thread | 0.069 ms | 0.139 ms | 75 B |
| ristretto255, Rust, 4 threads | 0.019 ms | 0.044 ms | 75 B |

10^4 elements with 8 threads: client 107 ms, aggregator 284 ms -- still one
group element per value (no packing), about 30 times the Ring-LWE MCFE.

## DDH, DCR and pairing families (design note)

No native port: 99.6% (client) and 97.6% (aggregator) of the DDH time is in
`gmpy2.powmod` (GMP), and DCR is likewise GMP-bound; docker measurements of
v0.3.0 and this branch agree within noise (DDH 229 / 223 vs 216 / 233 ms,
Paillier 794 / 768 vs 737 / 765 ms per 20 elements).

`bench_offline_potential.py` (host, per element, standard profile):

| | client: full / offline / online ms | aggregator: full / offline / online ms |
| --- | --- | --- |
| DDH (3072-bit group, CDGPP18) | 11.0 / 10.2 / 0.019 | 10.7 / 10.2 / 0.42 |
| DCR (3072-bit N, ABMPSW20) | 35.1 / 35.1 / 0.004 | 35.1 / 35.0 / 0.034 |

* **Offline precomputation.** The client mask `[u_l]^s` (DDH; two 3072-bit
  exponentiations of 5.1 ms) or `H(l)^s mod N^2` (DCR; a 7684-bit exponent)
  depends only on the label and the key; online encryption is then
  `g^x` times the mask (small `x`) or `(1 + x N)` times it, 500x to 9000x
  less. The aggregator's `[u_l]^d` or `H(l)^{-k}` depends on the label and
  the weights; online it multiplies the ciphertexts and, for DDH, solves the
  bounded dlog (0.42 ms). The outputs are the same (the script checks it);
  the total work is not: at 10^4 elements a DDH client spends 100 s and a DCR
  client 350 s per round offline. Implementable without changing outputs,
  like the Ring-LWE `precompute`.
* **Multi-exponentiation** (`u_0^{s_0} u_1^{s_1}` in one pass): about 1.3x on
  GMP's `mpn` layer, but gmpy2 has no such primitive and a Python-level
  Straus loop over gmpy2 products is slower than two `powmod` calls. Not
  worth it without a native GMP binding.
* **Elliptic-curve DDH.** Done: `*/ddh_ristretto255`, see *DDH over
  ristretto255* above (still one group element per value).
* **Pairing backend.** Done: BLS12-381 in Rust, see the next section.

## Pairing families on BLS12-381 (Rust)

`PairingGroup("BLS12-381")` comes from the Rust extension (arkworks 0.6,
`pyfe4ai.utils.bls12_381`) with charm's element interface, and is the
default of every pairing scheme under both profiles; other curve names still
come from charm. Native paths: products of pairings with one final
exponentiation (`pairing_backend.multi_pair`, used by every decryption), the
bounded discrete log in G_T with the baby-step table kept per (base, bound),
RFC 9380 hash to G1 / G2 (BLS12381G1/G2_XMD:SHA-256_SSWU_RO_; SHA-256 in
`rust/core`, checked against the RFC vectors). SGP decryption pairs `a_i`
with `prod_j b_j^{F_ij}` (2 n + 1 pairings instead of 2 n^2 pairings and n^2
G_T exponentiations), for every backend. Elements cross the boundary as
Python objects; serialisation is the compressed encoding (48 / 96 / 576
bytes for G1 / G2 / G_T, base64 behind charm's `"<type>:"` prefix), checked
for subgroup membership on deserialisation. Not constant-time.

Per operation (charm in the Docker image, Linux VM on the same Mac; Rust on
the host; all three in docker in the table after the next):

| | BLS12-381 (Rust) | MNT224 (charm) | BN254 (charm) |
| --- | ---: | ---: | ---: |
| pairing | 0.91 ms | 2.36 ms | 13.7 ms |
| product of 8 pairings | 2.61 ms | 19.4 ms | 110.8 ms |
| G1 / G2 / G_T exponentiation | 0.08 / 0.39 / 0.48 ms | 0.34 / 3.21 / 0.87 ms | 0.44 / 0.87 / 3.16 ms |
| G1 / G2 deserialisation | 0.06 / 0.11 ms | 0.04 / 7.8 ms | 0.03 / 0.06 ms |
| G_T dlog, bound 10^4 (table cached) | 0.05 ms | 1.33 ms | 4.81 ms |

End to end, one setup + key + encryption + decryption round
(`benchmark_quadratic_family.py`, `benchmark_fh_multi_ipe_family.py`,
`--rounds 2`):

| scheme | profile | v0.3 code, MNT224 (charm) | this branch, BLS12-381 (Rust) | |
| --- | --- | ---: | ---: | ---: |
| Quadratic SGP | small / report | 179 / 320 ms | 14 / 36 ms | 13x / 9x |
| Quadratic Quad | small / report | 302 / 506 ms | 32 / 67 ms | 9.5x / 7.6x |
| MIFE FH-Multi-IPE | small / report | 406 / 425 ms | 24 / 25 ms | 17x |
| MCFE FH-Multi-IPE | small / report | 889 / 889 ms | 79 / 82 ms | 11x |
| dMCFE FH-Multi-IPE | small / report | 909 / 945 ms | 80 / 79 ms | 12x |
| tMCFE FH-Multi-IPE | small / report | 1680 / 1565 ms | 137 / 136 ms | 12x |

All in docker (one container with charm and the Rust extension, `report`
profile, end to end):

| scheme | v0.3 code, MNT224 (charm) | BLS12-381 (Rust) | this branch, MNT224 / BN254 (charm) |
| --- | ---: | ---: | ---: |
| Quadratic SGP | 334 ms | 38 ms | 273 / 201 ms |
| Quadratic Quad | 538 ms | 69 ms | 554 / 537 ms |
| MIFE FH-Multi-IPE | 361 ms | 26 ms | 373 / 437 ms |
| MCFE FH-Multi-IPE | 947 ms | 86 ms | 998 / 1032 ms |
| dMCFE FH-Multi-IPE | 1016 ms | 87 ms | 936 / 1018 ms |
| tMCFE FH-Multi-IPE | 1726 ms | 151 ms | 1725 / 1833 ms |

Per operation in docker: pairing 0.96 ms (MNT224 2.63, BN254 14.3), product
of 8 pairings 3.3 ms (19.9 / 113.6), G2 exponentiation 0.41 ms (3.23 /
0.90). The FL path of `bench_fl_compare.py --families pairing` (MCFE
FH-Multi-IPE, `standard` profile, 5 elements, 3 clients): BN254 client 115
ms, aggregator 3703 ms -> BLS12-381 24 ms and 186 ms; ciphertexts 858 ->
1198 B per element (48-byte compressed G1 points against BN254's 33).

This branch on charm (same scripts, `--pairing-group`): MNT224 162 / 301 /
381 / 923 / 943 / 1657 ms and BN254 133 / 323 / 423 / 974 / 981 / 1756 ms
(small profile, same order), i.e. the scheme changes alone do not move the
charm numbers; the gain is the backend. The time left was in native group
operations (pairing products, checked G2 deserialisation, G2
exponentiations in key generation); the exponentiations are batched since,
see the next section.

### Multi-scalar multiplication

`pairing_backend.multiexp(bases, exps)` computes `prod_i b_i^{e_i}` in one
call (`bls_msm`), `pow_many(base, exps)` the powers of one base with one
table (`bls_pow_many`), and `deserialize_many(group, encodings)` decodes a
list in one call, equal encodings once (the 368 entries of Quad's `[M]_1`
have 21 distinct encodings); on charm groups they are the loops they
replace. The schemes use them for every vector: FH-IPE / FH-Multi-IPE
encryption and keys (MIFE, MCFE, dMCFE, tMCFE, with the tMCFE key shares as
one batch of powers of `g2`), SIFE FH-IPE and partial FH-IPE, SGP, Quad
(public key decoded once per encryptor object) and MIQFE (G_T products in
decryption). Keys and ciphertexts are unchanged: FH-IPE outputs with seeded
randomness are byte for byte those of the previous code, and the decryption
tests pass on BLS12-381 and on charm MNT224. Methods (`rust/native/src/msm.rs`):

* G1 / G2: Straus' interleaved wNAF (window 5) over the GLV decomposition
  `k = k1 + lambda k2` (`|k_i| ~ 2^128`), so the 128 doublings are shared by
  all terms; arkworks' Pippenger from 512 terms (its MSM does not use the
  endomorphism, and neither did its G2 scalar multiplication, plain
  double-and-add over 255 bits: a single G2 exponentiation is the one-term
  case and 2.3x faster).
* One base, many exponents: a table of `d 2^(w j) P` and its GLV images,
  each power `~2 * 129 / w` additions without doublings: the NAF (`w = 1`)
  below 12 exponents, windows of 4 / 5 / 6 bits from 12 / 32 / 96.
* G_T: `p = u (mod r)`, so the Frobenius map is `f -> f^u` on G_T, and
  `k = sum_{i<4} k_i |u|^i` (`k_i < 2^64`) turns `f^k` into a four-term
  product over 64-bit exponents (64 cyclotomic squarings instead of 255).
* G_T deserialisation tests membership by `f^(p^4) f = f^(p^2)` (cyclotomic
  subgroup) and `f^p = f^u` (Scott, ePrint 2021/1130; `gcd(Phi_12(p), p - u)
  = r` for BLS12-381, checked) instead of arkworks' `f^r = 1`; the tests
  reject elements of the cyclotomic subgroup outside G_T built in Python.
* `e(g1, g2)` of the FH-IPE, Quad and MIQFE contexts is computed when first
  needed (encryptors never need it), and Python ints of up to 64 bits cross
  into Rust without a call back into Python.

Per operation, host (M2 Max), both builds in one session, best of 5:

| | `perf/native-core` | this branch | |
| --- | ---: | ---: | ---: |
| G1 / G2 / G_T exponentiation | 0.079 / 0.367 / 0.492 ms | 0.058 / 0.159 / 0.215 ms | 1.4x / 2.3x / 2.3x |
| product of 4 powers, G1 / G2 / G_T | 0.34 / 1.63 / 2.02 ms | 0.14 / 0.39 / 0.75 ms | 2.4x / 4.1x / 2.7x |
| product of 16 powers, G1 / G2 / G_T | 1.44 / 6.52 / 8.24 ms | 0.47 / 1.33 / 2.79 ms | 3.0x / 4.9x / 3.0x |
| 16 powers of one base, G1 / G2 / G_T | 1.43 / 6.47 / 8.15 ms | 0.64 / 1.95 / 3.47 ms | 2.2x / 3.3x / 2.3x |
| G_T deserialisation | 0.81 ms | 0.078 ms | 10x |
| G1 / G2 deserialisation | 0.056 / 0.110 ms | 0.055 / 0.110 ms | -- |
| pairing / product of 8 pairings | 0.81 / 2.56 ms | 0.81 / 2.56 ms | -- |

End to end (`report` profile, `--rounds 2`; the two builds alternated twice
on the host, the faster run of each):

| scheme | `perf/native-core` | this branch | |
| --- | ---: | ---: | ---: |
| Quadratic SGP | 35.4 ms | 30.0 ms | 1.18x |
| Quadratic Quad | 64.7 ms | 47.8 ms | 1.35x |
| MIFE FH-Multi-IPE | 24.2 ms | 18.1 ms | 1.34x |
| MCFE FH-Multi-IPE | 79.2 ms | 46.6 ms | 1.70x |
| dMCFE FH-Multi-IPE | 79.4 ms | 48.3 ms | 1.64x |
| tMCFE FH-Multi-IPE | 138.4 ms | 70.2 ms | 1.97x |

Setup: Quad 15.8 -> 5.9 ms, the FH-Multi-IPE MCFE family 2.3-3.1 -> 1.2 ms.
Against v0.3 on charm MNT224 (table above): SGP 11x, Quad 11x, MIFE 23x,
MCFE 19x, dMCFE 20x, tMCFE 22x. Raw numbers:
`results/perf_native_bls_msm_2026-10-17.json`. Not repeated in docker: the
`pyfe4ai-pairing` image has no Rust toolchain.

Where the time goes now (one round, profiled): in the FH-Multi-IPE family,
products of pairings 35% (the Miller loops of the 17-pair FH-IPE
decryptions), checked deserialisation 25% (a G2 point costs a square root
in Fq2 and the `[x] P` subgroup test, ~0.11 ms, batched or not), powers of
one base 22% (tMCFE's 153 key-share powers of `g2`, FH-IPE key generation)
and products of powers 7%. In SGP and Quad, the G_T dlog takes 45%: its
baby-step table (16 sqrt(2 bound + 1) entries, ~17 ms at the report bounds)
is built per base and bound and pays off only over many decryptions, while
the benchmark decrypts once per fresh setup.

BLS12-381 gives about 120-126 bits after the exTNFS attacks (Barbulescu and
Duquesne, J. Cryptol. 2019), below the 128-bit target, so the `standard`
profile keeps a `FESecurityWarning`; a strict 128-bit curve (BLS12-446/461,
BN462) has no mature Rust crate. The `toy` default moved from MNT224 to
BLS12-381 too: pairing keys and ciphertexts of v0.3 are not compatible.
